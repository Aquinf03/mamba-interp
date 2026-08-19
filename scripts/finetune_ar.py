"""Finetune Mamba-130M on synthetic associative recall.

Trains next-token CE on the value position only. Master weights stay FP32;
``--dtype`` controls the autocast compute type. Saves to ``runs/ar_ft/checkpoint``
by default and refuses to clobber an existing checkpoint without ``--force``,
protecting the paper weights.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import MambaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.ar_train import build_ar_train_set
from src.eval.baseline import evaluate_split
from src.eval.io import load_jsonl
from src.eval.vocab import TokenPool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id",    default="state-spaces/mamba-130m-hf")
    p.add_argument("--out-dir",     type=Path, default=ROOT / "runs" / "ar_ft")
    p.add_argument("--eval-split",  type=Path,
                   default=ROOT / "data" / "splits" / "v1" / "ar_eval.jsonl")
    p.add_argument("--n-train",     type=int,   default=2048)
    p.add_argument("--steps",       type=int,   default=800)
    p.add_argument("--batch-size",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--dtype",       default="float16",
                   choices=("float16", "bfloat16", "float32"))
    p.add_argument("--device",      default="cuda")
    p.add_argument("--seed",        type=int,   default=1)
    p.add_argument("--eval-every",  type=int,   default=200)
    p.add_argument("--max-eval",    type=int,   default=128)
    p.add_argument("--grad-accum",  type=int,   default=1)
    p.add_argument("--save-every",  type=int,   default=0,
                   help="if >0, also save a snapshot at runs/.../step_{k}")
    p.add_argument("--force",       action="store_true",
                   help="allow overwriting an existing checkpoint (paper weights live in runs/ar_ft)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    torch.manual_seed(args.seed)
    existing = args.out_dir / "checkpoint" / "model.safetensors"
    if existing.exists() and not args.force:
        raise SystemExit(
            f"refusing to overwrite {existing}\n"
            "Paper checkpoint is runs/ar_ft/checkpoint. "
            "Use a new --out-dir (e.g. runs/ar_ft_s2) or pass --force."
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    amp_dtype  = getattr(torch, args.dtype)
    use_amp    = args.dtype in ("float16", "bfloat16") and args.device.startswith("cuda")
    use_scaler = args.dtype == "float16" and args.device.startswith("cuda")

    print(f"loading {args.model_id} ...", flush=True)
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = MambaForCausalLM.from_pretrained(args.model_id)
    model = model.to(args.device)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    pool      = TokenPool(tokenizer_name=args.model_id)
    train_set = build_ar_train_set(args.seed, pool, n=args.n_train)
    eval_ex   = load_jsonl(args.eval_split)
    if args.max_eval > 0:
        eval_ex = eval_ex[: args.max_eval]

    print(
        f"train n={len(train_set)} steps={args.steps} bs={args.batch_size} "
        f"lr={args.lr} amp={args.dtype} scaler={use_scaler}",
        flush=True,
    )

    history = []
    step = i_cursor = accum = 0
    opt.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    while step < args.steps:
        batch = [train_set[i_cursor % len(train_set)] for _ in range(args.batch_size)]
        i_cursor += args.batch_size
        lengths = [len(ex.input_ids) for ex in batch]
        max_len = max(lengths)
        ctx     = torch.zeros((len(batch), max_len), dtype=torch.long, device=args.device)
        targets = torch.tensor([ex.target_id for ex in batch], device=args.device)
        for j, ex in enumerate(batch):
            L = len(ex.input_ids)
            ctx[j, :L] = torch.tensor(ex.input_ids, dtype=torch.long, device=args.device)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(input_ids=ctx, use_cache=False)
            last_logits = torch.stack(
                [out.logits[j, L - 1] for j, L in enumerate(lengths)], dim=0
            )
            loss = F.cross_entropy(last_logits.float(), targets)

        scaler.scale(loss / args.grad_accum).backward()
        accum += 1
        if accum < args.grad_accum:
            continue
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        accum = 0
        step += 1

        if step % 50 == 0 or step == 1:
            print(f"step {step}/{args.steps} loss={loss.item():.4f}", flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                r = evaluate_split(model, eval_ex, name="ar_eval",
                                   device=args.device, batch_size=8)
            model.train()
            rec = {"step": step, "loss": float(loss.item()),
                   "eval_acc": r.accuracy, "eval_lp": r.mean_target_logprob}
            history.append(rec)
            print(f"  eval acc={r.accuracy:.3f} lp={r.mean_target_logprob:.3f}", flush=True)
            if args.save_every > 0:
                snap = args.out_dir / f"step_{step}"
                snap.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(snap)
                pool.tokenizer.save_pretrained(snap)
                print(f"  saved {snap}", flush=True)

    model.eval()
    save_dir = args.out_dir / "checkpoint"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    pool.tokenizer.save_pretrained(save_dir)

    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    meta = {
        "base_model": args.model_id,
        "n_train": args.n_train,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "history": history,
        "final_eval_acc": history[-1]["eval_acc"] if history else None,
        "peak_vram_gb": round(peak, 3),
        "wall_s": round(time.time() - t0, 2),
        "checkpoint": str(save_dir),
    }
    (args.out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"saved {save_dir}", flush=True)
    if meta["final_eval_acc"] is not None and meta["final_eval_acc"] < 0.3:
        print("NOTE: eval_acc still low - try more --steps or --n-train", flush=True)
    else:
        print("OK: checkpoint ready", flush=True)


if __name__ == "__main__":
    main()
