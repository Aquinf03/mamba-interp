"""Labeled n=16 demo: clean vs late-h wipe vs restore L17 h.

NOT paper numbers. Writes logs/demo.md only. Does not touch l17_restore.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import MambaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.io import load_jsonl
from src.intervene.run import results_to_markdown, run_condition, summarize_condition
from src.intervene.spec import Intervention
from src.intervene.windows import layer_group

BANNER = """
============================================================
DEMO  n=16  -  not paper numbers
Paper n=128 lives in logs/l17_restore.md (clean 0.961,
late wipe 0.242, restore L17 h 0.961).
============================================================
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo only. Do not cite these accuracies.")
    p.add_argument("--model-id", default=str(ROOT / "runs" / "ar_ft" / "checkpoint"))
    p.add_argument("--split", type=Path, default=ROOT / "data" / "splits" / "v1" / "ar_eval.jsonl")
    p.add_argument("--l17", type=int, default=17)
    p.add_argument("--n", type=int, default=16, help="demo cap; paper uses 128")
    p.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "logs" / "demo.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(BANNER.strip(), flush=True)
    if args.n >= 128:
        raise SystemExit("this is the demo. paper n=128 is scripts/run_l17.py (do not overwrite those logs)")
    ckpt = Path(args.model_id)
    if not (ckpt / "model.safetensors").exists() and not (ckpt / "pytorch_model.bin").exists():
        raise SystemExit(
            f"no checkpoint at {ckpt}\n"
            "Re-FT paper weights: python scripts\\finetune_ar.py --seed 1 --out-dir runs\\ar_ft"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    examples = load_jsonl(args.split)[: args.n]
    dtype = getattr(torch, args.dtype)
    print(f"loading {args.model_id} ...", flush=True)
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = MambaForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    model = model.to(args.device)
    model.eval()
    late = layer_group(len(model.backbone.layers), "late")
    L = args.l17

    print("== clean ==", flush=True)
    clean_scores, _, clean_h = run_condition(
        model, examples, None, device=args.device, layers=late, capture=True, seed=args.seed
    )
    results = [summarize_condition(None, clean_scores, None, layers=late)]
    print(f"  acc={results[-1].acc:.3f}  (DEMO)", flush=True)

    self_l17 = [{L: row[L]} for row in clean_h if L in row]
    wipe = Intervention("zero_h_write_late", "h", "zero", "write", late)
    print("== late h wipe ==", flush=True)
    scores, _, _ = run_condition(
        model, examples, wipe, device=args.device, layers=late, capture=False, seed=args.seed
    )
    results.append(summarize_condition(wipe, scores, clean_scores, layers=late))
    print(f"  acc={results[-1].acc:.3f}  (DEMO)", flush=True)

    rest = Intervention("restore_L17_after_late_zero", "h", "restore", "write", late)
    print("== restore this example's L17 h ==", flush=True)
    scores, _, _ = run_condition(
        model, examples, rest, device=args.device, layers=late, capture=False,
        donors=self_l17, seed=args.seed,
    )
    results.append(summarize_condition(rest, scores, clean_scores, layers=late))
    print(f"  acc={results[-1].acc:.3f}  (DEMO)", flush=True)

    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    wall = time.time() - t0
    md = [
        "# Demo (n=16) - not paper numbers",
        "",
        f"model: `{args.model_id}`  split: first {args.n} of `{args.split.name}`",
        "",
        "**Do not cite.** Paper restore table: `logs/l17_restore.md` (n=128).",
        "",
        results_to_markdown(results, title="demo"),
        f"peak_vram_gb: {peak:.3f}",
        f"wall_s: {wall:.1f}",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md), flush=True)
    print(f"wrote {args.out}  peak_vram_gb={peak:.3f} wall_s={wall:.1f}", flush=True)
    print("DEMO DONE - cite logs/l17_restore.md, not this file.", flush=True)


if __name__ == "__main__":
    main()
