"""L17 sufficiency test, key-matched value patch, and N-channel wipes.

Three experiments in one script:
  1. Restore: zero late h during write, paste this example's clean L17 h back at
     the last write position → accuracy recovers to clean. Proves L17 h is sufficient.
  2. Channels: wipe each of the 16 N-slots individually to find which slots carry the bind.
  3. Key-matched patch: donor and clean sequences share keys but differ on the queried
     value. Patching donor L17 h steers prediction toward the donor value.

Writes logs/l17_restore.md, l17_channels.md, l17_keypatch.md.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import MambaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.io import load_jsonl
from src.eval.schema import EvalExample
from src.eval.vocab import TokenPool, format_with_sep
from src.intervene.run import results_to_markdown, run_condition, summarize_condition
from src.intervene.spec import Intervention
from src.intervene.windows import layer_group


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id",    default=str(ROOT / "runs" / "ar_ft" / "checkpoint"))
    p.add_argument("--split",       type=Path,
                   default=ROOT / "data" / "splits" / "v1" / "ar_eval.jsonl")
    p.add_argument("--l17",         type=int, default=17)
    p.add_argument("--dtype",       default="float16",
                   choices=("float16", "bfloat16", "float32"))
    p.add_argument("--device",      default="cuda")
    p.add_argument("--max-examples", type=int, default=0)
    p.add_argument("--n-keypatch",  type=int, default=128)
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--out-dir",     type=Path, default=ROOT / "logs")
    return p.parse_args()


def _make_example(pool: TokenPool, pairs: list, query_key: str,
                  target_val: str, eid: str) -> EvalExample:
    pieces = []
    for k, v in pairs:
        pieces.extend([k, v])
    pieces.append(query_key)
    text = format_with_sep(pieces, pool.sep)
    ids  = pool.encode(text)
    tid  = pool.encode(target_val)[0]
    return EvalExample(
        example_id=eid,
        task="ar", split="keypatch",
        text=text, input_ids=ids,
        target_id=tid, target_text=target_val,
        meta={
            "n_pairs": len(pairs), "n_queries": 1, "filler_every": 0,
            "filler_n": 0, "ood_pad": 0,
            "pairs": [{"key": k, "value": v} for k, v in pairs],
            "final_key": query_key, "final_value": target_val,
        },
    )


def keypatch_pairs(pool: TokenPool, n: int, seed: int):
    """Create n matched (clean, donor) pairs that share keys but differ on the queried value."""
    rng = random.Random(seed)
    cleans, donors = [], []
    for i in range(n):
        k0, k1, v0, v1, v1p = rng.sample(pool.symbols, 5)
        cleans.append(_make_example(pool, [(k0, v0), (k1, v1)],  k1, v1,  f"kp_c_{i:04d}"))
        donors.append(_make_example(pool, [(k0, v0), (k1, v1p)], k1, v1p, f"kp_d_{i:04d}"))
    return cleans, donors


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    examples = load_jsonl(args.split)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    dtype = getattr(torch, args.dtype)
    print(f"loading {args.model_id} ...", flush=True)
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = MambaForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    model = model.to(args.device)
    model.eval()
    n_layers = len(model.backbone.layers)
    late = layer_group(n_layers, "late")
    L17 = args.l17
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Restore experiment ---
    print("== clean capture ==", flush=True)
    clean_scores, _, clean_h = run_condition(
        model, examples, None, device=args.device, layers=late, capture=True, seed=args.seed
    )
    clean_row = summarize_condition(None, clean_scores, None, layers=late)
    print(f"  clean acc={clean_row.acc:.3f}", flush=True)

    self_l17 = [{L17: row[L17]} for row in clean_h if L17 in row]
    if len(self_l17) != len(examples):
        raise SystemExit(f"missing L{L17} h capture ({len(self_l17)}/{len(examples)})")

    results = [clean_row]
    wipe = Intervention("zero_h_write_late", "h", "zero", "write", late)
    print("== zero late h ==", flush=True)
    scores, _, _ = run_condition(model, examples, wipe, device=args.device,
                                 layers=late, capture=False, seed=args.seed)
    results.append(summarize_condition(wipe, scores, clean_scores, layers=late))
    print(f"  acc={results[-1].acc:.3f}", flush=True)

    rest = Intervention("restore_L17_after_late_zero", "h", "restore", "write", late)
    print("== restore clean L17 h ==", flush=True)
    scores, _, _ = run_condition(model, examples, rest, device=args.device, layers=late,
                                 capture=False, donors=self_l17, seed=args.seed)
    results.append(summarize_condition(rest, scores, clean_scores, layers=late))
    print(f"  acc={results[-1].acc:.3f}", flush=True)

    only = Intervention("zero_h_write_L17", "h", "zero", "write", [L17])
    print("== zero L17 only ==", flush=True)
    scores, _, _ = run_condition(model, examples, only, device=args.device,
                                 layers=[L17], capture=False, seed=args.seed)
    results.append(summarize_condition(only, scores, clean_scores, layers=[L17]))
    print(f"  acc={results[-1].acc:.3f}", flush=True)

    md = results_to_markdown(results, title="L17 restore (sufficiency)")
    md += (
        f"\nZero h on late layers during write, then restore this example's clean "
        f"L{L17} h at the last write token. Accuracy returning to clean confirms "
        f"L{L17} h is sufficient for the bind.\n"
    )
    (args.out_dir / "l17_restore.md").write_text(md, encoding="utf-8")
    (args.out_dir / "l17_restore.json").write_text(
        json.dumps({"results": [r.to_dict() for r in results]}, indent=2), encoding="utf-8"
    )
    print(md)

    # --- 2. Channel wipes ---
    ch_rows = [summarize_condition(None, clean_scores, None, layers=[L17])]
    n_state = int(model.backbone.layers[L17].mixer.ssm_state_size)
    print(f"== L{L17} channel wipes N={n_state} ==", flush=True)
    for n in range(n_state):
        spec = Intervention(f"zero_h_L{L17}_N{n}", "h", "zero", "write", [L17], channels=[n])
        scores, _, _ = run_condition(model, examples, spec, device=args.device,
                                     layers=[L17], capture=False, seed=args.seed)
        row = summarize_condition(spec, scores, clean_scores, layers=[L17])
        ch_rows.append(row)
        print(f"  N={n} acc={row.acc:.3f}", flush=True)
    md_ch = results_to_markdown(ch_rows, title=f"L{L17} state-channel wipes (N={n_state})")
    (args.out_dir / "l17_channels.md").write_text(md_ch, encoding="utf-8")
    (args.out_dir / "l17_channels.json").write_text(
        json.dumps({"results": [r.to_dict() for r in ch_rows]}, indent=2), encoding="utf-8"
    )
    print(md_ch)

    # --- 3. Key-matched patch ---
    pool = TokenPool(tokenizer_name=args.model_id)
    nkp  = args.n_keypatch if args.max_examples <= 0 else min(args.n_keypatch, args.max_examples)
    cleans, d_exs = keypatch_pairs(pool, nkp, args.seed + 7)
    print(f"== keypatch n={nkp} ==", flush=True)
    d_scores, _, d_h = run_condition(model, d_exs, None, device=args.device,
                                     layers=[L17], capture=True, seed=args.seed)
    c_scores, _, _   = run_condition(model, cleans, None, device=args.device,
                                     layers=[L17], capture=False, seed=args.seed)
    donor_only = [{L17: row[L17]} for row in d_h]
    spec = Intervention("keypatch_L17_h", "h", "swap", "write", [L17])
    p_scores, _, _ = run_condition(model, cleans, spec, device=args.device, layers=[L17],
                                   capture=False, donors=donor_only, seed=args.seed)
    kp = {
        "n":                       nkp,
        "clean_acc":               sum(1 for s in c_scores if s.correct) / nkp,
        "donor_seq_acc":           sum(1 for s in d_scores if s.correct) / nkp,
        "patch_acc_original_value": sum(1 for s in p_scores if s.correct) / nkp,
        "patch_acc_donor_value":    sum(
            1 for s, d in zip(p_scores, d_exs) if int(s.pred_id) == int(d.target_id)
        ) / nkp,
    }
    print(json.dumps(kp, indent=2), flush=True)
    md_kp = "\n".join([
        "# L17 key-matched value patch",
        "",
        "Same keys; donor has a different value for the queried key. "
        "Patch donor L17 h at last write onto the clean sequence.",
        "",
        "| n | clean acc | donor-seq acc | still original V | **now donor V** |",
        "|---:|---:|---:|---:|---:|",
        f"| {nkp} | {kp['clean_acc']:.3f} | {kp['donor_seq_acc']:.3f} | "
        f"{kp['patch_acc_original_value']:.3f} | {kp['patch_acc_donor_value']:.3f} |",
        "",
        "If **now donor V** >> chance and original V drops, L17 h is content-addressable.",
        "",
    ])
    (args.out_dir / "l17_keypatch.md").write_text(md_kp, encoding="utf-8")
    (args.out_dir / "l17_keypatch.json").write_text(json.dumps(kp, indent=2), encoding="utf-8")
    print(md_kp)

    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    print(f"peak_vram_gb={peak:.3f} wall_s={time.time() - t0:.1f}", flush=True)
    print("wrote logs/l17_restore.md l17_channels.md l17_keypatch.md", flush=True)


if __name__ == "__main__":
    main()
