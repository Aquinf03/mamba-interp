"""B1-B3 replication: seed-2 restore, s2 neighbor wipes, ATR L17-only wipe.

Does not overwrite l17_restore.md / intervene_atr_*.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import MambaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_SCRIPTS = str(ROOT / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from src.eval.io import load_jsonl
from src.intervene.run import run_condition, summarize_condition
from src.intervene.spec import Intervention
from src.intervene.windows import layer_group

import run_l17_ctrl as l17c


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--part", default="all", choices=("all", "s2", "atr"))
    p.add_argument("--s2", default=str(ROOT / "runs" / "ar_ft_s2" / "checkpoint"))
    p.add_argument("--s1", default=str(ROOT / "runs" / "ar_ft" / "checkpoint"))
    p.add_argument("--ar-split", type=Path, default=ROOT / "data" / "splits" / "v1" / "ar_eval.jsonl")
    p.add_argument("--atr-split", type=Path, default=ROOT / "data" / "splits" / "v1" / "atr_short_eval.jsonl")
    p.add_argument("--l17", type=int, default=17)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-examples", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=ROOT / "logs")
    return p.parse_args()


def load_model(model_id: str, dtype, device: str):
    print(f"loading {model_id} ...", flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = MambaForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    return model.to(device).eval()


def load_split(path: Path, max_n: int):
    exs = load_jsonl(path)
    if max_n > 0:
        exs = exs[:max_n]
    return exs


def part_s2(args, dtype) -> None:
    examples = load_split(args.ar_split, args.max_examples)
    model = load_model(args.s2, dtype, args.device)
    n_layers = len(model.backbone.layers)
    late = layer_group(n_layers, "late")
    L = args.l17
    L16, L18 = L - 1, L + 1
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    cap: dict = {}
    print("== s2 clean capture ==", flush=True)
    clean_scores, _, _ = run_condition(
        model, examples, None, device=args.device, layers=late,
        capture=True, seed=args.seed, captures=cap,
    )
    clean_row = summarize_condition(None, clean_scores, None, layers=late)
    print(f"  clean acc={clean_row.acc:.3f}", flush=True)
    h_seqs = cap["h_seq"]
    res_seqs = cap["residual_seq"]
    h_lw = l17c.slice_h(h_seqs, examples, L, "last_write")
    r_lw = l17c.slice_res(res_seqs, examples, L, "last_write")
    r_q = l17c.slice_res(res_seqs, examples, L, "query")

    wipe = Intervention("zero_h_write_late", "h", "zero", "write", late)
    wipe_row = l17c.go(model, examples, wipe, device=args.device, layers=late,
                  clean_scores=clean_scores, seed=args.seed)
    rest_h = Intervention(
        "restore_L17_h_last_write", "h", "restore", "write", late, restore_window="last_write"
    )
    rest_h_row = l17c.go(model, examples, rest_h, device=args.device, layers=late,
                    clean_scores=clean_scores, seed=args.seed, donors=h_lw)
    rest_r = Intervention(
        "restore_L17_residual_last_write", "residual", "restore", "write", late,
        restore_window="last_write",
    )
    rest_r_row = l17c.go(model, examples, rest_r, device=args.device, layers=late,
                    clean_scores=clean_scores, seed=args.seed, donor_residuals=r_lw)
    rest_rq = Intervention(
        "restore_L17_residual_query", "residual", "restore", "write", late,
        restore_window="query",
    )
    rest_rq_row = l17c.go(model, examples, rest_rq, device=args.device, layers=late,
                     clean_scores=clean_scores, seed=args.seed, donor_residuals=r_q)

    rows_b1 = [clean_row, wipe_row, rest_h_row, rest_r_row, rest_rq_row]
    l17c.write_md(
        out / "l17_restore_s2.md",
        "B1 seed-2 L17 restore h vs residual",
        rows_b1,
        (
            f"model: `{args.s2}`  split: AR n={len(examples)}\n"
            "Must match seed 1: h restore ~clean, residual restore ~wipe."
        ),
    )
    (out / "l17_restore_s2.json").write_text(
        json.dumps({"model_id": args.s2, "results": [r.to_dict() for r in rows_b1]}, indent=2),
        encoding="utf-8",
    )

    z16 = Intervention("zero_h_write_L16", "h", "zero", "write", [L16])
    z17 = Intervention("zero_h_write_L17", "h", "zero", "write", [L])
    z18 = Intervention("zero_h_write_L18", "h", "zero", "write", [L18])
    r16 = l17c.go(model, examples, z16, device=args.device, layers=[L16],
             clean_scores=clean_scores, seed=args.seed)
    r17 = l17c.go(model, examples, z17, device=args.device, layers=[L],
             clean_scores=clean_scores, seed=args.seed)
    r18 = l17c.go(model, examples, z18, device=args.device, layers=[L18],
             clean_scores=clean_scores, seed=args.seed)
    rows_b2 = [clean_row, r16, r17, r18]
    l17c.write_md(
        out / "l17_neighbors_s2.md",
        "B2 seed-2 single-layer write wipe L16/L17/L18",
        rows_b2,
        (
            f"model: `{args.s2}`\n"
            "L17 must collapse; L16 and L18 must stay near clean (seed-1 pattern)."
        ),
    )
    (out / "l17_neighbors_s2.json").write_text(
        json.dumps({"results": [r.to_dict() for r in rows_b2]}, indent=2),
        encoding="utf-8",
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def part_atr(args, dtype) -> None:
    examples = load_split(args.atr_split, args.max_examples)
    model = load_model(args.s1, dtype, args.device)
    n_layers = len(model.backbone.layers)
    late = layer_group(n_layers, "late")
    L = args.l17
    L16, L18 = L - 1, L + 1
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print("== ATR-short clean ==", flush=True)
    clean_scores, _, _ = run_condition(
        model, examples, None, device=args.device, layers=late, capture=False, seed=args.seed
    )
    clean_row = summarize_condition(None, clean_scores, None, layers=late)
    print(f"  clean acc={clean_row.acc:.3f}", flush=True)

    late_wipe = Intervention("zero_h_write_late", "h", "zero", "write", late)
    late_row = l17c.go(model, examples, late_wipe, device=args.device, layers=late,
                  clean_scores=clean_scores, seed=args.seed)
    z16 = Intervention("zero_h_write_L16", "h", "zero", "write", [L16])
    z17 = Intervention("zero_h_write_L17", "h", "zero", "write", [L])
    z18 = Intervention("zero_h_write_L18", "h", "zero", "write", [L18])
    r16 = l17c.go(model, examples, z16, device=args.device, layers=[L16],
             clean_scores=clean_scores, seed=args.seed)
    r17 = l17c.go(model, examples, z17, device=args.device, layers=[L],
             clean_scores=clean_scores, seed=args.seed)
    r18 = l17c.go(model, examples, z18, device=args.device, layers=[L18],
             clean_scores=clean_scores, seed=args.seed)
    rows = [clean_row, late_row, r16, r17, r18]
    l17c.write_md(
        out / "l17_wipe_atr.md",
        "B3 ATR-short L17-only write wipe",
        rows,
        (
            f"model: `{args.s1}`  split: `{args.atr_split}`\n"
            "Late-block wipe already 0.039. If L17-only also collapses ATR, the store is the same layer as AR."
        ),
    )
    (out / "l17_wipe_atr.json").write_text(
        json.dumps({"results": [r.to_dict() for r in rows]}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    dtype = getattr(torch, args.dtype)
    t0 = time.time()
    if args.part in ("all", "s2"):
        part_s2(args, dtype)
    if args.part in ("all", "atr"):
        part_atr(args, dtype)
    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    print(f"peak_vram_gb={peak:.3f} wall_s={time.time() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
