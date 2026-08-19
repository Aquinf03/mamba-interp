"""Length/stuffing failure analysis: state geometry and Δ-clamp fix.

OOD pad failures respond to Δ-clamp (clamping junk Δ → 0 recovers accuracy).
Stuffing failures do not - the filler tokens are load-bearing and the bind simply
does not survive being displaced by noisy context. These two mechanisms produce
opposite signs of ``recovery``.
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

from src.eval.baseline import evaluate_split
from src.eval.io import load_jsonl
from src.intervene.failure import example_geometry, mean_by_correct, run_delta_clamp_junk
from src.intervene.windows import layer_group


SPLITS = [
    ("len_short",   "length_len_short_eval.jsonl"),
    ("len_mid",     "length_len_mid_eval.jsonl"),
    ("len_ood_pad", "length_len_ood_pad_eval.jsonl"),
    ("stuff_light", "length_stuff_light_eval.jsonl"),
    ("stuff_heavy", "length_stuff_heavy_eval.jsonl"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id",     default=str(ROOT / "runs" / "ar_ft" / "checkpoint"))
    p.add_argument("--splits-dir",   type=Path, default=ROOT / "data" / "splits" / "v1")
    p.add_argument("--dtype",        default="float16",
                   choices=("float16", "bfloat16", "float32"))
    p.add_argument("--device",       default="cuda")
    p.add_argument("--max-examples", type=int, default=0)
    p.add_argument("--geom-layers",  default="16,19,23")
    p.add_argument("--fix-group",    default="late")
    p.add_argument("--out-json",     type=Path, default=ROOT / "logs" / "failure.json")
    p.add_argument("--out-md",       type=Path, default=ROOT / "logs" / "failure.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    geom_layers = [int(x) for x in args.geom_layers.split(",") if x.strip()]
    dtype = getattr(torch, args.dtype)
    print(f"loading {args.model_id} ...", flush=True)
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = MambaForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    model = model.to(args.device)
    model.eval()
    n_layers   = len(model.backbone.layers)
    fix_layers = layer_group(n_layers, args.fix_group)

    payload_splits = []
    md_lines = [
        "# Failure modes",
        "",
        f"model: `{args.model_id}`",
        "",
        "| split | n | clean_acc | clamp_Δ_junk_acc | recovery | mean_len |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name, fname in SPLITS:
        path = args.splits_dir / fname
        if not path.exists():
            print(f"skip {path}", flush=True)
            continue
        exs = load_jsonl(path)
        if args.max_examples > 0:
            exs = exs[: args.max_examples]
        print(f"split {name} n={len(exs)} ...", flush=True)
        clean = evaluate_split(model, exs, name=name, device=args.device, batch_size=8)
        print(f"  clean acc={clean.accuracy:.3f}", flush=True)

        geom_rows = []
        for i, ex in enumerate(exs):
            geom_rows.append(example_geometry(model, ex, layers=geom_layers, device=args.device))
            if (i + 1) % 16 == 0:
                print(f"    geom {i + 1}/{len(exs)}", flush=True)

        scores  = run_delta_clamp_junk(model, exs, layers=fix_layers, device=args.device)
        fix_acc = sum(1 for s in scores if s.correct) / len(scores) if scores else 0.0
        recov   = fix_acc - clean.accuracy
        print(f"  clamp_delta_junk acc={fix_acc:.3f} recovery={recov:+.3f}", flush=True)

        keys = [
            f"L{geom_layers[-1]}_h_write_norm",
            f"L{geom_layers[-1]}_h_query_norm",
            f"L{geom_layers[-1]}_h_write_erank",
            f"L{geom_layers[-1]}_delta_junk",
            f"L{geom_layers[-1]}_delta_query",
        ]
        geom_sum = {k: mean_by_correct(geom_rows, k) for k in keys}
        payload_splits.append({
            "name": name, "n": len(exs),
            "clean_acc": clean.accuracy,
            "clean_lp": clean.mean_target_logprob,
            "clamp_delta_junk_acc": fix_acc,
            "recovery": recov,
            "mean_len": clean.mean_seq_len,
            "geometry": geom_sum,
        })
        md_lines.append(
            f"| {name} | {len(exs)} | {clean.accuracy:.3f} | "
            f"{fix_acc:.3f} | {recov:+.3f} | {clean.mean_seq_len:.1f} |"
        )

    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    wall = time.time() - t0
    payload = {
        "model_id": args.model_id,
        "geom_layers": geom_layers,
        "fix_layers": fix_layers,
        "peak_vram_gb": round(peak, 3),
        "wall_s": round(wall, 2),
        "splits": payload_splits,
        "note": (
            "clamp_Δ_junk: Δ←0 on filler/pad so those tokens neither write nor decay h. "
            "OOD pad failure: recovery > 0 (junk was overwriting the bind). "
            "Stuffing failure: recovery < 0 (filler is load-bearing)."
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines.extend(["", "## Geometry (mean on correct vs wrong)", ""])
    md_lines.append("| split | metric | correct | wrong |")
    md_lines.append("|---|---|---:|---:|")
    for sp in payload_splits:
        for k, v in (sp.get("geometry") or {}).items():
            c, w = v.get("correct"), v.get("wrong")
            md_lines.append(
                f"| {sp['name']} | {k} | "
                f"{'-' if c is None else f'{c:.4f}'} | {'-' if w is None else f'{w:.4f}'} |"
            )
    md_lines.extend(["", f"peak_vram_gb: {peak:.3f}  wall_s: {wall:.1f}", ""])
    md = "\n".join(md_lines) + "\n"
    args.out_md.write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
