"""Causal interventions on AR or ATR splits (C2 / C3 tables)."""

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

from src.eval.io import load_jsonl
from src.intervene.run import results_to_markdown
from src.intervene.spec import atr_core, late_core
from src.intervene.suite import run_causal_suite
from src.intervene.windows import layer_group


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id",     default=str(ROOT / "runs" / "ar_ft" / "checkpoint"))
    p.add_argument("--split",        type=Path,
                   default=ROOT / "data" / "splits" / "v1" / "ar_eval.jsonl")
    p.add_argument("--dtype",        default="float16",
                   choices=("float16", "bfloat16", "float32"))
    p.add_argument("--device",       default="cuda")
    p.add_argument("--max-examples", type=int, default=0, help="0 = all")
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--groups",       default="late,early,mid")
    p.add_argument("--core-group",   default="late")
    p.add_argument("--table",        default="ar", choices=("ar", "atr"))
    p.add_argument("--layer-sweep",  action="store_true",
                   help="also zero h at each layer individually (slow)")
    p.add_argument("--out-json",     type=Path, default=None)
    p.add_argument("--out-md",       type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    tag = "atr" if args.table == "atr" else "ar"
    if args.out_json is None:
        args.out_json = ROOT / "logs" / f"intervene_{tag}.json"
    if args.out_md is None:
        args.out_md = args.out_json.with_suffix(".md")

    examples = load_jsonl(args.split)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    if not examples:
        raise SystemExit(f"no examples in {args.split}")

    dtype = getattr(torch, args.dtype)
    print(f"loading {args.model_id} ...", flush=True)
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = MambaForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    model = model.to(args.device)
    model.eval()
    n_layers    = len(model.backbone.layers)
    core_layers = layer_group(n_layers, args.core_group)
    extra = {
        gname: layer_group(n_layers, gname)
        for gname in [g.strip() for g in args.groups.split(",") if g.strip()]
        if gname != args.core_group
    }
    specs = atr_core(core_layers) if args.table == "atr" else late_core(core_layers)
    print(
        f"n_layers={n_layers} core={args.core_group} {core_layers[0]}-{core_layers[-1]} "
        f"n={len(examples)} table={args.table}",
        flush=True,
    )

    pack = run_causal_suite(
        model, examples, specs,
        device=args.device,
        core_layers=core_layers,
        seed=args.seed,
        extra_zero_h_groups=extra if args.table == "ar" else None,
        layer_sweep=args.layer_sweep,
    )
    results = pack["results"]
    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    wall = time.time() - t0

    payload = {
        "model_id": args.model_id,
        "split": str(args.split),
        "n": len(examples),
        "dtype": args.dtype,
        "table": args.table,
        "core_group": args.core_group,
        "core_layers": core_layers,
        "peak_vram_gb": round(peak, 3),
        "wall_s": round(wall, 2),
        "budgets": pack["budgets"],
        "results": [r.to_dict() for r in results],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md  = results_to_markdown(results, title=f"Causal interventions ({args.table})")
    md += "## Mixer L2 budgets (zero_h_write vs clean)\n\n"
    md += json.dumps(pack["budgets"], indent=2) + "\n\n"
    md += "## Notes\n\n"
    md += f"- model: `{args.model_id}`\n"
    md += f"- split: `{args.split}`\n"
    md += f"- peak_vram_gb: {peak:.3f}\n"
    md += f"- wall_s: {wall:.1f}\n"
    md += "- residual_noise: random direction, L2 matched to zero_h_write mixer delta\n"
    md += "- swap donor_acc: P(pred = next example's target), excluding collisions\n"
    md += "- ci95: bootstrap over examples\n"
    args.out_md.write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
