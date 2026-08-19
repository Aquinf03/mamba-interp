"""Next-token baselines on frozen v1 splits, with optional coarse layer ablations."""

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

from src.eval.baseline import ablate_mixer_layers, evaluate_split, results_to_markdown
from src.eval.io import load_jsonl


DEFAULT_SPLITS = [
    ("ar",           "ar_eval.jsonl",                 "AR n_pairs=4"),
    ("atr_short",    "atr_short_eval.jsonl",           "induction n_marks=1"),
    ("atr_mid",      "atr_mid_eval.jsonl",             "induction n_marks=2"),
    ("len_short",    "length_len_short_eval.jsonl",    "AR pairs=2"),
    ("len_mid",      "length_len_mid_eval.jsonl",      "AR pairs=8"),
    ("len_ood_pad",  "length_len_ood_pad_eval.jsonl",  "AR pairs=4 + pad32"),
    ("stuff_light",  "length_stuff_light_eval.jsonl",  "AR fill light"),
    ("stuff_heavy",  "length_stuff_heavy_eval.jsonl",  "AR fill heavy"),
    ("factual",      "factual_eval.jsonl",              "fixed facts"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id",      default="state-spaces/mamba-130m-hf",
                   help="HF model id or local checkpoint dir")
    p.add_argument("--splits-dir",    type=Path, default=ROOT / "data" / "splits" / "v1")
    p.add_argument("--dtype",         default="float16",
                   choices=("float16", "bfloat16", "float32"))
    p.add_argument("--device",        default="cuda")
    p.add_argument("--batch-size",    type=int, default=8)
    p.add_argument("--max-examples",  type=int, default=0,
                   help="0 = all; >0 caps each split (for quick debug runs)")
    p.add_argument("--skip-ablation", action="store_true")
    p.add_argument("--out-json",      type=Path, default=ROOT / "logs" / "baseline.json")
    p.add_argument("--out-md",        type=Path, default=ROOT / "logs" / "baseline.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    dtype = getattr(torch, args.dtype)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    print(f"loading {args.model_id} ...", flush=True)
    model = MambaForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    model = model.to(args.device)
    model.eval()
    n_layers = len(model.backbone.layers)
    print(f"n_layers={n_layers}", flush=True)

    results = []
    for name, fname, note in DEFAULT_SPLITS:
        path = args.splits_dir / fname
        if not path.exists():
            print(f"skip missing {path}", flush=True)
            continue
        exs = load_jsonl(path)
        if args.max_examples > 0:
            exs = exs[: args.max_examples]
        print(f"eval {name} n={len(exs)} ...", flush=True)
        r = evaluate_split(model, exs, name=name, device=args.device,
                           batch_size=args.batch_size, config_note=note)
        results.append(r)
        print(f"  acc={r.accuracy:.3f} mean_len={r.mean_seq_len:.1f}", flush=True)

    # Coarse layer-group mixer ablations on AR + ATR-short + stuff_heavy.
    if not args.skip_ablation:
        third = max(1, n_layers // 3)
        groups = {
            "ablate_early": list(range(0, third)),
            "ablate_mid":   list(range(third, 2 * third)),
            "ablate_late":  list(range(2 * third, n_layers)),
        }
        for tname, fname in [("ar", "ar_eval.jsonl"),
                              ("atr_short", "atr_short_eval.jsonl"),
                              ("stuff_heavy", "length_stuff_heavy_eval.jsonl")]:
            path = args.splits_dir / fname
            if not path.exists():
                continue
            exs = load_jsonl(path)
            if args.max_examples > 0:
                exs = exs[: args.max_examples]
            for gname, layers in groups.items():
                print(f"eval {tname}/{gname} ...", flush=True)
                with ablate_mixer_layers(model, layers):
                    r = evaluate_split(model, exs, name=tname, device=args.device,
                                       batch_size=args.batch_size, ablation=gname,
                                       config_note=f"skip mixers {layers[0]}-{layers[-1]}")
                results.append(r)
                print(f"  acc={r.accuracy:.3f}", flush=True)

    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    wall = time.time() - t0
    payload = {
        "model_id": args.model_id,
        "dtype": args.dtype,
        "n_layers": n_layers,
        "peak_vram_gb": round(peak, 3),
        "wall_s": round(wall, 2),
        "results": [r.to_dict() for r in results],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md  = results_to_markdown(results, title="Behavioral baseline")
    md += "\n## Notes\n\n"
    md += f"- model: `{args.model_id}`\n"
    md += f"- peak_vram_gb: {peak:.3f}\n"
    md += f"- wall_s: {wall:.1f}\n"
    md += "- metric: teacher-forced next-token top-1 on frozen v1 splits\n"
    md += "- ablation rows: listed mixers become identity (residual pass-through)\n"
    args.out_md.write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
