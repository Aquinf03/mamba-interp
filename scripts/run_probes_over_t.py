"""Linear probes at multiple sequence positions: first / value_write / last_write / query.

Key question: at ``last_write`` (end of the key-value list, before the query key),
can a linear probe decode the queried value from residual or h? If not, the bind is
not a linearly-accessible letter in the residual stream - it is in h causally.
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

from src.probes.over_t import collect_features, fit_all, make_probe_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id",    default=str(ROOT / "runs" / "ar_ft" / "checkpoint"))
    p.add_argument("--n",           type=int, default=384)
    p.add_argument("--layers",      default="0,11,16,23")
    p.add_argument("--features",    default="residual,h_mean_n,delta")
    p.add_argument("--dtype",       default="float16",
                   choices=("float16", "bfloat16", "float32"))
    p.add_argument("--device",      default="cuda")
    p.add_argument("--train-frac",  type=float, default=0.75)
    p.add_argument("--seed",        type=int, default=99)
    p.add_argument("--probe-steps", type=int, default=400)
    p.add_argument("--out-json",    type=Path, default=ROOT / "logs" / "probes_over_t.json")
    p.add_argument("--out-md",      type=Path, default=ROOT / "logs" / "probes_over_t.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    layers     = [int(x) for x in args.layers.split(",") if x.strip()]
    feat_names = [x.strip() for x in args.features.split(",") if x.strip()]

    examples = make_probe_split(args.model_id, n=args.n, seed=args.seed)
    n = len(examples)
    g    = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_train  = max(8, int(n * args.train_frac))
    train_idx = perm[:n_train]
    test_idx  = perm[n_train:]
    if not test_idx:
        raise SystemExit("empty test set")

    dtype = getattr(torch, args.dtype)
    print(f"loading {args.model_id}  n={n}", flush=True)
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = MambaForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    model = model.to(args.device)
    model.eval()

    bucket, y = collect_features(model, examples, layers=layers,
                                  feat_names=feat_names, device=args.device)
    results   = fit_all(bucket, y, train_idx=train_idx, test_idx=test_idx,
                        probe_steps=args.probe_steps, seed=args.seed)

    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    payload = {
        "model_id": args.model_id,
        "n": n, "n_train": n_train, "n_test": len(test_idx),
        "layers": layers, "features": feat_names,
        "peak_vram_gb": round(peak, 3),
        "wall_s": round(time.time() - t0, 2),
        "results": [r.to_dict() for r in results],
        "note": (
            "Sites: first / value_write / last_write / query. "
            "Residual at last_write near chance = no linearly stored letter. "
            "Residual wins at query because the model has already assembled the answer."
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Probes over time (store vs readout)",
        "",
        f"model: `{args.model_id}`  n={n} (fresh AR seed={args.seed})",
        "",
        "| layer | site | feature | dim | train_acc | test_acc | n_classes |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.layer} | {r.when} | {r.feature} | {r.feat_dim} | "
            f"{r.train_acc:.3f} | {r.test_acc:.3f} | {r.n_classes} |"
        )
    lines.extend(["", "## Best test acc by site × feature (any layer)"])
    by: dict = {}
    for r in results:
        by.setdefault((r.when, r.feature), []).append(r.test_acc)
    for (site, feat), accs in sorted(by.items()):
        lines.append(f"- **{site} / {feat}**: {max(accs):.3f}")
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
