"""Linear probes on residual vs state at the AR query token (store vs readout)."""

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
from src.hooks import capture_mamba_states
from src.probes.linear import FEATURE_FNS, ProbeResult, fit_linear_probe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="state-spaces/mamba-130m-hf", help="HF id or local checkpoint dir")
    p.add_argument("--split", type=Path, default=ROOT / "data" / "splits" / "v1" / "ar_eval.jsonl")
    p.add_argument("--layers", default="0,11,23")
    p.add_argument("--features", default="residual,residual_out,h_mean_e,h_mean_n,h_flat,delta")
    p.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-examples", type=int, default=0, help="0=all")
    p.add_argument("--train-frac", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe-steps", type=int, default=400)
    p.add_argument("--out-json", type=Path, default=ROOT / "logs" / "probes.json")
    p.add_argument("--out-md", type=Path, default=ROOT / "logs" / "probes.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    dtype = getattr(torch, args.dtype)
    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    feat_names = [x.strip() for x in args.features.split(",") if x.strip()]
    for f in feat_names:
        if f not in FEATURE_FNS:
            raise SystemExit(f"unknown feature {f}; choose {list(FEATURE_FNS)}")

    examples = load_jsonl(args.split)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    n = len(examples)
    if n < 16:
        raise SystemExit(f"need more examples, got {n}")

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_train = max(8, int(n * args.train_frac))
    train_idx = set(perm[:n_train])
    test_idx = [i for i in perm[n_train:]]
    if not test_idx:
        raise SystemExit("empty test set; lower train-frac or raise n")

    print(f"loading {args.model_id} ...", flush=True)
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = MambaForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    model = model.to(args.device)
    model.eval()

    # Collect features at last context token (query key position).
    # X[layer][feat] = list of 1d tensors; y = target ids
    bucket: dict[int, dict[str, list]] = {L: {f: [] for f in feat_names} for L in layers}
    y_all: list[int] = []

    for i, ex in enumerate(examples):
        ids = torch.tensor([ex.input_ids], dtype=torch.long, device=args.device)
        with torch.no_grad():
            with capture_mamba_states(model, layers=layers, keep_on_cpu=True) as cap:
                _ = model(input_ids=ids, use_cache=False)
        y_all.append(int(ex.target_id))
        for L in layers:
            tr = cap.layers[L]
            for fname in feat_names:
                bucket[L][fname].append(FEATURE_FNS[fname](tr, t=-1))
        if (i + 1) % 16 == 0:
            print(f"  captured {i+1}/{n}", flush=True)

    results: list[ProbeResult] = []
    for L in layers:
        for fname in feat_names:
            xs = bucket[L][fname]
            X = torch.stack(xs, dim=0)
            y = torch.tensor(y_all, dtype=torch.long)
            tr_i = [i for i in range(n) if i in train_idx]
            te_i = test_idx
            X_tr, y_tr = X[tr_i], y[tr_i]
            X_te, y_te = X[te_i], y[te_i]
            # standardize with train stats
            mu = X_tr.mean(0, keepdim=True)
            sd = X_tr.std(0, keepdim=True).clamp_min(1e-6)
            X_tr = (X_tr - mu) / sd
            X_te = (X_te - mu) / sd
            tr_acc, te_acc, n_cls = fit_linear_probe(
                X_tr, y_tr, X_te, y_te, steps=args.probe_steps, seed=args.seed
            )
            r = ProbeResult(
                feature=fname,
                layer=L,
                when="query_last",
                train_acc=tr_acc,
                test_acc=te_acc,
                n_train=len(tr_i),
                n_test=len(te_i),
                n_classes=n_cls,
                feat_dim=int(X.shape[1]),
            )
            results.append(r)
            print(
                f"L{L} {fname}: train={tr_acc:.3f} test={te_acc:.3f} dim={X.shape[1]} ncls={n_cls}",
                flush=True,
            )

    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    payload = {
        "model_id": args.model_id,
        "split": str(args.split),
        "n": n,
        "n_train": n_train,
        "n_test": len(test_idx),
        "layers": layers,
        "features": feat_names,
        "peak_vram_gb": round(peak, 3),
        "wall_s": round(time.time() - t0, 2),
        "results": [r.to_dict() for r in results],
        "note": (
            "Even with 0% behavioral AR accuracy, probes test whether target value "
            "is linearly decodable from h vs residual at the query token."
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Linear probes (C1) - AR value @ query token",
        "",
        f"model: `{args.model_id}`  split: `{args.split.name}`  n={n}",
        "",
        "| layer | feature | dim | train_acc | test_acc | n_classes |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for r in sorted(results, key=lambda z: (z.layer, z.feature)):
        lines.append(
            f"| {r.layer} | {r.feature} | {r.feat_dim} | {r.train_acc:.3f} | {r.test_acc:.3f} | {r.n_classes} |"
        )
    lines.append("")
    lines.append("Chance ≈ 1/n_classes (multi-class over seen train targets).")
    lines.append("")
    # highlight best state vs residual
    by_feat = {}
    for r in results:
        by_feat.setdefault(r.feature, []).append(r.test_acc)
    lines.append("## Best test acc by feature (any layer)")
    for f, accs in sorted(by_feat.items(), key=lambda kv: -max(kv[1])):
        lines.append(f"- **{f}**: {max(accs):.3f}")
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {args.out_json}", flush=True)
    print(f"wrote {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
