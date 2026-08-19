"""C1-C5: pad vs filler interventions (does not overwrite failure.md)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from transformers import MambaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.baseline import evaluate_split
from src.eval.io import load_jsonl
from src.hooks import capture_mamba_states
from src.intervene.run import run_condition, summarize_condition
from src.intervene.spec import Intervention
from src.intervene.windows import layer_group, token_roles


SPLITS = [
    ("len_short", "length_len_short_eval.jsonl"),
    ("len_mid", "length_len_mid_eval.jsonl"),
    ("len_ood_pad", "length_len_ood_pad_eval.jsonl"),
    ("stuff_light", "length_stuff_light_eval.jsonl"),
    ("stuff_heavy", "length_stuff_heavy_eval.jsonl"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default=str(ROOT / "runs" / "ar_ft" / "checkpoint"))
    p.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits" / "v1")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-examples", type=int, default=0)
    p.add_argument("--fix-group", default="late")
    p.add_argument("--l17", type=int, default=17)
    p.add_argument("--large-delta", type=float, default=5.0)
    p.add_argument("--out-dir", type=Path, default=ROOT / "logs")
    return p.parse_args()


def specs(late: list[int], large: float) -> list[Intervention]:
    return [
        Intervention("clamp_delta_pad", "delta", "clamp", "pad", late, clamp=0.0),
        Intervention("clamp_delta_filler", "delta", "clamp", "filler", late, clamp=0.0),
        Intervention("zero_B_filler", "B", "zero", "filler", late),
        Intervention("clamp_delta_filler_large", "delta", "clamp", "filler", late, clamp=large),
    ]


def go(model, exs, spec, device, layers, seed):
    print(f"  {spec.name} ...", flush=True)
    scores, _, _ = run_condition(
        model, exs, spec, device=device, layers=layers, capture=False, seed=seed
    )
    row = summarize_condition(spec, scores, None, layers=layers)
    print(f"    acc={row.acc:.3f}", flush=True)
    return row


def _pct(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    t = torch.tensor(xs)
    return float(t.quantile(q))


def role_delta_stats(model, examples, *, layer: int, device: str) -> Dict[str, dict]:
    buckets: Dict[str, List[float]] = defaultdict(list)
    for i, ex in enumerate(examples):
        ids = torch.tensor([ex.input_ids], dtype=torch.long, device=device)
        with capture_mamba_states(model, layers=[layer], keep_on_cpu=True) as cap:
            model(input_ids=ids, use_cache=False)
        d = cap.layers[layer].delta[0].float().mean(dim=-1)  # [T]
        roles = token_roles(ex)
        t = min(len(roles), int(d.numel()))
        for j in range(t):
            buckets[roles[j]].append(float(d[j].item()))
        if (i + 1) % 16 == 0:
            print(f"    delta {i + 1}/{len(examples)}", flush=True)
    out = {}
    for role, xs in sorted(buckets.items()):
        out[role] = {
            "n": len(xs),
            "mean": sum(xs) / len(xs) if xs else float("nan"),
            "median": _pct(xs, 0.5),
            "p10": _pct(xs, 0.1),
            "p90": _pct(xs, 0.9),
        }
    return out


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    dtype = getattr(torch, args.dtype)
    print(f"loading {args.model_id} ...", flush=True)
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = MambaForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    model = model.to(args.device)
    model.eval()
    late = layer_group(len(model.backbone.layers), args.fix_group)
    ops = specs(late, args.large_delta)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    table_rows = []
    payload_splits = []
    md = [
        "# C4 pad vs filler ops",
        "",
        f"model: `{args.model_id}`",
        f"fix layers: {late[0]}-{late[-1]}  large_delta={args.large_delta}",
        "",
        "| split | n | clean | clamp Δ pad=0 | clamp Δ filler=0 | zero B filler | Δ filler="
        + f"{args.large_delta:g} |",
        "|---|---:|---:|---:|---:|---:|---:|",
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
        accs = {"clean": clean.accuracy}
        recs = {}
        for spec in ops:
            row = go(model, exs, spec, args.device, spec.layers, 0)
            accs[spec.name] = row.acc
            recs[spec.name] = {
                "acc": row.acc,
                "mean_lp": row.mean_lp,
                "recovery": row.acc - clean.accuracy,
            }
        payload_splits.append(
            {
                "name": name,
                "n": len(exs),
                "clean_acc": clean.accuracy,
                "ops": recs,
            }
        )
        md.append(
            f"| {name} | {len(exs)} | {clean.accuracy:.3f} | "
            f"{accs['clamp_delta_pad']:.3f} | {accs['clamp_delta_filler']:.3f} | "
            f"{accs['zero_B_filler']:.3f} | {accs['clamp_delta_filler_large']:.3f} |"
        )
        table_rows.append(accs)

    pad_md = md[:]
    pad_md.append("")
    pad_md.append(
        "C1: clamp Δ←0 on **pad** only. Expect help on len_ood_pad; no-op elsewhere.\n"
        "C2: clamp Δ←0 on **filler** only. Expect hurt on stuffing (fillers are used).\n"
        "C3: zero B on fillers (block writes from stuffing tokens).\n"
        "C4: large Δ on fillers (force forget). Opposite of C2.\n"
        "Empty windows are no-ops (acc should match clean)."
    )
    (out / "failure_pad_only.md").write_text("\n".join(pad_md) + "\n", encoding="utf-8")
    (out / "failure_filler_ops.md").write_text("\n".join(pad_md) + "\n", encoding="utf-8")

    # C5: L17 Δ by token role on the two failure families
    c5_names = ("len_ood_pad", "stuff_light", "stuff_heavy")
    c5 = {}
    md5 = [
        "# C5 L17 mean Δ by token role",
        "",
        f"layer {args.l17}. Per-token mean over E, pooled across examples.",
        "",
        "| split | role | n | mean | median | p10 | p90 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, fname in SPLITS:
        if name not in c5_names:
            continue
        path = args.splits_dir / fname
        exs = load_jsonl(path)
        if args.max_examples > 0:
            exs = exs[: args.max_examples]
        print(f"== C5 delta roles {name} ==", flush=True)
        stats = role_delta_stats(model, exs, layer=args.l17, device=args.device)
        c5[name] = stats
        for role, st in stats.items():
            md5.append(
                f"| {name} | {role} | {st['n']} | {st['mean']:.4f} | "
                f"{st['median']:.4f} | {st['p10']:.4f} | {st['p90']:.4f} |"
            )
    md5.append("")
    md5.append("If pad Δ looks like value Δ, pad is writing. Rank already failed; this is the Δ split.")
    (out / "failure_delta_roles.md").write_text("\n".join(md5) + "\n", encoding="utf-8")

    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    payload = {
        "model_id": args.model_id,
        "large_delta": args.large_delta,
        "fix_layers": late,
        "peak_vram_gb": round(peak, 3),
        "wall_s": round(time.time() - t0, 2),
        "splits": payload_splits,
        "delta_roles_L17": c5,
    }
    (out / "failure_split.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    text = "\n".join(pad_md + [""] + md5)
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))
    print(f"peak_vram_gb={peak:.3f} wall_s={payload['wall_s']}", flush=True)
    print("wrote logs/failure_pad_only.md failure_filler_ops.md failure_delta_roles.md", flush=True)


if __name__ == "__main__":
    main()
