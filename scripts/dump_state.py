"""Phase 2: dump ||h_t|| and mean Δ over time for selected layers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, MambaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hooks import capture_mamba_states, delta_mean_by_time, h_norm_by_time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="state-spaces/mamba-130m-hf")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--layers", default="0,11,23", help="comma layer idxs or 'all'")
    p.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "state_dump")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(f"loading {args.model_id} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = MambaForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    model = model.to(args.device)
    model.eval()

    n_layers = len(model.backbone.layers)
    if args.layers.strip().lower() == "all":
        layer_list = list(range(n_layers))
    else:
        layer_list = [int(x) for x in args.layers.split(",") if x.strip() != ""]
        bad = [i for i in layer_list if i < 0 or i >= n_layers]
        if bad:
            raise SystemExit(f"layers out of range 0..{n_layers-1}: {bad}")

    text = "The capital of France is Paris and the capital of Germany is Berlin."
    enc = tok(text, return_tensors="pt")
    ids = enc["input_ids"].to(args.device)
    if ids.shape[1] < args.seq_len:
        pad = torch.randint(
            0, model.config.vocab_size, (1, args.seq_len - ids.shape[1]), device=args.device
        )
        ids = torch.cat([ids, pad], dim=1)
    else:
        ids = ids[:, : args.seq_len]

    with torch.no_grad():
        with capture_mamba_states(model, layers=layer_list, keep_on_cpu=True) as cap:
            # use_cache=False → no incremental cache; full-sequence prefill only.
            out = model(input_ids=ids, use_cache=False)

    peak_gb = (
        torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "model_id": args.model_id,
        "seq_len": int(ids.shape[1]),
        "layers": layer_list,
        "n_layers": n_layers,
        "logits_shape": list(out.logits.shape),
        "peak_vram_gb": round(peak_gb, 3),
        "wall_s": round(time.time() - t0, 2),
        "summary": cap.summary().splitlines(),
        "tokenizer": type(tok).__name__,
    }

    series = {}
    for idx, tr in cap.layers.items():
        assert tr.h is not None and tr.delta is not None
        hn = h_norm_by_time(tr.h)
        dm = delta_mean_by_time(tr.delta)
        series[str(idx)] = {
            "h_norm": hn.tolist(),
            "delta_mean": dm.tolist(),
            "h_shape": list(tr.h.shape),
            "delta_shape": list(tr.delta.shape),
            "B_shape": list(tr.B.shape) if tr.B is not None else None,
            "C_shape": list(tr.C.shape) if tr.C is not None else None,
        }
        torch.save(
            {
                "h": tr.h,
                "delta": tr.delta,
                "B": tr.B,
                "C": tr.C,
                "conv_out": tr.conv_out,
                "residual": tr.residual,
                "mixer_out": tr.mixer_out,
                "residual_out": tr.residual_out,
                "gate": tr.gate,
            },
            args.out_dir / f"layer_{idx:02d}.pt",
        )

    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (args.out_dir / "series.json").write_text(json.dumps(series, indent=2), encoding="utf-8")

    print(cap.summary())
    print(json.dumps(meta, indent=2))
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
