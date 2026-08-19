"""Smoke test: load the model and run a dummy forward pass to measure peak VRAM."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, MambaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test: Mamba HF forward on CUDA")
    p.add_argument("--model-id", default="state-spaces/mamba-130m-hf")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log", type=Path, default=ROOT / "logs" / "smoke_forward.txt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA not available. Check torch install and nvidia-smi.")

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    print(f"loading {args.model_id} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = MambaForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    model = model.to(args.device)
    model.eval()

    vocab = model.config.vocab_size
    ids = torch.randint(0, vocab, (1, args.seq_len), device=args.device)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    logits = out.logits

    peak_gb = (
        torch.cuda.max_memory_allocated() / 1024**3
        if torch.cuda.is_available()
        else 0.0
    )
    wall_s = time.time() - t0
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    lines = [
        f"ok {args.model_id}",
        f"logits {tuple(logits.shape)}",
        f"dtype {args.dtype}",
        f"seq_len {args.seq_len}",
        f"peak_vram_gb {peak_gb:.3f}",
        f"wall_s {wall_s:.2f}",
        f"device {device_name}",
        f"torch {torch.__version__}",
        f"tokenizer {type(tok).__name__}",
    ]
    text = "\n".join(lines) + "\n"
    print(text, end="")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(text, encoding="utf-8")
    print(f"wrote {args.log}", flush=True)

    if peak_gb > 7.0:
        print("WARNING: peak VRAM > 7 GB; shrink seq_len or dtype before longer work.")


if __name__ == "__main__":
    main()
