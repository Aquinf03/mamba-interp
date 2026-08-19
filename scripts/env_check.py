"""Print torch/CUDA identity. Safe to run before model load."""

from __future__ import annotations

import platform
import sys


def main() -> None:
    print("python", sys.version.replace("\n", " "))
    print("platform", platform.platform())
    try:
        import torch
    except ImportError as e:
        raise SystemExit(f"torch missing: {e}") from e

    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device_name", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print("vram_gb", round(props.total_memory / 1024**3, 2))
        print("cuda_runtime", torch.version.cuda)
    else:
        print("device_name", "NO CUDA")

    try:
        import transformers

        print("transformers", transformers.__version__)
    except ImportError:
        print("transformers", "MISSING")

    try:
        import mamba_ssm  # noqa: F401

        print("mamba_ssm", "importable")
        print(
            "WARN: partial mamba_ssm often breaks HF Mamba on Windows. "
            "If smoke_forward fails, run: pip uninstall mamba_ssm -y"
        )
    except Exception as e:
        print("mamba_ssm", f"not usable ({type(e).__name__}: {e}) - OK for slow path")

    try:
        from transformers import MambaForCausalLM  # noqa: F401

        print("MambaForCausalLM", "importable")
    except Exception as e:
        print("MambaForCausalLM", f"FAILED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
