# Model A (frozen)

| Field | Value |
|---|---|
| id | `state-spaces/mamba-130m-hf` |
| role | Primary pure Mamba, all paper claims |
| expected VRAM (fp16, seq 128, batch 1) | **0.290 GB peak** (2026-08-08, Windows 4070) |
| stack | HuggingFace `MambaForCausalLM`, sequential / slow path (`use_mambapy=False`, no CUDA kernels) |
| wall smoke | 64.5 s first load (includes HF download); subsequent loads much faster |
| alternate | only if A hooks fail - same size class |

## Load command

```powershell
cd <repo>
python scripts\env_check.py
python scripts\smoke_forward.py
```

Do **not** install broken `mamba_ssm` CUDA packages on Windows - they break `import MambaForCausalLM` via a partial install / `vendor\mamba` shadow. Keep `vendor/` out of the path (gitignored).

## Smoke result

```
ok state-spaces/mamba-130m-hf
logits (1, 128, 50280)
dtype float16
seq_len 128
peak_vram_gb 0.290
wall_s 64.52
device NVIDIA GeForce RTX 4070 Laptop GPU
torch 2.7.1+cu128
```

## State dump (Phase 2)

```
python scripts\dump_state.py
peak_vram_gb 0.281  wall_s 3.48  seq_len 64
h: (1, 64, 1536, 16)   # E×N recurrent state per token
delta: (1, 64, 1536)
B, C: (1, 64, 16)
residual: (1, 64, 768)
layers: 0, 11, 23 (of 24)
```
