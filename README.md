# Reading the State

When AR-finetuned Mamba-130M does associative recall, the bind lives in **layer-17 recurrent state \(h\)**. Residual-matched noise misses it. Query residual probes are readout, not store.

This repo is the workshop paper, frozen eval splits, intervention hooks, and locked experiment receipts.

- Draft: [`paper/draft.md`](paper/draft.md)
- Numbers: [`logs/findings.md`](logs/findings.md)
- Figures: [`figs/`](figs/) (`python scripts/make_figs.py`)

Pretrained 130M and 370M score 0% on these splits. The store is a **task-finetune algorithm**, not a pretrained circuit. Venue default: workshop.

## Results

Paper checkpoint (`runs/ar_ft/checkpoint`, seed 1): AR accuracy **0.961** on frozen `data/splits/v1/` (n=128).

| Condition | AR acc |
|---|---:|
| Clean | 0.961 |
| Late \(h\) wipe on write (layers 16-23) | 0.242 |
| Residual-matched write noise | 0.961 |
| Restore this example's L17 \(h\) | 0.961 |
| Restore this example's L17 residual | 0.242 |
| L17-only write wipe | 0.297 |

L16 and L18 wipe/restore stay near clean or near wipe, respectively. Three finetune seeds give L17 wipe 0.297 / 0.289 / 0.320.

Padding overwrites the store (a \(\Delta \leftarrow 0\) clamp on pads helps in **sign** across seeds). Stuffing is a different failure (the same clamp hurts). Swap of \(h\) breaks recall; it does **not** copy the donor value (`donor_acc` 0.072).

Cite n=128 (AR/ATR) and n=64 (length). Do not cite `--max-examples 16` or `scripts/run_demo.py`.

## Setup

Needs CUDA, PyTorch, and HuggingFace `MambaForCausalLM`. Official fused `mamba-ssm` is **not** required. Claims use the sequential (slow) scan so \(h\), \(B\), \(C\), and \(\Delta\) can be hooked.

```bash
pip install -r requirements.txt
pip uninstall mamba_ssm -y
python scripts/env_check.py
python scripts/smoke_forward.py
```

On Windows PowerShell, set `$env:PYTHONIOENCODING = 'utf-8'` before scripts that print \(\Delta\). Ignore Triton `CUDA_HOME`, TensorFlow oneDNN, and "fast path is not available" warnings.

Smoke should print logits `(1, 128, 50280)` at about 0.3 GB peak.

### Hardware (measured)

| Job | Peak VRAM | Wall (RTX 4070 8 GB, sequential HF scan) |
|---|---:|---|
| AR finetune (800 steps, 2048 train) | 2.66 GB | about 5 min |
| Hooked eval n=128 | 0.26-0.33 GB | minutes |
| 370M zero-shot | about 1.1 GB | fits 8 GB |
| Demo n=16 | about 0.3 GB | under 10 min |

8 GB is enough. Wall time is the sequential scan, not VRAM.

## Paper checkpoint

Weights are **not** in git (`runs/` and `*.safetensors` are ignored). Re-finetune seed 1:

```bash
python scripts/finetune_ar.py --seed 1 --out-dir runs/ar_ft
```

Receipt: `runs/ar_ft/train_meta.json` (eval AR 0.961). Recipe: 800 steps, 2048 train examples, lr \(3\times10^{-4}\).

`finetune_ar.py` refuses to overwrite an existing `checkpoint/model.safetensors` unless you pass `--force`. Other seeds:

```bash
python scripts/finetune_ar.py --seed 2 --out-dir runs/ar_ft_s2
```

Eval splits are frozen in `data/splits/v1/` (SHA256 in `data/splits/v1/SHA256SUMS.txt` and `logs/LOCK.md`). Do not regenerate v1 in place. Multi-query AR is `data/splits/v2/` only.

## Demo (not paper n)

Clean vs late-\(h\) wipe vs restore L17 \(h\) on **n=16**. Do not cite these accuracies.

```bash
python scripts/run_demo.py --model-id runs/ar_ft/checkpoint
```

Writes `logs/demo.md`. Paper restore table: `logs/l17_restore.md` (n=128).

## Figures

```bash
python scripts/make_figs.py
```

Needs matplotlib. Numbers are hardcoded from `logs/findings.md`. This does not re-run the model.

## Reproduce paper tables

Point these at the paper checkpoint only. Do not write a different run into the original `logs/*.md` names.

```bash
python scripts/finetune_ar.py --seed 1 --out-dir runs/ar_ft
python scripts/run_baseline.py --model-id runs/ar_ft/checkpoint --out-json logs/baseline_ft.json --out-md logs/baseline_ft.md
python scripts/run_strong.py --model-id runs/ar_ft/checkpoint
python scripts/run_l17_ctrl.py --model-id runs/ar_ft/checkpoint
python scripts/run_e_content.py --model-id runs/ar_ft/checkpoint
python scripts/make_figs.py
python scripts/freeze_release.py
```

New experiment: new log filename. Never overwrite `logs/intervene_ar.md`, `failure.md`, `l17_restore.md`, `l17_channels.md`, `l17_keypatch.md`, `baseline.md`, or `probes.md`.

## Layout

| Path | Purpose |
|---|---|
| `paper/draft.md` | Workshop draft |
| `paper/RELEASE.md` | Tag and what is not in git |
| `figs/` | Five paper figures |
| `logs/findings.md` | Durable numbers |
| `logs/LOCK.md` | SHA256 of locked logs and splits |
| `data/splits/v1/` | Frozen eval |
| `runs/ar_ft/checkpoint` | Paper weights (local only) |
| `src/intervene/` | Hooks on \(h\), \(B\), \(C\), \(\Delta\), residual |
| `src/probes/` | Linear and MLP probes |
| `scripts/` | Entrypoints |

## Limitations

130M only. Synthetic AR/ATR. Task-finetuned (pretrained 130M and 370M are both 0% on these strings). Pad recovery magnitude is seed-sensitive. Swap does not copy. No portable key-value slot in \(N\) or \(C\). Sequential HuggingFace scan (no fused kernels).
