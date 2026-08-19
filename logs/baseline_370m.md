# Behavioral baseline (mamba-370m-hf, zero-shot)

Pretrained `state-spaces/mamba-370m-hf` (48 layers, d=1024, E=2048, N=16). Same frozen `data/splits/v1/`. No finetune.

| name | task | ablation | n | mean_len | acc | mean_lp(target) |
|---|---|---|---:|---:|---:|---:|
| ar | ar | none | 128 | 9.0 | 0.000 | -6.418 |
| atr_short | atr | none | 128 | 8.0 | 0.000 | -6.294 |
| atr_mid | atr | none | 128 | 19.0 | 0.000 | -7.018 |
| len_short | length | none | 64 | 5.0 | 0.000 | -5.702 |
| len_mid | length | none | 64 | 17.0 | 0.000 | -6.883 |
| len_ood_pad | length | none | 64 | 41.0 | 0.000 | -8.239 |
| stuff_light | length | none | 64 | 15.0 | 0.000 | -7.150 |
| stuff_heavy | length | none | 64 | 45.0 | 0.000 | -8.264 |
| factual | factual | none | 8 | 6.4 | 0.000 | -9.745 |

**All 0%.** Same floor as 130M pretrained. Scale does not unlock synthetic AR/ATR. D2 (layer sweep without FT) does not apply. Do not full-FT 370M on 8 GB.

## Notes

- model: `state-spaces/mamba-370m-hf`
- peak_vram_gb: 0.747 (load+eval); smoke fwd peak 1.084 GB @ seq 16
- wall_s: 81.5
- metric: teacher-forced next-token top-1 on frozen v1 splits

