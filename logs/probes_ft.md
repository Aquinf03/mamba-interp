# Linear probes (C1) - AR value @ query token

model: `runs\ar_ft\checkpoint`  split: `ar_eval.jsonl`  n=128

| layer | feature | dim | train_acc | test_acc | n_classes |
|---:|---|---:|---:|---:|---:|
| 0 | delta | 1536 | 0.740 | 0.071 | 40 |
| 0 | h_flat | 24576 | 0.969 | 0.036 | 40 |
| 0 | h_mean_e | 16 | 0.781 | 0.000 | 40 |
| 0 | h_mean_n | 1536 | 0.990 | 0.071 | 40 |
| 0 | residual | 768 | 0.458 | 0.036 | 40 |
| 0 | residual_out | 768 | 0.938 | 0.036 | 40 |
| 11 | delta | 1536 | 0.771 | 0.000 | 40 |
| 11 | h_flat | 24576 | 0.979 | 0.036 | 40 |
| 11 | h_mean_e | 16 | 0.729 | 0.000 | 40 |
| 11 | h_mean_n | 1536 | 0.938 | 0.071 | 40 |
| 11 | residual | 768 | 0.938 | 0.036 | 40 |
| 11 | residual_out | 768 | 0.854 | 0.071 | 40 |
| 23 | delta | 1536 | 0.917 | 0.429 | 40 |
| 23 | h_flat | 24576 | 0.969 | 0.500 | 40 |
| 23 | h_mean_e | 16 | 0.646 | 0.143 | 40 |
| 23 | h_mean_n | 1536 | 0.969 | 0.607 | 40 |
| 23 | residual | 768 | 1.000 | 0.929 | 40 |
| 23 | residual_out | 768 | 1.000 | 0.893 | 40 |

Chance ≈ 1/n_classes (multi-class over seen train targets).

## Best test acc by feature (any layer)
- **residual**: 0.929
- **residual_out**: 0.893
- **h_mean_n**: 0.607
- **h_flat**: 0.500
- **delta**: 0.429
- **h_mean_e**: 0.143
