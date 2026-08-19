# Causal interventions (C2, C3)

| name | layers | window | n | acc | mean_lp | acc\|clean_ok | Δlp\|clean_ok |
|---|---|---|---:|---:|---:|---:|---:|
| clean | 16-23 | none | 16 | 0.938 | -0.423 |  |  |
| zero_h_write | 16-23 | write | 16 | 0.500 | -2.889 | 0.467 | -3.044 |
| zero_h_query | 16-23 | query | 16 | 0.000 | -50.634 | 0.000 | -50.336 |
| swap_h_write_end | 16-23 | write | 16 | 0.500 | -5.896 | 0.467 | -6.236 |
| zero_B_write | 16-23 | write | 16 | 0.500 | -2.889 | 0.467 | -3.044 |
| clamp_delta_write | 16-23 | write | 16 | 0.500 | -2.889 | 0.467 | -3.044 |
| zero_C_query | 16-23 | query | 16 | 0.000 | -50.634 | 0.000 | -50.336 |
| local_only | 16-23 | all | 16 | 0.000 | -50.864 | 0.000 | -50.602 |
| residual_noise_matched | 16-23 | write | 16 | 0.938 | -0.413 | 1.000 | 0.001 |
| zero_h_write_early | 0-7 | write | 16 | 0.812 | -0.322 | 0.867 | -0.217 |
| zero_h_write_mid | 8-15 | write | 16 | 0.938 | -0.209 | 1.000 | 0.011 |
## Notes

- model: `runs\ar_ft\checkpoint`
- split: `C:\Users\fresh\OneDrive\Desktop\mamba\data\splits\v1\ar_eval.jsonl`
- peak_vram_gb: 0.255
- wall_s: 9.3
- clean_ok = examples the unpatched model already got right
- residual_noise_matched: random residual direction, per-token L2 matched to zero_h_write mixer delta
- swap_h_write_end: replace h at last write step with the next example's h
- local_only: SSM recurrence off; D·u and conv remain (C5 start)

## Read for C2/C3

If zero_h_write / swap_h drop accuracy a lot and residual_noise_matched does not, state (not residual geometry) carried the bind.
If local_only stays near clean, conv/D skip did the task, not h.
