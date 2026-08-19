# Causal interventions (C2, C3)

| name | layers | window | n | acc | mean_lp | acc\|clean_ok | Δlp\|clean_ok |
|---|---|---|---:|---:|---:|---:|---:|
| clean | 16-23 | none | 128 | 0.961 | -0.153 |  |  |
| zero_h_write | 16-23 | write | 128 | 0.242 | -4.857 | 0.228 | -4.973 |
| zero_h_query | 16-23 | query | 128 | 0.000 | -50.481 | 0.000 | -50.409 |
| swap_h_write_end | 16-23 | write | 128 | 0.234 | -8.151 | 0.220 | -8.305 |
| zero_B_write | 16-23 | write | 128 | 0.242 | -4.857 | 0.228 | -4.973 |
| clamp_delta_write | 16-23 | write | 128 | 0.242 | -4.857 | 0.228 | -4.973 |
| zero_C_query | 16-23 | query | 128 | 0.000 | -50.481 | 0.000 | -50.409 |
| local_only | 16-23 | all | 128 | 0.000 | -50.681 | 0.000 | -50.596 |
| residual_noise_matched | 16-23 | write | 128 | 0.961 | -0.156 | 1.000 | 0.000 |
| zero_h_write_early | 0-7 | write | 128 | 0.906 | -0.264 | 0.927 | -0.211 |
| zero_h_write_mid | 8-15 | write | 128 | 0.945 | -0.210 | 0.976 | -0.061 |
## Notes

- model: `runs\ar_ft\checkpoint`
- split: `C:\Users\fresh\OneDrive\Desktop\mamba\data\splits\v1\ar_eval.jsonl`
- peak_vram_gb: 0.255
- wall_s: 70.0
- clean_ok = examples the unpatched model already got right
- residual_noise_matched: random residual direction, per-token L2 matched to zero_h_write mixer delta
- swap_h_write_end: replace h at last write step with the next example's h
- local_only: SSM recurrence off; D·u and conv remain (C5 start)

## Read for C2/C3

If zero_h_write / swap_h drop accuracy a lot and residual_noise_matched does not, state (not residual geometry) carried the bind.
If local_only stays near clean, conv/D skip did the task, not h.
