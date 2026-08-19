# Causal interventions (ar)

| name | layers | window | n | acc | ci95 | mean_lp | acc\|clean_ok | Δlp\|clean_ok | donor_acc |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| clean | 16-23 | none | 128 | 0.961 | 0.930–0.992 | -0.153 |  |  |  |
| zero_h_write | 16-23 | write | 128 | 0.242 | 0.172–0.320 | -4.857 | 0.228 | -4.973 |  |
| zero_h_query | 16-23 | query | 128 | 0.000 | 0.000–0.000 | -50.481 | 0.000 | -50.409 |  |
| swap_h_write_end | 16-23 | write | 128 | 0.234 | 0.164–0.312 | -8.151 | 0.220 | -8.305 | 0.072 |
| zero_B_write | 16-23 | write | 128 | 0.242 | 0.172–0.320 | -4.857 | 0.228 | -4.973 |  |
| clamp_delta_write | 16-23 | write | 128 | 0.242 | 0.172–0.320 | -4.857 | 0.228 | -4.973 |  |
| zero_C_query | 16-23 | query | 128 | 0.000 | 0.000–0.000 | -50.481 | 0.000 | -50.409 |  |
| local_only | 16-23 | all | 128 | 0.000 | 0.000–0.000 | -50.681 | 0.000 | -50.596 |  |
| residual_noise_write | 16-23 | write | 128 | 0.961 | 0.930–0.992 | -0.156 | 1.000 | 0.000 |  |
| residual_noise_query | 16-23 | query | 128 | 0.883 | 0.828–0.938 | -0.895 | 0.902 | -0.817 |  |
| zero_h_write_early | 0-7 | write | 128 | 0.906 | 0.852–0.953 | -0.264 | 0.927 | -0.211 |  |
| zero_h_write_mid | 8-15 | write | 128 | 0.945 | 0.906–0.977 | -0.210 | 0.976 | -0.061 |  |
| zero_h_write_L0 | 0-0 | write | 128 | 0.938 | 0.891–0.977 | -0.147 | 0.959 | -0.090 |  |
| zero_h_write_L1 | 1-1 | write | 128 | 0.961 | 0.930–0.992 | -0.159 | 1.000 | -0.003 |  |
| zero_h_write_L2 | 2-2 | write | 128 | 0.969 | 0.938–0.992 | -0.152 | 1.000 | -0.001 |  |
| zero_h_write_L3 | 3-3 | write | 128 | 0.961 | 0.930–0.992 | -0.153 | 1.000 | -0.000 |  |
| zero_h_write_L4 | 4-4 | write | 128 | 0.961 | 0.930–0.992 | -0.152 | 1.000 | -0.000 |  |
| zero_h_write_L5 | 5-5 | write | 128 | 0.961 | 0.930–0.992 | -0.153 | 1.000 | 0.001 |  |
| zero_h_write_L6 | 6-6 | write | 128 | 0.961 | 0.930–0.992 | -0.155 | 1.000 | 0.000 |  |
| zero_h_write_L7 | 7-7 | write | 128 | 0.961 | 0.930–0.992 | -0.142 | 1.000 | -0.003 |  |
| zero_h_write_L8 | 8-8 | write | 128 | 0.961 | 0.930–0.992 | -0.158 | 1.000 | -0.003 |  |
| zero_h_write_L9 | 9-9 | write | 128 | 0.961 | 0.930–0.992 | -0.157 | 1.000 | -0.001 |  |
| zero_h_write_L10 | 10-10 | write | 128 | 0.961 | 0.930–0.992 | -0.153 | 1.000 | 0.000 |  |
| zero_h_write_L11 | 11-11 | write | 128 | 0.961 | 0.930–0.992 | -0.156 | 1.000 | -0.001 |  |
| zero_h_write_L12 | 12-12 | write | 128 | 0.969 | 0.938–0.992 | -0.154 | 1.000 | -0.004 |  |
| zero_h_write_L13 | 13-13 | write | 128 | 0.961 | 0.930–0.992 | -0.160 | 1.000 | -0.000 |  |
| zero_h_write_L14 | 14-14 | write | 128 | 0.961 | 0.930–0.992 | -0.163 | 0.992 | -0.003 |  |
| zero_h_write_L15 | 15-15 | write | 128 | 0.953 | 0.914–0.984 | -0.160 | 0.992 | -0.009 |  |
| zero_h_write_L16 | 16-16 | write | 128 | 0.969 | 0.938–0.992 | -0.146 | 1.000 | -0.001 |  |
| zero_h_write_L17 | 17-17 | write | 128 | 0.297 | 0.219–0.383 | -4.056 | 0.285 | -4.155 |  |
| zero_h_write_L18 | 18-18 | write | 128 | 0.953 | 0.914–0.984 | -0.170 | 0.992 | -0.008 |  |
| zero_h_write_L19 | 19-19 | write | 128 | 0.961 | 0.930–0.992 | -0.151 | 1.000 | -0.000 |  |
| zero_h_write_L20 | 20-20 | write | 128 | 0.953 | 0.914–0.984 | -0.147 | 0.992 | -0.002 |  |
| zero_h_write_L21 | 21-21 | write | 128 | 0.961 | 0.930–0.992 | -0.149 | 1.000 | 0.001 |  |
| zero_h_write_L22 | 22-22 | write | 128 | 0.961 | 0.930–0.992 | -0.154 | 1.000 | -0.000 |  |
| zero_h_write_L23 | 23-23 | write | 128 | 0.961 | 0.930–0.992 | -0.153 | 1.000 | 0.000 |  |
## Mixer L2 budgets (zero_h_write vs clean)

{
  "zero_h_write_mixer_L2_write": {
    "mean": 29.646541595458984,
    "median": 5.840880393981934,
    "p90": 85.49989318847656,
    "n": 8192
  },
  "zero_h_write_mixer_L2_query": {
    "mean": 54.42639923095703,
    "median": 25.30205535888672,
    "p90": 133.6435089111328,
    "n": 1024
  }
}

## Notes

- model: `runs\ar_ft\checkpoint`
- split: `C:\Users\fresh\OneDrive\Desktop\mamba\data\splits\v1\ar_eval.jsonl`
- peak_vram_gb: 0.255
- wall_s: 194.3
- residual_noise_write / _query: random residual direction, L2 matched to zero_h_write mixer delta on that window
- swap donor_acc: P(pred = next example's value), excluding target collisions
- ci95: bootstrap over examples
