# B2 seed-2 single-layer write wipe L16/L17/L18

| name | layers | window | n | acc | ci95 | mean_lp | acc\|clean_ok | Δlp\|clean_ok | donor_acc |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| clean | 16-23 | none | 128 | 0.961 | 0.922–0.992 | -0.120 |  |  |  |
| zero_h_write_L16 | 16-16 | write | 128 | 0.969 | 0.938–0.992 | -0.126 | 1.000 | -0.002 |  |
| zero_h_write_L17 | 17-17 | write | 128 | 0.289 | 0.211–0.367 | -4.318 | 0.301 | -4.198 |  |
| zero_h_write_L18 | 18-18 | write | 128 | 0.961 | 0.922–0.992 | -0.125 | 1.000 | -0.001 |  |

model: `C:\Users\fresh\OneDrive\Desktop\mamba\runs\ar_ft_s2\checkpoint`
L17 must collapse; L16 and L18 must stay near clean (seed-1 pattern).
