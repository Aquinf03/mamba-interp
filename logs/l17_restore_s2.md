# B1 seed-2 L17 restore h vs residual

| name | layers | window | n | acc | ci95 | mean_lp | acc\|clean_ok | Δlp\|clean_ok | donor_acc |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| clean | 16-23 | none | 128 | 0.961 | 0.922–0.992 | -0.120 |  |  |  |
| zero_h_write_late | 16-23 | write | 128 | 0.250 | 0.180–0.328 | -4.916 | 0.260 | -4.772 |  |
| restore_L17_h_last_write | 16-23 | write | 128 | 0.969 | 0.938–0.992 | -0.128 | 1.000 | -0.008 |  |
| restore_L17_residual_last_write | 16-23 | write | 128 | 0.250 | 0.180–0.328 | -4.914 | 0.260 | -4.769 |  |
| restore_L17_residual_query | 16-23 | write | 128 | 0.258 | 0.180–0.336 | -4.736 | 0.268 | -4.599 |  |

model: `C:\Users\fresh\OneDrive\Desktop\mamba\runs\ar_ft_s2\checkpoint`  split: AR n=128
Must match seed 1: h restore ~clean, residual restore ~wipe.
