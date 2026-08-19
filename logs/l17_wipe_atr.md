# B3 ATR-short L17-only write wipe

| name | layers | window | n | acc | ci95 | mean_lp | acc\|clean_ok | Δlp\|clean_ok | donor_acc |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| clean | 16-23 | none | 128 | 0.961 | 0.922–0.992 | -0.268 |  |  |  |
| zero_h_write_late | 16-23 | write | 128 | 0.039 | 0.008–0.078 | -6.384 | 0.041 | -6.197 |  |
| zero_h_write_L16 | 16-16 | write | 128 | 0.961 | 0.922–0.992 | -0.279 | 1.000 | -0.005 |  |
| zero_h_write_L17 | 17-17 | write | 128 | 0.133 | 0.078–0.188 | -5.297 | 0.138 | -5.094 |  |
| zero_h_write_L18 | 18-18 | write | 128 | 0.961 | 0.922–0.992 | -0.268 | 1.000 | -0.002 |  |

model: `C:\Users\fresh\OneDrive\Desktop\mamba\runs\ar_ft\checkpoint`  split: `C:\Users\fresh\OneDrive\Desktop\mamba\data\splits\v1\atr_short_eval.jsonl`
Late-block wipe already 0.039. If L17-only also collapses ATR, the store is the same layer as AR.
