# A2 L17 h restore over time

| name | layers | window | n | acc | ci95 | mean_lp | acc\|clean_ok | Δlp\|clean_ok | donor_acc |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| clean | 16-23 | none | 128 | 0.961 | 0.930–0.992 | -0.153 |  |  |  |
| zero_h_write_late | 16-23 | write | 128 | 0.242 | 0.172–0.320 | -4.857 | 0.228 | -4.973 |  |
| restore_L17_h_value | 16-23 | write | 128 | 0.969 | 0.938–0.992 | -0.148 | 1.000 | -0.004 |  |
| restore_L17_h_last_write | 16-23 | write | 128 | 0.961 | 0.929–0.992 | -0.150 | 0.992 | -0.004 |  |
| restore_L17_h_query | 16-23 | write | 128 | 0.953 | 0.914–0.984 | -0.154 | 0.992 | -0.004 |  |

After late h wipe, paste this example's clean L17 h at one token. Tokens before the site stay wiped; after the site, recurrence runs. Value vs last-write vs query localizes *when* the bind is in L17 h.
