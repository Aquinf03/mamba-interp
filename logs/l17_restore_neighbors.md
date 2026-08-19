# A3 restore L16 / L17 / L18 h

| name | layers | window | n | acc | ci95 | mean_lp | acc\|clean_ok | Δlp\|clean_ok | donor_acc |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| clean | 16-23 | none | 128 | 0.961 | 0.930–0.992 | -0.153 |  |  |  |
| zero_h_write_late | 16-23 | write | 128 | 0.242 | 0.172–0.320 | -4.857 | 0.228 | -4.973 |  |
| restore_L16_h_last_write | 16-23 | write | 128 | 0.258 | 0.188–0.336 | -4.666 | 0.244 | -4.778 |  |
| restore_L17_h_last_write | 16-23 | write | 128 | 0.961 | 0.929–0.992 | -0.150 | 0.992 | -0.004 |  |
| restore_L18_h_last_write | 16-23 | write | 128 | 0.242 | 0.172–0.320 | -4.914 | 0.228 | -5.033 |  |

Same late wipe + last-write restore, but paste neighbor-layer h instead of L17. Neighbors must not recover if sufficiency is L17-specific.
