# L17 restore (sufficiency)

| name | layers | window | n | acc | ci95 | mean_lp | acc\|clean_ok | Δlp\|clean_ok | donor_acc |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| clean | 16-23 | none | 128 | 0.961 | 0.930–0.992 | -0.153 |  |  |  |
| zero_h_write_late | 16-23 | write | 128 | 0.242 | 0.172–0.320 | -4.857 | 0.228 | -4.973 |  |
| restore_L17_after_late_zero | 16-23 | write | 128 | 0.961 | 0.929–0.992 | -0.150 | 0.992 | -0.004 |  |
| zero_h_write_L17 | 17-17 | write | 128 | 0.297 | 0.219–0.383 | -4.056 | 0.285 | -4.155 |  |

Restore: zero h on late layers during write, then copy **this example's** clean L17 state in at last write token. Acc back toward clean => L17 h is sufficient.
