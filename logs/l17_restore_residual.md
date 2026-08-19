# A1 L17 restore h vs residual

| name | layers | window | n | acc | ci95 | mean_lp | acc\|clean_ok | Δlp\|clean_ok | donor_acc |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| clean | 16-23 | none | 128 | 0.961 | 0.930–0.992 | -0.153 |  |  |  |
| zero_h_write_late | 16-23 | write | 128 | 0.242 | 0.172–0.320 | -4.857 | 0.228 | -4.973 |  |
| restore_L17_h_last_write | 16-23 | write | 128 | 0.961 | 0.929–0.992 | -0.150 | 0.992 | -0.004 |  |
| restore_L17_residual_last_write | 16-23 | write | 128 | 0.242 | 0.172–0.320 | -4.859 | 0.228 | -4.975 |  |
| restore_L17_residual_query | 16-23 | write | 128 | 0.258 | 0.188–0.336 | -4.618 | 0.244 | -4.725 |  |

Late h wipe on write, then copy **this example's** clean L17 vector at one site.
- restore h at last write: sufficiency (must ~clean).
- restore residual skip at last write: same site, residual is not recurrent (must stay near wipe).
- restore residual skip at query: fair readout control. If this recovers, residual at query is sufficient and C3 is only about the store during write.
