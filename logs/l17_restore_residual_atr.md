# A1 L17 restore h vs residual (atr)

| name | layers | window | n | acc | ci95 | mean_lp | acc\|clean_ok | Δlp\|clean_ok | donor_acc |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| clean | 16-23 | none | 128 | 0.961 | 0.922–0.992 | -0.268 |  |  |  |
| zero_h_write_late | 16-23 | write | 128 | 0.039 | 0.008–0.078 | -6.384 | 0.041 | -6.197 |  |
| restore_L17_h_last_write | 16-23 | write | 128 | 0.945 | 0.898–0.984 | -0.300 | 0.984 | -0.022 |  |
| restore_L17_residual_last_write | 16-23 | write | 128 | 0.039 | 0.008–0.078 | -6.385 | 0.041 | -6.197 |  |
| restore_L17_residual_query | 16-23 | write | 128 | 0.062 | 0.023–0.102 | -6.082 | 0.065 | -5.895 |  |

split: `data\splits\v1\atr_short_eval.jsonl`

Late h wipe on write, then copy **this example's** clean L17 vector at one site.
- restore h at last write: sufficiency (must ~clean).
- restore residual skip at last write: same site, residual is not recurrent (must stay near wipe).
- restore residual skip at query: fair readout control. If this recovers, residual at query is sufficient and C3 is only about the store during write.
