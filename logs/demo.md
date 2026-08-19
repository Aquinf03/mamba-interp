# Demo (n=16) - not paper numbers

model: `runs\ar_ft\checkpoint`  split: first 16 of `ar_eval.jsonl`

**Do not cite.** Paper restore table: `logs/l17_restore.md` (n=128).

# demo

| name | layers | window | n | acc | ci95 | mean_lp | acc\|clean_ok | Δlp\|clean_ok | donor_acc |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| clean | 16-23 | none | 16 | 0.938 | 0.812–1.000 | -0.423 |  |  |  |
| zero_h_write_late | 16-23 | write | 16 | 0.500 | 0.250–0.750 | -2.889 | 0.467 | -3.044 |  |
| restore_L17_after_late_zero | 16-23 | write | 16 | 0.938 | 0.812–1.000 | -0.344 | 1.000 | -0.001 |  |

peak_vram_gb: 0.255
wall_s: 4.0
