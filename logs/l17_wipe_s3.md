# Seed 3 - L17 write wipe

Model: `runs/ar_ft_s3/checkpoint` (same recipe, `--seed 3`). n=128 AR.

| seed | clean | L17 wipe | drop |
|---:|---:|---:|---:|
| 1 (`runs/ar_ft`) | 0.961 | 0.297 | 0.664 |
| 2 (`runs/ar_ft_s2`) | 0.961 | 0.289 | 0.672 |
| 3 (`runs/ar_ft_s3`) | **0.984** | **0.320** | 0.664 |

L17 write-wipe collapses AR on a third seed. Not a one-run fluke.
