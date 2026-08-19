# Seed 2 - L17 write wipe

Model: `runs/ar_ft_s2/checkpoint` (same recipe as seed 1, `--seed 2`). n=128 AR.

| seed | clean | L17 wipe | drop |
|---:|---:|---:|---:|
| 1 (`runs/ar_ft`) | 0.961 | 0.297 | 0.664 |
| 2 (`runs/ar_ft_s2`) | 0.961 | **0.289** | 0.672 |

L17 write-wipe collapses AR on a second seed. Not a one-run fluke.
