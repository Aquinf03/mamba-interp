# L17 store vs finetune step (dense, first 200)

`runs/ar_ft_trace50/` - same recipe as paper FT, seed 1, `--steps 200 --save-every 50`. Do not treat as a new seed. Steps 200–800: `logs/l17_emergence.md`.

| step | clean_acc | L17_wipe_acc | drop |
|---:|---:|---:|---:|
| pretrained | 0.000 | - | - |
| 50 | 0.961 | 0.352 | 0.609 |
| 100 | 0.914 | 0.297 | 0.617 |
| 150 | 0.945 | 0.367 | 0.578 |
| 200 | 0.938 | 0.367 | 0.570 |
| 400 (trace) | 0.945 | 0.359 | 0.586 |
| 600 (trace) | 0.953 | 0.320 | 0.633 |
| 800 (trace / paper) | 0.961 | 0.297 | 0.664 |

The L17 store is **already in place by step 50** (clean 0.961, drop 0.609). This table still does not show a birth. Step 200 matches the earlier trace (0.938 / 0.367). Later FT increases L17 dependence (wipe 0.352 → 0.297).
