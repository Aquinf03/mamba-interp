# Behavioral baseline (Model A)

| name | task | ablation | n | mean_len | acc | mean_lp(target) |
|---|---|---|---:|---:|---:|---:|
| ar | ar | none | 128 | 9.0 | 0.961 | -0.153 |
| atr_short | atr | none | 128 | 8.0 | 0.961 | -0.268 |
| atr_mid | atr | none | 128 | 19.0 | 0.844 | -0.712 |
| len_short | length | none | 64 | 5.0 | 1.000 | -0.004 |
| len_mid | length | none | 64 | 17.0 | 0.922 | -0.338 |
| len_ood_pad | length | none | 64 | 41.0 | 0.484 | -2.740 |
| stuff_light | length | none | 64 | 15.0 | 0.922 | -0.519 |
| stuff_heavy | length | none | 64 | 45.0 | 0.578 | -2.000 |
| factual | factual | none | 8 | 6.4 | 0.000 | -9.449 |
| ar | ar | ablate_early | 128 | 9.0 | 0.547 | -2.196 |
| ar | ar | ablate_mid | 128 | 9.0 | 0.875 | -0.523 |
| ar | ar | ablate_late | 128 | 9.0 | 0.000 | -82.228 |
| atr_short | atr | ablate_early | 128 | 8.0 | 0.602 | -1.673 |
| atr_short | atr | ablate_mid | 128 | 8.0 | 0.852 | -0.730 |
| atr_short | atr | ablate_late | 128 | 8.0 | 0.000 | -82.823 |
| stuff_heavy | length | ablate_early | 64 | 45.0 | 0.172 | -4.205 |
| stuff_heavy | length | ablate_mid | 64 | 45.0 | 0.438 | -2.970 |
| stuff_heavy | length | ablate_late | 64 | 45.0 | 0.000 | -81.552 |

## Notes

- model: `runs\ar_ft\checkpoint`
- peak_vram_gb: 0.592
- wall_s: 14.3
- metric: teacher-forced next-token top-1 on frozen v1 splits
- ablation: listed mixers become identity (residual pass-through)

## C4 hint

Compare `len_*` / `stuff_*` rows: drops with length or stuffing flag length-generalization candidates.

## Layer sensitivity (4.3)

Compare `ablate_early` / `ablate_mid` / `ablate_late` vs `none` on ar / atr_short / stuff_heavy.
