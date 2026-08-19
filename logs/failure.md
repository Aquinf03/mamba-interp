# Failure modes (C4)

model: `runs\ar_ft\checkpoint`

| split | n | clean_acc | clamp_Δ_junk_acc | recovery | mean_len |
|---|---:|---:|---:|---:|---:|
| len_short | 64 | 1.000 | 1.000 | +0.000 | 5.0 |
| len_mid | 64 | 0.922 | 0.922 | +0.000 | 17.0 |
| len_ood_pad | 64 | 0.484 | 0.703 | +0.219 | 41.0 |
| stuff_light | 64 | 0.922 | 0.328 | -0.594 | 15.0 |
| stuff_heavy | 64 | 0.578 | 0.109 | -0.469 | 45.0 |

## Geometry (mean on correct vs wrong)

| split | metric | correct | wrong |
|---|---|---:|---:|
| len_short | L23_h_write_norm | 44.1665 |  |
| len_short | L23_h_query_norm | 49.1478 |  |
| len_short | L23_h_write_erank | 1.0467 |  |
| len_short | L23_delta_junk |  |  |
| len_short | L23_delta_query | 0.2609 |  |
| len_mid | L23_h_write_norm | 45.5127 | 47.6848 |
| len_mid | L23_h_query_norm | 51.4822 | 53.0573 |
| len_mid | L23_h_write_erank | 1.1060 | 1.0821 |
| len_mid | L23_delta_junk |  |  |
| len_mid | L23_delta_query | 0.2657 | 0.2610 |
| len_ood_pad | L23_h_write_norm | 45.7645 | 46.3725 |
| len_ood_pad | L23_h_query_norm | 51.9217 | 51.5149 |
| len_ood_pad | L23_h_write_erank | 1.0681 | 1.0780 |
| len_ood_pad | L23_delta_junk | 0.2699 | 0.2701 |
| len_ood_pad | L23_delta_query | 0.2658 | 0.2634 |
| stuff_light | L23_h_write_norm | 45.5022 | 47.8591 |
| stuff_light | L23_h_query_norm | 52.2836 | 52.6872 |
| stuff_light | L23_h_write_erank | 1.0958 | 1.0758 |
| stuff_light | L23_delta_junk | 0.2796 | 0.2827 |
| stuff_light | L23_delta_query | 0.2652 | 0.2578 |
| stuff_heavy | L23_h_write_norm | 47.8644 | 49.3877 |
| stuff_heavy | L23_h_query_norm | 53.7238 | 53.9476 |
| stuff_heavy | L23_h_write_erank | 1.1106 | 1.1180 |
| stuff_heavy | L23_delta_junk | 0.2709 | 0.2716 |
| stuff_heavy | L23_delta_query | 0.2601 | 0.2568 |

peak_vram_gb: 0.592  wall_s: 53.8

