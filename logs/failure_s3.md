# Failure modes (C4)

model: `runs\ar_ft_s3\checkpoint`

| split | n | clean_acc | clamp_Δ_junk_acc | recovery | mean_len |
|---|---:|---:|---:|---:|---:|
| len_short | 64 | 0.984 | 0.984 | +0.000 | 5.0 |
| len_mid | 64 | 0.891 | 0.891 | +0.000 | 17.0 |
| len_ood_pad | 64 | 0.438 | 0.703 | +0.266 | 41.0 |
| stuff_light | 64 | 0.906 | 0.328 | -0.578 | 15.0 |
| stuff_heavy | 64 | 0.469 | 0.125 | -0.344 | 45.0 |

## Geometry (mean on correct vs wrong)

| split | metric | correct | wrong |
|---|---|---:|---:|
| len_short | L23_h_write_norm | 25.9166 | 25.5337 |
| len_short | L23_h_query_norm | 35.1646 | 25.3592 |
| len_short | L23_h_write_erank | 1.0932 | 1.0365 |
| len_short | L23_delta_junk |  |  |
| len_short | L23_delta_query | 0.2542 | 0.3272 |
| len_mid | L23_h_write_norm | 28.9638 | 28.6043 |
| len_mid | L23_h_query_norm | 34.8488 | 35.4679 |
| len_mid | L23_h_write_erank | 1.2097 | 1.1885 |
| len_mid | L23_delta_junk |  |  |
| len_mid | L23_delta_query | 0.2638 | 0.2597 |
| len_ood_pad | L23_h_write_norm | 26.6813 | 27.3260 |
| len_ood_pad | L23_h_query_norm | 34.9215 | 35.6531 |
| len_ood_pad | L23_h_write_erank | 1.1395 | 1.1741 |
| len_ood_pad | L23_delta_junk | 0.2731 | 0.2735 |
| len_ood_pad | L23_delta_query | 0.2683 | 0.2608 |
| stuff_light | L23_h_write_norm | 28.7167 | 28.0321 |
| stuff_light | L23_h_query_norm | 35.3796 | 34.5021 |
| stuff_light | L23_h_write_erank | 1.1938 | 1.1454 |
| stuff_light | L23_delta_junk | 0.2922 | 0.2977 |
| stuff_light | L23_delta_query | 0.2649 | 0.2621 |
| stuff_heavy | L23_h_write_norm | 32.6930 | 29.7736 |
| stuff_heavy | L23_h_query_norm | 35.5774 | 36.4049 |
| stuff_heavy | L23_h_write_erank | 1.2691 | 1.1951 |
| stuff_heavy | L23_delta_junk | 0.2797 | 0.2778 |
| stuff_heavy | L23_delta_query | 0.2633 | 0.2598 |

peak_vram_gb: 0.592  wall_s: 57.7

