# Failure modes (C4)

model: `runs\ar_ft_s2\checkpoint`

| split | n | clean_acc | clamp_Δ_junk_acc | recovery | mean_len |
|---|---:|---:|---:|---:|---:|
| len_short | 64 | 0.984 | 0.984 | +0.000 | 5.0 |
| len_mid | 64 | 0.781 | 0.781 | +0.000 | 17.0 |
| len_ood_pad | 64 | 0.578 | 0.656 | +0.078 | 41.0 |
| stuff_light | 64 | 0.938 | 0.297 | -0.641 | 15.0 |
| stuff_heavy | 64 | 0.625 | 0.141 | -0.484 | 45.0 |

## Geometry (mean on correct vs wrong)

| split | metric | correct | wrong |
|---|---|---:|---:|
| len_short | L23_h_write_norm | 45.9893 | 52.1605 |
| len_short | L23_h_query_norm | 55.1325 | 77.2190 |
| len_short | L23_h_write_erank | 1.0407 | 1.0419 |
| len_short | L23_delta_junk |  |  |
| len_short | L23_delta_query | 0.2616 | 0.2030 |
| len_mid | L23_h_write_norm | 47.9688 | 48.2816 |
| len_mid | L23_h_query_norm | 55.6788 | 58.3745 |
| len_mid | L23_h_write_erank | 1.0860 | 1.0998 |
| len_mid | L23_delta_junk |  |  |
| len_mid | L23_delta_query | 0.2616 | 0.2578 |
| len_ood_pad | L23_h_write_norm | 47.9543 | 48.7072 |
| len_ood_pad | L23_h_query_norm | 54.3909 | 60.7634 |
| len_ood_pad | L23_h_write_erank | 1.0628 | 1.0647 |
| len_ood_pad | L23_delta_junk | 0.2694 | 0.2697 |
| len_ood_pad | L23_delta_query | 0.2681 | 0.2640 |
| stuff_light | L23_h_write_norm | 48.6003 | 43.1590 |
| stuff_light | L23_h_query_norm | 61.4271 | 65.8770 |
| stuff_light | L23_h_write_erank | 1.0785 | 1.0962 |
| stuff_light | L23_delta_junk | 0.2775 | 0.2764 |
| stuff_light | L23_delta_query | 0.2603 | 0.2563 |
| stuff_heavy | L23_h_write_norm | 51.4533 | 50.9515 |
| stuff_heavy | L23_h_query_norm | 60.8622 | 64.5113 |
| stuff_heavy | L23_h_write_erank | 1.1021 | 1.0885 |
| stuff_heavy | L23_delta_junk | 0.2713 | 0.2706 |
| stuff_heavy | L23_delta_query | 0.2605 | 0.2632 |

peak_vram_gb: 0.592  wall_s: 60.1

