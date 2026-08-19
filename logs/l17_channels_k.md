# E1 L17 multi-channel wipes

N=16. Single-N already ~clean. Random = one fixed subset. top = highest mean energy over examples at last write.

| name | k | how | acc |
|---|---:|---|---:|
| clean | 0 | - | 0.961 |
| zero_h_L17_k2_random | 2 | random [7, 9] | 0.977 |
| zero_h_L17_k2_top_energy | 2 | top_energy [4, 15] | 0.953 |
| zero_h_L17_k2_top_per_ex | 2 | per-example top energy | 0.969 |
| zero_h_L17_k4_random | 4 | random [4, 5, 7, 9] | 0.977 |
| zero_h_L17_k4_top_energy | 4 | top_energy [1, 4, 7, 15] | 0.969 |
| zero_h_L17_k4_top_per_ex | 4 | per-example top energy | 0.953 |
| zero_h_L17_k8_random | 8 | random [0, 2, 3, 4, 7, 8, 9, 14] | 0.898 |
| zero_h_L17_k8_top_energy | 8 | top_energy [1, 2, 4, 5, 7, 9, 10, 15] | 0.953 |
| zero_h_L17_k8_top_per_ex | 8 | per-example top energy | 0.914 |

**Keep:** k=2 and k=4 stay at clean (~0.95–0.98). k=8 random dips to 0.898; top-energy k=8 stays 0.953. Energy ranking does not isolate the bind. Spread across \(N\), not a few high-norm slots.
