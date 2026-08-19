# E6 L17 Hidden-Attention-style map (control)

score_t = mean_E (h_t · C_query). Softmax over t. Frozen AR n=128. Not a reimplementation of Hidden Attention / LaTIM.

| mass on | mean softmax |
|---|---:|
| queried_value | 0.111 |
| other_value | 0.333 |
| key | 0.444 |
| query | 0.111 |
| other | 0.000 |

**Keep:** mass is on **keys** (0.444) and other values (0.333), not the queried value (0.111). This heatmap does not find the bind. Causal result remains L17 \(h\) wipe/restore, not Hidden Attention / LaTIM rebuilt.
