# C5 L17 mean Δ by token role

layer 17. Per-token mean over E, pooled across examples.

| split | role | n | mean | median | p10 | p90 |
|---|---|---:|---:|---:|---:|---:|
| len_ood_pad | key | 256 | 0.0107 | 0.0072 | 0.0043 | 0.0227 |
| len_ood_pad | pad | 2048 | 0.0049 | 0.0047 | 0.0031 | 0.0068 |
| len_ood_pad | query | 64 | 0.0047 | 0.0047 | 0.0029 | 0.0064 |
| len_ood_pad | value | 256 | 0.0070 | 0.0062 | 0.0040 | 0.0110 |
| stuff_light | filler | 384 | 0.0063 | 0.0059 | 0.0039 | 0.0090 |
| stuff_light | key | 256 | 0.0101 | 0.0064 | 0.0041 | 0.0227 |
| stuff_light | query | 64 | 0.0057 | 0.0052 | 0.0035 | 0.0075 |
| stuff_light | value | 256 | 0.0069 | 0.0062 | 0.0039 | 0.0111 |
| stuff_heavy | filler | 1792 | 0.0052 | 0.0049 | 0.0032 | 0.0074 |
| stuff_heavy | key | 512 | 0.0071 | 0.0050 | 0.0033 | 0.0224 |
| stuff_heavy | query | 64 | 0.0043 | 0.0042 | 0.0030 | 0.0057 |
| stuff_heavy | value | 512 | 0.0056 | 0.0049 | 0.0031 | 0.0099 |

If pad Δ looks like value Δ, pad is writing. Rank already failed; this is the Δ split.
