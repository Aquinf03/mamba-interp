# E2 L17 h SVD at last write

Replace last-write L17 h with SVD reconstruction (no late wipe). zero_top k = drop largest k singular values. keep_top k = keep only those.

| name | acc |
|---|---:|
| clean | 0.961 |
| svd_zero_top_1 | 0.961 |
| svd_zero_top_2 | 0.914 |
| svd_zero_top_4 | 0.672 |
| svd_zero_top_8 | 0.297 |
| svd_keep_top_1 | 0.312 |
| svd_keep_top_4 | 0.875 |
| svd_keep_top_8 | 0.961 |

**Keep:** not one principal component (zero_top 1 = clean; keep_top 1 = 0.312). Rank-8 reconstruction at last write is enough (keep_top 8 = 0.961); dropping the top 8 SVs matches L17 wipe (0.297). Distributed in rank, concentrated in the larger singular values.
