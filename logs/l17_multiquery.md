# E5 multi-query AR + L17 wipe

v2 split (`data/splits/v2/ar_mq_eval.jsonl`): 4 pairs, 2 queries. Intermediate query is answered in-transcript; final query is scored as usual. v1 untouched.

| condition | n | final-query acc | first-query acc (teacher-forced) |
|---|---:|---:|---:|
| clean | 128 | 0.922 | 0.984 |
| L17 write-wipe | 128 | 0.094 |  |

**Keep:** clean final-query 0.922 (first-query TF 0.984). L17 write-wipe → **0.094**. The second bind also lives in L17 \(h\). v1 untouched.

