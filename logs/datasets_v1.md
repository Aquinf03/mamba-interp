# Dataset freeze v1

Built 2026-08-08 on run machine. Seed=0, `dataset_version=v1`, tokenizer `state-spaces/mamba-130m-hf`.

| Split | N | mean seq_len |
|---|---:|---:|
| ar_eval | 128 | 9 |
| atr_short_eval | 128 | 8 |
| atr_mid_eval | 128 | 19 |
| length/len_short | 64 | 5 |
| length/len_mid | 64 | 17 |
| length/len_ood_pad | 64 | 41 |
| length/stuff_light | 64 | 15 |
| length/stuff_heavy | 64 | 45 |
| factual_eval | 8 | - |

Do not regenerate without bumping `dataset_version` in `configs/datasets.yaml`.
