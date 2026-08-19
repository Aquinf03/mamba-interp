# L17 store vs finetune step

| step | clean_acc | L17_wipe_acc | drop |
|---:|---:|---:|---:|
| 200 | 0.938 | 0.367 | 0.570 |
| 400 | 0.945 | 0.359 | 0.586 |
| 600 | 0.953 | 0.320 | 0.633 |
| 800 | 0.961 | 0.297 | 0.664 |
| final | 0.961 | 0.297 | 0.664 |

Pretrained AR is 0% (`logs/baseline.md`), so there is no pretrained AR circuit to inherit. By step **200** the L17 store is already there (clean 0.938, wipe 0.367, drop 0.570). From 200→800 clean only rises 2.3 points while L17 wipe **falls** (0.367 → 0.297): later FT consolidates onto L17 rather than spreading the algorithm. Step 800 matches the paper checkpoint wipe (0.297). This table does **not** show a birth between 200 and 800; first snapshot is already late.
