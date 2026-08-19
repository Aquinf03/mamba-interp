# Log + split freeze (2026-08-19)

Appendix archive of locked markdown receipts. **Do not regenerate `data/splits/v1/`.**
Paper numbers: `logs/findings.md`. Draft: `paper/draft.md`.

Weights (`*.safetensors`) are **not** in git (`runs/` is gitignored). Re-FT seed 1:

```powershell
python scripts\finetune_ar.py --seed 1 --out-dir runs\ar_ft
```

Suggested git tag after committing this freeze: `workshop-draft-2026-08-19`.

## data/splits/v1

| sha256 | file |
|---|---|
| `14977969dfec757192fd533bfe99edbbc895535f62a0d57d4291c9654a7fa68f` | `data/splits/v1/ar_eval.jsonl` |
| `9df73701748fe7c0d3b7f747e383860ed32d0448d9be64d2bca98d2620d995db` | `data/splits/v1/ar_eval.manifest.json` |
| `4e3795e4a91fe511dd7d57152ffcf9cf0aae575f25600dcd8cc0c48ec0b01545` | `data/splits/v1/atr_mid_eval.jsonl` |
| `924b89679512348d3c4a35b4229bf8479c053c0e5b265bfe099cf13e0b15dbd4` | `data/splits/v1/atr_mid_eval.manifest.json` |
| `5ea7032a6f2821bff790cc5abffc8ae712a3ead1dbc1f1a7a77fb959333e44cb` | `data/splits/v1/atr_short_eval.jsonl` |
| `abf29cee8a534baf035ebbd53d5769446407710229c9d4d5ad362616ac52f385` | `data/splits/v1/atr_short_eval.manifest.json` |
| `d076e997f82916ad7c288676a6db3a43fb4bc15e68543c5f87e82d11b107672a` | `data/splits/v1/factual_eval.jsonl` |
| `e9c5f7ea08ec14f4fec38c2ea8e8cdbd1108b7d581a35628c8580dfd8095363f` | `data/splits/v1/factual_eval.manifest.json` |
| `9669d17fc5183d7663af5483387378b4ba5e3097d40caa2ca660956d401016fb` | `data/splits/v1/index.json` |
| `286fa08d1b1714725f634d53f0ba7181642b1f839daed214a2f3955e4919f11b` | `data/splits/v1/length_len_mid_eval.jsonl` |
| `c1742771ffb5eca6532e9f1bcec55a70bd322feaf487b6a34fc9643473b161ae` | `data/splits/v1/length_len_mid_eval.manifest.json` |
| `a531527e4a4343d3517242fc4656e731b7838e01db750d18951453f43a6db45e` | `data/splits/v1/length_len_ood_pad_eval.jsonl` |
| `196952233963d13f0ada1c5f721d9bff48e8aaa3358fefdd1b18e8fc497c64a9` | `data/splits/v1/length_len_ood_pad_eval.manifest.json` |
| `36f359828302db8421b12005e71eb89ec62ea1d954d43ea28ece253a7ef296d6` | `data/splits/v1/length_len_short_eval.jsonl` |
| `93ed6348f6095e93d348c84508aa6fd5c76708e08fcd43d239291f1cd2391763` | `data/splits/v1/length_len_short_eval.manifest.json` |
| `1f3906747f900f9f2c95976a803dbf0f0df3871a8d52446661d5405c4739c367` | `data/splits/v1/length_stuff_heavy_eval.jsonl` |
| `4bd5f82fa8a5d7380453f8e2f5d004255d04a225e23f8d14d7db2dccb5e9eb82` | `data/splits/v1/length_stuff_heavy_eval.manifest.json` |
| `c73bded6a70d3030da0e6304673f343dbe2103b7fe4aa9bcb787969479a878d3` | `data/splits/v1/length_stuff_light_eval.jsonl` |
| `9795ff5b86c6081b582735610bc358385436feb3908176917dd115f2509b7bb8` | `data/splits/v1/length_stuff_light_eval.manifest.json` |

## data/splits/v2 (multi-query; v1 untouched)

| sha256 | file |
|---|---|
| `a71be9237904a4eaaa61039b508ada9e359a391c789ca4b2794e5c740dd0c4a3` | `data/splits/v2/ar_mq_eval.jsonl` |
| `a29da49bdf3a41e06f32b449c9dc2be87b4603e73844cafec2ac3fb21529fbf9` | `data/splits/v2/ar_mq_eval.manifest.json` |

## logs/*.md (locked receipts; demo.md excluded)

| sha256 | file |
|---|---|
| `3abbd8d34a2bf3163fd5446e4b96876624a74465c4cc698a966b2323fccddf14` | `logs/baseline.md` |
| `b48277dab6717bbede81ff90a2dcac630643725db5e25c7c35970a962d8ad4f6` | `logs/baseline_370m.md` |
| `fb3cfc5fd0b30e7c579e313f587e5062cfbcde1eb47ae88fb0fafb3497c50a17` | `logs/baseline_ft.md` |
| `8d471a2f0fe57ecc4fbc46fd6c5572664bc49850d8b1f5bf621f37b7d0ca8f03` | `logs/control_maps.md` |
| `8dd2b6cf7e3246cd513aa0f2a3f89b433ca460091fce931bae13fcc9c95eb03e` | `logs/datasets_v1.md` |
| `b31cbbf63c5092cd4df524d50dcc88bdcd64388008d06acb1cce772c9fdc6ec3` | `logs/failure.md` |
| `bb70dea397f2656bb44ed571a0644b4cc5258a211e9cb63e922d707d0d35b416` | `logs/failure_delta_roles.md` |
| `471677f120c2e425233566a26084bd1177e426f1b56fa80f4443cd1a656b823c` | `logs/failure_filler_ops.md` |
| `471677f120c2e425233566a26084bd1177e426f1b56fa80f4443cd1a656b823c` | `logs/failure_pad_only.md` |
| `275d71a0cfce189e007cfb782642c0c99acbd2f42ecf17c7ababd176fbf754da` | `logs/failure_s2.md` |
| `3d0b6eca3fa078dd9a02cf710ac582f86a8e09542ff79b64574319059faeae04` | `logs/failure_s3.md` |
| `4fa4df72ee6d96bf1a89619926e21d1b650b101847feb3d0054f91838a8f92e3` | `logs/findings.md` |
| `c67917e95aa7d50068f3a660db5169d8d0773fc7ad08bed4d382e0b7312a21fb` | `logs/intervene.md` |
| `458ce1147c9a7e57843290741ab83209e31b01149f89bf257a339135d8506bf7` | `logs/intervene_ar.md` |
| `35309d08d1374f399a1faf4a59fea18c03199bc430d6254bb641433e332d9411` | `logs/intervene_atr_mid.md` |
| `73d699377fd24686585266019959ad3049f061de52c072bd985c81f6f27c56f9` | `logs/intervene_atr_short.md` |
| `cc3d081e637c832452d540b50d1fb63b7283420685e0a7a9d68f331472013d3a` | `logs/intervene_smoke.md` |
| `bc6b9d021fb644f1ea18bd8946e6e931c54f20d2aee4ba946d54635964f53ac7` | `logs/l17_C_patch.md` |
| `d2f0ea8a4e69c3e106a39110bf1861b191d6437d77a7040260903ccf35bee870` | `logs/l17_channels.md` |
| `aca4ff86c3b1504b0b12f6542ced0cc3419a019d95f32505156915e39bf659be` | `logs/l17_channels_k.md` |
| `741393f56b9ce46a77d5be7739b7b47060673a1b8a4c8d2fb6caa4f1318abbe3` | `logs/l17_emergence.md` |
| `1ee025ddf140ba85693edd6c43e34fcf471a71b3b35600afecf5f072674f9269` | `logs/l17_emergence_dense.md` |
| `70386d9f0bee08776d08215f154b63481627ba1cb378109d69da005de72da202` | `logs/l17_keypatch.md` |
| `4ab7de5ed6de8feaeb05a36e92b030bb6221f99f9a3c1a80f555cc0aae694793` | `logs/l17_multiquery.md` |
| `542e10b1a92f0ecf0830e2fd08ad10ec5c3b8ac00642b4080803ef78a28eb718` | `logs/l17_neighbors_s2.md` |
| `329129e4b97d029ce0faf75e9fae06cbf32815e88835341da86e81b17fc5e143` | `logs/l17_restore.md` |
| `abe311c24ec40403eae1671661b2dab58bf09b3ad2d978a01e353abf2fab8363` | `logs/l17_restore_neighbors.md` |
| `629143d854557cab379bcf18958447fd752a3c52f2df4204a0c8236848f3cf3c` | `logs/l17_restore_residual.md` |
| `f8d2aba5da27c25cc3952a28d5cb393a56cd72a0224c940b57ca9a3be8fa89d2` | `logs/l17_restore_residual_atr.md` |
| `405d5560c236cf01713b596f6f28831e68d407c012c89aae85a7d9dea0550b53` | `logs/l17_restore_s2.md` |
| `dc3e74c4be30b0c940bbdc7f6c8d7a26add3bce193b161a31c0a8c0b611b1561` | `logs/l17_restore_time.md` |
| `1da1e714b0d3a238b25bb3643930d53f7a6df5803a639cde135287592a85df0d` | `logs/l17_svd.md` |
| `de93d72742ce678a9a92b76f943015b8a018b39e97f73c150250bb8cc64fc838` | `logs/l17_wipe_atr.md` |
| `6228b0e05b7b4d9bffa67158e7de410d1bd16e938ee7e7addd7c27b19d539672` | `logs/l17_wipe_s2.md` |
| `f70a1887bf3d08129ad4e2579cff90385b0b0db9d0d162c9a4339cf9bfd63a96` | `logs/l17_wipe_s3.md` |
| `f7c087f61e724fe592e825511074b7d757fa35769554f713008a78eaa52d8737` | `logs/model_A.md` |
| `ba949e7c91692f6238ec1e55ace65fc1dc3bf84254e3d33037ddd3d0b50e5f83` | `logs/probes.md` |
| `c711369bdb97f98a493520d7acd0c5477dabeb931f4d88d644468eb6dc168456` | `logs/probes_ft.md` |
| `f9c3882ee92f75bb3aec0067b92e7940b183c8db9c01cfa93f1bc4480c679616` | `logs/probes_mlp_lastwrite.md` |
| `f8a8279a7f4f3ca9c137d2e091b77a68de66a93f46d74c9a96db4d29e733888a` | `logs/probes_over_t.md` |
