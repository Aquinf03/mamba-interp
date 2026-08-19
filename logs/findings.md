# Findings lock (2026-08-18)

Durable record of results that later experiments must **not** overwrite in interpretation. Raw tables live in the log files listed at the end. Smoke runs (`--max-examples 16`) are **not** paper numbers. What to run next: venue tex if you want a PDF. Paper draft: `paper/draft.md`. Figures: `figs/`. Freeze: `logs/LOCK.md`. Demo (not paper n): `python scripts\run_demo.py`.

**Claim to keep:** When AR-finetuned `mamba-130m-hf` does associative recall / ATR, the bind is stored in **layer 17 recurrent state \(h\) during write**. Residual-matched noise misses it. Linear probes at the query token are **readout**, not store. Padding overwrites that state; stuffing is a different failure.

**Do not claim:** \(h\) linearly beats residual at the query token; swap transplants the donor value; stuffing is the same mechanism as pad; pretrained 130M **or 370M** solves AR; a portable KV slot in \(N\) or \(C\); last_write MLP “finds” the value.

---

## Setup (frozen)

| Item | Value |
|---|---|
| Base | `state-spaces/mamba-130m-hf` (24 layers, \(d=768\), \(E=1536\), \(N=16\)) |
| Task model | `runs/ar_ft/checkpoint` - full FT, 800 steps, 2048 AR train, seed 1, lr \(3\times10^{-4}\), AMP fp16 / FP32 master weights |
| Eval | frozen `data/splits/v1/` (do not regenerate without version bump) |
| Hardware | RTX 4070 8 GB, Windows, HF sequential scan (no fused `mamba-ssm`) |
| FT receipt | `runs/ar_ft/train_meta.json` - eval acc **0.961**, peak 2.66 GB, 312 s |

Pretrained Model A is at **0%** next-token AR/ATR/length/factual (`logs/baseline.md`). Pretrained **370M** is also **0%** on the same splits (`logs/baseline_370m.md`). All causal claims below are on the **finetuned** 130M checkpoint unless labeled pretrained.

---

## Behavior after AR finetune

`logs/baseline_ft.md` (n=128 AR/ATR, n=64 length):

| split | acc | mean_lp |
|---|---:|---:|
| ar | 0.961 | −0.153 |
| atr_short | 0.961 | −0.268 |
| atr_mid | 0.844 | −0.712 |
| len_short | 1.000 | −0.004 |
| len_mid | 0.922 | −0.338 |
| len_ood_pad | **0.484** | −2.740 |
| stuff_light | 0.922 | −0.519 |
| stuff_heavy | **0.578** | −2.000 |
| factual | 0.000 | −9.449 |

Mixer identity ablations on AR: early 0.547, mid 0.875, **late 0.000** (lp ≈ −82). Late mixers are required for the head; that is **not** the same as “memory lives in every late layer.”

---

## C1 - linear probes

### Pretrained (null)

`logs/probes.md`, AR value @ query, n=128, 40-way. Residual test **0.143** ≥ best \(h\) **0.107**. Train≈1, test≈chance. No C1. Same family as *Lost in State Space* (frozen 130M \(h\) anisotropic).

### Finetuned, query token only

`logs/probes_ft.md`. Chance ≈ 1/40.

| layer | residual test | best \(h\) test |
|---:|---:|---:|
| 0 / 11 | ~chance | ~chance (overfit) |
| 23 | **0.929** | 0.607 (`h_mean_n`) |

Residual wins at last layer last token. Expected: that is the bus the LM head reads.

### Finetuned, over \(t\) (the C1 figure)

`logs/probes_over_t.md`, n=384 fresh AR (seed 99, **not** frozen v1), 52-way, train 0.75.

| site | meaning | best test |
|---|---|---|
| `first` | t=0 | ~0.03 (nothing) |
| `value_write` | token of the bound value | residual **1.000** even at L0 - **the value token itself**, not storage |
| `last_write` | last pair token, before query | residual 0.073, \(h\) 0.104 - **no linear value code** after the list |
| `query` L23 | answer position | residual **0.979**, \(h\) 0.885 - residual is **readout** |

**Keep:** store in \(h\) is causal and **not** a linearly decodable “answer letter” at end of write (superposed / key-addressed). Residual wins when the model has already assembled the answer, and trivially at the value token.

**Do not keep:** “state probe beats residual” as a title.

---

## C2 / C3 - causal interventions (n=128)

Source: `logs/intervene_ar.md`. Bootstrap 95% CI. Core layers 16–23 unless named.

### Main table

| condition | acc | ci95 | acc on clean-ok | notes |
|---|---:|---|---:|---|
| clean | 0.961 | 0.930–0.992 | - | |
| zero_h_write | **0.242** | 0.172–0.320 | 0.228 | lp −4.86 (model still on) |
| swap_h_write_end | **0.234** | 0.164–0.312 | 0.220 | **donor_acc 0.072** |
| zero_B_write | 0.242 | 0.172–0.320 | 0.228 | **identical** to zero_h_write |
| clamp_delta_write | 0.242 | 0.172–0.320 | 0.228 | **identical** to zero_h_write |
| residual_noise_write | **0.961** | 0.930–0.992 | 1.000 | Δlp ≈ 0 |
| residual_noise_query | **0.883** | 0.828–0.938 | 0.902 | real but small |
| zero_h_write_early | 0.906 | 0.852–0.953 | 0.927 | |
| zero_h_write_mid | 0.945 | 0.906–0.977 | 0.976 | |
| zero_h_query / zero_C_query / local_only | **0.000** | 0–0 | 0 | lp ≈ −50; **readout death**, not the store result |

Write-block ops (`zero_h` / `zero_B` / `clamp_Δ`) are one mechanism: no writes ⇒ empty \(h\). Report **one** “block write” row in the paper.

Mixer L2 of `zero_h_write` vs clean is **not** a tiny budget: write mean 29.6 (median 5.8, p90 85.5); query mean 54.4 (median 25.3, p90 133.6). Residual-matched write noise still does nothing.

**Swap:** breaks recall; does **not** copy the donor bind (donor_acc 0.072 vs ~0.025 chance). C2 = necessity of \(h\), not a linearly swappable slot.

### Layer sweep (write-zero \(h\), one layer)

Almost every layer stays at clean (~0.96). **Only L17 collapses: acc 0.297** (CI 0.219–0.383, Δlp −4.16). Full late-block wipe 0.242 is L17 plus a little extra. L0 is a mild 0.938; L16/L18+ are clean.

**Keep:** AR memory is **layer-17 state during write**, not “late residual” and not “all 24 layers.”

### L17 sufficiency + channels + keypatch (2026-08-18)

`logs/l17_restore.md`, `l17_channels.md`, `l17_keypatch.md`.

| condition | acc |
|---|---:|
| clean | 0.961 |
| zero late \(h\) (16–23 write) | 0.242 |
| **restore this example’s L17 \(h\) at last write** | **0.961** (CI 0.929–0.992, Δlp −0.004) |
| zero L17 only | 0.297 |

**Keep:** L17 \(h\) is **necessary and sufficient**. Wiping late state kills AR; putting only this sequence’s L17 \(h\) back fully restores it.

Single \(N\)-channel wipes on L17: all stay ~0.96 (N12 = 0.945). The bind is **not** one of 16 slots; it is distributed in L17 (across \(E\) or the full \(N\)).

Key-matched patch (same keys, donor has different value for the query key): clean 0.984, after patch **still original V 0.984, donor V 0.000**. L17 \(h\) is sufficient **for this forward pass**, not a portable key–value register you can swap across sequences. Matches failed last_write linear probes. Do not claim a content-addressable dictionary in \(N\).

### A1–A3 - residual restore, time, neighbors (2026-08-18)

`logs/l17_restore_residual.md`, `l17_restore_time.md`, `l17_restore_neighbors.md`. Same late wipe (0.242), then paste **this example’s** clean vector.

| condition | acc |
|---|---:|
| clean | 0.961 |
| late \(h\) wipe | 0.242 |
| restore L17 **\(h\)** @ last write | **0.961** |
| restore L17 **residual** @ last write | **0.242** |
| restore L17 **residual** @ query | 0.258 |
| restore L17 \(h\) @ value token | **0.969** |
| restore L17 \(h\) @ query | **0.953** |
| restore L16 \(h\) @ last write | 0.258 |
| restore L18 \(h\) @ last write | 0.242 |

**Keep (C3):** copying the actual L17 residual skip does **not** recover, even at query. Copying L17 \(h\) does. Residual-matched noise was already a miss; this is the copy-the-real-vector control.

**Keep (time):** L17 \(h\) already holds the bind at the **value token** and still holds it at query. Not a last-write-only snapshot. Residual at those sites is not the store.

**Keep (layer):** neighbor restore fails. Sufficiency is **L17**, not “any late \(h\).”

ATR-short (`logs/l17_restore_residual_atr.md`), same protocol:

| condition | acc |
|---|---:|
| clean | 0.961 |
| late \(h\) wipe | 0.039 |
| restore L17 \(h\) @ last write | **0.945** (CI 0.898–0.984; acc\|clean_ok 0.984) |
| restore L17 residual @ last write | **0.039** |
| restore L17 residual @ query | 0.062 |

**Keep:** C3 is not AR-only. ATR wipe is deeper (0.039); L17 \(h\) restore returns to ~clean; residual copy stays at wipe.

### B - seed 2 restore, neighbors, ATR L17 wipe

`logs/l17_restore_s2.md`, `l17_neighbors_s2.md`, `l17_wipe_atr.md`.

Seed 2 restore (AR):

| condition | s1 | s2 |
|---|---:|---:|
| clean | 0.961 | 0.961 |
| late \(h\) wipe | 0.242 | 0.250 |
| restore L17 \(h\) | 0.961 | **0.969** |
| restore residual last write | 0.242 | **0.250** |
| restore residual query | 0.258 | 0.258 |

Seed 2 single-layer wipe: L16 **0.969**, L17 **0.289**, L18 **0.961**. Same L17 bottleneck (s1 L17 wipe was 0.297).

ATR-short L17-only wipe (seed 1): late-block 0.039; **L17-only 0.133**; L16/L18 stay 0.961. ATR store is L17-primary; the full late block adds a bit more than on AR (L17 0.297 vs late 0.242).

**Keep:** restore vs residual and the L17 layer id **replicate** on seed 2. ATR also dies at L17.

---

---

### L17 vs finetune step (emergence)

`logs/l17_emergence.md`. Second FT run, same recipe, `--save-every 200` into `runs/ar_ft_trace/` (paper checkpoint `runs/ar_ft` untouched). n=128 AR.

| step | clean | L17 wipe | drop |
|---:|---:|---:|---:|
| pretrained (Model A) | 0.000 | - | - |
| **50** (dense trace) | 0.961 | 0.352 | 0.609 |
| 100 | 0.914 | 0.297 | 0.617 |
| 150 | 0.945 | 0.367 | 0.578 |
| 200 | 0.938 | 0.367 | 0.570 |
| 400 | 0.945 | 0.359 | 0.586 |
| 600 | 0.953 | 0.320 | 0.633 |
| 800 / final | **0.961** | **0.297** | 0.664 |

Step 800 wipe **matches** the paper checkpoint (0.297). The store is **already in place by step 50** (`logs/l17_emergence_dense.md`: clean 0.961, wipe 0.352). Later steps increase L17 dependence (wipe 0.352 → 0.297). Pretrained AR is 0%, so this L17 store is a **finetune algorithm**. We did **not** catch a birth; first snapshot at 50 is already late. Do not claim a sharp phase transition.

---

## ATR (same checkpoint, not AR-only)

`logs/intervene_atr_short.md`, `logs/intervene_atr_mid.md`.

| | clean | zero_h_write | swap (donor_acc) | residual_write | residual_query | local_only |
|---|---:|---:|---|---:|---:|---:|
| ATR short | 0.961 | **0.039** | 0.008 (0.167) | 0.961 | 0.812 | 0.000 |
| ATR mid | 0.844 | **0.039** | 0.016 (0.071) | 0.836 | 0.680 | 0.000 |

ATR is **more** state-dependent than AR. Conv + D-skip cannot do it. Residual-write still ~clean. Residual-query hurts more on mid (0.680) but state wipe still goes to ~0.04.

---

## C4 - length / stuffing

`logs/failure.md`, n=64. Fix = clamp \(\Delta \leftarrow 0\) on filler+pad (junk) at late layers, test time.

| split | clean | clamp Δ junk | recovery |
|---|---:|---:|---:|
| len_short | 1.000 | 1.000 | 0 |
| len_mid | 0.922 | 0.922 | 0 |
| **len_ood_pad** | **0.484** | **0.703** | **+0.219** |
| stuff_light | 0.922 | 0.328 | **−0.594** |
| stuff_heavy | 0.578 | 0.109 | **−0.469** |

**Keep (seed 1):** OOD pad overwrites \(h\); freezing \(\Delta\) on pad recovers **+0.219**. Stuffing is **not** the same: the same clamp destroys accuracy (fillers are used, not just noise). Seeds 2–3: same *sign* on pad (+0.078 / +0.266) and stuffing (hurts); **do not lock +0.22 as the effect size**.

### Pad vs filler split (C1–C5)

`logs/failure_pad_only.md`, `failure_filler_ops.md`, `failure_delta_roles.md`. Late layers, paper checkpoint, n=64.

| split | clean | Δ←0 **pad** | Δ←0 **filler** | zero **B** filler | Δ=5 filler |
|---|---:|---:|---:|---:|---:|
| len_short / mid | 1.00 / 0.92 | no-op | no-op | no-op | no-op |
| **len_ood_pad** | 0.484 | **0.703 (+0.219)** | 0.484 | 0.484 | 0.484 |
| stuff_light | 0.922 | 0.922 | **0.328** | **0.328** | **0.000** |
| stuff_heavy | 0.578 | 0.578 | **0.109** | 0.125 | **0.000** |

Pad-only clamp **equals** the old junk clamp on OOD pad. Filler-only clamp **equals** the old junk clamp on stuffing. Two failures, surgically split.

**Keep:** pad is junk overwrite (freeze Δ helps). Fillers are **load-bearing**: zero-\(B\) hurts like freeze-Δ; force-forget (Δ=5) kills stuffing to **0%**. Stop hunting a stuffing “fix.”

**Do not keep:** “pads have huge Δ at L17.” L17 mean Δ: pad 0.0049 < value 0.0070; filler ≈ value. Causal split is the evidence, not a Δ histogram. L23 junk Δ (~0.27 in `failure.md`) also does not separate correct vs wrong.

Geometry (L23): effective rank of \(h\) ≈ **1.05–1.12** on all splits (anisotropic; *Lost in State Space*). Correct vs wrong **do not** separate on norm / rank / mean \(\Delta\). Do not tell a “rank collapse causes errors” story with these metrics.

---

## Seed 2 / 3 replication

`logs/l17_wipe_s2.md`, `failure_s2.md`, `l17_wipe_s3.md`, `failure_s3.md`. Recipe unchanged; `--seed 2` / `--seed 3`. Do not overwrite `runs/ar_ft`.

| | seed 1 | seed 2 | seed 3 |
|---|---:|---:|---:|
| AR clean | 0.961 | 0.961 | **0.984** |
| L17 write-wipe | **0.297** | **0.289** | **0.320** |

| split | s1 rec | s2 rec | s3 rec (clean → clamp) |
|---|---:|---:|---|
| len_ood_pad | **+0.219** | **+0.078** | **+0.266** (0.438 → 0.703) |
| stuff_light | −0.594 | −0.641 | −0.578 (0.906 → 0.328) |
| stuff_heavy | −0.469 | −0.484 | −0.344 (0.469 → 0.125) |

**Keep:** L17 wipe **replicates on three seeds** (~0.29–0.32). Pad clamp **helps** on all three; stuffing clamp **hurts** on all three. Pad recovery is **+0.08 to +0.27** - sign is the result, not +0.22. Seed 2 is weaker on len_mid (0.781).

---

## C5 - conv vs SSM

`local_only` (SSM recurrence off, D·u + conv remain) = **0%** on AR, ATR-short, ATR-mid. Short conv / skip is not doing these tasks. Combined with L17 write-wipe, the algorithm is **state-mediated**.

---

## D - 370M zero-shot

`logs/baseline_370m.md`. `state-spaces/mamba-370m-hf` (48 layers, d=1024, E=2048, N=16), fp16, peak ~1.1 GB, fits 8 GB.

Every frozen v1 split is **0%** (AR/ATR/length/factual), same as 130M pretrained. Mean target lp ≈ −6.4 on AR (130M was −6.4). Scale does not unlock this synthetic bind.

**Keep:** the L17 store is a **finetune algorithm**, not something 370M already does on these strings.

**Skip:** D2 layer sweep (no competent pretrained behavior). D3 extra induction eval (ATR-short is already that, and it is 0%). D4 full-FT 370M (8 GB sequential scan; would not kill the finetune objection anyway if we FT it).

---

## E - content / addressing (honest negatives)

Paper checkpoint, n=128 AR unless noted. New logs only; did not overwrite `l17_channels.md` / `l17_keypatch.md`. Wall 166 s, peak 0.33 GB.

### E1 multi-\(N\) wipe

`logs/l17_channels_k.md`. N=16. k=2 / k=4 (random, top-energy, per-example top) all stay ~clean (0.953–0.977). k=8 random **0.898**; k=8 top-energy still **0.953**.

**Keep:** the bind is spread across \(N\). High-energy slots are not the store. Matches single-\(N\) null.

### E2 SVD of L17 \(h\) at last write

`logs/l17_svd.md`. Swap reconstructed \(h\) at last write (no late wipe).

| condition | acc |
|---|---:|
| clean | 0.961 |
| zero_top 1 / 2 / 4 / 8 | 0.961 / 0.914 / 0.672 / **0.297** |
| keep_top 1 / 4 / 8 | 0.312 / 0.875 / **0.961** |

**Keep:** not one principal component. Rank-8 at last write is enough; dropping the top 8 SVs matches L17 wipe. Distributed in rank, concentrated in larger singular values. Do not claim a 1-D code.

### E3 \(C\) patch at query

`logs/l17_C_patch.md`. Same key-matched pairs as the \(h\) keypatch; paste donor L17 **\(C\) at query**, leave recipient \(h\) alone.

| n | clean | still original V | now donor V |
|---:|---:|---:|---:|
| 128 | 0.984 | **0.984** | **0.000** |

**Keep:** readout \(C\) is not a portable address. Same negative as patching \(h\). Store stays in this sequence’s \(h\).

### E4 MLP vs linear at last_write

`logs/probes_mlp_lastwrite.md`. Fresh AR n=384 seed=99 (not v1), 52-way, two-layer ReLU 256-d. Chance ≈ 0.019.

| feature | linear test | MLP test |
|---|---:|---:|
| residual | 0.052 | 0.062 |
| \(h\) mean-\(N\) | 0.010 | 0.031 |
| \(h\) flat | 0.042 | 0.042 |

Train = 1.0 on residual/MLP. **Keep:** last_write still has no easily decoded value. Superposition / key-addressing holds against a small MLP, not just a linear probe.

### E5 multi-query

`logs/l17_multiquery.md`. New split `data/splits/v2/ar_mq_eval.jsonl` (4 pairs, 2 queries, seed 2026). **v1 untouched.**

| condition | final-query acc | first-query (teacher-forced) |
|---|---:|---:|
| clean | 0.922 | 0.984 |
| L17 write-wipe | **0.094** | - |

**Keep:** the second bind also lives in L17 \(h\). Slightly harder than 1-query AR (0.922 vs 0.961) but the same layer.

### E6 implicit token map (control, not Hidden Attention)

`logs/control_maps.md`. score_t = mean_E(\(h_t \cdot C_{\mathrm{query}}\)); softmax over t; same 128 AR items.

| mass on | mean softmax |
|---|---:|
| queried_value | 0.111 |
| other_value | 0.333 |
| **key** | **0.444** |
| query | 0.111 |
| other | 0.000 |

**Keep:** this heatmap lights **keys**, not the stored value. Do not sell it as locating the bind. Causal evidence remains wipe/restore. Do not rebuild Hidden Attention / LaTIM.

---

## Paper framing (do not drift)

One sentence: *On this AR-finetuned Mamba-130M, associative recall binds live in layer-17 \(h\) from the value token through query (including a second query on the same list); copying that \(h\) restores accuracy, copying the residual or key-matched \(C\) does not; the code is distributed across \(N\) and SVD rank, not linearly/MLP-decodable at last write; padding overwrites the store with a seed-sensitive \(\Delta\)-clamp.*

| Claim | Status |
|---|---|
| C1 content in \(h\) linearly > residual @ query | **Fail** - residual wins at query; last_write linear **and MLP** ~chance |
| C1 store ≠ readout | **Hold** - probes-over-\(t\) + interventions |
| Portable KV in \(N\) or \(C\) | **Fail** - k-wipes, \(h\) keypatch, \(C\) patch, implicit maps |
| C2 causality of \(h\) | **Hold** - L17 wipe s1/s2/s3 = 0.297 / 0.289 / 0.320 |
| C3 residual miss | **Hold** - residual restore ~wipe on AR and ATR; seed-2 replicate |
| C4 pad overwrite + Δ fix | **Hold in sign** (s1 +0.219, s2 +0.078, s3 +0.266); magnitude not locked |
| C4 stuffing = same geometry | **Fail** - filler clamp / zero_B hurt; Δ=5 → 0%; rank unused |
| C5 conv stole the task | **Fail** (good) - local_only 0 |

Limitations (abstract, not footnote): 130M, synthetic AR/ATR, **task-finetuned** (pretrained 130M and 370M are both 0% on these splits), three FT seeds for L17 wipe / pad clamp, Windows slow scan, swap does not copy content.

Venue: **workshop**. L17 causality is not a one-run fluke. Main track is still blocked: 370M zero-shot is also 0%, so the store is learned at finetune, not uncovered in a pretrained net.

Related work to cite, not rehash: Hidden Attention, LaTIM, IOI-in-Mamba, Locating & Editing, *Lost in State Space*, Primacy–Recency, Activation Subspace Bottlenecks, Stuffed/LongMamba.

---

## What later experiments must not destroy

1. Do not overwrite `logs/baseline.md` (pretrained) or `logs/probes.md` (pretrained null).
2. Finetuned receipts: `baseline_ft.*`, `probes_ft.*`, `intervene_ar.*`, `intervene_atr_{short,mid}.*`, `probes_over_t.*`, `failure.*`, `failure_pad_only.*`, `failure_filler_ops.*`, `failure_delta_roles.*`, `failure_split.json`, `l17_restore.*`, `l17_channels.*`, `l17_keypatch.*`, `l17_emergence.*`, `l17_wipe_s2.*`, `failure_s2.*`, `l17_restore_residual.*`, `l17_restore_residual_atr.*`, `l17_restore_time.*`, `l17_restore_neighbors.*`, `l17_restore_s2.*`, `l17_neighbors_s2.*`, `l17_wipe_atr.*`, `l17_wipe_s3.*`, `failure_s3.*`, `l17_emergence_dense.*`, `l17_channels_k.*`, `l17_svd.*`, `l17_C_patch.*`, `probes_mlp_lastwrite.*`, `l17_multiquery.*`, `control_maps.*`. Do not overwrite `data/splits/v1/` or `data/splits/v2/`.
3. New runs: new filenames (`logs/intervene_ar_s2.md`, etc.) or a dated subfolder. Do not reuse `intervene_ar.md` for a different protocol without copying this file.
4. Checkpoint `runs/ar_ft/checkpoint` is the model these numbers belong to. Another seed → `runs/ar_ft_s2/`. Emergence snapshots: `runs/ar_ft_trace/step_*` (same recipe, seed 1; do not treat as seed 2). Split freeze: `data/splits/v1/SHA256SUMS.txt`.

---

## Log index

| file | what |
|---|---|
| `logs/model_A.md` | base checkpoint + dump shapes |
| `logs/baseline.md` | pretrained 130M behavior (all 0) |
| `logs/baseline_370m.md` | pretrained 370M zero-shot (all 0) |
| `logs/probes.md` | pretrained C1 null |
| `logs/baseline_ft.md` / `baseline_ft.json` | FT behavior + length drop |
| `logs/probes_ft.md` | FT query-token probes |
| `logs/intervene.md` | first AR causal table (no query-noise / donor / L-sweep) |
| `logs/intervene_ar.md` / `.json` | **canonical** AR C2/C3 + L0–L23 sweep |
| `logs/intervene_atr_short.md` | ATR short |
| `logs/intervene_atr_mid.md` | ATR mid |
| `logs/probes_over_t.md` | n=384 store vs readout |
| `logs/failure.md` | C4 pad/stuff + Δ-clamp |
| `logs/failure_pad_only.md` / `failure_filler_ops.md` | C1–C4 pad vs filler ops |
| `logs/failure_delta_roles.md` | C5 L17 Δ by token role |
| `logs/l17_restore.md` | L17 \(h\) restore after late wipe |
| `logs/l17_channels.md` | single-\(N\) wipes, all ~clean |
| `logs/l17_keypatch.md` | cross-seq L17 patch, donor V = 0 |
| `logs/l17_emergence.md` | L17 wipe vs FT step 200–800 |
| `logs/l17_wipe_s2.md` | seed 2 L17 write-wipe |
| `logs/failure_s2.md` | seed 2 C4 pad/stuff clamp |
| `logs/l17_restore_residual.md` | A1 \(h\) vs residual restore (AR) |
| `logs/l17_restore_residual_atr.md` | A1 on ATR-short |
| `logs/l17_restore_time.md` | A2 restore at value / last write / query |
| `logs/l17_restore_neighbors.md` | A3 restore L16 / L17 / L18 |
| `logs/l17_restore_s2.md` | B1 seed-2 h vs residual restore |
| `logs/l17_neighbors_s2.md` | B2 seed-2 L16/L17/L18 wipe |
| `logs/l17_wipe_atr.md` | B3 ATR-short L17-only wipe |
| `logs/l17_wipe_s3.md` | seed 3 L17 write-wipe |
| `logs/failure_s3.md` | seed 3 C4 pad/stuff clamp |
| `logs/l17_emergence_dense.md` | L17 wipe vs FT step 50–200 |
| `logs/l17_channels_k.md` | E1 multi-\(N\) wipes k=2/4/8 |
| `logs/l17_svd.md` | E2 SVD zero/keep top-k at last write |
| `logs/l17_C_patch.md` | E3 key-matched \(C\) @ query |
| `logs/probes_mlp_lastwrite.md` | E4 MLP vs linear @ last_write |
| `logs/l17_multiquery.md` | E5 2-query AR + L17 wipe |
| `logs/control_maps.md` | E6 implicit \(h{\cdot}C\) softmax mass |
| `data/splits/v2/ar_mq_eval.jsonl` | E5 multi-query split (v1 frozen) |
| `logs/LOCK.md` | SHA256 freeze of v1/v2 + locked logs |
| `logs/demo.md` | n=16 demo - **not paper** |
| `runs/ar_ft/train_meta.json` | FT hyperparameters |

Smoke leftovers (`intervene_smoke.md`, strong suite with `--max-examples 16`) are invalid for claims.

---

## Reproduce (full suite)

```powershell
python scripts\finetune_ar.py
python scripts\run_baseline.py --model-id runs\ar_ft\checkpoint --out-json logs\baseline_ft.json --out-md logs\baseline_ft.md
python scripts\run_probes.py --model-id runs\ar_ft\checkpoint --out-json logs\probes_ft.json --out-md logs\probes_ft.md
python scripts\run_strong.py --model-id runs\ar_ft\checkpoint
```

Ignore Triton CUDA_HOME / TensorFlow / “fast path” warnings on this Windows stack.
