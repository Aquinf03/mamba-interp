# Goated-paper checklist

This file is the lock of finished goat runs. Do not re-do them. Remaining optional work: venue tex (`paper/draft.md`).

Locked results: `logs/findings.md`. New runs → **new log filenames**. Hardware: 4070 8 GB, `runs/ar_ft/checkpoint` unless noted.

---

## Now (makes the paper)

- [x] **L17 restore (sufficiency).** Zero late \(h\) → 0.242; restore clean L17 \(h\) → **0.961**. Necessary and sufficient. `logs/l17_restore.md`.
- [x] **Key-matched value patch.** Donor V = **0.000**; original V unchanged (0.984). Not a portable slot. `logs/l17_keypatch.md`.
- [x] **L17 state channels.** No single \(N\) channel matters (all ~0.96). Distributed in L17. `logs/l17_channels.md`.
- [x] **FT emergence.** Store present by step 200 (clean 0.938, wipe 0.367); wipe falls to 0.297 by 800. Learned in FT, already late at first snapshot. `logs/l17_emergence.md`.
- [x] **Second seed.** AR 0.961; L17 wipe **0.289** (s1 0.297). Pad clamp +0.078 (s1 +0.219); stuffing clamp still hurts. L17 replicates; pad *magnitude* does not. `logs/l17_wipe_s2.md`, `failure_s2.md`.

## Next (main-track teeth)

- [x] **Restore vs residual.** AR: \(h\) 0.961 vs residual 0.242 / 0.258. ATR-short: wipe 0.039 → \(h\) **0.945**, residual 0.039 / 0.062. `logs/l17_restore_residual.md`, `l17_restore_residual_atr.md`.
- [x] **When in time.** Value 0.969 / last write 0.961 / query 0.953. Bind present from value through query. `logs/l17_restore_time.md`.
- [x] **Neighbor restore.** L16 0.258, L18 0.242. `logs/l17_restore_neighbors.md`.
- [x] **Stuffing ≠ pad.** Pad-only Δ←0 = +0.219; filler-only Δ←0 hurts (0.328 / 0.109); zero_B fillers hurts; Δ=5 fillers → **0%**. Two failures. Stop hunting a stuffing fix. `logs/failure_pad_only.md`.
- [x] **370M zero-shot.** All v1 splits **0%**. Fits 8 GB. Finetune objection stands. `logs/baseline_370m.md`.
- [x] **E content/addressing.** k-wipes spread across \(N\); SVD rank-8 not rank-1; \(C\) patch donor V=0; MLP last_write chance; 2-query wipe 0.094; maps light **keys** not values. `logs/l17_channels_k.md`, `l17_svd.md`, `l17_C_patch.md`, `probes_mlp_lastwrite.md`, `l17_multiquery.md`, `control_maps.md`.
- [x] **Hidden Attention / LaTIM maps** as a *control table* (keys 0.444, queried value 0.111). Do not rebuild those papers. `logs/control_maps.md`.

## Writing (required for “extremely good”)

- [x] **One sentence in the abstract** + limitations in the abstract (130M, synthetic, finetuned, pad magnitude seed-sensitive, swap does not copy). `paper/draft.md`.
- [x] **Five figures:** (1) write–store–read, (2) L17 vs other layers, (3) wipe vs residual vs restore + ATR, (4) probes over \(t\), (5) pad recovery vs stuffing collapse (three seeds, sign). `figs/`, `scripts/make_figs.py`.
- [x] **Do not title C1** “\(h\) beats residual.” Title: store ≠ readout. Draft §4.
- [x] **Swap subsection:** necessity, not copy (donor_acc 0.072). Draft §7.
- [x] Cite Hidden Attention, IOI-in-Mamba, Lost in State Space, Primacy–Recency, subspace bottlenecks, Stuffed/LongMamba - as neighbors. Draft §11.

Remaining: venue tex polish if you want a submission PDF. G is done (`README.md`, `scripts/run_demo.py`, `logs/LOCK.md`).

## Do not do (dilutes L17)

- Residual SAE suites, full IOI circuit, Jamba/hybrids, vision SSMs, ROME-first, re-deriving SSM theory, factual (still 0%), more letter-AR variants that do not touch L17 / pad / addressing.

---

## Venue bar

| If you have | Venue |
|---|---|
| Current `findings.md` + `paper/draft.md` + `figs/` | **Strong workshop / oral-plausible** |
| + a **pretrained-competent** natural task with the same L17 story | Main-track plausible |
| 370M zero-shot on these synthetics | **Done, 0%** - does not unlock main track |
| SAE/IOI zoo without the above | Worse paper |
