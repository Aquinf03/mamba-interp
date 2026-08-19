# C4 pad vs filler ops

model: `runs\ar_ft\checkpoint`
fix layers: 16-23  large_delta=5.0

| split | n | clean | clamp Δ pad=0 | clamp Δ filler=0 | zero B filler | Δ filler=5 |
|---|---:|---:|---:|---:|---:|---:|
| len_short | 64 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| len_mid | 64 | 0.922 | 0.922 | 0.922 | 0.922 | 0.922 |
| len_ood_pad | 64 | 0.484 | 0.703 | 0.484 | 0.484 | 0.484 |
| stuff_light | 64 | 0.922 | 0.922 | 0.328 | 0.328 | 0.000 |
| stuff_heavy | 64 | 0.578 | 0.578 | 0.109 | 0.125 | 0.000 |

C1: clamp Δ←0 on **pad** only. Expect help on len_ood_pad; no-op elsewhere.
C2: clamp Δ←0 on **filler** only. Expect hurt on stuffing (fillers are used).
C3: zero B on fillers (block writes from stuffing tokens).
C4: large Δ on fillers (force forget). Opposite of C2.
Empty windows are no-ops (acc should match clean).
