# E4 MLP vs linear at last_write L17

fresh AR n=384 seed=99, not v1. Chance ~ 1/n_classes. Linear last_write was ~0.10 in probes_over_t.

| feature | dim | linear train | linear test | MLP train | MLP test | n_classes |
|---|---:|---:|---:|---:|---:|---:|
| residual | 768 | 1.000 | 0.052 | 1.000 | 0.062 | 52 |
| h_mean_n | 1536 | 0.951 | 0.010 | 1.000 | 0.031 | 52 |
| h_flat | 24576 | 0.927 | 0.042 | 1.000 | 0.042 | 52 |

**Keep:** MLP test stays at chance (~0.03–0.06 on 52-way; train=1.0). Last-write L17 \(h\) / residual has no easily decoded value code. Superposition / key-addressing still holds vs a small MLP.
