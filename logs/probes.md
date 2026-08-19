# Linear probes (C1) - AR value @ query token

Run: 2026-08-08 · `state-spaces/mamba-130m-hf` · `ar_eval.jsonl` n=128 · 40-way over train targets  
Chance ≈ 1/40 = **0.025**

| layer | feature | dim | train_acc | test_acc | n_classes |
|---:|---|---:|---:|---:|---:|
| 0 | delta | 1536 | 0.927 | 0.036 | 40 |
| 0 | h_flat | 24576 | 0.979 | 0.036 | 40 |
| 0 | h_mean_e | 16 | 0.708 | 0.000 | 40 |
| 0 | h_mean_n | 1536 | 0.958 | 0.000 | 40 |
| 0 | residual | 768 | 0.438 | 0.036 | 40 |
| 0 | residual_out | 768 | 0.938 | 0.036 | 40 |
| 11 | delta | 1536 | 1.000 | 0.036 | 40 |
| 11 | h_flat | 24576 | 0.958 | 0.036 | 40 |
| 11 | h_mean_e | 16 | 0.771 | 0.071 | 40 |
| 11 | h_mean_n | 1536 | 1.000 | 0.000 | 40 |
| 11 | residual | 768 | 0.969 | 0.036 | 40 |
| 11 | residual_out | 768 | 1.000 | 0.036 | 40 |
| 23 | delta | 1536 | 0.990 | 0.036 | 40 |
| 23 | h_flat | 24576 | 0.917 | 0.107 | 40 |
| 23 | h_mean_e | 16 | 0.521 | 0.000 | 40 |
| 23 | h_mean_n | 1536 | 0.979 | 0.071 | 40 |
| 23 | residual | 768 | 1.000 | 0.143 | 40 |
| 23 | residual_out | 768 | 1.000 | 0.000 | 40 |

## Best test acc by feature (any layer)

- residual: **0.143**
- h_flat: 0.107
- h_mean_e / h_mean_n: 0.071
- residual_out / delta: 0.036

## Interpretation

1. **Huge train≫test gap** → probes mostly memorize (high-dim, small n, 40 classes). Treat test numbers as soft upper bounds on linear info, not solid evidence.
2. **No support for C1 yet.** Best state feature (h_flat 0.107) does **not** beat residual (0.143). Slight residual edge at L23 is weak (~5–6× chance) and may be noise.
3. Consistent with Phase 4 floor: pretrained Model A is not solving AR, and is not exposing a clean linear value code in \(h\) that generalizes.
4. **Mandatory next for a paper with causal claims:** make the model *use* AR (light finetune / task model), then re-probe and intervene. Probing a non-performing model only documents a null.

## C1 checkpoint

| Criterion | Result |
|---|---|
| State test-acc ≫ residual | **Fail** (null / residual ≥ state) |
| Clear above chance with generalization | **Marginal at best** (overfit) |
| Proceed to Phase 6 interventions on accuracy | **No** until non-zero task behavior |
