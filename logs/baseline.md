# Behavioral baseline (Model A)

Run: 2026-08-08 · `state-spaces/mamba-130m-hf` · fp16 · peak VRAM 0.592 GB · wall 15.8 s

Metric: teacher-forced next-token top-1 on frozen `data/splits/v1`.

| name | task | ablation | n | mean_len | acc | mean_lp(target) |
|---|---|---|---:|---:|---:|---:|
| ar | ar | none | 128 | 9.0 | 0.000 | -6.392 |
| atr_short | atr | none | 128 | 8.0 | 0.000 | -6.108 |
| atr_mid | atr | none | 128 | 19.0 | 0.000 | -7.026 |
| len_short | length | none | 64 | 5.0 | 0.000 | -5.437 |
| len_mid | length | none | 64 | 17.0 | 0.000 | -7.157 |
| len_ood_pad | length | none | 64 | 41.0 | 0.000 | -8.196 |
| stuff_light | length | none | 64 | 15.0 | 0.000 | -7.339 |
| stuff_heavy | length | none | 64 | 45.0 | 0.000 | -8.069 |
| factual | factual | none | 8 | 6.4 | 0.000 | -10.000 |
| ar | ar | ablate_early | 128 | 9.0 | 0.000 | -10.171 |
| ar | ar | ablate_mid | 128 | 9.0 | 0.023 | -7.097 |
| ar | ar | ablate_late | 128 | 9.0 | 0.000 | -77.514 |
| atr_short | atr | ablate_early | 128 | 8.0 | 0.000 | -10.604 |
| atr_short | atr | ablate_mid | 128 | 8.0 | 0.000 | -6.740 |
| atr_short | atr | ablate_late | 128 | 8.0 | 0.000 | -79.431 |
| stuff_heavy | length | ablate_early | 64 | 45.0 | 0.000 | -9.023 |
| stuff_heavy | length | ablate_mid | 64 | 45.0 | 0.000 | -8.044 |
| stuff_heavy | length | ablate_late | 64 | 45.0 | 0.000 | -79.212 |

## Interpretation (for paper design)

1. **Zero-shot synthetic AR/ATR is at floor on this checkpoint.** Expected for random letter→letter binds on a generic Pile-trained 130M. **Do not treat this as a failed experiment** - it means behavioral circuits on vanilla Model A need either task finetuning or tasks the LM already solves.
2. **Logprobs move with length/stuffing** (≈ −5.4 short → −8.1 stuffed/OOD). Targets become less favored as sequences get denser/longer - soft length-pressure signal even without accuracy above chance.
3. **Late-layer mixer ablation destroys LM head** (lp ≈ −77). Late residual pathway is critical for *any* next-token decoding. Mid ablation noise (acc 0.023) is not meaningful.
4. **Phase 5 implication:** probes can still test whether \(h_t\) *encodes* keys/values under teacher-forced context even when free readout fails. Causal claims (C2) that measure **answer accuracy** require a competent behavior (finetune small adapter / continue-train on AR, or switch to a natural task with non-zero baseline).

## Notes

- ablation: listed mixers become identity (residual pass-through)
- source: `logs/baseline.json`
