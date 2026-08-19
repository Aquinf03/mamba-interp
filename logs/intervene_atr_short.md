# Causal interventions (atr)

| name | layers | window | n | acc | ci95 | mean_lp | acc\|clean_ok | Δlp\|clean_ok | donor_acc |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| clean | 16-23 | none | 128 | 0.961 | 0.922–0.992 | -0.268 |  |  |  |
| zero_h_write | 16-23 | write | 128 | 0.039 | 0.008–0.078 | -6.384 | 0.041 | -6.197 |  |
| swap_h_write_end | 16-23 | write | 128 | 0.008 | 0.000–0.023 | -11.158 | 0.008 | -11.070 | 0.167 |
| local_only | 16-23 | all | 128 | 0.000 | 0.000–0.000 | -50.085 | 0.000 | -50.150 |  |
| residual_noise_write | 16-23 | write | 128 | 0.961 | 0.922–0.992 | -0.266 | 1.000 | -0.002 |  |
| residual_noise_query | 16-23 | query | 128 | 0.812 | 0.742–0.883 | -0.863 | 0.837 | -0.535 |  |
## Mixer L2 budgets (zero_h_write vs clean)

{
  "zero_h_write_mixer_L2_write": {
    "mean": 30.2890625,
    "median": 5.5860371589660645,
    "p90": 86.63245391845703,
    "n": 7168
  },
  "zero_h_write_mixer_L2_query": {
    "mean": 60.258602142333984,
    "median": 30.373619079589844,
    "p90": 151.67221069335938,
    "n": 1024
  }
}

## Notes

- model: `runs\ar_ft\checkpoint`
- split: `C:\Users\fresh\OneDrive\Desktop\mamba\data\splits\v1\atr_short_eval.jsonl`
- peak_vram_gb: 0.255
- wall_s: 38.2
- residual_noise_write / _query: random residual direction, L2 matched to zero_h_write mixer delta on that window
- swap donor_acc: P(pred = next example's value), excluding target collisions
- ci95: bootstrap over examples
