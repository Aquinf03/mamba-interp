# Causal interventions (atr)

| name | layers | window | n | acc | ci95 | mean_lp | acc\|clean_ok | Δlp\|clean_ok | donor_acc |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| clean | 16-23 | none | 128 | 0.844 | 0.773–0.898 | -0.712 |  |  |  |
| zero_h_write | 16-23 | write | 128 | 0.039 | 0.008–0.078 | -6.126 | 0.046 | -5.788 |  |
| swap_h_write_end | 16-23 | write | 128 | 0.016 | 0.000–0.039 | -11.207 | 0.009 | -11.248 | 0.071 |
| local_only | 16-23 | all | 128 | 0.000 | 0.000–0.000 | -50.227 | 0.000 | -49.831 |  |
| residual_noise_write | 16-23 | write | 128 | 0.836 | 0.766–0.898 | -0.751 | 0.991 | -0.007 |  |
| residual_noise_query | 16-23 | query | 128 | 0.680 | 0.594–0.766 | -2.289 | 0.787 | -1.054 |  |
## Mixer L2 budgets (zero_h_write vs clean)

{
  "zero_h_write_mixer_L2_write": {
    "mean": 38.88886260986328,
    "median": 14.200241088867188,
    "p90": 98.83445739746094,
    "n": 18432
  },
  "zero_h_write_mixer_L2_query": {
    "mean": 68.78827667236328,
    "median": 32.08912658691406,
    "p90": 170.66151428222656,
    "n": 1024
  }
}

## Notes

- model: `runs\ar_ft\checkpoint`
- split: `C:\Users\fresh\OneDrive\Desktop\mamba\data\splits\v1\atr_mid_eval.jsonl`
- peak_vram_gb: 0.260
- wall_s: 46.2
- residual_noise_write / _query: random residual direction, L2 matched to zero_h_write mixer delta on that window
- swap donor_acc: P(pred = next example's value), excluding target collisions
- ci95: bootstrap over examples
