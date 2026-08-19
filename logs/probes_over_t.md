# Probes over t (store vs readout)

model: `runs\ar_ft\checkpoint`  n=384 (fresh AR, seed=99, not frozen v1 eval)

| layer | site | feature | dim | train_acc | test_acc | n_classes |
|---:|---|---|---:|---:|---:|---:|
| 0 | first | delta | 1536 | 0.153 | 0.000 | 52 |
| 0 | first | h_mean_n | 1536 | 0.205 | 0.021 | 52 |
| 0 | first | residual | 768 | 0.219 | 0.021 | 52 |
| 0 | last_write | delta | 1536 | 0.747 | 0.031 | 52 |
| 0 | last_write | h_mean_n | 1536 | 1.000 | 0.042 | 52 |
| 0 | last_write | residual | 768 | 0.274 | 0.052 | 52 |
| 0 | query | delta | 1536 | 0.771 | 0.031 | 52 |
| 0 | query | h_mean_n | 1536 | 1.000 | 0.042 | 52 |
| 0 | query | residual | 768 | 0.198 | 0.010 | 52 |
| 0 | value_write | delta | 1536 | 0.688 | 0.208 | 52 |
| 0 | value_write | h_mean_n | 1536 | 0.951 | 0.042 | 52 |
| 0 | value_write | residual | 768 | 1.000 | 1.000 | 52 |
| 11 | first | delta | 1536 | 0.174 | 0.010 | 52 |
| 11 | first | h_mean_n | 1536 | 0.191 | 0.000 | 52 |
| 11 | first | residual | 768 | 0.208 | 0.000 | 52 |
| 11 | last_write | delta | 1536 | 0.938 | 0.073 | 52 |
| 11 | last_write | h_mean_n | 1536 | 0.993 | 0.042 | 52 |
| 11 | last_write | residual | 768 | 1.000 | 0.062 | 52 |
| 11 | query | delta | 1536 | 0.500 | 0.010 | 52 |
| 11 | query | h_mean_n | 1536 | 1.000 | 0.062 | 52 |
| 11 | query | residual | 768 | 0.990 | 0.010 | 52 |
| 11 | value_write | delta | 1536 | 0.976 | 0.417 | 52 |
| 11 | value_write | h_mean_n | 1536 | 0.858 | 0.052 | 52 |
| 11 | value_write | residual | 768 | 1.000 | 1.000 | 52 |
| 16 | first | delta | 1536 | 0.184 | 0.031 | 52 |
| 16 | first | h_mean_n | 1536 | 0.198 | 0.021 | 52 |
| 16 | first | residual | 768 | 0.219 | 0.031 | 52 |
| 16 | last_write | delta | 1536 | 0.931 | 0.021 | 52 |
| 16 | last_write | h_mean_n | 1536 | 0.990 | 0.052 | 52 |
| 16 | last_write | residual | 768 | 0.993 | 0.052 | 52 |
| 16 | query | delta | 1536 | 0.958 | 0.031 | 52 |
| 16 | query | h_mean_n | 1536 | 1.000 | 0.073 | 52 |
| 16 | query | residual | 768 | 1.000 | 0.146 | 52 |
| 16 | value_write | delta | 1536 | 0.927 | 0.104 | 52 |
| 16 | value_write | h_mean_n | 1536 | 0.976 | 0.042 | 52 |
| 16 | value_write | residual | 768 | 1.000 | 1.000 | 52 |
| 23 | first | delta | 1536 | 0.194 | 0.031 | 52 |
| 23 | first | h_mean_n | 1536 | 0.201 | 0.021 | 52 |
| 23 | first | residual | 768 | 0.201 | 0.010 | 52 |
| 23 | last_write | delta | 1536 | 0.417 | 0.062 | 52 |
| 23 | last_write | h_mean_n | 1536 | 0.951 | 0.104 | 52 |
| 23 | last_write | residual | 768 | 0.993 | 0.073 | 52 |
| 23 | query | delta | 1536 | 0.865 | 0.562 | 52 |
| 23 | query | h_mean_n | 1536 | 0.997 | 0.885 | 52 |
| 23 | query | residual | 768 | 1.000 | 0.979 | 52 |
| 23 | value_write | delta | 1536 | 0.993 | 0.823 | 52 |
| 23 | value_write | h_mean_n | 1536 | 1.000 | 0.875 | 52 |
| 23 | value_write | residual | 768 | 0.983 | 0.844 | 52 |

## Best test acc by site × feature (any layer)
- **first / delta**: 0.031
- **first / h_mean_n**: 0.021
- **first / residual**: 0.031
- **last_write / delta**: 0.073
- **last_write / h_mean_n**: 0.104
- **last_write / residual**: 0.073
- **query / delta**: 0.562
- **query / h_mean_n**: 0.885
- **query / residual**: 0.979
- **value_write / delta**: 0.823
- **value_write / h_mean_n**: 0.875
- **value_write / residual**: 1.000
