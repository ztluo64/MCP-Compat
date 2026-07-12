# Final Paper Tables and Sources

All thresholds are selected once on the validation set by maximizing Macro-F1 and then fixed for test evaluation.

## Table 1. COinCO-style baselines and structured variants

| Type | Method | Th. | Original Acc | Original Bal Acc | Original Macro-F1 | Original AUC | Balanced Acc | Balanced Macro-F1 | Balanced AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COinCO-style | COinCO-VisualNet* | 0.45 | 68.69 | 55.94 | 55.95 | 62.49 | 56.65 | 53.90 | 63.69 |
| COinCO-style | COinCO-SemanticNet* | 0.37 | 76.94 | 68.22 | 67.99 | 78.60 | 68.35 | 67.47 | 77.86 |
| COinCO-style | COinCO-VisSemanticNet* | 0.42 | 74.65 | 68.67 | 67.06 | 76.59 | 68.08 | 67.72 | 75.19 |
| Ours variant | Tri-view + Text | 0.38 | 81.43 | 75.73 | 74.80 | 85.18 | 74.28 | 74.06 | 83.52 |
| Ours | Ours | 0.37 | 81.93 | 76.93 | 75.71 | 85.45 | 75.99 | 75.82 | 84.05 |

## Table 2. Input decomposition ablation on original test

| Method | Th. | Acc | Bal Acc | Macro-F1 | F1-0 | F1-1 | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full only | 0.42 | 75.85 | 63.87 | 64.46 | 44.34 | 84.58 | 72.72 |
| Object crop only | 0.36 | 75.77 | 68.21 | 67.38 | 50.84 | 83.92 | 78.17 |
| Masked context only | 0.44 | 70.61 | 59.13 | 59.05 | 37.30 | 80.80 | 66.63 |
| Tri-view | 0.33 | 78.06 | 69.51 | 69.38 | 53.07 | 85.68 | 80.22 |
| Tri-view + Text | 0.38 | 81.43 | 75.73 | 74.80 | 61.88 | 87.73 | 85.18 |
| Ours | 0.37 | 81.93 | 76.93 | 75.71 | 63.41 | 88.00 | 85.45 |

## Table 3. Prior compatibility ablation on balanced test

| Method | Prior module | λ_prior | Th. | Acc | Bal Acc | Macro-F1 | F1-0 | F1-1 | AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tri-view + Text | none | - | 0.38 | 74.28 | 74.28 | 74.06 | 71.68 | 76.44 | 83.52 |
| + Decoupled Prior Prediction | prior prediction only, not used by classifier | 0.20 | 0.33 | 73.83 | 73.83 | 73.60 | 71.16 | 76.05 | 82.65 |
| + Latent Prior Compat. | latent prior-text compatibility | 0.10 | 0.38 | 73.47 | 73.47 | 73.21 | 70.59 | 75.84 | 83.50 |
| + Latent Prior Compat. | latent prior-text compatibility | 0.20 | 0.34 | 75.27 | 75.27 | 74.97 | 72.25 | 77.70 | 84.15 |
| Ours | latent + explicit scalar compatibility | 0.20 | 0.37 | 75.99 | 75.99 | 75.82 | 73.80 | 77.84 | 84.05 |
