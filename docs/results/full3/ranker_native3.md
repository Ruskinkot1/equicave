# Ranker results (native3)

1367 structures, 1017 clusters, 40986 candidates, ceiling 0.977; 236 features; 5-fold CV by 30 %-identity cluster, 3 seeds; 95 % CI by cluster bootstrap; success DCA <= 4 A; graded relevance True, within-structure z-scores True.

| method | top-1 | top-3 | top-N | top-(N+2) | MRR | gain top-1 vs native order |
|---|---|---|---|---|---|---|
| native order (detector score) | 0.523 [0.494, 0.552] | 0.747 [0.721, 0.773] | 0.634 [0.607, 0.663] | 0.800 [0.777, 0.825] | 0.657 [0.634, 0.679] | +0.000 [+0.000, +0.000] |
| largest cavity first | 0.207 [0.185, 0.232] | 0.383 [0.355, 0.412] | 0.296 [0.270, 0.324] | 0.443 [0.413, 0.474] | 0.345 [0.325, 0.368] | -0.316 [-0.350, -0.282] |
| most buried first | 0.368 [0.339, 0.396] | 0.695 [0.668, 0.724] | 0.504 [0.475, 0.534] | 0.762 [0.737, 0.787] | 0.556 [0.536, 0.578] | -0.155 [-0.191, -0.120] |
| LightGBM LambdaRank (236 features) | 0.787 [0.764, 0.808] | 0.890 [0.871, 0.908] | 0.826 [0.805, 0.847] | 0.905 [0.888, 0.922] | 0.846 [0.828, 0.863] | +0.264 [+0.235, +0.292] |
| LightGBM LambdaRank (236 features), seed ensemble | 0.791 [0.767, 0.813] | 0.890 [0.871, 0.909] | 0.825 [0.803, 0.847] | 0.908 [0.891, 0.925] | 0.848 [0.830, 0.866] | +0.268 [+0.238, +0.297] |
| LightGBM LambdaRank (236 features), calibrated | 0.791 [0.767, 0.814] | 0.892 [0.873, 0.910] | 0.827 [0.806, 0.849] | 0.910 [0.893, 0.927] | 0.849 [0.831, 0.867] | +0.268 [+0.239, +0.297] |

Calibration of the seed-ensemble score (isotonic, fitted out of fold): ECE 0.0020 after versus 0.0098 for a plain sigmoid of the score; Brier 0.0655 versus 0.0657; base rate 0.162, mean predicted 0.162.
