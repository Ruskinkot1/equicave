# Ranker results (native3)

1367 structures, 1017 clusters, 40986 candidates, ceiling 0.977; 272 features; 5-fold CV by 30 %-identity cluster, 3 seeds; 95 % CI by cluster bootstrap; success DCA <= 4 A; graded relevance True, within-structure z-scores True.

| method | top-1 | top-3 | top-N | top-(N+2) | MRR | gain top-1 vs native order |
|---|---|---|---|---|---|---|
| native order (detector score) | 0.523 [0.494, 0.552] | 0.747 [0.721, 0.773] | 0.634 [0.607, 0.663] | 0.800 [0.777, 0.825] | 0.657 [0.634, 0.679] | +0.000 [+0.000, +0.000] |
| largest cavity first | 0.207 [0.185, 0.232] | 0.383 [0.355, 0.412] | 0.296 [0.270, 0.324] | 0.443 [0.413, 0.474] | 0.345 [0.325, 0.368] | -0.316 [-0.350, -0.282] |
| most buried first | 0.368 [0.339, 0.396] | 0.695 [0.668, 0.724] | 0.504 [0.475, 0.534] | 0.762 [0.737, 0.787] | 0.556 [0.536, 0.578] | -0.155 [-0.191, -0.120] |
| LightGBM LambdaRank (272 features) | 0.796 [0.774, 0.818] | 0.894 [0.876, 0.911] | 0.841 [0.821, 0.861] | 0.908 [0.892, 0.923] | 0.851 [0.834, 0.868] | +0.273 [+0.244, +0.302] |
| LightGBM LambdaRank (272 features), seed ensemble | 0.797 [0.774, 0.820] | 0.890 [0.872, 0.907] | 0.841 [0.820, 0.861] | 0.908 [0.891, 0.924] | 0.851 [0.834, 0.869] | +0.274 [+0.244, +0.304] |
| LightGBM LambdaRank (272 features), calibrated | 0.796 [0.773, 0.819] | 0.890 [0.871, 0.907] | 0.841 [0.821, 0.861] | 0.908 [0.891, 0.924] | 0.851 [0.834, 0.869] | +0.273 [+0.243, +0.302] |

Calibration of the seed-ensemble score (isotonic, fitted out of fold): ECE 0.0018 after versus 0.0148 for a plain sigmoid of the score; Brier 0.0658 versus 0.0664; base rate 0.162, mean predicted 0.162.
