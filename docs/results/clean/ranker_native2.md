# Ranker results (native2)

432 structures, 383 clusters, 12955 candidates, ceiling 0.988; 218 features; 5-fold CV by 30 %-identity cluster, 3 seeds; 95 % CI by cluster bootstrap; success DCA <= 4 A; graded relevance True, within-structure z-scores True.

| method | top-1 | top-3 | top-N | top-(N+2) | MRR | gain top-1 vs native order |
|---|---|---|---|---|---|---|
| native order (detector score) | 0.516 [0.465, 0.565] | 0.745 [0.701, 0.787] | 0.641 [0.594, 0.684] | 0.812 [0.774, 0.851] | 0.654 [0.615, 0.691] | +0.000 [+0.000, +0.000] |
| largest cavity first | 0.181 [0.141, 0.220] | 0.354 [0.303, 0.400] | 0.278 [0.230, 0.324] | 0.428 [0.376, 0.478] | 0.319 [0.281, 0.355] | -0.336 [-0.396, -0.272] |
| most buried first | 0.352 [0.306, 0.402] | 0.685 [0.639, 0.732] | 0.525 [0.478, 0.577] | 0.766 [0.724, 0.807] | 0.549 [0.515, 0.586] | -0.164 [-0.225, -0.098] |
| LightGBM LambdaRank (218 features) | 0.782 [0.740, 0.823] | 0.900 [0.871, 0.928] | 0.835 [0.798, 0.871] | 0.925 [0.900, 0.950] | 0.848 [0.818, 0.877] | +0.265 [+0.217, +0.318] |
| LightGBM LambdaRank (218 features), seed ensemble | 0.792 [0.750, 0.832] | 0.898 [0.868, 0.928] | 0.845 [0.808, 0.880] | 0.928 [0.903, 0.953] | 0.853 [0.822, 0.883] | +0.275 [+0.228, +0.329] |
| LightGBM LambdaRank (218 features), calibrated | 0.787 [0.743, 0.829] | 0.900 [0.872, 0.930] | 0.847 [0.809, 0.883] | 0.928 [0.903, 0.953] | 0.852 [0.821, 0.881] | +0.271 [+0.223, +0.325] |

Calibration of the seed-ensemble score (isotonic, fitted out of fold): ECE 0.0040 after versus 0.0335 for a plain sigmoid of the score; Brier 0.0762 versus 0.0786; base rate 0.181, mean predicted 0.180.
