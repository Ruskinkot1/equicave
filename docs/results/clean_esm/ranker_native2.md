# Ranker results (native2)

432 structures, 383 clusters, 12955 candidates, ceiling 0.988; 288 features; 5-fold CV by 30 %-identity cluster, 3 seeds; 95 % CI by cluster bootstrap; success DCA <= 4 A; graded relevance True, within-structure z-scores True.

| method | top-1 | top-3 | top-N | top-(N+2) | MRR | gain top-1 vs native order |
|---|---|---|---|---|---|---|
| native order (detector score) | 0.516 [0.465, 0.565] | 0.745 [0.701, 0.787] | 0.641 [0.594, 0.684] | 0.812 [0.774, 0.851] | 0.654 [0.615, 0.691] | +0.000 [+0.000, +0.000] |
| largest cavity first | 0.181 [0.141, 0.220] | 0.354 [0.303, 0.400] | 0.278 [0.230, 0.324] | 0.428 [0.376, 0.478] | 0.319 [0.281, 0.355] | -0.336 [-0.396, -0.272] |
| most buried first | 0.352 [0.306, 0.402] | 0.685 [0.639, 0.732] | 0.525 [0.478, 0.577] | 0.766 [0.724, 0.807] | 0.549 [0.515, 0.586] | -0.164 [-0.225, -0.098] |
| LightGBM LambdaRank (288 features) | 0.802 [0.763, 0.842] | 0.907 [0.880, 0.934] | 0.846 [0.810, 0.881] | 0.929 [0.904, 0.953] | 0.863 [0.834, 0.891] | +0.286 [+0.238, +0.337] |
| LightGBM LambdaRank (288 features), seed ensemble | 0.799 [0.757, 0.840] | 0.905 [0.877, 0.934] | 0.847 [0.809, 0.883] | 0.924 [0.898, 0.949] | 0.861 [0.832, 0.889] | +0.282 [+0.234, +0.333] |
| LightGBM LambdaRank (288 features), calibrated | 0.794 [0.753, 0.833] | 0.905 [0.877, 0.934] | 0.854 [0.819, 0.887] | 0.924 [0.898, 0.949] | 0.860 [0.831, 0.888] | +0.278 [+0.230, +0.328] |

Calibration of the seed-ensemble score (isotonic, fitted out of fold): ECE 0.0073 after versus 0.0387 for a plain sigmoid of the score; Brier 0.0703 versus 0.0737; base rate 0.181, mean predicted 0.181.
