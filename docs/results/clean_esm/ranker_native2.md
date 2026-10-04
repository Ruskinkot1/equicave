# Ranker results (native2)

432 structures, 383 clusters, 12955 candidates, ceiling 0.988; 288 features; 5-fold CV by 30 %-identity cluster, 2 seeds; 95 % CI by cluster bootstrap; success DCA <= 4 A; graded relevance True, within-structure z-scores True.

| method | top-1 | top-3 | top-N | top-(N+2) | MRR | gain top-1 vs native order |
|---|---|---|---|---|---|---|
| native order (detector score) | 0.516 [0.465, 0.565] | 0.745 [0.701, 0.787] | 0.641 [0.594, 0.684] | 0.812 [0.774, 0.851] | 0.654 [0.615, 0.691] | +0.000 [+0.000, +0.000] |
| largest cavity first | 0.181 [0.141, 0.220] | 0.354 [0.303, 0.400] | 0.278 [0.230, 0.324] | 0.428 [0.376, 0.478] | 0.319 [0.281, 0.355] | -0.336 [-0.396, -0.272] |
| most buried first | 0.352 [0.306, 0.402] | 0.685 [0.639, 0.732] | 0.525 [0.478, 0.577] | 0.766 [0.724, 0.807] | 0.549 [0.515, 0.586] | -0.164 [-0.225, -0.098] |
| LightGBM LambdaRank (288 features) | 0.809 [0.770, 0.847] | 0.910 [0.882, 0.937] | 0.856 [0.820, 0.890] | 0.928 [0.903, 0.953] | 0.867 [0.838, 0.894] | +0.293 [+0.244, +0.344] |
| LightGBM LambdaRank (288 features), seed ensemble | 0.806 [0.764, 0.845] | 0.914 [0.888, 0.941] | 0.854 [0.817, 0.889] | 0.928 [0.902, 0.953] | 0.865 [0.836, 0.893] | +0.289 [+0.241, +0.340] |
| LightGBM LambdaRank (288 features), calibrated | 0.796 [0.754, 0.835] | 0.910 [0.883, 0.937] | 0.850 [0.813, 0.885] | 0.926 [0.900, 0.951] | 0.861 [0.832, 0.889] | +0.280 [+0.230, +0.332] |

Calibration of the seed-ensemble score (isotonic, fitted out of fold): ECE 0.0036 after versus 0.0387 for a plain sigmoid of the score; Brier 0.0709 versus 0.0739; base rate 0.181, mean predicted 0.181.
