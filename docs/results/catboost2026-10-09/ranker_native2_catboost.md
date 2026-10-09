# Ranker results (native2)

1367 structures, 1017 clusters, 40986 candidates, ceiling 0.977; 204 features; 5-fold CV by 30 %-identity cluster, 4 seeds; 95 % CI by cluster bootstrap; success DCA <= 4 A; graded relevance True, within-structure z-scores True.

| method | top-1 | top-3 | top-N | top-(N+2) | MRR | gain top-1 vs native order |
|---|---|---|---|---|---|---|
| native order (detector score) | 0.523 [0.494, 0.552] | 0.747 [0.721, 0.773] | 0.634 [0.607, 0.663] | 0.800 [0.777, 0.825] | 0.657 [0.634, 0.679] | +0.000 [+0.000, +0.000] |
| largest cavity first | 0.207 [0.185, 0.232] | 0.383 [0.355, 0.412] | 0.296 [0.270, 0.324] | 0.443 [0.413, 0.474] | 0.345 [0.325, 0.368] | -0.316 [-0.350, -0.282] |
| most buried first | 0.368 [0.339, 0.396] | 0.695 [0.668, 0.724] | 0.504 [0.475, 0.534] | 0.762 [0.737, 0.787] | 0.556 [0.536, 0.578] | -0.155 [-0.191, -0.120] |
| CatBoost YetiRank (204 features) | 0.775 [0.751, 0.798] | 0.882 [0.863, 0.901] | 0.817 [0.795, 0.839] | 0.898 [0.881, 0.916] | 0.837 [0.818, 0.855] | +0.252 [+0.223, +0.281] |
| CatBoost YetiRank (204 features), seed ensemble | 0.774 [0.750, 0.797] | 0.881 [0.861, 0.899] | 0.817 [0.794, 0.839] | 0.899 [0.881, 0.917] | 0.836 [0.817, 0.855] | +0.251 [+0.221, +0.281] |
| CatBoost YetiRank (204 features), calibrated | 0.775 [0.751, 0.798] | 0.884 [0.864, 0.903] | 0.821 [0.798, 0.843] | 0.898 [0.880, 0.916] | 0.837 [0.818, 0.855] | +0.252 [+0.223, +0.282] |

Calibration of the seed-ensemble score (isotonic, fitted out of fold): ECE 0.0021 after versus 0.3101 for a plain sigmoid of the score; Brier 0.0691 versus 0.1885; base rate 0.162, mean predicted 0.162.
