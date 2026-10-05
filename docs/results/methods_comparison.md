# Candidate generators and rankers on the same structures

1354 structures common to fpocket, native3, p2rank; labels, ligand filter and n_sites are identical for every method (DCA <= 4 A). Rankers are cross-validated by 30 %-identity cluster with 3 seeds, so no structure is scored by a model that saw it. 95 % CI by cluster bootstrap.

| method | candidates | ceiling | top-1 | top-3 | top-N | top-(N+2) | MRR |
|---|---|---|---|---|---|---|---|
| native3 (own order) | 30.0 | 0.979 | 0.526 [0.497, 0.556] | 0.751 [0.726, 0.778] | 0.638 [0.609, 0.667] | 0.805 [0.781, 0.829] | 0.660 [0.637, 0.683] |
| native3 + LambdaRank | 30.0 | 0.979 | 0.790 [0.767, 0.814] | 0.889 [0.871, 0.907] | 0.836 [0.814, 0.857] | 0.905 [0.888, 0.922] | 0.848 [0.830, 0.866] |
| fpocket (own order) | 38.1 | 0.960 | 0.393 [0.365, 0.420] | 0.575 [0.545, 0.602] | 0.479 [0.450, 0.507] | 0.620 [0.592, 0.648] | 0.514 [0.490, 0.536] |
| p2rank (own order) | 9.1 | 0.914 | 0.754 [0.727, 0.780] | 0.866 [0.845, 0.886] | 0.821 [0.797, 0.844] | 0.886 [0.865, 0.904] | 0.813 [0.791, 0.834] |

Paired differences against the first method listed (positive = better):

| method | top-1 difference | top-(N+2) difference |
|---|---|---|
| native3 + LambdaRank | +0.264 [+0.236, +0.293] | +0.100 [+0.079, +0.122] |
| fpocket (own order) | -0.133 [-0.171, -0.098] | -0.185 [-0.217, -0.153] |
| p2rank (own order) | +0.228 [+0.200, +0.257] | +0.081 [+0.058, +0.103] |
