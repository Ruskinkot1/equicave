# holo4k: our predictions against external tools on the same structures

3335 structures predicted by every method listed. Success is DCC <= 4 A to a ligand centroid of the set's own relevant-ligand list; N is the structure's own number of ligand sites. 95 % CI by 30 %-identity cluster bootstrap. `train-similar` means the structure shares a 30 %-identity cluster with **our** training manifest; it is leakage for us and not for the external tools, whose own training sets overlap this benchmark in ways this table does not measure. The comparable row for us is **not train-similar**.

| subset | method | predictions | ceiling | top-1 | top-3 | top-N | top-(N+2) | MRR | n |
|---|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 30.0 | 0.947 | 0.527 [0.485, 0.568] | 0.776 [0.748, 0.804] | 0.610 [0.573, 0.648] | 0.814 [0.790, 0.838] | 0.667 [0.638, 0.697] | 3335 |
| all | ours (native order) | 30.0 | 0.947 | 0.434 [0.387, 0.484] | 0.682 [0.643, 0.718] | 0.543 [0.502, 0.588] | 0.738 [0.706, 0.769] | 0.581 [0.545, 0.619] | 3335 |
| all | p2rank | 7.9 | 0.720 | 0.587 [0.543, 0.631] | 0.683 [0.641, 0.725] | 0.655 [0.613, 0.699] | 0.698 [0.656, 0.740] | 0.638 [0.595, 0.680] | 3335 |
| not train-similar | ours (ranker) | 30.0 | 0.947 | 0.494 [0.452, 0.539] | 0.748 [0.708, 0.783] | 0.582 [0.537, 0.629] | 0.795 [0.760, 0.826] | 0.641 [0.606, 0.674] | 1499 |
| not train-similar | ours (native order) | 30.0 | 0.947 | 0.454 [0.379, 0.524] | 0.674 [0.619, 0.724] | 0.568 [0.505, 0.626] | 0.731 [0.682, 0.773] | 0.589 [0.531, 0.645] | 1499 |
| not train-similar | p2rank | 8.2 | 0.766 | 0.613 [0.553, 0.672] | 0.711 [0.661, 0.760] | 0.690 [0.638, 0.740] | 0.736 [0.688, 0.783] | 0.668 [0.615, 0.720] | 1499 |
| train-similar | ours (ranker) | 30.0 | 0.947 | 0.553 [0.490, 0.618] | 0.800 [0.758, 0.837] | 0.633 [0.576, 0.687] | 0.830 [0.793, 0.863] | 0.688 [0.642, 0.733] | 1836 |
| train-similar | ours (native order) | 30.0 | 0.947 | 0.418 [0.349, 0.487] | 0.688 [0.631, 0.737] | 0.523 [0.462, 0.583] | 0.744 [0.695, 0.785] | 0.574 [0.523, 0.623] | 1836 |
| train-similar | p2rank | 7.7 | 0.682 | 0.565 [0.504, 0.623] | 0.660 [0.596, 0.718] | 0.626 [0.563, 0.684] | 0.666 [0.601, 0.724] | 0.613 [0.551, 0.670] | 1836 |

Paired differences against **p2rank** on the same structures (positive = we are better):

| subset | method | top-1 | top-N | top-(N+2) |
|---|---|---|---|---|
| all | ours (ranker) | -0.060 [-0.110, -0.014] | -0.045 [-0.095, -0.001] | +0.116 [+0.081, +0.150] |
| all | ours (native order) | -0.153 [-0.191, -0.114] | -0.112 [-0.151, -0.075] | +0.040 [+0.004, +0.074] |
| not train-similar | ours (ranker) | -0.119 [-0.188, -0.050] | -0.109 [-0.181, -0.043] | +0.059 [+0.015, +0.104] |
| not train-similar | ours (native order) | -0.159 [-0.206, -0.118] | -0.122 [-0.166, -0.078] | -0.005 [-0.046, +0.035] |
| train-similar | ours (ranker) | -0.012 [-0.074, +0.048] | +0.007 [-0.052, +0.062] | +0.163 [+0.118, +0.213] |
| train-similar | ours (native order) | -0.147 [-0.206, -0.087] | -0.103 [-0.162, -0.044] | +0.078 [+0.025, +0.132] |

Top-N by the structure's own number of sites, against **p2rank**. At N = 1 top-N is top-1 and no merging of predictions can change it, so a deficit in that row is ranking and a deficit confined to N >= 2 is prediction fragmentation:

| subset | method | N | n | top-N | p2rank top-N | difference | ceiling | p2rank ceiling |
|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 1 | 1704 | 0.526 | 0.681 | -0.156 | 0.941 | 0.754 |
| all | ours (ranker) | 2 | 1060 | 0.658 | 0.611 | +0.047 | 0.955 | 0.669 |
| all | ours (ranker) | >=3 | 571 | 0.771 | 0.658 | +0.112 | 0.949 | 0.713 |
| all | ours (native order) | 1 | 1704 | 0.518 | 0.681 | -0.163 | 0.941 | 0.754 |
| all | ours (native order) | 2 | 1060 | 0.548 | 0.611 | -0.063 | 0.955 | 0.669 |
| all | ours (native order) | >=3 | 571 | 0.609 | 0.658 | -0.049 | 0.949 | 0.713 |
| not train-similar | ours (ranker) | 1 | 738 | 0.473 | 0.713 | -0.240 | 0.946 | 0.789 |
| not train-similar | ours (ranker) | 2 | 479 | 0.676 | 0.672 | +0.004 | 0.962 | 0.749 |
| not train-similar | ours (ranker) | >=3 | 282 | 0.706 | 0.663 | +0.043 | 0.922 | 0.734 |
| not train-similar | ours (native order) | 1 | 738 | 0.531 | 0.713 | -0.182 | 0.946 | 0.789 |
| not train-similar | ours (native order) | 2 | 479 | 0.610 | 0.672 | -0.063 | 0.962 | 0.749 |
| not train-similar | ours (native order) | >=3 | 282 | 0.596 | 0.663 | -0.067 | 0.922 | 0.734 |
| train-similar | ours (ranker) | 1 | 966 | 0.566 | 0.657 | -0.091 | 0.937 | 0.727 |
| train-similar | ours (ranker) | 2 | 581 | 0.644 | 0.561 | +0.083 | 0.948 | 0.602 |
| train-similar | ours (ranker) | >=3 | 289 | 0.834 | 0.654 | +0.180 | 0.976 | 0.692 |
| train-similar | ours (native order) | 1 | 966 | 0.508 | 0.657 | -0.149 | 0.937 | 0.727 |
| train-similar | ours (native order) | 2 | 581 | 0.497 | 0.561 | -0.064 | 0.948 | 0.602 |
| train-similar | ours (native order) | >=3 | 289 | 0.623 | 0.654 | -0.031 | 0.976 | 0.692 |
