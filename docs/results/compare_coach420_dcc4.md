# coach420: our predictions against external tools on the same structures

269 structures predicted by every method listed. Success is DCC <= 4 A to a ligand centroid of the set's own relevant-ligand list; N is the structure's own number of ligand sites. 95 % CI by 30 %-identity cluster bootstrap. `train-similar` means the structure shares a 30 %-identity cluster with **our** training manifest; it is leakage for us and not for the external tools, whose own training sets overlap this benchmark in ways this table does not measure. The comparable row for us is **not train-similar**.

| subset | method | predictions | ceiling | top-1 | top-3 | top-N | top-(N+2) | MRR | n |
|---|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 30.0 | 0.892 | 0.424 [0.363, 0.485] | 0.677 [0.619, 0.733] | 0.502 [0.439, 0.561] | 0.736 [0.681, 0.787] | 0.575 [0.527, 0.624] | 269 |
| all | ours (native order) | 30.0 | 0.892 | 0.413 [0.353, 0.471] | 0.625 [0.565, 0.683] | 0.480 [0.414, 0.542] | 0.654 [0.596, 0.712] | 0.542 [0.490, 0.590] | 269 |
| all | p2rank | 7.0 | 0.677 | 0.535 [0.469, 0.598] | 0.639 [0.578, 0.699] | 0.602 [0.538, 0.663] | 0.651 [0.590, 0.711] | 0.590 [0.530, 0.647] | 269 |
| all | deeppocket | 27.9 | 0.662 | 0.454 [0.393, 0.515] | 0.602 [0.544, 0.662] | 0.517 [0.456, 0.579] | 0.625 [0.569, 0.682] | 0.536 [0.484, 0.592] | 269 |
| all | grasp | 2.0 | 0.528 | 0.424 [0.363, 0.488] | 0.528 [0.463, 0.592] | 0.457 [0.395, 0.523] | 0.528 [0.463, 0.592] | 0.475 [0.416, 0.535] | 269 |
| all | deepsurf | 1.8 | 0.572 | 0.528 [0.459, 0.594] | 0.572 [0.506, 0.640] | 0.558 [0.493, 0.625] | 0.572 [0.506, 0.640] | 0.550 [0.485, 0.614] | 269 |
| not train-similar | ours (ranker) | 30.0 | 0.876 | 0.373 [0.295, 0.451] | 0.634 [0.559, 0.708] | 0.466 [0.391, 0.543] | 0.689 [0.618, 0.762] | 0.531 [0.470, 0.594] | 161 |
| not train-similar | ours (native order) | 30.0 | 0.876 | 0.422 [0.346, 0.494] | 0.615 [0.537, 0.692] | 0.503 [0.423, 0.583] | 0.640 [0.562, 0.716] | 0.540 [0.476, 0.603] | 161 |
| not train-similar | p2rank | 7.6 | 0.727 | 0.559 [0.478, 0.636] | 0.683 [0.604, 0.755] | 0.640 [0.559, 0.715] | 0.689 [0.610, 0.764] | 0.621 [0.547, 0.691] | 161 |
| not train-similar | deeppocket | 30.4 | 0.677 | 0.460 [0.387, 0.531] | 0.621 [0.545, 0.696] | 0.540 [0.465, 0.615] | 0.634 [0.559, 0.708] | 0.548 [0.482, 0.614] | 161 |
| not train-similar | grasp | 2.0 | 0.484 | 0.404 [0.318, 0.483] | 0.484 [0.402, 0.562] | 0.453 [0.370, 0.530] | 0.484 [0.402, 0.562] | 0.443 [0.361, 0.518] | 161 |
| not train-similar | deepsurf | 2.0 | 0.596 | 0.565 [0.484, 0.643] | 0.596 [0.516, 0.673] | 0.596 [0.516, 0.673] | 0.596 [0.516, 0.673] | 0.581 [0.503, 0.657] | 161 |
| train-similar | ours (ranker) | 30.0 | 0.917 | 0.500 [0.404, 0.598] | 0.741 [0.648, 0.822] | 0.556 [0.464, 0.650] | 0.806 [0.727, 0.879] | 0.641 [0.566, 0.716] | 108 |
| train-similar | ours (native order) | 30.0 | 0.917 | 0.398 [0.306, 0.491] | 0.639 [0.540, 0.725] | 0.444 [0.348, 0.542] | 0.676 [0.583, 0.762] | 0.544 [0.466, 0.620] | 108 |
| train-similar | p2rank | 6.2 | 0.602 | 0.500 [0.402, 0.593] | 0.574 [0.471, 0.667] | 0.546 [0.444, 0.639] | 0.593 [0.486, 0.686] | 0.543 [0.444, 0.632] | 108 |
| train-similar | deeppocket | 24.1 | 0.639 | 0.444 [0.351, 0.537] | 0.574 [0.485, 0.673] | 0.481 [0.385, 0.576] | 0.611 [0.516, 0.703] | 0.519 [0.431, 0.606] | 108 |
| train-similar | grasp | 1.9 | 0.593 | 0.454 [0.347, 0.560] | 0.593 [0.489, 0.685] | 0.463 [0.355, 0.567] | 0.593 [0.489, 0.685] | 0.523 [0.425, 0.615] | 108 |
| train-similar | deepsurf | 1.6 | 0.537 | 0.472 [0.362, 0.584] | 0.537 [0.427, 0.648] | 0.500 [0.396, 0.609] | 0.537 [0.427, 0.648] | 0.505 [0.398, 0.608] | 108 |

Paired differences against **p2rank** on the same structures (positive = we are better):

| subset | method | top-1 | top-N | top-(N+2) |
|---|---|---|---|---|
| all | ours (ranker) | -0.112 [-0.181, -0.041] | -0.100 [-0.172, -0.031] | +0.086 [+0.018, +0.152] |
| all | ours (native order) | -0.123 [-0.186, -0.059] | -0.123 [-0.188, -0.061] | +0.004 [-0.058, +0.067] |
| all | deeppocket | -0.082 [-0.148, -0.011] | -0.086 [-0.148, -0.019] | -0.026 [-0.092, +0.040] |
| all | grasp | -0.112 [-0.186, -0.028] | -0.145 [-0.218, -0.064] | -0.123 [-0.195, -0.048] |
| all | deepsurf | -0.007 [-0.078, +0.064] | -0.045 [-0.113, +0.026] | -0.078 [-0.152, -0.007] |
| not train-similar | ours (ranker) | -0.186 [-0.283, -0.089] | -0.174 [-0.266, -0.078] | +0.000 [-0.087, +0.090] |
| not train-similar | ours (native order) | -0.137 [-0.221, -0.048] | -0.137 [-0.224, -0.054] | -0.050 [-0.135, +0.032] |
| not train-similar | deeppocket | -0.099 [-0.182, -0.019] | -0.099 [-0.181, -0.024] | -0.056 [-0.139, +0.027] |
| not train-similar | grasp | -0.155 [-0.263, -0.049] | -0.186 [-0.288, -0.082] | -0.205 [-0.308, -0.095] |
| not train-similar | deepsurf | +0.006 [-0.078, +0.098] | -0.043 [-0.131, +0.047] | -0.093 [-0.181, -0.006] |
| train-similar | ours (ranker) | +0.000 [-0.082, +0.082] | +0.009 [-0.083, +0.103] | +0.213 [+0.125, +0.308] |
| train-similar | ours (native order) | -0.102 [-0.204, +0.000] | -0.102 [-0.208, +0.000] | +0.083 [-0.019, +0.183] |
| train-similar | deeppocket | -0.056 [-0.158, +0.053] | -0.065 [-0.162, +0.040] | +0.019 [-0.087, +0.127] |
| train-similar | grasp | -0.046 [-0.170, +0.069] | -0.083 [-0.200, +0.031] | +0.000 [-0.099, +0.099] |
| train-similar | deepsurf | -0.028 [-0.142, +0.088] | -0.046 [-0.152, +0.065] | -0.056 [-0.167, +0.059] |

Top-N by the structure's own number of sites, against **p2rank**. At N = 1 top-N is top-1 and no merging of predictions can change it, so a deficit in that row is ranking and a deficit confined to N >= 2 is prediction fragmentation:

| subset | method | N | n | top-N | p2rank top-N | difference | ceiling | p2rank ceiling |
|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 1 | 142 | 0.465 | 0.585 | -0.120 | 0.852 | 0.655 |
| all | ours (ranker) | 2 | 86 | 0.488 | 0.581 | -0.093 | 0.919 | 0.674 |
| all | ours (ranker) | >=3 | 41 | 0.659 | 0.707 | -0.049 | 0.976 | 0.756 |
| all | ours (native order) | 1 | 142 | 0.408 | 0.585 | -0.176 | 0.852 | 0.655 |
| all | ours (native order) | 2 | 86 | 0.535 | 0.581 | -0.047 | 0.919 | 0.674 |
| all | ours (native order) | >=3 | 41 | 0.610 | 0.707 | -0.098 | 0.976 | 0.756 |
| all | deeppocket | 1 | 142 | 0.437 | 0.585 | -0.148 | 0.599 | 0.655 |
| all | deeppocket | 2 | 86 | 0.512 | 0.581 | -0.070 | 0.663 | 0.674 |
| all | deeppocket | >=3 | 41 | 0.805 | 0.707 | +0.098 | 0.878 | 0.756 |
| all | grasp | 1 | 142 | 0.345 | 0.585 | -0.239 | 0.479 | 0.655 |
| all | grasp | 2 | 86 | 0.581 | 0.581 | +0.000 | 0.581 | 0.674 |
| all | grasp | >=3 | 41 | 0.585 | 0.707 | -0.122 | 0.585 | 0.756 |
| all | deepsurf | 1 | 142 | 0.493 | 0.585 | -0.092 | 0.521 | 0.655 |
| all | deepsurf | 2 | 86 | 0.593 | 0.581 | +0.012 | 0.593 | 0.674 |
| all | deepsurf | >=3 | 41 | 0.707 | 0.707 | +0.000 | 0.707 | 0.756 |
| not train-similar | ours (ranker) | 1 | 77 | 0.377 | 0.597 | -0.221 | 0.805 | 0.675 |
| not train-similar | ours (ranker) | 2 | 54 | 0.500 | 0.630 | -0.130 | 0.926 | 0.741 |
| not train-similar | ours (ranker) | >=3 | 30 | 0.633 | 0.767 | -0.133 | 0.967 | 0.833 |
| not train-similar | ours (native order) | 1 | 77 | 0.364 | 0.597 | -0.234 | 0.805 | 0.675 |
| not train-similar | ours (native order) | 2 | 54 | 0.574 | 0.630 | -0.056 | 0.926 | 0.741 |
| not train-similar | ours (native order) | >=3 | 30 | 0.733 | 0.767 | -0.033 | 0.967 | 0.833 |
| not train-similar | deeppocket | 1 | 77 | 0.416 | 0.597 | -0.182 | 0.597 | 0.675 |
| not train-similar | deeppocket | 2 | 54 | 0.519 | 0.630 | -0.111 | 0.667 | 0.741 |
| not train-similar | deeppocket | >=3 | 30 | 0.900 | 0.767 | +0.133 | 0.900 | 0.833 |
| not train-similar | grasp | 1 | 77 | 0.325 | 0.597 | -0.273 | 0.390 | 0.675 |
| not train-similar | grasp | 2 | 54 | 0.556 | 0.630 | -0.074 | 0.556 | 0.741 |
| not train-similar | grasp | >=3 | 30 | 0.600 | 0.767 | -0.167 | 0.600 | 0.833 |
| not train-similar | deepsurf | 1 | 77 | 0.506 | 0.597 | -0.091 | 0.506 | 0.675 |
| not train-similar | deepsurf | 2 | 54 | 0.630 | 0.630 | +0.000 | 0.630 | 0.741 |
| not train-similar | deepsurf | >=3 | 30 | 0.767 | 0.767 | +0.000 | 0.767 | 0.833 |
| train-similar | ours (ranker) | 1 | 65 | 0.569 | 0.569 | +0.000 | 0.908 | 0.631 |
| train-similar | ours (ranker) | 2 | 32 | 0.469 | 0.500 | -0.031 | 0.906 | 0.562 |
| train-similar | ours (ranker) | >=3 | 11 | 0.727 | 0.545 | +0.182 | 1.000 | 0.545 |
| train-similar | ours (native order) | 1 | 65 | 0.462 | 0.569 | -0.108 | 0.908 | 0.631 |
| train-similar | ours (native order) | 2 | 32 | 0.469 | 0.500 | -0.031 | 0.906 | 0.562 |
| train-similar | ours (native order) | >=3 | 11 | 0.273 | 0.545 | -0.273 | 1.000 | 0.545 |
| train-similar | deeppocket | 1 | 65 | 0.462 | 0.569 | -0.108 | 0.600 | 0.631 |
| train-similar | deeppocket | 2 | 32 | 0.500 | 0.500 | +0.000 | 0.656 | 0.562 |
| train-similar | deeppocket | >=3 | 11 | 0.545 | 0.545 | +0.000 | 0.818 | 0.545 |
| train-similar | grasp | 1 | 65 | 0.369 | 0.569 | -0.200 | 0.585 | 0.631 |
| train-similar | grasp | 2 | 32 | 0.625 | 0.500 | +0.125 | 0.625 | 0.562 |
| train-similar | grasp | >=3 | 11 | 0.545 | 0.545 | +0.000 | 0.545 | 0.545 |
| train-similar | deepsurf | 1 | 65 | 0.477 | 0.569 | -0.092 | 0.538 | 0.631 |
| train-similar | deepsurf | 2 | 32 | 0.531 | 0.500 | +0.031 | 0.531 | 0.562 |
| train-similar | deepsurf | >=3 | 11 | 0.545 | 0.545 | +0.000 | 0.545 | 0.545 |
