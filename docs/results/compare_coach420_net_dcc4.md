# coach420: our predictions against external tools on the same structures

269 structures predicted by every method listed. Success is DCC <= 4 A to a ligand centroid of the set's own relevant-ligand list; N is the structure's own number of ligand sites. 95 % CI by 30 %-identity cluster bootstrap. `train-similar` means the structure shares a 30 %-identity cluster with **our** training manifest; it is leakage for us and not for the external tools, whose own training sets overlap this benchmark in ways this table does not measure. The comparable row for us is **not train-similar**.

| subset | method | predictions | ceiling | top-1 | top-3 | top-N | top-(N+2) | MRR | n |
|---|---|---|---|---|---|---|---|---|---|
| all | ours (network only) | 30.0 | 0.892 | 0.576 [0.515, 0.635] | 0.810 [0.759, 0.858] | 0.669 [0.612, 0.724] | 0.814 [0.762, 0.861] | 0.699 [0.650, 0.745] | 269 |
| all | ours (ranker+net) | 30.0 | 0.892 | 0.554 [0.496, 0.614] | 0.781 [0.725, 0.831] | 0.636 [0.581, 0.693] | 0.814 [0.763, 0.862] | 0.674 [0.627, 0.723] | 269 |
| all | ours (native order) | 30.0 | 0.892 | 0.413 [0.349, 0.471] | 0.625 [0.563, 0.684] | 0.480 [0.413, 0.542] | 0.654 [0.594, 0.713] | 0.542 [0.489, 0.591] | 269 |
| all | p2rank | 7.0 | 0.677 | 0.535 [0.473, 0.601] | 0.639 [0.578, 0.699] | 0.602 [0.540, 0.665] | 0.651 [0.590, 0.711] | 0.590 [0.530, 0.647] | 269 |
| all | grasp | 2.0 | 0.528 | 0.424 [0.361, 0.487] | 0.528 [0.462, 0.590] | 0.457 [0.392, 0.521] | 0.528 [0.462, 0.590] | 0.475 [0.412, 0.535] | 269 |
| all | deeppocket | 27.9 | 0.662 | 0.454 [0.394, 0.514] | 0.602 [0.542, 0.662] | 0.517 [0.457, 0.582] | 0.625 [0.564, 0.684] | 0.536 [0.483, 0.592] | 269 |
| all | deepsurf | 1.8 | 0.572 | 0.528 [0.463, 0.591] | 0.572 [0.508, 0.638] | 0.558 [0.492, 0.624] | 0.572 [0.508, 0.638] | 0.550 [0.487, 0.614] | 269 |
| not train-similar | ours (network only) | 30.0 | 0.886 | 0.557 [0.491, 0.621] | 0.803 [0.751, 0.852] | 0.649 [0.585, 0.707] | 0.803 [0.751, 0.852] | 0.685 [0.634, 0.733] | 228 |
| not train-similar | ours (ranker+net) | 30.0 | 0.886 | 0.522 [0.458, 0.586] | 0.772 [0.715, 0.827] | 0.601 [0.536, 0.664] | 0.803 [0.749, 0.853] | 0.652 [0.600, 0.703] | 228 |
| not train-similar | ours (native order) | 30.0 | 0.886 | 0.399 [0.330, 0.465] | 0.605 [0.540, 0.670] | 0.461 [0.392, 0.527] | 0.627 [0.562, 0.692] | 0.527 [0.471, 0.584] | 228 |
| not train-similar | p2rank | 6.8 | 0.684 | 0.531 [0.461, 0.600] | 0.640 [0.570, 0.705] | 0.596 [0.529, 0.665] | 0.654 [0.583, 0.716] | 0.588 [0.525, 0.650] | 228 |
| not train-similar | grasp | 2.0 | 0.500 | 0.390 [0.323, 0.462] | 0.500 [0.429, 0.567] | 0.421 [0.352, 0.491] | 0.500 [0.429, 0.567] | 0.445 [0.381, 0.510] | 228 |
| not train-similar | deeppocket | 27.0 | 0.662 | 0.461 [0.395, 0.527] | 0.605 [0.541, 0.672] | 0.522 [0.455, 0.587] | 0.623 [0.560, 0.688] | 0.541 [0.480, 0.599] | 228 |
| not train-similar | deepsurf | 1.8 | 0.575 | 0.539 [0.470, 0.610] | 0.575 [0.507, 0.642] | 0.561 [0.492, 0.629] | 0.575 [0.507, 0.642] | 0.557 [0.488, 0.625] | 228 |
| train-similar | ours (network only) | 30.0 | 0.927 | 0.683 [0.543, 0.829] | 0.854 [0.721, 0.973] | 0.780 [0.658, 0.914] | 0.878 [0.755, 1.000] | 0.772 [0.655, 0.891] | 41 |
| train-similar | ours (ranker+net) | 30.0 | 0.927 | 0.732 [0.595, 0.875] | 0.829 [0.696, 0.950] | 0.829 [0.702, 0.950] | 0.878 [0.755, 1.000] | 0.796 [0.678, 0.912] | 41 |
| train-similar | ours (native order) | 30.0 | 0.927 | 0.488 [0.353, 0.629] | 0.732 [0.595, 0.871] | 0.585 [0.429, 0.745] | 0.805 [0.673, 0.941] | 0.624 [0.519, 0.743] | 41 |
| train-similar | p2rank | 8.1 | 0.634 | 0.561 [0.412, 0.725] | 0.634 [0.476, 0.789] | 0.634 [0.476, 0.789] | 0.634 [0.476, 0.789] | 0.598 [0.446, 0.750] | 41 |
| train-similar | grasp | 2.1 | 0.683 | 0.610 [0.429, 0.784] | 0.683 [0.511, 0.854] | 0.659 [0.481, 0.829] | 0.683 [0.511, 0.854] | 0.642 [0.471, 0.810] | 41 |
| train-similar | deeppocket | 32.6 | 0.659 | 0.415 [0.250, 0.605] | 0.585 [0.425, 0.778] | 0.488 [0.318, 0.676] | 0.634 [0.471, 0.821] | 0.512 [0.360, 0.688] | 41 |
| train-similar | deepsurf | 1.8 | 0.561 | 0.463 [0.286, 0.658] | 0.561 [0.378, 0.778] | 0.537 [0.352, 0.750] | 0.561 [0.378, 0.778] | 0.512 [0.337, 0.713] | 41 |

Paired differences against **p2rank** on the same structures (positive = we are better):

| subset | method | top-1 | top-N | top-(N+2) |
|---|---|---|---|---|
| all | ours (network only) | +0.041 [-0.020, +0.100] | +0.067 [+0.008, +0.129] | +0.164 [+0.100, +0.226] |
| all | ours (ranker+net) | +0.019 [-0.049, +0.088] | +0.033 [-0.032, +0.096] | +0.164 [+0.103, +0.224] |
| all | ours (native order) | -0.123 [-0.185, -0.059] | -0.123 [-0.186, -0.063] | +0.004 [-0.060, +0.064] |
| all | grasp | -0.112 [-0.194, -0.033] | -0.145 [-0.221, -0.067] | -0.123 [-0.200, -0.048] |
| all | deeppocket | -0.082 [-0.148, -0.015] | -0.086 [-0.148, -0.021] | -0.026 [-0.091, +0.043] |
| all | deepsurf | -0.007 [-0.075, +0.064] | -0.045 [-0.112, +0.025] | -0.078 [-0.147, -0.007] |
| not train-similar | ours (network only) | +0.026 [-0.041, +0.091] | +0.053 [-0.014, +0.119] | +0.149 [+0.076, +0.222] |
| not train-similar | ours (ranker+net) | -0.009 [-0.083, +0.063] | +0.004 [-0.070, +0.075] | +0.149 [+0.080, +0.217] |
| not train-similar | ours (native order) | -0.132 [-0.201, -0.066] | -0.136 [-0.205, -0.068] | -0.026 [-0.096, +0.045] |
| not train-similar | grasp | -0.140 [-0.233, -0.051] | -0.175 [-0.264, -0.085] | -0.154 [-0.239, -0.063] |
| not train-similar | deeppocket | -0.070 [-0.143, -0.004] | -0.075 [-0.146, -0.005] | -0.031 [-0.105, +0.043] |
| not train-similar | deepsurf | +0.009 [-0.069, +0.084] | -0.035 [-0.114, +0.039] | -0.079 [-0.158, +0.000] |
| train-similar | ours (network only) | +0.122 [-0.020, +0.263] | +0.146 [+0.026, +0.286] | +0.244 [+0.128, +0.390] |
| train-similar | ours (ranker+net) | +0.171 [+0.049, +0.300] | +0.195 [+0.085, +0.326] | +0.244 [+0.128, +0.390] |
| train-similar | ours (native order) | -0.073 [-0.256, +0.093] | -0.049 [-0.229, +0.109] | +0.171 [+0.048, +0.316] |
| train-similar | grasp | +0.049 [-0.139, +0.225] | +0.024 [-0.140, +0.177] | +0.049 [-0.100, +0.194] |
| train-similar | deeppocket | -0.146 [-0.340, +0.077] | -0.146 [-0.304, +0.029] | +0.000 [-0.163, +0.175] |
| train-similar | deepsurf | -0.098 [-0.256, +0.065] | -0.098 [-0.240, +0.057] | -0.073 [-0.204, +0.086] |

Top-N by the structure's own number of sites, against **p2rank**. At N = 1 top-N is top-1 and no merging of predictions can change it, so a deficit in that row is ranking and a deficit confined to N >= 2 is prediction fragmentation:

| subset | method | N | n | top-N | p2rank top-N | difference | ceiling | p2rank ceiling |
|---|---|---|---|---|---|---|---|---|
| all | ours (network only) | 1 | 142 | 0.507 | 0.585 | -0.077 | 0.852 | 0.655 |
| all | ours (network only) | 2 | 86 | 0.826 | 0.581 | +0.244 | 0.919 | 0.674 |
| all | ours (network only) | >=3 | 41 | 0.902 | 0.707 | +0.195 | 0.976 | 0.756 |
| all | ours (ranker+net) | 1 | 142 | 0.521 | 0.585 | -0.063 | 0.852 | 0.655 |
| all | ours (ranker+net) | 2 | 86 | 0.721 | 0.581 | +0.140 | 0.919 | 0.674 |
| all | ours (ranker+net) | >=3 | 41 | 0.854 | 0.707 | +0.146 | 0.976 | 0.756 |
| all | ours (native order) | 1 | 142 | 0.408 | 0.585 | -0.176 | 0.852 | 0.655 |
| all | ours (native order) | 2 | 86 | 0.535 | 0.581 | -0.047 | 0.919 | 0.674 |
| all | ours (native order) | >=3 | 41 | 0.610 | 0.707 | -0.098 | 0.976 | 0.756 |
| all | grasp | 1 | 142 | 0.345 | 0.585 | -0.239 | 0.479 | 0.655 |
| all | grasp | 2 | 86 | 0.581 | 0.581 | +0.000 | 0.581 | 0.674 |
| all | grasp | >=3 | 41 | 0.585 | 0.707 | -0.122 | 0.585 | 0.756 |
| all | deeppocket | 1 | 142 | 0.437 | 0.585 | -0.148 | 0.599 | 0.655 |
| all | deeppocket | 2 | 86 | 0.512 | 0.581 | -0.070 | 0.663 | 0.674 |
| all | deeppocket | >=3 | 41 | 0.805 | 0.707 | +0.098 | 0.878 | 0.756 |
| all | deepsurf | 1 | 142 | 0.493 | 0.585 | -0.092 | 0.521 | 0.655 |
| all | deepsurf | 2 | 86 | 0.593 | 0.581 | +0.012 | 0.593 | 0.674 |
| all | deepsurf | >=3 | 41 | 0.707 | 0.707 | +0.000 | 0.707 | 0.756 |
| not train-similar | ours (network only) | 1 | 124 | 0.476 | 0.581 | -0.105 | 0.855 | 0.661 |
| not train-similar | ours (network only) | 2 | 70 | 0.814 | 0.571 | +0.243 | 0.900 | 0.686 |
| not train-similar | ours (network only) | >=3 | 34 | 0.941 | 0.706 | +0.235 | 0.971 | 0.765 |
| not train-similar | ours (ranker+net) | 1 | 124 | 0.484 | 0.581 | -0.097 | 0.855 | 0.661 |
| not train-similar | ours (ranker+net) | 2 | 70 | 0.686 | 0.571 | +0.114 | 0.900 | 0.686 |
| not train-similar | ours (ranker+net) | >=3 | 34 | 0.853 | 0.706 | +0.147 | 0.971 | 0.765 |
| not train-similar | ours (native order) | 1 | 124 | 0.387 | 0.581 | -0.194 | 0.855 | 0.661 |
| not train-similar | ours (native order) | 2 | 70 | 0.500 | 0.571 | -0.071 | 0.900 | 0.686 |
| not train-similar | ours (native order) | >=3 | 34 | 0.647 | 0.706 | -0.059 | 0.971 | 0.765 |
| not train-similar | grasp | 1 | 124 | 0.315 | 0.581 | -0.266 | 0.460 | 0.661 |
| not train-similar | grasp | 2 | 70 | 0.529 | 0.571 | -0.043 | 0.529 | 0.686 |
| not train-similar | grasp | >=3 | 34 | 0.588 | 0.706 | -0.118 | 0.588 | 0.765 |
| not train-similar | deeppocket | 1 | 124 | 0.444 | 0.581 | -0.137 | 0.605 | 0.661 |
| not train-similar | deeppocket | 2 | 70 | 0.500 | 0.571 | -0.071 | 0.657 | 0.686 |
| not train-similar | deeppocket | >=3 | 34 | 0.853 | 0.706 | +0.147 | 0.882 | 0.765 |
| not train-similar | deepsurf | 1 | 124 | 0.508 | 0.581 | -0.073 | 0.532 | 0.661 |
| not train-similar | deepsurf | 2 | 70 | 0.586 | 0.571 | +0.014 | 0.586 | 0.686 |
| not train-similar | deepsurf | >=3 | 34 | 0.706 | 0.706 | +0.000 | 0.706 | 0.765 |
| train-similar | ours (network only) | 1 | 18 | 0.722 | 0.611 | +0.111 | 0.833 | 0.611 |
| train-similar | ours (network only) | 2 | 16 | 0.875 | 0.625 | +0.250 | 1.000 | 0.625 |
| train-similar | ours (network only) | >=3 | 7 | 0.714 | 0.714 | +0.000 | 1.000 | 0.714 |
| train-similar | ours (ranker+net) | 1 | 18 | 0.778 | 0.611 | +0.167 | 0.833 | 0.611 |
| train-similar | ours (ranker+net) | 2 | 16 | 0.875 | 0.625 | +0.250 | 1.000 | 0.625 |
| train-similar | ours (ranker+net) | >=3 | 7 | 0.857 | 0.714 | +0.143 | 1.000 | 0.714 |
| train-similar | ours (native order) | 1 | 18 | 0.556 | 0.611 | -0.056 | 0.833 | 0.611 |
| train-similar | ours (native order) | 2 | 16 | 0.688 | 0.625 | +0.062 | 1.000 | 0.625 |
| train-similar | ours (native order) | >=3 | 7 | 0.429 | 0.714 | -0.286 | 1.000 | 0.714 |
| train-similar | grasp | 1 | 18 | 0.556 | 0.611 | -0.056 | 0.611 | 0.611 |
| train-similar | grasp | 2 | 16 | 0.812 | 0.625 | +0.188 | 0.812 | 0.625 |
| train-similar | grasp | >=3 | 7 | 0.571 | 0.714 | -0.143 | 0.571 | 0.714 |
| train-similar | deeppocket | 1 | 18 | 0.389 | 0.611 | -0.222 | 0.556 | 0.611 |
| train-similar | deeppocket | 2 | 16 | 0.562 | 0.625 | -0.062 | 0.688 | 0.625 |
| train-similar | deeppocket | >=3 | 7 | 0.571 | 0.714 | -0.143 | 0.857 | 0.714 |
| train-similar | deepsurf | 1 | 18 | 0.389 | 0.611 | -0.222 | 0.444 | 0.611 |
| train-similar | deepsurf | 2 | 16 | 0.625 | 0.625 | +0.000 | 0.625 | 0.625 |
| train-similar | deepsurf | >=3 | 7 | 0.714 | 0.714 | +0.000 | 0.714 | 0.714 |
