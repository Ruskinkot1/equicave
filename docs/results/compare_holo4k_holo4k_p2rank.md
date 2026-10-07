# holo4k: our predictions against external tools on the same structures

3335 structures predicted by every method listed. Success is DCA <= 4 A to a ligand of the set's own relevant-ligand list; N is the structure's own number of ligand sites. 95 % CI by 30 %-identity cluster bootstrap. `train-similar` means the structure shares a 30 %-identity cluster with **our** training manifest; it is leakage for us and not for the external tools, whose own training sets overlap this benchmark in ways this table does not measure. The comparable row for us is **not train-similar**.

| subset | method | predictions | ceiling | top-1 | top-3 | top-N | top-(N+2) | MRR | n |
|---|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 30.0 | 0.993 | 0.818 [0.790, 0.845] | 0.928 [0.912, 0.942] | 0.859 [0.834, 0.883] | 0.940 [0.925, 0.953] | 0.878 [0.858, 0.897] | 3335 |
| all | ours (native order) | 30.0 | 0.993 | 0.649 [0.604, 0.694] | 0.851 [0.820, 0.879] | 0.745 [0.708, 0.781] | 0.888 [0.864, 0.909] | 0.763 [0.730, 0.794] | 3335 |
| all | p2rank | 7.9 | 0.961 | 0.822 [0.789, 0.851] | 0.932 [0.914, 0.947] | 0.891 [0.868, 0.912] | 0.944 [0.927, 0.958] | 0.878 [0.855, 0.898] | 3335 |
| not train-similar | ours (ranker) | 30.0 | 0.988 | 0.785 [0.743, 0.823] | 0.903 [0.877, 0.925] | 0.829 [0.791, 0.863] | 0.923 [0.902, 0.942] | 0.851 [0.820, 0.878] | 1499 |
| not train-similar | ours (native order) | 30.0 | 0.988 | 0.635 [0.573, 0.691] | 0.832 [0.794, 0.864] | 0.736 [0.685, 0.782] | 0.873 [0.841, 0.900] | 0.746 [0.703, 0.787] | 1499 |
| not train-similar | p2rank | 8.2 | 0.967 | 0.819 [0.777, 0.853] | 0.927 [0.906, 0.945] | 0.893 [0.864, 0.917] | 0.946 [0.927, 0.962] | 0.875 [0.846, 0.899] | 1499 |
| train-similar | ours (ranker) | 30.0 | 0.998 | 0.845 [0.804, 0.882] | 0.948 [0.926, 0.966] | 0.885 [0.848, 0.916] | 0.954 [0.932, 0.971] | 0.900 [0.873, 0.925] | 1836 |
| train-similar | ours (native order) | 30.0 | 0.998 | 0.661 [0.595, 0.725] | 0.867 [0.819, 0.907] | 0.753 [0.699, 0.802] | 0.901 [0.864, 0.930] | 0.777 [0.730, 0.821] | 1836 |
| train-similar | p2rank | 7.7 | 0.957 | 0.825 [0.774, 0.869] | 0.935 [0.908, 0.958] | 0.889 [0.854, 0.919] | 0.942 [0.915, 0.964] | 0.880 [0.844, 0.910] | 1836 |

Paired differences against **p2rank** on the same structures (positive = we are better):

| subset | method | top-1 | top-N | top-(N+2) |
|---|---|---|---|---|
| all | ours (ranker) | -0.004 [-0.034, +0.026] | -0.032 [-0.056, -0.011] | -0.004 [-0.019, +0.010] |
| all | ours (native order) | -0.173 [-0.208, -0.141] | -0.146 [-0.177, -0.116] | -0.055 [-0.077, -0.037] |
| not train-similar | ours (ranker) | -0.033 [-0.063, -0.004] | -0.065 [-0.093, -0.039] | -0.023 [-0.043, -0.004] |
| not train-similar | ours (native order) | -0.183 [-0.228, -0.145] | -0.157 [-0.200, -0.122] | -0.073 [-0.101, -0.049] |
| train-similar | ours (ranker) | +0.020 [-0.028, +0.067] | -0.005 [-0.042, +0.028] | +0.012 [-0.008, +0.031] |
| train-similar | ours (native order) | -0.164 [-0.216, -0.113] | -0.136 [-0.180, -0.092] | -0.041 [-0.069, -0.014] |

Top-N by the structure's own number of sites, against **p2rank**. At N = 1 top-N is top-1 and no merging of predictions can change it, so a deficit in that row is ranking and a deficit confined to N >= 2 is prediction fragmentation:

| subset | method | N | n | top-N | p2rank top-N | difference | ceiling | p2rank ceiling |
|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 1 | 1704 | 0.824 | 0.870 | -0.046 | 0.996 | 0.954 |
| all | ours (ranker) | 2 | 1060 | 0.883 | 0.903 | -0.020 | 0.996 | 0.967 |
| all | ours (ranker) | >=3 | 571 | 0.921 | 0.933 | -0.012 | 0.979 | 0.974 |
| all | ours (native order) | 1 | 1704 | 0.710 | 0.870 | -0.160 | 0.996 | 0.954 |
| all | ours (native order) | 2 | 1060 | 0.780 | 0.903 | -0.123 | 0.996 | 0.967 |
| all | ours (native order) | >=3 | 571 | 0.786 | 0.933 | -0.147 | 0.979 | 0.974 |
| not train-similar | ours (ranker) | 1 | 738 | 0.804 | 0.875 | -0.072 | 0.997 | 0.963 |
| not train-similar | ours (ranker) | 2 | 479 | 0.843 | 0.902 | -0.058 | 0.992 | 0.969 |
| not train-similar | ours (ranker) | >=3 | 282 | 0.869 | 0.926 | -0.057 | 0.957 | 0.972 |
| not train-similar | ours (native order) | 1 | 738 | 0.699 | 0.875 | -0.176 | 0.997 | 0.963 |
| not train-similar | ours (native order) | 2 | 479 | 0.779 | 0.902 | -0.123 | 0.992 | 0.969 |
| not train-similar | ours (native order) | >=3 | 282 | 0.759 | 0.926 | -0.167 | 0.957 | 0.972 |
| train-similar | ours (ranker) | 1 | 966 | 0.840 | 0.865 | -0.026 | 0.996 | 0.946 |
| train-similar | ours (ranker) | 2 | 581 | 0.916 | 0.904 | +0.012 | 1.000 | 0.966 |
| train-similar | ours (ranker) | >=3 | 289 | 0.972 | 0.941 | +0.031 | 1.000 | 0.976 |
| train-similar | ours (native order) | 1 | 966 | 0.718 | 0.865 | -0.147 | 0.996 | 0.946 |
| train-similar | ours (native order) | 2 | 581 | 0.781 | 0.904 | -0.122 | 1.000 | 0.966 |
| train-similar | ours (native order) | >=3 | 289 | 0.813 | 0.941 | -0.128 | 1.000 | 0.976 |
