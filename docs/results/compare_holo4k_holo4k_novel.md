# holo4k: our predictions against external tools on the same structures

1018 structures predicted by every method listed. Success is DCA <= 4 A to a ligand of the set's own relevant-ligand list; N is the structure's own number of ligand sites. 95 % CI by 30 %-identity cluster bootstrap. `train-similar` means the structure shares a 30 %-identity cluster with **our** training manifest; it is leakage for us and not for the external tools, whose own training sets overlap this benchmark in ways this table does not measure. The comparable row for us is **not train-similar**.

| subset | method | predictions | ceiling | top-1 | top-3 | top-N | top-(N+2) | MRR | n |
|---|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 30.0 | 0.996 | 0.840 [0.811, 0.867] | 0.944 [0.925, 0.960] | 0.889 [0.865, 0.912] | 0.959 [0.942, 0.972] | 0.895 [0.875, 0.914] | 1018 |
| all | ours (native order) | 30.0 | 0.996 | 0.637 [0.582, 0.685] | 0.850 [0.812, 0.884] | 0.751 [0.704, 0.796] | 0.895 [0.863, 0.924] | 0.754 [0.714, 0.789] | 1018 |
| all | p2rank | 8.1 | 0.966 | 0.850 [0.817, 0.879] | 0.947 [0.930, 0.963] | 0.918 [0.896, 0.939] | 0.956 [0.939, 0.971] | 0.898 [0.874, 0.918] | 1018 |
| not train-similar | ours (ranker) | 30.0 | 0.994 | 0.817 [0.775, 0.853] | 0.935 [0.908, 0.958] | 0.869 [0.834, 0.900] | 0.950 [0.926, 0.969] | 0.878 [0.849, 0.904] | 618 |
| not train-similar | ours (native order) | 30.0 | 0.994 | 0.608 [0.549, 0.669] | 0.828 [0.789, 0.867] | 0.738 [0.686, 0.787] | 0.883 [0.850, 0.914] | 0.732 [0.688, 0.774] | 618 |
| not train-similar | p2rank | 9.0 | 0.966 | 0.814 [0.769, 0.855] | 0.940 [0.915, 0.962] | 0.905 [0.875, 0.932] | 0.951 [0.928, 0.972] | 0.876 [0.845, 0.905] | 618 |
| train-similar | ours (ranker) | 30.0 | 1.000 | 0.875 [0.837, 0.911] | 0.958 [0.934, 0.978] | 0.920 [0.890, 0.946] | 0.973 [0.952, 0.986] | 0.920 [0.894, 0.943] | 400 |
| train-similar | ours (native order) | 30.0 | 1.000 | 0.680 [0.579, 0.766] | 0.882 [0.814, 0.939] | 0.772 [0.678, 0.851] | 0.912 [0.851, 0.963] | 0.787 [0.712, 0.848] | 400 |
| train-similar | p2rank | 6.8 | 0.965 | 0.905 [0.866, 0.939] | 0.958 [0.931, 0.980] | 0.940 [0.907, 0.967] | 0.963 [0.936, 0.983] | 0.930 [0.900, 0.956] | 400 |

Paired differences against **p2rank** on the same structures (positive = we are better):

| subset | method | top-1 | top-N | top-(N+2) |
|---|---|---|---|---|
| all | ours (ranker) | -0.010 [-0.041, +0.023] | -0.029 [-0.054, -0.005] | +0.003 [-0.015, +0.020] |
| all | ours (native order) | -0.213 [-0.261, -0.168] | -0.167 [-0.215, -0.123] | -0.061 [-0.094, -0.031] |
| not train-similar | ours (ranker) | +0.003 [-0.040, +0.048] | -0.036 [-0.071, -0.002] | -0.002 [-0.028, +0.024] |
| not train-similar | ours (native order) | -0.206 [-0.256, -0.155] | -0.167 [-0.216, -0.121] | -0.068 [-0.102, -0.034] |
| train-similar | ours (ranker) | -0.030 [-0.071, +0.008] | -0.020 [-0.050, +0.011] | +0.010 [-0.008, +0.029] |
| train-similar | ours (native order) | -0.225 [-0.326, -0.138] | -0.168 [-0.263, -0.086] | -0.050 [-0.115, +0.003] |

Top-N by the structure's own number of sites, against **p2rank**. At N = 1 top-N is top-1 and no merging of predictions can change it, so a deficit in that row is ranking and a deficit confined to N >= 2 is prediction fragmentation:

| subset | method | N | n | top-N | p2rank top-N | difference | ceiling | p2rank ceiling |
|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 1 | 478 | 0.870 | 0.912 | -0.042 | 0.998 | 0.962 |
| all | ours (ranker) | 2 | 339 | 0.867 | 0.912 | -0.044 | 0.991 | 0.968 |
| all | ours (ranker) | >=3 | 201 | 0.970 | 0.945 | +0.025 | 1.000 | 0.970 |
| all | ours (native order) | 1 | 478 | 0.738 | 0.912 | -0.174 | 0.998 | 0.962 |
| all | ours (native order) | 2 | 339 | 0.729 | 0.912 | -0.183 | 0.991 | 0.968 |
| all | ours (native order) | >=3 | 201 | 0.821 | 0.945 | -0.124 | 1.000 | 0.970 |
| not train-similar | ours (ranker) | 1 | 248 | 0.835 | 0.903 | -0.069 | 0.996 | 0.980 |
| not train-similar | ours (ranker) | 2 | 233 | 0.850 | 0.893 | -0.043 | 0.987 | 0.957 |
| not train-similar | ours (ranker) | >=3 | 137 | 0.964 | 0.927 | +0.036 | 1.000 | 0.956 |
| not train-similar | ours (native order) | 1 | 248 | 0.681 | 0.903 | -0.222 | 0.996 | 0.980 |
| not train-similar | ours (native order) | 2 | 233 | 0.755 | 0.893 | -0.137 | 0.987 | 0.957 |
| not train-similar | ours (native order) | >=3 | 137 | 0.810 | 0.927 | -0.117 | 1.000 | 0.956 |
| train-similar | ours (ranker) | 1 | 230 | 0.909 | 0.922 | -0.013 | 1.000 | 0.943 |
| train-similar | ours (ranker) | 2 | 106 | 0.906 | 0.953 | -0.047 | 1.000 | 0.991 |
| train-similar | ours (ranker) | >=3 | 64 | 0.984 | 0.984 | +0.000 | 1.000 | 1.000 |
| train-similar | ours (native order) | 1 | 230 | 0.800 | 0.922 | -0.122 | 1.000 | 0.943 |
| train-similar | ours (native order) | 2 | 106 | 0.670 | 0.953 | -0.283 | 1.000 | 0.991 |
| train-similar | ours (native order) | >=3 | 64 | 0.844 | 0.984 | -0.141 | 1.000 | 1.000 |
