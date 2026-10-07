# holo4k: our predictions against external tools on the same structures

3335 structures predicted by every method listed. Success is DCC <= 10 A to a ligand centroid of the set's own relevant-ligand list; N is the structure's own number of ligand sites. 95 % CI by 30 %-identity cluster bootstrap. `train-similar` means the structure shares a 30 %-identity cluster with **our** training manifest; it is leakage for us and not for the external tools, whose own training sets overlap this benchmark in ways this table does not measure. The comparable row for us is **not train-similar**.

| subset | method | predictions | ceiling | top-1 | top-3 | top-N | top-(N+2) | MRR | n |
|---|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 30.0 | 0.996 | 0.843 [0.820, 0.867] | 0.939 [0.924, 0.953] | 0.882 [0.860, 0.902] | 0.951 [0.937, 0.963] | 0.896 [0.879, 0.912] | 3335 |
| all | ours (native order) | 30.0 | 0.996 | 0.666 [0.622, 0.708] | 0.875 [0.852, 0.897] | 0.760 [0.726, 0.795] | 0.909 [0.889, 0.925] | 0.779 [0.749, 0.807] | 3335 |
| all | p2rank | 7.9 | 0.984 | 0.854 [0.826, 0.878] | 0.953 [0.940, 0.963] | 0.913 [0.894, 0.930] | 0.963 [0.951, 0.973] | 0.905 [0.887, 0.921] | 3335 |
| not train-similar | ours (ranker) | 30.0 | 0.993 | 0.805 [0.764, 0.843] | 0.915 [0.890, 0.936] | 0.847 [0.811, 0.880] | 0.933 [0.913, 0.951] | 0.866 [0.837, 0.893] | 1499 |
| not train-similar | ours (native order) | 30.0 | 0.993 | 0.647 [0.587, 0.703] | 0.849 [0.813, 0.879] | 0.745 [0.696, 0.790] | 0.887 [0.857, 0.914] | 0.759 [0.717, 0.798] | 1499 |
| not train-similar | p2rank | 8.2 | 0.985 | 0.834 [0.795, 0.869] | 0.939 [0.918, 0.957] | 0.901 [0.872, 0.926] | 0.955 [0.936, 0.972] | 0.889 [0.863, 0.913] | 1499 |
| train-similar | ours (ranker) | 30.0 | 0.998 | 0.875 [0.843, 0.903] | 0.959 [0.938, 0.976] | 0.910 [0.882, 0.934] | 0.965 [0.946, 0.981] | 0.920 [0.898, 0.940] | 1836 |
| train-similar | ours (native order) | 30.0 | 0.998 | 0.681 [0.621, 0.739] | 0.897 [0.866, 0.924] | 0.773 [0.725, 0.817] | 0.926 [0.902, 0.947] | 0.795 [0.755, 0.835] | 1836 |
| train-similar | p2rank | 7.7 | 0.984 | 0.870 [0.832, 0.904] | 0.964 [0.949, 0.976] | 0.923 [0.896, 0.945] | 0.969 [0.955, 0.981] | 0.918 [0.896, 0.938] | 1836 |

Paired differences against **p2rank** on the same structures (positive = we are better):

| subset | method | top-1 | top-N | top-(N+2) |
|---|---|---|---|---|
| all | ours (ranker) | -0.010 [-0.032, +0.013] | -0.031 [-0.047, -0.016] | -0.012 [-0.022, -0.002] |
| all | ours (native order) | -0.188 [-0.223, -0.156] | -0.153 [-0.182, -0.126] | -0.054 [-0.071, -0.039] |
| not train-similar | ours (ranker) | -0.029 [-0.056, -0.001] | -0.055 [-0.081, -0.032] | -0.022 [-0.040, -0.005] |
| not train-similar | ours (native order) | -0.187 [-0.229, -0.149] | -0.156 [-0.195, -0.122] | -0.068 [-0.093, -0.046] |
| train-similar | ours (ranker) | +0.004 [-0.028, +0.040] | -0.013 [-0.032, +0.007] | -0.004 [-0.017, +0.007] |
| train-similar | ours (native order) | -0.190 [-0.241, -0.140] | -0.150 [-0.190, -0.108] | -0.043 [-0.065, -0.023] |

Top-N by the structure's own number of sites, against **p2rank**. At N = 1 top-N is top-1 and no merging of predictions can change it, so a deficit in that row is ranking and a deficit confined to N >= 2 is prediction fragmentation:

| subset | method | N | n | top-N | p2rank top-N | difference | ceiling | p2rank ceiling |
|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 1 | 1704 | 0.852 | 0.890 | -0.038 | 0.998 | 0.980 |
| all | ours (ranker) | 2 | 1060 | 0.904 | 0.927 | -0.024 | 0.998 | 0.988 |
| all | ours (ranker) | >=3 | 571 | 0.930 | 0.956 | -0.026 | 0.986 | 0.989 |
| all | ours (native order) | 1 | 1704 | 0.719 | 0.890 | -0.171 | 0.998 | 0.980 |
| all | ours (native order) | 2 | 1060 | 0.800 | 0.927 | -0.127 | 0.998 | 0.988 |
| all | ours (native order) | >=3 | 571 | 0.811 | 0.956 | -0.145 | 0.986 | 0.989 |
| not train-similar | ours (ranker) | 1 | 738 | 0.822 | 0.883 | -0.061 | 1.000 | 0.989 |
| not train-similar | ours (ranker) | 2 | 479 | 0.862 | 0.910 | -0.048 | 0.996 | 0.979 |
| not train-similar | ours (ranker) | >=3 | 282 | 0.883 | 0.933 | -0.050 | 0.972 | 0.982 |
| not train-similar | ours (native order) | 1 | 738 | 0.701 | 0.883 | -0.183 | 1.000 | 0.989 |
| not train-similar | ours (native order) | 2 | 479 | 0.791 | 0.910 | -0.119 | 0.996 | 0.979 |
| not train-similar | ours (native order) | >=3 | 282 | 0.784 | 0.933 | -0.149 | 0.972 | 0.982 |
| train-similar | ours (ranker) | 1 | 966 | 0.874 | 0.894 | -0.021 | 0.996 | 0.973 |
| train-similar | ours (ranker) | 2 | 581 | 0.938 | 0.941 | -0.003 | 1.000 | 0.995 |
| train-similar | ours (ranker) | >=3 | 289 | 0.976 | 0.979 | -0.003 | 1.000 | 0.997 |
| train-similar | ours (native order) | 1 | 966 | 0.733 | 0.894 | -0.161 | 0.996 | 0.973 |
| train-similar | ours (native order) | 2 | 581 | 0.807 | 0.941 | -0.134 | 1.000 | 0.995 |
| train-similar | ours (native order) | >=3 | 289 | 0.837 | 0.979 | -0.142 | 1.000 | 0.997 |
