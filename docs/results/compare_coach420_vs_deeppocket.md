# coach420: our predictions against external tools on the same structures

281 structures predicted by every method listed. Success is DCA <= 4 A to a ligand of the set's own relevant-ligand list; N is the structure's own number of ligand sites. 95 % CI by 30 %-identity cluster bootstrap. `train-similar` means the structure shares a 30 %-identity cluster with **our** training manifest; it is leakage for us and not for the external tools, whose own training sets overlap this benchmark in ways this table does not measure. The comparable row for us is **not train-similar**.

| subset | method | predictions | ceiling | top-1 | top-3 | top-N | top-(N+2) | MRR | n |
|---|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 30.0 | 0.989 | 0.740 [0.685, 0.794] | 0.875 [0.837, 0.913] | 0.797 [0.746, 0.845] | 0.890 [0.851, 0.926] | 0.821 [0.781, 0.858] | 281 |
| all | ours (native order) | 30.0 | 0.989 | 0.609 [0.550, 0.666] | 0.801 [0.750, 0.847] | 0.701 [0.645, 0.756] | 0.826 [0.779, 0.869] | 0.723 [0.679, 0.766] | 281 |
| all | p2rank | 8.4 | 0.940 | 0.769 [0.718, 0.817] | 0.890 [0.851, 0.927] | 0.840 [0.793, 0.883] | 0.904 [0.866, 0.940] | 0.833 [0.793, 0.869] | 281 |
| all | deeppocket | 32.6 | 0.964 | 0.726 [0.672, 0.779] | 0.872 [0.830, 0.913] | 0.776 [0.726, 0.825] | 0.900 [0.863, 0.936] | 0.808 [0.767, 0.848] | 281 |
| not train-similar | ours (ranker) | 30.0 | 0.988 | 0.699 [0.626, 0.768] | 0.867 [0.814, 0.916] | 0.775 [0.710, 0.837] | 0.873 [0.822, 0.920] | 0.797 [0.745, 0.845] | 173 |
| not train-similar | ours (native order) | 30.0 | 0.988 | 0.595 [0.520, 0.665] | 0.792 [0.729, 0.851] | 0.711 [0.639, 0.779] | 0.821 [0.760, 0.878] | 0.712 [0.655, 0.763] | 173 |
| not train-similar | p2rank | 9.6 | 0.931 | 0.751 [0.689, 0.817] | 0.873 [0.819, 0.918] | 0.827 [0.769, 0.883] | 0.879 [0.826, 0.925] | 0.816 [0.765, 0.866] | 173 |
| not train-similar | deeppocket | 37.1 | 0.965 | 0.699 [0.629, 0.767] | 0.861 [0.806, 0.912] | 0.775 [0.711, 0.833] | 0.884 [0.833, 0.930] | 0.791 [0.738, 0.840] | 173 |
| train-similar | ours (ranker) | 30.0 | 0.991 | 0.806 [0.717, 0.884] | 0.889 [0.820, 0.942] | 0.833 [0.752, 0.902] | 0.917 [0.857, 0.963] | 0.860 [0.794, 0.914] | 108 |
| train-similar | ours (native order) | 30.0 | 0.991 | 0.630 [0.525, 0.719] | 0.815 [0.733, 0.881] | 0.685 [0.585, 0.772] | 0.833 [0.755, 0.896] | 0.742 [0.665, 0.808] | 108 |
| train-similar | p2rank | 6.6 | 0.954 | 0.796 [0.711, 0.868] | 0.917 [0.858, 0.965] | 0.861 [0.780, 0.923] | 0.944 [0.897, 0.982] | 0.861 [0.798, 0.913] | 108 |
| train-similar | deeppocket | 25.3 | 0.963 | 0.769 [0.685, 0.845] | 0.889 [0.822, 0.945] | 0.778 [0.696, 0.854] | 0.926 [0.875, 0.972] | 0.836 [0.775, 0.891] | 108 |

Paired differences against **p2rank** on the same structures (positive = we are better):

| subset | method | top-1 | top-N | top-(N+2) |
|---|---|---|---|---|
| all | ours (ranker) | -0.028 [-0.078, +0.020] | -0.043 [-0.095, +0.011] | -0.014 [-0.052, +0.025] |
| all | ours (native order) | -0.160 [-0.211, -0.110] | -0.139 [-0.186, -0.093] | -0.078 [-0.119, -0.039] |
| all | deeppocket | -0.043 [-0.095, +0.011] | -0.064 [-0.115, -0.011] | -0.004 [-0.042, +0.037] |
| not train-similar | ours (ranker) | -0.052 [-0.122, +0.012] | -0.052 [-0.124, +0.017] | -0.006 [-0.055, +0.048] |
| not train-similar | ours (native order) | -0.156 [-0.224, -0.094] | -0.116 [-0.176, -0.058] | -0.058 [-0.112, +0.000] |
| not train-similar | deeppocket | -0.052 [-0.124, +0.017] | -0.052 [-0.118, +0.017] | +0.006 [-0.047, +0.061] |
| train-similar | ours (ranker) | +0.009 [-0.057, +0.078] | -0.028 [-0.103, +0.048] | -0.028 [-0.078, +0.018] |
| train-similar | ours (native order) | -0.167 [-0.252, -0.087] | -0.176 [-0.257, -0.103] | -0.111 [-0.177, -0.057] |
| train-similar | deeppocket | -0.028 [-0.104, +0.050] | -0.083 [-0.159, -0.009] | -0.019 [-0.070, +0.032] |

Top-N by the structure's own number of sites, against **p2rank**. At N = 1 top-N is top-1 and no merging of predictions can change it, so a deficit in that row is ranking and a deficit confined to N >= 2 is prediction fragmentation:

| subset | method | N | n | top-N | p2rank top-N | difference | ceiling | p2rank ceiling |
|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 1 | 142 | 0.775 | 0.824 | -0.049 | 1.000 | 0.930 |
| all | ours (ranker) | 2 | 87 | 0.782 | 0.828 | -0.046 | 0.989 | 0.966 |
| all | ours (ranker) | >=3 | 52 | 0.885 | 0.904 | -0.019 | 0.962 | 0.923 |
| all | ours (native order) | 1 | 142 | 0.676 | 0.824 | -0.148 | 1.000 | 0.930 |
| all | ours (native order) | 2 | 87 | 0.736 | 0.828 | -0.092 | 0.989 | 0.966 |
| all | ours (native order) | >=3 | 52 | 0.712 | 0.904 | -0.192 | 0.962 | 0.923 |
| all | deeppocket | 1 | 142 | 0.732 | 0.824 | -0.092 | 0.937 | 0.930 |
| all | deeppocket | 2 | 87 | 0.793 | 0.828 | -0.034 | 0.989 | 0.966 |
| all | deeppocket | >=3 | 52 | 0.865 | 0.904 | -0.038 | 1.000 | 0.923 |
| not train-similar | ours (ranker) | 1 | 77 | 0.701 | 0.792 | -0.091 | 1.000 | 0.909 |
| not train-similar | ours (ranker) | 2 | 56 | 0.786 | 0.821 | -0.036 | 1.000 | 0.964 |
| not train-similar | ours (ranker) | >=3 | 40 | 0.900 | 0.900 | +0.000 | 0.950 | 0.925 |
| not train-similar | ours (native order) | 1 | 77 | 0.649 | 0.792 | -0.143 | 1.000 | 0.909 |
| not train-similar | ours (native order) | 2 | 56 | 0.768 | 0.821 | -0.054 | 1.000 | 0.964 |
| not train-similar | ours (native order) | >=3 | 40 | 0.750 | 0.900 | -0.150 | 0.950 | 0.925 |
| not train-similar | deeppocket | 1 | 77 | 0.701 | 0.792 | -0.091 | 0.935 | 0.909 |
| not train-similar | deeppocket | 2 | 56 | 0.804 | 0.821 | -0.018 | 0.982 | 0.964 |
| not train-similar | deeppocket | >=3 | 40 | 0.875 | 0.900 | -0.025 | 1.000 | 0.925 |
| train-similar | ours (ranker) | 1 | 65 | 0.862 | 0.862 | +0.000 | 1.000 | 0.954 |
| train-similar | ours (ranker) | 2 | 31 | 0.774 | 0.839 | -0.065 | 0.968 | 0.968 |
| train-similar | ours (ranker) | >=3 | 12 | 0.833 | 0.917 | -0.083 | 1.000 | 0.917 |
| train-similar | ours (native order) | 1 | 65 | 0.708 | 0.862 | -0.154 | 1.000 | 0.954 |
| train-similar | ours (native order) | 2 | 31 | 0.677 | 0.839 | -0.161 | 0.968 | 0.968 |
| train-similar | ours (native order) | >=3 | 12 | 0.583 | 0.917 | -0.333 | 1.000 | 0.917 |
| train-similar | deeppocket | 1 | 65 | 0.769 | 0.862 | -0.092 | 0.938 | 0.954 |
| train-similar | deeppocket | 2 | 31 | 0.774 | 0.839 | -0.065 | 1.000 | 0.968 |
| train-similar | deeppocket | >=3 | 12 | 0.833 | 0.917 | -0.083 | 1.000 | 0.917 |
