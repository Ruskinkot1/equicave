# coach420: our predictions against external tools on the same structures

281 structures predicted by every method listed. Success is DCA <= 4 A to a ligand of the set's own relevant-ligand list; N is the structure's own number of ligand sites. 95 % CI by 30 %-identity cluster bootstrap. `train-similar` means the structure shares a 30 %-identity cluster with **our** training manifest; it is leakage for us and not for the external tools, whose own training sets overlap this benchmark in ways this table does not measure. The comparable row for us is **not train-similar**.

| subset | method | predictions | ceiling | top-1 | top-3 | top-N | top-(N+2) | MRR | n |
|---|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 30.0 | 0.989 | 0.740 [0.685, 0.794] | 0.875 [0.837, 0.913] | 0.797 [0.746, 0.845] | 0.890 [0.851, 0.926] | 0.821 [0.781, 0.858] | 281 |
| all | ours (native order) | 30.0 | 0.989 | 0.609 [0.550, 0.666] | 0.801 [0.750, 0.847] | 0.701 [0.645, 0.756] | 0.826 [0.779, 0.869] | 0.723 [0.679, 0.766] | 281 |
| all | deeppocket | 32.6 | 0.964 | 0.726 [0.672, 0.779] | 0.872 [0.830, 0.913] | 0.776 [0.726, 0.825] | 0.900 [0.863, 0.936] | 0.808 [0.767, 0.848] | 281 |
| all | p2rank | 8.4 | 0.940 | 0.769 [0.718, 0.817] | 0.890 [0.851, 0.927] | 0.840 [0.793, 0.883] | 0.904 [0.866, 0.940] | 0.833 [0.793, 0.869] | 281 |
| not train-similar | ours (ranker) | 30.0 | 0.988 | 0.699 [0.626, 0.768] | 0.867 [0.814, 0.916] | 0.775 [0.710, 0.837] | 0.873 [0.822, 0.920] | 0.797 [0.745, 0.845] | 173 |
| not train-similar | ours (native order) | 30.0 | 0.988 | 0.595 [0.520, 0.665] | 0.792 [0.729, 0.851] | 0.711 [0.639, 0.779] | 0.821 [0.760, 0.878] | 0.712 [0.655, 0.763] | 173 |
| not train-similar | deeppocket | 37.1 | 0.965 | 0.699 [0.629, 0.767] | 0.861 [0.806, 0.912] | 0.775 [0.711, 0.833] | 0.884 [0.833, 0.930] | 0.791 [0.738, 0.840] | 173 |
| not train-similar | p2rank | 9.6 | 0.931 | 0.751 [0.689, 0.817] | 0.873 [0.819, 0.918] | 0.827 [0.769, 0.883] | 0.879 [0.826, 0.925] | 0.816 [0.765, 0.866] | 173 |
| train-similar | ours (ranker) | 30.0 | 0.991 | 0.806 [0.717, 0.884] | 0.889 [0.820, 0.942] | 0.833 [0.752, 0.902] | 0.917 [0.857, 0.963] | 0.860 [0.794, 0.914] | 108 |
| train-similar | ours (native order) | 30.0 | 0.991 | 0.630 [0.525, 0.719] | 0.815 [0.733, 0.881] | 0.685 [0.585, 0.772] | 0.833 [0.755, 0.896] | 0.742 [0.665, 0.808] | 108 |
| train-similar | deeppocket | 25.3 | 0.963 | 0.769 [0.685, 0.845] | 0.889 [0.822, 0.945] | 0.778 [0.696, 0.854] | 0.926 [0.875, 0.972] | 0.836 [0.775, 0.891] | 108 |
| train-similar | p2rank | 6.6 | 0.954 | 0.796 [0.711, 0.868] | 0.917 [0.858, 0.965] | 0.861 [0.780, 0.923] | 0.944 [0.897, 0.982] | 0.861 [0.798, 0.913] | 108 |

Paired differences against **deeppocket** on the same structures (positive = we are better):

| subset | method | top-1 | top-N | top-(N+2) |
|---|---|---|---|---|
| all | ours (ranker) | +0.014 [-0.044, +0.068] | +0.021 [-0.026, +0.067] | -0.011 [-0.045, +0.022] |
| all | ours (native order) | -0.117 [-0.176, -0.059] | -0.075 [-0.127, -0.021] | -0.075 [-0.124, -0.029] |
| all | p2rank | +0.043 [-0.011, +0.095] | +0.064 [+0.011, +0.115] | +0.004 [-0.037, +0.042] |
| not train-similar | ours (ranker) | +0.000 [-0.071, +0.072] | +0.000 [-0.063, +0.059] | -0.012 [-0.047, +0.024] |
| not train-similar | ours (native order) | -0.104 [-0.177, -0.030] | -0.064 [-0.129, +0.000] | -0.064 [-0.128, -0.006] |
| not train-similar | p2rank | +0.052 [-0.017, +0.124] | +0.052 [-0.017, +0.118] | -0.006 [-0.061, +0.047] |
| train-similar | ours (ranker) | +0.037 [-0.052, +0.123] | +0.056 [-0.026, +0.134] | -0.009 [-0.079, +0.057] |
| train-similar | ours (native order) | -0.139 [-0.236, -0.045] | -0.093 [-0.184, -0.008] | -0.093 [-0.173, -0.021] |
| train-similar | p2rank | +0.028 [-0.050, +0.104] | +0.083 [+0.009, +0.159] | +0.019 [-0.032, +0.070] |

Top-N by the structure's own number of sites, against **deeppocket**. At N = 1 top-N is top-1 and no merging of predictions can change it, so a deficit in that row is ranking and a deficit confined to N >= 2 is prediction fragmentation:

| subset | method | N | n | top-N | deeppocket top-N | difference | ceiling | deeppocket ceiling |
|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 1 | 142 | 0.775 | 0.732 | +0.042 | 1.000 | 0.937 |
| all | ours (ranker) | 2 | 87 | 0.782 | 0.793 | -0.011 | 0.989 | 0.989 |
| all | ours (ranker) | >=3 | 52 | 0.885 | 0.865 | +0.019 | 0.962 | 1.000 |
| all | ours (native order) | 1 | 142 | 0.676 | 0.732 | -0.056 | 1.000 | 0.937 |
| all | ours (native order) | 2 | 87 | 0.736 | 0.793 | -0.057 | 0.989 | 0.989 |
| all | ours (native order) | >=3 | 52 | 0.712 | 0.865 | -0.154 | 0.962 | 1.000 |
| all | p2rank | 1 | 142 | 0.824 | 0.732 | +0.092 | 0.930 | 0.937 |
| all | p2rank | 2 | 87 | 0.828 | 0.793 | +0.034 | 0.966 | 0.989 |
| all | p2rank | >=3 | 52 | 0.904 | 0.865 | +0.038 | 0.923 | 1.000 |
| not train-similar | ours (ranker) | 1 | 77 | 0.701 | 0.701 | +0.000 | 1.000 | 0.935 |
| not train-similar | ours (ranker) | 2 | 56 | 0.786 | 0.804 | -0.018 | 1.000 | 0.982 |
| not train-similar | ours (ranker) | >=3 | 40 | 0.900 | 0.875 | +0.025 | 0.950 | 1.000 |
| not train-similar | ours (native order) | 1 | 77 | 0.649 | 0.701 | -0.052 | 1.000 | 0.935 |
| not train-similar | ours (native order) | 2 | 56 | 0.768 | 0.804 | -0.036 | 1.000 | 0.982 |
| not train-similar | ours (native order) | >=3 | 40 | 0.750 | 0.875 | -0.125 | 0.950 | 1.000 |
| not train-similar | p2rank | 1 | 77 | 0.792 | 0.701 | +0.091 | 0.909 | 0.935 |
| not train-similar | p2rank | 2 | 56 | 0.821 | 0.804 | +0.018 | 0.964 | 0.982 |
| not train-similar | p2rank | >=3 | 40 | 0.900 | 0.875 | +0.025 | 0.925 | 1.000 |
| train-similar | ours (ranker) | 1 | 65 | 0.862 | 0.769 | +0.092 | 1.000 | 0.938 |
| train-similar | ours (ranker) | 2 | 31 | 0.774 | 0.774 | +0.000 | 0.968 | 1.000 |
| train-similar | ours (ranker) | >=3 | 12 | 0.833 | 0.833 | +0.000 | 1.000 | 1.000 |
| train-similar | ours (native order) | 1 | 65 | 0.708 | 0.769 | -0.062 | 1.000 | 0.938 |
| train-similar | ours (native order) | 2 | 31 | 0.677 | 0.774 | -0.097 | 0.968 | 1.000 |
| train-similar | ours (native order) | >=3 | 12 | 0.583 | 0.833 | -0.250 | 1.000 | 1.000 |
| train-similar | p2rank | 1 | 65 | 0.862 | 0.769 | +0.092 | 0.954 | 0.938 |
| train-similar | p2rank | 2 | 31 | 0.839 | 0.774 | +0.065 | 0.968 | 1.000 |
| train-similar | p2rank | >=3 | 12 | 0.917 | 0.833 | +0.083 | 0.917 | 1.000 |
