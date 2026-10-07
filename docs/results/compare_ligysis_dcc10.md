# ligysis: our predictions against external tools on the same structures

1613 structures predicted by every method listed. Success is DCC <= 10 A to a ligand centroid of the set's own relevant-ligand list; N is the structure's own number of ligand sites. 95 % CI by 30 %-identity cluster bootstrap. `train-similar` means the structure shares a 30 %-identity cluster with **our** training manifest; it is leakage for us and not for the external tools, whose own training sets overlap this benchmark in ways this table does not measure. The comparable row for us is **not train-similar**.

| subset | method | predictions | ceiling | top-1 | top-3 | top-N | top-(N+2) | MRR | n |
|---|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 30.8 | 0.953 | 0.683 [0.658, 0.708] | 0.816 [0.796, 0.838] | 0.772 [0.750, 0.795] | 0.855 [0.837, 0.873] | 0.761 [0.741, 0.781] | 1613 |
| all | ours (native order) | 30.8 | 0.953 | 0.475 [0.447, 0.505] | 0.692 [0.665, 0.719] | 0.630 [0.599, 0.660] | 0.771 [0.747, 0.797] | 0.604 [0.580, 0.628] | 1613 |
| all | p2rank | 20.2 | 0.929 | 0.704 [0.676, 0.731] | 0.828 [0.807, 0.848] | 0.795 [0.770, 0.820] | 0.860 [0.841, 0.878] | 0.775 [0.754, 0.796] | 1613 |
| not train-similar | ours (ranker) | 31.1 | 0.945 | 0.641 [0.611, 0.670] | 0.790 [0.765, 0.817] | 0.745 [0.717, 0.774] | 0.834 [0.809, 0.857] | 0.728 [0.704, 0.752] | 1088 |
| not train-similar | ours (native order) | 31.1 | 0.945 | 0.457 [0.427, 0.487] | 0.669 [0.640, 0.699] | 0.612 [0.583, 0.642] | 0.754 [0.727, 0.781] | 0.587 [0.561, 0.612] | 1088 |
| not train-similar | p2rank | 20.7 | 0.919 | 0.689 [0.660, 0.718] | 0.805 [0.779, 0.830] | 0.780 [0.755, 0.807] | 0.838 [0.814, 0.862] | 0.758 [0.734, 0.782] | 1088 |
| train-similar | ours (ranker) | 30.1 | 0.970 | 0.770 [0.725, 0.813] | 0.870 [0.840, 0.899] | 0.829 [0.791, 0.865] | 0.899 [0.870, 0.926] | 0.828 [0.795, 0.860] | 525 |
| train-similar | ours (native order) | 30.1 | 0.970 | 0.512 [0.455, 0.573] | 0.739 [0.689, 0.789] | 0.667 [0.603, 0.728] | 0.808 [0.761, 0.852] | 0.640 [0.592, 0.688] | 525 |
| train-similar | p2rank | 19.2 | 0.950 | 0.733 [0.677, 0.784] | 0.874 [0.840, 0.906] | 0.825 [0.772, 0.870] | 0.905 [0.873, 0.934] | 0.810 [0.768, 0.846] | 525 |

Paired differences against **p2rank** on the same structures (positive = we are better):

| subset | method | top-1 | top-N | top-(N+2) |
|---|---|---|---|---|
| all | ours (ranker) | -0.021 [-0.046, +0.003] | -0.022 [-0.043, -0.002] | -0.005 [-0.020, +0.011] |
| all | ours (native order) | -0.229 [-0.257, -0.203] | -0.165 [-0.189, -0.141] | -0.089 [-0.109, -0.067] |
| not train-similar | ours (ranker) | -0.049 [-0.078, -0.018] | -0.035 [-0.059, -0.010] | -0.005 [-0.025, +0.016] |
| not train-similar | ours (native order) | -0.233 [-0.264, -0.201] | -0.168 [-0.197, -0.140] | -0.085 [-0.111, -0.060] |
| train-similar | ours (ranker) | +0.036 [-0.008, +0.082] | +0.004 [-0.036, +0.043] | -0.006 [-0.031, +0.018] |
| train-similar | ours (native order) | -0.221 [-0.271, -0.169] | -0.158 [-0.203, -0.110] | -0.097 [-0.137, -0.059] |

Top-N by the structure's own number of sites, against **p2rank**. At N = 1 top-N is top-1 and no merging of predictions can change it, so a deficit in that row is ranking and a deficit confined to N >= 2 is prediction fragmentation:

| subset | method | N | n | top-N | p2rank top-N | difference | ceiling | p2rank ceiling |
|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 1 | 536 | 0.685 | 0.711 | -0.026 | 0.951 | 0.901 |
| all | ours (ranker) | 2 | 489 | 0.771 | 0.810 | -0.039 | 0.955 | 0.937 |
| all | ours (ranker) | >=3 | 588 | 0.854 | 0.859 | -0.005 | 0.952 | 0.949 |
| all | ours (native order) | 1 | 536 | 0.519 | 0.711 | -0.192 | 0.951 | 0.901 |
| all | ours (native order) | 2 | 489 | 0.620 | 0.810 | -0.190 | 0.955 | 0.937 |
| all | ours (native order) | >=3 | 588 | 0.740 | 0.859 | -0.119 | 0.952 | 0.949 |
| not train-similar | ours (ranker) | 1 | 369 | 0.653 | 0.724 | -0.070 | 0.943 | 0.897 |
| not train-similar | ours (ranker) | 2 | 345 | 0.736 | 0.783 | -0.046 | 0.948 | 0.919 |
| not train-similar | ours (ranker) | >=3 | 374 | 0.845 | 0.834 | +0.011 | 0.944 | 0.941 |
| not train-similar | ours (native order) | 1 | 369 | 0.518 | 0.724 | -0.206 | 0.943 | 0.897 |
| not train-similar | ours (native order) | 2 | 345 | 0.594 | 0.783 | -0.188 | 0.948 | 0.919 |
| not train-similar | ours (native order) | >=3 | 374 | 0.722 | 0.834 | -0.112 | 0.944 | 0.941 |
| train-similar | ours (ranker) | 1 | 167 | 0.754 | 0.683 | +0.072 | 0.970 | 0.910 |
| train-similar | ours (ranker) | 2 | 144 | 0.854 | 0.875 | -0.021 | 0.972 | 0.979 |
| train-similar | ours (ranker) | >=3 | 214 | 0.869 | 0.902 | -0.033 | 0.967 | 0.963 |
| train-similar | ours (native order) | 1 | 167 | 0.521 | 0.683 | -0.162 | 0.970 | 0.910 |
| train-similar | ours (native order) | 2 | 144 | 0.681 | 0.875 | -0.194 | 0.972 | 0.979 |
| train-similar | ours (native order) | >=3 | 214 | 0.771 | 0.902 | -0.131 | 0.967 | 0.963 |
