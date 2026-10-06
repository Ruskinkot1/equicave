# coach420: our predictions against external tools on the same structures

279 structures predicted by every method listed. Success is DCA <= 4 A to a ligand of the set's own relevant-ligand list; N is the structure's own number of ligand sites. 95 % CI by 30 %-identity cluster bootstrap. `train-similar` means the structure shares a 30 %-identity cluster with **our** training manifest; it is leakage for us and not for the external tools, whose own training sets overlap this benchmark in ways this table does not measure. The comparable row for us is **not train-similar**.

| subset | method | predictions | ceiling | top-1 | top-3 | top-N | top-(N+2) | MRR | n |
|---|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 30.0 | 0.993 | 0.742 [0.684, 0.795] | 0.882 [0.841, 0.918] | 0.803 [0.749, 0.851] | 0.896 [0.858, 0.930] | 0.824 [0.783, 0.862] | 279 |
| all | ours (native order) | 30.0 | 0.993 | 0.616 [0.555, 0.673] | 0.803 [0.754, 0.849] | 0.706 [0.648, 0.760] | 0.828 [0.781, 0.871] | 0.729 [0.683, 0.772] | 279 |
| all | p2rank | 8.4 | 0.939 | 0.774 [0.720, 0.823] | 0.889 [0.850, 0.923] | 0.842 [0.794, 0.884] | 0.903 [0.866, 0.936] | 0.836 [0.795, 0.872] | 279 |
| all | deeppocket | 32.1 | 0.957 | 0.724 [0.669, 0.777] | 0.871 [0.831, 0.909] | 0.774 [0.724, 0.823] | 0.896 [0.861, 0.930] | 0.805 [0.767, 0.844] | 279 |
| all | grasp | 2.0 | 0.789 | 0.731 [0.677, 0.785] | 0.785 [0.734, 0.833] | 0.763 [0.710, 0.814] | 0.789 [0.738, 0.837] | 0.758 [0.705, 0.808] | 279 |
| not train-similar | ours (ranker) | 30.0 | 0.994 | 0.706 [0.634, 0.775] | 0.876 [0.824, 0.924] | 0.782 [0.715, 0.844] | 0.882 [0.832, 0.927] | 0.804 [0.751, 0.852] | 170 |
| not train-similar | ours (native order) | 30.0 | 0.994 | 0.606 [0.531, 0.679] | 0.794 [0.731, 0.853] | 0.718 [0.647, 0.781] | 0.824 [0.765, 0.879] | 0.719 [0.665, 0.772] | 170 |
| not train-similar | p2rank | 9.6 | 0.929 | 0.759 [0.691, 0.822] | 0.871 [0.818, 0.920] | 0.829 [0.770, 0.886] | 0.876 [0.826, 0.925] | 0.819 [0.768, 0.869] | 170 |
| not train-similar | deeppocket | 36.6 | 0.959 | 0.700 [0.629, 0.771] | 0.865 [0.808, 0.914] | 0.776 [0.712, 0.840] | 0.882 [0.829, 0.929] | 0.791 [0.736, 0.840] | 170 |
| not train-similar | grasp | 2.1 | 0.741 | 0.676 [0.602, 0.747] | 0.735 [0.665, 0.801] | 0.718 [0.647, 0.785] | 0.741 [0.673, 0.805] | 0.706 [0.637, 0.772] | 170 |
| train-similar | ours (ranker) | 30.0 | 0.991 | 0.798 [0.713, 0.875] | 0.890 [0.826, 0.944] | 0.835 [0.754, 0.905] | 0.917 [0.860, 0.964] | 0.857 [0.793, 0.912] | 109 |
| train-similar | ours (native order) | 30.0 | 0.991 | 0.633 [0.528, 0.726] | 0.817 [0.735, 0.887] | 0.688 [0.585, 0.776] | 0.835 [0.755, 0.901] | 0.744 [0.665, 0.812] | 109 |
| train-similar | p2rank | 6.5 | 0.954 | 0.798 [0.712, 0.876] | 0.917 [0.856, 0.965] | 0.862 [0.780, 0.925] | 0.945 [0.894, 0.983] | 0.863 [0.799, 0.915] | 109 |
| train-similar | deeppocket | 25.1 | 0.954 | 0.761 [0.674, 0.845] | 0.881 [0.813, 0.942] | 0.771 [0.685, 0.853] | 0.917 [0.860, 0.966] | 0.828 [0.766, 0.888] | 109 |
| train-similar | grasp | 1.9 | 0.862 | 0.817 [0.735, 0.886] | 0.862 [0.788, 0.923] | 0.835 [0.757, 0.903] | 0.862 [0.788, 0.923] | 0.839 [0.766, 0.904] | 109 |

Paired differences against **p2rank** on the same structures (positive = we are better):

| subset | method | top-1 | top-N | top-(N+2) |
|---|---|---|---|---|
| all | ours (ranker) | -0.032 [-0.082, +0.017] | -0.039 [-0.090, +0.011] | -0.007 [-0.044, +0.030] |
| all | ours (native order) | -0.158 [-0.212, -0.107] | -0.136 [-0.185, -0.091] | -0.075 [-0.115, -0.035] |
| all | deeppocket | -0.050 [-0.104, +0.007] | -0.068 [-0.119, -0.015] | -0.007 [-0.044, +0.032] |
| all | grasp | -0.043 [-0.103, +0.015] | -0.079 [-0.133, -0.025] | -0.115 [-0.167, -0.065] |
| not train-similar | ours (ranker) | -0.053 [-0.124, +0.017] | -0.047 [-0.121, +0.022] | +0.006 [-0.048, +0.057] |
| not train-similar | ours (native order) | -0.153 [-0.222, -0.083] | -0.112 [-0.169, -0.055] | -0.053 [-0.106, +0.000] |
| not train-similar | deeppocket | -0.059 [-0.135, +0.012] | -0.053 [-0.120, +0.012] | +0.006 [-0.053, +0.058] |
| not train-similar | grasp | -0.082 [-0.166, +0.000] | -0.112 [-0.192, -0.035] | -0.135 [-0.212, -0.064] |
| train-similar | ours (ranker) | +0.000 [-0.071, +0.073] | -0.028 [-0.105, +0.050] | -0.028 [-0.079, +0.019] |
| train-similar | ours (native order) | -0.165 [-0.252, -0.088] | -0.174 [-0.255, -0.102] | -0.110 [-0.178, -0.056] |
| train-similar | deeppocket | -0.037 [-0.113, +0.046] | -0.092 [-0.165, -0.016] | -0.028 [-0.081, +0.027] |
| train-similar | grasp | +0.018 [-0.076, +0.106] | -0.028 [-0.110, +0.049] | -0.083 [-0.157, -0.017] |

Top-N by the structure's own number of sites, against **p2rank**. At N = 1 top-N is top-1 and no merging of predictions can change it, so a deficit in that row is ranking and a deficit confined to N >= 2 is prediction fragmentation:

| subset | method | N | n | top-N | p2rank top-N | difference | ceiling | p2rank ceiling |
|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 1 | 142 | 0.775 | 0.824 | -0.049 | 1.000 | 0.930 |
| all | ours (ranker) | 2 | 87 | 0.793 | 0.839 | -0.046 | 0.989 | 0.966 |
| all | ours (ranker) | >=3 | 50 | 0.900 | 0.900 | +0.000 | 0.980 | 0.920 |
| all | ours (native order) | 1 | 142 | 0.676 | 0.824 | -0.148 | 1.000 | 0.930 |
| all | ours (native order) | 2 | 87 | 0.747 | 0.839 | -0.092 | 0.989 | 0.966 |
| all | ours (native order) | >=3 | 50 | 0.720 | 0.900 | -0.180 | 0.980 | 0.920 |
| all | deeppocket | 1 | 142 | 0.732 | 0.824 | -0.092 | 0.937 | 0.930 |
| all | deeppocket | 2 | 87 | 0.782 | 0.839 | -0.057 | 0.966 | 0.966 |
| all | deeppocket | >=3 | 50 | 0.880 | 0.900 | -0.020 | 1.000 | 0.920 |
| all | grasp | 1 | 142 | 0.782 | 0.824 | -0.042 | 0.831 | 0.930 |
| all | grasp | 2 | 87 | 0.816 | 0.839 | -0.023 | 0.816 | 0.966 |
| all | grasp | >=3 | 50 | 0.620 | 0.900 | -0.280 | 0.620 | 0.920 |
| not train-similar | ours (ranker) | 1 | 77 | 0.701 | 0.792 | -0.091 | 1.000 | 0.909 |
| not train-similar | ours (ranker) | 2 | 55 | 0.800 | 0.836 | -0.036 | 1.000 | 0.964 |
| not train-similar | ours (ranker) | >=3 | 38 | 0.921 | 0.895 | +0.026 | 0.974 | 0.921 |
| not train-similar | ours (native order) | 1 | 77 | 0.649 | 0.792 | -0.143 | 1.000 | 0.909 |
| not train-similar | ours (native order) | 2 | 55 | 0.782 | 0.836 | -0.055 | 1.000 | 0.964 |
| not train-similar | ours (native order) | >=3 | 38 | 0.763 | 0.895 | -0.132 | 0.974 | 0.921 |
| not train-similar | deeppocket | 1 | 77 | 0.701 | 0.792 | -0.091 | 0.935 | 0.909 |
| not train-similar | deeppocket | 2 | 55 | 0.800 | 0.836 | -0.036 | 0.964 | 0.964 |
| not train-similar | deeppocket | >=3 | 38 | 0.895 | 0.895 | +0.000 | 1.000 | 0.921 |
| not train-similar | grasp | 1 | 77 | 0.714 | 0.792 | -0.078 | 0.766 | 0.909 |
| not train-similar | grasp | 2 | 55 | 0.800 | 0.836 | -0.036 | 0.800 | 0.964 |
| not train-similar | grasp | >=3 | 38 | 0.605 | 0.895 | -0.289 | 0.605 | 0.921 |
| train-similar | ours (ranker) | 1 | 65 | 0.862 | 0.862 | +0.000 | 1.000 | 0.954 |
| train-similar | ours (ranker) | 2 | 32 | 0.781 | 0.844 | -0.062 | 0.969 | 0.969 |
| train-similar | ours (ranker) | >=3 | 12 | 0.833 | 0.917 | -0.083 | 1.000 | 0.917 |
| train-similar | ours (native order) | 1 | 65 | 0.708 | 0.862 | -0.154 | 1.000 | 0.954 |
| train-similar | ours (native order) | 2 | 32 | 0.688 | 0.844 | -0.156 | 0.969 | 0.969 |
| train-similar | ours (native order) | >=3 | 12 | 0.583 | 0.917 | -0.333 | 1.000 | 0.917 |
| train-similar | deeppocket | 1 | 65 | 0.769 | 0.862 | -0.092 | 0.938 | 0.954 |
| train-similar | deeppocket | 2 | 32 | 0.750 | 0.844 | -0.094 | 0.969 | 0.969 |
| train-similar | deeppocket | >=3 | 12 | 0.833 | 0.917 | -0.083 | 1.000 | 0.917 |
| train-similar | grasp | 1 | 65 | 0.862 | 0.862 | +0.000 | 0.908 | 0.954 |
| train-similar | grasp | 2 | 32 | 0.844 | 0.844 | +0.000 | 0.844 | 0.969 |
| train-similar | grasp | >=3 | 12 | 0.667 | 0.917 | -0.250 | 0.667 | 0.917 |
