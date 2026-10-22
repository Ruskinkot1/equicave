# coach420: our predictions against external tools on the same structures

269 structures predicted by every method listed. Success is DCA <= 4 A to a ligand of the set's own relevant-ligand list; N is the structure's own number of ligand sites. 95 % CI by 30 %-identity cluster bootstrap. `train-similar` means the structure shares a 30 %-identity cluster with **our** training manifest; it is leakage for us and not for the external tools, whose own training sets overlap this benchmark in ways this table does not measure. The comparable row for us is **not train-similar**.

| subset | method | predictions | ceiling | top-1 | top-3 | top-N | top-(N+2) | MRR | n |
|---|---|---|---|---|---|---|---|---|---|
| all | ours (network only) | 30.0 | 0.993 | 0.870 [0.822, 0.909] | 0.941 [0.910, 0.966] | 0.892 [0.850, 0.930] | 0.941 [0.910, 0.966] | 0.910 [0.877, 0.938] | 269 |
| all | ours (ranker+net) | 30.0 | 0.993 | 0.844 [0.795, 0.888] | 0.933 [0.902, 0.962] | 0.885 [0.843, 0.922] | 0.944 [0.915, 0.970] | 0.895 [0.862, 0.926] | 269 |
| all | p2rank | 7.0 | 0.937 | 0.781 [0.728, 0.831] | 0.888 [0.849, 0.924] | 0.840 [0.793, 0.884] | 0.903 [0.865, 0.937] | 0.840 [0.798, 0.878] | 269 |
| all | grasp | 2.0 | 0.803 | 0.755 [0.699, 0.805] | 0.803 [0.749, 0.849] | 0.777 [0.720, 0.826] | 0.803 [0.749, 0.849] | 0.778 [0.724, 0.826] | 269 |
| all | deeppocket | 27.9 | 0.959 | 0.729 [0.673, 0.779] | 0.877 [0.832, 0.915] | 0.773 [0.719, 0.822] | 0.900 [0.858, 0.933] | 0.810 [0.768, 0.847] | 269 |
| all | deepsurf | 1.8 | 0.862 | 0.818 [0.767, 0.865] | 0.862 [0.819, 0.902] | 0.836 [0.787, 0.882] | 0.862 [0.819, 0.902] | 0.840 [0.794, 0.882] | 269 |
| not train-similar | ours (network only) | 30.0 | 0.991 | 0.851 [0.802, 0.899] | 0.930 [0.895, 0.962] | 0.873 [0.828, 0.917] | 0.930 [0.895, 0.962] | 0.896 [0.861, 0.931] | 228 |
| not train-similar | ours (ranker+net) | 30.0 | 0.991 | 0.825 [0.772, 0.876] | 0.921 [0.882, 0.954] | 0.864 [0.816, 0.910] | 0.934 [0.897, 0.965] | 0.881 [0.842, 0.917] | 228 |
| not train-similar | p2rank | 6.8 | 0.930 | 0.759 [0.701, 0.814] | 0.873 [0.828, 0.915] | 0.816 [0.765, 0.868] | 0.890 [0.848, 0.931] | 0.822 [0.779, 0.865] | 228 |
| not train-similar | grasp | 2.0 | 0.781 | 0.732 [0.671, 0.789] | 0.781 [0.723, 0.835] | 0.754 [0.693, 0.810] | 0.781 [0.723, 0.835] | 0.756 [0.699, 0.810] | 228 |
| not train-similar | deeppocket | 27.0 | 0.956 | 0.715 [0.657, 0.775] | 0.873 [0.826, 0.914] | 0.763 [0.706, 0.819] | 0.895 [0.851, 0.932] | 0.800 [0.755, 0.844] | 228 |
| not train-similar | deepsurf | 1.8 | 0.842 | 0.807 [0.753, 0.859] | 0.842 [0.791, 0.890] | 0.816 [0.763, 0.866] | 0.842 [0.791, 0.890] | 0.824 [0.774, 0.872] | 228 |
| train-similar | ours (network only) | 30.0 | 1.000 | 0.976 [0.914, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.988 [0.957, 1.000] | 41 |
| train-similar | ours (ranker+net) | 30.0 | 1.000 | 0.951 [0.865, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.976 [0.932, 1.000] | 41 |
| train-similar | p2rank | 8.1 | 0.976 | 0.902 [0.792, 0.979] | 0.976 [0.914, 1.000] | 0.976 [0.914, 1.000] | 0.976 [0.914, 1.000] | 0.939 [0.860, 0.989] | 41 |
| train-similar | grasp | 2.1 | 0.927 | 0.878 [0.765, 0.971] | 0.927 [0.825, 1.000] | 0.902 [0.808, 0.976] | 0.927 [0.825, 1.000] | 0.902 [0.800, 0.974] | 41 |
| train-similar | deeppocket | 32.6 | 0.976 | 0.805 [0.674, 0.925] | 0.902 [0.794, 0.978] | 0.829 [0.705, 0.944] | 0.927 [0.829, 1.000] | 0.864 [0.767, 0.946] | 41 |
| train-similar | deepsurf | 1.8 | 0.976 | 0.878 [0.750, 0.973] | 0.976 [0.914, 1.000] | 0.951 [0.871, 1.000] | 0.976 [0.914, 1.000] | 0.927 [0.838, 0.984] | 41 |

Paired differences against **deepsurf** on the same structures (positive = we are better):

| subset | method | top-1 | top-N | top-(N+2) |
|---|---|---|---|---|
| all | ours (network only) | +0.052 [+0.008, +0.099] | +0.056 [+0.012, +0.103] | +0.078 [+0.036, +0.122] |
| all | ours (ranker+net) | +0.026 [-0.023, +0.077] | +0.048 [+0.008, +0.093] | +0.082 [+0.046, +0.122] |
| all | p2rank | -0.037 [-0.084, +0.011] | +0.004 [-0.038, +0.047] | +0.041 [+0.000, +0.082] |
| all | grasp | -0.063 [-0.118, -0.008] | -0.059 [-0.117, -0.004] | -0.059 [-0.110, -0.011] |
| all | deeppocket | -0.089 [-0.137, -0.044] | -0.063 [-0.113, -0.015] | +0.037 [-0.007, +0.083] |
| not train-similar | ours (network only) | +0.044 [-0.004, +0.094] | +0.057 [+0.009, +0.108] | +0.088 [+0.041, +0.138] |
| not train-similar | ours (ranker+net) | +0.018 [-0.036, +0.075] | +0.048 [+0.000, +0.099] | +0.092 [+0.049, +0.140] |
| not train-similar | p2rank | -0.048 [-0.099, +0.000] | +0.000 [-0.045, +0.047] | +0.048 [+0.004, +0.095] |
| not train-similar | grasp | -0.075 [-0.135, -0.017] | -0.061 [-0.121, -0.004] | -0.061 [-0.119, -0.004] |
| not train-similar | deeppocket | -0.092 [-0.142, -0.044] | -0.053 [-0.108, +0.000] | +0.053 [+0.000, +0.105] |
| train-similar | ours (network only) | +0.098 [+0.022, +0.214] | +0.049 [+0.000, +0.129] | +0.024 [+0.000, +0.086] |
| train-similar | ours (ranker+net) | +0.073 [-0.027, +0.200] | +0.049 [+0.000, +0.129] | +0.024 [+0.000, +0.086] |
| train-similar | p2rank | +0.024 [-0.053, +0.118] | +0.024 [+0.000, +0.085] | +0.000 [+0.000, +0.000] |
| train-similar | grasp | +0.000 [-0.149, +0.171] | -0.049 [-0.163, +0.073] | -0.049 [-0.158, +0.042] |
| train-similar | deeppocket | -0.073 [-0.196, +0.062] | -0.122 [-0.231, -0.026] | -0.049 [-0.128, +0.000] |

Top-N by the structure's own number of sites, against **deepsurf**. At N = 1 top-N is top-1 and no merging of predictions can change it, so a deficit in that row is ranking and a deficit confined to N >= 2 is prediction fragmentation:

| subset | method | N | n | top-N | deepsurf top-N | difference | ceiling | deepsurf ceiling |
|---|---|---|---|---|---|---|---|---|
| all | ours (network only) | 1 | 142 | 0.845 | 0.838 | +0.007 | 1.000 | 0.880 |
| all | ours (network only) | 2 | 86 | 0.942 | 0.837 | +0.105 | 0.988 | 0.849 |
| all | ours (network only) | >=3 | 41 | 0.951 | 0.829 | +0.122 | 0.976 | 0.829 |
| all | ours (ranker+net) | 1 | 142 | 0.859 | 0.838 | +0.021 | 1.000 | 0.880 |
| all | ours (ranker+net) | 2 | 86 | 0.919 | 0.837 | +0.081 | 0.988 | 0.849 |
| all | ours (ranker+net) | >=3 | 41 | 0.902 | 0.829 | +0.073 | 0.976 | 0.829 |
| all | p2rank | 1 | 142 | 0.824 | 0.838 | -0.014 | 0.930 | 0.880 |
| all | p2rank | 2 | 86 | 0.849 | 0.837 | +0.012 | 0.965 | 0.849 |
| all | p2rank | >=3 | 41 | 0.878 | 0.829 | +0.049 | 0.902 | 0.829 |
| all | grasp | 1 | 142 | 0.782 | 0.838 | -0.056 | 0.831 | 0.880 |
| all | grasp | 2 | 86 | 0.826 | 0.837 | -0.012 | 0.826 | 0.849 |
| all | grasp | >=3 | 41 | 0.659 | 0.829 | -0.171 | 0.659 | 0.829 |
| all | deeppocket | 1 | 142 | 0.732 | 0.838 | -0.106 | 0.937 | 0.880 |
| all | deeppocket | 2 | 86 | 0.791 | 0.837 | -0.047 | 0.977 | 0.849 |
| all | deeppocket | >=3 | 41 | 0.878 | 0.829 | +0.049 | 1.000 | 0.829 |
| not train-similar | ours (network only) | 1 | 124 | 0.823 | 0.823 | +0.000 | 1.000 | 0.863 |
| not train-similar | ours (network only) | 2 | 70 | 0.929 | 0.800 | +0.129 | 0.986 | 0.814 |
| not train-similar | ours (network only) | >=3 | 34 | 0.941 | 0.824 | +0.118 | 0.971 | 0.824 |
| not train-similar | ours (ranker+net) | 1 | 124 | 0.839 | 0.823 | +0.016 | 1.000 | 0.863 |
| not train-similar | ours (ranker+net) | 2 | 70 | 0.900 | 0.800 | +0.100 | 0.986 | 0.814 |
| not train-similar | ours (ranker+net) | >=3 | 34 | 0.882 | 0.824 | +0.059 | 0.971 | 0.824 |
| not train-similar | p2rank | 1 | 124 | 0.798 | 0.823 | -0.024 | 0.919 | 0.863 |
| not train-similar | p2rank | 2 | 70 | 0.814 | 0.800 | +0.014 | 0.957 | 0.814 |
| not train-similar | p2rank | >=3 | 34 | 0.882 | 0.824 | +0.059 | 0.912 | 0.824 |
| not train-similar | grasp | 1 | 124 | 0.758 | 0.823 | -0.065 | 0.806 | 0.863 |
| not train-similar | grasp | 2 | 70 | 0.800 | 0.800 | +0.000 | 0.800 | 0.814 |
| not train-similar | grasp | >=3 | 34 | 0.647 | 0.824 | -0.176 | 0.647 | 0.824 |
| not train-similar | deeppocket | 1 | 124 | 0.718 | 0.823 | -0.105 | 0.927 | 0.863 |
| not train-similar | deeppocket | 2 | 70 | 0.786 | 0.800 | -0.014 | 0.986 | 0.814 |
| not train-similar | deeppocket | >=3 | 34 | 0.882 | 0.824 | +0.059 | 1.000 | 0.824 |
| train-similar | ours (network only) | 1 | 18 | 1.000 | 0.944 | +0.056 | 1.000 | 1.000 |
| train-similar | ours (network only) | 2 | 16 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 |
| train-similar | ours (network only) | >=3 | 7 | 1.000 | 0.857 | +0.143 | 1.000 | 0.857 |
| train-similar | ours (ranker+net) | 1 | 18 | 1.000 | 0.944 | +0.056 | 1.000 | 1.000 |
| train-similar | ours (ranker+net) | 2 | 16 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 |
| train-similar | ours (ranker+net) | >=3 | 7 | 1.000 | 0.857 | +0.143 | 1.000 | 0.857 |
| train-similar | p2rank | 1 | 18 | 1.000 | 0.944 | +0.056 | 1.000 | 1.000 |
| train-similar | p2rank | 2 | 16 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 |
| train-similar | p2rank | >=3 | 7 | 0.857 | 0.857 | +0.000 | 0.857 | 0.857 |
| train-similar | grasp | 1 | 18 | 0.944 | 0.944 | +0.000 | 1.000 | 1.000 |
| train-similar | grasp | 2 | 16 | 0.938 | 1.000 | -0.062 | 0.938 | 1.000 |
| train-similar | grasp | >=3 | 7 | 0.714 | 0.857 | -0.143 | 0.714 | 0.857 |
| train-similar | deeppocket | 1 | 18 | 0.833 | 0.944 | -0.111 | 1.000 | 1.000 |
| train-similar | deeppocket | 2 | 16 | 0.812 | 1.000 | -0.188 | 0.938 | 1.000 |
| train-similar | deeppocket | >=3 | 7 | 0.857 | 0.857 | +0.000 | 1.000 | 0.857 |
