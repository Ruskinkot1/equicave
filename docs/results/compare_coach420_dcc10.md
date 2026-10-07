# coach420: our predictions against external tools on the same structures

269 structures predicted by every method listed. Success is DCC <= 10 A to a ligand centroid of the set's own relevant-ligand list; N is the structure's own number of ligand sites. 95 % CI by 30 %-identity cluster bootstrap. `train-similar` means the structure shares a 30 %-identity cluster with **our** training manifest; it is leakage for us and not for the external tools, whose own training sets overlap this benchmark in ways this table does not measure. The comparable row for us is **not train-similar**.

| subset | method | predictions | ceiling | top-1 | top-3 | top-N | top-(N+2) | MRR | n |
|---|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 30.0 | 0.993 | 0.766 [0.711, 0.820] | 0.896 [0.856, 0.931] | 0.829 [0.779, 0.876] | 0.903 [0.866, 0.938] | 0.843 [0.804, 0.879] | 269 |
| all | ours (native order) | 30.0 | 0.993 | 0.673 [0.615, 0.728] | 0.836 [0.790, 0.880] | 0.740 [0.684, 0.791] | 0.862 [0.817, 0.904] | 0.773 [0.730, 0.813] | 269 |
| all | p2rank | 7.0 | 0.955 | 0.796 [0.742, 0.845] | 0.914 [0.881, 0.947] | 0.859 [0.814, 0.900] | 0.926 [0.891, 0.956] | 0.859 [0.821, 0.892] | 269 |
| all | deeppocket | 27.9 | 0.978 | 0.773 [0.720, 0.821] | 0.914 [0.878, 0.944] | 0.818 [0.770, 0.863] | 0.926 [0.891, 0.955] | 0.849 [0.811, 0.882] | 269 |
| all | grasp | 2.0 | 0.822 | 0.777 [0.722, 0.828] | 0.822 [0.772, 0.866] | 0.796 [0.743, 0.844] | 0.822 [0.772, 0.866] | 0.799 [0.749, 0.846] | 269 |
| all | deepsurf | 1.8 | 0.888 | 0.855 [0.810, 0.898] | 0.885 [0.844, 0.924] | 0.866 [0.822, 0.907] | 0.888 [0.848, 0.926] | 0.871 [0.828, 0.911] | 269 |
| not train-similar | ours (ranker) | 30.0 | 0.994 | 0.733 [0.661, 0.802] | 0.882 [0.829, 0.927] | 0.807 [0.744, 0.866] | 0.888 [0.837, 0.933] | 0.822 [0.771, 0.870] | 161 |
| not train-similar | ours (native order) | 30.0 | 0.994 | 0.665 [0.594, 0.736] | 0.820 [0.758, 0.877] | 0.745 [0.677, 0.812] | 0.851 [0.794, 0.904] | 0.764 [0.711, 0.815] | 161 |
| not train-similar | p2rank | 7.6 | 0.944 | 0.783 [0.716, 0.844] | 0.907 [0.860, 0.950] | 0.851 [0.794, 0.904] | 0.907 [0.860, 0.950] | 0.845 [0.795, 0.890] | 161 |
| not train-similar | deeppocket | 30.4 | 0.969 | 0.739 [0.669, 0.805] | 0.894 [0.846, 0.938] | 0.801 [0.741, 0.860] | 0.901 [0.851, 0.942] | 0.823 [0.774, 0.868] | 161 |
| not train-similar | grasp | 2.0 | 0.783 | 0.727 [0.654, 0.795] | 0.783 [0.713, 0.844] | 0.752 [0.677, 0.815] | 0.783 [0.713, 0.844] | 0.755 [0.685, 0.816] | 161 |
| not train-similar | deepsurf | 2.0 | 0.863 | 0.845 [0.786, 0.897] | 0.857 [0.800, 0.909] | 0.851 [0.794, 0.904] | 0.863 [0.808, 0.913] | 0.852 [0.795, 0.904] | 161 |
| train-similar | ours (ranker) | 30.0 | 0.991 | 0.815 [0.728, 0.893] | 0.917 [0.858, 0.963] | 0.861 [0.780, 0.927] | 0.926 [0.869, 0.972] | 0.876 [0.816, 0.928] | 108 |
| train-similar | ours (native order) | 30.0 | 0.991 | 0.685 [0.594, 0.775] | 0.861 [0.790, 0.925] | 0.731 [0.640, 0.814] | 0.880 [0.814, 0.938] | 0.786 [0.717, 0.850] | 108 |
| train-similar | p2rank | 6.2 | 0.972 | 0.815 [0.731, 0.888] | 0.926 [0.869, 0.972] | 0.870 [0.796, 0.932] | 0.954 [0.908, 0.990] | 0.880 [0.822, 0.929] | 108 |
| train-similar | deeppocket | 24.1 | 0.991 | 0.824 [0.748, 0.893] | 0.944 [0.897, 0.983] | 0.843 [0.773, 0.910] | 0.963 [0.924, 0.992] | 0.886 [0.836, 0.933] | 108 |
| train-similar | grasp | 1.9 | 0.880 | 0.852 [0.774, 0.915] | 0.880 [0.804, 0.935] | 0.861 [0.785, 0.922] | 0.880 [0.804, 0.935] | 0.866 [0.793, 0.925] | 108 |
| train-similar | deepsurf | 1.6 | 0.926 | 0.870 [0.792, 0.933] | 0.926 [0.868, 0.973] | 0.889 [0.813, 0.947] | 0.926 [0.868, 0.973] | 0.898 [0.835, 0.947] | 108 |

Paired differences against **p2rank** on the same structures (positive = we are better):

| subset | method | top-1 | top-N | top-(N+2) |
|---|---|---|---|---|
| all | ours (ranker) | -0.030 [-0.080, +0.021] | -0.030 [-0.075, +0.019] | -0.022 [-0.058, +0.015] |
| all | ours (native order) | -0.123 [-0.172, -0.072] | -0.119 [-0.167, -0.071] | -0.063 [-0.107, -0.023] |
| all | deeppocket | -0.022 [-0.074, +0.029] | -0.041 [-0.088, +0.007] | +0.000 [-0.040, +0.039] |
| all | grasp | -0.019 [-0.075, +0.040] | -0.063 [-0.117, -0.011] | -0.104 [-0.156, -0.057] |
| all | deepsurf | +0.059 [+0.015, +0.106] | +0.007 [-0.031, +0.046] | -0.037 [-0.080, +0.004] |
| not train-similar | ours (ranker) | -0.050 [-0.121, +0.019] | -0.043 [-0.108, +0.019] | -0.019 [-0.073, +0.036] |
| not train-similar | ours (native order) | -0.118 [-0.187, -0.051] | -0.106 [-0.168, -0.045] | -0.056 [-0.109, -0.006] |
| not train-similar | deeppocket | -0.043 [-0.117, +0.029] | -0.050 [-0.117, +0.017] | -0.006 [-0.063, +0.048] |
| not train-similar | grasp | -0.056 [-0.135, +0.019] | -0.099 [-0.169, -0.031] | -0.124 [-0.194, -0.059] |
| not train-similar | deepsurf | +0.062 [+0.000, +0.127] | +0.000 [-0.057, +0.056] | -0.043 [-0.106, +0.018] |
| train-similar | ours (ranker) | +0.000 [-0.071, +0.071] | -0.009 [-0.077, +0.059] | -0.028 [-0.079, +0.019] |
| train-similar | ours (native order) | -0.130 [-0.212, -0.054] | -0.139 [-0.216, -0.071] | -0.074 [-0.144, -0.009] |
| train-similar | deeppocket | +0.009 [-0.050, +0.073] | -0.028 [-0.087, +0.032] | +0.009 [-0.039, +0.062] |
| train-similar | grasp | +0.037 [-0.050, +0.121] | -0.009 [-0.090, +0.068] | -0.074 [-0.151, -0.009] |
| train-similar | deepsurf | +0.056 [-0.009, +0.119] | +0.019 [-0.043, +0.076] | -0.028 [-0.071, +0.010] |

Top-N by the structure's own number of sites, against **p2rank**. At N = 1 top-N is top-1 and no merging of predictions can change it, so a deficit in that row is ranking and a deficit confined to N >= 2 is prediction fragmentation:

| subset | method | N | n | top-N | p2rank top-N | difference | ceiling | p2rank ceiling |
|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 1 | 142 | 0.810 | 0.859 | -0.049 | 1.000 | 0.965 |
| all | ours (ranker) | 2 | 86 | 0.802 | 0.849 | -0.047 | 0.988 | 0.965 |
| all | ours (ranker) | >=3 | 41 | 0.951 | 0.878 | +0.073 | 0.976 | 0.902 |
| all | ours (native order) | 1 | 142 | 0.718 | 0.859 | -0.141 | 1.000 | 0.965 |
| all | ours (native order) | 2 | 86 | 0.756 | 0.849 | -0.093 | 0.988 | 0.965 |
| all | ours (native order) | >=3 | 41 | 0.780 | 0.878 | -0.098 | 0.976 | 0.902 |
| all | deeppocket | 1 | 142 | 0.782 | 0.859 | -0.077 | 0.979 | 0.965 |
| all | deeppocket | 2 | 86 | 0.826 | 0.849 | -0.023 | 0.965 | 0.965 |
| all | deeppocket | >=3 | 41 | 0.927 | 0.878 | +0.049 | 1.000 | 0.902 |
| all | grasp | 1 | 142 | 0.817 | 0.859 | -0.042 | 0.866 | 0.965 |
| all | grasp | 2 | 86 | 0.814 | 0.849 | -0.035 | 0.814 | 0.965 |
| all | grasp | >=3 | 41 | 0.683 | 0.878 | -0.195 | 0.683 | 0.902 |
| all | deepsurf | 1 | 142 | 0.859 | 0.859 | +0.000 | 0.894 | 0.965 |
| all | deepsurf | 2 | 86 | 0.872 | 0.849 | +0.023 | 0.872 | 0.965 |
| all | deepsurf | >=3 | 41 | 0.878 | 0.878 | +0.000 | 0.902 | 0.902 |
| not train-similar | ours (ranker) | 1 | 77 | 0.740 | 0.844 | -0.104 | 1.000 | 0.948 |
| not train-similar | ours (ranker) | 2 | 54 | 0.833 | 0.852 | -0.019 | 1.000 | 0.963 |
| not train-similar | ours (ranker) | >=3 | 30 | 0.933 | 0.867 | +0.067 | 0.967 | 0.900 |
| not train-similar | ours (native order) | 1 | 77 | 0.688 | 0.844 | -0.156 | 1.000 | 0.948 |
| not train-similar | ours (native order) | 2 | 54 | 0.796 | 0.852 | -0.056 | 1.000 | 0.963 |
| not train-similar | ours (native order) | >=3 | 30 | 0.800 | 0.867 | -0.067 | 0.967 | 0.900 |
| not train-similar | deeppocket | 1 | 77 | 0.714 | 0.844 | -0.130 | 0.961 | 0.948 |
| not train-similar | deeppocket | 2 | 54 | 0.870 | 0.852 | +0.019 | 0.963 | 0.963 |
| not train-similar | deeppocket | >=3 | 30 | 0.900 | 0.867 | +0.033 | 1.000 | 0.900 |
| not train-similar | grasp | 1 | 77 | 0.753 | 0.844 | -0.091 | 0.818 | 0.948 |
| not train-similar | grasp | 2 | 54 | 0.778 | 0.852 | -0.074 | 0.778 | 0.963 |
| not train-similar | grasp | >=3 | 30 | 0.700 | 0.867 | -0.167 | 0.700 | 0.900 |
| not train-similar | deepsurf | 1 | 77 | 0.818 | 0.844 | -0.026 | 0.831 | 0.948 |
| not train-similar | deepsurf | 2 | 54 | 0.889 | 0.852 | +0.037 | 0.889 | 0.963 |
| not train-similar | deepsurf | >=3 | 30 | 0.867 | 0.867 | +0.000 | 0.900 | 0.900 |
| train-similar | ours (ranker) | 1 | 65 | 0.892 | 0.877 | +0.015 | 1.000 | 0.985 |
| train-similar | ours (ranker) | 2 | 32 | 0.750 | 0.844 | -0.094 | 0.969 | 0.969 |
| train-similar | ours (ranker) | >=3 | 11 | 1.000 | 0.909 | +0.091 | 1.000 | 0.909 |
| train-similar | ours (native order) | 1 | 65 | 0.754 | 0.877 | -0.123 | 1.000 | 0.985 |
| train-similar | ours (native order) | 2 | 32 | 0.688 | 0.844 | -0.156 | 0.969 | 0.969 |
| train-similar | ours (native order) | >=3 | 11 | 0.727 | 0.909 | -0.182 | 1.000 | 0.909 |
| train-similar | deeppocket | 1 | 65 | 0.862 | 0.877 | -0.015 | 1.000 | 0.985 |
| train-similar | deeppocket | 2 | 32 | 0.750 | 0.844 | -0.094 | 0.969 | 0.969 |
| train-similar | deeppocket | >=3 | 11 | 1.000 | 0.909 | +0.091 | 1.000 | 0.909 |
| train-similar | grasp | 1 | 65 | 0.892 | 0.877 | +0.015 | 0.923 | 0.985 |
| train-similar | grasp | 2 | 32 | 0.875 | 0.844 | +0.031 | 0.875 | 0.969 |
| train-similar | grasp | >=3 | 11 | 0.636 | 0.909 | -0.273 | 0.636 | 0.909 |
| train-similar | deepsurf | 1 | 65 | 0.908 | 0.877 | +0.031 | 0.969 | 0.985 |
| train-similar | deepsurf | 2 | 32 | 0.844 | 0.844 | +0.000 | 0.844 | 0.969 |
| train-similar | deepsurf | >=3 | 11 | 0.909 | 0.909 | +0.000 | 0.909 | 0.909 |
