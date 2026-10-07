# coach420: our predictions against external tools on the same structures

269 structures predicted by every method listed. Success is DCA <= 4 A to a ligand of the set's own relevant-ligand list; N is the structure's own number of ligand sites. 95 % CI by 30 %-identity cluster bootstrap. `train-similar` means the structure shares a 30 %-identity cluster with **our** training manifest; it is leakage for us and not for the external tools, whose own training sets overlap this benchmark in ways this table does not measure. The comparable row for us is **not train-similar**.

| subset | method | predictions | ceiling | top-1 | top-3 | top-N | top-(N+2) | MRR | n |
|---|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 30.0 | 0.993 | 0.747 [0.690, 0.804] | 0.888 [0.847, 0.926] | 0.807 [0.756, 0.856] | 0.900 [0.862, 0.937] | 0.830 [0.790, 0.870] | 269 |
| all | ours (native order) | 30.0 | 0.993 | 0.628 [0.567, 0.686] | 0.818 [0.768, 0.863] | 0.710 [0.651, 0.764] | 0.836 [0.789, 0.881] | 0.741 [0.695, 0.784] | 269 |
| all | p2rank | 7.0 | 0.937 | 0.781 [0.723, 0.832] | 0.888 [0.848, 0.925] | 0.840 [0.794, 0.884] | 0.903 [0.865, 0.937] | 0.840 [0.797, 0.878] | 269 |
| all | deeppocket | 27.9 | 0.959 | 0.729 [0.673, 0.783] | 0.877 [0.835, 0.915] | 0.773 [0.721, 0.825] | 0.900 [0.858, 0.935] | 0.810 [0.769, 0.850] | 269 |
| all | grasp | 2.0 | 0.803 | 0.755 [0.700, 0.807] | 0.803 [0.751, 0.851] | 0.777 [0.726, 0.827] | 0.803 [0.751, 0.851] | 0.778 [0.728, 0.827] | 269 |
| all | deepsurf | 1.8 | 0.862 | 0.818 [0.767, 0.865] | 0.862 [0.819, 0.903] | 0.836 [0.789, 0.882] | 0.862 [0.819, 0.903] | 0.840 [0.795, 0.880] | 269 |
| not train-similar | ours (ranker) | 30.0 | 0.994 | 0.708 [0.632, 0.776] | 0.882 [0.829, 0.926] | 0.783 [0.712, 0.846] | 0.888 [0.837, 0.932] | 0.808 [0.758, 0.855] | 161 |
| not train-similar | ours (native order) | 30.0 | 0.994 | 0.621 [0.548, 0.694] | 0.814 [0.752, 0.872] | 0.727 [0.656, 0.794] | 0.839 [0.781, 0.893] | 0.736 [0.682, 0.788] | 161 |
| not train-similar | p2rank | 7.6 | 0.925 | 0.764 [0.696, 0.827] | 0.870 [0.813, 0.921] | 0.826 [0.762, 0.881] | 0.876 [0.824, 0.925] | 0.822 [0.768, 0.872] | 161 |
| not train-similar | deeppocket | 30.4 | 0.963 | 0.708 [0.633, 0.775] | 0.876 [0.820, 0.923] | 0.776 [0.710, 0.836] | 0.888 [0.833, 0.932] | 0.799 [0.743, 0.845] | 161 |
| not train-similar | grasp | 2.0 | 0.764 | 0.708 [0.634, 0.778] | 0.764 [0.692, 0.827] | 0.739 [0.667, 0.805] | 0.764 [0.692, 0.827] | 0.735 [0.666, 0.800] | 161 |
| not train-similar | deepsurf | 2.0 | 0.826 | 0.801 [0.736, 0.860] | 0.826 [0.765, 0.884] | 0.814 [0.750, 0.874] | 0.826 [0.765, 0.884] | 0.814 [0.752, 0.871] | 161 |
| train-similar | ours (ranker) | 30.0 | 0.991 | 0.806 [0.715, 0.885] | 0.898 [0.836, 0.952] | 0.843 [0.765, 0.914] | 0.917 [0.858, 0.965] | 0.863 [0.799, 0.920] | 108 |
| train-similar | ours (native order) | 30.0 | 0.991 | 0.639 [0.532, 0.738] | 0.824 [0.745, 0.893] | 0.685 [0.583, 0.778] | 0.833 [0.755, 0.901] | 0.749 [0.669, 0.820] | 108 |
| train-similar | p2rank | 6.2 | 0.954 | 0.806 [0.720, 0.882] | 0.917 [0.857, 0.964] | 0.861 [0.784, 0.925] | 0.944 [0.896, 0.982] | 0.867 [0.808, 0.920] | 108 |
| train-similar | deeppocket | 24.1 | 0.954 | 0.759 [0.677, 0.840] | 0.880 [0.811, 0.938] | 0.769 [0.686, 0.848] | 0.917 [0.859, 0.964] | 0.827 [0.765, 0.886] | 108 |
| train-similar | grasp | 1.9 | 0.861 | 0.824 [0.745, 0.892] | 0.861 [0.784, 0.923] | 0.833 [0.757, 0.899] | 0.861 [0.784, 0.923] | 0.843 [0.766, 0.907] | 108 |
| train-similar | deepsurf | 1.6 | 0.917 | 0.843 [0.760, 0.912] | 0.917 [0.856, 0.963] | 0.870 [0.792, 0.935] | 0.917 [0.856, 0.963] | 0.878 [0.811, 0.932] | 108 |

Paired differences against **p2rank** on the same structures (positive = we are better):

| subset | method | top-1 | top-N | top-(N+2) |
|---|---|---|---|---|
| all | ours (ranker) | -0.033 [-0.083, +0.018] | -0.033 [-0.084, +0.019] | -0.004 [-0.042, +0.035] |
| all | ours (native order) | -0.152 [-0.206, -0.100] | -0.130 [-0.177, -0.083] | -0.067 [-0.108, -0.029] |
| all | deeppocket | -0.052 [-0.107, +0.004] | -0.067 [-0.121, -0.015] | -0.004 [-0.048, +0.038] |
| all | grasp | -0.026 [-0.083, +0.032] | -0.063 [-0.115, -0.011] | -0.100 [-0.151, -0.050] |
| all | deepsurf | +0.037 [-0.008, +0.085] | -0.004 [-0.046, +0.038] | -0.041 [-0.082, +0.000] |
| not train-similar | ours (ranker) | -0.056 [-0.131, +0.013] | -0.043 [-0.117, +0.030] | +0.012 [-0.042, +0.067] |
| not train-similar | ours (native order) | -0.143 [-0.212, -0.069] | -0.099 [-0.161, -0.042] | -0.037 [-0.091, +0.013] |
| not train-similar | deeppocket | -0.056 [-0.127, +0.019] | -0.050 [-0.118, +0.019] | +0.012 [-0.043, +0.068] |
| not train-similar | grasp | -0.056 [-0.137, +0.024] | -0.087 [-0.165, -0.012] | -0.112 [-0.187, -0.037] |
| not train-similar | deepsurf | +0.037 [-0.025, +0.101] | -0.012 [-0.074, +0.044] | -0.050 [-0.111, +0.006] |
| train-similar | ours (ranker) | +0.000 [-0.072, +0.070] | -0.019 [-0.094, +0.054] | -0.028 [-0.075, +0.019] |
| train-similar | ours (native order) | -0.167 [-0.250, -0.087] | -0.176 [-0.255, -0.104] | -0.111 [-0.180, -0.056] |
| train-similar | deeppocket | -0.046 [-0.124, +0.030] | -0.093 [-0.170, -0.018] | -0.028 [-0.085, +0.028] |
| train-similar | grasp | +0.019 [-0.070, +0.109] | -0.028 [-0.110, +0.051] | -0.083 [-0.157, -0.019] |
| train-similar | deepsurf | +0.037 [-0.036, +0.108] | +0.009 [-0.053, +0.067] | -0.028 [-0.080, +0.018] |

Top-N by the structure's own number of sites, against **p2rank**. At N = 1 top-N is top-1 and no merging of predictions can change it, so a deficit in that row is ranking and a deficit confined to N >= 2 is prediction fragmentation:

| subset | method | N | n | top-N | p2rank top-N | difference | ceiling | p2rank ceiling |
|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 1 | 142 | 0.775 | 0.824 | -0.049 | 1.000 | 0.930 |
| all | ours (ranker) | 2 | 86 | 0.802 | 0.849 | -0.047 | 0.988 | 0.965 |
| all | ours (ranker) | >=3 | 41 | 0.927 | 0.878 | +0.049 | 0.976 | 0.902 |
| all | ours (native order) | 1 | 142 | 0.676 | 0.824 | -0.148 | 1.000 | 0.930 |
| all | ours (native order) | 2 | 86 | 0.756 | 0.849 | -0.093 | 0.988 | 0.965 |
| all | ours (native order) | >=3 | 41 | 0.732 | 0.878 | -0.146 | 0.976 | 0.902 |
| all | deeppocket | 1 | 142 | 0.732 | 0.824 | -0.092 | 0.937 | 0.930 |
| all | deeppocket | 2 | 86 | 0.791 | 0.849 | -0.058 | 0.977 | 0.965 |
| all | deeppocket | >=3 | 41 | 0.878 | 0.878 | +0.000 | 1.000 | 0.902 |
| all | grasp | 1 | 142 | 0.782 | 0.824 | -0.042 | 0.831 | 0.930 |
| all | grasp | 2 | 86 | 0.826 | 0.849 | -0.023 | 0.826 | 0.965 |
| all | grasp | >=3 | 41 | 0.659 | 0.878 | -0.220 | 0.659 | 0.902 |
| all | deepsurf | 1 | 142 | 0.838 | 0.824 | +0.014 | 0.880 | 0.930 |
| all | deepsurf | 2 | 86 | 0.837 | 0.849 | -0.012 | 0.849 | 0.965 |
| all | deepsurf | >=3 | 41 | 0.829 | 0.878 | -0.049 | 0.829 | 0.902 |
| not train-similar | ours (ranker) | 1 | 77 | 0.701 | 0.792 | -0.091 | 1.000 | 0.909 |
| not train-similar | ours (ranker) | 2 | 54 | 0.815 | 0.852 | -0.037 | 1.000 | 0.963 |
| not train-similar | ours (ranker) | >=3 | 30 | 0.933 | 0.867 | +0.067 | 0.967 | 0.900 |
| not train-similar | ours (native order) | 1 | 77 | 0.649 | 0.792 | -0.143 | 1.000 | 0.909 |
| not train-similar | ours (native order) | 2 | 54 | 0.796 | 0.852 | -0.056 | 1.000 | 0.963 |
| not train-similar | ours (native order) | >=3 | 30 | 0.800 | 0.867 | -0.067 | 0.967 | 0.900 |
| not train-similar | deeppocket | 1 | 77 | 0.701 | 0.792 | -0.091 | 0.935 | 0.909 |
| not train-similar | deeppocket | 2 | 54 | 0.815 | 0.852 | -0.037 | 0.981 | 0.963 |
| not train-similar | deeppocket | >=3 | 30 | 0.900 | 0.867 | +0.033 | 1.000 | 0.900 |
| not train-similar | grasp | 1 | 77 | 0.714 | 0.792 | -0.078 | 0.766 | 0.909 |
| not train-similar | grasp | 2 | 54 | 0.815 | 0.852 | -0.037 | 0.815 | 0.963 |
| not train-similar | grasp | >=3 | 30 | 0.667 | 0.867 | -0.200 | 0.667 | 0.900 |
| not train-similar | deepsurf | 1 | 77 | 0.792 | 0.792 | +0.000 | 0.818 | 0.909 |
| not train-similar | deepsurf | 2 | 54 | 0.852 | 0.852 | +0.000 | 0.852 | 0.963 |
| not train-similar | deepsurf | >=3 | 30 | 0.800 | 0.867 | -0.067 | 0.800 | 0.900 |
| train-similar | ours (ranker) | 1 | 65 | 0.862 | 0.862 | +0.000 | 1.000 | 0.954 |
| train-similar | ours (ranker) | 2 | 32 | 0.781 | 0.844 | -0.062 | 0.969 | 0.969 |
| train-similar | ours (ranker) | >=3 | 11 | 0.909 | 0.909 | +0.000 | 1.000 | 0.909 |
| train-similar | ours (native order) | 1 | 65 | 0.708 | 0.862 | -0.154 | 1.000 | 0.954 |
| train-similar | ours (native order) | 2 | 32 | 0.688 | 0.844 | -0.156 | 0.969 | 0.969 |
| train-similar | ours (native order) | >=3 | 11 | 0.545 | 0.909 | -0.364 | 1.000 | 0.909 |
| train-similar | deeppocket | 1 | 65 | 0.769 | 0.862 | -0.092 | 0.938 | 0.954 |
| train-similar | deeppocket | 2 | 32 | 0.750 | 0.844 | -0.094 | 0.969 | 0.969 |
| train-similar | deeppocket | >=3 | 11 | 0.818 | 0.909 | -0.091 | 1.000 | 0.909 |
| train-similar | grasp | 1 | 65 | 0.862 | 0.862 | +0.000 | 0.908 | 0.954 |
| train-similar | grasp | 2 | 32 | 0.844 | 0.844 | +0.000 | 0.844 | 0.969 |
| train-similar | grasp | >=3 | 11 | 0.636 | 0.909 | -0.273 | 0.636 | 0.909 |
| train-similar | deepsurf | 1 | 65 | 0.892 | 0.862 | +0.031 | 0.954 | 0.954 |
| train-similar | deepsurf | 2 | 32 | 0.812 | 0.844 | -0.031 | 0.844 | 0.969 |
| train-similar | deepsurf | >=3 | 11 | 0.909 | 0.909 | +0.000 | 0.909 | 0.909 |
