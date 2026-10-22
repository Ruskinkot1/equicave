# coach420: our predictions against external tools on the same structures

120 structures predicted by every method listed. Success is DCA <= 4 A to a ligand of the set's own relevant-ligand list; N is the structure's own number of ligand sites. 95 % CI by 30 %-identity cluster bootstrap. `train-similar` means the structure shares a 30 %-identity cluster with **our** training manifest; it is leakage for us and not for the external tools, whose own training sets overlap this benchmark in ways this table does not measure. The comparable row for us is **not train-similar**.

| subset | method | predictions | ceiling | top-1 | top-3 | top-N | top-(N+2) | MRR | n |
|---|---|---|---|---|---|---|---|---|---|
| all | ours (network only) | 30.0 | 0.992 | 0.867 [0.803, 0.924] | 0.908 [0.853, 0.958] | 0.875 [0.815, 0.931] | 0.908 [0.853, 0.958] | 0.899 [0.851, 0.943] | 120 |
| all | p2rank | 7.1 | 0.900 | 0.758 [0.675, 0.833] | 0.833 [0.763, 0.894] | 0.800 [0.720, 0.868] | 0.858 [0.792, 0.916] | 0.806 [0.739, 0.867] | 120 |
| all | grasp | 1.9 | 0.742 | 0.675 [0.585, 0.760] | 0.742 [0.655, 0.818] | 0.708 [0.620, 0.788] | 0.742 [0.655, 0.818] | 0.708 [0.624, 0.787] | 120 |
| all | deeppocket | 27.6 | 0.967 | 0.717 [0.636, 0.795] | 0.858 [0.795, 0.918] | 0.742 [0.658, 0.820] | 0.875 [0.814, 0.932] | 0.795 [0.733, 0.854] | 120 |
| all | deepsurf | 1.8 | 0.833 | 0.792 [0.715, 0.866] | 0.833 [0.765, 0.894] | 0.808 [0.733, 0.877] | 0.833 [0.765, 0.894] | 0.812 [0.744, 0.877] | 120 |
| not train-similar | ours (network only) | 30.0 | 0.991 | 0.870 [0.805, 0.924] | 0.904 [0.850, 0.956] | 0.870 [0.805, 0.924] | 0.904 [0.850, 0.956] | 0.899 [0.848, 0.944] | 115 |
| not train-similar | p2rank | 7.1 | 0.904 | 0.757 [0.673, 0.833] | 0.835 [0.767, 0.898] | 0.800 [0.720, 0.872] | 0.861 [0.796, 0.921] | 0.806 [0.740, 0.869] | 115 |
| not train-similar | grasp | 1.9 | 0.730 | 0.661 [0.575, 0.746] | 0.730 [0.649, 0.809] | 0.696 [0.611, 0.778] | 0.730 [0.649, 0.809] | 0.696 [0.616, 0.774] | 115 |
| not train-similar | deeppocket | 27.8 | 0.965 | 0.713 [0.628, 0.795] | 0.861 [0.797, 0.920] | 0.739 [0.658, 0.817] | 0.878 [0.820, 0.932] | 0.793 [0.734, 0.852] | 115 |
| not train-similar | deepsurf | 1.8 | 0.835 | 0.791 [0.712, 0.868] | 0.835 [0.765, 0.902] | 0.809 [0.730, 0.879] | 0.835 [0.765, 0.902] | 0.813 [0.743, 0.883] | 115 |
| train-similar | ours (network only) | 30.0 | 1.000 | 0.800 [0.250, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.900 [0.625, 1.000] | 5 |
| train-similar | p2rank | 6.6 | 0.800 | 0.800 [0.250, 1.000] | 0.800 [0.250, 1.000] | 0.800 [0.250, 1.000] | 0.800 [0.250, 1.000] | 0.800 [0.250, 1.000] | 5 |
| train-similar | grasp | 1.6 | 1.000 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 5 |
| train-similar | deeppocket | 24.4 | 1.000 | 0.800 [0.250, 1.000] | 0.800 [0.250, 1.000] | 0.800 [0.250, 1.000] | 0.800 [0.250, 1.000] | 0.825 [0.344, 1.000] | 5 |
| train-similar | deepsurf | 1.0 | 0.800 | 0.800 [0.250, 1.000] | 0.800 [0.250, 1.000] | 0.800 [0.250, 1.000] | 0.800 [0.250, 1.000] | 0.800 [0.250, 1.000] | 5 |

Paired differences against **p2rank** on the same structures (positive = we are better):

| subset | method | top-1 | top-N | top-(N+2) |
|---|---|---|---|---|
| all | ours (network only) | +0.108 [+0.052, +0.171] | +0.075 [+0.025, +0.128] | +0.050 [-0.000, +0.106] |
| all | grasp | -0.083 [-0.175, +0.016] | -0.092 [-0.185, +0.000] | -0.117 [-0.200, -0.040] |
| all | deeppocket | -0.042 [-0.122, +0.041] | -0.058 [-0.138, +0.017] | +0.017 [-0.035, +0.074] |
| all | deepsurf | +0.033 [-0.041, +0.107] | +0.008 [-0.060, +0.074] | -0.025 [-0.090, +0.034] |
| not train-similar | ours (network only) | +0.113 [+0.053, +0.179] | +0.070 [+0.018, +0.123] | +0.043 [-0.009, +0.104] |
| not train-similar | grasp | -0.096 [-0.195, +0.000] | -0.104 [-0.204, -0.016] | -0.130 [-0.212, -0.050] |
| not train-similar | deeppocket | -0.043 [-0.125, +0.042] | -0.061 [-0.147, +0.018] | +0.017 [-0.043, +0.079] |
| not train-similar | deepsurf | +0.035 [-0.036, +0.112] | +0.009 [-0.061, +0.079] | -0.026 [-0.093, +0.043] |
| train-similar | ours (network only) | +0.000 [+0.000, +0.000] | +0.200 [+0.000, +0.750] | +0.200 [+0.000, +0.750] |
| train-similar | grasp | +0.200 [+0.000, +0.750] | +0.200 [+0.000, +0.750] | +0.200 [+0.000, +0.750] |
| train-similar | deeppocket | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| train-similar | deepsurf | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

**Composition of the hard-novelty subset.** Filtering by homology also filters by whatever correlates with it, so the aggregate difference below is not purely a leakage effect. These are the two populations the filter separated:

| population | structures | sites per structure | single-site | >= 3 sites |
|---|---|---|---|---|
| kept (novel) | 131 | 1.99 | 0.51 | 0.18 |
| removed (homologous) | 157 | 2.04 | 0.51 | 0.18 |

Top-N by the structure's own number of sites, against **p2rank**. At N = 1 top-N is top-1 and no merging of predictions can change it, so a deficit in that row is ranking and a deficit confined to N >= 2 is prediction fragmentation:

| subset | method | N | n | top-N | p2rank top-N | difference | ceiling | p2rank ceiling |
|---|---|---|---|---|---|---|---|---|
| all | ours (network only) | 1 | 63 | 0.841 | 0.794 | +0.048 | 1.000 | 0.889 |
| all | ours (network only) | 2 | 39 | 0.897 | 0.795 | +0.103 | 0.974 | 0.949 |
| all | ours (network only) | >=3 | 18 | 0.944 | 0.833 | +0.111 | 1.000 | 0.833 |
| all | grasp | 1 | 63 | 0.667 | 0.794 | -0.127 | 0.730 | 0.889 |
| all | grasp | 2 | 39 | 0.821 | 0.795 | +0.026 | 0.821 | 0.949 |
| all | grasp | >=3 | 18 | 0.611 | 0.833 | -0.222 | 0.611 | 0.833 |
| all | deeppocket | 1 | 63 | 0.698 | 0.794 | -0.095 | 0.937 | 0.889 |
| all | deeppocket | 2 | 39 | 0.769 | 0.795 | -0.026 | 1.000 | 0.949 |
| all | deeppocket | >=3 | 18 | 0.833 | 0.833 | +0.000 | 1.000 | 0.833 |
| all | deepsurf | 1 | 63 | 0.778 | 0.794 | -0.016 | 0.825 | 0.889 |
| all | deepsurf | 2 | 39 | 0.872 | 0.795 | +0.077 | 0.872 | 0.949 |
| all | deepsurf | >=3 | 18 | 0.778 | 0.833 | -0.056 | 0.778 | 0.833 |
| not train-similar | ours (network only) | 1 | 59 | 0.831 | 0.780 | +0.051 | 1.000 | 0.881 |
| not train-similar | ours (network only) | 2 | 39 | 0.897 | 0.795 | +0.103 | 0.974 | 0.949 |
| not train-similar | ours (network only) | >=3 | 17 | 0.941 | 0.882 | +0.059 | 1.000 | 0.882 |
| not train-similar | grasp | 1 | 59 | 0.644 | 0.780 | -0.136 | 0.712 | 0.881 |
| not train-similar | grasp | 2 | 39 | 0.821 | 0.795 | +0.026 | 0.821 | 0.949 |
| not train-similar | grasp | >=3 | 17 | 0.588 | 0.882 | -0.294 | 0.588 | 0.882 |
| not train-similar | deeppocket | 1 | 59 | 0.678 | 0.780 | -0.102 | 0.932 | 0.881 |
| not train-similar | deeppocket | 2 | 39 | 0.769 | 0.795 | -0.026 | 1.000 | 0.949 |
| not train-similar | deeppocket | >=3 | 17 | 0.882 | 0.882 | +0.000 | 1.000 | 0.882 |
| not train-similar | deepsurf | 1 | 59 | 0.763 | 0.780 | -0.017 | 0.814 | 0.881 |
| not train-similar | deepsurf | 2 | 39 | 0.872 | 0.795 | +0.077 | 0.872 | 0.949 |
| not train-similar | deepsurf | >=3 | 17 | 0.824 | 0.882 | -0.059 | 0.824 | 0.882 |
| train-similar | ours (network only) | 1 | 4 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 |
| train-similar | ours (network only) | >=3 | 1 | 1.000 | 0.000 | +1.000 | 1.000 | 0.000 |
| train-similar | grasp | 1 | 4 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 |
| train-similar | grasp | >=3 | 1 | 1.000 | 0.000 | +1.000 | 1.000 | 0.000 |
| train-similar | deeppocket | 1 | 4 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 |
| train-similar | deeppocket | >=3 | 1 | 0.000 | 0.000 | +0.000 | 1.000 | 0.000 |
| train-similar | deepsurf | 1 | 4 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 |
| train-similar | deepsurf | >=3 | 1 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 |
