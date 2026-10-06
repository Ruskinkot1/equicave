# coach420: our predictions against external tools on the same structures

283 structures predicted by every method listed. Success is DCA <= 4 A to a ligand of the set's own relevant-ligand list; N is the structure's own number of ligand sites. 95 % CI by 30 %-identity cluster bootstrap. `train-similar` means the structure shares a 30 %-identity cluster with **our** training manifest; it is leakage for us and not for the external tools, whose own training sets overlap this benchmark in ways this table does not measure. The comparable row for us is **not train-similar**.

| subset | method | predictions | ceiling | top-1 | top-3 | top-N | top-(N+2) | MRR | n |
|---|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 30.0 | 0.989 | 0.739 [0.681, 0.792] | 0.876 [0.837, 0.914] | 0.799 [0.746, 0.848] | 0.890 [0.853, 0.926] | 0.820 [0.779, 0.860] | 283 |
| all | ours (native order) | 30.0 | 0.989 | 0.611 [0.555, 0.668] | 0.802 [0.754, 0.847] | 0.703 [0.644, 0.757] | 0.827 [0.782, 0.869] | 0.725 [0.681, 0.768] | 283 |
| all | grasp | 2.0 | 0.777 | 0.721 [0.667, 0.774] | 0.774 [0.722, 0.824] | 0.753 [0.703, 0.803] | 0.777 [0.726, 0.827] | 0.748 [0.696, 0.797] | 283 |
| all | p2rank | 8.4 | 0.940 | 0.770 [0.719, 0.819] | 0.890 [0.850, 0.927] | 0.841 [0.795, 0.885] | 0.905 [0.866, 0.938] | 0.834 [0.795, 0.871] | 283 |
| all | deeppocket | 32.4 | 0.958 | 0.721 [0.664, 0.774] | 0.866 [0.824, 0.908] | 0.770 [0.720, 0.820] | 0.894 [0.855, 0.931] | 0.802 [0.763, 0.841] | 283 |
| not train-similar | ours (ranker) | 30.0 | 0.989 | 0.701 [0.630, 0.770] | 0.868 [0.815, 0.918] | 0.776 [0.712, 0.839] | 0.874 [0.821, 0.924] | 0.798 [0.748, 0.849] | 174 |
| not train-similar | ours (native order) | 30.0 | 0.989 | 0.598 [0.526, 0.669] | 0.793 [0.730, 0.850] | 0.713 [0.641, 0.778] | 0.822 [0.762, 0.877] | 0.713 [0.658, 0.766] | 174 |
| not train-similar | grasp | 2.0 | 0.724 | 0.661 [0.589, 0.734] | 0.718 [0.648, 0.784] | 0.701 [0.630, 0.770] | 0.724 [0.655, 0.790] | 0.690 [0.618, 0.756] | 174 |
| not train-similar | p2rank | 9.6 | 0.931 | 0.753 [0.688, 0.814] | 0.874 [0.822, 0.921] | 0.828 [0.771, 0.881] | 0.879 [0.830, 0.926] | 0.817 [0.766, 0.864] | 174 |
| not train-similar | deeppocket | 36.9 | 0.960 | 0.695 [0.624, 0.757] | 0.856 [0.799, 0.907] | 0.770 [0.705, 0.831] | 0.879 [0.827, 0.925] | 0.786 [0.732, 0.834] | 174 |
| train-similar | ours (ranker) | 30.0 | 0.991 | 0.798 [0.713, 0.875] | 0.890 [0.826, 0.944] | 0.835 [0.754, 0.905] | 0.917 [0.860, 0.964] | 0.857 [0.793, 0.912] | 109 |
| train-similar | ours (native order) | 30.0 | 0.991 | 0.633 [0.528, 0.726] | 0.817 [0.735, 0.887] | 0.688 [0.585, 0.776] | 0.835 [0.755, 0.901] | 0.744 [0.665, 0.812] | 109 |
| train-similar | grasp | 1.9 | 0.862 | 0.817 [0.735, 0.886] | 0.862 [0.788, 0.923] | 0.835 [0.757, 0.903] | 0.862 [0.788, 0.923] | 0.839 [0.766, 0.904] | 109 |
| train-similar | p2rank | 6.5 | 0.954 | 0.798 [0.712, 0.876] | 0.917 [0.856, 0.965] | 0.862 [0.780, 0.925] | 0.945 [0.894, 0.983] | 0.863 [0.799, 0.915] | 109 |
| train-similar | deeppocket | 25.1 | 0.954 | 0.761 [0.674, 0.845] | 0.881 [0.813, 0.942] | 0.771 [0.685, 0.853] | 0.917 [0.860, 0.966] | 0.828 [0.766, 0.888] | 109 |

Paired differences against **grasp** on the same structures (positive = we are better):

| subset | method | top-1 | top-N | top-(N+2) |
|---|---|---|---|---|
| all | ours (ranker) | +0.018 [-0.043, +0.076] | +0.046 [-0.011, +0.103] | +0.113 [+0.067, +0.163] |
| all | ours (native order) | -0.110 [-0.175, -0.045] | -0.049 [-0.114, +0.011] | +0.049 [-0.004, +0.106] |
| all | p2rank | +0.049 [-0.007, +0.111] | +0.088 [+0.035, +0.144] | +0.127 [+0.074, +0.181] |
| all | deeppocket | +0.000 [-0.059, +0.059] | +0.018 [-0.042, +0.076] | +0.117 [+0.064, +0.169] |
| not train-similar | ours (ranker) | +0.040 [-0.040, +0.118] | +0.075 [-0.006, +0.153] | +0.149 [+0.085, +0.215] |
| not train-similar | ours (native order) | -0.063 [-0.148, +0.018] | +0.011 [-0.070, +0.091] | +0.098 [+0.024, +0.174] |
| not train-similar | p2rank | +0.092 [+0.017, +0.169] | +0.126 [+0.052, +0.201] | +0.155 [+0.080, +0.229] |
| not train-similar | deeppocket | +0.034 [-0.049, +0.114] | +0.069 [-0.017, +0.147] | +0.155 [+0.077, +0.229] |
| train-similar | ours (ranker) | -0.018 [-0.112, +0.075] | +0.000 [-0.087, +0.084] | +0.055 [-0.017, +0.129] |
| train-similar | ours (native order) | -0.183 [-0.278, -0.084] | -0.147 [-0.245, -0.043] | -0.028 [-0.112, +0.054] |
| train-similar | p2rank | -0.018 [-0.106, +0.076] | +0.028 [-0.049, +0.110] | +0.083 [+0.017, +0.157] |
| train-similar | deeppocket | -0.055 [-0.146, +0.040] | -0.064 [-0.151, +0.029] | +0.055 [-0.027, +0.144] |

Top-N by the structure's own number of sites, against **grasp**. At N = 1 top-N is top-1 and no merging of predictions can change it, so a deficit in that row is ranking and a deficit confined to N >= 2 is prediction fragmentation:

| subset | method | N | n | top-N | grasp top-N | difference | ceiling | grasp ceiling |
|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 1 | 142 | 0.775 | 0.782 | -0.007 | 1.000 | 0.831 |
| all | ours (ranker) | 2 | 89 | 0.787 | 0.798 | -0.011 | 0.989 | 0.798 |
| all | ours (ranker) | >=3 | 52 | 0.885 | 0.596 | +0.288 | 0.962 | 0.596 |
| all | ours (native order) | 1 | 142 | 0.676 | 0.782 | -0.106 | 1.000 | 0.831 |
| all | ours (native order) | 2 | 89 | 0.742 | 0.798 | -0.056 | 0.989 | 0.798 |
| all | ours (native order) | >=3 | 52 | 0.712 | 0.596 | +0.115 | 0.962 | 0.596 |
| all | p2rank | 1 | 142 | 0.824 | 0.782 | +0.042 | 0.930 | 0.831 |
| all | p2rank | 2 | 89 | 0.831 | 0.798 | +0.034 | 0.966 | 0.798 |
| all | p2rank | >=3 | 52 | 0.904 | 0.596 | +0.308 | 0.923 | 0.596 |
| all | deeppocket | 1 | 142 | 0.732 | 0.782 | -0.049 | 0.937 | 0.831 |
| all | deeppocket | 2 | 89 | 0.775 | 0.798 | -0.022 | 0.966 | 0.798 |
| all | deeppocket | >=3 | 52 | 0.865 | 0.596 | +0.269 | 1.000 | 0.596 |
| not train-similar | ours (ranker) | 1 | 77 | 0.701 | 0.714 | -0.013 | 1.000 | 0.766 |
| not train-similar | ours (ranker) | 2 | 57 | 0.789 | 0.772 | +0.018 | 1.000 | 0.772 |
| not train-similar | ours (ranker) | >=3 | 40 | 0.900 | 0.575 | +0.325 | 0.950 | 0.575 |
| not train-similar | ours (native order) | 1 | 77 | 0.649 | 0.714 | -0.065 | 1.000 | 0.766 |
| not train-similar | ours (native order) | 2 | 57 | 0.772 | 0.772 | +0.000 | 1.000 | 0.772 |
| not train-similar | ours (native order) | >=3 | 40 | 0.750 | 0.575 | +0.175 | 0.950 | 0.575 |
| not train-similar | p2rank | 1 | 77 | 0.792 | 0.714 | +0.078 | 0.909 | 0.766 |
| not train-similar | p2rank | 2 | 57 | 0.825 | 0.772 | +0.053 | 0.965 | 0.772 |
| not train-similar | p2rank | >=3 | 40 | 0.900 | 0.575 | +0.325 | 0.925 | 0.575 |
| not train-similar | deeppocket | 1 | 77 | 0.701 | 0.714 | -0.013 | 0.935 | 0.766 |
| not train-similar | deeppocket | 2 | 57 | 0.789 | 0.772 | +0.018 | 0.965 | 0.772 |
| not train-similar | deeppocket | >=3 | 40 | 0.875 | 0.575 | +0.300 | 1.000 | 0.575 |
| train-similar | ours (ranker) | 1 | 65 | 0.862 | 0.862 | +0.000 | 1.000 | 0.908 |
| train-similar | ours (ranker) | 2 | 32 | 0.781 | 0.844 | -0.062 | 0.969 | 0.844 |
| train-similar | ours (ranker) | >=3 | 12 | 0.833 | 0.667 | +0.167 | 1.000 | 0.667 |
| train-similar | ours (native order) | 1 | 65 | 0.708 | 0.862 | -0.154 | 1.000 | 0.908 |
| train-similar | ours (native order) | 2 | 32 | 0.688 | 0.844 | -0.156 | 0.969 | 0.844 |
| train-similar | ours (native order) | >=3 | 12 | 0.583 | 0.667 | -0.083 | 1.000 | 0.667 |
| train-similar | p2rank | 1 | 65 | 0.862 | 0.862 | +0.000 | 0.954 | 0.908 |
| train-similar | p2rank | 2 | 32 | 0.844 | 0.844 | +0.000 | 0.969 | 0.844 |
| train-similar | p2rank | >=3 | 12 | 0.917 | 0.667 | +0.250 | 0.917 | 0.667 |
| train-similar | deeppocket | 1 | 65 | 0.769 | 0.862 | -0.092 | 0.938 | 0.908 |
| train-similar | deeppocket | 2 | 32 | 0.750 | 0.844 | -0.094 | 0.969 | 0.844 |
| train-similar | deeppocket | >=3 | 12 | 0.833 | 0.667 | +0.167 | 1.000 | 0.667 |
