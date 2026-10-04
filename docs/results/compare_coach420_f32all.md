# coach420: our predictions against external tools on the same structures

283 structures predicted by every method listed. Success is DCA <= 4 A to a ligand of the set's own relevant-ligand list; N is the structure's own number of ligand sites. 95 % CI by 30 %-identity cluster bootstrap. `train-similar` means the structure shares a 30 %-identity cluster with **our** training manifest; it is leakage for us and not for the external tools, whose own training sets overlap this benchmark in ways this table does not measure. The comparable row for us is **not train-similar**.

| subset | method | predictions | ceiling | top-1 | top-3 | top-N | top-(N+2) | MRR | n |
|---|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 30.0 | 0.989 | 0.739 [0.681, 0.792] | 0.876 [0.837, 0.914] | 0.799 [0.746, 0.848] | 0.890 [0.853, 0.926] | 0.820 [0.779, 0.860] | 283 |
| all | ours (native order) | 30.0 | 0.989 | 0.611 [0.555, 0.668] | 0.802 [0.754, 0.847] | 0.703 [0.644, 0.757] | 0.827 [0.782, 0.869] | 0.725 [0.681, 0.768] | 283 |
| all | p2rank | 8.4 | 0.940 | 0.770 [0.719, 0.819] | 0.890 [0.850, 0.927] | 0.841 [0.795, 0.885] | 0.905 [0.866, 0.938] | 0.834 [0.795, 0.871] | 283 |
| not train-similar | ours (ranker) | 30.0 | 0.989 | 0.701 [0.630, 0.770] | 0.868 [0.815, 0.918] | 0.776 [0.712, 0.839] | 0.874 [0.821, 0.924] | 0.798 [0.748, 0.849] | 174 |
| not train-similar | ours (native order) | 30.0 | 0.989 | 0.598 [0.526, 0.669] | 0.793 [0.730, 0.850] | 0.713 [0.641, 0.778] | 0.822 [0.762, 0.877] | 0.713 [0.658, 0.766] | 174 |
| not train-similar | p2rank | 9.6 | 0.931 | 0.753 [0.688, 0.814] | 0.874 [0.822, 0.921] | 0.828 [0.771, 0.881] | 0.879 [0.830, 0.926] | 0.817 [0.766, 0.864] | 174 |
| train-similar | ours (ranker) | 30.0 | 0.991 | 0.798 [0.713, 0.875] | 0.890 [0.826, 0.944] | 0.835 [0.754, 0.905] | 0.917 [0.860, 0.964] | 0.857 [0.793, 0.912] | 109 |
| train-similar | ours (native order) | 30.0 | 0.991 | 0.633 [0.528, 0.726] | 0.817 [0.735, 0.887] | 0.688 [0.585, 0.776] | 0.835 [0.755, 0.901] | 0.744 [0.665, 0.812] | 109 |
| train-similar | p2rank | 6.5 | 0.954 | 0.798 [0.712, 0.876] | 0.917 [0.856, 0.965] | 0.862 [0.780, 0.925] | 0.945 [0.894, 0.983] | 0.863 [0.799, 0.915] | 109 |

Paired differences against **p2rank** on the same structures (positive = we are better):

| subset | method | top-1 | top-N | top-(N+2) |
|---|---|---|---|---|
| all | ours (ranker) | -0.032 [-0.080, +0.017] | -0.042 [-0.093, +0.010] | -0.014 [-0.051, +0.024] |
| all | ours (native order) | -0.159 [-0.213, -0.109] | -0.138 [-0.187, -0.092] | -0.078 [-0.118, -0.040] |
| not train-similar | ours (ranker) | -0.052 [-0.121, +0.011] | -0.052 [-0.121, +0.017] | -0.006 [-0.059, +0.046] |
| not train-similar | ours (native order) | -0.155 [-0.222, -0.089] | -0.115 [-0.175, -0.056] | -0.057 [-0.113, -0.006] |
| train-similar | ours (ranker) | +0.000 [-0.071, +0.073] | -0.028 [-0.105, +0.050] | -0.028 [-0.079, +0.019] |
| train-similar | ours (native order) | -0.165 [-0.252, -0.088] | -0.174 [-0.255, -0.102] | -0.110 [-0.178, -0.056] |

Top-N by the structure's own number of sites, against **p2rank**. At N = 1 top-N is top-1 and no merging of predictions can change it, so a deficit in that row is ranking and a deficit confined to N >= 2 is prediction fragmentation:

| subset | method | N | n | top-N | p2rank top-N | difference | ceiling | p2rank ceiling |
|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 1 | 142 | 0.775 | 0.824 | -0.049 | 1.000 | 0.930 |
| all | ours (ranker) | 2 | 89 | 0.787 | 0.831 | -0.045 | 0.989 | 0.966 |
| all | ours (ranker) | >=3 | 52 | 0.885 | 0.904 | -0.019 | 0.962 | 0.923 |
| all | ours (native order) | 1 | 142 | 0.676 | 0.824 | -0.148 | 1.000 | 0.930 |
| all | ours (native order) | 2 | 89 | 0.742 | 0.831 | -0.090 | 0.989 | 0.966 |
| all | ours (native order) | >=3 | 52 | 0.712 | 0.904 | -0.192 | 0.962 | 0.923 |
| not train-similar | ours (ranker) | 1 | 77 | 0.701 | 0.792 | -0.091 | 1.000 | 0.909 |
| not train-similar | ours (ranker) | 2 | 57 | 0.789 | 0.825 | -0.035 | 1.000 | 0.965 |
| not train-similar | ours (ranker) | >=3 | 40 | 0.900 | 0.900 | +0.000 | 0.950 | 0.925 |
| not train-similar | ours (native order) | 1 | 77 | 0.649 | 0.792 | -0.143 | 1.000 | 0.909 |
| not train-similar | ours (native order) | 2 | 57 | 0.772 | 0.825 | -0.053 | 1.000 | 0.965 |
| not train-similar | ours (native order) | >=3 | 40 | 0.750 | 0.900 | -0.150 | 0.950 | 0.925 |
| train-similar | ours (ranker) | 1 | 65 | 0.862 | 0.862 | +0.000 | 1.000 | 0.954 |
| train-similar | ours (ranker) | 2 | 32 | 0.781 | 0.844 | -0.062 | 0.969 | 0.969 |
| train-similar | ours (ranker) | >=3 | 12 | 0.833 | 0.917 | -0.083 | 1.000 | 0.917 |
| train-similar | ours (native order) | 1 | 65 | 0.708 | 0.862 | -0.154 | 1.000 | 0.954 |
| train-similar | ours (native order) | 2 | 32 | 0.688 | 0.844 | -0.156 | 0.969 | 0.969 |
| train-similar | ours (native order) | >=3 | 12 | 0.583 | 0.917 | -0.333 | 1.000 | 0.917 |
