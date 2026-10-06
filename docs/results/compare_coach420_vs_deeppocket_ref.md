# coach420: our predictions against external tools on the same structures

283 structures predicted by every method listed. Success is DCA <= 4 A to a ligand of the set's own relevant-ligand list; N is the structure's own number of ligand sites. 95 % CI by 30 %-identity cluster bootstrap. `train-similar` means the structure shares a 30 %-identity cluster with **our** training manifest; it is leakage for us and not for the external tools, whose own training sets overlap this benchmark in ways this table does not measure. The comparable row for us is **not train-similar**.

| subset | method | predictions | ceiling | top-1 | top-3 | top-N | top-(N+2) | MRR | n |
|---|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 30.0 | 0.989 | 0.739 [0.681, 0.792] | 0.876 [0.837, 0.914] | 0.799 [0.746, 0.848] | 0.890 [0.853, 0.926] | 0.820 [0.779, 0.860] | 283 |
| all | ours (native order) | 30.0 | 0.989 | 0.611 [0.555, 0.668] | 0.802 [0.754, 0.847] | 0.703 [0.644, 0.757] | 0.827 [0.782, 0.869] | 0.725 [0.681, 0.768] | 283 |
| all | deeppocket | 32.4 | 0.958 | 0.721 [0.664, 0.774] | 0.866 [0.824, 0.908] | 0.770 [0.720, 0.820] | 0.894 [0.855, 0.931] | 0.802 [0.763, 0.841] | 283 |
| all | p2rank | 8.4 | 0.940 | 0.770 [0.719, 0.819] | 0.890 [0.850, 0.927] | 0.841 [0.795, 0.885] | 0.905 [0.866, 0.938] | 0.834 [0.795, 0.871] | 283 |
| not train-similar | ours (ranker) | 30.0 | 0.989 | 0.701 [0.630, 0.770] | 0.868 [0.815, 0.918] | 0.776 [0.712, 0.839] | 0.874 [0.821, 0.924] | 0.798 [0.748, 0.849] | 174 |
| not train-similar | ours (native order) | 30.0 | 0.989 | 0.598 [0.526, 0.669] | 0.793 [0.730, 0.850] | 0.713 [0.641, 0.778] | 0.822 [0.762, 0.877] | 0.713 [0.658, 0.766] | 174 |
| not train-similar | deeppocket | 36.9 | 0.960 | 0.695 [0.624, 0.757] | 0.856 [0.799, 0.907] | 0.770 [0.705, 0.831] | 0.879 [0.827, 0.925] | 0.786 [0.732, 0.834] | 174 |
| not train-similar | p2rank | 9.6 | 0.931 | 0.753 [0.688, 0.814] | 0.874 [0.822, 0.921] | 0.828 [0.771, 0.881] | 0.879 [0.830, 0.926] | 0.817 [0.766, 0.864] | 174 |
| train-similar | ours (ranker) | 30.0 | 0.991 | 0.798 [0.713, 0.875] | 0.890 [0.826, 0.944] | 0.835 [0.754, 0.905] | 0.917 [0.860, 0.964] | 0.857 [0.793, 0.912] | 109 |
| train-similar | ours (native order) | 30.0 | 0.991 | 0.633 [0.528, 0.726] | 0.817 [0.735, 0.887] | 0.688 [0.585, 0.776] | 0.835 [0.755, 0.901] | 0.744 [0.665, 0.812] | 109 |
| train-similar | deeppocket | 25.1 | 0.954 | 0.761 [0.674, 0.845] | 0.881 [0.813, 0.942] | 0.771 [0.685, 0.853] | 0.917 [0.860, 0.966] | 0.828 [0.766, 0.888] | 109 |
| train-similar | p2rank | 6.5 | 0.954 | 0.798 [0.712, 0.876] | 0.917 [0.856, 0.965] | 0.862 [0.780, 0.925] | 0.945 [0.894, 0.983] | 0.863 [0.799, 0.915] | 109 |

Paired differences against **deeppocket** on the same structures (positive = we are better):

| subset | method | top-1 | top-N | top-(N+2) |
|---|---|---|---|---|
| all | ours (ranker) | +0.018 [-0.036, +0.074] | +0.028 [-0.018, +0.079] | -0.004 [-0.037, +0.032] |
| all | ours (native order) | -0.110 [-0.166, -0.049] | -0.067 [-0.123, -0.017] | -0.067 [-0.116, -0.020] |
| all | p2rank | +0.049 [-0.003, +0.101] | +0.071 [+0.017, +0.119] | +0.011 [-0.029, +0.051] |
| not train-similar | ours (ranker) | +0.006 [-0.069, +0.081] | +0.006 [-0.054, +0.068] | -0.006 [-0.042, +0.030] |
| not train-similar | ours (native order) | -0.098 [-0.168, -0.029] | -0.057 [-0.124, +0.011] | -0.057 [-0.120, +0.000] |
| not train-similar | p2rank | +0.057 [-0.011, +0.129] | +0.057 [-0.011, +0.127] | +0.000 [-0.051, +0.053] |
| train-similar | ours (ranker) | +0.037 [-0.052, +0.119] | +0.064 [-0.010, +0.142] | +0.000 [-0.076, +0.071] |
| train-similar | ours (native order) | -0.128 [-0.229, -0.036] | -0.083 [-0.173, +0.000] | -0.083 [-0.163, -0.009] |
| train-similar | p2rank | +0.037 [-0.046, +0.113] | +0.092 [+0.016, +0.165] | +0.028 [-0.027, +0.081] |

Top-N by the structure's own number of sites, against **deeppocket**. At N = 1 top-N is top-1 and no merging of predictions can change it, so a deficit in that row is ranking and a deficit confined to N >= 2 is prediction fragmentation:

| subset | method | N | n | top-N | deeppocket top-N | difference | ceiling | deeppocket ceiling |
|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 1 | 142 | 0.775 | 0.732 | +0.042 | 1.000 | 0.937 |
| all | ours (ranker) | 2 | 89 | 0.787 | 0.775 | +0.011 | 0.989 | 0.966 |
| all | ours (ranker) | >=3 | 52 | 0.885 | 0.865 | +0.019 | 0.962 | 1.000 |
| all | ours (native order) | 1 | 142 | 0.676 | 0.732 | -0.056 | 1.000 | 0.937 |
| all | ours (native order) | 2 | 89 | 0.742 | 0.775 | -0.034 | 0.989 | 0.966 |
| all | ours (native order) | >=3 | 52 | 0.712 | 0.865 | -0.154 | 0.962 | 1.000 |
| all | p2rank | 1 | 142 | 0.824 | 0.732 | +0.092 | 0.930 | 0.937 |
| all | p2rank | 2 | 89 | 0.831 | 0.775 | +0.056 | 0.966 | 0.966 |
| all | p2rank | >=3 | 52 | 0.904 | 0.865 | +0.038 | 0.923 | 1.000 |
| not train-similar | ours (ranker) | 1 | 77 | 0.701 | 0.701 | +0.000 | 1.000 | 0.935 |
| not train-similar | ours (ranker) | 2 | 57 | 0.789 | 0.789 | +0.000 | 1.000 | 0.965 |
| not train-similar | ours (ranker) | >=3 | 40 | 0.900 | 0.875 | +0.025 | 0.950 | 1.000 |
| not train-similar | ours (native order) | 1 | 77 | 0.649 | 0.701 | -0.052 | 1.000 | 0.935 |
| not train-similar | ours (native order) | 2 | 57 | 0.772 | 0.789 | -0.018 | 1.000 | 0.965 |
| not train-similar | ours (native order) | >=3 | 40 | 0.750 | 0.875 | -0.125 | 0.950 | 1.000 |
| not train-similar | p2rank | 1 | 77 | 0.792 | 0.701 | +0.091 | 0.909 | 0.935 |
| not train-similar | p2rank | 2 | 57 | 0.825 | 0.789 | +0.035 | 0.965 | 0.965 |
| not train-similar | p2rank | >=3 | 40 | 0.900 | 0.875 | +0.025 | 0.925 | 1.000 |
| train-similar | ours (ranker) | 1 | 65 | 0.862 | 0.769 | +0.092 | 1.000 | 0.938 |
| train-similar | ours (ranker) | 2 | 32 | 0.781 | 0.750 | +0.031 | 0.969 | 0.969 |
| train-similar | ours (ranker) | >=3 | 12 | 0.833 | 0.833 | +0.000 | 1.000 | 1.000 |
| train-similar | ours (native order) | 1 | 65 | 0.708 | 0.769 | -0.062 | 1.000 | 0.938 |
| train-similar | ours (native order) | 2 | 32 | 0.688 | 0.750 | -0.062 | 0.969 | 0.969 |
| train-similar | ours (native order) | >=3 | 12 | 0.583 | 0.833 | -0.250 | 1.000 | 1.000 |
| train-similar | p2rank | 1 | 65 | 0.862 | 0.769 | +0.092 | 0.954 | 0.938 |
| train-similar | p2rank | 2 | 32 | 0.844 | 0.750 | +0.094 | 0.969 | 0.969 |
| train-similar | p2rank | >=3 | 12 | 0.917 | 0.833 | +0.083 | 0.917 | 1.000 |
