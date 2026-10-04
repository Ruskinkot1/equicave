# coach420: our predictions against external tools on the same structures

283 structures predicted by every method listed. Success is DCA <= 4 A to a ligand of the set's own relevant-ligand list; N is the structure's own number of ligand sites. 95 % CI by 30 %-identity cluster bootstrap. `train-similar` means the structure shares a 30 %-identity cluster with **our** training manifest; it is leakage for us and not for the external tools, whose own training sets overlap this benchmark in ways this table does not measure. The comparable row for us is **not train-similar**.

| subset | method | predictions | ceiling | top-1 | top-3 | top-N | top-(N+2) | MRR | n |
|---|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 30.0 | 0.989 | 0.753 [0.701, 0.806] | 0.901 [0.864, 0.935] | 0.781 [0.732, 0.831] | 0.915 [0.880, 0.947] | 0.836 [0.799, 0.872] | 283 |
| all | ours (native order) | 30.0 | 0.989 | 0.671 [0.615, 0.724] | 0.869 [0.826, 0.907] | 0.703 [0.647, 0.757] | 0.876 [0.834, 0.914] | 0.778 [0.736, 0.815] | 283 |
| all | p2rank | 8.4 | 0.940 | 0.770 [0.719, 0.819] | 0.890 [0.850, 0.927] | 0.841 [0.795, 0.885] | 0.905 [0.866, 0.938] | 0.834 [0.795, 0.871] | 283 |
| not train-similar | ours (ranker) | 29.9 | 0.989 | 0.707 [0.635, 0.773] | 0.891 [0.843, 0.933] | 0.741 [0.673, 0.807] | 0.908 [0.862, 0.948] | 0.807 [0.759, 0.851] | 174 |
| not train-similar | ours (native order) | 29.9 | 0.989 | 0.644 [0.571, 0.714] | 0.868 [0.816, 0.916] | 0.690 [0.618, 0.761] | 0.874 [0.820, 0.923] | 0.759 [0.706, 0.808] | 174 |
| not train-similar | p2rank | 9.6 | 0.931 | 0.753 [0.688, 0.814] | 0.874 [0.822, 0.921] | 0.828 [0.771, 0.881] | 0.879 [0.830, 0.926] | 0.817 [0.766, 0.864] | 174 |
| train-similar | ours (ranker) | 30.0 | 0.991 | 0.826 [0.743, 0.894] | 0.917 [0.861, 0.964] | 0.844 [0.766, 0.907] | 0.927 [0.873, 0.971] | 0.883 [0.827, 0.928] | 109 |
| train-similar | ours (native order) | 30.0 | 0.991 | 0.716 [0.624, 0.794] | 0.872 [0.800, 0.931] | 0.725 [0.631, 0.804] | 0.881 [0.811, 0.937] | 0.808 [0.741, 0.863] | 109 |
| train-similar | p2rank | 6.5 | 0.954 | 0.798 [0.712, 0.876] | 0.917 [0.856, 0.965] | 0.862 [0.780, 0.925] | 0.945 [0.894, 0.983] | 0.863 [0.799, 0.915] | 109 |

Paired differences against **p2rank** on the same structures (positive = we are better):

| subset | method | top-1 | top-N | top-(N+2) |
|---|---|---|---|---|
| all | ours (ranker) | -0.018 [-0.083, +0.046] | -0.060 [-0.120, +0.003] | +0.011 [-0.028, +0.052] |
| all | ours (native order) | -0.099 [-0.158, -0.039] | -0.138 [-0.194, -0.081] | -0.028 [-0.073, +0.018] |
| not train-similar | ours (ranker) | -0.046 [-0.133, +0.035] | -0.086 [-0.169, -0.006] | +0.029 [-0.030, +0.083] |
| not train-similar | ours (native order) | -0.109 [-0.192, -0.028] | -0.138 [-0.215, -0.064] | -0.006 [-0.070, +0.059] |
| train-similar | ours (ranker) | +0.028 [-0.061, +0.115] | -0.018 [-0.105, +0.067] | -0.018 [-0.065, +0.027] |
| train-similar | ours (native order) | -0.083 [-0.167, +0.000] | -0.138 [-0.220, -0.058] | -0.064 [-0.122, -0.018] |

Top-N by the structure's own number of sites, against **p2rank**. At N = 1 top-N is top-1 and no merging of predictions can change it, so a deficit in that row is ranking and a deficit confined to N >= 2 is prediction fragmentation:

| subset | method | N | n | top-N | p2rank top-N | difference | ceiling | p2rank ceiling |
|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 1 | 210 | 0.762 | 0.843 | -0.081 | 0.990 | 0.948 |
| all | ours (ranker) | 2 | 65 | 0.815 | 0.846 | -0.031 | 0.985 | 0.938 |
| all | ours (ranker) | >=3 | 8 | 1.000 | 0.750 | +0.250 | 1.000 | 0.750 |
| all | ours (native order) | 1 | 210 | 0.667 | 0.843 | -0.176 | 0.990 | 0.948 |
| all | ours (native order) | 2 | 65 | 0.785 | 0.846 | -0.062 | 0.985 | 0.938 |
| all | ours (native order) | >=3 | 8 | 1.000 | 0.750 | +0.250 | 1.000 | 0.750 |
| not train-similar | ours (ranker) | 1 | 124 | 0.718 | 0.823 | -0.105 | 0.992 | 0.935 |
| not train-similar | ours (ranker) | 2 | 43 | 0.767 | 0.860 | -0.093 | 0.977 | 0.953 |
| not train-similar | ours (ranker) | >=3 | 7 | 1.000 | 0.714 | +0.286 | 1.000 | 0.714 |
| not train-similar | ours (native order) | 1 | 124 | 0.637 | 0.823 | -0.185 | 0.992 | 0.935 |
| not train-similar | ours (native order) | 2 | 43 | 0.791 | 0.860 | -0.070 | 0.977 | 0.953 |
| not train-similar | ours (native order) | >=3 | 7 | 1.000 | 0.714 | +0.286 | 1.000 | 0.714 |
| train-similar | ours (ranker) | 1 | 86 | 0.826 | 0.872 | -0.047 | 0.988 | 0.965 |
| train-similar | ours (ranker) | 2 | 22 | 0.909 | 0.818 | +0.091 | 1.000 | 0.909 |
| train-similar | ours (ranker) | >=3 | 1 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 |
| train-similar | ours (native order) | 1 | 86 | 0.709 | 0.872 | -0.163 | 0.988 | 0.965 |
| train-similar | ours (native order) | 2 | 22 | 0.773 | 0.818 | -0.045 | 1.000 | 0.909 |
| train-similar | ours (native order) | >=3 | 1 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 |
