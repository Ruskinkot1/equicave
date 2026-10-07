# ligysis: our predictions against external tools on the same structures

1613 structures predicted by every method listed. Success is DCA <= 4 A to a ligand of the set's own relevant-ligand list; N is the structure's own number of ligand sites. 95 % CI by 30 %-identity cluster bootstrap. `train-similar` means the structure shares a 30 %-identity cluster with **our** training manifest; it is leakage for us and not for the external tools, whose own training sets overlap this benchmark in ways this table does not measure. The comparable row for us is **not train-similar**.

| subset | method | predictions | ceiling | top-1 | top-3 | top-N | top-(N+2) | MRR | n |
|---|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 30.8 | 0.943 | 0.659 [0.634, 0.685] | 0.803 [0.781, 0.824] | 0.748 [0.725, 0.771] | 0.847 [0.827, 0.865] | 0.742 [0.721, 0.763] | 1613 |
| all | ours (native order) | 30.8 | 0.943 | 0.456 [0.427, 0.484] | 0.670 [0.642, 0.697] | 0.601 [0.571, 0.632] | 0.743 [0.718, 0.769] | 0.584 [0.559, 0.607] | 1613 |
| all | p2rank | 20.2 | 0.909 | 0.679 [0.652, 0.705] | 0.810 [0.789, 0.831] | 0.778 [0.754, 0.803] | 0.843 [0.823, 0.863] | 0.752 [0.731, 0.774] | 1613 |
| not train-similar | ours (ranker) | 31.1 | 0.941 | 0.619 [0.589, 0.650] | 0.777 [0.750, 0.803] | 0.719 [0.689, 0.747] | 0.830 [0.805, 0.853] | 0.712 [0.687, 0.736] | 1088 |
| not train-similar | ours (native order) | 31.1 | 0.941 | 0.439 [0.409, 0.469] | 0.650 [0.620, 0.680] | 0.582 [0.550, 0.611] | 0.723 [0.695, 0.751] | 0.568 [0.542, 0.593] | 1088 |
| not train-similar | p2rank | 20.7 | 0.897 | 0.661 [0.631, 0.691] | 0.788 [0.761, 0.814] | 0.763 [0.736, 0.791] | 0.824 [0.800, 0.847] | 0.733 [0.708, 0.758] | 1088 |
| train-similar | ours (ranker) | 30.1 | 0.947 | 0.741 [0.694, 0.783] | 0.857 [0.824, 0.887] | 0.810 [0.770, 0.847] | 0.882 [0.850, 0.911] | 0.806 [0.770, 0.838] | 525 |
| train-similar | ours (native order) | 30.1 | 0.947 | 0.490 [0.434, 0.547] | 0.712 [0.664, 0.762] | 0.642 [0.578, 0.703] | 0.785 [0.740, 0.831] | 0.616 [0.570, 0.662] | 525 |
| train-similar | p2rank | 19.2 | 0.933 | 0.716 [0.661, 0.766] | 0.857 [0.820, 0.891] | 0.810 [0.759, 0.855] | 0.884 [0.848, 0.916] | 0.792 [0.750, 0.829] | 525 |

Paired differences against **p2rank** on the same structures (positive = we are better):

| subset | method | top-1 | top-N | top-(N+2) |
|---|---|---|---|---|
| all | ours (ranker) | -0.020 [-0.046, +0.006] | -0.030 [-0.051, -0.008] | +0.004 [-0.013, +0.020] |
| all | ours (native order) | -0.223 [-0.252, -0.198] | -0.177 [-0.202, -0.152] | -0.100 [-0.121, -0.078] |
| not train-similar | ours (ranker) | -0.041 [-0.073, -0.009] | -0.044 [-0.071, -0.017] | +0.006 [-0.016, +0.028] |
| not train-similar | ours (native order) | -0.222 [-0.252, -0.190] | -0.181 [-0.212, -0.153] | -0.100 [-0.128, -0.075] |
| train-similar | ours (ranker) | +0.025 [-0.021, +0.073] | +0.000 [-0.041, +0.040] | -0.002 [-0.028, +0.025] |
| train-similar | ours (native order) | -0.227 [-0.276, -0.175] | -0.168 [-0.212, -0.121] | -0.099 [-0.138, -0.059] |

Top-N by the structure's own number of sites, against **p2rank**. At N = 1 top-N is top-1 and no merging of predictions can change it, so a deficit in that row is ranking and a deficit confined to N >= 2 is prediction fragmentation:

| subset | method | N | n | top-N | p2rank top-N | difference | ceiling | p2rank ceiling |
|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 1 | 536 | 0.644 | 0.690 | -0.047 | 0.938 | 0.868 |
| all | ours (ranker) | 2 | 489 | 0.751 | 0.783 | -0.033 | 0.949 | 0.920 |
| all | ours (ranker) | >=3 | 588 | 0.842 | 0.854 | -0.012 | 0.942 | 0.937 |
| all | ours (native order) | 1 | 536 | 0.500 | 0.690 | -0.190 | 0.938 | 0.868 |
| all | ours (native order) | 2 | 489 | 0.593 | 0.783 | -0.190 | 0.949 | 0.920 |
| all | ours (native order) | >=3 | 588 | 0.701 | 0.854 | -0.153 | 0.942 | 0.937 |
| not train-similar | ours (ranker) | 1 | 369 | 0.612 | 0.702 | -0.089 | 0.938 | 0.862 |
| not train-similar | ours (ranker) | 2 | 345 | 0.713 | 0.757 | -0.043 | 0.939 | 0.899 |
| not train-similar | ours (ranker) | >=3 | 374 | 0.829 | 0.829 | +0.000 | 0.947 | 0.930 |
| not train-similar | ours (native order) | 1 | 369 | 0.501 | 0.702 | -0.201 | 0.938 | 0.862 |
| not train-similar | ours (native order) | 2 | 345 | 0.565 | 0.757 | -0.191 | 0.939 | 0.899 |
| not train-similar | ours (native order) | >=3 | 374 | 0.676 | 0.829 | -0.152 | 0.947 | 0.930 |
| train-similar | ours (ranker) | 1 | 167 | 0.713 | 0.665 | +0.048 | 0.940 | 0.880 |
| train-similar | ours (ranker) | 2 | 144 | 0.840 | 0.847 | -0.007 | 0.972 | 0.972 |
| train-similar | ours (ranker) | >=3 | 214 | 0.864 | 0.897 | -0.033 | 0.935 | 0.949 |
| train-similar | ours (native order) | 1 | 167 | 0.497 | 0.665 | -0.168 | 0.940 | 0.880 |
| train-similar | ours (native order) | 2 | 144 | 0.660 | 0.847 | -0.188 | 0.972 | 0.972 |
| train-similar | ours (native order) | >=3 | 214 | 0.743 | 0.897 | -0.154 | 0.935 | 0.949 |
