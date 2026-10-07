# ligysis: our predictions against external tools on the same structures

714 structures predicted by every method listed. Success is DCA <= 4 A to a ligand of the set's own relevant-ligand list; N is the structure's own number of ligand sites. 95 % CI by 30 %-identity cluster bootstrap. `train-similar` means the structure shares a 30 %-identity cluster with **our** training manifest; it is leakage for us and not for the external tools, whose own training sets overlap this benchmark in ways this table does not measure. The comparable row for us is **not train-similar**.

| subset | method | predictions | ceiling | top-1 | top-3 | top-N | top-(N+2) | MRR | n |
|---|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 30.9 | 0.940 | 0.648 [0.609, 0.687] | 0.796 [0.765, 0.826] | 0.739 [0.705, 0.772] | 0.850 [0.823, 0.876] | 0.733 [0.702, 0.764] | 714 |
| all | ours (native order) | 30.9 | 0.940 | 0.461 [0.423, 0.499] | 0.661 [0.623, 0.698] | 0.606 [0.568, 0.644] | 0.739 [0.704, 0.772] | 0.582 [0.550, 0.614] | 714 |
| all | p2rank | 19.1 | 0.896 | 0.651 [0.613, 0.688] | 0.786 [0.753, 0.816] | 0.773 [0.739, 0.806] | 0.833 [0.803, 0.861] | 0.729 [0.699, 0.758] | 714 |
| not train-similar | ours (ranker) | 31.1 | 0.940 | 0.625 [0.579, 0.666] | 0.786 [0.749, 0.820] | 0.723 [0.684, 0.762] | 0.846 [0.815, 0.877] | 0.717 [0.681, 0.749] | 546 |
| not train-similar | ours (native order) | 31.1 | 0.940 | 0.445 [0.404, 0.488] | 0.648 [0.606, 0.690] | 0.584 [0.541, 0.627] | 0.733 [0.693, 0.773] | 0.572 [0.536, 0.606] | 546 |
| not train-similar | p2rank | 19.3 | 0.888 | 0.645 [0.605, 0.687] | 0.773 [0.736, 0.809] | 0.766 [0.729, 0.802] | 0.821 [0.786, 0.855] | 0.720 [0.687, 0.754] | 546 |
| train-similar | ours (ranker) | 30.3 | 0.940 | 0.726 [0.645, 0.798] | 0.827 [0.763, 0.883] | 0.792 [0.723, 0.855] | 0.863 [0.804, 0.913] | 0.788 [0.726, 0.843] | 168 |
| train-similar | ours (native order) | 30.3 | 0.940 | 0.512 [0.423, 0.594] | 0.702 [0.620, 0.778] | 0.679 [0.593, 0.754] | 0.762 [0.692, 0.827] | 0.617 [0.543, 0.684] | 168 |
| train-similar | p2rank | 18.7 | 0.923 | 0.673 [0.589, 0.752] | 0.827 [0.763, 0.889] | 0.798 [0.726, 0.860] | 0.875 [0.816, 0.922] | 0.760 [0.695, 0.820] | 168 |

Paired differences against **p2rank** on the same structures (positive = we are better):

| subset | method | top-1 | top-N | top-(N+2) |
|---|---|---|---|---|
| all | ours (ranker) | -0.003 [-0.040, +0.036] | -0.034 [-0.062, -0.003] | +0.017 [-0.008, +0.042] |
| all | ours (native order) | -0.190 [-0.230, -0.152] | -0.167 [-0.202, -0.129] | -0.094 [-0.126, -0.063] |
| not train-similar | ours (ranker) | -0.020 [-0.065, +0.024] | -0.042 [-0.078, -0.004] | +0.026 [-0.004, +0.054] |
| not train-similar | ours (native order) | -0.200 [-0.243, -0.157] | -0.181 [-0.223, -0.139] | -0.088 [-0.125, -0.052] |
| train-similar | ours (ranker) | +0.054 [-0.022, +0.128] | -0.006 [-0.059, +0.048] | -0.012 [-0.058, +0.034] |
| train-similar | ours (native order) | -0.161 [-0.242, -0.080] | -0.119 [-0.186, -0.053] | -0.113 [-0.175, -0.053] |

**Composition of the hard-novelty subset.** Filtering by homology also filters by whatever correlates with it, so the aggregate difference below is not purely a leakage effect. These are the two populations the filter separated:

| population | structures | sites per structure | single-site | >= 3 sites |
|---|---|---|---|---|
| kept (novel) | 729 | 3.30 | 0.32 | 0.36 |
| removed (homologous) | 908 | 3.44 | 0.35 | 0.36 |

Top-N by the structure's own number of sites, against **p2rank**. At N = 1 top-N is top-1 and no merging of predictions can change it, so a deficit in that row is ranking and a deficit confined to N >= 2 is prediction fragmentation:

| subset | method | N | n | top-N | p2rank top-N | difference | ceiling | p2rank ceiling |
|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 1 | 222 | 0.635 | 0.671 | -0.036 | 0.941 | 0.824 |
| all | ours (ranker) | 2 | 227 | 0.762 | 0.797 | -0.035 | 0.947 | 0.925 |
| all | ours (ranker) | >=3 | 265 | 0.808 | 0.838 | -0.030 | 0.932 | 0.932 |
| all | ours (native order) | 1 | 222 | 0.509 | 0.671 | -0.162 | 0.941 | 0.824 |
| all | ours (native order) | 2 | 227 | 0.634 | 0.797 | -0.163 | 0.947 | 0.925 |
| all | ours (native order) | >=3 | 265 | 0.664 | 0.838 | -0.174 | 0.932 | 0.932 |
| not train-similar | ours (ranker) | 1 | 172 | 0.616 | 0.692 | -0.076 | 0.936 | 0.820 |
| not train-similar | ours (ranker) | 2 | 182 | 0.742 | 0.780 | -0.038 | 0.945 | 0.918 |
| not train-similar | ours (ranker) | >=3 | 192 | 0.802 | 0.818 | -0.016 | 0.938 | 0.922 |
| not train-similar | ours (native order) | 1 | 172 | 0.500 | 0.692 | -0.192 | 0.936 | 0.820 |
| not train-similar | ours (native order) | 2 | 182 | 0.604 | 0.780 | -0.176 | 0.945 | 0.918 |
| not train-similar | ours (native order) | >=3 | 192 | 0.641 | 0.818 | -0.177 | 0.938 | 0.922 |
| train-similar | ours (ranker) | 1 | 50 | 0.700 | 0.600 | +0.100 | 0.960 | 0.840 |
| train-similar | ours (ranker) | 2 | 45 | 0.844 | 0.867 | -0.022 | 0.956 | 0.956 |
| train-similar | ours (ranker) | >=3 | 73 | 0.822 | 0.890 | -0.068 | 0.918 | 0.959 |
| train-similar | ours (native order) | 1 | 50 | 0.540 | 0.600 | -0.060 | 0.960 | 0.840 |
| train-similar | ours (native order) | 2 | 45 | 0.756 | 0.867 | -0.111 | 0.956 | 0.956 |
| train-similar | ours (native order) | >=3 | 73 | 0.726 | 0.890 | -0.164 | 0.918 | 0.959 |
