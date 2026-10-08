# Per-point ligandability model (native)

2308491 cavity grid points from 1367 structures, 0.1513 of them within 2.0 A of a ligand heavy atom. Out-of-fold by 30 %-identity cluster, 32 point features.

| metric | value |
|---|---|
| point AUROC | 0.865 |
| point average precision | 0.542 |
| average precision, scores permuted | 0.151 |
| positive rate (the trivial baseline) | 0.1513 |

Ordering candidates by one aggregate alone, on the structures of this table. `nat_score` is the generator's own geometric score and is the baseline to beat; the learned ranker uses all 118 pocket features and is reported in `ranker_<tag>.md`.

| ordering | top-1 | top-3 | top-N | top-(N+2) | MRR | ceiling | n |
|---|---|---|---|---|---|---|
| pts_sum | 0.696 [0.670, 0.722] | 0.851 [0.830, 0.872] | 0.772 [0.748, 0.795] | 0.885 [0.867, 0.904] | 0.783 [0.763, 0.803] | 0.977 | 1367 |
| pts_blob_max | 0.732 [0.707, 0.758] | 0.870 [0.851, 0.888] | 0.793 [0.770, 0.815] | 0.893 [0.876, 0.910] | 0.808 [0.789, 0.827] | 0.977 | 1367 |
| pts_max | 0.729 [0.704, 0.755] | 0.876 [0.857, 0.895] | 0.784 [0.761, 0.808] | 0.898 [0.882, 0.915] | 0.809 [0.790, 0.828] | 0.977 | 1367 |
| nat_score | 0.473 [0.444, 0.503] | 0.691 [0.664, 0.719] | 0.585 [0.556, 0.615] | 0.747 [0.722, 0.773] | 0.610 [0.586, 0.634] | 0.977 | 1367 |

Gain of the fifteen most used point features:

| feature | gain |
|---|---|
| p_n12 | 5038717 |
| p_buried | 997176 |
| p_f_polar8 | 614131 |
| p_centrality | 540784 |
| p_d_cation | 387162 |
| p_n_hba | 324788 |
| p_n_aromatic | 292475 |
| p_n6 | 287841 |
| p_n8 | 260366 |
| p_d_anion | 243113 |
| p_d_aromatic | 201972 |
| p_d_hbd | 190398 |
| p_n_hydrophobic | 187755 |
| p_f_hydrophobic8 | 176827 |
| p_n_anion | 174457 |
