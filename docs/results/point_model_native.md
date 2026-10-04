# Per-point ligandability model (native)

2308510 cavity grid points from 1367 structures, 0.1514 of them within 2.0 A of a ligand heavy atom. Out-of-fold by 30 %-identity cluster, 32 point features.

| metric | value |
|---|---|
| point AUROC | 0.865 |
| point average precision | 0.543 |
| average precision, scores permuted | 0.151 |
| positive rate (the trivial baseline) | 0.1514 |

Ordering candidates by one aggregate alone, on the structures of this table. `nat_score` is the generator's own geometric score and is the baseline to beat; the learned ranker uses all 118 pocket features and is reported in `ranker_<tag>.md`.

| ordering | top-1 | top-3 | top-N | top-(N+2) | MRR | ceiling | n |
|---|---|---|---|---|---|---|
| pts_sum | 0.699 [0.674, 0.725] | 0.851 [0.831, 0.873] | 0.775 [0.751, 0.798] | 0.884 [0.865, 0.903] | 0.785 [0.766, 0.805] | 0.977 | 1367 |
| pts_blob_max | 0.735 [0.710, 0.760] | 0.869 [0.850, 0.889] | 0.796 [0.773, 0.818] | 0.893 [0.875, 0.911] | 0.810 [0.791, 0.828] | 0.977 | 1367 |
| pts_max | 0.741 [0.716, 0.766] | 0.878 [0.859, 0.897] | 0.793 [0.770, 0.816] | 0.902 [0.885, 0.919] | 0.815 [0.796, 0.834] | 0.977 | 1367 |
| nat_score | 0.473 [0.445, 0.503] | 0.691 [0.664, 0.719] | 0.585 [0.556, 0.615] | 0.747 [0.722, 0.773] | 0.611 [0.587, 0.634] | 0.977 | 1367 |

Gain of the fifteen most used point features:

| feature | gain |
|---|---|
| p_n12 | 5052495 |
| p_buried | 1003989 |
| p_f_polar8 | 617937 |
| p_centrality | 531574 |
| p_d_cation | 380793 |
| p_n_hba | 330234 |
| p_n_aromatic | 285610 |
| p_n6 | 280263 |
| p_n8 | 261219 |
| p_d_anion | 241621 |
| p_d_aromatic | 207547 |
| p_n_hydrophobic | 195782 |
| p_d_hbd | 186632 |
| p_n_anion | 176188 |
| p_f_hydrophobic8 | 175113 |
