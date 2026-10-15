# Per-point ligandability model (max)

18507192 cavity grid points from 10808 structures, 0.1499 of them within 2.0 A of a ligand heavy atom. Out-of-fold by 30 %-identity cluster, 32 point features.

| metric | value |
|---|---|
| point AUROC | 0.861 |
| point average precision | 0.537 |
| average precision, scores permuted | 0.150 |
| positive rate (the trivial baseline) | 0.1499 |

Ordering candidates by one aggregate alone, on the structures of this table. `nat_score` is the generator's own geometric score and is the baseline to beat; the learned ranker uses all 118 pocket features and is reported in `ranker_<tag>.md`.

| ordering | top-1 | top-3 | top-N | top-(N+2) | MRR | ceiling | n |
|---|---|---|---|---|---|---|
| pts_sum | 0.670 [0.659, 0.682] | 0.833 [0.825, 0.842] | 0.750 [0.740, 0.761] | 0.871 [0.863, 0.878] | 0.763 [0.755, 0.771] | 0.968 | 10808 |
| pts_blob_max | 0.707 [0.696, 0.718] | 0.853 [0.845, 0.861] | 0.771 [0.760, 0.780] | 0.880 [0.872, 0.887] | 0.788 [0.779, 0.796] | 0.968 | 10808 |
| pts_max | 0.713 [0.702, 0.724] | 0.856 [0.847, 0.863] | 0.775 [0.765, 0.785] | 0.882 [0.874, 0.889] | 0.793 [0.785, 0.801] | 0.968 | 10808 |
| nat_score | 0.464 [0.452, 0.476] | 0.684 [0.673, 0.695] | 0.585 [0.573, 0.597] | 0.740 [0.730, 0.749] | 0.600 [0.590, 0.610] | 0.968 | 10808 |

Gain of the fifteen most used point features:

| feature | gain |
|---|---|
| p_n12 | 36265422 |
| p_buried | 7483085 |
| p_n_hba | 2940511 |
| p_f_polar8 | 2906011 |
| p_d_cation | 2636553 |
| p_n_aromatic | 2304843 |
| p_centrality | 2265160 |
| p_n6 | 2077722 |
| p_n8 | 1616706 |
| p_d_hbd | 1472838 |
| p_n_hbd | 1353334 |
| p_n_anion | 1233237 |
| p_d_anion | 1185530 |
| p_f_hydrophobic8 | 988878 |
| p_n_cation | 946196 |
