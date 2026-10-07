# ligysis: our predictions against external tools on the same structures

1613 structures predicted by every method listed. Success is DCC <= 4 A to a ligand centroid of the set's own relevant-ligand list; N is the structure's own number of ligand sites. 95 % CI by 30 %-identity cluster bootstrap. `train-similar` means the structure shares a 30 %-identity cluster with **our** training manifest; it is leakage for us and not for the external tools, whose own training sets overlap this benchmark in ways this table does not measure. The comparable row for us is **not train-similar**.

| subset | method | predictions | ceiling | top-1 | top-3 | top-N | top-(N+2) | MRR | n |
|---|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 30.8 | 0.838 | 0.379 [0.355, 0.404] | 0.597 [0.572, 0.622] | 0.515 [0.489, 0.540] | 0.668 [0.643, 0.692] | 0.510 [0.489, 0.532] | 1613 |
| all | ours (native order) | 30.8 | 0.838 | 0.270 [0.245, 0.296] | 0.479 [0.451, 0.508] | 0.411 [0.383, 0.439] | 0.560 [0.531, 0.588] | 0.406 [0.383, 0.428] | 1613 |
| all | p2rank | 20.2 | 0.670 | 0.436 [0.410, 0.463] | 0.570 [0.544, 0.596] | 0.548 [0.522, 0.575] | 0.609 [0.584, 0.637] | 0.511 [0.486, 0.535] | 1613 |
| not train-similar | ours (ranker) | 31.1 | 0.821 | 0.352 [0.322, 0.383] | 0.566 [0.535, 0.598] | 0.482 [0.450, 0.514] | 0.644 [0.614, 0.675] | 0.482 [0.455, 0.508] | 1088 |
| not train-similar | ours (native order) | 31.1 | 0.821 | 0.255 [0.226, 0.283] | 0.450 [0.420, 0.483] | 0.384 [0.353, 0.416] | 0.536 [0.505, 0.566] | 0.385 [0.360, 0.410] | 1088 |
| not train-similar | p2rank | 20.7 | 0.653 | 0.429 [0.399, 0.462] | 0.550 [0.520, 0.581] | 0.538 [0.507, 0.571] | 0.591 [0.562, 0.621] | 0.498 [0.471, 0.527] | 1088 |
| train-similar | ours (ranker) | 30.1 | 0.872 | 0.436 [0.392, 0.485] | 0.661 [0.616, 0.709] | 0.583 [0.538, 0.629] | 0.716 [0.673, 0.758] | 0.569 [0.532, 0.610] | 525 |
| train-similar | ours (native order) | 30.1 | 0.872 | 0.303 [0.255, 0.356] | 0.539 [0.486, 0.600] | 0.467 [0.408, 0.528] | 0.611 [0.557, 0.671] | 0.449 [0.406, 0.497] | 525 |
| train-similar | p2rank | 19.2 | 0.703 | 0.450 [0.400, 0.502] | 0.611 [0.563, 0.660] | 0.570 [0.519, 0.622] | 0.648 [0.599, 0.695] | 0.536 [0.490, 0.582] | 525 |

Paired differences against **p2rank** on the same structures (positive = we are better):

| subset | method | top-1 | top-N | top-(N+2) |
|---|---|---|---|---|
| all | ours (ranker) | -0.056 [-0.088, -0.026] | -0.033 [-0.062, -0.005] | +0.058 [+0.031, +0.086] |
| all | ours (native order) | -0.166 [-0.195, -0.136] | -0.137 [-0.169, -0.106] | -0.049 [-0.079, -0.018] |
| not train-similar | ours (ranker) | -0.077 [-0.113, -0.040] | -0.056 [-0.088, -0.024] | +0.053 [+0.022, +0.083] |
| not train-similar | ours (native order) | -0.175 [-0.209, -0.143] | -0.153 [-0.189, -0.121] | -0.055 [-0.089, -0.023] |
| train-similar | ours (ranker) | -0.013 [-0.072, +0.046] | +0.013 [-0.044, +0.074] | +0.069 [+0.011, +0.126] |
| train-similar | ours (native order) | -0.147 [-0.206, -0.085] | -0.103 [-0.165, -0.033] | -0.036 [-0.102, +0.035] |

Top-N by the structure's own number of sites, against **p2rank**. At N = 1 top-N is top-1 and no merging of predictions can change it, so a deficit in that row is ranking and a deficit confined to N >= 2 is prediction fragmentation:

| subset | method | N | n | top-N | p2rank top-N | difference | ceiling | p2rank ceiling |
|---|---|---|---|---|---|---|---|---|
| all | ours (ranker) | 1 | 536 | 0.386 | 0.470 | -0.084 | 0.821 | 0.616 |
| all | ours (ranker) | 2 | 489 | 0.509 | 0.524 | -0.014 | 0.865 | 0.650 |
| all | ours (ranker) | >=3 | 588 | 0.636 | 0.639 | -0.003 | 0.830 | 0.735 |
| all | ours (native order) | 1 | 536 | 0.310 | 0.470 | -0.160 | 0.821 | 0.616 |
| all | ours (native order) | 2 | 489 | 0.409 | 0.524 | -0.115 | 0.865 | 0.650 |
| all | ours (native order) | >=3 | 588 | 0.505 | 0.639 | -0.134 | 0.830 | 0.735 |
| not train-similar | ours (ranker) | 1 | 369 | 0.360 | 0.485 | -0.125 | 0.813 | 0.612 |
| not train-similar | ours (ranker) | 2 | 345 | 0.464 | 0.493 | -0.029 | 0.832 | 0.612 |
| not train-similar | ours (ranker) | >=3 | 374 | 0.618 | 0.631 | -0.013 | 0.818 | 0.733 |
| not train-similar | ours (native order) | 1 | 369 | 0.304 | 0.485 | -0.182 | 0.813 | 0.612 |
| not train-similar | ours (native order) | 2 | 345 | 0.368 | 0.493 | -0.125 | 0.832 | 0.612 |
| not train-similar | ours (native order) | >=3 | 374 | 0.479 | 0.631 | -0.152 | 0.818 | 0.733 |
| train-similar | ours (ranker) | 1 | 167 | 0.443 | 0.437 | +0.006 | 0.838 | 0.623 |
| train-similar | ours (ranker) | 2 | 144 | 0.618 | 0.597 | +0.021 | 0.944 | 0.743 |
| train-similar | ours (ranker) | >=3 | 214 | 0.668 | 0.654 | +0.014 | 0.850 | 0.738 |
| train-similar | ours (native order) | 1 | 167 | 0.323 | 0.437 | -0.114 | 0.838 | 0.623 |
| train-similar | ours (native order) | 2 | 144 | 0.507 | 0.597 | -0.090 | 0.944 | 0.743 |
| train-similar | ours (native order) | >=3 | 214 | 0.551 | 0.654 | -0.103 | 0.850 | 0.738 |
