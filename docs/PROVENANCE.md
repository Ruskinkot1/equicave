# Provenance: borrowed ideas, our implementations, and what was verified

No third-party source code is copied into this repository. Every file in `src/` and `training/` was written here.
Where an idea comes from a publication, it is listed below with the form in which we implemented it.

| our component | borrowed idea | source | our implementation |
|---|---|---|---|
| `pockets._buried`, `detect.buriedness_grid` | grid-based buriedness by ray casting | LIGSITE (Hendlich 1997) | 26 lattice directions, 8 Å rays, 2 Å hit radius; computed by integer array shifts on an absolute 1 Å lattice; the lattice form is tested against an exact per-point ray caster |
| `detect.detect_sites` peak splitting | non-maximum suppression to split one cavity into sub-sites | common practice in object detection; P2Rank clusters SAS points instead | Gaussian smoothing of the buriedness field, greedy NMS at 6 Å, nearest-peak assignment, two depth tiers |
| `pockets.sas_points` | solvent-accessible surface sampling | Shrake & Rupley 1973 | Fibonacci sphere per atom, vdW + 1.4 Å probe, neighbour occlusion test; returns a point cloud with owner atom indices |
| `training/pockets/model.py` channels | scalar + vector channels gated by invariants | PaiNN (Schütt 2021), GVP (Jing 2021) | our own gating, plus degree-2 symmetric traceless tensors |
| `GeoTensorAttention` | attention over equivariant messages with high-degree channels (geometric tensor attention) | SE(3)-Transformer (Fuchs 2020), Equiformer (Liao 2023), GotenNet (Aykent & Xia 2025) | Cartesian l ≤ 2 primitives written out by hand (dot, `uᵀTu`, `Tu`, `sym_traceless`), segment softmax over incoming edges, pseudo-scalar triple products for chirality; no e3nn, no copied code |
| probe (virtual) nodes | virtual nodes that learn to sit at binding sites | VN-EGNN (Sestak 2024/2025) | probes are initialised on **free cavity lattice points** with their own buriedness and direction features, not on a sphere; heterogeneous edge types by node-type pair |
| surface module | a dedicated module over the molecular surface | EquiPocket (Zhang 2023) | SAS **point cloud** nodes with owner element/residue features and outward normals as initial vectors, inside the same attention layers |
| ESM-2 node features | frozen language-model embeddings as residue features | ESM-2 (Lin 2023); used by VN-EGNN and GDEGAN | `transformers` inference, cached as float16 per structure; ablated with `no_esm` |
| segmentation + centre losses | Dice loss for site segmentation, distance-to-centre regression with a set assignment | VN-EGNN (Sestak 2024/2025) | Dice + BCE with pos_weight; set loss over **all** sites of the structure using the nearest probe within 8 Å; confidence head with a detached target |
| hotspot field | per-point ligand-atom type probability (pharmacophore-style maps) | fragment hotspot maps (Radoux 2016), FTMap-type methods | focal BCE over 7 CCD-derived atom classes on cavity lattice points, labelled from crystal ligand atoms within 1.5 Å |
| top-(N+2) and the ligand rule | evaluation convention for pocket detection | P2Rank (Krivák 2018), DeepPocket (Aggarwal 2021) | implemented in `metrics.per_structure`; the `_mlig` lists are fetched, not redistributed |
| hybrid ranking | a learned re-ranker over candidates beats end-to-end detection | LIGYSIS comparison (Utgés & Barton 2024) | LightGBM LambdaRank over our own features, cluster CV, paired cluster bootstrap |
| peptide groove features | backbone-to-backbone recognition in peptide grooves | β-augmentation / PDZ, SH3, MHC structural literature | `peptide.backbone_exposure`: counts of receptor backbone N, O, Cα and side-chain atoms within 6 Å of each candidate point, plus shape anisotropy |

## Verification log
- RCSB and CCD licensing (CC0): read from the wwPDB policy pages and the file endpoints used, 2026-10-03 [проверено].
- p2rank-datasets has no LICENSE file at the tree root: checked by fetching the raw paths, 2026-10-03 [проверено].
  Only PDB ids are used; the lists are fetched by the user's own run and are not committed.
- LIGYSIS Zenodo record 13121414 metadata states CC-BY-4.0, 2026-10-03 [проверено].
- CryptoBench repository was not reachable from this environment, 2026-10-03 [не найдено].
- fpocket 4.x and P2Rank 2.5.1 were built/downloaded locally **only** for `scripts/baselines/`; neither `src/` nor
  `training/` imports or executes them, and no Java is required to train or run EquiCave.
- ESM-2 licence (MIT) is [из памяти] in this session: the model card was not opened from the primary source.
