# Literature: what EquiCave takes from where

Status column: **[проверено]** the licence or claim was read from the primary source in this project;
**[из памяти]** stated from memory, to be rechecked before publication; **[не найдено]** not found, component unused.
Numbers quoted from papers are **not** comparable to ours: protocols, splits and ligand filters differ. Every
comparison in our tables is re-run on our own splits.

## Equivariant architectures
| work | idea we use (or deliberately do not) | licence of code / weights | status |
|---|---|---|---|
| EGNN, Satorras et al. 2021 | coordinate-update message passing; our baseline notion of degree-1 equivariance | MIT (code) | [из памяти] |
| PaiNN, Schütt et al. 2021 | separate scalar and vector channels, gated by invariants | MIT | [из памяти] |
| GVP-GNN, Jing et al. 2021 | geometric vector perceptron; vector norms as invariants | MIT | [из памяти] |
| Tensor-Field Networks / e3nn, Thomas et al. 2018; Geiger & Smidt 2022 | Clebsch-Gordan couplings up to l = 2 (we implement the Cartesian equivalents ourselves, no e3nn dependency) | MIT | [из памяти] |
| SE(3)-Transformer, Fuchs et al. 2020 | attention over equivariant messages | MIT | [из памяти] |
| Equiformer, Liao & Smidt 2023 | depth-wise tensor products, equivariant normalisation | MIT | [из памяти] |
| **GotenNet, Aykent & Xia 2025 (ICLR 2025)** | geometric tensor attention: high-degree channels updated through invariant-gated tensor products — the style of our layer. Evaluated on QM9 / rMD17 / MD22 / Molecule3D only, no protein task | code MIT | [проверено] (repository LICENSE, ICLR proceedings page) |
| **GDEGAN, Animesh, Bhowmick & Mitra 2026 (arXiv:2603.19817)** | **GotenNet with l=2 steerable features and ESM-2 applied to ligand binding-site prediction. This is prior art for applying degree-2 tensor attention to pockets, so our claim is now only the controlled isolation of that contribution, which GDEGAN does not provide (it has no lmax ablation).** Preprint, no code, no licence; copies its classical baselines from EquiPocket; never compares against VN-EGNN, GrASP or DeepPocket; its own ablation puts its namesake attention at +0.4 points of COACH420 DCA; its Tables 1 and 2 disagree on the identical GotenNet row | code and weights **[не найдено]** | [проверено] (PDF read) |
| PeSTo, Krapp et al. 2023 | parameter-free geometric transformer over atoms; interface prediction without hand features | **CC BY-NC-SA 4.0** (non-commercial, share-alike) | [проверено] (repository LICENSE) — not MIT as previously recorded here; keep out of any permissively licensed release |
| Clifford / geometric-algebra networks, Ruhe et al. 2023 | considered as an alternative degree-2 mechanism; not used (Cartesian tensors are enough and cheaper) | MIT | [из памяти] |

## Pocket detection and pocket models
| work | idea | status |
|---|---|---|
| LIGSITE, Hendlich et al. 1997 | grid buriedness by rays; our 26-ray closure is this idea reimplemented from the description | [из памяти] |
| fpocket, Le Guilloux et al. 2009 | alpha spheres; used only as an optional baseline in `scripts/baselines/` | MIT [из памяти] |
| P2Rank, Krivák & Hoksza 2018 | random forest over surface points, SAS-point scoring; optional baseline only | MIT [из памяти] |
| DeepPocket, Aggarwal et al. 2021 | CNN rescoring of fpocket candidates; source of the top-(N+2) convention and the COACH420/HOLO4K ligand rule | [из памяти] |
| **VN-EGNN, Sestak et al. 2024/2025 (J. Cheminformatics; PMC12837241)** | **virtual nodes with heterogeneous message passing, Dice + centre-distance loss, ESM-2 node features. We keep the virtual-node idea but place probes on real cavity points instead of a sphere; code written by us** | code MIT, text CC BY-NC-ND [из памяти] |
| EquiPocket, Zhang et al. 2023 (ICML 2024 / PMLR v235) | a dedicated surface module over the solvent-accessible surface; we use an SAS **point cloud** rather than a mesh. **Its arXiv version (2302.12177) was withdrawn by the authors in August 2026** ("requires substantial restructuring"); the PMLR version stands. It displaces predicted centres 4 Å outward before scoring, and its retrained P2Rank baseline collapses above 3000 training samples, so it reports a 792-sample subset. It is the anchor baseline of both VN-EGNN and GDEGAN, which inherits the problem | code and licence [не найдено]; arXiv states no licence due to withdrawal | [проверено] (arXiv withdrawal notice, PMLR PDF) |
| GrASP, Smith et al. 2024 | graph attention over atoms with residue-level labels; evaluation lists on Zenodo (CC-BY-4.0) | [проверено] (licence), method [из памяти] |
| LIGYSIS / LBS-comparison, Utgés & Barton 2024 | benchmark and the finding that hybrid pipelines beat end-to-end detectors; our hybrid ranking follows that | CC-BY-4.0 [проверено] |
| CryptoBench, Skrhak et al. 2024 | cryptic-site benchmark; planned evaluation set | [не найдено] in this environment |
| ESM-2, Lin et al. 2023 | frozen residue embeddings as node features | MIT [из памяти] |

## Prior art found by the 2026-10-04 survey (`docs/LITERATURE_SURVEY.md`), all to be cited
| work | why it matters to us |
|---|---|
| GDEGAN 2026 (arXiv:2603.19817) | GotenNet + l=2 + ESM-2 on pockets: prior art for our degree-2 application claim |
| **PharmacoNet, Seo & Kim, Chem. Sci. 2025** | **predicts 7 pharmacophore types on a grid with labels from interactions PLIP actually detects, not proximity. This is prior art for interaction-based hotspot labels; it is a 3D CNN, not equivariant, and its labels come from PLIP on PDBbind rather than from the CCD plus explicit geometric tests** |
| Equivariant Scalar Fields (Jing et al., MLSB 2023, arXiv:2312.04323, MIT) | a learned equivariant multi-channel field over a protein; its channels are docking-score components, not interaction classes |
| GENEOnet, Sci. Rep. 15:34597 (2025) | an equivariant pocket detector with 17 parameters; success@1 0.764 against P2Rank 0.702 on 6854 PDBbind proteins |
| YuelPocket, PNAS 2026 123(10):e2524913123 | PLINDER-trained; **excluded COACH420 outright because most of its systems were already in the training data** |
| ProMoSite / ProMoNet, J. Cheminform. 18:93 (2026) | sequence-only, COACH420 top-N 68.3 %, HOLO4K 73.5 %: no 3D input is needed to reach the EquiPocket / DeepPocket band |
| Lee, Byun & Shin 2023 (arXiv:2303.08818) | the closest existing geometry ablation for pockets: geometric self-attention against plain attention at equal depth and width, worth 4.3 points. Removing their SE(3)-invariant grid alignment *improved* COACH420 |
| HEGNN (arXiv:2410.11443) | high-degree steerable features help in general; contains no protein pocket task |
| DeepDFT (npj Comput. Mater. 2022) | probe nodes at query grid points receiving equivariant messages — the mechanism of our cavity probes, from electron-density prediction in materials |
| LIGYSIS comparison (Utgés & Barton 2024) | on the only biological-unit-aware independent benchmark the hybrid fpocket+PRANK wins (60.4 % top-N+2) and VN-EGNN ranks 12th of 13 with **67 % redundant predictions** |

## Peptide-binding sites
| work | relevance | status |
|---|---|---|
| PepBDB, Wen et al. 2019; Propedia, Martins et al. 2021 | peptide-protein complex databases; own web terms, so we build our benchmark from RCSB instead | [из памяти] |
| PepNN, Abdin et al. 2022; InterPep2, Johansson-Åkhe et al. 2020 | peptide-site predictors; comparison targets once their weights and licences are verified | [из памяти] |
| β-augmentation / groove recognition literature (PDZ, SH3, MHC) | the backbone-exposure features of our groove tier | [из памяти] |

## What is ours, as the survey of 2026-10-04 leaves it
1. **Candidate generation** by lattice 26-ray closure with two depth tiers and peak splitting, calibrated on our own
   data, with its ceiling measured (0.977 at 30 candidates) against fpocket and P2Rank on identical structures.
   fpocket's published candidate recall is 80.8 % on COACH420, which caps every fpocket-based pipeline.
2. **Probe nodes on real cavity points.** Published placements are a sphere (VN-EGNN), SAS surface points
   (EquiPocket, P2Rank, DeepSurf), voxel grids (DeepSite, Kalasanty, GENEOnet) or alpha-sphere centres as candidates
   for rescoring (DeepPocket). Probe nodes *at cavity grid points receiving equivariant messages* were **not found**
   for proteins; the mechanism exists in DeepDFT for electron density. This is a negative search result: "we did not
   find", not "nobody has done".
3. **The controlled isolation of degree 2 and of equivariance** at equal depth, width and features. GDEGAN applies
   l=2 tensor attention to pockets but provides no lmax ablation; no equivariant-versus-invariant ablation at matched
   capacity exists for pocket detection, and the one adjacent measurement (EquiPNAS, protein-nucleic acid) is
   negligible. Our `no_tensors`, `no_vectors`, `invariant`, `achiral`, `e3nn_l2` and `e3nn_l3` arms are therefore the
   contribution, not the mechanism itself.
4. **Peptide binding sites**, the largest gap the survey found: the field is still per-residue classification with
   sequence models (PepNN, PepCA, PepBCL, InterPep2). No equivariant geometric network, no degree >= 2 features, no
   probe method and **no DCA/DCC site-localisation protocol** exists for peptide sites. Our groove tier, the
   homology-controlled RCSB benchmark and a DCA/DCC protocol for peptides are unoccupied ground.
5. **Interaction-validated hotspot labels as an equivariant field.** PharmacoNet already labels pharmacophore types
   by interactions PLIP detects, so label realism is not ours; a *learned equivariant* per-class field is not found,
   and our labels come from the CCD plus explicit geometric interaction tests rather than from a PLIP run, with the
   proximity variant kept as a control.
6. **The protocol**, which the survey shows the field lacks: top-N and top-(N+2), DCC at 4, 10 and 12 A, a redundancy
   statistic, the ligand rule and resulting counts stated, train-similar structures separated, and every baseline
   re-run in house rather than copied from a table.
