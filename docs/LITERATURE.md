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
| **GotenNet, Aykent & Xia 2025** | **geometric tensor attention: high-degree channels updated through invariant-gated tensor products — the style of our layer; never applied to binding-site detection, which is our novelty claim** | code MIT | [из памяти] |
| PeSTo, Krapp et al. 2023 | parameter-free geometric transformer over atoms; interface prediction without hand features | MIT | [из памяти] |
| Clifford / geometric-algebra networks, Ruhe et al. 2023 | considered as an alternative degree-2 mechanism; not used (Cartesian tensors are enough and cheaper) | MIT | [из памяти] |

## Pocket detection and pocket models
| work | idea | status |
|---|---|---|
| LIGSITE, Hendlich et al. 1997 | grid buriedness by rays; our 26-ray closure is this idea reimplemented from the description | [из памяти] |
| fpocket, Le Guilloux et al. 2009 | alpha spheres; used only as an optional baseline in `scripts/baselines/` | MIT [из памяти] |
| P2Rank, Krivák & Hoksza 2018 | random forest over surface points, SAS-point scoring; optional baseline only | MIT [из памяти] |
| DeepPocket, Aggarwal et al. 2021 | CNN rescoring of fpocket candidates; source of the top-(N+2) convention and the COACH420/HOLO4K ligand rule | [из памяти] |
| **VN-EGNN, Sestak et al. 2024/2025 (J. Cheminformatics; PMC12837241)** | **virtual nodes with heterogeneous message passing, Dice + centre-distance loss, ESM-2 node features. We keep the virtual-node idea but place probes on real cavity points instead of a sphere; code written by us** | code MIT, text CC BY-NC-ND [из памяти] |
| EquiPocket, Zhang et al. 2023 | a dedicated surface module over the solvent-accessible surface; we use an SAS **point cloud** rather than a mesh | [из памяти] |
| GrASP, Smith et al. 2024 | graph attention over atoms with residue-level labels; evaluation lists on Zenodo (CC-BY-4.0) | [проверено] (licence), method [из памяти] |
| LIGYSIS / LBS-comparison, Utgés & Barton 2024 | benchmark and the finding that hybrid pipelines beat end-to-end detectors; our hybrid ranking follows that | CC-BY-4.0 [проверено] |
| CryptoBench, Skrhak et al. 2024 | cryptic-site benchmark; planned evaluation set | [не найдено] in this environment |
| ESM-2, Lin et al. 2023 | frozen residue embeddings as node features | MIT [из памяти] |

## Peptide-binding sites
| work | relevance | status |
|---|---|---|
| PepBDB, Wen et al. 2019; Propedia, Martins et al. 2021 | peptide-protein complex databases; own web terms, so we build our benchmark from RCSB instead | [из памяти] |
| PepNN, Abdin et al. 2022; InterPep2, Johansson-Åkhe et al. 2020 | peptide-site predictors; comparison targets once their weights and licences are verified | [из памяти] |
| β-augmentation / groove recognition literature (PDZ, SH3, MHC) | the backbone-exposure features of our groove tier | [из памяти] |

## What is ours
1. Candidate generation by lattice 26-ray closure with two depth tiers and peak splitting, calibrated on our own data.
2. Probe nodes on **real cavity points** (not a sphere) carrying their own buriedness geometry.
3. Degree-2 Cartesian tensor channels with geometric tensor attention **for pocket detection**, including the
   equivariant-vs-invariant ablation at equal depth and width, which we did not find in the pocket literature.
4. A groove tier with backbone-exposure features for peptide-binder sites, and a homology-controlled peptide-site
   benchmark built from RCSB.
5. One hotspot field shared by all heads: per-class ligand-atom probability per lattice point, labelled from the CCD.
