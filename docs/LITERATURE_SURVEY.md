# Equivariant and geometric deep learning for protein structure, with emphasis on binding-site (pocket) detection

**State of the art as of 2026-10-04.** Compiled by literature search from primary sources (publisher pages, PMC
full text, arXiv PDFs, Europe PMC metadata API, repository `LICENSE` files fetched from
`raw.githubusercontent.com`).

## How to read this document

| tag | meaning |
|---|---|
| **[V]** | read from a primary source during this survey (publisher/PMC full text, arXiv PDF, repo LICENSE file, Europe PMC record). The exact source is named. |
| **[UNVERIFIED]** | stated but **not** confirmed from a primary source in this session. Do not cite without checking. |
| **[NOT FOUND]** | searched for and not found, or the page was unreachable. |

**Three warnings that apply to every number in this document.**

1. **Numbers from different papers on COACH420/HOLO4K are not comparable.** The same benchmark name hides
   different ligand filters, different chain handling, different numbers of systems, and in one case a
   coordinate post-processing step. §6 tabulates the same metric across papers to make the spread visible:
   P2Rank's DCA top-N on HOLO4K is reported as **68.6 %**, **70.6 %**, **78.7 %** and **81.2 %** by four
   different papers. Any claim of "state of the art" that mixes rows from different papers is unsound.
2. **GitHub repository licences cover code, not weights, and not training data.** Several methods ship MIT
   code trained on data with its own terms (scPDB, BioLiP, PDBbind). A permissive code licence does not make
   the weights redistributable.
3. **One search result in this survey was this project's own repository.** A web search for virtual-node /
   probe-refinement work returned `github.com/Ruskinkot1/equicave` and a description of "EquiCave-Net".
   That is EquiCave itself, indexed by the search engine; it is **not** independent literature and is excluded
   from every table and from the answers to A-G.

---

## 1. Equivariant backbones

| Method | Full name | Year | Venue | Mechanism | Trained for | Code licence | Weights | Pocket metrics? |
|---|---|---|---|---|---|---|---|---|
| **EGNN** | E(n) Equivariant Graph Neural Networks | 2021 (arXiv 2021-02-19) **[V]** arXiv:2102.09844 | ICML 2021 **[UNVERIFIED]** (arXiv listing gives no journal-ref) | Scalar messages from invariants (relative distance), separate coordinate update `x_i ← x_i + Σ (x_i−x_j) φ_x(m_ij)`. Degree-1 only, no spherical harmonics, no tensor products. | N-body, QM9, graph autoencoding | **MIT [V]** (`vgsatorras/egnn`, `main/LICENSE`) | n/a | No |
| **PaiNN** | Equivariant message passing for the prediction of tensorial properties and molecular spectra | 2021 (arXiv 2021-02-05) **[V]** arXiv:2102.03150 | **ICML 2021 [V]** (arXiv comment "Accepted at ICML 2021") | Parallel scalar and equivariant *vector* channels; vectors mixed by invariant gates and scalar products; degree ≤ 1. | Molecular property/spectra prediction | SchNetPack: custom MIT-style ("COPYRIGHT … All other contributions") **[V]** (`atomistic-machine-learning/schnetpack`) | n/a | No |
| **GVP-GNN** | Geometric Vector Perceptron GNN ("Learning from Protein Structure with Geometric Vector Perceptrons") | 2020 (arXiv 2020-09-03) **[V]** arXiv:2009.01411 | **ICLR 2021 [V]** (arXiv comment) | Perceptron acting jointly on scalar and vector features; rotation equivariance of vectors, invariance of scalars via vector *norms*. Degree ≤ 1. | CPD (computational protein design), Atom3D tasks | **MIT [V]** (`drorlab/gvp-pytorch`) | n/a | No (but see PocketMiner, §3) |
| **TFN** | Tensor field networks | 2018 (arXiv 2018-02-22) **[V]** arXiv:1802.08219 | arXiv / NIPS submission **[V]** (arXiv comment "changes for NIPS submission") | Point convolutions with filters factorised as radial function × spherical harmonic `Y^(l)`; features are SO(3) irreps combined by Clebsch–Gordan tensor products. Arbitrary degree `l`. | Toy 3D point tasks, molecular energies | **e3nn: MIT [V]** (`e3nn/e3nn`, LBNL copyright) | n/a | No |
| **SE(3)-Transformer** | SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks | 2020 (arXiv 2020-06-18) **[V]** arXiv:2006.10503 | NeurIPS 2020 **[UNVERIFIED]** | Attention over equivariant (irrep) messages; attention logits are invariant scalars, values are steerable, so output is equivariant. | N-body, ScanObjectNN, QM9 | **[NOT FOUND]** in this session | n/a | No |
| **Equiformer** | Equiformer: Equivariant Graph Attention Transformer for 3D Atomistic Graphs | 2022 (arXiv 2022-06-23) **[V]** arXiv:2206.11990 | ICLR 2023 **[UNVERIFIED]** | Transformer blocks over irreps with equivariant layer norm, depth-wise tensor products, and non-linear "equivariant graph attention". | QM9, OC20, MD17 | **MIT [V]** (`atomicarchitects/equiformer`) | n/a | No |
| **EquiformerV2** | EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations | 2023 (arXiv 2023-06-21) **[V]** arXiv:2306.12059 | **ICLR 2024 [V]** (arXiv comment) | Replaces SO(3) convolutions with eSCN SO(2) convolutions (arXiv:2302.03655 **[V]**), adds attention re-normalisation and separable `S²` activations; scales to higher `l` (up to 6-8). | OC20/OC22 catalysis | **MIT [V]** (`atomicarchitects/equiformer_v2`); also in FAIR-Chem (**MIT [V]**) | OC20 checkpoints **[UNVERIFIED]** terms | No |
| **GotenNet** | **GotenNet: Rethinking Efficient 3D Equivariant Graph Neural Networks** | 2025 | **ICLR 2025 [V]** (iclr.cc proceedings page; Sarp Aykent, Tian Xia) | "Geometric Tensor Network": unified structural embedding + **geometry-aware tensor attention** + **hierarchical tensor refinement** that iteratively updates *edge* representations by inner products on high-degree steerable features. Explicitly avoids irreps and Clebsch–Gordan transforms. | **QM9, rMD17, MD22, Molecule3D [V]** — no protein task, no pocket task | **MIT [V]** (`sarpaykent/GotenNet`) | n/a | No in the original paper. **But see GDEGAN (§3): GotenNet has since been applied to pocket detection.** |
| **NequIP** | E(3)-Equivariant GNNs for Data-Efficient and Accurate Interatomic Potentials | 2021 (arXiv 2021-01-08) **[V]** arXiv:2101.03164 | Nat. Commun. 2022 **[UNVERIFIED]** | Tensor-product message passing with `l ≤ 3` irreps for interatomic potentials. | MLIP (energies/forces) | NequIP/Allegro family **MIT [V]** (`mir-group/allegro`) | n/a | No |
| **MACE** | MACE: Higher Order Equivariant Message Passing NNs for Fast and Accurate Force Fields | 2022 (arXiv 2022-06-15) **[V]** arXiv:2206.07697 | **NeurIPS 2022 [V]** (arXiv comment) | Atomic cluster expansion in an equivariant MPNN: many-body messages formed by products of `l`-order features, giving higher body order per layer at fixed depth. | MLIP | **MIT [V]** (`ACEsuit/mace`, `main/LICENSE.md`) | MACE-MP/OFF foundation models **[UNVERIFIED]** terms | No |
| **Allegro** | Learning Local Equivariant Representations for Large-Scale Atomistic Dynamics | 2022 (arXiv 2022-04-11) **[V]** arXiv:2204.05249 | Nat. Commun. 2023 **[UNVERIFIED]** | Strictly *local*, many-body equivariant representation without message passing between atoms (no receptive-field growth), enabling parallel scaling. | MLIP | **MIT [V]** (`mir-group/allegro`) | n/a | No |
| **SEGNN** | Geometric and Physical Quantities Improve E(3) Equivariant Message Passing | 2021 (arXiv 2021-10-06) **[V]** arXiv:2110.02905 | **ICLR 2022 (Spotlight) [V]** (arXiv comment) | Steerable MLPs conditioned on geometric/physical quantities; non-linear steerable node and edge attributes. | N-body, QM9, OC20 | **[UNVERIFIED]** | n/a | No |
| **CGENN** | Clifford Group Equivariant Neural Networks | 2023 (arXiv 2023-05-18) **[V]** arXiv:2305.11141 | **NeurIPS 2023 (Oral) [V]** (arXiv comment) | Geometric (Clifford) algebra: multivectors as features; the Clifford group acts by (twisted) conjugation, and the geometric product gives an equivariant bilinear map, so all grades mix without Clebsch–Gordan machinery. | N-body, convex hulls, top tagging | **[UNVERIFIED]** | n/a | No |
| **Frame Averaging** | Frame Averaging for Invariant and Equivariant Network Design | 2021 (arXiv 2021-10-07) **[V]** arXiv:2110.03336 | ICLR 2022 **[UNVERIFIED]** | Symmetrise an *arbitrary* backbone by averaging over a small, input-dependent frame set (e.g. PCA frames) rather than over the whole group — gives exact equivariance with an unconstrained architecture. | Point clouds, n-body, graphs | **[UNVERIFIED]** | n/a | No |
| **FAENet** | FAENet: Frame Averaging Equivariant GNN for Materials Modeling | 2023 (arXiv 2023-04-28) **[V]** arXiv:2305.05577 | **ICML 2023 [V]** (arXiv comment) | Stochastic frame averaging applied to the *data*, so the GNN itself can be a plain, unconstrained message-passing network — "scalarisation by local frames" taken to its limit. | Materials property prediction (OC20, IS2RE) | **[UNVERIFIED]** | n/a | No |
| **HEGNN** | Are High-Degree Representations Really Unnecessary in Equivariant GNNs? | 2024 (arXiv 2024-10-15) **[V]** arXiv:2410.11443 | NeurIPS 2024 **[UNVERIFIED]** | Shows EGNN-style degree-1 models degenerate on symmetric graphs; adds high-degree steerable features while keeping EGNN's scalarisation style. **Concludes `l ≥ 2` does help.** Evaluated on toy symmetry, molecular and n-body tasks — **no protein pocket task [V]** (checked the PDF). | Expressivity study + molecular/n-body | **MIT [V]** (`GLAD-RUC/HEGNN`) | n/a | No |
| **EST** | Equivariant Spherical Transformer for Efficient Molecular Modeling | 2025 (arXiv 2025-05-29) **[V]** arXiv:2505.23086 | **[UNVERIFIED]** | Addresses limited non-linearity/degree of Clebsch–Gordan tensor-product convolutions. | Molecular modelling | **[UNVERIFIED]** | n/a | No |
| **DualEquiNet** | DualEquiNet: A Dual-Space Hierarchical Equivariant Network for Large Biomolecules | 2025 (arXiv 2025-06-10) **[V]** arXiv:2506.19862 | **[UNVERIFIED]** | Dual Euclidean/spherical space hierarchy for scaling E(3) GNNs to proteins and RNA. | Large biomolecules | **[UNVERIFIED]** | n/a | **[UNVERIFIED]** — not checked for pocket metrics |
| **SWSH** | Spin-Weighted Spherical Harmonics Enable Complete and Scalable E(3)-Equivariant Networks | 2026 (arXiv 2026-07-01) **[V]** arXiv:2607.01408 | **[UNVERIFIED]** | Replaces `O(L⁶)` Clebsch–Gordan tensor products; fixes the Gaunt tensor product's inability to capture antisymmetric paths, restoring completeness at lower cost. The most recent backbone found in this survey. | Atomistic systems | **[UNVERIFIED]** | n/a | No |

**Gap.** Of the 18 backbones above, **not one** reports a binding-site metric in its own paper. Every pocket
number attributed to a backbone (EGNN, SchNet, GAT, GotenNet) comes from a *pocket* paper that re-implemented
it as a baseline (§3, §6) — which means the backbone's hyper-parameters, node features and read-out were
chosen by the competing authors.

---

## 2. Protein-specific geometric models

| Method | Full name | Year | Venue | Mechanism | Trained for | Code licence | Weights | Pocket metrics? |
|---|---|---|---|---|---|---|---|---|
| **PeSTo** | PeSTo: parameter-free geometric deep learning for accurate prediction of protein binding interfaces | 2023 | **Nat. Commun. 14:2175 [V]** (PMC10113261; Krapp, Abriata, Cortés Rodriguez, Dal Peraro) | Geometric transformer on **atoms labelled only by element name**: pairwise distances and relative displacement vectors give translation invariance; the transformer linearly combines scaled local-geometry vectors, preserving **rotation equivariance of vector states**. Four stacked transformer sets with growing neighbourhoods (8/16/32/64 NN). **[V]** | Per-residue interface prediction: protein–protein, –nucleic acid, –ligand, –ion, –lipid | **CC BY-NC-SA 4.0 [V]** (`LBM-EPFL/PeSTo`, `main/LICENSE` = "Attribution-NonCommercial-ShareAlike 4.0 International") — **not MIT** | web server pesto.epfl.ch; Zenodo 10.5281/zenodo.7728869 **[V]** | **No. [V]** ROC-AUC only: protein–protein 0.93, nucleic acid 0.89, ligand 0.86, ion 0.87, lipid 0.77. No DCA/DCC, no COACH420/HOLO4K. |
| **ScanNet** | ScanNet: an interpretable geometric deep learning model for structure-based protein binding site prediction | 2022 | **Nat. Methods [V]** (pmid 35650259; Tubiana, Schneidman-Duhovny, Wolfson). Web-server paper J. Mol. Biol. 2022 **[V]** (pmid 36116806) | Learned "spatio-chemical arrangement of neighbours" filters at atom and amino-acid scale in local frames, plus MSA/PWM input; interpretable learned motifs. | Protein–protein, protein–disordered-protein, protein–antibody binding sites | **Apache-2.0 [V]** (`jertubiana/ScanNet`) | **[UNVERIFIED]** | No ligand-pocket DCA/DCC **[V]** (per-residue PPI metrics only) |
| **ProteinMPNN** (encoder) | Robust deep learning–based protein sequence design using ProteinMPNN | 2022 | Science **[UNVERIFIED]** | Message passing on backbone frames with **invariant** edge features (inter-atomic distances + relative orientations in local frames), i.e. scalarisation, not steerable features. | Fixed-backbone sequence design | **MIT [V]** (`dauparas/ProteinMPNN`) | in-repo weights, MIT **[UNVERIFIED]** scope | No |
| **IPA** (AlphaFold2) | Invariant Point Attention, in "Highly accurate protein structure prediction with AlphaFold" | 2021 | Nature **[UNVERIFIED]** | Attention augmented with query/key/value *points* produced in each residue's local frame and compared after mapping to global frame; the squared-distance term is invariant to a global rigid motion, hence "invariant point". Frames updated per recycling iteration. | Structure prediction | **Apache-2.0 [V]** (`google-deepmind/alphafold`); OpenFold **Apache-2.0 [V]** | AF2 params CC BY 4.0 **[UNVERIFIED]** | No |
| **AlphaFold3** | Accurate structure prediction of biomolecular interactions with AlphaFold 3 | 2024 | **Nature 630 [V]** (citation block in repo README) | Pairformer (triangle updates + pair-biased attention, no IPA) feeding an **atom-level diffusion module** that denoises coordinates directly; equivariance is obtained by *data augmentation over random global rotations*, not by architectural constraint. | Biomolecular complex structure prediction | **Apache-2.0 [V]** (`google-deepmind/alphafold3`) | **Non-commercial only; weights must be received directly from Google and must NOT be published or shared [V]** (`WEIGHTS_TERMS_OF_USE.md`, last modified 2024-11-09; also forbids training structure-prediction models on AF3 output) | No |
| **ESM-3 / structure tokens** | Simulating 500 million years of evolution with a language model | 2024/2025 | Science **[UNVERIFIED]** | Discrete structure tokens from a geometric VQ-VAE encoder; generative masked language model over sequence, structure and function tracks. | Multimodal protein generation/understanding | **MIT [V]** as of the current `evolutionaryscale/esm` repo — `LICENSE.md` reads "License (MIT), Copyright 2026 Chan Zuckerberg Biohub" and the README states "These models are available under the MIT license". | Earlier ESM3 releases used a non-commercial "Cambrian/EvolutionaryScale" licence **[UNVERIFIED]** — if you depend on the licence, re-check the specific checkpoint you download | No |
| **GearNet** | Protein Representation Learning by Geometric Structure Pretraining | 2022 (arXiv 2022-03-11) **[V]** arXiv:2203.06125 | ICLR 2023 **[UNVERIFIED]** | Relational message passing over sequential/radius/K-NN edge types plus an **edge-level** message-passing layer; geometry enters as invariant features. Multiview contrastive + self-prediction pretraining. | Protein function / fold / EC / GO | **MIT [V]** (`DeepGraphLearning/GearNet`); TorchDrug **Apache-2.0 [V]** | **[UNVERIFIED]** | No |
| **CDConv** | Continuous-Discrete Convolution for Geometry-Sequence Modeling in Proteins | 2023 | **ICLR 2023 [V]** (repo README: "the published ICLR23 paper", OpenReview `P5Z-Zl9XJ7`) | Independent learnable weights for *discrete* sequence displacements; direct encoding of *continuous* geometric displacements — decouples sequence and geometry modelling. | Fold / enzyme reaction / GO / EC | **MIT [V]** (`hehefan/Continuous-Discrete-Convolution`) | **[UNVERIFIED]** | No |
| **ProNet** | Learning Hierarchical Protein Representations via Complete 3D Graph Networks | 2022 (arXiv 2022-07-26) **[V]** arXiv:2207.12600 | **ICLR 2023 [V]** (arXiv comment) | Hierarchy of amino-acid / backbone / all-atom levels with provably *complete* invariant geometric representations (distances, angles, torsions, including side-chain torsions). | Fold, reaction, protein–protein identification | DIG: **GPL-3.0 [V]** (`divelab/DIG`) — copyleft, note for release | **[UNVERIFIED]** | No |
| **AtomSurf** | AtomSurf: Surface Representation for Learning on Protein Structures | 2023 (arXiv 2023-09-28) **[V]** arXiv:2309.16519 | **[UNVERIFIED]** | Joint surface-mesh and atom-graph encoders with feature sharing at node/vertex level in every layer. | Atom3D benchmark, binding-site identification, pocket classification **[UNVERIFIED]** numbers | **MIT [V]** (`Vincentx15/atomsurf`) | **[UNVERIFIED]** | Claims binding-site identification results; **numbers [UNVERIFIED]**, protocol not checked |

---

## 3. Pocket and binding-site specific methods

Metric column shows the *paper's own* reported DCA (distance to closest ligand atom, 4 Å) at top-N, unless
noted. See §6 for the cross-paper comparison and the protocol caveats.

| Method | Full name | Year | Venue | Mechanism | Trained for | Licence (code / weights) | Pocket metrics reported |
|---|---|---|---|---|---|---|---|
| **LIGSITE** | LIGSITE: automatic and efficient detection of potential small molecule-binding sites | 1997 | **J. Mol. Graph. Model. [V]** (pmid 9704298) | Grid "protein–solvent–protein" event counting along axes and cubic diagonals → buriedness. | — (geometric) | — | No (pre-dates DCA/DCC) |
| **fpocket** | Fpocket: an open source platform for ligand pocket detection | 2009 | **BMC Bioinformatics [V]** (pmid 19486540) | Voronoi alpha spheres, clustered into pockets, scored by geometric/physico-chemical descriptors. | — (geometric) | **MIT [V]** (within `rdk/p2rank` LICENSE.txt chain; fpocket's own repo **[UNVERIFIED]**) | Not in its own paper. Re-measured many times — see §6, where fpocket's COACH420 DCA top-N appears as **56.4 %** (P2Rank paper), **35.09 %** (DeepPocket paper) and **22.8 %** (EquiPocket paper). |
| **P2Rank** | P2Rank: machine learning based tool for rapid and accurate prediction of ligand binding sites from protein structure | 2018 | **J. Cheminform. [V]** (PMC6091426, CC BY) | Random forest scoring **solvent-accessible-surface points** from aggregated local neighbourhood features; points clustered into ranked pockets. Rotation-invariant by feature construction. | Ligandability of SAS points, trained on CHEN11, dev on JOINED | **MIT [V]** (`rdk/p2rank`, `master/LICENSE.txt`) | **[V] Table 3:** COACH420 DCA top-N **72.0**, top-(n+2) **78.3**; HOLO4K **68.6 / 74.0**. Baselines in the same table: fpocket 56.4/68.9 and 52.4/63.1; fpocket+PRANK 63.6/76.5 and 62.0/71.0; SiteHound 53.0/69.3 and 50.1/62.1; MetaPocket 2.0 63.4/74.6 and 57.9/68.6; DeepSite 56.4/63.4 and 45.6/48.2. |
| **DeepSite** | DeepSite: protein-binding site predictor using 3D-convolutional neural networks | 2017 | **Bioinformatics [V]** (pmid 28575181) | Voxelised protein channels → 3D CNN scoring sub-grids; **not rotation-equivariant**. | Binding-site voxel classification | **[UNVERIFIED]** | Numbers only via others: COACH420 DCA top-N 56.4 (P2Rank), 57.5 (DeepSurf), 53.07 (DeepPocket), 0.564 (EquiPocket/VN-EGNN) |
| **Kalasanty** | Improving detection of protein-ligand binding sites with 3D segmentation | 2020 | **Sci. Rep. [V]** (PMC7081267) | 3D U-Net *segmentation* of a fixed-size voxel grid; pocket = connected positive voxels. Rotation-sensitive. | Pocket voxel segmentation on scPDB | **[UNVERIFIED]** | Via others: COACH420 DCA top-N **68.0 / 70.4** (DeepSurf paper), **63.51 / 65.18** (DeepPocket paper), **0.636** (EquiPocket); HOLO4K **32.1 / 32.3** (DeepSurf), **61.21 / 62.63** (DeepPocket), **0.515** (EquiPocket). The 32 % vs 61 % spread on the same method and same dataset is the clearest single illustration of protocol incomparability. |
| **DeepPocket** | DeepPocket: Ligand Binding Site Detection and Segmentation using 3D CNNs | 2022 | **J. Chem. Inf. Model. [V]** (pmid 34374539; PDF via IIIT CDN) | **Hybrid**: fpocket generates candidates; a 3D CNN *re-scores* them, a second CNN segments the chosen cavity. | Candidate re-scoring + segmentation on scPDB; **Mlig subsets used for evaluation [V]** | **MIT [V]** (`devalab/DeepPocket`) | **[V] Table 1 (DCA):** COACH420 **67.96 / 79.94**; HOLO4K **73.36 / 82.97**; SC6K **64.58 / 83.01**. Same table: fpocket 35.09/51.25, 36.34/51.53; DeepSite 53.07/53.07, 51.65/51.67; Kalasanty 63.51/65.18, 61.21/62.63; P2Rank 68.24/75.48, 70.6/80.05. Also **DCC** success 81.31 % on COACH420, 80.24 % on SC6K. Reports fpocket's recall ceiling: pocket centre within 4 Å of a ligand atom for **80.78 %** of COACH420 ligands, **87.62 %** HOLO4K, **96.4 %** scPDB **[V]**. |
| **DeepSurf** | DeepSurf: a surface-based deep learning approach for the prediction of ligand binding sites on proteins | 2021 | **Bioinformatics 37(12):1681 [V]** (pmid 33471069) | Local 3D grids placed **on surface points** with a local coordinate system, scored by a ResNet; avoids the fixed global box. | Surface-point ligandability on scPDB | **[UNVERIFIED]** (repo `stemylonas/DeepSurf`; no LICENSE found at the paths checked) | **[V] Table 2 (DCA):** COACH420 ResNet-18 **72.1 / 73.3**, Bot-LDS-ResNet-18 71.7/72.7; HOLO4K **50.1 / 50.6** and 50.4/50.8. Homology control: training targets with >90 % sequence similarity to any test target removed **[V]**. |
| **PUResNet** | PUResNet: prediction of protein-ligand binding sites using deep residual neural network | 2021 | **J. Cheminform. [V]** (PMC8424938) | Deep residual 3D-CNN segmentation. | Pocket voxel segmentation | **[UNVERIFIED]** | Not in this survey's primary reads; LIGYSIS reports top-N+2 recall **41.1 %** and predicts a single pocket in ~90 % of cases **[V]** |
| **PointSite** | PointSite: A Point Cloud Segmentation Tool for Identification of Protein Ligand Binding Atoms | 2022 | **J. Chem. Inf. Model. [V]** (pmid 35621730) | Sparse-convolution point-cloud segmentation over protein **atoms** (not cavity points). | Binding-atom segmentation | **[UNVERIFIED]** | **[UNVERIFIED]** — not read |
| **RecurPocket** | RecurPocket: recurrent Lmser network with gating mechanism for protein binding site detection | 2022 | **IEEE BIBM 2022 [UNVERIFIED]** (Li, Cao, Tu, Xu) | Recurrent Lmser (bidirectional autoencoder-like) 3D CNN with gating, over voxelised protein. | Pocket detection | **[NOT FOUND]** | Via EquiPocket/GDEGAN tables: COACH420 DCC 0.354 / DCA 0.593; HOLO4K 0.277 / 0.616; PDBbind2020 0.492 / 0.663 **[V]** (as reported by EquiPocket) |
| **GrASP** | Graph Attention Site Prediction: Identifying Druggable Binding Sites Using Graph Neural Networks with Attention | 2024 (bioRxiv 2023) | **J. Chem. Inf. Model. [V]** (pmid 37546775 / 10.1021/acs.jcim.3c01698; full text via PMC10402091 JATS) | GATv2 semantic segmentation of **protein surface heavy atoms** (edges ≤ 5 Å, inverse-distance + bond-order edge features), ResNet + jumping-knowledge skips, Noisy-Nodes regularisation; average-linkage clustering (threshold 15 Å) over atoms scoring > 0.3 for instance segmentation. **Rotation-invariant, not equivariant.** | Atom-level ligandability on a **modified sc-PDB** (26,196 sites / 16,889 structures, adding ~9,000 ligands from symmetric multimers and aligned structures) | **MIT [V]** (`tiwarylab/GrASP`). Evaluation lists on Zenodo CC-BY-4.0 **[UNVERIFIED]** | **[V] (from the JATS tables):** COACH420(Mlig+) DCA recall top-N **77.5**, top-N+2 **80.6**, precision top-3 71.2, top-5 71.0, all-sites 71.0; P2Rank on the same set 74.9 / 79.4 / 41.0 / 33.2 / 28.3. HOLO4K(Mlig+) GrASP **81.3 / 84.3 / 72.8 / 71.6 / 71.4**; P2Rank **81.2 / 86.5 / 45.9 / 35.4 / 25.5**. sc-PDB 10-fold CV: 85.3 / 91.4 / 69.7 / 66.4 / 65.0. |
| **IF-SitePred** | Learnt representations of proteins can be used for accurate prediction of small molecule binding sites … | 2024 | **J. Cheminform. [V]** (Carbery, Buttenschoen, Skyner, von Delft, Deane; PMC10941399) | **ESM-IF1** inverse-folding per-residue embeddings → LightGBM ligandability → point-cloud clustering of predicted points into site centres. Backbone-only features, robust to predicted structures. | Residue ligandability | **BSD-3-Clause [V]** (`annacarbery/binding-sites`) | 93 % top-3 success on paired PDB/AF2 structures **[UNVERIFIED]**; LIGYSIS measures top-N+2 recall **25.7 %** (default) → **39 %** after redundancy removal + re-scoring **[V]** |
| **EquiPocket** | EquiPocket: an E(3)-Equivariant Geometric Graph Neural Network for Ligand Binding Site Prediction | 2024 | **ICML 2024, PMLR v235:60021–60039 [V]**. ⚠ **The arXiv version (2302.12177) was WITHDRAWN by the authors, v4 2026-08-17 / v5 2026-08-18, stating "the technical elaboration and experimental design in the current manuscript require substantial restructuring and revision" [V]** (arXiv abs page read directly). | Three modules: (i) local geometric extractor on **MSMS solvent-accessible-surface probe points** (probe radius 1.5 Å), (ii) global chemical/spatial GAT+EGNN over atoms, (iii) **Surface-EGNN** equivariant message passing over surface points; plus a dense (sigmoid) attention output layer for protein-size shift and an auxiliary **relative-direction (cosine) loss**. | Atom-level: protein atoms within 4 Å of any ligand atom are positive; scPDB 2017 (5,372 structures after UniProt clustering), mean-shift clustering to centres. **Mlig subsets for evaluation [V]** | **[NOT FOUND]** — no code repository or licence located; the arXiv record now says "No license for this version due to withdrawn" **[V]** | **[V] Table 1:** COACH420 DCC 0.423 / DCA 0.656; HOLO4K 0.337 / 0.662; PDBbind2020 0.545 / 0.721; failure rate 0.051, 1.70 M params. ⚠ **Protocol oddity [V]:** because EquiPocket labels *protein atoms*, its predicted centre is displaced 4 Å outward along the atom→surface direction (Eq. 11) before scoring — a hand-tuned correction with no counterpart in other methods. Its P2Rank baseline (COACH420 DCA 0.628, HOLO4K 0.621) is far below both P2Rank's own published numbers and VN-EGNN's re-run; the appendix shows their **retrained** P2Rank collapsed above 3,000 training samples (DCA 0.177/0.130/0.256), so they report a 792-sample subset. |
| **VN-EGNN** | VN-EGNN: E(3)- and SE(3)-Equivariant Graph Neural Networks with Virtual Nodes Enhance Protein Binding Site Identification | 2025 (arXiv 2024-04-10; NeurIPS 2023 MLSB workshop version) | **J. Cheminform. [V]** (10.1186/s13321-025-01127-9, PMC12837241, pmid 41398608, first published 2025-12-15; Sestak, Schneckenreiter, Brandstetter, Hochreiter, Mayr, Klambauer) | Residue nodes at Cα with frozen **ESM-2** features; **K virtual nodes initialised on a Fibonacci-grid sphere** of radius = centre-to-farthest-atom, connected to all residues. Three-phase **heterogeneous** message passing per layer (atom→atom, atom→virtual, virtual→atom); **virtual-node positions are updated at every one of the 5 layers [V]**. Loss = α·Dice (segmentation) + minimum-distance loss to observed site centres; per-virtual-node self-confidence for ranking. **Degree-1 only — no spherical harmonics, no `l ≥ 2` [V]**. 11 virtual nodes by default **[V]**. 1.20 M params. | Residue segmentation + site-centre regression; trained on **scPDB** | **Code MIT [V]** (`ml-jku/vnegnn`, `master/LICENSE`). **Article CC BY-NC-ND [V]** (Europe PMC licence field, confirmed by the PMC page). Weights in-repo; data Zenodo 17365855; demo HF `ml-jku/vnegnn` **[V]**. Weights licence **[UNVERIFIED]** | **[V] Table 1 (4 Å, top-M where M = number of known sites):** COACH420 DCC **0.605(0.009)** / DCA **0.750(0.008)**; HOLO4K DCC **0.532(0.021)** / DCA **0.659(0.026)**; PDBbind2020 DCC **0.669(0.015)** / DCA **0.820(0.010)**. **Critically, in VN-EGNN's own table P2Rank has the higher DCA on HOLO4K (0.787) and PDBbind2020 (0.826), and DeepPocket 0.734 on HOLO4K — VN-EGNN wins DCA only on COACH420 [V].** HOLO4K was run **per chain**, predictions then merged **[V]**. Ablation **[V]**: EGNN baseline COACH420 DCC 0.156 → +virtual nodes & heterogeneous MP 0.503 → +ESM-2 0.605. |
| **GENEOnet** | GENEOnet: a breakthrough in protein binding pocket detection using group equivariant non-expansive operators | 2025 | **Sci. Rep. 15:34597 [V]** (PMC12494700; Bocchi, Frosini, Micheletti et al.; open access CC BY 4.0) | Convex combination of **group equivariant non-expansive operators** (symmetric-kernel convolutions) on a voxel grid of the protein's empty space. **Equivariant to 3D translations and rotations; non-expansive (Lipschitz-stable). Only 17 learnable parameters [V].** Trained in ~6 min on 200 PDBbind complexes by maximising a volumetric Jaccard coefficient. | Pocket voxel occupancy | **[NOT FOUND]** for code (no LICENSE at `giovannibocchi/GENEOnet`); paper **CC BY 4.0 [V]**; pre-trained model via geneonet.exscalate.eu **[V]** | **[V]** On their own BINDTEST (6,854 PDBbind proteins): success@1 **0.764** vs P2Rank 0.702; success@3 **0.929**. BANK set (28,382 proteins) top-3 **0.875**. **Does not report COACH420 or HOLO4K.** |
| **GDEGAN** | GDEGAN: Gaussian Dynamic Equivariant Graph Attention Network for Ligand Binding Site Prediction | **2026** | **arXiv:2603.19817, submitted 2026-03-20 — PREPRINT, not peer-reviewed [V]** (Animesh, Bhowmick, Mitra) | Built **on GotenNet [V]**. Replaces dot-product attention with "Gaussian dynamic attention": per-node learned neighbourhood **variance** σ²ᵢ as an adaptive kernel bandwidth with learnable per-head temperatures. **`L_max = 2` steerable features initialised from spherical harmonics (`l=1` dipole, `l=2` quadrupole) [V]**; 4 layers, hidden 128, 8 heads, 32 neighbours; ESM-2 node features; an added chirality term makes it SE(3) rather than E(3); auxiliary directional loss (ADL) supervising `l=1` features towards ground-truth ligand directions. 1.90 M params. | Residue-level site prediction; scPDB-style training | **[NOT FOUND]** — no code URL or licence stated in the PDF | **[V] Table 1:** COACH420 DCC **0.580(0.008)** / DCA **0.707(0.009)**; HOLO4K DCC **0.560(0.013)** / DCA **0.788(0.011)**; PDBbind2020 DCC **0.675(0.010)** / DCA **0.826(0.011)**; failure 0.032. **GotenNet alone** (2.20 M params, Table 1): COACH420 0.464/0.624, HOLO4K 0.454/0.691, PDBbind2020 0.553/0.705. ⚠ **Internal inconsistency [V]:** the identical GotenNet row in Table 2 gives COACH420 DCC 0.454 and HOLO4K DCC 0.464 — the two values are **swapped between its own Tables 1 and 2** (lines verified in the PDF text). ⚠ **Its own ablation undercuts its headline claim [V]:** `GotenNet(full)` (SE(3) + ADL + ESM, i.e. the GotenNet backbone with all of GDEGAN's other additions but dot-product attention) reaches COACH420 DCA **0.703** against GDEGAN(full) **0.707** — the Gaussian dynamic attention that names the paper is worth **0.4 points of DCA on COACH420** and 3.9 points on HOLO4K (0.749 → 0.788). **It copies its fpocket/P2Rank/DeepSurf/Kalasanty/EquiPocket baselines verbatim from the EquiPocket paper [V], i.e. from the lowest-scoring P2Rank measurement in the literature, and it never compares against VN-EGNN, GrASP or DeepPocket [V]** (grep for "VN-EGNN"/"virtual node" in the PDF returns nothing). |
| **YuelPocket** | Unified protein–small molecule graph neural networks for binding site prediction | **2026** | **PNAS 123(10):e2524913123 [V]** (Jian Wang, Nikolay V. Dokholyan; PMC12974528) | Heterogeneous graph: protein sub-graph (Cα + one side-chain pseudo-atom per residue), **ligand** sub-graph, and a **virtual joint node** as a hub reducing protein×ligand coupling from quadratic to linear. 16 layers, hidden 128. **Not explicitly equivariant [V]**. | Multi-task: InfoNCE global protein–ligand pairing (1 true vs 50 decoys) + residue-level pocket (weighted BCE + Dice), ground truth = residues within 4 Å of any ligand atom. **PLINDER v2024-06**: 309,140 train / 1,036 test with 30 % sequence, 30 % PLI and 50 % ligand-Tanimoto split constraints **[V]** | `hust220/yuel_pocket`; licence **[UNVERIFIED]**; data Zenodo 18065818 / 16921425 **[V]** | **[V]** PLINDER test DCA top-1 ≈ 62 % vs P2Rank ≈ 55 %; DCC top-1 ≈ 45 % vs 40 %. Holo4k (340 systems after filtering) residue-level top-1 ≈ 55 %, top-3 ≈ 75 %. **Explicitly EXCLUDED COACH420: "the vast majority of COACH420 systems were already present in the PLINDER training data" [V]** — a direct, published leakage finding. |
| **ProMoSite / ProMoNet** | Sequence-based drug-target binding site pre-training enables cryptic pocket detection and improves binding affinity and kinetics prediction | **2026** | **J. Cheminform. 18:93 [V]** (Shuo Zhang, Li Xie, Daniel Tiourine, Lei Xie; PMC13371367) | **Sequence-only**, no 3D input: frozen ESM-2 protein + Uni-Mol ligand embeddings with trainable interaction modules; protein contact maps and ligand distance maps used as triangle-inequality constraints. | Binding-site annotation pre-training, then affinity/kinetics fine-tuning | `zetayue/ProMoNet`; **[NOT FOUND]** licence file | **[V]** COACH420 top-N success **68.30 ± 0.99 %**, HOLO4K **73.46 ± 0.65 %** (evaluated in 3D despite sequence-only input); residue classification COACH420 ROC-AUC 0.933 ± 0.001 / PR-AUC 0.530 ± 0.003, HOLO4K 0.926 ± 0.001 / 0.516 ± 0.009. Claims to beat all 3D baselines, but the baseline numbers are not tabulated in the text read. |
| **ProtGeoNet-Pocket** | ProtGeoNet-Pocket: A Binding Site Prediction Approach Integrating Sequence, Geometry, and Graph Structure | 2025 | **J. Chem. Inf. Model. 65(19):10736–10753 [V]** (pmid 40960030) | PointNet geometric features on residue coordinates + attention, fused with sequence and graph-edge features, then a **Graph Isomorphism Network**. Invariant, not equivariant. | Residue-level site prediction on scPDB | **[UNVERIFIED]** | F1 **72.87 %** on scPDB; evaluated on COACH420, HOLO4K, SC6K, PDBbind, ApoHolo. **DCA/DCC numbers [UNVERIFIED]** — paywalled, not read |
| **AstraBIND** | AstraBIND: Graph Attention Network for Predicting Ligand Binding Sites | 2025 | **bioRxiv 10.1101/2025.11.10.687555, 2025-11-11 [V]** (Goteti, Vasilyeva, Bozkurt; Orbion GmbH) | GATv2, 0.9 M params, edge attributes; three heads — binding-site (node binary), pocket (node multiclass), hierarchical ligand-class (graph level) with soft gating by binding probability. | Joint: binding residues + pocket assignment + 16 ligand categories, >250,000 curated complexes | Models behind `orbion.life`; **[NOT FOUND]** open licence | Weighted macro-F1 **0.47** across 16 ligand classes (nucleotides 0.79, porphyrins 0.74, cofactors 0.73) **[V]**. **No COACH420/HOLO4K comparison [V]** |
| **PocketMiner** | Predicting locations of cryptic pockets from single protein structures using the PocketMiner graph neural network | 2023 | **Nat. Commun. 14:1177 [V]** | **GVP-GNN** over backbone geometry, trained on labels derived from *molecular dynamics* (pocket-opening events), not from holo structures. | Per-residue probability that a cryptic pocket *will open* | **MIT [V]** (`bowman-lab/ae-pocketminer`) | **No DCA/DCC.** ROC-AUC **0.87** on 39 experimentally confirmed cryptic pockets **[V]**. EquiPocket's own appendix notes PocketMiner solves a different task (classification of opening, not localisation) **[V]** |
| **CryptoBench** | CryptoBench: cryptic protein–ligand binding sites dataset and benchmark | 2025 | **Bioinformatics 41(1):btae745 [V]** (Škrhák, Novotný, Feidakis, Krivák, Hoksza) | Benchmark, not a model. Apo–holo pairs from AHoJ-DB, resolution ≥ 2.5 Å, TM-score ≥ 0.5, centre distance ≤ 4 Å, R_g change ≤ 20 %, pocket RMSD > 2 Å, ligands ≥ 5 atoms; 40 % sequence clustering, 80:20 split (885 train in 4 folds / 222 test) plus a 10 % identity clustering to prevent leakage. 1,107 apo structures, 1,361 cryptic pockets. **[V]** | — | **MIT [V]** (`skrhakv/CryptoBench`); dataset on OSF `pz4a9` | **[V] Residue-level only, no DCA/DCC:** pLM-NN (ESM embeddings + NN) AUC 0.86 / AUPRC 0.36 / MCC 0.39 (CB-full), 0.88 / 0.43 / 0.44 (CB-PM); **PocketMiner** 0.76 / 0.19 / 0.22; **P2Rank** 0.81 / 0.21 / 0.27. The sequence-based baseline beats both structure-based methods. |
| **LIGYSIS / LBS-comparison** | Comparative evaluation of methods for the prediction of protein–ligand binding sites | 2024 | **J. Cheminform. [V]** (PMC11552181, pmid 39529176, **CC BY [V]**; Utgés & Barton) | Benchmark of 13 methods / 28 variants on the **human subset of LIGYSIS**: 3,448 proteins, 8,244 sites, aggregating *all* biologically relevant (BioLiP-defined) protein–ligand interfaces across **biological units** of all structures of a protein. | — | Paper CC BY 4.0 **[V]**; `bartongroup/LBS-comparison` **[NOT FOUND]** licence file | **[V] Top-N+2 recall at DCC ≤ 12 Å** (note: DCC, 12 Å, **not** DCA 4 Å): fpocket_PRANK **60.4**, DeepPocket_RESC **58.1**, P2Rank_CONS ≈ 54, P2Rank ≈ 52, GrASP **≈ 50**, DeepPocket_SEG-NR / Ligsite+_AA / PocketFinder+_AA ≈ 49, Surfnet+_AA ≈ 47, fpocket ≈ 47, **VN-EGNN_NR ≈ 46**, PUResNet_PRANK ≈ 41, **VN-EGNN 40.9**, IF-SitePred_RESC-NR 39, **IF-SitePred 25.7**. Maximum recall over all predictions: fpocket / fpocket_PRANK / DeepPocket_RESC ≈ 90 %; everything else ≈ 50–60 %; PUResNet 41 %. |
| **Sesame** | Sesame: Opening the door to protein pockets | 2025 | **ICLR 2025 GEM workshop [V]** (arXiv:2509.05302; Miñán, Perez-Lopez, Iglesias, Ciudad, Molina) | Generative model of apo→holo conformational change, ligand-agnostic, to make apo pockets dockable. **Pocket *conditioning*, not detection.** | Apo→holo geometry generation | **[UNVERIFIED]** | No DCA/DCC |
| **SE(3)-invariant CNN booster** | Boosting CNNs' Protein Binding Site Prediction Capacity Using SE(3)-invariant transformers, Transfer Learning and Homology-based Augmentation | 2023 | **arXiv:2303.08818, preprint [V]** (Lee, Byun, Shin) | 3D-CNN local features → **SE(3)-invariant geometric self-attention** over residues (query/key split into standard and geometric parts, built from residue frames `(R_i, t_i)`), plus BRI→BSD transfer learning and homology-based augmentation. | Binding-site detection (BSD) + binding-residue identification (BRI) on scPDB 2017 | **[UNVERIFIED]** (arXiv CC BY 4.0 for the paper) | **[V] Table 1, F1 success rate for detection** (their own metric, 5-fold): scPDB held-out / COACH420 / HOLO4K / CHEN-holo / CHEN-apo — DeepSurf 62.4/43.6/59.7/24.5/22.3; Kalasanty 70.0/50.8/44.9/28.5/27.1; DeepPocket 67.9/55.7/72.2/42.4/34.5; **Ours 70.1/59.1/77.0/41.2/36.5**. Crucial ablations below. |

### Why EquiPocket's withdrawal matters

EquiPocket (ICML 2024) is the anchor baseline of the entire equivariant-pocket literature: **VN-EGNN** and
**GDEGAN** both cite it as the previous state of the art, and GDEGAN copies its whole baseline table. Its
arXiv version was withdrawn by the authors in August 2026 because "the technical elaboration and experimental
design … require substantial restructuring and revision" **[V]**. The ICML/PMLR version remains published
and citable, but any claim of the form "we beat the previous equivariant state of the art" that rests on
EquiPocket's numbers now rests on a paper its own authors have asked to be treated as provisional. This is
independently corroborated by the internal evidence in the appendix: the 4 Å outward projection of predicted
centres (Eq. 11), and a retrained-P2Rank baseline that collapses to DCA 0.177 above 3,000 training samples.

---

## 4. Hotspot / interaction-field prediction

| Method | Full name | Year | Venue | Mechanism | Trained/computed for | Licence | Pocket metrics? |
|---|---|---|---|---|---|---|---|
| **SILCS** | Computational fragment-based binding site identification by ligand competitive saturation | 2009 | **PLoS Comput. Biol. [V]** (PMC2700966; Guvench & MacKerell) | **Physics-based**, not learned: all-atom MD of the protein in an aqueous solution of small probe molecules (benzene, propane, water); occupancy histograms → per-probe-type "FragMaps" (grid free-energy fields). | — (simulation) | Commercial (SilcsBio) **[UNVERIFIED]** | No |
| **FTMap** | Fragment-based identification of druggable 'hot spots' of proteins using Fourier domain correlation techniques | 2009 | **Bioinformatics [V]** (PMC2647826; Brenke et al.) | **Physics-based**: FFT rigid-body docking of 16 small organic probes over the whole surface, energy minimisation, clustering; consensus sites = hot spots. FTFlex (2013) adds side-chain flexibility **[V]** (PMC3634182). | — | Academic web server **[UNVERIFIED]** | No |
| **Fragment Hotspot Maps** | Identifying Interactions that Determine Fragment Binding at Protein Hotspots | 2016 | **J. Med. Chem. 59:4314–4325 [V]** (pmid 27043011; Radoux et al.) | **Knowledge-based, not learned**: CSD-mined interaction propensities applied to buriedness-weighted grid points; outputs **three maps — donor, acceptor, apolar** — i.e. an explicitly per-interaction-type field. | — | CCDC Hotspots API, commercial/academic **[UNVERIFIED]** | No |
| **AutoSite** | AutoSite: an automated approach for pseudo-ligands prediction — from ligand-binding sites identification to predicting key ligand atoms | 2016 | **Bioinformatics [V]** (PMC5048065; Ravindranath & Sanner) | Grid "fill points" typed by AutoDock affinity maps (C, OA, HD); clustered into pockets; **outputs a typed pseudo-ligand, i.e. per-atom-type points inside the pocket** — the closest classical analogue of a learned per-atom-class hotspot field. | — (force-field grids) | ADFR suite **[UNVERIFIED]** | No DCA top-N in the form used by P2Rank-lineage papers **[UNVERIFIED]** |
| **DeepFrag** | DeepFrag: a deep convolutional neural network for fragment-based lead optimization | 2021 | **Chem. Sci. [V]** (PMC8208308; Green & Durrant). Browser app: J. Chem. Inf. Model. 2021 **[V]** (PMC8243318) | 3D CNN on a voxelised receptor+parent-ligand context predicting a *fragment fingerprint* at a chosen growth vector. **Not equivariant; not a field; conditioned on an existing ligand.** | Fragment addition | **Apache-2.0 [V]** (`durrantlab/deepfrag`) | No |
| **PharmacoNet** | PharmacoNet: Accelerating Large-Scale Virtual Screening by Deep Pharmacophore Modeling | 2023 preprint → **Chemical Science 2025 [V]** | **3D CNN**: Feature Pyramid Network over a 3D Swin-Transformer-V2 encoder, doing **instance segmentation** on a 64³ grid; each predicted pharmacophore gets a centre and radius. **Labels come from non-covalent interactions actually detected by PLIP in PDBBind v2020 complexes (19,443), not from proximity [V].** Seven types: hydrophobic carbon, aromatic ring, H-bond donor, H-bond acceptor, halogen, cation, anion **[V]**. | Per-type pharmacophore/hotspot instances in a pocket | Code `SeonghwanSeo/PharmacoNet`, **CC BY 4.0 [V]** | No. Evaluated on DUD-E and DEKOIS2.0 by enrichment factor and AUROC **[V]** |
| **Equivariant Scalar Fields** | Equivariant Scalar Fields for Molecular Docking with Fast Fourier Transforms | 2023 (arXiv 2023-12-07) | arXiv:2312.04323; MLSB 2023 **[V]** | Scoring function = **cross-correlation of multi-channel ligand and protein scalar fields**, each field a sum of per-atom / per-Cα contributions expanded in Gaussian RBFs × real spherical harmonics, with coefficients produced by **equivariant GNNs**. Rigid-body optimisation by FFT. **A genuinely learned, equivariant, multi-channel spatial field over a protein** — but the channels are docking-score components, not interaction-type hotspot labels. | Docking pose scoring (vs Vina, Gnina) | **MIT [V]** (`bjing2016/scalar-fields`) | No |
| **DeepDFT** | Equivariant graph neural networks for fast electron density estimation of molecules, liquids, and solids | 2022 | **npj Comput. Mater. 8:183 [V]** (Jørgensen & Bhowmik) | **Probe nodes inserted at arbitrary query grid points**, receiving (but not sending) equivariant messages from atoms, to predict a scalar field value at each point. Different domain, but the exact mechanism of "equivariant message passing onto grid probes". | Electron density on grids | **[UNVERIFIED]** | No (materials) |

---

## 5. Peptide-binding-site predictors

| Method | Full name | Year | Venue | Mechanism | Trained for | Licence | Pocket metrics? |
|---|---|---|---|---|---|---|---|
| **PepBind** | PepBind: a comprehensive database and computational tool for analysis of protein–peptide interactions | 2013 | **Genomics Proteomics Bioinformatics [V]** (PMC4357787) | Database + consensus predictor; mainly used as a benchmark source by later work. | Peptide-binding residues | **[UNVERIFIED]** | No |
| **InterPep / InterPep2** | InterPep2: global peptide–protein docking using interaction surface templates | 2020 | **Bioinformatics [V]** (PMC7178396; Johansson-Åkhe et al.) | **Template/structural-homology** based: finds interaction-surface templates by structural alignment, scores with a random forest, then builds the full complex. Not a geometric neural network. | Peptide-binding site + complex | **[UNVERIFIED]** | No (uses its own success criteria) |
| **PepNN** | PepNN: a deep attention model for the identification of peptide binding sites | 2022 | **Commun. Biol. [V]** (PMC9135736; Abdin et al.) | Reciprocal multi-head attention between a protein representation and a peptide representation; **PepNN-Struct** uses graph attention on structure, **PepNN-Seq** sequence only; pretrained on protein–protein complexes, fine-tuned on protein–peptide, with contextual language-model features. | Per-residue peptide-binding probability | **[UNVERIFIED]** | ROC-AUC / MCC per residue; no DCA/DCC |
| **PepCA** | PepCA: unveiling protein–peptide interaction sites with a multi-input neural network model | 2024 | **[UNVERIFIED]** (ResearchGate record only) | Multi-input (sequence + embeddings) cross-attention. | Peptide-binding residues | **[UNVERIFIED]** | No |
| **PepBCL** | Prediction of Protein-Peptide Binding Sites Using PepBCL | 2025 | **[V]** pmid 40601263 (journal **[UNVERIFIED]**) | End-to-end, feature-design-free: pretrained protein language model representations + contrastive learning for the class-imbalanced residue task. | Peptide-binding residues | **[UNVERIFIED]** | No |
| **PepPCBench** | PepPCBench: a Comprehensive Benchmarking Framework for Protein–Peptide Complex Structure Prediction | 2025/2026 | **J. Chem. Inf. Model. [V]** (10.1021/acs.jcim.5c01084) | Benchmark of AlphaFold3-class folding networks on protein–peptide complexes. Reported finding: **binding-site identification is the main remaining failure mode [UNVERIFIED]**. | — | **[UNVERIFIED]** | No |
| **PepLLM** | PepLLM: ESM-Guided Llama for Structured Protein-Peptide Binding Interface Analysis | 2026 | **arXiv:2608.21367, preprint [V]** (listing only) | LLM conditioned on ESM representations for interface analysis. | Peptide interfaces | **[UNVERIFIED]** | **[UNVERIFIED]** |
| **"ScanNet-peptide"** | — | — | — | **[NOT FOUND].** ScanNet's published variants cover protein–protein, protein–**disordered protein** and protein–**antibody** interfaces **[V]**. No peptide-specific ScanNet model was found. The protein–disordered-protein head is the nearest thing, and EquiPocket's appendix explicitly says NodeCoder/ScanNet-class interface models are unsuitable for ligand-site prediction **[V]**. | | | | | |

**Summary for peptides.** The entire peptide-binding-site field is still **per-residue classification with
sequence/attention models**. No equivariant geometric network, no `l ≥ 2` features, no site-centre
localisation metric (DCA/DCC) and no probe/virtual-node method was found for peptide sites. This is the
largest open gap identified in this survey.

---

## 6. The same metric across papers: why "state of the art" is ill-defined

### COACH420, DCA ≤ 4 Å, top-N (= number of annotated sites)

| Reported value | Method | Reported by | Protocol differences that matter |
|---|---|---|---|
| **77.5** | GrASP | GrASP 2024 **[V]** | **COACH420(Mlig+): only 256 single-chain systems, 315 ligands** — both MOAD relevance *and* geometric criteria applied |
| 74.9 | P2Rank | GrASP 2024 **[V]** | same Mlig+ set |
| **75.0** | VN-EGNN | VN-EGNN 2025 **[V]** | mlig; top-M by self-confidence; per-chain for HOLO4K |
| 72.8 | P2Rank | VN-EGNN 2025 **[V]** | VN-EGNN's own re-run |
| 72.1 | DeepSurf (ResNet-18) | DeepSurf 2021 **[V]** | >90 % sequence-similarity training filter |
| 72.0 | P2Rank | P2Rank 2018 **[V]** | mlig; trained on CHEN11 |
| **70.7** | GDEGAN | GDEGAN 2026 preprint **[V]** | baselines copied from EquiPocket |
| 68.3 | ProMoSite | ProMoNet 2026 **[V]** | sequence-only input, 3D read-out |
| 68.24 | P2Rank | DeepPocket 2022 **[V]** | DeepPocket's re-run |
| 68.0 | Kalasanty | DeepSurf 2021 **[V]** | |
| 67.96 | DeepPocket | DeepPocket 2022 **[V]** | mlig |
| 65.6 | — (GotenNet alone, DCA 0.624) | GDEGAN 2026 **[V]** | |
| 65.6 | EquiPocket (DCA 0.656) | EquiPocket 2024 **[V]** | **4 Å outward projection of predicted centres (Eq. 11)** |
| 63.51 | Kalasanty | DeepPocket 2022 **[V]** | |
| 62.8 | P2Rank | EquiPocket 2024 **[V]** | lowest P2Rank measurement in the literature |
| 56.4 | fpocket | P2Rank 2018 **[V]** | |
| 44.4 | fpocket | EquiPocket 2024 **[V]** | |
| 35.09 | fpocket | DeepPocket 2022 **[V]** | |

**Spread for a single fixed, deterministic method (fpocket) on a single named dataset: 22.8 % to 56.4 %.**

### HOLO4K, DCA ≤ 4 Å, top-N

| Reported value | Method | Reported by | Note |
|---|---|---|---|
| **81.3** | GrASP | GrASP 2024 **[V]** | HOLO4K(Mlig+), **4,514 systems after splitting into single chains and interface-connected subsystems** |
| **81.2** | P2Rank | GrASP 2024 **[V]** | same set |
| **78.8** | GDEGAN | GDEGAN 2026 preprint **[V]** | |
| **78.7** | P2Rank | VN-EGNN 2025 **[V]** | — i.e. in VN-EGNN's own table P2Rank beats VN-EGNN on HOLO4K DCA |
| 73.46 | ProMoSite | ProMoNet 2026 **[V]** | |
| 73.4 | DeepPocket | DeepPocket 2022 **[V]** | |
| 73.4 | DeepPocket | VN-EGNN 2025 **[V]** | |
| 70.6 | P2Rank | DeepPocket 2022 **[V]** | |
| 69.1 | GotenNet alone (0.691) | GDEGAN 2026 **[V]** | |
| 68.6 | P2Rank | P2Rank 2018 **[V]** | |
| 66.2 | EquiPocket | EquiPocket 2024 **[V]** | |
| **65.9** | VN-EGNN | VN-EGNN 2025 **[V]** | per-chain, merged |
| 62.1 | P2Rank | EquiPocket 2024 **[V]** | |
| 51.5 | Kalasanty | EquiPocket 2024 **[V]** | |
| 50.1 | DeepSurf | DeepSurf 2021 **[V]** | |
| 32.1 | Kalasanty | DeepSurf 2021 **[V]** | vs 61.21 for the same method in DeepPocket's table |

### Top-(N+2), DCA ≤ 4 Å

| | COACH420 | HOLO4K | source |
|---|---|---|---|
| GrASP | **80.6** | 84.3 | GrASP 2024 **[V]** |
| P2Rank (GrASP's re-run) | 79.4 | **86.5** | GrASP 2024 **[V]** |
| DeepPocket | 79.94 | 82.97 | DeepPocket 2022 **[V]** |
| P2Rank (own) | 78.3 | 74.0 | P2Rank 2018 **[V]** |
| P2Rank (DeepPocket's re-run) | 75.48 | 80.05 | DeepPocket 2022 **[V]** |
| DeepSurf | 73.3 | 50.6 | DeepSurf 2021 **[V]** |
| fpocket+PRANK | 76.5 | 71.0 | P2Rank 2018 **[V]** |
| fpocket (own eval) | 68.9 | 63.1 | P2Rank 2018 **[V]** |

**No equivariant method (EquiPocket, VN-EGNN, GDEGAN, GENEOnet) reports top-(N+2) at all** — which is the
metric the largest independent benchmark (LIGYSIS) recommends as the universal standard **[V]**.

---

## 7. Answers to the specific questions

### A. Which mechanisms have *demonstrated* gains on pocket detection (with numbers), and which are only assumed to help?

**Demonstrated, with numbers, in a pocket paper's own ablation:**

| Mechanism | Evidence | Gain |
|---|---|---|
| **Hybrid: geometric candidate generation + learned re-scoring** | LIGYSIS **[V]**: fpocket_PRANK 60.4 % and DeepPocket_RESC 58.1 % top-N+2 recall beat every end-to-end detector; fpocket alone at top-N+2 is ≈ 47 %. P2Rank 2018 **[V]**: fpocket 56.4 → fpocket+PRANK 63.6 on COACH420 top-N. | **+10 to +13 points recall.** The single best-evidenced mechanism in the field. |
| **Removing redundant predictions and re-ranking** | LIGYSIS **[V]**: IF-SitePred 25.7 → 39 %; VN-EGNN 40.9 → ≈ 46 %; DeepPocket_SEG 43.8 → ≈ 49 %. 67 % of VN-EGNN's 13,582 predictions were redundant. | **+5 to +13 points recall**, for free, at inference time |
| **Virtual nodes with heterogeneous message passing** | VN-EGNN ablation **[V]**: COACH420 DCC 0.156 (EGNN) → 0.503 (virtual nodes + heterogeneous MP) → 0.605 (+ESM-2). Homogeneous-MP variant 0.497 vs heterogeneous 0.503 at matched conditions. | **Virtual nodes: ≈ +0.35 DCC. Heterogeneity specifically: only ≈ +0.006–0.03.** Most of the gain is the virtual-node read-out, not the heterogeneous schedule. |
| **Protein language model features** | VN-EGNN **[V]**: +0.06 DCC on COACH420 (0.503→0.605 includes ESM; homogeneous 0.497→0.575 with ESM = +0.078). GDEGAN **[V]**: ESM-2 instead of atomic numbers gives **+15.61 % DCC and +9.27 % DCA** averaged over three datasets. | **Large and consistent — the biggest single learned-feature gain reported.** |
| **Using geometric (frame) information in attention vs none** | Lee/Byun/Shin 2023 **[V]**: replacing SE(3)-invariant geometric attention with plain BERT attention at the same depth and width costs **4.3 percentage points** averaged over datasets (COACH420 F1 success 59.1 → 54.8; HOLO4K 77.0 → 70.8). | **+4.3 points.** This is the cleanest "geometry helps" measurement for pocket detection. |
| **Surface/local-geometry modules on top of an atom graph** | EquiPocket ablations **[V]**: removing the local geometric module → COACH420 DCA 0.656→0.546; removing the global module → 0.541; removing both → 0.502. Surface probe radius 1.0/1.5/2.0 Å → DCC 0.433/0.423/0.393. | **−10 to −15 % relative** per module removed |
| **An auxiliary direction loss towards the ligand** | EquiPocket **[V]**: removing the direction loss drops DCC from 0.428→0.319 for proteins <1000 atoms, 0.621→0.587 for 1000–2000 atoms. GDEGAN **[V]**: ADL adds ≈ 2 % DCC / 3.5 % DCA. | Material for small proteins; modest overall |
| **Variance-adaptive (Gaussian) attention instead of dot-product** | GDEGAN **[V]**, at matched backbone and features: GDEGAN+ESM vs GotenNet+ESM = **+3.33 % DCC, +3.32 % DCA** averaged over three sets. But against its strongest matched arm, `GotenNet(full)` vs `GDEGAN(full)`, COACH420 DCA moves only **0.703 → 0.707** **[V]**. | **+0.4 to +3.9 points DCA depending on the dataset** — in an unrefereed preprint with no code and an internal table inconsistency |
| **Transfer learning and homology-based augmentation** | Lee/Byun/Shin **[V]**: transfer learning +4.64 points; augmentation as a whole +3.84; homology augmentation specifically +1.74. | Comparable in size to the geometry term |
| **Dense/sigmoid attention read-out for protein-size shift** | EquiPocket **[V]**: helps proteins < 4000 atoms (e.g. 0.328→0.428 DCC for < 1000 atoms); **no benefit above 4000 atoms**. | Conditional |

**Assumed, not demonstrated, for pocket detection:**

1. **Exact equivariance as such.** No paper reports an equivariant-vs-invariant comparison at equal depth and
   width (see B). Worse, the one available measurement of an *invariance mechanism* is **negative**:
   Lee/Byun/Shin's "no alignment" ablation removes the grid-alignment step that gives their model
   SE(3)-invariance and COACH420 F1 success goes **up**, 59.1 → 59.4 **[V]**. The strong argument against
   voxel CNNs is rotation *sensitivity* (EquiPocket's stated motivation), but the gains in those papers are
   confounded with changing from grids to graphs, from CNNs to attention, and from no features to ESM-2.
2. **Degree ≥ 2 (`l ≥ 2`) tensor features.** Used by GDEGAN (`L_max = 2`) and never ablated (see B).
3. **Cavity-point probes** (as opposed to sphere probes or surface probes). No published comparison (see C).
4. **Chirality / pseudo-scalar features.** GDEGAN adds an SE(3) (chirality-aware) term, but its ablation table
   varies E(3)→SE(3) *together with* the ESM features, so the chirality contribution is not isolated **[V]**.
5. **The entire backbone-paper literature of §1.** Eighteen backbones, zero pocket numbers in their own
   papers. Their pocket performance exists only as baselines re-implemented by competitors.

A blunt conclusion: on current evidence, **features (ESM-2) and pipeline design (candidate generation +
re-scoring, redundancy removal, ranking) each buy more pocket-detection performance than any choice of
equivariant layer.**

### B. Has anyone published an ablation isolating equivariance (equivariant vs invariant at equal depth/width) for pocket detection? Has anyone used `l ≥ 2` features for pocket detection?

**Equivariance ablation: NO — not for pocket detection.** Nothing in this survey isolates equivariance at
matched depth and width for pocket detection. What exists:

- **Closest: Lee, Byun & Shin 2023 [V]** (arXiv:2303.08818) — *geometric vs non-geometric attention* at equal
  depth and width: SE(3)-invariant geometric self-attention replaced by plain BERT attention, **4.3
  percentage points** average loss (COACH420 F1 success 59.1 vs 54.8). This isolates *geometry*, not
  *equivariance* — both arms are at most invariant, neither is equivariant.
- **Negative data point in the same paper [V]:** removing the grid-alignment process "adopted to achieve
  SE(3)-invariance" slightly *improves* COACH420 (59.1 → 59.4) and slightly hurts HOLO4K (77.0 → 75.6).
- **EquiPocket / VN-EGNN / GDEGAN comparisons are confounded.** EquiPocket's and VN-EGNN's tables place GAT
  (0.130 DCA, COACH420) and GCN (0.139) against EGNN (0.361) **[V]**, which looks like a 23-point
  equivariance gain — but GAT and GCN there are *topological* graph networks with **no coordinate input at
  all**, not invariant geometric networks. That comparison measures "coordinates vs no coordinates", not
  "equivariant vs invariant". GrASP (invariant GATv2, 77.5 COACH420 top-N DCA) outscores every equivariant
  method on its own test set, and P2Rank (an invariant random forest) beats VN-EGNN on HOLO4K DCA inside
  VN-EGNN's own table **[V]** — so the aggregate evidence does not even show a consistent sign.
- **Nearest true equivariant-vs-invariant ablation in any protein task: EquiPNAS** (Nucleic Acids Research
  2024, protein–**nucleic acid** binding sites), which builds a baseline invariant network "with the
  coordinate updates of the equivariant graph convolution layers turned off" at 12 EGCL layers / hidden 768.
  Reported difference: ROC-AUC 0.938 vs 0.940, PR-AUC 0.565 vs 0.569 **[UNVERIFIED]** (from search summary
  only, primary table not read) — i.e. **negligible**. This is the single most relevant data point available
  and it is *not* for ligand pockets.
- **General (non-pocket) evidence that `l ≥ 2` helps: HEGNN [V]** (arXiv:2410.11443, "Are High-Degree
  Representations Really Unnecessary in Equivariant GNNs?"), which argues degree-1 models degenerate on
  symmetric graphs. Verified from the PDF that it contains **no protein binding-site task**.

**`l ≥ 2` for pocket detection: YES, once — and never ablated.**

- **GDEGAN (arXiv:2603.19817, 2026-03-20) [V]** uses `L_max = 2` steerable features initialised from
  spherical harmonics, explicitly `l = 1` dipole and `l = 2` quadrupole terms, on top of GotenNet, for ligand
  binding-site prediction. Its ablation table varies only {backbone attention type, auxiliary directional
  loss, ESM features} — **there is no `L_max = 1` vs `L_max = 2` row [V]**, and the layer-count sweep
  (Table 6) holds `L_max = 2` fixed. So the field has a method that *uses* `l = 2` for pockets and **zero
  evidence that `l = 2` is what helps**.
- GDEGAN also **does not compare against VN-EGNN, GrASP or DeepPocket**, and takes its classical baselines
  from EquiPocket, the lowest-P2Rank measurement in the literature **[V]**.

**Practical consequence.** A clean ablation — equivariant vs invariant, and `l ≤ 1` vs `l = 2`, at matched
depth, width, parameter count, features and read-out, on a fixed pocket protocol — **does not exist in the
literature** and would be a genuine contribution. But note that GDEGAN now establishes *priority* for
"GotenNet-style geometric tensor attention with `l = 2` applied to binding-site detection", so the novelty
claim must be reframed from "first application" to "first controlled isolation", and GDEGAN must be cited and
ideally reproduced.

### C. Has anyone placed virtual/probe nodes on actual cavity grid points (as opposed to a sphere around the protein)?

**Not found for pocket detection. [NOT FOUND]** The landscape of probe placements actually published:

| Where probes/nodes live | Method | Learned? | Equivariant message passing onto the probes? |
|---|---|---|---|
| **Sphere around the protein** (Fibonacci grid, radius = centre-to-farthest-atom) | **VN-EGNN [V]** | yes | yes (E(3)/SE(3) EGNN, positions updated each of 5 layers) |
| **Solvent-accessible surface points** (MSMS, probe radius 1.5 Å) | **EquiPocket [V]** (Surface-EGNN) | yes | yes |
| **Solvent-accessible surface points** | **P2Rank [V]** | yes (random forest) | no (invariant features) |
| **Local grids centred on surface points, with a local frame** | **DeepSurf [V]** | yes (3D CNN) | no (local-frame alignment, not equivariance) |
| **Voxel grid of the protein's empty space** — i.e. cavity volume | **GENEOnet [V]** | yes, 17 parameters | **operators are translation- and rotation-equivariant**, but they are fixed-form convolutions, not message passing, and the grid points are not graph nodes |
| **Voxel grid / fixed box** | DeepSite, Kalasanty, PUResNet, RecurPocket, PharmacoNet **[V]** | yes (3D CNN) | no |
| **fpocket alpha-sphere centres** (which *are* cavity points) used as candidates for a CNN | **DeepPocket, PRANK, DeepPocket_RESC [V]** | re-scoring only | no — candidates are not graph nodes receiving equivariant messages |
| **Grid fill points typed by force-field affinity** | AutoSite **[V]** | no (force field) | no |
| **Arbitrary query grid points as "probe nodes" receiving equivariant messages** | **DeepDFT [V]** (npj Comput. Mater. 2022) — **electron density in materials, not proteins** | yes | **yes — this is exactly the mechanism, in a different domain** |

So the two halves exist separately: cavity grid points as prediction sites (GENEOnet, AutoSite, LIGSITE,
fpocket), and equivariant message passing onto probe nodes (VN-EGNN on a sphere, EquiPocket on the surface,
DeepDFT on grid points). **The combination — probe nodes initialised on *cavity* grid points inside the
protein, carrying their own buriedness geometry, updated by equivariant message passing — was not found in
any published pocket paper.** Caveat: this was established by negative search, which is weaker than a
positive finding; the claim should be stated as "we did not find" rather than "nobody has done".

### D. Has anyone trained a per-ligand-atom-class interaction field (hotspots) with a learned equivariant network, and has anyone restricted such labels to interactions actually formed?

Two separate questions, two different answers.

**Per-interaction-class field, learned, with labels restricted to interactions actually formed: YES — but not equivariant.**
**PharmacoNet [V]** (Seo & Kim, Chemical Science 2025; arXiv:2310.00681) predicts instance-segmented
pharmacophore points of **seven types** (hydrophobic carbon, aromatic ring, H-bond donor, H-bond acceptor,
halogen, cation, anion) inside a protein pocket, and **its labels are the non-covalent interactions actually
detected by PLIP in PDBBind v2020 complexes — not proximity [V]**. Architecture: a 3D CNN (Feature Pyramid
Network over a 3D Swin-Transformer-V2 encoder) on a 64³ grid. **Not equivariant.** Code CC BY 4.0 **[V]**.

**Learned equivariant multi-channel spatial field over a protein: YES — but the channels are not interaction classes.**
**Equivariant Scalar Fields [V]** (Jing, Jaakkola & Berger, arXiv:2312.04323, MLSB 2023, MIT licence)
parameterises multi-channel protein and ligand scalar fields as sums of per-atom spherical-harmonic × RBF
expansions with coefficients from **equivariant GNNs**, and scores docking by FFT cross-correlation. The
channels are learned docking-score components, not donor/acceptor/apolar/aromatic classes, and the labels are
poses, not interactions.

**The combination — per-ligand-atom-class or per-interaction-type hotspot field, predicted by a learned
equivariant network, with labels restricted to interactions actually formed — was NOT FOUND.**

For completeness, the classical per-type field methods are all physics- or knowledge-based, not learned:
**SILCS FragMaps [V]** (MD occupancy of benzene/propane/water probes), **FTMap [V]** (FFT docking of 16
probes), **Fragment Hotspot Maps [V]** (CSD-derived propensities producing exactly three maps — donor,
acceptor, apolar), **AutoSite [V]** (typed pseudo-ligand of C/OA/HD fill points). **DeepFrag [V]** is learned
but is a fragment-fingerprint predictor conditioned on an existing parent ligand, not a field.

On the label question specifically: the pocket-detection literature is unanimous in using **proximity**
(protein atoms or residues within 4 Å of any ligand atom: EquiPocket **[V]**, DeepSurf **[V]**, GrASP
**[V]**, YuelPocket **[V]**, VN-EGNN **[V]**). PharmacoNet's PLIP-derived labels **[V]** and the LIGYSIS
dataset's use of **BioLiP biological-relevance definitions [V]** are the only two places in this survey where
labels are restricted beyond geometric proximity.

### E. Has anyone combined iterative refinement / recycling of probe positions with equivariant message passing for site detection?

**YES — VN-EGNN, and it is the only one. [V]**

VN-EGNN's virtual-node coordinates are updated at **every one of its 5 message-passing layers**, so the probe
positions migrate from the initial sphere towards pocket centres; the paper's own figure shows initial
(yellow) and post-`L` (turquoise) positions. This is layerwise refinement of probe positions under E(3)/SE(3)
equivariant message passing. VN-EGNN is degree-1 only **[V]**.

Distinctions worth keeping:

- **"Recycling" in the AlphaFold2/3 sense** — feeding whole-network outputs back as inputs for several full
  passes — is **NOT FOUND** for probe positions in site detection. VN-EGNN does layerwise updates within one
  forward pass, not outer-loop recycling.
- **GotenNet's "hierarchical tensor refinement" [V]** iteratively refines **edge representations**, not
  positions, and GDEGAN inherits that; neither has probes.
- **EquiPocket's Surface-EGNN [V]** does equivariant message passing over surface points and outputs
  coordinates, but surface-point positions are not iteratively driven towards site centres; centres come from
  mean-shift clustering of positive atoms plus a fixed 4 Å projection.
- **NEMP** (Node-Equivariant Message Passing, Chem. Sci. 2026 **[UNVERIFIED]**) iteratively refines
  coefficients against a virtual summed node — but for interatomic potentials, not site detection.
- **Full-atom pocket *design*/generation** methods (PocketGen, Pocket2Mol, "Full-Atom Protein Pocket Design
  via Iterative Refinement") use equivariant iterative refinement, but they *generate* pockets given a site,
  they do not detect sites **[UNVERIFIED]** in detail.

So: layerwise equivariant refinement of probes — done once (VN-EGNN, sphere initialisation, `l ≤ 1`).
Outer-loop recycling, cavity initialisation, or `l ≥ 2` combined with probe refinement — not found.

### F. Standard evaluation protocols and their known flaws

**The protocols in use.**

| Protocol | Definition | Used by |
|---|---|---|
| **DCA** | distance from predicted site centre to the **closest ligand heavy atom**; success if < 4 Å | P2Rank lineage: P2Rank, DeepSurf, DeepPocket, GrASP, EquiPocket, VN-EGNN, GDEGAN **[V]** |
| **DCC** | distance from predicted centre to the **ligand/site centroid**; success if < 4 Å | same papers, reported alongside DCA |
| **top-N** | keep exactly N predictions, N = number of annotated sites for that protein | all of the above **[V]** |
| **top-(N+2)** | keep N+2 predictions | P2Rank, DeepSurf, DeepPocket, GrASP, LIGYSIS **[V]** — **not** EquiPocket, VN-EGNN, GDEGAN, GENEOnet |
| **Failure rate** | fraction of proteins with no prediction at all | EquiPocket, VN-EGNN, GDEGAN **[V]** |
| **Residue-level F1/MCC/AUC/AP** | per-residue classification | ScanNet, PeSTo, CryptoBench, PocketMiner, ProtGeoNet-Pocket, ProMoSite **[V]** |
| **Recall at DCC ≤ 12 Å, top-N+2** | LIGYSIS's recommended standard | LIGYSIS **[V]** |

**Known flaws, each with a primary source.**

1. **"Success rate" is not a defined term.** LIGYSIS **[V]**: *"some methods use recall, whereas others use
   precision, both under the same term of success rate. This can be confusing."* Recommendation: standardise
   on **top-N+2 recall**.
2. **top-N assumes the annotation is complete, and it is not.** LIGYSIS **[V]** estimates **33–50 % of
   existing sites are yet to be observed**, so top-N penalises a method for ranking a *real but unannotated*
   pocket above an annotated one. DeepPocket independently observed a **17-point jump from top-N to
   top-(N+2)** and attributed it to unannotated sites **[V]**. Four of the newest and most-cited equivariant
   methods report **only top-N**.
3. **The 4 Å DCC threshold is too strict.** LIGYSIS **[V]**: *"A DCC threshold of 4 Å is too conservative,
   and to obtain comparable results between DCA and DCC recall, a threshold of DCC of 10–12 Å should be
   employed."* This matters: the ranking of equivariant methods is largely driven by DCC, where VN-EGNN's
   advantage is largest (0.605 vs EquiPocket 0.423 on COACH420), and DCC-at-4 Å is the metric LIGYSIS
   considers least meaningful.
4. **Redundant predictions corrupt both precision and recall.** LIGYSIS **[V]**: **67 % of VN-EGNN's 13,582
   predictions were redundant** (9,066 duplicates); IF-SitePred 49 %; DeepPocket_SEG 31 %. Example given:
   VN-EGNN predicts the same pocket of human creatine kinase **7 times**. Removing redundancy and re-ranking
   raises recall by 5–13 points. VN-EGNN's ROC₁₀₀ count of 1,301 TP is explicitly described as inflated by
   redundancy. **No equivariant pocket paper reports a redundancy statistic.**
5. **Ligand filters differ, and there is no single "COACH420".** The `mlig` subsets (P2Rank, from Binding
   MOAD relevance) are standard in the P2Rank lineage **[V]**. GrASP applies MOAD relevance **and** geometric
   criteria to produce `Mlig+`, leaving **COACH420(Mlig+) = 256 systems / 315 ligands** and
   **HOLO4K(Mlig+) = 4,514 systems / 6,368 ligands [V]** — a very different test set under the same name.
   EquiPocket says `mlig` but also projects predicted centres 4 Å outward **[V]**.
6. **HOLO4K's multimers are the single worst problem.**
   - LIGYSIS **[V]**: **1,811 of 4,550 (40 %) HOLO4K structures differ in chain count between asymmetric and
     biological unit**; and **234 of 417 (56 %) COACH420 structures** are affected too. Named examples:
     PDB 1JQY gives **14× redundancy** of identical interfaces, PDB 1PPR **3×**. Verdict: *"The use of
     duplicated protein–ligand interfaces in asymmetric units results in an overestimate of both precision
     and recall."*
   - GrASP **[V]**: *"HOLO4K contains many multimers with repetitions of the same binding mode"*; its fix is
     to connect all chains sharing an interfacial ligand and split everything else into single chains, which
     *"should more closely reflect the workflow used in practice avoiding evaluation on homomultimers while
     preserving evaluation on interfacial binding"*.
   - VN-EGNN **[V]**: *"Given the large complexes in HOLO4K, we process each chain individually … and then
     merge the predicted pocket centres."*
   - GDEGAN **[V]**: splits HOLO4K into per-chain components and calls it *"a strong distribution shift"*.
   Four papers, four different HOLO4K preparations, all reporting "HOLO4K DCA top-N".
7. **Homology control is inconsistent, and leakage is demonstrable.**
   - DeepSurf **[V]**: removes any training target with > 90 % sequence similarity to a test target.
   - CryptoBench **[V]**: 40 % sequence clustering plus a further 10 %-identity clustering across splits.
   - YuelPocket (PNAS 2026) **[V]**: **excluded COACH420 entirely** because *"the vast majority of COACH420
     systems were already present in the PLINDER training data"*. This is the strongest published statement
     that COACH420 is contaminated for models trained on large modern corpora.
   - LIGYSIS **[V]**: a different leakage direction — methods that train on 1:1 complexes mislabel as
     non-binding those residues that bind a ligand in *another* structure of the same protein; P2Rank and
     GrASP enrich their training data with ligands from other chains and homologues, while *"DeepPocket,
     PUResNet, VN-EGNN … seem to rely fully on 1:1 interactions"*, capping their achievable performance.
8. **Residue-level metrics reward conservatism.** LIGYSIS **[V]**: F1/MCC favour methods that predict only the
   obvious site (PUResNet F1 0.41 highest, fpocket 0.23 lowest) — the opposite of what matters for discovery.
9. **Training-set pathologies.** scPDB considers **one ligand per PDB entry**, is depleted in saccharides
   (< 1 %) and > 90 % of its sites exceed 20 residues **[V]**; COACH420/HOLO4K are biased towards cofactors
   (ATP/FAD/NAD > 30 % of top ligands) **[V]**; PDBbind overlaps LIGYSIS by only 3.8 %, Binding MOAD by 7.6 %
   **[V]**.
10. **Recall ceilings are rarely reported.** fpocket's candidate recall (pocket centre within 4 Å of a
    ligand atom) is **80.78 % on COACH420, 87.62 % on HOLO4K, 96.4 % on scPDB [V]** (DeepPocket) — a hard cap
    on any fpocket-based re-scoring pipeline, which LIGYSIS confirms: all three fpocket-derived methods
    plateau at ≈ 600 TP **[V]**.

**Minimum defensible protocol, implied by the above:** report **DCA top-N *and* top-(N+2)**; report **DCC at
both 4 Å and 10–12 Å**; report a **redundancy statistic and a redundancy-removed variant**; state the exact
ligand filter and give the resulting system/ligand counts; state the chain/biological-unit treatment of
HOLO4K explicitly; report the **homology-control threshold** used against every test set, and the overlap
between training data and each test set; report **failure rate** and **candidate recall ceiling**; and
re-run every baseline yourself rather than copying rows between papers.

### G. Current best reported number on COACH420 and HOLO4K DCA top-N, and by which method?

**Answer, with the honesty the question deserves: the question has no single answer, because no two of the
leading numbers were measured on the same test set.**

**Highest reported values [V]:**

- **COACH420 DCA top-N: GrASP, 77.5 %** (J. Chem. Inf. Model. 2024) — on **COACH420(Mlig+), 256 single-chain
  systems**, with P2Rank at 74.9 % on the same set. GrASP is an **invariant** GATv2, not equivariant.
  - Runner-up and the **highest among equivariant methods: VN-EGNN, DCA 0.750** (J. Cheminform. 2025) on the
    `mlig` subset.
  - Highest DCC on COACH420: VN-EGNN 0.605, then GDEGAN 0.580 (preprint).
- **HOLO4K DCA top-N: GrASP, 81.3 %**, with **P2Rank at 81.2 %** on the same HOLO4K(Mlig+) set of 4,514
  chain/interface-split systems — statistically a tie, and P2Rank wins top-(N+2) there, 86.5 % vs 84.3 %.
  - **GDEGAN claims 0.788** (arXiv preprint, March 2026, no code, no peer review, baselines copied from the
    withdrawn EquiPocket arXiv, no comparison to VN-EGNN/GrASP/DeepPocket).
  - **P2Rank is measured at 0.787 on HOLO4K DCA inside VN-EGNN's own published table [V]** — i.e. GDEGAN's
    headline HOLO4K number is within noise of a 2018 random forest as measured by VN-EGNN's authors.
  - **VN-EGNN itself scores 0.659 on HOLO4K DCA and is beaten there by both P2Rank (0.787) and DeepPocket
    (0.734) in its own table [V].**
  - Highest DCC on HOLO4K: GDEGAN 0.560 (preprint), then VN-EGNN 0.532.

**The defensible statements:**

1. **No equivariant method has been shown to beat a well-run P2Rank on HOLO4K DCA.** VN-EGNN's own table
   shows P2Rank ahead; GDEGAN only appears ahead because it uses EquiPocket's degraded P2Rank measurement.
2. **The equivariant advantage is concentrated in DCC**, the metric LIGYSIS argues is mis-thresholded at 4 Å.
   Equivariant models with a coordinate read-out land their centres nearer the centroid; that is not the same
   as finding more sites.
3. **On the one large, independent, biological-unit-aware benchmark (LIGYSIS), the best equivariant method
   ranks 12th of 13 at 40.9 % top-N+2 recall, rising to ≈ 46 % after redundancy removal, against 60.4 % for
   fpocket re-scored by PRANK [V].** That is the most protocol-robust comparison available, and it does not
   favour equivariant end-to-end detectors.
4. **Any new paper should report GrASP, P2Rank (re-run), DeepPocket, VN-EGNN and GDEGAN on its own single
   protocol**, and should not reuse any published row.

---

## 8. Searches that failed, and things left unverified

**Unreachable or blocked (reported, not worked around):**
- `link.springer.com` and `nature.com` article pages redirect to IdP authorisation endpoints; worked around via
  PMC where an open-access copy existed, otherwise **[NOT FOUND]**.
- `pubs.acs.org` returns **HTTP 403** — GrASP's JCIM version, ProtGeoNet-Pocket, SwinSite and SGLEPocket
  could not be read there. GrASP was recovered via the Europe PMC JATS full text of PMC10402091.
- `biorxiv.org/.../v2.full` returned **HTTP 429**; PDF endpoints returned HTML, not PDF.
- `api.github.com` is gated in this environment (403 via `curl`, "GitHub access not enabled" via `gh`).
  All licences were instead read from `raw.githubusercontent.com/<repo>/<branch>/LICENSE*`, which is a
  primary source but misses repositories that state their licence only in a README or `pyproject.toml`.
- `openreview.net` serves a browser-verification page to WebFetch.
- PMC direct `curl` is behind a reCAPTCHA; PMC was read via WebFetch instead.

**Explicitly not verified (do not cite from this document):**
- Venue/year for SE(3)-Transformer (NeurIPS 2020), Equiformer (ICLR 2023), NequIP (Nat. Commun. 2022),
  Allegro (Nat. Commun. 2023), HEGNN (NeurIPS 2024), Frame Averaging (ICLR 2022), EGNN (ICML 2021),
  GearNet (ICLR 2023), ProteinMPNN (Science 2022), AlphaFold2 (Nature 2021), ESM-3 (Science 2025),
  RecurPocket (IEEE BIBM 2022).
- Licences for SE(3)-Transformer, CGENN, Frame Averaging, FAENet, SEGNN, EST, DualEquiNet, SWSH,
  Kalasanty, DeepSurf, DeepSite, PUResNet, PointSite, RecurPocket, ProtGeoNet-Pocket, GENEOnet code,
  GDEGAN, ProMoNet, YuelPocket, and all peptide methods.
- **Weights** licences generally. Only AlphaFold3 (non-commercial, non-redistributable **[V]**) and the
  current ESM repo (MIT **[V]**) were read directly.
- EquiPNAS's equivariant-vs-invariant ablation numbers (ROC-AUC 0.938 vs 0.940) — from a search summary only.
- AtomSurf's binding-site identification numbers and protocol.
- DCA/DCC numbers for ProtGeoNet-Pocket, PointSite, PUResNet, YuelPocket (read as approximate "≈" values
  from figures in the PMC text, not from a table), and ProMoSite's baseline rows.
- Whether GrASP's Zenodo evaluation lists are CC-BY-4.0.
- Whether the historical ESM-3 "Cambrian/EvolutionaryScale" non-commercial licence still applies to any
  checkpoint one might download today.

**Primary sources consulted (selection):** arXiv abs/PDF for 2302.12177 (withdrawal notice), 2603.19817,
2303.08818, 2410.11443, 2404.07194, and the arXiv metadata API for 16 backbone papers; PMLR v235 page and
PDF for EquiPocket; PMC12837241 (VN-EGNN), PMC11552181 (LIGYSIS), PMC10402091 + Europe PMC JATS (GrASP),
PMC6091426 (P2Rank), PMC12494700 (GENEOnet), PMC12974528 (YuelPocket), PMC13371367 (ProMoNet),
PMC10113261 (PeSTo); `academic.oup.com` for DeepSurf and CryptoBench; the IIIT CDN PDF of DeepPocket;
the ICLR 2025 proceedings page for GotenNet; `WEIGHTS_TERMS_OF_USE.md` and READMEs in
`google-deepmind/alphafold3` and `evolutionaryscale/esm`; and `LICENSE` files in 30 repositories.
