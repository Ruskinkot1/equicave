---
title: "EquiCave: a self-contained equivariant multi-task model of protein binding pockets"
subtitle: "Ranked sites, pocket property classes and ligand-atom hotspot fields, with a peptide-groove tier"
status: "DRAFT — every number must come from docs/results/; placeholders read TBD"
---

# Abstract (draft, numbers TBD)

Binding-site prediction is usually solved either geometrically or by machine learning on top of an external pocket
finder, and the published numbers are not comparable because protocols, ligand filters and splits differ. We present
EquiCave, a model that takes a protein structure and returns three things: a ranked list of binding sites, a
multi-label description of each pocket, and a field of per-class ligand-atom probabilities inside it. The model is
self-contained: candidates come from a lattice closure analysis of the structure itself, and no external pocket
finder or Java runtime is used at training or inference. The network is an SO(3)-equivariant message-passing
architecture in which every node carries scalar, vector and symmetric traceless tensor channels updated by
invariant-gated geometric tensor attention; probe nodes are placed on free points of real cavities rather than on a
sphere, and a solvent-accessible-surface point cloud is a node type in the same attention stack. A separate groove
tier with backbone-exposure features addresses peptide-binder sites, for which we build a homology-controlled
benchmark from the PDB. On TBD structures with 30 %-identity cluster splits our candidate generator reaches a
ceiling of 0.977 at 30 candidates per structure, and a LambdaRank re-ranker over 32 structure-only features reaches
top-1 0.724 (95 % CI 0.701–0.748). Network results, ablations and external benchmarks: TBD.

# 1. Introduction

- Binding-site detection as the first step of structure-based design; three questions a designer actually asks:
  where, what kind, and where exactly inside the site.
- Three families of methods: geometric (LIGSITE, fpocket), learned over surface descriptors (P2Rank, DeepSurf),
  learned end-to-end on graphs (GrASP, EquiPocket, VN-EGNN). Hybrids win in independent comparisons (LIGYSIS).
- Two gaps we address. (i) Every learned method of the second and third family in practical use still depends on an
  external finder for candidates, or on a sphere of virtual nodes that ignores where cavities actually are.
  (ii) High-degree equivariant layers (degree 2 and above), standard in molecular property prediction, have not been
  applied to binding-site detection, and no ablation isolates equivariance at equal depth and width.
- A third gap: peptide-binding sites are shallow elongated grooves and are evaluated, when at all, with
  small-molecule conventions.
- Contributions: 1–5 of `docs/PAPER_PLAN.md`.

# 2. Methods

## 2.1 Candidate generation from closure
Full specification: `docs/ARCHITECTURE.md` §1. Absolute 1 Å lattice; free points at ≥ 3 Å from heavy atoms;
buriedness as the number of the 26 lattice directions along which protein is met within 8 Å, computed by integer
array shifts; 26-connected components of points above a closure threshold; splitting into sub-sites by peaks of the
smoothed buriedness field with 6 Å non-maximum suppression; two depth tiers (16 and 10 of 26 rays) and at most 30
candidates. Calibration of the thresholds (Table S1) was done on 55 structures and fixed before any ranking model
was trained.

## 2.2 Peptide grooves
`docs/ARCHITECTURE.md` §1.4. Closure threshold 10 of 26 rays; local peaks with 5 Å suppression; each candidate grown
to the patch points within 7 Å; kept only if its longest principal extent is ≥ 9 Å and its anisotropy ≥ 1.5; the
receptor's exposed backbone N, O and Cα counts within 6 Å of the candidate points are features. Groove and cavity
candidates are kept together, with their mutual distance as a feature, so the ranker sees both descriptions.

## 2.3 EquiCave-Net
`docs/ARCHITECTURE.md` §2, which gives the message equations. Node types: residues at Cα with frozen ESM-2
embeddings, cavity probes, SAS surface points. Channels: `x ∈ R^F`, `V ∈ R^{F×3}`, `T ∈ R^{F×3×3}` symmetric and
traceless. Messages are built only from the Cartesian couplings up to degree 2 (`⟨V,u⟩`, `uᵀTu`, `Tu`, `uuᵀ−I/3`,
`sym_traceless(V,u)`), each multiplied by a scalar gate produced from invariants, with attention normalised over the
incoming edges of each node. Chirality enters through pseudo-scalar triple products of vector channels, so the model
is SO(3)- but not O(3)-equivariant unless that term is switched off. Equivariance is verified numerically to 1e-4 for
every head (`tests/test_model.py`).

Heads and losses: residue segmentation (Dice + BCE), probe occupancy (Dice + BCE), site centre as an equivariant
vector offset read from `V` with a set loss over all sites of the structure, confidence, 14 pocket property classes,
and 7 hotspot classes per probe (focal BCE). Labels are derived from crystal ligands and the wwPDB Chemical
Component Dictionary (`docs/DATA_CARD.md`).

## 2.4 Hybrid ranking
LightGBM LambdaRank over 32 structure-only features in four groups (native, geometry, chemistry, context), plus
optional out-of-fold network features and, for peptide targets, 14 peptide features. One query per structure.

## 2.5 Data, splits and protocol
`docs/DATA_CARD.md`. RCSB PDB (CC0); 30 %-identity clusters; five folds assigned per cluster; 12 held-out families
excluded by cluster and UniProt accession; peptide benchmark clusters disjoint from the small-molecule training
clusters. Success is DCA ≤ 4 Å (DCC reported alongside); metrics are top-1, top-3, top-N, top-(N+2), MRR and the
candidate ceiling; confidence intervals are cluster bootstraps and every gain is tested with a paired cluster
bootstrap; at least three seeds; external-benchmark structures sharing a cluster with training are reported
separately.

# 3. Results

## 3.1 How far can geometry alone go? (candidate ceiling)
`docs/results/native_candidates_build.log`. 1367 structures, 30.0 candidates on average, ceiling **0.977**, median
best DCA 0.65 Å, 3.6 s per structure per core. Top-10 ceiling 0.918, top-20 0.964.

## 3.2 Ranking structure-only candidates
`docs/results/ranker_native.md`. Table 2. Detector order 0.523 top-1; LambdaRank **0.724 [0.701, 0.748]** top-1 and
**0.877** top-(N+2); chemistry is the most valuable feature group.

## 3.3 Same structures through fpocket and P2Rank
`docs/results/methods_comparison.md`. TBD (runs in progress at the time of writing).

## 3.4 The network and its ablations
TBD — requires GPU training. The ablation grid is specified and scripted (`scripts/train/run_ablations.sh`).

## 3.5 Pocket properties and the hotspot field
TBD: per-class AUROC/AP, ECE, enrichment of true ligand atoms in the top-k % of field points with a permutation
control. Label prevalences are in `data/processed/labels_summary.json`.

## 3.6 Peptide-binding sites
`docs/results/peptide_sites.md`. The benchmark holds 973 complexes in 685 receptor clusters. First finding: cavity
candidates already reach a high ceiling on peptide sites while groove candidates alone reach ~0.8, so detection is
not the limiting factor — ranking is. Ranking with and without the peptide feature group: TBD.

## 3.7 External benchmarks
TBD: COACH420, HOLO4K (DeepPocket ligand rule), LIGYSIS, CryptoBench, held-out families, one protocol.

# 4. Discussion

- What the ceiling result means: a geometry-only generator is sufficient as a candidate source, which removes the
  dependency on external finders that all comparable pipelines carry.
- Where equivariance should help and where it should not; the honest possibility that degree-2 channels do not pay
  for their cost, which the ablation is designed to detect.
- Peptide sites: why small-molecule conventions mislead, and what a groove-aware protocol changes.
- Limitations: crystal structures only, PDB-format parsing, heuristic property labels, no apo/holo pairing yet,
  no cryptic-site training signal.

# 5. Data and code availability
Code: this repository (private at submission; to be released under a licence chosen before publication).
Data: RCSB PDB and the wwPDB CCD (CC0); benchmark lists are fetched by the scripts from their original sources and
not redistributed. Models trained on non-commercial data would be labelled accordingly in their model cards.

# Figures
F1 pipeline and the three outputs. F2 the equivariant layer. F3 split and leakage design. F4 main curves.
F5 ablation bars. F6 qualitative pockets with hotspot fields including a peptide groove. F7 failure cases.

# Tables
T1 data. T2 main metrics. T3 ablations. T4 peptide benchmark. T5 licences and scope. S1 threshold calibration.
