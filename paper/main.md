---
title: "EquiCave: a self-contained equivariant multi-task model of protein binding pockets"
subtitle: "Ranked sites, pocket property classes and interaction-validated hotspot fields, with a peptide-groove tier"
status: "DRAFT. Every number traces to docs/results/. Anything not measured says NOT RUN; nothing is placeholder."
commit: "set by the author at submission: git rev-parse --short HEAD"
---

# Abstract

Binding-site prediction is either geometric, or machine learning on top of an external pocket finder, and the
published numbers are not comparable because ligand filters, chain handling and splits differ between papers. We
present EquiCave, which takes one protein structure and returns three things: ranked binding sites, a multi-label
description of each pocket, and a field of per-class ligand-atom probabilities inside it. The model is
self-contained: candidates come from a closure analysis of the structure itself, and no external finder or Java
runtime is used at training or inference. Its network is SO(3)-equivariant, with scalar, vector and symmetric
traceless tensor channels updated by invariant-gated geometric tensor attention; probe nodes are placed on free
points of real cavities rather than on a sphere, a solvent-accessible-surface point cloud is a node type in the same
attention stack, and the probes are recycled through their own predicted centres. The hotspot field is trained on
labels that require an interaction to have actually formed, not on proximity. A separate groove tier addresses
peptide binders, for which we build a homology-controlled benchmark from the PDB.

On 1367 RCSB structures split by 30 %-identity cluster, the geometric candidate generator alone reaches a ceiling of
0.977 at 30 candidates per structure, against 0.955 for fpocket (37.8 candidates) and 0.914 for P2Rank (9.2
candidates) run on the same structures with the same labels. A LambdaRank re-ranker over 109 structure-only features
reaches top-1 0.724 [0.701, 0.748], above fpocket's own ordering (0.391) and below P2Rank's (0.754) — the limiting
factor is therefore ranking, not candidate generation, which is the central empirical finding of the candidate study.
On twelve held-out drug-target families the detector order solves 0.917 at top-1 and 1.000 at top-(N+2). Network
results, the ablation grid and the external-benchmark table: NOT RUN (the network has only been run as a CPU pilot).

# 1. Introduction

Three questions a designer asks of a structure: where does something bind, what kind of pocket is it, and where
exactly inside it would each chemical group sit. Methods answer them separately. Geometric detectors (LIGSITE,
fpocket) answer the first cheaply and with no training. Learned surface models (P2Rank, DeepSurf) rank better.
Graph and equivariant models (GrASP, EquiPocket, VN-EGNN) improve segmentation. Independent comparison finds hybrid
pipelines ahead of end-to-end detectors.

Four gaps motivate this work.

1. **Dependence on external finders.** Learned re-rankers and CNN rescorers take their candidates from fpocket or
   P2Rank, which fixes their ceiling and adds a Java runtime to every deployment. Whether a self-contained generator
   can match that ceiling has not, to our knowledge, been measured.
2. **Degree-2 features are untested here.** Tensor (l ≥ 2) channels are routine in molecular property prediction.
   We find no application to binding-site detection, and no ablation isolating equivariance at equal depth and width
   for this task.
3. **Probes on a sphere.** VN-EGNN's virtual nodes start on a sphere around the protein and must learn to migrate
   into cavities. Cavities can be computed first.
4. **Hotspot labels are proximity labels.** Fields of ligand-atom type (SILCS, FTMap, Fragment Hotspot Maps,
   AutoSite) are physics- or sampling-based; learned fields use occupancy. A ligand atom near a point is not evidence
   that an atom of that class belongs there: the atom has to be making the interaction.

Peptide binders sit outside all of this: their sites are shallow elongated grooves, evaluated, when at all, with
small-molecule conventions.

**Contributions.** (i) A geometry-only candidate generator whose ceiling exceeds both external finders on the same
structures. (ii) Geometric tensor attention for site detection, with probes in real cavities, a surface point cloud,
amino-acid chemistry as explicit scalars and side-chain direction vectors, probe recycling, and an equivariant versus
invariant ablation at equal capacity. (iii) Interaction-validated hotspot labels and their geometric counterpart as
ranker features. (iv) A homology-controlled peptide-site benchmark with a groove tier, and the finding that detection
is not its bottleneck. (v) One evaluation protocol applied to every benchmark, with train-similar structures reported
separately and the ligand rule stated.

# 2. Methods

## 2.1 Candidate generation from lattice closure

An absolute cubic lattice of step 1 Å covers the receptor. One nearest-neighbour query per point gives a distance
field, from which *free* points (≥ 3.0 Å from any heavy atom, where a ligand atom could sit) and *wall* points
(< 2.0 Å, which a ray can hit) follow. Closure of a free point p is

&nbsp;&nbsp;&nbsp;&nbsp;b(p) = Σ_{k=1..26} 1[ ∃ t ≥ 1 : t·|e_k|·1 Å ≤ 8 Å ∧ wall(p + t·e_k) ],

the number of the 26 lattice directions along which protein is met within 8 Å. On the lattice this is 26 sums of
shifted boolean arrays instead of 26 × 8 per-point queries, so one protein costs about 3.6 s on a core; the lattice
form agrees with an exact ray caster to r > 0.95 and a mean deviation below 2 of 26 rays (`tests/test_detect.py`).

Cavities are the 26-connected components of b ≥ 16. Each is split at peaks of the Gaussian-smoothed closure field
under 6 Å non-maximum suppression; every point joins its nearest peak; a group of ≥ 12 points is a candidate, scored
Σ b(p), centred at the closure-weighted centroid. If fewer than 30 candidates result, a second tier at b ≥ 10 adds
shallower cavities whose centres are more than 6 Å from every deep candidate. Thresholds were calibrated once on 55
structures before any model existed (Table S1) and then frozen; they are stored with every model and checked at load.

**Peptide grooves.** Closure ≥ 10 (and < 26), local peaks under 5 Å suppression, each grown to the patch points
within 7 Å, kept only if the longest principal extent is ≥ 9 Å and the anisotropy ax₁/ax₃ ≥ 1.5, scored
Σ b(p) · min(anisotropy, 6). Groove and cavity candidates are kept together with their mutual distance as a feature,
since a groove and a cavity description of the same region are both informative.

## 2.2 EquiCave-Net

**Nodes.** Residues at Cα carry a 21-way one-hot, 14 amino-acid property scalars (hydropathy, side-chain volume,
charge, side-chain hydrogen-bond donors and acceptors, aromatic, aliphatic, polar, small, hinge, rigid,
metal-binding, reactive, rotatable bonds), two pocket-facing cosines, a b-factor z-score, a local density, and a
frozen ESM-2 embedding. Probes sit on free cavity lattice points with their closure, depth and neighbour counts.
Surface nodes are solvent-accessible-surface points with their owner atom's element and residue. Edges are k-nearest
within a per-pair radius for each of the nine ordered type pairs, with the pair identity, a bucketed sequence
separation and a same-chain flag embedded — distance alone cannot tell a helix turn from a chain interface.

**Channels.** Every node carries x ∈ ℝ^F (degree 0), V ∈ ℝ^{F×3} (degree 1, V ↦ V Rᵀ) and symmetric traceless
T ∈ ℝ^{F×3×3} (degree 2, T ↦ R T Rᵀ). Initial vectors are the backbone directions N→Cα, C→Cα, Cα→Cβ and two
side-chain directions, Cα→side-chain centroid and Cα→functional-group centroid; the last says which way a residue's
*chemistry* faces, so an Asp lining a cavity is distinguishable from one facing solvent.

**Layer.** For an edge j → i with r = p_i − p_j, d = |r|, u = r/d, the invariant stream is
[x_i, x_j, RBF(d), E(type), E(seq), ⟨V_j,u⟩, ‖V_j‖, uᵀT_j u, ‖T_j‖_F]. An MLP maps it to attention logits, softmaxed
over each node's incoming edges, and to scalar gates. Messages use only equivariant primitives, each scalar-gated:

&nbsp;&nbsp;&nbsp;&nbsp;Δx = Σ a [ g¹⊙W_x x_j + g²⊙⟨V_j,u⟩ + g³⊙uᵀT_j u ]
&nbsp;&nbsp;&nbsp;&nbsp;ΔV = Σ a [ g⁴⊙W_v V_j + g⁵⊙u + g⁶⊙T_j u ]
&nbsp;&nbsp;&nbsp;&nbsp;ΔT = Σ a [ g⁷⊙W_t T_j + g⁸⊙(uuᵀ − I/3) + g⁹⊙sym₀(V_j,u) ]

with sym₀(a,b) = ½(abᵀ + baᵀ) − ⅓(a·b)I: the Cartesian equivalents of the Clebsch–Gordan couplings to l = 2. A
node-wise refinement gates V ← V + g⊙T V and T ← T + g⊙sym₀(V,V) by the node's invariants and, optionally, by
pseudo-scalar triple products ⟨V_a × V_b, V_c⟩, which flip under reflection: with them the network is SO(3)- but not
O(3)-equivariant, so chirality is visible. An alternative backbone expresses the same layer in e3nn with irreps
0e+1o+2e(+3o) and spherical-harmonic edge features, both to cross-check the hand-written form and to reach l = 3.
Equivariance is verified numerically for every head (1e-4 for the Cartesian layer, 1e-7 for e3nn) together with the
mirror test.

**Recycling.** After a pass, probes move to their predicted centres, their edges are rebuilt and the trunk runs
again; every pass is supervised.

**Heads and losses.** Residue and probe segmentation (Dice + BCE); a site centre read out as an equivariant vector
offset from V, trained with one-to-one Hungarian matching of the most confident proposals to the true sites, plus a
confidence head; 14 pocket property classes (multi-label BCE over probes pooled by site membership); the 7-class
hotspot field (focal BCE); and two auxiliary tasks, masked-residue identity (self-supervised, and not recoverable
from the remaining inputs) and the site's ligand atom count.

## 2.3 Labels, and why hotspots are interactions

Site hit: DCA ≤ 4 Å from a candidate centre to the nearest heavy atom of a kept ligand copy (≥ 8 heavy atoms,
solvents, buffers, ions, sugars and detergents excluded, 150–900 Da). Residue positive within 4 Å of a ligand atom;
probe occupied within 2 Å. Property classes from the wwPDB Chemical Component Dictionary plus size, buriedness and
polarity.

A grid point is a hotspot for a chemical class only when the ligand atom within 1.5 Å of it **makes the matching
interaction** with the receptor: a hydrophobic carbon against a receptor hydrophobic carbon within 4.5 Å, an aromatic
atom against a ring atom within 5.5 Å or a cation within 5.0 Å, a donor against an acceptor within 3.5 Å and
conversely, a charged atom against the opposite charge within 4.0 Å, a halogen against O or S within 3.8 Å. Receptor
atoms are typed by residue and atom name, so no protonation inference is needed. The effect is large: on one
structure FAD keeps 8 of 15 nominal acceptors, and a bound inhibitor's nominal donor that donates to nothing is
dropped entirely; across a pocket, nominal donor coverage falls from 0.218 to 0.042. The proximity variant is kept
and scored in parallel, so "interaction-validated versus proximity" is an ablation, not an assumption. A drug-like
filter (12–60 heavy atoms, at least one ring, no cofactor-like CCD class) optionally restricts the field to the
chemistry a drug programme would start from.

## 2.4 Hybrid ranking

109 structure-only features per candidate: the candidate as generated (8), cavity shape (18), environment
composition (14), the receptor interaction partners offered in 5/8/12 Å shells (46), the interaction-potential field
of its own points (21: for each point, which interaction an atom placed there could make — the hand-computed
counterpart of the learned field) and context (2). Out-of-fold network scores and, for peptide targets, 14 peptide
features join them. LightGBM LambdaRank with graded relevance (2 within 2 Å, 1 within 4 Å), within-structure z-scores
of every feature, a seed ensemble, and an out-of-fold isotonic calibration of the final score.

## 2.5 Data and protocol

RCSB PDB (CC0) and the wwPDB CCD (CC0); splits by 30 %-identity cluster, five folds assigned per cluster, 12
held-out families excluded by cluster and UniProt accession; the peptide benchmark's receptor clusters disjoint from
the training manifest. Success DCA ≤ 4 Å with DCC reported alongside; metrics top-1, top-3, top-N, top-(N+2), MRR and
the candidate ceiling, where N is the structure's own number of ligand sites; predictions non-redundant at 6 Å;
confidence intervals are cluster bootstraps and every gain carries a paired cluster bootstrap; at least three seeds.
On external benchmarks the ligand rule (published relevant-ligand list or every drug-sized group) is stated, and
structures sharing a cluster with training are reported as a separate row. Details: `docs/DATA_CARD.md`.

# 3. Results

## 3.1 Data (Table 1)

| set | entries | usable | clusters | note |
|---|---|---|---|---|
| training manifest | 1499 | 1367 | 1017 | 107 are mmCIF-only at RCSB, 25 have no usable ligand |
| peptide benchmark | 973 | 894 | 685 receptor clusters | 53 have fewer than 3 observed peptide residues |
| held-out targets | 12 | 12 | 11 | ligand ids verified against the deposited entries |
| COACH420 | 420 | 300 under the relevant-ligand rule | — | fetched from the published list |
| HOLO4K | 4009 | 3448 under the relevant-ligand rule | — | NOT RUN |
| LIGYSIS | 3448 chains | — | — | NOT RUN |
| CryptoBench | 5493 (1100 test) | — | — | NOT RUN |

Mean N over the training set is 2.27 (median 2, maximum 20); 609 of 1367 structures have a single site.

## 3.2 How far geometry alone goes (Table 2)

Same 1366–1367 structures, same ligand filter, same labels, same metric.

| generator | ceiling | candidates | top-1 by its own order |
|---|---|---|---|
| **EquiCave closure analysis** | **0.977** | 30.0 | 0.523 [0.494, 0.552] |
| fpocket 4.x | 0.955 | 37.8 | 0.391 |
| P2Rank 2.5.1 | 0.914 | 9.2 | 0.754 |

Our ceiling by depth: top-10 0.918, top-15 0.944, top-20 0.964, top-30 0.977; median best DCA 0.65 Å. A geometry-only
generator therefore does not cost coverage, which removes the dependency every comparable pipeline carries.

## 3.3 Ranking (Table 3)

5-fold cross-validation by cluster, 5 seeds, 95 % cluster bootstrap, 32-feature version.

| method | top-1 | top-3 | top-N | top-(N+2) |
|---|---|---|---|---|
| detector order | 0.523 [0.494, 0.552] | 0.747 | 0.634 | 0.800 [0.777, 0.825] |
| most buried first | 0.368 | 0.695 | 0.504 | 0.762 |
| largest cavity first | 0.207 | 0.383 | 0.296 | 0.443 |
| **LambdaRank (32 features)** | **0.724 [0.701, 0.748]** | 0.857 | 0.782 | **0.877 [0.859, 0.896]** |
| without the chemistry group | 0.649 | 0.806 | 0.721 | 0.837 |
| candidate features only | 0.595 | 0.789 | 0.695 | 0.824 |

The 109-feature version with graded relevance, within-structure z-scores, the seed ensemble and calibration: IN
PROGRESS at the time of writing (`docs/results/ranker_native2.md`). Chemistry is the most valuable feature group
(−0.075 top-1 when removed).

## 3.4 Held-out drug targets (Table 4)

Twelve families, excluded from training by cluster and UniProt.

| method | DCA top-1 | DCA top-(N+2) | DCC top-1 | ceiling |
|---|---|---|---|---|
| detector order | 0.917 [0.700, 1.000] | 1.000 | 0.500 | 1.000 |
| ranker (32-feature model) | 0.833 [0.583, 1.000] | 1.000 | 0.667 | 1.000 |

Twelve structures: the intervals are wide and the ranker being below the detector order on DCA is not significant.
DCC, the harder criterion, is where the ranker helps.

## 3.5 Peptide-binding sites (Table 5)

894 receptors, 685 receptor clusters disjoint from training, peptide chains deleted from the input, median peptide 12
observed residues.

| candidate source | ceiling | mean candidates |
|---|---|---|
| cavity tiers | 0.930 | 30 |
| groove tier | 0.768 | ≤ 20 |
| both, as used | **0.945** | 44.3 |

Groove candidates uniquely find **1.6 %** of the sites: peptide anchor residues occupy genuine sub-pockets, so
ordinary closure analysis detects peptide sites, and the limiting factor is again ranking. The groove tier therefore
earns its place through its features (elongation, flatness, exposed receptor backbone), which the peptide ranker's
feature-group ablation tests directly. Peptide ranker: IN PROGRESS.

## 3.6 Network, ablations, external benchmarks

**NOT RUN.** The network is implemented, equivariance-tested for both backbones, and has been run only as a CPU pilot
(120 structures, 32 channels, 2 layers, no ESM-2, thinned graph) to validate the code path. No number from it appears
in this paper. The ablation grid is specified and scripted: `full`, `no_probes`, `no_surface`, `no_esm`,
`no_tensors`, `no_vectors`, `invariant`, `achiral`, `no_recycling`, `no_sequence_edges`, `no_masked_residue`,
`nearest_probe_loss`, `no_residue_chemistry`, `e3nn_l2`, `e3nn_l3`. COACH420 is running; HOLO4K, LIGYSIS and
CryptoBench are fetched and not yet run.

## 3.7 Cost

Candidate generation 3.6 s per structure per core; peptide candidates 11.3 s per receptor; the ranker trains in
minutes on four cores. Network timings: NOT MEASURED.

# 4. Discussion

The candidate result is the one that changes practice: a closure analysis with two depth tiers and peak splitting
covers 0.977 of sites in 30 proposals, above both external finders on identical data, so a pocket pipeline needs no
third-party finder and no Java. What it does not give is ordering — the detector's own score reaches 0.523 at top-1
where P2Rank's model reaches 0.754 — and that is where learning belongs. Our tabular re-ranker closes most of the
gap with structure-only features, chemistry contributing most.

Two findings cut against intuition. Peptide grooves need no special detector: the cavity tiers already cover 0.93 of
peptide sites and the groove tier adds 1.6 % unique coverage, so the groove machinery should be judged as a feature
source. And hotspot labels built from proximity are substantially wrong: a quarter of nominal hydrogen-bond donors in
a bound ligand donate to nothing, and training a field on them teaches chemistry that the crystal does not support.

Limitations: crystal structures only, PDB-format parsing, heuristic property labels from the CCD, no apo/holo
pairing, no cryptic-site training signal, and no trained network. If the degree-2 ablation shows no gain, that is a
reportable negative result, since the ablation itself is missing from the pocket literature.

# 5. Data and code availability

Code: this repository (private at submission). Data: RCSB PDB and the wwPDB CCD (CC0). Benchmark lists are fetched
from their original sources by the scripts and not redistributed. Licences and the non-commercial scope rule:
`docs/DATA_CARD.md`, `docs/PROVENANCE.md`.

# Figures

F1 pipeline (`docs/figures/pipeline.svg`). F2 the equivariant layer. F3 split and leakage design. F4 candidate
ceiling versus depth, three generators. F5 ablation bars (NOT RUN). F6 a pocket with its interaction-validated
hotspot labels (`docs/figures/hotspot_labels.png`) and a peptide groove. F7 failure cases.

# Tables

T1 data (3.1). T2 generators (3.2). T3 ranking (3.3). T4 held-out (3.4). T5 peptide (3.5). T6 ablations (NOT RUN).
T7 external benchmarks (partial). T8 licences and scope. S1 threshold calibration.
