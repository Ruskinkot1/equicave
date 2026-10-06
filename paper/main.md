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
candidates) run on the same structures with the same labels; on COACH420 it is 0.989 against P2Rank's 0.931. A
LambdaRank re-ranker over 32 structure-only features reaches top-1 0.701 on the COACH420 structures that are not
similar to our training set, where P2Rank reaches 0.753: the paired cluster bootstrap is −0.052 [−0.121, +0.011] on
top-1 and includes zero on top-N and top-(N+2) as well, so the two are not distinguishable. Across five benchmarks
under one protocol the ranker is worth +0.15 to +0.18 top-1 over the detector's own order on holo structures and
+0.034 on apo ones.

Three of our architectural claims are settled by the first trained network's ablation (3 seeds, seed sd 0.009), two
of them against us: removing the cavity probes costs **0.263** site top-1, an order of magnitude more than anything
else, while removing the degree-2 tensor channels costs **−0.006**, chirality +0.000, ESM-2 650M −0.010 and the
surface module −0.011 — all inside seed noise. Degree-2 attention applied to pockets is prior art; the ablation is
what the field lacks, so the measurement stands and the accuracy claim does not. The isolated equivariance arm
(local-frame scalarisation) is implemented and NOT RUN, so no equivariance effect size is claimed. Two
methodological results are reported because they invalidated numbers we had already recorded: evaluating on the
chain a benchmark row names while training on the assembly inverts the ordering of methods silently, and
cluster-split cross-validation moved *opposite* to benchmark transfer across three feature sets. Three of seven
modern deep-learning predictors can be run from published weights — GrASP, DeepPocket and DeepSurf, all three now
driven end to end in house; those runs are in progress and no number from them appears here.

# 1. Introduction

Three questions a designer asks of a structure: where does something bind, what kind of pocket is it, and where
exactly inside it would each chemical group sit. Methods answer them separately. Geometric detectors (LIGSITE,
fpocket) answer the first cheaply and with no training. Learned surface models (P2Rank, DeepSurf) rank better.
Graph and equivariant models (GrASP, EquiPocket, VN-EGNN, GDEGAN) improve segmentation. The one independent,
biological-unit-aware comparison [LIGYSIS, J. Cheminform. 2024] places the hybrid fpocket+PRANK first at 60.4 %
top-(N+2) recall and the equivariant VN-EGNN twelfth of thirteen, with 67 % of its predictions pointing at a site it
had already found; and inside VN-EGNN's own table P2Rank, a 2018 random forest, wins HOLO4K and PDBbind on DCA. The
reported advantage of equivariant detectors is concentrated in DCC at a 4 Å threshold that the same comparison argues
is too strict.

Four gaps motivate this work.

1. **Dependence on external finders.** Learned re-rankers and CNN rescorers take their candidates from fpocket or
   P2Rank, which fixes their ceiling and adds a Java runtime to every deployment. Whether a self-contained generator
   can match that ceiling has not, to our knowledge, been measured.
2. **Degree-2 features are applied but never isolated.** Tensor (l ≥ 2) channels are routine in molecular property
   prediction, and GDEGAN [arXiv:2603.19817, 2026] has since applied GotenNet with l = 2 and ESM-2 to binding-site
   prediction. What is missing is evidence that the degree is what helps: GDEGAN reports no l_max ablation, and no
   equivariant-versus-invariant comparison at equal depth, width and features exists for this task. The nearest
   adjacent measurement, EquiPNAS on protein–nucleic-acid binding, found the equivariant gain negligible. We supply
   the ablation, and it comes out null (3.6) — which makes the gap in the literature the result.
3. **Probes on a sphere.** VN-EGNN's virtual nodes start on a sphere around the protein and must learn to migrate
   into cavities. Cavities can be computed first.
4. **Hotspot labels are mostly proximity labels.** Fields of ligand-atom type (SILCS, FTMap, Fragment Hotspot Maps,
   AutoSite) are physics- or sampling-based; learned pocket fields use occupancy within 4 Å. The exception is
   PharmacoNet [Chem. Sci. 2025], which labels seven pharmacophore types from interactions PLIP actually detects, but
   with a 3D convolutional network rather than an equivariant one. A ligand atom near a point is not evidence that an
   atom of that class belongs there: the atom has to be making the interaction.

Peptide binders sit outside all of this: their sites are shallow elongated grooves, evaluated, when at all, with
small-molecule conventions.

**Contributions.** (i) A geometry-only candidate generator whose ceiling exceeds both external finders on the same
structures, so no third-party finder or Java runtime is needed at training or inference. (ii) **The ablation the
field does not have, with a null result for tensor degree**: degree-0/1/2 Cartesian channels with invariant-gated
tensor messages, ablated at equal depth, width and features on one protocol, where removing degree 2 costs −0.006
site top-1 and chirality +0.000. The contribution is the measurement, not a gain. (iii) **Probes placed in real
cavities, which the same ablation shows to be the architecture** (+0.263 site top-1, the largest effect by an order
of magnitude): a placement we did not find in the protein literature, where VN-EGNN starts its virtual nodes on a
sphere; the mechanism exists in DeepDFT for electron density. (iv) Interaction-validated hotspot labels in a learned
equivariant field, with their hand-computed geometric counterpart as ranker features. (v) **Two methodological
results about measuring pocket detection**, each of which cost us recorded numbers: evaluating on the chain a
benchmark row names while training on the assembly destroys every context feature and inverts the ordering of
methods silently, and cluster-split cross-validation does not predict benchmark transfer — across three feature sets
the two moved in opposite directions. (vi) A diagnosis of where first-rank accuracy fails, with three candidate
remedies measured and all three refuted. (vii) A homology-controlled peptide-site benchmark with a groove tier,
where the field currently has none, plus the finding that detection is not the bottleneck there. (viii) One
evaluation protocol applied to every benchmark — top-N and top-(N+2), DCC at 4, 10 and 12 Å, a redundancy statistic,
the ligand rule and its counts, train-similar structures separated — with every baseline re-run in house, and an
audit of which published methods can be run at all.

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

**Probe placement.** The probe budget is spent where a ligand atom is most likely to sit rather than uniformly:
half by a per-point ligandability model (2.4) and half at random within buriedness tiers, so negatives survive. The
point model is applied **out of fold** — a structure in fold k is scored by the model trained without fold k — so no
probe position is chosen with knowledge of that structure's own ligands. The uniform tiered placement is kept as the
comparison arm; it has NOT been run against the learned one.

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

## 2.4 Per-point ligandability, and hybrid ranking

**Per point.** Every free cavity lattice point is described by 32 rotation-invariant features — its closure, depth
and neighbour counts, the receptor chemistry in shells around it, and what interaction an atom placed there could
make — and labelled occupied if a ligand heavy atom lies within 2 Å. A LightGBM model over 19.7 M such points
(out-of-fold AUROC 0.865 over the 2.3 M points of the small manifest) scores points independently of any candidate.
The scores are then aggregated per candidate in 18 ways, including a union-find connected blob of high-scoring
points and a score-weighted centroid shift. This is the additive idea of P2Rank expressed in our own features; its
single best aggregate ranks candidates at 0.741 top-1 against the geometric score's 0.473, and it serves two roles:
probe placement (2.2) and candidate features.

**Ranking.** Up to 272 structure-only features per candidate: the candidate as generated (8), cavity shape (18),
environment composition (14), the receptor interaction partners offered in 5/8/12 Å shells (46), the
interaction-potential field of its own points (21 — the hand-computed counterpart of the learned field), the
per-point aggregates (18), ESM-2 projections, out-of-fold network scores, context (2), and 14 peptide features for
peptide targets. LightGBM LambdaRank with graded relevance (2 within 2 Å, 1 within 4 Å), within-structure z-scores
of every feature, a seed ensemble and an out-of-fold isotonic calibration. **The model that ships uses 32 of these
features**, selected on benchmark transfer rather than on cross-validated accuracy, for the reason measured in 3.8:
the richer sets score higher in cross-validation and lower on every benchmark.

## 2.5 Data and protocol

RCSB PDB (CC0) and the wwPDB CCD (CC0); splits by 30 %-identity cluster, five folds assigned per cluster, 12
held-out families excluded by cluster and UniProt accession; the peptide benchmark's receptor clusters disjoint from
the training manifest. Success DCA ≤ 4 Å with DCC reported alongside; metrics top-1, top-3, top-N, top-(N+2), MRR and
the candidate ceiling, where N is the structure's own number of ligand sites; predictions non-redundant at 6 Å;
confidence intervals are cluster bootstraps and every gain carries a paired cluster bootstrap; at least three seeds.
On external benchmarks the ligand rule (published relevant-ligand list or every drug-sized group) is stated, and
structures sharing a cluster with training are reported as a separate row.

**The receptor is always the whole deposited assembly**, at training and at evaluation, whether or not a benchmark
row names a chain. Using the named chain instead is available for reproducing earlier numbers and is why those
numbers were discarded (3.7).

**Apo structures.** CryptoBench pairs an apo entry with a holo one, and the ligand that defines the site is in the
holo entry. The holo chain is superposed onto the apo chain — Kabsch rotation without reflection over residue pairs
from a longest-common-subsequence sequence match, rejected above 4 Å RMSD — and the ligand is carried across by the
same transform. Without this the set is unmeasurable: 3 of 228 apo entries carry a ligand of their own, against 180
after the transfer. No external method is scored on this set, so the superposition affects only our own row.

Details: `docs/DATA_CARD.md`.

# 3. Results

Every number below is produced by a script in this repository and traces to a file in `docs/results/` that carries
its own protocol header. Where a measurement was taken under a protocol we later corrected, the row says so and the
number is not compared across protocols. Where a run has not happened, it says NOT RUN.

## 3.1 Data (Table 1)

| set | entries | usable | clusters | note |
|---|---|---|---|---|
| training manifest | 1499 | 1367 | 1017 | 107 are mmCIF-only at RCSB, 25 have no usable ligand |
| training manifest, full scale | 11 858 | 10 808 | — | the scale the network is trained at |
| peptide benchmark | 973 | 894 | 685 receptor clusters | 53 have fewer than 3 observed peptide residues |
| held-out targets | 12 | 12 | 11 | ligand ids verified against the deposited entries |
| COACH420 | 420 | 300 under the relevant-ligand rule | — | 179 not similar to our training manifest |
| HOLO4K | 4009 | 3448 under the relevant-ligand rule | — | 1501 not similar |
| LIGYSIS | 3448 chains | 1108 scored | — | the published chain list, Zenodo 13121414 |
| CryptoBench | 5493 (1100 test) | 228 apo entries, 180 scorable | — | 120 not similar; see 3.9 on the superposition |

Mean N over the training set is 2.27 (median 2, maximum 20); 609 of 1367 structures have a single site.

## 3.2 How far geometry alone goes (Table 2)

Same 1366–1367 structures, same ligand filter, same labels, same metric.

| generator | ceiling | candidates | top-1 by its own order |
|---|---|---|---|
| **EquiCave closure analysis** | **0.977** | 30.0 | 0.523 [0.494, 0.552] |
| fpocket 4.x | 0.955 | 37.8 | 0.391 |
| P2Rank 2.5.1 | 0.914 | 9.2 | 0.754 |

Our ceiling by depth: top-10 0.918, top-15 0.944, top-20 0.964, top-30 0.977; median best DCA 0.65 Å. A geometry-only
generator therefore does not cost coverage, which removes the dependency every comparable pipeline carries. On
COACH420 the ceiling is 0.989 against P2Rank's 0.931 on the same structures, and on the apo structures of
CryptoBench it falls to 0.850 — the only set in this paper where detection, not ranking, is the binding constraint.

## 3.3 Ranking in cross-validation, and why that is not the number to judge it by (Table 3)

5-fold cross-validation by 30 %-identity cluster, 5 seeds, 95 % cluster bootstrap, 32-feature version.

| method | top-1 | top-3 | top-N | top-(N+2) |
|---|---|---|---|---|
| detector order | 0.523 [0.494, 0.552] | 0.747 | 0.634 | 0.800 [0.777, 0.825] |
| most buried first | 0.368 | 0.695 | 0.504 | 0.762 |
| largest cavity first | 0.207 | 0.383 | 0.296 | 0.443 |
| **LambdaRank (32 features)** | **0.724 [0.701, 0.748]** | 0.857 | 0.782 | **0.877 [0.859, 0.896]** |
| without the chemistry group | 0.649 | 0.806 | 0.721 | 0.837 |
| candidate features only | 0.595 | 0.789 | 0.695 | 0.824 |

Richer feature sets score higher here — 0.787 [0.764, 0.808] at 236 features, 0.796 [0.774, 0.818] with the
per-point group at 272 — and **lower on every benchmark** (3.8). The cross-validated column measures whether a
feature group carries information; it is not evidence that the group will transfer, and in three consecutive cases
it pointed the wrong way. The model that ships is therefore the 32-feature one, chosen on benchmark transfer.

## 3.4 All five benchmarks under one protocol (Table 4)

One model (`models/ranker_native.txt`, 32 features), one protocol, the full deposited assembly as the receptor, the
set's own relevant-ligand rule. The row is the structures **not** sharing a 30 %-identity cluster with our training
manifest, except held-out, which is excluded from training by cluster and UniProt accession.

| benchmark | n | detector top-1 | + ranker top-1 | top-N | top-(N+2) | ceiling |
|---|---|---|---|---|---|---|
| held-out drug targets | 12 | 0.917 | 0.833 | 0.833 | 1.000 | 1.000 |
| HOLO4K | 1501 | 0.635 | **0.785** | 0.828 | 0.923 | 0.988 |
| COACH420 | 179 | 0.598 | 0.698 | 0.771 | 0.877 | 0.989 |
| LIGYSIS | 1108 | 0.433 | 0.610 | 0.709 | 0.821 | 0.941 |
| CryptoBench (apo) | 120 | 0.258 | 0.292 | 0.450 | 0.600 | 0.850 |

The ranker is worth +0.15 to +0.18 top-1 on the three holo benchmarks and +0.034 on CryptoBench, whose pockets are
closed in the apo form. Held-out has twelve structures: its intervals are too wide to order anything, and the
detector beating the ranker there is one structure.

## 3.5 Against external methods on identical structures (Table 5)

`scripts/eval/compare_on_set.py` intersects structure ids first and splits by train-similarity afterwards, so both
sides are scored on the same structures, the same labels and the same receptor. COACH420, the 174 structures not
similar to our training manifest:

| model | features | top-1 | top-N | top-(N+2) | predictions | ceiling |
|---|---|---|---|---|---|---|
| detector order | — | 0.598 | 0.713 | 0.822 | 30.0 | 0.989 |
| **`ranker_native`** | 32 | **0.701** | **0.776** | 0.874 | 30.0 | 0.989 |
| `ranker_native3_full` | 236 | 0.642 | 0.726 | 0.855 | 30.0 | 0.989 |
| `ranker_native3_full_points` | 272 | 0.626 | 0.704 | 0.855 | 30.0 | 0.989 |
| P2Rank 2.5.1 | — | 0.753 | 0.828 | 0.879 | 9.6 | 0.931 |

Paired cluster bootstrap of the 32-feature model against P2Rank on those structures: top-1 −0.052 [−0.121, +0.011],
top-N −0.052 [−0.121, +0.017], top-(N+2) −0.006 [−0.059, +0.046]. **All three intervals include zero**, so on the
corrected protocol we are not distinguishable from P2Rank on any of the three; at three or more sites the difference
is +0.000 on 40 structures. We do not claim to beat it, and the earlier significant top-N deficit of
−0.086 [−0.169, −0.006] was an artefact of the receptor bug in 3.7.

Three modern deep-learning methods are driven end to end in house and are running on COACH420 at the time of
writing — **GrASP**, **DeepPocket** and **DeepSurf** (3.10): **IN PROGRESS**, no number from any of them appears
here. fpocket and P2Rank are the only external methods with a completed run.

## 3.6 The trained network and its ablation (Table 6)

The first trained EquiCave-Net, 3 seeds, validation fold 0 of the training manifest, seed standard deviation about
0.009 — so a difference below roughly 0.02 is not distinguishable from seed noise. Site top-1 with one component
removed, at equal depth, width and features:

| component removed | site top-1 | effect |
|---|---|---|
| nothing (`full`) | 0.843 ± 0.009 | — |
| **cavity probes** | 0.580 ± 0.009 | **+0.263** |
| equivariance (`invariant`) | 0.803 ± 0.005 | +0.040 |
| degree-1 vector channels | 0.806 ± 0.009 | +0.037 |
| degree-2 tensor channels | 0.849 ± 0.008 | **−0.006** |
| chirality (pseudo-scalars) | 0.843 ± 0.005 | +0.000 |
| ESM-2 650M | 0.853 ± 0.003 | −0.010 |
| SAS surface module | 0.854 ± 0.002 | −0.011 |

Three of the project's claims are settled by this table, two of them against us.

**The probes are the architecture.** Removing them costs an order of magnitude more than any other component. This
is the one mechanism of the design with a large measured effect.

**Degree 2 does no measurable work on this task.** `full − no_tensors` is −0.006 against a seed sd of 0.008, and the
reduced arm is not even parameter-matched — it is *smaller* than `full`, which can only flatter the degree-2 side.
Applying l = 2 tensor attention to pockets is prior art (GDEGAN, 2026); the ablation is what the field does not
have, so the measurement stands as a result while the accuracy claim is retired.

**ESM-2, the surface module and chirality do no measurable work either.** ESM-2 650M is the most expensive component
of the whole pipeline — a frozen 650 M-parameter model, hours of embedding, a 710 MB cache — for −0.010.

**The equivariance figure of +0.040 is not reportable as an equivariance effect**, because this `invariant` arm
removes the steerable channels, the geometry from the input *and* the equivariant centre head at once.
`invariant_frames`, which scalarises the identical geometry in a local frame and isolates equivariance alone, is
implemented and **NOT RUN**. No number about equivariance appears in this paper's claims until it has been.

On the benchmarks the network beats the gradient-boosted ranker outright — COACH420 0.777 against 0.704 top-1,
HOLO4K 0.862 against 0.785 — and `network only` beats `ranker + network features` (0.777 against 0.704), the fourth
consecutive case of feature stacking losing on a benchmark while winning in cross-validation. **These three network
rows were produced on an older checkout under the single-chain receptor of 3.7**, so they are comparable with each
other (all three share that protocol) and **not** with the external rows of 3.5, which were given the whole
assembly. The re-run under the corrected receptor is the next measurement and is NOT RUN.

## 3.7 First methodological result: evaluating on the chain a benchmark names destroys context features

Every COACH420, LIGYSIS and CryptoBench row names a chain, and our evaluator passed it to the parser as a receptor
filter while the candidate tables are built on every chain. A truncated receptor moves the centroid and radius of
gyration, changes the residue and candidate counts, empties the 5/8/12 Å shells of a neighbouring chain's atoms, and
unburies every pocket at a chain interface — so centrality, the 46 shell features, the native ranks and every
within-structure z-score were computed in a context no model had been trained on. The external tools are given every
chain, so the truncation also biased the head-to-head against us.

The signature that identifies the cause rather than correlating with it: under the mismatch **every** rich ranker
scored *below* doing no ranking at all, the damage grew with how much context the feature set uses, and the
detector's own context-free score was identical in every run. Correcting the receptor reverses the ordering. Fixed
by a `--receptor-chains` option defaulting to the whole assembly; every number in 3.4 and 3.5 is post-fix, and the
pre-fix numbers for LIGYSIS and CryptoBench were discarded. HOLO4K names no chains and was never affected.

We report this because it is not specific to our code: any pipeline whose features see the environment, evaluated on
a benchmark whose rows name chains, faces the same mismatch, and the failure is silent — it produces plausible
numbers in the wrong order.

## 3.8 Second methodological result: cluster-fold cross-validation does not predict benchmark transfer

Three feature sets, one model family, one benchmark:

| features | cross-validated top-1 | COACH420 top-1 (not train-similar) |
|---|---|---|
| 32 | — (not re-measured on this table) | **0.701** |
| 236 | 0.787 [0.764, 0.808] | 0.642 |
| 272 (+ per-point ligandability) | **0.796 [0.774, 0.818]** | 0.626 |

The two columns move in opposite directions. The per-point group is the clearest case: it is the most valuable group
in the cross-validated ablation — +0.019 top-1 on the cleaned subset, worth more than the 46 shell columns — and it
**costs** 0.016 top-1 on the benchmark. Splitting folds by 30 %-identity cluster controls sequence similarity
between folds; it does not make a fold a sample of COACH420. Our manifest is built to its own criteria — resolution
limit, ligand size and type rules, a cap per cluster — so a rich model can exploit regularities every fold shares
and the benchmark does not, and the cross-validation cannot report it.

The consequence we adopt: **a feature group is accepted or rejected on benchmark transfer, not on cross-validated
top-1.** The per-point model's own out-of-fold point AUROC of 0.865 over 2.3 M cavity grid points is unaffected by
this — it is a measurement about points, not a claim about ranking transfer.

## 3.9 Where first-rank accuracy actually fails, and three remedies measured

On the COACH420 structures not similar to our training manifest our candidate set contains the answer for 0.992 of
them against P2Rank's 0.935, and we still lose top-N: we convert 72 % of our ceiling into a correct first prediction
where P2Rank converts 82 %. Detection has not been the bottleneck for some time; first-rank accuracy is. Three
explanations were measured and all three are refuted:

* **Prediction fragmentation cannot be it.** 210 of 283 structures have N = 1, where top-N *is* top-1 and no merging
  of predictions can change the result — and that is where the deficit is largest (−0.105, against −0.093 at N = 2
  and **+0.286** at N ≥ 3).
* **Mis-centring cannot be it.** Of the 53 structures whose first prediction is wrong, 4 are within 5 Å of the
  ligand and 34 are more than 8 Å away: a different pocket, not a near miss. Separately, six definitions of a
  candidate's centre all give DCC ≤ 4 Å between 0.505 and 0.577 and DCC ≤ 10 Å of 0.982 on 111 hitting candidates,
  so the low DCC at 4 Å is a property of the threshold — the cavity point cloud is larger than the ligand — and not
  of our centres. This supports the LIGYSIS recommendation to report DCC at 10–12 Å.
* **A cascade re-ranker over the top candidates is worse than one model over all of them.** The diagnosis invited
  it: the correct candidate is ranked second in 24 of 53 failures and within the first five in 41 of 53. Against the
  first stage's 0.796 top-1, re-ranking the top 3 gives 0.771 and the top 5 gives 0.766, top-(N+2) unchanged at
  0.919. The restriction discards about 97 % of the rows — and with them the easy negatives that put the hard pair
  on a scale. Kept behind a flag to be retried when the dataset is an order of magnitude larger.
* **Giving the ranker a shorter list does not help.** We emit 30 candidates where P2Rank emits 9.6, so the top-1
  decision might be easier over fewer. Restricting to the detector's own top-k gives 0.659 at k = 5 and returns to
  the unrestricted 0.698 by k = 15, while the ceiling falls from 0.989 to 0.872. The surplus candidates are not
  what it gets wrong.
* **Blending the detector's score with the ranker's does not help, and nearly passed for a result.** Swept on the
  benchmark the blend peaks at 0.721 against 0.698. Selected the honest way — on the 1367-structure out-of-fold
  cross-validation — the best weight is 1.00, the ranker alone, and every blend is worse. The benchmark peak was
  four structures of noise out of 179.
* **Telling the ranker how each candidate compares with its best competitor does nothing.** This is the cheap form
  of the site decoder's idea: besides each feature's within-structure z-score, its margin against the best other
  candidate, 96 columns instead of 64. Out of fold, 0.7286 against 0.7323, a paired cluster bootstrap of
  **−0.0037 [−0.0163, +0.0088]** — not even a cross-validation win that later fails to transfer. It does not refute
  the decoder, which learns a function of the whole list rather than a fixed comparison with one competitor, but it
  is evidence against the family, and we report it beside the decoder rather than only in its favour.

Seven remedies, seven nulls. What remains is how a candidate is scored as a whole, and the designed remedy is the
network — per-probe segmentation and confidence that score points inside the cavity instead of summarising the
pocket with features, and the site decoder of 3.6 that compares pockets against one another. Its single best
per-point aggregate already ranks candidates at 0.741 top-1 against the geometric score's 0.473. We state the
prior honestly: the margin result above is the cheap version of the decoder's hypothesis and it came out at zero.

## 3.10 Which modern methods can be run at all (Table 7)

A comparison against published deep-learning numbers is not available (the protocols differ, see below), so each
method has to be re-run here or left out. Which ones can be is a fact about the field's reproducibility and is
reported as one. Checked against primary sources on 2026-10-05 and extended on 2026-10-06:

| method | code | weights | verdict |
|---|---|---|---|
| **GrASP** (2024) | MIT | **in the repository**, `trained_models/` | **run in house** (`scripts/baselines/run_grasp.py`); COACH420 IN PROGRESS |
| **DeepPocket** (2021) | MIT | authors' SharePoint link answers 403 here; **mirrored on Zenodo** 13833813 (CC-BY-4.0) and loads into their `Model` exactly | **run in house** (`run_deeppocket.py`); its fpocket + 3D-CNN ranking works on CPU once libmolgrid comes from the `molgrid` wheel under numpy 1.x. The earlier verdict "blocked on libmolgrid" no longer holds |
| **DeepSurf** (2021) | AGPL-3.0; no licence stated for the weights | published (Google Drive, 373 MB) | **run in house** (`run_deepsurf.py` + `deepsurf_bridge.py`); needs its own Python 3.7 with TensorFlow 1.15, openbabel 2 and DMS |
| VN-EGNN (ICML 2024) | MIT | **none published** — the Zenodo record holds datasets only, and the evaluator loads a checkpoint from a Weights-and-Biases run id | not runnable without retraining |
| EquiPocket (ICML 2024) | ships as a baseline inside the VN-EGNN repository | none | not runnable without retraining; also needs MSMS |
| PointSite (2022) | repository, weights **committed** (`model/scale_80.pth`) | in the repository | its bundled SparseConvNet needs a C++20 build against current torch, which compiles but exhausts memory on this machine; deferred. It also segments binding *atoms* rather than ranking sites, so any top-N for it would depend on a clustering we would have to choose ourselves |
| GDEGAN (2026) | not located | not located | nothing to run |

Three of the seven are therefore comparable in house, against one when this project's audit was first written. The
four that are not divide into two kinds: no weights at all (VN-EGNN, EquiPocket, GDEGAN), where a comparison means
retraining on their data, and a different output type (PointSite), where it means choosing their post-processing
for them.

Their published numbers are reproduced here **only** as context, never compared with ours: DeepSurf reports
COACH420 DCA 72.1/73.3 and HOLO4K 50.1/50.6; GrASP reports COACH420 DCA top-N 77.5 / top-(N+2) 80.6 and HOLO4K
81.3/84.3; VN-EGNN reports COACH420 DCA 0.750 and HOLO4K 0.659; EquiPocket 0.656 and 0.662; GDEGAN 0.707 and 0.788.
These are not commensurable with each other, let alone with us. The sharpest evidence is a single method:
Kalasanty's HOLO4K DCA is reported as 32.1 by one paper and 61.21 by another — same method, same dataset, a 29-point
spread from protocol alone. VN-EGNN ran HOLO4K per chain and merged predictions; EquiPocket displaces its predicted
centre 4 Å outward along the atom-to-surface direction before scoring. This is why every baseline in this paper is
re-run in house or left out.

## 3.11 Peptide-binding sites (Table 8)

894 receptors, 685 receptor clusters disjoint from training, peptide chains deleted from the input, median peptide 12
observed residues.

| candidate source | ceiling | mean candidates |
|---|---|---|
| cavity tiers | 0.930 | 30 |
| groove tier | 0.768 | ≤ 20 |
| both, as used | **0.945** | 44.3 |

Groove candidates uniquely find **1.6 %** of the sites: peptide anchor residues occupy genuine sub-pockets, so
ordinary closure analysis detects peptide sites and the limiting factor is again ranking. The groove tier earns its
place through its features — elongation, flatness, exposed receptor backbone — which the peptide ranker's
feature-group ablation tests directly. The peptide ranker trains and is excluded from the claims of this paper: the
small-molecule evaluation path does not compute its features, so no benchmark number for it exists yet.

## 3.12 Cost

Candidate generation 3.6 s per structure per core; peptide candidates 11.3 s per receptor; the per-point table is
19.7 M rows at full scale; the ranker trains in minutes on four cores. Network training timings on an A100: NOT
MEASURED in a form fit to report.

# 4. Discussion

Two of this paper's results are measurements of our own design that came out against it, and we report them as the
substance rather than as caveats.

**The candidate result is the one that changes practice.** A closure analysis with two depth tiers and peak
splitting covers 0.977 of sites in 30 proposals on our manifest and 0.989 on COACH420, above both external finders
on identical data, so a pocket pipeline needs no third-party finder and no Java runtime. What it does not give is
ordering — the detector's own score reaches 0.523 at top-1 where P2Rank's model reaches 0.754 — and that is exactly
where learning belongs. Our 32-feature re-ranker closes the gap to the point where the paired interval against
P2Rank includes zero on top-1, top-N and top-(N+2); we are not distinguishable from it, and we do not claim more.

**The equivariant machinery is mostly inert on this task, and the probes are not.** Removing cavity probes costs
0.263 site top-1; removing the degree-2 tensor channels costs −0.006, chirality +0.000, ESM-2 650M −0.010 and the
surface module −0.011, all inside seed noise. The one intuition that survived contact with the measurement is the
cheap one: put the virtual nodes where a ligand atom could actually sit, which a cavity analysis can compute before
any learning happens, instead of on a sphere the network must learn to escape. The degree-2 null is worth stating
plainly because applying l = 2 attention to pockets is already in the literature without an l_max ablation, and the
nearest adjacent measurement — EquiPNAS on protein–nucleic-acid binding — found the equivariant gain negligible as
well. Our own equivariance figure remains unreported pending the local-frame arm that isolates it; the arm exists
and has not been run, and we would rather carry an open question than a number that confounds three changes.

**Two findings about measurement cost us numbers we had already written down.** Evaluating on the chain a benchmark
row names while training on the assembly destroys every context-dependent feature and silently inverts the ordering
of methods; and cross-validation split by sequence-identity cluster, the standard control in this field, moved in
the *opposite* direction to benchmark transfer across three feature sets. Both are properties of the protocol rather
than of our code, both are invisible in the output, and both are reasons to distrust any pocket comparison whose
protocol is not reported in full — which, given the 29-point spread between two papers' numbers for the same method
on the same dataset, is most of them.

**What we can and cannot claim about the state of the art.** Of seven modern deep-learning pocket predictors,
three can be run from published weights and are driven end to end here; for three the weights do not exist publicly
at all, and the fourth answers a different question (binding atoms, not ranked sites). A head-to-head against the
first three is therefore a matter of engineering, and it is in flight; against the rest it means retraining them on
their data, which is a different project. We record the audit rather than quietly comparing against numbers
measured under other protocols — which, where two papers disagree by 29 points about the same method on the same
dataset, is the only defensible choice. No number from the three runs appears in this version.

**Limitations.** Crystal structures only, PDB-format parsing, heuristic property labels from the CCD, and a
hotspot field and property head that are implemented and trained but whose benchmark evaluation is not yet
reportable. The network's benchmark numbers predate the receptor fix and must be re-measured before they can sit
beside an external baseline. The apo result is the weakest: 0.292 top-1 on CryptoBench, where the candidate ceiling
itself falls to 0.850, and cryptic sites are not in the training signal at all. The peptide tier has a benchmark and
a ranker but no evaluated number.

# 5. Data and code availability

Code: this repository (private at submission). Data: RCSB PDB and the wwPDB CCD (CC0). Benchmark lists are fetched
from their original sources by the scripts and not redistributed. Licences and the non-commercial scope rule:
`docs/DATA_CARD.md`, `docs/PROVENANCE.md`.

# Figures

F1 pipeline (`docs/figures/pipeline.svg`). F2 the equivariant layer. F3 split and leakage design. F4 candidate
ceiling versus depth, three generators. F5 ablation bars from 3.6, ordered by effect, with the seed-noise band drawn
— the figure's point is how few bars leave it. F6 a pocket with its interaction-validated hotspot labels
(`docs/figures/hotspot_labels.png`) and a peptide groove. F7 failure cases: the 34 of 53 first predictions that are
more than 8 Å from the ligand (3.9). F8 cross-validated top-1 against benchmark top-1 for the three feature sets,
which is the whole of 3.8 in one panel. F5 and F8: TO DRAW.

# Tables

T1 data (3.1). T2 generators (3.2). T3 cross-validated ranking (3.3). T4 five benchmarks (3.4). T5 against external
methods (3.5). T6 the network ablation (3.6). T7 which methods can be run (3.10). T8 peptide (3.11). T9 licences and
scope. S1 threshold calibration. S2 the feature-transfer table (3.8).
