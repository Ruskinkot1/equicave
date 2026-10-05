# EquiCave architecture (full specification)

Three stages: a geometric candidate generator, an equivariant multi-task network, and a learned hybrid ranker.
Nothing in this pipeline calls an external pocket finder. Every constant named below lives in one place
(`pockets.geometry()` / `pocket_features.geometry()`) and is stored next to each trained model, so inference cannot
silently drift from training.

---

## Stage 1. Native candidate generator (`src/equicave/detect.py`)

Input: heavy-atom coordinates of the receptor (all chains, hydrogens and waters dropped). No sequence, no scores.

### 1.1 Lattice and free space
An **absolute** lattice of step `GRID = 1.0 Å` (multiples of the step, not relative to the bounding box, so the same
protein always yields the same points) covers the protein plus a 4 Å margin. One nearest-neighbour query per lattice
point against a k-d tree of heavy atoms gives `dist`. Two masks follow:

- `free = dist >= CLASH` with `CLASH = 3.0 Å` — a ligand heavy atom can sit here;
- `occupied = dist < RAY_HIT` with `RAY_HIT = 2.0 Å` — the wall a ray can hit.

### 1.2 Buriedness by 26 rays
For each of the 26 lattice directions `d` (all `(±1,0,±1)`-type vectors, normalised) a ray is traced from every point
and marked as blocked if `occupied` is met within `RAY_LEN = 8.0 Å`:

```
buried(p) = Σ_{d ∈ 26 dirs}  1[ ∃ k ≥ 1 :  k·|d|·GRID ≤ RAY_LEN  ∧  occupied(p + k·d) ]
```

On the lattice this is computed with integer array shifts rather than per-point queries, which is what makes a whole
protein take seconds: 26 directions × ⌊RAY_LEN / (|d|·GRID)⌋ shifted boolean ORs. `tests/test_detect.py` checks the
lattice form against the exact per-point ray caster `pockets._buried` (Pearson r > 0.95, mean absolute difference < 2
of 26 rays). This is the LIGSITE closure idea, implemented from the description, not from any source code.

### 1.3 Cavities, peaks and two tiers
- **Tier 1 (deep)**: `cav = free ∧ buried ≥ DETECT_MIN_BURIED (16)`. 26-connected components of `cav` are cavities.
- Each cavity is split into sub-sites: the buriedness field is smoothed (Gaussian, σ = 1.5 Å), peaks are selected
  greedily with non-maximum suppression of radius `NMS_RADIUS = 6.0 Å`, and every cavity point is assigned to its
  nearest peak. A group of at least `MIN_POINTS = 12` points is a candidate.
- **Tier 2 (shallow fill)**: if fewer than `MAX_SITES = 30` candidates were produced, the same procedure runs on
  `buried ≥ FILL_MIN_BURIED (10)`, and a shallow candidate is appended only if its centre is farther than `NMS_RADIUS`
  from every deep candidate. `tier` records which tier produced a candidate.

Per candidate: `center` = buriedness-weighted centroid, `score` = Σ buriedness over its points, plus `n_points`,
`mean_buried`, `max_buried`, parent cavity size.

Calibration on 55 structures (sweep in the session log): `min_buried` 12 → ceiling 0.891, 14 → 0.945, **16 → 0.964**,
18 → 0.927, 20 → 0.926; adding the shallow tier at 10 → 0.982. On the full set of 1367 structures the ceiling is
**0.977** at a mean of 30.0 candidates (median best DCA 0.65 Å), above the 0.944 reported for fpocket+P2Rank in the
previous stage of this project.

### 1.4 Peptide-binder grooves (`src/equicave/peptide.py`)
A peptide keeps its own backbone, lies partly in solvent and extends 10–30 Å, so the deep-cavity settings are wrong
for it. The groove tier therefore changes three things:

- **Depth**: `GROOVE_MIN_BURIED = 10` of 26 rays (and `< 26`, so fully enclosed voids are excluded).
- **Shape instead of depth**: the shallow field of a protein is one connected sheet, so a global split is meaningless.
  The smoothed field is scanned for local peaks with NMS `PEAK_NMS = 5.0 Å`; each peak grows into a candidate from the
  patch points within `SEG_LEN / 2 = 7 Å`. A candidate survives only if its longest principal extent is
  ≥ `GROOVE_MIN_LENGTH (9 Å)` and its anisotropy `ax₁/ax₃` is ≥ `GROOVE_MIN_ANISOTROPY (1.5–1.8)`. A long groove thus
  yields a chain of overlapping candidates that together cover a bound peptide.
- **Wall chemistry**: `backbone_exposure` counts receptor backbone N, O and CA and side-chain C / polar atoms within
  6 Å of every candidate point. Peptide recognition pairs backbone with backbone (β-augmentation, PDZ, SH3, MHC), so
  the backbone-to-side-chain ratio separates a peptide groove from a side-chain-lined small-molecule cavity.

`score = Σ buriedness × min(anisotropy, 6)`. `merge_with_small_molecule` produces one list with `tier ∈ {1,2,3}`, keeping
overlapping candidates (a groove and a cavity description of the same region are both useful) and recording their
mutual distance, so the ranker can learn which generator to trust for a given receptor.

Measured on the 894-receptor benchmark with the peptide chains removed from the input (`docs/results/peptide_sites.md`):
cavity tiers reach a ceiling of 0.930, the groove tier alone 0.768, both together 0.945 at 44.3 candidates. Groove
candidates uniquely find only **1.6 %** of the sites, so **the candidate generator is not the bottleneck for peptide
sites; the ranking is**, and the groove tier contributes features rather than coverage.

---

## Stage 2. EquiCave-Net (`training/pockets/model.py`)

An SO(3)-equivariant message-passing network over a **heterogeneous** graph, written from scratch in plain PyTorch
(no e3nn, no torch_geometric, no copied code). What makes it different from published pocket models: probe nodes sit on
**real cavity points** rather than on a sphere, a **surface point cloud** is a first-class node type, and every node
carries **degree-2 tensor channels** with geometric tensor attention (GotenNet-style). Degree-2 tensor attention for
pockets is *not* new as of 2026: GDEGAN (arXiv:2603.19817) applies GotenNet with l=2 and ESM-2 to this task. What is
missing from the literature, and what the ablation grid here provides, is the **controlled isolation** of that
contribution: GDEGAN has no lmax ablation, and no equivariant-versus-invariant comparison at equal depth, width and
features exists for pocket detection at all (`docs/LITERATURE_SURVEY.md`).

### 2.1 Node types and inputs (`training/pockets/data.py`)
| type | nodes | scalar features | initial vectors `vec0` |
|---|---|---|---|
| 0 `res` | one per residue at Cα | one-hot residue (21), mean b-factor z-score, neighbour count within 8 Å / 60, **ESM-2 embedding** (480-d for `esm2_t12_35M`, 1280-d for `esm2_t33_650M`, frozen) | N→Cα, C→Cα, Cα→Cβ (ideal virtual Cβ when absent) |
| 1 `probe` | ≤ 768 free cavity lattice points (`buried ≥ FILL_MIN_BURIED`, deep points sampled first) | buriedness/26, distance to protein/8, deep flag, atom counts within 4/6/8 Å (scaled) | unit vector to the nearest atom, mean direction to atoms within 6 Å, zero |
| 2 `surf` | ≤ 512 SAS points (Shrake–Rupley, vdW + 1.4 Å probe, Fibonacci sampling) | one-hot owner element (5), one-hot owner residue (21), buriedness/26 | outward normal, owner Cα→atom, zero |

Edges: k-nearest within a per-pair radius for each of the 9 ordered type pairs (e.g. probe→probe k = 8 within 4 Å,
probe→res k = 24 within 10 Å, res→res k = 16 within 12 Å); `edge_type = 3·type(src) + type(dst)` is embedded.

### 2.2 Channels
Each node carries three fields, all with `F` channels:

```
x ∈ R^{F}          scalars                       (degree 0, invariant)
V ∈ R^{F×3}        vectors                       (degree 1, V → V Rᵀ)
T ∈ R^{F×3×3}      symmetric traceless tensors   (degree 2, T → R T Rᵀ)
```

`T` is initialised as the symmetric traceless part of `V ⊗ V`. Five independent components live in a 3×3 array for
simplicity; symmetry and tracelessness are preserved by construction because every update is built from
`sym_traceless(a, b) = ½(abᵀ + baᵀ) − ⅓(a·b)I`.

### 2.3 Geometric tensor attention (one layer)
For an edge `j → i` with `r = p_i − p_j`, `d = |r|`, `u = r/d`, the **invariant stream** is assembled only from
quantities that do not change under rotation:

```
inv_ij = [ x_i , x_j , RBF(d) , E(edge_type) , ⟨V_j, u⟩ , ‖V_j‖ , uᵀ T_j u , ‖T_j‖_F ]
```

`RBF` is 32 Gaussians on [0, cutoff] multiplied by a cosine envelope, so messages vanish smoothly at the cutoff.
An MLP maps `inv_ij` to `H` attention logits and `n_gates·F` scalar gates. Attention is a softmax over the incoming
edges of each destination node (`seg_softmax`, no leakage between structures), broadcast over channels:

```
a_ij = softmax_j( MLP_att(inv_ij) )            per head, repeated over F/H channels
```

Messages, each term being an equivariant primitive multiplied by a **scalar** gate `g`:

```
Δx_i = Σ_j a_ij [ g¹⊙(W_x x_j)  +  g²⊙⟨V_j,u⟩  +  g³⊙(uᵀT_j u) ]                       (degree 0)
ΔV_i = Σ_j a_ij [ g⁴⊙(W_v V_j)  +  g⁵⊙u        +  g⁶⊙(T_j u) ]                         (degree 1)
ΔT_i = Σ_j a_ij [ g⁷⊙(W_t T_j)  +  g⁸⊙(uuᵀ − I/3) + g⁹⊙sym_traceless(V_j, u) ]         (degree 2)
```

These are the Cartesian equivalents of the Clebsch–Gordan couplings up to l = 2: `1⊗1→0` (dot), `1⊗1→2`
(sym-traceless outer product), `2⊗1→1` (tensor·vector), `2⊗1→0` (`uᵀTu`), `1⊗1→1` (cross product, chiral only).

After aggregation a **node-wise refinement** uses the invariants of the node itself (and, when `chiral=True`, four
triple products `⟨V_a × V_b, V_c⟩`, which are pseudo-scalars and flip sign under reflection) to gate equivariant
self-interactions:

```
h, g¹, g² = MLP_node([ x , ‖V‖ , ‖T‖_F , (triple products) ])
x ← x + h ,   V ← V + g¹⊙(T V) ,   T ← T + g²⊙sym_traceless(V, V)
```

Normalisation: LayerNorm on `x`; RMS normalisation per node over channels for `V` and `T` with learnable gains
(`EqNorm`), which keeps degrees comparable without breaking equivariance. Residual connections on all three fields.

**Consequence**: the network is exactly SO(3)-equivariant, and O(3)-equivariant when `chiral=False`.
`tests/test_model.py` verifies both numerically (random rotation + translation, atol 1e-4, for every head; mirror test:
the achiral model must be reflection-invariant, the chiral one must not be).

### 2.4 Heads
| head | output | loss |
|---|---|---|
| residue segmentation | per residue, 1 if a heavy atom is within 4 Å of a ligand atom | Dice + BCE (pos_weight 3) |
| probe occupancy | per cavity point, 1 if a ligand atom is within 2 Å | Dice + BCE (pos_weight 3) |
| site centre | `center = p_probe + offset`, where `offset = Σ_f w_f V_f` is read out **from the vector channels** (an equivariant vector; both invariant arms read the centre from a plain MLP instead, which is stated in the paper because that head is then not equivariant) | set loss: every true site is approached by its nearest probe within 8 Å, `mean_s min_{p near s} ‖center_p − c_s‖` |
| confidence | per probe | BCE against `1[‖center_p − nearest site‖ < 4 Å]` (detached target) |
| pocket properties | multi-label over 14 classes, from probes pooled with the site's soft membership mask (scalars and `‖V‖` concatenated) | BCE |
| direction to ligand | a second equivariant vector per probe, read out from the vector channels exactly as the offset is | cosine loss towards the nearest ligand heavy atom, on every probe within 8 Å of **exactly one** site (ambiguous probes are dropped, not supervised towards an arbitrary site). The point is supervision density: the centre set loss matches one proposal per site, so the geometric gradient reaches at most ~3 of 768 probes per structure, while this reaches one to two orders of magnitude more, for 128 parameters. Two independent pocket ablations (EquiPocket ICML 2024, GDEGAN 2026) report a gain, both concentrated in DCC — the axis where a gradient-boosted ranker over pocket-level features structurally cannot help. The target is a direction, so it rotates and is not translated by the augmentation; `tests/test_model.py::test_direction_head_is_equivariant_and_its_loss_is_invariant` asserts that, because forgetting it would leave the loss finite and the supervision meaningless |
| listwise site order | reuses the confidence logits | softmax cross-entropy over the structure's own probes, with the probes whose predicted centre lands within 4 Å of a true site as the target set. Every other head is a per-node or per-site term, so nothing else in the loss says that *this* probe should outrank *that* one in the same protein — which is exactly what top-1 measures. Capped at the 64 most confident probes, and skipped when the list has nothing to reject |
| hotspot field | 7 ligand-atom classes per probe (hydrophobic C, aromatic, HBD, HBA, cation, anion, halogen), **interaction-validated**: a class is positive only where the witnessing ligand atom really makes that contact with the receptor | focal BCE (γ = 2, α = 0.75) |

Total loss is a weighted sum (`training/configs/pockets_net.yaml`: 1.0 / 1.0 / 0.5 / 0.5 / 1.0 / 0.5).
Labels come from `equicave.labels` + `equicave.ccd`: the wwPDB Chemical Component Dictionary (CC0) gives every ligand
atom's element, formal charge, aromatic flag and bonds, from which donors, acceptors, charged groups, hydrophobic
carbons and halogens are derived without any cheminformatics toolkit, and the ligand class (nucleotide, heme, peptide,
carbohydrate, lipid, metal) comes from the component type, name and composition.

**Hotspots are interactions, not proximity.** A point is a hotspot for a class only when the ligand atom that
witnesses it makes the matching contact with the receptor: a hydrophobic carbon against a receptor hydrophobic carbon
within 4.5 Å, an aromatic atom against a receptor ring atom within 5.5 Å (or a cation within 5.0 Å), a donor against a
receptor acceptor within 3.5 Å and vice versa, a charged atom against the opposite charge within 4.0 Å, a halogen
against O or S within 3.8 Å. Receptor atoms are typed by residue and atom name (`labels.protein_atom_types`), so no
protonation inference is needed. Measured effect on one structure: of FAD's 15 nominal H-bond acceptors only 8 are
validated, and a nominal donor of a bound inhibitor that donates to nothing is dropped entirely. The unvalidated
("proximity") target is kept as `y_hot_proximity` and scored in parallel, so the choice is an ablation, and a drug-like
ligand filter (`data.druglike_only`) restricts the field to molecules a drug programme would start from.

### 2.5 Training (`training/pockets/net_task.py`)
AdamW, cosine schedule with warm-up, gradient accumulation (one structure per step, 4 steps per update), gradient
clipping 1.0, EMA of weights (0.999), early stopping on the validation fold, bf16 autocast on CUDA, random rotation
augmentation (a consistency check more than a need, since the network is equivariant). Splits are the 5 cluster folds
of the manifest. `mode=oof` trains one model per fold and writes out-of-fold `net_seg`, `net_center_conf`,
`net_hot_mean` for every native candidate, so the ranker never sees a network that saw its structure.

### 2.6 Ablations (`training/configs/ablations.yaml`)
`full`, `no_probes`, `no_surface`, `no_esm`, `no_tensors` (degree ≤ 1), `no_vectors`, `achiral`, `no_recycling`,
`no_sequence_edges`, `no_masked_residue`, `no_residue_chemistry`, `nearest_probe_loss`, `e3nn_l2`, `e3nn_l3`, and the
two invariance arms below. Same optimiser, folds and ≥ 3 seeds; the table is built by
`scripts/train/collect_ablations.py` with paired differences against `full`.

**The invariant arm is fair by construction**, which took a correction. An ablation that turns equivariance off must
change the mechanism and nothing else. The first version of the switch also discarded the initial vectors, removing
the backbone directions, the side-chain chemistry vectors and the surface normals: it measured "coordinates versus no
coordinates", the confound this literature is criticised for (EquiPocket's and VN-EGNN's GAT/GCN baselines receive no
coordinates at all, which is why their 23-point "equivariance gain" means something else). `invariant_frames` now
scalarises each node's own vectors in a local frame built from two of them by Gram-Schmidt, so the invariant model
receives the same geometry at the same depth and width, as invariant numbers instead of steerable channels;
degenerate frames fall back to the axis-aligned one, a documented approximation for nodes whose own geometry is
missing. `tests/test_model.py` asserts both halves: the arm is rotation-invariant, **and** its output changes when the
initial vectors are zeroed, so it cannot silently become blind again. `invariant_blind` keeps the old behaviour as a
separate arm, so the difference between "invariant" and "blind" is itself a reported number.

---

## Stage 2b. Per-point ligandability (`src/equicave/point_score.py`)

A separate, deliberately small model that needs no GPU and answers one question about a **single empty cavity grid
point**: would a ligand heavy atom sit here? It exists because of a measurement. On the 174 COACH420 structures that
are not similar to our training set, our candidate set contains the right answer for 0.992 of them against P2Rank's
0.935, and yet our first prediction is correct less often — we turn 72 % of that ceiling into a correct top-1 where
P2Rank turns 82 % of its own. Splitting the deficit by the structure's number of sites rules out the obvious
explanation: 210 of 283 structures have one site, where top-N *is* top-1 and no merging of predictions can change
anything, and that is where the deficit is largest. The loss is in how a candidate is scored, and the reason is
visible in Stage 3's feature list: those are means, counts and fractions over a whole cavity, so a small well-formed
sub-pocket inside a large shallow hole is averaged away. P2Rank does not have that problem because its score is a sum
over individually classified surface points (credited in `docs/PROVENANCE.md`).

**Inputs per point** (32, all of them distances, counts or fractions, so the whole stage is rotation-invariant by
construction and `tests/test_point_score.py` asserts it): the seven interaction classes of Stage 3's potential group
as a partner count within 6 Å, a distance to the nearest partner and a strict flag; the number of classes available,
the nearest partner of any class and the total partner count; the lattice buriedness of the point itself; the
distance to the nearest protein heavy atom; atom counts within 6, 8 and 12 Å; the carbon and the N/O fraction within
8 Å; and the distance to the protein centroid in units of its radius of gyration.

**Labels.** 1 when a heavy atom of a kept ligand really sits within 2 Å of the grid point. Occupancy by a real
ligand, not a derived ligandability score — the same stance as the hotspot labels, which only count a hotspot where
the interaction is actually made.

**Aggregation to a candidate** (18 features, the `points` group of Stage 3). The additive sum, which is the part that
behaves like P2Rank's score and grows with how many points look ligandable; the distribution of the scores (mean,
standard deviation, median, 90th percentile, maximum, mean of the top decile, and the count and fraction above 0.3,
0.5 and 0.7); the **largest connected high-scoring blob** by union-find over the grid, which the sum alone cannot
distinguish from the same total scattered over the cavity, and which is what a real sub-pocket looks like; the point
count; and the displacement from the candidate centre to the score-weighted centroid, which is both a feature and a
proposed re-centring (measured separately, since geometric re-centring was already shown not to help DCC).

**Training and leakage.** `scripts/train/build_points.py` collects the rows — about one second per structure, and
since measured candidates reach 155 points the default cap drops nothing, so the sums the ranker sees are exact.
`scripts/train/train_point_model.py` fits one gradient-boosted model per 30 %-identity cluster fold and applies each
to its held-out fold only, so the aggregates handed to the ranker are out of fold and no candidate is ever scored by
a model that saw its structure. A model fitted on all folds is saved separately, for structures outside the manifest.
Negatives outnumber positives about six to one and are reweighted by the square root of the ratio rather than
discarded.

The guard matters here: a ranker trained with these columns and served without a point model would read zeros where
it learned a signal, which is exactly the failure `pocket_features.check_features` was added for, so the `points`
group is **not** exempt from it and `rank_sites` raises unless it is given a point model.

---

## Stage 3. Hybrid ranker (`src/equicave/pocket_features.py`, `scripts/train/train_ranker.py`)

A candidate is described by 118 protein-agnostic numbers in six groups (no external finder scores exist in
this project), plus three optional groups supplied when their upstream model is available:

- **native** (8): `nat_score`, `nat_rank`, `nat_rel` (score / best score in the structure), `nat_npts`,
  `nat_mean_bur`, `nat_max_bur`, `nat_cavity_npts`, `nat_tier`.
- **geometry** (18): cavity volume, three principal extents, mean buriedness of a 300-point subsample, volume rank,
  centrality (‖centre − protein centroid‖ / radius of gyration), depth (distance to the nearest atom), the spread of
  the buriedness (standard deviation, 10th and 90th percentiles, the deep and mouth fractions), the width to the
  nearest atom (mean and maximum), elongation and flatness of the principal extents, and points per unit volume.
- **chemistry** (14): atom and residue composition within 8 Å — element fractions, backbone fraction, residue count,
  and the fractions of hydrophobic, aromatic, polar, positive, negative, Gly and Pro residues.
- **shell** (46): the receptor interaction partners the candidate offers in 5, 8 and 12 Å shells — donors, acceptors,
  cations, anions, aromatic ring atoms and hydrophobic carbons, as counts and as fractions of all atoms in the shell —
  plus Kyte-Doolittle hydropathy (mean and sum), b-factor mean and standard deviation, residue counts at 5 and 12 Å,
  the donor/acceptor balance, the charge balance, the polar-to-apolar ratio and donors+acceptors per unit volume.
- **potential** (30): the interaction-potential field of the candidate's own points. For each point the same geometric
  tests as the hotspot labels are asked of the *empty* point: is there a receptor acceptor nearby (so a donor placed
  here would be satisfied), a donor, a hydrophobic carbon, an aromatic ring atom, an anion or a cation, an acceptor
  for a halogen. Per interaction class the features are the number of partners within 6 Å, the distance to the
  nearest one, and a strict flag at hydrogen-bond and salt-bridge distance; on top of those, the mean number of
  classes per point, the fraction of points offering two or three classes at once, the fraction offering both a polar
  and an apolar partner, the best point, the volume of the three-class region, the nearest partner of any class and
  the total partner count. The strict flag alone was the first version of this group and had to be replaced: measured
  on a real structure every cavity point satisfied every class, because a 4.5–5.5 Å cutoff inside a protein always
  is, so 21 of the then 109 features were constant and the group could not have shown up in an ablation. This is the
  physics-free prior that separates a real site from an equally deep but chemically featureless hole, and the
  geometric counterpart of the learned hotspot field.
- **context** (2): residue count of the protein, number of candidates in the structure.
- **network** (45, optional): from the out-of-fold network — `net_seg`, `net_center_conf`, `net_hot_mean`,
  `net_n_centers`, `net_center_dist`, then at 4 Å and 8 Å around the candidate centre the mean and maximum
  occupancy and confidence, the mean predicted offset and the probe count, and each of the seven hotspot classes
  separately as mean and maximum at both radii.
- **points** (18, optional): aggregates of the learned per-point ligandability score; see Stage 2b below.
- **peptide** (14, for peptide targets): tier, length, width, anisotropy, flatness, exposed receptor backbone N, O and
  Cα counts, side-chain carbon and polar counts, backbone total, backbone-to-side-chain ratio, backbone per point, and
  the distance to the nearest candidate of the other generator.

Model: LightGBM **LambdaRank** (one query = one structure, truncation level 10, 400 rounds, lr 0.05, 31 leaves,
feature and bagging fraction 0.8, L2 1.0, label gain [0, 1, 3]) with three additions, each switchable so its
contribution appears in the ablation table:

- **graded relevance**: 2 when the centre is within 2 Å of a ligand atom, 1 within 4 Å, 0 otherwise. Ranking a centre
  that sits on the ligand above one that merely touches it is what top-1 actually rewards.
- **within-structure z-scores**: every feature is also supplied as its z-score inside its own structure, because
  pocket ranking is a comparison between the cavities of one protein ("deeper than the others here"), not an absolute
  judgement.
- **seed ensemble**: the shipped score is the mean over seeds, reported next to the per-seed mean.

Afterwards the score is mapped to a probability by an **isotonic regression fitted out of fold** (fold k is calibrated
on the other folds' out-of-fold scores), which leaves the ranking inside a structure unchanged and makes the number
readable as P(this candidate hits a ligand); the expected calibration error and Brier score are reported before and
after. Protocol: 5-fold cross-validation **by 30 %-identity cluster**, 5 seeds, 95 % bootstrap CI over clusters, and a
**paired** cluster bootstrap of every gain against the detector order. Alternatives still to test: a listwise
transformer over the candidate set (softmax CE) and stacking; the cross-validation winner is kept.

Measured with the first 32-feature version (1367 structures, 1017 clusters, 40 986 candidates, ceiling 0.977);
the 109-feature version with graded relevance, z-scores and calibration is in `docs/results/ranker_native.md`:

| method | top-1 | top-3 | top-N | top-(N+2) |
|---|---|---|---|---|
| detector order | 0.523 [0.494, 0.552] | 0.747 | 0.634 | 0.800 |
| largest cavity first | 0.207 | 0.383 | 0.296 | 0.443 |
| most buried first | 0.368 | 0.695 | 0.504 | 0.762 |
| **LambdaRank (32 native features)** | **0.724 [0.701, 0.748]** | **0.857** | **0.782** | **0.877** |
| without chemistry | 0.649 | 0.806 | 0.721 | 0.837 |
| native features only | 0.595 | 0.789 | 0.695 | 0.824 |

The chemistry group is the largest single contributor (−0.075 top-1 when removed); the native generator features alone
already beat the detector order by +0.072.

---

## Inference and serving
`pocket_features.rank_sites(model, cands, st)` returns ranked sites with `ranker_score`; `equicave.mcp_server` exposes
`detect_pockets`, `cavity_mask`, `pocket_properties` and `hotspot_field` as MCP tools, falling back to geometry-only
estimates (and saying so in `note`) when no trained network is present.
