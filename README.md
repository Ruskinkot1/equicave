# EquiCave

Self-contained model of protein binding pockets. One structure in, three outputs:

Three operating modes (`equicave.modes`): **fast** for screening thousands of structures, **accurate** for a single
target you are designing against, **peptide** for peptide binders.

1. **Ranked binding sites** — own geometric candidate generator plus a learned re-ranker (and an equivariant network).
2. **Pocket property classes** — multi-label: ligand class, size, buriedness, polarity, charge.
3. **Hotspot field** — per grid point, the probability of a ligand atom of each chemical class. Labels are
   **interaction-validated**: a point counts only where the crystal ligand atom witnessing it really makes that
   contact with the receptor (H-bond, hydrophobic, stacking, salt bridge, halogen bond), optionally restricted to
   drug-like ligands.
4. **A site report** (`equicave.site_report`) — what a user acts on rather than what a benchmark scores: a docking
   box, the maximum inscribed radius, the lining residues, a ligand-type guess, and the hotspot field reduced to a
   **spatially resolved pharmacophore map** (seven classes per probe, not one label per pocket).

Peptide binders have their own candidate tier (shallow elongated grooves) and their own benchmark.

**No P2Rank, fpocket or Java** anywhere in `src/` or `training/`; they exist only in `scripts/baselines/` for comparison.

## Measured results

All on RCSB structures split by 30 %-identity cluster, 5-fold cross-validation, 5 seeds, 95 % CI by cluster bootstrap.
Success = DCA ≤ 4 Å from a predicted centre to a ligand heavy atom. N = the structure's own number of ligand sites
(mean 2.27 here), so top-1 is the strictest column.

### Settled on 2026-10-08

**A second stage over the top candidates is closed, not deferred.** 1367 structures, 40 986 candidates, 5-fold
cluster CV, 4 seeds. All six configurations stay below the single stage, and the column budget — the cause the
earlier failure named — acts in the predicted direction inside the cascade without recovering it:

| method | top-1 | top-N | top-(N+2) |
|---|---|---|---|
| one stage, LambdaRank over 204 features | **0.792 [0.770, 0.814]** | 0.833 | 0.903 |
| cascade over the top 3, 20 columns | 0.785 [0.761, 0.808] | 0.824 | 0.906 |
| cascade over the top 5, 20 columns | 0.786 [0.763, 0.810] | 0.830 | 0.901 |
| cascade over the top 5, 40 columns | 0.775 [0.752, 0.799] | 0.822 | 0.901 |

The published cascade gain (+7 to +14 Top-n, PRANK over fpocket) comes from putting a learned stage over a
*geometric* one whose candidate ceiling is 80.78 % on COACH420; ours is already learned over a list with a ceiling
of 0.977. The intervention is "geometry to learned", not "learned to learned twice".

**Two channels bounded before they were added.** A metal sits within 5 Å of 13.4 % of correct candidates against
1.8 % of decoys (7.3×, lifting a candidate's chance of being correct from a 6.9 % base rate to 35.5 %), but the AUC
of that distance alone is **0.544** because 87 % of correct candidates have no metal in range — so the arm caps
near 0.034 top-1. Long-range atom counts, by contrast, are **not** redundant with our closure field: single-count
correct-against-decoy AUC is 0.650 at 8 Å and **0.837 at 12 Å** against 0.755 for the 26-direction closure, and a
logistic model on our existing geometric inputs goes **0.7999 → 0.8469** (5-fold CV) when counts at 10, 12 and 15 Å
are added. Reproduced by `scripts/eval/probe_geometry_headroom.py`.

**The network ablation reference is void for new comparisons.** The table below (2026-10-06) stands as published,
but `full` must be re-measured before any new arm is differenced against it: that run predates the site decoder by
three hours and the checkpoint-selection fix by a day, and the old numbers were chosen by a rule that cannot see
site detection. See `docs/results/PREREGISTERED.md`, amendment (b).

### Candidate generation, 1367 structures
| | ceiling | candidates | median best DCA |
|---|---|---|---|
| EquiCave detector (geometry only) | **0.977** | 30.0 | 0.65 Å |
| fpocket 4.x, same structures | 0.955 | 37.8 | — |
| P2Rank 2.5.1, same structures | 0.914 | 9.2 | — |

### Ranking on the leakage-free subset (432 structures, 383 clusters, 118-feature table)
Cross-validated by 30 %-identity cluster, five seeds. Feature-group ablations, each the full model minus one group:

With the per-point ligandability group added (272 columns, top-1 0.806, calibrated 0.810, top-(N+2) 0.946):

| group removed | top-1 | cost |
|---|---|---|
| nothing (full model) | 0.806 [0.768, 0.842] | — |
| **points (18)** | 0.787 [0.747, 0.827] | **−0.019** |
| shell (46) | 0.789 [0.749, 0.830] | −0.017 |
| potential (30) | 0.793 [0.754, 0.833] | −0.013 |
| chemistry (14) | 0.800 [0.760, 0.837] | −0.006 |
| native (8) | 0.804 [0.767, 0.843] | −0.002 |
| geometry (18), context (2) | 0.808 | −0.000 |

The per-point group is the most valuable of the seven, and its 18 columns are worth more than the 46 shell columns.
The interaction-potential group, worth nothing at all before it was redefined — its features were availability flags
that a 4.5–5.5 Å cutoff inside a protein always satisfies, so 21 of the then 109 columns were constant and no
ablation could have shown it — is now third. Dropping the points group reproduces the independent 118-feature run
(0.787 against 0.788), which is the consistency check on the two runs.

> **These are cross-validation figures and they do not predict benchmark performance.** The same 272-feature model
> reaches 0.626 top-1 on COACH420, below the 32-feature model's 0.701, while cross-validation ranks them the other
> way round. Cluster-fold cross-validation measures whether a feature group carries information, not whether it
> transfers; see the second finding in `docs/results/README.md` before quoting any number in this section.

Earlier rows below were measured with the 109-feature table and are kept for comparison.
Structures sharing a 30 %-identity cluster with COACH420, HOLO4K, LIGYSIS, CryptoBench or a held-out family are
removed from training *and* from this cross-validation, so nothing here is memorised from a benchmark.

| method | top-1 | top-N | top-(N+2) | MRR |
|---|---|---|---|---|
| detector order | 0.516 [0.465, 0.565] | 0.641 | 0.812 | 0.654 |
| most buried first | 0.352 | 0.525 | 0.766 | 0.549 |
| LambdaRank, 109 features + z-scores | 0.782 [0.740, 0.823] | 0.835 | 0.925 | 0.848 |
| the same, seed ensemble | 0.792 [0.750, 0.832] | 0.845 [0.808, 0.880] | 0.928 [0.903, 0.953] | 0.853 |
| **+ ESM-2 features (288 columns)** | **0.809 [0.770, 0.847]** | **0.856 [0.820, 0.890]** | **0.928 [0.903, 0.953]** | **0.867** |

The language-model features are worth **+0.017 top-1 and +0.011 top-N** over the same model without them, which
matches the literature's finding that protein-language-model features buy more than any single architectural choice.
They are aggregated over the residues lining each candidate and reduced by a PCA fitted out of fold, so a tree model
can use them; no GPU and no trained network is involved.

Candidate ceiling 0.988. Calibration: expected calibration error 0.004 after isotonic regression against 0.034 for a
plain sigmoid of the score, with mean predicted 0.1805 against a base rate of 0.1806. The gain over the detector
order is +0.275 top-1 [+0.228, +0.329] by paired cluster bootstrap.

### Ranking, same 1367 structures and labels (earlier 32-feature model, retained for comparison)
| method | top-1 | top-(N+2) |
|---|---|---|
| detector order | 0.523 [0.494, 0.552] | 0.800 [0.777, 0.825] |
| most buried first | 0.368 [0.339, 0.396] | 0.762 [0.737, 0.787] |
| EquiCave + LambdaRank (32 features) | 0.724 [0.701, 0.748] | 0.877 [0.859, 0.896] |
| fpocket own ranking | 0.391 | — |
| P2Rank own ranking | 0.754 | — |
| EquiCave + LambdaRank (118 features) | rebuilding | rebuilding |
| **EquiCave network alone** | **0.777** [0.716, 0.837] | **0.927** [0.889, 0.962] |

### COACH420 head-to-head, identical structures and identical receptors (`scripts/eval/compare_on_set.py`)
174 structures predicted by both methods and not sharing a 30 %-identity cluster with our training manifest; both
sides see the full assembly, which is what the candidate table is built from and what the baseline wrappers pass to
the external tools.

| method | predictions | ceiling | top-1 | top-N | top-(N+2) |
|---|---|---|---|---|---|
| detector order, no ranker | 30.0 | **0.989** | 0.598 | 0.713 | 0.822 |
| **EquiCave + ranker (32 features)** | 30.0 | **0.989** | 0.701 [0.630, 0.770] | 0.776 [0.712, 0.839] | 0.874 [0.821, 0.924] |
| EquiCave + ranker (236 features) | 30.0 | **0.989** | 0.642 | 0.726 | 0.855 |
| EquiCave + ranker (272, + per-point) | 30.0 | **0.989** | 0.626 | 0.704 | 0.855 |
| P2Rank 2.5.1 | 9.6 | 0.931 | 0.753 [0.688, 0.814] | 0.828 [0.771, 0.881] | 0.879 [0.830, 0.926] |
| EquiCave network alone † | 30.0 | — | **0.777** [0.716, 0.837] | **0.804** [0.746, 0.862] | **0.927** [0.889, 0.962] |

† The network's row is **not comparable to P2Rank's** and is marked so deliberately: it was produced on an older
checkout under the single-chain receptor, while P2Rank is given the whole assembly. It *is* comparable to our own
ranker rows measured under that same protocol, where the ranker reached 0.704 — so the network beats the
gradient-boosted ranker by +0.073 top-1, and beats `ranker + network features` (0.704) by the same margin, which is
the fourth consecutive case of feature stacking losing on a benchmark while winning in cross-validation. Re-running
the network under the corrected receptor is the next measurement; until then no network-versus-P2Rank claim is made.

Paired cluster bootstrap, our best model against P2Rank: top-1 −0.052 [−0.121, +0.011], top-N −0.052 [−0.121,
+0.017], top-(N+2) −0.006 [−0.059, +0.046]. **All three include zero**: on this protocol the two are not
distinguishable on any of the three metrics. At three or more sites per structure they are equal (+0.000 on 40
structures).

**What is still not symmetric in this table**, stated because a reader should not have to find it:

1. **Leakage control applies to us only.** The row excludes structures sharing a 30 %-identity cluster with *our*
   training manifest. P2Rank's own training set (CHEN11 and others) is not under our control and its overlap with
   COACH420 is not measured here, so we pay a leakage penalty the baseline does not. The direction of that bias is
   against us, but it is a bias either way.
2. **The prediction budget is not equalised.** We emit 30 predictions per structure and P2Rank 9.6. This does not
   affect top-1 or top-N, where N is set by the structure, but it does inflate our **ceiling** relative to theirs —
   0.989 against 0.931 is partly a budget difference and should not be read as a pure detection advantage. A
   ceiling at a matched budget has not been measured; `--merge-radii` exists for it.
3. **Only fpocket and P2Rank are actually run by us.** No modern deep-learning predictor is in this table.
   DeepPocket, DeepSurf, GrASP and VN-EGNN have wrappers in `scripts/baselines/` and have never been executed, so
   the strongest claim this table supports is parity with a strong 2018 random-forest method — not with the state
   of the art.

**The first trained ablation (3 seeds, seed sd 0.009) settles what the design is actually made of**, and three
of its parts turn out not to matter: removing the cavity probes costs **+0.263** site top-1, the degree-1 vector
channels **+0.037**, while degree-2 tensors cost **−0.006**, chirality **+0.000**, ESM-2 650M **−0.010** and the
SAS surface module **−0.011**. The probes are the architecture; the degree-2 channels, the language model, the
surface module and chirality do no measurable work at this scale. Full table and the confounds in
`docs/results/README.md`.

**Honest state.** Detection is not the bottleneck: our candidate set contains the answer for 0.989 of these
structures against P2Rank's 0.931. Ranking is, and two results frame it:

1. **An earlier version of this table was wrong.** Evaluation was reading only the chain each benchmark row names
   while the model was trained on the whole assembly, so centrality, the 46 shell features, the native ranks and the
   z-scores were all served in a context the model had never seen — and the baselines were given the full assembly.
   Under that mismatch every rich ranker scored below doing no ranking at all. Fixed; `docs/results/README.md`
   carries the diagnosis and the evidence.
2. **More features give better cross-validation and worse benchmark transfer.** 32 features reach 0.701 here,
   236 reach 0.642, 272 reach 0.626 — while cross-validated top-1 moves the other way, 0.787 to 0.796. Splitting
   folds by 30 %-identity cluster controls sequence similarity; it does not make a fold a sample of COACH420. Feature
   groups are therefore accepted on benchmark transfer in this project, not on cross-validated top-1.

### Held-out drug targets (12 families, excluded from training by cluster and UniProt)
| method | DCA top-1 | DCA top-(N+2) | DCC top-1 | ceiling |
|---|---|---|---|---|
| detector order | 0.917 [0.700, 1.000] | 1.000 | 0.500 | 1.000 |
| ranker (32-feature model) | 0.833 [0.583, 1.000] | 1.000 | 0.667 | 1.000 |

Twelve structures only, so the intervals are wide and the ranker being below the detector order here is not
significant. The DCC column (centre within 4 Å of the ligand *centroid*) is the harder criterion and the one the
ranker improves.

### Peptide-binding sites, 973 complexes / 685 receptor clusters (clusters disjoint from training)
Peptide chains are removed from the input. Cavity candidates reach ~0.94 ceiling, groove candidates ~0.78, so for
peptide sites detection is not the bottleneck — ranking is. Full table: `docs/results/`.

### Published numbers of other methods
In `docs/LITERATURE.md` and `docs/LITERATURE_SURVEY.md`, quoted from the papers and marked as such. They are **not**
comparable to the table above, and the survey shows how badly: fpocket, a fixed deterministic program, is reported at
56.4 %, 35.1 % and 22.8 % on "COACH420" by three different papers, because the ligand filter and chain handling
differ. 40 % of HOLO4K and 56 % of COACH420 structures even differ in chain count between the asymmetric and the
biological unit, and one 2026 paper excluded COACH420 outright because most of it was already in its training data.
So we re-run every baseline ourselves (`scripts/baselines/run_external.py`) and never copy a published row.

That chain-count observation is not hypothetical: it is exactly what broke our own COACH420 numbers for a day.
Evaluating on the single chain each benchmark row names, while the model was trained on the whole assembly, moved
the protein centroid, emptied the shell features of a neighbouring chain's atoms and unburied every interface
pocket — and every richer ranker then scored *below* doing no ranking at all. The measurement is in
`docs/results/README.md`. If a difference of this size can come from one line of receptor handling, a published row
measured under an unstated chain convention cannot be compared with ours at all.

Protocol consequences we adopt from the independent LIGYSIS comparison: report top-N **and** top-(N+2), report DCC at
4 Å **and** 10-12 Å (4 Å is too strict for large ligands, and the reported advantage of equivariant detectors is
concentrated in DCC), report a **redundancy statistic** (67 % of one published method's predictions pointed at a site
it had already found), state the ligand rule and the resulting counts, and separate train-similar structures.

## Install and train

```bash
git clone <repo> && cd equicave
micromamba create -y -f training/environment.yml && micromamba activate equicave
pip install -e .

SCALE=max JOBS=$(nproc) bash scripts/train/train_all.sh
```

One command assembles the largest dataset, trains every stage and validates it: structures, the 118-feature
candidate table, the per-point ligandability model, two rankers, the network feature cache, the network on three
seeds, the out-of-fold network features, the hybrid ranker, then every benchmark under one protocol. Each stage is
skipped when its output exists and each long stage checkpoints internally, so a run that dies continues from the
same command.

| `SCALE` | structures | clusters | disk | CPU hours (16 cores) | GPU hours per seed (A100) |
|---|---|---|---|---|---|
| `small` (default) | 1 499 | 1 085 | ~6 GB | ~1 h | ~3.5 h |
| `big` | 3 965 | 2 824 | ~16 GB | ~3 h | ~10 h |
| **`max`** | **11 671** | **6 223** | **~45 GB** | **~8 h** | **~30 h** |

All three manifests are committed. Without a GPU the network stages fall back to a reduced configuration that
converges but is not publishable, and say so; every other stage is CPU work. Details and per-stage commands in
[`RUN.md`](RUN.md); the GPU experiment plan in [`GPU_EXPERIMENTS.md`](GPU_EXPERIMENTS.md).

### One run, and the ablation grid

```bash
export PYTHONPATH=src:.

# a single arm; without --out the directory is stamped with the launch time
python -m training pockets-net --config training/configs/pockets_net.yaml   --set ablation=full --set split.val_fold=0 --set optim.seed=0   --set stages.warmup_epochs=6 --set optim.epochs=30 --device cuda

DRY_RUN=1 GROUP=baseline SEEDS=3 bash scripts/train/run_ablations.sh   # the plan and the cost
GROUP=baseline SEEDS=3 bash scripts/train/run_ablations.sh             # full, full_next, noise_aug, probe_protrusion
ARMS="full no_site_decoder" SEEDS=3 bash scripts/train/run_ablations.sh
```

Arms live in `training/configs/ablations.yaml` (57 of them) and groups in `run_ablations.sh`: `baseline`, `inputs`,
`architecture`, `decoder`, `probes`, `equivariance`, `mechanisms`, `capacity`, `scale`, `chemistry`, or `all`.
`full` is always added, because every arm is a difference against it. An arm that overrides any `data.*` key gets
its own feature cache automatically — the cache is keyed by PDB id, so sharing one would measure the wrong
featurisation. `RESUME=1` continues the most recent dated grid instead of starting one; a run whose `metrics.json`
exists is skipped.

Two settings decide comparability and should be fixed once for a whole grid. The **schedule**: the default here is
the shortened one (`warmup_epochs=6`, `epochs=30`), and a full-schedule run measured site top-1 at 0.000 through
the entire 20-epoch warmup while `occ_ap` peaked at epoch 14–15 and fell by epoch 20 — so the long warmup costs
time and gives back nothing. `EXTRA=""` restores it, but then the arm is not comparable with arms run under the
default. The **selection criterion**: `net_top1` 0.5 / `occ_ap` 0.25 / `res_ap` 0.25, which matters because the
ranking terms buy ordering with dense accuracy (`occ_ap` falls monotonically through stage 2 while top-1 rises).

## Layout
| path | contents |
|---|---|
| `src/equicave/` | `detect` (candidates), `peptide` (grooves), `pockets` (grid, buriedness, SAS), `pocket_features` (ranker features, probe potentials, metals, screened Coulomb, protrusion, conservation), `labels` (+ interaction validation), `site_report` (the user-facing object and the pharmacophore map), `calibration` (temperature scaling and the sum-of-squares ranking it can move), `conservation`, `ccd`, `metrics`, `structure`, `viz`, `mcp_server` |
| `training/` | `pockets/model.py` (EquiCave-Net), `data.py`, `net_task.py`, `labels_task.py`, `configs/`, `environment.yml`, `Dockerfile` |
| `scripts/` | `data/` (manifests, structures, benchmark lists), `train/`, `eval/`, `baselines/` |
| `data/processed/` | small tables only; structures and third-party data are rebuilt, never committed |
| `docs/` | `ARCHITECTURE.md`, `PLAN_2026.md`, `PAPER_PLAN.md`, `DATA_CARD.md`, `LITERATURE.md`, `PROVENANCE.md`, `results/` |

## Architecture

![EquiCave pipeline](docs/figures/pipeline.svg)

Full specification with every constant and loss: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

### 1. Candidate generation from lattice closure

The receptor is embedded in an absolute cubic lattice of step 1 Å (multiples of the step, so the same protein always
yields the same points). One nearest-neighbour query per lattice point against the heavy atoms gives a distance field
`d(p)`, from which two masks follow: *free* points with `d(p) ≥ 3.0 Å`, where a ligand heavy atom could sit, and
*wall* points with `d(p) < 2.0 Å`, which a ray can hit. Closure of a free point is then

&nbsp;&nbsp;&nbsp;&nbsp;`b(p) = Σ_{k=1..26} 1[ ∃ t ≥ 1 : t·|e_k|·1Å ≤ 8 Å ∧ wall(p + t·e_k) ]`,

the number of the 26 lattice directions `e_k` along which protein is met within 8 Å. On the lattice this is a sum of
26 shifted boolean ORs rather than 26 × 8 per-point queries, which is why a whole protein takes about 3.6 s on one
core; the lattice form is verified against an exact ray caster (`tests/test_detect.py`: r > 0.95, mean absolute
deviation < 2 of 26 rays). Cavities are the 26-connected components of `b(p) ≥ 16`, each split into sub-sites at peaks
of the Gaussian-smoothed closure field with 6 Å non-maximum suppression; every cavity point is assigned to its nearest
peak, a group of ≥ 12 points is a candidate, its score is `Σ b(p)` and its centre the closure-weighted centroid. If
fewer than 30 candidates result, a second tier at `b(p) ≥ 10` fills the remaining slots with shallower cavities whose
centres are farther than 6 Å from every deep candidate. Peptide grooves form a third tier: closure ≥ 10, local peaks
with 5 Å suppression, each grown to the patch points within 7 Å and kept only if it is elongated (longest principal
extent ≥ 9 Å, anisotropy `ax₁/ax₃ ≥ 1.5`), scored by `Σ b(p) · min(anisotropy, 6)`.

Thresholds were calibrated once on 55 structures before any ranking model existed (ceiling 0.891 / 0.945 / **0.964** /
0.927 / 0.926 for closure 12 / 14 / 16 / 18 / 20, and 0.982 with the shallow tier added) and then frozen; they are
recorded next to every trained model and checked at load time, so inference cannot drift from training.

### 2. EquiCave-Net: an SO(3)-equivariant multi-task network

The graph has three node types: **residues** at Cα with frozen ESM-2 embeddings, one-hot identity, b-factor z-score and
local density; **probes** on free cavity lattice points, carrying their own closure, depth and neighbour counts;
**surface** points sampled on the solvent-accessible surface (Shrake–Rupley, vdW + 1.4 Å), carrying the element and
residue of the atom they belong to. Edges are k-nearest within a per-pair radius for each of the nine ordered type
pairs, and the pair identity is embedded. Placing probes on real cavity points, rather than on a sphere around the
protein as VN-EGNN does, means every virtual node starts where a ligand atom could actually be.

Each node carries three fields of `F` channels: scalars `x ∈ ℝ^F` (degree 0), vectors `V ∈ ℝ^{F×3}` (degree 1,
transforming as `V ↦ V Rᵀ`) and symmetric traceless tensors `T ∈ ℝ^{F×3×3}` (degree 2, `T ↦ R T Rᵀ`), initialised from
backbone directions, probe-to-atom directions and surface normals. For an edge `j → i` with `r = p_i − p_j`, `d = |r|`,
`u = r/d`, the invariant stream contains only rotation-invariant quantities,

&nbsp;&nbsp;&nbsp;&nbsp;`inv_ij = [ x_i, x_j, RBF(d), E(type), ⟨V_j, u⟩, ‖V_j‖, uᵀ T_j u, ‖T_j‖_F ]`,

from which an MLP produces attention logits (softmaxed over the incoming edges of each node) and scalar gates. The
messages use only equivariant primitives, each multiplied by such a gate:

&nbsp;&nbsp;&nbsp;&nbsp;`Δx_i = Σ_j a_ij [ g¹⊙W_x x_j + g²⊙⟨V_j,u⟩ + g³⊙(uᵀT_j u) ]`
&nbsp;&nbsp;&nbsp;&nbsp;`ΔV_i = Σ_j a_ij [ g⁴⊙W_v V_j + g⁵⊙u + g⁶⊙T_j u ]`
&nbsp;&nbsp;&nbsp;&nbsp;`ΔT_i = Σ_j a_ij [ g⁷⊙W_t T_j + g⁸⊙(uuᵀ − I/3) + g⁹⊙sym₀(V_j, u) ]`

where `sym₀(a,b) = ½(abᵀ + baᵀ) − ⅓(a·b)I`. These are the Cartesian equivalents of the Clebsch–Gordan couplings up to
`l = 2` (`1⊗1→0`, `1⊗1→2`, `2⊗1→1`, `2⊗1→0`), so no spherical-harmonic machinery or external equivariance library is
needed. A node-wise refinement then gates equivariant self-interactions (`V ← V + g⊙T V`, `T ← T + g⊙sym₀(V,V)`) by the
node's own invariants and, optionally, by pseudo-scalar triple products `⟨V_a × V_b, V_c⟩`, which change sign under
reflection: with them the network is SO(3)- but not O(3)-equivariant, so chirality is visible. Scalars are
layer-normalised, vector and tensor channels RMS-normalised per node. Equivariance is verified numerically for every
head (random rotation and translation, tolerance 1e-4), together with a mirror test that the achiral variant is
reflection-invariant and the chiral one is not.

Four input groups are implemented but **off by default**, each with its own ablation arm, its own feature cache
and a prediction recorded before it ran (`docs/results/PREREGISTERED.md`): metals and cofactor metals with
crystallographic occupancy as the confidence (`read_pdb` reads the polymer only, so until 2026-10-08 every ion in
every training structure was discarded before featurisation, while 26.5 % of our sites have one within 8 Å); a
Debye–Hückel screened-Coulomb term from formal charges, whose potential and magnitude enter as scalars and whose
**field enters as a degree-1 channel**, so a rotated structure gives a rotated field; per-probe MSA conservation
with the coverage of the lining by the alignment, so a thin alignment is not read as low conservation; and
long-range atom counts at 10, 12 and 15 Å — P2Rank's protrusion, its most important feature by a factor of six,
which our probe channels stopped short of at 8 Å. Metals exclude the ligand codes being predicted, because on a
benchmark where an ion is itself a site an unfiltered channel reads the label.

Two architectural switches are likewise off by default: `het_mp` gives each of the nine ordered type pairs its own
invariant gain per stream (indexed by the pair and not the destination, because a destination-only gain factors out
of the attention-weighted sum and would be a per-type scaling of the node update rather than a message function),
and `use_edge_type: false` is the removal that asks whether type awareness does anything at all.

Evaluation emits two things no paper in this field reports. A **calibration block**: the occupancy head's
temperature fitted on a hash-split half of the validation structures and reported on the other half, with the ECE
before and after, the reliability curve, and the sum-of-squares site ranking recomputed under the temperature —
calibration cannot reorder a ranking by maximum or mean, but sum of squares is not monotone-invariant, and it is
the default aggregator. And **per-class metrics** over the 14 property classes, with counts only below ten
structures, because an aggregate hides exactly the heterogeneity that is often the finding.

Five heads share the trunk: residue and probe segmentation (Dice + BCE), a site centre read out as an **equivariant
vector offset** from `V` and trained with a set loss over all sites of the structure (each true site is approached by
its nearest probe within 8 Å) plus a confidence head, a 14-class multi-label property head over probes pooled by site
membership, and the 7-class hotspot field (focal BCE). Ablation switches isolate every design decision at equal depth
and width: probes, surface module, ESM-2, tensor channels (degree ≤ 1), vector channels, full invariance
(distances only) and chirality.

### 3. Hybrid ranking

Each candidate is described by 109 numbers, none of them protein-family specific: the candidate as the generator made
it (8), the shape of its cavity (18), the atom and residue composition of its environment (14), the receptor
interaction partners it offers in 5, 8 and 12 Å shells (46: donors, acceptors, cations, anions, aromatic ring atoms and
hydrophobic carbons, as counts and fractions, with hydropathy, b-factor and balance ratios), the **interaction
potential of its own points** (21: per cavity point, which interaction a ligand atom placed there could actually make,
summarised over the cavity) and two context numbers. Out-of-fold network scores join them when a trained model exists.
A LightGBM LambdaRank model consumes these with graded relevance (2 within 2 Å of a ligand atom, 1 within 4 Å),
within-structure z-scores of every feature, and a seed ensemble; the cross-validation winner among LambdaRank, a
listwise transformer and stacking is kept, followed by isotonic calibration.

## Data and licences
Training data: RCSB PDB and the wwPDB Chemical Component Dictionary (CC0). ESM-2 weights: MIT. Benchmark lists are
fetched from their original sources by the scripts and never redistributed here. Details and leakage controls:
`docs/DATA_CARD.md`. Scores are computational hypotheses, not measured activity. The repository is private and has no
open-source licence yet.
