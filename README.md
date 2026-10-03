# EquiCave

Self-contained model of protein binding pockets. One structure in, three outputs:

1. **Ranked binding sites** — own geometric candidate generator plus a learned re-ranker (and an equivariant network).
2. **Pocket property classes** — multi-label: ligand class, size, buriedness, polarity, charge.
3. **Hotspot field** — per grid point, the probability of a ligand atom of each chemical class. Labels are
   **interaction-validated**: a point counts only where the crystal ligand atom witnessing it really makes that
   contact with the receptor (H-bond, hydrophobic, stacking, salt bridge, halogen bond), optionally restricted to
   drug-like ligands.

Peptide binders have their own candidate tier (shallow elongated grooves) and their own benchmark.

**No P2Rank, fpocket or Java** anywhere in `src/` or `training/`; they exist only in `scripts/baselines/` for comparison.

## Measured results

All on RCSB structures split by 30 %-identity cluster, 5-fold cross-validation, 5 seeds, 95 % CI by cluster bootstrap.
Success = DCA ≤ 4 Å from a predicted centre to a ligand heavy atom. N = the structure's own number of ligand sites
(mean 2.27 here), so top-1 is the strictest column.

### Candidate generation, 1367 structures
| | ceiling | candidates | median best DCA |
|---|---|---|---|
| EquiCave detector (geometry only) | **0.977** | 30.0 | 0.65 Å |
| fpocket 4.x, same structures | 0.955 | 37.8 | — |
| P2Rank 2.5.1, same structures | 0.914 | 9.2 | — |

### Ranking, same 1367 structures and labels
| method | top-1 | top-(N+2) |
|---|---|---|
| detector order | 0.523 [0.494, 0.552] | 0.800 [0.777, 0.825] |
| most buried first | 0.368 [0.339, 0.396] | 0.762 [0.737, 0.787] |
| EquiCave + LambdaRank (32 features) | 0.724 [0.701, 0.748] | 0.877 [0.859, 0.896] |
| fpocket own ranking | 0.391 | — |
| P2Rank own ranking | 0.754 | — |
| EquiCave + LambdaRank (88 features) | rebuilding | rebuilding |
| EquiCave + network | not run (needs a GPU) | not run |

**Honest state:** our candidate ceiling is the highest of the three (0.977 against 0.955 for fpocket and 0.914 for
P2Rank), and our ranker is far ahead of fpocket's ordering (0.724 against 0.391), but still behind P2Rank's own
ranking (0.754). The headroom is therefore in ranking, which is what the richer feature set and the network address.
The network has only been run as a CPU pilot; no trained network number exists yet.

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
In `docs/LITERATURE.md`, quoted from the papers and marked as such. They are **not** comparable to the table above:
different ligand filters, chain handling and splits. The comparison we trust is `scripts/baselines/run_external.py`,
which re-runs fpocket and P2Rank on our structures with our labels.

## Install and train

```bash
make setup                              # pip install -e ".[dev,train,viz]"
bash scripts/train/train_all.sh         # the whole pipeline; GPU stages are skipped automatically on a CPU
```
`train_all.sh` is restartable (each stage is skipped when its output exists) and takes
`JOBS=`, `SEEDS=`, `RUNS=`, `NET_CFG=` and `STAGES=` overrides:

| stage | what it does | needs |
|---|---|---|
| `data` | RCSB manifest (CC0) and structures | network, ~20 min |
| `candidates` | native candidates, 88 features, labels | CPU, ~3.6 s/structure/core |
| `ranker` | LambdaRank, cluster CV, 5 seeds, ablations | CPU, minutes |
| `peptide` | peptide benchmark, groove candidates, peptide ranker | CPU |
| `labels` | property and hotspot label statistics | CPU |
| `net` | EquiCave-Net, fold 0, 3 seeds | **GPU** |
| `net-oof` | out-of-fold network features | **GPU** |
| `hybrid` | ranker with the network's scores | CPU |
| `ablations` | full / no_probes / no_surface / no_esm / no_tensors / invariant / achiral | **GPU** |
| `baselines` | fpocket and P2Rank on the same structures | `FPOCKET=`, `PRANK=`, Java |
| `eval` | held-out, COACH420, HOLO4K with one protocol | CPU |

```bash
STAGES="net net-oof hybrid ablations" JOBS=8 bash scripts/train/train_all.sh   # a GPU box
FPOCKET=/opt/fpocket PRANK=/opt/p2rank/prank STAGES=baselines bash scripts/train/train_all.sh
```
Individual stages are also `make` targets (`make candidates`, `make ranker`, `make net`, ...) and
`python -m training list` shows the three training tasks. `notebooks/equicave_train.ipynb` runs everything with figures.

## Layout
| path | contents |
|---|---|
| `src/equicave/` | `detect` (candidates), `peptide` (grooves), `pockets` (grid, buriedness, SAS), `pocket_features` (88 ranker features), `labels` (+ interaction validation), `ccd`, `metrics`, `structure`, `viz`, `mcp_server` |
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
