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
| P2Rank 2.5.1, same structures | 0.914 | 9.2 | — |
| fpocket 4.x, same structures | in progress | — | — |

### Ranking, same 1367 structures and labels
| method | top-1 | top-(N+2) |
|---|---|---|
| detector order | 0.523 [0.494, 0.552] | 0.800 [0.777, 0.825] |
| most buried first | 0.368 [0.339, 0.396] | 0.762 [0.737, 0.787] |
| EquiCave + LambdaRank (32 features) | 0.724 [0.701, 0.748] | 0.877 [0.859, 0.896] |
| P2Rank own ranking | 0.754 | — |
| EquiCave + LambdaRank (88 features) | rebuilding | rebuilding |
| EquiCave + network | not run (needs a GPU) | not run |

**Honest state:** our candidate ceiling is well above P2Rank's (0.977 vs 0.914), but our current re-ranker is slightly
behind P2Rank's own ranking (0.724 vs 0.754). The headroom is in ranking, which is what the richer feature set and the
network address. The network has never been trained: its equivariance tests pass, nothing else about it is measured.

### Peptide-binding sites, 973 complexes / 685 receptor clusters (clusters disjoint from training)
Peptide chains are removed from the input. Cavity candidates reach ~0.94 ceiling, groove candidates ~0.78, so for
peptide sites detection is not the bottleneck — ranking is. Full table: `docs/results/`.

### Published numbers of other methods
In `docs/LITERATURE.md`, quoted from the papers and marked as such. They are **not** comparable to the table above:
different ligand filters, chain handling and splits. The comparison we trust is `scripts/baselines/run_external.py`,
which re-runs fpocket and P2Rank on our structures with our labels.

## Install and run
```bash
make setup      # pip install -e ".[dev,train,viz]"
make test       # 30 CPU tests on synthetic data, including exact rotation equivariance
make data       # RCSB manifest (CC0) + structures          ~20 min
make candidates # native candidates, features, labels        ~3.6 s/structure/core
make ranker     # LambdaRank, cluster CV, 5 seeds, ablations ~3 min
make labels     # property and hotspot label statistics
```
GPU (the network, out-of-fold features, ablations):
```bash
make net                                      # one fold
make net-oof && python scripts/train/train_ranker.py --features-extra net --seeds 5
make ablations                                # full | no_probes | no_surface | no_esm | no_tensors | invariant | achiral
```
Peptides: `make peptide-data && make peptide-ranker`. Benchmarks: `make eval`.
Optional baselines: `FPOCKET=... PRANK=... make baselines`.
`notebooks/equicave_train.ipynb` runs the whole thing with figures; `python -m training list` lists the tasks.

## Layout
| path | contents |
|---|---|
| `src/equicave/` | `detect` (candidates), `peptide` (grooves), `pockets` (grid, buriedness, SAS), `pocket_features` (88 ranker features), `labels` (+ interaction validation), `ccd`, `metrics`, `structure`, `viz`, `mcp_server` |
| `training/` | `pockets/model.py` (EquiCave-Net), `data.py`, `net_task.py`, `labels_task.py`, `configs/`, `environment.yml`, `Dockerfile` |
| `scripts/` | `data/` (manifests, structures, benchmark lists), `train/`, `eval/`, `baselines/` |
| `data/processed/` | small tables only; structures and third-party data are rebuilt, never committed |
| `docs/` | `ARCHITECTURE.md`, `PLAN_2026.md`, `PAPER_PLAN.md`, `DATA_CARD.md`, `LITERATURE.md`, `PROVENANCE.md`, `results/` |

## Architecture in one paragraph
Candidates: 1 Å lattice, points ≥ 3 Å from heavy atoms, buriedness = how many of 26 rays hit protein within 8 Å,
connected cavities split at smoothed-buriedness peaks (6 Å NMS), two depth tiers, ≤ 30 per structure.
Network (`docs/ARCHITECTURE.md`): residue nodes with frozen ESM-2, probe nodes **on real cavity points**, SAS surface
nodes; every node carries scalar, vector and symmetric-traceless-tensor channels updated by invariant-gated geometric
tensor attention (SO(3)-equivariant, verified numerically); heads for segmentation, site centre with confidence,
properties and the hotspot field. Ranker: LightGBM LambdaRank over candidate, shape, chemistry and
interaction-partner-shell features, plus the network's scores when available.

## Data and licences
Training data: RCSB PDB and the wwPDB Chemical Component Dictionary (CC0). ESM-2 weights: MIT. Benchmark lists are
fetched from their original sources by the scripts and never redistributed here. Details and leakage controls:
`docs/DATA_CARD.md`. Scores are computational hypotheses, not measured activity. The repository is private and has no
open-source licence yet.
