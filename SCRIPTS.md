# Every script in this repository

Run everything from the repository root with `export PYTHONPATH=src:.`, or through `make <target>`.
`bash scripts/train/train_all.sh` chains the whole pipeline; the table below is for running one step at a time.

## Data

| script | what it does | output |
|---|---|---|
| `scripts/data/build_manifest.py` | picks training structures from the RCSB PDB (X-ray, ≤ 2.5 Å, protein-only, drug-sized ligand), groups them by 30 %-identity cluster, assigns 5 cluster folds, removes the held-out families | `data/processed/manifest.csv`, `heldout_targets.json` |
| `scripts/data/build_peptide_manifest.py` | the peptide benchmark: protein-peptide complexes, receptor clusters disjoint from training (`--disjoint-from`) | `data/processed/manifest_peptide.csv` |
| `scripts/data/fetch_structures.py` | downloads the PDB files of any manifest, parallel and resumable | `data/pockets_ds/pdb/` (not in git) |
| `scripts/data/fetch_eval_sets.py` | fetches the benchmark lists: COACH420, HOLO4K (+ their `mlig` variants), LIGYSIS (from its pandas pickle, restricted unpickler), CryptoBench (from OSF) | `data/external/eval_sets/*.csv` |
| `scripts/data/check_targets.py` | verifies every held-out target's ligand id exists in its deposited entry; exits non-zero if not | stdout |

## Candidates and features

| script | what it does | output |
|---|---|---|
| `scripts/train/build_native.py` | geometric candidates for every manifest structure, 118 features, DCA/DCC and labels; checkpoints every 50 structures and resumes | `candidates_native.csv.gz`, `structures_native.csv` |
| `scripts/train/build_points.py` | one row per cavity grid point with its 32 ligandability features and occupancy label; restartable | `data/processed/points_<tag>.csv.gz` |
| `scripts/train/build_peptide.py` | cavity + groove candidates for the peptide benchmark, with the peptide feature group; peptide chains removed from the input | `candidates_peptide.csv.gz` |

## Training

| script | what it does | output |
|---|---|---|
| `scripts/train/train_ranker.py` | LambdaRank over candidates: 5-fold CV by cluster, several seeds, graded relevance, within-structure z-scores, seed ensemble, out-of-fold isotonic calibration, feature-group ablations | `docs/results/ranker_<tag>.md/.json`, `models/ranker_<tag>.txt` |
| `scripts/train/train_point_model.py` | the per-point ligandability model: one gradient-boosted model per cluster fold, out-of-fold AUROC/AP with a permutation control, and the candidate aggregates the ranker consumes | `models/point_<tag>*.txt`, `data/processed/point_features_<tag>.csv.gz`, `docs/results/point_model_<tag>.md/.json` |
| `scripts/train/train_properties.py` | the tabular baseline the network's property head must beat: one classifier per property class, out-of-fold, per-class AUROC/AP with cluster bootstrap, ECE, permutation control | `docs/results/properties_<tag>.md/.json` |
| `python -m training pockets-net` | **the network**: builds the feature cache, trains the multi-task model on four folds, validates on the fifth, keeps the best EMA weights; `--set mode=oof` trains one model per fold and writes the ranker's network features | `runs/training/<tag>_fold<k>_seed<s>/` |
| `python -m training pockets-labels` | property and hotspot label statistics, including what fraction of each hotspot class survives interaction validation | `labels_sites.csv`, `labels_summary.json` |
| `python -m training pockets-ranker` | thin wrapper that runs `train_ranker.py` with config overrides | as `train_ranker.py` |
| `scripts/train/run_ablations.sh` | the ablation grid (15 variants × seeds) | `runs/training/ablations/` |
| `scripts/train/collect_ablations.py` | turns the model cards of an ablation run into one table with paired differences against `full` | `docs/results/ablations.md` |
| `scripts/train/train_all.sh` | every stage in order, restartable, GPU stages skipped automatically on a CPU | all of the above |

## Evaluation

| script | what it does | output |
|---|---|---|
| `scripts/eval/evaluate.py` | one protocol for every benchmark (`--set heldout\|coach420\|holo4k\|ligysis\|cryptobench`): own candidates, ranker, network, DCA and DCC, top-1/3/N/(N+2), non-redundant predictions, train-similar structures as a separate row | `docs/results/eval_<set>.md/.json` |
| `scripts/eval/compare_on_set.py` | us against external tools on one benchmark, intersected to the structures both predicted, split by train-similarity and by the structure's number of sites, with paired cluster bootstraps | `docs/results/compare_<set>.md/.json` |
| `scripts/eval/compare_methods.py` | the paired table: native, fpocket and P2Rank candidates on the same structures, each with and without the learned ranker | `docs/results/methods_comparison.md/.json` |

## Baselines (optional, never imported by `src/` or `training/`)

| script | what it does |
|---|---|
| `scripts/baselines/run_external.py` | fpocket or P2Rank on our structures, written in our table layout with our labels |
| `scripts/baselines/run_competitors.py` | the same plus DeepPocket, DeepSurf, GrASP and VN-EGNN through wrapper scripts with a fixed JSON contract (`--suffix _probe` when testing on a few structures) |
| `scripts/baselines/wrappers/*.sh` | templates for those wrappers: `<wrapper> protein.pdb out.json`, each method's environment stays inside its own wrapper |

## Figures and serving

| script | what it does |
|---|---|
| `scripts/figures/make_figures.py` | the pipeline schematic plus real renders: candidate overview, the candidate at the ligand, a buriedness slice, interaction-validated hotspot labels, the ranker bar chart |
| `python -m equicave.mcp_server` | serves `detect_pockets`, `cavity_mask`, `pocket_properties` and `hotspot_field` as MCP tools |
| `notebooks/equicave_train.ipynb` | the whole pipeline with figures and the benchmark comparison table |

## Library entry points

```python
from equicave import predict, modes
modes.describe()                                  # fast / accurate / peptide
predict.predict("1m17.pdb", mode="fast")          # ranked sites
predict.predict("1m17.pdb", mode="accurate", model="runs/training/net/accurate_fold0_seed0/model.pt")
```
