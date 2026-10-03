# EquiCave

**Self-contained, equivariant, multi-task model of protein binding pockets.** One structure in, three outputs out:

1. **Where** — ranked binding sites (own geometric candidate generator + learned re-ranker + equivariant network).
2. **What kind** — multi-label pocket property classes (ligand class, size, buriedness, polarity, charge).
3. **Where exactly** — a hotspot field: probability of a ligand atom of each class at each grid point inside the pocket.
   The field is trained on **interaction-validated** labels: a point counts as a hotspot for a class only when the
   crystal ligand atom that witnesses it really makes that interaction with the receptor (H-bond, hydrophobic contact,
   stacking, salt bridge, halogen bond), optionally restricted to drug-like ligands — not on mere proximity.
4. **Peptide binders** — a separate groove candidate tier and peptide-specific features for peptide-binding sites.

No P2Rank, no fpocket, no Java at training or inference. `src/` and `training/` depend only on numpy, scipy, pandas,
LightGBM, scikit-learn and (for the network) PyTorch. fpocket and P2Rank appear only in `scripts/baselines/` as
optional comparison methods.

## Measured so far (CPU, this repository, 2026-10-03)
| result | number | protocol |
|---|---|---|
| native candidate ceiling | **0.977** (1367 structures, mean 30.0 candidates, median best DCA 0.65 Å) | DCA ≤ 4 Å to any ligand copy with ≥ 8 heavy atoms |
| ranker top-1 | **0.724 [0.701, 0.748]** vs detector order 0.523 [0.494, 0.552] | 5-fold CV by 30 %-identity cluster, 5 seeds, cluster bootstrap |
| ranker top-(N+2) | **0.877 [0.859, 0.896]** | same |
| peptide-site candidate ceiling | 0.98 with cavity tiers, 0.80 with groove tier alone (50-complex probe) | DCA ≤ 4 Å to any peptide heavy atom, peptide chains removed from the input |
| EquiCave-Net | **not trained** (exact equivariance tests pass; CPU pilot only) | needs a GPU; see below |

Numbers from unconverged runs are never published here. Anything missing is marked "not run".

## Published numbers of the competing architectures, and our slots (empty until measured)

Success rate at **top-N**, where **N is the number of ligand sites in that structure** (one prediction per true site,
no extra guesses): a structure counts as solved when one of its first N predicted centres lies within 4 Å of a heavy
atom of a ligand (DCA ≤ 4 Å). N therefore varies per structure — N = 1 for a single-ligand structure, N = 3 for a
structure with three distinct sites — and `top-(N+2)` in our own tables allows two extra predictions, the convention
used by P2Rank and DeepPocket. The values below are **as reported in the respective papers**, which differ in ligand
filter, chain handling and splits, so they are a bar to reproduce under our protocol, **not** a comparison.
`—` = not reported. Every "EquiCave" row is filled only from `docs/results/` after a converged run.

| method | type | COACH420 top-N | HOLO4K top-N | PDBbind2020 top-N | LIGYSIS |
|---|---|---|---|---|---|
| fpocket | geometry (alpha spheres) | 0.228 | 0.312 | 0.291 | —  |
| P2Rank | random forest over SAS points | 0.728 | 0.787 | 0.826 | —  |
| DeepPocket | 3D CNN rescoring of fpocket | 0.761 | 0.561 | — | —  |
| DeepSurf | surface CNN | 0.731 | 0.635 | 0.732 | —  |
| GrASP | graph attention over atoms | ≈0.78 | — | — | reported strong  |
| EquiPocket | equivariant GNN + surface module | — | — | — | —  |
| VN-EGNN | equivariant GNN, virtual nodes on a sphere | 0.750 | 0.659 | 0.820 | —  |
| **EquiCave (candidates only)** | our geometry-only generator | not run | not run | not run | not run  |
| **EquiCave (ranker)** | our candidates + LambdaRank | not run | not run | not run | not run  |
| **EquiCave (ranker + network)** | full model | not run | not run | not run | not run  |

All published values above are quoted from memory of the papers (fpocket and P2Rank as re-reported by VN-EGNN;
DeepPocket, DeepSurf, GrASP and VN-EGNN from their own papers) and **must be rechecked against the primary sources**
before they appear in the manuscript. The reason they are listed at all is to fix the
bar we have to clear under our own protocol, in which every baseline is re-run on our splits
(`scripts/baselines/run_external.py`, `scripts/eval/compare_methods.py`).

### What we have measured ourselves, on our own split
Here N is again the structure's own number of ligand sites (mean N = 2.27, median 2, maximum 20 over these 1367 structures; 609 have a single site), so top-1 is the
strictest column: the very first prediction must hit a site.

| method, same 1367 RCSB structures and labels | top-1 | top-(N+2) | ceiling | candidates |
|---|---|---|---|---|
| detector order (geometry only) | 0.523 [0.494, 0.552] | 0.800 [0.777, 0.825] | 0.977 | 30.0 |
| most buried first | 0.368 [0.339, 0.396] | 0.762 [0.737, 0.787] | 0.977 | 30.0 |
| largest cavity first | 0.207 [0.185, 0.232] | 0.443 [0.413, 0.474] | 0.977 | 30.0 |
| **EquiCave candidates + LambdaRank** | **0.724 [0.701, 0.748]** | **0.877 [0.859, 0.896]** | 0.977 | 30.0 |
| fpocket, same structures | in progress | in progress | in progress | in progress |
| P2Rank, same structures | in progress | in progress | in progress | in progress |
| EquiCave + network features | not run (needs a GPU) | not run | — | — |

## Install
```bash
git clone <this repo> && cd equicave
make setup                 # pip install -e ".[dev]" and ".[train]"
make test                  # 26 CPU tests on synthetic data, must stay green
```
GPU machines: `micromamba create -y -f training/environment.yml && micromamba activate equicave-train`
(or `docker build -t equicave-train -f training/Dockerfile .`).

## Run the whole pipeline
```bash
export PYTHONPATH=src:.

# 1. data: RCSB manifest (CC0) with 30 %-identity clusters, 5 cluster folds, held-out families; then structures
make data                  # ~20 min, writes data/processed/manifest.csv + data/pockets_ds/pdb (ignored by git)

# 2. native candidates: 1 Å lattice, 26-ray buriedness, connected cavities, peak splitting; features and labels
make candidates JOBS=8     # ~3.6 s per structure per core

# 3. ranker: LightGBM LambdaRank, cluster 5-fold CV, 5 seeds, feature-group ablations, paired bootstrap
make ranker                # ~3 min on 4 cores, writes docs/results/ranker_native.md and models/ranker_native.txt

# 4. peptide-binding sites: own benchmark (RCSB protein-peptide complexes), merged cavity + groove candidates
make peptide-data JOBS=8
make peptide-ranker

# 5. labels and their statistics (property classes from the CCD, hotspot classes per grid point)
make labels

# 6. GPU: the equivariant multi-task network
make net                                        # one fold, config training/configs/pockets_net.yaml
make net-oof                                    # out-of-fold network features for the hybrid ranker
python scripts/train/train_ranker.py --features-extra net --seeds 5   # hybrid ranker
make ablations                                  # the full ablation grid, 3 seeds each

# 7. evaluation: one protocol for every benchmark
make eval                                       # held-out families, COACH420, HOLO4K (+ LIGYSIS, CryptoBench when fetched)

# optional: the same structures through fpocket and P2Rank, for the comparison table
FPOCKET=/path/to/fpocket PRANK=/path/to/prank make baselines
```

### What your GPU machine should run
```bash
# one fold, 3 seeds, full config (dim 128, 5 layers, ESM-2 650M)
for s in 0 1 2; do python -m training pockets-net --device cuda --out runs/training/net \
    --set optim.seed=$s split.val_fold=0 tag=full; done
# all five folds of the winning setting, then the out-of-fold features for the ranker
for k in 0 1 2 3 4; do python -m training pockets-net --device cuda --set split.val_fold=$k tag=full; done
python -m training pockets-net --device cuda --set mode=oof tag=full
bash scripts/train/run_ablations.sh runs/training/ablations 0 3
```
Each run writes `runs/training/<tag>_fold<k>_seed<s>/` with `model.pt`, `history.json`, `metrics.json` and
`model_card.json` (config, geometry constants, data scope, metrics). `scripts/train/collect_ablations.py` turns the
cards into `docs/results/ablations.md`. A Colab notebook is in `notebooks/equicave_train.ipynb`.

## Layout
- `src/equicave/` — `structure.py` (PDB reader, peptide-chain ligands), `pockets.py` (free grid, 26-ray buriedness,
  cavity mask, SAS cloud), `detect.py` (native candidates, two depth tiers), `peptide.py` (groove tier, backbone
  exposure), `pocket_features.py` (ranker features and inference), `ccd.py` (ligand chemistry from the CCD),
  `labels.py`, `metrics.py`, `targets.py`, `mcp_server.py` (MCP tools).
- `training/` — `pockets/model.py` (EquiCave-Net), `pockets/data.py` (featurisation and cache), `pockets/net_task.py`
  (training loop, EMA, early stopping, out-of-fold features), `pockets/labels_task.py`, `configs/`, `environment.yml`,
  `Dockerfile`. One command per task: `python -m training list`.
- `scripts/` — `data/` (manifests, structures, benchmark lists), `train/` (candidate tables, ranker, ablations),
  `eval/evaluate.py` (one protocol for every benchmark), `baselines/` (optional fpocket / P2Rank).
- `data/processed/` — small derived tables only (manifests, candidate features, label statistics). Raw structures and
  third-party downloads are rebuilt by the scripts and never committed.
- `docs/` — `ARCHITECTURE.md` (the network in full), `PLAN_2026.md` (month plan), `PAPER_PLAN.md`, `DATA_CARD.md`,
  `LITERATURE.md`, `PROVENANCE.md`, `results/` (every measured table).
- `.claude/skills/equivariant-gnn-researcher/` — review checklist for equivariant layers and evaluation protocol.

## Licences and data policy
Training data for the shipped model is RCSB PDB (CC0) and the wwPDB Chemical Component Dictionary (CC0).
ESM-2 weights are MIT. Third-party benchmark sets are fetched by the user under their own terms and never
redistributed here; a model trained on non-commercial data is labelled "non-commercial scope" in its model card.
See `docs/DATA_CARD.md`. The repository is private and stays private; it has no open-source licence yet.
