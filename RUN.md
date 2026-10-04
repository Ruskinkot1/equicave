# How to run EquiCave

For a student or collaborator starting from a fresh clone. Every command is copy-paste; nothing needs editing.
If a step fails, the error is the deliverable: send it along with the log, do not work around it silently.

## 0. Setup (once, 5 minutes)

```bash
git clone <repo> && cd equicave
python -m venv .venv && source .venv/bin/activate     # or micromamba, see training/environment.yml
make setup                                            # installs the package, torch, e3nn, matplotlib
make test                                             # 41 tests, all must pass before anything else
```
On a GPU box prefer the pinned environment: `micromamba create -y -f training/environment.yml && micromamba activate equicave-train`.

Check what you have:
```bash
python -c "import torch; print('cuda:', torch.cuda.is_available())"
python -m training list                               # the three training tasks
python -c "from equicave import modes; print(modes.describe())"
```

## 1. Everything in one command

```bash
bash scripts/train/train_all.sh                       # CPU stages, then GPU stages if a GPU is present
```
It is restartable: each stage is skipped when its output already exists, and the two candidate builders checkpoint
every 50 structures. Useful overrides:

```bash
JOBS=8 SEEDS=5 bash scripts/train/train_all.sh                      # more cores, more seeds
STAGES="data candidates ranker" bash scripts/train/train_all.sh     # CPU part only
STAGES="net net-oof hybrid ablations" bash scripts/train/train_all.sh  # GPU part only
```

## 2. The stages, if you want them one at a time

| # | command | what it does | hardware | roughly |
|---|---|---|---|---|
| 1 | `make data` | RCSB manifest (CC0) + download structures | network | 20 min |
| 2 | `make candidates JOBS=8` | geometric candidates, 109 features, labels | CPU | 5 s per structure per core |
| 3 | `make ranker` | LambdaRank, 5-fold cluster CV, seeds, ablations, calibration | CPU | minutes |
| 4 | `make peptide-data && make peptide-ranker` | peptide benchmark and its ranker | CPU | 11 s per receptor per core |
| 5 | `make labels` | property and hotspot label statistics | CPU | minutes |
| 6 | `make net` | **the network**, fold 0 | **GPU** | hours |
| 7 | `make net-oof` | out-of-fold network features (5 models) | **GPU** | hours |
| 8 | `make ablations` | the 15-variant ablation grid, 3 seeds | **GPU** | a day |
| 9 | `make eval` | held-out, COACH420, HOLO4K under one protocol | CPU | hours |
| – | `FPOCKET=... PRANK=... make baselines` | fpocket and P2Rank on the same structures | CPU + Java | hours |

## 2b. Training at maximum scale (what to run if you have a day and many cores)

The measured results in the README come from 1367 structures and 1017 sequence clusters. Competing methods train on
roughly ten times that, and for a gradient-boosted ranker more clusters is the most reliable gain available without a
GPU. Three manifests ship with the repository:

| manifest | structures | 30 % clusters | build cost |
|---|---|---|---|
| `manifest.csv` | 1499 | 1017 | 20 min of downloads, 1.5 h of candidates on 4 cores |
| `manifest_big.csv` | 3965 | 2824 | 1 h of downloads, 4 h of candidates on 4 cores |
| `manifest_max.csv` | every matching RCSB entry | > 10 000 expected | hours of downloads, 10-20 h of candidates on 4 cores |

```bash
# the whole large-scale pipeline, one command
JOBS=16 make all-max

# or step by step
make data-max                                   # rebuild the manifest yourself (30-60 min of RCSB metadata queries)
JOBS=16 make candidates-max                     # streaming: downloads each structure, featurises, deletes it
TAG=max make esm                                # language-model features per candidate (no GPU)
make ranker-max                                 # the ranker on everything, with the ESM features and ablations
```

Three things to know before starting it.

1. **Disk.** A structure is 0.61 MB on average, so keeping 60 000 of them is about 37 GB. `--stream` (used by
   `candidates-max`) downloads each one, featurises it and deletes it, so the footprint stays at a few files. Do not
   drop `--stream` unless you have the space.
2. **It is restartable.** Rows are flushed to `data/processed/max_chunks/` every 50 structures and structures already
   present are skipped, so an interrupted run resumes where it stopped. Keep the chunk directory until the final
   table exists.
3. **Compare like with like.** A ranker trained on the maximum table must be evaluated with the same protocol as the
   small one, and the external benchmarks must be re-checked for cluster overlap with the *new* manifest
   (`scripts/eval/evaluate.py` does this automatically and reports train-similar structures separately). A gain that
   comes from new leakage is not a gain.

To hand the dataset to someone else without shipping tens of gigabytes:

```bash
python scripts/data/export_dataset.py --manifest manifest_max.csv --tag max --out dist/equicave_max.tar.gz
# on the other machine
python scripts/data/export_dataset.py --import dist/equicave_max.tar.gz && JOBS=16 make candidates-max
```
The archive holds the manifest, the geometry constants, the commit and checksums — everything that determines the
data. The import refuses to pretend it matches if the geometry constants of that checkout differ.

## 3. Training the network on a GPU

```bash
export PYTHONPATH=src:.

# one fold, three seeds, the accurate configuration
for s in 0 1 2; do
  python -m training pockets-net --config training/configs/mode_accurate.yaml \
      --out runs/training/net --device cuda --set split.val_fold=0 optim.seed=$s tag=accurate
done

# the screening configuration, for comparison
python -m training pockets-net --config training/configs/mode_fast.yaml --out runs/training/net \
    --device cuda --set split.val_fold=0 optim.seed=0 tag=fast

# the e3nn backbone instead of the hand-written one
python -m training pockets-net --config training/configs/mode_accurate.yaml --out runs/training/net \
    --device cuda --set ablation=e3nn_l3 tag=e3nn_l3

# all five folds of whichever setting won, then the features the ranker needs
for k in 0 1 2 3 4; do
  python -m training pockets-net --config training/configs/mode_accurate.yaml --device cuda --set split.val_fold=$k tag=accurate
done
python -m training pockets-net --config training/configs/mode_accurate.yaml --device cuda --set mode=oof tag=accurate
python scripts/train/train_ranker.py --tag native2 --features-extra net --seeds 5 --model models/ranker_hybrid.txt

# the ablation grid and its table
bash scripts/train/run_ablations.sh runs/training/ablations 0 3
python scripts/train/collect_ablations.py --runs runs/training/ablations --out docs/results/ablations.md
```

Each run writes `runs/training/<tag>_fold<k>_seed<s>/` with `model.pt`, `history.json`, `metrics.json` and
`model_card.json`. The first run also builds the feature cache (ESM-2 included), which later runs reuse.

## 4. Using a trained model

```python
from equicave import predict
sites = predict.predict("1m17.pdb", mode="fast")                       # ranked sites
sites = predict.predict("1m17.pdb", mode="accurate",
                        model="runs/training/net/accurate_fold0_seed0/model.pt")   # + properties and hotspots
```
As an MCP server for an agent: `PYTHONPATH=src python -m equicave.mcp_server`.
Figures: `python scripts/figures/make_figures.py`. The whole pipeline with plots: `notebooks/equicave_train.ipynb`.

## 5. Rules for reporting back

1. A number goes in a table only if a script in this repository produced it, into `docs/results/`.
2. Three seeds minimum for anything in a table; one seed is a debug run.
3. Never publish a number from a run that did not converge. Write "not run".
4. Say which mode and which commit produced each number (`git rev-parse --short HEAD`).
5. Send `model_card.json` and `history.json` with any result: they carry the configuration and the loss curves.

## 6. If something breaks

| symptom | cause and fix |
|---|---|
| `make test` fails on `test_model*` | torch or e3nn version; `pip install -U torch e3nn` and rerun |
| a build stops partway | it resumes: rerun the same command, the chunks are kept |
| `no space left on device` | delete `data/cache/net*` (rebuildable) and `data/pockets_ds/baselines` |
| P2Rank fails | it needs Java 17+; it is optional and nothing in `src/` or `training/` uses it |
| `knn_graph requires pyg-lib` | expected: graph construction uses scipy, PyTorch Geometric is not required |
| ESM-2 download fails | set `data.esm=null` to train without it; it is an ablation anyway |
