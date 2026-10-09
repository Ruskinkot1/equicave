# How to run EquiCave

> **Training the network on a GPU: see [`GPU_EXPERIMENTS.md`](GPU_EXPERIMENTS.md).** It has the ordered experiment
> plan, the gate that stops a wasted week, and the two rules for reading the numbers (the candidate ceiling is
> already 0.989, and cross-validated top-1 does not predict benchmark top-1).

## The whole thing, on the largest dataset, in one command

```bash
git clone <repo> && cd equicave
micromamba create -y -f training/environment.yml && micromamba activate equicave   # or: make setup
pip install -e .

SCALE=max JOBS=$(nproc) bash scripts/train/train_all.sh
```

That assembles the data, trains everything and validates it, in order: structures, the candidate table, the
per-point ligandability model, two rankers, the network feature cache, the network on three seeds, the
out-of-fold network features, the hybrid ranker, and then every benchmark under one protocol.

**Every stage is skipped when its output exists and every long stage checkpoints internally**, so if the run dies
or the machine reboots, the same command continues where it stopped. Nothing needs to be cleaned up first.

What `SCALE` costs, so the choice is made with numbers rather than optimism:

| `SCALE` | structures | 30 %-identity clusters | disk | CPU hours (16 cores) | GPU hours per seed (A100) |
|---|---|---|---|---|---|
| `small` (default) | 1 499 | 1 085 | ~6 GB | ~1 h | ~3.5 h |
| `big` | 3 965 | 2 824 | ~16 GB | ~3 h | ~10 h |
| **`max`** | **11 671** | **6 223** | **~45 GB** | **~8 h** | **~30 h** |

All three manifests are committed, so no RCSB metadata queries are needed. The script trains three network seeds,
so multiply the GPU column by three; `NET_SEEDS=1` for a first look. Without a GPU the network stages fall back to
a reduced configuration that converges but is not the publishable one, and say so; everything else is CPU work.

Useful variants:

```bash
STAGES="data candidates points ranker eval" SCALE=max bash scripts/train/train_all.sh   # skip the GPU stages
SCALE=max NET_SEEDS=1 bash scripts/train/train_all.sh                                   # one seed first
FPOCKET=/path/to/fpocket PRANK=/path/to/prank STAGES=baselines bash scripts/train/train_all.sh
```

**What is cached, and the one place that used to bite.** Structures (`data/pockets_ds/pdb`, `data/external/pdb`),
the candidate and point tables (in restartable chunks), ESM-2 embeddings (`data/cache/esm`), CCD entries and the
network feature cache (`data/cache/net*`) are all keyed by structure and skipped when present, so re-running any
stage costs nothing for work already done. The network cache is keyed by PDB id **alone**, so a changed
featurisation would have been served from the old files and the run would have measured the previous architecture
while the log claimed the new one. The settings that change the arrays are now pinned in
`<cache>/.featurisation.json` and a mismatch stops the run naming the fields that differ: delete the directory to
rebuild, or point `data.cache_dir` at a new one. Each arm that changes featurisation wants its own cache.

**Progress.** Every loop that takes more than a moment reports: the structure downloads, the RCSB metadata
queries, the CCD downloads, candidate and per-point featurisation, the network cache, each training epoch and each
benchmark. On a terminal it is a `tqdm` bar with the loop's own running numbers in it (the candidate ceiling so far,
the positive rate of the point table, the epoch's mean loss); piped to a log file it becomes one plain line every
30 seconds, because a redrawn bar in a file is unreadable. `EQUICAVE_PROGRESS=bar|plain|off` forces one of the
three — `bar` is worth setting under `tee`, where stderr may not look like a terminal. `tqdm` is a soft dependency:
without it the plain lines are used everywhere.

Two measured facts to keep in mind while reading whatever it produces, both of which cost this project a day:

* **Cross-validated top-1 on our manifest does not predict benchmark top-1.** Three feature sets ranked in the
  *opposite* order on COACH420. Judge a change on the `eval` stage.
* **The candidate ceiling is already 0.989 on COACH420.** Detection is not the bottleneck; ranking is.

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
| 2 | `make candidates JOBS=8` | geometric candidates, 118 features, labels | CPU | 5 s per structure per core |
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

# the ablation grid and its table (env vars, not positional arguments)
DRY_RUN=1 bash scripts/train/run_ablations.sh      # the plan and the cost first; the default group is `core`
bash scripts/train/run_ablations.sh                # 5 arms x 3 seeds = 15 runs
python scripts/train/collect_ablations.py --runs runs/training/ablations --out docs/results/ablations.md
```

**The grid is five arms, and that is the whole study.** `full` is the reference, `full_next` the candidate
baseline, `no_site_decoder` the one mechanism no published method has, `no_probes` the headline claim worth 0.263,
`no_tensors_matched` the ablation the field does not have. `training/configs/ablations.yaml` held 57 arms: at three
seeds that is 171 runs and about 770 GPU hours, and the expected maximum of pure noise across 57 arms at a seed sd
of 0.009 is **+0.026**, above this project's own +0.02 claim threshold — so that grid could not have supported a
claim whatever it returned. 33 arms were deleted, twelve more kept as named open questions (`GROUP=chemistry`,
`baseline`, `mechanisms`, `probes`, `architecture`), and the seven already published in the 2026-10-06 table kept
only so those rows stay reproducible.

Arms that change the featurisation the same way now **share one cache**, which is keyed by the `data.*` overrides
rather than by the arm name: `full_next` and `probe_protrusion` were each rebuilding their own copy of the same
1367 structures, over an hour apiece.

Each run writes `runs/training/<tag>_fold<k>_seed<s>/` with `model.pt`, `history.json`, `metrics.json` and
`model_card.json`. The first run also builds the feature cache (ESM-2 included), which later runs reuse.

## 3·0. One night on one GPU: what actually fits

Arithmetic first, because it decides the plan. At ~13 min per epoch on the shipped configuration, with
`warmup_epochs=6`, `epochs=30` and patience 10, a run stops around epoch 18–20, so **one run is about four
hours**. A twelve-hour night is therefore **three runs**. Nothing about agents, parallelism or tooling changes
that: the bottleneck is one card doing one structure at a time.

So pick one of these, not more:

```bash
export PYTHONPATH=src:.
# A. the reference, properly: three seeds of `full`. Gives a real seed spread and the number every
#    later arm is differenced against. This is the one to run first.
ARMS="full" SEEDS=3 bash scripts/train/run_ablations.sh

# B. the decisive question instead: one seed each of the baseline, the decoder arm and the candidate baseline
ARMS="full no_site_decoder full_next" SEEDS=1 bash scripts/train/run_ablations.sh
```

**A is the better night.** `no_site_decoder` already reads as +0.007 against `full` at one seed, which is a third
of the claim threshold, and a one-seed repeat cannot settle it — whereas without three seeds of `full` on the
current code there is no reference at all, and the one measured spread so far (0.179) was an artefact of a bug
rather than a spread.

Before starting, two checks that cost nothing and have each cost a night already:

```bash
nvidia-smi                 # another training process holding the card is the usual OOM; this run needs ~22 GiB
df -h .                    # a cache rebuild writes ~1.2 GB and the featurisation pass takes over an hour
```

The trainer now refuses to start when the card has less than `optim.need_gb` free (24 GiB by default), **before**
the featurisation pass rather than three structures into epoch 1.

## 3a. Making a network run faster without changing what it learns

Four things, in order of how much they give for how little risk. The first two are measured in this project; the
last two are not, and say so.

**0. Early stopping can no longer fire inside stage 1 — this is the reliability fix, not a speed one.** A plateau
is what stage 1 *is*, since the terms the model exists to optimise are held at zero there, so patience measured
against it ended runs before stage 2 began. Of three seeds of `full`, two reached stage 2 and scored 0.871 and
0.875 with a median centre error of 1.1 Å; the third stopped at epoch 16 of a 20-epoch warmup and reported its
stage-1 checkpoint — top-1 **0.562**, centre error **4.83 Å**. Averaged that reads as a seed standard deviation of
0.179, which would make every threshold in this project meaningless. It was one killed run.

**1. Shorten the warmup — free, and it improves the result.** `stages.warmup_epochs` was 20; the default is now 6.
Measured on a full run: site top-1 is 0.000 through the whole warmup and `occ_ap` peaks at epoch 14–15 and then
*falls*, so epochs 15–20 cost about 1.3 hours and give back a negative delta on both metrics. After the warmup,
top-1 goes from 0.551 to 0.871 in a single epoch of stage 2. At ~13 minutes per epoch that is **three hours saved
per run**, with a better model at the end.

**2. The cache is now prefetched — no effect on the maths.** The loop was `to_torch(load(file))` inside the inner
loop, which serialises reading the npz, inflating it (every array is zlib, from `savez_compressed`) and building
the tensors, all with the GPU idle, then runs the GPU step with the CPU idle. `data.prefetch` keeps the exact
order — so a seed's sequence of structures is unchanged and runs stay comparable with the measured ones — and
decompresses the next few on worker threads. How much it buys depends on how much of that second is data rather
than arithmetic, which is one command to check:

```bash
nvidia-smi --query-gpu=utilization.gpu --format=csv -l 2
```

Below ~40 % utilisation the loop is data-bound and this is the dominant fix; near 90 % it was already compute-bound
and the gain will be small.

**3. `no_recycling`, if the arm is null.** `model.recycles: 2` means the trunk runs three times per structure. The
arm that removes it is in the grid and has never been run. If it ties, that is close to a 3× saving on the trunk
for nothing — but it has to be measured, not assumed, because recycling is where the probes move to their
predicted centres and get relinked.

**4. Fewer probes, with the same caveat.** 768 probes plus 512 surface points dominate the ~1500 nodes, and the
probes are the one component the ablation values at 0.263 of site top-1 — so this is the setting least safe to
cut blind. There is no measured probe-count curve; `--set data.n_probe=512` would be roughly 1.5× faster per epoch
and needs its own arm before it is trusted.

What not to bother with: lowering `optim.accumulate` changes the optimiser step frequency, not the compute; and
`torch.compile` only pays off once several structures are batched into one padded forward pass, which is a real
change to the loop rather than a flag.

## 3c. Colab, and whether it is worth it

**Check the memory first, because it decides everything else.** A run at the shipped configuration peaked at
**22.5 GiB** (measured, from two torch OOM reports on an 80 GB A100).

| Colab GPU | memory | verdict |
|---|---|---|
| T4 (free tier) | 16 GB | **does not fit.** Only as a smoke test, with the overrides below, and the numbers are then not comparable to anything |
| L4 | 24 GB | fits with ~1.5 GiB to spare — the same thin margin that was OOMing on the A100 next to another process |
| A100 40 GB (Pro) | 40 GB | fits, the configuration this project measures |
| A100 80 GB | 80 GB | fits with room for two runs, which is still a bad idea: they contend and both slow down |

**Move the cache, do not rebuild it.** Featurising 1367 structures takes over an hour and downloading the
structures takes longer; the cache is just npz files keyed by PDB id plus `.featurisation.json`, so copy
`data/cache/net` (about 1.2 GB) and `data/processed/manifest.csv` to Drive and point the run at them. That turns a
three-hour setup into a five-minute one, and the signature file makes a mismatched cache refuse rather than
silently measure the wrong featurisation.

```python
from google.colab import drive; drive.mount('/content/drive')
D = '/content/drive/MyDrive/equicave'      # cache, manifest and runs live here, so a disconnect loses nothing
```

```bash
# a private repo needs a token; a tarball of the checkout works just as well
!git clone https://<token>@github.com/Ruskinkot1/equicave.git /content/equicave
%cd /content/equicave
!pip -q install -e ".[train]"
```

```bash
# one run, checkpointed to Drive so a dropped session resumes instead of restarting
!PYTHONPATH=src:. python -m training pockets-net \
    --config training/configs/pockets_net.yaml --device cuda \
    --out $D/runs --set ablation=full split.val_fold=0 optim.seed=0 \
    data.cache_dir=$D/cache/net data.manifest=$D/manifest.csv
```

The trainer writes `checkpoint.pt` after every epoch and resumes from it — weights, EMA, optimiser, schedule, the
RNG stream and the early-stopping counters — so re-running the same command after a disconnect continues rather
than starting over. That is what makes Colab usable here at all: a run is about four hours and a free session is
not reliably that long.

**On a 16 GB T4, only as a smoke test:**

```bash
--set model.use_tensors=false model.dim=96 data.n_probe=384 optim.need_gb=12
```

`use_tensors=false` is the single largest saving: the degree-2 edge messages are `[E, F, 3, 3]`, about 53 MiB per
tensor at 24 k edges and 128 channels, several per layer, five layers, three recycling passes, each kept by
autograd. It also costs −0.006 top-1, which is nothing — but every one of these overrides changes the
configuration, so such a run tells you the pipeline works and nothing about accuracy. Do not put its numbers in a
table.

**The notebook is `notebooks/equicave_colab.ipynb`** — fourteen cells that do one thing: check the GPU, mount
Drive, clone, point at the cache, run one arm, print the metrics including the per-class table. The launch command
it runs is this one, and it is safe to re-run after a disconnect:

```bash
PYTHONPATH=src:. python -m training pockets-net \
    --config training/configs/pockets_net.yaml --device cuda --out $D/runs \
    --set ablation=full split.val_fold=0 optim.seed=0 tag=full \
          data.cache_dir=$CACHE data.manifest=$MANIFEST
```

**Hours.** One run is **about 4 h** — 17 to 20 epochs at ~13.5 min, and that per-epoch figure was measured on an
A100 that was simultaneously running someone else's job at 89 % utilisation, so a clean card should be faster by
an unmeasured margin. Setup is **5 minutes** when the cache is on Drive, or **2–3 hours** when structures have to
be downloaded and featurised. A complete cache no longer loads ESM-2 at all, which saves 2.5 GB of download on
every run. Three seeds of `full` is therefore **12–14 h of GPU time**, i.e. more than one free session and about
one Pro session per seed.

**What Colab is actually good for here:** the CPU stages (the candidate table, the per-point model, the ranker,
the benchmark evaluation), which need no GPU and fit a session comfortably, and a single `full` run on an A100 Pro
instance while your own card is busy. What it is bad for: the grid, because 15 runs is 60 hours.

### On an Apple M-series machine

**Training the network: no.** Not because of a missing backend -- `--device mps` is plumbed and prints a warning
-- but because of two numbers. The configuration peaks at **22.5 GiB**, which on a unified-memory machine has to
come out of the same pool as the OS, so 16 GB is out and 32 GB is marginal. And the throughput is an order of
magnitude or more below a CUDA card, so a four-hour run becomes days. Several scatter and einsum paths in the
trunk may also fall back to the CPU; `PYTORCH_ENABLE_MPS_FALLBACK=1` works around that, slowly. None of this is
measured here, which is itself a reason not to spend a week finding out.

**Everything else: yes, and it is not a consolation prize.** The candidate table, the per-point model, both
rankers and the whole benchmark evaluation are CPU work, and they are where the open questions actually are --
the ranker is stuck at 0.792 for reasons that have nothing to do with the GPU. The test suite runs in about a
minute, and `training/configs/pockets_net_cpu.yaml` gives a network smoke test that converges and is not
publishable, and says so.

```bash
PYTHONPATH=src:. python -m pytest tests -q
PYTHONPATH=src:. python scripts/train/train_ranker.py --tag native2 --seeds 5 --out docs/results/native2
PYTHONPATH=src:. python scripts/eval/evaluate.py --set coach420 --ranker models/ranker_native2.txt
```

## 3b. The CatBoost arm of the ranker

The code path is exercised by a passing unit test on synthetic data (`tests/test_ranker_stages.py`,
`test_catboost_ranker_fits_and_orders`: YetiRank fits and puts the right candidate first in over 60 % of toy
structures), and the test skips where the package is absent. It has **never been run on the real candidate
table**, so treat the first real run as a smoke test as much as a measurement.

```bash
micromamba install -y catboost          # or: pip install catboost

# the reference, for the paired comparison -- same tag, same folds, same seeds
python scripts/train/train_ranker.py --tag native2 --seeds 4 --out runs/ranker_lgbm
python scripts/train/train_ranker.py --tag native2 --seeds 4 --learner catboost --out runs/ranker_cat
```

**What it is for.** Two mechanisms, both aimed at the failure this ranker actually has. *Ordered boosting*
estimates gradients on row permutations instead of on the rows the tree is then fitted to, removing the prediction
shift that standard boosting carries and that is worst on small data — and we have 1017 independent 30 %-identity
clusters against 204 columns. *Oblivious trees* use one feature and threshold per level, a much stronger
regulariser than LightGBM's free trees, which is the shape of model that keeps beating us: P2Rank's random forest
over 35 features with one dominant.

**What it is not.** "Ordered" in ordered boosting is the order of a permutation of training rows, not the order of
the pockets. Ranking is the loss function's job; for CatBoost that is `YetiRank` (set in `CAT_RANK_LOSS`), a
listwise scheme like LambdaRank and not a consequence of ordered boosting. Its other headline feature, ordered
target statistics for categorical columns, is useless here: all 204 features are numeric.

**How to judge it, and this part matters more than the run.** On a **benchmark**, not on cross-validation. Picking
the best of several learners by cross-validation is precisely the loop that gave us 0.701 top-1 at 32 features and
0.626 at 272 while cross-validation ranked the two the other way round. So:

```bash
# the only comparison that decides anything
python scripts/eval/evaluate.py --set coach420 --ranker <lgbm model> --tag _lgbm
python scripts/eval/evaluate.py --set coach420 --ranker <catboost model> --tag _cat
```

LightGBM stays the default until CatBoost wins that comparison, because every number in `docs/results` was
measured with it — including the 0.792 [0.770, 0.814] top-1 reference — and an arm with no reference is not a
measurement. If CatBoost wins, flip `LEARNER` in `scripts/train/train_ranker.py`, re-measure the published tables,
and keep `--learner lgbm` as the comparison arm. Expect a learner swap to be worth 0.005–0.02 on tabular ranking;
the gap we are chasing is 0.075 from the feature count alone, and the composition difference behind it (2.27 sites
per structure in training against 1.29 on COACH420, 44.6 % single-site against 74.7 %) is untouched by either
library.

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
