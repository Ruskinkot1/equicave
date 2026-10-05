# Running the network experiments on one A100

The network has **never been trained**. Everything below is therefore a first baseline plus the arms that judge
four new mechanisms. The CPU side (candidates, ranker, per-point model, benchmarks) is already measured and is not
repeated here; see `RUN.md` for that.

Before anything, two facts that decide how to read every number you get:

1. **Our candidate ceiling on COACH420 is 0.989 against P2Rank's 0.931.** Detection is solved. Every arm below is
   aimed at *ranking*, and the metric that matters is top-1 / top-N, not recall.
2. **Cross-validated top-1 on our manifest does not predict benchmark top-1.** Measured three times: 32 features
   give 0.701 on COACH420, 236 give 0.642, 272 give 0.626, while cross-validation ranks them the other way round
   (`docs/results/README.md`, second finding). So **no arm is accepted on its validation-fold score.** An arm wins
   only if it wins on COACH420 under step 5.

## 0. Setup and the gate

```bash
micromamba create -y -f training/environment.yml && micromamba activate equicave
pip install -e .
PYTHONPATH=src:. python -m pytest tests -q            # must be green before booking GPU hours
python -c "import torch; print(torch.cuda.get_device_name(0), torch.__version__)"
```

```bash
make data              # manifest + structures, ~20 min, network bound
make candidates        # 118-feature candidate table, ~1.5 h on 4 cores (skip if data/processed is populated)
```

**Build the feature cache once, deliberately.** It now contains the direction target (`probe_dir`), which did not
exist before, so an old cache must be deleted or the direction loss is silently skipped — the trainer raises if it
finds a stale cache, but deleting it up front is cleaner:

```bash
rm -rf data/cache/net                                 # only if it exists from an earlier checkout
python -m training pockets-net --set data.limit=8 optim.epochs=1 stages.enabled=false \
    --out runs/smoke --device cuda                    # ~10 min: builds part of the cache and proves the path
```

## 1. The baseline, and the gate that stops a wasted week

```bash
for s in 0 1 2; do
  python -m training pockets-net --set ablation=full optim.seed=$s split.val_fold=0 \
      --out runs/base --device cuda
done
```

≈3.5 h per seed at `dim 128 / layers 5 / recycles 2` on one A100 (an estimate — no A100 has been observed for this
code; if the first seed takes more than 6 h, halve `optim.epochs` and say so in the log).

**Gate.** Read `runs/base/full_fold0_seed0/metrics.json`. If `net_sites.top1` is below **0.516** — the detector's own
ordering on the training manifest — stop and debug. Every arm below is a difference against this baseline and is
uninterpretable if the baseline is broken.

## 2. The four new mechanisms, each as a removal

Each arm is the full model minus one thing, so the baseline is the configuration that ships.

| arm | what it removes | published prior | my expectation |
|---|---|---|---|
| `no_direction_loss` | the cosine target from every probe to its nearest ligand atom (P12) | EquiPocket ICML 2024: DCC 0.428→0.319 without it; GDEGAN 2026: ≈2 % DCC | **+0.03 to +0.08 DCC top-1**, little on DCA |
| `single_stage` | the two-stage schedule; all terms on from epoch 1 (P6) | Lee/Byun/Shin: +5.2/+7.0 F1 | +0.02 to +0.06, at zero cost |
| `no_listwise` | the only term that orders probes within one structure (P2) | none for pockets | +0.01 to +0.05, wide |
| `no_probe_potential` | the 21 per-probe interaction channels and their masked head (P1) | our own ranker: the group is worth −0.021 top-1 | +0.01 to +0.04 |

```bash
for arm in no_direction_loss single_stage no_listwise no_probe_potential; do
  for s in 0 1 2; do
    python -m training pockets-net --set ablation=$arm optim.seed=$s split.val_fold=0 \
        --out runs/abl --device cuda
  done
done
python scripts/train/collect_ablations.py --runs runs/abl --baseline runs/base
```

The expectations are **my estimates, not measurements**. A null result on any of them is a publishable finding,
because the survey establishes that none of these ablations exists for pocket detection.

## 3. The two arms the paper cannot do without

These are not expected to improve accuracy. They are what makes the equivariance and degree-2 claims survive review.

```bash
for arm in invariant_frames invariant_blind no_tensors_matched no_vectors_matched e3nn_l1 e3nn_l2 e3nn_l3; do
  for s in 0 1 2; do
    python -m training pockets-net --set ablation=$arm optim.seed=$s split.val_fold=0 --out runs/abl --device cuda
  done
done
```

Two things to know when you read these:

- `invariant_frames` scalarises the *same* geometry in a local frame, so it is a fair invariant arm.
  `invariant_blind` removes the geometry as well — the usual confound in the literature. The gap between the two is
  itself a result: it quantifies how much of a published "equivariance gain" that confound can account for.
- `no_tensors_matched` widens `dim` until it has **at least** as many parameters as `full` (152 against 128,
  +3.2 %; the step in `dim` is the head count, so no closer width exists). The arm is therefore over-provisioned,
  which makes a null result conclusive: if a model with 3 % more parameters and no degree-2 channels matches
  `full`, degree 2 is not what helps. The printed percentage goes in the paper; do not describe it as exact.
- `tests/test_model.py::test_degree2_separates_what_degree1_cannot` already proves the premise on an octahedron
  against a planar hexagon (identical distances, zero mean direction, different second moment). It passes, so the
  ablation is capable of concluding.

## 4. Out-of-fold features — run this only after the configuration is settled

This is the only run that writes `net_features_<tag>.csv`, and it must use the winning stack or be repeated.
≈17 h for five folds.

```bash
python -m training pockets-net --set mode=oof cand_tag=native3 --out runs/oof --device cuda
```

## 5. The step that decides whether any of it shipped

**Read this before quoting any number.** The network must be judged two ways, and the second one is the one that
counts:

```bash
# (a) the network ranking candidates by itself, with no gradient-boosted model involved
python scripts/eval/evaluate.py --set coach420 --net runs/base/full_fold0_seed0/model.pt --jobs 8

# (b) the network's 45 out-of-fold features added to the ranker
python scripts/train/train_ranker.py --tag native3 --features-extra net --seeds 5 --ablate \
    --model models/ranker_net.txt
python scripts/eval/evaluate.py --set coach420 --ranker models/ranker_net.txt --jobs 8
```

Path (b) is how the roadmap originally intended to deliver the network, and **today's evidence is against it**:
adding candidate-level columns to the ranker improved cross-validation and hurt COACH420 every time we tried it,
three times out of three, most recently with the per-point group (+0.019 in cross-validation, −0.016 on the
benchmark). So run both, and if (b) again loses on the benchmark while winning in cross-validation, the network
ships as a **direct site scorer** — path (a) — and that negative result about feature stacking goes in the paper.

Compare against the baselines on identical structures and an identical receptor:

```bash
python scripts/eval/compare_on_set.py --set coach420 --tools p2rank fpocket --ref p2rank
```

`--receptor-chains` defaults to `all`. Do not change it: evaluating on the single chain a benchmark row names,
while the model was trained on the whole assembly, made every rich ranker score *below* doing no ranking at all,
and the external tools are given every chain. That bug cost a day; the diagnosis is in `docs/results/README.md`.

## 6. What to report back

For every arm: `metrics.json`, the model card, and the `collect_ablations.py` table with its paired cluster
bootstrap. Numbers from a run that did not converge do not get published — say it did not converge instead. Three
seeds minimum for anything that goes in the paper.

If an arm wins in cross-validation and loses on COACH420, that is a result, not a failure, and it is the most
interesting thing this project has measured so far.
