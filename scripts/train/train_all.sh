#!/usr/bin/env bash
# EquiCave: clone the repository, run this, get a trained model and its validation tables.
#
#   bash scripts/train/train_all.sh                   # the committed 1499-structure manifest
#   SCALE=max bash scripts/train/train_all.sh         # the 11 671-structure manifest (see the cost table below)
#   SCALE=big JOBS=16 bash scripts/train/train_all.sh
#   STAGES="data candidates ranker" bash scripts/train/train_all.sh
#
# Every stage is skipped when its output already exists, and every long stage checkpoints internally, so the script
# is restartable: if it dies, or the machine reboots, run the same command again and it continues.
#
# SCALE picks the dataset. All three manifests are committed, so no RCSB metadata queries are needed.
#
#   SCALE   structures  clusters   disk      CPU hours (16 cores)   GPU hours per seed (A100, 60 epochs)
#   small        1 499     1 085   ~6 GB        ~1 h                  ~3.5 h
#   big          3 965     2 824   ~16 GB       ~3 h                  ~10 h
#   max         11 671     6 223   ~45 GB       ~8 h                  ~30 h
#
# The CPU hours cover structure download, the candidate table, the per-point model and the ranker; the GPU hours
# cover the network only, and the script trains three seeds, so multiply. Disk includes the structures, the
# candidate tables, the network feature cache and the ESM-2 embedding cache.
#
# Two things worth knowing before a long run, both measured in this project rather than assumed:
#   * Cross-validated top-1 on our manifest does NOT predict benchmark top-1. Three feature sets ranked in the
#     opposite order on COACH420 (docs/results/README.md). Judge a change on the `eval` stage, not on the CV table.
#   * The candidate ceiling is already 0.989 on COACH420, so detection is not the bottleneck. Ranking is.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src:.
PY=${PY:-python3}
JOBS=${JOBS:-$( (nproc 2>/dev/null || echo 4) )}
SEEDS=${SEEDS:-5}
NET_SEEDS=${NET_SEEDS:-3}
SCALE=${SCALE:-small}
STAGES=${STAGES:-"deps data candidates points ranker cache net net-oof hybrid eval"}
DS=data/processed
RUNS=${RUNS:-runs/training}
has() { [[ " $STAGES " == *" $1 "* ]]; }
say() { printf "\n=== %s (%s)\n" "$1" "$(date -u +%H:%M:%S)"; }

case "$SCALE" in
  small) MANIFEST=$DS/manifest.csv;     TAG=native3; CACHE=data/cache/net ;;
  big)   MANIFEST=$DS/manifest_big.csv; TAG=big;     CACHE=data/cache/net_big ;;
  max)   MANIFEST=$DS/manifest_max.csv; TAG=max;     CACHE=data/cache/net_max ;;
  *) echo "SCALE must be small, big or max (got '$SCALE')" >&2; exit 2 ;;
esac
[[ -f $MANIFEST ]] || { echo "$MANIFEST is missing; it should be committed. Rebuild with scripts/data/build_manifest.py" >&2; exit 2; }
N_STRUCT=$(( $(wc -l < "$MANIFEST") - 1 ))

if $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  NET_CFG=${NET_CFG:-training/configs/pockets_net.yaml}; DEVICE=cuda
else
  NET_CFG=${NET_CFG:-training/configs/pockets_net_cpu_real.yaml}; DEVICE=cpu
  cat >&2 <<'MSG'
No CUDA device. The network stages will use the reduced CPU configuration, which converges but is NOT the
publishable one: the full config measures at about 100 s per structure on four CPU cores, which is thousands of
hours. Everything else in this script -- candidates, the per-point model, the ranker, the benchmarks -- is CPU work
and is unaffected.
MSG
fi

printf 'SCALE=%s: %s structures from %s, tag %s, cache %s\n' "$SCALE" "$N_STRUCT" "$MANIFEST" "$TAG" "$CACHE"
printf 'device %s, %s CPU jobs, %s ranker seeds, %s network seeds\n' "$DEVICE" "$JOBS" "$SEEDS" "$NET_SEEDS"
printf 'free disk: %s\n' "$(df -h . | awk 'NR==2{print $4}')"

if has deps; then
  say "0. environment and tests"
  $PY -c "import numpy, scipy, pandas, lightgbm, sklearn; print('cpu stack ok')"
  $PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || echo "torch missing: network stages unavailable"
  $PY -m pytest tests -q -x
fi

if has data; then
  say "1. structures for $MANIFEST"
  $PY scripts/data/fetch_structures.py --manifest "$MANIFEST" --jobs 8
fi

if has candidates; then
  say "2. candidate table: geometry, 118 features, labels (restartable in chunks)"
  [[ -f $DS/candidates_$TAG.csv.gz ]] || \
    $PY scripts/train/build_native.py --manifest "$MANIFEST" --tag "$TAG" --jobs "$JOBS" --chunk 50 --keep-chunks
fi

if has points; then
  say "3. per-point ligandability: the point table, then the out-of-fold model and its aggregates"
  [[ -f $DS/points_$TAG.csv.gz ]] || \
    $PY scripts/train/build_points.py --manifest "$MANIFEST" --tag "$TAG" --jobs "$JOBS" --chunk 100 --keep-chunks
  [[ -f $DS/point_features_$TAG.csv.gz ]] || $PY scripts/train/train_point_model.py --tag "$TAG"
fi

if has ranker; then
  say "4. ranker: LambdaRank, cluster CV, $SEEDS seeds, feature-group ablations"
  # Two models, because which feature set transfers to a benchmark is an open question in this project and the
  # smaller set has so far won there. The eval stage reports both; do not pick one on the CV table alone.
  $PY scripts/train/train_ranker.py --tag "$TAG" --seeds "$SEEDS" --ablate --model "models/ranker_${TAG}.txt"
  $PY scripts/train/train_ranker.py --tag "$TAG" --seeds "$SEEDS" --ablate --features-extra points \
      --model "models/ranker_${TAG}_points.txt" --out docs/results/"${TAG}"_points
fi

if has cache; then
  say "5. network feature cache (probes, surface, ESM-2, labels) -- separate stage so a failure here is cheap"
  $PY -m training pockets-net --config "$NET_CFG" --out "$RUNS/cache" --device "$DEVICE" \
      --set "data.manifest=\"$MANIFEST\"" "data.cache_dir=\"$CACHE\"" "optim.epochs=0"
fi

if has net; then
  say "6. EquiCave-Net, fold 0, $NET_SEEDS seeds (device $DEVICE)"
  for s in $(seq 0 $((NET_SEEDS - 1))); do
    $PY -m training pockets-net --config "$NET_CFG" --out "$RUNS/net" --device "$DEVICE" \
        --set "data.manifest=\"$MANIFEST\"" "data.cache_dir=\"$CACHE\"" "split.val_fold=0" "optim.seed=$s" "tag=full"
  done
  echo "GATE: read $RUNS/net/full_fold0_seed0/metrics.json. If net_sites.top1 is below the detector's own"
  echo "ordering (0.52 on the small manifest), stop and debug -- every ablation is a difference against this."
fi

if has net-oof; then
  say "7. out-of-fold network features for the hybrid ranker (5 folds)"
  $PY -m training pockets-net --config "$NET_CFG" --out "$RUNS/net-oof" --device "$DEVICE" \
      --set "data.manifest=\"$MANIFEST\"" "data.cache_dir=\"$CACHE\"" "mode=oof" "cand_tag=\"$TAG\"" "tag=full"
fi

if has hybrid; then
  say "8. hybrid ranker: candidate features plus the network's scores"
  $PY scripts/train/train_ranker.py --tag "$TAG" --seeds "$SEEDS" --features-extra net \
      --model "models/ranker_${TAG}_hybrid.txt" --out docs/results/"${TAG}"_hybrid || \
    echo "skipped: net_features_full.csv not found (stage net-oof did not finish)"
fi

if has ablations; then
  say "9. ablation grid, $NET_SEEDS seeds each (GPU hours: multiply the per-seed cost by the arm count)"
  bash scripts/train/run_ablations.sh "$RUNS/ablations" 0 "$NET_SEEDS"
fi

if has baselines; then
  say "10. external baselines on the same structures (needs FPOCKET= and PRANK=)"
  [[ -n "${FPOCKET:-}" ]] && $PY scripts/baselines/run_external.py --tool fpocket --jobs "$JOBS" --manifest "$MANIFEST" || true
  [[ -n "${PRANK:-}" ]] && $PY scripts/baselines/run_external.py --tool p2rank --jobs "$JOBS" --manifest "$MANIFEST" || true
  $PY scripts/eval/compare_methods.py --tags "$TAG":nat_rank fpocket:tool_rank p2rank:tool_rank \
      --ranker-tag "$TAG" --seeds 3 || true
fi

if has eval; then
  say "11. validation: every benchmark, one protocol"
  $PY scripts/data/fetch_eval_sets.py || true
  NET_MODEL=$(ls -t "$RUNS"/net/*/model.pt 2>/dev/null | head -1 || true)
  for m in "ranker_${TAG}" "ranker_${TAG}_points"; do
    [[ -f models/$m.txt ]] || continue
    for set_name in heldout coach420 holo4k ligysis cryptobench_test; do
      $PY scripts/eval/evaluate.py --set "$set_name" --ranker "models/$m.txt" --jobs "$JOBS" --tag "_$m" || true
    done
  done
  if [[ -n "$NET_MODEL" ]]; then
    say "11b. the network as a site scorer in its own right, which is the delivery path the measurements favour"
    $PY scripts/eval/evaluate.py --set coach420 --net "$NET_MODEL" --jobs "$JOBS" --tag "_net" || true
  fi
  say "11c. head-to-head against the external tools on identical structures and receptors"
  $PY scripts/eval/compare_on_set.py --set coach420 --tag "_ranker_${TAG}" --tools p2rank fpocket --ref p2rank || true
fi

say "done"
echo "tables:      docs/results/*.md"
echo "model cards: $RUNS/*/*/model_card.json"
echo "read docs/results/README.md first: it lists what has been refuted, and the two rules for reading these numbers."
