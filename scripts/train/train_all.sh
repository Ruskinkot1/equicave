#!/usr/bin/env bash
# EquiCave: the whole training pipeline in one script. Run it from the repository root.
#
#   bash scripts/train/train_all.sh                  # everything the machine can do (GPU stages are skipped on CPU)
#   STAGES="data candidates ranker" bash scripts/train/train_all.sh
#   JOBS=8 SEEDS=5 bash scripts/train/train_all.sh
#
# Stages, in order: data candidates ranker peptide labels net net-oof hybrid ablations eval baselines
# Each stage is skipped when its output already exists, so the script is restartable.
# Every number it produces lands in docs/results/ with its protocol in the header.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src:.
PY=${PY:-python3}
JOBS=${JOBS:-$( (nproc 2>/dev/null || echo 4) )}
SEEDS=${SEEDS:-5}
STAGES=${STAGES:-"data candidates ranker peptide labels net net-oof hybrid ablations eval"}
DS=data/processed
RUNS=${RUNS:-runs/training}
NET_CFG=${NET_CFG:-}
has() { [[ " $STAGES " == *" $1 "* ]]; }
say() { printf "\n=== %s (%s)\n" "$1" "$(date -u +%H:%M:%S)"; }

if [[ -z "$NET_CFG" ]]; then
  if $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    NET_CFG=training/configs/pockets_net.yaml; DEVICE=cuda
  else
    NET_CFG=training/configs/pockets_net_cpu.yaml; DEVICE=cpu
    echo "no CUDA device: the network stages will use the CPU pilot config, whose numbers are not publishable"
  fi
fi

say "environment"
$PY -c "import numpy, scipy, pandas, lightgbm, sklearn; print('cpu stack ok')"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || echo "torch missing: network stages unavailable"
$PY -m pytest tests -q

if has data; then
  say "1. data: RCSB manifest and structures"
  [[ -f $DS/manifest.csv ]] || $PY scripts/data/build_manifest.py --n 3200 --seed 0
  $PY scripts/data/fetch_structures.py --jobs 8
fi

if has candidates; then
  say "2. native candidates, features and labels"
  [[ -f $DS/candidates_native.csv ]] || $PY scripts/train/build_native.py --jobs "$JOBS"
fi

if has ranker; then
  say "3. ranker: LambdaRank, cluster CV, $SEEDS seeds, ablations"
  $PY scripts/train/train_ranker.py --tag native --seeds "$SEEDS" --ablate --model models/ranker_native.txt
fi

if has peptide; then
  say "4. peptide-binding sites: benchmark, candidates, ranker"
  [[ -f $DS/manifest_peptide.csv ]] || $PY scripts/data/build_peptide_manifest.py --n 4000 --seed 0 --disjoint-from $DS/manifest.csv
  $PY scripts/data/fetch_structures.py --manifest $DS/manifest_peptide.csv --jobs 8
  [[ -f $DS/candidates_peptide.csv ]] || $PY scripts/train/build_peptide.py --jobs "$JOBS" --min-res 3
  $PY scripts/train/train_ranker.py --tag peptide --seeds "$SEEDS" --ablate --model models/ranker_peptide.txt
fi

if has labels; then
  say "5. property and hotspot label statistics"
  $PY -m training pockets-labels
fi

if has net; then
  say "6. EquiCave-Net on fold 0 (device: ${DEVICE:-auto}), 3 seeds"
  for s in 0 1 2; do
    $PY -m training pockets-net --config "$NET_CFG" --out "$RUNS/net" --device auto \
        --set "split.val_fold=0" "optim.seed=$s" "tag=full"
  done
fi

if has net-oof; then
  say "7. out-of-fold network features for the hybrid ranker"
  $PY -m training pockets-net --config "$NET_CFG" --out "$RUNS/net-oof" --device auto --set "mode=oof" "tag=full"
fi

if has hybrid; then
  say "8. hybrid ranker (candidate features + network scores)"
  $PY scripts/train/train_ranker.py --tag native --seeds "$SEEDS" --features-extra net --model models/ranker_hybrid.txt
fi

if has ablations; then
  say "9. ablation grid, 3 seeds each"
  bash scripts/train/run_ablations.sh "$RUNS/ablations" 0 3
fi

if has baselines; then
  say "10. optional external baselines (fpocket, P2Rank) on the same structures"
  [[ -n "${FPOCKET:-}" ]] && $PY scripts/baselines/run_external.py --tool fpocket --jobs "$JOBS" || true
  [[ -n "${PRANK:-}" ]] && $PY scripts/baselines/run_external.py --tool p2rank --jobs "$JOBS" || true
  $PY scripts/eval/compare_methods.py --tags native:nat_rank fpocket:tool_rank p2rank:tool_rank --seeds 3 || true
fi

if has eval; then
  say "11. benchmarks, one protocol"
  $PY scripts/data/fetch_eval_sets.py || true
  NET_MODEL=$(ls -t "$RUNS"/net/*/model.pt 2>/dev/null | head -1 || true)
  for set_name in heldout coach420 holo4k; do
    $PY scripts/eval/evaluate.py --set "$set_name" --ranker models/ranker_native.txt \
        ${NET_MODEL:+--net "$NET_MODEL"} --jobs "$JOBS" || true
  done
fi

say "done: tables in docs/results, model cards in $RUNS/*/model_card.json"
ls -1 docs/results/*.md 2>/dev/null || true
