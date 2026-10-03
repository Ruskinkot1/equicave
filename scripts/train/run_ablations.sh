#!/usr/bin/env bash
# The ablation grid of EquiCave-Net: every variant, 3 seeds, validation fold 0 (add folds for the final table).
# Usage: bash scripts/train/run_ablations.sh [OUT_DIR] [FOLD] [SEEDS]
set -euo pipefail
OUT=${1:-runs/training/ablations}
FOLD=${2:-0}
SEEDS=${3:-3}
export PYTHONPATH=src:.
for abl in full no_probes no_surface no_esm no_tensors no_vectors invariant achiral; do
  for s in $(seq 0 $((SEEDS - 1))); do
    echo "=== $abl seed $s fold $FOLD ==="
    python3 -m training pockets-net --config training/configs/pockets_net.yaml --out "$OUT" \
      --set "ablation=$abl" "split.val_fold=$FOLD" "optim.seed=$s" "tag=$abl"
  done
done
python3 scripts/train/collect_ablations.py --runs "$OUT" --out docs/results/ablations.md
