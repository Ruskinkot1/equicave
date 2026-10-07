#!/usr/bin/env bash
# Every training run of EquiCave-Net, in the order the measurements say is worth the GPU hours.
#
# One entry point, resumable, and it never silently redoes work: a stage whose output exists is skipped, so an
# interrupted run continues where it stopped.
#
#   bash scripts/train/run_all.sh                 # the whole thing
#   DRY_RUN=1 bash scripts/train/run_all.sh       # the plan and the cost, nothing executed
#   STAGES=grid bash scripts/train/run_all.sh     # only the ablation grid
#   SEEDS=1 bash scripts/train/run_all.sh         # one seed, for a first pass
#
# Stages, in order, each skippable by name in STAGES (default: cache,grid,collect):
#
#   cache    build the feature cache if it is missing. epochs=0, so this trains nothing.
#   grid     the ablation arms, in priority order (below).
#   collect  the summary table.
#
# The order of the arms is not arbitrary. `no_site_decoder` comes first because it decides whether the diagnosis
# the last three changes rest on is true at all: the error is which cavity binds, not where inside it. If that arm
# matches `full`, the site decoder, the ranking losses and the aggregator sweep are all answering a question that
# was not the question, and the rest of the grid matters much less.
#
# WARM shortens the warmup stage, and it is not a saving at the cost of quality. Measured on the first full run:
# site top-1 is 0.724 at epoch 1 and 0.022 at epoch 14, because stage 1 holds the centre loss at zero and lets the
# head drift; occ_ap reaches 0.760 by epoch 6 against 0.796 at epoch 14. Six warmup epochs buy 92 % of the dense
# heads' convergence for 30 % of the time, and stop the centre head being degraded for twenty epochs first.
set -euo pipefail
cd "$(dirname "$0")/../.."

STAGES=${STAGES:-cache,grid,collect}
SEEDS=${SEEDS:-3}
FOLD=${FOLD:-0}
DEVICE=${DEVICE:-cuda}
CFG=${CFG:-training/configs/pockets_net.yaml}
CACHE=${CACHE:-data/cache/net}
MANIFEST=${MANIFEST:-}
DRY_RUN=${DRY_RUN:-0}
PY=${PY:-python3}
WARM=${WARM:---set stages.warmup_epochs=6 --set optim.epochs=30}
export PYTHONPATH=src:.

# Priority order. Everything here is a measurement that does not exist in the published literature, except the
# first two, which are ours to settle before anything else is worth running.
ORDER=${ORDER:-"
no_site_decoder      does the decoder earn its place at all -- run this first
no_site_rank         the listwise term alone
no_site_margin       the hinge term alone
noise_aug            the only lever with a clean published measurement, +0.098 PR-AUC there
gaussian_attn        +3.3 % DCC in its own paper, on a backbone of our family, and it is smaller and faster
small                96x4, 2.18 M: if it ties, every later experiment costs half
agg_mean             the aggregator three leading methods disagree on and nobody has compared
agg_max              .
gaussian_attn_4      four layers, where the published depth sweep peaks
invariant_frames     the isolating equivariance ablation that exists nowhere in this field
no_probes_no_tensors does degree 2 work when the operator stops supplying enclosure
"}

say() { printf "\n\033[1m== %s\033[0m  %s\n" "$1" "$(date -u +%H:%M:%S)"; }
run() { if [[ "$DRY_RUN" == "1" ]]; then echo "   would run: $*"; else "$@"; fi; }

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

arms=$(echo "$ORDER" | awk 'NF {print $1}')
n_arms=$(echo "$arms" | wc -l)
echo "stages: $STAGES | seeds: $SEEDS | fold: $FOLD | device: $DEVICE | cache: $CACHE"
echo "$n_arms arms x $SEEDS seeds = $((n_arms * SEEDS)) runs"
echo "$ORDER" | awk 'NF {printf "   %-22s %s\n", $1, substr($0, index($0,$2))}'

if has_stage cache; then
  say "cache"
  if [[ -d "$CACHE" ]] && [[ -n "$(ls -A "$CACHE" 2>/dev/null)" ]]; then
    echo "   $CACHE exists with $(ls "$CACHE" | wc -l) files; skipped"
  else
    extra=()
    [[ -n "$MANIFEST" ]] && extra+=(--set "data.manifest=$MANIFEST")
    run $PY -m training pockets-net --config "$CFG" --device "$DEVICE" \
        --out "runs/training/cache-$(date +%Y%m%d-%H%M%S)" \
        --set "data.cache_dir=$CACHE" --set optim.epochs=0 "${extra[@]}"
  fi
fi

if has_stage grid; then
  say "grid"
  export CACHE_ROOT=$(dirname "$CACHE")
  run env ARMS="$(echo $arms)" SEEDS="$SEEDS" FOLD="$FOLD" DEVICE="$DEVICE" CFG="$CFG" \
      WARM="$WARM" bash scripts/train/run_ablations.sh
fi

if has_stage collect; then
  say "collect"
  latest=$(ls -1d runs/training/ablations/*/ 2>/dev/null | sort | tail -1)
  if [[ -n "$latest" ]]; then
    run $PY scripts/train/collect_ablations.py --runs "${latest%/}" --out "${latest%/}/ablations.md"
    echo "   table: ${latest%/}/ablations.md"
  else
    echo "   no grid directory yet"
  fi
fi

say "done"
