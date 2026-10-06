#!/usr/bin/env bash
# Train every model of EquiCave-Net: the baseline and every ablation arm, N seeds each. Resumable.
#
# The arm list is read from training/configs/ablations.yaml rather than written here, so this script cannot go
# stale the way its predecessor did -- that one still listed eight arms from before the project had trained the
# network once, and would have spent A100 hours re-testing the oldest claim while ignoring the newest change.
#
# Three things it does that matter for correctness, not convenience:
#
#   * **Per-arm caches.** An arm that overrides any `data.*` key changes the featurisation, and the feature cache is
#     keyed by PDB id alone. Such an arm gets its own cache directory automatically. Without that the run would hit
#     the featurisation guard and stop -- or, before that guard existed, would have measured the wrong architecture.
#   * **Resume.** A run whose metrics.json already exists is skipped, so an interrupted grid continues instead of
#     starting over, and adding a seed costs one seed.
#   * **Order by what the measurement says is worth an hour.** The site decoder arms run first: if `no_site_decoder`
#     matches `full`, the diagnosis behind the newest change was wrong and the rest of the grid matters less.
#
# Usage:
#   bash scripts/train/run_ablations.sh                       # every arm, 3 seeds, fold 0
#   SEEDS=1 GROUP=decoder bash scripts/train/run_ablations.sh # the decoder arms only, one seed
#   ARMS="full no_site_decoder" bash scripts/train/run_ablations.sh
#   DRY_RUN=1 bash scripts/train/run_ablations.sh             # print the plan and the cost, run nothing
#
# Env: OUT FOLD SEEDS ARMS GROUP DEVICE CFG CACHE_ROOT HOURS_PER_RUN DRY_RUN
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-runs/training/ablations}
FOLD=${FOLD:-0}
SEEDS=${SEEDS:-3}
GROUP=${GROUP:-all}
DEVICE=${DEVICE:-auto}
CFG=${CFG:-training/configs/pockets_net.yaml}
CACHE_ROOT=${CACHE_ROOT:-data/cache}
HOURS_PER_RUN=${HOURS_PER_RUN:-3.5}
DRY_RUN=${DRY_RUN:-0}
PY=${PY:-python3}
export PYTHONPATH=src:.

# The plan: one "arm cache_dir" line per arm, ordered, from the config rather than from this file.
PLAN=$($PY - "$GROUP" "${ARMS:-}" <<'PYEOF'
import json, pathlib, sys, yaml
group, arms_arg = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(pathlib.Path("training/configs/ablations.yaml").read_text())["ablations"]
# Priority groups. "full" always leads: every other arm is a difference against it.
GROUPS = {
    "decoder":      ["no_site_decoder", "no_site_margin", "no_site_rank", "site_layers_4", "probe_flow",
                     "probe_flow_4"],
    "probes":       ["probe_tiered", "no_probes", "lean"],
    "equivariance": ["invariant_frames", "invariant_blind", "no_tensors_matched", "no_vectors_matched",
                     "e3nn_l1", "e3nn_l2", "e3nn_l3"],
    "mechanisms":   ["no_direction_loss", "no_listwise", "single_stage", "no_probe_potential"],
}
known = set(cfg)
if arms_arg.strip():
    arms = [a for a in arms_arg.split() if a == "full" or a in known]
    missing = [a for a in arms_arg.split() if a not in arms]
    if missing:
        sys.exit(f"unknown arm(s): {' '.join(missing)}; see training/configs/ablations.yaml")
elif group == "all":
    ordered = [a for g in ("decoder", "probes", "equivariance", "mechanisms") for a in GROUPS[g] if a in known]
    arms = ["full"] + ordered + [a for a in cfg if a not in ordered and a != "full"]
elif group in GROUPS:
    arms = ["full"] + [a for a in GROUPS[group] if a in known]
else:
    sys.exit(f"unknown group {group!r}; one of: all {' '.join(GROUPS)}")
for a in arms:
    over = cfg.get(a, {})
    # An arm that touches data.* changes the arrays, so it may not share the default cache.
    touches_data = any(str(k).startswith("data.") for k in over)
    print(a, f"data/cache/net_{a}" if touches_data else "data/cache/net")
PYEOF
)

n_runs=$(echo "$PLAN" | wc -l)
echo "$n_runs arms x $SEEDS seeds = $((n_runs * SEEDS)) runs, about $($PY -c "print(f'{$n_runs * $SEEDS * $HOURS_PER_RUN:.0f}')") GPU hours at ${HOURS_PER_RUN} h per run"
echo "$PLAN" | awk '{printf "  %-22s cache %s\n", $1, $2}'
[[ "$DRY_RUN" == "1" ]] && { echo "DRY_RUN=1: nothing executed"; exit 0; }

done_n=0; skip_n=0; fail_n=0
while read -r arm cache; do
  [[ -z "$arm" ]] && continue
  cache="${CACHE_ROOT}/${cache#data/cache/}"
  for s in $(seq 0 $((SEEDS - 1))); do
    d="$OUT/${arm}_fold${FOLD}_seed${s}"
    if [[ -f "$d/metrics.json" ]]; then
      skip_n=$((skip_n + 1)); continue
    fi
    printf "\n=== %s seed %s fold %s (cache %s) %s\n" "$arm" "$s" "$FOLD" "$cache" "$(date -u +%H:%M:%S)"
    if $PY -m training pockets-net --config "$CFG" --out "$OUT" --device "$DEVICE" \
         --set "ablation=$arm" "split.val_fold=$FOLD" "optim.seed=$s" "tag=$arm" "data.cache_dir=\"$cache\""; then
      done_n=$((done_n + 1))
    else
      fail_n=$((fail_n + 1))
      echo "FAILED: $arm seed $s -- continuing with the rest of the grid"
    fi
  done
done <<< "$PLAN"

echo
echo "$done_n trained, $skip_n already present, $fail_n failed"
$PY scripts/train/collect_ablations.py --runs "$OUT" --out docs/results/ablations.md || \
  echo "collect_ablations.py failed; the runs are in $OUT"
