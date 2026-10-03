#!/usr/bin/env bash
# Template: VN-EGNN (equivariant GNN with virtual nodes). Fill in the environment and the call, then make executable.
# Contract: $1 = input protein PDB, $2 = output JSON [{"center": [x,y,z], "score": s, "rank": r}, ...] best first.
set -euo pipefail
IN=$1; OUT=$2
: "${VNEGNN_REPO:?set VNEGNN_REPO to the checkout of ml-jku/vnegnn}"
: "${VNEGNN_ENV:?set VNEGNN_ENV to the python interpreter of its environment}"
: "${VNEGNN_CKPT:?set VNEGNN_CKPT to the downloaded checkpoint}"

WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
# Replace the next line with the inference command of the release you downloaded; it must leave its predictions in $WORK.
"$VNEGNN_ENV" "$VNEGNN_REPO/inference.py" --input "$IN" --checkpoint "$VNEGNN_CKPT" --output "$WORK" >"$WORK/log" 2>&1

# Convert whatever it wrote into the contract above. The example expects a CSV with x,y,z,score columns.
"$VNEGNN_ENV" - "$WORK" "$OUT" <<'PY'
import csv, json, pathlib, sys
work, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
rows = []
for f in sorted(work.rglob("*.csv")):
    for d in csv.DictReader(open(f), skipinitialspace=True):
        d = {k.strip().lower(): v for k, v in d.items()}
        try:
            rows.append(dict(center=[float(d[k]) for k in ("x", "y", "z")], score=float(d.get("score", 0.0))))
        except (KeyError, ValueError):
            continue
rows.sort(key=lambda r: -r["score"])
for i, r in enumerate(rows, 1):
    r["rank"] = i
out.write_text(json.dumps(rows))
print(f"{len(rows)} predictions")
PY
