"""Label a tool's predictions that are already on disk, without running it again.

A baseline run costs hours, and the drivers re-run the tool whenever the table has to be rebuilt -- which is what
happens after a bug in the *labelling* rather than in the prediction. P2Rank's LIGYSIS predictions survived both
a non-zero exit that discarded them and the fix for it; re-running would have cost three hours to produce files
that were already there.

Only P2Rank's output layout is supported, because it is the only tool whose per-structure CSVs are
self-contained. Usage:

    PYTHONPATH=src:. python scripts/baselines/label_existing.py --tool p2rank --set ligysis \\
        --work data/pockets_ds/baselines_ligysis --manifest data/processed/manifest_ligysis.csv
"""
import argparse
import csv
import json
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts/train"))
sys.path.insert(0, str(REPO / "scripts/data")); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from equicave import tables  # noqa: E402
from run_external import label  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", default="p2rank", choices=["p2rank"])
    ap.add_argument("--set", required=True)
    ap.add_argument("--work", required=True, help="the run's work directory, holding <tool>/out/")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pdb-dir", default=str(REPO / "data/external/pdb"))
    a = ap.parse_args()

    out = pathlib.Path(a.work) / a.tool / "out"
    if not out.is_dir():
        sys.exit(f"{out} does not exist; there is nothing to label")
    man = {r["pdb"]: r for r in csv.DictReader(open(a.manifest))}
    rows, found, missing = [], 0, 0
    for r in man.values():
        f = out / f"{r['pdb']}.pdb_predictions.csv"
        if not f.exists():
            missing += 1
            continue
        found += 1
        preds = []
        for i, d in enumerate(csv.DictReader(open(f), skipinitialspace=True), 1):
            d = {k.strip(): v.strip() for k, v in d.items()}
            try:
                preds.append(dict(center=np.array([float(d["center_x"]), float(d["center_y"]), float(d["center_z"])]),
                                  tool_score=float(d["score"]), tool_rank=i, tool_extra=float(d["probability"])))
            except (KeyError, ValueError):
                continue
        raw = pathlib.Path(a.pdb_dir) / f"{r['pdb']}.pdb"
        if raw.exists():
            rows += label(preds, r["pdb"], raw, {l[0] for l in json.loads(r["ligands"])},
                          dict(cluster30=r["cluster30"], fold=int(r["fold"])))
    if not rows:
        sys.exit("no labelled rows; check the ligand rule and the manifest")
    df = pd.DataFrame(rows)
    path = tables.write_table(df, REPO / "data/processed", f"candidates_{a.tool}_{a.set}")
    g = df.groupby("pdb")
    print(f"{found} prediction files found, {missing} absent")
    print(f"  {len(df)} candidates in {g.ngroups} structures -> {path.name}; "
          f"ceiling {(g['label'].max() == 1).mean():.3f}; mean candidates {len(df) / g.ngroups:.1f}; "
          f"top-1 by its own order {(df[df.tool_rank == 1].groupby('pdb')['label'].max() == 1).sum() / g.ngroups:.3f}")


if __name__ == "__main__":
    main()
