#!/usr/bin/env python
"""Add the structures a tool answered "no site" for to its candidate table, as misses.

`compare_on_set.py` intersects the structures each method predicted, so a structure a method produced nothing for
vanishes from the comparison instead of counting against it. For a conservative method that is a large, silent
favour: GrASP returns no site at all on 9 % of COACH420, every one of them a guaranteed miss.

The drivers now write such a structure as a miss when it happens. This repairs a table produced before they did:
a structure the driver *attempted* (its id is in the chunk directory's attempted.txt) and which carries a usable
ligand (it appears in our own evaluation table, so the labels exist) but has no row in the tool's table was
answered with no prediction, and gets one row with no hit.

Usage: python scripts/baselines/add_missing_as_misses.py --tool grasp --set coach420 [--our-tag _f32all]
"""
import argparse, pathlib, sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from equicave import tables  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import failures  # noqa: E402

DS = REPO / "data/processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True); ap.add_argument("--set", required=True)
    ap.add_argument("--our-tag", default="_f32all")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    df = tables.read_table(DS, f"candidates_{a.tool}_{a.set}")
    attempted_file = DS / f"{a.tool}_{a.set}_chunks" / "attempted.txt"
    if not attempted_file.exists():
        sys.exit(f"{attempted_file} is missing: cannot tell what the tool was asked to predict")
    attempted = {l.strip() for l in attempted_file.read_text().splitlines() if l.strip()}
    # A structure our own pipeline failed on is not the method refusing to predict, and scoring it as a miss would
    # charge the method for our environment. The drivers record those separately.
    failed_file = attempted_file.parent / "failed.txt"
    failed = failures.read(failed_file)
    if failed:
        print(f"  excluding {len(failed)} structures our pipeline failed on, which are not refusals")
    attempted -= failed
    ours = pd.read_csv(DS / f"eval_candidates_{a.set}{a.our_tag}.csv")
    labelled = dict(ours.groupby("pdb")["n_sites"].first())      # structures whose labels exist, with their N

    missing = sorted((attempted & set(labelled)) - set(df["pdb"]))
    print(f"{a.tool}: {len(df)} rows for {df.pdb.nunique()} structures; {len(attempted)} attempted; "
          f"{len(missing)} answered with no site and are currently absent")
    if not missing:
        return
    before = df.groupby("pdb")["label"].max().mean()
    add = pd.DataFrame([dict(pdb=p, center="nan;nan;nan", tool_score=-np.inf, tool_rank=1, tool_rel=0.0,
                             dca=np.inf, dcc_min=np.inf, label=0, n_sites=int(labelled[p]), n_cands=0)
                        for p in missing])
    out = pd.concat([df, add], ignore_index=True)
    after = out.groupby("pdb")["label"].max().mean()
    print(f"  ceiling {before:.3f} -> {after:.3f} once the refusals count as misses")
    if a.dry_run:
        print("  dry run: nothing written")
        return
    tables.write_table(out, DS, f"candidates_{a.tool}_{a.set}")
    print(f"  wrote {len(out)} rows")


if __name__ == "__main__":
    main()
