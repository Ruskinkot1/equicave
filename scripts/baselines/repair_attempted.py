#!/usr/bin/env python
"""Drop structures from a run's attempted list when their predictions were never written.

A driver that records a structure as attempted before flushing its rows leaves an inconsistent pair behind if the
run is interrupted: the structure counts as done, its predictions are gone, and the resumed run neither redoes it
nor has anything for it -- so it reads back as the method refusing to predict, which scores the method zero for
our interruption. The drivers now write the two together. This repairs a directory left by one that did not.

A structure is kept in the list only if it has rows in some chunk, or is in failed.txt (our pipeline died on it,
which is recorded on purpose). Everything else is removed and will simply be run again.

Usage: python scripts/baselines/repair_attempted.py --tool deepsurf --set coach420 [--dry-run]
"""
import argparse, pathlib, sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import failures  # noqa: E402

DS = pathlib.Path(__file__).resolve().parents[2] / "data/processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True); ap.add_argument("--set", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    d = DS / f"{a.tool}_{a.set}_chunks"
    att = d / "attempted.txt"
    if not att.exists():
        sys.exit(f"{att} does not exist")
    ids = [l.strip() for l in att.read_text().splitlines() if l.strip()]
    have = set()
    for f in sorted(d.glob("part_*.csv")):
        try:
            have |= set(pd.read_csv(f)["pdb"])
        except Exception:                                # noqa: BLE001 -- a chunk being written right now
            print(f"  unreadable chunk {f.name}; ignored")
    failed = d / "failed.txt"
    have |= failures.read(failed)
    keep = [i for i in ids if i in have]
    lost = [i for i in ids if i not in have]
    print(f"{a.tool}: {len(ids)} attempted, {len(keep)} have rows or are recorded failures, "
          f"{len(lost)} were marked done with nothing written and will be run again")
    if lost and not a.dry_run:
        att.write_text("".join(f"{i}\n" for i in keep))
        print(f"  rewrote {att}")
    elif a.dry_run:
        print("  dry run: nothing written")


if __name__ == "__main__":
    main()
