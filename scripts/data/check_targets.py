#!/usr/bin/env python
"""Verify that every held-out target's ligand id exists in its deposited PDB entry (and report the alternatives).

An id that does not match makes the evaluation skip the entry silently, so this check runs before any evaluation.
Usage: python scripts/data/check_targets.py [--pdb-dir data/external/pdb] [--fetch]
Exit code 1 if any target is unusable.
"""
import argparse, pathlib, sys, urllib.request

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from equicave import structure, targets  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-dir", default=str(REPO / "data/external/pdb")); ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    d = pathlib.Path(a.pdb_dir); d.mkdir(parents=True, exist_ok=True)
    bad = 0
    for name, t in targets.TARGETS.items():
        f = d / f"{t['pdb']}.pdb"
        if not f.exists() and a.fetch:
            try:
                urllib.request.urlretrieve(f"https://files.rcsb.org/download/{t['pdb']}.pdb", f)
            except Exception as ex:  # noqa: BLE001
                print(f"{name:14s} {t['pdb']} download failed: {ex}"); bad += 1; continue
        if not f.exists():
            print(f"{name:14s} {t['pdb']} no local file (use --fetch)"); bad += 1; continue
        codes = sorted({l["comp"] for l in structure.read_ligands(f, min_heavy=8)})
        ok = t["lig"] in codes
        bad += not ok
        print(f"{name:14s} {t['pdb']} {t['lig']:4s} {'ok' if ok else 'NOT IN FILE'}  candidates in file: {codes}")
    print(f"\n{len(targets.TARGETS) - bad}/{len(targets.TARGETS)} targets usable")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
