#!/usr/bin/env python
"""Download the PDB files of a manifest into data/pockets_ds/pdb (not in git). Parallel, resumable.

Usage: python scripts/data/fetch_structures.py [--manifest data/processed/manifest.csv] [--jobs 8]
"""
import argparse, csv, pathlib, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = pathlib.Path(__file__).resolve().parents[2]


def fetch(pdb, d, tries=3):
    f = d / f"{pdb}.pdb"
    if f.exists() and f.stat().st_size > 1000:
        return pdb, True
    for t in range(tries):
        try:
            urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb}.pdb", f)
            if f.stat().st_size > 1000:
                return pdb, True
        except Exception:  # noqa: BLE001
            time.sleep(2 * (t + 1))
        f.unlink(missing_ok=True)
    return pdb, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(REPO / "data/processed/manifest.csv"))
    ap.add_argument("--out", default=str(REPO / "data/pockets_ds/pdb"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--column", default="pdb")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ids = sorted({r[a.column].upper()[:4] for r in csv.DictReader(open(a.manifest))})
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda p: fetch(p, out), ids))
    ok = sum(r[1] for r in res)
    print(f"{ok}/{len(ids)} files in {out}; missing: {[p for p, k in res if not k][:20]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
