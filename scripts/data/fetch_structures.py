#!/usr/bin/env python
"""Download the PDB files of a manifest into data/pockets_ds/pdb (not in git). Parallel, resumable.

Usage: python scripts/data/fetch_structures.py [--manifest data/processed/manifest.csv] [--jobs 8]
"""
import argparse, csv, pathlib, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from equicave import progress  # noqa: E402


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
    res, ok, bad = [], 0, 0
    with ThreadPoolExecutor(a.jobs) as ex:
        it = ex.map(lambda p: fetch(p, out), ids)
        with progress.Bar(f"downloading {len(ids)} structures", len(ids), unit="pdb") as bar:
            for r in it:
                res.append(r); ok += bool(r[1]); bad += not r[1]
                bar.update(1, postfix=f"{ok} ok, {bad} missing")
    print(f"{ok}/{len(ids)} files in {out}; missing: {[p for p, k in res if not k][:20]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
