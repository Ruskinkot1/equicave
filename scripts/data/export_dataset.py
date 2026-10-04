#!/usr/bin/env python
"""Export or import the dataset recipe, so a collaborator reproduces the exact same training set.

What travels and what does not. Raw PDB files are **never** committed: at full scale they are tens of gigabytes, and
they are not ours to redistribute. The *recipe* is small and fully determines the data: the manifest (PDB id, 30 %
cluster, fold, ligand components), the geometry constants, and the git commit. Derived feature tables are shipped
only when they fit (`--include-features`), optionally split into per-fold shards so no single file is large.

    # make an archive another machine can rebuild from (a few MB)
    python scripts/data/export_dataset.py --tag native2 --out dist/equicave_dataset.tar.gz

    # the same plus the feature table, sharded by fold (hundreds of MB at full scale)
    python scripts/data/export_dataset.py --tag native2 --include-features --shard-by fold --out dist/equicave_full.tar.gz

    # on the other machine
    python scripts/data/export_dataset.py --import dist/equicave_dataset.tar.gz
    make candidates JOBS=8          # or: python scripts/train/build_native.py --stream --jobs 8

`--stream` in the builder downloads each structure and deletes it after featurising, so rebuilding a full-PDB-scale
table needs time but almost no disk.
"""
import argparse, hashlib, json, pathlib, subprocess, sys, tarfile, time

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def sha256(p: pathlib.Path, limit: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(limit):
            h.update(chunk)
    return h.hexdigest()


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def export(a):
    from equicave import pocket_features as pf
    ds = REPO / "data/processed"
    members, meta = [], {}
    for name in (a.manifest, "heldout_targets.json"):
        p = ds / name
        if p.exists():
            members.append(p); meta[name] = dict(bytes=p.stat().st_size, sha256=sha256(p))
    if a.include_features:
        import pandas as pd
        from equicave import tables
        src = tables.find_table(ds, f"candidates_{a.tag}")
        if a.shard_by:
            df = pd.read_csv(src)
            shard_dir = ds / f"shards_{a.tag}"; shard_dir.mkdir(exist_ok=True)
            for v, g in df.groupby(a.shard_by):
                f = shard_dir / f"candidates_{a.tag}_{a.shard_by}{v}.csv.gz"
                g.to_csv(f, index=False); members.append(f)
                meta[f.name] = dict(rows=len(g), bytes=f.stat().st_size, sha256=sha256(f))
            print(f"sharded {len(df)} rows by {a.shard_by} into {len(members) - 2} files")
        else:
            members.append(src); meta[src.name] = dict(bytes=src.stat().st_size, sha256=sha256(src))
    recipe = dict(created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), commit=commit(), tag=a.tag,
                  geometry=pf.geometry(), files=meta,
                  rebuild=["python scripts/data/fetch_structures.py --manifest data/processed/" + a.manifest + " --jobs 8",
                           f"python scripts/train/build_native.py --manifest data/processed/{a.manifest} --jobs 8 --tag {a.tag}",
                           "# or, without the disk cost of keeping structures:",
                           f"python scripts/train/build_native.py --manifest data/processed/{a.manifest} --jobs 8 --stream --tag {a.tag}"],
                  note="Raw PDB files are not included: they are redistributed by the RCSB PDB (CC0) and the manifest "
                       "names exactly which ones to fetch. The geometry constants must match or the features drift.")
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    rec = REPO / "data/processed/RECIPE.json"; rec.write_text(json.dumps(recipe, indent=1))
    with tarfile.open(out, "w:gz") as t:
        t.add(rec, arcname="RECIPE.json")
        for m in members:
            t.add(m, arcname=str(m.relative_to(REPO)))
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB) with {len(members)} files; rebuild steps are in RECIPE.json")


def do_import(path):
    with tarfile.open(path, "r:gz") as t:
        names = t.getnames()
        t.extractall(REPO, filter="data")
    rec = json.loads((REPO / "data/processed/RECIPE.json").read_text())
    from equicave import pocket_features as pf
    print(f"imported {len(names)} files, built at commit {rec['commit'][:8]}")
    if rec["geometry"] != pf.geometry():
        print("WARNING: the geometry constants of this checkout differ from the export; features would not match:")
        print(f"  export:  {rec['geometry']}\n  current: {pf.geometry()}")
    else:
        print("geometry constants match this checkout")
    for step in rec["rebuild"]:
        print("  " + step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="native2"); ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--out", default=str(REPO / "dist/equicave_dataset.tar.gz"))
    ap.add_argument("--include-features", action="store_true"); ap.add_argument("--shard-by", default="")
    ap.add_argument("--import", dest="imp", default="")
    a = ap.parse_args()
    return do_import(a.imp) if a.imp else export(a)


if __name__ == "__main__":
    main()
