#!/usr/bin/env python
"""Per-point training data for the ligandability model: one row per sampled cavity grid point.

For every manifest structure: detect candidates, sample up to `--points-per-candidate` of each candidate's own grid
points, describe each point with `point_score.point_features`, and label it 1 when a ligand heavy atom of a kept
ligand really sits within `--occ-radius` of it. The structure's cluster and fold travel with every row, so the model
can be trained out of fold by 30 %-identity cluster exactly like the ranker.

The cap is per candidate rather than per structure so that small candidates are not swamped by large ones. Measured
candidate sizes reach 155 points, so at the default cap nothing is actually dropped and the candidate-level sums the
ranker consumes are exact; a lower cap makes them estimates and is only for a quick pilot. Negatives outnumber
positives about six to one and the training script reweights them rather than discarding any.

Restartable like build_native.py: rows flush to <tag>_point_chunks/part_*.csv every `--chunk` structures.

Usage: python scripts/train/build_points.py [--manifest data/processed/manifest.csv] [--tag native] [--jobs 4]
Output: data/processed/points_<tag>.csv.gz
"""
import argparse, csv, json, os, pathlib, sys, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from equicave import detect, labels as LB, point_score as ps, structure, tables  # noqa: E402


def one(task):
    pdb, path, lig_codes, meta, params = task
    t0 = time.time()
    try:
        st = structure.read_pdb(path)
        ligs = [l for l in structure.read_ligands(path, min_heavy=8) if l["comp"] in lig_codes]
        if len(st["xyz"]) < 50 or not ligs:
            return [], dict(pdb=pdb, status="no_protein_or_ligand")
        cands = detect.detect_sites(st["xyz"], min_buried=params["min_buried"], nms=params["nms"],
                                    max_sites=params["max_sites"], fill_min_buried=params["fill_min_buried"])
        if not cands:
            return [], dict(pdb=pdb, status="no_candidates")
        xyz = st["xyz"]; tp = cKDTree(xyz)
        types = LB.protein_atom_types(st)
        trees = {k: (cKDTree(xyz[m]) if m.sum() else None) for k, m in types.items()}
        centroid = xyz.mean(0); rg = float(np.sqrt(((xyz - centroid) ** 2).sum(1).mean())) or 1.0
        L = np.vstack([l["xyz"] for l in ligs])
        rng = np.random.default_rng(params["seed"])
        rows, n_pos = [], 0
        for ci, c in enumerate(cands):
            pts = np.atleast_2d(np.asarray(c.get("points", []), float))
            if not len(pts):
                continue
            if len(pts) > params["per_cand"]:
                pts = pts[rng.permutation(len(pts))[:params["per_cand"]]]
            X = ps.point_features(pts, st, tp, trees, rg, centroid)
            y = ps.point_labels(pts, L, params["occ_radius"])
            n_pos += int(y.sum())
            cen = ";".join(f"{x:.2f}" for x in c["center"])
            for p, x, lab in zip(pts, X, y):
                rows.append(dict(pdb=pdb, center=cen, cand=ci, x=round(float(p[0]), 2), y=round(float(p[1]), 2),
                                 z=round(float(p[2]), 2), occ=int(lab),
                                 **{f: float(v) for f, v in zip(ps.POINT_FEATURES, x)}, **meta))
        return rows, dict(pdb=pdb, status="ok", n_points=len(rows), n_pos=n_pos, n_cands=len(cands),
                          seconds=round(time.time() - t0, 1))
    except Exception as ex:  # noqa: BLE001
        return [], dict(pdb=pdb, status=f"error: {type(ex).__name__}: {ex}"[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(REPO / "data/processed/manifest.csv"))
    ap.add_argument("--pdb-dir", default=str(REPO / "data/pockets_ds/pdb"))
    ap.add_argument("--out", default=str(REPO / "data/processed"))
    ap.add_argument("--tag", default="native")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--points-per-candidate", type=int, default=400,
                    help="cap per candidate; measured candidate sizes top out near 155 points, so the "
                         "default takes every point and the candidate sums are exact")
    ap.add_argument("--occ-radius", type=float, default=ps.OCC_RADIUS)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--keep-chunks", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    chunks = out / f"{a.tag}_point_chunks"; chunks.mkdir(exist_ok=True)
    done = set()
    for f in sorted(chunks.glob("part_*.csv")):
        try:
            done |= set(pd.read_csv(f, usecols=["pdb"])["pdb"].unique())
        except Exception:  # noqa: BLE001
            print(f"  unreadable chunk {f.name}; ignored")
    pdb_dir = pathlib.Path(a.pdb_dir)
    man = list(csv.DictReader(open(a.manifest)))[: a.limit or None]
    params = dict(min_buried=detect.DETECT_MIN_BURIED, nms=detect.NMS_RADIUS, max_sites=detect.MAX_SITES,
                  fill_min_buried=detect.FILL_MIN_BURIED, per_cand=a.points_per_candidate,
                  occ_radius=a.occ_radius, seed=a.seed)
    tasks = []
    for r in man:
        p = pdb_dir / f"{r['pdb']}.pdb"
        if not p.exists() or r["pdb"] in done:
            continue
        codes = {l[0] for l in json.loads(r["ligands"])}
        tasks.append((r["pdb"], str(p), codes, dict(cluster30=r["cluster30"], fold=int(r["fold"])), params))
    print(f"{len(tasks)} structures to process ({len(man)} in the manifest, {len(done)} already in chunks)", flush=True)
    buf, summ, part = [], [], len(list(chunks.glob("part_*.csv")))
    with ProcessPoolExecutor(a.jobs) as ex:
        for k, (rows, s) in enumerate(ex.map(one, tasks, chunksize=1), 1):
            buf += rows; summ.append(s)
            if a.chunk and k % a.chunk == 0:
                pd.DataFrame(buf).to_csv(chunks / f"part_{part:04d}.csv", index=False); part += 1; buf = []
                ok = [x for x in summ if x["status"] == "ok"]
                print(f"  {k}: {sum(x['n_points'] for x in ok)} points, "
                      f"{sum(x['n_pos'] for x in ok) / max(1, sum(x['n_points'] for x in ok)):.3f} positive", flush=True)
    if buf:
        pd.DataFrame(buf).to_csv(chunks / f"part_{part:04d}.csv", index=False)
    df = pd.concat([pd.read_csv(f) for f in sorted(chunks.glob("part_*.csv"))], ignore_index=True)
    tables.write_table(df, out, f"points_{a.tag}")
    pd.DataFrame(summ).to_csv(out / f"point_build_{a.tag}.csv", index=False)
    pos = df["occ"].mean() if len(df) else float("nan")
    print(f"{len(df)} points from {df['pdb'].nunique()} structures, {pos:.4f} positive -> points_{a.tag}.csv.gz")
    (out / f"points_{a.tag}.meta.json").write_text(json.dumps(
        dict(occ_radius=a.occ_radius, points_per_candidate=a.points_per_candidate, seed=a.seed,
             features=list(ps.POINT_FEATURES), n_points=int(len(df)), positive_fraction=float(pos)), indent=1))
    if not a.keep_chunks:
        for f in chunks.glob("part_*.csv"):
            f.unlink()
        chunks.rmdir()


if __name__ == "__main__":
    main()
