#!/usr/bin/env python
"""Step 2: native candidates for every manifest structure, featurised and labelled. No external tools.

Per structure: protein heavy atoms (all chains) -> `detect.detect_sites` -> `pocket_features.featurize` -> label.
Label: DCA (centre to nearest heavy atom of any kept ligand copy, >= 8 heavy atoms, listed in the manifest) <= 4 A.
Also per structure: n_sites (distinct ligand sites; copies within 8 A merged) for the top-N / top-(N+2) metrics,
and the best DCA of any candidate (ceiling analysis).

Usage: python scripts/train/build_native.py [--manifest data/processed/manifest.csv] [--jobs 4] [--limit N]
       [--min-buried 14] [--nms 6] [--tag native]
Output: data/processed/candidates_<tag>.csv + .geometry.json, data/processed/structures_<tag>.csv (per-structure summary)
"""
import argparse, csv, json, os, pathlib, sys, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from equicave import detect, pocket_features as pf, pockets as pk, structure  # noqa: E402


def n_sites(copies, link=8.0):
    cents = [c.mean(0) for c in copies]; grp = []
    for c in cents:
        if not any(np.linalg.norm(c - g) < link for g in grp):
            grp.append(c)
    return len(grp)


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
            return [], dict(pdb=pdb, status="no_candidates", n_sites=n_sites([l["xyz"] for l in ligs]))
        rows = pf.featurize(cands, st)
        L = np.vstack([l["xyz"] for l in ligs])
        copies = [l["xyz"] for l in ligs]
        ns = n_sites(copies)
        for r in rows:
            r["dca"] = pk.dca(r["center"], L)
            r["dcc_min"] = min(pk.dcc(r["center"], c) for c in copies)
            r["label"] = int(r["dca"] <= 4.0)
            r.update(pdb=pdb, n_sites=ns, **meta); r["center"] = ";".join(f"{x:.2f}" for x in r["center"])
        return rows, dict(pdb=pdb, status="ok", n_sites=ns, n_cands=len(rows), best_dca=min(r["dca"] for r in rows),
                          hit=int(any(r["label"] for r in rows)), n_atoms=len(st["xyz"]), seconds=round(time.time() - t0, 1))
    except Exception as ex:  # noqa: BLE001
        return [], dict(pdb=pdb, status=f"error: {type(ex).__name__}: {ex}"[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(REPO / "data/processed/manifest.csv"))
    ap.add_argument("--pdb-dir", default=str(REPO / "data/pockets_ds/pdb"))
    ap.add_argument("--out", default=str(REPO / "data/processed"))
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-buried", type=int, default=detect.DETECT_MIN_BURIED)
    ap.add_argument("--nms", type=float, default=detect.NMS_RADIUS)
    ap.add_argument("--max-sites", type=int, default=detect.MAX_SITES)
    ap.add_argument("--fill", type=int, default=detect.FILL_MIN_BURIED)
    ap.add_argument("--tag", default="native")
    a = ap.parse_args()
    man = list(csv.DictReader(open(a.manifest)))
    if a.limit:
        man = man[:a.limit]
    pdb_dir = pathlib.Path(a.pdb_dir)
    params = dict(min_buried=a.min_buried, nms=a.nms, max_sites=a.max_sites, fill_min_buried=a.fill)
    tasks = []
    for r in man:
        p = pdb_dir / f"{r['pdb']}.pdb"
        if p.exists():
            codes = {l[0] for l in json.loads(r["ligands"])}
            tasks.append((r["pdb"], p, codes, dict(cluster30=r["cluster30"], fold=int(r["fold"])), params))
    print(f"{len(tasks)}/{len(man)} structures have a PDB file", flush=True)
    rows, summ = [], []
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, (rr, s) in enumerate(ex.map(one, tasks, chunksize=2), 1):
            rows += rr; summ.append(s)
            if i % 100 == 0:
                ok = [x for x in summ if x["status"] == "ok"]
                print(f"  {i}: ceiling so far {np.mean([x['hit'] for x in ok]):.3f}, mean cands {np.mean([x['n_cands'] for x in ok]):.1f}", flush=True)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows); sm = pd.DataFrame(summ)
    df.to_csv(out / f"candidates_{a.tag}.csv", index=False)
    sm.to_csv(out / f"structures_{a.tag}.csv", index=False)
    (out / f"candidates_{a.tag}.geometry.json").write_text(json.dumps(dict(pf.geometry(), **params), indent=1))
    ok = sm[sm.status == "ok"]
    bad = sm[sm.status != "ok"]
    print(f"{len(ok)} structures ok, {len(bad)} skipped ({bad.status.value_counts().to_dict() if len(bad) else {}})")
    print(f"ceiling (any candidate DCA <= 4 A): {ok.hit.mean():.3f}; mean candidates per structure: {ok.n_cands.mean():.1f}; "
          f"median best DCA {ok.best_dca.median():.2f} A; mean time {ok.seconds.mean():.1f} s")
    for n in (10, 15, 20, 30):
        sub = df[df.nat_rank <= n].groupby("pdb")["label"].max()
        print(f"  ceiling with top-{n} native candidates: {sub.mean():.3f}")


if __name__ == "__main__":
    main()
