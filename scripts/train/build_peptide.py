#!/usr/bin/env python
"""Peptide-site candidate table: merged cavity + groove candidates for every protein-peptide complex, labelled.

Receptor = the long chains of the entry (`rec_chains`); the peptide chains are removed from the protein input, so the
detector never sees the ligand it must find (the apo-like setting a real prediction faces). Ligands = those peptide
chains (>= 8 heavy atoms). Label: DCA <= 4 A from a candidate centre to any peptide heavy atom. n_sites = number of
distinct peptide sites (chains whose atoms lie within 8 A of each other are one site).

Candidates: `detect.detect_sites` (cavity tiers 1-2, <= --small) merged with `peptide.detect_peptide_sites`
(groove tier 3, <= --groove) by 6 A non-maximum suppression. Features: the shared set of `pocket_features.featurize`
plus `peptide.groove_features` for every candidate of every tier.

Usage: python scripts/train/build_peptide.py [--jobs 4] [--limit N]
Output: data/processed/candidates_peptide.csv (+ .geometry.json), data/processed/structures_peptide.csv
"""
import argparse, csv, json, os, pathlib, sys, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from equicave import detect, peptide as PEP, pocket_features as pf, pockets as pk, structure  # noqa: E402


def one(task):
    pdb, path, rec_chains, pep_chains, meta, params = task
    t0 = time.time()
    try:
        found = structure.read_peptide_ligands(path, min_res=params["min_res"], max_res=params["max_res"])
        # the manifest chain ids come from RCSB entities and need not match the deposited file, so they are a hint
        ligs = [l for l in found if l["chain"] in pep_chains] or found
        rec = "".join(c for c in structure.receptor_chains(path) if c not in {l["chain"] for l in ligs})
        st = structure.read_pdb(path, rec or (rec_chains or None))
        if len(st["xyz"]) < 100 or not ligs:
            return [], dict(pdb=pdb, status="no_receptor_or_peptide")
        small = detect.detect_sites(st["xyz"], max_sites=params["small"])
        groove = PEP.detect_peptide_sites(st["xyz"], st, max_sites=params["groove"], min_points=20,
                                          min_anisotropy=1.5, nms=4.0)
        cands = PEP.merge_with_small_molecule(groove, small, max_sites=params["small"] + params["groove"])
        for c in cands:
            c.setdefault("tier", 1)
        if not cands:
            return [], dict(pdb=pdb, status="no_candidates")
        rows = pf.featurize(cands, st)
        for r, g in zip(rows, PEP.groove_features(cands, st)):
            r.update(g)
        copies = [l["xyz"] for l in ligs]
        L = np.vstack(copies)
        sites, used = [], set()
        for i, c in enumerate(copies):                      # peptide chains within 8 A are one site
            if i in used:
                continue
            grp = [i]; used.add(i)
            for j in range(i + 1, len(copies)):
                if j not in used and np.linalg.norm(copies[j][:, None] - np.vstack([copies[k] for k in grp])[None], axis=2).min() < 8.0:
                    grp.append(j); used.add(j)
            sites.append(grp)
        for r in rows:
            r["dca"] = pk.dca(r["center"], L)
            r["dcc_min"] = min(pk.dcc(r["center"], np.vstack([copies[k] for k in s])) for s in sites)
            r["label"] = int(r["dca"] <= 4.0)
            r.update(pdb=pdb, n_sites=len(sites), pep_res=int(sum(l["n_res"] for l in ligs)), **meta)
            r["center"] = ";".join(f"{x:.2f}" for x in r["center"])
        return rows, dict(pdb=pdb, status="ok", n_sites=len(sites), n_cands=len(rows), best_dca=min(r["dca"] for r in rows),
                          hit=int(any(r["label"] for r in rows)),
                          hit_groove=int(any(r["label"] and r["pep_tier"] == 3 for r in rows)),
                          hit_cavity=int(any(r["label"] and r["pep_tier"] < 3 for r in rows)),
                          pep_res=int(sum(l["n_res"] for l in ligs)), seconds=round(time.time() - t0, 1))
    except Exception as ex:  # noqa: BLE001
        return [], dict(pdb=pdb, status=f"error: {type(ex).__name__}: {ex}"[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(REPO / "data/processed/manifest_peptide.csv"))
    ap.add_argument("--pdb-dir", default=str(REPO / "data/pockets_ds/pdb"))
    ap.add_argument("--out", default=str(REPO / "data/processed"))
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--small", type=int, default=30); ap.add_argument("--groove", type=int, default=20)
    ap.add_argument("--min-res", type=int, default=4); ap.add_argument("--max-res", type=int, default=30)
    ap.add_argument("--tag", default="peptide")
    a = ap.parse_args()
    man = list(csv.DictReader(open(a.manifest)))
    if a.limit:
        man = man[:a.limit]
    params = dict(small=a.small, groove=a.groove, min_res=a.min_res, max_res=a.max_res)
    tasks = []
    for r in man:
        p = pathlib.Path(a.pdb_dir) / f"{r['pdb']}.pdb"
        if p.exists():
            tasks.append((r["pdb"], p, r.get("rec_chains", ""), r.get("pep_chains", ""),
                          dict(cluster30=r["cluster30"], fold=int(r["fold"])), params))
    print(f"{len(tasks)}/{len(man)} entries have a PDB file", flush=True)
    rows, summ = [], []
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, (rr, s) in enumerate(ex.map(one, tasks, chunksize=2), 1):
            rows += rr; summ.append(s)
            if i % 100 == 0:
                ok = [x for x in summ if x["status"] == "ok"]
                print(f"  {i}: ceiling {np.mean([x['hit'] for x in ok]):.3f} (groove {np.mean([x['hit_groove'] for x in ok]):.3f}, "
                      f"cavity {np.mean([x['hit_cavity'] for x in ok]):.3f})", flush=True)
    out = pathlib.Path(a.out)
    df = pd.DataFrame(rows); sm = pd.DataFrame(summ)
    df.to_csv(out / f"candidates_{a.tag}.csv", index=False); sm.to_csv(out / f"structures_{a.tag}.csv", index=False)
    (out / f"candidates_{a.tag}.geometry.json").write_text(json.dumps(dict(pf.geometry(), **params,
        GROOVE_MIN_BURIED=PEP.GROOVE_MIN_BURIED, SEG_LEN=PEP.SEG_LEN, PEAK_NMS=PEP.PEAK_NMS), indent=1))
    ok = sm[sm.status == "ok"]
    print(f"{len(ok)} receptors ok, {len(sm) - len(ok)} skipped ({sm[sm.status != 'ok'].status.value_counts().to_dict()})")
    print(f"ceiling {ok.hit.mean():.3f} (groove candidates alone {ok.hit_groove.mean():.3f}, cavity candidates alone {ok.hit_cavity.mean():.3f}); "
          f"mean candidates {ok.n_cands.mean():.1f}; median best DCA {ok.best_dca.median():.2f} A; median peptide length {ok.pep_res.median():.0f} residues")


if __name__ == "__main__":
    main()
