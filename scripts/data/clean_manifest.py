#!/usr/bin/env python
"""Clean a training manifest: drop what should never have been a training example, and say why for each entry.

Four filters, each reported separately so the cost of every one is visible.

1. **Leakage against the evaluation sets** (the rule the DeepPocket and P2Rank papers apply, and the one our own
   COACH420 run showed matters: 38 % of that benchmark shared a cluster with our training set). Every 30 %-identity
   cluster that appears in COACH420, HOLO4K, LIGYSIS, CryptoBench or the held-out target families is removed from
   training. Clusters of the evaluation entries are fetched once from RCSB and cached.

2. **Ligand relevance.** A ligand is a training signal only if it is actually bound: at least `--min-contacts`
   receptor heavy atoms within 4.5 A, mean occupancy at least `--min-occupancy`, and a mean B-factor no more than
   `--max-bfactor-ratio` times the receptor's own mean. A molecule sitting on the surface at half occupancy with
   twice the b-factor of its protein is crystallisation noise, and a model trained to find it learns noise.

3. **Burial.** The ligand's own mean 26-ray closure must be at least `--min-buried`, which removes ligands lying in
   a crystal contact rather than in a pocket.

4. **Redundancy.** At most `--per-cluster-ligand` entries per (cluster, ligand component) pair, so a protein
   crystallised twenty times with the same inhibitor contributes a few examples rather than twenty.

Usage: python scripts/data/clean_manifest.py --manifest data/processed/manifest_max.csv --out data/processed/manifest_clean.csv
Output: the cleaned manifest, plus <out>.report.json and a printed table of what each filter removed.
"""
import argparse, collections, csv, json, pathlib, sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from equicave import pockets as pk, structure  # noqa: E402
from build_manifest import groups, info  # noqa: E402

EVAL_SETS = ("coach420", "holo4k", "ligysis", "cryptobench")
CACHE = REPO / "data/cache/eval_clusters.json"


def eval_clusters(sets=EVAL_SETS, refresh: bool = False) -> dict:
    """{pdb: [clusters]} for every structure of the evaluation sets, cached on disk."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() and not refresh else {}
    ids = set()
    for s in sets:
        f = REPO / "data/external/eval_sets" / f"{s}.csv"
        if f.exists():
            ids |= {r["pdb"].upper() for r in csv.DictReader(open(f)) if r["pdb"]}
    todo = sorted(ids - set(cache))
    if todo:
        print(f"  fetching clusters for {len(todo)} evaluation structures", flush=True)
        for i in range(0, len(todo), 100):
            try:
                for k, e in info(todo[i:i + 100]).items():
                    cache[k] = groups(e, "sequence_identity", 30.0)
            except Exception as ex:  # noqa: BLE001
                print(f"    batch {i}: {type(ex).__name__}: {ex}")
            if i and i % 1000 == 0:
                CACHE.parent.mkdir(parents=True, exist_ok=True); CACHE.write_text(json.dumps(cache))
        CACHE.parent.mkdir(parents=True, exist_ok=True); CACHE.write_text(json.dumps(cache))
    return cache


def ligand_quality(path, codes, min_contacts, min_occ, max_bf_ratio, min_buried):
    """Per ligand copy: contacts, occupancy, b-factor ratio and closure; returns (kept_codes, stats)."""
    st = structure.read_pdb(path)
    if len(st["xyz"]) < 50:
        return set(), dict(reason="no_receptor")
    ligs = [l for l in structure.read_ligands(path, min_heavy=8) if l["comp"] in codes]
    if not ligs:
        return set(), dict(reason="no_ligand")
    from scipy.spatial import cKDTree
    tree = cKDTree(st["xyz"])
    prot_bf = float(st["bfactor"].mean()) or 1.0
    # occupancy and b-factor of ligand atoms, read from the same records
    occ, bf = {}, {}
    for line in pathlib.Path(path).read_text(errors="ignore").splitlines():
        if line[:6] != "HETATM":
            continue
        comp = line[17:20].strip()
        if comp not in codes:
            continue
        try:
            occ.setdefault(comp, []).append(float(line[54:60]))
            bf.setdefault(comp, []).append(float(line[60:66]))
        except ValueError:
            pass
    kept, stats = set(), {}
    for l in ligs:
        comp = l["comp"]
        n_contacts = int(sum(len(x) for x in tree.query_ball_point(l["xyz"], 4.5)))
        o = float(np.mean(occ.get(comp, [1.0])))
        b = float(np.mean(bf.get(comp, [prot_bf]))) / prot_bf
        bur = float(pk._buried(l["xyz"], tree).mean())
        ok = (n_contacts >= min_contacts and o >= min_occ and b <= max_bf_ratio and bur >= min_buried)
        s = stats.setdefault(comp, dict(contacts=n_contacts, occupancy=round(o, 2), bfactor_ratio=round(b, 2),
                                        buried=round(bur, 1), kept=False))
        if ok:
            kept.add(comp); s["kept"] = True
    return kept, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(REPO / "data/processed/manifest.csv"))
    ap.add_argument("--pdb-dir", default=str(REPO / "data/pockets_ds/pdb"))
    ap.add_argument("--out", default=str(REPO / "data/processed/manifest_clean.csv"))
    ap.add_argument("--min-contacts", type=int, default=15); ap.add_argument("--min-occupancy", type=float, default=0.5)
    ap.add_argument("--max-bfactor-ratio", type=float, default=2.0); ap.add_argument("--min-buried", type=float, default=10.0)
    ap.add_argument("--per-cluster-ligand", type=int, default=2); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-quality", action="store_true", help="only the leakage and redundancy filters (no PDB files needed)")
    ap.add_argument("--refresh-clusters", action="store_true")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.manifest)))
    if a.limit:
        rows = rows[:a.limit]
    print(f"{len(rows)} entries in {pathlib.Path(a.manifest).name}")

    held = REPO / "data/processed/heldout_targets.json"
    bad = set()
    if held.exists():
        h = json.loads(held.read_text())
        bad |= {c for v in h.values() for c in v.get("cluster30", [])}
    ec = eval_clusters(refresh=a.refresh_clusters)
    eval_cl = {c for v in ec.values() for c in v}
    print(f"  evaluation sets occupy {len(eval_cl)} clusters; held-out families {len(bad)}")
    bad |= eval_cl

    dropped = collections.Counter()
    per_pair = collections.Counter()
    kept_rows, quality = [], {}
    for i, r in enumerate(rows, 1):
        if r["cluster30"] in bad:
            dropped["leakage_eval_or_heldout"] += 1; continue
        codes = [l[0] for l in json.loads(r["ligands"])]
        if not a.no_quality:
            p = pathlib.Path(a.pdb_dir) / f"{r['pdb']}.pdb"
            if not p.exists():
                dropped["no_pdb_file"] += 1; continue
            kept_codes, stats = ligand_quality(p, set(codes), a.min_contacts, a.min_occupancy,
                                               a.max_bfactor_ratio, a.min_buried)
            quality[r["pdb"]] = stats
            if not kept_codes:
                dropped["ligand_quality"] += 1; continue
            codes = [c for c in codes if c in kept_codes]
            r = dict(r, ligands=json.dumps([l for l in json.loads(r["ligands"]) if l[0] in kept_codes]))
        key = (r["cluster30"], tuple(sorted(codes)))
        if per_pair[key] >= a.per_cluster_ligand:
            dropped["cluster_ligand_cap"] += 1; continue
        per_pair[key] += 1
        kept_rows.append(r)
        if i % 500 == 0:
            print(f"  {i}/{len(rows)} inspected, {len(kept_rows)} kept", flush=True)

    if not kept_rows:
        sys.exit("every entry was filtered out; loosen the thresholds")
    out = pathlib.Path(a.out)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(kept_rows)
    folds = collections.Counter(int(r["fold"]) for r in kept_rows)
    report = dict(manifest=str(a.manifest), kept=len(kept_rows), input=len(rows), dropped=dict(dropped),
                  clusters=len({r["cluster30"] for r in kept_rows}), folds={k: folds[k] for k in sorted(folds)},
                  thresholds=dict(min_contacts=a.min_contacts, min_occupancy=a.min_occupancy,
                                  max_bfactor_ratio=a.max_bfactor_ratio, min_buried=a.min_buried,
                                  per_cluster_ligand=a.per_cluster_ligand),
                  eval_clusters=len(eval_cl), quality_checked=len(quality))
    pathlib.Path(str(out) + ".report.json").write_text(json.dumps(report, indent=1))
    print(f"\nkept {len(kept_rows)}/{len(rows)} entries in {report['clusters']} clusters -> {out.name}")
    for k, v in dropped.most_common():
        print(f"  dropped {v:6d}  {k}")
    print(f"  folds {report['folds']}")


if __name__ == "__main__":
    main()
