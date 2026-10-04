#!/usr/bin/env python
"""One protocol for every benchmark: native candidates (+ ranker, + network) vs optional external tools.

For each structure of a set: download PDB (RCSB), protein = all chains (or the listed chain), ligands = HETATM
groups with >= 8 heavy atoms (or the listed codes; waters, ions and the manifest EXCLUDE list are skipped), predict
sites, report DCA and DCC success at 4 A for top-1, top-3, top-N and top-(N+2) with N = number of distinct ligand sites,
MRR and the candidate ceiling. Predictions are non-redundant (no two centres within 6 A). Bootstrap 95 % CI by
30 % cluster (RCSB cluster ids fetched for the set) and paired bootstrap of every method against the native order.
Similarity filter: structures of the set whose 30 % cluster appears in the training manifest are reported separately
("train-similar"), so the headline row of an external benchmark is the one with those removed.

Ligand rule. COACH420 and HOLO4K are published in two forms: the full list, where every HETATM group that passes the
solvent/additive filter counts as a site, and the `mlig` ("relevant ligand") list used by P2Rank and DeepPocket, where
each entry names the ligands that define its sites. `--ligand-rule mlig` (the default when an `_mlig` list exists)
follows the published convention, `--ligand-rule all` uses every drug-sized HETATM group. The two give different
numbers, so the rule is printed in the table header and stored in the JSON; papers that do not state theirs are not
comparable to either.

Usage: python scripts/eval/evaluate.py --set heldout|coach420|holo4k|ligysis|cryptobench [--ranker models/ranker_native.txt]
       [--net runs/training/pockets-net/full_fold0_seed0/model.pt] [--external fpocket,p2rank] [--limit N] [--jobs 4]
Output: docs/results/eval_<set>.json / .md
"""
import argparse, csv, json, os, pathlib, shutil, subprocess, sys, tempfile, time, urllib.request
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/data"))
from equicave import detect, labels as LB, metrics as M, pocket_features as pf, pockets as pk, structure, targets  # noqa: E402
from build_manifest import EXCLUDE, info, groups  # noqa: E402

EXT = REPO / "data/external/eval_sets"; RAW = REPO / "data/external/pdb"


def fetch(pdb):
    RAW.mkdir(parents=True, exist_ok=True); f = RAW / f"{pdb}.pdb"
    if not f.exists():
        try:
            urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb}.pdb", f)
        except Exception:  # noqa: BLE001
            return None
    return f if f.exists() and f.stat().st_size > 1000 else None


def load_set(name, limit, ligand_rule="mlig"):
    """Rows of a benchmark. With `ligand_rule='mlig'` the ligand codes of the published relevant-ligand list are used."""
    if name == "heldout":
        rows = [dict(pdb=t["pdb"], chain="", ligand_codes=t["lig"], note=k) for k, t in targets.TARGETS.items()]
    else:
        f = EXT / f"{name}.csv"
        if not f.exists():
            sys.exit(f"{f} missing: run scripts/data/fetch_eval_sets.py")
        rows = [r for r in csv.DictReader(open(f)) if r["pdb"]]
        mlig = EXT / f"{name}_mlig.csv"
        if ligand_rule == "mlig" and mlig.exists():
            codes = {(r["pdb"], r["chain"]): r["ligand_codes"] for r in csv.DictReader(open(mlig)) if r["ligand_codes"]}
            keep = []
            for r in rows:
                c = codes.get((r["pdb"], r["chain"]))
                if c is None and not r["chain"]:
                    c = next((v for (p, _), v in codes.items() if p == r["pdb"]), None)
                if c:
                    keep.append(dict(r, ligand_codes=c, note=(r.get("note", "") + " mlig").strip()))
            if keep:
                print(f"  relevant-ligand rule: {len(keep)} of {len(rows)} entries are in {mlig.name}")
                rows = keep
    seen, out = set(), []
    for r in rows:
        key = (r["pdb"], r["chain"])
        if key not in seen:
            seen.add(key); out.append(r)
    return out[:limit] if limit else out


def predict_native(st, ranker, net_model, net_cfg, pdb_path):
    cands = detect.detect_sites(st["xyz"])
    if not cands:
        return []
    rows = pf.featurize(cands, st)
    extra = None
    if net_model is not None:
        import torch
        from training.pockets import data as D, net_task as NT
        d = D.featurize(pdb_path, None, None, net_cfg["data"]["n_probe"], net_cfg["data"]["n_surf"])
        if d is not None and net_cfg["data"].get("esm"):
            from training.pockets.esm_embed import Embedder, cached
            rt = structure.residue_table(st)
            d = D.featurize(pdb_path, None, cached(Embedder(net_cfg["data"]["esm"]), pathlib.Path(pdb_path).stem, rt["seq"], rt["chain"], REPO / "data/cache/esm"), net_cfg["data"]["n_probe"], net_cfg["data"]["n_surf"])
        if d is not None:
            with torch.no_grad():
                out = net_model(D.to_torch(d))
            extra = NT.net_features(out, d["pos"][d["n_res"]:d["n_res"] + d["n_probe"]], np.array([c["center"] for c in cands]))
            for r, e in zip(rows, extra):
                r.update(e)
    scores = {"native order": [-r["nat_rank"] for r in rows]}
    if ranker is not None:
        booster, feats = ranker
        X = np.array([[r.get(f, 0.0) for f in feats] for r in rows], float)
        scores["ranker"] = booster.predict(X).tolist()
        if extra is not None:
            scores["ranker+net"] = scores.pop("ranker")
    if extra is not None:
        scores["network only"] = [e["net_center_conf"] + e["net_seg"] for e in extra]
    return [dict(center=r["center"], **{k: v[i] for k, v in scores.items()}) for i, r in enumerate(rows)]


def one(task):
    r, chains, args = task
    pdb = r["pdb"]; path = fetch(pdb)
    if path is None:
        return dict(pdb=pdb, status="no_pdb"), []
    st = structure.read_pdb(path, chains or None)
    codes = set(filter(None, r.get("ligand_codes", "").replace(";", ",").split(","))) or None
    ligs = [l for l in structure.read_ligands(path, min_heavy=8, exclude=EXCLUDE if codes is None else set()) if (codes is None or l["comp"] in codes)]
    if chains:
        ligs = [l for l in ligs if np.linalg.norm(st["xyz"][:, None] - l["xyz"][None], axis=2).min() <= 6.0] if len(st["xyz"]) else []
    if len(st["xyz"]) < 50 or not ligs:
        return dict(pdb=pdb, status="no_ligand" if len(st["xyz"]) >= 50 else "no_protein"), []
    copies = [l["xyz"] for l in ligs]; L = np.vstack(copies); ns = len(LB.group_sites(ligs))
    ranker = pf.load_ranker(args["ranker"]) if args["ranker"] else None
    net_model = net_cfg = None
    if args["net"]:
        from training.pockets import net_task as NT
        import torch
        net_model, net_cfg = NT.load_model(args["net"], torch.device("cpu"))
    preds = predict_native(st, ranker, net_model, net_cfg, path)
    out = []
    for p in preds:
        d = pk.dca(p["center"], L)
        out.append(dict(pdb=pdb, chain=chains, n_sites=ns, dca=d, dcc=min(pk.dcc(p["center"], c) for c in copies), label=int(d <= 4.0),
                        label_dcc=int(min(pk.dcc(p["center"], c) for c in copies) <= 4.0), **{k: v for k, v in p.items() if k != "center"}))
    return dict(pdb=pdb, status="ok", n_sites=ns, n_cands=len(preds)), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True); ap.add_argument("--ranker", default=""); ap.add_argument("--net", default="")
    ap.add_argument("--external", default=""); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--no-similarity-filter", action="store_true"); ap.add_argument("--out", default=str(REPO / "docs/results"))
    ap.add_argument("--ligand-rule", choices=["mlig", "all"], default="mlig")
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. _all for the other ligand rule")
    a = ap.parse_args()
    rows = load_set(a.set, a.limit, a.ligand_rule)
    print(f"{a.set}: {len(rows)} entries", flush=True)
    ids = sorted({r["pdb"] for r in rows})
    meta = {}
    for i in range(0, len(ids), 200):
        try:
            for k, e in info(ids[i:i + 200]).items():
                meta[k] = dict(cluster30=groups(e, "sequence_identity", 30.0), uniprot=groups(e, "matching_uniprot_accession"))
        except Exception as ex:  # noqa: BLE001
            print(f"  RCSB metadata failed for a batch: {ex}")
    train_cl = {r["cluster30"] for r in csv.DictReader(open(REPO / "data/processed/manifest.csv"))} if (REPO / "data/processed/manifest.csv").exists() else set()
    args = dict(ranker=a.ranker, net=a.net)
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(one, [(r, r.get("chain", ""), args) for r in rows], chunksize=2))
    summ = pd.DataFrame([s for s, _ in res]); cand = pd.DataFrame([c for _, cs in res for c in cs])
    if cand.empty:
        sys.exit("no predictions")
    cand["cluster30"] = cand["pdb"].map(lambda p: (meta.get(p, {}).get("cluster30") or [p])[0])
    cand["train_similar"] = cand["pdb"].map(lambda p: bool(set(meta.get(p, {}).get("cluster30", [])) & train_cl))
    methods = [c for c in ("native order", "ranker", "ranker+net", "network only") if c in cand]
    results = {}
    for subset, df in (("all", cand), ("not train-similar", cand[~cand.train_similar]), ("train-similar", cand[cand.train_similar])):
        if df.empty or (a.no_similarity_filter and subset != "all"):
            continue
        results[subset] = {}
        per_ref = None
        for m in methods:
            for crit in ("label", "label_dcc"):
                per = M.per_structure(df, m, label_col=crit)
                st = M.summarize(per)
                if m == "native order" and crit == "label":
                    per_ref = per
                if per_ref is not None and crit == "label":
                    st["paired_vs_native_order_top1"] = M.paired_boot(per.set_index("pdb").loc[per_ref["pdb"], "top1"].to_numpy(float), per_ref["top1"].to_numpy(float), per_ref["cluster30"].to_numpy())
                results[subset][f"{m} ({'DCA' if crit == 'label' else 'DCC'})"] = st
    for tool in filter(None, a.external.split(",")):     # external tables produced by scripts/baselines/run_external.py --manifest <set list>
        f = REPO / "data/processed" / f"candidates_{tool}_{a.set}.csv"
        if f.exists():
            ext = pd.read_csv(f)
            results.setdefault("all", {})[f"{tool} (DCA)"] = M.summarize(M.per_structure(ext.assign(s=-ext["tool_rank"]), "s"))
        else:
            print(f"  external table {f.name} not found; skipped")
    lines = [f"# Evaluation: {a.set}", "",
             f"{(summ.status == 'ok').sum()} structures evaluated, skipped: {summ.status.value_counts().to_dict()}. "
             f"Ligand rule: {a.ligand_rule} ({'published relevant-ligand list' if a.ligand_rule == 'mlig' else 'every drug-sized HETATM group'}). "
             f"Success is DCA (or DCC) <= 4 A; N is the structure's own number of ligand sites; predictions are "
             f"non-redundant (6 A). The headline row is **not train-similar**: structures sharing a 30 %-identity "
             f"cluster with the training manifest are listed separately.", "",
             "| subset | method | top-1 | top-3 | top-N | top-(N+2) | MRR | n | ceiling |", "|---|---|---|---|---|---|---|---|---|"]
    for subset, d in results.items():
        for m, st in d.items():
            lines.append(f"| {subset} | {m} | {M.fmt(st['top1'])} | {M.fmt(st['top3'])} | {M.fmt(st['topN'])} | {M.fmt(st['topN2'])} | {M.fmt(st['mrr'])} | {st['n']} | {st['ceiling']:.3f} |")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"eval_{a.set}{a.tag}.md").write_text("\n".join(lines) + "\n")
    (out / f"eval_{a.set}{a.tag}.json").write_text(json.dumps(dict(set=a.set, ligand_rule=a.ligand_rule, n=len(rows),
        status=summ.status.value_counts().to_dict(), results=results), indent=1))
    cand.to_csv(REPO / "data/processed" / f"eval_candidates_{a.set}{a.tag}.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
