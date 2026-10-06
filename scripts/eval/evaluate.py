#!/usr/bin/env python
"""One protocol for every benchmark: native candidates (+ ranker, + network) vs optional external tools.

For each structure of a set: download PDB (RCSB), protein = all chains (or the listed chain), ligands = HETATM
groups with >= 8 heavy atoms (or the listed codes; waters, ions and the manifest EXCLUDE list are skipped), predict
sites, report DCA and DCC success at 4 A for top-1, top-3, top-N and top-(N+2) with N = number of distinct ligand sites,
MRR and the candidate ceiling. Predictions are non-redundant (no two centres within 6 A). Bootstrap 95 % CI by
30 % cluster (RCSB cluster ids fetched for the set) and paired bootstrap of every method against the native order.
Similarity filter: structures of the set whose 30 % cluster appears in the training manifest are reported separately
("train-similar"), so the headline row of an external benchmark is the one with those removed.

Thresholds. Success is reported at DCA <= 4 A (centre to the nearest ligand heavy atom) and at DCC <= 4 A, 10 A and
12 A (centre to the ligand centroid). The LIGYSIS comparison argues 4 A is too strict for DCC, because a correct
prediction on a large or elongated ligand can sit more than 4 A from its centroid, and recommends 10-12 A; since the
reported advantage of equivariant detectors is concentrated in DCC, the threshold is not a detail.

Redundancy. Predictions are non-redundant by construction (no two centres within 6 A), and the table also reports how
many of the predictions that hit a site hit one that a better-ranked prediction already found, which the LIGYSIS
comparison showed dominates the apparent ranking of pocket methods.

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
from equicave import (detect, labels as LB, metrics as M, pocket_features as pf, pockets as pk, progress,  # noqa: E402
                      structure, superpose as SP, targets)
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


def _holo_spec(row) -> dict:
    """The apo/holo pairing a benchmark row carries in its note, or {} when the row is an ordinary holo entry."""
    note = row.get("note", "") or ""
    if "holo_pdb_id" not in note:
        return {}
    try:
        d = json.loads(note[note.index("{"):note.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return {}
    return d if d.get("holo_pdb_id") else {}


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


def predict_native(st, ranker, net_model, net_cfg, pdb_path, esm=None, point_model=None):
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
    if point_model is not None:               # the ranker was trained with the per-point aggregates
        for r, agg in zip(rows, pf.point_aggregates(cands, st, point_model)):
            r.update(agg)
    if esm is not None:                       # the ranker was trained with language-model columns
        for r, ef in zip(rows, esm(pdb_path, np.array([c["center"] for c in cands]))):
            r.update(ef)
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
    """One benchmark entry. Never raises: a single unreadable structure must not abort a 4000-structure run, so a
    failure becomes a status row and shows up in the skipped counts of the report instead."""
    try:
        return _one(task)
    except Exception as ex:                                     # noqa: BLE001
        return dict(pdb=task[0]["pdb"], status=f"error: {type(ex).__name__}: {ex}"[:200]), []


def _one(task):
    r, chains, args = task
    pdb = r["pdb"]; path = fetch(pdb)
    if path is None:
        return dict(pdb=pdb, status="no_pdb"), []
    # Receptor context. Every COACH420 entry names a chain, and reading only that chain was costing us the benchmark:
    # the candidate table is built on the whole assembly (scripts/train/build_native.py reads every chain), so a
    # single-chain receptor moves the protein centroid and radius of gyration, changes prot_n_res and the candidate
    # count, empties the 5/8/12 A shells of a neighbouring chain's atoms, and -- worst -- unburies every pocket that
    # sits at a chain interface. Measured on the 179 comparable structures, the richer the feature set the worse the
    # damage: the detector's own context-free score was unchanged at 0.642 top-1, a 32-feature ranker reached 0.704,
    # and 236- and 272-feature rankers fell to 0.598 and 0.570, below no ranking at all. scripts/baselines/run_external.py
    # gives the external tools every chain, so the truncation also biased the head-to-head against us.
    # The chain in a benchmark row identifies the relevant *ligand*, not a receptor to cut down, so the default reads
    # the assembly the model was trained on; --receptor-chains entry restores the old behaviour for comparison.
    entry_only = args.get("receptor_chains") == "entry"
    st = structure.read_pdb(path, chains or None) if (entry_only and chains) else structure.read_pdb(path)
    codes = set(filter(None, r.get("ligand_codes", "").replace(";", ",").split(","))) or None
    # Apo/holo benchmarks (CryptoBench): the entry names an apo structure, where the point of the benchmark is that
    # the pocket is closed and no ligand is bound, and a separate holo structure that actually holds the ligand.
    # Prediction happens on the apo form and the label has to be carried over by superposition. Without this the
    # evaluation found no ligand in the apo file and silently skipped the entry: 222 of 228 published test entries.
    holo = _holo_spec(r)
    if holo:
        hpath = fetch(holo["holo_pdb_id"])
        if hpath is None:
            return dict(pdb=pdb, status="no_holo_pdb"), []
        tr = SP.align_chains(path, hpath, holo.get("apo_chain", ""), holo.get("holo_chain", ""))
        if not tr.get("ok"):
            return dict(pdb=pdb, status=f"holo alignment failed: {tr.get('reason', '')}"[:120]), []
        ligs = SP.transfer_ligands(hpath, tr, codes)
        st = structure.read_pdb(path)                         # the apo receptor, every chain
        if not ligs:
            return dict(pdb=pdb, status="no_ligand_in_holo", holo=holo["holo_pdb_id"]), []
    else:
        ligs = [l for l in structure.read_ligands(path, min_heavy=8, exclude=EXCLUDE if codes is None else set()) if (codes is None or l["comp"] in codes)]
    if chains and entry_only and not holo:
        ligs = [l for l in ligs if np.linalg.norm(st["xyz"][:, None] - l["xyz"][None], axis=2).min() <= 6.0] if len(st["xyz"]) else []
    if len(st["xyz"]) < 50 or not ligs:
        return dict(pdb=pdb, status="no_ligand" if len(st["xyz"]) >= 50 else "no_protein"), []
    sites = LB.group_sites(ligs)
    copies = [l["xyz"] for l in ligs]; L = np.vstack(copies); ns = len(sites)
    site_atoms = [np.vstack([ligs[i]["xyz"] for i in s]) for s in sites]
    ranker = pf.load_ranker(args["ranker"]) if args["ranker"] else None
    esm = None
    if ranker is not None and any(f.startswith("esm_") for f in ranker[1]):
        from equicave.esm_features import EsmFeatures
        esm = EsmFeatures(args.get("esm_tag", "native2"))
    point_model = None
    if ranker is not None and any(f in pf.POINT_AGG for f in ranker[1]):
        import lightgbm as lgb
        path_pm = args.get("point_model") or str(REPO / "models/point_native.txt")
        if not pathlib.Path(path_pm).exists():
            return dict(pdb=pdb, status=f"point model {pathlib.Path(path_pm).name} missing"), []
        # The default is the model fitted on all folds, which is correct for a benchmark structure outside the
        # manifest and is leakage for one inside it -- those are the `train-similar` rows, reported separately. Note
        # that the ranker learned these columns from *out-of-fold* point scores, so the full-fold model serves
        # slightly sharper values than it was trained on; the shift is in the optimistic direction and is the reason
        # the headline row is the one that excludes train-similar structures.
        point_model = lgb.Booster(model_file=path_pm)
    net_model = net_cfg = None
    if args["net"]:
        from training.pockets import net_task as NT
        import torch
        net_model, net_cfg = NT.load_model(args["net"], torch.device("cpu"))
    preds = predict_native(st, ranker, net_model, net_cfg, path, esm, point_model)
    out = []
    for p in preds:
        d = pk.dca(p["center"], L)
        dcc_per_site = [pk.dcc(p["center"], a) for a in site_atoms]
        dca_per_site = [pk.dca(p["center"], a) for a in site_atoms]
        j = int(np.argmin(dca_per_site))                       # the site this prediction is closest to
        out.append(dict(pdb=pdb, chain=chains, n_sites=ns, dca=d, dcc=min(dcc_per_site), site_idx=j,
                        label=int(d <= 4.0), label_dcc=int(min(dcc_per_site) <= 4.0),
                        label_dcc10=int(min(dcc_per_site) <= 10.0), label_dcc12=int(min(dcc_per_site) <= 12.0),
                        center=";".join(f"{x:.2f}" for x in p["center"]),
                        **{k: v for k, v in p.items() if k != "center"}))
    return dict(pdb=pdb, status="ok", n_sites=ns, n_cands=len(preds)), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True); ap.add_argument("--ranker", default=""); ap.add_argument("--net", default="")
    ap.add_argument("--external", default=""); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--no-similarity-filter", action="store_true"); ap.add_argument("--out", default=str(REPO / "docs/results"))
    ap.add_argument("--ligand-rule", choices=["mlig", "all"], default="mlig")
    ap.add_argument("--receptor-chains", choices=["all", "entry"], default="all",
                    help="all (default): the whole assembly, as the candidate table was built and as the external "
                         "baselines are run; entry: only the chain the benchmark row names, which mismatches training")
    ap.add_argument("--point-model", default="", help="per-point ligandability booster; defaults to "
                    "models/point_native.txt and is only loaded when the ranker needs its columns")
    ap.add_argument("--esm-tag", default="native2", help="which esm_features_<tag> projection to use at inference")
    ap.add_argument("--merge-radii", default="8,12", help="prediction-merging radii to report in addition to as-generated")
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. _all for the other ligand rule")
    a = ap.parse_args()
    rows = load_set(a.set, a.limit, a.ligand_rule)
    print(f"{a.set}: {len(rows)} entries", flush=True)
    ids = sorted({r["pdb"] for r in rows})
    meta = {}
    with progress.Bar("RCSB metadata", len(ids), unit="entry") as bar:
        for i in range(0, len(ids), 200):
            try:
                for k, e in info(ids[i:i + 200], desc=None).items():   # the outer bar reports the whole set
                    meta[k] = dict(cluster30=groups(e, "sequence_identity", 30.0), uniprot=groups(e, "matching_uniprot_accession"))
            except Exception as ex:  # noqa: BLE001
                print(f"  RCSB metadata failed for a batch: {ex}")
            bar.update(len(ids[i:i + 200]), postfix=f"{len(meta)} resolved")
    train_cl = {r["cluster30"] for r in csv.DictReader(open(REPO / "data/processed/manifest.csv"))} if (REPO / "data/processed/manifest.csv").exists() else set()
    args = dict(ranker=a.ranker, net=a.net, esm_tag=a.esm_tag, point_model=a.point_model,
                receptor_chains=a.receptor_chains)
    tasks = [(r, r.get("chain", ""), args) for r in rows]
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(progress.track(ex.map(one, tasks, chunksize=2), f"predicting on {a.set}", len(tasks), unit="pdb"))
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
        crits = dict(DCA="label", DCC4="label_dcc", DCC10="label_dcc10", DCC12="label_dcc12")
        radii = [float(x) for x in a.merge_radii.split(",") if x.strip()]
        for m in methods:
            for radius in radii:                       # the same ranking, merged, so prediction count is comparable
                if radius <= 0 or "center" not in df:
                    continue
                sub = M.nms_by_score(df, m, radius)
                st_m = M.summarize(M.per_structure(sub, m, label_col="label"))
                st_m["mean_predictions"] = float(len(sub) / sub["pdb"].nunique())
                st_m["merge_radius"] = radius
                if "site_idx" in sub:
                    st_m["redundancy_all"] = M.redundancy(sub, m, label_col="label")
                results[subset][f"{m} (DCA, merged at {radius:.0f} A)"] = st_m
            for name, crit in crits.items():
                if crit not in df:
                    continue
                per = M.per_structure(df, m, label_col=crit)
                st = M.summarize(per)
                if m == "native order" and name == "DCA":
                    per_ref = per
                if per_ref is not None and name == "DCA":
                    st["paired_vs_native_order_top1"] = M.paired_boot(per.set_index("pdb").loc[per_ref["pdb"], "top1"].to_numpy(float), per_ref["top1"].to_numpy(float), per_ref["cluster30"].to_numpy())
                if "site_idx" in df:
                    st["redundancy_all"] = M.redundancy(df, m, label_col=crit)
                    st["redundancy_topN2"] = M.redundancy(df, m, label_col=crit, top=None)
                results[subset][f"{m} ({name})"] = st
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
             "| subset | method | top-1 | top-3 | top-N | top-(N+2) | MRR | n | ceiling | redundant hits |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for subset, d in results.items():
        for m, st in d.items():
            red = st.get("redundancy_all", {})
            rtxt = f"{red.get('fraction', float('nan')):.3f}" if red else "—"
            if "mean_predictions" in st:
                m = f"{m}, {st['mean_predictions']:.1f} predictions"
            lines.append(f"| {subset} | {m} | {M.fmt(st['top1'])} | {M.fmt(st['top3'])} | {M.fmt(st['topN'])} | "
                         f"{M.fmt(st['topN2'])} | {M.fmt(st['mrr'])} | {st['n']} | {st['ceiling']:.3f} | {rtxt} |")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"eval_{a.set}{a.tag}.md").write_text("\n".join(lines) + "\n")
    (out / f"eval_{a.set}{a.tag}.json").write_text(json.dumps(dict(set=a.set, ligand_rule=a.ligand_rule, n=len(rows),
        status=summ.status.value_counts().to_dict(), results=results), indent=1))
    cand.to_csv(REPO / "data/processed" / f"eval_candidates_{a.set}{a.tag}.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
