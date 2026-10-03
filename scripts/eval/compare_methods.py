#!/usr/bin/env python
"""One table comparing every candidate-generation method on exactly the same structures and labels.

Reads data/processed/candidates_<tag>.csv for each tag given, keeps only the structures present in **all** of them
(so the comparison is paired), and reports for each method: candidate ceiling, mean candidates, and top-1 / top-3 /
top-N / top-(N+2) under the method's own ordering (`--order` column per tag) and, for the native table, under the
trained ranker (out-of-fold cross-validation, so no structure is scored by a model that saw it).

Paired cluster bootstrap compares every method against the first tag given.

Usage: python scripts/eval/compare_methods.py --tags native:nat_rank fpocket:tool_rank p2rank:tool_rank \
           [--ranker-tag native] [--seeds 5]
Output: docs/results/methods_comparison.md / .json
"""
import argparse, json, pathlib, sys
import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts/train"))
from equicave import metrics as M, pocket_features as pf  # noqa: E402
from train_ranker import cv_scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["native:nat_rank", "fpocket:tool_rank", "p2rank:tool_rank"])
    ap.add_argument("--ds", default=str(REPO / "data/processed")); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ranker-tag", default="native"); ap.add_argument("--out", default=str(REPO / "docs/results"))
    a = ap.parse_args()
    ds = pathlib.Path(a.ds)
    tables = {}
    for spec in a.tags:
        tag, order = spec.split(":")
        f = ds / f"candidates_{tag}.csv"
        if not f.exists():
            print(f"  {f.name} missing, skipped"); continue
        tables[tag] = (pd.read_csv(f), order)
    if not tables:
        sys.exit("no candidate tables")
    common = set.intersection(*[set(df["pdb"]) for df, _ in tables.values()])
    print(f"{len(common)} structures present in all of {sorted(tables)}")
    rows, results, ref_per = [], {}, None
    for tag, (df, order) in tables.items():
        d = df[df["pdb"].isin(common)].copy()
        if "cluster30" not in d:
            d["cluster30"] = d["pdb"]
        per = M.per_structure(d.assign(_s=-d[order]), "_s")
        st = M.summarize(per); st["mean_candidates"] = float(len(d) / d["pdb"].nunique())
        if ref_per is None:
            ref_per = per
        else:
            st["paired_vs_first"] = {c: M.paired_boot(per.set_index("pdb").loc[ref_per["pdb"], c].to_numpy(float),
                                                      ref_per[c].to_numpy(float), ref_per["cluster30"].to_numpy())
                                     for c in ("top1", "topN2")}
        results[f"{tag} (own order)"] = st
        # the same candidates re-ranked by a LambdaRank model trained out of fold on that table's own features
        feats = [f for f in pf.FEATURES if f in d] or [c for c in d.columns if c.startswith("tool_")]
        if "fold" in d and len(feats) >= 3:
            scores = [cv_scores(d.reset_index(drop=True), feats, s) for s in range(a.seeds)]
            pers = [M.per_structure(d.reset_index(drop=True).assign(_s=s), "_s") for s in scores]
            avg = pd.concat(pers).groupby("pdb", sort=False).agg({c: "mean" for c in ("top1", "top3", "topN", "topN2", "mrr", "ceiling")} | {"cluster30": "first"}).reset_index()
            st2 = M.summarize(avg); st2["mean_candidates"] = st["mean_candidates"]; st2["seeds"] = a.seeds
            st2["paired_vs_first"] = {c: M.paired_boot(avg.set_index("pdb").loc[ref_per["pdb"], c].to_numpy(float),
                                                      ref_per[c].to_numpy(float), ref_per["cluster30"].to_numpy())
                                      for c in ("top1", "topN2")}
            results[f"{tag} + LambdaRank"] = st2
    lines = ["# Candidate generators and rankers on the same structures", "",
             f"{len(common)} structures common to {', '.join(sorted(tables))}; labels, ligand filter and n_sites are identical "
             f"for every method (DCA <= 4 A). Rankers are cross-validated by 30 %-identity cluster with {a.seeds} seeds, so no "
             "structure is scored by a model that saw it. 95 % CI by cluster bootstrap.", "",
             "| method | candidates | ceiling | top-1 | top-3 | top-N | top-(N+2) | MRR |", "|---|---|---|---|---|---|---|---|"]
    for name, st in results.items():
        lines.append(f"| {name} | {st['mean_candidates']:.1f} | {st['ceiling']:.3f} | {M.fmt(st['top1'])} | {M.fmt(st['top3'])} | "
                     f"{M.fmt(st['topN'])} | {M.fmt(st['topN2'])} | {M.fmt(st['mrr'])} |")
    lines += ["", "Paired differences against the first method listed (positive = better):", "",
              "| method | top-1 difference | top-(N+2) difference |", "|---|---|---|"]
    for name, st in results.items():
        p = st.get("paired_vs_first")
        if p:
            lines.append(f"| {name} | {p['top1']['diff']:+.3f} [{p['top1']['lo']:+.3f}, {p['top1']['hi']:+.3f}] | "
                         f"{p['topN2']['diff']:+.3f} [{p['topN2']['lo']:+.3f}, {p['topN2']['hi']:+.3f}] |")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "methods_comparison.md").write_text("\n".join(lines) + "\n")
    (out / "methods_comparison.json").write_text(json.dumps(dict(n_structures=len(common), tags=list(tables), results=results), indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
