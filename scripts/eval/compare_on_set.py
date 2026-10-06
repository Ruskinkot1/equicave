#!/usr/bin/env python
"""Compare our predictions against external tools on one benchmark, structure for structure.

The per-set evaluation writes our candidates to data/processed/eval_candidates_<set>.csv and the baseline wrappers
write theirs to data/processed/candidates_<tool>_<set>.csv. Both carry the same labels (DCA <= 4 A against the same
ligand list), so the two tables can be compared directly -- but only on the structures that appear in both, and only
within the same subset. Reporting our number on the structures that are not similar to our training set against a
tool's number on every structure would flatter whichever method had the easier set, so this script intersects the
structure ids first and splits by train-similarity afterwards.

`train_similar` marks a structure sharing a 30 %-identity cluster with data/processed/manifest.csv. That is a
property of *our* training set, so for an external tool it does not mean leakage; the subset exists so the same
structures are compared on both sides. A tool's own training set may overlap the benchmark differently, which the
table notes rather than corrects.

Usage: python scripts/eval/compare_on_set.py --set coach420 --tools p2rank fpocket [--ours ranker] [--ref p2rank]
Output: docs/results/compare_<set>.md / .json
"""
import argparse, csv, json, pathlib, sys
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from equicave import metrics as M, tables  # noqa: E402

DS = REPO / "data/processed"


def train_clusters() -> set:
    f = DS / "manifest.csv"
    return {r["cluster30"] for r in csv.DictReader(open(f))} if f.exists() else set()


def mark(df: pd.DataFrame, cluster: dict, similar: dict) -> pd.DataFrame:
    """Give every table the same cluster and subset labels, taken from one shared map.

    Each table is built separately and can pick a different representative for the same 30 %-identity cluster, so
    deriving the subsets per table puts the same structure in different subsets for different methods and the
    comparison stops being paired. One map, keyed by structure, keeps the subsets and the bootstrap clusters
    identical for every method.
    """
    df["cluster30"] = df["pdb"].map(lambda p: cluster.get(p, p))
    df["train_similar"] = df["pdb"].map(lambda p: bool(similar.get(p, False)))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--tools", nargs="*", default=["p2rank", "fpocket"])
    ap.add_argument("--ours", nargs="+", default=["ranker", "native order"], help="score columns in our eval table")
    ap.add_argument("--ref", default="", help="method to take paired differences against (default: the first tool)")
    ap.add_argument("--tag", default="", help="suffix of the eval_candidates_<set><tag>.csv to read, as passed to "
                    "evaluate.py --tag; the receptor protocol must match the baselines', which see every chain")
    ap.add_argument("--out", default=str(REPO / "docs/results"))
    # `--tag` names an input, so without this a probe run -- a partial tool table, a different ligand rule --
    # overwrites the published comparison for that tag and nothing says so.
    ap.add_argument("--out-tag", dest="out_tag", default=None,
                    help="suffix of the written compare_<set><out-tag>.md/.json; defaults to --tag")
    a = ap.parse_args()
    if a.out_tag is None:
        a.out_tag = a.tag

    ours = pd.read_csv(DS / f"eval_candidates_{a.set}{a.tag}.csv")
    train_cl = train_clusters()
    if "cluster30" not in ours:
        ours["cluster30"] = ours["pdb"]
    if "train_similar" not in ours:
        ours["train_similar"] = ours["cluster30"].isin(train_cl)
    first = ours.drop_duplicates("pdb").set_index("pdb")
    cluster = first["cluster30"].to_dict()                 # one cluster and subset label per structure, ours
    similar = first["train_similar"].astype(bool).to_dict()
    scored = {}                                  # name -> (table, score column, higher is better)
    for col in a.ours:
        if col in ours:
            scored[f"ours ({col})"] = (mark(ours.copy(), cluster, similar), col)
        else:
            print(f"  our eval table has no column {col!r}; skipped")
    for tool in a.tools:
        try:
            df = tables.read_table(DS, f"candidates_{tool}_{a.set}")
        except FileNotFoundError:
            print(f"  candidates_{tool}_{a.set} not found; run scripts/baselines/run_external.py for this set")
            continue
        df = mark(df, cluster, similar)
        scored[tool] = (df.assign(_s=-df["tool_rank"]), "_s")        # the tool's own ranking

    if len(scored) < 2:
        sys.exit("need our table and at least one tool table")
    common = set.intersection(*[set(df["pdb"]) for df, _ in scored.values()])
    print(f"{len(common)} structures predicted by all of {', '.join(scored)}")

    results = {}
    for name, (df, col) in scored.items():
        d = df[df["pdb"].isin(common)]
        for subset, sub in (("all", d), ("not train-similar", d[~d.train_similar]), ("train-similar", d[d.train_similar])):
            if sub.empty:
                continue
            per = M.per_structure(sub, col)
            st = M.summarize(per)
            st["mean_predictions"] = float(len(sub) / sub["pdb"].nunique())
            results.setdefault(subset, {})[name] = dict(st, _per=per)

    ref = a.ref or next((t for t in a.tools if t in scored), "")
    for subset, d in results.items():
        if ref not in d:
            continue
        base = d[ref]["_per"]
        for name, st in d.items():
            if name == ref:
                continue
            per = st["_per"].set_index("pdb").loc[base["pdb"]]
            st["paired_vs_ref"] = {c: M.paired_boot(per[c].to_numpy(float), base[c].to_numpy(float),
                                                    base["cluster30"].to_numpy())
                                   for c in ("top1", "topN", "topN2")}

    lines = [f"# {a.set}: our predictions against external tools on the same structures", "",
             f"{len(common)} structures predicted by every method listed. Success is DCA <= 4 A to a ligand of the "
             f"set's own relevant-ligand list; N is the structure's own number of ligand sites. 95 % CI by 30 %-"
             f"identity cluster bootstrap. `train-similar` means the structure shares a 30 %-identity cluster with "
             f"**our** training manifest; it is leakage for us and not for the external tools, whose own training "
             f"sets overlap this benchmark in ways this table does not measure. The comparable row for us is "
             f"**not train-similar**.", "",
             "| subset | method | predictions | ceiling | top-1 | top-3 | top-N | top-(N+2) | MRR | n |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for subset, d in results.items():
        for name, st in d.items():
            lines.append(f"| {subset} | {name} | {st['mean_predictions']:.1f} | {st['ceiling']:.3f} | "
                         f"{M.fmt(st['top1'])} | {M.fmt(st['top3'])} | {M.fmt(st['topN'])} | {M.fmt(st['topN2'])} | "
                         f"{M.fmt(st['mrr'])} | {st['n']} |")
    if ref:
        lines += ["", f"Paired differences against **{ref}** on the same structures (positive = we are better):", "",
                  "| subset | method | top-1 | top-N | top-(N+2) |", "|---|---|---|---|---|"]
        for subset, d in results.items():
            for name, st in d.items():
                p = st.get("paired_vs_ref")
                if not p:
                    continue
                cells = " | ".join(f"{p[c]['diff']:+.3f} [{p[c]['lo']:+.3f}, {p[c]['hi']:+.3f}]" for c in ("top1", "topN", "topN2"))
                lines.append(f"| {subset} | {name} | {cells} |")

    # Where the difference lives. At N = 1 top-N is top-1, so merging predictions cannot move it: any deficit there
    # is the ranker putting the wrong candidate first, not fragments eating the budget. Splitting by the structure's
    # own site count separates the two explanations, which the aggregate top-N hides.
    if ref:
        lines += ["", f"Top-N by the structure's own number of sites, against **{ref}**. At N = 1 top-N is top-1 and "
                  "no merging of predictions can change it, so a deficit in that row is ranking and a deficit "
                  "confined to N >= 2 is prediction fragmentation:", "",
                  "| subset | method | N | n | top-N | " + f"{ref} top-N | difference | ceiling | {ref} ceiling |",
                  "|---|---|---|---|---|---|---|---|---|"]
        sites = {p: int(n) for p, n in ours.drop_duplicates("pdb").set_index("pdb")["n_sites"].items()}
        for subset, d in results.items():
            base = d.get(ref)
            for name, st in d.items():
                if name == ref or not st.get("paired_vs_ref"):
                    continue
                per = st["_per"].set_index("pdb"); ref_per = base["_per"].set_index("pdb")
                n_of = pd.Series({p: sites.get(p, 1) for p in per.index})
                for label, sel in (("1", n_of == 1), ("2", n_of == 2), (">=3", n_of >= 3)):
                    idx = n_of.index[sel]
                    if len(idx) == 0:
                        continue
                    o, r = per.loc[idx], ref_per.loc[idx]
                    lines.append(f"| {subset} | {name} | {label} | {len(idx)} | {o.topN.mean():.3f} | {r.topN.mean():.3f} | "
                                 f"{o.topN.mean() - r.topN.mean():+.3f} | {o.ceiling.mean():.3f} | {r.ceiling.mean():.3f} |")

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"compare_{a.set}{a.out_tag}.md").write_text("\n".join(lines) + "\n")
    clean = {s: {m: {k: v for k, v in st.items() if k != "_per"} for m, st in d.items()} for s, d in results.items()}
    # Which of our eval tables was compared against which tool tables, so the row cannot be read without knowing
    # the protocol behind it.
    (out / f"compare_{a.set}{a.out_tag}.json").write_text(json.dumps(dict(
        set=a.set, n_common=len(common), ref=ref, run=dict(vars(a)), results=clean), indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
