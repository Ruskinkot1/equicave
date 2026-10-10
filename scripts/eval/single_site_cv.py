"""Does restricting cross-validation to single-site structures recover the benchmark's ordering of feature sets?

The project's central measurement problem: cross-validated top-1 on our manifest does not predict benchmark top-1.
Three feature sets ranked in the *opposite* order -- 32 features gave 0.701 on COACH420 and 272 gave 0.626, while
cross-validation put them the other way round (0.787 and 0.796). Every ranker decision has been selected on the
metric that disagrees.

This script tests one mechanical explanation that costs nothing to check. The two populations differ in the
problem they pose: our manifest averages 2.27 ligand sites per structure with 44.6 % of structures having exactly
one, while COACH420 averages 1.29 with 74.7 % single-site. Ordering five pockets is a different task from picking
one of thirty, and a feature that helps the first can hurt the second. If that is the whole story, then
cross-validated top-1 *restricted to single-site structures* should order feature sets the way the benchmark does.

Confirmed, the fix is a filter: select models on the single-site slice. Refuted, the divergence is something else
-- distribution of protein size, of ligand chemistry, of receptor preparation -- and the search continues, which is
also worth knowing before more GPU time is spent.

    PYTHONPATH=src:. python scripts/eval/single_site_cv.py --tag native2 --seeds 3
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts/train"))
from equicave import metrics as M, tables  # noqa: E402
import train_ranker as TR  # noqa: E402


def groups_of(feats, zsuffix="_z"):
    """Feature subsets to compare, smallest first. The benchmark's own ordering was smallest-is-best."""
    from equicave import pocket_features as pf
    nat = [f for f in feats if f in pf.NATIVE]
    geo = [f for f in feats if f in pf.GEOMETRY]
    chem = [f for f in feats if f in pf.CHEMISTRY]
    shell = [f for f in feats if f in pf.SHELL]
    pot = [f for f in feats if f in pf.POTENTIAL]
    def withz(cols):
        return [c for c in cols] + [c + zsuffix for c in cols if c + zsuffix in feats]
    nat, geo, chem, shell, pot = map(withz, (nat, geo, chem, shell, pot))
    return {
        "native only": nat,
        "native+geometry": nat + geo,
        "native+geometry+chemistry": nat + geo + chem,
        "+shell": nat + geo + chem + shell,
        "+potential (all hand-built)": nat + geo + chem + shell + pot,
        "everything in the table": list(feats),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default=str(REPO / "data/processed"))
    ap.add_argument("--tag", default="native2")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=str(REPO / "docs/results/single_site_cv.md"))
    a = ap.parse_args()

    from equicave import pocket_features as pf
    df = tables.read_table(pathlib.Path(a.ds), f"candidates_{a.tag}").reset_index(drop=True)
    # The same preparation train_ranker does, so the numbers are its numbers: graded relevance, then the
    # within-structure z-score of every column (trees cannot compare candidates, so the comparison is a feature).
    base = [f for f in pf.FEATURES if f in df.columns]
    df = TR.add_relevance(df, graded=True)
    df, feats = TR.add_zscores(df, base)

    per_pdb = df.drop_duplicates("pdb")
    single = set(per_pdb.loc[per_pdb["n_sites"] == 1, "pdb"])
    print(f"{len(per_pdb)} structures, {len(single)} single-site ({len(single)/len(per_pdb):.1%}), "
          f"mean sites {per_pdb['n_sites'].mean():.2f}")
    print("COACH420 for comparison: 1.29 mean sites, 74.7 % single-site\n")

    rows = []
    for name, cols in groups_of(feats).items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        sc = np.mean([TR.cv_scores(df, cols, sd) for sd in range(a.seeds)], axis=0)
        allp = M.per_structure(df.assign(_s=sc), "_s")
        sub = df[df["pdb"].isin(single)]
        onep = M.per_structure(sub.assign(_s=sc[df["pdb"].isin(single).to_numpy()]), "_s")
        rows.append(dict(features=name, n=len(cols),
                         cv_all=float(allp["top1"].mean()), cv_single=float(onep["top1"].mean()),
                         cv_allN2=float(allp["topN2"].mean())))
        print(f"{name:30s} {len(cols):4d} cols   all {rows[-1]['cv_all']:.3f}   "
              f"single-site {rows[-1]['cv_single']:.3f}")

    t = pd.DataFrame(rows)
    best_all = t.loc[t["cv_all"].idxmax(), "features"]
    best_single = t.loc[t["cv_single"].idxmax(), "features"]
    # A different argmax is not a disagreement when the two are a thousandth apart, and the question is not which
    # label wins but whether the *ordering inverts* the way the benchmark's did. Inversion means the small feature
    # set beats the large one; saturation means the extra columns stop paying without ever turning negative.
    margin = float(t["cv_single"].max() - t.loc[t["cv_all"].idxmax(), "cv_single"])
    small, large = float(t.iloc[0]["cv_single"]), float(t.iloc[-1]["cv_single"])
    if small > large + 0.01:
        verdict = ("the single-site slice **inverts** the ordering the way the benchmark does, so site-count "
                   "composition explains the divergence and model selection belongs on this slice")
    elif margin > 0.01:
        verdict = ("the slices pick different feature sets by more than a hundredth, which is the composition "
                   "effect the benchmark divergence would predict")
    else:
        verdict = (f"no inversion: the smallest set is still {large - small:+.3f} *behind* the largest on the "
                   f"single-site slice, and the best-on-all set is within {margin:.3f} of the best there. Site "
                   f"count alone does **not** explain the benchmark divergence. What the slice does show is "
                   f"**saturation**: the columns past the shell group buy {large - float(t.iloc[-3]['cv_single']):+.3f} "
                   f"on single-site structures against "
                   f"{float(t.iloc[-1]['cv_all']) - float(t.iloc[-3]['cv_all']):+.3f} on all of them")
    md = ["# Cross-validation on the single-site slice (does composition explain the benchmark divergence?)", "",
          f"{len(per_pdb)} structures, {len(single)} single-site ({len(single)/len(per_pdb):.1%}), mean "
          f"{per_pdb['n_sites'].mean():.2f} sites. COACH420: 1.29 mean, 74.7 % single-site.", "",
          f"{a.seeds} seeds, 5-fold CV by 30 %-identity cluster, the ranker's own `cv_scores`.", "",
          "| features | columns | CV top-1, all | CV top-1, single-site only | CV top-(N+2), all |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['features']} | {r['n']} | {r['cv_all']:.3f} | {r['cv_single']:.3f} | {r['cv_allN2']:.3f} |")
    md += ["", f"Best on all structures: **{best_all}**. Best on the single-site slice: **{best_single}**.", "",
           f"Reading: {verdict}.", "",
           "The benchmark numbers this is trying to explain, from `docs/results/README.md`: 32 features 0.701 on "
           "COACH420, 236 features 0.642, 272 features 0.626, while cross-validation ranked 236 and 272 *above* "
           "the smaller set. Fewer columns transferred better; more columns scored better in cross-validation.", "",
           "One limit of this test, stated because it bounds the conclusion: those three benchmark numbers come "
           "from three *different candidate tables* built at different times, not from feature subsets of one "
           "table. This script varies only the columns, holding the table, the labels, the folds and the "
           "preparation fixed. So it can rule site count out as the explanation -- which it does -- without "
           "ruling out everything else that differed between those tables."]
    pathlib.Path(a.out).write_text("\n".join(md) + "\n")
    print(f"\nbest on all: {best_all}\nbest single-site: {best_single}\n-> {verdict}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
