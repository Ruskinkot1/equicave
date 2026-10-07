"""Where the first-rank gap to a reference method actually sits, structure by structure.

The aggregate says we trail P2Rank on COACH420 top-1 by about five points. That number alone cannot say whether
the pocket is missing from our list, buried deep in it, or sitting one place below our own first pick -- and those
three call for different work. This splits the gap:

* the win/loss/net counts, because a net gap of eight structures out of 174 built from 52 disagreements is a
  different object from a systematic loss;
* how deep our first correct candidate sits when we lose, which separates a ranking problem from a detection one;
* how far our wrong first pick is from the nearest true site, which separates a near-miss on the right cavity
  from a confident vote for a different one;
* what a perfect choice among our own top k would score, which is the ceiling of any re-ranker over our shortlist.

Both methods are ranked over their own candidate sets, so the ranks are not comparable between them; only the
hit/miss per structure is. Structures similar to the training manifest are dropped by default for the same reason
the headline table drops them.

    PYTHONPATH=src:. python scripts/eval/rank1_residual.py --set coach420 --ref p2rank
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from equicave import metrics as M, tables  # noqa: E402

DS = REPO / "data/processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--ref", default="p2rank", help="the method to split the gap against")
    ap.add_argument("--ours", default="ranker", help="score column in our eval table")
    ap.add_argument("--tag", default="", help="suffix of eval_candidates_<set><tag>.csv")
    ap.add_argument("--train-similar", dest="train_similar", action="store_true",
                    help="keep the structures similar to our training manifest, which the headline table drops")
    ap.add_argument("--out", default="", help="write the numbers to this JSON as well as printing them")
    a = ap.parse_args()

    ours = pd.read_csv(DS / f"eval_candidates_{a.set}{a.tag}.csv")
    ref = tables.read_table(DS, f"candidates_{a.ref}_{a.set}")
    first = ours.drop_duplicates("pdb").set_index("pdb")
    cl, sim = first["cluster30"].to_dict(), first["train_similar"].astype(bool).to_dict()
    ref["cluster30"] = ref["pdb"].map(cl)
    common = sorted(set(ours["pdb"]) & set(ref["pdb"]))
    if not a.train_similar:
        common = [p for p in common if not sim.get(p, False)]
    if not common:
        sys.exit("no structures in common")

    po = M.per_structure(ours[ours["pdb"].isin(common)], a.ours).set_index("pdb").loc[common]
    r = ref[ref["pdb"].isin(common)]
    pr = M.per_structure(r.assign(_s=-r["tool_rank"]), "_s").set_index("pdb").loc[common]

    lose = po.index[(~po["top1"]) & pr["top1"]]
    win = po.index[po["top1"] & (~pr["top1"])]
    out = dict(set=a.set, ref=a.ref, n=len(common), train_similar_kept=bool(a.train_similar),
               top1_ours=float(po["top1"].mean()), top1_ref=float(pr["top1"].mean()),
               lose=len(lose), win=len(win), agree=int((po["top1"] == pr["top1"]).sum()))
    print(f"{out['n']} structures{'' if a.train_similar else ', not train-similar'}: "
          f"top-1 ours {out['top1_ours']:.3f}, {a.ref} {out['top1_ref']:.3f}")
    print(f"we lose on {out['lose']}, win on {out['win']}, net {out['win'] - out['lose']}, "
          f"agree on {out['agree']}")

    # Where the answer sits in our own list on the structures we lose: a ranking problem looks different from a
    # detection one, and only the first is a re-ranker's to solve.
    f = po.loc[lose, "first"]
    depth = {"2-3": int(((f >= 2) & (f <= 3)).sum()), "4-10": int(((f >= 4) & (f <= 10)).sum()),
             ">10": int((f > 10).sum()), "absent": int(f.isna().sum()), "exactly 2": int((f == 2).sum())}
    out["our_rank_of_the_answer_when_we_lose"] = depth
    print("our first correct candidate there:",
          ", ".join(f"{k} {v}" for k, v in depth.items()))

    # And how wrong our first pick is: a few angstroms off the right cavity is a different error from a vote for
    # a cavity on the other side of the protein, and re-centring only ever addressed the first.
    top = ours.sort_values(a.ours, ascending=False).groupby("pdb", sort=False).head(1).set_index("pdb")
    d = top.loc[lose, "dca"].to_numpy(float)
    miss = {"4-8 A": int(((d > 4) & (d <= 8)).sum()), "8-15 A": int(((d > 8) & (d <= 15)).sum()),
            ">15 A": int((d > 15).sum()), "median": float(np.median(d))}
    out["dca_of_our_first_pick_when_we_lose"] = miss
    print(f"distance from our first pick to the nearest true site: median {miss['median']:.1f} A, "
          f"4-8 A {miss['4-8 A']}, 8-15 A {miss['8-15 A']}, >15 A {miss['>15 A']}")

    # The ceiling of any re-ranker over our own shortlist. Not reachable -- a learned chooser loses structures the
    # current order gets right -- but it bounds the idea, and the k=2 row says how narrow the decision can be.
    out["oracle_over_our_top_k"] = {str(k): float((po["first"] <= k).mean()) for k in (1, 2, 3, 5, 10)}
    out["oracle_over_our_whole_list"] = float(po["ceiling"].mean())
    print("a perfect choice among our own top k:",
          ", ".join(f"k={k} {v:.3f}" for k, v in out["oracle_over_our_top_k"].items()),
          f"| whole list {out['oracle_over_our_whole_list']:.3f}")

    ns = ours.drop_duplicates("pdb").set_index("pdb")["n_sites"]
    out["by_site_count"] = {}
    for lab, pick in (("1", lambda v: v == 1), ("2", lambda v: v == 2), (">=3", lambda v: v >= 3)):
        sel = [p for p in common if pick(int(ns.get(p, 1)))]
        if sel:
            out["by_site_count"][lab] = dict(n=len(sel), lost=len(set(sel) & set(lose)),
                                             ours=float(po.loc[sel, "top1"].mean()),
                                             ref=float(pr.loc[sel, "top1"].mean()))
            s = out["by_site_count"][lab]
            print(f"  sites {lab}: {s['lost']}/{s['n']} lost, ours {s['ours']:.3f}, {a.ref} {s['ref']:.3f}")

    out["structures_we_lose"] = list(lose)
    out["structures_we_win"] = list(win)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
