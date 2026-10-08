"""Does a P2Rank-style long-range protrusion count add anything over the counts and closure we already have?

The question is not whether protrusion works -- P2Rank measured that, and it is its most important feature by a
factor of six -- but whether *we already have it* under another name. Our probe channels carry atom counts at 4, 6
and 8 A and a 26-direction closure field, so a 10 A count could be redundant, and adding a redundant channel is
how the ESM-2, surface and degree-2 arms each cost a week for a null.

Correct-against-decoy discrimination at our own detector's candidate centres, the same setup as the metal check.
A candidate is correct when its centre is within 4 A of a site centre, the rule the benchmark scores. This is a
proxy for top-1, not top-1 itself: it says how separable the two populations are by these inputs alone.

Result on 2026-10-08, 268 structures, 8039 candidates, 565 correct (7.0 %):
    single-count AUC   cnt4 0.376   cnt6 0.466   cnt8 0.650   cnt10 0.792   cnt12 0.837   cnt15 0.832
                       buried26 0.755
    5-fold CV AUC      ours (cnt4,6,8 + buried26) 0.7999 +- 0.0189
                       + cnt10                    0.8224 +- 0.0132
                       + cnt10,12,15              0.8469 +- 0.0154
So: not redundant. The long radius carries 0.047 of AUC that nothing we had was carrying, which is why
`data.probe_protrusion` exists.

Usage: PYTHONPATH=src:. python scripts/eval/probe_geometry_headroom.py
"""
import csv, json, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from equicave import structure, labels as LB, detect, pockets as pk

RADII = (4.0, 6.0, 8.0, 10.0, 12.0, 15.0)
rows = list(csv.DictReader(open("data/processed/manifest.csv")))
rng = np.random.default_rng(1)
rows = [rows[i] for i in rng.permutation(len(rows))[:300]]
X, y = [], []
n = 0
for r in rows:
    p = Path("data/pockets_ds/pdb") / f"{r['pdb']}.pdb"
    if not p.exists():
        continue
    try:
        codes = {l[0] for l in json.loads(r["ligands"])}
        ligs = [l for l in structure.read_ligands(p, min_heavy=8) if l["comp"] in codes]
        if not ligs:
            continue
        st = structure.read_pdb(p)
        cen = LB.site_centres(ligs, LB.group_sites(ligs))
        cands, field = detect.detect_sites(st["xyz"], return_field=True)
        c = np.array([x["center"] for x in cands], float).reshape(-1, 3)
        if not len(c):
            continue
        tree = cKDTree(st["xyz"])
        cnt = np.stack([[len(q) for q in tree.query_ball_point(c, rad)] for rad in RADII], 1).astype(float)
        bur = pk._buried(c, tree).astype(float)          # our 26-direction closure
        X.append(np.c_[cnt, bur]); y.append((cKDTree(cen).query(c)[0] <= 4.0).astype(int))
        n += 1
    except Exception:
        continue
X = np.vstack(X); y = np.concatenate(y)
names = [f"cnt{int(r)}" for r in RADII] + ["buried26"]
print(f"structures {n}; candidates {len(y)}; correct {y.sum()} ({y.mean():.1%})")
print("\nsingle-feature AUC (correct vs decoy):")
for i, nm in enumerate(names):
    print(f"  {nm:9s} {roc_auc_score(y, X[:, i]):.3f}")
print("\ncorrelation with buried26 and with cnt8:")
for i, nm in enumerate(names):
    print(f"  {nm:9s} r(buried26)={np.corrcoef(X[:, i], X[:, -1])[0,1]:+.3f}  r(cnt8)={np.corrcoef(X[:, i], X[:, 2])[0,1]:+.3f}")
# does cnt10 add over what we have? paired logistic comparison
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
have = [0, 1, 2, 6]                                    # cnt4, cnt6, cnt8, buried26
for label, cols in (("ours (cnt4,6,8 + buried26)", have), ("+ cnt10", have + [3]), ("+ cnt10,12,15", have + [3, 4, 5])):
    s = cross_val_score(LogisticRegression(max_iter=2000), X[:, cols], y, cv=5, scoring="roc_auc")
    print(f"{label:28s} AUC {s.mean():.4f} +- {s.std():.4f}")
