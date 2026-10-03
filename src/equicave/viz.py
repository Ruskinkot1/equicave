"""Plots and tables for notebooks and the paper: pockets in 3D, hotspot fields, and benchmark comparison tables.

Needs matplotlib (`pip install -e ".[viz]"`). Every function returns the Matplotlib figure or the pandas DataFrame,
so a notebook can show it and a script can save it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import detect, labels as LB, pockets as pk, structure

REPO = Path(__file__).resolve().parents[2]
PALETTE = ["#1f9e8f", "#d99a2b", "#7a5cc4", "#c0504d", "#4b7fb3", "#8aa63a", "#b5651d"]


def _axes3d(ax=None, figsize=(7, 6)):
    import matplotlib.pyplot as plt
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 1))
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_alpha(0.0)
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    return ax


def _equal_limits(ax, pts, pad=2.0):
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)


def plot_pocket(pdb_path, candidate: dict | None = None, ligands: list[dict] | None = None, radius: float = 10.0,
                show_protein: bool = True, ax=None, title: str | None = None):
    """One candidate pocket: receptor Cα trace, the cavity points coloured by buriedness, and the crystal ligand.

    `candidate`: a record from `detect.detect_sites` (or any dict with `center`); None picks the top candidate.
    """
    st = structure.read_pdb(pdb_path)
    rt = structure.residue_table(st)
    if candidate is None:
        cands = detect.detect_sites(st["xyz"])
        if not cands:
            raise ValueError("no candidate found")
        candidate = cands[0]
    c = np.asarray(candidate["center"], float)
    pts = np.asarray(candidate.get("points", np.empty((0, 3))), float)
    if len(pts) == 0:
        pts = pk.cavity_box(c, st["xyz"], 8.0)[2]
    bur = np.asarray(candidate.get("buried", pk._buried(pts, __import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(st["xyz"]))), float)
    ax = _axes3d(ax)
    near = np.linalg.norm(rt["ca"] - c, axis=1) < radius + 12
    if show_protein and near.any():
        ax.plot(*rt["ca"][near].T, color="#9aa5b1", lw=1.2, alpha=0.8, label="receptor Cα")
    s = ax.scatter(*pts.T, c=bur, cmap="viridis", s=14, alpha=0.75, label="cavity points (buriedness)")
    ax.scatter(*c[None].T, color="#c0504d", s=90, marker="X", label="predicted centre")
    if ligands is None:
        ligands = structure.read_ligands(pdb_path, min_heavy=8)
    for lg in ligands or []:
        if np.linalg.norm(lg["xyz"] - c, axis=1).min() < radius + 8:
            ax.scatter(*lg["xyz"].T, color="#d99a2b", s=28, depthshade=False, label=f"ligand {lg['comp']}")
    _equal_limits(ax, np.vstack([pts, c[None]]))
    ax.set_title(title or f"{Path(pdb_path).stem}: candidate rank {candidate.get('rank', 1)}, "
                          f"score {candidate.get('score', float('nan')):.0f}", fontsize=10)
    h, l = ax.get_legend_handles_labels()
    seen = dict(zip(l, h))
    ax.legend(seen.values(), seen.keys(), loc="upper left", fontsize=7, framealpha=0.3)
    ax.figure.colorbar(s, ax=ax, shrink=0.55, label="rays blocked (of 26)")
    return ax.figure


def plot_sites_overview(pdb_path, top_k: int = 5, ax=None):
    """The top candidates of a structure at once: Cα trace plus one coloured cloud per candidate."""
    st = structure.read_pdb(pdb_path)
    rt = structure.residue_table(st)
    cands = detect.detect_sites(st["xyz"])[:top_k]
    ax = _axes3d(ax, figsize=(7.5, 6.5))
    ax.plot(*rt["ca"].T, color="#9aa5b1", lw=0.9, alpha=0.7)
    allp = [rt["ca"]]
    for i, c in enumerate(cands):
        ax.scatter(*c["points"].T, s=10, alpha=0.6, color=PALETTE[i % len(PALETTE)], label=f"#{c['rank']} score {c['score']:.0f}")
        allp.append(c["points"])
    for lg in structure.read_ligands(pdb_path, min_heavy=8):
        ax.scatter(*lg["xyz"].T, color="black", s=20, marker="^", depthshade=False)
    _equal_limits(ax, np.vstack(allp))
    ax.set_title(f"{Path(pdb_path).stem}: top {len(cands)} candidates (ligands as black triangles)", fontsize=10)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.3)
    return ax.figure


def plot_hotspot_field(points: np.ndarray, probabilities: np.ndarray, classes: list[str] | None = None,
                       ligand: np.ndarray | None = None, threshold: float = 0.3, ncols: int = 4):
    """One panel per hotspot class: points above `threshold`, size and colour by probability.

    `probabilities`: [n_points, n_classes] (network output) or [n_points] for a single field.
    """
    import matplotlib.pyplot as plt
    P = np.atleast_2d(np.asarray(probabilities, float))
    if P.shape[0] != len(points):
        P = P.T
    classes = classes or (LB.HOTSPOT_CLASSES if P.shape[1] == len(LB.HOTSPOT_CLASSES) else [f"class {i}" for i in range(P.shape[1])])
    n = P.shape[1]; ncols = min(ncols, n); nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(3.4 * ncols, 3.2 * nrows))
    for j in range(n):
        ax = fig.add_subplot(nrows, ncols, j + 1, projection="3d")
        _axes3d(ax)
        m = P[:, j] >= threshold
        if m.any():
            ax.scatter(*points[m].T, c=P[m, j], cmap="magma", vmin=threshold, vmax=1.0, s=10 + 40 * P[m, j], alpha=0.8)
        if ligand is not None:
            ax.scatter(*np.asarray(ligand).T, color="#1f9e8f", s=18, marker="^", depthshade=False)
        _equal_limits(ax, points)
        ax.set_title(f"{classes[j]}  (p ≥ {threshold}, n={int(m.sum())})", fontsize=9)
    fig.suptitle("Predicted hotspot field: probability of a ligand atom of each class", fontsize=11)
    fig.tight_layout()
    return fig


def plot_buriedness_slice(pdb_path, axis: int = 2, index: int | None = None, ax=None):
    """A 2D slice of the buriedness field: what the candidate generator actually sees."""
    import matplotlib.pyplot as plt
    st = structure.read_pdb(pdb_path)
    cands, field = detect.detect_sites(st["xyz"], return_field=True)
    bur = field["buried"] * field["free"]
    if index is None:
        index = int(np.argmax(bur.sum(axis=tuple(i for i in range(3) if i != axis))))
    sl = [slice(None)] * 3; sl[axis] = index
    img = bur[tuple(sl)]
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(img.T, origin="lower", cmap="viridis", vmin=0, vmax=26)
    ax.set_title(f"{Path(pdb_path).stem}: buriedness slice (axis {axis}, index {index})", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.figure.colorbar(im, ax=ax, shrink=0.8, label="rays blocked (of 26)")
    return ax.figure


def plot_metric_bars(table: pd.DataFrame, metric: str = "top-1", ax=None, title: str | None = None):
    """Horizontal bars with confidence intervals from a comparison table ('0.724 [0.701, 0.748]' cells)."""
    import matplotlib.pyplot as plt
    def parse(x):
        if not isinstance(x, str) or "[" not in x:
            return (np.nan, np.nan, np.nan)
        m, ci = x.split("[")
        lo, hi = ci.strip("] ").split(",")
        return float(m), float(lo), float(hi)
    vals = [parse(v) for v in table[metric]]
    names = table.iloc[:, 0].tolist()
    ok = [i for i, v in enumerate(vals) if not np.isnan(v[0])]
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 0.5 * len(ok) + 1.5))
    y = np.arange(len(ok))
    m = np.array([vals[i][0] for i in ok]); lo = np.array([vals[i][1] for i in ok]); hi = np.array([vals[i][2] for i in ok])
    ax.barh(y, m, color=[PALETTE[i % len(PALETTE)] for i in range(len(ok))], alpha=0.85)
    ax.errorbar(m, y, xerr=[m - lo, hi - m], fmt="none", ecolor="#333", capsize=3, lw=1)
    ax.set_yticks(y); ax.set_yticklabels([names[i] for i in ok], fontsize=8)
    ax.set_xlabel(f"{metric} success rate (DCA ≤ 4 Å), 95 % CI by cluster bootstrap")
    ax.set_title(title or f"{metric} on the same structures", fontsize=10)
    ax.set_xlim(0, 1); ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    return ax.figure


# ---- tables --------------------------------------------------------------------------------------------------------
PUBLISHED = pd.DataFrame([
    ("fpocket", "geometry (alpha spheres)", 0.228, 0.312, 0.291),
    ("P2Rank", "random forest over SAS points", 0.728, 0.787, 0.826),
    ("DeepPocket", "3D CNN rescoring of fpocket", 0.761, 0.561, np.nan),
    ("DeepSurf", "surface CNN", 0.731, 0.635, 0.732),
    ("GrASP", "graph attention over atoms", 0.78, np.nan, np.nan),
    ("VN-EGNN", "equivariant GNN, sphere virtual nodes", 0.750, 0.659, 0.820),
], columns=["method", "type", "COACH420 top-N", "HOLO4K top-N", "PDBbind2020 top-N"])
PUBLISHED.attrs["note"] = ("Values quoted from the respective papers from memory; protocols, ligand filters and splits "
                           "differ, so they are a bar to reproduce, not a comparison. Recheck against the primary "
                           "sources before publication.")


def published_reference() -> pd.DataFrame:
    """Published top-N DCA numbers of competing methods (context only, see the `note` attribute)."""
    return PUBLISHED.copy()


def load_results(path="docs/results") -> dict:
    """Every result JSON written by the scripts, keyed by file stem."""
    out = {}
    for f in sorted((REPO / path if not Path(path).is_absolute() else Path(path)).glob("*.json")):
        try:
            out[f.stem] = json.loads(f.read_text())
        except json.JSONDecodeError:
            pass
    return out


def comparison_table(results: dict | None = None, keys=("ranker_native", "methods_comparison"),
                     metrics=("top1", "top3", "topN", "topN2", "mrr")) -> pd.DataFrame:
    """One tidy table from the result JSONs: method, n, ceiling and each metric as 'mean [lo, hi]'."""
    results = results or load_results()
    rename = dict(top1="top-1", top3="top-3", topN="top-N", topN2="top-(N+2)", mrr="MRR")
    rows = []
    for key in keys:
        r = results.get(key)
        if not r:
            continue
        for name, st in (r.get("results") or {}).items():
            row = dict(method=name, source=key, n=st.get("n"), ceiling=round(st.get("ceiling", float("nan")), 3))
            for m in metrics:
                v = st.get(m)
                row[rename.get(m, m)] = f"{v['mean']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]" if isinstance(v, dict) else None
            p = st.get("paired_vs_native_order") or st.get("paired_vs_first") or {}
            if p.get("top1"):
                row["gain top-1"] = f"{p['top1']['diff']:+.3f} [{p['top1']['lo']:+.3f}, {p['top1']['hi']:+.3f}]"
            rows.append(row)
    return pd.DataFrame(rows)


def eval_table(results: dict | None = None, subset: str = "not train-similar") -> pd.DataFrame:
    """Benchmark evaluations (`eval_<set>.json`) in one table, one row per set and method."""
    results = results or load_results()
    rename = dict(top1="top-1", top3="top-3", topN="top-N", topN2="top-(N+2)")
    rows = []
    for key, r in results.items():
        if not key.startswith("eval_"):
            continue
        d = (r.get("results") or {})
        for sub in (subset, "all"):
            if sub in d:
                for name, st in d[sub].items():
                    rows.append(dict(benchmark=key[5:], subset=sub, method=name, n=st.get("n"),
                                     ceiling=round(st.get("ceiling", float("nan")), 3),
                                     **{rename[m]: f"{st[m]['mean']:.3f} [{st[m]['lo']:.3f}, {st[m]['hi']:.3f}]"
                                        for m in rename if isinstance(st.get(m), dict)}))
                break
    return pd.DataFrame(rows)
