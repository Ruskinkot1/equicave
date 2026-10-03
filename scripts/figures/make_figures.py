#!/usr/bin/env python
"""Build the repository figures from real data: the pipeline schematic and rendered examples.

Usage: python scripts/figures/make_figures.py [--pdb 10QV] [--out docs/figures]
Writes pipeline.svg (schematic), candidates_overview.png, pocket_example.png, buriedness_slice.png,
hotspot_labels.png (interaction-validated labels of a real ligand) and ranker_top1.png when results exist.
"""
import argparse, pathlib, sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
SVG = REPO / "docs/figures/pipeline.svg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default=""); ap.add_argument("--out", default=str(REPO / "docs/figures"))
    ap.add_argument("--pdb-dir", default=str(REPO / "data/pockets_ds/pdb"))
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    from equicave import ccd, detect, labels as LB, pockets as pk, structure, viz
    pdb_dir = pathlib.Path(a.pdb_dir)
    path = (pdb_dir / f"{a.pdb}.pdb") if a.pdb else None
    if path is None or not path.exists():
        path = next((p for p in sorted(pdb_dir.glob("*.pdb")) if structure.read_ligands(p, min_heavy=12)), None)
    if path is None:
        print("no structure with a ligand available; only the schematic exists"); return 0
    st = structure.read_pdb(path)
    ligs = structure.read_ligands(path, min_heavy=12)
    cands = detect.detect_sites(st["xyz"])
    L = np.vstack([l["xyz"] for l in ligs])
    best = min(cands, key=lambda c: pk.dca(c["center"], L))
    viz.plot_sites_overview(path, top_k=5).savefig(out / "candidates_overview.png", dpi=130, bbox_inches="tight")
    viz.plot_pocket(path, best, ligands=ligs,
                    title=f"{path.stem}: candidate {best['rank']} at the ligand (DCA {pk.dca(best['center'], L):.2f} A)"
                    ).savefig(out / "pocket_example.png", dpi=130, bbox_inches="tight")
    viz.plot_buriedness_slice(path).savefig(out / "buriedness_slice.png", dpi=130, bbox_inches="tight")
    entries = {l["comp"]: ccd.load(l["comp"]) for l in ligs}
    _, hot, prox = LB.point_labels(best["points"], ligs, entries, st=st)
    viz.plot_hotspot_field(best["points"], hot.astype(float), LB.HOTSPOT_CLASSES, ligand=L, threshold=0.5
                           ).savefig(out / "hotspot_labels.png", dpi=100, bbox_inches="tight")
    print(f"{path.stem}: interaction-validated coverage {dict(zip(LB.HOTSPOT_CLASSES, hot.mean(0).round(3).tolist()))}")
    print(f"  proximity-only coverage      {dict(zip(LB.HOTSPOT_CLASSES, prox.mean(0).round(3).tolist()))}")
    if (REPO / "docs/results/ranker_native.json").exists():
        viz.plot_metric_bars(viz.comparison_table(keys=("ranker_native",)), metric="top-1",
                             title="Ranking candidates: 5-fold cross-validation by sequence cluster"
                             ).savefig(out / "ranker_top1.png", dpi=130, bbox_inches="tight")
    print("figures:", ", ".join(sorted(p.name for p in out.iterdir())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
