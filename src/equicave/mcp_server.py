"""EquiCave as an MCP server: pocket detection, ranking, properties and hotspots as tools for agents.

Run:  PYTHONPATH=src python -m equicave.mcp_server            (stdio transport; needs the `mcp` package)
Tools
  detect_pockets(pdb_path, top_k=10, ranker=None)      ranked sites: centre, box, residues, native + ranker scores
  pocket_properties(pdb_path, center, model=None)      multi-label property probabilities (network) or geometry classes
  hotspot_field(pdb_path, center, model=None, radius=8) grid points with per-class ligand-atom probabilities
  cavity_mask(pdb_path, center, radius=8)              irregular cavity points + docking box
All tools are pure functions over the structure file; nothing external is called. Without a trained network the
property and hotspot tools fall back to geometry-only estimates and say so in `note`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import detect, labels as LB, pocket_features as pf, pockets as pk, structure

MODELS = Path(__file__).resolve().parents[2] / "models"


def _json(o):
    if isinstance(o, np.ndarray):
        return o.round(3).tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def detect_pockets(pdb_path: str, top_k: int = 10, ranker: str | None = None, chains: str | None = None) -> dict:
    st = structure.read_pdb(pdb_path, chains)
    cands = detect.detect_sites(st["xyz"])
    rk = ranker or (str(MODELS / "ranker_native.txt") if (MODELS / "ranker_native.txt").exists() else None)
    rows = pf.rank_sites(rk, cands, st) if rk else pf.featurize(cands, st)
    from scipy.spatial import cKDTree
    tp = cKDTree(st["xyz"]); out = []
    for i, r in enumerate(rows[:top_k], 1):
        c = np.asarray(r["center"]); ctr, box, cav = pk.cavity_box(c, st["xyz"], 8.0)
        near = tp.query_ball_point(c, 8.0)
        out.append(dict(rank=i, center=c.round(2).tolist(), box_center=ctr.round(2).tolist(), box=np.round(box, 1).tolist(),
                        residues=sorted(set(st["resid"][near].tolist())), native_score=r["nat_score"], native_rank=int(r["nat_rank"]),
                        ranker_score=r.get("ranker_score"), cavity_points=int(len(cav)), shape_axes=np.round(pk.shape_axes(cav), 1).tolist()))
    return dict(pdb=Path(pdb_path).stem, n_candidates=len(rows), ranker=rk, sites=out)


def cavity_mask(pdb_path: str, center: list[float], radius: float = 8.0) -> dict:
    st = structure.read_pdb(pdb_path)
    ctr, box, cav = pk.cavity_box(np.asarray(center, float), st["xyz"], radius)
    return dict(box_center=ctr.round(2).tolist(), box=np.round(box, 1).tolist(), points=np.round(cav, 1).tolist())


def _net(model_path):
    import torch
    from training.pockets import net_task as NT
    return NT.load_model(model_path, torch.device("cpu"))


def pocket_properties(pdb_path: str, center: list[float], model: str | None = None) -> dict:
    st = structure.read_pdb(pdb_path); c = np.asarray(center, float)
    model = model or (str(MODELS / "pockets_net.pt") if (MODELS / "pockets_net.pt").exists() else None)
    if model:
        import torch
        from training.pockets import data as D
        net, cfg = _net(model)
        d = D.featurize(pdb_path, None, None, cfg["data"]["n_probe"], cfg["data"]["n_surf"],
                        **D.featurisation_kwargs(cfg["data"]))
        b = D.to_torch(d)
        ppos = d["pos"][d["n_res"]:d["n_res"] + d["n_probe"]]
        b["site_probe_mask"] = torch.tensor((np.linalg.norm(ppos - c, axis=1) <= 8.0)[None].astype(np.float32))
        with torch.no_grad():
            p = torch.sigmoid(net(b)["prop_logit"])[0].numpy()
        return dict(classes=dict(zip(LB.PROPERTY_CLASSES, p.round(3).tolist())), note="network probabilities (validate calibration per model card)")
    _, _, cav = pk.cavity_box(c, st["xyz"], 8.0)
    from scipy.spatial import cKDTree
    bur = pk._buried(cav[:300], cKDTree(st["xyz"])).mean()
    near = cKDTree(st["xyz"]).query_ball_point(c, 8.0)
    el = st["element"][near]; polar = float(np.isin(el, ["N", "O"]).mean()) if len(el) else 0.0
    return dict(classes=dict(size_small=float(len(cav) < 60), size_large=float(len(cav) > 250), buried_deep=float(bur >= 16), buried_shallow=float(bur < 10),
                             polar=float(polar > 0.35), apolar=float(polar < 0.2)), note="geometry-only estimate; no network model available")


def hotspot_field(pdb_path: str, center: list[float], model: str | None = None, radius: float = 8.0) -> dict:
    st = structure.read_pdb(pdb_path); c = np.asarray(center, float)
    model = model or (str(MODELS / "pockets_net.pt") if (MODELS / "pockets_net.pt").exists() else None)
    _, _, cav = pk.cavity_box(c, st["xyz"], radius)
    if model:
        import torch
        from training.pockets import data as D
        net, cfg = _net(model)
        d = D.featurize(pdb_path, None, None, cfg["data"]["n_probe"], cfg["data"]["n_surf"],
                        **D.featurisation_kwargs(cfg["data"]))
        with torch.no_grad():
            out = net(D.to_torch(d))
        ppos = d["pos"][d["n_res"]:d["n_res"] + d["n_probe"]]
        m = np.linalg.norm(ppos - c, axis=1) <= radius
        hot = torch.sigmoid(out["hot_logit"]).numpy()[m]
        return dict(classes=LB.HOTSPOT_CLASSES, points=np.round(ppos[m], 1).tolist(), probabilities=hot.round(3).tolist(), note="network field")
    from scipy.spatial import cKDTree
    bur = pk._buried(cav, cKDTree(st["xyz"])) / 26.0
    return dict(classes=["buriedness"], points=np.round(cav, 1).tolist(), probabilities=[[float(x)] for x in bur], note="geometry-only field (buriedness); no network model available")


def main():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise SystemExit("pip install mcp   (the MCP server needs the `mcp` package; the functions above work without it)")
    srv = FastMCP("equicave")
    srv.tool()(detect_pockets); srv.tool()(cavity_mask); srv.tool()(pocket_properties); srv.tool()(hotspot_field)
    srv.run()


if __name__ == "__main__":
    main()
