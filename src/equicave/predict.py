"""One inference entry point for every consumer: the CLI, the MCP server and the evaluation script.

    sites = predict(pdb_path, mode="fast")          # ranked sites
    sites = predict(pdb_path, mode="accurate")      # plus properties and the hotspot field

The mode (`equicave.modes`) decides the candidate tiers, the network configuration and the ranker. Whatever is
missing degrades explicitly: with no trained network the sites come from the geometric detector and the learned
ranker, and `note` says the network was not used; with no ranker the detector order is returned, and `note` says so.
Nothing silently falls back to a different definition of a site.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import detect, labels as LB, modes, peptide as PEP, pocket_features as pf, pockets as pk, structure


def _load_point_model(tag: str):
    """The full-data per-point booster, or None. Loud about which one, because the choice is a leakage question.

    At inference on a structure this project never trained on, the full-data model is the right one. Scoring our
    own training structures needs the fold model that did not see them instead -- that is what `build_cache` uses
    -- so this loader is deliberately not reached from there.
    """
    f = Path(__file__).resolve().parents[2] / "models" / f"point_{tag or 'native'}.txt"
    if not f.exists():
        return None
    try:
        import lightgbm as lgb
        return lgb.Booster(model_file=str(f))
    except Exception:                      # noqa: BLE001  -- missing lightgbm must not break plain detection
        return None


def _network_features(pdb_path, cands, cfg: dict, model_path):
    """Network scores per candidate, or None when no model is available."""
    if model_path is None or not Path(model_path).exists():
        return None, None
    import torch
    from training.pockets import data as D, net_task as NT
    model, mcfg = NT.load_model(model_path, torch.device("cpu"))
    esm = None
    if mcfg["data"].get("esm"):
        from training.pockets.esm_embed import Embedder, cached
        st = structure.read_pdb(pdb_path); rt = structure.residue_table(st)
        esm = cached(Embedder(mcfg["data"]["esm"]), Path(pdb_path).stem, rt["seq"], rt["chain"],
                     Path(__file__).resolve().parents[2] / "data/cache/esm")
    pm = _load_point_model(mcfg["data"].get("point_model_tag", "")) \
        if mcfg["data"].get("probe_sampling") == "ligandable" else None
    d = D.featurize(pdb_path, None, esm, mcfg["data"]["n_probe"], mcfg["data"]["n_surf"],
                    **D.featurisation_kwargs(mcfg["data"], point_model=pm, log=print))
    if d is None:
        return None, None
    rots = cfg["test_time"]["rotations"]
    probe_pos = d["pos"][d["n_res"]:d["n_res"] + d["n_probe"]]
    centers = np.array([c["center"] for c in cands]).reshape(-1, 3)
    acc = None
    rng = np.random.default_rng(0)
    for r in range(max(1, rots)):
        b = D.to_torch(d)
        if r:
            b = D.random_rotation(b, rng)       # the network is equivariant, so this is a consistency check
        with torch.no_grad():
            out = model(b)
        feats = NT.net_features(out, probe_pos, centers)
        acc = feats if acc is None else [{k: a[k] + f[k] for k in a} for a, f in zip(acc, feats)]
    feats = [{k: v / max(1, rots) for k, v in a.items()} for a in acc]
    return feats, (out, d)


def predict(pdb_path, mode: str = "fast", model: str | Path | None = None, ranker: str | Path | None = None,
            chains: str | None = None, top_k: int = 10) -> dict:
    """Ranked binding sites of one structure, with pocket properties and hotspots when the mode asks for them."""
    cfg = modes.mode(mode)
    st = structure.read_pdb(pdb_path, chains)
    if len(st["xyz"]) < 50:
        return dict(pdb=Path(pdb_path).stem, sites=[], note="fewer than 50 heavy atoms: nothing to predict")
    det = cfg["detect"]
    cands = detect.detect_sites(st["xyz"], min_buried=det["min_buried"], fill_min_buried=det["fill_min_buried"],
                               max_sites=det["max_sites"])
    if det.get("groove"):
        grooves = PEP.detect_peptide_sites(st["xyz"], st, max_sites=det.get("groove_max_sites", 20),
                                           min_points=20, min_anisotropy=1.5, nms=4.0)
        cands = PEP.merge_with_small_molecule(grooves, cands, max_sites=det["max_sites"] + det.get("groove_max_sites", 20))
    if not cands:
        return dict(pdb=Path(pdb_path).stem, sites=[], note="no cavity passed the closure threshold")
    notes = []
    feats, net_state = _network_features(pdb_path, cands, cfg, model)
    if feats is None:
        notes.append("network not used (no model given or available)")
    # Peptide mode's ranker is trained with the 14 peptide features, and only `peptide.groove_features` produces
    # them -- the small-molecule featuriser does not. Without this the peptide path served a 264-feature model
    # without the columns that distinguish a groove from a cavity, which is the whole point of the mode. It stayed
    # invisible while the repository carried a peptide ranker predating those features; a freshly trained one made
    # the stale-feature guard fire instead, which is what the guard is for.
    extra = feats
    if det.get("groove"):
        pep_rows = PEP.groove_features(cands, st)
        extra = [dict(g, **(f or {})) for g, f in zip(pep_rows, feats or [{}] * len(cands))]
    rk = ranker or modes.ranker_path(mode)
    if rk and Path(rk).exists():
        rows = pf.rank_sites(rk, cands, st, extra=extra)
        score_key = "ranker_score"
    else:
        rows = pf.featurize(cands, st)
        if extra:
            for r, f in zip(rows, extra):
                r.update(f)
        for r in rows:
            r["ranker_score"] = -r["nat_rank"]
        score_key = "ranker_score"
        notes.append(f"ranker {Path(rk).name if rk else 'not found'}: using the detector order")
    out_sites = []
    from scipy.spatial import cKDTree
    tp = cKDTree(st["xyz"])
    for rank, r in enumerate(sorted(rows, key=lambda r: -r[score_key])[:top_k], 1):
        c = np.asarray(r["center"], float)
        ctr, box, cav = pk.cavity_box(c, st["xyz"], 8.0)
        near = tp.query_ball_point(c, 8.0)
        site = dict(rank=rank, center=c.round(2).tolist(), score=float(r[score_key]),
                    native_rank=int(r["nat_rank"]), tier=int(r.get("nat_tier", 1)),
                    box_center=ctr.round(2).tolist(), box=np.round(box, 1).tolist(),
                    residues=sorted(set(st["resid"][near].tolist())), cavity_points=int(len(cav)),
                    shape_axes=np.round(pk.shape_axes(cav), 1).tolist())
        if feats:
            site.update({k: round(float(r[k]), 4) for k in pf.NET_FEATURES if k in r})
        out_sites.append(site)
    result = dict(pdb=Path(pdb_path).stem, mode=mode, n_candidates=len(rows), sites=out_sites,
                  note="; ".join(notes) or "network and ranker both used")
    if "properties" in cfg["heads"] or "hotspots" in cfg["heads"]:
        result["heads"] = _heads(net_state, cfg, out_sites)
    return result


def _heads(net_state, cfg: dict, sites: list[dict]) -> dict:
    """Property probabilities and the hotspot field per returned site, when a network produced them."""
    if net_state is None:
        return dict(note="no trained network: properties and hotspots not predicted")
    import torch
    out, d = net_state
    probe_pos = d["pos"][d["n_res"]:d["n_res"] + d["n_probe"]]
    hot = torch.sigmoid(out["hot_logit"]).detach().cpu().numpy()
    res = {}
    for s in sites:
        c = np.asarray(s["center"], float)
        m = np.linalg.norm(probe_pos - c, axis=1) <= 8.0
        if not m.any():
            continue
        res[s["rank"]] = dict(hotspot_classes=LB.HOTSPOT_CLASSES,
                              points=np.round(probe_pos[m], 1).tolist(),
                              probabilities=hot[m].round(3).tolist())
    return dict(hotspots=res, note="property head needs site membership masks; use training/pockets for the full output")
