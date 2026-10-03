"""Features of native pocket candidates for the learned ranker; shared by training and inference.

Every feature is a plain number computed from the protein coordinates and the candidate itself. There are no
finder scores from external programs: the first five `nat_*` features describe the candidate as produced by
`equicave.detect`; `net_*` features (optional, `NET_FEATURES`) come from the network heads when a model is available.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import detect, pockets as pk

HYDROPHOBIC = set("ALA VAL LEU ILE MET PHE TRP CYS".split())
AROMATIC = set("PHE TRP TYR HIS".split())
POLAR = set("SER THR ASN GLN".split())
POS, NEG = set("LYS ARG HIS".split()), set("ASP GLU".split())
ENV_R = 8.0
CAV_R = 8.0

NATIVE = ["nat_score", "nat_rank", "nat_rel", "nat_npts", "nat_mean_bur", "nat_max_bur", "nat_cavity_npts", "nat_tier"]
GEOMETRY = ["cav_volume", "ax1", "ax2", "ax3", "buried_mean", "vol_rank", "centrality", "depth"]
CHEMISTRY = ["n_atoms", "f_C", "f_N", "f_O", "f_S", "f_backbone", "n_res", "r_hydrophobic", "r_aromatic", "r_polar",
             "r_pos", "r_neg", "r_gly", "r_pro"]
CONTEXT = ["prot_n_res", "n_cands"]
FEATURES = NATIVE + GEOMETRY + CHEMISTRY + CONTEXT
NET_FEATURES = ["net_seg", "net_center_conf", "net_hot_mean"]          # appended when a network model scores the sites
PEPTIDE = ["pep_tier", "pep_length", "pep_width", "pep_anisotropy", "pep_flatness", "pep_bb_n", "pep_bb_o", "pep_bb_ca",
           "pep_sc_c", "pep_sc_polar", "pep_bb_total", "pep_bb_ratio", "pep_bb_per_point", "pep_overlap_dist"]
GROUPS = dict(native=NATIVE, geometry=GEOMETRY, chemistry=CHEMISTRY, context=CONTEXT, network=NET_FEATURES, peptide=PEPTIDE)


def featurize(cands: list[dict], st: dict, radius: float = CAV_R) -> list[dict]:
    """One feature dict per candidate from `detect.detect_sites`; `st` is `structure.read_pdb(...)`."""
    xyz = st["xyz"]; tp = cKDTree(xyz)
    centroid = xyz.mean(0); rg = float(np.sqrt(((xyz - centroid) ** 2).sum(1).mean())) or 1.0
    n_res_total = len(set(st["resid"]))
    table = detect.candidate_table(cands)
    rows = []
    for c, t in zip(cands, table):
        _, _, cav = pk.cavity_box(c["center"], xyz, radius)
        sub = cav[np.random.default_rng(0).permutation(len(cav))[:300]]
        near = tp.query_ball_point(c["center"], ENV_R)
        el, rn, rid, bb = st["element"][near], st["resname"][near], st["resid"][near], st["backbone"][near]
        names = list({r: n for r, n in zip(rid, rn)}.values()); nr = max(1, len(names))
        ax = pk.shape_axes(cav) + [0.0, 0.0, 0.0]
        fr = lambda e: float((el == e).mean()) if len(el) else 0.0
        rows.append(dict(
            center=c["center"], **{k: t[k] for k in NATIVE}, n_cands=t["n_cands"],
            cav_volume=float(len(cav)), ax1=ax[0], ax2=ax[1], ax3=ax[2],
            buried_mean=float(pk._buried(sub, tp).mean()) if len(sub) else 0.0,
            depth=float(tp.query(c["center"])[0]),
            n_atoms=len(near), f_C=fr("C"), f_N=fr("N"), f_O=fr("O"), f_S=fr("S"),
            f_backbone=float(bb.mean()) if len(bb) else 0.0, n_res=len(names),
            r_hydrophobic=sum(n in HYDROPHOBIC for n in names) / nr, r_aromatic=sum(n in AROMATIC for n in names) / nr,
            r_polar=sum(n in POLAR for n in names) / nr, r_pos=sum(n in POS for n in names) / nr,
            r_neg=sum(n in NEG for n in names) / nr, r_gly=sum(n == "GLY" for n in names) / nr,
            r_pro=sum(n == "PRO" for n in names) / nr, prot_n_res=n_res_total,
            centrality=float(np.linalg.norm(c["center"] - centroid) / rg)))
    order = np.argsort(np.argsort([-r["cav_volume"] for r in rows]))
    for r, o in zip(rows, order):
        r["vol_rank"] = int(o) + 1
    return rows


def geometry() -> dict:
    """Parameters the features depend on; stored next to every ranker model and checked at load time."""
    return dict(pk.geometry(), DETECT_MIN_BURIED=detect.DETECT_MIN_BURIED, NMS_RADIUS=detect.NMS_RADIUS,
                MIN_POINTS=detect.MIN_POINTS, MAX_SITES=detect.MAX_SITES, FILL_MIN_BURIED=detect.FILL_MIN_BURIED, CAV_R=CAV_R, ENV_R=ENV_R)


def check_geometry(path) -> None:
    gj = Path(str(path) + ".geometry.json")
    if not gj.exists():
        warnings.warn(f"{gj.name} missing: cannot verify that geometry matches training", stacklevel=3)
        return
    want, have = json.loads(gj.read_text()), geometry()
    if want != have:
        raise RuntimeError(f"ranker was trained with geometry {want}, current is {have}; retrain or restore them")


def load_ranker(path):
    """LightGBM booster plus its feature list (`<path>.features.json` if present, else FEATURES)."""
    try:
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(path))
    except Exception as ex:  # noqa: BLE001
        raise RuntimeError(f"cannot load ranker model {path}: {ex}") from ex
    fj = Path(str(path) + ".features.json")
    feats = json.loads(fj.read_text()) if fj.exists() else FEATURES
    return booster, feats


def rank_sites(model_path, cands: list[dict], st: dict, extra: list[dict] | None = None) -> list[dict]:
    """Score native candidates with the learned ranker; returns feature rows best-first with `ranker_score`.

    `extra`: optional per-candidate dicts with network features (`NET_FEATURES`), same order as `cands`.
    """
    booster, feats = load_ranker(model_path)
    check_geometry(model_path)
    rows = featurize(cands, st)
    if extra:
        for r, e in zip(rows, extra):
            r.update(e)
    X = np.array([[r.get(f, 0.0) for f in feats] for r in rows], float)
    if len(X):
        for r, v in zip(rows, booster.predict(X)):
            r["ranker_score"] = float(v)
    return sorted(rows, key=lambda r: -r.get("ranker_score", 0.0))
