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

# Kyte-Doolittle hydropathy, used as a per-residue scalar around the candidate
KD = {"ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5, "GLN": -3.5, "GLU": -3.5, "GLY": -0.4,
      "HIS": -3.2, "ILE": 4.5, "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8, "PRO": -1.6, "SER": -0.8,
      "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2, "MSE": 1.9}
SHELLS = (5.0, 8.0, 12.0)     # A: nested spheres around the candidate centre for the interaction-type counts

NATIVE = ["nat_score", "nat_rank", "nat_rel", "nat_npts", "nat_mean_bur", "nat_max_bur", "nat_cavity_npts", "nat_tier"]
GEOMETRY = ["cav_volume", "ax1", "ax2", "ax3", "buried_mean", "vol_rank", "centrality", "depth",
            "bur_std", "bur_p10", "bur_p90", "frac_deep", "frac_mouth", "width_mean", "width_max", "elongation",
            "flatness", "pts_per_volume"]
CHEMISTRY = ["n_atoms", "f_C", "f_N", "f_O", "f_S", "f_backbone", "n_res", "r_hydrophobic", "r_aromatic", "r_polar",
             "r_pos", "r_neg", "r_gly", "r_pro"]
# receptor interaction partners in nested shells: what a ligand atom in this pocket could actually bind to
SHELL = [f"{t}_{int(r)}" for r in SHELLS for t in ("donor", "acceptor", "cation", "anion", "aromatic", "hydrophobic")] + \
        [f"f_{t}_{int(r)}" for r in SHELLS for t in ("donor", "acceptor", "cation", "anion", "aromatic", "hydrophobic")] + \
        ["kd_mean", "kd_sum", "bfac_mean", "bfac_std", "n_res_5", "n_res_12", "hb_balance", "charge_balance",
         "polar_apolar_ratio", "donor_acceptor_per_volume"]
# interaction potential: for each cavity point, which interaction a ligand atom placed there could actually make
POT_CLASSES = ("hbd", "hba", "hydrophobic", "aromatic", "cation", "anion", "halogen")
POTENTIAL = ([f"pot_f_{c}" for c in POT_CLASSES]            # fraction of points with a partner at the strict distance
             + [f"pot_d_{c}" for c in POT_CLASSES]           # mean distance to the nearest partner of that type
             + [f"pot_c_{c}" for c in POT_CLASSES]           # mean number of partners within 6 A
             + ["pot_classes_mean", "pot_f_multi2", "pot_f_multi3", "pot_f_polar_apolar", "pot_f_hbd_hba",
                "pot_best_point_classes", "pot_volume_multi3", "pot_d_min_overall", "pot_c_total"])
CONTEXT = ["prot_n_res", "n_cands"]
FEATURES = NATIVE + GEOMETRY + CHEMISTRY + SHELL + POTENTIAL + CONTEXT
# Network features, all rotation-invariant scalars. The three originals are kept first for compatibility; the rest
# preserve what the three threw away: each hotspot class separately, at two radii, with mean and max.
NET_FEATURES = (["net_seg", "net_center_conf", "net_hot_mean", "net_n_centers", "net_center_dist"]
                + [f"net_{k}_{r}" for r in (4, 8) for k in ("occ_mean", "occ_max", "conf_mean", "conf_max",
                                                            "offset_mean", "n_probes")]
                + [f"net_hot{j}_{s}_{r}" for r in (4, 8) for j in range(7) for s in ("mean", "max")])
PEPTIDE = ["pep_tier", "pep_length", "pep_width", "pep_anisotropy", "pep_flatness", "pep_bb_n", "pep_bb_o", "pep_bb_ca",
           "pep_sc_c", "pep_sc_polar", "pep_bb_total", "pep_bb_ratio", "pep_bb_per_point", "pep_overlap_dist"]
GROUPS = dict(native=NATIVE, geometry=GEOMETRY, chemistry=CHEMISTRY, shell=SHELL, potential=POTENTIAL,
              context=CONTEXT, network=NET_FEATURES, peptide=PEPTIDE)


POT_SPEC = (("hbd", "acceptor"), ("hba", "donor"), ("hydrophobic", "hydrophobic"), ("aromatic", "aromatic"),
            ("cation", "anion"), ("anion", "cation"), ("halogen", "acceptor"))


def point_potential(points: np.ndarray, trees: dict, count_radius: float = 6.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per point: how many partners of each interaction type are near, how near the nearest is, and a strict flag.

    Returns (count [P, 7], distance to the nearest [P, 7], strict availability [P, 7]).

    A plain "is a partner within the cutoff" flag turned out to be **useless**: measured on a real structure, every
    cavity probe had all seven interaction types available, because a 4.5-5.5 A cutoff inside a protein is always
    satisfied and the flag is constant. The information is in *how many* partners and *how close* the nearest one is,
    so those are what the model gets. The strict flag uses a tight threshold (hydrogen bonds 3.2 A, salt bridges
    3.5 A, halogen 3.5 A, stacking 4.5 A, hydrophobic contact 4.0 A) where it does vary.
    """
    from . import labels as LB
    strict = dict(hbd=3.2, hba=3.2, hydrophobic=4.0, aromatic=4.5, cation=3.5, anion=3.5, halogen=3.5)
    pts = np.atleast_2d(np.asarray(points, float))
    n, k = len(pts), len(POT_SPEC)
    count = np.zeros((n, k), np.float32)
    dist = np.full((n, k), count_radius * 2, np.float32)
    avail = np.zeros((n, k), np.float32)
    for j, (name, partner) in enumerate(POT_SPEC):
        tr = trees.get(partner)
        if tr is None or n == 0:
            continue
        q = tr.query(pts)[0]
        dist[:, j] = np.minimum(q, count_radius * 2)
        avail[:, j] = (q <= strict[name]).astype(np.float32)
        count[:, j] = [len(x) for x in tr.query_ball_point(pts, count_radius)]
    return count, dist, avail


def interaction_potential(points: np.ndarray, trees: dict, max_points: int = 200, seed: int = 0) -> dict:
    """What a ligand atom could bind to at each cavity point, summarised over the cavity.

    The same geometric criteria as the hotspot labels (`labels.interaction_classes`), but asked of the *empty* point
    instead of a ligand atom: is there a receptor acceptor within 3.5 A (so a donor placed here would be satisfied),
    a donor within 3.5 A, a hydrophobic carbon within 4.5 A, an aromatic ring atom within 5.5 A, an anion or a cation
    within 4.0 A, an acceptor within 3.8 A for a halogen. A physics-free prior that separates a plausible binding site
    from empty space, and the geometric counterpart of the learned hotspot field.
    """
    from . import labels as LB
    pts = np.atleast_2d(np.asarray(points, float))
    if len(pts) == 0:
        return {f: 0.0 for f in POTENTIAL}
    n_total = len(pts)
    if n_total > max_points:
        pts = pts[np.random.default_rng(seed).permutation(n_total)[:max_points]]
    C, D, A = point_potential(pts, trees)
    avail = {name: A[:, j].astype(bool) for j, (name, _) in enumerate(POT_SPEC)}
    n_cls = A.sum(1)
    out = {f"pot_f_{n}": float(A[:, j].mean()) for j, (n, _) in enumerate(POT_SPEC)}
    out.update({f"pot_d_{n}": float(D[:, j].mean()) for j, (n, _) in enumerate(POT_SPEC)})
    out.update({f"pot_c_{n}": float(C[:, j].mean()) for j, (n, _) in enumerate(POT_SPEC)})
    out["pot_d_min_overall"] = float(D.min(1).mean())
    out["pot_c_total"] = float(C.sum(1).mean())
    out["pot_classes_mean"] = float(n_cls.mean())
    out["pot_f_multi2"] = float((n_cls >= 2).mean())
    out["pot_f_multi3"] = float((n_cls >= 3).mean())
    out["pot_f_polar_apolar"] = float(((avail["hbd"] | avail["hba"]) & avail["hydrophobic"]).mean())
    out["pot_f_hbd_hba"] = float((avail["hbd"] & avail["hba"]).mean())
    out["pot_best_point_classes"] = float(n_cls.max())
    out["pot_volume_multi3"] = float((n_cls >= 3).mean() * n_total)
    return out


def featurize(cands: list[dict], st: dict, radius: float = CAV_R) -> list[dict]:
    """One feature dict per candidate from `detect.detect_sites`; `st` is `structure.read_pdb(...)`.

    Four groups of numbers, none of them family specific: the candidate as the generator produced it (`nat_*`), the
    shape of its cavity, the atom and residue composition of its 8 A environment, and the receptor interaction
    partners it offers in nested shells (`SHELL`) — how many donors, acceptors, cations, anions, aromatic ring atoms
    and hydrophobic carbons a ligand atom placed here could actually bind to. The last group is what tells a real
    binding site from an equally deep but chemically featureless hole.
    """
    from . import labels as LB
    xyz = st["xyz"]; tp = cKDTree(xyz)
    types = LB.protein_atom_types(st)
    trees = {k: (cKDTree(xyz[m]) if m.sum() else None) for k, m in types.items()}
    centroid = xyz.mean(0); rg = float(np.sqrt(((xyz - centroid) ** 2).sum(1).mean())) or 1.0
    n_res_total = len(set(st["resid"]))
    table = detect.candidate_table(cands)
    rows = []
    for c, t in zip(cands, table):
        _, _, cav = pk.cavity_box(c["center"], xyz, radius)
        sub = cav[np.random.default_rng(0).permutation(len(cav))[:300]]
        bur_sub = pk._buried(sub, tp) if len(sub) else np.zeros(0)
        near = tp.query_ball_point(c["center"], ENV_R)
        el, rn, rid, bb = st["element"][near], st["resname"][near], st["resid"][near], st["backbone"][near]
        names = list({r: n for r, n in zip(rid, rn)}.values()); nr = max(1, len(names))
        ax = pk.shape_axes(cav) + [0.0, 0.0, 0.0]
        fr = lambda e: float((el == e).mean()) if len(el) else 0.0
        # shape of the candidate's own points
        cpts = np.asarray(c.get("points", cav)); cbur = np.asarray(c.get("buried", bur_sub), float)
        width = tp.query(cpts)[0] if len(cpts) else np.zeros(1)
        shell = {}
        for r in SHELLS:
            n_all = max(1, len(tp.query_ball_point(c["center"], r)))
            for k, tr in trees.items():
                cnt = len(tr.query_ball_point(c["center"], r)) if tr is not None else 0
                shell[f"{k}_{int(r)}"] = float(cnt)
                shell[f"f_{k}_{int(r)}"] = cnt / n_all
        res5 = {r for r in st["resid"][tp.query_ball_point(c["center"], 5.0)]}
        res12 = {r for r in st["resid"][tp.query_ball_point(c["center"], 12.0)]}
        kd = [KD.get(str(n), 0.0) for n in names]
        bf = st["bfactor"][near]
        d8, a8 = shell["donor_8"], shell["acceptor_8"]
        pos8, neg8 = shell["cation_8"], shell["anion_8"]
        vol = max(1.0, float(len(cav)))
        pot = interaction_potential(cpts if len(cpts) else cav, trees)
        rows.append(dict(
            center=c["center"], **{k: t[k] for k in NATIVE}, n_cands=t["n_cands"], **shell, **pot,
            cav_volume=float(len(cav)), ax1=ax[0], ax2=ax[1], ax3=ax[2],
            buried_mean=float(bur_sub.mean()) if len(bur_sub) else 0.0,
            bur_std=float(cbur.std()) if len(cbur) else 0.0,
            bur_p10=float(np.percentile(cbur, 10)) if len(cbur) else 0.0,
            bur_p90=float(np.percentile(cbur, 90)) if len(cbur) else 0.0,
            frac_deep=float((cbur >= detect.DETECT_MIN_BURIED).mean()) if len(cbur) else 0.0,
            frac_mouth=float((cbur < detect.FILL_MIN_BURIED + 2).mean()) if len(cbur) else 0.0,
            width_mean=float(width.mean()), width_max=float(width.max()),
            elongation=float(ax[0] / max(ax[2], 1.0)), flatness=float(ax[1] / max(ax[2], 1.0)),
            pts_per_volume=float(len(cpts)) / vol,
            kd_mean=float(np.mean(kd)) if kd else 0.0, kd_sum=float(np.sum(kd)),
            bfac_mean=float(bf.mean()) if len(bf) else 0.0, bfac_std=float(bf.std()) if len(bf) else 0.0,
            n_res_5=float(len(res5)), n_res_12=float(len(res12)),
            hb_balance=(d8 - a8) / max(1.0, d8 + a8), charge_balance=(pos8 - neg8) / max(1.0, pos8 + neg8),
            polar_apolar_ratio=(d8 + a8) / max(1.0, shell["hydrophobic_8"]),
            donor_acceptor_per_volume=(d8 + a8) / vol,
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


def check_features(model_features: list[str], produced: dict, model_path="") -> None:
    """Fail loudly when a trained model asks for a feature this checkout no longer produces.

    Feature definitions change (the interaction-potential group was redefined once because it was saturated), and a
    model trained before such a change asks for columns that no longer exist. Filling them with zeros gives a
    prediction that looks valid and is not: the model is reading a constant where it learned a signal. Network and
    language-model columns are exempt, since those are supplied by the caller only when a network or an embedding is
    available, and their absence is reported separately.
    """
    optional = tuple(NET_FEATURES) + ("esm_",)
    missing = [f for f in model_features
               if f not in produced and not f.startswith(optional) and not f.endswith(("_z", "_m"))
               and f not in NET_FEATURES]
    if missing:
        raise RuntimeError(
            f"model {Path(model_path).name or model_path} needs {len(missing)} features this checkout does not "
            f"produce: {missing[:8]}{'...' if len(missing) > 8 else ''}. The feature definitions changed since it was "
            f"trained; retrain it (scripts/train/train_ranker.py) or check out the commit recorded in its "
            f"geometry file.")


def rank_sites(model_path, cands: list[dict], st: dict, extra: list[dict] | None = None) -> list[dict]:
    """Score candidates with the learned ranker; returns feature rows best-first with `ranker_score`.

    `extra`: optional per-candidate dicts with network features (`NET_FEATURES`), same order as `cands`.
    Within-structure z-score features, when the model was trained with them (`<model>.preprocess.json`), are
    recomputed here exactly as in training, so inference and training see the same columns.
    """
    booster, feats = load_ranker(model_path)
    check_geometry(model_path)
    rows = featurize(cands, st)
    if extra:
        for r, e in zip(rows, extra):
            r.update(e)
    if not rows:
        return rows
    check_features(feats, rows[0], model_path)
    pre = Path(str(model_path) + ".preprocess.json")
    if pre.exists():
        cfg = json.loads(pre.read_text())
        if cfg.get("zscore"):
            base = [f for f in cfg["base_features"] if f in rows[0]]
            M = np.array([[r.get(f, 0.0) for f in base] for r in rows], float)
            Z = (M - M.mean(0)) / (M.std(0) + 1e-6)
            for r, z in zip(rows, Z):
                r.update({f"{f}_z": float(v) for f, v in zip(base, z)})
    X = np.array([[r.get(f, 0.0) for f in feats] for r in rows], float)
    for r, v in zip(rows, booster.predict(X)):
        r["ranker_score"] = float(v)
    return sorted(rows, key=lambda r: -r.get("ranker_score", 0.0))
