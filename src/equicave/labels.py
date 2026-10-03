"""Label builders for the three outputs: site hits, pocket property classes, and the hotspot field.

Sites: a ligand site is a group of ligand copies whose centroids lie within 8 A (one site may hold several ligands).
Residue segmentation: a residue is positive when a heavy atom of it is within SEG_R of a ligand heavy atom.
Probe (grid point) occupancy: a point is positive when it is within OCC_R of a ligand heavy atom.
Property classes per site (multi-label): CCD ligand classes of all ligands in the site + size / buriedness /
polarity / aromatic / charged classes computed from the ligand atoms. Hotspot classes per grid point: a point is
positive for class k when a ligand heavy atom of class k lies within HOT_R.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import ccd, pockets as pk

SEG_R = 4.0
OCC_R = 2.0
HOT_R = 1.5
SITE_LINK = 8.0
PROPERTY_CLASSES = ccd.LIGAND_CLASSES + ["size_small", "size_large", "buried_deep", "buried_shallow", "polar", "apolar",
                                         "aromatic_ligand", "charged_ligand"]
HOTSPOT_CLASSES = ccd.ATOM_CLASSES


def group_sites(ligs: list[dict], link: float = SITE_LINK) -> list[list[int]]:
    """Indices of ligand copies per site (single linkage on centroids within `link` A)."""
    cents = [l["xyz"].mean(0) for l in ligs]
    sites: list[list[int]] = []
    for i, c in enumerate(cents):
        for s in sites:
            if any(np.linalg.norm(c - cents[j]) < link for j in s):
                s.append(i); break
        else:
            sites.append([i])
    return sites


def ligand_atom_classes(lig: dict, entry: dict | None) -> np.ndarray:
    """[n_atoms, len(ATOM_CLASSES)] booleans; falls back to element rules when the CCD entry is missing."""
    n = len(lig["xyz"]); out = np.zeros((n, len(ccd.ATOM_CLASSES)), bool)
    if entry is not None:
        ac = ccd.atom_classes(entry)
        for i, name in enumerate(lig.get("atom", [])):
            if name in ac:
                out[i] = ac[name]
        if out.any():
            return out
    el = lig["element"]
    out[:, 0] = el == "C"; out[:, 3] = el == "O"; out[:, 2] = np.isin(el, ["N", "O"]); out[:, 6] = np.isin(el, list(ccd.HALOGENS))
    return out


def site_properties(ligs: list[dict], entries: dict[str, dict | None], prot_xyz: np.ndarray) -> np.ndarray:
    """Multi-label property vector of one site (union over its ligands)."""
    v = np.zeros(len(PROPERTY_CLASSES), bool)
    atoms = np.vstack([l["xyz"] for l in ligs])
    cls = np.vstack([ligand_atom_classes(l, entries.get(l["comp"])) for l in ligs])
    for l in ligs:
        e = entries.get(l["comp"])
        if e is not None:
            v[:len(ccd.LIGAND_CLASSES)] |= ccd.ligand_classes(e)
    n = len(atoms)
    v[PROPERTY_CLASSES.index("size_small")] = n <= 15
    v[PROPERTY_CLASSES.index("size_large")] = n >= 35
    bur = pk._buried(atoms, cKDTree(prot_xyz)).mean()
    v[PROPERTY_CLASSES.index("buried_deep")] = bur >= 18
    v[PROPERTY_CLASSES.index("buried_shallow")] = bur < 12
    el = np.concatenate([l["element"] for l in ligs])
    polar = np.isin(el, ["N", "O"]).mean()
    v[PROPERTY_CLASSES.index("polar")] = polar >= 0.35
    v[PROPERTY_CLASSES.index("apolar")] = polar <= 0.15
    v[PROPERTY_CLASSES.index("aromatic_ligand")] = cls[:, 1].sum() >= 5
    v[PROPERTY_CLASSES.index("charged_ligand")] = cls[:, 4].any() or cls[:, 5].any()
    return v


def residue_labels(st: dict, lig_xyz: np.ndarray, r: float = SEG_R) -> np.ndarray:
    """Per residue (order of `structure.residue_table`): 1 if any heavy atom within r of a ligand heavy atom."""
    d = cKDTree(lig_xyz).query(st["xyz"])[0]
    hit = d <= r
    order, lab = [], {}
    for rid, h in zip(st["resid"], hit):
        if rid not in lab:
            order.append(rid); lab[rid] = False
        lab[rid] |= bool(h)
    return np.array([lab[r] for r in order], bool)


def point_labels(points: np.ndarray, ligs: list[dict], entries: dict[str, dict | None], occ_r: float = OCC_R,
                 hot_r: float = HOT_R) -> tuple[np.ndarray, np.ndarray]:
    """Per grid point: occupancy (any ligand atom within occ_r) and hotspot classes (atom of class k within hot_r)."""
    atoms = np.vstack([l["xyz"] for l in ligs])
    cls = np.vstack([ligand_atom_classes(l, entries.get(l["comp"])) for l in ligs])
    tree = cKDTree(atoms)
    occ = tree.query(points)[0] <= occ_r
    hot = np.zeros((len(points), cls.shape[1]), bool)
    for i, nb in enumerate(tree.query_ball_point(points, hot_r)):
        if nb:
            hot[i] = cls[nb].any(0)
    return occ, hot


def site_centres(ligs: list[dict], sites: list[list[int]]) -> np.ndarray:
    return np.array([np.vstack([ligs[i]["xyz"] for i in s]).mean(0) for s in sites]).reshape(-1, 3)
