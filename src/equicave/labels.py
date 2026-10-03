"""Label builders for the three outputs: site hits, pocket property classes, and the hotspot field.

**Hotspots are interactions, not geometry.** A ligand atom near a grid point is not evidence that the point matters:
what matters is that an atom of that chemical class sitting there actually *makes the corresponding interaction* with
the receptor, the way the atoms of a real drug do. So `point_labels(..., require_interaction=True)` (the default)
keeps a ligand atom as a hotspot witness only when `interaction_classes` confirms the matching contact:

    hydrophobic_c  a protein hydrophobic carbon within HYDROPHOBIC_D (4.5 A)
    aromatic       a protein aromatic ring atom within AROMATIC_D (5.5 A), or a protein cation within CATION_PI_D
    hbd            a protein acceptor within HBOND_D (3.5 A)
    hba            a protein donor within HBOND_D
    cation         a protein anion within SALT_D (4.0 A)
    anion          a protein cation within SALT_D
    halogen        a protein acceptor (O or S) within HALOGEN_D (3.8 A)

Ligands can also be restricted to drug-like components (`druglike_ligand`), so the field is trained on the chemistry
medicinal chemists actually exploit rather than on crystallisation additives or cofactor scaffolds. Both targets are
produced, `y_hot` (interaction-validated) and `y_hot_proximity` (the plain geometric variant), so the choice can be
ablated instead of assumed.

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
HBOND_D = 3.5         # A: heavy-atom donor-acceptor distance (no hydrogens in crystal structures)
HYDROPHOBIC_D = 4.5   # A: carbon-carbon contact
AROMATIC_D = 5.5      # A: ring-atom to ring-atom, a proxy for stacking without ring perception on the protein side
CATION_PI_D = 5.0     # A
SALT_D = 4.0          # A
HALOGEN_D = 3.8       # A

# receptor atom typing by residue and atom name: the partner sets every interaction is checked against
BB_DONOR = {"N"}
BB_ACCEPTOR = {"O", "OXT"}
SC_DONOR = {"SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"}, "CYS": {"SG"}, "ASN": {"ND2"}, "GLN": {"NE2"},
            "LYS": {"NZ"}, "ARG": {"NE", "NH1", "NH2"}, "HIS": {"ND1", "NE2"}, "TRP": {"NE1"}}
SC_ACCEPTOR = {"ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}, "ASN": {"OD1"}, "GLN": {"OE1"}, "SER": {"OG"},
               "THR": {"OG1"}, "TYR": {"OH"}, "HIS": {"ND1", "NE2"}, "MET": {"SD"}, "CYS": {"SG"}}
SC_CATION = {"LYS": {"NZ"}, "ARG": {"NE", "NH1", "NH2", "CZ"}, "HIS": {"ND1", "NE2", "CE1"}}
SC_ANION = {"ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}}
SC_AROMATIC = {"PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"}, "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
               "TRP": {"CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2", "CG", "CD1"}, "HIS": {"CG", "ND1", "CD2", "CE1", "NE2"}}
HYDROPHOBIC_RES = set("ALA VAL LEU ILE MET PHE TRP PRO CYS TYR".split())
PROPERTY_CLASSES = ccd.LIGAND_CLASSES + ["size_small", "size_large", "buried_deep", "buried_shallow", "polar", "apolar",
                                         "aromatic_ligand", "charged_ligand"]
HOTSPOT_CLASSES = ccd.ATOM_CLASSES


def protein_atom_types(st: dict) -> dict[str, np.ndarray]:
    """Boolean masks over the receptor heavy atoms: donor, acceptor, cation, anion, aromatic, hydrophobic carbon.

    Typing is by residue and atom name (no protonation state inference), which is what a crystal structure supports.
    HIS nitrogens count both as donor and acceptor, the usual conservative choice.
    """
    name, res, el = st["atom"], st["resname"], st["element"]
    def in_map(m):
        return np.array([a in m.get(str(r), ()) for a, r in zip(name, res)])
    donor = np.isin(name, list(BB_DONOR)) | in_map(SC_DONOR)
    acceptor = np.isin(name, list(BB_ACCEPTOR)) | in_map(SC_ACCEPTOR)
    cation, anion, aromatic = in_map(SC_CATION), in_map(SC_ANION), in_map(SC_AROMATIC)
    hydrophobic = (el == "C") & ~st["backbone"] & np.isin(res, list(HYDROPHOBIC_RES))
    hydrophobic |= (name == "CB") & (el == "C")
    return dict(donor=donor, acceptor=acceptor, cation=cation, anion=anion, aromatic=aromatic, hydrophobic=hydrophobic)


def interaction_classes(lig: dict, lig_cls: np.ndarray, st: dict, types: dict | None = None) -> np.ndarray:
    """[n_lig_atoms, 7]: which of the atom's chemical classes are *confirmed by a contact* with the receptor.

    The output is `lig_cls` masked by the interaction test of each class, so an aromatic ligand atom that stacks
    against nothing stops being an aromatic hotspot witness, while the same atom in a real drug pose keeps the label.
    """
    types = types or protein_atom_types(st)
    xyz = st["xyz"]; L = lig["xyz"]
    out = np.zeros_like(lig_cls, bool)
    def near(mask, d):
        if mask.sum() == 0:
            return np.zeros(len(L), bool)
        return cKDTree(xyz[mask]).query(L, distance_upper_bound=d)[0] <= d
    tests = [near(types["hydrophobic"], HYDROPHOBIC_D),                                   # hydrophobic_c
             near(types["aromatic"], AROMATIC_D) | near(types["cation"], CATION_PI_D),    # aromatic
             near(types["acceptor"], HBOND_D),                                            # hbd
             near(types["donor"], HBOND_D),                                               # hba
             near(types["anion"], SALT_D),                                                # cation
             near(types["cation"], SALT_D),                                               # anion
             near(types["acceptor"], HALOGEN_D)]                                          # halogen
    for j, t in enumerate(tests[:lig_cls.shape[1]]):
        out[:, j] = lig_cls[:, j] & t
    return out


DRUGLIKE_MIN_HEAVY, DRUGLIKE_MAX_HEAVY = 12, 60
NON_DRUGLIKE_CLASSES = ("nucleotide", "heme", "carbohydrate", "lipid", "metal")


def druglike_ligand(lig: dict, entry: dict | None, min_heavy: int = DRUGLIKE_MIN_HEAVY,
                    max_heavy: int = DRUGLIKE_MAX_HEAVY) -> bool:
    """Is this ligand copy the kind of molecule a drug programme would start from?

    Size window, at least one ring, and none of the cofactor-like CCD classes (nucleotide, heme, carbohydrate, lipid,
    metal). Peptide ligands are handled separately by the peptide pipeline. Without a CCD entry only size is checked.
    """
    n = len(lig["xyz"])
    if not (min_heavy <= n <= max_heavy):
        return False
    if entry is None:
        return True
    cls = dict(zip(ccd.LIGAND_CLASSES, ccd.ligand_classes(entry)))
    if any(cls.get(k) for k in NON_DRUGLIKE_CLASSES):
        return False
    A = entry["atoms"]
    heavy = {k: v for k, v in A.items() if v["element"] not in ("H", "D")}
    bonds = sum(len([n_ for n_, _ in v["nbrs"] if n_ in heavy]) for v in heavy.values()) // 2
    return bonds >= len(heavy)            # at least one ring (cyclomatic number >= 1)


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
                 hot_r: float = HOT_R, st: dict | None = None, require_interaction: bool = True):
    """Per grid point: occupancy, interaction-validated hotspot classes, and the plain proximity variant.

    Returns (occ [n], hot [n, 7], hot_proximity [n, 7]). `hot` counts a class only where the witnessing ligand atom
    makes the matching interaction with the receptor (`interaction_classes`), which needs `st`; without `st` or with
    `require_interaction=False` the two outputs are identical. With no ligands every label is negative, which is what
    an apo structure should contribute.
    """
    n_cls = len(ccd.ATOM_CLASSES)
    if not ligs:
        z = np.zeros((len(points), n_cls), bool)
        return np.zeros(len(points), bool), z, z.copy()
    atoms = np.vstack([l["xyz"] for l in ligs])
    cls = np.vstack([ligand_atom_classes(l, entries.get(l["comp"])) for l in ligs])
    if require_interaction and st is not None:
        types = protein_atom_types(st)
        inter = np.vstack([interaction_classes(l, ligand_atom_classes(l, entries.get(l["comp"])), st, types) for l in ligs])
    else:
        inter = cls
    tree = cKDTree(atoms)
    occ = tree.query(points)[0] <= occ_r
    hot = np.zeros((len(points), n_cls), bool); prox = np.zeros((len(points), n_cls), bool)
    for i, nb in enumerate(tree.query_ball_point(points, hot_r)):
        if nb:
            hot[i] = inter[nb].any(0); prox[i] = cls[nb].any(0)
    return occ, hot, prox


def site_centres(ligs: list[dict], sites: list[list[int]]) -> np.ndarray:
    return np.array([np.vstack([ligs[i]["xyz"] for i in s]).mean(0) for s in sites]).reshape(-1, 3)
