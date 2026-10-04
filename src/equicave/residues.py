"""Amino-acid chemistry as node features: what each residue can do, not just which one it is.

A 21-way one-hot makes the network learn from scratch that Asp and Glu are both anionic carboxylates, that Phe, Tyr
and Trp all stack, and that Gly is a hinge. Those facts are known, so they are given directly. Two kinds of feature
come out of this module:

**Scalars** (`PROPERTIES`, per residue type): Kyte-Doolittle hydropathy, side-chain volume, formal charge at pH 7,
counts of side-chain hydrogen-bond donors and acceptors, flags for aromatic / aliphatic / polar / small / hinge
(Gly) / rigid (Pro) / metal-coordinating / reactive (Cys, Lys, Ser for covalent chemistry), and the number of
side-chain rotatable bonds as a flexibility proxy.

**Vectors** (`side_chain_vectors`, per residue instance, degree 1 and therefore rotating with the structure):
Cα→Cβ, Cα→side-chain centroid, and Cα→functional-group centroid, where the functional group is the set of the
residue's polar, charged or aromatic side-chain atoms. The third vector is the one that matters for pockets: it says
which way the residue's *chemistry* faces, so the network can distinguish an Asp whose carboxylate points into the
cavity from one that points at the solvent — a distinction no scalar feature and no distance can express.

Values are standard tabulated ones [из памяти: hydropathy from Kyte & Doolittle 1982, volumes from Zamyatnin 1972;
recheck against the primary sources before publication]. They are inputs, never labels.
"""
from __future__ import annotations

import numpy as np

# name: (hydropathy, side-chain volume A^3, charge, sc donors, sc acceptors, aromatic, aliphatic, polar,
#        small, hinge, rigid, metal, reactive, rotatable bonds)
_T = {
    "ALA": (1.8, 88.6, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0),
    "ARG": (-4.5, 173.4, 1, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 4),
    "ASN": (-3.5, 114.1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 2),
    "ASP": (-3.5, 111.1, -1, 0, 2, 0, 0, 1, 1, 0, 0, 1, 0, 2),
    "CYS": (2.5, 108.5, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1),
    "GLN": (-3.5, 143.8, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 3),
    "GLU": (-3.5, 138.4, -1, 0, 2, 0, 0, 1, 0, 0, 0, 1, 0, 3),
    "GLY": (-0.4, 60.1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0),
    "HIS": (-3.2, 153.2, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 2),
    "ILE": (4.5, 166.7, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2),
    "LEU": (3.8, 166.7, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2),
    "LYS": (-3.9, 168.6, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 4),
    "MET": (1.9, 162.9, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 3),
    "PHE": (2.8, 189.9, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2),
    "PRO": (-1.6, 112.7, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0),
    "SER": (-0.8, 89.0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1),
    "THR": (-0.7, 116.1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1),
    "TRP": (-0.9, 227.8, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2),
    "TYR": (-1.3, 193.6, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 2),
    "VAL": (4.2, 140.0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1),
    "MSE": (1.9, 162.9, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 3),
}
PROPERTIES = ["hydropathy", "volume", "charge", "sc_donors", "sc_acceptors", "aromatic", "aliphatic", "polar",
              "small", "hinge", "rigid", "metal_binding", "reactive", "rotatable"]
_SCALE = np.array([4.5, 227.8, 1.0, 3.0, 2.0, 1, 1, 1, 1, 1, 1, 1, 1, 4.0], float)   # to roughly [-1, 1]
_UNKNOWN = np.zeros(len(PROPERTIES))

# side-chain atoms that carry the residue's chemistry: polar, charged or aromatic
FUNCTIONAL_ATOMS = {
    "ARG": {"NE", "NH1", "NH2", "CZ"}, "LYS": {"NZ"}, "ASP": {"OD1", "OD2", "CG"}, "GLU": {"OE1", "OE2", "CD"},
    "ASN": {"OD1", "ND2"}, "GLN": {"OE1", "NE2"}, "SER": {"OG"}, "THR": {"OG1"}, "CYS": {"SG"}, "MET": {"SD"},
    "HIS": {"ND1", "CD2", "CE1", "NE2", "CG"}, "TYR": {"OH", "CZ", "CE1", "CE2"},
    "TRP": {"NE1", "CD2", "CE2", "CZ2", "CZ3", "CH2"}, "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
}
BACKBONE = ("N", "CA", "C", "O")


def properties(resnames) -> np.ndarray:
    """[n_residues, len(PROPERTIES)] scaled property vector; unknown residues get zeros."""
    return np.array([_T.get(str(r), None) or _UNKNOWN for r in resnames], float) / _SCALE


def side_chain_vectors(st: dict, rt: dict) -> np.ndarray:
    """[n_residues, 3, 3] unit vectors: Cα→Cβ, Cα→side-chain centroid, Cα→functional-group centroid.

    Each is a degree-1 quantity, so it rotates with the structure and can be consumed by the vector channels of the
    network directly. Missing groups (Gly has no side chain, an unresolved side chain has no atoms) give zeros, which
    the network reads as "no direction".
    """
    idx = {r: i for i, r in enumerate(rt["resid"])}
    n = len(rt["resid"])
    sc_sum = np.zeros((n, 3)); sc_cnt = np.zeros(n)
    fn_sum = np.zeros((n, 3)); fn_cnt = np.zeros(n)
    cb = np.zeros((n, 3)); has_cb = np.zeros(n, bool)
    for i, (rid, name, rn) in enumerate(zip(st["resid"], st["atom"], st["resname"])):
        k = idx.get(rid)
        if k is None:
            continue
        if name == "CB":
            cb[k] = st["xyz"][i]; has_cb[k] = True
        if name in BACKBONE:
            continue
        sc_sum[k] += st["xyz"][i]; sc_cnt[k] += 1
        if name in FUNCTIONAL_ATOMS.get(str(rn), ()):
            fn_sum[k] += st["xyz"][i]; fn_cnt[k] += 1
    out = np.zeros((n, 3, 3))
    ca = rt["ca"]
    out[has_cb, 0] = cb[has_cb] - ca[has_cb]
    m = sc_cnt > 0
    out[m, 1] = sc_sum[m] / sc_cnt[m, None] - ca[m]
    m = fn_cnt > 0
    out[m, 2] = fn_sum[m] / fn_cnt[m, None] - ca[m]
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    return np.where(norms > 1e-6, out / np.maximum(norms, 1e-6), 0.0)


def pocket_facing(st: dict, rt: dict, probe_xyz: np.ndarray, radius: float = 8.0) -> np.ndarray:
    """[n_residues, 2]: does the residue's side chain, and its functional group, point towards the cavity?

    cos of the angle between the Cα→side-chain (and Cα→functional-group) vector and the direction from Cα to the
    nearest cavity probe. +1 means the chemistry faces the pocket, -1 means it faces away. This is an invariant built
    from two degree-1 inputs, so it is safe to feed as a scalar.
    """
    if len(probe_xyz) == 0:
        return np.zeros((len(rt["resid"]), 2))
    from scipy.spatial import cKDTree
    v = side_chain_vectors(st, rt)
    nearest = probe_xyz[cKDTree(probe_xyz).query(rt["ca"])[1]]
    to_pocket = nearest - rt["ca"]
    to_pocket /= np.maximum(np.linalg.norm(to_pocket, axis=1, keepdims=True), 1e-6)
    return np.stack([(v[:, 1] * to_pocket).sum(1), (v[:, 2] * to_pocket).sum(1)], 1)
