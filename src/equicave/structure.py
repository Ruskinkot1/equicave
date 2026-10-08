"""Minimal PDB reader: heavy protein atoms, ligand copies, residue tables. No external dependencies beyond numpy.

Only what the pipeline needs. mmCIF is not read here; the build scripts download legacy PDB files and skip entries
that are PDB-format only on the RCSB side.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
    "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}
BACKBONE = ("N", "CA", "C", "O")


def _element(line: str) -> str:
    e = line[76:78].strip().upper()
    if not e:
        name = line[12:16].strip()
        e = name[1].upper() if name[:1].isdigit() and len(name) > 1 else name[:1].upper()
    return e


def read_pdb(path, chains: str | None = None, model: int = 1) -> dict:
    """Protein heavy atoms of one model.

    Returns xyz [N,3], element, resname, resid ('A_123'), chain, atom name, backbone flag, b-factor.
    Metal ions and cofactor metals are *not* here: they are not polymer atoms, and `read_metals` reads them.
    MSE (selenomethionine) is kept as a residue; its HETATM records are read because it is part of the chain.
    """
    xyz, el, rn, rid, ch, an, bb, bf = [], [], [], [], [], [], [], []
    cur_model = 1
    for line in Path(path).read_text(errors="ignore").splitlines():
        tag = line[:6]
        if tag == "MODEL ":
            cur_model = int(line[10:14])
            continue
        if tag == "ENDMDL" and cur_model >= model:
            break
        if cur_model != model:
            continue
        if not (tag.startswith("ATOM") or (tag.startswith("HETATM") and line[17:20].strip() in ("MSE", "SEC"))):
            continue
        if len(line) < 54:          # too short to hold the coordinate columns: a truncated or malformed record
            continue                # (one such line in a downloaded file used to abort the whole structure)
        if chains and line[21] not in chains:
            continue
        alt = line[16]
        if alt not in (" ", "A", "1"):
            continue
        e = _element(line)
        if e in ("H", "D"):
            continue
        name = line[12:16].strip()
        xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        el.append(e); rn.append(line[17:20].strip()); ch.append(line[21])
        rid.append(f"{line[21]}_{line[22:27].strip()}"); an.append(name); bb.append(name in BACKBONE)
        try:
            bf.append(float(line[60:66]))
        except ValueError:
            bf.append(0.0)
    return dict(xyz=np.asarray(xyz, float).reshape(-1, 3), element=np.array(el), resname=np.array(rn),
                resid=np.array(rid), chain=np.array(ch), atom=np.array(an), backbone=np.array(bb, bool),
                bfactor=np.asarray(bf, float))


def residue_table(st: dict) -> dict:
    """One row per residue in file order: resid, resname, chain, CA coordinate (centroid if CA is missing), 1-letter code."""
    order, first = [], {}
    for i, r in enumerate(st["resid"]):
        if r not in first:
            first[r] = len(order); order.append(r)
    n = len(order)
    ca = np.full((n, 3), np.nan)
    cen = np.zeros((n, 3)); cnt = np.zeros(n)
    rn = np.empty(n, object); chain = np.empty(n, object)
    for i, r in enumerate(st["resid"]):
        k = first[r]
        cen[k] += st["xyz"][i]; cnt[k] += 1
        rn[k] = st["resname"][i]; chain[k] = st["chain"][i]
        if st["atom"][i] == "CA":
            ca[k] = st["xyz"][i]
    miss = np.isnan(ca[:, 0])
    ca[miss] = cen[miss] / np.maximum(cnt[miss, None], 1)
    seq = "".join(AA3.get(str(x), "X") for x in rn)
    return dict(resid=np.array(order), resname=rn.astype(str), chain=chain.astype(str), ca=ca, seq=seq)


# ligand records: non-polymer HETATM groups, keyed by (comp, chain, resseq+icode)
WATER = {"HOH", "DOD", "WAT"}


def read_ligands(path, min_heavy: int = 8, exclude: set[str] | None = None, model: int = 1) -> list[dict]:
    """Ligand copies from HETATM records with at least `min_heavy` heavy atoms.

    `exclude`: component ids to skip (waters are always skipped). Returns dicts with comp, chain, xyz, element.
    """
    exclude = (exclude or set()) | WATER | {"MSE", "SEC"}
    copies: dict[tuple, dict] = {}
    cur_model = 1
    for line in Path(path).read_text(errors="ignore").splitlines():
        if line[:6] == "MODEL ":
            cur_model = int(line[10:14]); continue
        if line[:6] == "ENDMDL" and cur_model >= model:
            break
        if cur_model != model or line[:6] != "HETATM":
            continue
        comp = line[17:20].strip()
        if comp in exclude:
            continue
        e = _element(line)
        if e in ("H", "D"):
            continue
        if line[16] not in (" ", "A", "1"):
            continue
        key = (comp, line[21], line[22:27].strip())
        c = copies.setdefault(key, dict(comp=comp, chain=line[21], resseq=line[22:27].strip(), xyz=[], element=[]))
        c["xyz"].append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        c["element"].append(e)
        c.setdefault("atom", []).append(line[12:16].strip())
    out = []
    for c in copies.values():
        if len(c["xyz"]) >= min_heavy:
            c["xyz"] = np.asarray(c["xyz"], float); c["element"] = np.array(c["element"]); c["atom"] = np.array(c["atom"])
            out.append(c)
    return out


def write_pdb(path, xyz, element="C", resname="UNK", record="HETATM", resseq: int | None = None) -> Path:
    """Write points as a PDB file (tests, cavity visualisation). `resseq=None`: one residue per point; else one residue."""
    xyz = np.atleast_2d(np.asarray(xyz, float))
    el = [element] * len(xyz) if isinstance(element, str) else list(element)
    lines = [f"{record:<6}{i + 1:5d}  {el[i]:<3}{resname:>4} A{(i + 1 if resseq is None else resseq):4d}    "
             f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el[i]:>2}"
             for i, ((x, y, z)) in enumerate(xyz)]
    Path(path).write_text("\n".join(lines) + "\nEND\n")
    return Path(path)


# ---- short polymer chains as peptide ligands ----------------------------------------------------------------------
def chain_lengths(path, model: int = 1) -> dict[str, int]:
    """Residues per chain from ATOM records (one model), for telling receptors from peptide chains."""
    seen: dict[str, set] = {}
    cur = 1
    for line in Path(path).read_text(errors="ignore").splitlines():
        if line[:6] == "MODEL ":
            cur = int(line[10:14]); continue
        if line[:6] == "ENDMDL" and cur >= model:
            break
        if cur != model or not (line[:4] == "ATOM" or (line[:6] == "HETATM" and line[17:20].strip() in ("MSE", "SEC"))):
            continue
        seen.setdefault(line[21], set()).add(line[22:27].strip())
    return {c: len(r) for c, r in seen.items()}


def read_peptide_ligands(path, min_res: int = 3, max_res: int = 30, min_heavy: int = 8, model: int = 1,
                         receptor_min_res: int = 50) -> list[dict]:
    """Short polymer chains that act as peptide ligands of a longer receptor chain in the same file.

    A chain qualifies when it has `min_res`..`max_res` residues, at least `min_heavy` heavy atoms, the file also holds a
    chain of at least `receptor_min_res` residues, and the chain lies within 5 A of that receptor. Returned in the same
    record format as `read_ligands` (comp = 'PEP:<chain>', xyz, element, atom) plus `n_res`, `seq` and `chain`.
    """
    import numpy as _np
    lens = chain_lengths(path, model)
    if not any(n >= receptor_min_res for n in lens.values()):
        return []
    short = {c for c, n in lens.items() if min_res <= n <= max_res}
    long_ = {c for c, n in lens.items() if n >= receptor_min_res}
    if not short:
        return []
    st = read_pdb(path, model=model)
    out = []
    rec = st["xyz"][_np.isin(st["chain"], list(long_))]
    if len(rec) == 0:
        return []
    from scipy.spatial import cKDTree as _KD
    tree = _KD(rec)
    for c in sorted(short):
        m = st["chain"] == c
        if m.sum() < min_heavy:
            continue
        xyz = st["xyz"][m]
        if tree.query(xyz)[0].min() > 5.0:
            continue
        sub = {k: st[k][m] for k in ("xyz", "element", "atom", "resname", "resid")}
        order, rn = [], []
        for r, n in zip(sub["resid"], sub["resname"]):
            if r not in order:
                order.append(r); rn.append(n)
        out.append(dict(comp=f"PEP:{c}", chain=c, resseq="", xyz=xyz, element=sub["element"], atom=sub["atom"],
                        n_res=len(order), seq="".join(AA3.get(str(x), "X") for x in rn), is_peptide=True))
    return out


def receptor_chains(path, receptor_min_res: int = 50, model: int = 1) -> str:
    """Chain ids of the chains long enough to be the receptor (used to exclude peptide chains from the protein input)."""
    return "".join(c for c, n in chain_lengths(path, model).items() if n >= receptor_min_res)


# Metal ions and the metal centres of cofactors. `read_pdb` reads the polymer only, so none of these atoms reaches
# anything built on it -- which is the state the literature is in as well: of the eight site predictors surveyed in
# docs/LITERATURE.md, two carry a metal input bit (DeepSite, Kalasanty/DeepSurf), two strip heteroatoms outright and
# none publishes an ablation of it, while ions account for about 40 % of ligand sites in LIGYSIS.
METAL_ELEMENTS = frozenset("ZN MG CA FE MN CU NA K CO NI CD HG MO W V SR BA PT AU AG LI".split())


def read_metals(path, model: int = 1, min_occupancy: float = 0.0) -> dict:
    """Metal atoms of one model, from any HETATM group: xyz, element, occupancy, comp.

    Read separately from the polymer because a metal is not a protein atom -- it has no residue, no atom name worth
    typing and no backbone -- and because its occupancy is the only cheap confidence available that the site is real.
    That matters here: Metal3D's authors estimate about a third of the zinc sites in the PDB are crystallisation
    artefacts, so a hard presence flag would assert more than the file supports. Cofactor metals are included (the
    iron of a haem is an element FE atom in a HEM group), since to a cavity they are chemistry like any other.
    """
    xyz, el, occ, comp = [], [], [], []
    cur_model = 1
    for line in Path(path).read_text(errors="ignore").splitlines():
        tag = line[:6]
        if tag == "MODEL ":
            cur_model = int(line[10:14]); continue
        if tag == "ENDMDL" and cur_model >= model:
            break
        if cur_model != model or tag != "HETATM" or len(line) < 54:
            continue
        if line[16] not in (" ", "A", "1"):
            continue
        e = _element(line)
        if e.upper() not in METAL_ELEMENTS:
            continue
        try:
            o = float(line[54:60])
        except ValueError:
            o = 1.0
        if o < min_occupancy:
            continue
        xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        el.append(e.upper()); occ.append(o); comp.append(line[17:20].strip())
    return dict(xyz=np.asarray(xyz, float).reshape(-1, 3), element=np.array(el, dtype="<U2"),
                occupancy=np.asarray(occ, float), comp=np.array(comp, dtype="<U3"))
