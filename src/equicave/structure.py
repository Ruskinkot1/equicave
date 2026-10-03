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

    Returns xyz [N,3], element, resname, resid ('A_123'), chain, atom name, backbone flag, b-factor, occupancy.
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
