"""Ligand chemistry from the wwPDB Chemical Component Dictionary (CCD, CC0): atom classes and ligand classes.

A CCD entry gives every atom (element, formal charge, aromatic flag) and every bond, including hydrogens, so
hydrogen-bond donors and acceptors, aromatic atoms, charged groups and hydrophobic carbons can be assigned without
a cheminformatics toolkit. Files are fetched once from files.rcsb.org/ligands and cached in data/cache/ccd (not in git).

Hotspot atom classes (multi-label per heavy atom):
  hydrophobic_c  carbon bonded only to C/H (aliphatic) or aromatic carbon with no heteroatom neighbour
  aromatic       atom with the CCD aromatic flag
  hbd            N/O/S with at least one bonded hydrogen
  hba            O (any), N without hydrogen that is not amide/aromatic-substituted with three heavy neighbours
  cation         positive formal charge, or sp3 amine N with >= 1 H and 3 heavy/H neighbours not next to C=O, or guanidine/amidine N
  anion          negative formal charge, or O of a carboxylic / phosphoric / sulfonic acid group
  halogen        F, Cl, Br, I
"""
from __future__ import annotations

import re
import time
import urllib.request
from pathlib import Path

import numpy as np

CCD_URL = "https://files.rcsb.org/ligands/download/{}.cif"
CACHE = Path(__file__).resolve().parents[2] / "data/cache/ccd"
ATOM_CLASSES = ["hydrophobic_c", "aromatic", "hbd", "hba", "cation", "anion", "halogen"]
HALOGENS = {"F", "CL", "BR", "I"}
METALS = set("LI NA K RB CS MG CA SR BA MN FE CO NI CU ZN CD HG AL GA PB PT AU AG MO W V CR TL YB LA GD SM EU TB OS IR RU PD RH RE U".split())


def fetch(comp: str, cache: Path = CACHE, tries: int = 3) -> str | None:
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{comp}.cif"
    if f.exists():
        return f.read_text()
    for t in range(tries):
        try:
            txt = urllib.request.urlopen(CCD_URL.format(comp), timeout=60).read().decode()
            f.write_text(txt)
            return txt
        except Exception:  # noqa: BLE001
            time.sleep(1 + t)
    return None


def _int(x) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def _loop(txt: str, prefix: str) -> list[dict]:
    """Rows of a CIF loop whose tags start with `prefix` (e.g. '_chem_comp_atom.')."""
    m = re.search(r"loop_\s*((?:%s\S+\s*)+)" % re.escape(prefix), txt)
    if not m:
        # single-row (non-loop) block
        tags = re.findall(r"^%s(\S+)\s+(.+)$" % re.escape(prefix), txt, re.M)
        return [{k: v.strip().strip('"') for k, v in tags}] if tags else []
    tags = [t[len(prefix):] for t in m.group(1).split()]
    body = txt[m.end():]
    end = re.search(r"^(loop_|_\S+|#)", body, re.M)
    body = body[:end.start()] if end else body
    tokens = re.findall(r'"[^"]*"|\'[^\']*\'|\S+', body)
    rows = []
    for i in range(0, len(tokens) - len(tags) + 1, len(tags)):
        rows.append({k: v.strip('"\'') for k, v in zip(tags, tokens[i:i + len(tags)])})
    return rows


def parse(txt: str) -> dict:
    """Atoms (name, element, charge, aromatic), bonds (a1, a2, order, aromatic), component type and name."""
    atoms = _loop(txt, "_chem_comp_atom.")
    bonds = _loop(txt, "_chem_comp_bond.")
    head = {k: v for k, v in re.findall(r"^_chem_comp\.(\S+)\s+(.+)$", txt, re.M)}
    A = {a["atom_id"]: dict(element=a.get("type_symbol", "C").upper(), charge=_int(a.get("charge", "0")),
                            aromatic=a.get("pdbx_aromatic_flag", "N") == "Y", nbrs=[]) for a in atoms}
    for b in bonds:
        a1, a2 = b.get("atom_id_1"), b.get("atom_id_2")
        if a1 in A and a2 in A:
            A[a1]["nbrs"].append((a2, b.get("value_order", "SING")))
            A[a2]["nbrs"].append((a1, b.get("value_order", "SING")))
    return dict(atoms=A, type=head.get("type", "").strip('"').upper(), name=head.get("name", "").strip('"').upper(),
                formula=head.get("formula", "").strip('"'))


def atom_classes(ccd: dict) -> dict[str, np.ndarray]:
    """Per heavy-atom name: boolean vector over ATOM_CLASSES."""
    A = ccd["atoms"]
    out = {}
    for name, a in A.items():
        el = a["element"]
        if el == "H" or el == "D":
            continue
        nb = [(A[n]["element"], o) for n, o in a["nbrs"] if n in A]
        heavy = [(e, o) for e, o in nb if e not in ("H", "D")]
        n_h = sum(e in ("H", "D") for e, _ in nb)
        v = np.zeros(len(ATOM_CLASSES), bool)
        if el == "C" and all(e in ("C", "H", "D") for e, _ in nb):
            v[0] = True
        if a["aromatic"]:
            v[1] = True
        if el in ("N", "O", "S") and n_h > 0:
            v[2] = True
        if el == "O":
            v[3] = True
        if el == "N" and n_h == 0 and len(heavy) < 3 and not _amide_n(name, A):
            v[3] = True
        if el == "N" and a["aromatic"] and n_h == 0 and len(heavy) == 2:
            v[3] = True
        if a["charge"] > 0 or (el == "N" and n_h >= 1 and not a["aromatic"] and _amine_like(name, A)):
            v[4] = True
        if a["charge"] < 0 or (el == "O" and _acid_oxygen(name, A)):
            v[5] = True
        if el in HALOGENS:
            v[6] = True
        out[name] = v
    return out


def _carbonyl_c(name, A) -> bool:
    return A[name]["element"] == "C" and any(A[n]["element"] == "O" and o.upper().startswith("DOUB") for n, o in A[name]["nbrs"])


def _amide_n(name, A) -> bool:
    return any(A[n]["element"] == "C" and _carbonyl_c(n, A) for n, _ in A[name]["nbrs"])


def _amine_like(name, A) -> bool:
    """sp3 amine (all single bonds, no neighbouring carbonyl / aromatic / sp2 C) or guanidine / amidine N."""
    a = A[name]
    if any(o.upper().startswith(("DOUB", "AROM")) for _, o in a["nbrs"]):
        return _amidine_n(name, A)
    for n, _ in a["nbrs"]:
        c = A[n]
        if c["element"] == "C" and (_carbonyl_c(n, A) or c["aromatic"] or any(o.upper().startswith("DOUB") for _, o in c["nbrs"])):
            return _amidine_n(name, A)
        if c["element"] in ("S", "P"):
            return False
    return True


def _amidine_n(name, A) -> bool:
    for n, _ in A[name]["nbrs"]:
        c = A[n]
        if c["element"] == "C":
            ns = [m for m, _ in c["nbrs"] if A[m]["element"] == "N"]
            if len(ns) >= 2 and any(o.upper().startswith("DOUB") and A[m]["element"] == "N" for m, o in c["nbrs"]):
                return True
    return False


def _acid_oxygen(name, A) -> bool:
    """O attached to a C/P/S that carries another O (carboxylate, phosphate, sulfonate / sulfate)."""
    for n, _ in A[name]["nbrs"]:
        c = A[n]
        if c["element"] in ("C", "P", "S"):
            os_ = [m for m, _ in c["nbrs"] if A[m]["element"] == "O"]
            if c["element"] == "C" and len(os_) >= 2 and _carbonyl_c(n, A):
                return True
            if c["element"] in ("P", "S") and len(os_) >= 3:
                return True
    return False


# ---- ligand (component) classes for the property head -------------------------------------------------------------
LIGAND_CLASSES = ["nucleotide", "heme", "peptide", "carbohydrate", "lipid", "metal"]
_NUC_NAMES = ("ADENOSINE", "GUANOSINE", "CYTIDINE", "URIDINE", "THYMIDINE", "INOSINE", "ADENINE", "GUANINE", "NICOTINAMIDE-ADENINE",
              "FLAVIN", "COENZYME A", "ACETYL COENZYME", "S-ADENOSYL", "ADENYL", "GUANYL", "CYTIDYL", "URIDYL", "DEOXY")


def ligand_classes(ccd: dict) -> np.ndarray:
    """Boolean vector over LIGAND_CLASSES from the CCD entry (type, name, composition)."""
    A = ccd["atoms"]; t = ccd["type"]; nm = ccd["name"]
    el = [a["element"] for a in A.values() if a["element"] not in ("H", "D")]
    n_p = el.count("P"); n_c = el.count("C"); n_n = el.count("N"); n_o = el.count("O")
    v = np.zeros(len(LIGAND_CLASSES), bool)
    v[0] = ("NUCLEOTIDE" in nm or any(k in nm for k in _NUC_NAMES) or "RNA" in t or "DNA" in t) and n_p >= 1
    v[1] = ("FE" in el or "CO" in el or "MG" in el) and n_n >= 4 and any(a["aromatic"] for a in A.values()) and ("HEM" in nm or "PORPH" in nm or "CHLOROPHYLL" in nm or "COBALAMIN" in nm or n_c >= 30)
    v[2] = "PEPTIDE" in t or "PEPTIDE" in nm
    v[3] = "SACCHARIDE" in t or "GLUCOS" in nm or "GALACTOS" in nm or "MANNOS" in nm or "FUCOS" in nm or "SIALIC" in nm
    v[4] = _longest_aliphatic_chain(A) >= 8 and n_c >= 12 and (n_n + n_o) <= max(6, n_c // 3)
    v[5] = any(e in METALS for e in el)
    return v


def _longest_aliphatic_chain(A) -> int:
    """Longest path of non-aromatic carbons bonded only to C/H (a crude acyl-chain detector)."""
    ok = {n for n, a in A.items() if a["element"] == "C" and not a["aromatic"] and all(A[m]["element"] in ("C", "H", "D") for m, _ in a["nbrs"] if m in A)}
    best = 0
    for s in ok:
        seen = {s}; frontier = [(s, 1)]
        while frontier:
            n, d = frontier.pop()
            best = max(best, d)
            for m, _ in A[n]["nbrs"]:
                if m in ok and m not in seen:
                    seen.add(m); frontier.append((m, d + 1))
    return best


def load(comp: str, cache: Path = CACHE) -> dict | None:
    txt = fetch(comp, cache)
    return parse(txt) if txt else None
