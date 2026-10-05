"""Superpose one structure onto another by sequence-matched Cα, and carry a ligand across with the transform.

Why this exists. CryptoBench is a cryptic-site benchmark: each entry names an **apo** structure, where the pocket is
closed and no ligand is present, and separately the **holo** structure where the ligand actually sits. Prediction is
made on the apo form -- that is the whole point of the benchmark -- but the label can only come from the holo form.
Without a superposition the evaluation looks for a ligand in the apo file, finds none, and silently reports the
entry as having no ligand: measured on the published test split, 222 of 228 entries were skipped that way and 3
were scored, so the benchmark was not being measured at all.

The alignment is deliberately simple and is stated in the paper as such: residues common to both chains by their
one-letter code and sequence position under a longest-common-subsequence match, Kabsch rotation on their Cα atoms,
and a reported RMSD so an entry whose alignment failed can be dropped on a number rather than on faith. No
structural alignment library, no flexible fitting -- an apo/holo pair of the same UniProt entry is the easy case for
superposition, and an entry whose Cα RMSD exceeds the threshold is excluded and counted.
"""
from __future__ import annotations

import numpy as np

from . import structure


def kabsch(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Rotation and translation taking `p` onto `q` (both [n,3], paired), plus the resulting RMSD."""
    pc, qc = p.mean(0), q.mean(0)
    h = (p - pc).T @ (q - qc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    t = qc - r @ pc
    rmsd = float(np.sqrt((((p @ r.T + t) - q) ** 2).sum(1).mean()))
    return r, t, rmsd


def _lcs_pairs(a: str, b: str) -> list[tuple[int, int]]:
    """Index pairs of a longest common subsequence of two sequences; O(len(a)*len(b)) and fine at protein length."""
    n, m = len(a), len(b)
    if not n or not m:
        return []
    dp = np.zeros((n + 1, m + 1), np.int32)
    for i in range(n):
        for j in range(m):
            dp[i + 1, j + 1] = dp[i, j] + 1 if a[i] == b[j] else max(dp[i, j + 1], dp[i + 1, j])
    out, i, j = [], n, m
    while i and j:
        if a[i - 1] == b[j - 1] and dp[i, j] == dp[i - 1, j - 1] + 1:
            out.append((i - 1, j - 1)); i -= 1; j -= 1
        elif dp[i - 1, j] >= dp[i, j - 1]:
            i -= 1
        else:
            j -= 1
    return out[::-1]


def align_chains(apo_path, holo_path, apo_chain: str = "", holo_chain: str = "",
                 max_rmsd: float = 4.0) -> dict:
    """Transform taking the holo structure onto the apo one, with the diagnostics to judge it.

    Chains are given as the benchmark writes them, which may be several joined by '-' (CryptoBench does this for a
    multi-chain binding site); every listed chain is used. Returns `ok`, the rotation and translation, the RMSD, the
    number of matched residues, and a reason when it is not usable.
    """
    chains = lambda s: set(filter(None, str(s).replace(",", "-").split("-"))) or None
    apo = structure.read_pdb(apo_path, chains(apo_chain))
    holo = structure.read_pdb(holo_path, chains(holo_chain))
    if len(apo["xyz"]) < 30 or len(holo["xyz"]) < 30:
        return dict(ok=False, reason="a chain is missing or too short")
    ta, th = structure.residue_table(apo), structure.residue_table(holo)
    pairs = _lcs_pairs("".join(ta["seq"]), "".join(th["seq"]))
    if len(pairs) < 20:
        return dict(ok=False, reason=f"only {len(pairs)} residues matched by sequence", n_matched=len(pairs))
    ia = np.array([p[0] for p in pairs]); ih = np.array([p[1] for p in pairs])
    pa, ph = ta["ca"][ia], th["ca"][ih]
    good = np.isfinite(pa).all(1) & np.isfinite(ph).all(1)
    if good.sum() < 20:
        return dict(ok=False, reason=f"only {int(good.sum())} matched residues have a C-alpha", n_matched=int(good.sum()))
    r, t, rmsd = kabsch(ph[good], pa[good])            # holo -> apo
    if rmsd > max_rmsd:
        return dict(ok=False, reason=f"C-alpha RMSD {rmsd:.2f} A exceeds {max_rmsd} A", rmsd=rmsd,
                    n_matched=int(good.sum()))
    return dict(ok=True, rotation=r, translation=t, rmsd=rmsd, n_matched=int(good.sum()),
                apo_atoms=len(apo["xyz"]), holo_atoms=len(holo["xyz"]))


def transfer_ligands(holo_path, transform: dict, codes=None, min_heavy: int = 8) -> list[dict]:
    """Ligands of the holo structure, moved into the apo frame by `transform` from `align_chains`."""
    if not transform.get("ok"):
        return []
    r, t = transform["rotation"], transform["translation"]
    out = []
    for lig in structure.read_ligands(holo_path, min_heavy=min_heavy):
        if codes and lig["comp"] not in codes:
            continue
        out.append(dict(lig, xyz=lig["xyz"] @ r.T + t))
    return out
