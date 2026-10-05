"""Apo/holo superposition: the Kabsch fit, the sequence match, and the ligand transfer that makes CryptoBench
measurable at all. Before this existed, 222 of the 228 published test entries were silently skipped as having no
ligand, because the ligand is in the holo structure and prediction happens on the apo one."""
import numpy as np
import pytest

from equicave import superpose as SP, structure


def test_kabsch_recovers_an_exact_rigid_transform():
    rng = np.random.default_rng(0)
    p = rng.normal(size=(60, 3)) * 10
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    t_true = np.array([5.0, -3.0, 2.0])
    r, t, rmsd = SP.kabsch(p, p @ q.T + t_true)
    assert rmsd == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(r, q, atol=1e-6) and np.allclose(t, t_true, atol=1e-6)


def test_kabsch_never_returns_a_reflection():
    """A mirrored point set must be fitted by a rotation, not a reflection: a reflected 'fit' would move a ligand
    into a mirror-image pocket and the DCA would be meaningless rather than obviously wrong."""
    rng = np.random.default_rng(1)
    p = rng.normal(size=(40, 3))
    mirrored = p * np.array([1.0, 1.0, -1.0])
    r, _, _ = SP.kabsch(p, mirrored)
    assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-6)


def test_lcs_matches_shared_residues_in_order():
    pairs = SP._lcs_pairs("ACDEFGH", "AXCDEYGH")
    a, b = "ACDEFGH", "AXCDEYGH"
    assert all(a[i] == b[j] for i, j in pairs)
    assert [i for i, _ in pairs] == sorted(i for i, _ in pairs)      # strictly increasing: an alignment, not a bag
    assert len(pairs) >= 6
    assert SP._lcs_pairs("", "ABC") == [] and SP._lcs_pairs("ABC", "") == []


def _chain_pdb(tmp_path, name, coords, resnames, chain="A", ligand=None):
    lines = []
    for i, ((x, y, z), rn) in enumerate(zip(coords, resnames)):
        lines.append(f"ATOM  {i + 1:5d}  CA  {rn} {chain}{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 10.00           C")
    if ligand is not None:
        for i, (x, y, z) in enumerate(ligand):
            lines.append(f"HETATM{900 + i:5d}  C{i:<2d} LIG X 900    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 10.00           C")
    p = tmp_path / f"{name}.pdb"; p.write_text("\n".join(lines) + "\nEND\n")
    return p


def test_a_ligand_is_carried_from_the_holo_frame_into_the_apo_frame(tmp_path):
    """The whole point: the apo file has no ligand, and after alignment the holo ligand lands where it belongs."""
    rng = np.random.default_rng(2)
    n = 60
    apo_ca = rng.normal(size=(n, 3)) * 12
    res = ["ALA", "LEU", "SER", "ASP", "LYS", "PHE", "GLY", "VAL"] * ((n // 8) + 1)
    res = res[:n]
    # the holo structure is the same protein in a rotated and translated frame, with a ligand at a known site
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    shift = np.array([30.0, -10.0, 7.0])
    site_apo = apo_ca[:6].mean(0)                              # the pocket, expressed in the apo frame
    lig_apo = site_apo + rng.normal(0, 0.8, size=(10, 3))
    holo_ca = apo_ca @ q.T + shift
    lig_holo = lig_apo @ q.T + shift
    apo = _chain_pdb(tmp_path, "apo", apo_ca, res, chain="A")
    hol = _chain_pdb(tmp_path, "holo", holo_ca, res, chain="B", ligand=lig_holo)

    assert structure.read_ligands(apo, min_heavy=8) == []       # the apo form really has no ligand
    tr = SP.align_chains(apo, hol, "A", "B")
    assert tr["ok"], tr
    assert tr["rmsd"] == pytest.approx(0.0, abs=1e-3)
    assert tr["n_matched"] == n
    back = SP.transfer_ligands(hol, tr, codes={"LIG"})
    assert len(back) == 1
    assert np.allclose(back[0]["xyz"], lig_apo, atol=1e-3)      # landed on the apo-frame pocket
    assert np.linalg.norm(back[0]["xyz"].mean(0) - site_apo) < 1.0


def test_a_mismatched_pair_is_rejected_with_a_reason(tmp_path):
    """An entry that cannot be aligned must be excluded on a number, not scored on a bad transform."""
    rng = np.random.default_rng(3)
    res = ["ALA", "LEU", "SER", "ASP"] * 15
    apo = _chain_pdb(tmp_path, "a", rng.normal(size=(60, 3)) * 12, res, chain="A")
    other = _chain_pdb(tmp_path, "b", rng.normal(size=(60, 3)) * 12, res, chain="B")
    tr = SP.align_chains(apo, other, "A", "B")
    assert not tr["ok"] and "RMSD" in tr["reason"]
    assert SP.transfer_ligands(other, tr) == []                 # and nothing is transferred from a failed fit
    short = _chain_pdb(tmp_path, "c", rng.normal(size=(5, 3)), ["ALA"] * 5, chain="C")
    assert not SP.align_chains(apo, short, "A", "C")["ok"]


def test_multi_chain_specifications_are_accepted(tmp_path):
    """CryptoBench writes a multi-chain site as 'M-N-O', so the chain field is a set, not a single letter."""
    rng = np.random.default_rng(4)
    res = ["ALA", "LEU", "SER", "ASP"] * 15
    ca = rng.normal(size=(60, 3)) * 12
    lines = []
    for i, ((x, y, z), rn) in enumerate(zip(ca, res)):
        ch = "A" if i < 30 else "B"
        lines.append(f"ATOM  {i + 1:5d}  CA  {rn} {ch}{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 10.00           C")
    p = tmp_path / "two.pdb"; p.write_text("\n".join(lines) + "\nEND\n")
    tr = SP.align_chains(p, p, "A-B", "A-B")
    assert tr["ok"] and tr["rmsd"] == pytest.approx(0.0, abs=1e-6)
    assert tr["n_matched"] == 60                                 # both chains were used, not just the first
