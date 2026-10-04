"""Two operating modes of EquiCave, because the two real uses have opposite constraints.

**fast** — screening. You have thousands of structures (a benchmark, a proteome, a docking campaign's receptor set)
and you want ranked sites, quickly, on whatever hardware is at hand. No language model, no recycling, a thinner graph
and fewer probes; the Cartesian backbone with vector channels only. Everything needed for site ranking, nothing else.

**accurate** — one target. You are about to design against a pocket and you want its properties, its hotspot field
and a calibrated confidence. ESM-2 embeddings, degree-2 (or e3nn degree-3) channels, probe recycling, the full probe
and surface clouds, rotation averaging at test time and an ensemble over the cluster folds.

The modes differ only in configuration, never in definitions: the lattice, the closure thresholds, the labels and the
metrics are identical, so a number measured in one mode is comparable with the other. `cost` is an order-of-magnitude
note, not a benchmark; the measured timings live in `docs/results/`.
"""
from __future__ import annotations

from pathlib import Path

MODELS = Path(__file__).resolve().parents[2] / "models"

MODES = {
    "fast": dict(
        description="screening: ranked sites only, no language model, no recycling",
        detect=dict(max_sites=30, min_buried=16, fill_min_buried=10),
        network=dict(backbone="cartesian", dim=64, layers=3, lmax=1, use_tensors=False, recycles=0,
                     n_probe=384, n_surf=256, k_scale=0.6, esm=None),
        heads=("sites",),
        ranker="ranker_native.txt",
        test_time=dict(rotations=1, fold_ensemble=False),
        cost="seconds per structure on a CPU core; the geometry stage dominates",
    ),
    "accurate": dict(
        description="single target: sites, pocket properties and the hotspot field, calibrated",
        detect=dict(max_sites=30, min_buried=16, fill_min_buried=10),
        network=dict(backbone="cartesian", dim=128, layers=5, lmax=2, use_tensors=True, recycles=2,
                     n_probe=768, n_surf=512, k_scale=1.0, esm="facebook/esm2_t33_650M_UR50D"),
        heads=("sites", "properties", "hotspots"),
        ranker="ranker_hybrid.txt",
        test_time=dict(rotations=4, fold_ensemble=True),
        cost="tens of seconds per structure with a GPU, minutes on a CPU; ESM-2 and recycling dominate",
    ),
    "peptide": dict(
        description="peptide binders: grooves as well as cavities, peptide-specific features",
        detect=dict(max_sites=30, min_buried=16, fill_min_buried=10, groove=True, groove_max_sites=20),
        network=dict(backbone="cartesian", dim=128, layers=5, lmax=2, use_tensors=True, recycles=2,
                     n_probe=768, n_surf=512, k_scale=1.0, esm="facebook/esm2_t33_650M_UR50D"),
        heads=("sites", "properties"),
        ranker="ranker_peptide.txt",
        test_time=dict(rotations=4, fold_ensemble=True),
        cost="as accurate, plus the groove tier (about 11 s per receptor on a CPU core in the benchmark)",
    ),
}


def mode(name: str) -> dict:
    if name not in MODES:
        raise KeyError(f"unknown mode {name!r}; available: {sorted(MODES)}")
    return MODES[name]


def ranker_path(name: str) -> Path | None:
    p = MODELS / mode(name)["ranker"]
    return p if p.exists() else None


def describe() -> str:
    lines = ["mode       heads                        ranker                 note"]
    for k, v in MODES.items():
        lines.append(f"{k:10s} {','.join(v['heads']):28s} {v['ranker']:22s} {v['description']}")
    return "\n".join(lines)
