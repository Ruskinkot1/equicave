"""The prediction object a user receives, as opposed to the row a benchmark scores.

A benchmark needs one number per site: did the centre land within 4 A. A person deciding whether to run a
screen needs to know where to put the docking box, which residues line the pocket, what chemistry it offers, and
whether to believe any of it. Those are different objects, and conflating them is why most pocket predictors
return a centre and a score that means nothing outside their own ranking.

Three decisions shape this schema.

**The score is a calibrated probability, never the model's raw output.** A LambdaRank score is an ordering
device: it has no units and no meaning across structures, so "0.83" tells a user nothing. The isotonic map fitted
out of fold brings expected calibration error to 0.0019 against 0.0095 for the raw sigmoid, which makes the
number mean what a user will read into it -- that about 80 of 100 sites scored 0.8 are real.

**The pharmacophore map is per sub-region, not per site.** The network predicts seven chemical classes on every
probe, so a pocket can say "a donor fits here, an aromatic ring there" rather than reporting one averaged
character. None of the methods we benchmarked against outputs this, and for a medicinal chemist it is the part
that decides what to screen.

**Caveats travel with the site.** An apo structure has a detection recall about fifteen points lower than a holo
one, and a user who does not know that will read a missing pocket as the absence of a pocket. Where the
prediction is weaker, the object says so, in the same place as the number.
"""
from __future__ import annotations

import numpy as np

SCHEMA = "equicave.site/1"

# The seven chemical classes the hotspot head predicts per probe, and the fourteen site-level property classes.
PHARMACOPHORE = ["hydrophobic_c", "aromatic", "hbd", "hba", "cation", "anion", "halogen"]
LIGAND_TYPE = ["nucleotide", "heme", "peptide", "carbohydrate", "lipid", "metal", "size_small", "size_large",
               "buried_deep", "buried_shallow", "polar", "apolar", "aromatic_ligand", "charged_ligand"]


def docking_box(points: np.ndarray, pad: float = 4.0, min_size: float = 12.0) -> dict:
    """An axis-aligned box around the pocket, in the form AutoDock Vina and most docking tools expect.

    Padded because a pocket's lattice points stop where the free space does, while a ligand's atoms sit against
    the walls and its hydrogens beyond them. The floor keeps a small pocket from producing a box too tight to
    sample: Vina's search degenerates below about 12 A on a side.
    """
    p = np.asarray(points, float)
    if len(p) == 0:
        return {}
    lo, hi = p.min(0) - pad, p.max(0) + pad
    size = np.maximum(hi - lo, min_size)
    centre = (lo + hi) / 2
    return dict(center=[round(float(x), 2) for x in centre], size=[round(float(x), 2) for x in size])


def max_inscribed_radius(points: np.ndarray, protein_xyz: np.ndarray) -> float:
    """Radius of the largest sphere fitting inside the pocket: what size of fragment the site can hold.

    Taken as the greatest distance from any pocket point to the nearest protein heavy atom, which is the usual
    definition and needs no optimisation. It answers a question volume alone cannot -- a long narrow groove and
    a compact cavity of equal volume take very different molecules.
    """
    from scipy.spatial import cKDTree
    p, x = np.asarray(points, float), np.asarray(protein_xyz, float)
    if len(p) == 0 or len(x) == 0:
        return 0.0
    return float(cKDTree(x).query(p, k=1)[0].max())


def pharmacophore(hot_logits: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """The site's chemical character as a distribution over the seven classes, from per-probe predictions.

    Normalised to sum to one so sites of different sizes are comparable: a user reads it as "this pocket is 30 %
    hydrogen-bond acceptor by character", not as a count that grows with volume.
    """
    h = np.asarray(hot_logits, float)
    if h.ndim == 1:
        h = h[None]
    p = 1.0 / (1.0 + np.exp(-h))
    w = np.ones(len(p)) if weights is None else np.asarray(weights, float)
    agg = (p * w[:, None]).sum(0) / max(w.sum(), 1e-9)
    tot = agg.sum()
    return {k: round(float(v / tot if tot > 0 else 0.0), 3) for k, v in zip(PHARMACOPHORE, agg)}


def hotspot_map(points: np.ndarray, hot_logits: np.ndarray, top: int = 12, min_p: float = 0.5) -> list:
    """The strongest per-class sub-regions: where in the pocket each chemistry is favoured.

    This is the part a chemist uses and the part no method we benchmarked against produces. Returned as a short
    list rather than a dense grid, because a list of a dozen points is something a person can look at while a
    field of a thousand is something only another program can.
    """
    p = 1.0 / (1.0 + np.exp(-np.asarray(hot_logits, float)))
    pts = np.asarray(points, float)
    out = []
    for j, name in enumerate(PHARMACOPHORE):
        if j >= p.shape[1]:
            break
        order = np.argsort(-p[:, j])[:max(1, top // len(PHARMACOPHORE))]
        for i in order:
            if p[i, j] >= min_p:
                out.append(dict(center=[round(float(x), 2) for x in pts[i]], type=name, p=round(float(p[i, j]), 3)))
    return sorted(out, key=lambda d: -d["p"])[:top]


def ligand_type(prop_logits: np.ndarray) -> dict:
    """What kind of ligand the site looks built for, as independent probabilities rather than a softmax.

    The classes are not mutually exclusive -- a site can be both deeply buried and aromatic -- so they are not
    normalised against each other, and a user should read each as its own yes/no.
    """
    p = 1.0 / (1.0 + np.exp(-np.asarray(prop_logits, float).ravel()))
    return {k: round(float(v), 3) for k, v in zip(LIGAND_TYPE, p)}


def lining_residues(points: np.ndarray, st: dict, radius: float = 5.0, max_n: int = 40) -> list:
    """The residues that line the pocket, nearest first, with chain and author numbering.

    The first thing a structural biologist looks at, and the thing a centre alone cannot give. Author numbering
    rather than a zero-based index, because that is what the literature, the PDB entry and the user's viewer all
    use; an index would have to be translated by hand every time.
    """
    from scipy.spatial import cKDTree
    p, xyz = np.asarray(points, float), np.asarray(st["xyz"], float)
    if len(p) == 0 or len(xyz) == 0:
        return []
    d, _ = cKDTree(p).query(xyz, k=1)
    near = np.where(d <= radius)[0]
    # resid is "<chain>_<number>" in author numbering, which is what the PDB entry, the literature and the
    # user's viewer all use; a zero-based index would have to be translated by hand every time.
    best: dict = {}
    for i in near:
        key = str(st["resid"][i])
        if key not in best or d[i] < best[key][0]:
            best[key] = (float(d[i]), str(st["resname"][i]), str(st["chain"][i]))
    rows = []
    for key, (dd, resn, chain) in best.items():
        num = key.split("_", 1)[1] if "_" in key else key
        rows.append(dict(chain=chain, resi=num, resn=resn, distance_A=round(dd, 2)))
    return sorted(rows, key=lambda x: x["distance_A"])[:max_n]


def build_site(idx: int, rank: int, probability: float, points: np.ndarray, center, st: dict,
               hot_logits=None, prop_logits=None, caveats=None) -> dict:
    """One site, as a user receives it.

    `probability` must already be calibrated; this function does not transform it, because a transformation
    applied here would be invisible to whoever fitted the calibration.
    """
    pts = np.asarray(points, float)
    site = dict(
        id=f"site-{idx}", rank=rank,
        score=round(float(probability), 3), score_type="calibrated_probability",
        center=[round(float(x), 2) for x in np.asarray(center, float)],
        docking_box=docking_box(pts),
        volume_A3=int(len(pts)),                       # one lattice point is 1 A^3 at the project's grid step
        max_inscribed_radius_A=round(max_inscribed_radius(pts, st["xyz"]), 2),
        residues=lining_residues(pts, st),
        caveats=list(caveats or []),
    )
    if hot_logits is not None and len(pts):
        site["pharmacophore"] = pharmacophore(hot_logits)
        site["hotspots"] = hotspot_map(pts, hot_logits)
    if prop_logits is not None:
        site["ligand_type"] = ligand_type(prop_logits)
    return site


def build_report(sites: list, structure: dict, model: dict) -> dict:
    """The whole answer for one structure: what was predicted, by what, and under which caveats."""
    return dict(schema=SCHEMA, structure=structure, model=model, sites=sites)
