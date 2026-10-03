# Peptide-binding sites

A benchmark and a candidate generator for peptide binders, built because peptide sites are shallow elongated grooves
and the small-molecule conventions mislead on them.

## The benchmark (built here, from RCSB primary data)

No redistributable, homology-controlled benchmark of *peptide binding sites* was found (PepBDB and Propedia are web
databases under their own terms; BioLiP has no machine-readable licence), so the set is assembled from the RCSB PDB
(CC0) and only PDB ids are stored.

| | |
|---|---|
| selection | X-ray, resolution ≤ 2.5 Å, protein-only polymers, ≥ 2 protein entities, one entity of 4–25 residues (the peptide) and one of ≥ 50 residues (the receptor) |
| entries | 973 in the manifest, 947 with a legacy PDB file, **894 usable** (53 have fewer than 3 observed peptide residues or no receptor chain) |
| redundancy | receptor clusters at 30 % sequence identity, ≤ 2 entries per cluster, **685 receptor clusters**, 5 folds assigned per cluster |
| leakage control | every receptor cluster of the small-molecule training manifest is excluded (`--disjoint-from`), as are the 12 held-out target families by cluster and UniProt |
| peptide length | median 12 observed residues (10th percentile 5, 90th 30) |
| sites per receptor | mean 1.66, median 1 |
| label | DCA ≤ 4 Å from a candidate centre to any peptide heavy atom; peptide chains are **removed from the receptor input**, so the detector never sees what it must find |

Build: `scripts/data/build_peptide_manifest.py` then `scripts/train/build_peptide.py` (`make peptide-data`).

## Candidate generation by tier

Candidates are the union of the cavity tiers (`detect.detect_sites`, ≤ 30) and the groove tier
(`peptide.detect_peptide_sites`, ≤ 20), merged without suppressing overlaps: a groove and a cavity candidate may
describe the same region in two different ways and both descriptions are given to the ranker, with their mutual
distance as a feature.

| | ceiling (any candidate within 4 Å) | mean candidates |
|---|---|---|
| cavity tiers alone | 0.930 | 30 |
| groove tier alone | 0.768 | ≤ 20 |
| both (as used) | **0.945** | 44.3 |

11.3 s per receptor on one core; median best DCA 0.67 Å.

**The informative negative result:** groove candidates uniquely find only **1.6 %** of peptide sites that the cavity
tiers miss (0.753 of receptors are hit by both tiers). Peptide grooves are therefore *detectable* by ordinary cavity
closure analysis — anchor residues of a peptide sit in genuine sub-pockets — and the limiting factor for peptide-site
prediction is **ranking**, not detection. The groove tier earns its place through the features it contributes
(elongation, flatness, and the exposed receptor backbone a peptide pairs with), not through extra coverage. Whether
those features actually improve ranking is measured by the `without peptide` row of `ranker_peptide.md`; a null result
there is reported as a null result.

## What is not done yet
- Ranking with the network's scores (the network has not been trained).
- A comparison against dedicated peptide-site predictors (PepNN, InterPep2): their weights and licences have to be
  verified first, and they are run, if at all, through the wrapper contract in `scripts/baselines/wrappers/`.
- Apo receptors: here the receptor is the holo structure with the peptide deleted, which is easier than a true apo
  conformation. CryptoBench (fetched, 1100 test entries) is the set for that question.
