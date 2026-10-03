# Data card

Small derived tables live in `data/processed/` (in git). Raw structures, ESM-2 caches, network feature caches and
third-party databases are **never** committed (`.gitignore`); every one of them is rebuilt by a script in `scripts/`.

## Tables in the repository
| file | content | origin |
|---|---|---|
| `manifest.csv` | 1499 PDB entries: resolution, 30 %-identity cluster, fold (0-4), UniProt, chains, ligands as `[comp, chains, type, weight, charge]` | RCSB PDB search + GraphQL (CC0) |
| `heldout_targets.json` | 12 held-out benchmark families with their 30 % clusters (11) and UniProt accessions (12) | `scripts/data/build_manifest.py` |
| `manifest_peptide.csv` | 973 protein-peptide complexes in 685 receptor clusters, folds, peptide chains, peptide lengths and sequences | RCSB (CC0); clusters disjoint from `manifest.csv` |
| `candidates_native.csv` (+ `.geometry.json`) | 40 986 native candidates of 1367 structures with 32 features, DCA, DCC and label | `scripts/train/build_native.py` |
| `candidates_peptide.csv` | merged cavity + groove candidates of the peptide set with the peptide feature group | `scripts/train/build_peptide.py` |
| `structures_*.csv` | per-structure summary: status, n_sites, candidates, best DCA, runtime | the same scripts |
| `labels_sites.csv`, `labels_summary.json` | one row per ligand site with its property vector; class prevalences | `python -m training pockets-labels` |
| `candidates_fpocket.csv`, `candidates_p2rank.csv` | optional baseline tables in the same layout | `scripts/baselines/run_external.py` |

## Sources and licences
| source | use | licence | status |
|---|---|---|---|
| RCSB PDB structures and metadata | training and evaluation structures, 30 % clusters | CC0 (wwPDB policy) | [проверено] fetched from files.rcsb.org / search.rcsb.org in this session |
| wwPDB Chemical Component Dictionary | ligand atom and ligand class labels | CC0 | [проверено] fetched from files.rcsb.org/ligands |
| ESM-2 (`facebook/esm2_t12_35M_UR50D`, `esm2_t33_650M_UR50D`) | frozen residue embeddings | MIT | [из памяти] — recheck the model card before release |
| rdk/p2rank-datasets (COACH420, HOLO4K id lists) | evaluation lists | **no licence file found** | [проверено] no LICENSE in the repository tree; ids only, cited, not redistributed here |
| LIGYSIS (Zenodo 13121414) | evaluation | CC-BY-4.0 (Zenodo metadata) | [проверено] record metadata read; files are pandas pickles, parsed locally, not committed |
| CryptoBench | cryptic-site evaluation | repository not reachable from this environment | [не найдено] — the user downloads it and places the list in `data/external/eval_sets/cryptobench.csv` |
| PepBDB, Propedia | candidate external peptide-site sets | web databases with their own terms | not used; the peptide benchmark is built from RCSB instead |
| BioLiP | possible training extension | no machine-readable licence ("freely available") | not used for the shipped model; a model trained on it would be labelled non-commercial scope |
| fpocket, P2Rank | optional baselines only | MIT | [из памяти] — recheck before release; both built/downloaded locally, never imported by `src/` or `training/` |

## Labels
- **Site hit**: a candidate counts as a hit when DCA ≤ 4 Å, the distance from its centre to the nearest heavy atom of
  any kept ligand copy. Kept ligand copies have ≥ 8 heavy atoms and a component id that is not a solvent, buffer, ion,
  sugar or detergent (the `EXCLUDE` list of `build_manifest.py`); molecular weight 150–900 Da at manifest level.
- **n_sites**: number of distinct ligand sites per structure (copies whose centroids lie within 8 Å are one site);
  used for top-N and top-(N+2).
- **Residue segmentation**: a residue is positive when one of its heavy atoms is within 4 Å of a ligand heavy atom.
- **Probe occupancy**: a cavity lattice point is positive when a ligand heavy atom is within 2 Å.
- **Property classes** (14, multi-label per site): nucleotide, heme, peptide, carbohydrate, lipid, metal (from the CCD
  component type, name and composition) plus size_small (≤ 15 atoms), size_large (≥ 35), buried_deep (mean 26-ray
  buriedness ≥ 18), buried_shallow (< 12), polar (N+O fraction ≥ 0.35), apolar (≤ 0.15), aromatic_ligand (≥ 5 aromatic
  atoms), charged_ligand.
- **Hotspot classes** (7 per lattice point): a point is positive for a class when a ligand heavy atom of that class is
  within 1.5 Å. Classes from the CCD: hydrophobic C, aromatic, H-bond donor, H-bond acceptor, cation, anion, halogen.
- **Peptide sites**: ligands are polymer chains of 3–30 observed residues lying within 5 Å of a chain of ≥ 50 residues;
  the peptide chains are removed from the receptor input, so the model never sees what it must find.

## Leakage controls
- Splits are by RCSB 30 %-identity cluster; a cluster never spans folds; at most 3 entries per cluster (2 for the peptide set).
- The 12 held-out families are excluded by cluster **and** UniProt accession from every training manifest.
- The peptide benchmark excludes every receptor cluster that occurs in the small-molecule manifest (`--disjoint-from`).
- `scripts/eval/evaluate.py` marks every external-benchmark structure whose 30 % cluster occurs in the training
  manifest as "train-similar" and reports it as a separate row, excluded from the main table.
- Ligand overlap between train and test is reported, never used to filter data.

## Known limitations
- 107 of 1499 manifest entries have no legacy PDB file at RCSB (large structures, mmCIF only) and are skipped;
  25 more have no usable ligand after parsing. The pipeline reads PDB format only.
- RCSB 30 % clusters are recomputed by RCSB over time; the ids here are from 2026-10-03.
- Property and hotspot labels are heuristics over the CCD, not curated annotations; the lipid and heme rules in
  particular are composition-based. Class prevalences are in `labels_summary.json`.
- Crystal structures only, one model, altloc A: no apo/holo pairing, no waters, no metal coordination geometry.
- Peptide sites with fewer than 3 observed residues are excluded, which removes weakly ordered peptides.
