# Data card

Small derived tables live in `data/processed/` (in git). Raw structures, ESM-2 caches, network feature caches and
third-party databases are **never** committed (`.gitignore`); every one of them is rebuilt by a script in `scripts/`.

## Dataset scale and what travels in git

Raw PDB files are never committed: at full scale they are tens of gigabytes (0.61 MB per structure measured), and
they belong to the RCSB PDB, which distributes them under CC0 far better than we could. What is committed is the
**recipe**, which fully determines the data: the manifest (PDB id, 30 %-identity cluster, fold, ligand components),
the geometry constants, and the commit. `scripts/data/export_dataset.py` packages that recipe (about 0.1 MB for 4000
structures) and, with `--include-features`, the derived feature table, optionally sharded by fold so no single file
is large. On the other side `--import` restores it and warns if the geometry constants of that checkout differ, which
would silently change the features.

`scripts/train/build_native.py --stream` downloads each structure and deletes it after featurising, so rebuilding at
full-PDB scale costs time rather than disk.

| manifest | entries | 30 % clusters | state |
|---|---|---|---|
| `manifest.csv` | 1499 (1367 usable) | 1017 | the measured results use this |
| `manifest_big.csv` | 3965 | 2824 | built 2026-10-04; 2076 clusters new; candidates building |
| `manifest_max.csv` | **11 671** (60 000 entries inspected, ≤ 3 per cluster) | **6223** | built 2026-10-04; folds 2349 / 2415 / 2384 / 2178 / 2345 |

The RCSB query behind all three: X-ray, resolution ≤ 2.5 Å, protein-only polymers, ≤ 8 polymer instances, at least
one non-polymer ligand of 150-900 Da that is not a solvent, buffer, ion, sugar or detergent.

## Tables in the repository
| file | content | origin |
|---|---|---|
| `manifest.csv` | 1499 PDB entries: resolution, 30 %-identity cluster, fold (0-4), UniProt, chains, ligands as `[comp, chains, type, weight, charge]` | RCSB PDB search + GraphQL (CC0) |
| `heldout_targets.json` | 12 held-out benchmark families with their 30 % clusters (11) and UniProt accessions (12) | `scripts/data/build_manifest.py` |
| `manifest_peptide.csv` | 973 protein-peptide complexes in 685 receptor clusters, folds, peptide chains, peptide lengths and sequences | RCSB (CC0); clusters disjoint from `manifest.csv` |
| `candidates_native3.csv.gz` (+ `.geometry.json`) | 40 986 native candidates of 1367 structures with 118 features, DCA, DCC and label; `candidates_native.csv.gz` is the earlier table whose interaction-potential group was saturated | `scripts/train/build_native.py` |
| `point_features_native.csv.gz` | the 18 out-of-fold aggregates of the per-point ligandability score, one row per candidate, joinable on (pdb, center) | `scripts/train/train_point_model.py` |
| `points_native.csv.gz` | **not committed**: one row per cavity grid point (about 2.3 M rows) with its 32 features and occupancy label; regenerable by `make points` in about half an hour | `scripts/train/build_points.py` |
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
| LIGYSIS (Zenodo 13121414) | evaluation: 3448 chains | CC-BY-4.0 (Zenodo metadata) | [проверено] record metadata read; the chain list is a pandas pickle, read with a restricted unpickler that refuses anything outside numpy/pandas reconstruction, and not committed |
| CryptoBench (OSF 10.17605/OSF.IO/PZ4A9) | cryptic-site evaluation: 5493 apo entries, 1100 in the held-out test fold | OSF project; licence file in the GitHub repository | [проверено] annotation JSON fetched 2026-10-03; the 1.1 GB CIF archive is not downloaded, structures come from RCSB |
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
- **Hotspot classes** (7 per lattice point), **interaction-validated**: a point is positive for a class when a ligand
  heavy atom of that class lies within 1.5 Å **and that atom actually makes the corresponding interaction with the
  receptor**. Classes and their tests: hydrophobic C (receptor hydrophobic carbon within 4.5 Å), aromatic (receptor
  aromatic ring atom within 5.5 Å or a cation within 5.0 Å), H-bond donor (receptor acceptor within 3.5 Å), H-bond
  acceptor (receptor donor within 3.5 Å), cation (receptor anion within 4.0 Å), anion (receptor cation within 4.0 Å),
  halogen (receptor O or S within 3.8 Å). Receptor atoms are typed by residue and atom name
  (`labels.protein_atom_types`), with HIS nitrogens counted as both donor and acceptor.
  The plain proximity variant is stored alongside as `y_hot_proximity`, so "interaction-validated vs proximity" is an
  ablation rather than an assumption; `labels_summary.json` reports what fraction of each class survives validation.
- **Drug-like ligand filter** (`labels.druglike_ligand`, off by default, `data.druglike_only: true` to enable):
  12–60 heavy atoms, at least one ring, and none of the cofactor-like CCD classes (nucleotide, heme, carbohydrate,
  lipid, metal). The purpose is a field trained on the chemistry that real drugs exploit rather than on cofactor
  scaffolds and crystallisation additives.
- **Peptide sites**: ligands are polymer chains of 3–30 observed residues lying within 5 Å of a chain of ≥ 50 residues;
  the peptide chains are removed from the receptor input, so the model never sees what it must find.

## Dataset cleaning, and how much leaks (measured 2026-10-04)

`scripts/data/clean_manifest.py` applies four filters and reports what each one costs. Run on the 1499-entry
manifest it keeps **432 entries in 383 clusters**:

| filter | entries removed |
|---|---|
| shares a 30 %-identity cluster with COACH420, HOLO4K, LIGYSIS, CryptoBench or a held-out family | **986** |
| ligand quality: fewer than 15 receptor contacts within 4.5 Å, mean occupancy below 0.5, mean b-factor above twice the receptor's, or mean 26-ray closure below 10 | 44 |
| no legacy PDB file at RCSB | 36 |
| more than two entries for the same (cluster, ligand) pair | 1 |

**Two thirds of a PDB-wide drug-bound training set leaks into the standard pocket benchmarks** at the 30 % identity
level, because those benchmarks were themselves drawn from the same part of the PDB: COACH420, HOLO4K, LIGYSIS and
CryptoBench together occupy 5295 clusters. Any method trained on PDB-wide data and evaluated on these sets without
this filter is reporting a partly memorised number — which is also why our own COACH420 evaluation reports the
`not train-similar` subset as the headline row, and why the cleaned manifest is small enough that a larger raw
manifest is a requirement rather than an optimisation.

The ligand-quality thresholds encode a simple rule: a molecule with half occupancy, twice the protein's b-factor and
a handful of contacts is crystallisation noise, and a model trained to find it learns noise.

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
