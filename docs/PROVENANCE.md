# Provenance: borrowed ideas, our implementations, and what was verified

No third-party source code is copied into this repository. Every file in `src/` and `training/` was written here.
Where an idea comes from a publication, it is listed below with the form in which we implemented it.

| our component | borrowed idea | source | our implementation |
|---|---|---|---|
| `pockets._buried`, `detect.buriedness_grid` | grid-based buriedness by ray casting | LIGSITE (Hendlich 1997) | 26 lattice directions, 8 Å rays, 2 Å hit radius; computed by integer array shifts on an absolute 1 Å lattice; the lattice form is tested against an exact per-point ray caster |
| `detect.detect_sites` peak splitting | non-maximum suppression to split one cavity into sub-sites | common practice in object detection; P2Rank clusters SAS points instead | Gaussian smoothing of the buriedness field, greedy NMS at 6 Å, nearest-peak assignment, two depth tiers |
| `point_score`, `scripts/train/build_points.py` | score every point of a candidate individually and add the scores up, so a pocket's score grows with how many of its points look ligandable | P2Rank (Krivák & Hoksza 2018), which classifies SAS points with a random forest and sums them per cluster | our points are the detector's own 1 Å cavity grid points, not a SAS triangulation; 32 features per point (the seven interaction-potential counts, distances and strict flags, lattice buriedness, local composition, centrality); labels are occupancy by a real ligand heavy atom within 2 Å, not a derived ligandability score; aggregation adds the distribution shape, the largest connected high-scoring blob and the score-weighted centroid shift, and the model is trained out of fold by 30 %-identity cluster |
| `pockets.sas_points` | solvent-accessible surface sampling | Shrake & Rupley 1973 | Fibonacci sphere per atom, vdW + 1.4 Å probe, neighbour occlusion test; returns a point cloud with owner atom indices |
| `training/pockets/model.py` channels | scalar + vector channels gated by invariants | PaiNN (Schütt 2021), GVP (Jing 2021) | our own gating, plus degree-2 symmetric traceless tensors |
| `GeoTensorAttention` | attention over equivariant messages with high-degree channels (geometric tensor attention) | SE(3)-Transformer (Fuchs 2020), Equiformer (Liao 2023), GotenNet (Aykent & Xia 2025) | Cartesian l ≤ 2 primitives written out by hand (dot, `uᵀTu`, `Tu`, `sym_traceless`), segment softmax over incoming edges, pseudo-scalar triple products for chirality; no e3nn, no copied code |
| probe (virtual) nodes | virtual nodes that learn to sit at binding sites | VN-EGNN (Sestak 2024/2025) | probes are initialised on **free cavity lattice points** with their own buriedness and direction features, not on a sphere; heterogeneous edge types by node-type pair |
| surface module | a dedicated module over the molecular surface | EquiPocket (Zhang 2023) | SAS **point cloud** nodes with owner element/residue features and outward normals as initial vectors, inside the same attention layers |
| ESM-2 node features | frozen language-model embeddings as residue features | ESM-2 (Lin 2023); used by VN-EGNN and GDEGAN | `transformers` inference, cached as float16 per structure; ablated with `no_esm` |
| segmentation + centre losses | Dice loss for site segmentation, distance-to-centre regression with a set assignment | VN-EGNN (Sestak 2024/2025) | Dice + BCE with pos_weight; set loss over **all** sites of the structure using the nearest probe within 8 Å; confidence head with a detached target |
| per-point ligandability (`src/equicave/point_score.py`) | classifying an **empty grid point** as pocket or not | **SiteRadar** (Evteev 2023), which scores grid points with a GNN over the atoms around each one; P2Rank's additive per-point scoring | our 32 rotation-invariant features per free cavity point, a LightGBM model, out-of-fold per fold, and 18 aggregates per candidate including a union-find blob. The idea of scoring the empty point is theirs; the features, the model and the aggregation are ours. We found no code or weights for SiteRadar, so no comparison was run |
| site decoder losses (`site_rank_loss`, `site_margin_loss`) | contrastive training of probe scores against decoy probes | **YuelPocket** (Wang & Dokholyan, PNAS 2026) trains SAS probes with InfoNCE: the probe near the true centre against probes beyond 4 A, and against the true probe paired with a wrong ligand | our terms are listwise cross-entropy over the decoder's own site list plus a hinge on the single best-wrong against best-right comparison, with no ligand conditioning anywhere; read after our implementation was written, and recorded here because the objective is of the same family |
| hotspot field | per-point ligand-atom type probability (pharmacophore-style maps) | fragment hotspot maps (Radoux 2016), FTMap-type methods | focal BCE over 7 CCD-derived atom classes on cavity lattice points, labelled from crystal ligand atoms within 1.5 Å |
| top-(N+2) and the ligand rule | evaluation convention for pocket detection | P2Rank (Krivák 2018), DeepPocket (Aggarwal 2021) | implemented in `metrics.per_structure`; the `_mlig` lists are fetched, not redistributed |
| hybrid ranking | a learned re-ranker over candidates beats end-to-end detection | LIGYSIS comparison (Utgés & Barton 2024) | LightGBM LambdaRank over our own features, cluster CV, paired cluster bootstrap |
| heterogeneous message passing (`GeoTensorAttention`, `het_mp`) | separate message functions per node-type pair in a graph with virtual nodes | **VN-EGNN** (Sestak 2024/2025): separate transforms for its virtual nodes, worth +0.02 to +0.05 DCC in its own ablation and null-to-harmful on 3 of 6 metrics without rich protein features | ours is one invariant gain per edge type per stream, zero-initialised, multiplying the existing message rather than replacing the function. Indexed by the ordered pair and not the destination, because a destination-only gain factors out of the attention-weighted sum and would be a per-type scaling of the node update rather than a message function -- a test checks that arithmetic. Their virtual nodes sit on a sphere around the protein; ours are cavity lattice points, so the analogy covers the mechanism and not the construction |
| per-probe conservation (`pocket_features.point_conservation`) | per-residue MSA conservation as a pocket-ranking feature | **P2Rank_CONS** (Krivák & Hoksza), whose shipped conservation-enabled model reaches top-(N+2) 53.9 % against 51.9 % on LIGYSIS; PRANK (2015) deliberately excludes it, so the choice is contested in the same lineage | ours is per cavity probe rather than per candidate: 1/r-weighted mean over the lining atoms, the maximum, and the *coverage* of the lining by the alignment, so a thinly aligned pocket is not read as an unconserved one. The scores themselves are read from a file in P2Rank's own two-column format; we compute no alignment and ship none |
| metal channels (`pocket_features.point_metal`, `structure.read_metals`) | a metal ion as a model input at all | **DeepSite** (Jiménez 2017) has a `metallic` grid channel; **Kalasanty** (Stepniewska-Dziubinska 2020) and **DeepSurf** (Mylonas 2021) inherit a `metal` bit from Pafnucy's 18 atom features. None of the three ablates it, and DeepPocket and GrASP strip heteroatoms instead | per probe, per chemical role (transition / alkaline earth / alkali): distance to the nearest ion and an **occupancy-weighted** count. The grouping, the occupancy weighting and the per-probe form are ours. The weighting is motivated by **Metal3D**'s (Dürr 2023) estimate that about a third of PDB zinc sites are crystallisation artefacts, so a presence bit would assert more than the file supports |
| screened-Coulomb channels (`pocket_features.point_field`, `formal_charges`) | electrostatics as a network input without a solver | **dMaSIF** (Sverrisson, CVPR 2021) removed MaSIF's Poisson–Boltzmann solve, replaced it with a learned function of atom types and inverse distances, and *gained* accuracy (0.85 → 0.87 on the same interface split) at ~1/1000 of the preprocessing cost. What we take from that is the negative claim: a solver is not worth paying for here | our term is neither theirs nor a PB solve: formal charges spread over the atoms of each ionised group, a Debye–Hückel screened Coulomb sum (λ = 7.8 Å), the potential and ‖E‖ as scalars and **E itself as a degree-1 channel**. No force-field parameter file is read, so no force-field licence applies. We found no published comparison of a cheap Coulomb field against a PB solve as a network input for any of the three tasks, which is why the arm exists |
| peptide groove features | backbone-to-backbone recognition in peptide grooves | β-augmentation / PDZ, SH3, MHC structural literature | `peptide.backbone_exposure`: counts of receptor backbone N, O, Cα and side-chain atoms within 6 Å of each candidate point, plus shape anisotropy |

## Verification log
- RCSB and CCD licensing (CC0): read from the wwPDB policy pages and the file endpoints used, 2026-10-03 [проверено].
- p2rank-datasets has no LICENSE file at the tree root: checked by fetching the raw paths, 2026-10-03 [проверено].
  Only PDB ids are used; the lists are fetched by the user's own run and are not committed.
- LIGYSIS Zenodo record 13121414 metadata states CC-BY-4.0, 2026-10-03 [проверено].
- CryptoBench repository was not reachable from this environment, 2026-10-03 [не найдено].
- fpocket 4.x and P2Rank 2.5.1 were built/downloaded locally **only** for `scripts/baselines/`; neither `src/` nor
  `training/` imports or executes them, and no Java is required to train or run EquiCave.
- SiteRadar (J. Chem. Inf. Model. 63:1124-1132, 2023) could not be read at the primary source: ACS returns
  403 from this environment and no preprint was found, so its architecture is recorded from a secondary
  description in the YuelPocket preprint, 2026-10-06 [из вторичного источника]. Its DCC 0.760 likewise comes
  from someone else's baseline table with no stated threshold and must not be compared with our numbers.
- YuelPocket's published weights and its mandatory ligand input were checked in its repository, 2026-10-06
  [проверено]; its licence is still not stated there [не найдено].
- ESM-2 licence (MIT) is [из памяти] in this session: the model card was not opened from the primary source.
- Gaussian Dynamic Attention is taken as an idea from GDEGAN (Wang et al., arXiv 2603.19817, 2026), read from the
  arXiv PDF on 2026-10-07 [проверено]. What is borrowed is the form of the attention score — a Gaussian kernel on
  variance-normalised feature differences with a learnable width per head, in place of a learned key-query
  projection — and the observation that it preserves the backbone's equivariance because it reads only invariant
  features. No code was consulted or copied; `GeoTensorAttention.attn_logits` and `seg_mean` are written here, and
  the per-destination normalisation is our own choice, made so the statistics cannot cross a structure boundary.
  Their Table 1 and Figure 4a are used as evidence in `docs/results/README.md`; their licence is not stated in the
  preprint [не найдено], which is why only the idea and the published numbers are used.
- The electrostatics, hydration, protonation/metal, chemistry-in-network and pharmacophore-output literature was
  read on 2026-10-08; the notes with per-claim verification marks are in
  `research_notes/Химия в архитектуре предсказания/` and the synthesis in `reports/`. Three numbers used above
  were read from the primary PDFs [проверено]: dMaSIF Table 1 (feature step 19.69 ± 16.08 s per protein against a
  36 ms forward pass), Kalasanty's "18 atomic features used in our previous project" and DeepSurf's matching
  sentence, which is how we know both inherit Pafnucy's single partial-charge channel without testing it. That
  **no** site predictor conditions on protonation state, and that **no** metal input channel has ever been
  ablated, are absence claims from that search [не найдено], not proofs.
- **AllMetal3D** (MIT licence, pip-installable, 11 metals) is the candidate for *predicted* metal positions in apo
  structures, where there are no HETATM ions to read. Not implemented and not run: `point_metal` reads the file.
- The Debye–Hückel screening length (7.8 Å at 150 mM monovalent salt, 298 K) and the formal charges of Arg, Lys,
  Asp and Glu are textbook, taken from no paper in particular [из памяти]. Nothing here uses Amber, CHARMM,
  Gasteiger or AM1-BCC parameters, so the charge set carries no third-party licence.

## Architecture audit, 2026-10-08

Reviewed under `.claude/skills/equivariant-gnn-researcher` after the chemistry additions. Four findings, all acted
on in the same session:

- **A leakage path in the metal channel [исправлено].** `read_metals` read every HETATM metal, and on the
  benchmarks where ions are a large share of the sites -- about 40 % of LIGYSIS ligand sites -- such an atom can be
  the very thing the prediction is scored against. `exclude_comps` now removes the ligand codes being predicted,
  the featuriser passes them, and a test pins it. Our own manifest requires eight heavy atoms per ligand, so a
  monatomic ion was never one of *our* labels; the featuriser is also what runs on the external sets, where it was.
- **The featurisation did not reach inference [исправлено].** `predict.py`, `scripts/eval/evaluate.py` and the MCP
  server each passed `n_probe`, `n_surf` and `k_scale` and nothing else, so a checkpoint trained with any other
  featurisation was fed different arrays than it was trained on. For the width-changing flags that is a loud shape
  error. For `probe_sampling` it was silent and much worse: training placed probes by predicted ligandability
  while inference placed them at random within buriedness tiers, in the one component the ablation values at
  0.263 of site top-1. `data.featurisation_kwargs` is now the single carrier, a test asserts every width-changing
  flag is in it, and a config asking for learned placement with no point model supplied says so instead of falling
  back quietly.
- **`het_mp` was not heterogeneous message passing [исправлено].** The first version indexed its gain by the
  destination node type, which factors straight out of the attention-weighted sum, so VN-EGNN's number could not
  be attached to it. Now indexed by edge type. The caveat is recorded in the arm itself: the edge MLP already
  consumes a 16-dimensional embedding of the same nine types, so this may be another redundancy null, which is
  why `no_edge_type` -- the removal that asks whether type awareness matters at all -- was added beside it.
- **One confound left in place deliberately [принято].** `vec0` channel 2 carries Cα→Cβ for residues, zeros for
  surface points and, with `probe_electrostatic` on, the screened-Coulomb field for probes, and `vec_in` is shared
  across node types. So the field competes for the same projection weights as a backbone direction. This predates
  the change -- channel 0 is already N→Cα for residues and the direction to the nearest atom for probes -- and
  giving probes their own vector projection is a larger change than the arm it would serve. Recorded here so the
  arm is not read as a clean test of "electrostatics as a degree-1 input"; it is a test of that input *through a
  shared projection*.
