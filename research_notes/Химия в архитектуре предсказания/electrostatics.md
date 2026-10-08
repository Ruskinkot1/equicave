# Electrostatics in deep models for binding-site prediction, affinity prediction and pose scoring

**Scope note.** Current as of 2026-10-08. Every finding below is tagged **[PRIMARY]** when I read the
paper's own text (PDF text layer or publisher HTML) in this session, or **[SECOND-HAND]** when it comes
from a search engine's synthesis of a source I did not open directly. Two further tags are used
throughout because the assignment turns on the distinction:

- **[LISTED]** — the paper names this as an input channel/feature.
- **[MEASURED]** — the paper ran an experiment that isolates what this input contributes.

The single most load-bearing result in these notes is the MaSIF → dMaSIF pair, because it is the only
case I found where the *same task, same data, same authors' lineage* were run **with** a Poisson–Boltzmann
solve as an input feature and **without** it, with wall-clock numbers for both.

---

## PARTIAL CHARGES: which models use force-field / empirical charges as atom features, what ablations measured their contribution, and how charges are assigned at inference for an unparameterised protein

### Takeaway

A single 18/19-feature atom descriptor invented for Pafnucy (2018) — which contains exactly **one** float
partial-charge channel — propagated verbatim into Kalasanty and DeepSurf, and in modified form into
SwinSite (2026); so three of the best-known voxel site predictors do carry a partial charge, but **none of
Pafnucy, Kalasanty or DeepSurf ever measured what that channel contributes**. The other major family
(DeepSite, KDEEP, DeepPocket, gnina) carries **no partial charge at all** — only binary
`positive_ionizable`/`negative_ionizable` flags or bare atom types. Only SwinSite ran a per-channel
ablation, and its partial-charge number sits in supplementary material I could not retrieve.

### Cited Findings

**Pafnucy — the origin of the 18/19-feature descriptor (affinity, not site detection)**

- Pafnucy's atom descriptor is 19 features, quoted verbatim from the paper: "9 bits (one-hot or all null)
  encoding atom types: B, C, N, O, P, S, Se, halogen, and metal"; "1 integer (1, 2, or 3) with atom
  hybridization: hyb"; "1 integer counting the numbers of bonds with other heavyatoms: heavy valence";
  "1 integer counting the numbers of bonds with other heteroatoms: hetero valence"; "5 bits (1 if present)
  encoding properties defined with SMARTS patterns: hydrophobic, aromatic, acceptor, donor, and ring";
  "**1 float with partial charge: partialcharge**"; "1 integer (1 for ligand, -1 for protein) to distinguish
  between the two molecules: moltype" — [PRIMARY] [LISTED] — [Pafnucy, arXiv:1712.07042](https://arxiv.org/pdf/1712.07042)
- **How the charges were produced** (this is the clearest published answer to the inference-time question):
  "All complexes used in this study were protonated and charged using UCSF Chimera (Pettersen et al., 2004)
  with **Amber ff14SB for standard residues and AM1-BCC for non-standard residues and ligands**. No additional
  improvements nor calibration was performed on the complexes" — [PRIMARY] — [Pafnucy, arXiv:1712.07042](https://arxiv.org/pdf/1712.07042)
- "The partial charges were scaled by the training set's standard deviation in order to get a distribution
  with a unit standard deviation, which improves learning. In case of collisions (multiple atoms in a single
  grid point), which rarely occur for a 1-Å grid, features from all colliding atoms were added." — [PRIMARY] — [Pafnucy, arXiv:1712.07042](https://arxiv.org/pdf/1712.07042)
- "Atomic features were calculated using Open Babel (O'Boyle et al., 2011), and the complexes were
  transformed into grids." — [PRIMARY] — [Pafnucy, arXiv:1712.07042](https://arxiv.org/pdf/1712.07042)
- **What Pafnucy actually measured about charges.** Figure 3 is titled "Range of weights for each input
  channel (feature)". The channels in descending order of weight range are: `donor, aromatic, N,
  partialcharge, heterovalence, ring, halogen, S, O, acceptor, C, hyb, heavyvalence, metal, P, B, Se`,
  on an axis spanning roughly −0.04 to +0.04. `partialcharge` is **4th of 17**. This is an inspection of
  first-layer weight magnitudes, **not** a retrained ablation — [PRIMARY] [NOT an ablation] — [Pafnucy, arXiv:1712.07042](https://arxiv.org/pdf/1712.07042)
- Pafnucy's only perturbation experiment is spatial occlusion, not feature removal: "we produced 343
  corrupted complexes with some missing data and predicted the binding affinity for each. The missing data
  were produced by deleting a 5-Å cubic box from the original data. We slid the box with a 3-Å step (in
  every direction), thus yielding 7^3 = 343 corrupted inputs." — [PRIMARY] — [Pafnucy, arXiv:1712.07042](https://arxiv.org/pdf/1712.07042)
- Code: "Helper functions used to prepare the data and Jupyter Notebook with all preprocessing steps are
  available at http://gitlab.com/cheminfIBB/pafnucy" — [PRIMARY]; the paper does **not** state a software
  licence — [Pafnucy, arXiv:1712.07042](https://arxiv.org/pdf/1712.07042); repo at [gitlab.com/cheminfIBB/pafnucy](https://gitlab.com/cheminfIBB/pafnucy/-/tree/master)
- The Pafnucy README independently confirms the dependency: "Pafnucy uses protonation and partial charges
  to calculate features, so make sure that your files contain this information", and lists
  "partial charge (*partialcharge*, scaled by training set std)" among the input properties — [SECOND-HAND] — [gitlab.com/cheminfIBB/pafnucy](https://gitlab.com/cheminfIBB/pafnucy/-/tree/master)

**Kalasanty — inherits Pafnucy's descriptor, so it DOES carry a partial charge**

- Verbatim: "Finally, all the resulting protein structures and segmentations were represented with 3D grids
  with 2 Å resolution. The grids were centered on a protein center and had 70 Å in each direction.
  **Proteins were described with 18 atomic features used in our previous project [17].**" Reference [17] is
  Stepniewska-Dziubinska et al., "Development and evaluation of a deep learning model for protein–ligand
  binding affinity prediction" = **Pafnucy** — [PRIMARY] [LISTED] — [Kalasanty, arXiv:1904.06517](https://arxiv.org/pdf/1904.06517)
- Therefore Kalasanty's 18 channels = Pafnucy's 19 minus the ligand/protein `moltype` indicator, i.e. the
  `partialcharge` float is channel-for-channel present in Kalasanty. The Kalasanty paper never enumerates
  the 18 features itself — [INFERENCE from PRIMARY text]
- **Kalasanty ran no input-feature ablation of any kind.** Its experiments are 10-fold cross-validation for
  hyperparameter/stability assessment plus a held-out comparison with DeepSite on the Chen benchmark
  (149 structures, 269 binding sites) — [PRIMARY] [NOT MEASURED] — [Kalasanty, arXiv:1904.06517](https://arxiv.org/pdf/1904.06517)
- Architecture/grid: U-Net, 9 convolutional blocks (4 encoder, 1 bottleneck, 4 decoder), filters
  32/64/128/256/512, kernel 3×3×3; "for input of 36x36x36 pixels that was used in this work, feature maps
  in the middle of the network (bottleneck) have spacial sizes of 1x1x1 and can be used as feature vectors."
  — [PRIMARY] — [Kalasanty, arXiv:1904.06517](https://arxiv.org/pdf/1904.06517)
- Code and weights: "Source code and network's parameters are freely available at
  http://gitlab.com/cheminfIBB/kalasanty." The paper does **not** state a licence — [PRIMARY] — [Kalasanty, arXiv:1904.06517](https://arxiv.org/pdf/1904.06517)
- Inference-time structure preparation described by Kalasanty for the Chen benchmark: "for each structure
  ligand(s) and protein were split into separate files using UCSF Chimera [24]. Solvent and ions were
  assigned to the protein. Then, we used VolSite [22] (available in IChem toolkit) to describe a cavity for
  each ligand." **No charging protocol is stated** for the test structures — [PRIMARY] — [Kalasanty, arXiv:1904.06517](https://arxiv.org/pdf/1904.06517)
- Robustness note relevant to unparameterised inputs: "For 79 (0.5%) entries in the sc-PDB database this
  procedure lead to empty pocket grids. After manually inspecting several of such cases it turned out that
  it affects large protein complexes and less carefully prepared protein structures" — [PRIMARY] — [Kalasanty, arXiv:1904.06517](https://arxiv.org/pdf/1904.06517)

**DeepSurf — same 18 features again**

- Verbatim: the featurization is "the … scheme initially introduced by [17] and used also in Kalasanty [23],
  which consists of **18 chemical features calculated per protein atom**. Each grid voxel receives the
  features of the atoms inside it." [17] = Pafnucy, [23] = Kalasanty — [PRIMARY] [LISTED] — [DeepSurf, arXiv:2002.05643](https://arxiv.org/pdf/2002.05643)
- Protonation is a hard requirement: "Prior to importing in DeepSurf, proteins should also be suitably
  pre-processed. Specifically, water, ions and ligands are removed from the PDB structures, and the
  remaining structure is **protonated, if needed, enabling the proper computation of the necessary input
  features for the 3D-CNN [17]**. Before the final step of binding sites extraction …, hydrogen atoms are
  removed from the protein in order binding sites to maintain only heavy atoms." — [PRIMARY] — [DeepSurf, arXiv:2002.05643](https://arxiv.org/pdf/2002.05643)
- **DeepSurf's ablations are entirely architectural**, not chemical: ResNet vs LDS-ResNet variants, bottleneck
  vs non-bottleneck, depth, and the surface representation. Example reported number: "DeepSurf achieves a
  higher top-n prediction score of 68.1%" for the larger architecture; "although bottleneck LDS-ResNet-18 has
  more than 10 times fewer parameters, it achieves similar, if not better, CV performance than its
  competitor." No feature-channel ablation — [PRIMARY] [NOT MEASURED] — [DeepSurf, arXiv:2002.05643](https://arxiv.org/pdf/2002.05643)
- Surface machinery: solvent accessible surface from DMS, mesh simplification with factor f = 10 raising
  "the average minimum distance of the remaining surface points to 2.3 Å"; sliding 16×16×16 cuboid grids;
  ligandability threshold T = 0.9 — [PRIMARY] — [DeepSurf, arXiv:2002.05643](https://arxiv.org/pdf/2002.05643)
- Code and weights: "The source code of the method along with **trained models** are available at
  https://github.com/stemylonas/DeepSurf.git". Licence not stated in the paper — [PRIMARY] — [DeepSurf, arXiv:2002.05643](https://arxiv.org/pdf/2002.05643)

**SwinSite (JCIM 2026) — the only site model with a per-channel ablation**

- 18-dimensional per-heavy-atom vector including atom type, hybridization, heavy and heteroatom neighbour
  counts, "**estimated partial charge**", and functional-group features such as aromaticity and
  hydrogen-bonding potential. The full list is in Table S1 of the Supporting Information, which I could not
  retrieve — [PRIMARY of main text] [LISTED] — [SwinSite, PMC12977039](https://pmc.ncbi.nlm.nih.gov/articles/PMC12977039/)
- The paper does **not** say how the partial charge is computed — it says only that it is "estimated"
  — [PRIMARY] — [SwinSite, PMC12977039](https://pmc.ncbi.nlm.nih.gov/articles/PMC12977039/)
- Ablation protocol: each input channel independently masked to zero at inference with model parameters
  fixed — [PRIMARY] [MEASURED, but see caveat] — [SwinSite, PMC12977039](https://pmc.ncbi.nlm.nih.gov/articles/PMC12977039/)
- Main-text numbers (Figure 4a; Top-1 DCC success, averaged over five benchmarks): `atom:O` 39.7% drop,
  `atom:C` 25.45%, `atom:N` 14.88%, `atom:S` 10.35%, `Ring` 10.43%; heterodegree, H-bond donor, aromatic,
  H-bond acceptor and hydrophobic roughly 15–23% each; hybridization and heavy-degree masking produced zero
  success rates and low recall on all benchmarks. **The partial-charge result is not in the main text** —
  the group-wise ablation that includes charge appears only in the Supporting Information — [PRIMARY] — [SwinSite, PMC12977039](https://pmc.ncbi.nlm.nih.gov/articles/PMC12977039/)
- Authors' own summary of the ablation: "chemical and atom-type features also substantially influenced
  prediction accuracy, **although to a lesser extent than geometric cues**" — [PRIMARY] — [SwinSite, PMC12977039](https://pmc.ncbi.nlm.nih.gov/articles/PMC12977039/)
- Cost: inference ≈ 1.7 s per protein with the 4-fold ensemble, ≈ 1.4 s for the CNN-only variant
  (SwinSite-CNN); training ≈ 4 days per fold with four folds in parallel on four GPUs. Per-protein
  preprocessing time is **not** reported — [PRIMARY] — [SwinSite, PMC12977039](https://pmc.ncbi.nlm.nih.gov/articles/PMC12977039/)
- Code at github.com/ding-oh/SwinSite; trained weights not mentioned; no software licence stated (the
  CC-BY-NC-ND 4.0 notice covers the article) — [PRIMARY] — [SwinSite, PMC12977039](https://pmc.ncbi.nlm.nih.gov/articles/PMC12977039/)
- **Source conflict to flag:** two indexed versions of the SwinSite paper report different Kalasanty
  baseline numbers on the same benchmarks — one table gives 36.29 / 26.96 / 38.01 / 31.75 / 32.35 success
  rates, another 56.45 / 53.91 / 61.99 / 52.33 / 54.39. Anyone citing SwinSite's comparison table must
  check which version — [SECOND-HAND, flagged as a discrepancy] — [SwinSite, ACS](https://pubs.acs.org/jcisd8/article/66/5/2551/5080425/SwinSite-3D-Structure-Based-Prediction-of-Protein) / [SwinSite, PMC12977039](https://pmc.ncbi.nlm.nih.gov/articles/PMC12977039/)

**The other lineage (DeepSite / KDEEP / DeepPocket / gnina): no partial charges at all**

- moleculekit's `getVoxelDescriptors` — the library behind DeepSite and KDEEP — defaults to eight channels:
  "If no channels are given, the default ('hydrophobic', 'aromatic', 'hbond_acceptor', 'hbond_donor',
  '**positive_ionizable**', '**negative_ionizable**', 'metal', 'occupancies') channels will be used."
  These are binary pharmacophoric flags, i.e. a **formal-charge / ionizability proxy, not a partial charge
  and not an electrostatic potential** — [PRIMARY of docs] [LISTED] — [moleculekit voxeldescriptors docs](https://software.acellera.com/moleculekit/moleculekit.tools.voxeldescriptors.html)
- The same docs present these as the inputs "used in KDeep, DeepSite and more" — [PRIMARY of docs] — [moleculekit voxelization tutorial](https://software.acellera.com/moleculekit/tutorials/voxelization_tutorial.html)
- A user-supplied `userchannels` array of shape (numAtoms, nchannels) is supported, so charges *could* be
  injected — but no published site model in this family does so — [PRIMARY of docs] — [moleculekit voxeldescriptors docs](https://software.acellera.com/moleculekit/moleculekit.tools.voxeldescriptors.html)
- The eight-feature set descends from AutoDock 4 atom typing: a contemporaneous account of the HTMD/KDEEP
  workflow states the parameters come from AutoDock 4, so "the performance of the predictive models using
  the voxel features depends on performance of Autodock4" — [SECOND-HAND, ~9-year-old blog, treat as
  historical context] — [iwatobipen blog](https://iwatobipen.wordpress.com/2018/02/03/get-3d-feature-descriptors-from-pdb-file/)
- DeepPocket: 14 atom-type channels, 48×48×48 bounding box for the classifier; segmentation re-voxelizes at
  0.5 Å with 14 channels on a 65×65×65 grid; preprocessing goes through libmolgrid "gninatypes" and
  "molcache2" binaries. **No charge or electrostatic channel** — [SECOND-HAND] — [DeepPocket, JCIM](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00799) / [DeepPocket repo](https://github.com/devalab/DeepPocket/blob/main/README.md)
- gnina: CNN scoring uses one grid channel per atom type; the original default used smina atom types,
  "34 distinct types with 16 receptor types and 18 ligand types" (a later paper lists 35). Types depend on
  element, aromaticity, adjacency to polar atoms and hydrogen-bonding propensity — not on charge
  — [SECOND-HAND] — [gnina/libmolgrid, JCIM](https://pubs.acs.org/doi/10.1021/acs.jcim.9b01145) / [GNINA 1.3, PMC11874439](https://pmc.ncbi.nlm.nih.gov/articles/PMC11874439/)
- libmolgrid itself "supports atom typing according to XS atom typing, atomic element, or a user-provided
  callback function", and types "may be represented by a single integer or a vector encoding" — so a charge
  channel is reachable but unused in the published models — [SECOND-HAND] — [libmolgrid, arXiv:1912.04822](https://arxiv.org/pdf/1912.04822)
- Vina-style scoring, which gnina's empirical term inherits, **has no electrostatic term**; its
  electrostatic-like contribution enters only through the hydrogen-bond term — [SECOND-HAND] — [gnina docking workshop notes](https://gnina.github.io/gnina/rsc_workshop2021/)

**Graph models: formal charge, not partial charge**

- GrASP's atom-scale features: "Atom Type, Local Density of Atoms (9 features, density within spheres
  ranging from 2 - 10 Å), Solvent Accessible Surface Area, **Formal Charge**", plus ring membership,
  aromaticity, mass, hybridization, H-bond donor/acceptor status and hydrophobicity; residue features
  include residue name; edges carry inverse distance and bond order, connecting heavy atoms within 5 Å.
  **Partial charge is not mentioned anywhere in the paper** — [PRIMARY of main text] [LISTED] — [GrASP, PMC10402091](https://pmc.ncbi.nlm.nih.gov/articles/PMC10402091/)
- GrASP reports **no feature ablation or feature-importance analysis**, for charge or anything else; results
  compare GrASP with P2Rank at the metric level. Runtime and preprocessing cost are not reported
  — [PRIMARY] [NOT MEASURED] — [GrASP, PMC10402091](https://pmc.ncbi.nlm.nih.gov/articles/PMC10402091/)
- GrASP code and data: "The GrASP model as well as all training and test data is available at"
  github.com/tiwarylab/GrASP. The paper names no software licence (the CC-BY-4.0 notice covers the article).
  The repo's featurizer is `featurize_protein.py` and the README names **MDAnalysis and OpenBabel 2.4.1**,
  not moleculekit — [PRIMARY of paper, SECOND-HAND of README] — [GrASP, PMC10402091](https://pmc.ncbi.nlm.nih.gov/articles/PMC10402091/) / [GrASP repo](https://github.com/tiwarylab/GrASP)
- **Name collision worth flagging:** "GRaSP" (Santana et al. 2020) and GRaSP-web are a *different*,
  residue-level extremely-randomized-trees method using "solvent relative accessibility, atom types,
  interaction level" from an atomic graph. Results do not transfer between GrASP and GRaSP — [SECOND-HAND] — [GRaSP-web, NAR](https://academic.oup.com/nar/article/50/W1/W392/6582175)
- VN-EGNN uses residue nodes at α-carbon positions with **pre-trained ESM-2 embeddings** as initial
  features; virtual-node features are "derived … by averaging the residue node features across the entire
  protein". Section 2.2 mentions only "e.g., the atom or residue type" as a generic option. **No charge,
  no electrostatics** — [PRIMARY] — [VN-EGNN, arXiv:2404.07194](https://arxiv.org/html/2404.07194)
- An independent comparative evaluation lists VN-EGNN's features simply as "ESM-2 embeddings | 1280"
  — [SECOND-HAND] — [Comparative evaluation, J Cheminform](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00923-z)
- **EquiPocket: arXiv:2302.12177 has been WITHDRAWN and no PDF is served.** The abstract page mentions only
  that one module models "both the chemical and spatial structure of protein", with no feature list, no
  mention of charges or electrostatic potential, and no ablation. Treat EquiPocket as uncitable for feature
  claims — [PRIMARY of the abstract page] — [EquiPocket, arXiv:2302.12177](https://arxiv.org/abs/2302.12177)
- P2Rank (the strongest classical site ranker, and the usual baseline) projects "distance weighted
  properties of nearby protein atoms onto SAS points (6 Å neighbourhood is considered, w(d) = 1 - d/6)" into
  a **35-feature** vector of residue-level and atomic-level properties — physico-chemical properties of
  standard amino acids, hydropathy index, VolSite pharmacophore labels, statistical ligand-binding
  propensities, aromaticity, protrusion. **Neither partial charge nor electronegativity appears in any
  P2Rank description I could find**, and "the single most important feature turned out to be a geometric
  feature termed protrusion" — [SECOND-HAND] — [P2Rank, J Cheminform](https://link.springer.com/article/10.1186/s13321-018-0285-8) / [P2Rank repo](https://github.com/rdk/p2rank)
- CAT-Site uses eight property channels of 16×16×16, each "a chemical property derived from atom types set
  by the openbabel library" — i.e. the same type-derived rather than charge-derived scheme — [SECOND-HAND] — [CAT-Site, Pharmaceutics](https://doi.org/10.3390/pharmaceutics15010119)

**Charge assignment at inference for an unparameterised protein — what is actually published**

- Pafnucy's protocol is the one explicit, reproducible recipe: UCSF Chimera, **Amber ff14SB for standard
  residues, AM1-BCC for non-standard residues and ligands** — [PRIMARY] — [Pafnucy, arXiv:1712.07042](https://arxiv.org/pdf/1712.07042)
- MaSIF's protocol: protonation with **Reduce**; "**PDB2PQR** was used to prepare protein files for
  electrostatic calculations and **APBS (v.1.5)** was used to compute Poisson–Boltzmann electrostatics for
  each protein. The corresponding charge at each vertex of the meshed surface was assigned using
  **Multivalue**, provided within the APBS suite. Charge values above +30 and below −30 were capped at those
  values and then values were normalized between −1 and 1." — [PRIMARY] — [MaSIF, Nat Methods 2020 (PDF)](https://www-cbi.cs.uni-saarland.de/wp-content/uploads/2025/03/MaSIF-paper_SS2025.pdf)
- APBS's own documented prerequisite: "Before running APBS, a structure should be prepared by
  reconstructing missing heavy atoms, adding hydrogens, and assigning atomic charges and radii."
  — [SECOND-HAND] — [APBS calculation input docs](https://ics.uci.edu/~dock/manuals/apbs/html/tutorial/x78.html)
- EspalomaCharge is the current ML route around AM1-BCC's cost for ligands and non-standard residues: a GNN
  for "ultrafast partial charge assignment", motivated by partial charges being "central to the
  electrostatic contributions to intermolecular energies" — [SECOND-HAND] — [EspalomaCharge, arXiv:2302.06758](https://arxiv.org/pdf/2302.06758) / [EspalomaCharge, PMC11129294](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11129294/)
- A 2026 GNN charge model aimed at "accurate electrostatic properties of organic molecules" is a further
  entry in the same category — [SECOND-HAND] — [GNN charge model, PMC12834533](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12834533/)

### Inferences

- The field has effectively **two featurization monocultures**. One (Pafnucy → Kalasanty → DeepSurf, and
  SwinSite's variant) ships a partial charge and therefore inherits a hard dependency on a protonation +
  charging pipeline. The other (DeepSite/KDEEP/moleculekit, DeepPocket/gnina/libmolgrid) ships only atom
  types and ionizability flags and therefore has no such dependency. Because these two families reach
  broadly comparable site-detection numbers, the existence of the charge channel is **not** what separates
  good from bad site predictors.
- The partial-charge channel in Kalasanty and DeepSurf is best described as **inherited, unexamined
  baggage**. It arrived because the same lab reused its affinity descriptor for a segmentation task, and no
  paper in the chain ever tested whether removing it changes anything.
- Pafnucy's Figure 3 weight-range ranking is the only quantitative signal that the charge channel is used at
  all (4th of 17). It is weak evidence: a large first-layer weight range can reflect input scaling (charges
  were std-normalised; most other channels are 0/1 bits or small integers) rather than predictive value.
- SwinSite's zero-masking protocol is a **biased estimator of channel importance**: zeroing one channel of a
  trained network at inference creates an out-of-distribution input, so the reported "drops" bound
  importance from above. The fact that masking `hybridization` or `heavy-degree` drives success to *zero*
  is a symptom of that pathology, not a claim that hybridization is more informative than oxygen position.
- For anyone running Kalasanty or DeepSurf on a raw apo PDB with missing residues, non-standard residues or
  unparameterised cofactors, the charge channel is the fragile part of the pipeline and the papers give no
  guidance. Pafnucy's Chimera/ff14SB/AM1-BCC recipe is the only published fallback, and AM1-BCC on a large
  non-standard residue is itself a semi-empirical QM step.

### Gaps

- **SwinSite's partial-charge ablation number.** It exists (the paper says the group-wise ablation including
  charge is in the SI) but the Supporting Information was not retrievable in this session. This is the single
  most directly relevant missing number in the whole assignment.
- No retrained leave-one-feature-out ablation of the `partialcharge` channel exists for Pafnucy, Kalasanty
  or DeepSurf. I found none, and the papers state none.
- KDEEP's eight channels are described second-hand as "pharmacophoric-like properties"; I did not open the
  KDEEP paper itself to confirm they are exactly moleculekit's eight defaults.
- The exact 14 atom types in DeepPocket's channels are in its Supporting Information, which I did not read.
- Software **licences** are the weakest part of the evidence: Pafnucy, Kalasanty, DeepSurf, GrASP, SwinSite
  and VN-EGNN all publish repository URLs, and Kalasanty and DeepSurf explicitly publish trained weights,
  but **none of the papers states a software licence**, and I did not verify the licence files in the repos.
  Treat all licence questions as unresolved.
- A bioRxiv scaffold-aware affinity model reports that "partial charge is correlated to the electrostatic
  potential energy and its contribution is slightly higher than the rest of features on CASF-2016 but shows
  no difference on CSAR-HiQ" — [SECOND-HAND], a feature-importance analysis on a different architecture, not
  an ablation, and I could not verify the numbers — [Leveraging Scaffold Information, bioRxiv](https://www.biorxiv.org/content/10.1101/2022.08.19.504617v1.full.pdf)

---

## CONTINUUM ELECTROSTATICS: Poisson–Boltzmann and generalised Born potentials as 3D/surface inputs — which models, what they measured, and the real cost

### Takeaway

**MaSIF is the canonical case and essentially the only one**: it feeds an APBS Poisson–Boltzmann potential,
sampled at every surface vertex, as one of five channels, and it ran a genuine retrained feature-subset
ablation. Its successor dMaSIF then **removed the PB solve entirely**, replaced it with a learned function
of atom type and inverse distance, and matched then exceeded MaSIF's site ROC-AUC while running the whole
preprocessing chain ~1000× faster. Published wall-clock numbers show MaSIF's feature step (19.7 s/protein,
the step containing the PB solve) costs **~120× its own network forward pass** and **~550× dMaSIF's**.
I found **no** binding-site or affinity network that uses a generalised Born potential as an input grid.

### Cited Findings

**MaSIF — the PB-as-input reference implementation**

- Feature set, verbatim: "For each vertex within the patch, we compute two geometric features (shape index
  and distance-dependent curvature) and **three chemical features (hydropathy index, continuum
  electrostatics and the location of free electrons and proton donors)**" — [PRIMARY] [LISTED] — [MaSIF, Nat Methods 2020 (PDF)](https://www-cbi.cs.uni-saarland.de/wp-content/uploads/2025/03/MaSIF-paper_SS2025.pdf)
- Figure 1b labels these exactly as: geometric = "Shape index", "Distance-dependent curvature"; chemical =
  "Hydropathy", "**Continuum electrostatics**", "Free electrons/protons" — [PRIMARY] — [MaSIF, Nat Methods 2020 (PDF)](https://www-cbi.cs.uni-saarland.de/wp-content/uploads/2025/03/MaSIF-paper_SS2025.pdf)
- PB pipeline, verbatim (Methods, "Poisson–Boltzmann continuum electrostatics"): "PDB2PQR was used to
  prepare protein files for electrostatic calculations and APBS (v.1.5) was used to compute
  Poisson–Boltzmann electrostatics for each protein. The corresponding charge at each vertex of the meshed
  surface was assigned using Multivalue, provided within the APBS suite. Charge values above +30 and below
  −30 were capped at those values and then values were normalized between −1 and 1." — [PRIMARY] — [MaSIF, Nat Methods 2020 (PDF)](https://www-cbi.cs.uni-saarland.de/wp-content/uploads/2025/03/MaSIF-paper_SS2025.pdf)
- Surface pipeline: "All proteins in the datasets were protonated using Reduce, and triangulated using the
  MSMS program with a density of 3.0 and a water probe radius of 1.5 Å. Protein meshes were then
  downsampled and regularized to a resolution of 1.0 Å using pymesh." — [PRIMARY] — [MaSIF, Nat Methods 2020 (PDF)](https://www-cbi.cs.uni-saarland.de/wp-content/uploads/2025/03/MaSIF-paper_SS2025.pdf)
- Patch radius r = 9 Å or 12 Å, application-dependent; each of the five feature types "was run through a
  separate neural network channel", with a learned soft grid of 16 angular bins and 5 radial bins followed
  by a convolutional layer with 80 filters and angular max pooling — [PRIMARY] — [MaSIF, Nat Methods 2020 (PDF)](https://www-cbi.cs.uni-saarland.de/wp-content/uploads/2025/03/MaSIF-paper_SS2025.pdf)
- **MaSIF-site feature-subset ablation (Fig. 3d) — a real retrained ablation.** Caption verbatim: "ROC AUC
  scores for ablation studies with networks trained with different subsets of features: only geometric
  (Geom), only the location of free electrons/proton donors (hbond), **Poisson–Boltzmann electrostatics
  (elec)**, the hydropathy index (hpathy) and all features (G+C) (surface points, no. of positives = 218,246,
  no. of negatives = 1,973,624)" — [PRIMARY] [MEASURED] — [MaSIF, Nat Methods 2020 (PDF)](https://www-cbi.cs.uni-saarland.de/wp-content/uploads/2025/03/MaSIF-paper_SS2025.pdf)
- The five bar values recoverable from the figure's text layer are **0.68, 0.75, 0.75, 0.78, 0.80 ROC AUC**.
  The full-feature bar (G+C) is the highest at 0.80; the lowest, 0.68, corresponds to the single weakest
  subset (hbond). **I could not reliably assign each remaining value to its specific bar** from the PDF text
  layer — the column positions are ambiguous. So: all five feature subsets individually land in the
  0.68–0.78 band, and all features together give 0.80 — [PRIMARY, with explicit bar-assignment
  uncertainty] — [MaSIF, Nat Methods 2020 (PDF)](https://www-cbi.cs.uni-saarland.de/wp-content/uploads/2025/03/MaSIF-paper_SS2025.pdf)
- A search engine's reading of the same figure asserted that "the electrostatics result appears to be the
  strongest single feature" with all-features at ROC AUC = 0.80. **I was unable to confirm the first half of
  that claim and explicitly do not endorse it** — [SECOND-HAND, unconfirmed, contradicted by my own
  inability to resolve the bars] — [search synthesis over MaSIF sources](https://github.com/pablogainza/masif_paper/blob/master/README.md)
- MaSIF-site per-protein median ROC AUCs reported elsewhere in Fig. 3e–g: 0.87, 0.89, 0.81, 0.81 across
  protein subsets (All / Transient / Large-hydrophobic / Small-hydrophobic), with an illustrative example at
  ROC AUC 0.84, and 0.81 vs 0.65 vs 0.62 in the comparison with SPPIDER and PSIVER on 53 single-chain
  transient interactions — [PRIMARY] — [MaSIF, Nat Methods 2020 (PDF)](https://www-cbi.cs.uni-saarland.de/wp-content/uploads/2025/03/MaSIF-paper_SS2025.pdf)
- MaSIF-ligand (pocket classification into 7 cofactor classes) also has a chemistry-vs-geometry comparison:
  Fig. 2c reports "balanced accuracy of the prediction of the specificity of binding sites using all features
  (G+C, geometry and chemistry), only geometric features (Geom) or only chemical features (Chem)". **The
  numeric values were not recoverable from the text layer** — [PRIMARY of caption, values missing] — [MaSIF, Nat Methods 2020 (PDF)](https://www-cbi.cs.uni-saarland.de/wp-content/uploads/2025/03/MaSIF-paper_SS2025.pdf)
- MaSIF-search's feature comparison (Fig. 5c) similarly reports ROC AUC for GIF descriptors vs Geom vs Chem
  vs G+C on 13,338 positive and 13,338 negative patch pairs — values not recoverable — [PRIMARY of caption] — [MaSIF, Nat Methods 2020 (PDF)](https://www-cbi.cs.uni-saarland.de/wp-content/uploads/2025/03/MaSIF-paper_SS2025.pdf)
- Code: github.com/LPDI-EPFL/masif. The README pins **PDB2PQR 2.1.1, multivalue and APBS 1.5** and states
  these "are necessary to compute electrostatics charges". A later fork, MaSIF-neosurf, moved to PDB2PQR
  3.5.2 — [SECOND-HAND] — [masif repo](https://github.com/LPDI-EPFL/masif) / [masif-neosurf](https://github.com/arcimboldo-team/masif-neosurf)

**dMaSIF — the controlled removal of the PB solve, with wall-clock numbers**

- Framing, verbatim: "MaSIF tackles this problem as a surface segmentation problem. The binding site (red)
  is the ground truth signal that MaSIF tries to predict from precomputed chemical and geometric features,
  **such as the electrostatic potential**. … Our method predicts the binding site **without using any
  precomputed mesh structure or features**." — [PRIMARY] — [dMaSIF, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Sverrisson_Fast_End-to-End_Learning_on_Protein_Surfaces_CVPR_2021_paper.pdf)
- dMaSIF's replacement for the PB channel, verbatim: "For each point x_i, we then find the 16 nearest atom
  centers {a_i1, …, a_i16} with types {t_i1, …, t_i16} encoded as one-hot vectors in R^6. We compute a
  vector of chemical features f_i in R^6 by applying a Multi-Layer Perceptron (MLP) to the vectors
  **[t_ik, 1/‖x_i − a_ik‖] in R^7**, performing a summation over the indices k = 1, …, 16 and applying a
  second MLP to the result. … using simple MLPs with a single hidden layer of dimension 12 is enough to
  learn rich chemical features, **such as the Poisson-Boltzmann electrostatic potential**." — [PRIMARY] — [dMaSIF, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Sverrisson_Fast_End-to-End_Learning_on_Protein_Surfaces_CVPR_2021_paper.pdf)
- Full input vector: 10 geometric (mean and Gaussian curvatures "at 5 scales σ ranging from 1 Å to 10 Å")
  + 6 learned chemical = 16 dimensions; surfaces sampled at 1 Å giving N = 6K–15K points for proteins of
  A = 3K–15K atoms — [PRIMARY] — [dMaSIF, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Sverrisson_Fast_End-to-End_Learning_on_Protein_Surfaces_CVPR_2021_paper.pdf)
- **Table 1, "Average 'pre-processing' time per protein"** — the actual cost of the PB-bearing pipeline:

  | Computation | MaSIF | dMaSIF |
  |---|---|---|
  | Surface generation | 6.11 ± 6.18 s | 59.0 ± 15.2 ms* |
  | **Input features (contains the APBS PB solve)** | **19.69 ± 16.08 s** | 6.59 ± 1.22 ms* |
  | Local coordinates | 50.65 ± 45.15 s | 0.46 ± 0.09 ms* |

  (*with batches of 128 proteins at a time). Caption: "Our method is about **1000 times faster** than MaSIF
  and allows these computations to be performed on the fly, as opposed to the offline precomputations of
  MaSIF." — [PRIMARY] — [dMaSIF, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Sverrisson_Fast_End-to-End_Learning_on_Protein_Surfaces_CVPR_2021_paper.pdf)
- Storage cost of the precomputation: "the pre-processed files used to train the MaSIF networks weigh
  **more than 1TB**" — [PRIMARY] — [dMaSIF, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Sverrisson_Fast_End-to-End_Learning_on_Protein_Surfaces_CVPR_2021_paper.pdf)
- **Network forward-pass cost, the denominator for the comparison**: "if we use a single convolution layer
  with a Gaussian window of deviation σ = 15 Å, our method matches the best accuracy of **0.85 ROC-AUC**
  produced by MaSIF — with 3 successive convolutional layers on patches of radius 9 Å. In this
  configuration, our network runs 10 times faster than MaSIF with an average time in the forward pass of
  **16 ms vs. 164 ms per protein**. At the price of a modest increase of the model complexity (three
  convolution layers, and **36 ms** on average per protein), **we outperform MaSIF with a 0.87 ROC-AUC**."
  — [PRIMARY] [MEASURED] — [dMaSIF, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Sverrisson_Fast_End-to-End_Learning_on_Protein_Surfaces_CVPR_2021_paper.pdf)
- **PB-recovery experiment (Fig. 6a):** "we show in Figure 6 the results of an experiment where our chemical
  feature extractor is used to regress the Poisson-Boltzmann electrostatic potential on surface points."
  Reported: "**Correlation cofactor r=0.83 and RMSE=0.16**". Authors' conclusion: "The quality of our
  prediction suggests that our data-driven chemical features are of similar quality to the descriptors used
  by MaSIF – or better." — [PRIMARY] [MEASURED] — [dMaSIF, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Sverrisson_Fast_End-to-End_Learning_on_Protein_Surfaces_CVPR_2021_paper.pdf)
- **dMaSIF's own chemical-vs-geometric ablation (Fig. 6b):** "We also note the results of an ablation study
  for chemical and geometric features, depicted in Figure 6. They suggest that **the concatenation of
  geometric curvatures to the vector of learned chemical features does not significantly improve the
  performance of the network for the site prediction task**: we will investigate this point in future works."
  The per-bar numbers are in the figure, not the text — [PRIMARY of text, values missing] [MEASURED] — [dMaSIF, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Sverrisson_Fast_End-to-End_Learning_on_Protein_Surfaces_CVPR_2021_paper.pdf)
- Dataset and hardware: site identification uses 2958 train / 356 test proteins with a sequence-and-structure
  similarity split; interaction prediction 4614 / 912 complexes; "All models are trained on either a single
  NVIDIA GeForce RTX 2080 Ti GPU or a single Tesla V100. Run times and memory consumption are measured on a
  single Tesla V100." Point counts: N = 11549 ± 1853 for dMaSIF's clouds vs 6321 ± 1028 for MaSIF's
  — [PRIMARY] — [dMaSIF, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Sverrisson_Fast_End-to-End_Learning_on_Protein_Surfaces_CVPR_2021_paper.pdf)
- Independent corroboration: ProteinMAE states that "Sverrisson et al. (2021) demonstrated that chemical
  properties such as Poisson–Boltzmann electrostatics can be learned from raw chemical features like atom
  type distribution", and describes dMaSIF as achieving "competitive performance without complex
  preprocessing" — [SECOND-HAND] — [ProteinMAE, Bioinformatics](https://academic.oup.com/bioinformatics/article/39/12/btad724/7455256)

**DeepDrug3D — knowledge-based energy grids, NOT Coulomb/AMBER (a correction)**

- DeepDrug3D voxelizes a **spherical grid of radius 15 Å at 1 Å spacing** centred on the ligand centroid,
  removes points within 2 Å of any protein atom, outside the convex hull, or disconnected, then discretizes
  to **32×32×32 voxels with 14 channels** of interaction energies for 14 SYBYL atom types (C, N, O, P, S, F
  variants) — [SECOND-HAND] [LISTED] — [DeepDrug3D, PLoS Comput Biol](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006718)
- The energies are **DFIRE** (distance-scale finite ideal-gas reference) knowledge-based statistical
  potentials, computed by a Fortran module, Linux-only. **No AMBER parameters and no Coulomb term appear
  anywhere in the method** — my initial hypothesis that DeepDrug3D uses a Coulomb/vdW field was wrong
  — [SECOND-HAND, explicit correction] — [DeepDrug3D, PLoS Comput Biol](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006718) / [DeepDrug3D repo](https://github.com/pulimeng/DeepDrug3D)
- **Shape-only ablation (relevant by analogy):** replacing the 14 interaction-potential channels with a
  binary occupancy (2 channels) gave 0.824 for nucleotide-binding and 0.952 for heme-binding pockets, against
  a reported ~0.95 accuracy for the full 14-channel model. So the energy channels buy roughly **+0.13 on
  nucleotide pockets and ~0 on heme pockets** — [SECOND-HAND] [MEASURED] — [DeepDrug3D, PLoS Comput Biol](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006718)

**Other PB-adjacent ML work (none of it feeding a site/affinity model)**

- DIS-PB: a GNN trained on the **difference between all-atom MD and PB energies**, with the APBS PB
  electrostatic potential as the prior. Targets solvation and DNA, not binding sites — [SECOND-HAND] — [Achieving all-atom MD accuracy from PB through ML, J Chem Phys](https://pubs.aip.org/aip/jcp/article/164/5/054107/3378487/Achieving-all-atom-molecular-dynamics-accuracy)
- XPPBE, a physics-informed neural network that **solves** the PB equation grid-free; on full proteins the
  difference from a reference boundary-element solution was "of the order of 10^-2 in both solvation free
  energy and surface potential". This is a PB *surrogate*, not a feature for a structural model
  — [SECOND-HAND] — [PINNs for the PB equation, arXiv:2410.12810](https://arxiv.org/html/2410.12810v2)
- A 2025 voxel-based local site-detection method lists "hydropathy, hydrogen bond potential, **electrostatics**
  and atom types" as its chemical features. I could not open it to determine whether "electrostatics" means a
  PB solve, a Coulomb field or an ionizability flag, nor whether it was ablated — [SECOND-HAND, unresolved] — [Voxel-Based Deep Learning Method for Local Detection of Protein-Ligand Binding Sites](https://doi.org/10.1145/3807503.3819493)
- A review notes a structure-informed model using "van der Waals, hydrogen bonding, and electrostatic terms,
  built into a molecular graph framework" — unnamed in the retrieved text, so not actionable — [SECOND-HAND] — [Structure-informed ML for drug discovery, Brief Bioinform](https://academic.oup.com/bib/article-pdf/27/1/bbag081/67075818/bbag081.pdf)

**The real cost of a PB solve**

- The **best empirical number in a DL context is MaSIF's own**: the "Input features" step, which contains
  PDB2PQR + the APBS 1.5 solve + Multivalue surface interpolation + hydropathy + H-bond assignment, costs
  **19.69 ± 16.08 s per protein** — [PRIMARY] — [dMaSIF, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Sverrisson_Fast_End-to-End_Learning_on_Protein_Surfaces_CVPR_2021_paper.pdf)
- Scaling: an APBS `mg-auto` benchmark found that "the memory cost grows linearly and the time to solution
  grows **quadratically** with respect to the grid size" — [SECOND-HAND] — [Bempp-Exafmm, arXiv:2103.01048](https://arxiv.org/pdf/2103.01048)
- A large-system anchor: the PyMOL APBS plugin wiki states that a virus assembly (PDB 3j7l) at **2.0 Å**
  spacing "takes about 20 minutes" — much larger system, much coarser grid than a 300-residue protein, so not
  directly comparable — [SECOND-HAND] — [APBS Electrostatics Plugin, PyMOL Wiki](https://pymolwiki.org/APBS_Electrostatics_Plugin)
- Grid guidance: "Grid spacings of 0.5 Å or smaller are recommended for quantitative calculations", the grid
  centre "should coincide with the region of interest, such as a binding site", and grid lengths must be
  large enough for the boundary conditions to hold — [SECOND-HAND] — [APBS calculation input docs](https://ics.uci.edu/~dock/manuals/apbs/html/tutorial/x78.html)
- **Accuracy caveat on the standard grid:** a JCTC study raised the concern that "the widely used grid
  spacing of 0.5 Å may not give reliable binding free energies with APBS, DelPhi, and PBSA", and that no
  confirmation exists for its reliable use in binding-energy calculations. Anyone using PB-derived grids as
  features should test grid-spacing sensitivity first — [SECOND-HAND] — [reported via search over APBS/PB literature](https://arxiv.org/pdf/1707.00027)
- APBS itself: "a software package for the numerical solution of the Poisson-Boltzmann equation, one of the
  most popular continuum models for describing electrostatic interactions between molecular solutes in salty,
  aqueous media", solved with finite-difference or finite-element methods, output as a `.dx` potential map
  — [SECOND-HAND] — [Improvements to the APBS suite, arXiv:1707.00027](https://arxiv.org/pdf/1707.00027)

### Inferences

- **The preprocessing/model cost ratio is extreme and is published.** Using dMaSIF's Table 1 and forward-pass
  numbers: MaSIF's feature step alone (19.69 s) is ≈ **120×** MaSIF's own 164 ms forward pass and ≈ **550×**
  dMaSIF's 36 ms three-layer forward pass. MaSIF's *total* precomputation (6.11 + 19.69 + 50.65 = **76.45 s**)
  is ≈ **466×** its own inference. These ratios are my arithmetic on their published numbers, not quoted
  figures — [INFERENCE from PRIMARY numbers]
- **For a per-protein budget, the PB solve is not even the dominant preprocessing cost in MaSIF** — the
  geodesic local-coordinate computation (50.65 s, Dijkstra + MDS in MATLAB) is 2.6× more expensive than the
  whole feature step. A paper that cites MaSIF's preprocessing burden as "the cost of electrostatics" is
  misattributing it.
- **The dMaSIF result is the strongest controlled evidence in this entire literature** that an explicit PB
  solve is dispensable for a *site-level* task: same task, same split, same lineage, PB removed, accuracy
  equal (0.85) then better (0.87). And the r = 0.83 regression shows *why*: on a protein surface the PB
  potential is largely a smooth function of nearby atom identities and inverse distances, which is exactly
  what a small MLP over the 16 nearest atoms can reconstruct. The PB solve is, in this regime, a
  deterministic preprocessing of information the network already has.
- The r = 0.83 figure also bounds the claim honestly: ~31% of the variance in the PB potential is *not*
  recovered. Whatever lives in that residual — long-range, salt-screening and buried-charge effects that a
  16-nearest-atom local function cannot see — did not help site prediction here, but that is a null result
  on one task, not a proof of irrelevance.
- The absence of any **generalised Born** input grid in this literature is itself a finding. GB is cheaper
  than PB and is the standard implicit-solvent choice in MD, yet nobody appears to have voxelized a GB
  potential as a network input for site or affinity prediction.
- DeepDrug3D's shape-only ablation is the closest analogue to "what do physics channels buy": +0.13 on one
  pocket class, ~0 on another. That heterogeneity — large gain where chemistry discriminates (nucleotide vs
  other), none where shape already does (heme) — is the pattern one should expect for electrostatic channels
  too, and argues against reporting a single aggregate number.

### Gaps

- **No published APBS wall-clock for a ~300-residue protein at 0.5 Å.** I searched specifically for it and
  found none. The usable surrogates are MaSIF's 19.69 ± 16.08 s (which bundles PB with other features at
  mesh-vertex resolution, not a 0.5 Å volumetric grid) and the quadratic-in-grid-size scaling law. A
  domain-decomposition paper benchmarked APBS on the 141-atom protein 1etn on a 2.5 GHz Core i7 MacBook Pro
  and reported run times in a table, but the table values were not retrievable.
- MaSIF Fig. 3d's per-bar values could not be unambiguously assigned to Geom / hbond / elec / hpathy / G+C
  from the PDF text layer. **The exact ROC AUC of the electrostatics-only MaSIF-site model is therefore not
  established in these notes.** Recovering it requires reading the published figure image or the source data.
- MaSIF-ligand (Fig. 2c) and MaSIF-search (Fig. 5c) chemistry-vs-geometry numbers likewise not recovered.
  MaSIF-ligand is the most site-relevant of the three for small-molecule work.
- dMaSIF Fig. 6b's ablation bar values not recovered; only the authors' verbal summary.
- No model found that feeds a **generalised Born** potential as an input channel. I did not find a negative
  statement either — this is an absence of evidence.
- dMaSIF's code licence not verified. MaSIF's dependency chain (MSMS in particular) may carry
  non-commercial restrictions, but I did not verify any licence text in this session.

---

## SIMPLE COULOMB FIELDS: a plain 1/r or screened Coulomb grid as a substitute for a PB solve

### Takeaway

**I found no paper that directly compares a plain or screened Coulomb field against a PB solve as the input
to a binding-site, affinity or pose-scoring network.** This is a genuine hole in the literature. The nearest
thing is dMaSIF, whose learned chemical feature *is* a learned function of inverse distances and atom types
and which recovers the PB potential at r = 0.83 while matching/beating MaSIF — strong indirect evidence that
the cheap version loses little on a site task. From the force-field side, PhysNet's with/without ablation
shows a screened-Coulomb long-range term matters for energies in the asymptotic regime, but that is a
different task.

### Cited Findings

- dMaSIF's chemical features are explicitly built from **inverse distances**: an MLP on
  "[t_ik, 1/‖x_i − a_ik‖] in R^7" summed over the 16 nearest atoms, with a single hidden layer of dimension
  12 — i.e. a learned, atom-type-weighted 1/r field — [PRIMARY] — [dMaSIF, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Sverrisson_Fast_End-to-End_Learning_on_Protein_Surfaces_CVPR_2021_paper.pdf)
- That learned inverse-distance field regresses the PB potential at **r = 0.83, RMSE = 0.16**, and gives
  equal-or-better site ROC-AUC (0.85 → 0.87 vs MaSIF's 0.85) at ~1000× lower preprocessing cost
  — [PRIMARY] [MEASURED] — [dMaSIF, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Sverrisson_Fast_End-to-End_Learning_on_Protein_Surfaces_CVPR_2021_paper.pdf)
- **PhysNet's with/without-long-range ablation** (the closest thing to a direct cheap-vs-expensive
  electrostatics comparison anywhere in the retrieved literature): the paper tabulates PhysNet's performance
  "with and without explicit long-range electrostatic interactions", averaged over five runs and ensembles of
  five models. "The model without explicit inclusion of long-range interactions performs significantly
  worse", showing errors of about **1 kcal/mol in the asymptotic regions of the potential energy surface"
  — [SECOND-HAND] [MEASURED] — [PhysNet, arXiv:1902.08408](https://arxiv.org/pdf/1902.08408)
- The effect is dataset-dependent and physics-interpretable: largest for halides, where "ion-dipole
  interactions, which decay with the square of the distance, play an important role … Their influence extends
  well beyond the cut-off distance". For SN2 reactions of methyl halides with halide anions, "including
  long-range electrostatic interactions explicitly in the model significantly improves the qualitative shape
  of the predicted potential energy surface close to and beyond the cutoff radius" — [SECOND-HAND] — [PhysNet, arXiv:1902.08408](https://arxiv.org/pdf/1902.08408)
- A critique relevant to the "how cheap can you go" question: in PhysNet "the nonlocality only provides a
  statistical map between electrostatic energy and local structural features, and no far-field information
  is used to predict the atomic charges" — [SECOND-HAND] — [Long-range electrostatics in atomistic ML: a physical perspective, arXiv:2602.11071](https://arxiv.org/pdf/2602.11071)
- **OnionNet deliberately refuses charges and still performs competitively**, and its stated reason is a
  direct argument against cheap empirical charges: "the distance-based contacts and chemical element type of
  each atoms (from both the protein and the ligand) are the only information considered", and "hybrid
  empirical methods, such as AM1-BCC charges, are usually adopted to calculate the 'partial charge' of each
  atom **without considering the solvent environment and dipole moments**" — [SECOND-HAND] — [OnionNet, arXiv:1906.02418](https://arxiv.org/pdf/1906.02418) / [OnionNet, ACS Omega](https://pubs.acs.org/doi/10.1021/acsomega.9b01997)
- **Conflicting secondary claim about OnionNet:** a review describes OnionNet's distance shells as capturing
  "both short-range (van der Waals) and long-range (electrostatic) interactions". The OnionNet paper itself
  describes no explicit electrostatic term; the review is loosely describing distance shells
  — [contradiction flagged] — [OnionNet, arXiv:1906.02418](https://arxiv.org/pdf/1906.02418), contradicted by a review summary surfaced in search
- A 2025 recent-advances review also notes that a 2024 npj paper classifies long-range-aware models into
  those that "include the Coulomb form with environment-dependent charges" — placing PhysNet, BAMBOO and
  AIMNET2 in that class — [SECOND-HAND] — [Long-range electrostatics for MLIPs is easier than we thought, J Chem Phys](https://pubs.aip.org/aip/jcp/article/164/6/060901/3379367/Long-range-electrostatics-for-machine-learning)
- Also surfaced: "A universal augmentation framework for long-range electrostatics in machine learning
  interatomic potentials" and "Differentiable Particle-Mesh Ewald with Cartesian Tensor Message Passing",
  both of which replace explicit solves with learned/Ewald-style long-range terms — [SECOND-HAND, not read] — [arXiv:2507.14302](https://arxiv.org/pdf/2507.14302) / [arXiv:2606.01598](https://arxiv.org/pdf/2606.01598)

### Inferences

- The dMaSIF result can be read as an **implicit** cheap-vs-expensive comparison: a learned 1/r field over
  the 16 nearest atoms vs an APBS PB solve, same task, same data. The cheap version loses **nothing
  measurable** on site ROC-AUC and is ~1000× cheaper to prepare. But it is *learned*, not a fixed
  1/r Coulomb sum, so it is not the experiment the key question asks for — it is strictly stronger than a
  fixed 1/r field in expressivity and strictly weaker in interpretability.
- The PhysNet evidence runs the other way, but on a task where it should: absolute energies in asymptotic
  regions depend on the far field, and site/pocket scoring on a protein surface does not obviously. The
  honest reading is that the value of explicit long-range electrostatics scales with how much the target
  quantity depends on interactions beyond the network's receptive field — large for PES tails and
  ion–dipole systems, plausibly small for "is this surface patch a pocket".
- A screened Coulomb field (Debye–Hückel style, exp(−κr)/εr) occupies exactly the design niche nobody has
  tested: it costs the same O(N·M) as a 1/r sum, encodes the salt screening that a bare 1/r sum misses, and
  would be a one-line change to any voxel pipeline. The absence of such a comparison is the clearest
  actionable gap in the assignment.
- OnionNet's position is informative for the affinity side: an explicit, published refusal of partial charges
  on physical grounds, from a model that remained competitive. This is the nearest thing to a negative result
  on charges in affinity prediction, though it is a design choice, not an ablation.

### Gaps

- **No controlled comparison of a plain or screened Coulomb grid vs a PB grid as a network input exists, for
  any of the three tasks.** I searched for it from several directions (binding-site, affinity, PB-vs-Coulomb)
  and found nothing. This is a clean, cheap experiment that nobody has published.
- No paper quantifies "how much accuracy is lost by the cheap version" for a structural task. The only
  number of that shape anywhere is dMaSIF's r = 0.83 / RMSE = 0.16 for *reconstructing* PB, which is a
  feature-fidelity number, not a downstream-accuracy loss.
- PhysNet's ablation table values (the actual MAEs with and without the long-range term, per dataset) were
  not retrieved; only the authors' qualitative summary and the ~1 kcal/mol asymptotic figure.
- Dielectric treatment is unaddressed in every site model I examined. None of Pafnucy, Kalasanty, DeepSurf,
  SwinSite, GrASP or VN-EGNN represents the protein/solvent dielectric discontinuity in any form; MaSIF gets
  it implicitly via APBS and dMaSIF discards it.

---

## LEARNED CHARGES: charge-equilibration layers, learned electronegativity, charge prediction as an auxiliary task

### Takeaway

There is a mature and well-benchmarked literature on learned charges — QEq/EEM layers, learned
electronegativities, charges fitted to dipoles — but **essentially all of its controlled evidence is on
energies, forces, dipoles and charge-transfer in small molecules and materials.** I found **no evidence that
learned or equilibrated charges beat fixed force-field charges on a downstream structural task** (site
detection, site ranking, affinity, or pose scoring). The one near-relevant data point runs the other way:
dMaSIF's learned chemistry replaced an APBS solve with no loss, but it predicts no charges at all.

### Cited Findings

**Charge equilibration in neural network potentials**

- 4G-HDNNP (Ko, Finkler, Goedecker, Behler) introduced "charges from a charge equilibration method based on
  **electronegativities**, in the spirit of CENT", combining short-range atomic energies in the style of
  2G-HDNNPs with QEq-determined charges, and capturing **non-local charge transfer** — [SECOND-HAND] — [4G-HDNNP, Nat Commun 2021](https://www.nature.com/articles/s41467-020-20427-2)
- The charge distribution is "determined by minimizing the electrostatic energy with respect to the atomic
  charges, subject to total charge conservation" — [SECOND-HAND] — [4G-HDNNP, Nat Commun 2021](https://www.nature.com/articles/s41467-020-20427-2)
- Key architectural distinction: "unlike CENT, where the QEq charges serve merely as **auxiliary
  variables**, 4G-HDNNP fits the electronegativities to reproduce reference **Hirshfeld** charges from DFT"
  — [SECOND-HAND] — [Long-range electrostatics in atomistic ML: a physical perspective, arXiv:2602.11071](https://arxiv.org/pdf/2602.11071)
- Iterative charge equilibration (iQEq) implements 4G-HDNNP QEq in LAMMPS as a parallel pair style for
  large-scale periodic MD; benchmarked on NaCl, where "a dataset with only neutral and negatively charged
  NaCl clusters resolves even small energy differences between cluster geometries, and the potential
  transfers well to positively charged clusters and the melt" — [SECOND-HAND] — [iQEq, arXiv:2502.07907](https://arxiv.org/pdf/2502.07907)
- Variational charge equilibration (npj Comput Mater 2024) reports head-to-head numbers against 4G-HDNNP:
  carbon chain validation RMSE **1.167 meV/atom** and **78.98 meV Å⁻¹** for energies and forces,
  "comparable to those obtained with 4G-HDNNP"; on silver clusters "the new approach attains a lower RMSE in
  energy, forces, and Hirshfeld charges (when used as a target) than 4G-HDNNP"; on Na₈Cl₈⁺ and Na₉Cl₈⁺
  "RMSEs on energy and forces are lower than those obtained with 4G-HDNNP" — [SECOND-HAND] [MEASURED] — [Variational charge equilibration, npj Comput Mater](https://www.nature.com/articles/s41524-024-01225-6)
- Where QEq is *necessary*: for charged/doped slabs, "local MLPs and long-range models based only on
  environment-dependent charges cannot represent this effect and give a qualitatively incorrect potential
  energy surface", whereas 4G-HDNNP's QEq "lets charges respond to the global slab structure" and matches DFT
  including doping-induced changes in binding strength and site preference — [SECOND-HAND] [MEASURED] — [RuNNer 2.0, arXiv:2607.17978](https://arxiv.org/html/2607.17978)
- Where QEq charges are *wrong*: against the self-consistent-field NN (SCFNN) in bulk water, "the 4G-HDNNP
  models give significantly narrower distributions than the SCFNN model, and their average molecular dipole
  moment is either too large (Hirshfeld) or too small (Mulliken)" — [SECOND-HAND] — [Self-consistent determination of long-range electrostatics in NNPs, PMC8943018](https://pmc.ncbi.nlm.nih.gov/articles/PMC8943018)
- A silver-cluster comparison also establishes the 2G baseline failure: "the 4G-HDNNP model accurately
  reproduced the energy and forces of these systems, while the 2G model yielded a very large error in both"
  — [SECOND-HAND] — [Variational charge equilibration, npj Comput Mater](https://www.nature.com/articles/s41524-024-01225-6)

**Charge prediction as an auxiliary task**

- PhysNet learns charges rather than consuming them: "PhysNet fits the atomic charges to reproduce both the
  total dipoles and the total electronic energies, which avoids a separate learning task for the short-range
  energy", with a loss enforcing total-charge conservation, so "because of the partial charge correction
  scheme, **total charge is always predicted exactly**" — [SECOND-HAND] — [PhysNet, arXiv:1902.08408](https://arxiv.org/pdf/1902.08408)
- A 2025 classification places "PhysNet, BAMBOO and AIMNET2" together as models that "include DFT dipoles in
  the loss function and predict atomic partial charges" — [SECOND-HAND] — [Long-range electrostatics for MLIPs is easier than we thought, J Chem Phys](https://pubs.aip.org/aip/jcp/article/164/6/060901/3379367/Long-range-electrostatics-for-machine-learning)
- Critique of PhysNet's charge scheme: it "misses a non-local charge redistribution other than the average"
  — [SECOND-HAND] — [Variational charge equilibration, npj Comput Mater](https://www.nature.com/articles/s41524-024-01225-6)
- A 2025 Nature Communications paper demonstrates "machine learning of charges and long-range interactions
  **from energies and forces**" — i.e. charges as a latent, learned without charge labels — [SECOND-HAND, not read] — [Nat Commun 2025](https://www.nature.com/articles/s41467-025-63852-x)
- EspalomaCharge: a GNN for "ultrafast partial charge assignment", positioned as a drop-in replacement for
  AM1-BCC, motivated by partial charges being "central to the electrostatic contributions to intermolecular
  energies" — [SECOND-HAND] — [EspalomaCharge, arXiv:2302.06758](https://arxiv.org/pdf/2302.06758)
- Earlier work in the same line: "Graph Nets for Partial Charge Prediction" — [SECOND-HAND, not read] — [arXiv:1909.07903](https://arxiv.org/pdf/1909.07903)
- A 2026 "linear-scaling, charge-aware foundation potential for atomistic simulations" continues the trend
  toward charge-awareness in general-purpose potentials — [SECOND-HAND, not read] — [arXiv:2511.07249](https://arxiv.org/pdf/2511.07249)
- Background on why empirical charges vary: "Gasteiger-Marsili charges, which are computed from
  electronegativity, depend on both the type of atom and its molecular environment" — i.e. EEM-style charges
  are already a (shallow) equilibration scheme — [SECOND-HAND] — [Partial charge, Wikipedia](https://en.wikipedia.org/wiki/Partial_charge)
- Also relevant to the "which charges" question at all: empirical charge models for cheminformatics have
  been benchmarked for universality, and the spread between QM-derived net atomic charge schemes in
  reproducing the surrounding electrostatic potential has been quantified — [SECOND-HAND, not read] — [High-quality and universal empirical atomic charges, PMC4667495](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4667495/) / [How well do QM-derived net atomic charges reproduce the ESP, PMC12230677](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12230677/) / [Atomic Partial Charges Arboretum, PMC7317385](https://pmc.ncbi.nlm.nih.gov/articles/PMC7317385/)

### Inferences

- The learned-charge literature and the protein-structure-task literature **do not touch**. Every controlled
  comparison I found (4G-HDNNP vs 2G, variational QEq vs 4G-HDNNP, PhysNet with vs without long-range,
  SCFNN vs 4G-HDNNP) is scored on energies, forces, dipoles or Hirshfeld charges, on NaCl clusters, carbon
  chains, silver clusters, slabs or bulk water. None is scored on a pocket, an affinity, or a pose.
- The physical condition under which QEq demonstrably beats environment-dependent fixed charges — global,
  non-local charge redistribution in response to overall system state (the doped-slab result) — has **no
  obvious analogue in site detection**, where the question is local and the global charge state of the
  protein is fixed. This is a principled reason to expect the transfer to be weak, not merely untested.
- Where learned charges plausibly *do* help a protein pipeline is not as a better feature but as a **cheaper
  and more robust charge assigner**: EspalomaCharge-style models remove the AM1-BCC bottleneck for ligands
  and non-standard residues, which is exactly the failure mode in Pafnucy's charging protocol. That is an
  engineering argument about coverage and cost, not an accuracy argument, and nobody has measured it
  end-to-end on a site or affinity benchmark.
- The SCFNN-vs-4G-HDNNP dipole result is a useful warning: charges fitted to reproduce one population
  analysis (Hirshfeld or Mulliken) inherit that analysis's biases. Since Amber ff14SB charges are themselves
  RESP-fitted and AM1-BCC charges are bond-charge-corrected AM1, a model trained on one and run on the other
  is already inconsistent — a problem that applies directly to Pafnucy/Kalasanty/DeepSurf, which mix ff14SB
  (standard residues) with AM1-BCC (everything else) in the same channel.

### Gaps

- **No downstream structural-task evidence at all.** I found no paper showing learned charges (QEq, EEM,
  learned electronegativity, auxiliary charge prediction) outperforming fixed force-field charges on binding-
  site detection, site ranking, affinity prediction or pose scoring. This is the clearest negative finding in
  the assignment and should be reported as such rather than hedged.
- No charge-prediction auxiliary-task result on a protein site model. Nobody appears to have added
  "predict the partial charge / PB potential" as an auxiliary head on a pocket network and measured the
  effect on site metrics — even though dMaSIF's Fig. 6a shows such a head trains easily (r = 0.83).
- Exact benchmark tables for 4G-HDNNP and iQEq were not retrieved; the numbers above come from search
  syntheses and the authors' own comparisons, which are not independent.
- Licences and weight availability for the learned-charge models (4G-HDNNP/RuNNer, PhysNet, EspalomaCharge)
  were not checked in this session.

---

## MULTIPOLES: dipole and quadrupole moments as equivariant features in E(3)/SO(3) networks

### Takeaway

Atomic multipoles appear in equivariant networks almost exclusively as **outputs** — quantities predicted to
reproduce an electrostatic potential or a long-range energy — not as **input features**. I found **no
controlled ablation isolating the effect of multipole input features in any protein–ligand site, affinity or
pose-scoring model**, and only one ablation-shaped result anywhere: a 2026 QM9 study where adding atomic
dipoles alongside quadrupoles made things *worse*.

### Cited Findings

- "Learning Atomic Multipoles" (Thürlemann, Böselt, Riniker) predicts atomic multipoles up to the quadrupole
  with an equivariant GNN to reproduce the electrostatic potential; the equivariant dipole update is built
  from neighbour vectors weighted by learned functions, and "models for monopoles, dipoles, and quadrupoles
  were optimized **independently**". No multipole-feature ablation table was found — [SECOND-HAND] — [Learning Atomic Multipoles, JCTC 2021](https://pubs.acs.org/doi/10.1021/acs.jctc.1c01021) / [arXiv:2110.05417](https://arxiv.org/abs/2110.05417)
- **The one ablation-shaped multipole result**: a 2026 study of molecular electrostatic potentials from ML
  dipole and quadrupole predictions reports that "the AC-AD-DQ models performed **worse** than the AC-DQ
  models and were more prone to overfitting" — i.e. atomic charges + atomic dipoles + quadrupoles did worse
  than charges + quadrupoles — while "adding a dipole contribution in the dw100 models **slightly increased**
  the model accuracy" — [SECOND-HAND] [MEASURED, but on an ESP-reproduction task, not a structural one] — [Molecular ESPs from ML dipole and quadrupole predictions, IOPscience](https://iopscience.iop.org/article/10.1088/3050-287X/ae531a)
- PaiNN has the best-known ablation of **equivariant vector features in general** (not multipoles
  specifically): "We evaluate the impact of equivariant vector features at the example of the aspirin MD
  trajectory" (MD17), and the ICML slide deck states that the ablations show equivariant features matter even
  for scalar properties. The table values were not retrieved — [SECOND-HAND] [MEASURED, equivariance not
  multipoles] — [PaiNN, PMLR v139](http://proceedings.mlr.press/v139/schutt21a/schutt21a.pdf) / [PaiNN slides](https://icml.cc/media/icml-2021/Slides/8499.pdf)
- "Polarizable atomic multipoles for learning long-range electrostatics" reads dipoles and quadrupoles out of
  equivariant features, with quadrupoles made traceless before prediction. No ablation of those channels was
  found in the retrieved excerpts — [SECOND-HAND] — [arXiv:2605.05746](https://arxiv.org/html/2605.05746)
- "Differentiable Particle-Mesh Ewald with Cartesian Tensor Message Passing" states plainly that "the two
  example studies reported here focus on **monopoles and dipoles; quadrupolar channels are a natural next
  extension**" — i.e. quadrupoles are not yet tested even in the long-range MLIP literature
  — [SECOND-HAND] — [arXiv:2606.01598](https://arxiv.org/html/2606.01598)
- Multipole expansion for ground and excited states in molecular simulation (2026) and neural-network
  augmentation of the AMOEBA polarizable force field (2023) are the two further lines where multipoles meet
  NNs — both on energies/spectra, neither on protein–ligand structural tasks — [SECOND-HAND, not read] — [Incorporating long-range interactions via the multipole expansion, npj Comput Mater](https://www.nature.com/articles/s41524-026-02048-3) / [NNs into AMOEBA, ChemRxiv](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/6556373b6e0ec7777f1349df/original/incorporating-neural-networks-into-the-amoeba-polarizable-force-field.pdf)
- On the protein side, HeMeNet (heterogeneous multichannel equivariant network for protein multitask
  learning) surfaced in the same search space but with no indication of multipole features — [SECOND-HAND, not read] — [HeMeNet, arXiv:2404.01693](https://arxiv.org/pdf/2404.01693)

### Inferences

- The asymmetry is systematic: in the MLFF literature multipoles are **targets** (fit to reproduce an ESP or
  a long-range energy), whereas the key question asks about them as **inputs**. For a protein, inputting an
  atomic dipole would require having computed it from QM or a polarizable force field first — a preprocessing
  cost well above a PB solve — which plausibly explains why nobody has tried it on a 300-residue protein.
- The one directly relevant number points the wrong way for multipoles: adding atomic dipoles on top of
  charges + quadrupoles **degraded** accuracy and increased overfitting. Even granting that this is an
  ESP-reproduction task, it argues that higher multipoles are not free — they add parameters and variance,
  and the information they carry overlaps heavily with what charges plus geometry already encode.
- The honest framing for a protein site model is that the equivariant-feature literature (PaiNN, e3nn-style
  tensor networks, GVP) gives good evidence that **vector/tensor internal features** help, and no evidence at
  all that **physically-derived multipole inputs** help. These are different claims that are easy to conflate:
  an l=1 internal channel is not a dipole moment.

### Gaps

- **No controlled ablation isolating multipole input features in any protein–ligand model** — site, affinity
  or pose. None found, from several search directions.
- PaiNN's MD17-aspirin ablation numbers not retrieved.
- Whether "Learning Atomic Multipoles" contains an internal ablation of its dipole/quadrupole heads is
  unresolved; the retrieved excerpts suggest independent optimization rather than an ablation.
- No paper found that uses a *protein's* computed atomic multipoles as network inputs for any task.

---

## THE KEY QUESTION: is there evidence that electrostatics helps SITE DETECTION AND RANKING specifically?

### Takeaway

**Almost none, and what little exists is partly contradicted by a direct controlled refutation.** For
small-molecule pocket detection and ranking on COACH420 / HOLO4K / sc-PDB / CHEN there is **no published
ablation isolating any electrostatic input**, despite three of the most-used models carrying a partial-charge
channel. The only genuine retrained feature-subset ablation involving Poisson–Boltzmann electrostatics on a
*site* task is MaSIF-site — and that is protein–protein interface prediction, where all-features beats the
best single-feature subset by roughly 0.02 ROC AUC. Its successor dMaSIF then removed the PB solve on the
identical task and split and **matched then exceeded** MaSIF's accuracy. Meanwhile the models that top
current site-ranking benchmarks (P2Rank, VN-EGNN) contain no electrostatics at all, and their own ablations
attribute their gains to geometry, architecture and evolutionary embeddings.

### Cited Findings

**Evidence that electrostatics helps a site-level task**

- MaSIF-site's Fig. 3d is the only retrained feature-subset ablation on a site task that includes PB
  electrostatics as its own arm (Geom / hbond / elec / hpathy / G+C; 218,246 positive and 1,973,624 negative
  surface points). All-features (G+C) is the best at **0.80 ROC AUC**; the five bars span **0.68–0.80**, so
  every single-feature model is within ~0.02–0.12 of the full model — [PRIMARY] [MEASURED] — [MaSIF, Nat Methods 2020 (PDF)](https://www-cbi.cs.uni-saarland.de/wp-content/uploads/2025/03/MaSIF-paper_SS2025.pdf)
- MaSIF-ligand reports a chemistry-vs-geometry comparison for **pocket classification** (balanced accuracy,
  G+C vs Geom vs Chem over seven cofactor classes) — the closest MaSIF result to small-molecule pocket work —
  but the values were not recoverable from the PDF text layer — [PRIMARY of caption] — [MaSIF, Nat Methods 2020 (PDF)](https://www-cbi.cs.uni-saarland.de/wp-content/uploads/2025/03/MaSIF-paper_SS2025.pdf)
- DeepDrug3D's shape-only ablation on **pocket classification**: replacing 14 knowledge-based interaction-
  energy channels with a 2-channel binary occupancy gave 0.824 (nucleotide) and 0.952 (heme) against ~0.95
  for the full model — the energy channels help one pocket class and not the other — [SECOND-HAND] [MEASURED] — [DeepDrug3D, PLoS Comput Biol](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006718)
- SwinSite's channel masking shows the chemical/atom-type channels do carry signal for Top-1 DCC success
  (H-bond donor, aromatic, H-bond acceptor, hydrophobic, heterodegree each ~15–23% drop) — but the charge
  channel's number is in the unretrieved SI, and the authors conclude the chemical channels matter "to a
  lesser extent than geometric cues" — [PRIMARY] [partially MEASURED] — [SwinSite, PMC12977039](https://pmc.ncbi.nlm.nih.gov/articles/PMC12977039/)

**Evidence that it is unnecessary for site detection**

- **dMaSIF, the direct refutation**: same site-identification task, same 2958/356 split, PB input removed and
  replaced by a learned MLP over atom types and inverse distances. Single-conv dMaSIF "matches the best
  accuracy of 0.85 ROC-AUC produced by MaSIF"; three-conv dMaSIF "outperform[s] MaSIF with a 0.87 ROC-AUC".
  Preprocessing ~1000× cheaper — [PRIMARY] [MEASURED] — [dMaSIF, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Sverrisson_Fast_End-to-End_Learning_on_Protein_Surfaces_CVPR_2021_paper.pdf)
- And the mechanism: the learned features recover the PB potential at **r = 0.83, RMSE = 0.16**, so the PB
  solve was largely re-deriving information already present in atom types and distances — [PRIMARY] — [dMaSIF, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Sverrisson_Fast_End-to-End_Learning_on_Protein_Surfaces_CVPR_2021_paper.pdf)
- **VN-EGNN's ablation — what actually moves site DCC/DCA — contains no electrostatics anywhere**
  (Table 2, mean ± s.d.):

  | Variant | VN | Heterog. MP | ESM | COACH420 DCC / DCA | HOLO4K DCC / DCA | PDBbind2020 DCC / DCA |
  |---|---|---|---|---|---|---|
  | EGNN | ✗ | ✗ | ✗ | 0.156 (0.017) / 0.361 (0.020) | 0.127 (0.005) / 0.406 (0.004) | 0.143 (0.007) / 0.302 (0.006) |
  | VN-EGNN (residue emb.) | ✓ | ✓ | ✗ | 0.503 (0.022) / 0.684 (0.016) | 0.438 (0.019) / 0.605 (0.013) | 0.551 (0.017) / 0.751 (0.009) |
  | VN-EGNN (homog.) | ✓ | ✗ | ✗ | 0.497 (0.014) / 0.700 (0.013) | 0.414 (0.023) / 0.618 (0.024) | 0.502 (0.029) / 0.717 (0.025) |
  | VN-EGNN (homog.) | ✓ | ✗ | ✓ | 0.575 (0.008) / 0.708 (0.009) | 0.479 (0.012) / 0.595 (0.010) | 0.649 (0.010) / 0.805 (0.006) |
  | VN-EGNN (full) | ✓ | ✓ | ✓ | **0.605 (0.009) / 0.750 (0.008)** | **0.532 (0.021) / 0.659 (0.026)** | **0.669 (0.015) / 0.820 (0.010)** |

  The levers are virtual nodes (0.156 → ~0.50 COACH420 DCC), ESM-2 embeddings (0.497 → 0.575) and
  heterogeneous message passing (0.575 → 0.605) — [PRIMARY] [MEASURED] — [VN-EGNN, arXiv:2404.07194](https://arxiv.org/html/2404.07194)
- **P2Rank**, the standard strong baseline for site *ranking*, has 35 features, none electrostatic, and "the
  single most important feature turned out to be a geometric feature termed protrusion" — [SECOND-HAND] — [P2Rank, J Cheminform](https://link.springer.com/article/10.1186/s13321-018-0285-8)
- SwinSite's own verdict on its channel ablation: chemical and atom-type features matter "to a lesser extent
  than geometric cues", and the largest single drop is `atom:O` (39.7%) — an **atom-type occupancy** channel,
  not a charge channel — [PRIMARY] — [SwinSite, PMC12977039](https://pmc.ncbi.nlm.nih.gov/articles/PMC12977039/)
- The entire DeepSite / KDEEP / DeepPocket / gnina lineage reaches competitive site numbers with **no partial
  charge and no electrostatic potential**, only atom types and binary ionizability flags — [PRIMARY of
  moleculekit docs; SECOND-HAND for DeepPocket/gnina] — [moleculekit voxeldescriptors docs](https://software.acellera.com/moleculekit/moleculekit.tools.voxeldescriptors.html) / [DeepPocket repo](https://github.com/devalab/DeepPocket/blob/main/README.md)
- Kalasanty and DeepSurf carry the `partialcharge` channel and **never measured it**; neither paper contains
  an input-feature ablation of any kind — [PRIMARY] [NOT MEASURED] — [Kalasanty, arXiv:1904.06517](https://arxiv.org/pdf/1904.06517) / [DeepSurf, arXiv:2002.05643](https://arxiv.org/pdf/2002.05643)
- A 2024 comparative evaluation of site-prediction methods exists and tabulates the methods' feature sets
  (e.g. "VN-EGNN | EGNN + VN | ESM-2 embeddings | 1280"), which is the right place to check whether any
  evaluated method carries electrostatics — [SECOND-HAND] — [Comparative evaluation, J Cheminform](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00923-z)

**Why the evidence does not transfer from affinity/rescoring**

- On the affinity side, the leading models are also mostly electrostatics-free: OnionNet uses only
  distance-based contacts and element types and explicitly rejects empirical partial charges; RTMScore uses a
  "residue–atom distance likelihood potential" from a graph transformer plus a mixture density network, with
  no electrostatic term found; Vina-style scoring in gnina has no electrostatic term at all
  — [SECOND-HAND] — [OnionNet, arXiv:1906.02418](https://arxiv.org/pdf/1906.02418) / [RTMScore repo](https://github.com/sc8668/RTMScore/blob/main/README.md) / [gnina workshop notes](https://gnina.github.io/gnina/rsc_workshop2021/)
- Pafnucy, the one affinity model with a charge channel, measured only weight ranges (partialcharge 4th of
  17) and spatial occlusion — never a charge ablation — [PRIMARY] — [Pafnucy, arXiv:1712.07042](https://arxiv.org/pdf/1712.07042)

### Inferences

- **The answer to the key question is: no, there is no good evidence that electrostatics helps site detection
  and ranking.** The correct summary for the report is: (i) three widely used site models carry a
  partial-charge channel but never measured it; (ii) one site-level task (protein–protein interface) has a
  real PB ablation showing all-features > any single feature by ~0.02 AUC; (iii) on that same task, removing
  the PB solve entirely cost nothing and gained speed; (iv) no small-molecule pocket benchmark has an
  electrostatics ablation at all.
- The evidence genuinely does not transfer across tasks, and the mechanism is clear. In affinity prediction
  and pose rescoring the ligand is present, so charge **complementarity** between two bodies is a
  first-order feature of the input. In apo/holo site detection there is only one body, and the question is
  whether a surface patch could host a ligand — the electrostatic field of the protein alone carries far less
  of that answer than its shape does. P2Rank's protrusion result, SwinSite's "geometric cues dominate", and
  dMaSIF's finding that curvature concatenation adds little on top of learned chemistry all point the same
  way from three different directions.
- There is a second, subtler reason electrostatics underperforms on site tasks specifically: at 1–2 Å voxel
  resolution a std-normalised partial charge is one scalar per heavy atom, while the atom-type one-hots
  already encode most of the charge's variance (an O is negative, an N-H is positive). The charge channel is
  nearly collinear with the type channels, which is exactly what dMaSIF's r = 0.83 PB reconstruction from
  types + distances demonstrates quantitatively.
- **For EquiCave specifically** (ranking pocket candidates on apo and holo proteins), the published record
  gives no reason to pay for a PB solve, and no reason to pay for a charging pipeline either. If
  electrostatics is to be included, the defensible design is dMaSIF's: let the network learn a
  chemistry channel from atom types and inverse distances, which costs milliseconds and reproduces PB at
  r = 0.83. A fixed screened-Coulomb channel would be the cheap, untested middle option — and because nobody
  has published that comparison, running it on COACH420/HOLO4K with a paired test would be a genuinely novel
  controlled result rather than a replication.
- A caveat in the other direction: all of the above is evidence about *aggregate* benchmark metrics. The
  DeepDrug3D pattern (+0.13 on nucleotide pockets, ~0 on heme) suggests electrostatics could still matter for
  specific, charge-driven site classes — metal sites, phosphate/nucleotide pockets, highly charged
  substrate channels — which aggregate DCC/DCA on COACH420 would average away. No paper I found stratifies a
  site-prediction ablation by pocket electrostatic character.

### Gaps

- **No electrostatics ablation exists for small-molecule pocket detection or ranking.** Not for Kalasanty, not
  for DeepSurf, not for DeepPocket, not for GrASP, not for VN-EGNN, not for P2Rank. SwinSite's is the only
  candidate and its charge arm is in unretrieved supplementary material.
- No published study stratifies site-prediction performance by whether the pocket is charge-driven, so the
  "electrostatics helps metal and nucleotide sites" hypothesis is untested.
- MaSIF-ligand's G+C / Geom / Chem balanced accuracies — the single most transferable MaSIF number for
  small-molecule pocket work — were not recovered.
- No apo-vs-holo comparison of electrostatic feature value. Since apo structures have different side-chain
  rotamers and protonation, a charge channel computed on an apo structure may be noisier than on holo, which
  would matter for cryptic-site work. Nobody has measured this.
- The EquiPocket paper is withdrawn, so a commonly cited "equivariant pocket model with chemical features"
  cannot be used as evidence either way.
- Licence and weight-availability status is unverified for every method in this section except Kalasanty and
  DeepSurf, which state that weights are published but name no licence.

---

## Cross-cutting summary table

| Model | Task | Electrostatic input | [LISTED] or [MEASURED] | Preprocessing cost | Code / weights |
|---|---|---|---|---|---|
| Pafnucy | affinity | `partialcharge` float, 1 of 19 features; Chimera ff14SB + AM1-BCC | LISTED; weight-range rank 4/17; **no ablation** | charging via Chimera (not timed) | gitlab.com/cheminfIBB/pafnucy; licence not stated |
| Kalasanty | site segmentation | inherits Pafnucy's 18 features incl. partial charge | LISTED only | not reported | gitlab, source + **weights**; licence not stated |
| DeepSurf | site, surface | same 18 features; requires protonation | LISTED only; ablations are architectural | not reported | github.com/stemylonas/DeepSurf, source + **trained models**; licence not stated |
| SwinSite (2026) | site | "estimated partial charge", 1 of 18 | MEASURED by zero-masking, **charge number in SI only** | not reported; inference 1.7 s/protein | github.com/ding-oh/SwinSite; weights not mentioned; licence not stated |
| DeepSite / KDEEP | site / affinity | **none**; `positive_ionizable`/`negative_ionizable` flags only | n/a | moleculekit voxelization | moleculekit (Acellera); PlayMolecule web service |
| DeepPocket | site + segmentation | **none**; 14 atom-type channels | n/a | libmolgrid gninatypes/molcache2 | github.com/devalab/DeepPocket |
| gnina | docking / pose scoring | **none**; 34–35 smina atom types; Vina term has no electrostatics | n/a | libmolgrid, GPU | github.com/gnina/gnina |
| GrASP | site, graph | **formal** charge only | LISTED; **no ablation at all** | not reported | github.com/tiwarylab/GrASP, model + data; licence not stated |
| VN-EGNN | site | **none**; ESM-2 embeddings on Cα | n/a; full ablation over VN/MP/ESM | ESM-2 forward pass | github.com/ml-jku/vnegnn; weights not mentioned |
| P2Rank | site + **ranking** | **none** in 35 features; top feature is protrusion | n/a | seconds, CPU | github.com/rdk/p2rank |
| MaSIF | PPI interface site; pocket classification | **APBS 1.5 PB potential** per surface vertex via PDB2PQR + Multivalue, capped ±30, normalised | **MEASURED** (Fig. 3d retrained subsets; G+C 0.80, bars span 0.68–0.80) | 6.11 s surface + **19.69 s features** + 50.65 s coords = 76.45 s/protein; >1TB cached | github.com/LPDI-EPFL/masif; needs PDB2PQR 2.1.1, APBS 1.5, MSMS, Reduce |
| dMaSIF | same site task | **none** — learned MLP on [atom type, 1/r] over 16 nearest atoms | **MEASURED**: 0.85 matches MaSIF, 0.87 beats it; recovers PB at r=0.83 | 59.0 + 6.59 + 0.46 ms = **66 ms/protein** | referenced in CVPR paper; licence not verified |
| DeepDrug3D | pocket classification | **none** — 14 DFIRE knowledge-based energy channels (NOT Coulomb/AMBER) | MEASURED: shape-only 0.824 / 0.952 vs ~0.95 | DFIRE Fortran, Linux-only | github.com/pulimeng/DeepDrug3D |
| OnionNet | affinity | **explicitly refused** — element-pair distance shells only | n/a (design choice, argued) | trivial | — |
| RTMScore | pose / screening | none found | n/a | — | github.com/sc8668/RTMScore |

**The headline cost numbers**, all from dMaSIF Table 1 and §4.2, with the ratios being my arithmetic:
MaSIF's PB-bearing feature step is **19.69 ± 16.08 s/protein** against a **164 ms** MaSIF forward pass
(**≈120×**) and a **36 ms** dMaSIF forward pass (**≈550×**); MaSIF's full precomputation is **76.45 s**,
**≈466×** its own inference, and the cached features exceed **1TB** —
[dMaSIF, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Sverrisson_Fast_End-to-End_Learning_on_Protein_Surfaces_CVPR_2021_paper.pdf).
