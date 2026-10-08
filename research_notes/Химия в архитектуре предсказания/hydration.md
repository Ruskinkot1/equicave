# Hydration and Desolvation as a Signal for Binding-Site Prediction and Pocket Ranking

**Provenance convention used throughout.** Every finding is tagged:
- `[PRIMARY-FULL]` — I fetched and read the primary source's full text (or the first ~100k characters of it) in this research pass.
- `[PRIMARY-ABSTRACT]` — I read an authoritative abstract/catalogue record of the primary source, not the full text.
- `[SECOND-HAND]` — the fact comes from a search-engine summary of a source I could not open (paywall/403/cookie wall). Treat as a lead to verify, not as established.

Full texts I could **not** open in this pass, and whose contents are therefore `[SECOND-HAND]` below: Halgren 2009 SiteMap (`10.1021/ci800324m`), Balius et al. 2017 PNAS GIST-in-DOCK, Schneider & Zacharias 2012 dPredGB full text (ScienceDirect 403), the 2019 *JCAMD* water-tool comparison (Springer), HydraMap v1/v2, SiteMap vendor manual, the Nature Comms Chem version of SuperWater (Nature IdP redirect).

---

## THE CLAIM TO TEST: has a hydration-thermodynamics descriptor ever been turned into a cavity-ranking feature and benchmarked?

### Takeaway

The claim is **confirmed in its strict form and refuted in its loose form**. I found **no** method that computes an explicit-water hydration-thermodynamics quantity (WaterMap/GIST/3D-RISM/SZMAP/JAWS/WaterFLAP-class) and uses it to rank a protein's full list of candidate cavities on COACH420, HOLO4K, LIGYSIS or scPDB — that part of the claim stands. But there **is** one genuine counterexample at the level of a *continuum desolvation* descriptor: **dPredGB** (Schneider & Zacharias, *J. Struct. Biol.* 2012) detects candidate cavities geometrically and then ranks that full candidate list by a generalized-Born desolvation free energy, reporting top-1 success on a standard bound/unbound binding-site benchmark. That is exactly the architecture the objective is looking for, implemented 14 years ago, with a cheap descriptor — and essentially never followed up.

### Cited Findings

**The counterexample (continuum desolvation, cavity-list ranking, benchmarked):**
- dPredGB "combines rapid geometric detection with an evaluation of the desolvation properties of the putative binding pocket", i.e. stage 1 identifies putative cavities and stage 2 "identifies putative cavities for which the desolvation free energies are subsequently evaluated", with probes placed on the cavity surface — [OSTI record 1059618](https://www.osti.gov/biblio/1059618) `[PRIMARY-ABSTRACT]`; same wording in [ScienceDirect listing](https://www.sciencedirect.com/science/article/abs/pii/S1047847712002547) `[SECOND-HAND]`
- Performance: "the known ligand binding cavity [is] the top ranking prediction in 69% of the unbound cases and 85% of the bound cases" — [OSTI record 1059618](https://www.osti.gov/biblio/1059618) `[PRIMARY-ABSTRACT]`
- Citation: Schneider S. & Zacharias M., *Journal of Structural Biology* 180(3):546-550, 2012, DOI `10.1016/j.jsb.2012.09.010`, PMID 23023089 — [OSTI record 1059618](https://www.osti.gov/biblio/1059618) `[PRIMARY-ABSTRACT]`
- Benchmark used: "optimized for a small set of proteins (used to classify binding site predictors in Leis et al., 2010) and validated on a test set of bound and unbound proteins, widely used to benchmark the performance of several binding site predictors" — [search summary of ScienceDirect/PubMed records](https://pubmed.ncbi.nlm.nih.gov/23023089) `[SECOND-HAND]`. **This is NOT COACH420/HOLO4K/LIGYSIS/scPDB** — it is the older Huang–Schroeder / Leis-style bound-unbound set, almost certainly ~48–210 proteins.
- The paper also claims spatial output, not just a scalar: it provides "the spatial characterization of the desolvation properties of a binding region" — [OSTI record 1059618](https://www.osti.gov/biblio/1059618) `[PRIMARY-ABSTRACT]`

**Near-misses that do NOT count (recorded as such, per the brief):**
- **GIST-in-DOCK (Balius et al., PNAS 2017)** — a GIST water-displacement term was added to DOCK3.7 and precomputed on a lattice, but it scores *ligand poses inside an already-known site* in docking screens, not candidate cavities. Authors: "inclusion of this water-displacement term can substantially improve the hit rates and ligand geometries from docking screens, although the magnitude of its effects can be small"; GIST energy was ~8% of total docking score for the top 100 docked molecules in the CcP-ga cavity — [PNAS 114:E6839](https://lehman.edu/faculty/tkurtzman/pdfs/PNAS-2017-Balius-E6839-46.pdf) `[SECOND-HAND]`. **Classification: known-pocket scoring, and the lead system (CcP-ga) is an engineered model cavity.**
- **GIST's original validation system was cucurbit[7]uril**, a host-guest cavitand, not a protein — [Nguyen/Kurtzman/Gilson, ResearchGate record](https://www.researchgate.net/publication/230593833_Grid_inhomogeneous_solvation_theory_Hydration_structure_and_thermodynamics_of_the_miniature_receptor_cucurbit7uril) `[SECOND-HAND]`. **Classification: host-guest, excluded by the brief.**
- **3D-RISM over 3,706 apo structures (Yoshidome et al., *J. Comput. Chem.* 2020)** — the largest water-thermodynamics survey of binding sites I found. It computed g_O(r) at ~620,000 crystallographic-water positions and at ligand heavy atoms across 2,403 structures with waters, from 4,154 PDBbind refined-set v2017 complexes. **It did not rank binding sites against other surface regions and reports no accuracy, enrichment or AUC.** Its only pose-level result was qualitative: polar ligand heteroatoms in correct crystallographic poses overlapped highly hydrated regions more often than in AutoDock Vina decoy poses (RMSD 4.5–5.5 Å); the authors say correct poses "might be distinguished" — [PMC7540010](https://pmc.ncbi.nlm.nih.gov/articles/PMC7540010/) `[PRIMARY-FULL]`. **Classification: large-scale hydration survey of known sites; no cavity ranking.**
- **DeepWATsite / hydration-channel CNN (*Commun. Chem.* 2020)** — puts hydration occupancy, enthalpy and entropy on a 3D grid as CNN input channels alongside protein and ligand density, trained "to separate active from decoy poses" — [Nature Commun. Chem. s42004-020-0261-x](https://www.nature.com/articles/s42004-020-0261-x) `[SECOND-HAND]`. **This is the closest existing instance of "water thermodynamics as a network input channel", but the task is pose ranking in a known site, not pocket detection.**
- **ColdBrew (2026)** — "uses ColdBrew for ranking and prioritizing waters in known binding sites to guide ligand design. It does not describe using it for pocket detection" — [PMC12822489](https://pmc.ncbi.nlm.nih.gov/articles/PMC12822489) `[PRIMARY-FULL]`. **Classification: water ranking inside known sites.**
- **CrypToth (bioRxiv 2024)** — ranks *hotspots* from mixed-solvent MD with topological data analysis; "in 6 out of 9 cases, the correct hotspots were ranked 1" — [bioRxiv 2024.07.10.602991](https://www.biorxiv.org/content/10.1101/2024.07.10.602991v2.full) `[SECOND-HAND]`. **Classification: ranks a candidate list, but the probes are organic co-solvents (isopropanol, resorcinol, acetic acid), not water thermodynamics, N=9, and the task is cryptic-site specific.**
- **OpenEye Orion** advertises a "ligandability prediction model [that] helps rank cryptic pockets and guide conformation selection" — [eyesopen.com/cryptic-pocket](https://www.eyesopen.com/cryptic-pocket) `[SECOND-HAND]`. Vendor claim, no published benchmark found, commercial.
- **SuperWater** explicitly lists "binding site predictions" as a *possible future application*; pockets appear only as case-study contexts — [bioRxiv 2024.11.18.624208](https://www.biorxiv.org/content/10.1101/2024.11.18.624208v1.full) `[PRIMARY-FULL]`

**Independent confirmation that the gap is real:**
- Two separate extended searches, phrased differently, returned explicit "no such method found" conclusions: "I didn't find a single method that combines hydration-site thermodynamics with pocket-level ranking. The hydration-thermodynamics literature (HydraMap, 3D-RISM, GridGCMC-style methods, HydroRank) and the pocket-ranking literature (Fpocket/ConCavity re-ranking) appear to be separate threads"; and for GIST specifically, "the results I found don't include a study that scores druggability with GIST directly". `[SECOND-HAND — negative evidence from search coverage, not proof of absence]`
- Fpocket's own authors concede the ranking problem that a solvation term would address: pockets are ranked by "putative capacity to bind a small molecule, which does not reflect drugability" — [Fpocket, BMC Bioinformatics 10:168](https://link.springer.com/article/10.1186/1471-2105-10-168) `[SECOND-HAND]`

### Inferences

- The strict claim survives. The single counterexample (dPredGB) uses a **continuum GB desolvation energy**, not a hydration-thermodynamics descriptor in the WaterMap/GIST sense, and its benchmark predates COACH420/HOLO4K/LIGYSIS. So "no hydration-thermodynamics descriptor turned into a cavity-ranking feature and evaluated on a modern binding-site benchmark" is still true.
- dPredGB is nevertheless the most important finding for the project, for two opposite reasons: (a) it is prior art that must be cited and positioned against, and (b) its 69%/85% top-1 is *not* obviously competitive with modern geometry-only baselines, so the 2012 result does not by itself establish that desolvation adds signal on top of a strong modern geometric ranker. It was never re-run on COACH420/HOLO4K.
- The cleanest publishable claim available is therefore: *re-evaluating a desolvation ranking feature on modern benchmarks with a proper geometry-only control* — which is an ablation nobody has done, not a new idea.

### Gaps

- I could not read dPredGB's full text (ScienceDirect 403, PubMed cookie wall). Unverified: the exact GB formulation, the probe placement scheme, the identity and size of the test set, the per-structure wall-clock cost, and whether any geometry-only control was reported. **This is the single highest-value follow-up: get `10.1016/j.jsb.2012.09.010` in full.**
- I could not confirm whether dPredGB code was ever released, or whether it still exists.
- I did not exhaustively check scPDB-era druggability papers (DrugPred, DoGSiteScorer, PockDrug) for a solvation term inside their descriptor sets. DrugPred and DoGSite are known to use buriedness/hydrophobicity but I did not verify whether any includes a continuum-solvation term, and whether they rank full cavity lists or classify known pockets.
- `[CONFLICT/AMBIGUITY]` The PMC record PMC12585415, titled "Making waves in structure-based ligand discovery", returned abstract text essentially identical to the ColdBrew paper's content. It may be a commentary on ColdBrew rather than an independent review. I could not resolve which. Do not cite it as an independent source.

---

## WATER THERMODYNAMICS METHODS: what each computes, cost, licence, and whether ever used for detection/ranking

### Takeaway

All six named methods were built for, and are used for, lead optimisation on a known site. None has been turned into a cavity-ranking feature. The MD-based ones (WaterMap, GIST, JAWS) cost hours to days per protein and are disqualified for a per-protein network input channel; 3D-RISM is ~0.5–2 h/structure; only grid/statistical methods (SZMAP, HydraMap) approach a usable budget, and HydraMap is the only one that is both fast and non-commercial.

### Cited Findings

**WaterMap (Schrödinger; Young/Abel/Friesner lineage)**
- Explicit-solvent MD plus inhomogeneous solvation theory to locate hydration sites and assign them enthalpy/entropy/free energy. **Commercial, Schrödinger.** — [schrodinger.com/platform/products/watermap](https://www.schrodinger.com/platform/products/watermap) `[SECOND-HAND]`
- Cost is the stated barrier: it "hinders its application to a large number (e.g., thousands) of proteins" — `[SECOND-HAND]`, attributed to the HydraMap line of papers
- C&EN: the tool "has a reputation among some users for being time-consuming and expensive" — [C&EN 90(11)](https://cen.acs.org/articles/90/i11/Waters-Role-Drug-Discovery.html) `[SECOND-HAND]`
- **No published per-run wall-clock figure found.** Typical protocol in the surrounding literature involves ns-scale MD on a solvated box.
- Pocket detection/ranking: **no.** Applications found are lead optimisation and repurposing on known sites, e.g. PAK1 inhibitors — [ACS Omega 2021](https://pubs.acs.org/doi/10.1021/acsomega.1c02032) `[SECOND-HAND]`

**GIST (Gilson/Kurtzman)**
- "Discretizes the equations of inhomogeneous solvation theory onto a three-dimensional grid situated around a solute molecule or complex. Snapshots from explicit solvent simulations are used to estimate localized solvation entropies, energies, and free energies associated with the grid boxes." Crucially it covers low-occupancy regions, unlike earlier IST work "largely restricted to analyzing discrete, high-occupancy water sites" — [GIST in AmberTools, *J. Comput. Chem.*](https://www.lehman.edu/faculty/tkurtzman/pdfs/2016-08-05-JCC-GIST-AmberTools.pdf) `[SECOND-HAND]`
- Implementation: `GIST-cpptraj` in CPPTRAJ/AmberTools, plus `GISTPP` post-processing tools; both described by the authors as "open source and freely distributed" — same source `[SECOND-HAND]`. Tutorial exists for the Factor Xa active site — [ambermd.org tutorial 25](https://ambermd.org/tutorials/advanced/tutorial25/) `[SECOND-HAND]`
- Cost is dominated by the MD that feeds it, and convergence is uneven: translational entropy and solute-water energy converge relatively quickly but "orientational entropy requires significantly more sampling" — `[SECOND-HAND]`
- Grid output is lattice-shaped and precomputable, which is why Balius et al. could drop it into docking — "water-displacement energies can be precalculated and stored on a lattice of points, supporting the rapid scoring necessary for large library screens" — [PNAS 2017](https://lehman.edu/faculty/tkurtzman/pdfs/PNAS-2017-Balius-E6839-46.pdf) `[SECOND-HAND]`
- Authors' own caution: "its impact in drug binding sites merits further controlled studies" — `[SECOND-HAND]`
- Pocket detection/ranking: **no.** Used for docking-score augmentation and hotspot reading on known sites.

**3D-RISM**
- Integral-equation theory giving a water oxygen distribution function g_O(r) on a grid, no MD sampling. In the 3,706-structure survey: AmberTools18, ff99SB, SPC/E water, 310 K — [PMC7540010](https://pmc.ncbi.nlm.nih.gov/articles/PMC7540010/) `[PRIMARY-FULL]`
- **Real wall-clock, primary-verified:** "the computation for each ligand-free structure took less than 2 hr", run on the HOKUSAI supercomputer; core counts not given — [PMC7540010](https://pmc.ncbi.nlm.nih.gov/articles/PMC7540010/) `[PRIMARY-FULL]`
- Other reported figures span a wide range: "~30 minutes" average on standard CPUs with AmberTools in one study; "a few hours with a single central processing unit"; and a review's "hours to tens of hours" — `[SECOND-HAND, mutually inconsistent — these are different systems/hardware]`
- A known limitation: the density "can be calculated within minutes, but is difficult to convert it into explicit water molecules" — `[SECOND-HAND]`
- Licence: ships in **AmberTools, free of charge**; exact licence text not verified in this pass.
- Pocket detection/ranking: **no** — see the Yoshidome entry above; the largest attempt stopped short of ranking.

**SZMAP (OpenEye)**
- Semi-continuum: a Poisson-Boltzmann engine (OEZap) with explicit-probe-water sampling, giving per-grid-point water free energy/entropy/enthalpy. **Commercial, OpenEye/Cadence.** — [eyesopen.com/SZMAP](https://www.eyesopen.com/SZMAP), [SZMAP semi-continuum docs](https://docs.eyesopen.com/toolkits/python/szmaptk/semicontinuum.html) `[SECOND-HAND]`
- Only runtime number found is vendor-supplied and backward-looking: "significant speedups are possible for a calculation that took close to 3 hours on one CPU with the previous version of SZMAP" — current-version timing not stated — [OpenEye docs](https://docs.eyesopen.com/toolkits/python/szmaptk/semicontinuum.html) `[SECOND-HAND]`
- Pocket detection/ranking: **no.** Published use is "Evaluating Free Energies of Binding and Conservation of Crystallographic Waters" — i.e. known-site water scoring — [ResearchGate record](https://www.researchgate.net/publication/280103635_Evaluating_Free_Energies_of_Binding_and_Conservation_of_Crystallographic_Waters_Using_SZMAP) `[SECOND-HAND]`

**JAWS / grand-canonical methods**
- `[GAP]` I found no primary source on JAWS in this pass. One protocol description mentions "a short 2 ns MD simulation ... using the Grand Canonical Monte Carlo (GCMC) sampling" `[SECOND-HAND]`. No cost, licence or detection/ranking evidence gathered. **Needs a dedicated pass.**

**WaterFLAP (Cresset/Molecular Discovery)**
- GRID-based water prediction and energetics. **Commercial.** Case study: in carbonic anhydrase II, two ligands with unsubstituted five-membered heteroaromatic rings both displaced a "very unhappy" water with ΔG of **+4.69 kcal/mol**, increasing affinity — [Cresset science resource](https://cresset-group.com/science/science-resources/computational-exploration-of-water-in-ligand-design/) `[SECOND-HAND]`
- Accuracy caveat: one comparison reports WaterFLAP "showed a reduced accuracy in water prediction compared with other approaches" — `[SECOND-HAND]`
- Pocket detection/ranking: **no.**

**Fast non-commercial alternatives in the same family**
- **HydraMap**: statistical potentials derived from ~10,987 PDB crystal structures score positions in a pocket, then cluster into discrete hydration sites; reported to match 3D-RISM and WATsite on external tests while running **30–1,000× faster**; outputs were also used to estimate desolvation energy on water displacement — [HydraMap v2, PubMed 37433022](https://pubmed.ncbi.nlm.nih.gov/37433022/) and v1 `[SECOND-HAND]`. **This is the best candidate for a runnable, fast, non-vendor hydration-thermodynamics input.** Licence not verified.
- **PyRod** — traces waters in MD trajectories, generates "dMIFs"; identified a displaceable-water site in CDK2 — [arXiv 1904.01903](https://arxiv.org/pdf/1904.01903) `[SECOND-HAND]`. MD-dependent, so not cheap.
- **HydroRank** — open-source, clusters MD waters, estimates entropy, heuristically ranks displacement favourability — [github.com/AlessioPrunotto/HydroRank](https://github.com/AlessioPrunotto/HydroRank) `[SECOND-HAND]`. MD-dependent; heuristic scoring, no benchmark seen.
- **GridSolvate** web server for biomolecular hydration properties — [JCIM 2020](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00779) `[SECOND-HAND]`
- `[NOTEWORTHY BUT SPECULATIVE]` A December 2025 arXiv preprint proposes protein-pocket hydration-site prediction on a quantum computer — [arXiv 2512.08390](https://arxiv.org/pdf/2512.08390). Not usable; recorded only so it is not mistaken for a practical route.

### Inferences

- For a per-protein, sub-second-per-structure network input channel, **every method in the brief's list is disqualified by cost** except possibly a HydraMap-class statistical potential. GIST/WaterMap/JAWS require MD; 3D-RISM is ~0.5–2 h even on an HPC node.
- The one architectural lesson worth taking from GIST is the *representation*, not the physics: a per-voxel scalar field on a lattice around the protein is exactly the shape of a CNN/grid input channel, and Balius et al. already proved a precomputed GIST lattice can be consumed by a downstream scorer without disrupting performance.
- The licence picture is decisive for the project: of the six named methods, **three are commercial (WaterMap/Schrödinger, SZMAP/OpenEye, WaterFLAP), two ship free inside AmberTools (GIST, 3D-RISM), and one (JAWS) is unverified.** Anything built on the commercial three cannot be a reproducible benchmark contribution.

### Gaps

- No head-to-head controlled wall-clock benchmark of these tools on the same protein and hardware exists in what I could reach. The 2019 *JCAMD* paper ("Water molecules in protein-ligand interfaces. Evaluation of software tools and SAR comparison", `10.1007/s10822-019-00187-y`) evaluates 3D-RISM, SZMAP, WaterFLAP, WaterRank and WaterMap on common targets and is the right place to look; **I could not open it.**
- No WaterMap per-run wall-clock figure found from any source.
- JAWS entirely uncovered.
- HydraMap licence and code availability unverified.

---

## THE "UNHAPPY WATER" HYPOTHESIS: is there measured evidence that cavities with unfavourable water are more likely to bind?

### Takeaway

The evidence is strong **at the level of individual waters inside an already-known pocket** — including one clean prospective SAR series and one million-water statistical study — but I found **no measurement at the level of whole cavities**, i.e. nobody has shown that a cavity's aggregate water unhappiness predicts whether that cavity binds a ligand at all. The mechanism that would justify the feature is established for water selection, assumed for pocket selection.

### Cited Findings

**Strongest prospective structural evidence (single water, known site):**
- FKBP51: crystal structures showed a conserved "unhappy" water; "of the 28 synthesized sulfonamides, compounds with a methyl group in (S)-configuration displayed significantly improved binding affinity, while all (R)-isomers displayed decreased binding affinity", and co-crystal structures confirmed the (S)-methyl displaced the targeted water — reported via [ESRF spotlight 399](https://www.esrf.fr/home/news/spotlight/content-news/spotlight/spotlight399.html) and the water-in-drug-design review `[SECOND-HAND]`
- Carbonic anhydrase II / WaterFLAP: displacement of a water with ΔG = **+4.69 kcal/mol** raised affinity for two ligands — [Cresset](https://cresset-group.com/science/science-resources/computational-exploration-of-water-in-ligand-design/) `[SECOND-HAND]`

**Largest statistical evidence (ColdBrew, 2026) — primary-verified:**
- Dataset: 242 matched cryo-/room-temperature structure pairs; **71,145 classified cryo waters** (32,901 present at RT, 38,244 absent). Ligand-grouped analysis: 162 groups, 5,604 cryo structures, **>1 million holo waters** and ~0.5 million apo waters. Precomputed predictions released for **>46 million waters** across >100,000 structures — [PMC12822489](https://pmc.ncbi.nlm.nih.gov/articles/PMC12822489) `[PRIMARY-FULL]`
- Water-category counts: holo — 11,981 conserved, 23,299 absent, 20,395 displaceable; apo — 8,409 conserved, 2,834 absent, 6,435 displaceable — [PMC12822489](https://pmc.ncbi.nlm.nih.gov/articles/PMC12822489) `[PRIMARY-FULL]`
- **The key quantitative link between "unhappy" and "displaced":** waters displaced by **polar** ligand atoms had median ColdBrew probability **0.71**, vs **0.38** for those displaced by **nonpolar** atoms — i.e. ligands displace *stable* waters with polar groups and *unstable* waters with hydrophobic groups — [PMC12822489](https://pmc.ncbi.nlm.nih.gov/articles/PMC12822489) `[PRIMARY-FULL]`
- **Agreement between a cheap empirical proxy and full MD+GIST:** Pearson **r = −0.60** overall across four systems (endothiapepsin, Hsp90α, PTP1B, β-lactamase), and **r = −0.62** for waters near the ligand, between apo ColdBrew probability and MD+GIST energy. MD reference was 20 ns production in a 20 Å TIP3P box, NAMD 3.0 on GPUs — [PMC12822489](https://pmc.ncbi.nlm.nih.gov/articles/PMC12822489) `[PRIMARY-FULL]`
- Behavioural evidence from medicinal chemistry practice: "waters with high ColdBrew probabilities were empirically avoided, with some notable exceptions" — [PMC12822489](https://pmc.ncbi.nlm.nih.gov/articles/PMC12822489) `[PRIMARY-FULL]`
- Conformational dependence: in Hsp90α one water was conserved in **88%** of loop-in structures but displaced in **18%** of helical-conformation structures — [PMC12822489](https://pmc.ncbi.nlm.nih.gov/articles/PMC12822489) `[PRIMARY-FULL]`
- The authors explicitly recommend "using relative rankings rather than binary calls" — [PMC12822489](https://pmc.ncbi.nlm.nih.gov/articles/PMC12822489) `[PRIMARY-FULL]`

**Counter-evidence and complications:**
- Not every unhappy water is displaceable: in a GIST case study some waters "had positive solvation free energies but were still enthalpically 'happy', indicating strong interaction with the protein or surrounding water molecules", requiring polar replacement groups — `[SECOND-HAND]`
- Displacement can cost affinity: in a WaterFLAP analysis "one favourable water is displaced by all ligands, which works against their overall affinity" — `[SECOND-HAND]`
- Network, not per-water, effects: in BRD4 "the gains or losses in affinity upon displacement or replacement of water molecules cannot be explained by considering protein-ligand interactions alone" — [JCIM 2024, 10.1021/acs.jcim.4c01291](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01291) `[SECOND-HAND]`
- Terminology is unstandardised: the same concept appears as "high free energy", "frustrated", "unhappy", "low positional stability" — [Expert Opin. Drug Discov. 2025 review](https://www.tandfonline.com/doi/full/10.1080/17460441.2025.2497912) `[SECOND-HAND]`
- Indirect support for the apolar-pocket link: an enhanced-sampling study notes "druggable pockets are known to be apolar/hydrophobic" and altered protein-water interactions to mimic a small ligand — [JACS enhanced-sampling/ligandability paper](https://pubs.acs.org/jacsat/article/doi/10.1021/jacs.6c05778/5326462/Enhanced-Sampling-and-Ligandability-Assessment-to) `[SECOND-HAND]`
- Older review-level claim: including hydration in docking improved binding-pose ranking "to about 89% accuracy" — `[SECOND-HAND, system unspecified, do not cite without finding the source]`

### Inferences

- The mechanism is real but its *direction* is more subtle than the simple hypothesis. ColdBrew's polar/nonpolar split (0.71 vs 0.38) says ligands displace both stable and unstable waters, and which one tells you about the *chemistry of the displacing group*, not about whether the site is bindable. A cavity-level feature built on "mean water unhappiness" would therefore be conflating two signals.
- The more defensible cavity-level formulation suggested by the data is **compositional**: a cavity with a *population* of low-stability waters is a cavity whose volume can be taken over by hydrophobic ligand atoms at low desolvation cost. That is close to, and may be largely redundant with, SiteMap's hydrophobic-enclosure term.
- The ColdBrew r = −0.60 result is the single most useful number in this whole research pass for the project: **a random forest over five cheap crystallographic/geometric metrics recovers ~60% correlation with a 20 ns MD + GIST calculation.** That is direct evidence that the expensive physics is largely approximable by cheap descriptors — the premise the project needs.
- The 88%-vs-18% Hsp90α result warns that any water-derived feature is conformation-dependent, so on apo or predicted structures it will be noisier than on holo crystals. This matters because cavity ranking is evaluated largely on apo/holo-mixed benchmarks.

### Gaps

- **No study measures whether cavities with unfavourable water bind ligands more often than cavities without.** This is the exact gap the project would fill, and it is unfilled. I found no AUC, odds ratio or enrichment for that proposition anywhere.
- ColdBrew's MD+GIST comparison covers only four systems; the r = −0.60 is not a large-N validation.
- No licence stated for the ColdBrew code in the text I read (code at `github.com/TheFischerLab/ColdBrew`, results at Zenodo DOI `10.5281/zenodo.13909324`). The last ~15,700 characters of that paper were not read.

---

## EXPLICIT WATER-SITE PREDICTION: deep models, published weights, usability as an input channel

### Takeaway

This field is mature enough to use: several deep models predict water oxygen positions from structure alone with sub-angstrom accuracy, and at least SuperWater and WaterFlow ship code publicly. But **none** reports per-protein inference wall-clock time, none predicts water *thermodynamics* (only positions), and none has been applied to pocket detection — SuperWater's authors list it as future work.

### Cited Findings

- **SuperWater** (bioRxiv 2024 / *Commun. Chem.* 2025): score-based diffusion model with equivariant GNNs sampling candidate water positions, an **SE(3)-equivariant confidence model** scoring and filtering them, then clustering to merge neighbours. Trained on 17,092 curated PDB files split 13,674 / 1,709 / 1,709, 300 epochs — [bioRxiv 2024.11.18.624208](https://www.biorxiv.org/content/10.1101/2024.11.18.624208v1.full) `[PRIMARY-FULL]`
  - Accuracy (figure-derived, not headline): at 1.0 Å cutoff ~90% precision at ~27% coverage, ~66% precision at 54% coverage; at 0.5 Å ~70% precision at 27% coverage, ~46% at 40% coverage; MAD for true positives at cap 0.5 is **0.3 ± 0.06 Å** — [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.11.18.624208v1.full) `[PRIMARY-FULL]`
  - Authors' own comparison: "at a fixed 25% coverage SuperWater maintains 95% precision, outperforming HydraProt by about eight percentage points and GalaxyWater-CNN by roughly fifteen" — `[SECOND-HAND, authors' benchmark]`
  - Code: `github.com/kuangxh9/SuperWater`; processed datasets/caches on Zenodo DOI `10.5281/zenodo.14166655`. **Licence not stated in the text; model weights not explicitly mentioned in the manuscript body I read.** Hardware: DGX A100 cluster acknowledged. **No inference wall-clock reported.** — [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.11.18.624208v1.full) `[PRIMARY-FULL]`
  - Pocket use: "Neither. ... binding pockets and protein-protein interfaces appear only as case-study contexts, and the paper lists 'binding site predictions' as a possible future application" — [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.11.18.624208v1.full) `[PRIMARY-FULL]`
- **WaterFlow** (`github.com/prism-science/WaterFlow`): two-stage — a candidate generator trained with flow matching, plus a confidence model that ranks and scores candidates; README states shipped models can be run on user structures. **Clearest open-weights candidate; licence unconfirmed.** — [GitHub](https://github.com/prism-science/WaterFlow) `[SECOND-HAND]`
- **HydraProt** (*JCIM* 2024, `10.1021/acs.jcim.3c01559`): predicts water oxygen positions with a 3D U-Net plus an MLP; authors claim it "surpasses existing state-of-the-art approaches in precision and recall". Weight availability unconfirmed. — [JCIM](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01559) `[SECOND-HAND]`
- **GalaxyWater-CNN** (*JCIM* 2022, `10.1021/acs.jcim.2c00306`): 3D-CNN predicting water positions on protein chains, protein-protein interfaces **and protein-compound binding sites**, trained on high-resolution crystal structures with waters — [JCIM](https://pubs.acs.org/doi/10.1021/acs.jcim.2c00306) `[SECOND-HAND]`. Earlier sibling **GalaxyWater-wKGB** uses a knowledge-based potential accounting for solvent accessibility and protein-water H-bond orientation `[SECOND-HAND]`
- **WatGNN**: SE(3)-equivariant GNN predicting water positions from protein structure using probe nodes placed near non-carbon atoms, "avoiding the dense grid calculations used in earlier CNN-based approaches" — [Zenodo record 20780410](https://zenodo.org/records/20780410) `[SECOND-HAND]`. Architecturally the closest to an EquiCave-style backbone.
- **hotWater** (bioRxiv 2020): residual network identifying surface "water hot spots"; can score existing waters from a PDB file **or scan a protein surface de novo**; blind-tested on a newly solved structure with waters removed, and reported higher recall than two electron-density-based algorithms at resolutions >2.6 Å — [bioRxiv 2020.04.20.050393](https://www.biorxiv.org/content/10.1101/2020.04.20.050393v1.full) `[SECOND-HAND]`
- **Instantaneous hydration properties from static structures** (*Commun. Chem.* 2020, `s42004-020-00435-5`): computes protein interaction energies with several probe atoms on a grid, feeds them to a CNN to predict **hydration occupancy**, framed as segmentation with thresholds such as "0.02 represents approximately bulk water density" — [Nature Commun. Chem.](https://www.nature.com/articles/s42004-020-00435-5) `[SECOND-HAND]`. **This is the only model found that predicts a hydration *property field* rather than discrete positions, and it is the nearest existing thing to a drop-in input channel.**
- **gr Predictor** (bioRxiv 2022): deep model predicting hydration *structures* (g(r)) around proteins — [bioRxiv 2022.04.18.488616](https://www.biorxiv.org/content/10.1101/2022.04.18.488616.full.pdf) `[SECOND-HAND]`. Also a field-valued predictor; worth checking for weights.
- Historical baseline worth knowing: **Consolv** reached "75% accuracy in predicting whether a binding site water molecule would be displaced or not", but "as Consolv used crystallographic temperature factors as structural descriptors, it cannot be applied to predicted water sites" — `[SECOND-HAND]`
- Cheap non-ML baseline: **WaterDock** predicted water sites after stripping ligands from the Astex Diverse Set — [PLoS ONE 7:e32036](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0032036) `[SECOND-HAND]`
- MD-derived hydration sites reproduced **73%** of crystal binding-site waters with the ligand present, and **58%** when simulations were repeated without the co-crystallised ligand — [JCIM 2017, 10.1021/acs.jcim.7b00520](https://pubs.acs.org/doi/10.1021/acs.jcim.7b00520) `[SECOND-HAND]`. **This 73→58% drop is the right order-of-magnitude expectation for how much water signal survives on apo structures.**

### Inferences

- Two of these — the 2020 `s42004-020-00435-5` occupancy CNN and gr Predictor — output a *continuous field*, which is directly consumable as an extra input channel on an existing lattice, with no clustering or discrete-water step. Discrete-position predictors (SuperWater, HydraProt, GalaxyWater-CNN, WaterFlow) need a density-smearing step first, which adds a hyperparameter and discards the confidence scores unless they are carried through as weights.
- Using a predicted-water channel creates a **dependency-and-leakage problem**: every one of these models was trained on crystal waters from the PDB, so their training sets overlap heavily with COACH420/HOLO4K/scPDB. Any gain from such a channel must be checked against the water model's own training-set membership, or reviewers will call leakage.
- The 58% apo recall figure plus the Hsp90α 88%/18% conformational result jointly suggest a predicted-water channel will be substantially weaker on apo and AlphaFold structures than the holo numbers imply — which is where cavity ranking is hardest.
- Nothing in this family predicts water *free energy*. Turning a predicted-position model into a thermodynamics channel would require either ColdBrew-style stability prediction on top, or training against GIST maps directly, which nobody has published.

### Gaps

- **No inference wall-clock for any of these models.** This is the single missing number needed to decide whether a predicted-water channel fits a sub-second budget. SuperWater is a diffusion model with a confidence rescoring stage, so it is almost certainly seconds-to-minutes per protein, not sub-second, but I could not verify this.
- Licences unconfirmed for SuperWater, WaterFlow, HydraProt, GalaxyWater-CNN, WatGNN, gr Predictor. **All require a repository check before any build decision.**
- Whether SuperWater's Zenodo deposit contains trained weights or only processed data is unresolved.

---

## HYDROPHOBIC ENCLOSURE, SiteMap's Dscore, and cheap geometric desolvation proxies

### Takeaway

SiteMap's Dscore is the classic druggability score and is, in effect, already a cheap desolvation proxy: it is a weighted sum of enclosure, size and a **negative hydrophilicity** term, which is precisely "buriedness discounted by polarity". It is commercial, and — important for the brief — the published Dscore thresholds are inconsistent across the literature, so cross-study Dscore comparisons are unsafe.

### Cited Findings

- Dscore is "computed from physiochemical descriptors generated by SiteMap, and is a weighted sum with contributions from three components": **enclosure, size (number of site points), and a negative hydrophilic term** — reported across the druggability literature `[SECOND-HAND]`
- Dscore differs from SiteScore mainly in how hard it penalises polarity: Schrödinger's own note says "undruggable and difficult sites typically are much more hydrophilic and much less hydrophobic than druggable sites" — [Schrödinger support article 320](https://my.schrodinger.com/support/article/320) `[SECOND-HAND]`
- Mechanically, "SiteScore uses a cap of 1.0 for the hydrophilic score, whereas Dscore is not capped" (bromodomain study); and "highly polar active sites are more penalized by the Dscore function than SiteScore" (PTP study) — [J. Med. Chem. 2012 bromodomains](https://pubs.acs.org/doi/10.1021/jm300346w); [DDDT PTP study](https://www.dovepress.com/druggability-analysis-and-classification-of-protein-tyrosine-phosphata-peer-reviewed-fulltext-article-DDDT) `[SECOND-HAND]`
- **`[CONFLICT]` Published Dscore thresholds disagree sharply:**
  - Halgren's commonly cited values: druggable > **1.108**, difficult-to-drug < **0.871** `[SECOND-HAND]`
  - A patent using SiteMap: undruggable < **0.83**, difficult **0.83–0.98**, druggable > **0.98** `[SECOND-HAND]`
  - A three-tier scheme in other papers: difficult ≤ **0.7**, intermediate **0.7–0.8**, highly druggable > **0.8** `[SECOND-HAND]`
  - A PPI study added a "moderately druggable" class for "marginal targets that obtain a Dscore of just less than 0.8" — [Sci. Rep. 12, s41598-022-12105-8](https://www.nature.com/articles/s41598-022-12105-8) `[SECOND-HAND]`
- Primary reference: Halgren T.A., "Identifying and Characterizing Binding Sites and Assessing Druggability", *J. Chem. Inf. Model.* 49(2):377-389, 2009, DOI `10.1021/ci800324m` — [Google Scholar record](https://scholar.google.com/scholar_lookup?title=Identifying+and+characterizing+binding+sites+and+assessing+druggability&amp=&author=TA+Halgren&amp=&publication_year=2009&amp=&journal=J+Chem+Inf+Model&amp=&pages=377-389&amp=&doi=10.1021/ci800324m&amp=&pmid=19434839) `[SECOND-HAND — full text not read]`
- SiteMap is **commercial (Schrödinger)**; the underlying method ("Understanding and Predicting Druggability. A High-Throughput Method for Detection of Drug Binding Sites") is the companion paper — [ResearchGate record](https://www.researchgate.net/publication/45504065_Understanding_and_Predicting_Druggability_A_High-Throughput_Method_for_Detection_of_Drug_Binding_Sites) `[SECOND-HAND]`
- **Fpocket**, the open-source comparator, reports top-1/top-3 of **83%/92%** on bound and **69%/94%** on unbound structures, and its authors state its ranking "does not reflect drugability"; its separate druggability score "accounts for not only the geometry of a pocket but also the chemical environment of a pocket" — [Fpocket, BMC Bioinformatics 10:168](https://link.springer.com/article/10.1186/1471-2105-10-168) `[SECOND-HAND]`
- Depth-weighted (not solvation) ranking as a cheap alternative: the Euclidean Distance Transform method scores candidate pockets with "a weighted volume score, which sums the squared distance values from the nearest voxel of the outer solvent-accessible surface across the pocket's voxels" — [PeerJ preprint 27314](https://peerj.com/preprints/27314.pdf) `[SECOND-HAND]`
- Scale of structure-based druggability assessment already done with SiteMap-class tools: a mammalian-proteome study with light flexibility — [PLoS Comput. Biol. 10:e1003741](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1003741) `[SECOND-HAND]`
- For context on what modern learned rankers achieve on an adjacent task: PocketMiner reaches **ROC AUC 0.87** discriminating cryptic-pocket-forming residues — [Nat. Commun. 14, s41467-023-36699-3](https://www.nature.com/articles/s41467-023-36699-3) `[SECOND-HAND]`

### Inferences

- **Dscore already is the cheap desolvation feature.** Enclosure × (hydrophobic − hydrophilic) measures something buriedness alone does not: it distinguishes a deep *apolar* pocket (where water is frustrated and displaceable at low cost) from a deep *polar* pocket (where water is well satisfied and expensive to displace). Pure buriedness cannot tell those apart, and that distinction is the entire physical content of the unhappy-water hypothesis.
- This has a sharp consequence for the project's novelty claim: **any new cheap hydration feature must be ablated against an enclosure-weighted hydrophobicity baseline**, not against raw buriedness, or the result will be a reinvention of Dscore. A reviewer who knows SiteMap will ask this first.
- The threshold disagreement means Dscore cannot be used as a label or as a comparator number taken from another paper; it must be recomputed in-house (which requires a Schrödinger licence) or replaced with an open reimplementation of the same three terms.
- Fpocket's 83%/92% bound and 69%/94% unbound top-1/top-3 sit suspiciously close to dPredGB's 85%/69% bound/unbound top-1 on a similarly vintage benchmark. **This is the strongest reason to doubt that dPredGB's desolvation term added anything over geometry** — the numbers are nearly indistinguishable, on different test sets. Resolving this requires the dPredGB full text.

### Gaps

- I could not read Halgren 2009, so the actual Dscore coefficients, the definition of "enclosure", and the authoritative thresholds are unverified. The SiteMap user manual is the other authoritative source and is licence-gated.
- No benchmark of Dscore as a *cavity-list ranker* found — the druggability literature uses it to classify known pockets, which is the distinction the brief insists on. I found no paper applying Dscore to rank a full candidate cavity list on COACH420/HOLO4K/LIGYSIS.
- No wall-clock cost for SiteMap found.

---

## BURIED WATERS IN CRYSTAL STRUCTURES: how many, are they predictive, and is stripping them losing signal?

### Takeaway

Ordered waters are abundant in deposited structures (hundreds of thousands to millions across the PDB, with ~620,000 in 2,403 PDBbind refined-set structures alone), and ColdBrew establishes that they carry real, extractable information about ligand displaceability. But **no source I found measures whether the mere presence of ordered waters in a cavity predicts that the cavity binds a ligand**, so whether pocket predictors lose signal by stripping waters is formally unanswered — and there is a strong methodological reason to strip them anyway.

### Cited Findings

- **Counts.** 2,403 of 3,706 ligand-free structures derived from the PDBbind refined set v2017 (4,154 complexes) had crystallographic waters, totalling **~620,000 CWs** — [PMC7540010](https://pmc.ncbi.nlm.nih.gov/articles/PMC7540010/) `[PRIMARY-FULL]`
- ColdBrew released predictions for **>46 million waters** across **>100,000 structures** (the paper says ">100,000" in one place and ">107,000" in another) — [PMC12822489](https://pmc.ncbi.nlm.nih.gov/articles/PMC12822489) `[PRIMARY-FULL]`
- **95% of all PDB structures were collected under cryogenic conditions** — [PMC12585415 abstract](https://pmc.ncbi.nlm.nih.gov/articles/PMC12585415/) `[PRIMARY-ABSTRACT]`. This is the reason raw water counts are not a clean physical signal: cryo-trapping inflates them relative to room temperature.
- **Of ColdBrew's 71,145 classified cryo waters, 32,901 were present at RT and 38,244 were not** — i.e. **54% of ordered cryo waters are artefacts of freezing** — [PMC12822489](https://pmc.ncbi.nlm.nih.gov/articles/PMC12822489) `[PRIMARY-FULL]`
- **Which hydration quantities are predictive, and how much.** ColdBrew's random forest over five metrics (Bnorm, RSCC, EDIA, SASA, H-bond count) reaches **AUC 0.810 train / 0.799 test / 0.804 overall**, accuracy 73%/72%; AUC rises to **0.838** for waters within 7 Å of a ligand. Single-metric AUCs: **EDIA 0.747, Bnorm 0.734, RSCC 0.725, SASA 0.720, H-bond count 0.672** — [PMC12822489](https://pmc.ncbi.nlm.nih.gov/articles/PMC12822489) `[PRIMARY-FULL]`
- Calibration: 84% of RT-present waters scored >0.35 and 91% of absent waters scored <0.65; binary accuracy 80% in the 0–0.35 band, 82% in 0.65–1, and only **58% in the ambiguous 0.35–0.65 band** — [PMC12822489](https://pmc.ncbi.nlm.nih.gov/articles/PMC12822489) `[PRIMARY-FULL]`
- Positional corroboration: median displacement of stable waters **0.23 Å** vs **0.89 Å** for low-probability waters — [PMC12822489](https://pmc.ncbi.nlm.nih.gov/articles/PMC12822489) `[PRIMARY-FULL]`
- Which CWs are structurally meaningful: highly hydrated CWs (g_O > 4) "tend to form hydrogen bonds with protein N/O atoms", moderately hydrated ones "tend to form weaker CH···O contacts" — [PMC7540010](https://pmc.ncbi.nlm.nih.gov/articles/PMC7540010/) `[PRIMARY-FULL]`
- The Consolv precedent is the explicit warning about B-factor-based features: 75% accuracy, but "as Consolv used crystallographic temperature factors as structural descriptors, it cannot be applied to predicted water sites" — `[SECOND-HAND]`

### Inferences

- **Stripping waters is methodologically correct for a pocket predictor, and the question in the brief is slightly mis-posed.** Three of ColdBrew's five predictive metrics (Bnorm, RSCC, EDIA) are *crystallographic refinement quantities*, not physical ones. A model that reads deposited waters learns resolution, temperature, refinement protocol and depositor practice. Since 54% of cryo waters are not present at RT, and 95% of the PDB is cryo, a water-count feature is substantially a proxy for crystallisation conditions. On apo or AlphaFold inputs it is unavailable, so a model that depends on it cannot be deployed where cavity ranking is actually needed.
- **Signal is nevertheless being lost, but it is recoverable without reading the deposited waters.** The two ColdBrew metrics that do *not* require experimental data — **SASA (AUC 0.720) and H-bond count (AUC 0.672)** — are computable from coordinates alone in milliseconds, and together they are not far behind the full five-metric model's 0.804. That is the quantitative case for the cheap version: most of the water-stability signal is geometric and chemical, not crystallographic.
- Any experiment using deposited waters must be framed as an **upper-bound/oracle study** ("how much would a perfect water channel be worth?"), with the deployable version using predicted or geometric surrogates. Reporting the oracle as a result would be indefensible.

### Gaps

- **Unanswered and important: the base rates.** I found no number for what fraction of *true* binding cavities contain ≥1 ordered water versus what fraction of *decoy* cavities do. Without those two numbers the "are buried waters predictive of bindability" question cannot be answered, and I could not find them. This is computable in-house from any of the standard benchmarks in an afternoon and would be a genuinely novel measurement.
- I found no paper that explicitly ablates water-stripping in a pocket predictor. The water-prediction literature ablates *ligands* (73% → 58% crystal-water recall) but not waters-as-input-to-pocket-detection.
- Whether COACH420/HOLO4K/LIGYSIS/scPDB preprocessing retains or discards HOH records, and whether the predictors evaluated on them received waters, was not established from sources in this pass; it should be read off the benchmark construction scripts directly.

---

## THE CHEAP VERSION: the least expensive defensible quantity, assessed

### Takeaway

The best-supported cheap option is **not** a new invention: it is a continuum/generalized-Born desolvation free energy per cavity, which **already has a benchmarked precedent in dPredGB** (69% apo / 85% holo top-1), combined with a per-lattice-point polar/apolar contact ratio. The evidence that cheap descriptors suffice is quantitative and comes from two independent places: ColdBrew's geometric-only metrics reach AUC 0.72/0.67 against a full-model 0.80 and correlate r = −0.60 with 20 ns MD+GIST, and HydraMap matches 3D-RISM 30–1000× faster.

### Cited Findings

Assessing each candidate named in the brief:

**(1) Continuum solvation term (generalized Born) — strongest evidence, has precedent**
- dPredGB ranks the full candidate cavity list by GB desolvation free energy, reaching top-1 of 69% (unbound) / 85% (bound) — [OSTI 1059618](https://www.osti.gov/biblio/1059618) `[PRIMARY-ABSTRACT]`
- GB is routinely used as a *rapid* desolvation term in docking: one approach relates "the Generalized-Born effective Born radii for every ligand atom to a fractional desolvation and then [uses] this fraction to scale an atom-by-atom decomposition of the full transfer free energy", published as "**Rapid** Context-Dependent Ligand Desolvation in Molecular Docking" — [JCIM, 10.1021/ci100214a](https://pubs.acs.org/doi/abs/10.1021/ci100214a) `[SECOND-HAND]`
- SZMAP's engine is Poisson-Boltzmann (OEZap), which OpenEye describes as "simple, effective, and ... fast" — [OpenEye docs](https://docs.eyesopen.com/toolkits/python/szmaptk/semicontinuum.html) `[SECOND-HAND]`

**(2) Per-point count of H-bond partners vs hydrophobic contacts — direct quantitative support**
- ColdBrew single-metric AUCs for water displaceability: **SASA 0.720, H-bond count 0.672**, against a 5-metric model at **0.804** — [PMC12822489](https://pmc.ncbi.nlm.nih.gov/articles/PMC12822489) `[PRIMARY-FULL]`
- The polar/apolar distinction is the operative one empirically: median ColdBrew probability **0.71** for waters displaced by polar ligand atoms vs **0.38** for nonpolar — [PMC12822489](https://pmc.ncbi.nlm.nih.gov/articles/PMC12822489) `[PRIMARY-FULL]`
- GalaxyWater-wKGB's knowledge-based potential is built on exactly these two ingredients: "solvent accessibility and protein-water hydrogen-bond orientation" `[SECOND-HAND]`
- 3D-RISM's own structural finding supports an H-bond-count proxy: highly hydrated sites (g_O > 4) are those forming H-bonds to protein N/O — [PMC7540010](https://pmc.ncbi.nlm.nih.gov/articles/PMC7540010/) `[PRIMARY-FULL]`

**(3) Kyte-Doolittle-weighted enclosure — supported by analogy to Dscore, not directly measured**
- Dscore is already "a weighted sum with contributions from ... enclosure, size, and a negative hydrophilic term" `[SECOND-HAND]`
- P2Rank already computes distance-weighted sums of atomic properties "within a 6 Å neighborhood" including "partial charge, hydrophobicity", over "more than 30 attributes describing physical-chemical properties of amino acids in the local neighborhood" — [P2Rank repo](https://www.github.com/rdk/p2rank), [P2Rank slides, UPOL](https://www.kfc.upol.cz/wp-content/uploads/2024/01/7ADD_24_Marian.pdf) `[SECOND-HAND]`

**(4) Simple hydration-shell count — weakest; confounded**
- 54% of ordered cryo waters are absent at RT, and 95% of the PDB is cryo — [PMC12822489](https://pmc.ncbi.nlm.nih.gov/articles/PMC12822489) `[PRIMARY-FULL]`, [PMC12585415](https://pmc.ncbi.nlm.nih.gov/articles/PMC12585415/) `[PRIMARY-ABSTRACT]`

**Speed evidence that cheap works:**
- HydraMap: statistical potentials from ~10,987 crystal structures, "30~1000 times faster than 3D-RISM and WATsite while giving comparable results on external test sets", and its output was used to estimate desolvation energy on water displacement — [HydraMap v2, PubMed 37433022](https://pubmed.ncbi.nlm.nih.gov/37433022/) `[SECOND-HAND]`
- ColdBrew's full model is a random forest over five scalars and the authors contrast it with "expensive MD/GIST calculations", describing it as scalable — though **no wall-clock is given** — [PMC12822489](https://pmc.ncbi.nlm.nih.gov/articles/PMC12822489) `[PRIMARY-FULL]`

### Inferences

Ranked recommendation, with the reasoning made explicit so the report-writer can attribute it as inference rather than finding:

1. **Per-lattice-point polar/apolar contact asymmetry** is the best first move. It is O(neighbours) per point, trivially sub-second, needs no external tool or licence, and has the most direct quantitative backing (AUC 0.672–0.720 for the per-water displaceability task that the hypothesis rests on). It is also the only candidate that produces a *field*, so it drops straight into an existing lattice as an extra channel rather than becoming a per-cavity scalar.
2. **A GB/PB desolvation scalar per cavity** is the right second channel, because it is the one with a benchmarked cavity-ranking precedent. Its cost is the open question — GB over a whole protein with per-cavity decomposition is plausibly seconds, not sub-second — so it may belong as an auxiliary per-cavity feature in the ranker rather than a per-point input.
3. **Kyte-Doolittle-weighted enclosure** should be built as the *control*, not the contribution, because it is approximately Dscore and approximately already inside P2Rank's 30+ attributes. If the new water channel does not beat it, there is no finding.
4. **Hydration-shell count from deposited waters** should be used only as an oracle upper bound, for the reasons in the previous section.

On the auxiliary-target option, which the objective raises: predicting a *water-stability field* as an auxiliary task is more defensible than predicting water *positions*, because stability is what the hypothesis is about and because ColdBrew provides >46 million labelled waters with calibrated probabilities as a ready-made, openly deposited supervision signal (Zenodo `10.5281/zenodo.13909324`). That is, to my knowledge from this pass, an unexploited labelled resource at a scale that suits auxiliary-task pretraining. Two cautions: the labels are cryo-vs-RT persistence, not free energy, and the ambiguous 0.35–0.65 band is only 58% accurate, so those waters should be down-weighted or masked.

### Gaps

- **No wall-clock measurement exists for any of the four candidate cheap features.** The "<1 s per protein" target in the brief is my own assumption for options (1) and (3) based on their O(N) neighbour-counting form, and is **unverified** for (2) — GB has no cited per-protein timing anywhere I could reach. This must be measured in-house before the budget claim is made.
- Whether a continuum-solvation channel adds anything over P2Rank's existing hydrophobicity attributes is unmeasured by anyone, including in the P2Rank papers.
- I could not obtain P2Rank's exact feature table, so I cannot state whether a desolvation-like term is already present. The authoritative sources are `config/default.groovy` in the repo and the feature table in the 2018 *J. Cheminform.* paper — both checkable locally.
- HydraMap's licence and runnability are unverified, and it is the one fast tool that would let a true hydration-thermodynamics channel be tested at benchmark scale.
