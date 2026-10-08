# Protonation states and metals/cofactors in structure-based binding-site prediction

**Verification key** (applied to every bullet):
- `[P-READ]` = I read the primary source text myself in this session (full text, extracted PDF text, or repository file).
- `[P-SNIP]` = primary source, but only a search-engine snippet/abstract of it; I did not read the surrounding text.
- `[2ND]` = second-hand (review, another paper describing a third tool, aggregator page).

**Scope note:** current as of 2026-10-08. Where a paper does not state something, it is recorded as **not stated** rather than inferred.

---

## PART A, Q1 — How do binding-site predictors treat His/Asp/Glu/Cys/Lys protonation? Does any model condition on predicted protonation state?

### Takeaway
No structure-based binding-site predictor I could find conditions on a predicted protonation state or pKa. Protonation enters, at most, indirectly and silently: through Open Babel partial charges and SMARTS donor/acceptor flags computed on heavy atoms (Kalasanty, DeepSurf), through a `formal_charge` node feature (GrASP), or not at all (VN-EGNN, which is residue-level). Several methods explicitly strip hydrogens or heteroatoms. The one place protonation is done properly is *upstream dataset curation* (sc-PDB), whose protonated structures are then reduced back to heavy atoms by every consumer.

### Cited Findings
- Kalasanty: "Proteins were described with 18 atomic features used in our previous project [17]" — i.e. the Pafnucy featurization; no residue protonation state, no pKa, no hydrogens in the grid. `[P-READ]` — [Kalasanty preprint, arXiv:1904.06517](https://arxiv.org/pdf/1904.06517) (text extracted locally)
- The Pafnucy feature set that Kalasanty and DeepSurf inherit is 19 features: "9 bits (one-hot or all null) encoding atom types: B, C, N, O, P, S, Se, halogen and metal"; plus hybridization (1–3), heavy-atom valence, heteroatom valence, five SMARTS bits (hydrophobic, aromatic, acceptor, donor, ring), partial charge (float), and molecule type (+1 ligand / −1 protein). Binding-site models use the 18 without the molecule-type bit. Protonation can only be felt through `partial charge` and the `acceptor`/`donor` SMARTS bits. `[P-READ]` — [Stepniewska-Dziubinska et al., Bioinformatics 34(21):3666 (Pafnucy)](https://academic.oup.com/bioinformatics/article/34/21/3666/4994792)
- Kalasanty's structures are loaded with Open Babel; 304 binding sites were discarded "because of errors when loading their corresponding protein structures with Open Babel." So partial charges / donor-acceptor bits come from Open Babel defaults, not from a pKa calculation. `[P-READ]` — [Kalasanty preprint](https://arxiv.org/pdf/1904.06517)
- DeepSurf adopts the same scheme: "We adopt here the featurization scheme initially introduced by [17] and used also in Kalasanty [23], which consists of 18 chemical features calculated per protein atom. Each grid voxel receives the features of the atoms inside it." `[P-READ]` — [DeepSurf, arXiv:2002.05643](https://arxiv.org/pdf/2002.05643)
- DeepSurf explicitly removes hydrogens: "Before the final step of binding sites extraction (step 12 in Algorithm 1), hydrogen atoms are removed from the protein in order binding sites to maintain only heavy atoms." `[P-READ]` — [DeepSurf, arXiv:2002.05643](https://arxiv.org/pdf/2002.05643)
- GrASP: "nodes represent heavy atoms, and edges are drawn between all pairs of atoms within 5 Å"; node features are "both atomic features such as formal charge and residue features such as residue name." Formal charge is the only charge-related feature named in the main text; whether it reflects an assigned protonation state is **not stated**. `[P-READ]` — [GrASP, J Chem Inf Model / PMC10402091](https://pmc.ncbi.nlm.nih.gov/articles/PMC10402091/)
- VN-EGNN is residue-level and cannot represent protonation at all: "We used the position of the α-carbons as residue node locations"; "For the initial residue node features we used pre-trained ESM-2 protein embeddings." `[P-READ]` — [VN-EGNN, arXiv:2404.07194 HTML v1](https://arxiv.org/html/2404.07194v1)
- DeepPocket discards all heteroatoms before featurization: "We first clean the input structure by removing all heteroatoms and solvent molecules from the protein structure using the Biopython47 library." `[P-READ]` — [DeepPocket, J Chem Inf Model 2022 (author PDF)](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2022_01/Deeppocket.pdf)
- P2Rank: I grepped the complete Europe PMC full text of the primary paper. There is **no mention of hydrogens, protonation, pKa, or HETATM atom-level handling anywhere in the paper** — only a HET-group filter for defining *ligands* (see Part B Q4). Protonation handling is therefore **not stated**. `[P-READ]` — [P2Rank, J Cheminform 10:39 (2018), Europe PMC full text PMC6091426](https://europepmc.org/article/MED/30109435)
- Upstream, sc-PDB *does* protonate: hydrogens are added, "arginine and lysine are positively charged and aspartic and glutamic acids are negatively charged," other residues are protonated "according to ionized templates built from HET group dictionary," and the hydrogen-bond network is optimized with BioSolveIT Hydescorer. **No pH is stated.** `[P-READ]` — [sc-PDB: a 3D-database of ligandable binding sites — 10 years on, NAR 2015 / PMC4384012](https://pmc.ncbi.nlm.nih.gov/articles/PMC4384012/)

### Inferences
- The sc-PDB protonation effort is largely wasted on the downstream models: sc-PDB protonates, VolSite derives pharmacophoric cavity labels from the protonated structure, and then Kalasanty/DeepPocket/DeepSurf/GrASP throw the hydrogens away and learn from heavy atoms. The protonation information survives only inside the *labels*, not the *inputs* — an asymmetry nobody in these papers comments on.
- Because His/Asp/Glu/Cys/Lys are distinguished only by residue name and heavy-atom topology, a model cannot in principle tell HID from HIE from HIP, or neutral Cys from thiolate. The pharmacophore of a pocket is therefore represented at residue-type resolution, not at protonation resolution.

### Gaps
- **No binding-site predictor that conditions on predicted protonation state was found.** I searched for this directly and found none; no paper reports a measurement of such a conditioning. This is a genuine hole in the literature, not just a hole in my search — but I cannot prove absence.
- GrASP's full feature table is in Supporting Information which I could not retrieve; whether it contains a protonation or pKa feature is unverified.
- EquiPocket's preprocessing is unavailable: arXiv v5 is **withdrawn** by the authors ("technical elaboration and experimental design" need "substantial restructuring and revision") and no PDF is served. `[P-READ]` — [arXiv:2302.12177](https://arxiv.org/abs/2302.12177)

---

## PART A, Q2 — pKa prediction tools: accuracy, wall-clock cost, code, licence

### Takeaway
Six usable families: PROPKA3 (empirical, fastest classical, LGPL-2.1), H++ / PypKa / DelPhiPKa (Poisson–Boltzmann, minutes to tens of minutes per protein), and the ML generation DeepKa / pKAI / pKALM (sub-second to seconds). Accuracy differences are small (RMSE ~0.8–1.1 pKa units) and an independent benchmark warns that no method clearly beats null baselines. For a binding-site pipeline, the decisive fact is cost: PypKa needs up to 24 min for a 1359-residue protein, while pKALM claims ~5000 pKa/s.

### Cited Findings

**PROPKA 3**
- Licence: GNU LGPL v2.1 per the official docs; PyPI metadata agrees ("GNU Lesser General Public License v2 (LGPLv2) (LGPL v2.1)"); one older conda listing says LGPL-3.0 — conflict, check the repo LICENSE. `[P-SNIP]` — [propka.readthedocs.io](https://propka.readthedocs.io/); [PyPI propka](https://pypi.org/project/propka/)
- Speed: the pKALM benchmark reports ~38 pKa values/s for PROPKA (fastest of the classical set). `[P-SNIP]` — [pKALM preprint, bioRxiv 2024.09.16.613101](https://www.biorxiv.org/content/10.1101/2024.09.16.613101v1.full.pdf)
- A historical author claim is "within a couple of seconds" per protein. `[2ND]` — [VEGA PropKa service page](https://www.ddl.unimi.it/vegaol/propka_about.htm)

**H++**
- Wall clock, from the server FAQ: "approximately 18 sec for a molecule of 12 titratable sites such as 1vii, 5 minutes for 111 titratable sites (1AD2), and 45 minutes for 360 titratable sites (1KX5)." Server-load dependent and dated. `[P-SNIP]` — [H++ FAQ](http://newbiophysics.cs.vt.edu/H++/faq.php)
- Licence: **not stated** on any page I reached. It is a web service; the 2012 paper describes it as a freely available open-source web server, but no current licence text was found. `[2ND]` — [bio.tools/hplusplus](https://bio.tools/hplusplus)

**PypKa**
- Licence: LGPL-3.0 for the Python module, but the DelPhi solver dependency is **proprietary** and needs its own licence download. This is a real blocker for redistribution. `[P-SNIP]` — [PypKa docs](https://pypka.readthedocs.io/); [PypKa GitHub mirror](https://github.com/sailfish009/PypKa)
- Wall clock (the only properly tabulated numbers I found for any PB method): 12 s for PDB 1A1W (83 residues, 23 titratable) up to **24 min 06 s** for 1CB5 (1359 residues, 405 titratable), run in parallel on 16 cores. pKAI on the same set: ~1–4 s. pKPDB lookup: ~0.5 s. `[P-SNIP]` — [PypKa server, NAR 52(W1):W294 (2024)](https://academic.oup.com/nar/article/52/W1/W294/7645774)
- pKPDB provides precomputed pKa for >200k PDB and AlphaFold DB structures — i.e. for benchmark proteins you can often skip the calculation entirely. Server is free including commercial use; API rate-limited to 100 requests/hour. `[P-SNIP]` — [PypKa server, NAR 2024](https://academic.oup.com/nar/article/52/W1/W294/7645774)
- Throughput in the pKALM comparison: 0.86 pKa/s, 3 h 44 min for its test set. `[P-SNIP]` — [pKALM preprint](https://www.biorxiv.org/content/10.1101/2024.09.16.613101v1.full.pdf)

**DelPhiPKa**
- "DelPhi based open source C++ program, allowing to predict pKa's of ionizable groups of proteins, RNAs and DNAs." Repository `delphi001/DelphiPka`, latest release v2.3, 18 Apr 2018 — effectively unmaintained. Licence file **not verified**. `[P-SNIP]` — [DelPhiPKa manual](http://compbio.clemson.edu/pka_webserver/assets/manual/DelPhiPKa_User_Manual.html); [GitHub delphi001/DelphiPka](https://github.com/delphi001/DelphiPka)

**DeepKa**
- Developer-reported accuracy vs PROPKA on experimental data: MAE 0.81 vs 0.87, RMSD 1.05 vs 1.12; better on His and Lys, PROPKA better on Glu. `[P-SNIP]` — [DeepKa, ACS Omega 2021](https://pubs.acs.org/doi/10.1021/acsomega.1c05440)
- A web server exists (2024). Throughput ~7 pKa/s per the pKALM comparison. Code licence **not verified**. `[2ND]` — [DeepKa Web Server, JCIM 2024](https://www.researchgate.net/publication/379307545_DeepKa_Web_Server_High-Throughput_Protein_pKa_Prediction); [pKALM preprint](https://www.biorxiv.org/content/10.1101/2024.09.16.613101v1.full.pdf)

**pKAI / pKAI+** (2022, often the right speed/accuracy trade-off)
- pKAI+ RMSE 0.98 vs PypKa 1.07 vs PROPKA3 1.11 — but reported by the pKAI authors. `[P-SNIP]` — [pKAI, J Chem Theory Comput 2022](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00308); [pKAI GitHub](https://github.com/bayer-science-for-a-better-life/pKAI)
- Code at `bayer-science-for-a-better-life/pKAI`, `pip install pkai`. Licence **not verified**. `[P-SNIP]` — [pKAI GitHub](https://github.com/bayer-science-for-a-better-life/pKAI)

**pKALM** (2024 preprint; the 2024–2026 successor)
- Sequence-only (no structure needed), frozen ESM-2 + pI models + residue embedding → BiLSTM predicting a shift from the standard pKa. RMSE 0.8658; throughput ~4,965 pKa/s; covers Asp, Glu, His, Lys, Cys, Tyr plus N-/C-termini. Preprint text is CC-BY-NC-ND 4.0 (document licence, not code). `[P-SNIP]` — [pKALM preprint, bioRxiv 10.1101/2024.09.16.613101](https://www.biorxiv.org/content/10.1101/2024.09.16.613101v1.full.pdf); server at [Onoda Lab](https://onodalab.ees.hokudai.ac.jp/pkalm)
- Its ranking: pKALM second-best, then DeepKa, PypKa, pKAI, with "PROPKA performs the worst, due to its simplistic empirical rules and limited model capacity." For cysteine, PypKa RMSE is above 3. `[P-SNIP]` — [pKALM preprint](https://www.biorxiv.org/content/10.1101/2024.09.16.613101v1.full.pdf)

**Independent benchmark (the one to cite for honest numbers)**
- Seven predictors on a curated PKAD subset (408 residues). DeepKa best on the Large Set: MUE 0.60, MeUE 0.45, RMSE 0.81. PROPKA3 close to pKAI+. Crucially: "no method clearly outperformed the null hypotheses with confidence," though DeepKa, pKAI+, PROPKA3 and H++ "showed some utility"; a simple consensus of the best empirical predictors improved transferability. `[P-SNIP]` — [Comparative Performance of High-Throughput Methods for Protein pKa Predictions, J Chem Inf Model 2023 / PMC10466379](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10466379/)
- Other 2024–2026 entrants I saw but did not evaluate: KaMLs (tree-based, PMC11601431) and pKaLearn (Commun Chem 2026). `[2ND]` — [KaMLs](https://pmc.ncbi.nlm.nih.gov/articles/PMC11601431/); [pKaLearn](https://www.nature.com/articles/s42004-026-01983-y)

### Inferences
- The accuracy spread between tools (RMSE ~0.81–1.11) is smaller than the ~1 pKa-unit uncertainty needed to call a His protonated at pH 7, so **tool choice matters less than the fact that any single tool's answer is often not decisive**. The JCIM-2023 null-hypothesis finding supports treating a pKa prediction as a soft feature, not a hard assignment.
- For a per-protein preprocessing budget on a 4000-structure benchmark (HOLO4K), PypKa is impractical (hours to days), PROPKA3 and pKAI are practical (seconds), and pKPDB lookups are nearly free where coverage exists.

### Gaps
- No measured wall-clock for PROPKA 3 or pdb2pqr on a defined protein from a primary source. The only PROPKA figures are throughput (38 pKa/s) and a historical "couple of seconds."
- Licences for DeepKa, pKAI, DelPhiPKa code and for H++ are not stated on sources I reached.

---

## PART A, Q3 — Does protonation matter for *locating and ranking* pockets, as opposed to docking/affinity?

### Takeaway
I found **no measurement at all** of protonation state changing which pocket a binding-site predictor picks or how it ranks pockets. Every piece of evidence is from docking, virtual screening or MD — downstream tasks. The argument that a donor-vs-acceptor His changes a pocket's pharmacophore is sound but, as of 2026, unmeasured for pocket detection.

### Cited Findings
- **Virtual screening is measurably sensitive.** In M. tuberculosis RmlC, varying the protonation and rotamer states of two histidines produced receptor models with "AUC values range from 0.868 to 0.996." `[P-SNIP]` — [Effects of histidine protonation and rotameric states on virtual screening of M. tuberculosis RmlC](https://www.researchgate.net/publication/236196122_Effects_of_histidine_protonation_and_rotameric_states_on_virtual_screening_of_M_tuberculosis_RmlC)
- **Which site gets occupied can flip in MD.** For trypsin, benzamidine "binds the S1 pocket in 48% of HID57 simulations (24 out of 50 replicas of 200 ns), 44% of HIE57 (22 out of 50 replicas), and 10% of HIP57 (5 out of 50 replicas)" — a histidine more than 10 Å from the S1 pocket. The authors conclude that "when His57 is positively charged, binding will preferentially occur through direct diffusion from the solvent." This is the closest thing to "protonation changes which pocket is picked" that I found, and it is MD, not a predictor. `[P-SNIP]` — [Changes in Protonation States of In-Pathway Residues can Alter Ligand Binding Pathways, PMC9289141](https://pmc.ncbi.nlm.nih.gov/articles/PMC9289141/)
- **Long-range electrostatics of a single His matters at a real drug site.** At the CK2 ATP site, "the protonated His160 also contributes to the binding of such ligands by long-range electrostatic interactions," and "His 115 indirectly affects ligand binding, placing the hinge region in open/closed conformations." `[P-SNIP]` — [Effect of histidine protonation state on ligand binding at the ATP-binding site of human protein kinase CK2, Sci Rep 2024](https://www.nature.com/articles/s41598-024-51905-y)
- **pKa tools disagree exactly where it matters.** For His-12 in an Exd-Hox–DNA complex, "PROPKA 3, H++, and DelPhiPKa ... predicted pKa to be 6.16, 5.75, and 6.77, respectively" — all straddling physiological pH, so no single tool settles the state. `[P-SNIP]` — [Probing the role of the protonation state of a minor groove-linker histidine in Exd-Hox–DNA binding, Biophys J](https://www.sciencedirect.com/science/article/pii/S0006349523041516)
- **A static pre-docking assignment can be wrong once a ligand binds.** In a kinase survey, "significant residues, predominantly Lys and Cys residues, are identified sites where inhibitor binding induces significant shifts in protonation states." `[P-SNIP]` — [Kinase inhibitors can change protonation or tautomeric state upon binding, PMC13483969](https://pmc.ncbi.nlm.nih.gov/articles/PMC13483969/)

### Inferences
- Pocket *detection* may be more robust to protonation than docking is, because detection is dominated by shape/buriedness and residue identity, and because the standard DCA/DCC success criteria are coarse (4 Å). A protonation error that reverses a single H-bond vector is unlikely to move a pocket centre by 4 Å. This is a plausible reason the field has not measured it — but it is my inference, not a finding.
- The trypsin result suggests the effect, where it exists, would show up in *ranking* (which of several real pockets scores highest) rather than in *recall*. A cheap experiment: run a predictor on HID/HIE/HIP variants of the same structure and measure rank churn. I found no paper that has done this.

### Gaps
- **No benchmark quantifies how often pocket-detection output changes with protonation state.** Stated explicitly as missing.
- No study of Asp/Glu/Cys/Lys protonation effects on pocket detection either; the literature is His-heavy.

---

## PART A, Q4 — Hydrogen placement (Reduce, pdb2pqr, OpenMM Modeller): is adding hydrogens standard practice, and do papers report it?

### Takeaway
Adding hydrogens is **not** standard practice in structure-based binding-site prediction. Where papers mention hydrogens at all, it is to remove them. The papers that say nothing are the majority, so "not stated" is the most common honest answer.

### Cited Findings
- DeepSurf removes hydrogens (quoted above, Q1). `[P-READ]` — [DeepSurf](https://arxiv.org/pdf/2002.05643)
- GrASP uses heavy atoms as nodes (quoted above). `[P-READ]` — [GrASP PMC10402091](https://pmc.ncbi.nlm.nih.gov/articles/PMC10402091/)
- DeepPocket removes all heteroatoms and solvent (quoted above); hydrogen handling **not stated**. `[P-READ]` — [DeepPocket PDF](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2022_01/Deeppocket.pdf)
- P2Rank: hydrogens **not stated** anywhere in the paper (verified by grepping the full text). The paper does state "No preprocessing steps on part of the user are needed," which is the opposite of a protonation step. `[P-READ]` — [P2Rank full text, PMC6091426](https://europepmc.org/article/MED/30109435)
- VN-EGNN: Cα-only, so hydrogens are moot; **not stated**. `[P-READ]` — [VN-EGNN HTML](https://arxiv.org/html/2404.07194v1)
- Kalasanty: hydrogens **not stated**; the 18-feature scheme is per heavy atom with Open Babel-derived charges. `[P-READ]` — [Kalasanty preprint](https://arxiv.org/pdf/1904.06517)
- sc-PDB is the exception in the pipeline, and it does add hydrogens and optimize the H-bond network (quoted in Q1). `[P-READ]` — [sc-PDB 2015, PMC4384012](https://pmc.ncbi.nlm.nih.gov/articles/PMC4384012/)
- pdb2pqr licence: the project "switched license from GPL to BSD in version 1.3.0" (2008); PDB2PQR 3.x takes PROPKA as a pip dependency rather than bundling it, so PROPKA's LGPL applies to that component. The public server caps input at 10,000 atoms; the CLI is recommended above that. Per-protein runtime **not found**. `[P-SNIP]` — [pdb2pqr release history](https://pdb2pqr.readthedocs.io/en/latest/releases.html); [pdb2pqr examples](https://pdb2pqr.readthedocs.io/en/latest/using/examples.html)

### Inferences
- The field's de facto convention is "heavy atoms, no explicit protonation, charges from a cheminformatics toolkit default." That convention is inherited, not argued for: it propagates from Pafnucy's 2018 grid featurization into Kalasanty and then DeepSurf, and from `remove heteroatoms` boilerplate into DeepPocket and GrASP.

### Gaps
- I did not verify Reduce's or OpenMM Modeller's licence or runtime from primary sources in this session. Reduce (Richardson lab) and OpenMM are both open source to my knowledge, but **treat that as unverified** — no primary source read.
- No binding-site paper I read reports hydrogen placement *either way* with a named tool.

---

## PART A, Q5 — His tautomers: how often is the assignment ambiguous, and is there measured downstream impact?

### Takeaway
Ambiguity is routine in principle (three states, HID/HIE/HIP, and pKa predictions that straddle pH 7), but I found **no published frequency** of ambiguous His assignment, and no measured downstream impact on pocket detection. Measured downstream impact exists only for docking/VS/MD.

### Cited Findings
- Three independent predictors gave 6.16, 5.75 and 6.77 for the same histidine — i.e. ambiguous by construction at pH 7. `[P-SNIP]` — [Exd-Hox study](https://www.sciencedirect.com/science/article/pii/S0006349523041516)
- Tautomers can be assigned experimentally but rarely are: solution NMR or neutron diffraction, "with the latter limited to very large single crystals and requiring a neutron source"; solid-state MAS NMR is offered as a practical alternative, where "the tautomeric state of histidines can be unambiguously determined from a unique combination of 15N sidechain chemical shifts." `[P-SNIP]` — [Determination of Histidine Protonation States in Proteins by Fast Magic Angle Spinning NMR, PMC8703106](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8703106/)
- A pocket can be geometrically indifferent to the tautomer: for histamine at H1R, "each tautomer (i.e., τ- or π-protonated) or even a diprotonated imidazole ring could be accommodated in the binding pocket." `[P-SNIP]` — [Computational Analysis of Histamine Protonation Effects on H1R Binding, PMC10180022](https://pmc.ncbi.nlm.nih.gov/articles/PMC10180022)
- Measured VS impact of His tautomer+rotamer: AUC 0.868–0.996 on RmlC (as in Q3). `[P-SNIP]` — [RmlC study](https://www.researchgate.net/publication/236196122_Effects_of_histidine_protonation_and_rotameric_states_on_virtual_screening_of_M_tuberculosis_RmlC)

### Gaps
- **No statistic for "what fraction of His residues in a benchmark have an ambiguous tautomer/protonation assignment."** Not found; would have to be computed (e.g. count His with predicted pKa in 6.0–8.0 across COACH420).
- No measured downstream impact of His tautomer choice on pocket location or ranking.

---

## PART B, Q1 — How are Zn, Mg, Fe, Ca, Mn, Na treated as input, per method?

### Takeaway
Three regimes. (a) **Explicit metal channel:** DeepSite has a dedicated `metallic` channel; Kalasanty and DeepSurf carry a `metal` bit inside their 9-way atom-type one-hot — so metals are represented *if present in the input file*. (b) **Stripped:** DeepPocket removes all heteroatoms; GrASP cleans heteroatoms during parsing. (c) **Structurally impossible:** VN-EGNN is Cα-only with ESM-2 features, so a metal cannot be represented at all. P2Rank and EquiPocket: **not stated**.

### Cited Findings per method
- **DeepSite** — eight channels: voxel occupancies "depending on their excluded volume and other seven atom properties: hydrophobic, aromatic, hydrogen bond acceptor or donor, positive or negative ionizable and **metallic**"; metal assignment via AutoDock 4 atom types, whose table lists Mg, Zn, Mn, Ca and Fe as non-H-bonding; "Non-protein atoms are filtered out of the calculation." Note the tension: a metallic channel exists, yet non-protein atoms are filtered — so whether an isolated Zn HETATM reaches the grid is **ambiguous from the snippet**. `[P-SNIP]` — [DeepSite, Bioinformatics 33(19):3036](https://academic.oup.com/bioinformatics/article/33/19/3036/3859178)
- **Kalasanty** — 18 atomic features from Pafnucy, whose atom-type one-hot is "B, C, N, O, P, S, Se, halogen and **metal**." Metals in the input file therefore occupy their own feature bit. Whether metals are present depends on sc-PDB's prepared protein files (see below). HETATM filtering is **not stated** in the Kalasanty paper. `[P-READ]` (feature list from Pafnucy, read directly; Kalasanty's adoption of it read directly) — [Pafnucy](https://academic.oup.com/bioinformatics/article/34/21/3666/4994792); [Kalasanty preprint](https://arxiv.org/pdf/1904.06517)
- **DeepSurf** — identical 18-feature scheme, so identical metal bit. HETATM filtering **not stated**. `[P-READ]` — [DeepSurf](https://arxiv.org/pdf/2002.05643)
- **DeepPocket** — "We first clean the input structure by removing all heteroatoms and solvent molecules from the protein structure using the Biopython47 library." Metals are therefore **absent from the input**, regardless of what the 14 libmolgrid receptor channels contain. `[P-READ]` — [DeepPocket PDF](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2022_01/Deeppocket.pdf)
- **DeepPocket channels** — "C = 14 is the number of atom-type channels"; "libmolgrid default receptor atom types were used as grid channels and are stated in Table S1 of the Supporting Information." Table S1 not retrieved; libmolgrid/gnina types derive from AutoDock 4, which does include metal types — but this is an inference, not a read. `[P-SNIP]` / `[2ND]` — [DeepPocket PDF](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2022_01/Deeppocket.pdf); [libmolgrid docs](https://gnina.github.io/libmolgrid/python/)
- **P2Rank** — I grepped the complete full text: **no mention of metals, ions as input features, or HETATM atom-level handling**. Features are chemical/geometric properties of SAS-point neighbourhoods; metal handling is **not stated**. `[P-READ]` — [P2Rank full text, PMC6091426](https://europepmc.org/article/MED/30109435)
- **EquiPocket** — operates on "surface atoms" with "chemical and spatial structure" features; HETATM/metal handling **not stated**, and the arXiv version is withdrawn so no methods text is currently served. `[P-READ]` (abstract + withdrawal notice) — [arXiv:2302.12177](https://arxiv.org/abs/2302.12177)
- **VN-EGNN** — "We used the position of the α-carbons as residue node locations"; features are "pre-trained ESM-2 protein embeddings" (ablation: one-hot amino-acid type "complemented by an additional category for the virtual nodes"). There is no atom-level channel, so **metals and cofactors are structurally unrepresentable**. `[P-READ]` — [VN-EGNN HTML v1](https://arxiv.org/html/2404.07194v1)
- **GrASP** — heavy-atom nodes; repository instructions state: "Heteroatoms do not need to be removed, they will be cleaned during parsing." So metals are removed by the parser. Whether the SI feature table has a metal indicator is **not verified**. `[P-READ]` (repository README) — [GrASP GitHub README](https://github.com/tiwarylab/GrASP); [GrASP paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10402091/)
- **Context for Kalasanty/DeepSurf/DeepPocket inputs:** sc-PDB's prepared protein and site files "may include cofactor(s), metallic ion(s) and covalently bound residue(s)." So in sc-PDB training data, metals and cofactors are *part of the protein*, not the ligand. `[P-READ]` — [sc-PDB 2015, PMC4384012](https://pmc.ncbi.nlm.nih.gov/articles/PMC4384012/)

### Inferences
- There is an unremarked train/test inconsistency in the most-used pipeline: Kalasanty and DeepSurf train on sc-PDB protein files that *contain* metals and cofactors (and have a metal feature bit to encode them), but are evaluated on COACH420/HOLO4K PDB files whose heteroatom content is handled differently or not stated. DeepPocket trains on the same sc-PDB but strips heteroatoms, so its metal bit is always zero.
- A metal-containing cavity presented to VN-EGNN or GrASP looks, to the model, like an empty cavity lined by the same residues. For Zn/Mg sites where the metal is the pharmacophoric anchor, the model is being asked to detect a site whose defining feature has been deleted.

### Gaps
- DeepPocket Table S1 (the 14 channel names) not retrieved.
- GrASP SI feature table not retrieved.
- EquiPocket methods currently unavailable (withdrawn preprint).

---

## PART B, Q2 — Is there evidence the metal-input choice matters for site-prediction accuracy? Any ablation?

### Takeaway
**No ablation of a metal input channel exists for any general binding-site predictor**, as far as I can find. The only quantitative evidence that metals matter to site prediction is indirect: removing ion sites from a benchmark raises every method's recall by 5–10 points.

### Cited Findings
- Removing ions from LIGYSIS "raises top-N+2 recall by 5–10% for all methods except fpocket, without changing the overall method ranking." The benchmark keeps ions deliberately "to challenge the methods as they have not been trained on such sites." `[P-READ]` — [Comparative evaluation of methods for the prediction of protein–ligand binding sites, J Cheminform 2024 / PMC11552181](https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/)
- P2Rank's own ablation table covers feature subsets (e.g. removing atom-type propensity features, which contribute "minimal at best") but **no metal-related feature is ablated** because none is named. `[P-READ]` — [P2Rank Supplementary Information](https://static-content.springer.com/esm/art%3A10.1186%2Fs13321-018-0285-8/MediaObjects/13321_2018_285_MOESM1_ESM.pdf)
- DeepSurf's ablations cover grid alignment and network architecture (ResNet-18 vs LDS-ResNet), not chemistry channels. `[P-READ]` — [DeepSurf](https://arxiv.org/pdf/2002.05643)
- The closest analogue in the *metal-site* literature: PRIME (2025 preprint) reports that "its probe-based surface molecules significantly improve metal-binding site prediction accuracy" — an ablation, but for a dedicated metal-site predictor. `[P-SNIP]` — [Probe-Based Identification of Metal-Binding Sites Using Deep Learning Representations, bioRxiv 2025.10.04.680417](https://www.biorxiv.org/content/10.1101/2025.10.04.680417v1.full)

### Inferences
- The 5–10 point recall gap between LIGYSIS and LIGYSIS_NI is the best available price tag for ignoring ion/metal sites, but it conflates two causes — metals absent from the input, and ion sites being small/geometrically atypical — and so cannot be attributed to the input-representation choice alone. An ablation (same model, metal channel on/off, on metal-containing holo sites) is an open and cheap experiment.

### Gaps
- **No metal-channel ablation published for DeepSite, Kalasanty, DeepSurf, DeepPocket, P2Rank, EquiPocket, VN-EGNN or GrASP.** Stated as absent.

---

## PART B, Q3 — How common are metal-containing binding sites in the standard benchmarks?

### Takeaway
Only LIGYSIS publishes the number, and it is large: ions are ≈40% of its ligand binding sites. For COACH420, HOLO4K and sc-PDB no percentage is published — and for COACH420/HOLO4K the figure is near zero *by construction*, because P2Rank's relevant-ligand filter requires ≥5 ligand atoms, which excludes every monatomic ion.

### Cited Findings
- **LIGYSIS:** "LIGYSIS differs from all other datasets since biologically relevant ions are considered, comprising ≈ 40% of the ligand sites." `[P-READ]` — [J Cheminform 2024 / PMC11552181](https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/)
- Four of LIGYSIS's five most frequent ligands are ions (Zn²⁺, Ca²⁺, Mg²⁺, Mn²⁺), "together representing 19.2% of all ligands." `[P-READ]` — [PMC11552181](https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/)
- Ions skew site size: small sites (1–10 residues) are "56% of sites, falling to 36% without ions." `[P-READ]` — [PMC11552181](https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/)
- LIGYSIS_NI (no ions) size: 2275 structures, 4572 sites, 38,595 ligands. LIGYSIS-web overall: 64,782 binding sites from 435,038 biologically relevant ligands across 25,003 proteins (different scope/version from the benchmark subset — do not mix the numbers). `[P-READ]` — [PMC11552181](https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/); [LIGYSIS-web, NAR 2025](https://academic.oup.com/nar/article/53/W1/W351/8133629)
- **PDB-wide reference point:** in BioLiP2, "regular small molecules account for more than half (52.5%) of all ligands; metal ions are the second largest group (34.3%)" (2023 snapshot). `[P-SNIP]` — [BioLiP2, NAR 52(D1):D404](https://academic.oup.com/nar/article/52/D1/D404/7233921)
- **Zinc alone:** "present in about 10% of deposited structures," and Metal3D estimates "about one third of zinc sites in the PDB are artifacts." `[P-READ]` — [Metal3D, PMC10175565](https://pmc.ncbi.nlm.nih.gov/articles/PMC10175565/)
- **Why COACH420/HOLO4K are near zero:** P2Rank's relevant-ligand filter requires "number of ligand atoms is greater or equal than 5," and Binding MOAD is described as "the strictest of them, not accepting any small ions for example." `[P-READ]` — [P2Rank SI](https://static-content.springer.com/esm/art%3A10.1186%2Fs13321-018-0285-8/MediaObjects/13321_2018_285_MOESM1_ESM.pdf)

### Inferences
- The field's headline numbers (COACH420/HOLO4K DCA success) are measured on a ligand population from which metal sites have been filtered out. LIGYSIS's ≈40% ion-site figure is the first honest look at what a predictor faces on real protein surfaces, and it is where methods lose 5–10 points.

### Gaps
- **No published percentage of metal-containing (as opposed to metal-as-ligand) binding sites in COACH420, HOLO4K or sc-PDB.** Would need to be computed from the structures. sc-PDB states only that prepared sites "may include ... metallic ion(s)" without a count.

---

## PART B, Q4 — Cofactors (haem, FAD, NAD, ATP): how do the benchmarks' relevant-ligand rules treat them, and does excluding them remove real sites?

### Takeaway
Cofactors are *kept* by the standard rules, not excluded — and that is itself the problem: COACH420 and HOLO4K are "dominated by co-factor ligands." P2Rank's ignore list excludes water, sugars, glycerol, MPD, sulfate and phosphate but **not** HEM, FAD, NAD or ATP. Ions, by contrast, are excluded, and the P2Rank authors say so explicitly as a design choice.

### Cited Findings
- **P2Rank's exact relevant-ligand filter** (the authoritative statement for COACH420/HOLO4K). A ligand is relevant if:
  - "number of ligand atoms is greater or equal than 5"
  - "distance from any atom of the ligand to the closest protein atom is at least 4Å (to remove 'floating' ligands)" — note this reads as a typo in the SI; the intent must be *at most* 4 Å
  - "distance form the center of the mass of the ligand to the closest protein atom is not greater than 5.5Å (to remove ligands that 'stick out')"
  - "name of the PDB group is not on the list of ignored groups: (HOH, DOD, WAT, NAG, MAN, UNK, GLC, ABA, MPD, GOL, SO4, PO4)"
  `[P-READ]` — [P2Rank Supplementary Information](https://static-content.springer.com/esm/art%3A10.1186%2Fs13321-018-0285-8/MediaObjects/13321_2018_285_MOESM1_ESM.pdf)
- HEM, FAD, NAD, NAP, ATP, ADP are **not** on that ignore list and all have ≥5 atoms — so they are *relevant ligands* and define real target sites in COACH420 and HOLO4K. `[P-READ]` (inference from the quoted list, which I read in full)
- The authors' stated position on ions: "We believe that predicting binding sites for ions, peptides and other specific types of binding partners would be better served by specialized methods." And: Binding MOAD is "the strictest of them, not accepting any small ions for example." `[P-READ]` — [P2Rank SI](https://static-content.springer.com/esm/art%3A10.1186%2Fs13321-018-0285-8/MediaObjects/13321_2018_285_MOESM1_ESM.pdf)
- **How much the ligand-relevance rule changes the target set:** "HOLO4K(Mlig), which has approx. by 1/3 less relevant binding sites to be predicted than HOLO4K." Exact counts from the SI table — COACH420: 420 proteins / 511 ligands; HOLO4K: 4009 / 9584; COACH420(Mlig): 300 / 378; HOLO4K(Mlig): 3448 / 6886. `[P-READ]` — [P2Rank SI](https://static-content.springer.com/esm/art%3A10.1186%2Fs13321-018-0285-8/MediaObjects/13321_2018_285_MOESM1_ESM.pdf)
- The `*(mlig)*` dataset variants "contain explicitly specified relevant ligands"; valid codes come from MOAD 2013, and "Proteins unknown to MOAD and proteins with conflicting ligand codes (valid&invalid) were removed." `[P-READ]` — [p2rank-datasets README](https://github.com/rdk/p2rank-datasets)
- **The cofactor-domination finding:** most datasets (all except LIGYSIS, LIGYSIS_NI and PDBbind_REF) are "dominated by co-factor ligands," such as FAD, NAD and heme (HEM), or energy carriers such as ATP, ADP and AMP. By contrast, LIGYSIS_NI's most common ligands include cholesterol, mannose and fucose. `[P-READ]` — [PMC11552181](https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/)
- **sc-PDB's rule:** ligands are "a small synthetic or natural ligand" of 140–800 Da, well buried and biologically relevant, and since 2013 also predicted ligandable by an ML model. Site = all protein residues with ≥1 heavy atom within 6.5 Å of any ligand heavy atom. Cofactors and metals sit on the *protein* side: the protein and binding site "may include cofactor(s), metallic ion(s) and covalently bound residue(s)." Waters are present in ~2/3 of complexes and retained when they make ≥2 hydrogen bonds with the site. `[P-READ]` — [sc-PDB 2015, PMC4384012](https://pmc.ncbi.nlm.nih.gov/articles/PMC4384012/)
- **BioLiP/BioLiP2's rule** (what LIGYSIS inherits): an artifact list of "463 commonly used non-biological ligands" (353 from BioLiP v1 plus 110 added since), covering crystallization additives and purification buffers. "A ligand on the artifact list will be discarded if it appears >15 times in the structure, or if its binding site only contains two consecutive residues"; survivors go to a literature check against the PubMed title/abstract of the primary citation. `[P-SNIP]` — [BioLiP2, NAR 52(D1):D404](https://academic.oup.com/nar/article/52/D1/D404/7233921)
- BioLiP treats metals case by case: "Most metal ions, such as sodium ion, are first listed as possible artifacts," then the abstract decides — a sodium in 1ET1 is discarded as a crystallization artifact, while a sodium in 193L is kept because the abstract describes its role in stabilizing loop Ser60-Leu75. `[P-SNIP]` — [BioLiP, NAR 41(D1):D1096](https://academic.oup.com/nar/article/41/D1/D1096/1074898)
- DeepPocket's own re-curation of COACH420/HOLO4K: it removed standard amino acids mistakenly annotated as ligands ("reported as ligands due to poor preparation of the PDB files") and any ligand unparseable by RDKit or BioPandas, yielding 291 proteins / 359 ligands (COACH420) and 3413 / 4288 (HOLO4K) — numbers that differ from P2Rank's, so cross-paper comparisons on "COACH420" are not strictly like-for-like. `[P-READ]` — [DeepPocket PDF](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2022_01/Deeppocket.pdf)

### Inferences
- Excluding cofactors as "not drug-like" would remove a *large fraction* of the standard benchmarks' positives, not a few edge cases — the LIGYSIS authors' "dominated by co-factor ligands" characterization implies a substantial share. The real distortion runs the other way: COACH420/HOLO4K over-represent large, buried, highly conserved cofactor pockets (FAD/NAD/HEM) and under-represent small ion sites and shallow drug-like pockets. A method tuned on them is tuned for big buried cavities.
- A cofactor is simultaneously part of the functional site and a target ligand, and the two benchmark families resolve this inconsistently: sc-PDB puts cofactors on the protein side (so a model sees HEM as protein context), while COACH420/HOLO4K put them on the ligand side (so the same HEM is a prediction target). Training on sc-PDB and testing on HOLO4K therefore inverts the role of the same chemical entity. I did not find any paper that notes this.

### Gaps
- No published count of how many COACH420/HOLO4K sites are cofactor sites (only the qualitative "dominated by").
- LIGYSIS's specific treatment of cofactors beyond inheriting BioLiP relevance is not described in the text I read.

---

## PART B, Q5 — Metal-site predictors: runnable, weights published, usable as a comparison or input channel?

### Takeaway
Yes, several are runnable today, and **AllMetal3D (MIT, pip-installable, 11 metals) is the sensible choice for an input channel**; Metal3D is the better-documented accuracy reference but is zinc-only and does not state a code licence. AlphaFill (BSD-2-Clause) is the right tool if the goal is to transplant metals/cofactors into apo or predicted structures rather than to predict them.

### Cited Findings

**Metal3D / Metal1D**
- Accuracy: predictions "within 0.70 ± 0.64 Å of experimental locations"; at p = 0.9, MAD 0.70 ± 0.64 Å, median 0.52 Å, 1 false positive. At p = 0.75 it identified 85 test-set sites vs MIB 78 and BioMetAll 75, with 9 false positives. `[P-READ]` — [Metal3D, Nat Commun 2023 / PMC10175565](https://pmc.ncbi.nlm.nih.gov/articles/PMC10175565/)
- Training data: 2,085 structures / 252,324 voxelized environments; validation 3,067, test 6,550 environments (59 test structures, 26 validation). `[P-READ]` — [PMC10175565](https://pmc.ncbi.nlm.nih.gov/articles/PMC10175565/)
- **Wall clock: "typically 25 seconds for a 250 residue protein on a multicore GPU workstation" (20 CPUs, GTX2070).** `[P-READ]` — [PMC10175565](https://pmc.ncbi.nlm.nih.gov/articles/PMC10175565/)
- Code: "Code is available under https://github.com/lcbc-epfl/metal-site-prediction", archived at Zenodo 10.5281/zenodo.7015849. There is no dedicated web server; a Google Colab notebook and a Hugging Face Space exist. **The paper does not state a code licence and does not say whether trained weights are released separately** (the article text is CC BY 4.0, which does not cover code). `[P-READ]` — [PMC10175565](https://pmc.ncbi.nlm.nih.gov/articles/PMC10175565/)
- Scope: trained only on zinc; high recall for transition metals, lower for alkali/alkaline-earth; it "does not distinguish zinc from other metals." Independent evaluation at threshold 0.75: recall ~80%, precision ~82% for sites with ≥3 ligands. `[P-READ]` (training scope) / `[2ND]` (independent recall/precision) — [PMC10175565](https://pmc.ncbi.nlm.nih.gov/articles/PMC10175565/); [PinMyMetal, Nat Commun 2025](https://www.nature.com/articles/s41467-025-57637-5)

**AllMetal3D (2025)**
- "available as ChimeraX extension, standalone web app as well as python package"; `pip install allmetal3d`; ChimeraX Toolshed distribution; "Data and Code are archived under DOI https://doi.org/10.5281/zenodo.14809847". **Licence: MIT** ("allmetal3d is licensed under the terms of the MIT license"), confirmed by PyPI metadata. Supports 11 metal ions. `[P-SNIP]` — [AllMetal3D preprint, bioRxiv 2025.02.05.636627](https://www.biorxiv.org/content/10.1101/2025.02.05.636627v1); [GitHub lcbc-epfl/allmetal3d](https://github.com/lcbc-epfl/allmetal3d); [PyPI allmetal3d](https://pypi.org/project/allmetal3d/)

**BioMetAll**
- Command-line application, GitHub `insilichem/biometall`, on PyPI. **Licence conflict:** the page text says "BioMetAll is an open-source software licensed under the BSD-3 Clause License" and PyPI says BSD-3-Clause, while the GitHub sidebar shows LGPL-3.0. Check the LICENSE file before relying on either. Backbone-preorganization method (no trained weights in the ML sense). `[P-SNIP]` — [GitHub insilichem/biometall](https://github.com/insilichem/biometall); [PyPI BioMetAll](https://pypi.org/project/BioMetAll/)

**MIB / MIB2**
- Web server at combio.life.nctu.edu.tw/MIB2. **Code, weights and licence: not found.** Used only as a comparison baseline in the Metal3D and PRIME papers. `[2ND]` — [PRIME preprint](https://www.biorxiv.org/content/10.1101/2025.10.04.680417v1.full)

**AlphaFill**
- Transplants "missing small molecules and ions from experimentally determined structures to predicted protein models," drawing candidates from the most common PDB ligands plus cofactors and analogs from the CoFactor database; homologs at >25% identity over ≥85 aligned residues. "A total of 12,029,789 transplants were performed on 995,411 AlphaFold models," served at alphafill.eu. **Code licence: BSD-2-Clause** (GitHub PDB-REDO/alphafill). Underlying AlphaFold DB data is CC-BY-4.0. **The AlphaFill databank's own data licence is not stated on sources I reached.** `[P-SNIP]` — [AlphaFill, Nat Methods 2023 / PMC9911346](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9911346/); [GitHub PDB-REDO/alphafill](https://github.com/PDB-REDO/alphafill)

**Others seen (2025–2026), not evaluated**
- PinMyMetal (hybrid learning, transition metals, Nat Commun 2025); ESMBind / ESMBind-DL (7 metals, ESM-2 + ESM-IF); MetalSiteHunter (ensemble 3D CNN, Fe/Zn/Mg/Mn/Ca/Na); MetalBind (Cell Rep Methods 2026, surface-point module); PRIME (probe-based, 2025 preprint, with an ablation). `[2ND]` — [PinMyMetal](https://www.nature.com/articles/s41467-025-57637-5); [MetalSiteHunter, Cell Rep Phys Sci 2022](https://www.cell.com/cell-reports-physical-science/fulltext/S2666-3864(22)00340-X); [PRIME](https://www.biorxiv.org/content/10.1101/2025.10.04.680417v1.full)

### Inferences
- As an **input channel**, AllMetal3D is the only option that is simultaneously multi-metal, pip-installable and unambiguously licensed (MIT). Metal3D's 25 s / 250 residues is acceptable for a benchmark-scale preprocessing pass (roughly a day of single-GPU time for HOLO4K at 4000 structures), but zinc-only output limits it.
- As a **comparison**, metal-site predictors are not directly comparable to pocket predictors: they predict metal *positions* (sub-angstrom MAD), not ranked ligand pockets, and are evaluated with precision/recall on coordination sites rather than DCA/DCC top-n. Any comparison would have to be reframed.

### Gaps
- Metal3D code licence and whether weights are published as a release artifact: **not stated** in the paper; unverified.
- MIB2 code/weights/licence: not found.
- AlphaFill databank data licence: not stated.

---

## PART B, Q6 — Does a metal change what a *small-molecule* binding site looks like? Should a metal-containing cavity be a different class?

### Takeaway
The quantitative evidence says metal/ion sites are a geometrically different population (much smaller, much harder for existing methods), and the qualitative evidence says metal coordination is a known failure mode for docking and for deep models. There is no published study that tests "metal-containing cavity as a separate class" in a binding-site predictor — the question is open.

### Cited Findings
- Ion sites are structurally atypical: with ions, 56% of LIGYSIS sites are small (1–10 residues); without them, 36%. `[P-READ]` — [PMC11552181](https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/)
- They are measurably harder: removing ions raises top-N+2 recall by 5–10% for all methods except fpocket, with no change in method ranking. The authors keep them "to challenge the methods as they have not been trained on such sites." `[P-READ]` — [PMC11552181](https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/)
- Metal coordination is a recognized modelling failure: "conventional docking tools and existing deep-learning models fail to reliably capture metal–ligand interactions." `[2ND]` — [MetalProGNet, Chem Sci](https://www.sciencedirect.com/org/science/article/pii/S2041652023059886)
- sc-PDB's own curation treats a metal in the site as protein context, not as a distinct site class: the site "may include cofactor(s), metallic ion(s) and covalently bound residue(s)." `[P-READ]` — [PMC4384012](https://pmc.ncbi.nlm.nih.gov/articles/PMC4384012/)
- One third of PDB zinc sites are estimated to be artifacts, and ~62% of the Metal3D test set consists of "well-coordinated" sites — so a metal present in a crystal structure is not reliably a functional feature of the pocket. `[P-READ]` — [PMC10175565](https://pmc.ncbi.nlm.nih.gov/articles/PMC10175565/)
- BioLiP's per-structure literature check for metals (the 1ET1 vs 193L sodium example) is the field's acknowledgement that a metal's status as real-site-component vs artifact cannot be decided from the structure alone. `[P-SNIP]` — [BioLiP, NAR 2013](https://academic.oup.com/nar/article/41/D1/D1096/1074898)

### Inferences
- Two distinct design decisions are being conflated in the literature and should be separated: (i) whether a metal *ion* is a prediction target (it is in LIGYSIS, not in COACH420/HOLO4K/MOAD), and (ii) whether a metal present in a cavity is an *input feature* that shapes a small-molecule site. Decision (ii) is unexamined. A metalloenzyme inhibitor pocket (e.g. a zinc-chelating warhead site) is a drug-like small-molecule site whose pharmacophore is dominated by the metal; deleting the metal from the input, as DeepPocket and GrASP do, removes the feature that makes the site what it is.
- Given that ~1/3 of crystallographic zinc sites are artifacts, a metal input channel should probably carry confidence (e.g. occupancy, coordination number, or an AllMetal3D/Metal3D probability) rather than a hard presence bit — otherwise the channel learns from crystallization noise.
- The 5–10 point recall gap and the small-site skew together suggest that treating metal-containing cavities as a separate class is worth testing, but the evidence currently supports it only as a hypothesis.

### Gaps
- **No study treats metal-containing cavities as a distinct class in a binding-site predictor**, and no study measures whether doing so helps. This is the clearest open experiment identified in this research.
- No measurement separating "metal site is hard because it is small" from "metal site is hard because the metal is missing from the input."

---

## Cross-cutting summary of "not stated" findings (for the report writer)

| Method | Hydrogens / protonation | HETATM & metals | Metal feature channel | Source read |
|---|---|---|---|---|
| DeepSite | not stated | "non-protein atoms are filtered out" | **yes** — `metallic` channel (8 total) | `[P-SNIP]` |
| Kalasanty | not stated (heavy atoms, Open Babel charges) | not stated | **yes** — `metal` bit in 9-way atom type (18 feats) | `[P-READ]` |
| DeepSurf | hydrogens **removed** | not stated | **yes** — same 18-feature scheme | `[P-READ]` |
| DeepPocket | not stated | **all heteroatoms + solvent removed** | 14 libmolgrid channels (Table S1 unread); moot — metals stripped | `[P-READ]` |
| P2Rank | **not stated** (grepped full text) | not stated for input; ions excluded from *ligands* by ≥5-atom rule | not stated | `[P-READ]` |
| EquiPocket | not stated (preprint withdrawn) | not stated | not stated | `[P-READ]` |
| VN-EGNN | n/a — Cα nodes, ESM-2 features | n/a — no atoms | **no** — structurally impossible | `[P-READ]` |
| GrASP | heavy atoms only; `formal_charge` feature | heteroatoms "cleaned during parsing" | SI table unread; not in main text | `[P-READ]` |

**The single strongest quantitative claim available for "getting it wrong costs something":** removing ion sites from LIGYSIS raises top-(N+2) recall by 5–10 percentage points for every method except fpocket — [J Cheminform 2024 / PMC11552181](https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/) `[P-READ]`.

**The single strongest gap:** no ablation, for any binding-site predictor, of protonation state or of a metal input channel.
