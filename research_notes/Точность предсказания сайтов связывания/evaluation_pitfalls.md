# Published critiques and methodological pitfalls of ligand binding-site prediction benchmarking (as of 2026)

**Verification key used throughout:** `[PRIMARY]` = read directly from the paper's own text (full text or PDF fetched in this session); `[SECOND-HAND]` = taken from a search-engine synthesis or another paper's description, not confirmed against the original; `[DEMONSTRATED]` = critique backed by numbers in the source; `[ASSERTED]` = critique stated without supporting measurement.

**The single most important source** for almost every question below is Utgés & Barton, *"Comparative evaluation of methods for the prediction of protein–ligand binding sites"*, J. Cheminform. 16:126 (2024) — the LIGYSIS benchmark paper. It is open access and is the first independent, third-party benchmark of this field in over a decade. Full text read directly: https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/

---

## Q1. Train/test leakage between scPDB (training) and COACH420 / HOLO4K / PDBbind (test)

### Takeaway
Leakage is widely acknowledged but **poorly quantified**: no paper I found reports a clean "X% of COACH420 clusters at 30/40/50% identity also appear in scPDB" figure. The closest things to hard numbers are (a) individual papers' *ad hoc* removals (e.g. "122 structures removed from COACH420 because present in training"), (b) the P2Rank authors' set-level disjointness claims, which are **by PDB code / protein, not by sequence-identity clustering**, and (c) leak-proof re-splitting work in the adjacent affinity-prediction field (LP-PDBBind) where the effect size *is* measured (30–50% performance drops). The re-evaluation that changed rankings is the LIGYSIS benchmark — but it changed them by fixing **redundancy and receptor definition**, not leakage.

### Cited findings
- COACH420 and HOLO4K were constructed by the P2Rank authors to be *"disjunct with the CHEN11 and JOINED datasets"* (P2Rank's own training/validation sets). This is a set-difference on proteins, with **no sequence-identity clustering details given**: *"we have taken COACH test set and removed proteins contained in CHEN11 and JOINED. However, no sequence-identity clustering details are provided."* — [PRIMARY] [P2Rank, J. Cheminform. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6091426/)
  - **Critical consequence:** the disjointness guarantee is relative to *P2Rank's* training data (CHEN11/JOINED), **not** to scPDB. Every deep method trained on scPDB inherits no such guarantee, yet reuses COACH420/HOLO4K as if it were an independent test set.
- *"Although COACH420 and HOLO4K benchmarks are used by many methods, most of them perform additional filtering (e.g. removing irrelevant ligands or addressing data leakage between the training and test sets), resulting in slightly different subsets, which means a direct comparison of methods based on these benchmarks may not be straightforward."* — [SECOND-HAND] [ASSERTED], surfaced in search synthesis of the binding-site literature; the same point is made independently in the UniSite paper (below).
- *"Commonly used validation sets such as HOLO4K and COACH420 exhibit data leakage problems, with the use of global protein similarity metrics resulting in ligandability prediction models with strong biases, and data leakage between training and validation data naturally occurring."* — [SECOND-HAND] [ASSERTED]; I could not confirm the originating paper (the likely source, an OpenReview "pickpocket" submission, was behind a bot-verification wall: https://openreview.net/pdf?id=c3eb9I0kWQ).
- Reported *ad hoc* leakage removals by individual method papers (all [SECOND-HAND] unless noted):
  - *"122 protein structures were removed from COACH420 since they were present in the training dataset"* — attributed to a method trained on scPDB (PUResNet/DeepPocket lineage). **This is the largest concrete leakage figure I found: 122/420 ≈ 29% of COACH420.** Not confirmed against the original paper; **flag for verification before citing.**
  - LaMPSite *"excludes scPDB structures with more than 50% sequence identity or 0.9 ligand similarity and removes proteins from COACH420"* — [PRIMARY, as quoted in] [Frontiers in Bioinformatics 2025 review](https://www.frontiersin.org/journals/bioinformatics/articles/10.3389/fbinf.2025.1520382/full)
  - Pseq2Sites *"using unseen test datasets and filtering proteins with ≤40% structural similarity for unbiased evaluation"*; HoTS *"reporting results at various similarity thresholds"* — same source.
- The 2025 Frontiers review states *"addressing data leakage is essential, especially the similarity between training and test datasets"* but **provides no quantified effect**: it *"does not provide quantified effects of removing data leakage on model rankings or performance."* — [PRIMARY] [ASSERTED] [Frontiers in Bioinformatics 2025](https://www.frontiersin.org/journals/bioinformatics/articles/10.3389/fbinf.2025.1520382/full)
- The 50% identity cutoff appears to be the de facto convention: *"For PDBbind, proteins with more than 50% sequence identity to those in scPDB are removed to avoid data leakage... The 50% sequence identity threshold appears to be a standard cutoff used in this field."* — [SECOND-HAND]. Note 50% is far laxer than the 30% used elsewhere in structural bioinformatics.
- **Independent re-training to remove leakage exists but is small-scale.** Lee, Byun & Shin re-trained *both their model and the DeepPocket and DeepSurf baselines* on a fresh split for a case study: *"we re-trained the Deeppocket and DeepSurf models with a new dataset split to avoid data leakeage"*, after *"we removed all 42 'albumin' structures from the scPDB v.2017 dataset"* and regenerating the homology-augmentation set so that *"there is no leakage coming from the homology-based augmentation."* — [PRIMARY] [DEMONSTRATED for a single target family] [arXiv:2303.08818](https://arxiv.org/pdf/2303.08818)
  - They also measure homology's contribution to performance directly: ablating their homology augmentation costs **1.74 percentage points**. This is an upper-bound-ish indicator that homology information is worth low-single-digit points, i.e. leakage of homologues into training plausibly inflates scores by a similar order. [PRIMARY] [DEMONSTRATED]
- **Adjacent-field evidence of the effect size.** In binding-*affinity* prediction, the leak-proof LP-PDBBind re-split *"exposes 30–50% drops for many state-of-the-art scorers, but few sequence-only papers have re-evaluated their models under this regime."* — [SECOND-HAND] [DEMONSTRATED in the original] [LP-PDBBind, arXiv:2308.09639](https://arxiv.org/html/2308.09639v2); see also [*Resolving data bias improves generalization in binding affinity prediction*, Nat. Mach. Intell. 2025](https://www.nature.com/articles/s42256-025-01124-5) and [HonestAffinity, arXiv:2606.03422](https://arxiv.org/html/2606.03422).
- **Protein-similarity-controlled splits do move numbers in a 2026 binding-site-localisation benchmark.** InteractBind builds four OOD splits with maximum train–test protein similarity of **25%, 28%, 31%, 33%** (Needleman–Wunsch global alignment). FusionDTI's binary-binding AUROC falls **91.8% → 81.8%** going from the 33% to the 25% split; its binding-residue hit rate BRHR@5 moves only **30.1% → 28.4%**. Ligand-similarity splits (Tanimoto/ECFP, max 8–59%) have *"substantially smaller"* effects, *"suggesting models exploit protein sequence patterns more than ligand chemistry."* — [PRIMARY] [DEMONSTRATED] [arXiv:2605.24045](https://arxiv.org/html/2605.24045v1)

### Inferences
- The field's leakage controls are **protein-identity-based, not cluster-based**: the usual guarantee is "this PDB code is not in my training list", which does nothing about close homologues. A new paper that clusters at 30% identity with MMseqs2/Foldseek and reports the overlap explicitly would be ahead of essentially all published baselines.
- Because every scPDB-trained method filters COACH420/HOLO4K differently (or not at all), **the published tables are not comparing methods on the same test set**. This is the strongest defensible statement available, and it is made in the literature ([ASSERTED], multiple sources) rather than measured.

### Gaps
- No paper found that reports the scPDB↔COACH420/HOLO4K/PDBbind overlap as a clustered percentage at a stated identity threshold with a stated clustering tool. This is a genuine hole in the literature and worth measuring in the new paper.
- No paper found that re-evaluated the *published binding-site* methods after leakage removal and reported changed rankings. (The ranking-change result that does exist is LIGYSIS's, driven by redundancy/receptor definition — Q2/Q4.)
- The "122 structures removed from COACH420" figure needs primary confirmation.

---

## Q2. Receptor definition: single chain vs deposited (asymmetric) unit vs biological assembly

### Takeaway
This is the one pitfall that **has been quantified**, by the LIGYSIS paper, and the numbers are large: **40% of HOLO4K structures and 56% of COACH420 structures have a different number of chains in the asymmetric unit than in the biological unit**, with per-structure interface redundancy up to **14×**. Treating redundant copies as independent sites inflates measured performance.

### Cited findings
- *"1811 (40%) of structures present different numbers of chains between the asymmetric and biological units"* (HOLO4K); for COACH420 the figure is *"234 (56%)"*. — [PRIMARY] [DEMONSTRATED] [LIGYSIS / Utgés & Barton 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/)
- Explicit statement of the inflation mechanism: *"Considering predictions of these interfaces as independent can lead to an overestimate of the performance of a predictor."* — [PRIMARY] [same]
- Worked examples with redundancy factors: PDB **1JQY** — *"the asymmetric unit is formed by three copies of a homo-pentamer, whereas the biological unit comprises a single pentamer"*, giving *"the same protein–ligand interface repeated 14 times (14× redundancy)"*; *"In the asymmetric unit of PDB: 1JQY 14 molecules of BMSC-0010 (A32) interact with 14 copies of E. coli heat-labile enterotoxin B chain."* PDB **1PPR** — chlorophyll A, peridinin and digalactosyl diacyl glycerol *"bind to the three copies... resulting in a redundancy of 3×"*. LIGYSIS's correction: *"so 1/14 interfaces are considered for PDB: 1JQY and 12/36 for PDB: 1PPR."* — [PRIMARY] [DEMONSTRATED]
- LIGYSIS's design choice is to *"aggregate biologically relevant unique protein–ligand interfaces across biological units"*, mapping to UniProt residue numbers, i.e. the benchmark unit is a **UniProt-level site**, not a PDB-level ligand. — [PRIMARY]
- **Predictor-side redundancy is also huge and method-specific.** LIGYSIS counts redundant predicted pockets (defined as centroid distance ≤ 5 Å **or** residue Jaccard index > 0.75): *"VN-EGNN presents the highest percentage of redundant pockets with 9066/13,582 (67%)"*, *"IF-SitePred with 22,232/44,948 (49%)"*, DeepPocket(SEG) **−31%**. — [PRIMARY] [DEMONSTRATED]
- **Effect of fixing it on measured accuracy:** *"Removing redundant predictions and re-ranking those remaining resulted in a significant increase in recall of +5.2% for VN-EGNN, +13.4% for IF-SitePred, and +5.6% for DeepPocketSEG"*; re-scoring IF-SitePred's predictions added a further **+2.1%**. *"Removing redundancy increases significantly the recall of VN-EGNN, IF-SitePred and DeepPocketSEG by > 5%"* (p ≤ 0.05). — [PRIMARY] [DEMONSTRATED]
- Methods differ in what receptor they can even accept: *"PUResNet, DeepPocket, P2RankCONS and P2Rank often don't predict on smaller proteins (< 100 amino acids)"*; *"IF-SitePred only predicts pockets on 75% of the chains."* — [PRIMARY] [DEMONSTRATED]
- P2Rank, by contrast, *"is able to work directly with multi-chain structures and thus find potential binding sites that consist of residues from multiple chains"* — i.e. the geometric baseline was designed for assemblies while many deep methods are single-chain. — [PRIMARY] [P2Rank 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6091426/)
- A 2026 method paper openly changes the receptor definition to make HOLO4K tractable: *"We have split the HOLO4K dataset into per-chain components and aggregated the predictions in our evaluated results."* — [PRIMARY] [GDEGAN, arXiv:2603.19817](https://arxiv.org/pdf/2603.19817)

### Inferences
- The "truncated receptor unburies interface pockets" mechanism is **implied but not isolated** in the literature: no paper reports the same model scored on single-chain vs biological-assembly receptors with burial features held otherwise fixed. LIGYSIS quantifies the *reference-side* consequence (redundant sites) and the *prediction-side* consequence (redundant pockets), not the feature-distribution consequence.
- Chain-splitting + aggregation (GDEGAN) and assembly-level aggregation (LIGYSIS) are **different tasks**, so a number produced under one is not comparable to a number produced under the other, even on "HOLO4K".

### Gaps
- No published controlled ablation of single-chain vs deposited vs biological assembly input for a fixed model with burial-based features. This is a clean, publishable experiment the new paper could run.

---

## Q3. Ligand relevance: how "relevant" is defined and how much it moves the numbers

### Takeaway
There are at least four incompatible definitions in circulation (P2Rank/Binding MOAD "valid" codes distributed as the `mlig` lists; a heavy-atom + contact-distance rule; BioLiP's biological-relevance filter; and "all HET groups"). Where the effect has been measured, removing ions alone moves top-N+2 recall by **5–10 percentage points** for nearly every method — large enough to swamp most claimed improvements — though **LIGYSIS reports the ranking does not change**.

### Cited findings
- **P2Rank `mlig` provenance is Binding MOAD, not a geometric rule.** *"Relevant ligands are identified from the MOAD 2013 database"*; the `(mlig)` datasets *"contain explicitly specified relevant ligands"* with codes sourced from MOAD 2013; *"Proteins unknown to MOAD and proteins with conflicting ligand codes (valid&invalid) were removed."* — [PRIMARY] [p2rank-datasets README](https://raw.githubusercontent.com/rdk/p2rank-datasets/master/README.md)
- The P2Rank paper itself: *"P2Rank is focused on predicting binding sites for biologically relevant ligands and PDB files in considered datasets often contain ligands (or HET groups) that are not relevant. To determine which ligands are relevant we use a custom filter and alternatively the binding MOAD database."* The filter details are deferred to Additional file 1. — [PRIMARY] [P2Rank 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6091426/)
- A heavy-atom + distance rule is widely attributed to P2Rank: *"Ligands are considered relevant if (1) the number of ligand atoms is ≥ 5, and (2) the distance from any atom of the ligand to the closest protein atom is at least 4 Å."* — [SECOND-HAND]. **Treat with caution:** criterion (2) as phrased ("at least 4 Å") is almost certainly a garbled rendering of a *contact* criterion (≤ 4 Å) or of an exclusion of ligands too far from the protein. The P2Rank supplementary PDF that would settle this returned HTTP 403 in this session. **Verify against Additional file 1 before citing.**
- **BioLiP filter** is what LIGYSIS uses: *"only biologically relevant ligands, as defined by BioLiP"*, explicitly excluding compounds that are a *"byproduct of crystallisation."* — [PRIMARY] [LIGYSIS 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/)
- **Ions are ~40% of the reference sites** in LIGYSIS: ions comprise *"≈ 40% of the ligand sites."* — [PRIMARY] [DEMONSTRATED]
- **Measured effect of excluding ions:** *"When ions are removed, all methods, except fpocket, experience an increase in (top-N+2) recall of 5–10% but the overall ranking of methods does not change."* — [PRIMARY] [DEMONSTRATED]
- Excluding ions also changes the *shape* of the benchmark: *"54% of the protein chains present more than one binding site... decreases for LIGYSIS-NI (38%)"* — a 16-point shift in the multi-site fraction, which in turn changes what top-N and top-N+2 mean. — [PRIMARY] [DEMONSTRATED]
- Ligand-composition differences across datasets are documented: SC6K and sc-PDB-FULL are *"depleted in saccharides (< 1%)"*; in PDBbind-REF *"short peptides (<10 aas) are the most common ligands... 5%."* — [PRIMARY]
- Dataset sizes after relevance filtering, from the UniSite paper: **HOLO4K-mlig = 3,204 structures / 1,259 unique proteins**; **COACH420-mlig = 284 structures / 265 unique proteins**; scPDB (2017) = 17,594 structures / 5,550 unique proteins; PDBbind v2020 = 19,443 structures / 3,888 unique proteins. — [PRIMARY] [UniSite, arXiv:2506.03237v2](https://arxiv.org/html/2506.03237v2)
  - Note the **COACH420-mlig = 284** figure versus the "420 complexes" universally quoted in method papers: a paper reporting "COACH420" without saying whether it means 420 or 284 structures is not interpretable.

### Inferences
- Because ion inclusion alone is worth 5–10 points of recall, **any claimed improvement smaller than ~10 points is uninterpretable unless the ligand-relevance rule is stated exactly**. This is a directly citable, number-backed argument for stating the protocol.
- The fact that ranking survived ion removal in LIGYSIS is the strongest evidence that ligand relevance is a *comparability* problem (numbers incomparable across papers) more than a *validity* problem (wrong winner).

### Gaps
- No paper found that reports the same method on the all-ligands and `mlig` versions of COACH420/HOLO4K side by side. The 5–10 point ion figure is from LIGYSIS/LIGYSIS-NI, a different dataset pair, so it is an analogy, not a direct measurement of the `mlig` effect.
- The exact P2Rank relevant-ligand filter rules (heavy-atom minimum, any blacklist of buffer/cryoprotectant codes) could not be verified from primary text; the supplementary PDF was inaccessible (403).

---

## Q4. HOLO4K specifically: multi-chain structures, multiple copies, and how site/prediction counting differs

### Takeaway
HOLO4K is the least comparable benchmark in the field. Even its **size is reported inconsistently across papers (4,009 vs 4,288 vs 4,543 vs 3,204)**, 40% of its entries have an asymmetric unit that differs from the biological unit, 62% are multi-ligand, and ~20% of proteins are affected by double-counting under the standard DCC/DCA protocol. "Strong distribution shift", in GDEGAN's footnote, means concretely: multi-chain oligomeric assemblies and repeated copies of the same complex that are absent from the single-chain scPDB training distribution.

### Cited findings
- **Inconsistent structure counts, traced to the source distribution.** The P2Rank dataset repository: the holo4k *"directory holds 4,543 PDB files, but the corresponding .ds file contains 4,009 lines, representing the actual dataset size used in publications"*; also *"1xgf.pdb removed from holo4k datasets (all UNK groups, no ligands)."* — [PRIMARY] [p2rank-datasets README](https://raw.githubusercontent.com/rdk/p2rank-datasets/master/README.md)
  - Meanwhile GDEGAN states *"HOLO4K comprises 4,288 structures"* — [PRIMARY] [arXiv:2603.19817](https://arxiv.org/pdf/2603.19817); UniSite reports HOLO4K-mlig at **3,204** — [PRIMARY] [arXiv:2506.03237v2](https://arxiv.org/html/2506.03237v2); and search-level sources quote "4009 structures". **Four different cardinalities for "HOLO4K" are in print.**
- *"HOLO4K presents the highest proportion (62%) of multi-ligand entries"* and *"1811 (40%) of structures present different numbers of chains between the asymmetric and biological units."* — [PRIMARY] [DEMONSTRATED] [LIGYSIS 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/)
- **Double-counting is measured.** *"The metrics lack proper matching criteria between predictions and ground truth"*, causing predicted sites to be counted multiple times; on HOLO4K-sc *"approximately 20% of proteins experience double-counting during evaluation"* (their Table S1). — [PRIMARY] [DEMONSTRATED] [UniSite, arXiv:2506.03237v2](https://arxiv.org/html/2506.03237v2)
- **What the GDEGAN footnote means in practice.** The footnote reads, verbatim from the PDF: *"holo4k contains multi chains and complex with multiple copies, presenting a strong distribution shift."* Their body text expands it: *"Notably, HOLO4K presents significant distribution shift challenges as it contains numerous multi-chain assemblies and oligomeric proteins absent from typical training sets. We have split the HOLO4K dataset into per-chain components and aggregated the predictions in our evaluated results."* Training is on scPDB (*"17,594 protein-ligand complex structures from the 2017 release, representing 4,782 unique proteins and 6,326 ligands"*), preprocessed *"using the steps described in EquiPocket"*. — [PRIMARY] [arXiv:2603.19817](https://arxiv.org/pdf/2603.19817)
  - So in practice: GDEGAN's HOLO4K number is a **per-chain-decomposed, prediction-aggregated** score, not an assembly-level score. It is not comparable to a number produced on whole deposited files (P2Rank's native mode) or to LIGYSIS's UniProt-aggregated sites.
- **Does the choice change rankings?** Yes, at least under a matching-aware metric: on HOLO4K-sc *"DeepPocket and GrASP showed nearly identical DCA performance (< 0.01 difference) but diverged substantially in AP₀.₃ (> 0.10 difference)"*. — [PRIMARY] [DEMONSTRATED] [UniSite](https://arxiv.org/html/2506.03237v2)
  - And in LIGYSIS, fixing prediction-side redundancy moved recall by +5.2 to +13.4 points for three methods and nothing for the others — which reorders them. — [PRIMARY] [DEMONSTRATED]

### Inferences
- Reporting "HOLO4K DCA top-n" without stating (i) which `.ds` list, (ii) mlig or not, (iii) whole file vs per-chain vs biological assembly, and (iv) whether redundant predictions were merged, specifies none of the four degrees of freedom that each move the number by ≥5 points. A defensible protocol must pin all four.

### Gaps
- No source directly compares the *same* method on HOLO4K scored assembly-wise vs per-chain-aggregated. GDEGAN uses per-chain aggregation but reports no whole-file counterpart.

---

## Q5. Metric definitions: DCC vs DCA, the 4 Å threshold, top-n vs top-(n+2), failure rate

### Takeaway
The 4 Å threshold traces to the Chen et al. (2011) independent benchmark and is reused by P2Rank, from which nearly everyone else inherited it. Two independent 2024–2026 papers now argue it is wrong: LIGYSIS says **4 Å is too strict for DCC and 10–12 Å should be used for parity with DCA = 4 Å**, and UniSite says DCC/DCA are structurally blind and unmatched, proposing residue-level IoU-based Average Precision instead. Top-(n+2) exists because the reference annotation is incomplete; LIGYSIS proposes **top-N+2 recall as the universal metric**.

### Cited findings
- **Origin of the protocol.** P2Rank: evaluation uses the *"DCC(distance between the center of the pocket and any ligand atom) pocket identification criterion with 4 Å threshold"* and *"This evaluation methodology is the same as the one that was used in independent benchmarking study [21]"* — reference [21] being Chen et al.'s 2011 *Large-Scale Comparison of Four Binding Site Detection Algorithms*. — [PRIMARY] [P2Rank 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6091426/); Chen et al.: https://www.researchgate.net/publication/46190262_Large-Scale_Comparison_of_Four_Binding_Site_Detection_Algorithms
  - Note P2Rank's own text conflates the labels (it writes "DCC" while describing distance-to-closest-ligand-atom, i.e. DCA). **The label confusion is in the foundational paper.** — [PRIMARY] [DEMONSTRATED by the quote itself]
- **Top-n / top-(n+2) definition and rationale (P2Rank).** *"Top-n and Top-(n+2) rank cutoffs where n is the number of relevant ligands in the evaluated target protein structure (for proteins with only one ligand this corresponds to the usual Top-1 and Top-3 cutoffs)."* — [PRIMARY] [P2Rank 2018]
- **Why n+2 at all (LIGYSIS's justification).** *"Considering only the top-N predicted pockets assumes that there are exactly N real pockets for a given protein, which might not be the case... 33–50% of existing sites yet to be observed with ligands bound."* And: *"By considering the top-N+2 pockets, we are controlling to some extent for this noise in the reference data and obtaining a more accurate representation of the performance."* They conclude by proposing *"top-N+2 recall as the universal benchmark metric for ligand binding site prediction."* — [PRIMARY] [DEMONSTRATED: the 33–50% unobserved-site estimate is the quantitative basis] [LIGYSIS 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/)
- **The 4 Å DCC threshold is attacked with numbers.** *"a DCC threshold of 4 Å is too conservative, and a more flexible DCC threshold of 10–12 Å should be used for comparable performance with DCA = 4 Å"*; and in Methods, *"12 Å was chosen as 4 Å is too strict a threshold when using DCC"*, while 4 Å *"works well for the distance to closest ligand atom (DCA)."* — [PRIMARY] [DEMONSTRATED] [LIGYSIS 2024]
- **DCC/DCA are shown to be biased even on ground truth.** UniSite measures the metrics against the reference itself: mean ground-truth DCC = **2.15 Å (92.65% < 4 Å)** and mean ground-truth DCA = **1.57 Å (98.88% < 4 Å)**, where *"neither should deviate from 0 theoretically"* — i.e. ~7% of true sites fail the DCC test by construction. — [PRIMARY] [DEMONSTRATED] [UniSite](https://arxiv.org/html/2506.03237v2)
- **Two further UniSite criticisms:** (i) *"lack proper matching criteria between predictions and ground truth"* → double counting (~20% of HOLO4K-sc proteins); (ii) DCC and DCA *"completely disregard the structural properties such as shape, size, and residue composition of binding sites"* — *"Predictions can appear successful by center-distance while missing critical residues"*, and *"Different ligands bound to identical sites produce inconsistent DCC/DCA values."* — [PRIMARY] [DEMONSTRATED, with failure cases in their Figure 4]
- **UniSite's proposed replacement:** residue-level **IoU-based Average Precision** with one-to-one matching following COCO protocols, reported at IoU thresholds **0.3 and 0.5**; they show it is more discriminative than DCA (the DeepPocket/GrASP case above). Their own scores: UniSite-1D AP₀.₃ = 0.5121, AP₀.₅ = 0.3033; UniSite-3D AP₀.₃ = 0.5603, AP₀.₅ = 0.3835. — [PRIMARY] [DEMONSTRATED]
- **LIGYSIS's metric suite:** 10 metrics over 13 original methods and 15 variants; pocket-level top-N and top-N+2 recall, plus residue-level **F1 and MCC** (*"combines precision and recall into a unified metric, capturing the accuracy and completeness of predictions at the residue level"*), plus mean ROC/PR AUC across protein chains. Redundancy definition: *"A predicted pocket i is considered redundant if there exists a pocket j ≠ i so that the distance between their centroids D_i,j ≤ 5 Å or their residue overlap JI_i,j > 0.75."* Headline results: fpocket re-scored by PRANK and by DeepPocket give the highest recall (**60%**); IF-SitePred the lowest (**39%**); *"stronger scoring schemes improved performance by up to 14% in recall and 30% in precision."* — [PRIMARY] [DEMONSTRATED]
- **"Failure rate" is not well defined in the literature.** GDEGAN's Appendix D.3 gives, verbatim, `Failure Rate = |{Predicted sites | DCC/DCA < τ}| / |{True sites}|` — which is the *success* rate formula, not a failure rate; the inequality is evidently inverted. They report *"GDEGAN decreases the failure rate to 3.2%, in contrast to 5.1% for EquiPocket and 4.9% for GotenNet."* — [PRIMARY] [DEMONSTRATED: the formula as printed is self-contradictory with the reported direction]
  - **Inference, flagged as mine:** since the printed formula contradicts the reported numbers, "failure rate" as circulated in the EquiPocket→GotenNet→GDEGAN lineage cannot be reconstructed from the papers. Any new paper quoting a published failure rate is quoting an undefined quantity.
- Prediction-success convention in the same lineage: *"Predictions are successful when DCC or DCA < 4Å"*, with mean-shift clustering on high-scoring residues *"following (Krivák & Hoksza, 2018)"* to obtain pockets. — [PRIMARY] [GDEGAN](https://arxiv.org/pdf/2603.19817)

### Inferences
- There are now **two independent, numerically supported calls (2024 LIGYSIS, 2025–26 UniSite) to abandon the 4 Å DCC / unmatched-DCA protocol**, from different groups and with different replacements (top-N+2 recall + residue F1/MCC vs IoU-AP). A new paper is on solid ground reporting both families and noting they disagree on the remedy.
- The metric inherited by the whole deep-learning lineage comes from a 2011 four-method comparison, via P2Rank, with the DCC/DCA labels already muddled at the source.

### Gaps
- Chen et al. (2011) itself was not read in this session (ResearchGate only); the "4 Å originates there" claim is [PRIMARY as cited by P2Rank] but [SECOND-HAND] as to Chen et al.'s own rationale for choosing 4 Å. Worth reading the original to state the origin cleanly.

---

## Q6. Statistical practice: confidence intervals, seeds, paired tests

### Takeaway
Practice is thin but not absent. LIGYSIS reports **95% confidence intervals and significance tests (p ≤ 0.05)** but no seeds or run-to-run variance. Method papers in the EquiPocket/GDEGAN lineage report **standard deviations in parentheses** but no paired tests and no cluster-level resampling. I found **no paper that performs a cluster-level bootstrap** of binding-site results, and therefore **no published estimate of what fraction of claimed improvements would survive one** — that question is, as far as I can tell, unanswered in the literature.

### Cited findings
- LIGYSIS: *"Error bars represent 95% confidence interval"* and *"95% CI of the recall... which is 100 × proportion"* — i.e. binomial CIs over protein chains. Significance is asserted with p-values: *"Removing redundancy increases significantly the recall of VN-EGNN, IF-SitePred and DeepPocketSEG by > 5%"* with *"p ≤ 0.05"* / *"p > 0.05"* annotations. — [PRIMARY] [LIGYSIS 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/)
- LIGYSIS has **no discussion of random seeds, repeated runs, or training variance** (it benchmarks released weights, so there is nothing to seed). — [PRIMARY, by absence]
- GDEGAN's main table reports **standard deviations in brackets** (e.g. values printed as `0.788(0.011)`, `0.675(0.010)`, `0.826(0.011)`), with the footnote *"The standard deviation is indicated in brackets."* No paired test against baselines, no seed count stated in the extracted text. — [PRIMARY] [arXiv:2603.19817](https://arxiv.org/pdf/2603.19817)
- The CI/proportion basis in LIGYSIS is **per-chain binomial**, which treats homologous chains as independent observations — i.e. even the best statistical reporting in the field does not cluster. — [PRIMARY] [inference about its consequence is mine]
- For contrast, the adjacent affinity field has produced explicit bias-resolution and leak-aware re-evaluation papers with effect sizes ([LP-PDBBind](https://arxiv.org/html/2308.09639v2); [Nat. Mach. Intell. 2025](https://www.nature.com/articles/s42256-025-01124-5); [HonestAffinity](https://arxiv.org/html/2606.03422)) — [SECOND-HAND as to their internal statistics].

### Inferences
- "What fraction of claimed improvements would survive a cluster-level bootstrap?" is **not answered anywhere I could find**. Given that (a) ion inclusion alone is worth 5–10 points, (b) redundancy handling is worth 5–13 points, and (c) typical claimed margins in method papers are a few points (GDEGAN claims 7.7–19.0% relative DCA improvement over EquiPocket, i.e. a few absolute points), the honest statement is that the protocol degrees of freedom are larger than the claimed effects. That is an argument from measured effect sizes, not from a bootstrap anyone has run.

### Gaps
- No critique paper specifically targeting statistical reporting in *binding-site* prediction was found. The nearest things are the general ML-in-biology reporting-standards literature (DOME-style recommendations) which I did not verify in this session, and the affinity-prediction bias papers.
- No published cluster-level bootstrap of binding-site benchmarks.

---

## Q7. Copied baseline numbers

### Takeaway
It is demonstrably standard practice and is **visible in the table footnotes of current papers**: GDEGAN (2026) marks part of its comparison table as *"Results from the EquiPocket (Zhang et al., 2024) paper."* I found **no paper that explicitly criticises the practice by name**, but LIGYSIS and UniSite both demonstrate *why* it is unsafe — the same "HOLO4K DCA" number means different things under different receptor and redundancy conventions.

### Cited findings
- GDEGAN's Table 1 footnote, verbatim from the PDF: *"a The standard deviation is indicated in brackets. b Results from the EquiPocket (Zhang et al., 2024) paper."* The baseline roster so inherited includes Fpocket, P2Rank, DeepSite, Kalasanty, RecurPocket, GAT, GCN, GCN2, SchNet, EGNN and EquiPocket. — [PRIMARY] [DEMONSTRATED — this is a direct instance] [arXiv:2603.19817](https://arxiv.org/pdf/2603.19817)
  - Note also that GDEGAN cites "DeepSite (Aggarwal et al., 2021)" — the Aggarwal reference is DeepPocket, not DeepSite — a citation error consistent with a baseline block carried over rather than re-run. [PRIMARY] [my observation from their reference list]
- GDEGAN preprocesses scPDB and selects the PDBbind refined set *"as per (Zhang et al., 2024)"*, i.e. the data pipeline is also inherited from EquiPocket. — [PRIMARY]
- LIGYSIS positions itself as *"the first independent ligand site prediction benchmark for over a decade, since Chen et al. (2012)"* — an implicit indictment of a decade of self-reported, mutually-copied comparisons. They do **not** state whether they could reproduce prior authors' published numbers. — [PRIMARY] [ASSERTED]
- The mechanism that makes copying misleading is demonstrated rather than asserted: differing filtering *"result[s] in slightly different subsets, which means a direct comparison of methods based on these benchmarks may not be straightforward"* [SECOND-HAND / ASSERTED], while LIGYSIS's +5.2/+13.4/+5.6 redundancy effects and UniSite's 20% double-counting figure show the size of the discrepancy [PRIMARY / DEMONSTRATED].

### Inferences
- The defensible claim for a new paper: baseline numbers copied across the EquiPocket→GotenNet→GDEGAN chain carry an unknown and unstated protocol (receptor granularity, mlig or not, redundancy merging, failure-rate definition), and at least one quantity in that chain ("failure rate") is printed with a formula that contradicts the reported values. Re-running baselines under one's own fixed protocol is therefore not optional politeness but a correctness requirement.

### Gaps
- No explicit published critique of number-copying in this subfield. The strongest available support is the direct footnote evidence plus the demonstrated protocol sensitivity.

---

## Q8. Reproducibility: independent re-runs, and how many methods cannot be run at all

### Takeaway
LIGYSIS is the one large independent re-run, and its exclusion list is the hard evidence: **at least 12 published methods could not be included "due to technical reasons"**, and of those that ran, several silently fail on large fractions of inputs (IF-SitePred predicts on only 75% of chains). The authors explicitly call on the community to release code, models and benchmarking scripts.

### Cited findings
- **Methods excluded for technical reasons (verbatim list):** *"RefinePocket, EquiPocket, GLPocket, SiteRadar, NodeCoder, RecurPocket, PointSite, DeepSurf, Kalasanty, BiteNet, GRaSP, or DeepSite were not included... due to technical reasons."* Inclusion criteria were *"open-source methods with publicly accessible code, clear installation instructions, well defined dependencies, accessible command line interfaces and trained machine learning models."* — [PRIMARY] [DEMONSTRATED] [LIGYSIS 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/)
  - **This is the answer to "how many published methods cannot be run at all for want of weights": ≥12, named, in one 2024 benchmark — including EquiPocket, whose numbers are nonetheless the ones being copied forward into 2026 papers (Q7).**
- Of the 13 that did run: *"VN-EGNN failed with an error for PDB: 6BCU chain: A. The rest of the methods ran successfully for all protein chains."* But coverage gaps remain: *"If-SitePred only predicts pockets on 75% of the chains"*; *"PUResNet, DeepPocket, P2RankCONS and P2Rank often don't predict on smaller proteins (< 100 amino acids)."* — [PRIMARY] [DEMONSTRATED]
- All methods were run *"with their default settings"*, with a supplementary table recording which methods even output pocket centroids, residues, scores and rankings — a prerequisite for comparable scoring that is not uniformly met. — [PRIMARY]
- Explicit community recommendation: *"All authors of ligand site prediction tools should... Benchmarking code should also be shared by the authors for the sake of reproducibility."* — [PRIMARY]
- A second, smaller independent re-run: Lee, Byun & Shin re-trained DeepPocket and DeepSurf themselves on a leakage-controlled split rather than citing their numbers. — [PRIMARY] [arXiv:2303.08818](https://arxiv.org/pdf/2303.08818)
- LIGYSIS's companion results are archived: [Zenodo record 13121414, "LBS-Comparison results"](https://zenodo.org/records/13121414) — [SECOND-HAND, surfaced in search; not opened].
- The LIGYSIS data resource itself: [LIGYSIS-web, Nucleic Acids Research 2025](https://academic.oup.com/nar/advance-article/doi/10.1093/nar/gkaf411/8133629) — [SECOND-HAND, not opened].

### Inferences
- The combination of Q7 and Q8 is the field's sharpest internal contradiction: EquiPocket is simultaneously (a) unrunnable by an independent benchmarking group in 2024 and (b) the source of the baseline column in 2026 method tables.

### Gaps
- No systematic reproducibility audit (e.g. "we attempted to install N methods, M worked") exists beyond LIGYSIS's inclusion/exclusion statement; LIGYSIS does not report *why* each of the 12 failed.

---

## Q9. Is there a recognised community blind assessment / CASP-like exercise?

### Takeaway
Yes, historically: ligand-binding-*residue* prediction was assessed blind in the **CASP function-prediction (FN) category in CASP8, CASP9 and CASP10**, using MCC and the Binding-site Distance Test (BDT); best groups reached mean MCC ≈ 0.62 in CASP9. The category then lapsed, and CASP15/CASP16 moved to **protein–ligand complex and pose/affinity prediction**, not site detection. So there is currently **no active blind assessment of binding-site detection** — which is exactly the vacuum LIGYSIS says it is filling.

### Cited findings
- *"Methods for ligand binding site residue predictions are assessed in the function prediction category of the biennial CASP experiment, utilizing the Matthews Correlation Coefficient (MCC) and Binding-site Distance Test (BDT) metrics."* — [SECOND-HAND synthesis over the CASP assessment papers]
- CASP9: *"The overall accuracy of ligand-binding site predictions in CASP9 showed average Matthews correlation coefficient of 0.62 for the 10 top performing groups."* — [SECOND-HAND] [DEMONSTRATED in the original] [Assessment of ligand-binding residue predictions in CASP9](https://www.researchgate.net/publication/51707057_Assessment_of_ligand-binding_residue_predictions_in_CASP9)
- CASP8: blind assessment of ligand binding site prediction included; homologous structures + conservation were the successful approach. — [SECOND-HAND] [PMC2814558](https://pmc.ncbi.nlm.nih.gov/articles/PMC2814558/)
- CASP10: assessed within the *"function prediction (prediction of binding sites, FN)"* category. — [SECOND-HAND] [Assessment of ligand binding site predictions in CASP10, Proteins 2014](https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.24495) / [PubMed 24339001](https://pubmed.ncbi.nlm.nih.gov/24339001/) (PubMed page was cookie-walled in this session)
- CASP15/16 shifted scope: *"The ligand prediction challenge in CASP15 differed from past ligand prediction challenges, with newer challenges focusing on protein-ligand complex prediction."* — [SECOND-HAND] [Assessment of protein–ligand complexes in CASP15](https://www.researchgate.net/publication/374489672_Assessment_of_protein-ligand_complexes_in_CASP15); [Assessment of Pharmaceutical Protein–Ligand Pose and Affinity Predictions in CASP16](https://pmc.ncbi.nlm.nih.gov/articles/PMC12750038/)
- LIGYSIS: *"This work represents the first independent ligand site prediction benchmark for over a decade, since Chen et al. (2012)"* — consistent with the FN category having lapsed. — [PRIMARY]

### Inferences
- A new paper can accurately say: binding-site *detection* has had no blind community assessment since CASP10 (2012); the current de facto standards are self-reported COACH420/HOLO4K tables and two independent third-party benchmarks (LIGYSIS 2024, UniSite 2025–26).
- CASP's metrics (MCC, BDT) are residue-level, which is closer to LIGYSIS's residue F1/MCC and UniSite's IoU-AP than to DCC/DCA — i.e. the blind-assessment community never used the pocket-centre-distance metric the method papers standardised on.

### Gaps
- I did not confirm from primary text why the FN category was discontinued, nor read the CASP10 assessment's conclusions directly (both pages were inaccessible in this session). CAMEO does not, as far as I could determine, run a continuous ligand-binding-site category — unverified.

---

## Q10. Position papers / reviews arguing that simple geometric or feature-based methods remain competitive

### Takeaway
There is no polemical position paper, but **LIGYSIS's result is the substantive version of that argument, demonstrated with numbers**: the best performers in the only independent 2024 benchmark are **fpocket (a 2009 geometric method) re-scored by PRANK or DeepPocket**, at 60% top-N+2 recall, while the newest end-to-end deep method tested (IF-SitePred) is last at 39%. Separately, fpocket was the only method *not* helped by removing ions, and P2Rank (a 2018 random-forest method) handles multi-chain assemblies that several deep methods cannot.

### Cited findings
- *"re-scoring of fpocket predictions by PRANK and DeepPocket display the highest recall (60%)"*, while *"IF-SitePred achieved the lowest at 39%"*; *"stronger scoring schemes improved performance by up to 14% in recall and 30% in precision."* — [PRIMARY] [DEMONSTRATED] [LIGYSIS 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/)
  - **Reading:** the gain is attributed to *scoring*, applied on top of a classical geometric pocket generator — not to learned pocket generation.
- The deep methods are the ones generating the most redundant output (VN-EGNN 67%, IF-SitePred 49%, DeepPocketSEG 31%), and fixing that is what gives them their largest single improvement. — [PRIMARY] [DEMONSTRATED]
- *"When ions are removed, all methods, except fpocket, experience an increase in (top-N+2) recall of 5–10%"* — fpocket is uniquely insensitive to the ligand-relevance convention. — [PRIMARY] [DEMONSTRATED]
- Counter-position (the method-paper consensus, [SECOND-HAND], [ASSERTED] on its own benchmarks): *"FPocket is clearly inferior to other methods in both datasets"*; DeepSurf, DeepPocket etc. report outperforming it on COACH420/HOLO4K. Also noted as a real limitation: *"FPocket typically predicts many, often overlapping alpha spheres, making it difficult or even impossible to segment the protein surface clearly"* — see e.g. [DeepSurf, arXiv:2002.05643](https://arxiv.org/pdf/2002.05643), [EquiPocket, arXiv:2302.12177](https://arxiv.org/pdf/2302.12177).
  - **The conflict is direct:** self-reported method-paper tables put fpocket last; the independent benchmark puts fpocket-plus-rescoring first. This contradiction should be stated, not resolved by picking a side.
- The 2026 shortcut-learning result in the adjacent sequence-based setting is the strongest "deep models are not learning sites" evidence: FusionDTI reaches **98.3% AUROC / 90.1% accuracy** on binary binding prediction but only **21.6% BRHR@1** for binding-site localisation — *"strong binary binding prediction does not yet translate into reliable top-ranked binding-site localization."* — [PRIMARY] [DEMONSTRATED] [arXiv:2605.24045](https://arxiv.org/html/2605.24045v1)
- Broader review context (not a critique paper per se): [Computational methods for binding site prediction on macromolecules, Q. Rev. Biophys.](https://www.cambridge.org/core/journals/quarterly-reviews-of-biophysics/article/computational-methods-for-binding-site-prediction-on-macromolecules/9B72B038954130A5B4BE810FAFEDE31E) — [SECOND-HAND, not opened].

### Inferences
- The honest framing for a new paper: in the only independent head-to-head of the last decade, a 2009 geometric detector with a learned re-scorer beats every end-to-end deep detector tested. That is a demonstrated finding, not a polemic, and it is the single most useful citation for justifying a geometric-candidate + learned-ranking architecture.

### Gaps
- No explicit position/opinion paper arguing "deep learning for binding-site prediction is overclaimed" was found. The argument exists only as benchmark results plus the shortcut-learning evidence.

---

## Cross-cutting: a defensible protocol checklist implied by the above

Every item below is a degree of freedom for which the literature gives a measured effect size, so each must be stated explicitly:

1. **Receptor granularity** — single chain / deposited asymmetric unit / biological assembly. Affects 40% of HOLO4K and 56% of COACH420 entries [LIGYSIS, PRIMARY].
2. **Reference-site de-duplication** — up to 14× interface redundancy within one PDB entry [LIGYSIS, PRIMARY].
3. **Prediction de-duplication** — centroid ≤ 5 Å or residue JI > 0.75; worth +5.2 to +13.4 recall points depending on method [LIGYSIS, PRIMARY].
4. **Ligand relevance rule** — MOAD-derived `mlig`, BioLiP, heavy-atom rule, or all HET. Ion exclusion alone is worth 5–10 recall points [LIGYSIS, PRIMARY]. State the structure count (COACH420 = 420 or 284? HOLO4K = 3,204 / 4,009 / 4,288 / 4,543?).
5. **Metric and threshold** — DCA 4 Å and/or DCC 10–12 Å (not DCC 4 Å) [LIGYSIS, PRIMARY]; plus a matched, structure-aware metric (residue F1/MCC, or IoU-AP at 0.3/0.5) [UniSite, PRIMARY].
6. **Rank cutoff** — top-N and top-N+2, with N defined as the number of *relevant* ligands/sites; justify n+2 by the 33–50% unobserved-site estimate [LIGYSIS, PRIMARY].
7. **Leakage control** — cluster-level, with the tool and threshold named; note that the field's convention (50% identity, protein-code set difference) is lax and that COACH420/HOLO4K are disjoint from *P2Rank's* training data only, not from scPDB [P2Rank, PRIMARY].
8. **Baselines re-run, not copied** — at minimum flag which published numbers are inherited and from where, as GDEGAN does [PRIMARY], and note that ≥12 published methods were unrunnable for an independent group in 2024 [LIGYSIS, PRIMARY].
9. **Statistics** — per-site binomial CIs are the current best practice [LIGYSIS, PRIMARY]; cluster-level resampling and paired tests are not done by anyone found, so doing them is a differentiator rather than a convention.

## Overall gaps (for the report writer)
- The four most valuable missing numbers in the literature: (i) clustered scPDB↔COACH420/HOLO4K overlap at a stated threshold; (ii) same-model accuracy under single-chain vs biological-assembly receptors; (iii) same-model accuracy on all-ligands vs `mlig`; (iv) fraction of published improvements surviving a cluster-level bootstrap. None of these is answered in print, which makes each an opportunity rather than a citation.
- Sources that were inaccessible in this session and should be checked before final write-up: P2Rank Additional file 1 (relevant-ligand filter, HTTP 403); CASP10 FN assessment (cookie wall); the OpenReview "pickpocket" submission (bot wall); Chen et al. 2011 full text; LIGYSIS-web NAR 2025; Zenodo 13121414.
