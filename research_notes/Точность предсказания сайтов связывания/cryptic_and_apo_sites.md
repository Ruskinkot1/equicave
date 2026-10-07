# CRYPTIC and APO binding-site prediction as of 2026: what works when the pocket is closed

> **Provenance convention used throughout.**
> `[V]` = verified from a primary source (paper full text, publisher/PMC page, official repo page, or publisher API record) that I fetched in this session.
> `[2H]` = second-hand (search-result summary, review article reporting someone else's number, or an aggregator) — the number is reported but I did not open the primary paper.
> `[UNVERIFIED]` = could not reach / could not confirm.
>
> **Task separation enforced throughout.** Three different tasks are routinely conflated in this literature:
> - **(A) Site detection on apo input** — "given a closed, unbound structure, point at the place where a ligand will bind." Measured with pocket-level hit criteria (DCA/DCC, overlap, top-n).
> - **(B) Per-residue crypticity propensity** — "which residues will participate in a pocket that is not there now." Measured with residue-level ROC-AUC / AUPRC. This is what CryptoSite, PocketMiner and the CryptoBench baseline actually do.
> - **(C) Quantitative opening thermodynamics** — "what fraction of the time is this pocket open, and will this mutation open it more." Measured against experimental populations.
> Nearly all headline AUCs in this field are task (B), not task (A). A model whose apo candidate-generation ceiling is the bottleneck is squarely a **task (A)** problem, and task (A) numbers are much scarcer.

---

## BENCHMARKS: standard cryptic-site datasets, sizes, pair definitions, criteria, downloadability

### Takeaway
There are four established cryptic-site datasets (CryptoSite 2016, PocketMiner 2023, CryptoBench 2024, CryptoBank 2025) plus ASBench for allosteric sites; they differ by ~2 orders of magnitude in size, and **all of them define the task per-residue, not per-pocket**, which is why almost no published number is directly comparable to a top-n candidate-generation ceiling. CryptoBench is the only one shipping predefined leakage-controlled CV splits, and it is the only one that explicitly reports the same method on apo and on holo input.

### Cited Findings

**CryptoSite (Cimermancic et al., JMB 2016)**
- Built on a benchmark of **93 unbound–bound (apo–holo) protein pairs**; CryptoBench's own comparison table lists it as **93 apo structures / 98 cryptic pockets**, average pocket RMSD **2.65 ± 2.36 Å**, **0 promiscuous pockets, 0 multi-chain pockets** — [CryptoBench, PMC11725321](https://pmc.ncbi.nlm.nih.gov/articles/PMC11725321/) `[V]`
- The 2025 Bioinformatics Advances review describes the composition as "84 known examples of cryptic binding sites, 92 binding sites, and 705 concave surface patches", drawn from the PDB and MOAD — [Bioinformatics Advances vbaf156](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf156/8180504) `[V]`
- Definition of cryptic used: "a site that forms a pocket in a holo structure, but not in the apo structure" — [search summary of CryptoSite](https://www.researchgate.net/publication/293195942_CryptoSite_Expanding_the_Druggable_Proteome_by_Characterization_and_Prediction_of_Cryptic_Binding_Sites) `[2H]`
- Download page `https://modbase.compbio.ucsf.edu/cryptosite/download` **returned HTTP 200** when probed on 2026-10-07 (reachability only; contents not inspected) — probe in this session `[V]`

**PocketMiner validation set (Meller et al., Nat Commun 2023)**
- **38 apo/holo protein pairs containing 39 cryptic pockets**, filtered from the PDB with resolution ≥2.5 Å and sequence-identity criteria; average pocket RMSD **3.46 ± 1.99 Å**; 0 promiscuous, 0 multi-chain pockets — [Nat Commun / PMC9977097](https://pmc.ncbi.nlm.nih.gov/articles/PMC9977097/) `[V]`; RMSD/promiscuity figures from [CryptoBench Table 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC11725321/) `[V]`
- The *evaluation* set is residue-level and deliberately includes hard negatives: **24 apo structures with ligand-binding pockets, 4 hyper-rigid proteins, 7 extensively ligand-screened proteins — 563 positive residues and 1,283 negative residues in total** — [PMC9977097](https://pmc.ncbi.nlm.nih.gov/articles/PMC9977097/) `[V]`
- It distinguishes **"forward pockets"** (residues farther apart in apo) from **"reverse pockets"** (residues closer in apo) — [PMC9977097](https://pmc.ncbi.nlm.nih.gov/articles/PMC9977097/) `[V]`
- Negatives were *simulation-verified* rigid proteins (ubiquitin, gamma-B crystallin, designed proteins) plus ligand-screened MOAD proteins — [PMC9977097](https://pmc.ncbi.nlm.nih.gov/articles/PMC9977097/) `[V]`

**CryptoBench (Škrhák et al., Bioinformatics 2024/2025; btae745)**
- **1,107 apo structures, 1,361 cryptic pockets** — the largest curated CBS dataset at publication; average pocket RMSD **2.89 ± 0.87 Å**, **16.60 ± 7.22 binding residues per protein**; includes **371 promiscuous pockets and 197 multi-chain pockets**, both absent from CryptoSite and PocketMiner sets — [PMC11725321](https://pmc.ncbi.nlm.nih.gov/articles/PMC11725321/) `[V]`
- **Pair definition and filtering pipeline:** apo–holo pairs come from **AHoJ-DB**, starting from **14,054,029 candidate pairs**, then sequentially filtered on resolution ≥2.5 Å, matching binding-residue counts between apo and holo, geometric sanity (**TM-score ≥0.5, centre distance ≤4 Å, ≤20% change in radius of gyration**), and ligand heavy-atom count ≥5 — [PMC11725321](https://pmc.ncbi.nlm.nih.gov/articles/PMC11725321/) `[V]`
- **Crypticity threshold:** a site counts as cryptic if the **binding residues differ by ≥2 Å RMSD between holo and apo**. Justification given: general (non-cryptic) sites typically stay under ~1.5 Å pocket RMSD, cryptic ones above 2 Å — [PMC11725321](https://pmc.ncbi.nlm.nih.gov/articles/PMC11725321/) `[V]`
- **Leakage control:** two clustering rounds, **40% sequence identity** first, then **10% identity** to separate train from test — [PMC11725321](https://pmc.ncbi.nlm.nih.gov/articles/PMC11725321/) `[V]`
- **Splits:** 80:20 → **222 apo test structures / 885 train**; train further split into 4 folds (222/222/222/219) — [PMC11725321](https://pmc.ncbi.nlm.nih.gov/articles/PMC11725321/) `[V]`
- **Success criterion: binary per-residue classification** (binding vs non-binding residue of a cryptic site), scored with AUC, AUPRC, accuracy, FPR, TPR, MCC, F1 — [PMC11725321](https://pmc.ncbi.nlm.nih.gov/articles/PMC11725321/) `[V]`
- **Downloadable:** dataset (CIF files, splits, visualisation scripts) at **https://osf.io/pz4a9/** (HTTP 200 on probe 2026-10-07); scripts at **https://github.com/skrhakv/CryptoBench/** under **MIT** — [OSF probe, this session] `[V]`; [repo page](https://github.com/skrhakv/CryptoBench/) `[V]`

**CryptoBank (Febrer Martinez, Borsatto, Fröhlking, Gervasio; bioRxiv 2025-04-26, published in Science Advances, DOI 10.1126/sciadv.ady6364)**
- Built by applying an ML model for ligand-induced conformational change to **over 5.5 million apo/holo structural alignments from the PDB** — [bioRxiv API record, DOI 10.1101/2025.04.23.650184](https://www.biorxiv.org/content/10.1101/2025.04.23.650184v1) `[V]`
- Headline statistic: **cryptic pockets occur in ~16.3% of protein clusters** — [same record](https://www.biorxiv.org/content/10.1101/2025.04.23.650184v1) `[V]`
- Described elsewhere as "the largest known database of cryptic sites", 5.5M alignments — [Zhang & Bowman review, PMC12959236](https://pmc.ncbi.nlm.nih.gov/articles/PMC12959236/) `[2H]`
- Accessible "via a web server" per the abstract; **I could not confirm the web-server URL** — my guessed host `cryptobank.sysbio.unige.ch` failed to connect (proxy 502). Treat the access route as `[UNVERIFIED]`; the Science Advances paper's data-availability section is the place to look.
- A 286-chain subset of CryptoBank is used as a second test set by the 2026 world-model paper (below), implying chain-level data are extractable — [bioRxiv API, DOI 10.64898/2026.09.21.752781](https://www.biorxiv.org/content/10.64898/2026.09.21.752781v1) `[V]`

**ASBench (allosteric, not strictly cryptic)**
- **Core set: 235 unique allosteric sites; Core-Diversity set: 147 structurally diverse allosteric sites**; companion Allosteric Database (ASD) holds 1,949 allosteric proteins — [search summary via PMC8767309 and related](https://pmc.ncbi.nlm.nih.gov/articles/PMC8767309) `[2H]`
- **Download location not confirmed.** The ASD-group URL I probed (`https://mdl.shsmu.edu.cn/ASBench/`) returned **404**. `[UNVERIFIED]`
- Note: ASBench targets *allosteric* sites, which overlap with but are not the same population as cryptic sites — many allosteric sites are open in apo, and many cryptic sites are orthosteric.

**Other / newer sets**
- **TACTICS** rebuilt a database from CryptoSite with a controlled negative ratio: "a fixed number of 50 apo structures per 1 holo structure" — [vbaf156 review](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf156/8180504) `[V, review reporting]`
- **BioEmu cryptic-pocket benchmark:** a set of **34 apo/holo pairs** where a binding site forms only in holo — [search summary of BioEmu preprint](https://www.biorxiv.org/content/10.1101/2024.12.05.626885v1.full) `[2H]`
- **Bowman-lab 2026 thermodynamic benchmark:** only **two systems** (Ebola VP35 and TEM β-lactamase) but with experimentally measured open-state populations and mutational series — [bioRxiv API, DOI 10.64898/2026.01.21.700870](https://www.biorxiv.org/content/10.64898/2026.01.21.700870v1) `[V]`

### Inferences
- For a pipeline whose constraint is **apo candidate generation**, CryptoBench is the only dataset with enough apo structures (1,107) *and* published splits to retrain on; but its native criterion is per-residue, so you have to define your own pocket-level hit rule on top of it — exactly what the 2026 world-model paper did (231 held-out apo chains, pocket-level overlap).
- CryptoBench's inclusion of **197 multi-chain** and **371 promiscuous** pockets is the practical reason PocketMiner could not even be scored on 60 of 222 test structures; any apo evaluation built on CryptoBench must decide explicitly how to handle multi-chain sites.
- The ~2 Å pocket-RMSD cut is a *soft* definition. CryptoSite's 2.65 ± 2.36 Å and PocketMiner's 3.46 ± 1.99 Å means those sets contain much more extreme (and much more variable) motion than CryptoBench's 2.89 ± 0.87 Å. Cross-dataset AUC comparisons are therefore not apples-to-apples, and the CryptoBench authors say so.

### Gaps
- Exact CryptoSite file inventory and its own train/test split definition — I confirmed only that the download page responds, not its contents.
- ASBench canonical download URL and current reachability.
- CryptoBank's public web-server URL and licence.
- Whether any 2026 benchmark exists that is natively **pocket-level on apo input**; the world-model paper appears to define its own rule rather than adopting a community one.

---

## PUBLISHED RESULTS: absolute numbers with protocol attached

### Takeaway
Residue-level ROC-AUC on cryptic sites sits in a narrow 0.76–0.89 band for every method ever published, and the single cleanest apo-vs-holo contrast in the literature — P2Rank on CryptoBench — is **AUC 0.89 holo → 0.81 apo, AUPRC 0.34 → 0.21**. At pocket level on apo input, the only 2026 numbers are top-1 ≈0.85 / top-5 ≈0.95, which is strikingly close to the 0.85 apo ceiling in the user's own system.

### Cited Findings

**CryptoBench benchmark, Table 5 — per-residue, binary, test set of 222 apo structures (subsets as defined above)** — all from [PMC11725321](https://pmc.ncbi.nlm.nih.gov/articles/PMC11725321/) `[V]`

| Method | Dataset | AUC | AUPRC | ACC | FPR | TPR | MCC | F1 |
|---|---|---|---|---|---|---|---|---|
| pLM-NN (ESM2-3B + NN) | CB-full | 0.86 | 0.36 | 0.93 | 0.05 | 0.48 | 0.39 | 0.92 |
| pLM-NN | CB-PM | 0.88 | 0.43 | 0.93 | 0.04 | 0.52 | 0.44 | 0.93 |
| **PocketMiner** | CB-PM | **0.76** | **0.19** | 0.82 | 0.16 | 0.51 | 0.22 | 0.78 |
| pLM-NN | CB-P2RANK-apo | 0.88 | 0.42 | 0.93 | 0.04 | 0.51 | 0.43 | 0.93 |
| **P2Rank (apo input)** | CB-P2RANK-apo | **0.81** | **0.21** | 0.85 | 0.14 | 0.62 | 0.27 | 0.81 |
| **P2Rank (holo input)** | CB-P2RANK-holo | **0.89** | **0.34** | 0.85 | 0.15 | 0.84 | 0.38 | 0.81 |

- Subset definitions: **CB-PM** = the test structures PocketMiner could actually run on (**22 structures errored, 38 multi-chain structures excluded**, out of 222); **CB-P2RANK-apo/holo** = single-chain apo structures, with the holo variant using the paired bound structure, selecting the partner with the largest pocket RMSD — [PMC11725321](https://pmc.ncbi.nlm.nih.gov/articles/PMC11725321/) `[V]`
- Note the mechanism of the holo→apo drop for P2Rank: **TPR collapses 0.84 → 0.62 at essentially unchanged FPR (0.15 → 0.14)**. That is a *detection/recall* loss, not a ranking loss. `[V — arithmetic on the cited table]`
- The pLM-NN baseline "outperformed PocketMiner and P2Rank across key metrics, including AUC, AUPRC, MCC and F1" — [PMC11725321](https://pmc.ncbi.nlm.nih.gov/articles/PMC11725321/) `[V]`

**CryptoSite (2016)** — residue-level
- "ROC-AUC = 0.83" for classifying whether a residue participates in a cryptic pocket; a feature-selection variant reported AUC 0.77 — [search summaries of the JMB paper](http://cdn.fraserlab.com/publications/2016_cimermancic.pdf) `[2H]`
- PocketMiner's own head-to-head reports **CryptoSite ROC-AUC 0.85** on PocketMiner's experimental test set, vs PocketMiner 0.87 — [PMC9977097](https://pmc.ncbi.nlm.nih.gov/articles/PMC9977097/) `[V]`
- Zhang & Bowman's review also lists CryptoSite at **ROC-AUC 0.85** with "one day per input structure" — [PMC12959236](https://pmc.ncbi.nlm.nih.gov/articles/PMC12959236/) `[V, review reporting]`
- Conflict to flag: 0.77 / 0.83 (CryptoSite's own paper, on its own benchmark) vs 0.85 (as re-measured by PocketMiner on the PocketMiner test set). These are different test sets; do not average them.

**PocketMiner (Nat Commun 2023)** — all from [PMC9977097](https://pmc.ncbi.nlm.nih.gov/articles/PMC9977097/) `[V]`
- On experimental apo structures (563 pos / 1,283 neg residues): **ROC-AUC 0.87 ± 0.04**, **PR-AUC 0.44 ± 0.12**
- On its native training task (predicting pocket formation within a 40 ns simulation window from the starting structure): **ROC-AUC 0.83**
- **>1,000-fold faster than CryptoSite**; **<1 s per protein**
- Protocol: 5-fold CV split by **CATH topology code**; test proteins <55% sequence identity to training proteins
- Applied proteome-wide, the authors conclude "over half of proteins thought to lack pockets based on available structures likely contain cryptic pockets" — [search summary / Nat Commun abstract](https://www.nature.com/articles/s41467-023-36699-3) `[2H]`

**Pocket-level, apo input, 2026 — "Ability of a Structural World Model to Detect Cryptic Pockets from Apo Structure" (Shihabi, Taraman, Vaughan; Eratos Therapeutics; bioRxiv v1 2026-09-22, v2 2026-09-24, DOI 10.64898/2026.09.21.752781, licence CC-BY-NC-ND, not yet journal-published)**
- **CryptoBench, 231 held-out apo chains, pocket-level overlap rule: top-1 0.848, top-3 0.931, top-5 0.952** — [search summary of the preprint](https://www.researchgate.net/publication/414630935_A_World_Model_of_Molecular_Organization_Detects_Cryptic_Pockets_from_Apo_Structure) `[2H — numbers appear in the search result summary, not in the abstract I verified]`
- **CryptoBank 286-chain subset: 0.846 / 0.933 / 0.990** — same source `[2H]`
- Verified from the bioRxiv API abstract: "localizes cryptic sites with **top-1 accuracy as high as 0.848 and top-5 accuracy exceeding 0.93**"; "on **190 targets held out of both training sets it recovers 119 sites OpenDDE misses, against 2 in the other direction**, and at top-5 co-folding's recovered sites are a subset of ours" — [bioRxiv API record](https://www.biorxiv.org/content/10.64898/2026.09.21.752781v1) `[V]`
- **The exact pocket-level overlap rule is not stated in the abstract and I could not fetch the full text** (bioRxiv returned HTTP 429 and Cloudflare error 1015 on repeated attempts). Treat the top-n definition as `[UNVERIFIED]`.

**General pocket finders, apo vs holo (not cryptic-specific) — DeepSurf, CHEN apo/holo subsets, DCA criterion with Dcut = 4 Å, 104 holo + 104 apo proteins, apo structurally aligned onto holo and holo ligands transferred** — all from the DeepSurf paper PDF (Mylonas, Axenopoulos, Daras), extracted locally `[V]`

| Method | CHEN holo Top-n | CHEN holo Top-(n+2) | CHEN apo Top-n | CHEN apo Top-(n+2) |
|---|---|---|---|---|
| Jiang et al. | 34.5 | 35 | 28.4 | 30.5 |
| Kalasanty | 35.5 | 36 | 33 | 34 |
| DeepSurf (ResNet-18) | 40.6 | 40.6 | 39.6 | 39.6 |
| DeepSurf (Bot-LDS-ResNet-18) | 39.1 | 39.1 | 37.6 | 37.6 |

- Authors' own summary: "although all methods perform worse on the apo case, DeepSurf exhibits the **smallest decrease in accuracy (1–1.5%)**" — DeepSurf PDF `[V]`
- The same paper walks through a cryptic case explicitly: apo `2iyt` vs holo `2iyq`, a pair from Cimermancic's set, which DeepSurf hits in both forms — DeepSurf PDF `[V]`
- Caveat: these CHEN absolute values (28–41%) are low because CHEN is a hard, sequence-diverse set; they are **not** comparable to COACH420/HOLO4K numbers.

**Quantitative opening probability (task C), 2026** — Zhang, Miller, Bowman; bioRxiv DOI 10.64898/2026.01.21.700870 (v1 2026-01-23 "AI-Based Methods for Cryptic Pocket Detection Are Fast and Qualitative Compared to Quantitatively Predictive Simulations"; v2 2026-04-01 retitled "How Well Can AI and Physics-Based Simulations Predict the Probability a Cryptic Pocket Is Open?"); **published in J. Chem. Theory Comput., DOI 10.1021/acs.jctc.6c00135** — [bioRxiv API record](https://www.biorxiv.org/content/10.64898/2026.01.21.700870v1) `[V]`
- Methods benchmarked: **AlphaFlow, BioEmu, PocketMiner, CryptoBank, and physics-based MD**, on Ebola VP35 and TEM β-lactamase with experimentally characterised open-state probabilities and mutational effects `[V]`
- Findings, verbatim from the abstract: "Multiple methods are remarkably successful at predicting **whether a mutation will increase or decrease** the probability of cryptic pocket opening. However, **none can reliably predict the absolute probability of pocket opening**. MD is very close for the two wild-type proteins but all the methods struggle for pockets with small probabilities of opening experimentally (e.g. less than 1%). BioEmu and PocketMiner capture some trends between variants with experimental populations over 1% but have **systematic errors and poorer performance for rare pockets (<1% open)**" `[V]`
- No per-system numeric table verified — full text not fetched. `[Gap]`

### Inferences
- The best-documented apo penalty for a *general* pocket finder on cryptic-enriched data is CryptoBench's P2Rank row: **−0.08 AUC and −0.13 AUPRC**, driven almost entirely by recall. That is the right reference point for a system losing 0.99 → 0.85 in candidate-generation ceiling: the literature says the loss is real, is recall-shaped, and is roughly this size.
- A sequence-only model with no coordinates (pLM-NN, AUC 0.86–0.88) beats both a cryptic-specialist GNN (0.76) and a general pocket finder on apo input (0.81) on the same test structures. On apo input, structure is at best neutral and at worst actively misleading, because the apo geometry encodes the *closed* state.
- At pocket level with top-n scoring, 2026 state of the art on apo is ~0.85 top-1 and ~0.95 top-5. If a candidate generator's apo ceiling is 0.85, that ceiling is roughly at the level of a *published 2026 top-1 detector* — i.e. the headroom is in recall of the candidate set (top-5-style coverage), not in the ranker.

### Gaps
- No published method reports pocket-level DCA/DCC top-n on CryptoBench apo except the 2026 world-model preprint, whose criterion I could not verify.
- PocketMiner's own paper does not report a pocket-level top-n hit rate, so it cannot be compared to candidate-generation ceilings directly.
- AE-PocketMiner's numbers (see below) are not verified at all.

---

## DEDICATED METHODS: what they predict, and how they differ from pocket finders

### Takeaway
Every dedicated cryptic-site method before 2026 predicts a **per-residue propensity**, not a pocket — which is why they do not drop into a candidate-generation pipeline without an extra clustering/scoring step. The 2024–2026 wave splits into sequence-only PLM models (CryptoBench pLM-NN, CryptoBank's fine-tuned PLM) and latent-readout structure models (AE-PocketMiner, the Eratos world-model detector).

### Cited Findings

**CryptoSite (2016)** — SVM with a quadratic kernel over sequence, structure and dynamics features, classifying each residue as participating in a cryptic site or not; runtime **1–2 days per structure**; known weakness: false positives — [vbaf156 review](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf156/8180504) `[V, review reporting]`. The pipeline requires simulation/dynamics input, hence the cost — [PMC12959236](https://pmc.ncbi.nlm.nih.gov/articles/PMC12959236/) `[V]`

**PocketMiner (Meller et al., Nat Commun 2023)** — all from [PMC9977097](https://pmc.ncbi.nlm.nih.gov/articles/PMC9977097/) `[V]`
- Output: **per-residue likelihood that the residue will participate in a pocket that opens**, i.e. an *opening-event propensity*, not a pocket object and not a ligandability score.
- Architecture: **GVP-GNN** (geometric vector perceptron graph net); node features = backbone dihedrals and unit vectors; edge features = RBF-encoded distances; message passing over the **30 nearest neighbours**.
- Training labels come from **MD, not from crystallography**: **2,400 independent MD simulations over 37 proteins (35 used for training)**, adaptively sampled with **FAST**, giving **941,650 residue-level training examples**; a residue is positive if **LIGSITE pocket volume increases >40 Å³ within a 40 ns window**.
- Training sources: prior SARS-CoV-2 proteome simulations, ebolavirus VP35 simulations, and 16 newly simulated proteins with known cryptic pockets.
- Key difference from an ordinary pocket finder: an ordinary finder scores *existing* concavity in the input coordinates; PocketMiner scores the *propensity of currently-flat geometry to become concave*. On a closed apo structure the two objectives disagree by construction.

**CryptoBench's own baseline (pLM-NN, 2024)** — [PMC11725321](https://pmc.ncbi.nlm.nih.gov/articles/PMC11725321/) `[V]`
- Input: **ESM2-3B embeddings only** — no coordinates at all. Only one PLM was tested; **no ProtT5/Ankh comparison and no sequence-vs-structure ablation** was run. `[V]`
- Output: per-residue binary cryptic-binding-residue label.
- Framework lives in a separate repo, `github.com/skrhakv/apolo`, branch `cryptobench-v2`; **no pre-trained weights are hosted in the CryptoBench repo** — [repo page](https://github.com/skrhakv/CryptoBench/) `[V]`

**CryptoBank's predictor (2025, Science Advances)** — [bioRxiv API record](https://www.biorxiv.org/content/10.1101/2025.04.23.650184v1) `[V]`
- A **fine-tuned protein language model predicting cryptic sites from sequence alone**.
- Reported: **PR-AUC 0.8 when the query shares >20% identity with a CryptoBank entry** — an explicitly identity-conditioned result, i.e. the authors concede the number degrades below 20%.
- Prospective validation: predicted a cryptic site in **human TPP1** (<20% identity to any CryptoBank entry) and confirmed its opening by MD.

**AE-PocketMiner (Zhang, Mishra, Kelly, Kumar, Bowman; bioRxiv 2026-05-23, DOI 10.64898/2026.05.21.726899, licence cc_no, not yet journal-published)** — [bioRxiv API record](https://www.biorxiv.org/content/10.64898/2026.05.21.726899) `[V]`
- Adds an **attention mechanism to the GVP-GNN backbone** and predicts, **simultaneously from a single input structure, (i) cryptic pocket locations and (ii) their allosteric coupling to the rest of the protein**.
- Claims to "outperform past methods for identifying cryptic pockets" and to recapitulate known allosteric interactions, with **experimental confirmation of newly predicted pockets and of mutations that allosterically control opening**. **No numbers in the abstract, and the README carries none either** — [repo page](https://github.com/bowman-lab/ae-pocketminer) `[V]`

**Structural world-model detector (Eratos, 2026)** — [bioRxiv API record](https://www.biorxiv.org/content/10.64898/2026.09.21.752781v1) `[V]`
- Reads cryptic pockets **directly from the latent representation of a structural world model applied to a single apo structure** — "without conformational sampling or external pocket finders" and "no protein language model".
- Architectural claim that matters for a candidate-generation pipeline: it **"finds pockets directly rather than by first predicting a bound complex"**, and it is scored at pocket level with top-n, not per-residue.
- Also notes it runs "from the apo coordinates alone, including the mmCIF-only entries all recent depositions carry", and that accuracy "keeps climbing as the training corpus grows".
- Stated intended use: "prospective cryptic-site nomination on the targets that sequence and static structure leave without a starting point".

**Also-rans reported by the 2025 review** — [vbaf156](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf156/8180504) `[V, review reporting]`
- **SSnet** — DNN over backbone torsion + curvature with grad-CAM; predicted cryptic sites in three CryptoSite benchmark proteins; review's criticism: "blindness to conformation".
- **TACTICS** — Random Forest over MD trajectory frames + ConCavity pockets + conservation; ~1 h per target on an 8-core desktop; found sites in SARS-CoV-2 nsp5, MTase, ArCP; review's criticism: "assumes all cryptic binding sites are closed in apo forms".

### Inferences
- Of the dedicated methods, **only the 2026 Eratos detector and (nominally) AE-PocketMiner output something pocket-shaped**. Everything else needs a residue→pocket aggregation step that the papers do not specify, and that step is where a candidate-generation pipeline would differ from theirs.
- PocketMiner's training labels being *MD-derived* rather than *crystallographic* is the likeliest explanation of its weak 0.76 AUC on CryptoBench: its positives are "opens in 40 ns", CryptoBench's are "a ligand binds here in some holo PDB entry". These are different label distributions, and the CryptoBench authors flag the training-set mismatch.

### Gaps
- No verified metrics for AE-PocketMiner.
- SSnet and TACTICS numbers are only qualitative in the review; I did not reach their primary papers.
- No 2024–2026 method found that outputs both a pocket *and* a calibrated crypticity flag for that pocket.

---

## MD AND ENSEMBLES: evidence, cost, and whether the benefit is obtainable without MD

### Takeaway
MD and enhanced sampling are the only approaches with *quantitative* success on opening thermodynamics, and MD is reported as "very close" to experiment for wild-type pockets — but conventional MD is explicitly insufficient for the slow classes, and the cost is orders of magnitude above any ML method. The cheap substitute that actually works is **training on MD and then inferring without it** (PocketMiner's whole design), which buys a 1,000× speedup at the price of 0.76–0.87 AUC rather than quantitative accuracy.

### Cited Findings
- Conventional MD alone is insufficient: pockets arising from **secondary-structure change** open on timescales "ranging from microseconds to minutes, often exceeding the practical limits of conventional MD simulations" — [Zhang & Bowman, PMC12959236](https://pmc.ncbi.nlm.nih.gov/articles/PMC12959236/) `[V]`
- **FAST** adaptive sampling iteratively biases simulation toward pocket-opening features; it is what generated PocketMiner's training data — [PMC12959236](https://pmc.ncbi.nlm.nih.gov/articles/PMC12959236/) `[V]`, [PMC9977097](https://pmc.ncbi.nlm.nih.gov/articles/PMC9977097/) `[V]`
- **SWISH / SWISH-X** (replica exchange with scaled protein–water interactions) "successfully induced known pockets in **TEM-1 β-lactamase, IL-2, and Polo-like kinase-1**"; cost quoted as **~1 µs of sampling with 6–8 replicas** — [PMC12959236](https://pmc.ncbi.nlm.nih.gov/articles/PMC12959236/) `[V]`; replica/time figures from [vbaf156](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf156/8180504) `[V, review reporting]`
- **Mixed-solvent / cosolvent MD** (benzene, acetic acid, isopropanol probes) detects and quantifies pockets via probe-occupancy mapping and "does not rely on a priori knowledge", but "requires high computing power" and is sensitive to probe choice and protein destabilisation; demonstrated on **ricin, ALBP, AR, BACE-1, Hsp90** — [vbaf156](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf156/8180504) `[V]`, [PMC12959236](https://pmc.ncbi.nlm.nih.gov/articles/PMC12959236/) `[V]`
- **Markov state models** discretise the landscape and enable virtual screening against open states; led to ligand discovery for **TEM-1 β-lactamase and the 5-HT3A receptor** — [PMC12959236](https://pmc.ncbi.nlm.nih.gov/articles/PMC12959236/) `[V]`. Also used to find a cryptic pocket in **MetAP-II** via adaptive-bandit MD + MSM — [PMC11223136](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11223136/) `[2H, title/search only]`
- **CV-dependent enhanced sampling** (metadynamics, umbrella sampling, aMD, ABMD) works but "requires prior pocket knowledge" — i.e. it cannot do detection, only characterisation. CV-independent methods (T-REMD, SWISH, OPES) are "highly efficient for systems with slow and rare events" and were applied to **IL2, PLK1, NPC2, LfrR, p38α, hPNMT, DYRK1A** — [vbaf156](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf156/8180504) `[V]`
- Full-stack case study: ebolavirus **VP35** allosteric cryptic pocket controlling RNA binding, found by combining FAST, Folding@home, exposon analysis and DiffNets — [PMC12959236](https://pmc.ncbi.nlm.nih.gov/articles/PMC12959236/) `[V]`
- **Cost contrast, verified:** CryptoSite ≈ **one day per input structure**; PocketMiner ≈ **seconds**, **>1,000× faster**, **<1 s per protein** — [PMC12959236](https://pmc.ncbi.nlm.nih.gov/articles/PMC12959236/) `[V]`, [PMC9977097](https://pmc.ncbi.nlm.nih.gov/articles/PMC9977097/) `[V]`
- **Does the MD benefit transfer without MD?** Yes, partially and measurably: PocketMiner's GVP-GNN reaches **ROC-AUC 0.83** at predicting 40 ns pocket-formation events *from the starting structure alone*, and **0.87** on experimental apo structures — [PMC9977097](https://pmc.ncbi.nlm.nih.gov/articles/PMC9977097/) `[V]`. The 2026 Eratos detector claims pocket-level top-1 0.848 on apo "without conformational sampling" — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.09.21.752781v1) `[V]`
- **But not quantitatively:** on the two systems with experimental open-state populations, "MD is very close for the two wild-type proteins but all the methods struggle for pockets with small probabilities of opening (<1%)" — [bioRxiv DOI 10.64898/2026.01.21.700870](https://www.biorxiv.org/content/10.64898/2026.01.21.700870v1) `[V]`

### Inferences
- The evidence supports a clean division of labour: **ML (no MD at inference) for detection/nomination; MD or enhanced sampling for characterisation, prioritisation and pose generation.** The 2026 JCTC benchmark is explicit that the AI methods are "fast and qualitative" (v1 title) while MD is the quantitative one.
- For a candidate-generation ceiling problem, MD is the wrong tool: it costs µs-scale sampling per target and still needs a pocket finder on the resulting frames. The transferable trick is MD-*derived labels* on static input — the thing PocketMiner showed works at 0.83–0.87 residue AUC.

### Gaps
- No paper found that quantifies **how much a candidate-generation recall improves** when you add an MD/ensemble frame set, i.e. a "ceiling with ensemble vs ceiling with single apo" number. This is the single most decision-relevant missing measurement for the stated objective.
- No wall-clock/GPU-hour figures verified for SWISH, cosolvent MD or MSM construction beyond "1 µs, 6–8 replicas".

---

## GENERATIVE AND PREDICTED STRUCTURES: what is measured, not claimed

### Takeaway
Measured, not claimed: AlphaFold2 with MSA subsampling samples the open state in **6 of 10** known cryptic-pocket cases; BioEmu recovers **86% of holo** but only **56% of apo** conformations on a 34-case set; and on absolute opening probabilities AlphaFlow/BioEmu fail. Generative ensembles help detection, do not solve it, and are systematically biased toward the bound state.

### Cited Findings
- **AlphaFold2 (Meller, Bhakat, Solieva, Bowman; JCTC 2023, DOI 10.1021/acs.jctc.2c01189):** ensembles generated for **10 known cryptic-pocket examples, 5 of them deposited after AF2's training cutoff**; "in **6 out of 10 cases AlphaFold samples the open state**". For **plasmepsin II** AF2 captured only *partial* opening; MD launched from the AF2 ensemble then sampled full opening, "even though an equivalent amount of simulations launched from a ligand-free experimental structure fails to do so", and the resulting MSM free-energy landscape agreed with well-tempered metadynamics — [bioRxiv API record, DOI 10.1101/2022.11.23.517577](https://www.biorxiv.org/content/10.1101/2022.11.23.517577v1) `[V]`. Authors' own conclusion: "many cryptic pockets may remain difficult to sample using AlphaFold alone." `[V]`
- Independent re-reading of the same 10-protein result: "some predicted conformations with **less than 1.2 Å RMSD from the open (holo) structure in six of the 10 cases**"; and critically, "**subsampling improved the results only for a single protein**, but very shallow MSA led to incorrect structures" — [AlphaFold-SFA, bioRxiv/PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0307226) `[2H, search summary]`
- **MSA subsampling settings actually used** in the AlphaFold-SFA work: `max_msa = 32:64` (32 cluster centres, 64 extra sequences); this produced ensemble diversity for plasmepsin II but "failed to sample the 'deep' cryptic pocket opening" — [AlphaFold-SFA](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0307226) `[2H]`
- **AlphaFold-SFA** then used slow feature analysis on the AF2 ensemble to define metadynamics CVs, reaching deep opening "within a few hundreds of nanoseconds", beyond the reach of µs unbiased MD — [AlphaFold-SFA](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0307226) `[2H]`
- **BioEmu:** trained on "over 200 ms of simulation data, static structures, and experimental protein stabilities"; on **34 cryptic-pocket cases it recovered 86% of holo structures but only 56% of apo** — [Zhang & Bowman, PMC12959236](https://pmc.ncbi.nlm.nih.gov/articles/PMC12959236/) `[V, review reporting]`; same figures restated with the explanation that this "highlight[s] the need for … better balancing apo and holo representations during training" — [search summary of BioEmu preprint](https://www.biorxiv.org/content/10.1101/2024.12.05.626885v1.full) `[2H]`
- **AlphaFlow, BioEmu, PocketMiner, CryptoBank vs MD on opening thermodynamics:** none predicts absolute open probability reliably; all degrade badly below 1% open population; directionality of mutational effects is predicted well — [bioRxiv DOI 10.64898/2026.01.21.700870, published JCTC 10.1021/acs.jctc.6c00135](https://www.biorxiv.org/content/10.64898/2026.01.21.700870v1) `[V]`
- **AlphaFold2 "remembers too much":** a PNAS paper argues AF2's ability to produce multiple ligand-binding-site conformations is partly memorisation — [PNAS 10.1073/pnas.2412719121](https://www.pnas.org/doi/10.1073/pnas.2412719121) `[2H, title/search only — not fetched]`
- **AlphaFold2's limitation as an ensemble generator:** predicted ensembles "lack information about the relative weights of different conformations"; AF2 "recapitulated cryptic pockets in only 6 out of 10 proteins"; AlphaFold3 supports joint protein–ligand prediction with "potential for improved identification of open pocket conformations" — the latter stated as *potential*, not measured — [PMC12959236](https://pmc.ncbi.nlm.nih.gov/articles/PMC12959236/) `[V]`
- **Co-folding is not a substitute for detection:** the 2026 Eratos detector, compared against the co-folding engine **OpenDDE** on 190 targets held out of both training sets, **recovered 119 sites OpenDDE missed against 2 the other way**, and at top-5 OpenDDE's recovered sites were a strict subset — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.09.21.752781v1) `[V]`

### Inferences
- The **86% holo / 56% apo** asymmetry in BioEmu is the generative-model analogue of the user's own 0.99 holo / 0.85 apo ceiling. It is strong evidence that the apo deficit is a *training-distribution* artefact (PDB holo bias) rather than an information-theoretic limit.
- Nothing in the verified record supports "generate an ensemble and the cryptic pocket appears". The best-measured outcome is 6/10 with hand-tuned MSA depth, and subsampling helped only 1 of 10 proteins. Using AF2 ensembles as *MD seeds* is the use with the cleanest positive evidence.
- Detect-directly beats predict-the-complex-then-find-the-pocket, by a large margin, in the one 2026 head-to-head available.

### Gaps
- No verified number for AlphaFold3 or Boltz-family models on cryptic pocket opening. The AF3 statement I found is explicitly aspirational.
- AlphaFlow's own cryptic-pocket numbers in the 2026 JCTC benchmark are not verified (full text not fetched).
- The PNAS "remembers too much" paper was not fetched; its claim is reported second-hand.

---

## SEQUENCE SIGNAL: do PLMs carry cryptic-site information structure does not?

### Takeaway
Yes, and this is the single most surprising verified result in the field: on CryptoBench's 222 apo test structures a **sequence-only ESM2-3B model (AUC 0.86–0.88)** beats a cryptic-specialist structural GNN (0.76) and a structural pocket finder on apo input (0.81). The evidence is *comparative*, not *controlled* — no paper runs the ablation that would isolate the sequence contribution.

### Cited Findings
- pLM-NN = **ESM2-3B embeddings only, no coordinates**; AUC 0.86 (CB-full), 0.88 (CB-PM and CB-P2RANK-apo); AUPRC 0.36–0.43 vs PocketMiner 0.19 and P2Rank-apo 0.21 — [PMC11725321](https://pmc.ncbi.nlm.nih.gov/articles/PMC11725321/) `[V]`
- **Only one PLM was tested; there was no ProtT5/Ankh comparison and no sequence-only vs structure-based ablation.** The paper offers **no mechanistic explanation** for why sequence wins, and attributes part of the gap to pLM-NN being trained on CryptoBench while PocketMiner was pre-trained elsewhere — [PMC11725321](https://pmc.ncbi.nlm.nih.gov/articles/PMC11725321/) `[V]`
- Corroborating, from a different group: **ProtT5-XL-U50 and ESM-1b embeddings "outperformed" structure-based CryptoSite** on test sets — [Zhang & Bowman, PMC12959236](https://pmc.ncbi.nlm.nih.gov/articles/PMC12959236/) `[V, review reporting]`
- **CryptoBank's fine-tuned PLM: PR-AUC 0.8 above 20% sequence identity to the database**, plus one prospective hit (human TPP1, <20% identity, MD-confirmed) — [bioRxiv API record](https://www.biorxiv.org/content/10.1101/2025.04.23.650184v1) `[V]`
- The 2025 review's caveat on sequence models: "generalizability to novel sequences remained limited" — [PMC12959236](https://pmc.ncbi.nlm.nih.gov/articles/PMC12959236/) `[V]`
- Counter-signal: the 2026 Eratos detector deliberately uses **no protein language model**, relying on structural world-model latents, and reaches pocket-level top-1 0.848 — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.09.21.752781v1) `[V]`

### Inferences
- The honest reading of the CryptoBench result is **confounded**: pLM-NN was trained in-domain on CryptoBench; PocketMiner and P2Rank were not. The comparison shows "a model trained on this dataset beats two models trained elsewhere", which is weaker than "sequence beats structure". The CryptoBench authors say as much.
- Nonetheless the direction is consistent across two independent groups (CryptoBench pLM-NN; CryptoBank's PLM; ProtT5/ESM-1b vs CryptoSite), and it is mechanistically plausible: the apo coordinate set is actively *wrong* about the pocket, whereas sequence is state-agnostic. **A sequence channel is the cheapest known hedge against apo-closed geometry.**
- CryptoBank's identity-conditioned PR-AUC 0.8 is really a nearest-neighbour statement. Below 20% identity the PLM signal is undemonstrated at scale (one anecdote).

### Gaps
- **No controlled evidence exists** of the form "same architecture, same split, sequence-only vs structure-only vs both." This is the experiment the field is missing, and it would be a cheap and publishable contribution.
- No verified result on whether PLM features *added to* a structural apo model raise pocket-level recall.

---

## WHAT MAKES A SITE CRYPTIC: determinants and taxonomy

### Takeaway
There is a working four-class mechanistic taxonomy (loop motion / secondary-structure motion / secondary-structure change / interdomain rearrangement) with named exemplars, and the classes differ by orders of magnitude in timescale — which is precisely why they need different methods. No paper found reports per-class accuracy, so the "different classes need different methods" claim is mechanistically argued but not quantitatively demonstrated.

### Cited Findings
- **Four-class taxonomy with exemplars** — [Zhang & Bowman, PMC12959236](https://pmc.ncbi.nlm.nih.gov/articles/PMC12959236/) `[V]`:
  1. **Loop motion** — dihydrofolate reductase
  2. **Secondary-structure motion** — lipoprotein LpqN
  3. **Secondary-structure change** — calcium-integrin-binding protein 1
  4. **Interdomain rearrangement** — nopaline-binding periplasmic protein
- Timescale separation: class 3 (secondary-structure change) spans **microseconds to minutes**, "often exceeding the practical limits of conventional MD" — [PMC12959236](https://pmc.ncbi.nlm.nih.gov/articles/PMC12959236/) `[V]`
- PocketMiner's dataset was curated to span "loop motions, secondary structure shifts, domain movements" and additionally splits sites by *direction*: **forward** (apo residues farther apart) vs **reverse** (apo residues closer together) — [PMC9977097](https://pmc.ncbi.nlm.nih.gov/articles/PMC9977097/) `[V]`
- Physicochemical determinants from CryptoSite: cryptic sites are **as evolutionarily conserved as ordinary pockets, but less hydrophobic and more flexible** — [search summary of Cimermancic 2016](http://cdn.fraserlab.com/publications/2016_cimermancic.pdf) `[2H]`
- Formation-mechanism dichotomy: **induced fit vs conformational selection**; location dichotomy: **near active site vs remote/allosteric** — [vbaf156](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf156/8180504) `[V]`
- The 2025 review explicitly states it does **not** present a taxonomy of cryptic-site classes — [vbaf156](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf156/8180504) `[V]`
- Allosteric coupling is treated as a *separate predictable property* of a cryptic site, not a class of it: AE-PocketMiner predicts pocket location and allosteric coupling jointly, because "it is difficult to predict the functional relevance of a cryptic site as this often requires insight into allostery" — [bioRxiv DOI 10.64898/2026.05.21.726899](https://www.biorxiv.org/content/10.64898/2026.05.21.726899) `[V]`
- Druggability criterion sometimes applied to candidate cryptic sites (Kazakov/Beglov): a site binding **16 or more probe clusters** — [vbaf156](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf156/8180504) `[V, review reporting]`
- Magnitude proxies, per dataset: mean pocket RMSD **2.65 Å (CryptoSite)**, **3.46 Å (PocketMiner)**, **2.89 Å (CryptoBench)**; CryptoBench's much tighter SD (±0.87 vs ±1.99/±2.36) means it under-samples the extreme interdomain class — [PMC11725321](https://pmc.ncbi.nlm.nih.gov/articles/PMC11725321/) `[V]`

### Inferences
- The method/class mapping that the evidence supports: **side-chain rotamer and loop classes are reachable by single-structure ML and by short MD**; **secondary-structure-change and interdomain classes are not reachable by conventional MD at all** and are where AF2/generative seeding plus enhanced sampling earns its cost. The 6/10 AF2 result and the plasmepsin II failure are consistent with the deep/slow classes being the residual.
- CryptoBench's ≥2 Å RMSD cut with a 2.89 ± 0.87 Å mean means it is dominated by moderate motion. A model tuned on CryptoBench should not be expected to generalise to hinge/domain cases, and benchmark success there may overstate real-world coverage.

### Gaps
- **No per-class accuracy breakdown found for any method.** Nobody reports "PocketMiner on loop-motion sites vs interdomain sites". This is a genuine hole and an easy win if CryptoBench structures are reclassified by motion type.
- No quantitative definition distinguishing the four classes (e.g. RMSD/DSSP-change thresholds) was found; the taxonomy is exemplar-based.

---

## APO-HOLO GENERALISATION: how much is lost, and does apo training close it?

### Takeaway
The best-controlled measurement is P2Rank on CryptoBench: **AUC 0.89 holo → 0.81 apo, AUPRC 0.34 → 0.21, with TPR falling 0.84 → 0.62 at constant FPR** — a recall collapse, not a ranking failure. There is no published experiment that directly trains the same architecture on apo vs holo and reports the gap closing; the strongest indirect evidence that it would is BioEmu's 86%/56% split, which its own authors attribute to apo under-representation in training.

### Cited Findings
- **Same method, same proteins, apo vs holo input (P2Rank, CryptoBench, per-residue):** AUC 0.89→0.81, AUPRC 0.34→0.21, TPR 0.84→0.62, FPR 0.15→0.14, ACC 0.85→0.85 — [PMC11725321](https://pmc.ncbi.nlm.nih.gov/articles/PMC11725321/) `[V]`
- CryptoBench's stated motivation is exactly this problem: current methods "rely on holo conformations for training and evaluation, overlooking the significance of the apo states", which "in the case of cryptic binding sites … yields unrealistic performance expectations" — [PMC11725321](https://pmc.ncbi.nlm.nih.gov/articles/PMC11725321/) `[V]`
- **General pocket finders, CHEN apo vs holo, DCA Dcut = 4 Å, 104+104 proteins, apo aligned to holo with ligands transferred:** Jiang et al. 34.5→28.4 (top-n), Kalasanty 35.5→33, DeepSurf ResNet-18 40.6→39.6, DeepSurf Bot-LDS 39.1→37.6; "all methods perform worse on the apo case, DeepSurf exhibits the smallest decrease (1–1.5%)" — DeepSurf paper, text extracted from the PDF in this session `[V]`
- **Generative model trained on a holo-heavy corpus:** BioEmu recovers **86% of holo but 56% of apo** over 34 cryptic cases, with the stated remedy being "better balancing apo and holo representations during training" — [PMC12959236](https://pmc.ncbi.nlm.nih.gov/articles/PMC12959236/) `[V, review reporting]`; [BioEmu preprint summary](https://www.biorxiv.org/content/10.1101/2024.12.05.626885v1.full) `[2H]`
- **Training on apo pairs (indirect evidence it works):** pLM-NN was trained on CryptoBench's 885 *apo* training structures and reaches AUC 0.88 / AUPRC 0.43 on CB-PM, vs PocketMiner's 0.76 / 0.19 — the CryptoBench authors explicitly attribute part of the gap to pLM-NN having been trained on CryptoBench while PocketMiner was pre-trained elsewhere — [PMC11725321](https://pmc.ncbi.nlm.nih.gov/articles/PMC11725321/) `[V]`
- The Eratos 2026 detector is trained and evaluated on apo chains and reports improving accuracy with corpus scale — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.09.21.752781v1) `[V]`
- Mechanism of the apo penalty, as stated in the literature: "the shape of cryptic binding sites can differ significantly between apo and holo forms, resulting in decreased performance when measured on the apo form"; P2Rank specifically "is significantly affected by the protrusion of surface regions" — [search summaries referencing CryptoBench and P2Rank](https://link.springer.com/article/10.1186/s13321-024-00923-z) `[2H]`

### Inferences
- The two independent apo/holo contrasts disagree in *magnitude*: on cryptic-enriched CryptoBench the penalty is large (−0.08 AUC, −22 pp TPR); on the general CHEN set it is small (−1 to −6 pp DCA). The reconciliation is that **the penalty scales with how cryptic the site is**, which is exactly the regime of interest. A system losing 0.14 of candidate-generation ceiling on apo is behaving like the cryptic-enriched case, not the CHEN case.
- Because the loss is recall at fixed FPR, the levers that should work are **recall-side**: more permissive candidate generation on apo (larger top-n, lower concavity threshold), a sequence channel that does not depend on apo geometry, and apo-side augmentation (train on apo structures with holo-derived labels — the CryptoBench label scheme).
- Evidence that apo-training closes the gap is **suggestive but confounded** everywhere I looked. Nobody has run the clean experiment. This is a defensible gap to claim in a paper.

### Gaps
- **No paper found that trains one architecture on holo-only vs apo-only vs apo+holo pairs and reports the three apo test numbers.** This is the central missing control for the stated objective.
- No published "ceiling" numbers (candidate-generation recall at top-k) on apo vs holo for any general pocket finder, so the user's 0.99→0.85 figure has no direct literature counterpart — only the analogues above.

---

## PRACTICAL: code, weights, licences

### Takeaway
Only **AE-PocketMiner ships checkpoints in-repo under MIT**. PocketMiner's original code is MIT-adjacent/CC-BY via the paper but its weights were distributed via a branch plus Zenodo; CryptoBench ships data and scripts (MIT) but **no weights**; CryptoSite and CryptoBank are server-only in practice.

### Cited Findings

| Method | Repository | Weights in repo? | Licence |
|---|---|---|---|
| **AE-PocketMiner** (2026) | https://github.com/bowman-lab/ae-pocketminer | **Yes** — `models/` contains an AE-PocketMiner checkpoint *and* a copy of the original PocketMiner checkpoint | **MIT** — [repo page](https://github.com/bowman-lab/ae-pocketminer) `[V]` |
| **PocketMiner** (2023) | https://github.com/Mickdub/gvp (branch `pocket_pred`); also a Zenodo archive referenced in the paper; 3D-CNN baseline at https://github.com/meghana-kshirsagar/3DCNN_protein_structures | Not confirmed in that repo; **the checkpoint is obtainable from the AE-PocketMiner repo's `models/` directory** | Paper is CC-BY 4.0 (Nat Commun); repo licence not confirmed — [PMC9977097](https://pmc.ncbi.nlm.nih.gov/articles/PMC9977097/) `[V]`, [ae-pocketminer](https://github.com/bowman-lab/ae-pocketminer) `[V]` |
| **PocketMiner web server** | https://pocketminer.azurewebsites.net/ — HTTP 200 on probe 2026-10-07 | n/a | n/a `[V, reachability only]` |
| **CryptoBench** (dataset + pLM-NN baseline) | https://github.com/skrhakv/CryptoBench/ (scripts, tutorial notebook); data at https://osf.io/pz4a9/ (HTTP 200); baseline framework at https://github.com/skrhakv/apolo branch `cryptobench-v2` | **No** — "no pre-trained weights are hosted here" | **MIT** (source) — [repo page](https://github.com/skrhakv/CryptoBench/) `[V]` |
| **CryptoSite** (2016) | ModBase: https://modbase.compbio.ucsf.edu/cryptosite/download (HTTP 200) and https://modbase.compbio.ucsf.edu/cryptosite/help; web server runtime 1–2 days | Not assessed | Not confirmed — `[UNVERIFIED]` |
| **CryptoBank** (2025) | "publicly accessible via a web server" per the abstract; **URL not confirmed** (my guessed host returned proxy 502) | Unknown | Preprint CC-BY-NC-ND; journal version Science Advances DOI 10.1126/sciadv.ady6364 — [bioRxiv API](https://www.biorxiv.org/content/10.1101/2025.04.23.650184v1) `[V]` |
| **Eratos structural world-model detector** (2026) | **No repository mentioned in the abstract** | Unknown | Preprint **CC-BY-NC-ND** (non-commercial, no-derivatives — note this restricts reuse) — [bioRxiv API](https://www.biorxiv.org/content/10.64898/2026.09.21.752781v1) `[V]` |
| **SSnet** | GitHub (Verma et al. 2021) — URL not given in the review | Unknown | Unknown `[2H]` |
| **TACTICS** | GitHub (Evans et al. 2021) — URL not given in the review | Unknown | Unknown `[2H]` |
| **DeepSurf** (general finder, apo-robust) | https://github.com/stemylonas/DeepSurf.git — stated in the paper | Not assessed | Not confirmed — DeepSurf PDF `[V for URL]` |

- The AE-PocketMiner checkpoint provenance is worth noting: the AE-PocketMiner model was "trained through sequential steps using **LIGSITE and fpocket labels**" — [repo page](https://github.com/bowman-lab/ae-pocketminer) `[V]`
- The AE-PocketMiner README carries **no quantitative metrics** and points to the paper (bioRxiv 10.64898/2026.05.21.726899) — [repo page](https://github.com/bowman-lab/ae-pocketminer) `[V]`

### Inferences
- **If you need a runnable cryptic-site baseline with weights today, it is AE-PocketMiner (MIT, checkpoints included, and it also ships the original PocketMiner checkpoint).** That single repo gives both generations of the Bowman-lab model.
- The CC-BY-NC-ND licence on the 2026 Eratos preprint, with no repository, means the current pocket-level SOTA on apo is **not reproducible** — its numbers can be cited but not matched or audited.

### Gaps
- Could not verify repository licences via the GitHub API (blocked for this session: "GitHub access to this repository is not enabled"); the licence facts above come from WebFetch of the rendered repo pages, which is reliable for the README/licence badge but not for file-level inspection.
- CryptoSite's and DeepSurf's repo licences; the Eratos code availability; CryptoBank's web-server URL.

---

## Cross-cutting notes for the report writer

- **Detection vs crypticity-prediction, explicitly.** Every pre-2026 headline number (CryptoSite 0.83/0.85, PocketMiner 0.87, pLM-NN 0.86–0.88, P2Rank 0.81 apo) is **per-residue crypticity propensity**, task (B). The only verified **task (A)** pocket-level apo numbers in the literature as of 2026 are the Eratos detector's top-1 0.848 / top-5 0.952 on CryptoBench and the DeepSurf CHEN-apo DCA figures (37.6–39.6% top-n). The 2025 Bioinformatics Advances review states that it does not treat these as separate tasks, and the Zhang & Bowman review "does not explicitly state this as a separate methodological distinction" — the conflation is real and documented. [PMC12959236](https://pmc.ncbi.nlm.nih.gov/articles/PMC12959236/) `[V]`, [vbaf156](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf156/8180504) `[V]`
- **Datasets that could not be reached or confirmed:** ASBench download (404 at the URL I tried); CryptoBank web server (no confirmed URL); full texts of the two 2026 bioRxiv preprints (HTTP 429 / Cloudflare 1015 — abstracts obtained via the bioRxiv API instead); the DeepSurf OUP journal page (returned an unrelated article). Recorded rather than assumed.
- **Chronology of the field, verified dates:** CryptoSite JMB Feb 2016 → PocketMiner Nat Commun Feb 2023 → AlphaFold-for-cryptic-pockets JCTC 2023 → CryptoBench Bioinformatics btae745 (Dec 2024) → CryptoBank bioRxiv Apr 2025, Science Advances 2025 → Zhang & Bowman review (Curr Opin Struct Biol, PMC12959236, 2025) → Bowman-lab AI-vs-MD thermodynamics benchmark bioRxiv Jan 2026, JCTC 10.1021/acs.jctc.6c00135 → AE-PocketMiner bioRxiv May 2026 → Eratos structural world-model detector bioRxiv Sep 2026.
