# Pocket ranking mechanisms and what specifically improves top-1 accuracy (state as of 2026)

**Scope note on verification.** Every bullet is tagged `[PRIMARY]` (read from the paper / its supplementary PDF text in this session), `[PRIMARY-SUMMARISED]` (fetched the primary page but the numbers came through a summarising pass over the full text, so transcription risk is non-zero), or `[SECOND-HAND]` (search-engine snippet or aggregator; not confirmed against the paper). Where a ranking choice is a hand-set hyperparameter rather than a learned component, it is said so explicitly.

**Framing caveat that matters for the whole document.** Most of this literature does **not** report top-1. P2Rank, PRANK, DeepPocket, DeepSurf, EquiPocket and VN-EGNN all report **Top-n** (n = number of relevant ligands in the structure, so n ≥ 1 and often > 1) and **Top-(n+2)**; Utgés & Barton 2024 report **Top-(N+2)** and argue for it as the standard. Only YuelPocket (2026) was found reporting Top-1 explicitly. So "what improves top-1" has to be partly inferred from Top-n versus Top-(n+2) gaps, which are the field's proxy for ranking headroom.

---

## Q1. How does P2Rank rank pockets, what drives it, and why can a 2018 random forest beat deep models at rank 1?

### Takeaway
P2Rank's ranking is a **sum of squared per-point ligandability probabilities over a single-linkage cluster** — a cumulative, size-coupled score, not a learned pocket-level head. Its discriminative power is overwhelmingly carried by one geometric feature (**protrusion**, RF importance 0.0845 versus 0.0139 for the runner-up), and a protrusion-only model already reaches ~90 % of the full model's Top-n. The ranking is therefore a very robust, low-variance buriedness-plus-chemistry prior that does not overfit to training-set pocket identity — which is the most plausible mechanical reason it survives at rank 1 where deep models with learned pocket heads do not.

### Cited Findings
- Pocket score formula, stated in the PRANK paper as `PScore = Σᵢ (P₁(Vᵢ))²` — sum over inner pocket points of the **squared** positive-class probability. Authors' stated rationale: "Squaring the probabilities puts more emphasis on the points with probability closer to 1." They explicitly tested and rejected normalised (mean) scoring: cumulative scoring "consistently outperformed normalized approaches", because an oversized predicted pocket that contains the true site still retains an adequate score. `[PRIMARY-SUMMARISED]` — [PRANK, Krivák & Hoksza 2015, J Cheminform 7:12 (PMC4414931)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4414931/)
- P2Rank inherits the same aggregation: pockets are ranked by "cumulative ligandability score of their points (sum of squared ligandability scores of all points in the cluster)". `[PRIMARY-SUMMARISED]` — [P2Rank 2018, J Cheminform 10:39 (PMC6091426)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6091426/)
- Clustering is **single-linkage with a 3 Å cutoff** over SAS points whose RF ligandability exceeds a threshold. These are **hand-set hyperparameters**, not learned: the paper states parameters were "optimized with respect to the performance on JOINED dataset", and the supplementary repeats that "parameter optimization and final model selection was done with respect to the results on JOINED dataset". No ablation over the cutoff or threshold is published. `[PRIMARY-SUMMARISED]` for the 3 Å / single-linkage; `[PRIMARY]` for the JOINED-tuning statement — [P2Rank PMC6091426](https://pmc.ncbi.nlm.nih.gov/articles/PMC6091426/); [P2Rank supplementary PDF](https://static-content.springer.com/esm/art%3A10.1186%2Fs13321-018-0285-8/MediaObjects/13321_2018_285_MOESM1_ESM.pdf)
- Feature set is **35 features** on SAS points: 17 residue-level binary/scalar physicochemical attributes projected from exposed atoms (hydrophobic, hydrophilic, Kyte–Doolittle hydropathy, aliphatic, aromatic, sulfur, hydroxyl, basic, acidic, amide, pos/negCharge, H-bond donor/acceptor/both, polar, ionizable), 6 VolSite atom-type features (vsAromatic, vsCation, vsAnion, vsHydrophobic, vsAcceptor, vsDonor), atomicHydrophobicity (Kapcha–Rossky), two Khazanov–Carlson atom-type ligand-binding propensities (apRawValids, apRawInvalids), **bfactor**, and 9 point-level geometric counts (atoms within 6 Å, atomDensity, atomC/O/N, hDonorAtoms, hAcceptorAtoms, **protrusion** = number of *all* protein atoms within 10 Å of the point, after Pintar's Cx). `[PRIMARY]` — [P2Rank supplementary, Table 4](https://static-content.springer.com/esm/art%3A10.1186%2Fs13321-018-0285-8/MediaObjects/13321_2018_285_MOESM1_ESM.pdf)
- **Feature importances (RF, CHEN11, avg of 10 runs)** — protrusion 0.084528; bfactor 0.013888; apRawInvalids 0.011785; vsAromatic 0.010165; apRawValids 0.009403; atomO 0.009275; hydrophobic 0.008630; hydrophilic 0.007643; vsAcceptor 0.006244; vsHydrophobic 0.005273; atoms 0.005188; … down to amide 0.000831. Protrusion is **6.1× the second-ranked feature** and ~2.5× the sum of the next three. `[PRIMARY]` — [P2Rank supplementary, Table 5](https://static-content.springer.com/esm/art%3A10.1186%2Fs13321-018-0285-8/MediaObjects/13321_2018_285_MOESM1_ESM.pdf)
- **P2Rank's own feature-set ablation (SI Table 3, DCA 4 Å, identification success rate %, avg of 10 train/eval runs):**

  | feature set | JOINED Top-n | JOINED Top-(n+2) | COACH420 Top-n | COACH420 Top-(n+2) | HOLO4K Top-n | HOLO4K Top-(n+2) |
  |---|---|---|---|---|---|---|
  | [protrusion] only | 62.8 | 73.4 | 64.2 | 73.0 | 59.3 | 67.7 |
  | [full − protrusion] | 64.3 | 75.9 | 60.5 | 71.8 | 68.2 | 75.9 |
  | [full − propensities] | 73.9 | 80.5 | 71.6 | 77.9 | 69.1 | 74.7 |
  | [full] | 74.0 | 80.2 | 71.4 | 78.1 | 70.1 | 75.4 |
  | P2Rank default shipped model | 74.4 | 80.2 | 72.0 | 78.3 | 68.6 | 74.0 |

  `[PRIMARY]` — [P2Rank supplementary, Table 3](https://static-content.springer.com/esm/art%3A10.1186%2Fs13321-018-0285-8/MediaObjects/13321_2018_285_MOESM1_ESM.pdf)
- The paper's own framing of this: protrusion is "the single most important feature … a proxy for point's buriedness", and "even a simplified version of the algorithm, based only on this feature alone, seem to outperform many of the other methods". `[SECOND-HAND]` (phrasing surfaced via search over the P2Rank/LIGYSIS texts; the numeric backing is `[PRIMARY]` above) — [search over jcheminf/PMC texts](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0285-8)
- The atom-type propensity features — the only features with any PDB-wide statistical training behind them — contribute **essentially nothing**: removing them changes COACH420 Top-n by +0.2 and *improves* HOLO4K Top-n by −1.0 → 69.1 vs 70.1 (within noise), and the authors say "contribution of those features is minimal at best". They ran this ablation specifically to rule out leakage from the propensity tables. `[PRIMARY]` — [P2Rank supplementary §2.3](https://static-content.springer.com/esm/art%3A10.1186%2Fs13321-018-0285-8/MediaObjects/13321_2018_285_MOESM1_ESM.pdf)
- P2Rank Top-n vs competing methods (DCA 4 Å): COACH420 72.0 (P2Rank) vs 56.4 (fpocket), 53.0 (SiteHound, on its subset), 63.4 (MetaPocket 2.0), 56.4 (DeepSite); HOLO4K 68.6 vs 52.4 / 50.1 / 57.9 / 45.6. `[PRIMARY]` — [P2Rank supplementary Tables 1–2](https://static-content.springer.com/esm/art%3A10.1186%2Fs13321-018-0285-8/MediaObjects/13321_2018_285_MOESM1_ESM.pdf)
- PRANK's own explicit statement of the ranking-is-the-bottleneck thesis: native methods use "simple" scorers (single descriptors or linear combinations); pocket **detection** reaches near-100 % total coverage, so **ranking quality dominates final performance**. It also argues that classifying *individual points* rather than whole pockets is what gives generalisation despite training-set bias toward known ligands. `[PRIMARY-SUMMARISED]` — [PRANK PMC4414931](https://pmc.ncbi.nlm.nih.gov/articles/PMC4414931/)
- P2Rank's 2018 paper does **not** itself contain a ranking-vs-detection bottleneck analysis; it emphasises speed as the practical advantage. `[PRIMARY-SUMMARISED]` — [P2Rank PMC6091426](https://pmc.ncbi.nlm.nih.gov/articles/PMC6091426/)

### Inferences
- The squaring exponent is a **tunable knob nobody has swept publicly**. `Σ p²` sits between `Σ p` (pure size×quality, favours big pockets) and `max p` (size-blind). For a system whose wrong first pick is a distant large cavity, the exponent is a one-line, zero-training lever: raising it toward 3–4, or interpolating `Σ pᵏ / |cluster|^α`, directly changes the size/quality trade-off that decides rank 1. The PRANK authors chose cumulative over normalised for a *detection-tolerance* reason (oversized pockets still score), which is exactly the failure mode that hurts rank-1 discrimination.
- The ablation is the strongest available answer to "why does a 2018 RF win at rank 1": ~90 % of P2Rank's COACH420 Top-n (64.2 of 71.4) comes from **one hand-crafted buriedness count with no learned parameters at all**. A ranker whose dominant signal is a parameter-free geometric statistic cannot overfit pocket identity, cannot be distribution-shifted by a new fold, and degrades gracefully. Deep pocket-level heads have far more capacity to key on training-set-specific cues, and (see Q2) most of them actually rank by something *cruder* than P2Rank does.
- Protrusion's contribution is **dataset-dependent in sign**: removing it costs 10.9 Top-n points on COACH420 (71.4 → 60.5) but only 1.9 on HOLO4K (70.1 → 68.2), and protrusion-only is *worse* than full-minus-protrusion on HOLO4K. So buriedness is doing most of the discrimination on single-chain/small-protein benchmarks and much less on HOLO4K's larger multi-chain structures. If the system in question is being judged on COACH420, buriedness calibration is disproportionately load-bearing.
- `bfactor` ranking second is notable and under-discussed: it is a crystallographic artefact feature (and is meaningless/constant on predicted structures). Any comparison against P2Rank on experimental structures gives P2Rank a signal that a geometry+chemistry model does not have.

### Gaps
- No published sweep of the exponent in `Σ pᵏ`, nor of size-normalisation variants, for either PRANK or P2Rank.
- No published ablation of the 3 Å single-linkage cutoff or the SAS-point score threshold, so their contribution to rank-1 is unquantified.
- P2Rank feature importances are reported on CHEN11 only (the training set, 251 proteins); no importance breakdown on the test sets, and RF Gini importance is known to be biased toward high-cardinality continuous features (protrusion is continuous, most others are binary) — the 6× gap is therefore likely **inflated**, though the protrusion-only ablation independently confirms the feature is genuinely dominant.

---

## Q2. Scoring and ranking schemes in the deep methods, and the evidence for each

### Takeaway
Across the deep methods, the pocket-level ranking function is almost always the **weakest, least-learned part of the pipeline** — a mean of residue probabilities (GrASP), a count of cloud points or an averaged MLP confidence over mean-shift clusters (VN-EGNN), or a connected-component/density blob with no real ordering (Kalasanty, PUResNet). The single deep method that genuinely *learns to rank* — DeepPocket, a 3D CNN classifier re-scoring fpocket candidates — is also the one that beats P2Rank, and it loses to P2Rank on exactly one number: COACH420 Top-n.

### Cited Findings
- **DeepPocket** = fpocket candidate barycentres → constant-sized 23.5 Å / 0.5 Å-resolution voxel grids at each barycentre → 3D CNN **binary classifier** score → re-rank → U-Net segmentation of top-ranked pockets only. Training labels: any fpocket barycentre within 4 Å of any ligand atom is positive; 518,460 candidates, 22,030 positive (4.25 %), class imbalance handled by oversampling to 50/50 batches. The scoring and segmentation stages are **separate models**; segmentation is applied *after* ranking and does not feed back into it. `[PRIMARY]` — [DeepPocket, Aggarwal et al., J Chem Inf Model 2022 (PDF text)](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2022_01/Deeppocket.pdf)
- **DeepPocket DCA results (Table 1, success rate %):**

  | method | COACH420 Top-n | COACH420 Top-(n+2) | HOLO4K Top-n | HOLO4K Top-(n+2) | SC6K Top-n | SC6K Top-(n+2) |
  |---|---|---|---|---|---|---|
  | Fpocket | 35.09 | 51.25 | 36.34 | 51.53 | 23.99 | 37.23 |
  | DeepSite | 53.07 | 53.07 | 51.65 | 51.67 | 52.94 | 65.41 |
  | Kalasanty | 63.51 | 65.18 | 61.21 | 62.63 | 61.75 | 61.75 |
  | P2Rank | 68.24 | 75.48 | 70.60 | 80.05 | 62.90 | 75.74 |
  | DeepPocket | 67.96 | 79.94 | 73.36 | 82.97 | 64.58 | 83.01 |

  `[PRIMARY]` — [DeepPocket PDF, Table 1](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2022_01/Deeppocket.pdf)
- The authors' own reading of their single loss: "DeepPocket outperforms all other state-of-the-art methods in all the data sets **except in the Top-n score for COACH420, where P2Rank detects only one extra pocket therefore beating DeepPocket by just 0.28 %**." They attribute it to the **candidate ceiling**: on COACH420, fpocket places a centre within 4 Å of a ligand heavy atom for only **80.78 %** of binding sites, versus 87.62 % on HOLO4K and 91.64 % on SC6K; "DeepPocket has successfully ranked 85 % of the Fpocket detected binding sites in the Top-n ranks". `[PRIMARY]` — [DeepPocket PDF](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2022_01/Deeppocket.pdf)
- DeepPocket's own cross-validation ranking curve on scPDB v.2017: Top-n ≈ 70.8 % rising to **87.77 % at Top-(n+2)** — "a big jump of 17 % in success rate from Top-n to Top-(n+2)". `[PRIMARY]` (Top-(n+2) value verbatim; the Top-n value is implied by "a jump of 17 %") — [DeepPocket PDF](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2022_01/Deeppocket.pdf)
- **GrASP**: per-atom/residue binding probability from an attention GNN → **mean-shift clustering** over surface atoms → residues assigned to clusters by **majority vote** of their surface atoms → each pocket scored by the **mean predicted probability of its residues** → rank by that mean. `[SECOND-HAND]` (mechanism description assembled from search over the GrASP paper; not read in full this session) — [GrASP, Smith et al., J Chem Inf Model 2023 (PMC10402091)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10402091/)
- **VN-EGNN**: ~12 **virtual nodes** per protein, each with a learned **self-confidence** `c_k = ψ(v_k)` (an MLP on the virtual-node embedding); at inference, spatially proximate virtual nodes are merged by **mean shift**, and the cluster's representative confidence and position are the **averages** of its members'. Evaluation takes "the top M predicted binding sites for each protein, selected based on their highest self-confidence scores". Reported: COACH420 DCC 0.605 ± 0.009 / DCA 0.750 ± 0.008; HOLO4K DCC 0.532 ± 0.021 / DCA 0.659 ± 0.026; PDBbind2020 DCC 0.669 ± 0.015 / DCA 0.820 ± 0.010. **No ablation of the confidence/ranking module is reported, and top-1 is not separated from top-N.** `[PRIMARY-SUMMARISED]` — [VN-EGNN (PMC12837241)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12837241); preprint [arXiv:2404.07194](https://arxiv.org/pdf/2404.07194)
- A conflicting second-hand description says VN-EGNN pockets are "scored and ranked based on the **number of cloud points**" rather than by the learned confidence. These two descriptions are incompatible; the primary fetch supports the learned-confidence-averaged-over-cluster account, so treat the point-count account as unreliable. `[SECOND-HAND, flagged as conflicting]` — [search snippet](https://portal.valencelabs.com/blogs/post/vn-egnn-protein-binding-site-identification-using-graph-neural-networks-JKfoQld2n5BwVQZ) vs. [PMC12837241](https://pmc.ncbi.nlm.nih.gov/articles/PMC12837241)
- **DeepSurf**: SAS computed, small grids placed along surface **normal vectors**, 3D CNN features per grid cell. **EquiPocket**: among the first to use message-passing GNNs for the task. Reported DCC: DeepSurf 0.386 (COACH420) / 0.289 (HOLO4K) / 0.510 (PDBbind2020); EquiPocket 0.423 / 0.337 / 0.545 — both well below VN-EGNN. `[SECOND-HAND]` — [search over VN-EGNN comparison tables](https://arxiv.org/pdf/2404.07194)
- **Kalasanty** is a one-step U-Net segmentation over the whole protein with no separate ranking stage; its Top-n and Top-(n+2) are nearly identical (COACH420 63.51 → 65.18; SC6K 61.75 → 61.75), i.e. **it has almost no ranked list at all**. `[PRIMARY]` (numbers) — [DeepPocket PDF, Table 1](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2022_01/Deeppocket.pdf)
- Utgés & Barton 2024 corroborate the same pattern on an independent 2,775-chain / 8,244-site benchmark (LIGYSIS): **PUResNet** Top-N 40.6 % → all-predictions 41.1 % (0.5-point gap, no useful ranking headroom), **P2Rank** 46.7 % → 57.0 % (10.3), **fpocket** 38.8 % → **91.3 %** (52.5-point gap — essentially perfect detection, catastrophic ranking). `[PRIMARY-SUMMARISED]` — [Utgés & Barton, J Cheminform 2024 (PMC11552181)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/)

### Inferences
- There is a clean ordering by **how much learning is in the ranker**: no ranker (Kalasanty/PUResNet, flat curves) < mean-of-probabilities over a clustering (GrASP, VN-EGNN) < sum-of-squared point scores (P2Rank) < a dedicated discriminative pocket-level classifier trained on candidate-vs-ligand labels (DeepPocket, PRANK). The two methods at the top of that list are the two that lead the benchmarks. That is the most defensible generalisation in this whole area.
- **Mean-of-probabilities ranking is the specific weakness to attack.** A mean is size-blind and is dragged down by the cluster's boundary points; it also cannot express "this cavity is large *and* uniformly ligandable", which is what separates a real site from a buried void. P2Rank's `Σ p²` beats a mean on exactly this axis. GrASP and VN-EGNN both use means (VN-EGNN averages confidences within a mean-shift cluster) — which is consistent with their being strong at *precision over the whole list* (see Q4) while not dominating Top-n.
- Mean-shift bandwidth, the single-linkage cutoff, the mean-vs-sum choice and the majority-vote residue assignment are all **hyperparameters, not learned components**, in GrASP, VN-EGNN and P2Rank respectively. None of these papers publishes a sensitivity analysis for them. For a system at a 0.99 candidate ceiling, this is the cheapest unexplored search space in the field.
- DeepPocket's Top-n → Top-(n+2) jump of 17 points on its *own* cross-validation set, with a candidate ceiling of ~91 % on SC6K, says that even the best learned re-ranker leaves roughly half its headroom on the table at the top of the list. The described failure profile (correct pocket at rank 2–3 about 70 % of the time) is the normal state of the art, not an anomaly.

### Gaps
- Could not retrieve EquiPocket's or DeepSurf's clustering/scoring sections from primary sources in this session; their ranking mechanisms here are second-hand only.
- No paper found that ablates its own clustering/aggregation choice (mean vs sum vs max, bandwidth, cutoff) against top-ranked accuracy. This appears to be a genuine hole in the literature rather than a search failure.

---

## Q3. Does sequence conservation help ranking, and does it hold on train-dissimilar structures?

### Takeaway
Conservation is an **optional, measurable but modest** ranking gain in P2Rank (~+2 recall points on LIGYSIS, +346 true positives at a fixed 100-false-positive budget), and critically it is a feature with **no training-set dependence at all** — it is computed per-target from an MSA — so it is one of the few ranking signals that cannot degrade on train-dissimilar structures the way a learned head does.

### Cited Findings
- P2Rank's 2018 paper did **not** use conservation; it is listed as future work: "Sequence conservation and energetic calculations could be used to further enrich the feature vector". The `conservation` feature does not appear in SI Table 4's 35-feature list. `[PRIMARY-SUMMARISED]` for the quote; `[PRIMARY]` for its absence from Table 4 — [P2Rank PMC6091426](https://pmc.ncbi.nlm.nih.gov/articles/PMC6091426/); [P2Rank supplementary](https://static-content.springer.com/esm/art%3A10.1186%2Fs13321-018-0285-8/MediaObjects/13321_2018_285_MOESM1_ESM.pdf)
- **P2Rank_CONS** (the shipped `conservation`-enabled model) on LIGYSIS: Top-(N+2) recall **53.9 %** vs **51.9 %** for default P2Rank (+2.0 points); and **+346 true positives relative to default P2Rank** at the 100-false-positive operating point. `[PRIMARY-SUMMARISED]` — [Utgés & Barton 2024 (PMC11552181)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/)
- At the top-1000-predictions precision metric, P2Rank_CONS sits in the 900–1,000-TP band alongside GrASP and IF-SitePred, above default P2Rank. `[PRIMARY-SUMMARISED]` — [Utgés & Barton 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/)
- PRANK (2015) **deliberately excluded** conservation, using "only local geometric and physico-chemical features", on the stated rationale that this distinguishes it from ConCavity and lets it find novel, non-conserved but physicochemically ligandable sites. `[PRIMARY-SUMMARISED]` — [PRANK PMC4414931](https://pmc.ncbi.nlm.nih.gov/articles/PMC4414931/)
- LIGYSIS was explicitly constructed with low overlap to the methods' training sets (0.5–9.7 % overlap with sc-PDB, Binding MOAD and CHEN11), so the P2Rank_CONS gain above is measured on largely train-dissimilar structures. `[PRIMARY-SUMMARISED]` — [Utgés & Barton 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/)

### Inferences
- The +2.0-point Top-(N+2) gain is real but small relative to what re-ranking buys (see Q7, +7 to +14 points). Conservation is a *tie-breaker*, not a fix for "which cavity is ligandable".
- Its value is nevertheless structurally different from every learned feature: because it is computed from the target's own MSA at inference time, its contribution is not a function of train/test similarity. For a system whose stated problem is a distant-cavity first pick, conservation is the natural asymmetric signal — crystallographic artefact cavities and surface voids are not conserved, functional sites are.
- PRANK's and P2Rank's opposite choices here (exclude for novelty vs include for accuracy) are a genuine design trade-off, not a settled question; conservation will systematically penalise cryptic and allosteric sites with weak conservation signal.

### Gaps
- No measurement found that isolates conservation's contribution **at top-1 specifically**, nor one that stratifies it by train/test sequence identity bin. The LIGYSIS result is a single aggregate.
- No measurement found of conservation's contribution on predicted (AlphaFold) structures, where the MSA is available but the geometry is not experimental.

---

## Q4. Druggability / ligandability as a separate ranking problem

### Takeaway
Dedicated druggability scorers (fpocket score, DoGSiteScorer, SiteMap Dscore, PockDrug) are built and validated as **binary druggable/non-druggable classifiers on curated pocket sets**, not as rankers over all candidate cavities of a protein, and no study was found that substitutes a druggability score for a geometric score in a pocket-ranking pipeline and reports the effect on top-1. The strong indirect evidence is that P2Rank's whole design *is* a ligandability ranker — it calls its per-point score "ligandability" — and it beats pure geometry by 15+ Top-n points, which is the field's real answer to "is ligandability better than geometry".

### Cited Findings
- **SiteMap Dscore** is a **linear combination of exactly three descriptors**: pocket size, enclosure, and a hydrophilicity penalty; Dscore > 0.83 indicates a druggable site. `[SECOND-HAND]` — [search over SiteMap/druggability literature](https://www.nature.com/articles/s41598-022-12105-8)
- **fpocket** ranks by a score derived from α-sphere descriptors of the Delaunay/α-shape complex; **DoGSiteScorer** uses a machine-learned druggability score over grid-based pocket descriptors. `[SECOND-HAND]` — [search over druggability literature](https://academic.oup.com/nar/article/43/W1/W436/2467926)
- **PockDrug** vs the other two on a druggability classification task: PockDrug 93.5 % accuracy / MCC 0.515, DoGSiteScorer 79.1 % / MCC 0.328; PockDrug reported to exceed fpocket score and DoGSiteScorer "by at least 10 % points in accuracy and 0.2 in MCC". `[SECOND-HAND]` — numbers from a ResearchGate reproduction of the PockDrug table; primary is [PockDrug-Server, NAR 2015](https://academic.oup.com/nar/article/43/W1/W436/2467926)
- PockDrug's stated design purpose is to be robust to **pocket estimation uncertainty** (i.e. the same site delineated differently by different detectors gives a similar druggability score). `[SECOND-HAND]` — [PockDrug](https://academic.oup.com/nar/article/43/W1/W436/2467926)
- A one-class-learning probabilistic druggability model exists (Fpharm. 2022), framing druggability as **positive-unlabelled / one-class** rather than binary — relevant because non-druggable pockets are not reliably labelled. `[SECOND-HAND]` — [Probabilistic Pocket Druggability Prediction via One-Class Learning (PMC9278401)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9278401/)
- The strongest in-field evidence for "ligandability beats geometry as a ranking signal": fpocket's α-sphere geometric score gives COACH420 Top-n 35.09–56.4 % (depending on version/evaluation), while re-scoring *the same candidates* with a learned ligandability model gives 63.6 % (PRANK) or 67.96 % (DeepPocket). The candidate set is identical; only the ranking signal changed. `[PRIMARY]` — [P2Rank supplementary Table 1](https://static-content.springer.com/esm/art%3A10.1186%2Fs13321-018-0285-8/MediaObjects/13321_2018_285_MOESM1_ESM.pdf) and [DeepPocket Table 1](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2022_01/Deeppocket.pdf)
- VolSite pharmacophore-annotated cavity descriptors (Desaphy et al. 2012), explicitly a **druggability-prediction** descriptor set, are embedded in P2Rank as the `vs*` features — and vsAromatic is P2Rank's 4th-most-important feature (0.0102), with vsAcceptor (0.0062) and vsHydrophobic (0.0053) also in the top ten. `[PRIMARY]` — [P2Rank supplementary Tables 4–5](https://static-content.springer.com/esm/art%3A10.1186%2Fs13321-018-0285-8/MediaObjects/13321_2018_285_MOESM1_ESM.pdf)

### Inferences
- The right reading is not "druggability score instead of geometry" but "**train the score against the ligand-occupancy label, on the candidate distribution you will actually rank**". DeepPocket's label definition — fpocket barycentre within 4 Å of any ligand atom, 4.25 % positive rate over 518k candidates — is the operative detail: it makes the model a discriminator over *realistic negatives*, including the large distant cavities that cause rank-1 errors. A druggability model trained on curated druggable-vs-undruggable pocket pairs has never seen that negative distribution.
- Published druggability scorers are all **small linear or shallow models over a handful of descriptors** (Dscore: three descriptors). That they perform respectably supports the Q1 inference that rank-1 discrimination is a low-dimensional, low-variance problem, not a capacity problem.
- The one-class / PU framing is the honest one for this problem and is directly applicable: a cavity with no ligand in *this* crystal is unlabelled, not negative. Treating it as a hard negative is a known mis-specification that no top-performing binding-site method addresses.

### Gaps
- **No study found that uses a druggability score as the ranking function over a protein's full candidate cavity list and reports top-1.** This is the clearest literature hole relevant to the stated objective.
- SiteMap, DoGSiteScorer and PockDrug numbers here are second-hand; none of the primary papers was read in this session.

---

## Q5. Probability calibration and its effect on pocket ranking

### Takeaway
**No paper was found that calibrates pocket-level scores and reports the effect on top-1 or on ranked success rate.** This appears to be a genuine absence in the binding-site literature, not a search failure; calibration work in the adjacent protein–ligand space is all about *affinity* regression and *pose* confidence, not pocket ordering.

### Cited Findings
- Uncertainty-quantification work on protein–ligand **binding affinity** evaluates with Miscalibration Area, RMSCE and MACE, and finds Bayes-by-Backprop most reliable among the compared UQ methods. This is affinity, not pocket ranking. `[SECOND-HAND]` — [Uncertainty quantification for protein–ligand binding affinity, Sci Rep 2025](https://www.nature.com/articles/s41598-025-27167-7)
- Docking-score literature treats scores as "heuristic ranking signals" that are "uncalibrated … used primarily for ranking rather than absolute predictions". `[SECOND-HAND]` — [search over docking-score literature](https://www.rowansci.com/blog/how-to-predict-binding-affinity)
- AlphaFold3-based affinity ranking correlates with structural confidence (pLDDT), i.e. confidence is informative for *ranking* in a neighbouring task. `[SECOND-HAND]` — [How Good is AlphaFold3 at Ranking Drug Binding Affinities?, bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.05.27.656341v1.full.pdf)
- VN-EGNN has a learned self-confidence head and ranks by it, but publishes **no calibration analysis and no ablation of the head**. `[PRIMARY-SUMMARISED]` — [VN-EGNN PMC12837241](https://pmc.ncbi.nlm.nih.gov/articles/PMC12837241)

### Inferences
- Monotone post-hoc calibration (Platt, isotonic) applied to a *pocket-level* score cannot change a ranking at all — it is order-preserving. Calibration can only change top-1 if it is applied **per-point before a non-linear aggregation**: because `Σ p²` is convex in `p`, recalibrating the per-point probabilities changes the relative contribution of many-mediocre-points vs few-excellent-points, and therefore *does* reorder pockets. This is a concrete, cheap, untested lever, and the literature's silence on it is an opportunity rather than evidence of futility.
- The corollary is a warning: if a system's per-point scores are miscalibrated in a size-correlated way (e.g. systematically over-confident on large shallow surfaces), a cumulative aggregator will amplify that into exactly the observed failure — a large distant cavity ranked first.

### Gaps
- No primary source found on pocket-score calibration or on ECE/reliability diagrams for binding-site predictors. Report this as unstudied.

---

## Q6. Learning-to-rank (listwise / pairwise) applied to pocket candidates

### Takeaway
**No published binding-site method was found that uses a pairwise or listwise ranking loss over pocket candidates.** Every learned ranker in this field (PRANK, P2Rank, DeepPocket, GrASP, VN-EGNN) is **pointwise** — per-point or per-candidate binary classification, then sort. Pairwise ranking losses do appear in adjacent structural-biology ranking tasks and work there, which makes this a clearly identified, unexploited transfer.

### Cited Findings
- PRANK, P2Rank: per-SAS-point binary classification (ligandable / not), then sort pockets by aggregated score. **Pointwise.** `[PRIMARY-SUMMARISED]` — [PRANK PMC4414931](https://pmc.ncbi.nlm.nih.gov/articles/PMC4414931/); [P2Rank PMC6091426](https://pmc.ncbi.nlm.nih.gov/articles/PMC6091426/)
- DeepPocket: per-candidate binary classification with oversampled balanced batches, Adam, lr 1e-3, weight decay 1e-3, 200k iterations, rotational augmentation. **Pointwise, with the class balance handled by sampling rather than by a ranking objective.** `[PRIMARY]` — [DeepPocket PDF](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2022_01/Deeppocket.pdf)
- Adjacent task with a working pairwise approach: ranking clusters of docked protein–protein complexes by **pairwise cluster comparison** — clusters are described by the distribution of molecular descriptors within them, and a model is trained to discriminate near-native from incorrect clusters *by comparing pairs*, rather than scoring each in isolation. `[SECOND-HAND]` — [A machine learning approach for ranking clusters of docked protein-protein complexes by pairwise cluster comparison (PMC5396268)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5396268/)
- The canonical LTR taxonomy and the argument for listwise over pairwise (ranking is a prediction task on a *list*, so the loss should be defined on lists) is Cao et al., ListNet. `[SECOND-HAND, methodological reference only]` — [Learning to Rank: From Pairwise Approach to Listwise Approach, Microsoft Research TR-2007-40](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf)
- Multi-criteria GNN-based algorithm selection for molecular docking (MC-GNNAS-Dock, 2025) applies ranking-style selection in the docking domain. `[SECOND-HAND, not read]` — [arXiv:2509.26377](https://arxiv.org/pdf/2509.26377)

### Inferences
- The structure of this problem is textbook LTR: one **query** = one protein, one **candidate list** = its cavities, **one relevant item** (or n), and the metric of interest is top-1 / MRR / NDCG@1. A pointwise binary classifier optimises a per-candidate threshold that nothing in the evaluation cares about; it spends capacity separating easy negatives from each other instead of separating the best cavity from the second-best. A listwise loss (softmax-over-candidates / ListNet, or LambdaRank with NDCG@1 weighting) puts the gradient exactly where the described failure is — the rank-1 vs rank-2/3 boundary that accounts for ~70 % of this system's errors.
- Because the correct pocket is already at rank 2–3 in ~70 % of failures, the quantity to optimise is a **within-protein** margin, which only a pairwise/listwise objective expresses. A pointwise loss cannot represent "pocket A must outscore pocket B of the same protein" at all.
- The absence of LTR from this field is most plausibly historical: PRANK framed the task as point classification in 2015 and every subsequent method inherited the framing, including the deep ones.

### Gaps
- No in-field evidence on whether listwise beats pointwise for pocket ranking; the inference above is architectural reasoning from the loss function, not a measured effect, and should be labelled as such in the report.
- Virtual-screening rescoring with ranking losses was searched for but returned only generic LTR methodology and docking-cluster work; no VS-rescoring paper with a clean pointwise-vs-listwise ablation was found.

---

## Q7. Two-stage / cascade re-ranking: does a second model on a shortlist help?

### Takeaway
**Yes — this is the single best-evidenced intervention in the entire literature, and it is the only one with three independent confirmations.** Re-scoring a geometric detector's candidate list with a learned model buys +7 to +14 Top-n points (PRANK over fpocket), and on the largest independent benchmark (LIGYSIS, 2024) the two best methods of all 13+15 evaluated are **both** fpocket-plus-a-re-ranker. The documented failure mode is not that re-ranking fails, but that it is **hard-capped by the first stage's candidate ceiling**.

### Cited Findings
- **PRANK re-scoring fpocket's own candidates (DCA 4 Å, Top-n → Top-n):** CHEN11 47.1 → 58.2 (+11.1); JOINED 53.8 → 68.2 (+14.4); COACH420 56.4 → 63.6 (+7.2); HOLO4K 52.4 → 62.0 (+9.6); COACH420(Mlig) 57.4 → 64.0 (+6.6); HOLO4K(Mlig) 56.9 → 68.3 (+11.4). Top-(n+2): COACH420 68.9 → 76.5; HOLO4K 63.1 → 71.0. **The candidate set is unchanged; only the ordering changed.** `[PRIMARY]` — [P2Rank supplementary, Table 1](https://static-content.springer.com/esm/art%3A10.1186%2Fs13321-018-0285-8/MediaObjects/13321_2018_285_MOESM1_ESM.pdf)
- PRANK's original paper reports the same effect with its own numbers, including +24.1 points on U48 Top-n (53.7 → 77.8) over fpocket and +18.8 on DT198 (37.5 → 56.2); over ConCavity, +15.6 on DT198 (45.8 → 61.5). It states PRANK recovered **40–86 % of the theoretical improvement margin** (i.e. of the gap between native-ranking Top-n and perfect ranking of the same candidates), with larger gains on fpocket's longer candidate lists. `[PRIMARY-SUMMARISED]` — [PRANK PMC4414931](https://pmc.ncbi.nlm.nih.gov/articles/PMC4414931/)
- PRANK can re-score the output of fpocket, ConCavity, SiteHound, MetaPocket 2.0, LISE and DeepSite — i.e. the cascade is detector-agnostic. `[SECOND-HAND]` — [P2Rank project page](https://siret.ms.mff.cuni.cz/p2rank)
- **DeepPocket** (3D-CNN re-scoring of fpocket candidates) beats P2Rank on 5 of 6 dataset×metric cells, most decisively at Top-(n+2): COACH420 79.94 vs 75.48; HOLO4K 82.97 vs 80.05; SC6K 83.01 vs 75.74. `[PRIMARY]` — [DeepPocket Table 1](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2022_01/Deeppocket.pdf)
- **Independent confirmation on LIGYSIS (2024), 13 methods + 15 variants, 2,775 chains / 8,244 sites:** the top two Top-(N+2) recalls are **fpocket+PRANK 60.4 %** and **DeepPocket_RESC 58.1 %** — both cascades over fpocket — ahead of P2Rank_CONS 53.9 %, P2Rank 51.9 %, GrASP 49.9 %, and IF-SitePred 25.7 % (lowest). The authors' summary: "Re-scoring fpocket predictions by PRANK and DeepPocket display the highest recall (60 %)". `[PRIMARY-SUMMARISED]` — [Utgés & Barton 2024 (PMC11552181)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/)
- **The documented limit of the idea** (the nearest thing to a reported failure): in the same LIGYSIS study, "DeepPocket_RESC, DeepPocket_SEG-NR and fpocket_PRANK, all starting from fpocket candidates, achieve no more than ≈600 TP" at the fixed-false-positive operating point — i.e. all three cascades plateau at the same ceiling because they share the same first stage. The authors read this as **the quality of the base candidates mattering more than the scoring methodology**. `[PRIMARY-SUMMARISED]` — [Utgés & Barton 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/)
- The same ceiling effect is what DeepPocket itself blames for its one loss: fpocket covers only 80.78 % of COACH420 sites, DeepPocket ranks 85 % of what fpocket found into Top-n, and that product is what P2Rank's 68.24 % narrowly beats. `[PRIMARY]` — [DeepPocket PDF](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2022_01/Deeppocket.pdf)
- No paper was found reporting that cascade re-ranking **hurt** performance.

### Inferences
- For the system described in the brief, this is the decisive finding: the published cascades are all bottlenecked by an ~81–92 % first-stage ceiling, and a system at a **0.99 candidate ceiling has strictly more headroom than any of them**. Applying PRANK's accounting (40–86 % of the theoretical margin recovered) to a 0.99 ceiling rather than fpocket's 0.81 is the single highest-expected-value framing available.
- The result also implies the second-stage model class is **not** the lever: a random forest on 35 hand-crafted features (PRANK) and a 3D CNN on voxel grids (DeepPocket) land within ~2 points of each other on LIGYSIS Top-(N+2) and plateau at the same TP count. What differs is the candidate set and the label definition — not the architecture. This should temper any expectation that a better equivariant backbone fixes rank 1.
- `0.99 ceiling + rank-1 trailing a 2018 RF` is diagnostically the *inverse* of fpocket's profile (great detection, bad ranking) and the *same* profile as fpocket itself. The literature's answer to fpocket's profile was, twice, "bolt on a discriminative re-ranker trained on the real candidate distribution".

### Gaps
- No ablation found that isolates how much of DeepPocket's gain comes from re-scoring versus from the segmentation stage. The paper's structure (segmentation applied only to already-top-ranked pockets, evaluated with DCC/DVO rather than ranked DCA) means segmentation **cannot** be contributing to the DCA Top-n/Top-(n+2) numbers — so all of the Table 1 gain is attributable to re-scoring. This is an inference from the pipeline order, not a stated ablation.
- Whether a *third* stage (re-rank the top 3–5 with a more expensive model) has ever been tried: no source found either way.

---

## Q8. What distinguishes a ligand-binding cavity from an equally buried non-binding one?

### Takeaway
The in-field evidence says the discriminating signal is **buriedness-with-chemistry** rather than buriedness alone — protrusion plus aromatic/hydrophobic/H-bond-acceptor character — and that is measured, with effect sizes. The dynamics and water-displacement-thermodynamics accounts are mechanistically attractive but I found **no evidence that either has been turned into a ranking feature and measured on a binding-site benchmark**; the water-thermodynamics literature I could reach is about synthetic cavitand hosts, not protein cavity discrimination.

### Cited Findings
- **Buriedness alone is strong but not sufficient:** protrusion-only P2Rank gets COACH420 Top-n 64.2 vs 71.4 for the full feature set — so chemistry adds 7.2 points on top of pure buriedness; conversely buriedness alone retrieves 90 % of full performance. `[PRIMARY]` — [P2Rank supplementary Table 3](https://static-content.springer.com/esm/art%3A10.1186%2Fs13321-018-0285-8/MediaObjects/13321_2018_285_MOESM1_ESM.pdf)
- **Which chemistry:** after protrusion (0.0845) and bfactor (0.0139), the ranked contributors are aromaticity-related and H-bond/polarity-related — apRawInvalids 0.0118, vsAromatic 0.0102, apRawValids 0.0094, atomO 0.0093, hydrophobic 0.0086, hydrophilic 0.0076, vsAcceptor 0.0062, vsHydrophobic 0.0053. Charge and amide features are the *least* important (vsCation 0.00083, amide 0.00083, posCharge 0.0010, hBondDonor 0.0011). `[PRIMARY]` — [P2Rank supplementary Table 5](https://static-content.springer.com/esm/art%3A10.1186%2Fs13321-018-0285-8/MediaObjects/13321_2018_285_MOESM1_ESM.pdf)
- **Enclosure as an explicit descriptor:** SiteMap's Dscore is size + **enclosure** + hydrophilicity penalty, three descriptors only — i.e. the commercial state of the art also reduces the discrimination to buriedness + hydrophobic enclosure. `[SECOND-HAND]` — [search over SiteMap/druggability literature](https://www.nature.com/articles/s41598-022-12105-8)
- **Conservation** adds ~+2.0 Top-(N+2) points on train-dissimilar structures (Q3), i.e. functional-site evolutionary signal is a real but small discriminator on top of geometry+chemistry. `[PRIMARY-SUMMARISED]` — [Utgés & Barton 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/)
- **Hydration/dewetting:** the available primary work is on deep-cavity **cavitands** (synthetic hosts), where nonpolar pockets are "largely dry in water with only transient (sub-nanosecond) filling events", portal methyl groups favour dewetting and hydroxyl groups favour wetting, and dewetting transitions mediate cavity–ligand recognition. Computed free-energy cost of displacing water from binding sites spans **0 to +37 kcal/mol**. `[SECOND-HAND, and on a non-protein model system]` — [Spontaneous drying of non-polar deep-cavity cavitand pockets (PMC8170693)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8170693/); [Role of water and steric constraints in cavity–ligand unbinding, PNAS 2015](https://www.pnas.org/doi/10.1073/pnas.1516652112); [Thermodynamics of Water Displacement from Binding Sites (PMC12377430)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12377430/)
- **B-factor** — a crude dynamics proxy — is P2Rank's **second** most important feature (0.0139), 1.6× the next. `[PRIMARY]` — [P2Rank supplementary Table 5](https://static-content.springer.com/esm/art%3A10.1186%2Fs13321-018-0285-8/MediaObjects/13321_2018_285_MOESM1_ESM.pdf)

### Inferences
- The measured hierarchy is: **buriedness ≫ local flexibility (B-factor) > aromatic/hydrophobic/acceptor character > conservation ≫ charge**. The near-irrelevance of charge features is the most actionable and least intuitive item: electrostatic complementarity, which one might expect to matter, carries almost no discriminative weight in the one place it has been measured.
- B-factor ranking second is a double-edged finding. It is the only dynamics-adjacent feature with a measured effect in this field, which supports adding a flexibility/dynamics channel. But it is also an experimental-structure artefact, so a model that is competing against P2Rank on crystal structures is competing against a model that gets a free dynamics proxy it may not have.
- The water-displacement account is the best mechanistic story for "equally buried, one binds" — a cavity whose waters are high-energy and displaceable is ligandable, one whose waters are well-coordinated is not — and a ~37 kcal/mol spread is far larger than any geometric difference. But it has not been reduced to a cheap per-cavity feature and validated on COACH420/HOLO4K, so it should be presented as a hypothesis with a mechanism, not as an established ranking signal.

### Gaps
- Found **no** paper that computes a hydration-thermodynamic or MD-derived descriptor per candidate cavity and measures the resulting change in top-1 / Top-n on a standard binding-site benchmark. If such work exists it was not surfaced by these searches.
- No study found that constructs the controlled comparison the question asks for — matched pairs of equally buried binding and non-binding cavities — and reports which descriptors separate them.

---

## Q9. Does ligand identity or a ligand prior change the ranking, and what does it cost?

### Takeaway
Yes, demonstrably and substantially: **YuelPocket (PNAS 2026)** conditions on the ligand graph and reports **Top-1 DCA ~62 % vs ~55 % for P2Rank** and **Top-1 DCC ~45 % vs ~40 %** on a PLINDER test set — the only clean Top-1 comparison found in this literature. The applicability cost is that you must supply a ligand; the authors' mitigation is a hand-picked **15-ligand "minimal probe set"**, but they publish **no ablation quantifying the loss when the true ligand is unknown**, which leaves the practical gain unquantified for the apo / unknown-ligand setting that binding-site prediction is normally used in.

### Cited Findings
- YuelPocket takes a protein graph and a small-molecule graph jointly, with residue-level and coordinate-level prediction modes; candidates are **SAS-point probes** scored per protein–ligand pair, grouped by a **hill-climbing + union-find** procedure, and "clusters are then ranked by their **peak scores**" — note: peak (max), not sum or mean. `[PRIMARY-SUMMARISED]` — [Unified protein–small molecule GNNs for binding site prediction, PNAS 2026 (PMC12974528)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12974528/)
- Ligand identity changes **which** site is returned, not just the score: for 1D1V, the native ligand H4B retrieved only its own pocket, while probe KI2 "successfully identifies both the H4B and PTU binding sites"; the paper states "the same protein can exhibit dramatically different binding site predictions depending on the input ligand". `[PRIMARY-SUMMARISED]` — [PNAS 2026 (PMC12974528)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12974528/)
- PLINDER test set (1,036 systems): **Top-1 DCA @4 Å ≈ 62 % (YuelPocket) vs ≈ 55 % (P2Rank); Top-1 DCC @4 Å ≈ 45 % vs ≈ 40 %**. Comparisons are to P2Rank only — no numbers vs DeepPocket or VN-EGNN. Values read off figures, so treat as approximate. `[PRIMARY-SUMMARISED, approximate]` — [PNAS 2026 (PMC12974528)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12974528/)
- Trained on PLINDER; claimed linear complexity O(C + m + n). `[PRIMARY-SUMMARISED]` — [PNAS 2026 (PMC12974528)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12974528/)
- The "minimal probe set" is **15 selected ligands** intended to cover binding space across diverse targets without knowing the true ligand. `[PRIMARY-SUMMARISED]` — [PNAS 2026 (PMC12974528)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12974528/)
- Code at [github.com/hust220/yuel_pocket](https://github.com/hust220/yuel_pocket); archived release at [Zenodo 18065818](https://zenodo.org/records/18065818). `[SECOND-HAND]`
- Adjacent 2025 ligand-aware work exists (LABind, "identifying protein binding ligand-aware sites via learning interactions between ligand and protein"), confirming this is an active direction rather than one paper. `[SECOND-HAND, not read]` — [LABind (PMC12365077)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12365077)

### Inferences
- The ~7-point Top-1 DCA gain over P2Rank is the largest reported Top-1 improvement found anywhere in this search, but it is **not comparable** to the apo/unknown-ligand benchmarks (COACH420/HOLO4K/LIGYSIS) that P2Rank's 72 % Top-n comes from — different dataset, different metric definition, and the ligand is given. It should be reported as evidence that *a ligand prior reorders candidates*, not as evidence that ligand-conditioning beats P2Rank on the standard task.
- YuelPocket's ranking by **peak score** is a third distinct aggregation choice (max), alongside P2Rank's `Σ p²` and GrASP/VN-EGNN's mean. Three leading methods, three different aggregators, zero published comparisons between them — again a hyperparameter masquerading as a design decision.
- A ligand prior is plausibly a cheap partial substitute even without a known ligand: scoring each candidate against a small probe panel and taking the max over probes gives a per-cavity "is this ligandable by *anything* drug-like" score, which is closer to the quantity that decides rank 1 than a ligand-agnostic geometric score is. The 15-probe set is evidence the authors think this works; the absence of an ablation is evidence nobody has measured it.

### Gaps
- **No ablation quantifying YuelPocket's performance with the 15-probe set versus the true ligand versus ligand-agnostic baselines.** Without it, the applicability cost is unquantified and the headline Top-1 number cannot be transferred to the apo setting.
- No statement in the paper about the practical cost of requiring a ligand.
- Could not retrieve a 2026 bioRxiv preprint (10.64898/2026.01.28.702257) that surfaced in a GrASP-ranking search and may be directly on-topic; the server returned HTTP 429 on two attempts. Worth a retry by whoever picks this up.

---

## Cross-cutting notes for the report writer

- **The metric is part of the problem.** Top-n with n = number of ligands is *not* top-1, and the field's own preferred metric (Utgés & Barton: Top-(N+2)) deliberately looks past the first rank. A system judged on top-1 is being held to a standard most of this literature does not report. When comparing, state which metric.
- **Ordering of interventions by strength of published evidence**, highest first:
  1. **Cascade re-ranking of a high-recall candidate set with a discriminative model trained on realistic negatives** — three independent confirmations, +7 to +14 Top-n points, best two methods on the largest 2024 benchmark. `[PRIMARY]`
  2. **Getting buriedness right** — protrusion alone = 90 % of P2Rank; its removal costs 10.9 Top-n points on COACH420. `[PRIMARY]`
  3. **Aggregation function choice** (`Σ p²` vs mean vs max) — strong indirect evidence (PRANK tested and rejected normalised scoring; the three leading methods use three different aggregators), but **no published sweep**. `[PRIMARY for the PRANK choice, unstudied otherwise]`
  4. **Conservation** — +2.0 Top-(N+2) points, measured on train-dissimilar structures, no training-set dependence. `[PRIMARY-SUMMARISED]`
  5. **Ligand/probe conditioning** — ~+7 Top-1 DCA points, but on a different benchmark with the ligand given and no apo ablation. `[PRIMARY-SUMMARISED, not transferable]`
  6. **Listwise/pairwise ranking losses** — zero in-field evidence; strong architectural argument; adjacent-field precedent only. `[inference]`
  7. **Per-point calibration before a convex aggregator** — zero evidence either way; mechanically capable of reordering, unlike post-hoc pocket-level calibration which is order-preserving and therefore cannot. `[inference]`
  8. **Hydration thermodynamics / dynamics descriptors** — best mechanistic story, no benchmark evidence, and the reachable literature is on synthetic hosts. `[inference, weak]`
- **Things that are hyperparameters, not learned components**, and that no paper ablates: P2Rank's 3 Å single-linkage cutoff and SAS score threshold (tuned on JOINED); GrASP's mean-shift bandwidth and majority-vote residue assignment; VN-EGNN's mean-shift parameters and its ~12 virtual-node count; YuelPocket's hill-climbing/union-find grouping; the exponent 2 in `Σ p²`; DeepPocket's 4 Å positive-label radius and 23.5 Å grid.
- **The one negative result worth reporting:** three different second-stage models over the same fpocket candidates all plateau at ≈600 TP on LIGYSIS, which the benchmark authors read as base-candidate quality mattering more than scoring methodology. This is the cautionary counterpart to finding (1) — re-ranking is worth a lot, and then it stops, at the first stage's ceiling.
