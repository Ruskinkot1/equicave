# Chemistry in the architecture: where chemical information enters an equivariant GNN, and evidence on channel redundancy

**Scope note / verification key.** Every finding below is tagged:
- **[PRIMARY]** = I read the paper's own text/table (PDF text extraction or full-text HTML fetch) and am reporting numbers as printed.
- **[PRIMARY-FETCH]** = full text was fetched and the table reproduced by the fetch tool; I did not see the raw PDF bytes, so transcription error is possible.
- **[SECOND-HAND]** = search-result snippet or summary only; **not** verified against the paper.

**Overall epistemic warning for the report writer.** The literature that bears directly on this brief is thin, recent, and dominated by arXiv preprints (several from 2026 that are not yet peer-reviewed). There is **no** paper I could find that does the exact experiment the user wants (hand-built pharmacophore channels vs. learned atom-type embeddings, parameter-matched, multi-seed, on a protein structure task). The useful evidence is adjacent: atom-type-embedding vs. frozen PLM, surface vs. graph, auxiliary-loss-on/off. I flag the substitution each time.

---

## WHERE CHEMISTRY GOES: scalar node features, edge features, separate chemical graph, or learned atom-type embeddings

### Takeaway
Two 2024–2026 binding-site papers (VN-EGNN, GDEGAN) run clean on/off comparisons of the **node-scalar** channel with the rest of the architecture fixed, and in both the node-scalar channel is where chemical/evolutionary information pays off (+0.06 to +0.12 DCC). I found **no** paper that holds everything else fixed and moves the same chemical information between node scalars, edge features, and a separate chemical graph — that specific placement comparison appears not to exist. The de-facto convention in the papers that measure anything is: **chemistry on node scalars, geometry on edges and in the steerable/tensor channels.**

### Cited Findings
- **[PRIMARY]** GDEGAN's architecture puts chemistry and geometry in explicitly separate places: node scalar features `h_i ∈ R^{n_d}` are initialised from pre-trained ESM-2 embeddings, while **edge** scalar features are geometry-only (RBF of distance, smooth cutoff `φ(r̃_ij)`, and inner products of degree-`l` steerable features), and spatial information lives in spherical-harmonic steerable tensors up to `L_max = 2`. The message aggregation is `m_i = Σ_j (W_a(h_j) ∘ ϕ(r̃_ij) W_rbf ∘ φ(r̃_ij))` — i.e. the chemical node channel is multiplicatively gated by the geometric edge channel rather than concatenated with it. — [GDEGAN, arXiv 2603.19817](https://arxiv.org/pdf/2603.19817)
- **[PRIMARY]** GDEGAN Table 2 (ablation), COACH420 / HOLO4K / PDBbind2020, DCC and DCA, std dev in brackets:

  | Method | Equiv. | Aux. dir. loss | ESM | C420 DCC | C420 DCA | H4K DCC | H4K DCA | PDBb DCC | PDBb DCA |
  |---|---|---|---|---|---|---|---|---|---|
  | GotenNet | E(3) | No | No | 0.454(0.007) | 0.624(0.014) | 0.464(0.001) | 0.691(0.005) | 0.553(0.008) | 0.705(0.007) |
  | GotenNet+ADL | E(3) | Yes | No | 0.485(0.004) | 0.642(0.011) | 0.468(0.004) | 0.732(0.004) | 0.592(0.010) | 0.748(0.003) |
  | GotenNet+ESM | SE(3) | No | Yes | 0.543(0.008) | 0.693(0.006) | 0.520(0.011) | 0.753(0.004) | 0.637(0.005) | 0.760(0.004) |
  | GotenNet(full) | SE(3) | Yes | Yes | 0.556(0.002) | 0.703(0.005) | 0.529(0.011) | 0.749(0.006) | 0.649(0.005) | 0.801(0.014) |
  | GDEGAN+ESM | SE(3) | No | Yes | 0.572(0.001) | 0.702(0.002) | 0.532(0.010) | 0.769(0.006) | 0.652(0.010) | 0.810(0.011) |
  | GDEGAN(full) | SE(3) | Yes | Yes | 0.580(0.008) | 0.707(0.009) | 0.560(0.013) | 0.788(0.011) | 0.675(0.010) | 0.826(0.011) |

  — [GDEGAN, arXiv 2603.19817](https://arxiv.org/pdf/2603.19817)
- **[PRIMARY]** GDEGAN's own summary of that table: "Using ESM embeddings (GotenNet+ESM) instead of atomic numbers embeddings (GotenNet) improves the results by 15.61% DCC and 9.27% DCA averaged across datasets." — [GDEGAN, arXiv 2603.19817](https://arxiv.org/pdf/2603.19817)
- **[PRIMARY]** **Placement has an equivariance cost that the paper states explicitly.** Putting ESM features on node scalars **breaks E(3) and leaves only SE(3)**: "The E(3) equivariance breaks after the introduction of ESM-embeddings because, now node features do not encode chirality information." The `EQc` column in Table 2 above confirms the drop from E(3) to SE(3) for every ESM arm. — [GDEGAN, arXiv 2603.19817](https://arxiv.org/pdf/2603.19817)
- **[PRIMARY]** GDEGAN's design note on **choosing a chemical channel to avoid redundancy with the geometric channel**: ESM-2 was chosen because it provides "sequential information orthogonal to our geometric processing, unlike structure aware alternatives like ProstT5 (Heinzinger et al., 2024) that would create geometric features redundancy (Mallet et al., 2025)." This is an *a priori* design argument, **not** a measured ablation — they did not run the ProstT5 arm. — [GDEGAN, arXiv 2603.19817](https://arxiv.org/pdf/2603.19817)
- **[PRIMARY-FETCH]** VN-EGNN Table 2, same question (initial node embedding swap, architecture fixed), DCC/DCA with std dev over training re-runs:

  | Variant | VN | Heterog. MP | ESM | C420 DCC/DCA | H4K DCC/DCA | PDBb DCC/DCA |
  |---|---|---|---|---|---|---|
  | EGNN | ✗ | ✗ | ✗ | 0.156(0.017)/0.361(0.020) | 0.127(0.005)/0.406(0.004) | 0.143(0.007)/0.302(0.006) |
  | Residue emb. (one-hot AA) | ✓ | ✓ | ✗ | 0.503(0.022)/0.684(0.016) | 0.438(0.019)/0.605(0.013) | 0.551(0.017)/0.751(0.009) |
  | Homog. MP | ✓ | ✗ | ✗ | 0.497(0.014)/0.700(0.013) | 0.414(0.023)/0.618(0.024) | 0.502(0.029)/0.717(0.025) |
  | Homog. MP | ✓ | ✗ | ✓ | 0.575(0.008)/0.708(0.009) | 0.479(0.012)/0.595(0.010) | 0.649(0.010)/0.805(0.006) |
  | Full | ✓ | ✓ | ✓ | 0.605(0.009)/0.750(0.008) | 0.532(0.021)/0.659(0.026) | 0.669(0.015)/0.820(0.010) |

  — [VN-EGNN, arXiv 2404.07194 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2404.07194)
- **[PRIMARY-FETCH]** AtomSurf places chemistry and geometry on *different node sets*: surface vertices carry **only geometric** descriptors (Gaussian and mean curvature, shape index, normal, heat kernel signature); the residue graph carries the chemistry (one-hot residue type, secondary structure, **hydrophobicity**, ESM-650M); bipartite surface↔graph edges carry geometry only (direction, distance and normal angle in 16 RBFs, each vertex to its 16 nearest graph nodes). — [AtomSurf, arXiv 2309.16519v4](https://arxiv.org/html/2309.16519v4)
- **[SECOND-HAND]** EquiPocket's own ablation is at **module** level, not feature level; its global module bundles chemical and spatial information together (atom type and chemical bonds), so it does not isolate the chemical contribution. Reported module effects: global module ≈ +10% DCC / +20% DCA; local geometric module alone "negligible"; surface message passing ≈ +20% DCC and DCA. I could **not** verify these numbers — the arXiv PDF returned HTTP 404 on fetch. — [EquiPocket arXiv page](https://arxiv.org/pdf/2302.12177); [ICML 2024 abstract](https://icml.cc/virtual/2024/oral/35580)

### Inferences
- The only placement question with measured answers is "node scalars: rich or poor?", and the answer is consistently "rich wins by a lot." The user's question of *node scalars vs. edge features vs. separate chemical graph* is **unanswered by the literature**; GDEGAN and AtomSurf both simply assume node-scalars-for-chemistry and reserve edges/tensors for geometry.
- GDEGAN's multiplicative gating (`W_a(h_j) ∘ geometric edge terms`) is a third placement option that neither "node" nor "edge" describes cleanly: chemistry sets *what* is sent, geometry sets *how strongly*. For a model that already has pharmacophore availability on probe nodes, this is the cheapest untested variant — gate the existing geometric edge terms by the pharmacophore vector rather than concatenating a new channel.
- **Actionable warning:** if EquiCave currently claims E(3) (not merely SE(3)) equivariance, adding any achiral per-node chemical descriptor is fine, but adding PLM embeddings would, by GDEGAN's own argument, silently demote the guarantee to SE(3). Worth an equivariance unit test on whatever is added.

### Gaps
- No paper found that moves the *same* chemical information between node scalars, edge features, and a parallel chemical graph with parameters and data held fixed. This comparison appears genuinely absent from the literature.
- EquiPocket's Table 1/ablation numbers could not be verified from the primary PDF (404). GDEGAN's Table 1 does reproduce EquiPocket's published results (COACH420 DCC 0.423(0.014), DCA 0.656(0.007); HOLO4K 0.337(0.006)/0.662(0.007); PDBbind2020 0.545(0.010)/0.721(0.004), 1.70M params) — **[PRIMARY]** as a reproduction, second-hand as to EquiPocket itself.
- PaiNN, GVP, NequIP, MACE, Equiformer and GotenNet were not reachable within budget for their own feature-placement ablations. Their ablations are, from prior knowledge, about tensor degree and message form rather than chemical-feature placement — but I did **not** verify this, so it should be stated as unverified or dropped.

---

## HAND-BUILT VERSUS LEARNED: pharmacophore counts and distances vs. learned embeddings

### Takeaway
There is **no** controlled comparison of P2Rank/GRID-style pharmacophore features against learned atom-type embeddings on a protein structure task that I could find. What does exist is the adjacent comparison — **learned end-to-end atom-type/residue-type embedding vs. frozen pre-trained PLM embedding** — run twice with multiple seeds, and the PLM wins decisively *in distribution*. Critically, two independent sources find that the PLM advantage **shrinks or reverses under strict similarity-filtered splits**, which means the "learned wins" conclusion is split-conditioned, not general.

### Cited Findings
- **[PRIMARY]** GDEGAN: learned **atomic-number embeddings** (trained end to end) vs. frozen ESM-2 on node scalars, same architecture: COACH420 DCC 0.454(0.007) → 0.543(0.008); HOLO4K 0.464(0.001) → 0.520(0.011); PDBbind2020 0.553(0.008) → 0.637(0.005). Every delta is 5–50× the larger std dev. — [GDEGAN, arXiv 2603.19817](https://arxiv.org/pdf/2603.19817)
- **[PRIMARY-FETCH]** VN-EGNN: one-hot amino-acid encoding vs. ESM-2, full architecture otherwise identical: COACH420 DCC 0.503(0.022) → 0.605(0.009) (Δ +0.102); HOLO4K 0.438(0.019) → 0.532(0.021) (Δ +0.094); PDBbind2020 0.551(0.017) → 0.669(0.015) (Δ +0.118). The paper also notes its method beats prior work "regardless of the initial embeddings used." — [VN-EGNN, arXiv 2404.07194](https://ar5iv.labs.arxiv.org/html/2404.07194)
- **[PRIMARY-FETCH]** Non-monotonicity in the same VN-EGNN table: on **HOLO4K DCA**, the one-hot arm (0.605±0.013) *beats* the Homog.-MP+ESM arm (0.595±0.010). So the PLM advantage is not uniform across metric/dataset even within one paper.
- **[PRIMARY]** **HonestAffinity** is the cleanest controlled swap in the set: "Same schedule, same data, same pocket marker; only the protein token representation is swapped (ESM-2 → 21-vocabulary residue embedding)." Pearson R, mean±std over **three seeds**, ΔR = (Pocket-NoESM) − (Pocket); positive = replacing ESM-2 **improves**:

  | Split | Pocket (ESM-2) | Pocket-NoESM (learned 21-vocab) | ΔR | rel. |
  |---|---|---|---|---|
  | val | 0.548 ± 0.011 | 0.529 ± 0.015 | −0.019 | −3.5% |
  | test cl1 | 0.507 ± 0.014 | 0.531 ± 0.033 | **+0.024** | +4.7% |
  | test cl2 | 0.496 ± 0.021 | 0.538 ± 0.059 | **+0.042** | +8.5% |
  | test cl3 | 0.433 ± 0.023 | 0.497 ± 0.076 | **+0.064** | +14.9% |
  | CASF-2016 | 0.747 ± 0.031 | 0.713 ± 0.039 | −0.034 | −4.6% |
  | CASF non-train | 0.646 ± 0.027 | 0.632 ± 0.044 | −0.014 | −2.2% |

  Training set: 11,513 leak-proof complexes; ~3 GPU-hours on one Tesla V100. — [HonestAffinity, arXiv 2606.03422](https://arxiv.org/pdf/2606.03422)
- **[PRIMARY]** HonestAffinity's mechanistic reading: PLM features encode "within-family discriminators, such as residue conservation, secondary-structure cues, and binding-site signatures, that are most informative when the test target's protein family overlaps the training distribution. Under LP-style similarity filtering, that overlap is reduced by construction, and signals that were useful within-family discriminators can become correlated with target identity rather than ligand affinity." — [HonestAffinity, arXiv 2606.03422](https://arxiv.org/pdf/2606.03422)
- **[PRIMARY-FETCH]** AtomSurf's ESM ablation reaches the same conclusion independently: removing ESM lowers performance ("truly an important feature"), but the model "stays competitive on most tasks" without it, and **on PINDER, which has stricter splits, the drop from removing ESM is smaller**. The authors hypothesise that "ESM-derived features may aid memorization, while structural features may generalize better," and label this as future work rather than a result. — [AtomSurf, arXiv 2309.16519v4](https://arxiv.org/html/2309.16519v4)
- **[PRIMARY]** A **small-data counter-case** where hand-built chemistry is actively harmful: DegradoMap (PROTAC degradability, 3,260 labeled entries, 28-dim node features, target-unseen split, consistent hyperparameters LR=1e-3/4 layers/hidden 128 across arms). Table 12, Δ positive = removing the feature *improves*:

  | Configuration | Val | Test | AUPRC | Δ |
  |---|---|---|---|---|
  | Full model (baseline) | 0.584 | 0.477 | 0.406 | – |
  | − pLDDT | 0.558 | 0.528 | 0.434 | +0.051 |
  | − SASA | 0.518 | 0.403 | 0.386 | −0.074 |
  | − Lysine indicator | 0.541 | 0.578 | 0.526 | **+0.101** |
  | − Physicochemical | 0.580 | 0.551 | 0.447 | **+0.074** |
  | − Disorder | 0.546 | 0.565 | 0.450 | +0.088 |
  | Only AA one-hot | 0.529 | 0.532 | 0.437 | +0.055 |

  — [DegradoMap, arXiv 2606.04021](https://arxiv.org/pdf/2606.04021)
- **[PRIMARY]** **…and why that table must not be cited as evidence.** DegradoMap's own multi-seed numbers show the seed spread swamps every one of those deltas: 3-seed mean 0.646 ± **0.124** and 6-seed mean 0.603 ± **0.097** AUROC on target-unseen (best seed 0.7449); 5-fold × 3 seeds gives overall 0.565 ± 0.052, 95% CI [0.490, 0.650]. Table 12 is single-run per arm. By the user's own criterion (effect < seed spread ⇒ not an effect), **none** of +0.101, +0.088, +0.074, +0.055, −0.074 is an effect. — [DegradoMap, arXiv 2606.04021](https://arxiv.org/pdf/2606.04021)
- **[SECOND-HAND]** Crossover with dataset size: on ChEMBL20, Chemprop (learned) was beaten by a fingerprint baseline below ~1024 datapoints and became competitive/better above it. Reported as a single-benchmark figure, not a constant. — [Pappu et al., arXiv 2011.12203](https://arxiv.org/pdf/2011.12203)
- **[SECOND-HAND]** Yang et al. (2019) report their learned graph model "struggled on low-label targets compared with the baseline, which relied on human-engineered descriptors," attributing it to learning features from scratch. — [Analyzing Learned Molecular Representations, arXiv 1904.01561](https://arxiv.org/pdf/1904.01561)
- **[SECOND-HAND]** Sabando et al. (2021), >25,000 models: "predictive performance using molecular embeddings was not better than traditional representations, contrary to what was expected." The retrieved snippet was truncated; verify before citing.

### Inferences
- **The comparison the user actually wants is not in the literature, and the available substitute is biased.** In both VN-EGNN and GDEGAN, the "learned" arm is **not learned end to end from the task data** — it is frozen transfer from ESM-2, pre-trained on UniRef. GDEGAN's only genuinely end-to-end-learned chemical embedding (atomic number) **loses** to the frozen PLM. So the measured result is *transferred beats hand-built* and *transferred beats learned-from-scratch*; it says nothing about *hand-built vs. learned-from-scratch at 1,100–25,000 structures*. The information budgets are wildly unmatched (see parameter-matching note below).
- Three independent lines (HonestAffinity ΔR sign flip, AtomSurf PINDER, Yang et al. low-label) converge on: **the advantage of a rich learned/transferred channel is largest where train and test share family, and smallest or negative where similarity is filtered out.** For EquiCave, whose evaluation is similarity-filtered, this predicts exactly the near-zero ESM-2 result the user measured. The user's −0.010 is the expected value under this literature, not an anomaly.
- At 1,100–25,000 structures EquiCave straddles the Pappu-style crossover region, and the DegradoMap case (3,260 entries) shows what the low-data failure mode looks like in practice — hand-built channels that are individually plausible but collectively indistinguishable from noise. The defensible read is: **hand-built pharmacophore features are the right default at EquiCave's scale, and additional hand-built channels should be expected to add nothing measurable.**

### Gaps
- No controlled, parameter-matched, multi-seed comparison of pharmacophore-type counts/distances (P2Rank, GRID, FTMap-style) against learned atom-type embeddings on any protein structure task. This is a real hole in the literature and worth saying so in the report.
- I could not verify whether GrASP ablates its own pharmacophore atom features. The GrASP PMC page redirected and I did not have budget to re-fetch. The paper does say it "re-evaluates assumptions in nearly every step" **[SECOND-HAND]**, which is suggestive but not an ablation. Worth one follow-up fetch of [PMC10402091](https://pmc.ncbi.nlm.nih.gov/articles/PMC10402091/).
- DeepPocket, Pafnucy and RTMScore were not reached within budget. No claim should be made about their ablations.

---

## WHICH NODES CARRY IT: protein atoms, virtual/probe nodes, surface points, or all

### Takeaway
Only one paper in the set varies this, and only indirectly. The measured convention is: **chemistry on the protein nodes; virtual/probe nodes get it by initialisation from a pool of protein features, not from their own chemical descriptors; surface points get geometry only.** No paper tests putting chemical descriptors directly on virtual or surface nodes, which is precisely what EquiCave does.

### Cited Findings
- **[PRIMARY-FETCH]** VN-EGNN: physical nodes are **residues** at Cα positions and carry the chemistry (ESM-2, or one-hot AA in the ablation, plus an extra category for virtual nodes). Virtual nodes are initialised as "the average of the residue features … by averaging the residue node features across the entire protein," start on a **Fibonacci sphere**, connect to residue nodes only and **not to each other**. "Chemical information enters only through the residue nodes." — [VN-EGNN, arXiv 2404.07194](https://ar5iv.labs.arxiv.org/html/2404.07194)
- **[PRIMARY-FETCH]** The sequential **heterogeneous** message-passing scheme (separate MLPs for physical and virtual nodes per layer) vs. a **homogeneous** variant (identical MLPs for both) is the closest thing to a which-nodes-carry-what ablation. With ESM on: COACH420 DCC 0.575(0.008) → 0.605(0.009), HOLO4K 0.479(0.012) → 0.532(0.021), PDBbind2020 0.649(0.010) → 0.669(0.015). All three favour giving virtual nodes their **own** transform. — [VN-EGNN, arXiv 2404.07194](https://ar5iv.labs.arxiv.org/html/2404.07194)
- **[PRIMARY-FETCH]** AtomSurf, surface vertices vs. graph nodes, ATOM3D protocol with ~200k parameters and **fixed input features** "with the aim of a fair comparison" (Table 3):

  | Model | PIP (AuROC) | MSP (AuROC) | PSR R_l | PSR R_g |
  |---|---|---|---|---|
  | 3DCNN | 0.844 | 0.574 | 0.431 | 0.789 |
  | Graph | 0.669 | 0.609 | 0.411 | 0.75 |
  | Surface Diff (surface only) | 0.837 | 0.500 | 0.330 | 0.643 |
  | AtomSurf-bench (combined) | **0.876** | **0.707** | **0.452** | **0.831** |

  Surface-only "consistently falls short," which the authors say "challenges assertions in previous purely surface-based methods"; the combined model wins on all tasks and they attribute it to "synergy." — [AtomSurf, arXiv 2309.16519v4](https://arxiv.org/html/2309.16519v4)
- **[PRIMARY-FETCH]** AtomSurf's own caveat on the surface arm: all arms use **minimal** inputs, and surface networks "may do better with richer input features." They explicitly do **not** claim surface features are redundant. — [AtomSurf, arXiv 2309.16519v4](https://arxiv.org/html/2309.16519v4)
- **[PRIMARY-FETCH]** AtomSurf architecture ablations: sequential (vs. bipartite) design is "underwhelming" (one quoted MSP comparison 70.7 vs. 60.9, arm labelling unclear in the fetched text); attention-only message passing without a geometric neighbourhood is "consistently outperformed by localized message passing"; best setting is several message-passing blocks, with "the most interconnected networks performing best." Appendix F.4 tables were truncated. — [AtomSurf, arXiv 2309.16519v4](https://arxiv.org/html/2309.16519v4)
- **[PRIMARY-FETCH]** AtomSurf parameter accounting: AtomSurf-bench ≈ **200k** parameters under the ATOM3D protocol; full AtomSurf ≈ **600k**, which the authors say is "less than half the parameters of other methods they list"; hybrid models use **equal channel widths** for the surface and graph encoders. Three replicates, reported as **standard error** (not std dev): PIP 90.9 ± 0.088, MSP 71.5 ± 1, PSR R_l 61.7 ± 0.27, PSR R_g 85.7 ± 0.22. No replicates for competing methods, so "no formal statistical tests were done." — [AtomSurf, arXiv 2309.16519v4](https://arxiv.org/html/2309.16519v4)

### Inferences
- EquiCave's design — hand-built pharmacophore availability **on probe nodes** — has no direct precedent in the measured literature. VN-EGNN deliberately does the opposite: probe/virtual nodes carry *pooled* protein features, and the chemistry stays on the protein nodes. The one thing VN-EGNN's heterogeneous-MP ablation does support is that probe nodes benefit from their own transform, which is consistent with giving them their own feature semantics.
- The gap between AtomSurf's surface gain (+0.207 AuROC on MSP over graph-only, parameter-matched at ~200k) and EquiCave's surface module at −0.011 is best explained by *how* the surface is coupled, not by whether surfaces help. AtomSurf's gain comes from **bipartite feature sharing between surface vertices and graph nodes at every layer** (16 nearest graph nodes per vertex, distance and normal angle in 16 RBFs), plus genuinely surface-specific descriptors (curvature, shape index, HKS). A bolted-on surface module that sees the same geometry the main graph already sees is the configuration AtomSurf's own "sequential is underwhelming" finding predicts will fail.
- **Actionable:** before concluding the surface channel is dead, test (a) surface-specific descriptors the atom graph cannot compute (Gaussian/mean curvature, shape index, heat kernel signature) and (b) per-layer bipartite coupling rather than a late fusion.

### Gaps
- No paper varies *which* node set carries a chemical descriptor while holding the descriptor itself fixed. Nothing measures pharmacophore features on probe nodes vs. on protein atoms vs. on both.
- AtomSurf Appendix F.4 (architecture) and F.5 (ESM ablation) tables were truncated in the fetch; I have the qualitative ESM conclusions but not the per-task numbers. Fetch with `offset=100000` to recover them.

---

## REDUNDANCY: is a chemical channel contributing nothing because another channel already carries it a known, named phenomenon?

### Takeaway
**The phenomenon is real and has been measured several times, but it has no single accepted name and no dedicated literature.** The user is not seeing something unusual — they are seeing the best-documented version of it (PLM-embedding redundancy under similarity-filtered evaluation), which two independent papers report with the same sign and the same mechanism. What is unusual is only that the user measured it carefully enough to notice; most papers that hit it either report the channel as "important" from an in-distribution split or, like GDEGAN, avoid the redundant channel *a priori* and cite the avoidance rather than measure it.

Names in circulation, none canonical: *feature redundancy*, *geometric features redundancy* (GDEGAN/AtomSurf usage), *split-conditioned reversal* (HonestAffinity's own coinage), *negative transfer* (multi-task framing), *subadditivity / redundancy ratio* (mechanistic-interpretability framing).

### Cited Findings
- **[PRIMARY]** **The sharpest measured case, and it matches the user's ESM-2 result in sign and mechanism.** HonestAffinity: "both the pocket marker and the 1280-dimensional ESM-2 input help on familiar/CASF-style splits but **reduce Pearson R on strict no-leak tiers**." ESM-2 vs. learned residue embedding: val +0.019, CASF-2016 +0.034, CASF non-train +0.014 in favour of ESM, but **−0.024 (test_cl1), −0.042 (test_cl2), −0.064 (test_cl3)**. Three seeds, mean±std, same schedule and data. — [HonestAffinity, arXiv 2606.03422](https://arxiv.org/pdf/2606.03422)
- **[PRIMARY]** HonestAffinity's second, independent reversal — the **pocket-position marker**, a channel whose information the structure arguably already carries (Table V, Pearson R, 3-seed mean±std):

  | Split | Pocket | NoPocket | ΔR |
  |---|---|---|---|
  | val | 0.548 ± 0.011 | 0.506 ± 0.003 | −0.042 |
  | test cl1 | 0.507 ± 0.014 | 0.525 ± 0.027 | +0.018 |
  | test cl2 | 0.496 ± 0.021 | 0.503 ± 0.031 | +0.008 |
  | test cl3 | 0.433 ± 0.023 | 0.455 ± 0.033 | +0.023 |
  | CASF-2016 | 0.747 ± 0.031 | 0.685 ± 0.043 | −0.063 |
  | CASF non-train | 0.646 ± 0.027 | 0.567 ± 0.017 | −0.079 |

  — [HonestAffinity, arXiv 2606.03422](https://arxiv.org/pdf/2606.03422)
- **[PRIMARY]** HonestAffinity's methodological framing, which is the most quotable sentence in this whole brief: "two structurally unrelated architectural choices, the binary pocket-position marker and the 1280-dimensional frozen ESM-2 input, both change sign between canonical and leak-proof evaluations in the same direction … **Single-regime reporting would have concealed both reversals.**" They recommend "paired canonical and leak-proof ablations" as standard practice. — [HonestAffinity, arXiv 2606.03422](https://arxiv.org/pdf/2606.03422)
- **[PRIMARY]** HonestAffinity's own honesty about variance, which bears directly on the user's ±0.009: the NoESM arm's std devs grow to **0.033, 0.059, 0.076** on test_cl1/cl2/cl3. The headline +0.064 at test_cl3 is **within one std dev** of that arm. The authors say so: "small best-mean margins … should be read descriptively," and they report paired bootstrap checks only for selected comparisons (ΔR = +0.037, +0.060, +0.082 with intervals > 0 for one comparison; significant on test_cl2/cl3 but **not** test_cl1). — [HonestAffinity, arXiv 2606.03422](https://arxiv.org/pdf/2606.03422)
- **[PRIMARY-FETCH]** AtomSurf, independently: the ESM drop is **smaller on PINDER's stricter splits**; "ESM-derived features may aid memorization, while structural features may generalize better" — explicitly flagged as a hypothesis for future work, not a result. — [AtomSurf, arXiv 2309.16519v4](https://arxiv.org/html/2309.16519v4)
- **[PRIMARY]** Redundancy avoided by design rather than measured: GDEGAN rejects ProstT5 (a *structure-aware* PLM) in favour of ESM-2 precisely because a structure-aware sequence embedding "would create geometric features redundancy," citing [AtomSurf (Mallet, Miao, Attaiki, Correia, Ovsjanikov), ICLR 2025](https://arxiv.org/html/2309.16519v4). This is the clearest statement in the set that practitioners treat "a chemical/sequence channel can be redundant with the geometric channel" as a known design hazard — while still not measuring it. — [GDEGAN, arXiv 2603.19817](https://arxiv.org/pdf/2603.19817)
- **[SECOND-HAND]** Redundancy quantified directly (non-protein, mechanistic interpretability): a single-cell foundation model study reports that "pairwise combinatorial ablation of same-pathway features produces universally subadditive effects (median **redundancy ratio 0.74**)." This is the only source I saw that defines a redundancy *metric* rather than describing the symptom. — [arXiv 2603.11940](https://arxiv.org/pdf/2603.11940)
- **[SECOND-HAND]** A PLM-already-encodes-it case: reduced amino-acid alphabets fed to ESMFold improved pseudo-perplexity but "no corresponding enhancement in structural accuracy is observed." — [Protein language models meet reduced amino acid alphabets, Bioinformatics 40(2):btae061](https://academic.oup.com/bioinformatics/article/40/2/btae061/7600424)
- **[SECOND-HAND]** A null-ablation-is-a-method-artifact warning worth repeating in the report: in a vision-language-action model study, a strong intervention "disrupts the residual stream's skip connections, producing inflated effect sizes," and after correction "ablation of 2–5 concept-associated features produces no statistically significant effect." The reverse caution also applies — how you ablate determines what you measure. — [arXiv 2603.19233](https://arxiv.org/pdf/2603.19233)
- **[SECOND-HAND]** Enzyme-classification feature ablation: ESM-2 gave the single largest gain (+0.200 macro-F1), but the **full** feature set gave the best AUROC and a *slightly lower* macro-F1 than ESM-2 + one-hot alone — read by the authors as "mild redundancy between ESM-2 and handcrafted features that the model resolves differently across metrics." Snippet-only; I could not verify the table.
- **[SECOND-HAND]** Salty-peptide benchmark: fusing ESM-2 with traditional descriptors gave a small CV gain but held-out ROC-AUC fell 0.704 → 0.700; authors concluded the information is complementary but "weak relative to fold-to-fold uncertainty in a 580-peptide benchmark." — [SaltyMeta, arXiv 2609.16809](https://arxiv.org/pdf/2609.16809)

### Inferences
- **Direct answer to the user's question.** Three near-zero ablations (ESM-2 650M at −0.010, surface module at −0.011, degree-2 tensor channels at −0.006, seed sd 0.009) is a *normal* result, not an anomaly, and it is the result this literature predicts for a similarity-filtered protein structure task. Two of the three have close published analogues: ESM-2 → HonestAffinity (−0.024 to −0.064 on strict tiers) and AtomSurf (smaller ESM drop on stricter splits); the surface module → AtomSurf's "sequential is underwhelming" and surface-only underperformance. The degree-2 tensor result has **no** published analogue I could find.
- There is **no name to adopt** and no body of work to cite as "the X phenomenon." The report should say this plainly. The nearest defensible framings, in descending order of how well they are supported: (1) HonestAffinity's **split-conditioned reversal**, which has actual numbers and a mechanism; (2) **feature redundancy**, generic and widely used but undefined; (3) **negative transfer**, which is multi-task-specific and does not fit an input-channel ablation.
- The literature's **default response to redundancy is avoidance, not measurement** (GDEGAN choosing ESM-2 over ProstT5 on redundancy grounds, without running either arm). This means the user's measured nulls are more informative than most published positive results, and are a contribution worth writing up — a three-channel null panel with a stated seed sd is rarer in this literature than a new architecture.
- **The parameter-matching problem is systematically unaddressed.** Of everything read: AtomSurf is the only work with a genuinely matched protocol (~200k params, fixed inputs, equal channel widths across hybrid arms). GDEGAN's param column (GotenNet 2.20M, GDEGAN 1.90M, EquiPocket 1.70M) covers trainable parameters only — the ESM arms add a frozen 650M-parameter encoder and the UniRef pre-training corpus behind it, none of which appears in the comparison. So "learned beats hand-built" in GDEGAN and VN-EGNN is substantially "pre-trained-on-far-more-data beats hand-built." DegradoMap holds hyperparameters fixed across arms but removes parameters with each feature group and runs one seed. The report should state that **no paper in this set meets the user's own standard** of parameter-matched, multi-seed feature ablation except AtomSurf, and AtomSurf does not ablate chemical features.

### Gaps
- No paper found where adding **partial charges** does nothing because atom types are already embedded. I searched for this specific pattern and did not find it; it should be reported as not found rather than asserted.
- No paper found measuring that geometric features already encode what a chemical feature was meant to add. GDEGAN *asserts* the converse risk (sequence-structure embedding redundant with geometry) but measures neither direction.
- No analogue found for the degree-2 steerable-tensor null. The equivariant-architecture papers (NequIP, MACE, Equiformer) are the place to look for `L_max` ablations and were not reached within budget.

---

## INPUT/OUTPUT OVERLAP: the model receives seven protein pharmacophore-availability channels and predicts seven complementary ligand-atom classes

### Takeaway
**I found essentially nothing.** The structural-biology literature appears silent on input/output correspondence of this kind, and I could not locate a study of it as either a failure mode or a well-posed auxiliary task. The one on-point source is a 2019 speech-recognition paper that treats input-copying as a real hazard and engineers around it — which at least establishes that practitioners in another field consider the risk live enough to mitigate. This should be reported as a genuine gap, not padded.

### Cited Findings
- **[SECOND-HAND]** The closest analogue found: an INTERSPEECH 2019 end-to-end ASR paper adds an auxiliary branch that **reconstructs the input features** from the shared encoder, and addresses the identity-copy risk by **deliberately distorting the input in the auxiliary branch only** — "swapping the former and latter parts of an utterance, or using a part of an utterance by stripping the beginning or end parts" — arguing these distortions "intentionally suppress long-span dependencies in the time domain," avoiding overfitting. This is a stated design choice, not a measured comparison of with/without distortion. — [Multi-task CTC training with auxiliary feature reconstruction, IBM Research](https://research.ibm.com/publications/multi-task-ctc-training-with-auxiliary-feature-reconstruction-for-end-to-end-speech-recognition)
- **[SECOND-HAND]** On why identity shortcuts are a special case at all: networks "are designed to learn general input-output associations, and in many of the tasks they are applied to there is no identity function"; even with matched dimensionality, input and output "remain distinct spaces and there need not be a single, unambiguous identity function." Background, not analysis of auxiliary shortcuts. — [Generalisation in Neural Networks Does not Require Feature Overlap, arXiv 2107.06872](https://arxiv.org/pdf/2107.06872)
- **[SECOND-HAND]** Auxiliary tasks as regularisation/implicit augmentation is the standard favourable framing: in a clinical multi-task model, "the auxiliary tasks can be viewed as a regularization method as well as implicit data augmentation." — [BMC Med Inform Decis Mak 18 (2018)](https://bmcmedinformdecismak.biomedcentral.com/track/pdf/10.1186/s12911-018-0676-9)
- **[SECOND-HAND]** The corresponding unfavourable framing: auxiliary learning "may also lead to the propagation of biases inherent in the auxiliary tasks." Retrieved from a dataset page, which is a weak source; I would not cite it without a better origin.

### Inferences
- The user's framing is the right one and the literature does not resolve it. Because a protein donor implies a ligand acceptor, the input→output map for the seven classes is close to a **fixed permutation plus a geometric availability mask**. That makes a near-identity solution reachable, which the ASR precedent says is worth defending against.
- The cheapest diagnostic, and it needs no literature: **train the hotspot head on the pharmacophore input alone, with the geometric channels removed or randomised.** Whatever AUROC/AP that degenerate model reaches is the echo floor. If the full model's hotspot metric sits near that floor, the head is echoing; if it sits well above, the head is doing geometry. This directly measures what the literature does not, and mirrors the input-distortion logic of the ASR paper.
- A second, complementary check from the same logic: permute the seven input channels at evaluation and see whether the seven output channels permute with them. A near-perfect permutation response is strong evidence of echo.
- Because the primary metric is a *different* metric (binding-site ranking, not hotspot AP), the overlap is more likely a **dilution** risk than a correctness risk — see the next section, where GDEGAN gives the relevant measured pattern.

### Gaps
- No structural-model study of input/output channel correspondence found, in either direction. I searched for shortcut/echo framings and for auxiliary-reconstruction framings and found no protein-structure instance.
- No self-distillation literature found that addresses the case where the *input* rather than a teacher's output supplies the target. The self-distillation framing may simply be the wrong lens; the ASR input-reconstruction framing fits better.

---

## MULTI-TASK DILUTION: ten output heads, judged on one metric

### Takeaway
GDEGAN provides the one well-seeded, structure-task measurement of an auxiliary head's effect on the primary metric, and the pattern is clear and directly relevant: **the auxiliary loss helps a lot when the main input channel is weak (+0.031 DCC), helps marginally when it is strong (+0.013), and vanishes or goes slightly negative in the best configuration (+0.008 against a 0.008 std dev; −0.004 on one metric).** For gradient-surgery methods, nothing I could verify was measured on a protein structure task; the molecular evidence is mixed and the clearest PCGrad-vs-GradNorm comparison is in materials science, not biology.

### Cited Findings
- **[PRIMARY]** GDEGAN formulates binding-site prediction explicitly "as a multi-task learning problem," with Dice loss on node-level binding probability plus an **Auxiliary Directional Loss (ADL)** that extracts direction from the `l = 1` steerable features and supervises it against ground-truth ligand directions. — [GDEGAN, arXiv 2603.19817](https://arxiv.org/pdf/2603.19817)
- **[PRIMARY]** The ADL effect shrinks as the primary input strengthens (COACH420 DCC, from Table 2 above): weak input, GotenNet 0.454(0.007) → +ADL 0.485(0.004), **Δ +0.031** (~4–8× sd); strong input, GotenNet+ESM 0.543(0.008) → GotenNet(full) 0.556(0.002), **Δ +0.013**; strongest, GDEGAN+ESM 0.572(0.001) → GDEGAN(full) 0.580(0.008), **Δ +0.008 — equal to the arm's own std dev, i.e. not an effect by the user's criterion.** — [GDEGAN, arXiv 2603.19817](https://arxiv.org/pdf/2603.19817)
- **[PRIMARY]** A measured **sign reversal** from the same auxiliary loss: HOLO4K DCA, GotenNet+ESM 0.753(0.004) → GotenNet(full, +ADL) **0.749(0.006)** — the auxiliary head costs 0.004 on that metric, within the std dev. The paper's own headline is "+2% DCC and 3.5% DCA averaged across datasets," which averages over this reversal. — [GDEGAN, arXiv 2603.19817](https://arxiv.org/pdf/2603.19817)
- **[PRIMARY]** DegradoMap's module ablation is a second structure-task case of an added head contributing nothing to one split while helping another (Table 7, identical default hyperparameters LR=1e-3 across arms): SUG only 0.536 / 0.708 / 0.739 (target-unseen / E3-unseen / random); SUG+E3 0.540 / 0.806 / 0.741; Full (A+B+C+D) **0.540** / 0.811 / 0.774. Adding modules C and D changes target-unseen AUROC by **0.000** while improving random-split AUROC by 0.033 — with a 3-seed std of ±0.124 on that metric, neither number means anything. — [DegradoMap, arXiv 2606.04021](https://arxiv.org/pdf/2606.04021)
- **[SECOND-HAND]** AIM (NeurIPS 2025 workshop) argues negative transfer in molecular MTL comes from conflicting task gradients, treats PCGrad as a "static heuristic," and reports statistically significant improvements over multi-task baselines on QM9 subsets and targeted-protein-degrader benchmarks, "most pronounced in data-scarce regimes." I did **not** retrieve its numbers. — [AIM, arXiv 2509.25955](https://arxiv.org/pdf/2509.25955)
- **[SECOND-HAND]** MatSciML (Materials Project): "PCGrad provides little performance improvement compared to multi-task learning with additive losses," attributed to the tasks being highly correlated so there are few gradient conflicts to resolve. — [MatSciML, arXiv 2309.05934](https://arxiv.org/pdf/2309.05934)
- **[SECOND-HAND]** The clearest PCGrad-vs-GradNorm head-to-head is **not** on a molecular or protein task: in a metal-alloy property study, PCGrad improved hardness R² by 12.4% (0.761 → 0.855) while LDS+GradNorm "achieves the best overall balance across all tasks," with the authors recommending PCGrad for minority tasks and LDS+GradNorm when balance matters. — [When Does Multi-Task Learning Fail?, arXiv 2512.22740](https://arxiv.org/html/2512.22740v2)
- **[SECOND-HAND]** A 2026 preprint argues gradient-based task-affinity measurement is only meaningful when tasks share enough samples, reporting MoleculeNet at <5% overlap and TDC at 8–14%, which would undermine conflict measurements on those benchmarks. Single preprint; treat as hypothesis. — [arXiv 2512.22740](https://arxiv.org/html/2512.22740v2)
- **[SECOND-HAND]** RCGrad (J. Cheminformatics 2024) extends PCGrad for pre-trained molecular GNNs and reports improvements "up to 7.7% over fine-tuning"; the retrieved appendix table did not name its dataset, so the specific numbers should be checked before citing.
- **[SECOND-HAND]** GradNorm's original results are not molecular. — [GradNorm, PMLR v80](https://proceedings.mlr.press/v80/chen18a.html)

### Inferences
- **The GDEGAN ADL pattern is the single most transferable result for EquiCave.** An auxiliary head on a structure task buys most where the primary representation is impoverished and approximately nothing once it is strong. If EquiCave's main channels are already good, the prior should be that its ten heads are near-neutral on the primary metric — neither the rescue nor the catastrophe. That also means the *cost* of auxiliary heads is mostly opportunity cost (capacity, tuning surface, engineering time), not primary-metric damage.
- **No gradient-surgery method has been measured on a protein structure task** in anything I could verify. PCGrad/GradNorm guidance for EquiCave would be extrapolated from materials science and molecular property prediction. The MatSciML null is the most relevant warning: when the auxiliary tasks are highly *correlated* with the primary one — which seven pharmacophore hotspot classes plus a site score almost certainly are — gradient surgery has few conflicts to fix and should be expected to do nothing. Given EquiCave's seed sd of 0.009, detecting a PCGrad effect would likely require more seeds than it is worth.
- Practical ordering implied by the evidence: measure the simple thing first (does dropping heads change the primary metric beyond ±0.009, across enough seeds?), and only reach for PCGrad/GradNorm/uncertainty weighting if a conflict is actually demonstrated. No paper read supports reaching for them pre-emptively on a structure task.

### Gaps
- No measured case of PCGrad, GradNorm or uncertainty weighting on a protein **structure** task (binding site, pocket property, hotspot field). This is a clean gap.
- AIM's and RCGrad's actual numbers were not retrieved; both should be read before citing. AIM's "most pronounced in data-scarce regimes" claim is the one most relevant to EquiCave's 1,100–25,000 structures and is currently unverified.
- No loss-weighting guidance specific to many-head hotspot-field models was found.

---

## FEATURE SATURATION AND DATA SIZE: does a richer input need more data to pay off?

### Takeaway
The direction of the effect is well attested — fixed/hand-built featurisation wins at small sample sizes, learned representations overtake as data grows — but every reported crossover point is benchmark-specific and all of the quantitative evidence I could find is from **molecular** property prediction, not protein structure. The one protein-structure-adjacent data point (DegradoMap, 3,260 entries) shows the small-data failure mode but has variance too large to support any conclusion.

### Cited Findings
- **[SECOND-HAND]** The only numeric crossover found: on ChEMBL20, Chemprop was outperformed by a fingerprint baseline below ~**1024** datapoints and became competitive/better above it; the authors note GNNs "may overfit in low-resource settings" while fingerprint methods have fewer parameters and engineered features. Explicitly a single-benchmark figure. — [Making GNNs Worth It for Low-Data Molecular ML, arXiv 2011.12203](https://arxiv.org/pdf/2011.12203)
- **[SECOND-HAND]** Yang et al. (2019): their learned model matched or outperformed fixed-descriptor models overall, but "struggled on low-label targets compared with the baseline, which relied on human-engineered descriptors." — [arXiv 1904.01561](https://arxiv.org/pdf/1904.01561)
- **[SECOND-HAND]** Representation value plateaus at a task-dependent scale: in a VAE study, reconstruction accuracy kept rising from 5K to 250K source molecules while **predictive** performance plateaued around **25K**; GRALE by contrast reports downstream gains continuing past 1M pre-training samples. — [Improving VAE based molecular representations, arXiv 2201.04929](https://arxiv.org/pdf/2201.04929); [GRALE, arXiv 2505.22109](https://arxiv.org/pdf/2505.22109)
- **[PRIMARY]** DegradoMap, 3,260 labeled entries, 28-dim node features: removing four of six hand-built feature groups *improved* target-unseen AUROC (lysine indicator +0.101, disorder +0.088, physicochemical +0.074, pLDDT +0.051; "only AA one-hot" +0.055), with the authors concluding amino-acid identity alone carries substantial signal and "additional features introduce noise in this small-data regime." **But**: single run per arm against a 3-seed std of ±0.124 and a 6-seed std of ±0.097, so this supports the *narrative* and none of the numbers. — [DegradoMap, arXiv 2606.04021](https://arxiv.org/pdf/2606.04021)
- **[PRIMARY-FETCH]** AtomSurf's variance scales with dataset size, which matters for how EquiCave should budget seeds: "standard error is higher for smaller datasets (MSP, AbAg)," with MSP at ±1 against PIP at ±0.088 over three replicates. — [AtomSurf, arXiv 2309.16519v4](https://arxiv.org/html/2309.16519v4)
- **[SECOND-HAND]** SaltyMeta makes the variance point explicitly for small benchmarks: the ESM-2 + descriptor fusion gain was "weak relative to fold-to-fold uncertainty in a 580-peptide benchmark." — [arXiv 2609.16809](https://arxiv.org/pdf/2609.16809)

### Inferences
- EquiCave's 1,100–25,000 structures straddles the regime where this literature predicts hand-built chemistry is *competitive or superior* to additional learned capacity. At the 1,100 end it sits above Pappu's ~1024 crossover but barely; at 25,000 it is around the scale where the VAE study saw downstream performance plateau. The honest expectation for adding more chemistry at this scale is **no measurable gain**, which is exactly what the user observed three times.
- Combining this with the variance findings gives the most important practical conclusion in the brief: at EquiCave's data scale, with a seed sd of 0.009, the *detectable* effect size is roughly ±0.02 at a few seeds. Most published feature-ablation deltas in this literature are smaller than their own seed spread. **The binding constraint on EquiCave's feature decisions is statistical power, not feature design** — and the cheapest way to make the next "does channel X help?" question answerable is more seeds, not more channels.

### Gaps
- No protein-structure study found that sweeps training-set size against feature richness. The data-size evidence is entirely from molecular property prediction and does not transfer cleanly to binding-site or hotspot tasks.
- No study found that reports a saturation point for *hand-built* feature count (as opposed to learned-representation pre-training scale) on any protein task.
- Pafnucy, DeepPocket and RTMScore were not reached; no claim should be made about their feature-ablation or data-scaling behaviour.

---

## Cross-cutting methodological notes for the report writer

### Takeaway
Judged by the user's own standard — parameter-matched arms, seed counts reported, effects larger than seed spread — almost none of this literature qualifies, and the two papers that come closest are the ones whose conclusions most support the user's observations.

### Cited Findings
- **[PRIMARY-FETCH]** **Only AtomSurf is genuinely parameter-matched**: ~200k parameters with fixed input features under the ATOM3D protocol "with the aim of a fair comparison," equal channel widths for surface and graph encoders in hybrid arms. It reports 3 replicates as **standard error**, not std dev, runs no replicates for competitors, and states that "no formal statistical tests were done." — [AtomSurf, arXiv 2309.16519v4](https://arxiv.org/html/2309.16519v4)
- **[PRIMARY]** **GDEGAN reports std devs on every ablation cell** (the strongest variance reporting in the set) but does **not** state its seed count, and its parameter column (GotenNet 2.20M, GDEGAN 1.90M, EquiPocket 1.70M) counts trainable parameters only — the frozen ESM-2 650M encoder and its pre-training corpus are outside the comparison entirely. Training: 4 layers, hidden dim 128, `L_max = 2`, 8 attention heads, 100 epochs, AdamW. — [GDEGAN, arXiv 2603.19817](https://arxiv.org/pdf/2603.19817)
- **[PRIMARY-FETCH]** **VN-EGNN** reports std dev "across training re-runs" but **does not state the number of re-runs**, and its EGNN baseline row is quoted from Zhang et al. 2023b rather than re-run by the authors — so the headline no-virtual-node comparison is not a controlled arm. — [VN-EGNN, arXiv 2404.07194](https://ar5iv.labs.arxiv.org/html/2404.07194)
- **[PRIMARY]** **HonestAffinity** is the best-designed ablation in the set: three seeds for every variant *and* every reproduced baseline, mean±std throughout, identical schedule/data/architecture template with only the protein token representation swapped, paired bootstrap checks on selected comparisons, and explicit instruction to read small margins descriptively. — [HonestAffinity, arXiv 2606.03422](https://arxiv.org/pdf/2606.03422)
- **[PRIMARY]** **DegradoMap** is the cautionary example: hyperparameters held fixed across ablation arms (good), but one run per arm against a ±0.124 3-seed std (fatal), and ablation arms performed at a non-tuned learning rate so the ablation baseline (0.540 target-unseen) differs from the headline result (0.657) — the authors flag this themselves. — [DegradoMap, arXiv 2606.04021](https://arxiv.org/pdf/2606.04021)

### Inferences
- **Removing a feature removes parameters**, and no paper in this set controls for it in a feature ablation. AtomSurf controls parameters but ablates representations and modules, not chemical features. So every "feature X contributes Δ" number quotable in this report confounds the feature with the capacity it carried. For EquiCave this cuts both ways: a *null* result (−0.010) is more trustworthy than a positive one, because the null survives despite also losing parameters.
- The frozen-PLM arms in GDEGAN, VN-EGNN and HonestAffinity are not "learned" in the sense the user means. A report section on hand-built vs. learned should distinguish three things the literature conflates: hand-built descriptors, embeddings learned end to end on the task (1.1k–25k structures), and representations transferred from a corpus many orders of magnitude larger. Only the second is a realistic alternative at EquiCave's scale, and **the one paper that tests it (GDEGAN's atomic-number embedding) finds it loses to transfer and the comparison to hand-built chemistry is never run.**

### Gaps
- Several key sources in this brief are 2026 arXiv preprints (GDEGAN 2603.19817, HonestAffinity 2606.03422, DegradoMap 2606.04021) whose peer-review status I could not establish. DegradoMap lists ACM-BCB '26 (June 30 – July 3, 2026, Calabria). GDEGAN's formatting suggests an ICML-style submission. These should be labelled as preprints in the report.
- Papers in the suggested-source list that I could not reach within the tool budget and about which **nothing** should be claimed: PaiNN, GVP, NequIP, MACE, Equiformer, GotenNet (as primary sources), DeepPocket, Pafnucy, RTMScore, GrASP's ablation, and EquiPocket's own text.
