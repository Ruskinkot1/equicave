# Training Side of Ligand Binding-Site Prediction: Datasets, Scale, Losses, Optimisation (as of 2026)

Scope note on verification. Claims below are marked **[PRIMARY]** when I read the number in the
paper's own PDF/HTML (I extracted full text from the PDFs of GDEGAN, EquiPocket, VN-EGNN, DeepPocket,
DeepSurf and the TMLR equivariance-scaling paper with `pdftotext`), and **[SECOND-HAND]** when the
number came only through a search summary or a fetch summariser rather than my own reading of the
primary text. Where a paper is silent I write "not stated".

---

## TRAINING SETS: which set, which release, exact structure counts, and overlap handling

### Takeaway

Almost every accurate structure-based method trains on **scPDB 2017**, and — crucially — after
de-duplication they all collapse it to roughly **5,000–8,600 structures**, which is exactly the
regime the EquiCave model is already in. The two methods that broke out of that regime did so by
*re-mining the PDB* rather than by using a bigger slice of scPDB: GrASP (16,889 structures /
26,196 sites), the PoSSuM-trained graph transformer (22,599 entries) and UniSite-DS (11,510 UniProt
IDs). Overlap handling is wildly inconsistent: DeepPocket removes test homologues at 50%/30%
sequence identity, the graph transformer at 50%, UniSite only at 90% MMseqs2 similarity, and
**EquiPocket / VN-EGNN / GDEGAN do not remove test-set homologues at all** — they cluster only
*within* scPDB by UniProt ID.

### Cited Findings — the scPDB 2017 release itself

- scPDB 2017 release: **17,594 structures, 16,034 entries, 4,782 proteins, 6,326 ligands**. This
  exact quadruple is quoted identically by EquiPocket, VN-EGNN and GDEGAN. **[PRIMARY]** —
  [EquiPocket arXiv:2302.12177](https://arxiv.org/pdf/2302.12177v1);
  [VN-EGNN App. E.1, arXiv:2404.07194](https://arxiv.org/abs/2404.07194);
  [GDEGAN App. D.1, arXiv:2603.19817](https://arxiv.org/abs/2603.19817)
- **Conflict to flag:** DeepPocket describes the *same* scPDB v.2017 as "17 594 binding sites, which
  corresponds to **16 612 proteins and 5 540 UniProt IDs**". The "16,612 proteins" is inconsistent
  with the "4,782 proteins / 16,034 entries" figure the other three papers use; the papers are
  probably counting different objects (structures vs. entries vs. unique sequences) but none
  reconciles it. **[PRIMARY]** —
  [DeepPocket, J. Chem. Inf. Model. 10.1021/acs.jcim.1c00799](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2022_01/Deeppocket.pdf)
- EquiPocket's per-dataset size table (average atom counts), which is the basis of its
  distribution-shift argument: scPDB 4,205 atoms / 2,317 surface atoms / 24,010 surface points /
  **47 target (positive) atoms**; COACH420 2,123 / 1,217 / 12,325 / 58; HOLO4k 3,845 / 2,052 /
  20,023 / 106; PDBbind 3,104 / 1,677 / 17,357 / 37. **[PRIMARY]** —
  [EquiPocket Table 1](https://arxiv.org/pdf/2302.12177v1)

### Cited Findings — what each method actually trains on

| Method | Set | Count actually trained on | Overlap with test benchmarks |
|---|---|---|---|
| EquiPocket | scPDB 2017 | **5,372 structures** | UniProt clustering *within* scPDB only; test homologues not removed |
| VN-EGNN | scPDB 2017 (via PUResNet's `scpdb_subset.zip`) | same clustering recipe; count not stated | same as EquiPocket; no test-homologue removal stated |
| GDEGAN | scPDB 2017, EquiPocket preprocessing, 90:10 split | count not stated (EquiPocket pipeline ⇒ ~5,372) | inherits EquiPocket settings; no test-homologue removal stated |
| PUResNet | scPDB 2017 | **5,020 structures** of 16,034 | UniProt clusters + Tanimoto ≥ 80% dedup |
| DeepPocket | scPDB 2017, 10-fold CV by UniProt | 518,460 candidate pockets; removes 2,418–7,951 structures per benchmark | **explicit**: seq-id > 50%, OR ligand similarity > 0.9 AND seq-id > 30% |
| DeepSurf | scPDB 2017, 5-fold CV | not stated (full scPDB, undersampled) | reports homologous / non-homologous test subsets at 40% global seq-id |
| GrASP | re-curated scPDB v2017 | **16,889 structures / 26,196 binding sites** | 10-fold CV by UniProt + binding-site similarity; external-benchmark overlap not detailed |
| Graph transformer (PLOS ONE 2024) | scPDB **and** PoSSuM | scPDB 8,537 entries; PoSSuM 22,599 entries | seq-id > 50% vs. test removed, both sets |
| UniSite-3D | UniSite-DS (new) | 11,510 UniProt IDs | MMseqs2 similarity > 0.9 removed (loose) |

- **EquiPocket: 5,372 structures.** "i) Cluster the structures in scPDB by their Uniprot IDs, and
  select the longest sequenced protein structures from every cluster as the train data. Finally,
  **5,372 structures** are selected out." **[PRIMARY]** —
  [EquiPocket §5.1.3](https://arxiv.org/pdf/2302.12177v1)
- VN-EGNN uses the *same* recipe and takes the subset file directly from the PUResNet repo:
  "Structures were clustered based on their Uniprot IDs. From each cluster, protein structures with
  the longest sequences were selected... (Source:
  `https://github.com/jivankandel/PUResNet/blob/main/scpdb_subset.zip`)". It does **not** restate the
  resulting count. **[PRIMARY]** — [VN-EGNN App. E.1](https://arxiv.org/abs/2404.07194)
- GDEGAN: "We used the most frequently used dataset for LBS identification ... for training and
  validation, with a split of **90 : 10**. Final dataset was preprocessed using the steps described
  in EquiPocket." GDEGAN never states its own training structure count. **[PRIMARY]** —
  [GDEGAN App. D.1](https://arxiv.org/abs/2603.19817)
- PUResNet: "Among 16034 protein structures present in scPDB, **5020 structures were selected**",
  by grouping on UniProt ID then keeping the longest sequence per cluster with a Tanimoto ≥ 80%
  similarity criterion. **[SECOND-HAND]** —
  [PUResNet, J. Cheminform. 2021](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00547-7)
- GrASP deliberately enlarged the set because the model is deeper: "As a deeper model, GrASP
  requires a larger dataset for training ... a new publicly available version of the sc-PDB database
  containing **26,196 binding sites across 16,889 protein structures**" (≈ 9,000 ligands added over
  scPDB's original annotation), with 10-fold CV splits preventing leakage on "UniProt IDs as well as
  binding site similarity". **[SECOND-HAND]** —
  [GrASP, PMC10402091](https://pmc.ncbi.nlm.nih.gov/articles/PMC10402091/)
- DeepPocket's leakage protocol is the strictest I found and quantifies what it costs: "we removed
  all proteins from the training set that had either sequence identity greater than 50% or ligand
  similarity greater than 0.9 and sequence identity greater than 30% to any of the structures in the
  test set ... This resulted in the removal of **2418 structures for COACH420, 7951 structures for
  HOLO4k, 6285 structures for SC6K, and 5801 structures for the Refined set** from the scPDB
  training set." **[PRIMARY]** — [DeepPocket Methods](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2022_01/Deeppocket.pdf)
- VN-EGNN itself flags the comparability problem in its main results table: P2Rank and DeepPocket
  rows are footnoted "Uses different training set and, thus, limited comparability." **[PRIMARY]** —
  [VN-EGNN Table 1](https://arxiv.org/abs/2404.07194)
- UniSite-DS construction: started from **143,197 protein–ligand interaction entries**, resolution
  filter ≤ 2.5 Å, ligands < 5 atoms removed, NMS with IoM 0.7 / IoU 0.5, sequence-length threshold
  800 residues, manual inspection of proteins with > 10 sites ⇒ **11,510 valid UniProt IDs, 3,670
  with multiple binding sites**; "4.81 times more multi-site data and 2.08 times more overall data
  compared to the previously most widely used datasets". Train/test separated with MMseqs2 at
  "sequence similarity above 0.9". **[SECOND-HAND]** —
  [UniSite, arXiv:2506.03237 / NeurIPS 2025](https://arxiv.org/html/2506.03237v2)
- PDBbind v2020 as used as a *test* set: general set 14,127 complexes, refined set 5,316 complexes;
  all three equivariant papers test on the refined set. **[PRIMARY]** —
  [EquiPocket §5.1.1](https://arxiv.org/pdf/2302.12177v1); [VN-EGNN App. E.1](https://arxiv.org/abs/2404.07194); [GDEGAN App. D.1](https://arxiv.org/abs/2603.19817)
- COACH420 = 420 protein–ligand complexes; HOLO4K = **4,288 structures** (GDEGAN's count); both
  evaluated on the `mlig` subsets from the P2Rank datasets repo. **[PRIMARY]** —
  [GDEGAN App. D.1](https://arxiv.org/abs/2603.19817); source repo `https://github.com/rdk/p2rank-datasets`
- BioLiP/BioLiP2 is far larger than anything used for training in this literature: the weekly-updated
  BioLiP (BioLiP2 since Feb 2023) reports **989,724 total entries** and **512,101 protein receptors**,
  split as 501,072 regular-ligand, 213,844 metal, 41,570 peptide, 50,999 DNA, 182,239 RNA entries.
  **[SECOND-HAND]** — [BioLiP, Zhang Group](https://www.zhanggroup.org/BioLiP/);
  [BioLiP2 paper, NAR 2023](https://citedrive.com/en/discovery/biolip2-an-updated-structure-database-for-biologically-relevant-ligandprotein-interactions/)

### Inferences

- The "accurate equivariant" family (EquiPocket → VN-EGNN → GDEGAN) is **all trained on roughly the
  same ~5.4k structures**, so their large headline gains over each other are *architecture and
  features*, not data. A model trained on 1,100–8,600 structures is therefore not at a data
  disadvantage relative to the published SOTA on COACH420/HOLO4k/PDBbind — it is at parity.
- The three groups that enlarged the data (GrASP, PoSSuM graph transformer, UniSite) all did so by
  re-mining the PDB with their own ligand-relevance filters. If EquiCave wants >10k structures,
  BioLiP2 or a GrASP/PoSSuM-style re-mining is the realistic route; scPDB simply does not contain
  more non-redundant protein targets.
- The leakage asymmetry is a *reporting* hazard rather than a modelling one: DeepPocket drops 7,951
  of its scPDB training structures to evaluate HOLO4k cleanly, while EquiPocket/VN-EGNN/GDEGAN drop
  none. Any paired comparison of EquiCave against those three numbers on HOLO4k is comparing a
  (potentially) leak-free model against leaky baselines.

### Gaps

- VN-EGNN and GDEGAN never state their own final training structure count — only the pipeline.
  "Not stated."
- No paper I read trains on **ChEN** or on **holo4k training splits**; holo4k is used purely as a
  test set in this literature, and I found no "CASF-trained" site predictor. I found no evidence
  these are used as training sets at all, so I cannot report counts for them.
- GrASP's loss function, optimiser, learning rate, epochs and early stopping were **not recoverable**
  from the PMC HTML I fetched (the summariser reported them as "not specified"). I did not read the
  GrASP SI.
- The 16,612-vs-4,782 "proteins" discrepancy between DeepPocket and the other papers is unresolved.

---

## SCALING: is accuracy known to scale with the number of training structures? (priority question)

### Takeaway

**Yes — exactly one clean, published, same-architecture data-scaling experiment exists for binding-site
prediction, and its answer is "diminishing".** Going from 8,537 to 22,599 training entries (2.6×)
bought **+0.026 combined test PR-AUC** (0.7640 → 0.7896), while *noise augmentation alone on the
smaller set* bought **+0.098 PR-AUC** (0.6139 → 0.7115) — roughly **four times** the gain of 2.6× the
data. Scaling the model 1.28M → 7.34M parameters on the larger set bought only +0.014. There are **no
published learning curves** (accuracy vs. N) for any of the equivariant site predictors.

### Cited Findings — the one real data-scaling experiment

Shiota et al., "Protein ligand binding site prediction using graph transformer neural network",
PLOS ONE 2024. Same graph-transformer architecture trained on two differently sized corpora, with
identical >50%-sequence-identity-to-test filtering, 5-fold CV, and the same test set
(COACH420 ∪ HOLO4k).

- Dataset sizes: scPDB **8,537 train/valid entries → 276,531 pocket candidates, 7.1% positive**;
  PoSSuM **22,599 train/valid entries → 729,853 pocket candidates, 6.3% positive** (PoSSuM built by
  extracting PoSSuM entries whose ligands appear in scPDB, 37,067 PDB entries before redundancy
  removal; ≈ 2.6× the processed scPDB). **[SECOND-HAND, from fetch of the primary article]** —
  [PLOS ONE 10.1371/journal.pone.0308425](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0308425)
- Table 2, test-set PR-AUC / ROC-AUC (combined column in bold text below): **[SECOND-HAND, from two
  independent fetches of the primary article which agreed]** —
  [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0308425);
  [PMC11302905](https://pmc.ncbi.nlm.nih.gov/articles/PMC11302905/)

  | Config | Coach420 PR-AUC | Holo4K PR-AUC | **Combined PR-AUC** | Coach420 ROC-AUC | Holo4K ROC-AUC |
  |---|---|---|---|---|---|
  | Unbal (scPDB) | 0.5808 | 0.6172 | **0.6127** | 0.9036 | 0.9095 |
  | Bal (class balancing) | 0.5915 | 0.6176 | **0.6139** | 0.9028 | 0.9079 |
  | Bal+aug (noise aug.) | 0.6907 | 0.7158 | **0.7115** | 0.9342 | 0.9305 |
  | Bal+aug+SA (+SASA feats) | 0.7236 | 0.7691 | **0.7640** | 0.9450 | 0.9424 |
  | PoSSuM/M (2.6× data, same model) | 0.7632 | 0.7933 | **0.7896** | 0.9543 | 0.9500 |
  | PoSSuM/L (2.6× data, 5.7× model) | 0.7816 | 0.8067 | **0.8035** | 0.9531 | 0.9489 |

- Derived deltas (my arithmetic on the table above): **class balancing alone +0.0012**; **noise
  augmentation +0.0976**; **SASA features +0.0525**; **2.6× training data +0.0256**; **5.7× model
  size on top of that +0.0139**.
- The authors' own reading of the model-size row: "The capacity of the baseline model is likely
  sufficiently large even for the PoSSuM dataset, and thus increasing the model size did not
  significantly improve performance." **[SECOND-HAND]** —
  [PMC11302905](https://pmc.ncbi.nlm.nih.gov/articles/PMC11302905/)
- The authors do **not** claim saturation on the data axis — they note "larger datasets were shown to
  contribute to better prediction performance" and suggest "inclusion of a wide range of uncurated PDB
  entries" could help further. **[SECOND-HAND]** —
  [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0308425)
- Downstream success rates from the same paper, Table 3 (top-n / top-(n+2)): Fpocket 35.09% / 51.25%
  (COACH420) and 36.34% / 51.53% (HOLO4k); DeepPocket 67.96% / 79.94% and 73.36% / 82.97%; P2Rank
  68.24% / 75.48% and 70.6% / 80.05%; PoSSuM/L **77.67% / 88.14%** and **81.33% / 89.43%**.
  **[SECOND-HAND]** — [PMC11302905](https://pmc.ncbi.nlm.nih.gov/articles/PMC11302905/)

### Cited Findings — the nearest general-ML evidence

- "Does equivariance matter at scale?" (TMLR 07/2025, Brehmer et al., Qualcomm AI Research) trains
  an equivariant transformer (GATr) and a standard transformer on a rigid-body mesh-dynamics task
  (4·10⁵ trajectories × 96 steps, 3–10 objects, mean 5,470 mesh faces) across model size, training
  steps and dataset size. Three conclusions, verbatim from the abstract: "First, equivariance
  improves data efficiency, but **training non-equivariant models with data augmentation can close
  this gap given sufficient epochs**. Second, scaling with compute follows a power law, with
  equivariant models outperforming non-equivariant ones at each tested compute budget. Finally, the
  optimal allocation of a compute budget onto model size and training duration differs between
  equivariant and non-equivariant models." **[PRIMARY]** —
  [arXiv:2410.23179](https://arxiv.org/abs/2410.23179)
- Caveat the report writer should keep: that benchmark is **synthetic rigid-body physics, not
  proteins**, and the authors chose it explicitly because it needs "a large number of training
  samples" and "a low floor and a high ceiling" — the opposite of the 5k-structure protein regime.
  **[PRIMARY]** — [arXiv:2410.23179 §3.1](https://arxiv.org/abs/2410.23179)
- NequIP (Batzner et al., "E(3)-Equivariant Graph Neural Networks for **Data-Efficient** and Accurate
  Interatomic Potentials") is the canonical data-efficiency-from-equivariance result in molecular ML
  and is cited in the equivariance-at-scale related work as one of the force-field cases where
  equivariance improved data efficiency. **[PRIMARY for existence/framing]** —
  [arXiv:2101.03164](https://arxiv.org/abs/2101.03164)

### Inferences

- At the 5k–20k structure scale, the published evidence says **data is the weakest of the available
  levers**: ~2.6× more structures ≈ +0.026 PR-AUC, versus +0.098 for augmentation and +0.053 for a
  single extra input feature family (SASA). For a model at 1,100–8,600 structures, the expected
  return on pushing to ~20k is on the order of a couple of PR-AUC points, not a step change.
- The one confound to state honestly: PoSSuM is not scPDB-scaled-up, it is a *different corpus* with
  different ligand/target composition. So +0.026 bundles "more data" with "somewhat different data",
  and cannot be cleanly attributed to N alone. It is the best available estimate, not a learning curve.
- Combined with the GDEGAN/VN-EGNN ESM ablations (see Transfer section), the ranked levers implied by
  the literature are: **(1) pretrained sequence representations ≫ (2) augmentation ≈ input features
  ≫ (3) 2–3× more structures ≈ auxiliary losses > (4) model size**.

### Gaps

- **No learning curve exists** for any of EquiPocket, VN-EGNN, GDEGAN, GrASP, DeepPocket, DeepSurf,
  Kalasanty or PUResNet. None of them trains the same architecture on subsampled fractions of its
  training set. I searched for this specifically and found nothing; I consider this a genuine hole in
  the literature rather than a search failure.
- UniSite, despite introducing a 2.08×-larger dataset, reports **no ablation on training-set size** —
  so the single largest new dataset in the field was not used to produce a scaling curve.
  **[SECOND-HAND]** — [UniSite](https://arxiv.org/html/2506.03237v2)
- I found no protein-ML scaling study (AlphaFold- or ESM-adjacent) that measures *structure-based
  pocket* accuracy against corpus size. Search results for "learning curves" led overwhelmingly to
  binding-**affinity** prediction, which is a different task; I am not reporting those numbers as
  site-prediction evidence.

---

## LOSSES: what is optimised, and what the ablations say

### Takeaway

The accurate equivariant methods converge on an almost identical recipe: **Dice as the segmentation
loss, plus one auxiliary geometric term, summed with weight 1:1 and no tuning**. The auxiliary term
is a cosine/direction loss (EquiPocket, GDEGAN) or a min-distance centre-regression loss on virtual
nodes (VN-EGNN). Ablations put the directional loss at a real but modest **+2% DCC / +3.5% DCA**,
concentrated on the hardest (HOLO4K) split and on small proteins. Only UniSite uses a true
set-matching (Hungarian) loss, and it is the only method with explicit loss weights.

### Cited Findings — EquiPocket

- Label definition: `y_i = 1` if a **surface atom** is within **4 Å** of any ligand atom.
  **[PRIMARY]** — [EquiPocket §4.4](https://arxiv.org/pdf/2302.12177v1)
- Dice loss: `L_b = 1 − 2·Σ(ŷ_i·y_i) / (Σŷ_i + Σy_i + ε)`, with ε a small stabiliser.
  **[PRIMARY]** — [EquiPocket Eq. 17](https://arxiv.org/pdf/2302.12177v1)
- Direction loss: label `d_i = (m_i − x_i)/‖m_i − x_i‖` where `m_i` is the nearest ligand atom;
  prediction `d̂_i = (x_i^out − x_i)/‖x_i^out − x_i‖`; `L_d = Σ(1 − cos(d̂_i, d_i))`.
  **[PRIMARY]** — [EquiPocket Eqs. 18–19](https://arxiv.org/pdf/2302.12177v1)
- Total: "The eventual loss is **L = L_b + L_d**. We train the parameters of all the three modules
  end to end." No weighting hyperparameter. **[PRIMARY]** —
  [EquiPocket §4.4](https://arxiv.org/pdf/2302.12177v1)
- Direction-loss ablation is **reported only as a figure, broken down by protein size**, not as a
  table: "The result of the EquiPocket (w/o Direction Loss) in Figure 6 demonstrates conclusively
  that the prediction performance of **small proteins with fewer than 3,000 atoms** is diminished in
  the absence of this task." No scalar DCC/DCA delta is given. **[PRIMARY]** —
  [EquiPocket §5.2.2](https://arxiv.org/pdf/2302.12177v1)
- Module ablation (full Table 2, DCC/DCA on COACH420 / HOLO4K / PDBbind2020, failure rate):
  EquiPocket-L 0.552 fail, 0.070/0.171, 0.044/0.138, 0.051/0.132 → EquiPocket-G 0.292 fail,
  0.159/0.373, 0.129/0.411, 0.145/0.311 → EquiPocket-LG 0.220 fail, 0.212/0.443, 0.183/0.502,
  0.274/0.462 → **EquiPocket 0.051 fail, 0.423/0.656, 0.337/0.662, 0.545/0.721**. The surface
  message-passing module is the big win: "DCC and DCA have increased by approximately 20% on
  average, and the failure rate has been significantly reduced." **[PRIMARY]** —
  [EquiPocket Table 2, §5.2.2](https://arxiv.org/pdf/2302.12177v1)
- Centre decoding is itself loss-coupled: EquiPocket converts atom-level predictions to a centre via
  `p̂os_i^L = pos_i + threshold · (center_i − pos_i)/|center_i − pos_i|` with `threshold = 4`,
  "because we label the protein atoms within 4 Å of any ligand atom as positive", then mean-shift
  clusters. Atoms with probability < **T = 0.5** are discarded. **[PRIMARY]** —
  [EquiPocket App. A.2.4, Algorithm](https://arxiv.org/pdf/2302.12177v1)

### Cited Findings — GDEGAN (arXiv:2603.19817, read in full)

- "For binding site identification, a node-level prediction task, we compute `ŷ_i =
  Sigmoid(MLP(h_i^(L)))` and use **Dice Loss** ... **to address inherent class imbalance**."
  **[PRIMARY]** — [GDEGAN §3.5](https://arxiv.org/abs/2603.19817)
- Auxiliary Directional Loss (ADL): predicted direction read off the **l = 1 steerable features**,
  `d̂_i = X̃_i^channel / (‖X̃_i^channel‖₂ + ε)` where `X̃_i^channel = (1/h_d) Σ_k X̃_{i,k}^(1)` averages
  across feature channels; ground truth `d_i^true = (p*_lig − p_i)/‖p*_lig − p_i‖₂` with `p*_lig` the
  **nearest ligand heavy atom**; `L_Dir` is a cosine-similarity loss. **[PRIMARY]** —
  [GDEGAN §3.5](https://arxiv.org/abs/2603.19817)
- Total: "**L = L_Dice + L_Dir**" — again 1:1, no weight stated. **[PRIMARY]** —
  [GDEGAN §3.5](https://arxiv.org/abs/2603.19817)
- **ADL ablation, full Table 2** (DCC/DCA, COACH420 / HOLO4K / PDBbind2020, std in parentheses):
  **[PRIMARY]** — [GDEGAN Table 2](https://arxiv.org/abs/2603.19817)

  | Variant | Equivariance | ADL | ESM | COACH420 DCC/DCA | HOLO4K DCC/DCA | PDBbind2020 DCC/DCA |
  |---|---|---|---|---|---|---|
  | GotenNet | E(3) | No | No | 0.454(.007) / 0.624(.014) | 0.464(.001) / 0.691(.005) | 0.553(.008) / 0.705(.007) |
  | GotenNet+ADL | E(3) | **Yes** | No | 0.485(.004) / 0.642(.011) | 0.468(.004) / 0.732(.004) | 0.592(.010) / 0.748(.003) |
  | GotenNet+ESM | SE(3) | No | Yes | 0.543(.008) / 0.693(.006) | 0.520(.011) / 0.753(.004) | 0.637(.005) / 0.760(.004) |
  | GotenNet(full) | SE(3) | Yes | Yes | 0.556(.002) / 0.703(.005) | 0.529(.011) / 0.749(.006) | 0.649(.005) / 0.801(.014) |
  | GDEGAN+ESM | SE(3) | No | Yes | 0.572(.001) / 0.702(.002) | 0.532(.010) / 0.769(.006) | 0.652(.010) / 0.810(.011) |
  | **GDEGAN(full)** | SE(3) | Yes | Yes | **0.580(.008) / 0.707(.009)** | **0.560(.013) / 0.788(.011)** | **0.675(.010) / 0.826(.011)** |

- The paper's own summary of the ADL effect: adding ADL on top of either attention variant "further
  improves the results by an **average of 2% DCC and 3.5% DCA** respectively averaged across
  datasets." Note the ADL-only gain *without* ESM is larger (GotenNet → GotenNet+ADL: +0.031 DCC on
  COACH420, +0.039 on PDBbind, +0.041 DCA on HOLO4K) than the ADL gain on top of ESM + Gaussian
  attention (GDEGAN+ESM → GDEGAN(full): +0.008 DCC COACH420, +0.028 HOLO4K, +0.023 PDBbind).
  **[PRIMARY]** — [GDEGAN §4.5](https://arxiv.org/abs/2603.19817)
- GDEGAN headline table also records a **failure rate** column: GDEGAN 0.032 vs. GotenNet 0.049 vs.
  EquiPocket 0.051 vs. Kalasanty 0.120 vs. GCN2 0.466. **[PRIMARY]** —
  [GDEGAN Table 1](https://arxiv.org/abs/2603.19817)
- Multi-pocket extraction is thresholded, not learned: candidates are the residues with `ŷ_i > τ`,
  with **τ = 0.5** listed in the hyperparameter table, following P2Rank's procedure. **[PRIMARY]** —
  [GDEGAN §4.2, Table 5](https://arxiv.org/abs/2603.19817)

### Cited Findings — VN-EGNN (set-style centre loss + confidence head)

- Segmentation term can be cross-entropy `L_segm = (1/N) Σ CE(y_n, ŷ_n)` **or** Dice
  `L_dice = 1 − (2 Σ y_n ŷ_n + ε)/(Σ y_n + Σ ŷ_n + ε)` with **ε = 1**; the final model uses Dice.
  **[PRIMARY]** — [VN-EGNN §2.5](https://arxiv.org/abs/2404.07194)
- Binding-site-centre loss is a **min-assignment (not Hungarian)** set loss:
  `L_bsc = (1/M) Σ_{m=1..M} min_{k∈1..K} ‖y_m − ŷ_k‖²`, i.e. each true centre must be captured by at
  least one virtual node; virtual nodes are not forced to be distinct. **[PRIMARY]** —
  [VN-EGNN Eq. 19](https://arxiv.org/abs/2404.07194)
- Total: "**L = L_bsc + L_dice**, in which the two terms could also be balanced against each other
  through a hyperparameter, **which we found was not necessary** though." **[PRIMARY]** —
  [VN-EGNN §2.5](https://arxiv.org/abs/2404.07194)
- **Confidence / self-confidence head** (an AlphaFold-style auxiliary): target
  `c_k = 1 − (1/2γ)·‖y_k − ŷ_k‖` if `‖y_k − ŷ_k‖ ≤ γ`, else `c_0`, with **c_0 = 0.001** and
  **γ = 4** chosen "to align with the commonly accepted threshold value for the DCC/DCA success rates
  of 4 Å"; loss is squared error `L_confidence = (1/K) Σ (c_k − ĉ_k)²`. This score is what ranks the
  predictions. **[PRIMARY]** — [VN-EGNN Eqs. 20–21](https://arxiv.org/abs/2404.07194)
- Coordinate-loss detail worth copying: coordinates are normalised (÷5) and unnormalised (×5), and
  "used **Huber loss** (Huber, 1964) for the coordinates, which **empirically proved to be slightly
  more effective**" than plain squared error; Huber δ searched over {1} only. **[PRIMARY]** —
  [VN-EGNN §3.3, Table E1](https://arxiv.org/abs/2404.07194)
- VN-EGNN component ablation (DCC/DCA): EGNN baseline 0.156/0.361, 0.127/0.406, 0.143/0.302 →
  VN-EGNN (residue emb., VN + heterog. MP, no ESM) 0.503/0.684, 0.438/0.605, 0.551/0.751 →
  VN-EGNN (homog., no ESM) 0.497/0.700, 0.414/0.618, 0.502/0.717 → VN-EGNN (homog. + ESM)
  0.575/0.708, 0.479/0.595, 0.649/0.805 → **VN-EGNN (full) 0.605/0.750, 0.532/0.659, 0.669/0.820**.
  **[PRIMARY]** — [VN-EGNN Table 2](https://arxiv.org/abs/2404.07194)
- The paper attributes its own gain to the loss/output design rather than the architecture: "We
  attribute our improvement largely to the **direct prediction of binding site centers, rather than
  inferring them from the geometric center of segmented areas**, a common practice in earlier
  methods. Relying on segmentation can lead to inaccuracies, especially if a single erroneous
  prediction impacts the cal[culation]..." **[PRIMARY]** —
  [VN-EGNN §4](https://arxiv.org/abs/2404.07194)

### Cited Findings — UniSite (the only Hungarian/bipartite set loss)

- "UniSite is the first end-to-end ligand binding site detection framework supervised by **set
  prediction loss with bijective matching**"; during training the **Hungarian algorithm** finds the
  optimal one-to-one match between predicted and ground-truth sites by minimising a matching cost,
  "which forces the model to learn distinct, non-redundant pockets". **[SECOND-HAND]** —
  [UniSite](https://arxiv.org/html/2506.03237v2)
- Loss composition and **explicit weights**: matching cost = classification loss + mask loss
  (BCE + Dice); **λ_bce = λ_dice = 5.0**; **λ_cls = 2.0 with 10× down-weighting for the negative
  ("no object") class**. Architecture: d_model = 256, 6 encoder / 6 decoder layers, **32 site
  queries**. **[SECOND-HAND]** — [UniSite](https://arxiv.org/html/2506.03237v2)
- UniSite also changes the *metric*, introducing an IoU-based AP. On UniSite-DS AP₀.₅: UniSite-3D
  0.3835, GrASP 0.2848, DeepPocket 0.2334, P2Rank 0.2157. On HOLO4K-sc AP₀.₃: UniSite-3D 0.7091,
  GrASP 0.6668, P2Rank 0.6011. **[SECOND-HAND]** — [UniSite](https://arxiv.org/html/2506.03237v2)

### Cited Findings — CNN-era methods

- DeepPocket splits the problem into a **ranking/rescoring CNN** over Fpocket candidates and a
  **U-Net segmentation CNN** trained only on positives. The actual loss functional forms are not
  written out in the paper text I extracted — the Methods specify labels, sampling, grids and
  optimiser but not the loss expression. **Not stated.** **[PRIMARY, absence]** —
  [DeepPocket Methods](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2022_01/Deeppocket.pdf)
- DeepSurf likewise does not state its loss expression in the text I extracted; it specifies labels
  (surface points within 4 Å of any ligand atom), 50/50 class balancing, L2 regularisation
  (λ = 10⁻⁴), batch norm, 20 epochs, batch 64, Adam at lr 10⁻³, and a **ligandability threshold
  T = 0.9**. **Loss: not stated.** **[PRIMARY]** —
  [DeepSurf arXiv:2002.05643](https://arxiv.org/abs/2002.05643)
- Graph transformer (PoSSuM/scPDB): "Binary cross-entropy with sigmoid activation", and the authors
  note "the loss function ℒ satisfies the E(3) invariance according to the translation and rotation".
  **[SECOND-HAND]** — [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0308425)
- Focal loss is **widely described as standard for this class-imbalanced task**, but I could not
  verify a specific γ/α value in any of the primary site-prediction papers I read. Treat "focal loss
  with γ" as a community practice claim, not a verified setting from GrASP or any of the above.
  **[SECOND-HAND, low confidence]** — [GrASP search summary / PMC10402091](https://pmc.ncbi.nlm.nih.gov/articles/PMC10402091/)

### Inferences

- Every accurate method sums its terms at **weight 1.0 and says so explicitly** (EquiPocket, GDEGAN)
  or says tuning was unnecessary (VN-EGNN). There is no published evidence that loss-weight tuning
  matters for this task. The only method with non-trivial weights is UniSite, and those weights
  (5/5/2 with 10× negative down-weighting) are inherited from the DETR/Mask2Former lineage, not
  derived from a binding-site ablation.
- The directional/cosine auxiliary is the **most consistently validated auxiliary loss** in the
  field: three independent papers use it, two ablate it, and both ablations show it helps. But both
  ablations also show the gain is *conditional* — EquiPocket: only proteins < 3,000 atoms; GDEGAN:
  largest on HOLO4K and when ESM features are absent. Expect it to shrink if you already have strong
  sequence features.
- VN-EGNN's `min_k` centre loss and UniSite's Hungarian loss are the two set-level designs, and they
  differ in a way that matters: `min_k` permits several virtual nodes to collapse onto the same site
  (VN-EGNN then mean-shift-clusters them at inference), whereas Hungarian matching penalises that
  directly. If EquiCave predicts a fixed number of sites, UniSite's formulation is the stricter and
  better-motivated choice; VN-EGNN's is the one with published DCC numbers.

### Gaps

- **Tversky loss**: I found no binding-site paper using it. No evidence either way.
- **Listwise / pairwise ranking losses**: no paper in this set optimises a ranking loss. DeepPocket
  and P2Rank *rank* pockets but via a pointwise classifier score; VN-EGNN ranks via the confidence
  head's squared loss. I found no published ablation of pointwise vs. pairwise vs. listwise ranking
  for pocket ranking.
- **IoU-prediction heads**: UniSite introduces IoU-based *evaluation* but I did not verify an
  IoU-*prediction* head. VN-EGNN's confidence head is distance-calibrated, not IoU-calibrated.
- Exact focal-loss parameters anywhere in the site-prediction literature: not found.

---

## CLASS IMBALANCE: how skewed is it, and what is done

### Takeaway

Positives are **4–7% of candidates and roughly 2% of surface atoms**, and the field splits into two
camps: the **CNN/candidate-based methods resample to 50/50**, while the **graph/equivariant methods
do not resample at all and rely entirely on Dice loss** to absorb the imbalance. Nobody has published
a head-to-head of resampling vs. Dice vs. focal on this task, and the one paper that measured
class-balancing in isolation found it **almost worthless** (+0.0012 PR-AUC).

### Cited Findings

- **Measured imbalance, residue level:** GDEGAN's validation set is **205,791 residues = 11,107
  binding + 194,684 non-binding**, i.e. **5.4% positive**. **[PRIMARY]** —
  [GDEGAN App. F](https://arxiv.org/abs/2603.19817)
- **Measured imbalance, pocket-candidate level:** DeepPocket had "a total of **5 18 460** data points
  across the cross-validation set, out of which, **22 030 (4.25%) were positive** and the rest
  negative." **[PRIMARY]** — [DeepPocket Methods](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2022_01/Deeppocket.pdf)
- Graph transformer: **7.1% positive** among 276,531 scPDB pocket candidates; **6.3% positive** among
  729,853 PoSSuM candidates. **[SECOND-HAND]** —
  [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0308425)
- **Atom level (inferred from EquiPocket Table 1):** scPDB averages 2,317 surface atoms and 47 target
  atoms per structure ⇒ **≈ 2.0% positive surface atoms**. *(My arithmetic on primary numbers.)* —
  [EquiPocket Table 1](https://arxiv.org/pdf/2302.12177v1)
- **DeepSurf — explicit undersampling to 50/50:** "the resulting dataset would be quite imbalanced,
  since the non-binding samples outnumber by far the binding ones. The class imbalance problem is a
  well-known problem in machine learning ... The most common tactic lies on the data level and
  consists of either undersampling the main class or oversampling the secondary one. **Due to the
  required time efficiency during training, the former technique was herein followed.** For each
  protein, from the set of non-binding samples a number equal to the binding samples was randomly
  chosen in order to obtain a **50/50 balance** between the two classes." **[PRIMARY]** —
  [DeepSurf §4](https://arxiv.org/abs/2002.05643)
- **DeepPocket — oversampling + receptor stratification:** "This data imbalance is handled by
  **oversampling the positive examples** such that each batch contains equal amounts of positive and
  negative samples while training the model... Each iteration contained a batch of **50 samples** that
  contained an equal number of positive and negative samples through oversampling. Furthermore, we
  used the **stratify option in Libmolgrid to ensure equivalent sampling of all the receptors**
  during training." **[PRIMARY]** — [DeepPocket Methods](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2022_01/Deeppocket.pdf)
- **Dice-only camp:** EquiPocket, VN-EGNN and GDEGAN describe no resampling, reweighting or focal
  term. GDEGAN states the Dice loss is chosen specifically "to address inherent class imbalance".
  **[PRIMARY]** — [GDEGAN §3.5](https://arxiv.org/abs/2603.19817);
  [EquiPocket §4.4](https://arxiv.org/pdf/2302.12177v1); [VN-EGNN §2.5](https://arxiv.org/abs/2404.07194)
- **The one isolated measurement of class balancing:** in the graph transformer ablation, Unbal →
  Bal moved combined test PR-AUC from **0.6127 to 0.6139 (+0.0012)** and actually *lowered* COACH420
  ROC-AUC (0.9036 → 0.9028). **[SECOND-HAND]** —
  [PMC11302905](https://pmc.ncbi.nlm.nih.gov/articles/PMC11302905/)

### Inferences

- At 2–7% positives the imbalance is mild by segmentation standards, which is consistent with the
  measured near-zero benefit of explicit rebalancing. The evidence favours **Dice (or Dice + an
  auxiliary) over resampling**, and gives no reason to spend effort on focal/Tversky tuning.
- Note a protocol asymmetry: the resampling methods (DeepSurf, DeepPocket) resample *candidate
  points/pockets*, while the Dice methods score *every* surface atom or residue. These are not the
  same imbalance, so "50/50" and "Dice" are not strictly alternative answers to one question.

### Gaps

- No paper measures a Dice-vs-weighted-CE-vs-focal comparison on binding sites. "Not stated"
  everywhere.
- GrASP's class-imbalance treatment and its positive-atom fraction were **not reported** in the text
  I could access. It does use "a sigmoid function on the distance between the ligand and protein
  atom" for soft target scores, which is a label-smoothing-like alternative to hard 4 Å labels, but
  no accompanying imbalance statistic. **[SECOND-HAND]** —
  [GrASP, PMC10402091](https://pmc.ncbi.nlm.nih.gov/articles/PMC10402091/)

---

## AUGMENTATION: what helps, and the rotation-vs-equivariance question

### Takeaway

Augmentation is, by the numbers, **the single best-evidenced lever in this literature**: coordinate
noise + node dropout + feature noise bought **+0.098 PR-AUC** on a fixed 8,537-structure set — about
four times the gain from 2.6× more data. Rotation augmentation is used by the CNN methods
(DeepPocket) and avoided by the surface-aligned ones (DeepSurf). On the specific question
"rotation-augmented non-equivariant vs. equivariant at equal data", **there is no protein
binding-site experiment** — the only direct evidence is a synthetic rigid-body study (TMLR 2025)
which finds augmentation *largely closes* the data-efficiency gap given enough epochs, while
equivariance still wins per unit compute.

### Cited Findings

- **Noise augmentation, the big win.** Graph transformer: positional noise **σ_pos = 0.5** (Gaussian
  on atom coordinates), node dropping **σ_node = 0.03** (random node removal/duplication), SASA noise
  **σ_SASA = 0.3** (multiplicative Gaussian on surface-area values). Effect, combined test PR-AUC:
  Bal 0.6139 → Bal+aug **0.7115 (+0.0976)**. **[SECOND-HAND]** —
  [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0308425);
  [PMC11302905](https://pmc.ncbi.nlm.nih.gov/articles/PMC11302905/)
- **DeepPocket uses rotation (and translation):** the classification CNN was trained "for 200 000
  iterations, with **rotational augmentation**"; the segmentation CNN "for 200 epochs with
  **rotational and translational augmentation** using the same optimizers and hyperparameters as the
  classification model." **[PRIMARY]** —
  [DeepPocket Methods](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2022_01/Deeppocket.pdf)
- **DeepSurf argues against relying on rotation augmentation** and replaces it with surface-aligned
  local grids: "3D cuboid grids are always rotation-sensitive and [this is addressed] by augmenting
  the data with random rotations during [training]... With this approach, the rotation issue is not
  eliminated, since random rotations are still applied... rotated across one of the three axes by
  90°." Its stated motivation is that "a major problem in the grid representation of a protein is the
  lack of rotation invariance." **[PRIMARY]** —
  [DeepSurf §3](https://arxiv.org/abs/2002.05643)
- **EquiPocket makes the same argument explicitly** as its "Issue 2 — sensitive to rotations":
  existing methods handle it by "augmenting training data with random rotations", which "conflicts
  with the fact that any rotation of the protein [should give the same answer]", and claims
  equivariance delivers "rotation invariance **in theory**." **[PRIMARY]** —
  [EquiPocket §1, Issues 2–3](https://arxiv.org/pdf/2302.12177v1)
- **VN-EGNN uses rotation augmentation *inside* an equivariant model** — of the virtual-node
  initialisation, not the protein: "we **randomly rotate this grid of initial virtual node
  coordinates for each sample in every epoch**", which "leads to approximate invariance to different
  initializations". **[PRIMARY]** — [VN-EGNN §2.4, App. E.4](https://arxiv.org/abs/2404.07194)
- **VN-EGNN measures that augmentation's effect** (App. G, Table G1): rotating the protein within the
  Fibonacci grid at inference and averaging over random rotations gives COACH420 **0.612(0.005)** DCC
  / 0.741(0.006) DCA, HOLO4K **0.524(0.002)** / 0.632(0.002), PDBbind2020 **0.702(0.001)** /
  0.833(0.002) — "hardly any change of the DCC and DCA metric across different random rotations.
  Therefore we conclude that our method VN-EGNN achieves **approximate invariance to the initial
  coordinates of the virtual nodes via data augmentation during training**." (Compare the headline
  numbers 0.605 / 0.532 / 0.669 DCC — the rotation-averaged numbers are within noise.)
  **[PRIMARY]** — [VN-EGNN Table G1](https://arxiv.org/abs/2404.07194)
- **The equal-data equivariance-vs-augmentation question, general ML:** "equivariance improves data
  efficiency, but training non-equivariant models with data augmentation can close this gap given
  sufficient epochs. Second, scaling with compute follows a power law, with equivariant models
  outperforming non-equivariant ones at each tested compute budget." Figure caption: "All experiments
  use the same training compute budget, which means that the number of epochs reduces from left to
  right. **Equivariance improves data efficiency compared to the baseline, but data augmentation can
  close this gap.**" The benchmark is synthetic rigid-body mesh dynamics (Kubric/MOVi-B-like, 4·10⁵
  trajectories), GATr vs. a pre-LN multi-query-attention transformer. **[PRIMARY]** —
  [arXiv:2410.23179](https://arxiv.org/abs/2410.23179)
- Within the binding-site papers, the *implicit* answer: the non-equivariant 3D-CNNs that do use
  rotation augmentation (Kalasanty 70.64M params, DeepSurf 33.06M) are beaten by far smaller
  equivariant models at equal data — EquiPocket 1.70M, VN-EGNN 1.20M, GDEGAN 1.90M. E.g. COACH420
  DCC: Kalasanty 0.335, DeepSurf 0.386 vs. EquiPocket 0.423, VN-EGNN 0.605, GDEGAN 0.580.
  **[PRIMARY]** — [GDEGAN Table 1](https://arxiv.org/abs/2603.19817); [VN-EGNN Table 1](https://arxiv.org/abs/2404.07194)

### Inferences

- The strongest actionable finding in this whole note: at ~8.5k structures, **noise augmentation
  (coordinate jitter σ ≈ 0.5 Å, node dropout 3%, feature noise) was worth four times more than 2.6×
  the data.** For a model at 1,100–8,600 structures this is the cheapest large lever, and the
  specific σ values are published.
- The Kalasanty/DeepSurf-vs-EquiPocket/VN-EGNN/GDEGAN comparison is *suggestive* of equivariance
  beating rotation augmentation at equal data, but it is **confounded** by representation (voxel vs.
  graph), features (ESM vs. atom types), and output parameterisation (segmentation vs. direct centre
  regression). It is not a controlled test, and no controlled test exists for proteins.
- VN-EGNN is the existence proof that augmentation and equivariance are complementary rather than
  alternatives: an E(3)-equivariant model still needed rotation augmentation to become invariant to
  a *non-equivariant* part of its own construction (the virtual-node grid).

### Gaps

- **Side-chain perturbation and conformer ensembles: no evidence found.** No paper in this set
  augments with rotamer perturbation or multiple conformers, and none ablates it.
- **Cropping: not stated anywhere** as an augmentation. (The CNN methods *crop* to a 70 Å × 70 Å ×
  70 Å box as a representation constraint, which EquiPocket identifies as a failure mode for large
  proteins, not as augmentation.)
- No protein-structure-task study measures rotation-augmented non-equivariant vs. equivariant at
  matched data **and** matched everything else. This is the clearest missing experiment, and it is
  one EquiCave could run.

---

## TRAINING SCHEDULES: optimisers, epochs, checkpoint selection

### Takeaway

Checkpoint selection is **uniformly "best on a held-out validation split"**, but the *criterion*
differs and is almost never justified: EquiPocket selects on validation **loss**, the graph
transformer on validation **PR-AUC**, VN-EGNN and GDEGAN on unspecified "validation" performance. No
paper publishes an ablation of checkpoint-selection criteria, no paper uses EMA, and no paper uses
staged/curriculum training with frozen loss terms.

### Cited Findings

| Method | Optimiser | LR | Schedule | Epochs / iters | Batch | Checkpoint criterion |
|---|---|---|---|---|---|---|
| EquiPocket | Adam | **1e-4** | not stated | not stated | **8** | **validation loss**, 5-fold CV |
| VN-EGNN | AdamW | **1e-3** | ÷10 after 100 epochs if no improvement for **10 epochs** | **1500** | **64 per GPU × 4 A100** | "best checkpoint based on the validation dataset" (10% split) |
| GDEGAN | AdamW | **5e-4** (→ min 1e-6) | Cosine Annealing Warm Restarts, **10 warmup epochs** | **100** max, **early-stopping patience 30** | **16** | "best checkpoints based on the validation set" (10% split) |
| DeepPocket | Adam, wd **1e-3** | **1e-3** | ReduceLROnPlateau ÷10 after **15 contiguous test intervals** without improvement | **200,000 iters** (cls), **200 epochs** (seg), tested every 1000 iters | **50** | not stated explicitly; PDBbind-Refined masks used to pick the substructure-benchmark checkpoint |
| DeepSurf | Adam, L2 λ **1e-4** | **1e-3** | not stated | **20** | **64** | not stated; 5-fold CV |
| Graph transformer | Adam | max **2e-4** | cosine annealing, **25-epoch warmup** | **300** | **128** | **highest validation PR-AUC** |
| UniSite | AdamW, wd **0.05** | **1e-4** | not stated | not stated | not stated | not stated |

- EquiPocket: "we use Adam optimizer for model training with a learning rate of **0.0001** and set the
  batch size as **8**. The basic dimensions of node and edge embeddings are both set to **128**. The
  dropout rate is set to **0.1**. The probe radius in MSMS ... is set to **1.5**." and "We take
  **5-fold cross validation** on training data scPDB and **use valid loss to save checkpoint**."
  Cutoff θ swept over {2,4,6,8,10}; surface-EGNN depth 4. **[PRIMARY]** —
  [EquiPocket §5.1.7, App. A.2.2](https://arxiv.org/pdf/2302.12177v1)
- GDEGAN full hyperparameter table (search space; bold-equivalent selected values in the text): LR
  {0.003, 0.0003, **0.0005**}, min LR 1e-6, batch {8, **16**, 32}, optimiser {Adam, **AdamW**},
  scheduler Cosine Annealing Warm Restarts, warmup **10** epochs, max **100** epochs, **early
  stopping patience 30**, gradient clipping {10, 15}, weight decay {0.01, **0.05**}, dropout
  {0.1, 0.2, 0.5}, node hidden dim **128**, edge dim **128**, **L_max = 2**, layers
  {3, **4**, 5, 6}, **32 RBFs**, **max 32 neighbours**, heads {4, **8**}, activation {ReLU, **SiLU**},
  **τ = 0.5**, edge spatial cutoff **r_c = 10 Å**. Trained on one NVIDIA H100 NVL.
  **[PRIMARY]** — [GDEGAN App. G, Table 5](https://arxiv.org/abs/2603.19817)
- GDEGAN depth sensitivity (Table 6, average DCC): 3 layers 0.528/0.534/0.643 (COACH/HOLO/PDBbind
  DCC), **4 layers (default) 0.580/0.560/0.675**, 5 layers 0.564/0.544/0.658, 6 layers
  0.550/0.541/0.660 — "performance peaks at **4 layers (Average DCC = 0.6050)** and degrades with
  deeper architectures (5 layers: 0.588, 6 layers: 0.583), despite each additional layer adding
  nearly 20% more parameters", attributed to oversmoothing. **[PRIMARY]** —
  [GDEGAN App. H, Table 6](https://arxiv.org/abs/2603.19817)
- GDEGAN reports a **validation-test gap in the right direction**: "This validation-test gap
  indicates that our method learns the right features for ... better test performance", and shows
  train/validation loss curves to ~50 epochs "consistent with our early [stopping criteria]".
  **[PRIMARY]** — [GDEGAN App. E, Fig. 5](https://arxiv.org/abs/2603.19817)
- VN-EGNN: "We used AdamW ... for **1500 epochs**, selecting the best checkpoint based on the
  validation dataset. We used **5 VN-EGNN layers** ... the feature and message size was set to
  **100** ... The learning rate was set to **10⁻³**, after 100 epochs we reduced the learning rate by
  factor of 10⁻¹ if the model did not improve for 10 epochs. For training we used **4× NVIDIA A100
  40GB GPU with batch size set to 64 on each GPU. The training time was about 8 hours.**
  Hyperparameters were selected based on a validation dataset where we split **10% of the training
  data**." Searched: layers {2,3,4,**5**}, **K virtual nodes {4, 8}**, node dim {20,30,**100**},
  message dim {40,50,**100**}, Huber δ {1}. **[PRIMARY]** —
  [VN-EGNN §3.3, Table E1](https://arxiv.org/abs/2404.07194)
- DeepPocket: "trained using the **Adam** optimizer with a base learning rate of **0.001** and weight
  decay of **0.001** ... for **200 000 iterations**, with rotational augmentation, and tested the
  model at **every 1000 iterations** ... The Pytorch learning rate scheduler **ReduceLROnPlateau**
  was also implemented to reduce the learning rate by a factor of **10** if model performance did not
  improve after **15 contiguous test intervals**." Grids: 23.5 Å side at 0.5 Å resolution for
  classification (input 14@48×48×48), 32 Å side at 0.5 Å for segmentation (input 14@65×65×65).
  **[PRIMARY]** — [DeepPocket Methods](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2022_01/Deeppocket.pdf)
- DeepSurf: "L2 regularization was applied on the weights of all convolutional layers (**λ = 10⁻⁴**),
  while batch normalization was applied with its default parameters. All models were trained for
  **20 epochs**, with batch size of **64** samples, and were optimized by the **Adam** optimizer with
  a learning rate of **10⁻³**." **[PRIMARY]** — [DeepSurf §4](https://arxiv.org/abs/2002.05643)
- Graph transformer: "Adam optimizer; cosine annealing with warm-up over **25 epochs**; maximum
  learning rate **2×10⁻⁴**; batch size **128**; total epochs **300**; model selection: **highest
  PR-AUC value for the validation dataset**." Model sizes: baseline/M ≈ **1.28M params (L = 8,
  d = 128)**; L ≈ **7.34M params (L = 12, d = 256)**. **[SECOND-HAND]** —
  [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0308425);
  [PMC11302905](https://pmc.ncbi.nlm.nih.gov/articles/PMC11302905/)
- UniSite: "AdamW optimizer, learning rate **1.0×10⁻⁴**, weight decay **0.05**", trained on **8×
  NVIDIA RTX 4090**. **[SECOND-HAND]** — [UniSite](https://arxiv.org/html/2506.03237v2)

### Inferences

- There is a 15× spread in epoch budgets (DeepSurf 20, GDEGAN 100, graph transformer 300, VN-EGNN
  1500) with no paper explaining the choice, which suggests nobody has found it to be a sensitive
  axis — or that nobody has looked.
- The one clear design signal: the two most recent equivariant papers both **select on a 10%
  validation split with early stopping**, and GDEGAN explicitly notes it stops well before its
  100-epoch cap (loss curves shown to 50 epochs). For a 5k-structure set, long schedules with
  patience-based stopping are the norm.
- GDEGAN's layer sweep is the clearest "don't go deeper" evidence in the field for this task:
  4 layers, and performance *monotonically degrades* past that despite more parameters.

### Gaps

- **EMA: not used or not stated by any paper I read.** No evidence.
- **Staged / curriculum training, freezing loss terms early: no evidence found.** EquiPocket, GDEGAN
  and VN-EGNN all state the full objective is optimised end-to-end from the start ("We train the
  parameters of all the three modules end to end").
- **No published ablation of checkpoint-selection criterion** (validation loss vs. validation DCC vs.
  validation PR-AUC) for this task. Given that EquiPocket selects on loss and the graph transformer
  on PR-AUC, and that DCC success rate is the headline metric, this is an unexamined and plausibly
  non-trivial choice.
- DeepSurf's and UniSite's checkpoint criteria, and UniSite's epoch count: not stated.

---

## NEGATIVE SAMPLING: how negatives are chosen, and whether it matters

### Takeaway

Two distinct regimes. **Candidate-based methods** (DeepPocket, the graph transformer, P2Rank-style)
take negatives from a geometric detector's output — which imposes a hard recall ceiling that
DeepPocket quantifies: Fpocket covers only **80.78% of COACH420 ligands** and 87.62% of HOLO4k, so a
rescoring model can never exceed that. **Dense methods** (EquiPocket, VN-EGNN, GDEGAN) do no negative
sampling at all. The only isolated measurement of balancing says it barely matters (+0.0012 PR-AUC).

### Cited Findings

- **DeepPocket's negative definition:** "we processed the data set using the pocket centers predicted
  by **Fpocket** [version 3.0]. **Any center that was within 4 Å of any ligand atom was marked as a
  positive data point and the rest were marked as negative.**" Redundancy is handled by labelling
  against *all* binding sites of a redundant protein: "Due to the presence of redundant proteins in
  the scPDB database, we processed the data set to take account of all the binding sites for a
  redundant protein and labeled the candidate pocket centers from Fpocket according to the
  information from all of these binding sites." **[PRIMARY]** —
  [DeepPocket Methods](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2022_01/Deeppocket.pdf)
- **The recall ceiling the choice of generator imposes** — this is the most actionable number in this
  section: "Fpocket found pocket centers that were within 4 Å of any ligand atom for **96.4% of
  ligands in scPDB, 80.78% of ligands in COACH420, 87.62% of ligands in HOLO4k, 91.64% of ligands in
  SC6K, and 95.89% of ligands in the Refined set**." **[PRIMARY]** —
  [DeepPocket Methods](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2022_01/Deeppocket.pdf)
- **DeepPocket's segmentation stage uses positives only:** "Positive data points of the classification
  data set were used to train the segmentation CNN models." **[PRIMARY]** —
  [DeepPocket Methods](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2022_01/Deeppocket.pdf)
- **DeepSurf:** negatives are surface points *not* within 4 Å of any ligand atom, randomly
  undersampled per protein to match the positive count (50/50). Surface points come from DMS at
  density d = 0.2, simplified with f = 10 to a ~2.3 Å mean nearest-neighbour spacing. **[PRIMARY]** —
  [DeepSurf §4](https://arxiv.org/abs/2002.05643)
- **Graph transformer:** rather than subsampling negatives, it **augments positives up to the negative
  count** using the noise augmentations: true-label samples "were augmented to a level equal to the
  number of false-label samples" via noise-based augmentation rather than simple repetition.
  **[SECOND-HAND]** — [PMC11302905](https://pmc.ncbi.nlm.nih.gov/articles/PMC11302905/)
- **Dense methods:** EquiPocket labels every *surface* atom (within 4 Å of a ligand atom = positive)
  and trains Dice over all of them; GDEGAN labels every *residue*; VN-EGNN labels every residue and
  additionally supervises K virtual nodes. None describes negative sampling. **[PRIMARY]** —
  [EquiPocket §4.4](https://arxiv.org/pdf/2302.12177v1); [GDEGAN §3.5](https://arxiv.org/abs/2603.19817); [VN-EGNN §2.5](https://arxiv.org/abs/2404.07194)
- **Does the choice matter?** The only controlled evidence is the Unbal → Bal row: **+0.0012 combined
  PR-AUC**, and a slight *decrease* in COACH420 ROC-AUC. **[SECOND-HAND]** —
  [PMC11302905](https://pmc.ncbi.nlm.nih.gov/articles/PMC11302905/)
- Relatedly, DeepSurf notes that unannotated-but-real sites make "negatives" noisy: it observes a
  structure counted as a "negative sample due to the absence of known bind[ing site annotation]",
  and DeepPocket similarly notes it detects "sites that have not been annotated in the data set".
  **[PRIMARY]** — [DeepSurf](https://arxiv.org/abs/2002.05643);
  [DeepPocket](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2022_01/Deeppocket.pdf)

### Inferences

- For a candidate-ranking architecture, the **candidate generator's recall is the binding constraint,
  not the loss or the negative-sampling scheme.** 80.78% Fpocket recall on COACH420 caps DeepPocket's
  achievable top-n at ~81%, and its measured 67.96% top-n leaves ~13 points of headroom inside that
  cap. For EquiCave, measuring and raising native-candidate recall is a better-evidenced lever than
  tuning negative sampling.
- "Negatives" in this task are systematically **false negatives** (unannotated real sites), which is
  the likely reason explicit rebalancing does nothing: sharpening the decision boundary against noisy
  negatives has no payoff. This also argues against hard-negative mining.

### Gaps

- **No paper ablates hard-negative mining, decoy-pocket construction, or negative:positive ratio.**
  The ratio question is answered only by "we made it 1:1" or "we didn't resample".
- GrASP's negative construction: not recoverable from what I read.

---

## TRANSFER AND PRETRAINING

### Takeaway

**This is the best-evidenced large lever in the whole literature, and it is not structural
pretraining — it is swapping hand-made atom features for frozen ESM-2 embeddings.** GDEGAN measures
**+15.61% DCC and +9.27% DCA** averaged across the three benchmarks from that one change, larger than
every other component in its ablation; VN-EGNN's ESM ablation agrees (+0.078 DCC on COACH420,
+0.147 on PDBbind). I found **no** paper that pretrains on a larger structural corpus, or on
protein–protein interfaces / conservation / solvent accessibility, and then fine-tunes on ligand
sites.

### Cited Findings

- **GDEGAN, ESM-2 as the dominant component:** "Using ESM embeddings (GotenNet+ESM) instead of atomic
  number embeddings (GotenNet) improves the results by **15.61% DCC and 9.27% DCA** averaged across
  datasets, indicating **evolutionary information helps capture better binding regions than a purely
  geometric approach**." Compare: Gaussian attention over dot-product attention **+3.33% DCC /
  +3.32% DCA**, and the auxiliary directional loss **+2% DCC / +3.5% DCA**. **[PRIMARY]** —
  [GDEGAN §4.5](https://arxiv.org/abs/2603.19817)
- GDEGAN's use of ESM also changes the symmetry group: "The scalar nature of ESM embeddings ensures
  that reflection equivariance is not required, **reducing E(3) to SE(3)**." **[PRIMARY]** —
  [GDEGAN App. C](https://arxiv.org/abs/2603.19817)
- **VN-EGNN, ESM ablation (same architecture, ESM on/off):** VN-EGNN (homog., no ESM) 0.497 / 0.414 /
  0.502 DCC → VN-EGNN (homog., **+ESM**) **0.575 / 0.479 / 0.649** DCC on COACH420 / HOLO4K /
  PDBbind2020. Deltas **+0.078 / +0.065 / +0.147 DCC**. **[PRIMARY]** —
  [VN-EGNN Table 2](https://arxiv.org/abs/2404.07194)
- VN-EGNN sources its node features exactly this way: "For the initial residue node features we used
  **pre-trained ESM-2** protein embeddings following Corso et al. (2023); Pei et al. (2023). For
  virtual nodes, we derived their features by **averaging the residue node features across the entire
  protein**." Residue positions are **α-carbons**. **[PRIMARY]** —
  [VN-EGNN §3.3](https://arxiv.org/abs/2404.07194)
- VN-EGNN flags the un-exploited structural corpus explicitly, as future work rather than a result:
  it points at "the 200 million structures [of the] AlphaFold DB, with predicted [structures]" as a
  direction. **[PRIMARY]** — [VN-EGNN §4](https://arxiv.org/abs/2404.07194)
- GDEGAN's framing names the same opportunity: "Equivariant GNNs have emerged as a powerful paradigm
  ... due to the **large-scale availability of 3D structures of proteins via protein databases and
  AlphaFold predictions**" — but GDEGAN itself trains only on scPDB. **[PRIMARY]** —
  [GDEGAN Abstract](https://arxiv.org/abs/2603.19817)
- **SASA as a transferred/engineered feature is worth a lot:** adding solvent-accessibility features
  (Bal+aug → Bal+aug+SA) moved combined test PR-AUC **0.7115 → 0.7640 (+0.0525)** — twice the gain
  from 2.6× more training data. **[SECOND-HAND]** —
  [PMC11302905](https://pmc.ncbi.nlm.nih.gov/articles/PMC11302905/)
- **Weak, off-task evidence on supervised pretraining:** search summaries report that in
  binding-**affinity** prediction, "pretrained graph-based models consistently outperform randomly
  initialized models across all data splits, highlighting the benefits of supervised pretraining on
  heterogeneous binding affinity data." I did not verify this in the primary text and it is a
  *different task*; the report writer should not present it as site-prediction evidence.
  **[SECOND-HAND, low confidence]** —
  [Harnessing pre-trained models for binding affinity, PMC11834573](https://pmc.ncbi.nlm.nih.gov/articles/PMC11834573/)

### Inferences

- Ranked by measured effect size across the primary ablations I read: **ESM-2 features (+15.6% DCC
  relative) > SASA features (+0.053 PR-AUC) ≈ noise augmentation (+0.098 PR-AUC) > attention/
  architecture refinement (+3.3% DCC) ≈ directional auxiliary loss (+2% DCC) ≈ 2.6× more structures
  (+0.026 PR-AUC) > model size (+0.014 PR-AUC) > class rebalancing (+0.001 PR-AUC)**. These come
  from different papers and metrics so the ordering is indicative, not a single controlled ranking —
  but the top and bottom of the list are robust across sources.
- The most striking gap-as-opportunity: everyone in this literature *cites* AlphaFold DB's 200M
  structures as the reason structure-based site prediction is now tractable, and then trains on 5,372
  experimental structures. No one has published a result from pretraining on that corpus for this
  task.

### Gaps

- **No published result on pretraining on a larger structural corpus (PDB-wide, AlphaFold DB) then
  fine-tuning on ligand sites.** Searched; found none.
- **No published result on pretraining on a related task** (protein–protein interfaces,
  conservation, solvent accessibility) then fine-tuning on ligand sites. SASA appears only as an
  *input feature*, never as a pretraining target.
- Whether ESM embeddings are frozen or fine-tuned is not stated by GDEGAN; VN-EGNN says only that a
  linear layer maps them to the model dimension, implying frozen, but does not state it.

---

## Cross-cutting notes the report writer may need

### Protocol differences that confound the headline numbers

- **HOLO4K is evaluated per-chain by the recent methods.** GDEGAN: "We have split the HOLO4K dataset
  into per-chain components and aggregated the predictions in our evaluated results." VN-EGNN:
  "Because of the large complexes in HOLO4K, we ran VN-EGNN for each chain and merged the predicted
  pocket centers." Both also label HOLO4K a "strong distribution shift" / "strong domain shift" from
  scPDB. P2Rank's published HOLO4K numbers do not use this protocol. **[PRIMARY]** —
  [GDEGAN App. D.1, Table 1 footnote](https://arxiv.org/abs/2603.19817);
  [VN-EGNN §3.3, Table 1 footnote](https://arxiv.org/abs/2404.07194)
- **Metric definitions.** GDEGAN states DCC/DCA success rate as
  `|{predicted sites with DCC/DCA < τ}| / |{true sites}|` and failure rate as
  `|{proteins with zero predicted centers}| / |{proteins}|`, with **τ = 4 Å**. **[PRIMARY]** —
  [GDEGAN App. D.3, Eqs. 25–28](https://arxiv.org/abs/2603.19817)
- **The prediction-object gap.** EquiPocket devotes an appendix to the fact that CNN methods predict
  *grids* (including grids over ligand atoms) while graph methods predict *protein atoms*, creating
  "a natural gap in the prediction object ... which also lead[s] to the natural gap for the center of
  predicted binding site", and corrects for it with the 4 Å outward projection (Eq. 21). Any
  cross-family DCC comparison inherits this. **[PRIMARY]** —
  [EquiPocket App. A.2.4](https://arxiv.org/pdf/2302.12177v1)

### Train-distribution bias on protein size (directly relevant at small N)

- EquiPocket measures that the scPDB size distribution biases the learned model: "According to the
  train data (scPDB), the majority of proteins contain **fewer than 2,000 protein atoms** ...
  Consequently, the model's parameters will be biased toward this protein size. In addition, for
  proteins with **more than 8000 atoms**, the prediction effect is not even as good as the
  geometric-based method." **[PRIMARY]** — [EquiPocket §5.2.3](https://arxiv.org/pdf/2302.12177v1)
- Both of EquiPocket's mitigations are size-conditional, which is a useful caution about expecting
  flat gains: the Dense Attention module helps only below ~3,000 atoms ("when the number of atoms
  contained in a protein is less than 3000, the result of the EquiPocket (w/o Dense Attention) is
  weaker ... whereas when the protein is larger, there is no significant distinction between the two
  models"), and the direction loss likewise helps only below ~3,000 atoms. **[PRIMARY]** —
  [EquiPocket §5.2.2](https://arxiv.org/pdf/2302.12177v1)

### Inference cost, for budgeting

- GDEGAN Table 3, seconds per 100 proteins: **GDEGAN 1.90** (residue-level nodes), GotenNet 4.12,
  **EquiPocket 37.00** (atom-level nodes), Fpocket 23.00, Kalasanty 86.00, **DeepSurf 641.00**.
  **[PRIMARY]** — [GDEGAN App. D.4](https://arxiv.org/abs/2603.19817)
- VN-EGNN per-protein: 3LPK (910 residues) 2.367 s; 1ODI (1,410 residues) 3.809 s. **[PRIMARY]** —
  [VN-EGNN App. E.3](https://arxiv.org/abs/2404.07194)

### Overall gaps in this literature

- No learning curves. No data-size ablations within any single paper on this task except the
  cross-corpus scPDB-vs-PoSSuM comparison.
- No loss-weight ablations (every method uses 1:1 and two say tuning was unnecessary).
- No EMA, no curriculum/staged loss schedules, no checkpoint-criterion ablation.
- No side-chain or conformer augmentation evidence.
- No structural or auxiliary-task pretraining evidence.
- Inconsistent and often absent test-homologue removal, which makes the published
  COACH420/HOLO4k/PDBbind league table only loosely comparable across methods — VN-EGNN is the only
  paper that footnotes this explicitly.
