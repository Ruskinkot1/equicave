# State of the art in ligand binding-site prediction: absolute accuracy numbers with their exact evaluation protocols (as of October 2026)

**How to read this file.** Every number carries a protocol tag. Numbers from different protocol tags must never be
placed in the same column. Verification status is marked on each finding:

- **[P]** = primary, I read the paper PDF/HTML or the repository file myself in this session.
- **[S]** = second-hand, the number is quoted in another paper's baseline table or in a search-engine summary; the
  originating paper was not read.

Protocol tag format used throughout: `{metric}/{threshold}/{rank cutoff}/{receptor}/{ligand rule}/{leakage control}`.

**Headline warning found during this research:** the arXiv version of EquiPocket (2302.12177) — the paper whose
protocol and whose baseline numbers are copied by most 2024–2026 papers — **has been withdrawn by its authors**.
The abstract page states: "This paper has been withdrawn by the authors. After further internal evaluation, we find
that the technical elaboration and experimental design in the current manuscript require substantial restructuring
and revision," with v4 and v5 (August 2026) marked withdrawn. The v3 full text remains readable and is what I quote
below. — [P] [arXiv:2302.12177](https://arxiv.org/abs/2302.12177)

---

## Q1. Published DCC and DCA success rates per method, per benchmark

### Takeaway

There are at least **four mutually incomparable protocol families** on COACH420/HOLO4K, and the same method moves by
10–25 absolute points between them: (A) the P2Rank 2018 family (whole deposited multi-chain file, MOAD-based
relevant-ligand list, top-n and top-(n+2)); (B) the Kalasanty/DeepSurf family (scPDB-trained CNNs, top-n, 90 %
sequence-similarity filter applied only to the authors' own model); (C) the EquiPocket family (mlig subsets, top-n
capped at the true site count, no leakage control, **P2Rank retrained by the EquiPocket authors on 792 scPDB
samples**); (D) the LIGYSIS family (biological assemblies, BioLiP relevant ligands, DCC 10–12 Å, top-N+2). GDEGAN
(March 2026) is the current top reported number inside family (C); VN-EGNN is the top inside family (C) with
re-run P2Rank/DeepPocket; GrASP and fpocket+PRANK lead in families (A) and (D) respectively.

### Cited Findings

#### Family C — EquiPocket protocol, COACH420 / HOLO4K (mlig) / PDBbind2020 refined. DCC and DCA, 4 Å, top-n

EquiPocket Table 1 ("Experimental and ablation results of baseline models and our framework"), footnote b = "We use
their published results, codes, or pretrained models." — [P] [arXiv:2302.12177v3](https://arxiv.org/html/2302.12177v3)

| Method | Param (M) | Failure rate | COACH420 DCC | COACH420 DCA | HOLO4K DCC | HOLO4K DCA | PDBbind2020 DCC | PDBbind2020 DCA |
|---|---|---|---|---|---|---|---|---|
| Fpocket (published tool) | – | 0.000 | 0.228 | 0.444 | 0.192 | 0.457 | 0.253 | 0.371 |
| **P2Rank (retrained by EquiPocket authors on 792 scPDB samples)** | – | 0.000 | **0.366** | **0.628** | 0.314 | 0.621 | 0.503 | 0.677 |
| DeepSite (number taken from DeepSurf paper) | 1.00 | – | – | 0.564 | – | 0.456 | – | – |
| Kalasanty (published pretrained model) | 70.6 | 0.120 | 0.335 | 0.636 | 0.244 | 0.515 | 0.416 | 0.625 |
| DeepSurf (published pretrained model) | 33.1 | 0.054 | 0.386 | 0.658 | 0.289 | 0.635 | 0.510 | 0.708 |
| RecurPocket (kalasanty-variant pretrained model) | 21.2 | 0.075 | 0.354 | 0.593 | 0.277 | 0.616 | 0.492 | 0.663 |
| GAT | 0.03 | 0.110 | 0.039(0.005) | 0.130(0.009) | 0.036(0.003) | 0.110(0.010) | 0.032(0.001) | 0.088(0.011) |
| GCN | 0.06 | 0.163 | 0.049(0.001) | 0.139(0.010) | 0.044(0.003) | 0.174(0.003) | 0.018(0.001) | 0.070(0.002) |
| GCN2 | 0.11 | 0.466 | 0.042(0.098) | 0.131(0.017) | 0.051(0.004) | 0.163(0.008) | 0.023(0.007) | 0.089(0.013) |
| SchNet | 0.49 | 0.140 | 0.168(0.019) | 0.444(0.020) | 0.192(0.005) | 0.501(0.004) | 0.263(0.003) | 0.457(0.004) |
| EGNN | 0.41 | 0.270 | 0.156(0.017) | 0.361(0.020) | 0.127(0.005) | 0.406(0.004) | 0.143(0.007) | 0.302(0.006) |
| EquiPocket-L (ablation) | 0.15 | 0.552 | 0.070(0.009) | 0.171(0.008) | 0.044(0.004) | 0.138(0.006) | 0.051(0.003) | 0.132(0.009) |
| EquiPocket-G (ablation) | 0.42 | 0.292 | 0.159(0.016) | 0.373(0.021) | 0.129(0.005) | 0.411(0.005) | 0.145(0.007) | 0.311(0.007) |
| EquiPocket-LG (ablation) | 0.50 | 0.220 | 0.212(0.016) | 0.443(0.011) | 0.183(0.004) | 0.502(0.008) | 0.274(0.004) | 0.462(0.005) |
| **EquiPocket (full)** | 1.70 | 0.051 | **0.423(0.014)** | **0.656(0.007)** | **0.337(0.006)** | **0.662(0.007)** | **0.545(0.010)** | **0.721(0.004)** |

EquiPocket probe-radius sensitivity (MSMS probe radius, DCC only): r=1 → 0.433/0.338/0.549 (failure 0.053);
r=1.5 (used) → 0.423/0.337/0.545 (failure 0.051); r=2 → 0.393/0.289/0.524 (failure 0.096). — [P] EquiPocket Table 6

#### Family C′ — VN-EGNN, same benchmarking setting, with P2Rank and DeepPocket **re-run** under their own training sets

VN-EGNN Table 1, footnote a = std over training re-runs, footnote b = "Results from Zhang et al. (2023b)"
[= EquiPocket], footnote c = "Uses different training set and, thus, limited comparability", footnote d = "This
dataset represents a strong domain shift from the training data for all methods (except for P2Rank)."
— [P] [arXiv:2404.07194](https://arxiv.org/abs/2404.07194) (PDF read locally)

| Method | Source of number | COACH420 DCC | COACH420 DCA | HOLO4K DCC | HOLO4K DCA | PDBbind2020 DCC | PDBbind2020 DCA |
|---|---|---|---|---|---|---|---|
| Fpocket | b (copied from EquiPocket) | 0.228 | 0.444 | 0.192 | 0.457 | 0.253 | 0.371 |
| **P2Rank** | **c (re-run, released model)** | **0.464** | **0.728** | **0.474** | **0.787** | **0.653** | **0.826** |
| DeepSite | b | – | 0.564 | – | 0.456 | – | – |
| Kalasanty | b | 0.335 | 0.636 | 0.244 | 0.515 | 0.416 | 0.625 |
| DeepSurf | b | 0.386 | 0.658 | 0.289 | 0.635 | 0.510 | 0.708 |
| **DeepPocket** | **c (re-run)** | 0.399 | 0.645 | 0.456 | 0.734 | 0.644 | 0.813 |
| GAT | b | 0.039(0.005) | 0.130(0.009) | 0.036(0.003) | 0.110(0.010) | 0.032(0.001) | 0.088(0.011) |
| GCN | b | 0.049(0.001) | 0.139(0.010) | 0.044(0.003) | 0.174(0.003) | 0.018(0.001) | 0.070(0.002) |
| GAT + GCN | b | 0.036(0.009) | 0.131(0.021) | 0.042(0.003) | 0.152(0.020) | 0.022(0.008) | 0.074(0.007) |
| GCN2 | b | 0.042(0.098) | 0.131(0.017) | 0.051(0.004) | 0.163(0.008) | 0.023(0.007) | 0.089(0.013) |
| SchNet | b | 0.168(0.019) | 0.444(0.020) | 0.192(0.005) | 0.501(0.004) | 0.263(0.003) | 0.457(0.004) |
| EGNN | b | 0.156(0.017) | 0.361(0.020) | 0.127(0.005) | 0.406(0.004) | 0.143(0.007) | 0.302(0.006) |
| EquiPocket | b | 0.423(0.014) | 0.656(0.007) | 0.337(0.006) | 0.662(0.007) | 0.545(0.010) | 0.721(0.004) |
| **VN-EGNN (1.20 M)** | own run | **0.605(0.009)** | **0.750(0.008)** | **0.532(0.021)** | 0.659(0.026) | **0.669(0.015)** | 0.820(0.010) |

VN-EGNN ablation (Table 2), all own runs: EGNN 0.156/0.361, 0.127/0.406, 0.143/0.302; VN-EGNN (residue emb.)
0.503(0.022)/0.684(0.016), 0.438(0.019)/0.605(0.013), 0.551(0.017)/0.751(0.009); VN-EGNN (homog., no ESM)
0.497(0.014)/0.700(0.013), 0.414(0.023)/0.618(0.024), 0.502(0.029)/0.717(0.025); VN-EGNN (homog., ESM)
0.575(0.008)/0.708(0.009), 0.479(0.012)/0.595(0.010), 0.649(0.010)/0.805(0.006); VN-EGNN (full)
0.605(0.009)/0.750(0.008), 0.532(0.021)/0.659(0.026), 0.669(0.015)/0.820(0.010). — [P]

VN-EGNN's own text concedes the P2Rank comparison is not clean: "Note that there is limited comparability with
P2rank since this method uses a different training set that might be closer to HOLO4K. HOLO4K contains many
complexes of symmetric proteins … which should be considered as a strong domain shift to the training data and thus
pose a problem for all methods except P2rank." — [P] [arXiv:2404.07194](https://arxiv.org/abs/2404.07194)

#### Family C″ — GDEGAN (arXiv 2603.19817, 20 March 2026; Animesh, Bhowmick, Mitra), GotenNet backbone

GDEGAN Table 1; footnote b = "Results from the EquiPocket (Zhang et al., 2024) paper"; footnote k = "holo4k contains
multi chains and complex with multiple copies, presenting a strong distribution shift."
— [P] [arXiv:2603.19817](https://arxiv.org/html/2603.19817) (PDF read locally)

| Method | Source | COACH420 DCC | COACH420 DCA | HOLO4K DCC | HOLO4K DCA | PDBbind2020 DCC | PDBbind2020 DCA |
|---|---|---|---|---|---|---|---|
| Fpocket, P2rank, DeepSite, Kalasanty, DeepSurf, RecurPocket, GAT, GCN, GCN2, SchNet, EGNN, EquiPocket | all footnote b — **copied verbatim from EquiPocket Table 1** | (identical to the EquiPocket table above) | | | | | |
| GotenNet (2.20 M, failure 0.049) | own run | 0.464(0.007) | 0.624(0.014) | 0.454(0.001) | 0.691(0.005) | 0.553(0.008) | 0.705(0.007) |
| **GDEGAN (1.90 M, failure 0.032)** | own run | **0.580(0.008)** | **0.707(0.009)** | **0.560(0.013)** | **0.788(0.011)** | **0.675(0.010)** | **0.826(0.011)** |

GDEGAN ablation (Table 2), own runs: GotenNet E(3), no ADL, no ESM → 0.454(0.007)/0.624(0.014),
0.464(0.001)/0.691(0.005), 0.553(0.008)/0.705(0.007); GotenNet+ADL → 0.485/0.642, 0.468/0.732, 0.592/0.748;
GotenNet+ESM (SE(3)) → 0.543/0.693, 0.520/0.753, 0.637/0.760; GotenNet(full) → 0.556/0.703, 0.529/0.749,
0.649/0.801; GDEGAN+ESM → 0.572/0.702, 0.532/0.769, 0.652/0.810; GDEGAN(full) → 0.580/0.707, 0.560/0.788,
0.675/0.826. — [P]

**Two internal inconsistencies in the GDEGAN tables (verified by direct comparison, flag these):** (1) the GotenNet
row in Table 1 gives COACH420 DCC 0.464 and HOLO4K DCC 0.454, while Table 2 gives the same model COACH420 DCC
0.454 and HOLO4K DCC 0.464 — the two values appear transposed between tables; (2) the copied GAT row reports
PDBbind2020 DCC 0.018 in GDEGAN but 0.032 in EquiPocket's original table (0.018 is GCN's value) — a transcription
error in the copied baseline block. — [P] GDEGAN Tables 1–2 vs EquiPocket Table 1

GDEGAN inference time (Table 3): GDEGAN 1.90 s / 100 proteins, GotenNet 4.12 s, EquiPocket 37.00 s, Fpocket 23.00 s,
Kalasanty 86.00 s, DeepSurf 641.00 s (the last four marked p = "Results from the EquiPocket paper"). — [P]

#### Family A — P2Rank 2018 protocol, whole deposited structures, MOAD relevant-ligand filter

P2Rank Table 3, DCA, 4 Å, top-n and top-(n+2), n = number of relevant ligands in the structure.
— [P] [Krivák & Hoksza 2018, J Cheminform 10:39, PMC6091426](https://pmc.ncbi.nlm.nih.gov/articles/PMC6091426/)

| Method | COACH420 top-n | COACH420 top-(n+2) | HOLO4K top-n | HOLO4K top-(n+2) |
|---|---|---|---|---|
| **P2Rank** | **72.0 %** | **78.3 %** | **68.6 %** | **74.0 %** |
| Fpocket | 56.4 % | 68.9 % | 52.4 % | 63.1 % |
| Fpocket + PRANK rescoring | 63.6 % | 76.5 % | 62.0 % | 71.0 % |
| SiteHound | 53.0 % | 69.3 % | 50.1 % | 62.1 % |
| MetaPocket 2.0 | 63.4 % | 74.6 % | 57.9 % | 68.6 % |
| DeepSite | 56.4 % | 63.4 % | 45.6 % | 48.2 % |

#### Family B — DeepSurf protocol (scPDB-trained CNNs, full COACH420 = 420 structures, HOLO4K = 4009)

DeepSurf Table 2, DCA, Dcut = 4 Å, top-n and top-(n+2).
— [P] [Mylonas et al. 2021, arXiv:2002.05643](https://arxiv.org/pdf/2002.05643) (PDF read locally; journal version
Bioinformatics 37(12):1681)

| Method | COACH420 top-n | COACH420 top-(n+2) | HOLO4K top-n | HOLO4K top-(n+2) |
|---|---|---|---|---|
| DeepSite | 57.5 | 65.1 | 45.6 | 48.2 |
| Jiang et al. | 55 | 58.7 | 38.2 | 41.5 |
| Kalasanty | 68 | 70.4 | 32.1 | 32.3 |
| **DeepSurf (ResNet-18)** | **72.1** | **73.3** | **50.1** | **50.6** |
| DeepSurf (Bot-LDS-ResNet-18) | 71.7 | 72.7 | 50.4 | 50.8 |

DeepSurf also reports, for the most dissimilar similarity bin (< 40 % global sequence similarity to training),
**67 % on COACH420 and 44.5 % on HOLO4K** — "about 5 % lower to its overall performance". — [P]

DeepSurf CHEN apo/holo (Table 5, DCA 4 Å, top-n / top-(n+2)): Jiang et al. 34.5/35 holo, 28.4/30.5 apo; Kalasanty
35.5/36 holo, 33/34 apo; DeepSurf ResNet-18 40.6/40.6 holo, 39.6/39.6 apo; DeepSurf Bot-LDS 39.1/39.1 holo,
37.6/37.6 apo. Note the absolute level: **all methods are below 41 % on CHEN**, apo or holo. — [P]

DeepSurf overlap (OVR, computed only on correctly located sites with DCA < 4 Å, Table 3): Kalasanty 0.21 COACH420 /
0.15 HOLO4K; DeepSurf ResNet-18 0.29 / 0.17; DeepSurf Bot-LDS 0.28 / 0.17. The authors conclude "the extraction of
properly shaped binding sites is still an open issue." — [P]

DeepSurf failure counts (Table 4): DeepSite 3 COACH420 / 21 HOLO4K; Jiang 12 / 65; **Kalasanty 16 / 475**; DeepSurf
ResNet-18 7 / 5; DeepSurf Bot-LDS 8 / 10. Average predicted pockets vs true average (1.2 COACH420, 2.8 HOLO4K):
DeepSite 3.2/2.8, Jiang 1.4/3.4, Kalasanty 1.1/1.2, DeepSurf 1.1/1.8. — [P]

#### Family A′ — GrASP protocol (own Mlig+ sets, DCA recall + DCA precision)

GrASP (Smith, Tiwary et al., JCIM 2024) built **its own** relevant-ligand sets, COACH420(Mlig+) with 256
single-chain systems and HOLO4K(Mlig+) with 6,368 ligands across 4,514 systems, using geometric criteria plus a
Binding MOAD check. DCA = distance to any ligand heavy atom, 4 Å, top-N and top-N+2. Training: modified sc-PDB,
26,196 binding sites across 16,889 structures, 10-fold CV split **by UniProt ID**.
— [P] [PMC11182664](https://pmc.ncbi.nlm.nih.gov/articles/PMC11182664/)

| Dataset | Metric | GrASP | P2Rank |
|---|---|---|---|
| COACH420(Mlig+) | top-N recall | **77.5 %** | 74.9 % |
| COACH420(Mlig+) | DCA precision (all predictions) | **71.0 %** | 28.3 % |
| HOLO4K(Mlig+) | top-N recall | **81.3 %** | 81.2 % |
| HOLO4K(Mlig+) | DCA precision (all predictions) | **71.4 %** | 25.5 % |
| sc-PDB (own validation) | top-N / top-N+2 recall | 85.3 % / 91.4 % | – |

#### Family D — LIGYSIS independent benchmark (Utgés & Barton, J Cheminform 2024, 16:126)

Largest independent benchmark since Chen et al. 2010: 13 original methods + 15 variants, >10 metrics, 2,775
proteins (human subset of LIGYSIS). Receptor = **biological assembly** (PISA), ligands = **BioLiP**-relevant;
ions are ~40 % of LIGYSIS sites and a LIGYSIS-NI variant excludes them. The authors explicitly reject the 4 Å
convention: "a DCC threshold of 4 Å is too conservative, and a more flexible DCC threshold of 10–12 Å should be
used", and recommend **top-N+2 recall as the universal benchmark metric**.
— [P] [PMC11552181](https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/)

| Method | top-N+2 recall (DCC 12 Å) |
|---|---|
| fpocket + PRANK rescoring | **60.4 %** |
| DeepPocket (rescoring variant) | 58.1 % |
| P2Rank + conservation | 53.9 % |
| P2Rank | 51.9 % |
| GrASP | 49.9 % |
| DeepPocket (segmentation variant) | 43.8 % |
| PUResNet | 41.1 % |
| VN-EGNN | 40.9 % |
| IF-SitePred | 25.7 % |

Also reported: precision ~92.5 % for VN-EGNN and GrASP (few, well-placed predictions) against 39–47 % for
Surfnet+/fpocket; rescoring improvements up to +14 % recall (IF-SitePred) and +30 % precision (Surfnet). Raw
per-method results are deposited: [Zenodo 14645504 "LBS-Comparison results"](https://zenodo.org/records/14645504). — [P]

#### Independent re-run under whole-assembly + mlig (this project's own measurement)

P2Rank 2.5.1 on COACH420, mlig relevant-ligand list, whole deposited assembly, 288 structures evaluated (12 of 300
dropped as `no_ligand`), DCA ≤ 4 Å, non-redundant predictions: **top-1 = 0.753 [0.688, 0.814], top-N = 0.828
[0.771, 0.881], top-(N+2) = 0.879 [0.830, 0.926]**, mean 9.6 predictions per structure, candidate ceiling 0.931.
A second table in the same repository records P2Rank's own ranking at **0.754** top-1 on the 1367-structure internal
set. — [P] `/home/user/equicave/docs/results/eval_coach420.md`, `/home/user/equicave/README.md` (lines 86, 101),
`/home/user/equicave/docs/results/eval_coach420_allchain.json`

#### 2025–2026 methods and cryptic-site benchmarks

- **PUResNetV2.0** (2024): DCA success 85.4 % and F1 74.7 % on Holo801; on Coach100, 59.0 % (DCA ≤ 4 Å) against
  P2Rank 44.0 %, DeepSurf 51.0 %, DeepSite 27.0 %. Note these are **non-standard sets** (Holo801, Coach100), not
  HOLO4K/COACH420, so they belong to no family above. — [S] search summary of
  [J Cheminform 2024 PUResNetV2.0](https://doaj.org/article/3156ae56aece40bd9a0eb50a2df23f1e)
- **CryptoBench** (2025): 1,107 structures built from apo-holo pairs, grouped by UniProt ID, clustered by sequence
  identity, filtered for substantial binding-site structural change, with predefined CV splits; described as the
  most extensive cryptic binding-site dataset to date. On the CB-PM subset **PocketMiner: AUC 0.76, AUPRC 0.19,
  MCC 0.22, F1 0.78**, beaten by a protein-language-model sequence baseline (pLM-NN). — [S]
  [PMC11725321](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11725321/)
- **Sequence-based cryptic-pocket pretraining** (J Cheminform 2026, s13321-026-01227-0) claims cryptic pocket
  detection plus affinity/kinetics gains; numbers not extracted. — [S]
  [link](https://link.springer.com/article/10.1186/s13321-026-01227-0)
- **ProMoSite**: claimed to outperform all 3D-based baselines on COACH420 and HOLO4K top-n using sequence only;
  numbers not extracted. — [S] search summary
- **S1-Omni** (2026): COACH420 AUROC 0.880 and MCC 0.303 → 0.366 — these are **residue-level classification**
  metrics on COACH420, not DCC/DCA site localisation, and must not be put in the same table. — [S]
  [arXiv:2607.15686](https://arxiv.org/pdf/2607.15686)

### Inferences

- Inside family C, the ranking is GDEGAN > VN-EGNN > GotenNet > EquiPocket > DeepSurf > Kalasanty > P2Rank(retrained)
  > Fpocket. Inside family A/A′, the ranking is GrASP ≈ P2Rank > fpocket+PRANK > MetaPocket2 > fpocket > DeepSite.
  **No method is top in both.**
- GDEGAN's claimed "37–66 % relative DCC improvement" is measured against baselines copied from a withdrawn paper,
  including a P2Rank row that is a weakened retrain. Against VN-EGNN's re-run P2Rank (0.728 COACH420 DCA, 0.787
  HOLO4K DCA, 0.826 PDBbind DCA), GDEGAN's DCA numbers (0.707 / 0.788 / 0.826) are **level or marginally behind**
  on DCA and ahead only on DCC.
- HOLO4K DCA is the clearest case of protocol dominance: P2Rank 0.621 (EquiPocket's retrain) vs 0.686 (P2Rank's own
  paper, whole structures) vs 0.787 (VN-EGNN's re-run of the released model) vs 81.2 % (GrASP's Mlig+ set).

### Gaps

- **scPDB-internal** DCC/DCA for most methods: only GrASP reports it (85.3 / 91.4 % top-N / top-N+2). No
  cross-method scPDB table was found.
- **PointSite, SiteRadar, YuelPocket**: no COACH420/HOLO4K DCC or DCA numbers found. Two searches returned nothing
  relevant; these methods may report only residue-level or internal metrics.
- **DeepSite and PDBbind2020**: no DCC and no PDBbind DCA number exists anywhere in family C (the cell is blank in
  EquiPocket, VN-EGNN and GDEGAN alike).
- **LIGYSIS per-method DCC at 4 Å** was not extracted; only the 12 Å top-N+2 recall column was read.
- CryptoBench numbers are second-hand; the primary PDF was not read.

---

## Q2. Protocol annotation for every number (the fields that differ)

### Takeaway

Six protocol axes move the numbers: rank cutoff, receptor definition, relevant-ligand rule, HOLO4K multi-copy
handling, leakage control, and whether the baseline model is the released one or a retrain. EquiPocket-family papers
state the first and third explicitly, are silent on the receptor, and have **no leakage control at all**.

### Cited Findings

**EquiPocket protocol, verbatim** — [P] [arXiv:2302.12177v3](https://arxiv.org/html/2302.12177v3) App. A.4:
- Metric/threshold: "The predicted ligand binding center with the DCC/DCA falls below a predetermined threshold,
  typically set to 4, is classified as successful."
- Rank cutoff: "We evaluate based on the **top-n predicted binding site centers, where n does not exceed the actual
  number of binding sites**. If no predicted binding sites center is found, it is considered a failure." → top-n
  only; **no top-(n+2) number exists in family C**.
- Site centre: "we define the center of a binding site as the mean position of the ligand atoms" (citing P2Rank,
  DeepSite, DeepSurf, DeepPocket, fpocket).
- Training label: "Following DeepSurf (Mylonas et al., 2021), the protein atoms within 4 Ångströms of any ligand
  atom are set as positive, otherwise negative."
- Relevant-ligand rule: "We use the mlig subsets for evaluation (Mylonas et al., 2021)."
- Test-set sizes implied by the binding-site histogram (Table 3): COACH420 = 235 + 36 + 7 + 4 + 2 = **284
  proteins**; HOLO4K = 2442 + 635 + 67 + 22 + 38 = **3204 proteins**; PDBbind = **5025 proteins, every one with
  exactly 1 site**.
- Average structure size (Table 2): scPDB 4205 atoms, COACH420 2123, HOLO4K 3845, PDBbind 3104; surface points
  24010 / 12325 / 20023 / 17357; target atoms 47 / 58 / 106 / 37.
- Surface: MSMS, probe radius 1.5; clustering: mean-shift; model selection: 5-fold CV on scPDB with validation loss.
- Receptor definition (single chain vs assembly): **not stated**.
- HOLO4K multi-copy handling: **not stated**.
- Train/test similarity removal: **not stated anywhere in the paper** — the only dedup mentioned is UniProt-based
  dedup *within* scPDB training.

**GDEGAN protocol, verbatim** — [P] [arXiv:2603.19817](https://arxiv.org/html/2603.19817) App. D:
- DCC = ‖p̂ᵢ − p_ligand‖₂; DCA = min_{b∈L} ‖p̂ᵢ − p_b‖₂; "τ = 4 Å is the standard threshold"; Success rate =
  |{predicted sites with DCC/DCA < τ}| / |{true sites}|; Failure rate = |{proteins with 0 predicted centers}| /
  |{proteins}|.
- Rank cutoff: inherits EquiPocket's top-n; no top-(n+2).
- Receptor / HOLO4K: "**We have split the HOLO4K dataset into per-chain components and aggregated the predictions
  in our evaluated results.**" This is the only explicit HOLO4K-decomposition statement in family C.
- Relevant ligands: "Both datasets use the M_LIG subsets … containing biologically relevant ligands as defined by
  the original curation."
- Preprocessing: "we exclude solvent molecules and apply standard preprocessing, such as removing hydrogen atoms.
  Structures with missing coordinates or ambiguous ligand positions are filtered during preprocessing using rDkit."
- Set sizes: COACH420 "420 protein-ligand complexes", HOLO4K "4,288 structures", PDBbind2020 refined "5,316
  complexes … resolution better than 2.5 Å and complete ligand electron density", from a general set of 14,127.
- Representation: residue graph on Cα, 10 Å cutoff, ESM-2 node features. Binding residue = "any of its constituent
  atoms lies within a threshold distance" dbind = 4 Å of a ligand atom.
- Training: scPDB with a 90:10 split; the fetched text states **17,594 protein-ligand complexes**, preprocessed
  "using the steps described in EquiPocket".
- Leakage control: **not stated**.

**VN-EGNN protocol, verbatim** — [P] [arXiv:2404.07194](https://arxiv.org/abs/2404.07194) §3.2–3.3:
- "We use the benchmarking setting of Zhang et al. (2023b)" [EquiPocket].
- "The DCC is defined as the distance between the predicted and known binding site centers, whereas the DCA is
  defined as the shortest [distance to any ligand atom]"; "we maintained a threshold of 4 Å throughout our
  experiments (for other thresholds, see Fig. 2)".
- Rank cutoff: "for each protein only **M predicted binding sites with the highest self-confidence scores** are
  considered, where M is the number of known binding sites of the protein. Subsequently, each predicted binding
  site was aligned with the closest real binding site and DCC/DCA success rate was calculated." → top-n, greedy
  nearest-site assignment.
- HOLO4K: "**Because of the large complexes in HOLO4K, we ran VN-EGNN for each chain and merged the predicted
  pocket centers.**"
- Representation: Cα positions as residue node locations, ESM-2 initial features, virtual nodes connected only to
  residue nodes, mean-shift clustering of virtual nodes, 5 layers, 1500 epochs, AdamW.
- Leakage control: **not stated**.

**P2Rank 2018 protocol, verbatim** — [P] [PMC6091426](https://pmc.ncbi.nlm.nih.gov/articles/PMC6091426/):
- DCA = "distance between the center of the pocket and any ligand atom", 4 Å threshold.
- Rank cutoff: "Top-n and Top-(n+2) rank cutoffs where n is the number of relevant ligands in the evaluated target
  protein structure (for proteins with only one ligand this corresponds to the usual Top-1 and Top-3 cutoffs)."
- Receptor: COACH420 = "420 single chain structures"; HOLO4K = "larger multi-chain structures downloaded directly
  from PDB"; "P2Rank is able to automatically produce predictions for any PDB file (single or multi chained)" and
  "able to work directly with multi-chain structures and thus find potential binding sites that consist of residues
  from multiple chains."
- Relevant-ligand rule: "PDB files in considered datasets often contain ligands (or HET groups) that are not
  relevant. To determine which ligands are relevant we use a custom filter and alternatively the binding MOAD
  database." The precise heavy-atom minimum and exclusion lists are in Additional file 1, which I did not read →
  **heavy-atom minimum and exclusion lists: not read / not stated in the main text**.
- mlig definition (dataset repository): "`*(mlig)*` datasets: datasets that contain explicitly specified relevant
  ligands. Valid ligand codes come from MOAD 2013 database. **Proteins unknown to MOAD and proteins with
  conflicting ligand codes (valid & invalid) were removed.**" — [P]
  [rdk/p2rank-datasets README](https://github.com/rdk/p2rank-datasets)
- HOLO4K multi-copy handling: the repository warns the dataset files are subsets — "`holo4k/` directory contains
  4543 pdb files but `holo4k.ds` contains 4009 lines. For reproducibility, **4009 is the correct number of proteins
  in the HOLO4K dataset used in P2Rank/PrankWeb papers**"; `1xgf.pdb` was removed (all UNK groups, no ligands). A
  separate `holo4k-chains` style variant exists for per-chain evaluation. — [P]
- Baseline versions used in that paper: "Fpocket … v1.0 with default parameters"; SiteHound, MetaPocket 2.0 and
  DeepSite predictions obtained from their web servers in Fall 2017; P2Rank predictions "correspond to P2Rank 2.0
  with default parameters". `*-XXsubset-*` datasets restrict to structures where a given method produced output. — [P]

**DeepSurf protocol, verbatim** — [P] [arXiv:2002.05643](https://arxiv.org/pdf/2002.05643):
- Receptor/sets: "COACH420 has been derived from the COACH test set and consists of **420 single chain structures**
  containing a mix of drug targets and naturally occurring ligands. HOLO4K is a larger dataset (**4009 structures**)
  containing larger multi-chain structures."
- Surface: DMS, density d = 0.2; hydrogens removed before binding-site extraction so sites keep only heavy atoms.
- Training labels: scPDB surface points within 4 Å of any ligand atom are "binding".
- **Leakage control (the only explicit one in family A/B):** "a global sequence alignment between all targets from
  the training and the three test sets was performed, and **any training target with more than 90 % sequence
  similarity with a testing one was removed. The remaining dataset, consisting of 9444 targets, was used to train**
  the two variants of DeepSurf. Although all of the competing methods have been trained on the same database
  (scPDB), any proteins common to our testing datasets have not been removed. This means that **these methods have
  a slight advantage due to this specific data leakage**."
- Failure handling: "In case that a method fails to produce any binding site, an adequately large value of DCA is
  assigned for each ligand of this protein ensuring that this solution will be regarded erroneous."
- Binding-site atom definition for overlap: "binding sites are defined as the non-hydrogen atoms of a residue that
  are within 4 Å to a non-hydrogen atom of the ligand."
- DeepSite's numbers in DeepSurf's table are themselves copied: "The provided DeepSite results are those obtained
  by [14]."

**GrASP protocol** — [P] [PMC11182664](https://pmc.ncbi.nlm.nih.gov/articles/PMC11182664/):
- DCA to any ligand **heavy** atom, 4 Å, top-N and top-N+2; introduces DCA precision = correctly predicted sites /
  total predicted sites.
- Scores only surface atoms (by SASA), uses buried atoms within 5 Å as context.
- Relevant ligands: own COACH420(Mlig+) / HOLO4K(Mlig+) sets — "only bound, biologically/pharmacologically relevant
  ligands using geometric criteria and Binding MOAD database checks"; COACH420(Mlig+) = 256 single-chain systems;
  HOLO4K(Mlig+) = 6,368 ligands across 4,514 systems.
- Leakage control: 10-fold CV with **UniProt-ID splitting** of the training set.

**LIGYSIS protocol** — [P] [PMC11552181](https://pmc.ncbi.nlm.nih.gov/articles/PMC11552181/): biological assemblies
(PISA); BioLiP relevant ligands; sites aggregated across multiple structures of the same protein so that one
interface is counted once; DCC 12 Å headline (10–12 Å recommended, 4 Å rejected as too conservative); top-N+2 recall
as the headline; ions ≈ 40 % of sites with an ion-free LIGYSIS-NI variant; pocket metrics (recall, precision,
ROC100, #TP100FP) plus residue metrics (F1, MCC, AUC, AP).

### Inferences

- Family C's top-n-only convention systematically *understates* every method relative to families A/A′/D, which
  report top-(n+2) as well; on COACH420 the top-n → top-(n+2) step is worth +6.3 points for P2Rank, +12.5 for
  fpocket (P2Rank's own table) and +12.9 for this project's independent run (0.753 → 0.879).
- Only DeepSurf (90 % global-identity filter, 9444 remaining targets) and GrASP (UniProt-ID 10-fold split) apply any
  leakage control. EquiPocket, VN-EGNN and GDEGAN state none, so their scPDB-trained numbers on COACH420/HOLO4K and
  especially on **PDBbind2020 refined** are optimistic by an unknown amount.
- Family C is also the only family in which **PDBbind2020 refined** is used as a test set; since every PDBbind entry
  has exactly one site, its "top-n" is top-1, which is why PDBbind numbers are systematically the highest DCC
  numbers reported anywhere.

### Gaps

- Receptor definition (single chain / single model / deposited assembly) is **not stated** in EquiPocket or VN-EGNN
  for COACH420 and PDBbind; only GDEGAN states per-chain splitting, and only for HOLO4K.
- The exact P2Rank "custom filter" thresholds (heavy-atom minimum, buffer/ion/sugar exclusion list) live in
  Additional file 1 of the 2018 paper, which I did not retrieve.
- Whether family C's mlig lists are P2Rank's `*(mlig)*` files or DeepSurf's re-derivation of them is not stated;
  EquiPocket cites Mylonas 2021 for the mlig subsets, not Krivák & Hoksza.

---

## Q3. Why P2Rank scores 0.628 in the EquiPocket/GDEGAN protocol and ~0.75 under a whole-assembly + mlig run

### Takeaway

**The decisive reason is that EquiPocket's 0.628 is not P2Rank. It is a P2Rank model that the EquiPocket authors
retrained themselves on a randomly chosen 792-structure subset of scPDB**, because their attempt to retrain on the
full scPDB diverged. The released, properly trained P2Rank model — the one everyone else runs — scores 0.728 (DCA,
COACH420, top-n) when VN-EGNN re-ran it inside the same benchmark, 0.749 under GrASP's Mlig+ protocol, 0.720 in
P2Rank's own 2018 paper, and 0.753 in this project's whole-assembly mlig run. Secondary contributions are the
top-n-only cutoff and, for HOLO4K, per-chain splitting.

### Cited Findings

- EquiPocket, Appendix A.4.6, verbatim: "The machine-learning method P2Rank (Krivák & Hoksza, 2018) was originally
  trained on the CHEN11 and JOINED datasets, totaling **792 samples**. In contrast, deep learning methods such as
  DeepSite, Kalasanty, DeepSurf, and EquiPocket commonly used the scPDB dataset for training, which had 5000 samples
  after processing and deduplication. P2Rank's paper highlighted the greater diversity of CHEN11 and JOINED compared
  to scPDB, affecting model performance. **To ensure a fair comparison, we attempted to retrain P2Rank on the scPDB
  dataset. However, as indicated in Table 5, we encountered convergence issues when the training set size exceeded
  3000 samples. Therefore, we report the test results of retrained P2rank based on a randomly selected subset of 792
  samples from the scPDB**, which matches the quantity of number to the CHEN11 and JOINED datasets of P2rank
  papers." — [P] [arXiv:2302.12177v3](https://arxiv.org/html/2302.12177v3)
- EquiPocket Table 5, "The detailed DCA results of retrained P2rank" (columns COACH420 / HOLO4K / PDBbind2020) — [P]:

  | scPDB samples used to retrain | COACH420 | HOLO4K | PDBbind2020 |
  |---|---|---|---|
  | 100 | 0.611 | 0.602 | 0.655 |
  | 1000 | 0.648 | 0.613 | 0.672 |
  | 2000 | 0.639 | 0.622 | 0.671 |
  | 3000 | 0.177 | 0.130 | 0.256 |
  | 4000 | 0.177 | 0.130 | 0.256 |
  | 5000 | 0.177 | 0.130 | 0.256 |

  The Table 1 figures 0.628 / 0.621 / 0.677 sit inside the 100–2000-sample band, consistent with the stated
  792-sample retrain, and the >3000 rows show an outright training collapse.
- The same table block in EquiPocket confirms the other baselines were **not** retrained: "For geometric-based
  method Fpocket, we use its published tool. For CNN-based methods kalasanty and DeepSurf, we use their published
  pre-train models." and "The results of DeepSite come from (Mylonas et al., 2021)." — [P]
- VN-EGNN re-ran P2Rank with its own (released) training set and got **COACH420 DCC 0.464 / DCA 0.728**, HOLO4K
  0.474 / 0.787, PDBbind2020 0.653 / 0.826, with footnote c "Uses different training set and, thus, limited
  comparability". — [P] [arXiv:2404.07194](https://arxiv.org/abs/2404.07194)
- P2Rank's own 2018 paper: COACH420 DCA top-n **72.0 %**, top-(n+2) 78.3 %; HOLO4K 68.6 % / 74.0 %. — [P]
  [PMC6091426](https://pmc.ncbi.nlm.nih.gov/articles/PMC6091426/)
- GrASP's independent run of P2Rank on COACH420(Mlig+): top-N recall **74.9 %**. — [P]
  [PMC11182664](https://pmc.ncbi.nlm.nih.gov/articles/PMC11182664/)
- This project's independent run, P2Rank 2.5.1, whole assembly, mlig list, 288 COACH420 structures: top-1 DCA
  **0.753 [0.688, 0.814]**, top-N 0.828, top-(N+2) 0.879. — [P]
  `/home/user/equicave/docs/results/eval_coach420.md`
- Contributing (but smaller) protocol differences, all verified: family C counts **top-n only** and never
  top-(n+2); GDEGAN splits HOLO4K into per-chain components and aggregates, VN-EGNN runs per chain and merges, while
  P2Rank's own paper runs the whole multi-chain file (4009 entries); P2Rank's dataset repo warns that the correct
  HOLO4K size is 4009, whereas GDEGAN reports 4,288 structures and EquiPocket's histogram implies 3,204 proteins —
  i.e. **three different HOLO4K populations are in play**. — [P] (sources as cited above)

### Inferences

- The gap decomposes approximately as: retrained-vs-released P2Rank model ≈ 0.628 → 0.728 (+0.10, the dominant
  term, measured inside the same family by VN-EGNN); remaining 0.728 → 0.753 (+0.025) attributable to receptor
  scope (whole assembly vs the family-C preprocessed structure), the exact mlig/`no_ligand` filtering, and
  prediction-merging radius. The 4 Å threshold and the DCA definition itself are identical across all of these, so
  they explain none of the gap.
- Consequence for the literature: **every paper that copies EquiPocket's baseline block (GDEGAN explicitly, and any
  2025–2026 paper citing "P2Rank 0.628/0.621/0.677") is comparing itself against a crippled P2Rank**, and its
  claimed improvement over P2Rank is overstated by roughly 10 DCA points on COACH420 and 17 on HOLO4K. VN-EGNN and
  GrASP are the only two methods examined here that beat a correctly configured P2Rank on COACH420 DCA.
- EquiPocket's own stated motive was fairness (matching training-set size). The effect was the opposite, because
  P2Rank's released model is the artefact the field actually uses, and because the retrain visibly failed
  (divergence above 3000 samples) rather than merely being smaller.

### Gaps

- The exact 0.759 value quoted in the assignment was not reproduced; this project's recorded figures are 0.753 and
  0.754 under the whole-assembly mlig protocol. The difference is probably a different structure count or
  merge-radius variant of the same run, but I could not locate a 0.759 cell in the repository.
- EquiPocket does not state which P2Rank version or training command was used for the retrain, so the divergence
  above 3000 samples cannot be attributed to a specific cause.

---

## Q4. Which protocol each number comes from: re-run vs copied

### Takeaway

Of the 2024–2026 family-C papers, **GDEGAN copies the entire baseline block from EquiPocket** (12 methods), VN-EGNN
copies 11 rows from EquiPocket but **re-runs P2Rank and DeepPocket**, and EquiPocket itself re-runs fpocket,
Kalasanty, DeepSurf and RecurPocket from released artefacts, retrains P2Rank, and copies DeepSite from DeepSurf.

### Cited Findings

| Number | Appears in | Provenance | Evidence |
|---|---|---|---|
| Fpocket 0.228/0.444 … | EquiPocket, VN-EGNN, GDEGAN | EquiPocket's own run of the published fpocket tool; copied by the other two | EquiPocket A.4.5 "we use its published tool"; VN-EGNN fn b; GDEGAN fn b — [P] |
| P2Rank 0.366/0.628 … | EquiPocket, GDEGAN | **EquiPocket retrain on 792 scPDB samples**; copied by GDEGAN | EquiPocket A.4.6 + Table 5; GDEGAN fn b — [P] |
| P2Rank 0.464/0.728 … | VN-EGNN | VN-EGNN's own run, released model | VN-EGNN fn c — [P] |
| DeepSite 0.564 / 0.456 | EquiPocket, VN-EGNN, GDEGAN | Originates in **DeepSurf's** table (which itself says the DeepSite numbers came from another paper); copied twice more | EquiPocket A.4.6 "The results of DeepSite come from (Mylonas et al., 2021)"; DeepSurf: "The provided DeepSite results are those obtained by [14]" — [P] |
| Kalasanty 0.335/0.636 …, DeepSurf 0.386/0.658 … | EquiPocket, VN-EGNN, GDEGAN | EquiPocket's re-run of the **published pretrained models**; copied twice | EquiPocket A.4.5/A.4.6 — [P] |
| RecurPocket 0.354/0.593 … | EquiPocket, GDEGAN | EquiPocket's run of RecurPocket's **kalasanty-branch** pretrained model ("To ensure a fair comparison, we chose to test with the former pretrained model of RecurPocket (kalasanty)") — i.e. not the DeepPocket branch | EquiPocket A.4.6 — [P] |
| DeepPocket 0.399/0.645 … | VN-EGNN | VN-EGNN's own run | VN-EGNN fn c — [P] |
| GAT/GCN/GCN2/SchNet/EGNN | EquiPocket, VN-EGNN, GDEGAN | EquiPocket's own PyG implementations; copied twice (with one transcription error in GDEGAN's GAT row) | EquiPocket Table 4 code sources — [P] |
| GotenNet 0.464/0.624 … | GDEGAN | GDEGAN's own run (two values transposed between its Tables 1 and 2) | GDEGAN Tables 1–2 — [P] |
| DeepSurf 72.1 / 73.3 (COACH420) | DeepSurf paper | DeepSurf's own run under its own protocol — **not comparable** to the 0.658 attributed to DeepSurf in family C | DeepSurf Table 2 — [P] |
| P2Rank 72.0 / 78.3 | P2Rank 2018 | P2Rank's own run | PMC6091426 Table 3 — [P] |
| GrASP 77.5 % / P2Rank 74.9 % | GrASP | GrASP's own runs on its own Mlig+ sets | PMC11182664 — [P] |
| LIGYSIS recall table | Utgés & Barton 2024 | **All 13 methods re-run independently by the benchmark authors** — the only fully independent re-run of this scale | PMC11552181 — [P] |
| DeepLBS | — | Excluded by EquiPocket: "did not provide pretrained models or public codes, so it was not included in the baselines" | EquiPocket A.4.6 — [P] |

EquiPocket Table 4, "Sources of baseline codes and pre-train models" (verbatim list): Fpocket
`github.com/Discngine/fpocket`; kalasanty `gitlab.com/cheminfIBB/kalasanty`; DeepSurf
`github.com/stemylonas/DeepSurf`; RecurPocket `github.com/CMACH508/RecurPocket`; P2rank `github.com/rdk/p2rank`;
GAT/GCN/SchNet from PyTorch Geometric; GCN2 `github.com/chennnM/GCNII`; EGNN `github.com/vgsatorras/egnn`. — [P]

### Inferences

- The field has a single-source dependency: a 12-row baseline block produced once, in 2023, by the EquiPocket
  authors, propagated through at least two subsequent "state of the art" papers, and now sitting behind a withdrawn
  arXiv submission. Any new paper reporting against that block inherits both the retrained-P2Rank artefact and the
  absence of leakage control.
- GDEGAN's own protocol statement (per-chain HOLO4K splitting and aggregation) is **not** the protocol under which
  the copied HOLO4K baselines were produced — EquiPocket never states per-chain splitting — so GDEGAN's HOLO4K
  column mixes two receptor definitions in one table.

### Gaps

- I could not read the J Cheminform journal version of VN-EGNN (s13321-025-01127-9); Springer requires
  authentication through all three URL forms I tried. The arXiv version's tables are what I report; whether the
  journal version re-ran more baselines is **unverified**.

---

## Q5. Code, trained weights and licences

### Takeaway

P2Rank, GrASP, DeepPocket, DeepSurf and PUResNet all ship code and weights under permissive or copyleft licences.
VN-EGNN ships code and weights with data on Zenodo. **EquiPocket has no repository URL in its paper, and GDEGAN has
no code or weights at all** — the two methods defining the family-C numbers are the two that cannot be re-run.

### Cited Findings

| Method | Repository | Weights | Licence | Status |
|---|---|---|---|---|
| **P2Rank** | [github.com/rdk/p2rank](https://github.com/rdk/p2rank) | Shipped inside the binary release (default model + `alphafold` config model); current release **2.5.1**, with a 2.6-alpha adding pocket-grid/per-pocket descriptor exports and cofactor-as-surface handling | **MIT** ("Copyright (c) 2017-2025 Radoslav Krivák, David Hoksza, Lukáš Jendele, Petr Škoda and other contributors") — verified in `LICENSE.txt` | [P] README + LICENSE.txt read |
| P2Rank datasets | [github.com/rdk/p2rank-datasets](https://github.com/rdk/p2rank-datasets) | n/a — contains the `.ds` dataset definitions, the `(mlig)` relevant-ligand variants, and stored predictions by fpocket v1.0, SiteHound, MetaPocket 2.0, DeepSite and P2Rank 2.0 | not stated in README | [P] README read |
| **GrASP** | [github.com/tiwarylab/GrASP](https://github.com/tiwarylab/GrASP) | Trained model **in the repo**; datasets on Zenodo — [scPDB 15571599](https://zenodo.org/records/15571599), [COACH420 15572019](https://zenodo.org/records/15572019), [HOLO4K 15571950](https://zenodo.org/records/15571950); plus a Colab web interface | **MIT**, "Copyright (c) 2024 Tiwary Research Group" | [P] README + LICENSE read |
| **VN-EGNN** | [github.com/ml-jku/vnegnn](https://github.com/ml-jku/vnegnn) | README: "This repository contains all code, instructions and **model weights** necessary to run the method or to retrain a model"; processed datasets on [Zenodo 17365855](https://zenodo.org/records/17365855); runs logged via W&B | README has a `## License` section but **no `LICENSE` file at the repository root** (raw fetch returned 404) — licence text not read | [P] README read; licence [S]/unverified |
| **DeepPocket** | [github.com/devalab/DeepPocket](https://github.com/devalab/DeepPocket) | Checkpoints, prepared types and molcache on a **Pitt SharePoint link** (not Zenodo, not in-repo), including `first_model_fold1_best_test_auc_85001.pth.tar`, `seg0_best_test_IOU_91.pth.tar`, `refined_best_test_IOU_88.pth.tar`, and `SC6K.tar.gz`; points to p2rank-datasets for COACH420/HOLO4K | **MIT**, "Copyright (c) 2021 Deva Lab" | [P] README + LICENSE read |
| **DeepSurf** | [github.com/stemylonas/DeepSurf](https://github.com/stemylonas/DeepSurf) | Pretrained models distributed with the repo (EquiPocket and others used "their published pre-train models") | **GNU AGPL v3** — verified from `LICENSE` header. Note this is the most restrictive licence among the methods surveyed | [P] LICENSE read |
| **RecurPocket** | [github.com/CMACH508/RecurPocket](https://github.com/CMACH508/RecurPocket) | Checkpoint paths referenced in README commands (`ckpt_kalasanty/baseline_model/1050_best_model.pth`, `ckpt_kalasanty/kalasanty-ite2/i2-recurrent-1300.pth`, `ckpt_deeppocket/coach420_best_test_IOU_44.pth.tar`, `ckpt_deeppocket/seg0_best_test_IOU_91.pth.tar`); the README's data paths are the authors' absolute cluster paths | **no LICENSE file** at repo root (404) | [P] README read |
| **PUResNet** | [github.com/jivankandel/PUResNet](https://github.com/jivankandel/PUResNet) | Weights `whole_trained_model1.hdf` in-repo via **git LFS** — README warns "Since this was uploaded with git lfs, git clone won't download the full file" and you must download it manually | **MIT**, declared in the README `## License` section; no separate root `LICENSE` file | [P] README read |
| Kalasanty | [gitlab.com/cheminfIBB/kalasanty](https://gitlab.com/cheminfIBB/kalasanty) | Published pretrained model (used by EquiPocket) | not checked | [S] via EquiPocket Table 4 |
| fpocket | [github.com/Discngine/fpocket](https://github.com/Discngine/fpocket) | n/a (geometric, no weights) | not checked | [S] via EquiPocket Table 4 |
| **EquiPocket** | **no repository URL given in the paper**; Table 4 lists only baseline URLs | absent | n/a | [P] arXiv:2302.12177v3 read; no code statement found |
| **GDEGAN** | none | absent — "**We will release the full code based on the acceptance of the work**" | n/a | [P] arXiv:2603.19817 read |
| LIGYSIS benchmark results | [Zenodo 14645504](https://zenodo.org/records/14645504) "LBS-Comparison results" | n/a (results, not weights) | not checked | [P] linked from the paper's search record |

### Inferences

- Reproducibility inverts the leaderboard: the methods with the highest family-C numbers (GDEGAN, EquiPocket) are
  the least reproducible, while the method used as the whipping boy (P2Rank, MIT, versioned releases, published
  datasets and stored competitor predictions) is the most reproducible.
- DeepSurf's AGPL-3.0 is a practical constraint for anyone embedding it in a pipeline that is itself distributed;
  all other weight-bearing methods here are MIT.
- DeepPocket's weights behind a personal SharePoint link are the most fragile hosting arrangement found; a dead link
  would make its family-C′ numbers (0.399/0.645 …) permanently unreproducible.

### Gaps

- VN-EGNN's licence text was not read (no root `LICENSE` file; the README's License section content was not
  retrieved).
- RecurPocket has no licence file, so its redistribution status is undetermined.
- GitHub's REST API was unreachable from this environment (empty responses), so I could not confirm SPDX licence
  identifiers, last-push dates or weight file sizes programmatically; the licence findings above come from reading
  `LICENSE`/`README.md` over raw.githubusercontent.com.
- PointSite, SiteRadar and YuelPocket repositories were not located; no code/weights/licence information found.

---

## Cross-cutting caveats the report writer should carry

- **[P]** The EquiPocket arXiv submission is **withdrawn** (v4/v5, August 2026) for reasons the authors give as
  "the technical elaboration and experimental design in the current manuscript require substantial restructuring
  and revision". Its numbers are still the de-facto baseline table of the field.
  — [arXiv:2302.12177](https://arxiv.org/abs/2302.12177)
- **[P]** Three different HOLO4K populations circulate: 4009 (P2Rank/PrankWeb papers and DeepSurf), 4,288 (GDEGAN),
  ~3,204 proteins (implied by EquiPocket's site histogram), with 4543 files present in the dataset directory.
  HOLO4K numbers across papers are therefore not on the same denominator.
- **[P]** Family C reports top-n only; families A/A′/D report top-n and top-(n+2). LIGYSIS argues the 4 Å DCC
  convention is wrong in principle and recommends 10–12 Å with top-N+2.
- **[P]** Only DeepSurf (90 % global sequence identity filter → 9444 training targets) and GrASP (UniProt-ID
  10-fold split) document train/test similarity control. For EquiPocket, VN-EGNN and GDEGAN the answer is
  "not stated".
- **[P]** DeepSurf states explicitly that its competitors benefit from leakage it removed for itself — so even
  within family B the comparison is asymmetric by the authors' own admission.
