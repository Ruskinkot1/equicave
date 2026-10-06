# Paper plan

**Working title.** *EquiCave: a self-contained equivariant multi-task model of binding pockets — ranked sites,
pocket properties and ligand-atom hotspot fields, with a peptide-groove tier and homology-controlled evaluation.*

Nothing in this plan is a result unless it appears in `docs/results/` with its protocol. Numbers taken from other
papers are never compared directly to ours: every baseline is re-run on our splits.

## Contributions claimed
1. **A pocket model with no external dependencies.** Candidates, features, network and ranker all come from the
   structure alone; no P2Rank, fpocket or Java at training or inference. The native generator reaches a **0.977**
   candidate ceiling on 1367 RCSB structures at 30 candidates per structure.
2. **The ablation the field does not have, with a null result for degree 2.** Degree-0/1/2 Cartesian channels with
   invariant-gated tensor messages, plus an e3nn backbone to l=3, ablated at equal depth, width and features on one
   protocol. Applying l=2 tensor attention to pockets is prior art (GDEGAN 2026); what no paper provides is the
   ablation. **Measured (2026-10-06, 3 seeds, seed sd 0.009): removing the degree-2 channels costs −0.006 site
   top-1, and the arm is not even parameter-matched but smaller, which can only flatter degree 2.** Chirality
   likewise costs +0.000. So the contribution is the measurement, not a gain: a method the field has adopted for
   this task does no measurable work on it, which the nearest adjacent result (EquiPNAS on protein–nucleic acid,
   negligible equivariant gain) makes plausible rather than surprising. Degree-1 vector channels do work (+0.037).
   The overall equivariance figure (+0.040) is **not yet reportable**: that arm removes the steerable channels, the
   geometry from the input and the equivariant centre head at once. `invariant_frames`, which scalarises the same
   geometry in a local frame, isolates it and has still to be run.
3. **Probes in real cavities are the architecture, and we can say so with a number.** Removing them costs
   **+0.263 site top-1** — an order of magnitude more than any other component, and the single largest effect in
   the whole ablation. VN-EGNN puts its virtual nodes on a sphere around the protein; ours start on free lattice
   points of detected cavities, so every probe begins where a ligand atom could sit. Following the measurement, the
   probes are now placed by a learned per-point ligandability model out of fold rather than at random within
   buriedness tiers, with the fraction of the budget spent that way as the knob (`probe_tiered` is the comparison
   arm). The SAS surface module, bundled with this claim before it was measured, costs −0.011 and is reported as
   doing no measurable work.
4. **One hotspot field** of per-class ligand-atom probabilities per lattice point, labelled from the wwPDB CCD, shared
   with the property head.
5. **Two methodological results about measuring pocket detection**, both of which cost us numbers we had already
   written down. (a) Evaluating on the chain a benchmark row names, while training on the whole assembly, destroys
   every context-dependent feature and makes a rich ranker score below no ranking at all; the baselines are given
   the assembly, so it also biases the comparison. Under the corrected protocol our best model is not
   distinguishable from P2Rank on top-1, top-N or top-(N+2) (all three paired intervals include zero). (b) Splitting
   folds by 30 %-identity cluster controls sequence similarity but does not make a fold a sample of a benchmark:
   across three feature sets, cross-validated top-1 and benchmark top-1 move in **opposite** directions, so a
   feature group cannot be accepted on cross-validation alone. Both are reported as findings because every
   published comparison in this field faces them.
6. **A diagnosis of where pocket detection actually fails, and a per-point remedy for it** (the remedy raises
   cross-validated ranking and does not transfer -- see (b) above). On the COACH420
   structures that are not similar to our training set, our candidate set contains the right answer for 0.992 of
   them against P2Rank's 0.935, and we still lose top-N: we convert 72 % of our ceiling into a correct first
   prediction where P2Rank converts 82 %. Three candidate explanations are measured and two are refuted --
   prediction fragmentation cannot be it (210 of 283 structures have one site, where top-N is top-1 and merging
   changes nothing, and that is where the deficit is largest), and mis-centring cannot be it (34 of 53 failing first
   predictions are more than 8 A from the ligand, a different pocket rather than a near miss). What remains is how
   a candidate is scored as a whole, and the remedy is a per-point ligandability model: out-of-fold AUROC 0.865 over
   2.3 M cavity grid points, whose single best aggregate ranks candidates at 0.741 top-1 against the geometric
   score's 0.473. A second-stage cascade over the top candidates, the other obvious remedy, is also measured and
   also fails -- reported, because the negative results are part of the contribution.
7. **Peptide-binder sites as a first-class problem**: a groove candidate tier with backbone-exposure features and a
   homology-controlled peptide-site benchmark built from RCSB (973 complexes, 685 receptor clusters, disjoint from the
   small-molecule training clusters). First result: cavity candidates already reach a high ceiling on peptide sites, so
   the limiting factor is ranking, not detection.

## Claims and their falsification tests
| id | claim | test | fails if |
|---|---|---|---|
| C1 | native candidates + learned ranker match or beat fpocket/P2Rank candidates + the same ranker, with no external tool | same 1367 structures, same labels, cluster CV, paired cluster bootstrap | CI of the difference in top-1 or top-(N+2) lies below 0 |
| C2 | network features raise the ranker | out-of-fold `net_*` features added, paired bootstrap | CI includes 0 |
| C3 | degree-2 tensor channels beat degree ≤ 1 and the invariant model | ablation grid, 3 seeds, equal depth/width | **failed**: full − no_tensors = −0.006 at seed sd 0.008, on an arm smaller than full. Degree 2 does no measurable work; chirality +0.000. Reported as a null result | full − no_tensors CI includes 0 |
| C4 | probes on cavity points beat no probes (and a sphere-probe variant) | ablation | CI includes 0 |
| C5 | property classes and the hotspot field beat geometry-only and tabular baselines on held-out clusters | per-class AUROC/AP, ECE, enrichment of ligand atoms in top-k % points, permutation control | no gain |
| C8 | interaction-validated hotspot labels train a better field than proximity labels | the same network trained on `y_hot` and on `y_hot_proximity`, both scored against the validated target | no difference |
| C6 | peptide-specific features (groove shape + backbone exposure) improve peptide-site ranking | peptide benchmark, feature-group ablation | CI of the gain includes 0 |
| C9 | the per-point ligandability aggregates raise the ranker beyond the 118 pocket features | the group added, feature-group ablation and paired cluster bootstrap, then COACH420 with and without it | **failed on the benchmark**: +0.019 cross-validated top-1, -0.016 on COACH420; the group carries information and does not transfer |
| C10 | feature groups accepted on cross-validated top-1 transfer to a benchmark | every model evaluated on COACH420 under the matched receptor protocol | **failed**: 32 features 0.701, 236 0.642, 272 0.626 on the benchmark while cross-validation ranks them in the opposite order |
| C7 | the model generalises across benchmarks | COACH420, HOLO4K, LIGYSIS, CryptoBench, held-out families, one protocol, train-similar structures separated | macro-average below the re-run baselines |

## Protocol requirements adopted from the 2026-10-04 survey
The survey (`docs/LITERATURE_SURVEY.md`) documents ten flaws in how this field reports results. Our tables must
therefore carry, without exception: top-N **and** top-(N+2); DCC at **4, 10 and 12 Å** as well as DCA at 4 Å; a
**redundancy statistic** and the non-redundant variant; the **ligand rule** with the resulting system and ligand
counts; the explicit **chain / biological-unit treatment** for HOLO4K; the homology threshold against every test set
**and** the measured train/test cluster overlap; the failure rate; and the **candidate recall ceiling**. Every
baseline is re-run in house; no published row is copied into a table of ours. Where a competing method reports only
top-N, or only DCC at 4 Å, that is stated next to its number.

## Data and splits
`docs/DATA_CARD.md`. Training: RCSB (CC0), 30 %-identity clusters, 5 cluster folds, 12 held-out families excluded by
cluster and UniProt. Evaluation: held-out families, COACH420, HOLO4K (DeepPocket ligand rule), LIGYSIS, CryptoBench,
and the peptide benchmark. Structures sharing a 30 % cluster with training are reported separately as "train-similar".

## Experiments
- **E1 main table.** DCA and DCC at 4 Å, top-1 / top-3 / top-N / top-(N+2), MRR, ceiling: detector order, geometry
  baselines, LambdaRank, LambdaRank + network, network alone, fpocket, P2Rank. Cluster bootstrap CI, ≥ 3 seeds.
- **E2 ablations.** `full`, `no_probes`, `no_surface`, `no_esm`, `no_tensors`, `no_vectors`, `invariant`, `achiral`;
  and feature-group ablations of the ranker (already measured for the ranker: chemistry is the largest group).
- **E3 properties and hotspots.** Per-class AUROC/AP, ECE, reliability curves; enrichment of true ligand atoms in the
  top-k % of field points with a permutation control.
- **E4 peptide sites.** Ceiling by tier, ranking with and without the peptide feature group, per-receptor-family breakdown.
- **E5 failure analysis.** Apo vs holo, cryptic sites, multi-site proteins, large ligands, membrane proteins, metals.
- **E6 robustness.** Predicted (AlphaFold) structures, protein-only input, rotation averaging at test time.
- **E7 cost.** Seconds per structure and memory for each stage (the native generator is 3.6 s per structure per core).

## Figures and tables
F1 pipeline and the three outputs. F2 the equivariant layer (degrees, messages, gates). F3 split and leakage design.
F4 main curves. F5 ablation bars. F6 qualitative pockets with hotspot fields, including one peptide groove.
F7 failure cases. T1 data. T2 main metrics. T3 ablations. T4 peptide benchmark. T5 licences and scope.

## Current status of the tables
| table | status |
|---|---|
| T1 data | done (1499 small-molecule entries / 1085 clusters; 973 peptide complexes / 685 receptor clusters; 12 held-out families verified against the deposited files) |
| T2 main | partial: native candidates (ceiling 0.977) + ranker (top-1 0.724 with 32 features) + geometry baselines + P2Rank (top-1 0.754, ceiling 0.914) measured on the same 1392 structures; 109-feature ranker rebuilding; network **not trained** |
| T3 ablations | ranker feature groups measured on the corrected 118-feature table (chemistry −0.039, shell −0.033, geometry and potential −0.021 each, native −0.006, context −0.001 top-1); network ablations need a GPU |
| T6 negative results | done: prediction merging, re-centring (twice), the cascade re-ranker — each with the measurement that refutes it (`docs/results/README.md`) |
| T4 peptide | 973 complexes built; candidate tiers measured on 400 of them (ceiling 0.952, groove tier alone 0.781, cavity tiers 0.939); ranker pending |
| T5 licences | done (`docs/DATA_CARD.md`, `docs/PROVENANCE.md`) |

## Venues
J. Cheminformatics or J. Chem. Inf. Model. (methods); Bioinformatics (application note); an ML4Science workshop for the
equivariance ablation alone. Preprint once C1 and C3 are reproduced with ≥ 3 seeds.

## Honesty rules
Report what was not run. Never publish numbers from an unconverged run. Scores are computational hypotheses, not
measured affinity or activity. If C2 or C3 fail, the paper is C1 + C5 + C6 with a negative-result section on
high-degree channels — which is itself a contribution, since the ablation is missing from the literature.
