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
   score's 0.473. A second-stage cascade over the top candidates, the other obvious remedy, is measured twice and
   fails twice, and the second measurement closes it rather than deferring it. The first attempt blamed the row
   count; the retry at 3.2x the data (1367 structures, 40 986 candidates, 4 seeds) swept the column budget that
   explanation names and every one of six configurations stays below the single stage, 0.775 to 0.786 against
   **0.792 [0.770, 0.814]** top-1. What replaces the explanation reconciles the result with the literature instead
   of contradicting it: the published cascade gain (+7 to +14 Top-n, PRANK over fpocket, three independent
   confirmations) comes from putting a learned second stage over a **geometric** first stage whose candidate
   ceiling is 80.78 % on COACH420, while ours is already a learned ranker over a list with a ceiling of 0.977. The
   intervention is "geometry to learned", not "learned to learned twice", and this project is on the far side of
   it. Reported, because the negative results are part of the contribution.
7. **Peptide-binder sites as a first-class problem**: a groove candidate tier with backbone-exposure features and a
   homology-controlled peptide-site benchmark built from RCSB (973 complexes, 685 receptor clusters, disjoint from the
   small-molecule training clusters). First result: cavity candidates already reach a high ceiling on peptide sites, so
   the limiting factor is ranking, not detection.

8. **The object a user receives, not the row a benchmark scores** (`src/equicave/site_report.py`). A benchmark
   needs one number per site: did the centre land within 4 A. Someone deciding whether to run a screen needs the
   docking box, the maximum inscribed radius, the lining residues, the ligand-type guess and a **spatially
   resolved pharmacophore map** -- seven hotspot classes per probe rather than one label per pocket. The literature
   review (`reports/`) establishes what this is and is not: the output type is **not** novel as of 2026, since
   PharmRL, PharmacoNet and PharmacoForge all emit typed pharmacophore fields with code released, one of them dense
   at 0.5 A with per-class ROC-AUC 0.951-0.982. What is unoccupied is the **evaluation**: prior art is validated on
   122 structures (SuperStar, 2001) and 21 fragments (Fragment Hotspot Maps, 2016), nobody reports per-type
   confusion or calibration, and no benchmark for map accuracy exists. So the map is presented as table stakes and
   the contribution is the measurement protocol, not the field.
9. **Three measurement results the field does not have, each cheap and each absent from every paper surveyed.**
   (a) **Checkpoint selection.** No paper in this literature ablates the selection criterion, none uses EMA and
   none uses staged training with frozen loss terms; we do all three, and selecting blind to the task once reported
   top-1 0.022 for a model that reached 0.868, which is the strongest argument for stating the criterion at all.
   (b) **Calibration under a non-linear aggregate.** Temperature scaling cannot reorder a structure's candidates --
   it is monotone -- so calibration is cosmetic for a ranking by maximum or mean. It is not cosmetic for P2Rank's
   rule, the **sum of squares** of per-point ligandability, which is not invariant under a monotone
   reparameterisation; that is our default aggregator, and the effect on ranked success is measured rather than
   assumed. A corollary found in our own ranker table: an isotonic fit's apparent +0.007 at top-(N+2) is a
   tie-breaking artefact, because a step function ties previously distinct scores and the stable sort then breaks
   them by the detector's own order. (c) **Stratified reporting by pocket chemical class.** No paper stratifies a
   site-prediction ablation this way, and the aggregate hides the finding: DeepDrug3D loses ~0.13 on nucleotide
   pockets and nothing on haem, and our own metal cue is a 7.3x enrichment that yields an AUC of 0.544.
10. **Chemistry added as pre-registered arms with their upside bounded first.** The review found that the three
   most-used voxel site models carry a partial-charge channel and **none of them ablates it**, that dMaSIF removed a
   Poisson-Boltzmann solve and *gained* accuracy at ~1/1000 of the cost, and that no metal input channel has ever
   been ablated by any site predictor. Our four additions (metals with occupancy as confidence, a screened-Coulomb
   field entering as a degree-1 channel, per-probe MSA conservation, long-range protrusion) are therefore shipped
   **off by default**, each with its own arm, its own feature cache and a prediction written before it ran
   (`docs/results/PREREGISTERED.md`). Two were bounded by our own measurement before any GPU time was spent: the
   metal cue is 7.3x enriched at correct candidates but caps near 0.034 top-1 because 87 % of correct candidates
   have no metal in range, while the long-range counts are **not** redundant with our closure field (single-count
   correct-against-decoy AUC 0.650 at 8 A against 0.837 at 12 A; a logistic model on our existing geometric inputs
   goes 0.7999 to 0.8469). The methodological point is the discipline: a channel's headroom is measured before it
   is added, because three of our own ablations have already returned nulls from redundancy.

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
| C11 | the chemistry channels raise site ranking | each arm against `full` on the current commit, 3 seeds, paired cluster bootstrap, +0.02 threshold fixed in advance | CI of the gain includes 0 -- predicted for `probe_electrostatic`, which may re-encode the ionisable-group distances already in `probe_potential` |
| C12 | calibration changes the ranking under a sum-of-squares aggregate | the same predictions ranked by sum of squares with and without the fitted temperature, on structures the temperature was not fitted on | the delta is zero, which would say the head is already calibrated enough for the aggregate not to care -- also a result |
| C13 | a site-prediction result is heterogeneous across pocket chemical classes | per-class top-1 and DCC over the 14 property classes, counts reported and no metrics claimed below ten structures | the per-class spread is within the seed spread, i.e. the aggregate was sufficient after all |
| C7 | the model generalises across benchmarks | COACH420, HOLO4K, LIGYSIS, CryptoBench, held-out families, one protocol, train-similar structures separated | macro-average below the re-run baselines |

## Protocol requirements adopted from the 2026-10-04 survey
The survey (`docs/LITERATURE_SURVEY.md`) documents ten flaws in how this field reports results. Our tables must
therefore carry, without exception: top-N **and** top-(N+2); DCC at **4, 10 and 12 Å** as well as DCA at 4 Å; a
**redundancy statistic** and the non-redundant variant; the **ligand rule** with the resulting system and ligand
counts; the explicit **chain / biological-unit treatment** for HOLO4K; the homology threshold against every test set
**and** the measured train/test cluster overlap; the failure rate; and the **candidate recall ceiling**. Every
baseline is re-run in house; no published row is copied into a table of ours. Three requirements were added on
2026-10-08: every homology-filtered subset carries its **composition** beside its accuracy (structures, clusters,
mean sites, single-site fraction, site-count histogram), because an earlier version of this analysis read a
closing gap as leakage when it was a site-count shift; the **hard-novelty** subset drops structures homologous to
our training clusters *and* to the pre-2017 ligand-bound superset the scPDB-trained competitors were drawn from
(17 980 clusters, a superset that over-removes and never under-removes); and every ablation table is reported
**stratified by pocket chemical class** as well as in aggregate. Where a competing method reports only
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
- **E8 chemistry arms.** `no_probe_potential` first -- it asks whether the chemistry already in the model does any
  work -- then `probe_protrusion`, `probe_metal`, `probe_electrostatic`, `probe_chemistry` and, where an alignment
  can be produced, `probe_conservation`. Each as an addition to `full`, 3 seeds, against the +0.02 threshold.
  Featurisation cost measured: +0.07 s per structure for the metals, nothing measurable for the field or the
  protrusion counts, against 0.71 s for the rest of the featuriser.
- **E9 calibration.** Temperature fitted on a hash-split half of the validation structures and reported on the
  other half: the occupancy head's ECE before and after, the reliability curve, and the sum-of-squares site
  ranking recomputed with and without the temperature. A boundary-valued temperature is flagged as a boundary, not
  published as a fit.
- **E10 stratified reporting.** Every ablation and benchmark table split by the 14 property classes, with counts
  only below ten structures per class, plus the metal arm reported on the metal stratum specifically.

## Figures and tables
F1 pipeline and the three outputs. F2 the equivariant layer (degrees, messages, gates). F3 split and leakage design.
F4 main curves. F5 ablation bars. F6 qualitative pockets with hotspot fields, including one peptide groove.
F7 failure cases. T1 data. T2 main metrics. T3 ablations. T4 peptide benchmark. T5 licences and scope.

## Current status of the tables
| table | status |
|---|---|
| T1 data | done (1499 small-molecule entries / 1085 clusters; 973 peptide complexes / 685 receptor clusters; 12 held-out families verified against the deposited files) |
| T2 main | partial: native candidates (ceiling 0.977) + ranker (**top-1 0.792 [0.770, 0.814]** on 1367 structures, 204 features, 4 seeds) + geometry baselines + P2Rank + three deep baselines on COACH420 (DeepSurf 0.801 beats P2Rank 0.764, which retires this project's earlier claim that no learned method does); the network is trained but its `full` reference needs re-measuring, see the row below |
| T3 ablations | ranker feature groups measured on the corrected 118-feature table (chemistry −0.039, shell −0.033, geometry and potential −0.021 each, native −0.006, context −0.001 top-1). Network: 8 arms measured 2026-10-06 at 3 seeds (probes +0.263, equivariance +0.040, vectors +0.037, tensors −0.006, chirality +0.000, ESM-2 −0.010, surface −0.011). **That reference is void for new comparisons**: it predates the site decoder by three hours and predates the checkpoint-selection fix by a day, so `full` must be re-run on the current commit before any new arm is differenced against it (`docs/results/PREREGISTERED.md`, amendment b) |
| T6b cascade | **done and closed** (2026-10-08): six configurations over the top 3 and top 5 at three column budgets, all below the single stage; the stated cause (row-to-column ratio) retired and replaced by the ceiling argument |
| T7 chemistry arms | **not run.** Code, tests, per-arm caches, measured featurisation costs and written-down predictions are in place; `probe_conservation` additionally needs an alignment per target, which nothing in this repository produces |
| T8 calibration and stratification | **not run** as tables. Implemented in `net_task.evaluate` (`calibration`, `net_sites_by_class`) and in `scripts/eval/evaluate.py` (the hard-novelty subset and the per-subset composition block); no table carries them yet |
| T6 negative results | done: prediction merging, re-centring (twice), the cascade re-ranker — each with the measurement that refutes it (`docs/results/README.md`) |
| T4 peptide | 973 complexes built; candidate tiers measured on 400 of them (ceiling 0.952, groove tier alone 0.781, cavity tiers 0.939); ranker pending |
| T5 licences | done (`docs/DATA_CARD.md`, `docs/PROVENANCE.md`) |

## Venues
J. Cheminformatics or J. Chem. Inf. Model. (methods); Bioinformatics (application note); an ML4Science workshop for the
equivariance ablation alone. Preprint once C1 and C3 are reproduced with ≥ 3 seeds.

## Honesty rules
Report what was not run. Never publish numbers from an unconverged run. Never difference an arm against a
reference produced by different code or a different selection rule -- if the reference moves, say so and re-measure
it, rather than re-baselining quietly. Predictions for every arm are written down before the arm runs, and the
claim threshold is derived from this grid's own seed spread (16 arms at sd 0.009 put the expected maximum of pure
noise near +0.017, hence +0.02) rather than borrowed. Scores are computational hypotheses, not
measured affinity or activity. If C2 or C3 fail, the paper is C1 + C5 + C6 with a negative-result section on
high-degree channels — which is itself a contribution, since the ablation is missing from the literature.
