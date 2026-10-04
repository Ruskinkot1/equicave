# Results

Every file here is written by a script and carries its protocol in the header. Nothing is edited by hand.
A number that is absent is absent on purpose: the run did not happen, or did not converge.

| file | produced by | content |
|---|---|---|
| `native_candidates_build.log` | `scripts/train/build_native.py` | candidate ceiling, mean candidates, timing over the whole manifest |
| `ranker_native.md` / `.json` | `scripts/train/train_ranker.py` | LambdaRank cross-validation by cluster, seeds, feature-group ablations, paired bootstrap |
| `ranker_peptide.md` / `.json` | the same, `--tag peptide` | the peptide-site benchmark, with and without the peptide feature group |
| `methods_comparison.md` / `.json` | `scripts/eval/compare_methods.py` | native, fpocket and P2Rank candidates on the same structures, each with and without the learned ranker |
| `eval_<set>.md` / `.json` | `scripts/eval/evaluate.py` | one benchmark under the shared protocol; `train-similar` structures are a separate row |
| `ablations.md` / `.json` | `scripts/train/collect_ablations.py` | the network ablation grid, mean ± seed sd, paired against `full` |
| `compare_<set>.md` / `.json` | `scripts/eval/compare_on_set.py` | us against external tools on one benchmark, same structures, split by site count |

Conventions in every table: success is DCA ≤ 4 Å (DCC reported where available); N is the structure's own number of
ligand sites; intervals are 95 % cluster bootstraps; gains carry a paired cluster bootstrap; "not run" means not run.

## Open discrepancy: the cleaned-subset number does not transfer to COACH420

The ranker trained on the 432 cleaned structures reaches 0.806 top-1 in its own cross-validation (0.810 calibrated)
and **0.637** on the 179 comparable COACH420 structures -- below the detector's own ordering there (0.642), and
below the 0.704 that an older model with 32 features reached on the same subset
(`docs/results/eval_coach420_n3.md` against `docs/results/eval_coach420.md`). Two things changed at once between
those two models, so the gap is not yet attributable: the feature count (32 to 236 with z-scores) and the training
set (all 1367 manifest structures to the 432 that share no 30 %-identity cluster with any benchmark). A model with
236 columns fitted on 432 structures may simply be data-starved, in which case the cleaned-subset
cross-validation is measuring a model that cannot be served.

A silent serving bug is ruled out: the 118 features assembled at inference were compared against the stored
training table for 180 candidates of six structures and agree exactly (maximum absolute difference 0 across every
column), so the model is served the values it was trained on.

The remaining candidate cause, besides training-set size, is in the protocol itself. The cleaned subset is not a
random subsample: it is **selected** as the structures sharing no 30 %-identity cluster with any benchmark, so a
cross-validation inside that pool trains and tests on families chosen for being unlike the benchmarks, and measures
generalisation within an unusual pool rather than to a benchmark. If that is the explanation, then 0.806 was never a
benchmark prediction and 0.637 is the honest out-of-family number -- and the protocol that gives both an honest and
a usable figure is to train on the whole manifest and report the benchmark structures that are not train-similar,
which is leakage-free because those structures are absent from training by construction.

Until that is separated, **the cleaned-subset top-1 figures should not be read as benchmark performance.** The runs
that separate them -- the same 272-column feature set trained on all 1367 structures, with and without the
per-point group, evaluated on the same COACH420 subset -- are in progress.

## Negative results worth keeping

**A cascade re-ranker over the top candidates is worse than one model over all of them** (measured 2026-10-04 on the
cleaned 432 structures, `docs/results/clean3/`). The diagnosis invited it: when our first prediction is wrong the
correct candidate is ranked second in 24 of 53 COACH420 cases and within the first five in 41 of 53, so a second
stage trained only on the top k -- where typically exactly one candidate is correct and the whole gradient is the
choice that decides top-1 -- should have sharpened exactly that decision. It does the opposite. Against the
first stage's 0.796 top-1 (seed ensemble), re-ranking the top 3 gives 0.771 and the top 5 gives 0.766, with top-(N+2)
unchanged at 0.919. The restriction throws away about 97 % of the rows (432 structures x 3 instead of 12 955
candidates) and with them the easy negatives that place the hard pair on a scale, and a few thousand rows cannot
support a 236-column model. The implementation is kept behind `--cascade` because the same idea should be retried
once the dataset is an order of magnitude larger, where the row count stops being the binding constraint.

**Moving a candidate's centre cannot fix a wrong first prediction** (measured 2026-10-04). Of the 53 non-train-similar
COACH420 structures whose first prediction is wrong, 4 are within 5 A of the ligand and 34 are more than 8 A away:
the first prediction is usually a different pocket, not a near miss. This is the second negative result for
re-centring, after the earlier one showing six centre definitions all within noise for DCC.

**Merging our predictions cannot close the gap to P2Rank on COACH420** (measured 2026-10-04,
`docs/results/compare_coach420.md`). We emit 30 predictions per structure to P2Rank's 9.6 and lose top-N by
0.086 [0.006, 0.169] on the 174 structures that are not similar to our training set, so the obvious reading was
that fragments of one pocket consume the top-N budget and that merging nearby predictions would recover it.
Splitting the same comparison by the structure's own number of sites refutes it: 210 of the 283 structures have
N = 1, where top-N is top-1 and no merging of predictions can change the result, and that is exactly where the
deficit is largest (-0.105 on the comparable subset, against -0.093 at N = 2 and +0.286 at N >= 3). The gap is the
ranker choosing the wrong candidate first, not fragmentation.

The same table sizes the headroom: on those structures our candidate set contains the answer for 0.992 of them
against P2Rank's 0.935, and we convert 72 % of that ceiling into a correct first prediction where P2Rank converts
82 %. Detection is not the bottleneck and has not been for some time; first-rank accuracy is, and the designed
remedy for it -- the network's per-probe segmentation and confidence, which scores points inside the cavity rather
than summarising the pocket with features -- has still never been trained.

**Re-centring a candidate does not improve DCC** (measured 2026-10-04 on 111 hitting candidates). Six definitions of
a candidate's centre — buriedness-weighted centroid (the one we use), plain centroid, the peak point, the centroid of
the deepest half, an interaction-potential-weighted centroid, and the centroid of points that can make three or more
interactions — all give DCC ≤ 4 Å between 0.505 and 0.577 and DCC ≤ 10 Å of 0.982, with DCA ≤ 4 Å essentially 1.0.
The distance from any reasonable centre to the ligand *centroid* is about 3.6 Å because the cavity point cloud is
larger than the ligand, so the low DCC at a 4 Å threshold is a property of the threshold, not of our centres. This
supports the LIGYSIS recommendation to report DCC at 10-12 Å, and it closes re-centring as an avenue for improving
that metric.
