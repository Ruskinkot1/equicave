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
| `compare_<set><tag>.md` / `.json` | `scripts/eval/compare_on_set.py` | us against external tools on one benchmark, same structures and same receptor protocol, split by site count |

Conventions in every table: success is DCA ≤ 4 Å (DCC reported where available); N is the structure's own number of
ligand sites; intervals are 95 % cluster bootstraps; gains carry a paired cluster bootstrap; "not run" means not run.

## All five benchmarks under the corrected protocol (2026-10-05)

One model (`models/ranker_native.txt`, 32 features — the set that transfers best), one protocol, the full assembly
as the receptor. The row is the structures **not** sharing a 30 %-identity cluster with our training manifest,
except held-out, which is excluded from training by construction.

| benchmark | n | detector top-1 | + ranker top-1 | top-N | top-(N+2) | ceiling |
|---|---|---|---|---|---|---|
| held-out drug targets | 12 | 0.917 | 0.833 | 0.833 | 1.000 | 1.000 |
| HOLO4K | 1501 | 0.635 | **0.785** | 0.828 | 0.923 | 0.988 |
| COACH420 | 179 | 0.598 | 0.698 | 0.771 | 0.877 | 0.989 |
| LIGYSIS | 1108 | 0.433 | 0.610 | 0.709 | 0.821 | 0.941 |
| CryptoBench (apo) | 120 | 0.258 | 0.292 | 0.450 | 0.600 | 0.850 |

Read across it rather than down: the ranker is worth +0.15 to +0.18 top-1 on HOLO4K, COACH420 and LIGYSIS, and
almost nothing on CryptoBench (+0.034), where the pockets are closed. The candidate ceiling holds near 0.99 on the
holo benchmarks and falls to 0.850 on apo structures, which is the only set where detection, not ranking, is the
binding constraint. Held-out has twelve structures and its intervals are too wide to rank anything; the detector
beating the ranker there is one structure.

The two sets whose earlier numbers were invalidated by the single-chain receptor bug were LIGYSIS and CryptoBench,
since every row of theirs names a chain. HOLO4K names none, so its earlier numbers were never affected by it.

## Resolved: the benchmark numbers were measured on a receptor the model was never trained on

**Cause.** Every COACH420 entry names a chain and `evaluate.py` passed it to `read_pdb` as a receptor filter, while
the candidate table is built on every chain. A truncated receptor moves the protein centroid and radius of gyration,
changes the residue and candidate counts, empties the 5/8/12 A shells of a neighbouring chain's atoms, and unburies
every pocket at a chain interface -- so `centrality`, the 46 shell features, the native ranks and every
within-structure z-score were computed in a context the model had never seen. `scripts/baselines/run_external.py`
gives the external tools every chain, so the truncation also biased the head-to-head against us. Fixed:
`--receptor-chains` defaults to `all`, with `entry` kept to reproduce the old behaviour.

The signature that identifies the cause rather than merely correlating with it: under the mismatch every rich ranker
scored *below* doing no ranking at all, and the damage grew with how much context the feature set uses, while the
detector's own context-free score was identical in every run. Correcting the receptor reverses the ordering.

**COACH420, 174 structures not similar to our training manifest, both sides on the full assembly:**

| model | features | top-1 | top-N | top-(N+2) | ceiling |
|---|---|---|---|---|---|
| detector order | — | 0.598 | 0.713 | 0.822 | 0.989 |
| `ranker_native` | 32 | **0.701** | **0.776** | 0.874 | 0.989 |
| `ranker_native3_full` | 236 | 0.642 | 0.726 | 0.855 | 0.989 |
| `ranker_native3_full_points` | 272 | 0.626 | 0.704 | 0.855 | 0.989 |
| P2Rank 2.5.1 | — | 0.753 | 0.828 | 0.879 | 0.931 |

Paired cluster bootstrap of the 32-feature model against P2Rank: top-1 -0.052 [-0.121, +0.011], top-N
-0.052 [-0.121, +0.017], top-(N+2) -0.006 [-0.059, +0.046]. **All three intervals include zero**, so on the
corrected protocol we are not distinguishable from P2Rank on any of the three, against a significant top-N deficit
of -0.086 [-0.169, -0.006] under the broken one. At three or more sites we match it exactly (+0.000 on 40
structures).

## Second finding: our feature engineering does not transfer, and cluster-fold cross-validation cannot see it

The same table read down the feature column is the uncomfortable result. More features give monotonically **better**
cross-validation and monotonically **worse** benchmark transfer:

| features | cross-validated top-1 | COACH420 top-1 |
|---|---|---|
| 32 | — (not re-measured on this table) | 0.701 |
| 236 | 0.787 [0.764, 0.808] | 0.642 |
| 272 (+ per-point) | 0.796 [0.774, 0.818] | 0.626 |

The per-point ligandability group is the clearest case: it is the most valuable group in the ablation (+0.019 top-1
on the cleaned subset, +0.009 on the full manifest, worth more than the 46 shell columns) and it **costs** 0.016
top-1 on the benchmark. Splitting by 30 %-identity cluster controls sequence similarity between folds; it does not
make a fold a sample of COACH420. Our manifest is built to its own criteria -- resolution limits, ligand size and
type rules, a cap of entries per cluster -- so a rich model can exploit regularities that every fold shares and the
benchmark does not, and the cross-validation will not report it.

The consequence for the project: **feature groups must be accepted or rejected on benchmark transfer, not on
cross-validated top-1.** The cross-validated numbers remain the right way to measure whether a group carries
information; they are not evidence that it will help on a benchmark, and in these three cases they pointed the wrong
way. The per-point model's own out-of-fold point AUROC of 0.865 is unaffected by this -- that is a measurement about
points, not a claim about ranking transfer.

## Which modern methods can actually be run (checked 2026-10-05, primary sources)

The comparison was against fpocket (2009) and P2Rank (2018) only. Of the deep-learning methods, most cannot be run
at all without retraining them, which is a fact about the field's reproducibility rather than a limitation of this
project, so it is recorded with what was checked:

| method | code | weights | verdict |
|---|---|---|---|
| **GrASP** (Tiwary 2024) | MIT | **in the repository**, `trained_models/`, 1.6 MB per arm | **runnable** -- and run, see `scripts/baselines/run_grasp.py` |
| VN-EGNN (ICML 2024) | MIT | **none published**: the Zenodo record (17365855) holds datasets only, 763 MB, and there are no GitHub releases; `src/eval.py` loads a checkpoint from a Weights-and-Biases run id | not runnable without retraining on a GPU |
| EquiPocket (ICML 2024) | shipped as a baseline inside the VN-EGNN repository | none | not runnable without retraining; also needs MSMS for surfaces |
| DeepPocket (2021) | MIT | published (OneDrive) | blocked on libmolgrid, whose build is CUDA-oriented |
| GDEGAN (2026) | not located | not located | nothing to run |

So a claim about "beating the state of the art" currently rests on GrASP alone. The other three are not a matter of
effort: without published weights, reproducing them means retraining on their data, which is a different project.

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
`docs/results/compare_coach420.md`; measured under the single-chain receptor mismatch, so the deficit it refutes an
explanation for was itself overstated — the conclusion survives because it rests on the N = 1 split, which the
protocol does not affect). We emit 30 predictions per structure to P2Rank's 9.6 and lose top-N by
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
