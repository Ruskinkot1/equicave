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

## The network is trained, and its ablation settles three of the project's novelty claims (2026-10-06)

First trained EquiCave-Net, 3 seeds on validation fold 0, seed standard deviation about 0.009 — so a difference
below roughly 0.02 is not distinguishable from noise:

| component removed | site top-1 | drop |
|---|---|---|
| nothing (`full`) | 0.843 ± 0.009 | — |
| **cavity probes** | 0.580 ± 0.009 | **+0.263** |
| equivariance (`invariant`) | 0.803 ± 0.005 | +0.040 |
| degree-1 vector channels | 0.806 ± 0.009 | +0.037 |
| degree-2 tensor channels | 0.849 ± 0.008 | **−0.006** |
| chirality | 0.843 ± 0.005 | +0.000 |
| ESM-2 650M | 0.853 ± 0.003 | −0.010 |
| SAS surface module | 0.854 ± 0.002 | −0.011 |

**The probes are the architecture.** They are the only component with a large effect, and the rest of the design is
either small (degree-1 channels, +0.037) or indistinguishable from zero.

**The degree-2 claim fails.** `full − no_tensors` is −0.006 with seed sd 0.008, so the lower bound is nowhere near
zero and the arm is not even parameter-matched — it is *smaller* than `full`, which can only flatter the degree-2
side. Degree 2 does no measurable work on this task. This was the project's second claimed contribution and the
survey established that no such ablation exists in the literature, so the measurement is still a result; what is
retired is the accuracy claim.

**ESM-2, the surface module and chirality do no measurable work either.** ESM-2 650M is the most expensive part of
the whole pipeline — a frozen 650 M-parameter model, hours of embedding, a 710 MB cache — for −0.010.

**The equivariance gain of +0.040 is real but its size is unknown**, because this `invariant` arm is the
distances-only one: it removes the steerable channels, the geometry from the input, *and* makes the centre head a
plain MLP. `invariant_frames`, which scalarises the same geometry in a local frame and isolates equivariance alone,
is implemented and has never been run. No number about equivariance goes in the paper until it has.

On the benchmarks the network beats the gradient-boosted ranker outright: COACH420 0.777 against 0.704 top-1,
HOLO4K 0.862 against 0.785. And `network only` beats `ranker + network features` (0.777 against 0.704), which is
the fourth consecutive case of feature stacking losing on a benchmark while winning in cross-validation. The
network ships as the predictor; the ranker is a baseline.

Protocol caveat: these were produced on an older checkout (`c8df939`) under the single-chain receptor, which the
detector and ranker numbers confirm (0.642 and 0.704, matching the pre-fix runs). The comparison *between* the
three methods is sound because all three share that protocol; the comparison against P2Rank is not, since P2Rank
was given the whole assembly. Re-running the network under `--receptor-chains all` is the next measurement.

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

## Our training set does not overlap the benchmarks by identity (2026-10-06)

YuelPocket (PNAS 2026) excluded COACH420 from its own paper because "the vast majority of COACH420 systems were
already present in the PLINDER training data" — a published leakage finding against a 309,140-system training set.
That is a reason to check our own, so here it is, by direct PDB id against `data/processed/manifest.csv` (1499
entries):

| benchmark | entries | share also in our manifest |
|---|---|---|
| COACH420 | 420 | 6 (1.4 %) |
| HOLO4K | 4009 | 73 (1.8 %) |
| LIGYSIS | 3376 | 17 (0.5 %) |
| CryptoBench | 1107 | 5 (0.5 %) |

Every one of those structures already falls in the `train-similar` row, which the 30 %-identity cluster rule puts
there regardless of identity, so the headline row — structures *not* similar to our training manifest — is unaffected.
The risk the finding describes is real for anyone training at PLINDER scale; at our scale it is not what limits us.

## The first finished comparison with a modern deep-learning method (2026-10-06)

DeepPocket (Aggarwal 2021) run end to end through its own pipeline -- its `clean_pdb`, fpocket, its `get_centers`,
its gridding and its published CNN weights -- on COACH420, converted into our candidate table and scored with the
identical labels, ligand rule, non-redundancy and receptor as every other method here. 286 structures predicted,
281 shared with every method in the table.

**The 173 structures not similar to our training manifest:**

| method | top-1 | top-N | top-(N+2) | predictions | ceiling |
|---|---|---|---|---|---|
| **ours** (ranker, 32 features) | **0.701** | 0.776 | 0.874 | 30.0 | 0.989 |
| **DeepPocket** (2021 CNN, published weights) | **0.695** | 0.770 | 0.879 | 36.9 | 0.960 |
| P2Rank 2.5.1 (2018 random forest) | 0.753 | 0.828 | 0.879 | 9.6 | 0.931 |
| our detector, no ranking | 0.598 | 0.713 | 0.822 | 30.0 | 0.989 |

Paired cluster bootstrap, us against DeepPocket on the same 174 structures: **top-1 +0.006 [-0.069, +0.081]**,
top-N +0.006 [-0.054, +0.068]. A tie. DeepPocket against P2Rank is -0.049 [-0.101, +0.003] on all 283, the same
deficit as ours with an interval that includes zero.

**One correction was needed before these numbers could be trusted.** `compare_on_set.py` intersects the structures
each method predicted, so a structure a method answers with *no site at all* disappears from the comparison rather
than counting against it -- a silent favour to any conservative method, and every such structure is a guaranteed
miss. DeepPocket refuses 2 of 300; GrASP, whose clustering returns nothing when no atom passes its threshold,
refuses 9 %. The three drivers now write a refusal as a miss, `add_missing_as_misses.py` repairs a table produced
before they did, and the numbers above are after that repair (it moved DeepPocket's ceiling from 0.962 to 0.960
and its top-1 from 0.699 to 0.695).

Two things follow, and the second is the uncomfortable one. We are **not worse** than a modern deep-learning
predictor with published weights, which is the first such statement this project can make from its own paired
measurement rather than from someone else's table. And neither of us is distinguishable from a random forest from
2018 -- which is the pattern the literature already shows, with P2Rank winning DCA on HOLO4K and PDBbind inside
VN-EGNN's own table and the fpocket+PRANK hybrid placing first of thirteen in LIGYSIS.

Detection is again not where we lose: our ceiling is 0.988 against DeepPocket's 0.965 and P2Rank's 0.931. The same
diagnosis as before, now confirmed against a learned method and not only against a geometric one.

### GrASP as well: three learned methods, and none of them beats a 2018 random forest

GrASP (Tiwary 2024) driven through its own parse, inference and mean-shift clustering. All four methods on the
**170 structures not similar to our training manifest** (279 in common overall), same labels, ligand rule, receptor
and non-redundancy:

| method | top-1 | top-N | top-(N+2) | predictions | ceiling |
|---|---|---|---|---|---|
| P2Rank 2.5.1 (2018) | **0.759** | **0.829** | 0.876 | 9.6 | 0.929 |
| **ours** (ranker) | 0.706 | 0.782 | **0.882** | 30.0 | **0.994** |
| DeepPocket (2021) | 0.700 | 0.776 | **0.882** | 36.6 | 0.959 |
| GrASP (2024) | 0.676 | 0.718 | 0.741 | 2.1 | 0.741 |

Paired cluster bootstrap against P2Rank on those structures: ours **-0.053 [-0.124, +0.017]**, DeepPocket
**-0.059 [-0.135, +0.012]**, GrASP **-0.082 [-0.166, +0.000]**. All three intervals reach zero. On this benchmark a
random forest from 2018 is not beaten by a 2021 CNN, a 2024 graph attention network, or us -- and none of the three
differences is significant either.

**GrASP's shape, not its ranking, is what the top-N column measures.** It emits 2.1 predictions per structure to
our 30 and P2Rank's 9.6, because it is built for precision -- its own paper reports precision at top-3 of 71.2
against P2Rank's 41.0. Split by the structure's own number of sites, on the 77 single-site structures it reaches
0.714, above our 0.706 and DeepPocket's 0.700; at two sites 0.800; and on the 38 with three or more it falls to
0.605 against our 0.900, because two predictions cannot cover four pockets. Comparing it on top-N compares designs.

**Two corrections were needed before any of this could be quoted, and the second reversed a claim.**

*Refusals count as misses.* `compare_on_set.py` intersects the structures each method predicted, so a structure a
method answers with no site at all would vanish from the comparison instead of counting against it -- a silent
favour to the most conservative method. DeepPocket refuses 2 of 300, GrASP 22. The drivers now record a refusal as
a miss and `add_missing_as_misses.py` repairs an older table.

*Our failures do not.* A structure **our** pipeline died on is not the method refusing, and scoring it as a miss
charges the method for our environment. Five of the structures first counted against GrASP were of that kind. With
them excluded its ceiling rises from 0.771 to 0.784 and its paired difference against P2Rank moves from
-0.092 [-0.169, -0.017] to -0.082 [-0.166, +0.000] -- **from significant to not**. An earlier version of this
section claimed GrASP was significantly worse than P2Rank; that claim was an artefact of our own crashes and is
withdrawn. The drivers now keep a `failed.txt` apart from the attempted list.

Files: `compare_coach420_vs_dl.md` (all four, differences against P2Rank), `compare_coach420_vs_deeppocket_ref.md`
and `compare_coach420_vs_grasp_ref.md`. DeepSurf is still running; the comparison will be reissued with it.

### DeepSurf: two kinds of failure, and only one of them is DeepSurf's

The correction above -- that our crashes must not be charged to a method -- needs the converse too: a limit written
into the method's own code *is* the method's, and excluding it would flatter it. DeepSurf's run on COACH420 produced
both kinds, and they are separated by reading the traceback rather than by counting.

*Their limit (3 of 300 structures).* `readSurfPoints` in `utils.py` returns `None` when the DMS surface carries more
than 100000 points, and `simplify_dms` then unpacks that `None`, so their own guard reaches us as
`TypeError: cannot unpack non-iterable NoneType object`. It fires on exactly the three largest receptors in the set
-- 2WVA (34156 atoms, 167163 surface points), 1E5Q (27528) and 1Q51 (25697) -- and on no others: these are the only
three above 20000 atoms, and the point count scales with the surface, so the cap and the atom count tell the same
story. Nothing we can configure changes it; the structures are reported as excluded, with the reason, rather than
quietly dropped or scored as refusals.

*Our limit (11 structures, retried).* The remaining failures were the bridge process dying without reporting, and
`dmesg` names them: `oom-kill ... constraint=CONSTRAINT_MEMCG`, 5-11 GB resident against this container's 15 GB,
with two shards resident at once. Those structures are 9555-16096 atoms -- ordinary for the set, far below their
cap -- so the cause is our parallelism, not their code. They were rerun serially after the shards finished, and the
count of any that still fail is reported with the table.

Keeping the two apart matters for the same reason the GrASP correction did: 14 structures scored as misses would
move DeepSurf's numbers by about 5 percentage points, in a comparison whose differences are already inside their
confidence intervals.

### The top-1 gap is a choice between our own first two candidates (2026-10-07)

`scripts/eval/rank1_residual.py` splits the gap instead of quoting it. On the 174 not-train-similar COACH420
structures, ours 0.707 against P2Rank's 0.753:

* **The net gap is eight structures built from fifty-two disagreements.** We lose 30 and win 22; the two methods
  agree on 122. P2Rank is not systematically better here, it is differently wrong, with a small tilt. Any claim
  that one method ranks pockets better than the other has to survive that, and the paired interval
  (-0.053 [-0.124, +0.017]) already says it does not.
* **When we lose, the answer is almost always just below our pick.** Our first correct candidate is at rank 2 or 3
  in 21 of the 30, at rank 2 alone in 14, and absent in 1. This is a ranking failure among a few plausible
  pockets, not a detection failure -- consistent with the ceiling of 0.989 over our full list.
* **But the pick we make instead is often not a near miss.** Its distance to the nearest true site is 4-8 A in 8
  cases, 8-15 A in 11 and above 15 A in 11, median 11.1 A. A third of the time we confidently prefer a cavity
  elsewhere in the protein, which is why re-centring a prediction never helped: the error is which cavity, not
  where in it.
* **A perfect choice among our own top two would score 0.839**, and among the top three 0.891, against P2Rank's
  0.753. The decision that would close the gap is narrow enough to state as a binary one.
* **The gap does not depend on how many sites the structure has** (21/124 lost at one site, 8/43 at two, 1/7 at
  three or more), so it is not the multi-site behaviour that the top-N column measures.

Two things follow. The first is that a shortlist re-ranker is worth one more attempt, and `--cascade-feats` makes
the attempt the earlier failure pointed at rather than a repeat of it (see the negative result below). The second
is that "which cavity is ligandable, given the others" is exactly the comparison the site decoder was built to
make, since it scores site tokens against each other rather than scoring candidates independently -- so this
diagnosis is a prediction the `no_site_decoder` ablation tests, not a new experiment to design.

Numbers and the structure lists: `rank1_residual_coach420.json`.

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

**The probe-placement ablation was weighted for the wrong arm** (measured and fixed 2026-10-06). The occupancy
loss carries a positive-class weight, and the plan said to retune it "once placement is settled" because learned
placement was expected to raise the positive rate. Measured directly on 19 structures at the shipped settings:

| probe placement | occupancy positive rate | balancing weight (1-p)/p |
|---|---|---|
| `ligandable` (the default) | 0.230 | 3.34 |
| `tiered` (the comparison arm) | 0.133 | 6.49 |

So the shipped `occ_pos_weight: 3.0` is right for the default arm and **half of what the comparison arm needs**.
Run as it stood, `probe_tiered` would have been handicapped by a class weight tuned for its opponent, and the one
component with a large measured effect (+0.263) would have been compared against a deliberately under-weighted
baseline. The arm now carries its own weight. The worry in the plan was real but pointed the wrong way: the rate
did not rise to a third under learned placement, it is the *random* placement that is the sparse one.

**Telling the ranker how each candidate compares with its best competitor does nothing** (measured 2026-10-06).
This was the cheap version of the idea behind the network's site decoder: besides the within-structure z-score of
every feature, give the ranker its **margin** -- the value minus the best value among the *other* candidates of the
same structure, 96 columns instead of 64. Out of fold on 1367 structures the result is 0.7286 against 0.7323, a
paired cluster bootstrap of **-0.0037 [-0.0163, +0.0088]**: indistinguishable from zero, and this time it does not
even win in cross-validation before failing to transfer.

It does not refute the decoder, which lets every site attend to every other and compute an arbitrary function of
the list, where this gives a fixed per-feature comparison with a single competitor. But it does say that the
comparison has to be *learned over the list* to be worth anything, and it lowers the prior on the decoder helping.

**Combining the detector's score with the ranker's does not help either, and this one nearly fooled us**
(measured 2026-10-06). Swept directly on COACH420, a blend of the two within-structure z-scores peaks at
0.721 against the ranker's 0.698 at a weight of 0.75 -- a tempting +0.023. Selected the honest way instead, on the
1367-structure out-of-fold cross-validation (`train_ranker.py --save-oof`), the best weight is **1.00**: the
ranker alone, 0.7323, with every blend below it (0.7293 at 0.95, 0.7169 at 0.80, 0.6225 at 0.50). The benchmark
peak was four structures of noise out of 179. The parameter is chosen on cross-validation and the benchmark is
measured once, which is the only reason this was caught.

**Giving the ranker fewer candidates to choose from does not help** (measured 2026-10-06 on the 179 COACH420
structures not similar to our training set). The reasoning was that we hand the ranker 30 candidates where P2Rank
emits 9.6, so most of what it sees is a distractor, and the top-1 decision might be easier over a shorter list.
Restricting it to the detector's own top-k reaches 0.659 at k = 5, 0.687 at k = 8 and 10, and 0.698 from k = 15
upward, which is exactly the unrestricted number -- so truncation never wins and only costs ceiling (0.872 at
k = 5 against 0.989). The candidates the ranker does not need are not the ones it is getting wrong.

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
