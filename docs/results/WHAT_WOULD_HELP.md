# What would actually improve this model, ranked

Assembled 2026-10-08 from the two literature reviews in `research_notes/` and from our own measurements. The
point of the file is to stop the question "should we add X" being answered by plausibility. Every row carries an
evidence class, a cost, and a status, and the ordering is by expected effect on **site top-1 at one site**, which
is where our measured deficit is (−0.07 to −0.09 against P2Rank and DeepSurf; level at three or more sites).

Evidence classes, used strictly:

* **controlled** — someone retrained the same architecture with the component on and off and published both numbers.
* **comparative** — different methods differ, and the difference is attributed to the component. Confounded.
* **absent** — nobody has measured it. Says nothing about sign.
* **ours** — measured in this repository, with the script named.

Two constraints bound everything below. Our seed standard deviation is **0.009**, so at three seeds the
detectable effect is about **±0.02**, and most published feature-ablation deltas in this literature are smaller
than that. And our pre-registered claim threshold (`docs/results/PREREGISTERED.md`) is **+0.02** because 11 arms
at σ 0.009 give an expected noise maximum of +0.015. A lever worth running is one whose evidence predicts more
than +0.02, or one that costs nothing.

## Tier A — already built, never run. Run these before writing any new code.

| # | Lever | Evidence | Expected | Cost |
|---|---|---|---|---|
| 1 | **Noise augmentation** (`noise_aug` arm: pos 0.5 Å, feat 0.3, drop 0.03) | **controlled**, and the largest clean number in this literature: +0.0976 combined PR-AUC on a fixed 8,537-structure set, against **+0.0256 for 2.6× the training structures** and +0.014 for 5.7× the parameters (Shiota et al., PLOS ONE 2024, via the PoSSuM-trained graph transformer) | **+0.02 to +0.08**. Rotation is a no-op for an equivariant model, so today we augment by nothing at all | one config line, zero preprocessing |
| 2 | **`no_site_decoder`** | **ours**: on the COACH420 structures we get wrong, the right candidate is second in **24 of 53** cases. The decoder exists to fix exactly that | decides whether the diagnosis was right. If it matches `full`, pocket-against-pocket comparison was not what was missing and the next three rows matter more | already in the grid |
| 3 | **Cascade re-ranking** (finish the column-budget sweep of `train_ranker`) | **controlled, three independent confirmations** — the best-evidenced intervention in the field: PRANK over fpocket is +7 to +14 Top-n points, and on LIGYSIS (2024) the two best of 13+15 evaluated methods are **both** fpocket-plus-a-re-ranker | the largest single number available to us, and larger than every chemical channel put together | 8+ h CPU, no GPU |
| 4 | **`probe_tiered` / placement arms** | **ours**: the probes are the only component with a large effect — removing them costs **0.263** site top-1, while every other component measured so far is within two seed σ of zero | placement is the highest-leverage knob we have | already in the grid |

The cascade's documented failure mode is not that re-ranking fails but that it is **hard-capped by the first
stage's ceiling** — fpocket covers only 80.78 % of COACH420 ligands and 87.62 % of HOLO4K, and DeepPocket
inherits that cap. Ours is measured and better: **ceiling 1.000** with `grow_volumes` at 6 Å, 0.921 with
`merge_by_cavity` (`docs/results/`), which is the one place our pipeline is structurally ahead of the
best-evidenced design in the field.

**Promoted out of Tier B after measurement:** `probe_protrusion` now ranks with Tier A — it is the only addition
backed by a measurement of ours, it is free, and the feature it supplies is the one carrying P2Rank's ranking on
the benchmark where we lose. Run it alongside row 1.

## Tier B — cheap additions whose evidence is controlled

| # | Lever | Evidence | Expected | Status |
|---|---|---|---|---|
| 5 | **Protrusion at 10, 12 and 15 Å** (counts of all protein atoms around the point, after P2Rank's protrusion / Pintar's Cx) | **controlled** in the field — P2Rank RF importance 0.0845 against 0.0139 for the runner-up, protrusion-only reaches COACH420 Top-n 64.2 against 71.4 for all 35 features, removing it costs **10.9 Top-n points on COACH420** against 1.9 on HOLO4K — **and now `ours`**: on 268 of our structures and 8039 of our own candidates, a single count's correct-against-decoy AUC is 0.650 at 8 Å and **0.837 at 12 Å**, against 0.755 for our 26-direction closure, and a logistic model on our four existing geometric inputs goes **0.7999 → 0.8469** (5-fold CV) when the three long counts are added | the largest measured headroom of any addition here. It is a proxy — candidate discrimination, not top-1 — so it will not transfer one for one | **implemented** (`data.probe_protrusion`, arm `probe_protrusion`), costs nothing measurable at featurisation time. `scripts/eval/probe_geometry_headroom.py` reproduces the numbers |
| 6 | **Per-probe MSA conservation** (`probe_conservation`, implemented) | **controlled on train-dissimilar structures**, the only such result here: P2Rank_CONS top-(N+2) 53.9 % against 51.9 % on LIGYSIS, +346 true positives at a fixed 100-false-positive budget, on a benchmark with 0.5–9.7 % training overlap. Computed per target at inference, so it cannot be leakage | **+0.01 to +0.02**, and it is top-(N+2) evidence, not top-1 | code and tests in place; **cannot run here** — no alignment tool and no sequence database. PRANK excluded conservation deliberately, arguing it penalises cryptic and allosteric pockets |

The RF Gini importance behind row 5 is biased toward continuous features and protrusion is the only continuous
one, so the 6× gap is likely inflated — but the protrusion-only and full-minus-protrusion ablations are
independent of that bias and both confirm the feature is dominant on COACH420.

## Tier C — costs nothing and improves the evidence rather than the model

| # | Lever | Why | Evidence class |
|---|---|---|---|
| 7 | **Stratified reporting by pocket chemical class** (metal, nucleotide/phosphate, charged, apolar) | an aggregate DCC averages away exactly the heterogeneity that matters: DeepDrug3D's shape-only ablation loses ~0.13 on nucleotide pockets and ~0 on heme, and our own metal measurement is a 7.3× enrichment that yields AUC 0.544 | **absent** — no paper stratifies a site-prediction ablation by pocket chemical character |
| 8 | **Calibrate the per-probe scores and report the effect on top-1** | this is not cosmetic for us, and the reason is mechanical: a monotone recalibration cannot change an argmax, so it would be pointless if a site score were a max or a mean — but our default aggregator is **`sum_sq`**, P2Rank's, which is *nonlinear in the per-point probabilities*, so calibration changes the site ordering. The `site_agg` arms already cross the aggregators | **absent** — no paper calibrates pocket scores and reports the effect on ranking |
| 9 | **Keep the hard-novelty homology control in the headline table** | EquiPocket, VN-EGNN and GDEGAN do **not** remove test-set homologues at all; they cluster only within scPDB by UniProt id. DeepPocket removes at 50 %/30 %, UniSite only at 90 % | **ours** — and the one place where our protocol is stricter than the methods we compare against, which is worth stating rather than burying |

## Tier D — the evidence argues against spending time on these

* **Partial charges, PB or GB grids, atomic multipoles, learned/equilibrated charges.** Charges are nearly
  collinear with atom types (dMaSIF recovers the PB potential at r = 0.83 from types and inverse distances), and
  in P2Rank's own table the charge features are the *least* important of 35 (vsCation 0.00083, posCharge 0.0010
  against protrusion 0.0845). Kalasanty, DeepSurf and Pafnucy all carry a charge channel and **none ablates it**.
* **More frozen protein-language-model channels.** Our −0.010 for frozen ESM-2 650M is the expected value, not an
  anomaly: HonestAffinity swaps only the protein token representation and finds ESM ahead by 0.019–0.034 on
  familiar splits and **behind by 0.024–0.064 on leak-proof tiers**; AtomSurf reaches the same conclusion on
  PINDER. The large published ESM gains (GDEGAN +15.61 % DCC, VN-EGNN 0.503 → 0.605) replace *weak* baselines —
  atomic numbers and one-hot amino acids — where ours already carries residue chemistry and pocket-facing terms.
* **More training data as the main lever.** The one clean scaling experiment gives +0.0256 for 2.6× the
  structures, a quarter of what noise augmentation gives on the smaller set. Widening the manifest was right to
  do and is not where the next 0.05 comes from.
* **Higher steerable degrees.** Our degree-2 null (−0.006) has **no published analogue**, because no
  binding-site paper has ever swept `l_max` — so it is the field's first datapoint and worth reporting, but not
  worth re-litigating with more channels.

## Measured here, so it does not need re-arguing

* **Metals**: a metal sits within 5 Å of 13.4 % of correct candidates against 1.8 % of decoys (7.3×), lifting a
  candidate's chance of being correct from a 6.9 % base rate to 35.5 % — but the AUC of that distance alone is
  **0.544**, because 87 % of correct candidates have no metal near them. Sparse high-precision cue, bounded at
  about 0.034 top-1 even if it fixed every metal-bearing error. 47.3 % of our structures carry a metal, 26.5 % of
  sites have one within 8 Å, and 96 of 646 metal-bearing structures carry an ion below full occupancy.
* **Volume output**: growing at 6 Å gives DVO 0.294 at an unchanged ceiling of 1.000; merging by cavity gives
  0.265 and costs 0.079 of the ceiling. Growing wins on both axes.
* **Checkpoint selection**: selecting blind to the task reported top-1 0.022 for a model that reached 0.868.
  Fixed in `net_task.select_score`. No paper in this literature ablates the selection criterion, none uses EMA
  and none uses staged training with frozen loss terms, so all three of ours are unmeasured elsewhere.

## One bookkeeping consequence

Each arm added raises the expected maximum of pure noise across the grid. At 11 arms and σ 0.009 that maximum was
+0.015, which is where the pre-registered +0.02 threshold came from; at 15 arms it is about +0.017, still under
the threshold, so the threshold stands. It would not survive many more arms, which is another reason the answer to
"should we add a channel" has to keep being "measure the headroom first".

## Order of work

1. Tier A rows 1–4 plus `probe_protrusion`, in that order, 3 seeds, fold 0. Row 1 is one config line and has the
   largest published evidence; `probe_protrusion` has the largest evidence of ours.
2. Read row 2 before deciding anything else: it decides whether our own diagnosis of the top-1 deficit was right.
3. Tier C in parallel — it needs no GPU and changes what the paper can claim.
4. Tier B row 5 only if the redundancy measurement says a 10 Å count is not already carried by the closure field.
5. A fourth and fifth seed on the arms that already exist beats a sixth channel. At σ 0.009 the power, not the
   feature set, is what makes these questions answerable.
