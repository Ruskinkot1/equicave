# Predictions made before the grid was read

Written 2026-10-08, before any arm of the current grid had produced a number. The point is to make the result
informative in both directions: a prediction recorded afterwards cannot be wrong, and an arm picked as the winner
after the fact is a lottery ticket rather than a finding.

Baseline: `full` at site top-1 **0.843 ± 0.009** over three seeds (ablation of 2026-10-06, seeds 0.849 / 0.849 /
0.831). Eleven arms in the grid.

**Amended 2026-10-08, additively.** Five arms were added after the literature review (`probe_metal`,
`probe_electrostatic`, `probe_chemistry`, `probe_conservation`, `probe_protrusion`) plus `full_next`. Nothing
above is withdrawn or reworded; the baseline is unchanged and still `full` at 0.843 ± 0.009. Two consequences,
both recorded here rather than discovered later:

* **The threshold still holds, with less room.** Sixteen arms at σ 0.009 put the expected maximum of pure noise at
  about +0.017 against +0.015 for eleven. Still under +0.02, so the rule below stands unaltered — but it would not
  survive many more arms, and that is now a reason to stop adding them rather than a footnote.
* **The promotion rule for `full_next`.** `full_next` bundles noise augmentation with the protrusion channels. If
  it beats `full` by **≥ +0.02 on three seeds**, it becomes the baseline and every arm is re-run against it; the
  single-change arms (`noise_aug`, `probe_protrusion`) then say which half earned it. Below +0.02 the defaults do
  not move. The defaults were deliberately **not** flipped when these levers were implemented, because changing
  `full` mid-grid would leave no arm comparable to the reference recorded above.

Predictions for the six new arms, written now and before any of them has run: `probe_protrusion` **+0.01 to
+0.03** (the only one with a measurement of ours behind it: candidate-discrimination AUC 0.7999 → 0.8469, which
is a proxy and will not transfer one for one); `full_next` **+0.02 to +0.08**, carried mostly by the augmentation;
`probe_metal` **+0.00 to +0.02** (bounded by our own measurement: a metal is within 5 Å of 13.4 % of correct
candidates against 1.8 % of decoys, but the AUC of that distance alone is 0.544, which caps the arm near 0.034
even if it fixed every metal-bearing error); `probe_electrostatic` **0.000 ± 0.01, predicted null** — the
ionisable-group distances and counts are already in `probe_potential`, so this is the fourth redundancy arm;
`probe_chemistry` no more than the better of its two halves; `probe_conservation` **not run** — it needs an
alignment per target and nothing in this repository produces one.

Predictions for the architecture arms added the same day, also before any ran: `no_edge_type` **−0.00 to −0.03**
(a removal, so a drop is the expected direction; a null would say the nine edge types are decoration, which would
be worth more than a small confirmation); `het_mp` **0.000 ± 0.01, predicted null for the same reason** -- the
edge MLP already embeds those nine types, so this adds a mechanism it may already carry; the three `site_agg`
arms **within ±0.02 of each other**, with `max` the likeliest to lose, on the reasoning that it discards how
large a site's agreement is. The aggregator comparison is run because three accurate methods use three different
rules and none of them ablated it, not because we expect a winner.

One prediction about the calibration block, which is a measurement rather than an arm: the sum-of-squares site
ranking **will move** under temperature scaling, because that aggregate is not monotone-invariant, and the sign
is genuinely unknown. No paper in this literature calibrates pocket scores and reports the effect on ranked
success, so either sign is a result. A null there would also be informative: it would mean the head is already
close enough to calibrated that the aggregate does not care.

## The threshold, fixed in advance

With eleven arms and a seed standard deviation of 0.009, the expected maximum of eleven draws from noise alone
sits about 1.6 sd above the mean -- **+0.015**. An arm that beats `full` by 0.015 is therefore exactly what pure
chance produces. At three seeds the standard error of a mean is about 0.005.

**Nothing below +0.02 is claimed without a confirming run on fresh seeds.** This is not a convention borrowed from
elsewhere; it is what this grid's own arithmetic requires, and no paper in this field applies one.

## Ranked expectation

1. **`noise_aug`** -- the strongest prior. The only lever in this literature with a clean controlled measurement
   (+0.098 PR-AUC, Shiota et al. 2024), and our position amplifies it: we currently augment by rotation only,
   which is a no-op against an equivariant model, so this is a move from no effective augmentation to three
   kinds rather than from weak to strong.
2. **`gaussian_attn_4`** -- the attention kernel and the four-layer depth together, if both hold. Two changes at
   once, so a win here does not say which, which is why the single-change arms exist beside it.
3. **`gaussian_attn`** -- +3.3 % DCC in GDEGAN's own ablation on a backbone of our family, with the winning arm
   smaller and faster so capacity cannot explain it. Measured against weaker node features than ours, and one of
   their reported effects did not clear their own noise floor.
4. **`small`** -- expected to tie, not to win. Its value is elsewhere: a tie halves the cost of every later
   experiment.

## Expected not to help

* `agg_mean` and `agg_max` -- PRANK tested the mean against sum-of-squares and rejected it; this repeats that
  comparison on our model expecting their answer. Informative either way, because nobody has published it.
* `no_site_decoder`, `invariant_frames`, `no_probes_no_tensors` -- removals. By construction they should be worse
  or level. They are measurements, not candidates for best.

## What would change the plan

If `no_site_decoder` matches `full`, the diagnosis that our error is which cavity binds rather than where inside
it is wrong, and the decoder, the ranking losses and the aggregator sweep are all answering the wrong question.
That arm is first in the order for this reason.
