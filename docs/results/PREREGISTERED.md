# Predictions made before the grid was read

Written 2026-10-08, before any arm of the current grid had produced a number. The point is to make the result
informative in both directions: a prediction recorded afterwards cannot be wrong, and an arm picked as the winner
after the fact is a lottery ticket rather than a finding.

Baseline: `full` at site top-1 **0.843 ± 0.009** over three seeds (ablation of 2026-10-06, seeds 0.849 / 0.849 /
0.831). Eleven arms in the grid.

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
