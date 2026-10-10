# Cross-validation on the single-site slice: site count does not explain the benchmark divergence

Measured 2026-10-10. 1367 structures, 609 single-site (44.6 %), mean 2.27 sites. COACH420 for comparison:
1.29 mean sites, 74.7 % single-site. Three seeds, 5-fold CV by 30 %-identity cluster, the ranker's own
`cv_scores`, graded relevance and within-structure z-scores — so these are the ranker's numbers, not a
re-implementation.

| features | columns | CV top-1, all | CV top-1, single-site |
|---|---|---|---|
| native only | 16 | 0.592 | 0.608 |
| native+geometry | 52 | 0.647 | 0.647 |
| native+geometry+chemistry | 80 | 0.738 | 0.714 |
| +shell | 172 | 0.781 | 0.752 |
| +potential (all hand-built) | 200 | 0.787 | 0.754 |
| everything in the table | 204 | 0.791 | 0.754 |

**The hypothesis is refuted.** Our manifest poses a different problem from the benchmark — 2.27 sites per
structure against 1.29, 44.6 % single-site against 74.7 % — and the obvious explanation for cross-validation
ranking feature sets opposite to COACH420 was that ordering five pockets rewards features that picking one of
thirty does not. If that were the story, the single-site slice would invert the ordering. It does not: the
smallest set is **0.146 behind** the largest there (0.608 against 0.754), the same direction as on all
structures. Selecting models on the single-site slice would not have fixed the divergence.

**What the slice does show is saturation.** The 104 columns past the shell group buy **+0.002** on single-site
structures against **+0.010** on all of them, and the gap between the two slices widens monotonically with the
column count: +0.016 in favour of single-site at 16 columns, −0.029 against it at 172, −0.037 at 204. So the
extra columns do help the multi-site structures and stop paying on the single-site ones — consistent with
over-parameterisation relative to the number of independent *structures* rather than candidates, which is the
same reading the CatBoost null supports (a stronger regulariser did not help either).

**One limit that bounds the conclusion.** The three benchmark numbers this tries to explain (32 features 0.701
on COACH420, 236 features 0.642, 272 features 0.626) come from three *different candidate tables* built at
different times, not from feature subsets of one table. This test varies only the columns, holding the table,
the labels, the folds and the preparation fixed. It rules site count out; it does not rule out the rest of what
differed between those tables — protein-size distribution, receptor preparation, the geometry constants.

**So the measurement problem stands.** Cross-validated top-1 still cannot be used to accept a feature group, and
the next candidate explanations are the ones this test holds fixed. Reproduce with:

```bash
PYTHONPATH=src:. python scripts/eval/single_site_cv.py --tag native2 --seeds 3
```
