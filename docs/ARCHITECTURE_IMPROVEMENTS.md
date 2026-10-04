# EquiCave-Net: concrete architecture improvements, each with a source and a falsifiable test

**Written 2026-10-04.** Companion to `docs/LITERATURE_SURVEY.md` (which establishes what the field has and has not
shown) and `docs/ARCHITECTURE.md` (which specifies what we have built). This document does **not** survey; it
proposes. Every proposal is sized against our files and has an ablation name, a metric that must move, and a
statement of what result would falsify it.

Tags as in the survey: **[V]** read from a primary source in this session; **[UNVERIFIED]** stated but not
confirmed from a primary source here; **[NOT FOUND]** searched for and not found. A proposal with no source says so
and argues from first principles instead.

---

## 0. The bottleneck, stated precisely, because everything below is ranked against it

Four measured facts from this repository fix what is worth doing.

1. **Coverage is not the problem.** The geometric generator's ceiling is **0.977** on our 1367-structure set and
   **0.989** on COACH420 (`docs/ARCHITECTURE.md` §1.3, `docs/results/`). For peptide grooves the groove tier
   uniquely finds only **1.6 %** of sites. Nothing that increases recall of candidates can buy more than 2 points.
2. **Ordering is the problem.** On the leakage-free 432-structure subset
   (`docs/results/clean/ranker_native2.md`): detector order top-1 **0.516**, ranker top-1 **0.792**, top-N
   **0.845**, ceiling **0.988**. There are **0.196 of top-1** and **0.143 of top-N** left on the table, and all of
   it is ordering and per-prediction precision.
3. **Chemistry, not geometry, is what the ranker eats.** In the 32-feature study
   (`docs/results/ranker_native.md`) removing the `chemistry` group costs **−0.075 top-1**; removing `geometry`
   costs **−0.009**; removing `native` *improves* top-1 by **+0.007**, i.e. the detector's own score contributes
   nothing once the environment is described. The `potential` group's contribution has **never been measured** —
   it only exists in the 109/218-feature version, whose results table carries no per-group ablation. **Measure it
   first** (§Roadmap step 0); it conditions the value of P1.
4. **Centring is the weak axis.** On the 12-structure held-out set (`docs/results/eval_heldout.md`): ranker top-1
   is **0.833 by DCA** but **0.667 by DCC**, against a DCC ceiling of 0.833. DCC is also precisely where
   equivariant models have their published advantage (survey answer G, statement 2: "the equivariant advantage is
   concentrated in DCC"). The network has an equivariant coordinate read-out; the ranker does not. **Centring is
   the one axis where the network can do something the gradient-boosted ranker structurally cannot.**

And one fact about the network: `net_task.evaluate` scores the network's own site list by **DCA ≤ 4 Å only**
(`net_task.py:133`). There is no DCC metric for the network anywhere in the repo. Several proposals below are
untestable until that is added — it is four lines and it is in the roadmap as step 0.

Finally, a sobering frame from the survey that should be held throughout: **LIGYSIS [V]** ranks the best
equivariant end-to-end detector 12th of 13, and the best-evidenced mechanism in the whole field is "geometric
candidate generation + learned re-scoring" at **+10 to +13 points recall**. We already *are* that mechanism. The
network's plausible job is therefore to **supply better features to the re-scorer**, plus to fix DCC, plus to
deliver the controlled ablation the field lacks. A realistic expected gain for one GPU week is **+0.02 to +0.05
top-1** over the 218-feature ranker, not +0.2. Any plan that implies otherwise is not credible.

---

## P1. Per-probe interaction-potential channels, plus a masked-potential auxiliary target

**1. Mechanism.** Feed the *un-aggregated* interaction potential — the seven per-point "what could a ligand atom
here actually bind to" flags that `pocket_features.interaction_potential` already computes, plus the seven
distances to the nearest matching partner — as scalar inputs on every probe node, and add a head that must
reconstruct those seven flags for probes whose own potential channels have been masked out.

**2. Primary source.**
- **PharmacoNet** (Seo & Kim, *Chemical Science* 2025; arXiv:2310.00681) **[V in survey §4/answer D]** predicts
  seven pharmacophore types inside a pocket with labels restricted to interactions PLIP actually detected in
  PDBBind v2020 — establishing that per-interaction-class fields over empty pocket space are learnable and
  useful, and that the class vocabulary we use is the right one. It is a 3D CNN, not equivariant, so the
  architecture is not the transferable part; the *input/target design* is.
- **P2Rank** (Krivák & Hoksza, *J. Cheminform.* 10:39, 2018) **[V in survey]** reaches 74.9 % COACH420 DCA top-N
  with a random forest over **local physicochemical features aggregated at surface points** — i.e. the whole
  method is "per-point chemistry descriptors", and it still beats every equivariant detector on HOLO4K inside
  VN-EGNN's own table (0.787). Strong evidence that per-point chemistry is the high-value input.
- **Ishitani, Takemoto & Tomii**, *PLOS ONE* 19(8):e0308425, 2024 **[V]** — a graph transformer that ranks
  fpocket candidates; its ablation finds that **including solvent-accessible-surface-area features gave a
  significant improvement**, and that positional-noise augmentation suppresses overfitting. Top-1 success 0.571
  on their holo4k+coach420 set (not comparable to our protocol). MIT licence stated.

**3. Why it should help this task.** Our bottleneck is per-prediction precision, and fact 3 above says the single
most valuable feature group in the *ranker* is chemistry. The *network* currently sees no chemistry at probe level
at all: `feat_probe` is six numbers, all geometric (buriedness, distance to protein, deep flag, three atom
counts). To know that a probe sits opposite an Asp carboxylate the trunk must infer it from residue one-hots
through up to 5 layers of 10 Å message passing — when the answer is one KD-tree query. This is the cheapest
available route to the quantity that the ranker says matters most. It also directly conditions the hotspot head:
the hotspot target *is* an interaction class, and the potential is the "is this class even possible here" prior
for the same class.

**4. Implementation sketch.**
- `src/equicave/pocket_features.py`: factor the per-point part of `interaction_potential` out into
  `point_potential(points, trees) -> (avail [P,7] bool, dmin [P,7] float32)`. `interaction_potential` then calls
  it and aggregates as now, so the 21 ranker features are bit-identical. ~20 lines, net new ~12.
- `training/pockets/data.py` in `featurize`, after `ppos` is built and before `feat_probe` is assembled: build
  `trees = {k: cKDTree(xyz[m]) for k, m in LB.protein_atom_types(st).items() if m.sum()}` (already the pattern in
  `pocket_features.featurize:103-104`), call `point_potential(ppos, trees)`, and concatenate
  `[avail.astype(float32), np.minimum(dmin, 8.0)/8.0]` → `feat_probe` goes **[P,6] → [P,20]**. Guard with
  `if potential_channels:` from `data.potential_channels` (default `true`). ~12 lines.
- Same treatment for `feat_surf` is **not** proposed: surface points sit on atoms, so their potential is
  degenerate.
- `training/pockets/model.py`: `self.head_pot = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 7))`,
  emitted as `o["pot_logit"] = self.head_pot(x[probe])` → **[P,7]**. 3 lines.
- `training/pockets/data.py`: `mask_probe_potential(b, frac=0.2, rng)` mirroring `mask_residues` — zero columns
  6:20 of `feat_probe` for a random 20 % of probes, return the mask and the pre-mask `avail` as `y_pot`. ~18
  lines. Masking is what makes this **supervision rather than an identity map**: an unmasked model could read the
  answer off its own input.
- `training/pockets/net_task.py` `losses()`: `part["pot"] = Fn.binary_cross_entropy_with_logits(o["pot_logit"][m],
  b["y_pot"][m])` on the last pass only, weight `loss.w_pot: 0.3`. ~5 lines.
- `in_dims["probe"]` is read from the cache at `net_task.py:164`, so nothing else changes. **The cache must be
  rebuilt** (`data/cache/net`) — that is a ~1367-structure CPU job, not a GPU job, and it can run before the GPU
  is booked.
- Config keys: `data.potential_channels: true`, `loss.w_pot: 0.3`, `loss.pot_mask_frac: 0.2`.
- Total ≈ **70 lines** across four files.

**5. Equivariance impact.** **None — these are scalars**, invariant by construction (every one is a function of
interpoint distances only). The hazard is clerical, not mathematical: a wrong concatenation order between cache
build and `in_dims`, or accidentally putting a *direction* to the nearest partner in the scalar block. The
existing `test_so3_equivariance` catches the latter immediately (it would break `res_logit` invariance), but it
constructs `feat_probe` with a hardcoded width of 4 (`tests/test_model.py:28`). Add
`test_potential_channels_are_invariant`: build the batch with `feat_probe` of width 20 whose last 14 columns are
computed from `pos` via `point_potential`, rotate and translate, assert all scalar heads and `pot_logit` agree to
atol 1e-4 and that `offset` rotates. ~20 lines.

**6. Cost.** Parameters: `embed["probe"]` grows by 14×128 = 1,792; `head_pot` adds 128×128+128×7 ≈ 17.3 k. Against
a `dim=128, layers=5` trunk (order 4–5 M parameters) that is **<0.5 %**. Memory: unchanged. Time: the GPU step is
unchanged to within noise; **featurisation** costs 7 extra KD-tree range queries over ≤768 points, which on the
numbers in `pocket_features.featurize` is tens of milliseconds per structure — call it **+2 % on a one-off cache
build**.

**7. Falsifiable test.** Ablation `no_probe_potential: {data.potential_channels: false, loss.w_pot: 0.0}` in
`training/configs/ablations.yaml`. Metric that must move: **`net_sites.top1`** (in `collect_ablations.py`'s KEYS
already) and **hotspot mean AP**. Then, downstream, **ranker top-1 with `--features-extra net`**.
*What would mean the idea is wrong:* `net_sites.top1` flat (within 1 seed-sd) while hotspot mean AP jumps. That
pattern means the potential channels were absorbed by the hotspot head as a shortcut and taught the trunk nothing
about site identity — the feature is then a label proxy, not information, and should be removed (or kept only as
a ranker feature, where it already is).

**8. Risk, and what I would drop it for.** The real risk is exactly that shortcut: `y_hot` and the potential share
the same distance thresholds (`LB.HBOND_D` etc.), so the potential is a *superset indicator* of the hotspot
label — every true hotspot point necessarily has the matching potential. The hotspot head could collapse to
"copy the input channel", inflating `hot_ap` while the trunk learns less than before. Mitigations: (a) the
masking, which makes the auxiliary task non-trivial; (b) report `hot_ap` for masked and unmasked probes
separately; (c) keep the `hot_ap_per_class_proximity_target` cross-check that already exists
(`net_task.py:147`). I would drop P1 for nothing — it is the cheapest proposal here and the only one whose GPU
cost is literally zero. But if the step-0 CPU measurement shows the `potential` ranker group is worth **< 0.005
top-1**, I would cut its scope to the 7 `avail` flags only and skip the auxiliary head.

---

## P2. A within-structure listwise ranking loss on candidate-level scores

**1. Mechanism.** Pool probes into the ~30 native candidates the structure already carries (`cand_centers` is
cached by `data.featurize:182`), score each candidate with one head, and train that score with a *listwise* loss
over the candidates of the structure — softmax cross-entropy against graded relevance, plus a top-3-truncated
LambdaRank-style pairwise term — instead of relying only on per-probe pointwise BCE.

**2. Primary source.**
- **Oksuz, Cam, Akbas & Kalkan, "Rank & Sort Loss for Object Detection and Instance Segmentation", ICCV 2021
  [V]** (arXiv:2107.11669; verified venue, oral, and the headline claims from the arXiv abstract): a
  ranking-based loss that supervises the classifier to rank each positive above all negatives *and* sort the
  positives among themselves by localisation quality improves **Faster R-CNN by ≈3 box AP**, and **Mask R-CNN
  with RFS on LVIS by 3.5 mask AP (≈7 AP for rare classes)**, beating the ranking-based aLRP baseline by ≈2 box
  AP, with **tuning-free task-balancing coefficients**. Search-summary detail, not read from the PDF (CVF
  returned 403): replacing Focal Loss by RS Loss is reported as ≈1 AP, with gains of 3.4 / 1.0 on Faster R-CNN /
  Cascade R-CNN and ≈1 AP on ATSS / PAA **[UNVERIFIED]**.
- **Wang, Li, Golbandi, Bendersky & Najork, "The LambdaLoss Framework for Ranking Metric Optimization", CIKM
  2018 [V on venue, authors and claim]** — a probabilistic framework giving theoretically grounded losses for
  NDCG@K, and the proof that LambdaMART optimises a lower bound on DCG. This is the formal justification for
  truncated, metric-aligned ranking losses.
- **Cao, Qin, Liu, Tsai & Li, "Learning to Rank: From Pairwise Approach to Listwise Approach" (ListNet), ICML
  2007 [UNVERIFIED]** — the softmax-cross-entropy listwise loss itself.
- **Our own measurement**, which is the strongest evidence of all: the 218-feature LightGBM arm uses
  **LambdaRank with truncation level 10, graded relevance and within-structure z-scores** and converts detector
  order 0.516 → **0.792** top-1 (`docs/results/clean/ranker_native2.md`). Listwise, within-structure,
  graded-relevance training is the thing that works on *this* data.

**3. Why it should help this task.** The network is currently trained with **no ordering objective whatsoever**.
`head_conf` is supervised by `Fn.binary_cross_entropy_with_logits` against a detached per-probe indicator
(`model.py:314-315`) — a *pointwise* loss, which optimises calibration of each probe independently and is
indifferent to whether probe A outranks probe B. Then `predict_sites` sorts by exactly that score
(`net_task.py:88`). We are reading a ranking off a quantity that was never trained to rank. Meanwhile the
LightGBM arm proves that on this data a listwise objective over the candidates of one structure is worth
+0.28 top-1 over the unordered alternative. Pointwise→listwise is the single largest *loss-function* lever
available, and it is aimed at the exact metric we are short on.

**4. Implementation sketch.**
- `training/pockets/data.py`: alongside `site_probe_mask`, cache `cand_probe_mask` = `[C, n_probe]` float, 1
  where a probe is within `CAV_R = 8 Å` of a candidate centre (one `cKDTree(ppos).query_ball_point` loop,
  mirroring line 198), and `y_cand` = `[C]` graded relevance {0,1,2} computed exactly as
  `scripts/train/train_ranker.py:add_relevance` does it (2 if the centre is within 2 Å of a ligand heavy atom, 1
  within 4 Å, else 0), so the network and the ranker optimise the same target. ~15 lines.
- `training/pockets/model.py`: `self.head_cand = nn.Sequential(nn.Linear(2*dim, dim), nn.SiLU(), nn.Linear(dim,
  1))` reusing the `head_prop` pooling pattern (`model.py:238-244`): `pooled = cat([w @ x[probe], w @
  V[probe].norm(-1)]) / w.sum(1,keepdim=True)` → `[C, 2*dim]`, `o["cand_logit"] = head_cand(pooled).squeeze(-1)`
  → `[C]`. 6 lines.
- New loss in `model.py`:
  ```
  def listwise_rank_loss(logit, rel, truncate=3, tau=1.0):
      """ListNet softmax CE against graded relevance + a top-k-truncated pairwise hinge."""
      if logit.numel() < 2 or rel.max() == 0: return logit.sum() * 0
      p = Fn.log_softmax(logit / tau, 0)
      q = rel / rel.sum()
      ce = -(q * p).sum()                                   # ListNet
      k = min(truncate, len(logit))
      top = logit.topk(k).indices                           # LambdaLoss-style truncation
      d = logit[top][:, None] - logit[None, :]              # [k, C]
      m = rel[top][:, None] < rel[None, :]                  # a worse item ranked above a better one
      pair = Fn.softplus(d[m]).mean() if m.any() else d.sum() * 0
      return ce + pair
  ```
  ~16 lines.
- `net_task.py` `losses()`: `part["rank"] = M.listwise_rank_loss(o["cand_logit"], b["y_cand"],
  w.get("rank_truncate", 3))`, on the last pass only; config `loss.w_rank: 1.0`, `loss.rank_truncate: 3`.
- `net_task.py` `net_features()` and `evaluate()`: use `cand_logit` as the network's own site score, which makes
  `net_sites.top1` finally measure what the network was trained for. ~8 lines.
- Total ≈ **55 lines**.

**5. Equivariance impact.** **Preserved exactly.** `cand_logit` is built from `x` (invariant) and `V.norm(-1)`
(invariant), pooled with distance-derived weights (invariant). The hazard is pooling with anything direction-
dependent. Test `test_candidate_score_is_invariant`: add `cand_probe_mask` to `_batch()`, assert
`allclose(o1["cand_logit"], o2["cand_logit"], atol=1e-4)` under rotation+translation, and add `"cand_logit"` to
the loop at `tests/test_model.py:45`. ~8 lines.

**6. Cost.** Parameters: `head_cand` = 2·128·128 + 128 ≈ 33 k, **<1 %**. Memory: `[C, 2·dim]` with C ≤ 30 —
nothing. Time: the pooled matmul is `[30,768] @ [768,128]`, negligible; **1.00× wall clock** to within noise.

**7. Falsifiable test.** Ablation `no_listwise_rank: {loss.w_rank: 0.0}`. Metrics that must move: **`net_sites.top1`
and `net_sites.topN2`** (the network's own ranking, which is what the loss targets) and, downstream, **ranker
top-1 with the `cand_logit` column exported** (P3).
*What would mean the idea is wrong:* `net_sites.top1` improves but ranker top-1 with `--features-extra net` does
**not** — that would mean the network has merely rediscovered, less well, what LightGBM already extracts from
109 hand features, and the right conclusion is that the network should be trained for the things the ranker
*cannot* compute (centring, per-class hotspots) and not for ranking at all. That outcome is informative and
should be reported, not buried.

**8. Risk, and what I would drop it for.** Signal density: ~30 candidates per structure against ~768 probes, so
the ranking gradient is two orders of magnitude sparser than the segmentation gradient and may simply be drowned
at `w_rank = 1.0` — which is also the argument for P6 (staged training) and P7 (loss weighting) as companions.
Second risk: `y_cand` is computed from the same labels the ranker uses, so if the network is then used as a
ranker feature we are stacking two models on one target and must keep the out-of-fold discipline that
`mode=oof` already enforces (`net_task.py:242-258`) — no shortcuts there. I would drop P2 for P1 and P6 if GPU
time runs short, because both are cheaper and P6 is better evidenced.

---

## P3. Export ~40 candidate-level features from the network instead of 3 scalars

**1. Mechanism.** Replace the three numbers `net_features()` currently returns with a per-candidate block: a
low-dimensional projection of the pooled trunk embedding, per-class hotspot aggregates (not a max-then-mean),
occupancy and confidence order statistics, the predicted site size, the 14 property logits, the listwise
candidate score, and invariant summaries of the predicted centre offset.

**2. Primary source.** No single paper proposes this exact export; the justification is composed.
- **LIGYSIS** (Utgés & Barton, *J. Cheminform.*, PMC11552181) **[V in survey]**: the best-performing methods on
  the only large biological-unit-aware benchmark are **geometric candidates re-scored by a learned model**
  (fpocket_PRANK 60.4 %, DeepPocket_RESC 58.1 % top-N+2 recall) and re-ranking is worth **+5 to +13 points**.
  The quality of the re-scorer's *inputs* is therefore the lever.
- **DeepPocket** (Aggarwal et al.) **[V in survey]**: a CNN's segmentation score is itself the ranking feature for
  fpocket candidates — precedent for exporting network outputs as candidate features, which is what we do, only
  with 3 numbers.
- **Our own measurement**: the `chemistry` group (14 numbers) is worth **−0.075 top-1** when removed, and the
  whole 109-feature upgrade plus z-scores took top-1 from 0.724 to 0.792. Feature *count and specificity* at the
  candidate level is demonstrably what this ranker converts into top-1.

**3. Why it should help this task.** `net_features()` (`net_task.py:97-109`) is a hard information bottleneck
placed at exactly the decision point we are short on. It computes `hot = sigmoid(hot_logit).max(1)` — collapsing
seven interaction classes to one number — then takes its mean over probes within 4 Å. A pocket that is a strong
hydrophobic-plus-aromatic site and one that is a strong donor site become the same number. `net_center_conf` is a
max over a 4 Å ball, so it throws away the *distribution* of confidence, which is what distinguishes a tight
single-mode site from a smeared one. And the 128-dim pooled representation that the property and size heads use
is never exported at all. Three scalars out of a 128-dim per-probe field is not a feature extractor; it is a
summary.

**4. Implementation sketch.**
- `training/pockets/net_task.py`, `net_features()` (currently 13 lines → ~45): for each candidate centre, with
  `nb = tp.query_ball_point(ctr, 8.0)` (use `CAV_R`, not 4 Å, to match `pocket_features`):
  - `net_occ_mean`, `net_occ_max`, `net_occ_p90`, `net_occ_frac_gt50` — 4
  - `net_conf_max`, `net_conf_mean`, `net_conf_p90`, `net_conf_n_gt50` — 4
  - `net_hot_{cls}_mean` and `net_hot_{cls}_max` for the 7 classes — 14
  - `net_hot_classes_mean` (mean count of classes over 0.5 per probe), `net_hot_multi3_frac` — 2, deliberate
    mirrors of `pot_classes_mean` / `pot_f_multi3` so the ranker can compare learned against geometric
  - `net_prop_{j}` for the 14 property logits, pooled with the candidate mask — 14
  - `net_size_pred` — 1
  - `net_cand_score` (P2's `cand_logit`) — 1
  - `net_offset_norm_mean`, `net_offset_norm_std`, `net_center_spread` (mean pairwise distance between the
    predicted centres of the candidate's probes), `net_center_to_cand` (‖mean predicted centre − candidate
    centre‖) — 4. **All invariant.**
  - `net_emb_{0..7}`: 8 dims of the pooled trunk embedding, obtained by a `torch.pca_lowrank` basis fitted on the
    *training* folds inside `mode=oof` and stored in the checkpoint — 8.
  Total **52 columns**; trim to the ~40 that survive a LightGBM gain threshold.
- `src/equicave/pocket_features.py`: extend `NET_FEATURES` to the full list; `GROUPS["network"]` picks it up
  automatically, so `train_ranker.py --ablate` measures the whole block for free (`train_ranker.py:250`).
- `rank_sites(..., extra=...)` already merges arbitrary per-candidate dicts (`pocket_features.py:206-208`), and
  `mcp_server` already falls back when no network exists, so serving needs no change.
- Total ≈ **60 lines**, all in two files, **zero model change**.

**5. Equivariance impact.** **Preserved, conditionally.** Every exported number must be a rotation invariant.
`net_offset_norm_*`, `net_center_spread` and `net_center_to_cand` are norms/distances and are fine; exporting a
raw offset *component* would silently make the whole pipeline frame-dependent and would not be caught by any
existing test because the ranker is not under equivariance test at all. Add
`test_net_features_are_rotation_invariant` in `tests/test_model.py`: run `net_features()` on a batch and on its
rotated+translated copy with the candidate centres transformed the same way, and assert every returned value
agrees to atol 1e-4. ~18 lines. This is the single most valuable new test in this document, because it guards a
seam that is currently untested.

**6. Cost.** Parameters: **zero**. Memory: zero. Training time: zero. The CSV
`data/processed/net_features_<tag>.csv` grows from 3 to ~40 numeric columns — for 40,986 candidates that is a few
MB. LightGBM training time grows roughly linearly in feature count: 218 → ~258 features, **+18 %** on a job that
is minutes.

**7. Falsifiable test.** Not a network ablation — a **ranker** ablation, run by
`python scripts/train/train_ranker.py --tag native2 --features-extra net --ablate`, which already emits
`without network`. Report three rows: no network features, the 3 legacy scalars, the ~40 new ones. Metric:
**ranker top-1 and top-N on the 432-structure leakage-free subset, with the paired cluster bootstrap against the
3-scalar version**.
*What would mean the idea is wrong:* the 40-column version is not better than the 3-column version outside the
bootstrap CI. Given 432 structures and 383 clusters, 40 columns is a real overfitting risk, so this is a live
possibility. If it happens, the diagnosis is sample size, not concept; the fallback is the 8-dim embedding
projection plus the 14 per-class hotspot means only (22 columns).

**8. Risk, and what I would drop it for.** Overfitting at 383 clusters, and double-dipping: the network was
trained on folds of the same clusters. The `mode=oof` discipline handles the second (`net_task.py:242`) but only
if the fold assignment in `candidates_native2.csv` matches the network's — add an assertion that it does, because
silently mismatched folds would manufacture a gain that evaporates on any external set. I would **not** drop P3:
it is the only proposal with zero GPU cost that attacks the measured bottleneck, and it must be implemented
*before* the final `mode=oof` run or that 17-hour run has to be repeated.

---

## P4. A *fair* invariant arm: local-frame scalarisation plus frame averaging for the vector read-out

**1. Mechanism.** Add `model.invariant_mode: "frames"`, an arm that keeps every geometric input but expresses it
in invariant coordinates — per-node orthonormal local frames built by Gram–Schmidt from `vec0`, all input vectors
and edge directions projected into those frames as scalars — and that recovers an exactly equivariant centre
read-out by **frame averaging** the offset MLP over the sign-flip frame group of the probe's local PCA frame.

**2. Primary source.**
- **Puny, Atzmon, Smith, Misra, Grover, Ben-Hamu & Lipman, "Frame Averaging for Invariant and Equivariant
  Network Design", ICLR 2022 (oral) [V on venue, authors and the core claim]** (arXiv:2110.03336;
  openreview `zIUyj55nXR`): averaging an arbitrary backbone over a small *frame* of group elements yields
  **exact** invariance or equivariance while keeping the backbone's expressive power. This is what makes an
  invariant trunk able to emit an equivariant vector without becoming equivariant internally — i.e. what makes
  the comparison fair.
- **Lee, Byun & Shin 2023 (arXiv:2303.08818; journal version PMC11416008) [V, table read this session]**: their
  "no alignment" row removes the local-frame grid alignment that gives their model SE(3)-invariance, and COACH420
  F1 success goes **59.1 → 59.4 (up)** while HOLO4K goes **77.0 → 75.6 (down)**. Local-frame invariance for this
  task is, on the only published measurement, worth roughly nothing on one set and 1.4 points on another.
- **GVP** (Jing, Eismann, Suriana, Townshend & Dror, ICLR 2021, arXiv:2009.01411) **[V on venue/authors;
  ablation numbers UNVERIFIED — the PDF did not extract]** — the canonical vector↔scalar interface (norms in,
  scaled vectors out) that a scalarised arm should use.
- Survey answer B **[V]**: no equivariant-vs-invariant comparison at matched depth and width exists for pocket
  detection, and the nearest protein analogue (EquiPNAS, protein–nucleic-acid sites) reports a **negligible**
  difference (ROC-AUC 0.938 vs 0.940) **[UNVERIFIED]**.

**3. Why it should help this task.** It probably will **not** improve accuracy — and that is the point. The
project's stated novelty (`ARCHITECTURE.md` §2, survey answer B) is the **controlled isolation** of equivariance
and of degree-2. As implemented, `ablations.yaml`'s `invariant: {model.equivariant: false}` changes **three
things at once**, because `model.py:176-177` makes `use_vectors = equivariant and use_vectors`:
 (i) steerable messages are removed; (ii) **`vec0` is dropped from the inputs entirely** — the N→Cα, C→Cα,
 Cα→Cβ backbone directions, the two side-chain-chemistry direction vectors, the probe's direction to the nearest
 atom, and the surface normals all vanish, so the invariant arm is not merely invariant, it is *blind*; and
 (iii) the centre read-out silently switches from the equivariant `off_vec` to the non-equivariant `off_inv`
 MLP (`model.py:233-236`), which `ARCHITECTURE.md` honestly flags but which also means the arm's DCC is
 crippled for a reason unrelated to equivariance. A reviewer will say, correctly, that this ablation measures
 "coordinates vs no coordinates" — the exact confound the survey criticises EquiPocket and VN-EGNN for
 (answer B). Fixing it is the highest-value *scientific* change in this document, independent of any accuracy
 gain.

**4. Implementation sketch.**
- `training/pockets/model.py`, new module-level function:
  ```
  def local_frames(vec0):                      # vec0 [N, K, 3] with K >= 2
      e1 = Fn.normalize(vec0[:, 0] + 1e-8, dim=-1)
      a  = vec0[:, 1] - (vec0[:, 1] * e1).sum(-1, keepdim=True) * e1
      bad = a.norm(dim=-1) < 1e-3             # degenerate: vec0[:,1] parallel to vec0[:,0] or zero
      a = torch.where(bad[:, None], torch.stack([e1[:,1], -e1[:,0], torch.zeros_like(e1[:,0])], -1), a)
      e2 = Fn.normalize(a + 1e-8, dim=-1)
      return torch.stack([e1, e2, torch.cross(e1, e2, dim=-1)], 1)        # [N, 3, 3], rows orthonormal
  ```
  Every node type already supplies two usable vectors: residues have N→Cα and C→Cα (`data.backbone_vectors`),
  probes have `v1` (to nearest atom) and `v2` (mean direction to atoms within 6 Å) (`data.py:146-152`), surface
  points have the outward normal and owner-Cα→atom (`data.py:163-164`). The degenerate branch is essential:
  `v2` is zero whenever no atom lies within 6 Å. ~12 lines.
- `scalarise(vec0, F)` → `einsum("nkc,ndc->nkd", vec0, F).reshape(N, 3K)`: the input vectors as 3K invariant
  scalars, appended to `x` at embedding time. For edges, append `einsum("ec,edc->ed", u, F[dst])` — 3 invariant
  scalars per edge — to `inv` in `GeoTensorAttention.forward` via the existing `edge_scalar` channel, so the
  invariant arm *does* know the edge direction, just in a frame. ~10 lines.
- `frame_average(fn, pos, idx, k=16)`: for each probe, PCA of its k nearest nodes gives three axes; the frame set
  is the **4 sign-flip combinations with determinant +1**; `offset = ¼ Σ_g g · fn(scalarised-in-g)`. For SO(3)
  this is exact equivariance (Puny et al.). ~30 lines.
- `net_task.train_one`: when `cfg["model"]["invariant_mode"] == "frames"`, compute `n_extra = 3 * n_init_vec` and
  widen `in_dims` accordingly, and set `n_edge_scalar += 3`. ~8 lines.
- `ablations.yaml`: rename `invariant` → `invariant_distances` (keep as the "blind" reference) and add
  `invariant_frames: {model.equivariant: false, model.invariant_mode: frames}`.
- Total ≈ **90 lines**.

**5. Equivariance impact.** The arm must be **exactly invariant for all scalar heads** (because the frames rotate
with the structure) and **exactly equivariant for `offset`/`center`** (via frame averaging). Two new tests:
`test_frame_invariant_arm_scalars_are_invariant_and_readout_is_equivariant` — the same assertions as
`test_so3_equivariance` but on the `invariant_mode="frames"` model, including `offset @ R.T == offset'`; and
`test_local_frames_handle_degenerate_vectors` — build `vec0` with `vec0[:,1] = 2*vec0[:,0]` and with
`vec0[:,1] = 0`, assert `local_frames` returns orthonormal determinant-+1 frames and that the model's scalar
outputs stay invariant. The degeneracy test is the one that will catch real bugs: a naive Gram–Schmidt produces
NaN there, and NaN propagates silently through `index_add_`. ~45 lines.

**6. Cost.** `invariant_mode="frames"` with a **head-only** frame average: parameters +3K·dim at the embedding
(≈ 1,920 for K = 5) and +3 edge-scalar inputs; the offset head runs 4× on ≤768 probes, which is <1 % of the step.
**1.00–1.03× time.** If frame averaging were applied to the whole trunk it would be 4× everything — **do not do
that**; report the arm honestly as "invariant trunk, frame-averaged equivariant head", which is the comparison
that isolates the trunk.

**7. Falsifiable test.** Ablations `invariant_distances` and `invariant_frames`, both at matched depth, width and
features, ≥3 seeds. Metrics: **`net_sites.top1`, `occ_ap`, `res_ap`, and the new DCC-based site metric**.
*What would mean the idea is wrong — or rather, what would falsify the project's equivariance claim:* if
`invariant_frames` matches `full` within the paired seed CI, then equivariance buys nothing for pocket detection
at matched expressivity. **That is a publishable result and the survey says the field has no answer to this
question** (answer B). The wrong move would be to keep only the `invariant_distances` arm because it flatters the
equivariant model.

**8. Risk, and what I would drop it for.** Risk to the *paper*, not the model: the honest comparison may delete
the headline. The engineering risk is frame instability — a probe whose neighbourhood PCA has near-degenerate
eigenvalues has an ill-conditioned frame, so the frame-averaged output is continuous but can be high-variance;
Puny et al. note frames must be chosen so the averaging is well-defined, and the sign-flip group handles sign but
not eigenvalue ties. Log the PCA eigenvalue gaps. I would drop P4 only if the baseline `full` run fails to beat
detector order at all — in that case there is nothing whose equivariance is worth isolating.

---

## P5. Candidate tokens with pairwise attention (DETR-style), and an argued refusal of triangle updates

**1. Mechanism.** Create ~32 candidate tokens from pooled probes, run two layers of full self-attention over them
with an invariant pair bias from inter-candidate distance RBFs, and keep the Hungarian one-to-one matching that
`center_set_loss` already implements — so duplicate suppression is *learned* instead of done by the fixed 6 Å NMS
in `predict_sites`.

**2. Primary source.**
- **Carion, Massa, Synnaeve, Usunier, Kirillov & Zagoruyko, "End-to-End Object Detection with Transformers"
  (DETR), ECCV 2020 [V on venue and the core claim]**: detection as direct set prediction with a **set-based
  global loss that forces unique predictions via bipartite matching**, which **removes the need for
  non-maximum suppression**; the transformer's self-attention over the query set is explicitly credited with
  removing duplicate predictions.
- **LIGYSIS [V in survey]**: **67 % of VN-EGNN's 13,582 predictions were redundant** (9,066 duplicates; human
  creatine kinase predicted 7 times); IF-SitePred 49 %, DeepPocket_SEG 31 %; removing redundancy and re-ranking
  raises recall by **5–13 points**, and **no equivariant pocket paper reports a redundancy statistic**.
- **AlphaFold2** triangle multiplicative updates: Fig. 4a of Jumper et al., *Nature* 2021 does contain a
  "No triangles, biasing or gating (use axial attention)" ablation row **[UNVERIFIED — nature.com and PMC are
  gated in this environment; this is from a search summary, and the numbers were not read]**.
- **AlphaFold3 Pairformer** **[V in survey §2]**: triangle updates plus pair-biased attention, with equivariance
  obtained by **data augmentation over random rotations rather than by architectural constraint** — i.e. the
  AF3 lineage deliberately gives up exact equivariance, which our constraints forbid.

**3. Why it should help this task.** Our whole bottleneck is "which of these 30 candidates, in order". That is a
*set* decision, and the only set-level mechanism we have is a scalar NMS radius. `predict_sites`
(`net_task.py:84-94`) sorts by a pointwise confidence and greedily suppresses within 6 Å — so two probes 7 Å
apart in the same elongated cavity both survive, which is exactly the redundancy LIGYSIS quantified at 67 % for
the nearest comparable method. Pairwise attention lets a candidate's score depend on its competitors, which is
the learned version of the within-structure z-scores that we have *already measured* to help in LightGBM. At
C ≤ 30 tokens, all-pairs attention is 900 interactions — free.

**On triangle updates specifically, I recommend against them, and the reason is structural, not budgetary.** A
triangle multiplicative update enforces consistency of a pair representation that the network is *constructing*
— in AF2 the pairwise distances are unknown and must satisfy the triangle inequality. Our candidate-pair
distances are **exact Euclidean distances between known coordinates**: the triangle inequality holds by
construction and there is nothing to enforce. A triangle update would therefore spend O(C³) compute re-deriving a
constraint the input already satisfies. Plain pair-biased attention captures everything we need; triangle updates
are a mechanism for a problem we do not have. This is a direction I would decline.

**4. Implementation sketch.**
- `training/pockets/model.py`, new `class CandidateHead(nn.Module)`: input `pooled [C, 2*dim]` (from P2's mask
  pooling) and `cand_pos [C, 3]`; two blocks of `nn.MultiheadAttention(dim, heads, batch_first=True)` on a
  `[1, C, dim]` sequence with an additive bias `pair_bias = MLP(RBF(cdist(cand_pos, cand_pos)))` → `[heads, C,
  C]` passed as `attn_mask`; output `score [C]` and `emb [C, dim]`. ~70 lines.
- `data.py`: `cand_probe_mask` and `cand_centers` (the latter already cached, `data.py:182`). Note the mask must
  be rebuilt after recycling moves the probes — simplest is to pool on the **original** probe positions, which
  are what the candidate centres refer to.
- `net_task.py`: `cand_logit` comes from `CandidateHead` instead of a plain MLP; everything else as in P2.
- `ablations.yaml`: `no_candidate_tokens: {model.candidate_tokens: false}`.
- Total ≈ **90 lines** on top of P2.

**5. Equivariance impact.** **Preserved** provided the pair bias is a function of `cdist` only and the pooled
input uses `x` and `V.norm(-1)`. The failure mode worth testing is a pair bias that uses relative *vectors*. Test
`test_candidate_tokens_are_invariant_and_permutation_equivariant`: (a) rotate+translate, assert `cand_logit`
invariant to 1e-4; (b) permute the candidate order, assert the scores permute identically — this second
assertion catches an accidental positional encoding over the candidate index, which would make the output depend
on the detector's arbitrary ordering. ~22 lines.

**6. Cost.** Parameters: two attention blocks at dim 128 ≈ 2 × 66 k, plus the bias MLP ≈ 5 k → **≈ 140 k, ~3 %**.
Memory: `[heads, 30, 30]` — nothing. Time: **≤1.02×**.

**7. Falsifiable test.** Ablation `no_candidate_tokens`. Metrics: **`net_sites.top1`**, **`net_sites.topN2`**, and
— this is the one that makes the proposal honest — **the redundancy statistic**, for which
`equicave.metrics.redundancy` already exists (`metrics.py:109`) and should be added to `collect_ablations.KEYS`
and reported for every arm. We would then be the first pocket method to report it (survey flaw 4).
*What would mean the idea is wrong:* redundancy drops but top-1 does not. That would mean NMS at 6 Å was already
doing the job and the attention is decoration; drop the module and keep the redundancy metric, which is worth
having regardless.

**8. Risk, and what I would drop it for.** With ~30 tokens and ~1–3 positives per structure, the attention has
very little to learn from and may collapse to "score by cavity volume", reproducing the `largest cavity first`
baseline (top-1 **0.181**, the worst row in our table). Second risk: the candidate set at training time comes
from `detect_sites` on the *holo* structure, so the token set is not available at inference for a structure the
detector handles differently — it is, since the detector is deterministic and ours, but the fold discipline must
hold. I would drop P5 for P2 alone: a listwise loss over mask-pooled candidates gets most of the within-structure
comparison with none of the new machinery.

---

## P6. Staged training: pretrain the trunk on dense residue/probe segmentation, then fine-tune the ranking heads

**1. Mechanism.** Replace the single fixed-weight sum over 8 losses with a two-phase schedule: phase 1 trains the
trunk on the dense per-residue and per-probe segmentation targets (plus the masked-residue task); phase 2 drops
those weights to ~0.2 and trains the sparse site-level heads (centre, confidence, candidate ranking, properties).

**2. Primary source.** **Lee, Byun & Shin 2023 (arXiv:2303.08818v2, Table 1; journal version
"Turbocharging protein binding site prediction…", PMC11416008) [V — table read this session]**:
"inter-resolution transfer learning" initialises the **Binding-Site-Detection** module from a module first
trained on **Binding-Residue-Identification** (residues within 4 Å of a ligand). Measured effect, F1 success rate:

| arm | COACH420 | HOLO4K |
|---|---|---|
| full | 59.1 ± 0.3 | 77.0 ± 0.6 |
| no transfer | 53.9 (**−5.2**) | 70.0 (**−7.0**) |
| plain BERT attention instead of geometric | 54.8 (−4.3) | 70.8 (−6.2) |
| no augmentation at all | 55.8 (−3.3) | 71.3 (−5.7) |
| no homology augmentation | 57.3 (−1.8) | 75.1 (−1.9) |
| no perturbation augmentation | 58.9 (−0.2) | 76.2 (−0.8) |
| no grid alignment (removes their SE(3)-invariance) | 59.4 (**+0.3**) | 75.6 (−1.4) |

**Residue→site transfer is the largest single term in that table — larger than their geometry term, larger than
all augmentation combined.** This is the best-evidenced architectural recommendation available for our task.

**3. Why it should help this task.** We have the same two resolutions and the same asymmetry: hundreds of residue
labels and ~768 probe labels per structure against **1–3 site labels**. We currently mix them from epoch 1 at
fixed weights (`pockets_net.yaml` `w_res: 1.0, w_occ: 1.0, w_center: 0.5, w_conf: 0.5`), so for the first many
epochs the gradient is dominated by the dense terms and the sparse ranking heads train on a representation that
is still moving. Lee/Byun/Shin's result says the *schedule*, not the weights, is where the 5–7 points are. It
also costs nothing: same data, same model, same total epochs.

**4. Implementation sketch.**
- `training/configs/pockets_net.yaml`: add
  ```
  optim:
    stages:
      - {epochs: 25, weights: {w_res: 1.0, w_occ: 1.0, w_seq: 0.3, w_hot: 1.0, w_center: 0.0, w_conf: 0.0, w_rank: 0.0, w_prop: 0.0, w_size: 0.0}}
      - {epochs: 35, weights: {w_res: 0.2, w_occ: 0.2, w_seq: 0.0, w_hot: 1.0, w_center: 0.5, w_conf: 0.5, w_rank: 1.0, w_prop: 0.5, w_size: 0.1}}
  ```
- `net_task.train_one`: build the epoch→weights map once before the loop; inside the epoch loop replace
  `cfg["loss"]` with `dict(cfg["loss"], **stage_weights[ep])` when passing to `losses()`. The cosine schedule
  (`net_task.py:179`) should be **restarted at the stage boundary** with a short warm-up, otherwise phase 2 trains
  at a learning rate already decayed to near zero — this is the bug most likely to make the experiment
  inconclusive. Also reset `best`/`bad` at the boundary so early stopping does not fire on phase 1's score.
  ~35 lines.
- `ablations.yaml`: `single_stage: {optim.stages: null}` (i.e. today's behaviour, as the reference arm).

**5. Equivariance impact.** **None.** Nothing about the architecture changes. No new test needed; the existing
`test_so3_equivariance` still covers the model.

**6. Cost.** Parameters: zero. Memory: zero. Time: **1.00×** — same epoch count. The only cost is that the
best-checkpoint selection logic must be made stage-aware, which is bookkeeping.

**7. Falsifiable test.** Ablation `single_stage`, ≥3 seeds. Metrics: **`net_sites.top1`** primarily, with
`occ_ap` / `res_ap` watched for regression.
*What would mean the idea is wrong:* `net_sites.top1` unchanged and `occ_ap` / `res_ap` *down* — that would mean
phase 2 forgot the dense tasks without buying any site-level ordering, and the fixed-weight sum was fine. Given
the +5.2/+7.0 in the source this would be a genuine negative worth reporting, because our setting differs in one
important way: Lee/Byun/Shin *transfer weights between two separate modules*, whereas we share one trunk, so
forgetting is a real hazard they did not face.

**8. Risk, and what I would drop it for.** Catastrophic forgetting of the segmentation heads (mitigated by
keeping weights at 0.2 rather than 0.0) and the learning-rate-schedule interaction above. Secondary risk: this
interacts with P7 — if we later add uncertainty weighting, the stage weights become priors on the learned
weights and the two mechanisms can fight. Decide: **stages first, uncertainty weighting only if stages are
inconclusive.** I would drop P6 for nothing; it is free and the best-evidenced item in this document.

---

## P7. Multi-task weighting: uncertainty weighting at most — and an argued refusal of GradNorm/PCGrad

**1. Mechanism.** Learn one `log σ_i` per loss term and minimise `Σ_i exp(−2 log σ_i) L_i + log σ_i`, six to eight
extra scalar parameters, instead of the hand-set weights in `pockets_net.yaml`.

**2. Primary source.**
- **Kurin, De Palma, Kostrikov, Whiteson & Kumar, "In Defense of the Unitary Scalarization for Deep Multi-Task
  Learning", NeurIPS 2022 [V on venue and claim]**: a comprehensive evaluation finds that **unitary
  scalarisation (plain sum of losses) combined with standard regularisation consistently matches or outperforms
  specialised multi-task optimisers** (MGDA, IMTL, PCGrad, GradDrop); the analysis argues those optimisers act
  largely as *regularisers*; a sign-agnostic GradDrop shows per-task gradient sign conflicts do **not**
  significantly affect performance; and the authors explicitly recommend testing the plain sum with early
  stopping, weight decay and dropout before adopting an SMTO.
- **Kendall, Gal & Cipolla, "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and
  Semantics", CVPR 2018, pp. 7482–7491 [V on venue, authors and claim; the comparison table was NOT read — CVF
  returned 403, so the exact margin over grid search is UNVERIFIED]**: homoscedastic-uncertainty weighting
  learns the task weights and is reported to outperform separately trained single-task models, with the
  motivation that hand-tuning the weights is "difficult and expensive".
- **Chen, Badrinarayanan, Lee & Rabinovich, GradNorm, ICML 2018 [UNVERIFIED]** — listed only to be declined.

**3. Why it should help this task.** We have eight weighted terms (`w_res, w_occ, w_center, w_conf, w_hot, w_prop,
w_seq, w_size`) whose values were **chosen, never tuned on this data** — the network has never been trained. If
the ranking terms (P2) are added, that becomes nine, and the ranking loss is the one whose scale we understand
least. Uncertainty weighting is the cheap insurance: six scalars, one backward pass, no extra memory. The
*specific* reason it may matter here is that `center` is divided by a magic 4.0 (`net_task.py:68`) and `hot` is a
focal loss whose scale depends on `alpha=0.75, gamma=2` — these are not commensurable quantities and no one has
checked that the ranking gradient is not three orders of magnitude smaller than the segmentation gradient.

**But GradNorm and PCGrad I would decline.** Both require per-task gradients, i.e. 8–9 backward passes per step
on a model where one structure is already one step with gradient accumulation 4 (`pockets_net.yaml`). That is a
**6–9× training cost** for a family of methods that a NeurIPS 2022 paper **[V]** found to be matched by the plain
sum plus regularisation. At our budget (one GPU week, a 3.5-hour run) that trade is indefensible. The honest
ordering is: (1) log the per-term gradient norms for one epoch — a diagnostic, not a method; (2) if they span
more than ~2 orders of magnitude, try uncertainty weighting; (3) do not try GradNorm.

**4. Implementation sketch.**
- `training/pockets/model.py`:
  ```
  class UncertaintyWeights(nn.Module):
      def __init__(self, names): super().__init__(); self.names = list(names)
      ; self.log_s = nn.Parameter(torch.zeros(len(names)))
      def forward(self, parts):
          out = 0.0
          for i, n in enumerate(self.names):
              if n in parts:
                  s = self.log_s[i].clamp(-3.0, 3.0)      # guard: stops a term being switched off entirely
                  out = out + torch.exp(-2 * s) * parts[n] + s
          return out
  ```
  ~15 lines.
- `net_task.train_one`: construct it, add its parameters to the AdamW group with **weight_decay 0** (decaying
  `log σ` biases every weight towards 1 and defeats the mechanism), and have `losses()` return the `part` dict so
  the module can consume it. Also **log the learned weights every epoch** into `history.json` — without that the
  experiment is uninterpretable. ~20 lines.
- Also add a one-epoch diagnostic script `scripts/train/grad_norms.py` that backprops each term separately on 20
  structures and prints the per-term gradient norm — ~40 lines, CPU-runnable on the pilot config, and it may
  make the whole proposal unnecessary.
- `ablations.yaml`: `uncertainty_weights: {loss.uncertainty: true}`.

**5. Equivariance impact.** **None.** Loss weighting cannot break equivariance. No new test.

**6. Cost.** Parameters: **8**. Memory: zero. Time: **1.00×**.

**7. Falsifiable test.** Ablation `uncertainty_weights`, ≥3 seeds. Metrics: **`net_sites.top1`** and **hotspot
mean AP** (the two heads most likely to be starved).
*What would mean the idea is wrong:* neither metric moves outside the seed sd. Then the hand weights were
adequate and this direction is **closed** — report it, cite Kurin et al., and move on. That is a cheap, clean
negative and it saves anyone else the experiment.

**8. Risk, and what I would drop it for.** Uncertainty weighting is scale-sensitive and degenerate for losses
that can reach zero: a term with high variance gets a large learned `σ` and is effectively switched off, which is
exactly what could happen to the sparse ranking loss — the opposite of what we want. Hence the `clamp(-3, 3)`
and the per-epoch logging. I would drop P7 for P6: a stage schedule is a *hand-designed* answer to the same
question with far better evidence behind it for this specific task.

---

## P8. Treat unlabelled candidates as unlabelled, not negative (non-negative PU risk on the ranking loss)

**1. Mechanism.** Train the candidate-level ranking/classification loss with the **non-negative positive-unlabelled
risk estimator** rather than treating every non-annotated candidate as a confirmed negative, with a class prior
π for "unannotated candidate is actually a real site".

**2. Primary source.**
- **Kiryo, Niu, du Plessis & Sugiyama, "Positive-Unlabeled Learning with Non-Negative Risk Estimator", NIPS 2017
  (oral) [V on venue, authors and claim]** (arXiv:1703.00593): the unbiased PU risk goes negative and overfits
  badly for flexible models; the non-negative estimator fixes this and enables training very flexible models from
  limited positive data. Reference code exists but **we will not copy it** — the estimator is three lines of
  algebra.
- **LIGYSIS [V in survey, flaw 2]**: estimates that **33–50 % of existing binding sites are yet to be
  observed**, so top-N "penalises a method for ranking a *real but unannotated* pocket above an annotated one".
- **DeepPocket [V in survey]**: independently observed a **17-point jump from top-N to top-(N+2)** and attributed
  it to unannotated sites.
- **Our own numbers** corroborate the magnitude: on the 432-structure subset, ranker top-N **0.845** against
  top-(N+2) **0.928** — an 8.3-point gap that unannotated sites would produce.

**3. Why it should help this task.** This attacks the bottleneck from a direction **no pocket paper in the survey
has tried** [NOT FOUND]. With ~30 candidates per structure and a label base rate of **0.181**
(`ranker_native2.md`), roughly 25 candidates per structure are labelled 0. If LIGYSIS is right that a third to a
half of real sites are unannotated, then on the order of **1 of those 25 is a real pocket being actively pushed
down the ranking** every single step. For a *pointwise* loss that is mild label noise. For a *ranking* loss it is
directly adversarial: the loss explicitly penalises putting that candidate above the annotated one, which is the
decision top-1 measures. This is the cleanest available argument that our training signal, not our architecture,
is what caps top-1.

**4. Implementation sketch.**
- `training/pockets/model.py`:
  ```
  def nnpu_loss(logit, y, prior: float = 0.08):
      """Non-negative PU risk (Kiryo et al. 2017). y = 1 for annotated positives, 0 for unlabelled."""
      p, u = y > 0.5, y <= 0.5
      if not p.any() or not u.any():
          return Fn.binary_cross_entropy_with_logits(logit, y)
      rp  = Fn.softplus(-logit[p]).mean()                      # risk of positives as positive
      rpn = Fn.softplus( logit[p]).mean()                      # risk of positives as negative
      run = Fn.softplus( logit[u]).mean()                      # risk of unlabelled as negative
      neg = run - prior * rpn                                  # unbiased negative risk
      return prior * rp + torch.clamp(neg, min=0.0)            # the non-negative correction
  ```
  ~12 lines.
- Apply to `cand_logit` (P2) under `loss.pu_prior`; `0.0` recovers plain BCE so the ablation is a one-key change.
  For the listwise term, the PU analogue is simpler: **exclude** the unlabelled items from the pairwise hinge with
  probability π (a stochastic "don't penalise this comparison"), which is 4 lines in `listwise_rank_loss`.
- `ablations.yaml`: `pu_negatives: {loss.pu_prior: 0.08}` plus a sweep `pu_prior_{0.04,0.08,0.16}`.
- Total ≈ **25 lines**.

**5. Equivariance impact.** **None** — a loss on an invariant scalar. No new test. (Add `pu_prior` to the finite-
loss test `test_invariant_ablation_runs_and_losses_are_finite` so a negative clamp or an empty-positive structure
cannot produce NaN; ~6 lines.)

**6. Cost.** Parameters: zero. Memory: zero. Time: **1.00×**.

**7. Falsifiable test.** Ablations `pu_negatives` and the π sweep. Metrics: **`net_sites.top1`**, **`net_sites.topN`**,
and specifically **the gap between top-N and top-(N+2)**, which should *shrink* if the mechanism is real — the
model should stop being punished for the predictions that top-(N+2) forgives.
*What would mean the idea is wrong:* top-1 flat across the whole π sweep including π = 0. The estimator is known
to be sensitive to a misspecified prior, so a flat sweep means either the prior is irrelevant here (label noise
is not the binding constraint) or our candidate labels are cleaner than LIGYSIS's estimate implies. Either way
the direction closes, and the sweep is the test that closes it.

**8. Risk, and what I would drop it for.** π is unknown. LIGYSIS's 33–50 % is a statement about *sites*, not about
*our candidates*, and converting between them requires an assumption we cannot check. A wrong π makes everything
worse, not just neutral. Hence: **always run the sweep, never ship a single π.** Second risk: this is the most
speculative proposal here and the easiest to over-claim. I would drop P8 for P2 + P3 if time is short, because
PU learning on top of an untrained ranking head is two unvalidated things stacked.

---

## P9. Make the degree-2 and equivariance ablations capable of concluding

**1. Mechanism.** Three changes that are about *experimental validity*, not accuracy: (a) match the parameter
count of the `no_tensors` arm to `full`; (b) add an `e3nn_l1` arm so the e3nn family has a real degree sweep
l = 1/2/3 at matched channels; (c) add a synthetic unit test exhibiting a pair of configurations that a degree-≤1
model provably cannot separate and a degree-2 model can.

**2. Primary source.**
- **Joshi, Bodnar, Mathis, Cohen & Liò, "On the Expressive Power of Geometric Graph Neural Networks", ICML 2023
  [V on venue, year and the claim]**: "higher order tensors and scalarisation enable maximally powerful
  geometric GNNs", and equivariant layers distinguish a larger class of graphs by propagating geometric
  information beyond local neighbourhoods. Their synthetic discrimination experiments are the template for the
  unit test. (The specific k-chain/rotationally-symmetric construction was **[NOT FOUND]** in the accessible
  abstract; the PDF did not extract in this environment.)
- **HEGNN** (arXiv:2410.11443) **[V in survey]**: argues degree-1 models **degenerate on symmetric graphs**, and
  contains **no protein binding-site task** — so the mechanism is motivated but untested for our setting.
- **GDEGAN** (arXiv:2603.19817) **[V in survey answer B]**: uses `L_max = 2` for binding sites and **has no
  `L_max = 1` vs `L_max = 2` row**; the layer sweep holds `L_max = 2` fixed. The field has a method that uses
  l = 2 for pockets and **zero** evidence that l = 2 is what helps.

**3. Why it should help this task.** It will not improve accuracy at all. It is what makes the project's claim
survive review. Two concrete defects:
- **`no_tensors` is not a matched arm.** Setting `use_tensors=False` removes `self.wt` (dim×dim), removes 2·dim of
  invariant edge input and 3·dim of gate outputs from `edge_mlp` (`model.py:85-87`), and removes one term from
  `node_mlp`'s input. The arm is strictly **smaller**, so `full − no_tensors` measures *degree 2 plus extra
  capacity*. Any positive result is confounded in exactly the way the survey criticises in others.
- **`no_vectors`'s own comment admits confusion**: `ablations.yaml:8` reads "degree 0 only but still gated by
  edge directions? no: distances only" — a question mark in a config file is a sign the arm's semantics are not
  pinned down. With P4's `invariant_frames` arm, `no_vectors` becomes well-defined as "no steerable channels but
  full geometric input in local frames".

**4. Implementation sketch.**
- `net_task.train_one`: when `model.match_params` is set, solve for `dim` such that the arm's parameter count is
  within 2 % of a reference count stored in the ablation entry — a 10-line bisection over `dim` calling
  `sum(p.numel() for p in EquiCaveNet(...).parameters())`. ~20 lines.
- `ablations.yaml`:
  ```
  no_tensors_matched: {model.use_tensors: false, model.match_params: full}
  no_vectors_matched: {model.use_vectors: false, model.match_params: full, model.invariant_mode: frames}
  e3nn_l1: {model.backbone: e3nn, model.lmax: 1}
  ```
  (`e3nn_l2` and `e3nn_l3` already exist, so l = 1/2/3 becomes a real sweep.)
- `tests/test_model.py`, new `test_degree2_separates_what_degree1_cannot`: build two probe neighbourhoods with
  **identical multisets of pair distances and identical mean direction** but different second moments — e.g.
  six neighbours at the vertices of a regular octahedron (second moment ∝ I, traceless part zero) versus six
  neighbours in a planar regular hexagon of the same radius (traceless second moment non-zero). For both,
  `Σ u = 0` so every degree-1 aggregate vanishes, and all |u| are equal so every distance-based invariant is
  identical; only `Σ (u uᵀ − I/3)` differs. Assert: a `use_tensors=True` model gives different `occ_logit` for
  the two, and the **same weights** with `use_tensors=False` give the same `occ_logit` to 1e-5. ~35 lines.
  **Write this test before booking any GPU time** — it is the cheapest possible check that our degree-2 channels
  are wired to do something a degree-1 model cannot, and if it fails the whole degree-2 story needs rethinking
  before a single epoch is trained.

**5. Equivariance impact.** No change to the model. The new test is itself an equivariance-adjacent test: it
checks *expressivity*, which is the thing the equivariance tests cannot see. A mistake it would catch: if
`sym_traceless` were accidentally symmetric-but-not-traceless, or if `EYE.to(a)` were on the wrong device/dtype
under bf16 autocast (`model.py:33, 143` — `EYE` is a module-level `torch.eye(3)` in fp32 and is `.to()`-ed per
call, which is correct but fragile).

**6. Cost.** `no_tensors_matched` is *larger* than today's `no_tensors` (wider to match parameters), so that arm
costs perhaps **1.15×** a normal run. The test is milliseconds. `e3nn_l1` is cheaper than `e3nn_l2`.

**7. Falsifiable test.** The ablations themselves are the test. Metrics: **`net_sites.top1`, `occ_ap`, `res_ap`,
DCC-based site metric**, with the paired difference against `full` that `collect_ablations.py` already computes.
*What would mean the degree-2 idea is wrong:* `no_tensors_matched` matches `full` within the paired seed CI. Given
GDEGAN has priority on using l = 2 for pockets and **nobody has ablated it**, a well-powered negative is a real
contribution and should be the headline if that is what we measure. Conversely, if `test_degree2_separates_...`
fails on CPU today, the implementation is wrong and no training run can rescue it.

**8. Risk, and what I would drop it for.** The synthetic test may be hard to make tight: with random weights the
degree-1 model's outputs could differ for numerical reasons, or the two configurations could be separated through
the *scalar* input channels if those are not held identical. Construct it with identical scalar features and
identical `vec0`, differing only in `pos`. If the degree-1 model separates them, the test is wrong, not the
model — debug the test, do not weaken the claim. I would not drop P9; parameter matching is 20 lines and without
it the ablation table cannot be published.

---

## P10. Apo/holo and cryptic sites: measure the generator ceiling on CryptoBench first; side-chain jitter is low-value

**1. Mechanism.** Two separable things. (a) A CPU-only measurement: run the existing detector on CryptoBench's apo
structures and report the candidate-recall **ceiling**, which tells us whether cryptic sites are a *generator*
problem or a *ranking* problem. (b) Only if (a) says ranking: side-chain jitter augmentation — rotate side-chain
atoms beyond Cβ about the Cα–Cβ axis by N(0, 15°) before featurisation.

**2. Primary source.**
- **Lee, Byun & Shin 2023 [V — table read this session]**: removing *perturbation* augmentation (random
  perturbation of residue orientations) costs only **0.2 points on COACH420 and 0.8 on HOLO4K**, against 1.8/1.9
  for homology augmentation and 5.2/7.0 for transfer learning. **Geometric perturbation augmentation is the
  weakest of their three mechanisms by a factor of 5–25.** This is the single most relevant measurement and it
  argues *against* the direction.
- **Ishitani, Takemoto & Tomii, PLOS ONE 19(8):e0308425, 2024 [V]**: positional-noise augmentation "may be
  effective in suppressing overfitting" in a candidate-ranking graph transformer — a hedged, regularisation-
  flavoured claim, not a robustness claim.
- **CryptoBench** (Škrhák, Novotný, Feidakis, Krivák & Hoksza, *Bioinformatics* 41(1), January 2025) **[V on
  venue, size and the headline comparison]**: 1,107 structures built from apo–holo pairs grouped by UniProt,
  clustered by sequence identity, filtered to substantial binding-site conformational change; predefined CV
  splits; and **a sequence-based (protein-language-model) predictor outperformed PocketMiner and P2Rank on AUC,
  AUPRC, MCC and F1**. Distributed via OSF — **licence [UNVERIFIED], must be read before use** under this
  project's own rule that unverified-licence training data is rejected. Note the benchmark model is sequence-based,
  which means **a structure-based method's advantage on cryptic sites is not established**.
- **PocketMiner** (Meller, Ward, Borowsky, Kshirsagar, Lotthammer, Oviedo, Lavista Ferres & Bowman, *Nature
  Communications* 14:1177, 2023) **[V on venue, article number and the number]**: a GNN trained to predict where
  pockets open in MD, **ROC-AUC 0.87** on 39 experimentally confirmed cryptic pockets, >1,000× faster than
  simulation.

**3. Why it should help this task — and why I rank it low.** Our probes are free-space lattice points of the
structure we are given (`data.py:131-141`). **If a cryptic site is closed, there is no free space, so there is no
probe, so no amount of network improvement can find it** — the failure is upstream of everything in `model.py`.
Side-chain jitter of 15° will not open a closed cavity; opening cryptic pockets requires backbone motion, which
is what PocketMiner needed MD to learn. So the honest first move is the **ceiling measurement**, which costs CPU
hours and zero GPU hours and which decides whether the direction exists at all. Note also that cryptic sites are
orthogonal to our stated bottleneck: they are a *coverage* problem, and §0 fact 1 says coverage is not where our
0.196 of top-1 is hiding.

**4. Implementation sketch.**
- `scripts/eval/eval_cryptobench.py` (~90 lines, CPU): download CryptoBench's apo list **after reading its OSF
  licence**, run `detect.detect_sites`, and report `ceiling` (fraction of structures with any candidate within
  4 Å of a holo ligand atom, transferring ligand coordinates through the apo–holo superposition the benchmark
  provides), mean candidates per structure, and the same for `FILL_MIN_BURIED` lowered to 8 and 6. This is the
  decision-making number.
- `training/pockets/data.py` for (b): `sidechain_jitter(st, sigma_deg, rng)` rotating atoms beyond Cβ about the
  Cα–Cβ axis, called at the top of `featurize` under `data.sidechain_jitter: 0.0`. ~25 lines. **Important
  consequence:** jittering changes the lattice, hence `detect_sites` must re-run per sample, which on CPU is
  seconds per structure — the cache (`build_cache`) would have to be rebuilt per epoch or pre-built as N jittered
  copies. That cost is real and may exceed the GPU step. Pre-build 3 jittered copies per structure, which
  triples the cache (acceptable) and makes the augmentation static (weaker, but honest).
- `ablations.yaml`: `sidechain_jitter: {data.sidechain_jitter: 15.0}`.

**5. Equivariance impact.** **None** — augmentation acts on inputs before featurisation. The one hazard: jitter
must be applied **before** `detect_sites` and before label computation, or probe positions and hotspot labels
decouple and the targets become wrong. A test is warranted: `test_sidechain_jitter_preserves_backbone_and_labels`
— assert N, Cα, C, Cβ coordinates are unchanged, bond lengths within the side chain are preserved to 1e-4, and
`y_occ` recomputed on the jittered structure still has the same positive count to within 10 %. ~20 lines.

**6. Cost.** The ceiling measurement: **zero GPU**, a few CPU hours. The augmentation: **3× cache size** and one
extra cache build; GPU step unchanged.

**7. Falsifiable test.** Two. (i) The ceiling measurement itself: **if the apo candidate ceiling on CryptoBench is
already > 0.90, the generator is fine and side-chain augmentation is pointless** — the direction closes on a CPU
job. (ii) Ablation `sidechain_jitter`: metric `net_sites.top1` on the normal validation fold **must not drop**,
and CryptoBench residue-level AUPRC (a new metric, computed from `res_logit`) must rise.
*What would mean the idea is wrong:* given the source measures only **−0.2 / −0.8** points for removing
perturbation augmentation, anything less than +0.01 top-1 or +0.02 CryptoBench AUPRC means it is noise. I expect
this outcome and say so in advance.

**8. Risk, and what I would drop it for.** Licence risk is the serious one: CryptoBench's terms are unverified and
this project rejects unverified-licence training data. It can be used for **evaluation** reporting under fair
use-style academic citation only if its licence permits, which must be checked first — and it must **never** enter
training until it is. Second risk: the cache triples and the detector re-run may dominate wall clock. I would
drop (b) immediately for P6, and keep (a), because (a) is free and tells us something we do not know.

---

## P11. Protein–protein interface as an auxiliary target, from our own PDB files; and an argued refusal of conservation

**1. Mechanism.** Add a per-residue head supervised by "is this residue in a protein–protein interface", computed
from the chains already present in each manifest entry (a heavy atom of chain A within 4 Å of a heavy atom of
chain B), plus relative-SASA regression. Both targets are free — no new dataset, no licence question.

**2. Primary source.**
- **PeSTo** (Krapp, Abriata, Cortés Rodriguez & Dal Peraro, *Nature Communications* 14:2175, 2023; PMC10113261)
  **[V in survey §2]**: a geometric transformer on **atoms labelled only by element name** reaches accurate
  per-residue protein-binding-interface prediction, with rotation-equivariant vector states — establishing that
  interface labels are learnable from precisely the inputs we already have (coordinates + element), with no
  additional features.
- **ScanNet** (Tubiana, Schneidman-Duhovny & Wolfson, *Nature Methods* 2022) **[V in survey §2]**: per-residue
  protein-binding-site prediction from local-frame geometry plus sequence profiles.
- **Ishitani et al., PLOS ONE 2024 [V]**: SASA features gave a **significant improvement** in a candidate-ranking
  graph transformer — the direct evidence for the rSASA half of this proposal.
- **Lee, Byun & Shin 2023 [V]**: the +5.2/+7.0 residue→site transfer result, which is the same mechanism (a dense
  residue-level auxiliary target improving a sparse site-level one) applied to a different label.

**3. Why it should help this task.** The dense signal most correlated with "this surface patch binds something" is
"this surface patch binds a protein". Our current dense auxiliary tasks are residue segmentation (which is the
ligand target, so it adds no *new* information, only a different resolution of the same label), masked-residue
recovery (real extra supervision but about sequence, not bindability), and probe occupancy. An interface head
adds a **different, dense, bindability-flavoured label** from the same file. And unlike P10's CryptoBench, the
licence question does not arise: the labels come from the structures already in `data/pockets_ds/pdb`, which the
data card already covers.

**On conservation I would decline.** PrankWeb 3 (Jakubec et al., *Nucleic Acids Research* 50:W593–W597, 2022)
**[V on the claim]** offers a P2Rank+Conservation model with "a new, more accurate evolutionary conservation
estimation pipeline based on the UniRef50 sequence database and the HMMER3 package", described as "capable of
improving the default P2Rank predictions" — but **the magnitude is [NOT FOUND]** in this session. Against that:
it requires HMMER3 plus a UniRef50 copy **at inference**, which is a heavy new dependency that cuts against this
project's "nothing external at inference" constraint as much as P2Rank/Java would; and we already carry frozen
**ESM-2** features, which the survey records as **the biggest single learned-feature gain in the field** (VN-EGNN
+0.06 DCC; GDEGAN +15.61 % DCC / +9.27 % DCA **[V]**) and which encode an evolutionary prior by construction. The
expected marginal information from an explicit MSA conservation column, given ESM-2 is already in, is small, and
the cost is large. **Rank last; I would not do it.**

**4. Implementation sketch.**
- `training/pockets/data.py` in `featurize`: `rt["chain"]` is already available (used at line 175). Build
  `y_iface [L]` with one `cKDTree` per chain pair over heavy atoms — for ≤10 chains this is a few queries. Also
  `y_rsasa [L]`: per-residue SAS point count from `pk.sas_points` (already computed at line 155) divided by a
  per-residue-type maximum. ~25 lines. **Single-chain entries get an all-zero `y_iface` and must be masked out of
  the loss, not trained as all-negative** — otherwise the head learns "monomers have no interfaces", which is
  circular. Store `has_iface: bool`.
- `training/pockets/model.py`: `self.head_iface = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim,
  2))` emitting interface logit and rSASA, `o["iface_logit"]`, `o["rsasa_pred"]` → `[L]`, `[L]`. (The e3nn
  backbone already has an unused `head_aux` of exactly this width, `model_e3nn.py:98` — mirror it.) 4 lines.
- `net_task.losses()`: `part["iface"] = seg_loss(...)` when `b["has_iface"]`, `part["rsasa"] =
  Fn.smooth_l1_loss(...)`; `loss.w_iface: 0.3`, `loss.w_rsasa: 0.2`. ~8 lines.
- `ablations.yaml`: `no_interface_aux: {loss.w_iface: 0.0, loss.w_rsasa: 0.0}`.
- Total ≈ **45 lines**.

**5. Equivariance impact.** **None** — scalar heads on invariant features. Existing `test_so3_equivariance` covers
them once `"iface_logit"` is added to the loop at `tests/test_model.py:45` (1 line).

**6. Cost.** Parameters: ≈ 16.8 k, **<0.5 %**. Memory: zero. Time: **1.00×** at the GPU step; featurisation gains a
few KD-tree queries.

**7. Falsifiable test.** Ablation `no_interface_aux`, ≥3 seeds. Metrics: **`net_sites.top1`** and **`res_ap`**; also
report interface AUPRC itself so we know the head is learning anything at all (a head that does not learn its own
task cannot be regularising the trunk).
*What would mean the idea is wrong:* interface AUPRC is high (the head learns) but `net_sites.top1` is flat or
down. That is the expected failure mode and it has a specific mechanism: ligand pockets and protein interfaces
**overlap**, so the head may push the trunk to call every interface a pocket — raising top-1 on interfacial
ligands and lowering it on buried ones. **Report per-subset** (structures whose annotated site is within 5 Å of
an interface vs not); if the two subsets move in opposite directions, the head is trading, not helping, and
should be dropped.

**8. Risk, and what I would drop it for.** The overlap confound above, and the monomer-masking trap. Also: our
manifest's chain composition is whatever the PDB entries happen to contain, and the survey's flaw 6 **[V]**
records that **40 % of HOLO4K and 56 % of COACH420 structures differ in chain count between asymmetric and
biological unit** — so "interface" computed from an asymmetric unit may be a crystallographic artefact rather
than biology, which would teach the head noise. Mitigation: restrict `y_iface` to chain pairs with ≥ 20
interface residues. I would drop P11 for P1 and P3; it is the weakest-evidenced of the auxiliary-target
proposals for *our* metric even though its source is the strongest paper in the list.

---

## P12. An equivariant direction-to-ligand loss on every probe (the only mechanism with two independent positive measurements in the pocket literature)

**1. Mechanism.** A second equivariant vector read-out on every probe, supervised by a cosine loss to point at the
nearest ligand heavy atom of its site, for every probe within 8 Å of exactly one site — not only the ~2 proposals
the Hungarian matching selects.

**2. Primary source.**
- **EquiPocket** (Zhang et al., ICML 2024; PMLR v235) **[V in survey answer A]**: removing the auxiliary
  **direction loss** towards the ligand drops **DCC 0.428 → 0.319** for proteins under 1000 atoms and
  **0.621 → 0.587** for 1000–2000 atoms.
- **GDEGAN** (arXiv:2603.19817) **[V in survey answer A]**: the auxiliary directional loss (ADL) adds
  **≈2 % DCC and 3.5 % DCA**.
- These are **two independent pocket papers with their own ablations both reporting a positive effect** for the
  same mechanism. In the survey's answer-A table, only three mechanisms have that status, and this is the only one
  of the three that we do not already implement.

**3. Why it should help this task.** Two distinct arguments. First, **supervision density**: `center_set_loss`
with `hungarian=True` takes the top-32 proposals by confidence, matches them one-to-one to the S≈1–3 true sites,
and takes the mean of S matched distances (`model.py:303-309`). So **per structure, the geometric gradient reaches
at most 3 of 768 probes.** A direction loss over all probes within 8 Å of a site reaches ~50–200 probes — one to
two orders of magnitude more geometric gradient per step, at zero extra cost. Second, **it targets the axis where
we are measurably weak and where the ranker structurally cannot help**: §0 fact 4, ranker top-1 is 0.833 by DCA
but **0.667 by DCC** against a 0.833 DCC ceiling. The survey's answer G, statement 2, says the published
equivariant advantage *is* concentrated in DCC. This is the proposal where our architecture choice and our
measured weakness line up.

**4. Implementation sketch.**
- `training/pockets/data.py`: cache `probe_dir [n_probe, 3]` (unit vector from each probe to the nearest ligand
  heavy atom) and `probe_dir_mask [n_probe]` (True for probes within 8 Å of exactly one site, computed from the
  existing `site_probe_mask`, whose row sums give the site count directly: `site_probe_mask.sum(0) == 1`).
  ~12 lines. `data.random_rotation` must rotate `probe_dir` — **this is the easiest thing to forget and it would
  silently destroy the loss**, so it goes in the same commit as the test.
- `training/pockets/model.py`: `self.dir_vec = nn.Linear(dim, 1, bias=False)` and
  `o["dir"] = einsum("nfc,f->nc", V[probe], self.dir_vec.weight[0])` → `[P, 3]` (the same pattern as `off_vec`,
  `model.py:234`). 3 lines. Plus
  ```
  def direction_loss(d_pred, d_true, mask):
      if mask.sum() == 0: return d_pred.sum() * 0
      p = Fn.normalize(d_pred[mask] + 1e-8, dim=-1)
      return (1.0 - (p * d_true[mask]).sum(-1)).mean()
  ```
  ~6 lines.
- `net_task.losses()`: `part["dir"] = M.direction_loss(o["dir"], b["probe_dir"], b["probe_dir_mask"])` on every
  pass (it is a per-probe target, so it supervises recycling passes too); `loss.w_dir: 0.5`.
- `net_task.evaluate()`: **add the DCC metric**, without which neither this nor P4 can be judged — label each
  predicted centre by distance to the nearest *site centroid* (`d["site_centers"]`) at 4 Å and at 10 Å, alongside
  the existing DCA label at `net_task.py:133`, and add `net_sites_dcc.*` to `collect_ablations.KEYS`. ~12 lines.
  LIGYSIS **[V, flaw 3]** explicitly recommends a DCC threshold of 10–12 Å rather than 4 Å, so report both.
- `ablations.yaml`: `no_direction_loss: {loss.w_dir: 0.0}`.
- Total ≈ **45 lines**.

**5. Equivariance impact.** `o["dir"]` is a **vector read-out from `V` and must be exactly equivariant**. Extend
`test_so3_equivariance` with `assert allclose(o1["dir"] @ R.T, o2["dir"], atol=1e-4)` (1 line) and add
`test_direction_target_rotates_with_the_structure`: apply `data.random_rotation` to a batch containing
`probe_dir` and assert `probe_dir` rotated — this is the test that catches the forgotten rotation above, and it
is the single highest-value-per-line test in this document. ~12 lines. Note that for the `invariant_distances`
arm `V` is None, so `head_dir` must fall back to an invariant MLP and be **reported as not equivariant there**,
exactly as `off_inv` already is (`model.py:192`) — or, better, use P4's frame-averaged read-out so the invariant
arm gets a fair direction head too.

**6. Cost.** Parameters: **128** (one linear row). Memory: `[768, 3]` per structure — nothing. Time: **1.00×**.

**7. Falsifiable test.** Ablation `no_direction_loss`, ≥3 seeds. Metrics: **the new DCC-based `net_sites` top-1 at
4 Å and 10 Å** (primary — this is what EquiPocket measured), and `net_sites.top1` by DCA (secondary).
*What would mean the idea is wrong:* DCC top-1 flat. EquiPocket's effect was **size-dependent** (large for
proteins under 1000 atoms, small for 1000–2000), so stratify by protein size before concluding — a flat *average*
with a positive effect on small proteins is a partial confirmation, not a refutation, and our manifest's size
distribution differs from theirs.

**8. Risk, and what I would drop it for.** The target is ambiguous for a large or multi-ligand site, where
"nearest ligand heavy atom" can point to a different atom for adjacent probes, giving a discontinuous field that
no equivariant read-out can fit — hence the single-site mask. It also competes with `center`: both are geometric
read-outs from `V`, and at `w_dir: 0.5, w_center: 0.5` they may blur. Mitigation: use **separate linear rows**
(`off_vec` and `dir_vec` are independent, which the sketch does) and report both losses separately in
`history.json`. I would drop P12 for P1 and P6 only; of all the model-side proposals it has the best
source-to-cost ratio.

---

## Directions I would decline, with reasons

| Direction | Why not |
|---|---|
| **Triangle multiplicative updates between probes/candidates (AF2/AF3-style)** | Triangle updates enforce metric consistency on a pair representation being *constructed*. Our candidate-pair distances are exact Euclidean distances between known coordinates, so the constraint already holds and there is nothing to enforce; the cost is O(C³). Plain pair-biased attention (P5) delivers the within-structure comparison with none of that. Also: the AF2 ablation numbers are **[UNVERIFIED]** here (Nature and PMC are both gated in this environment), so the direction would be adopted on an unread citation. |
| **GradNorm / PCGrad / MGDA** | Require per-task gradients — 8–9 backward passes per step, a 6–9× training cost — for a mechanism that **Kurin et al., NeurIPS 2022 [V]** found to be matched by the plain sum with standard regularisation, and which they characterise as a regulariser in disguise. Indefensible at a one-GPU-week budget. Use uncertainty weighting (P7) if anything. |
| **Explicit MSA conservation features** | Needs HMMER3 + UniRef50 at inference (a heavy new dependency, against the project's own constraint), the magnitude of the gain is **[NOT FOUND]**, and frozen ESM-2 — already in the model and recorded by the survey as the field's biggest learned-feature gain — encodes an evolutionary prior by construction. Low expected marginal information, high cost. |
| **Side-chain / coordinate perturbation augmentation as a priority** | The one direct measurement **[V]** puts it at **−0.2 (COACH420) / −0.8 (HOLO4K)** points, 5–25× smaller than homology augmentation or transfer learning in the same table; and it cannot open a *closed* cryptic pocket, which is what the free-space probe generator actually needs. Keep as a cheap robustness check (P10b), not a priority. |
| **Going above degree 2 (e3nn l = 3+) as an accuracy play** | The e3nn `lmax` sweep is worth running **as a scientific result** (P9) because the field has no answer, but as an accuracy bet it is poor: GDEGAN stops at l = 2 and never ablates it; HEGNN's degree argument is demonstrated on symmetric graphs with **no protein task [V]**; and the survey's blunt conclusion is that features and pipeline design each buy more than any choice of equivariant layer. l = 3 roughly doubles the irrep width for a mechanism with zero pocket evidence. |
| **Training on CryptoBench / any external set before reading its licence** | Project rule: anything trained on data with an unverified licence is rejected. CryptoBench is on OSF and its terms are **[UNVERIFIED]**. Evaluation-only use, after reading the licence, is the most that is permissible now. |

---

## Ranked roadmap for one GPU week

**Assumptions, stated because they are guesses, not measurements** (there is no GPU in this environment, so no
timing has been observed): one structure per step, 1367 structures, ~40 k edges, `dim 128 / layers 5 / recycles 2`
→ **≈3.5 h per 60-epoch single-fold single-seed run on one A100**; 3 seeds ≈ 10.5 h; a 5-fold `mode=oof` run
≈ 17 h. One GPU week = 168 h. **If the real per-run time is 2× this estimate, cut in the order given from the
bottom.**

### Step 0 — before the GPU is booked (CPU only, ~1 day, no GPU hours)

| # | Task | Why it must come first |
|---|---|---|
| 0a | `train_ranker.py --tag native2 --features-extra net --ablate` to get the **`potential` group's contribution** to the existing ranker | Never measured. It is the prior on P1, the highest-ranked proposal. One CPU job. |
| 0b | Write `test_degree2_separates_what_degree1_cannot` (P9) | If our degree-2 channels cannot separate an octahedron from a hexagon, no training run can fix it, and P9's entire ablation is void. Milliseconds to run. |
| 0c | Add the **DCC metric** to `net_task.evaluate` and `collect_ablations.KEYS` (P12) | P4, P9 and P12 are all unjudgeable without it, and it is 12 lines. |
| 0d | Add the **redundancy statistic** to the ablation keys (P5) — `metrics.redundancy` already exists | No pocket paper reports it (survey flaw 4); free to add; needed to judge P5. |
| 0e | Implement **P1** and **P3** and rebuild the cache | Both are cache/CPU work with zero GPU cost; P1 must be in the cache before the baseline, or the baseline is the wrong baseline. |
| 0f | **P10a**: CryptoBench apo candidate-recall ceiling, after reading its licence | Closes or opens a whole direction for CPU hours. |
| 0g | `scripts/train/grad_norms.py` (P7's diagnostic) on the CPU pilot config | May close P7 before it costs a GPU hour. |

### GPU week

| order | h | arm | expected gain (my estimate, not a measurement) | reason for this position |
|---|---|---|---|---|
| 1 | 0–11 | **`full` baseline with P1 included, 3 seeds, fold 0** | — (reference) | The network has **never been trained**. Without a converged baseline no ablation means anything. **Gate: if `net_sites.top1` < 0.516 (detector order), stop and debug; everything downstream is uninterpretable.** P1 is folded in because it is cache-level and free, and its own ablation (`no_probe_potential`) is run later as the *comparison*. |
| 2 | 11–22 | **`no_probe_potential` (isolates P1), 3 seeds** | P1 worth **+0.01 to +0.04** `net_sites.top1` | Cheapest proposal, aimed at the feature group our own ranker says matters most (−0.075 top-1 for chemistry). Second only because it is measured *as a removal* from the baseline. |
| 3 | 22–33 | **P6 staged training, 3 seeds** (arm `single_stage` is the comparison; staged is the new default if it wins) | **+0.02 to +0.06** | Best-evidenced item in the document (+5.2/+7.0 F1 in Lee/Byun/Shin **[V]**), **zero parameter and zero time cost**, and it fixes a structural defect (sparse ranking signal competing with a dense one from epoch 1). Before P2 because P2's sparse loss benefits from a trunk that is already converged. |
| 4 | 33–44 | **P12 direction loss, 3 seeds** | **+0.03 to +0.08 on DCC top-1**, +0.00 to +0.02 on DCA top-1 | Two independent pocket ablations **[V]**, 45 lines, 128 parameters, and it is the only proposal aimed at the axis (DCC 0.667 vs DCA 0.833) where the equivariant read-out can beat the gradient-boosted ranker structurally. Also multiplies the geometric gradient by ~50×. |
| 5 | 44–55 | **P2 listwise ranking loss, 3 seeds** | **+0.01 to +0.05** on `net_sites.top1`; unknown on ranker top-1 | Our bottleneck is ordering and the network currently has no ordering objective at all. Behind P6 and P12 because its signal is sparse (~30 items/structure) and it is the proposal most likely to need the schedule fix from P6 to work. |
| 6 | 55–66 | **P4 `invariant_frames`, 3 seeds** | expected **0.00**; value is scientific, not numeric | **The paper-critical arm.** Today's `invariant` ablation confounds three things and a reviewer will say so. Placed here, not later, because if `invariant_frames` matches `full` the entire framing of the paper changes and we want to know that with 100 GPU hours still in hand. |
| 7 | 66–78 | **P9 `no_tensors_matched` + `e3nn_l1`, 3 seeds each at ~1.15×** | expected **0.00 to 0.02**; value is scientific | Same reasoning: the degree-2 claim needs a matched arm or it is unpublishable. Paired with the `e3nn` degree sweep so the two backbones corroborate each other. Behind P4 because P4 fixes the larger confound. |
| 8 | 78–91 | **P8 PU prior sweep, π ∈ {0, 0.04, 0.08, 0.16}, 1 seed each** | **+0.00 to +0.04**; wide | Highest-variance idea here: attacks the bottleneck from a direction nobody in the field has tried, costs 25 lines, but π is unknown and the estimator is prior-sensitive. A 1-seed sweep is the cheapest way to find out whether a 3-seed run is justified; promote to 3 seeds in the reserve block if any π wins. |
| 9 | 91–108 | **`mode=oof` 5-fold on the winning stack → P3's ~40 candidate features → ranker** | **+0.01 to +0.04 on ranker top-1** (the number that actually ships) | This is the only run that produces `net_features_*.csv`, and it must come **after** the configuration is settled or it has to be repeated. P3 itself costs zero GPU hours; the 17 hours are the OOF training. The ranker that consumes it is a CPU job. **This is the step that converts everything above into the shipped metric.** |
| 10 | 108–119 | **P5 candidate tokens, 3 seeds** | **+0.00 to +0.03** on top-1; **−20 to −50 % redundancy** | Behind the OOF run because it is additive to P2 and its main measurable (redundancy) is new and interesting regardless of top-1. If it wins, the OOF run is repeated in the reserve block. |
| 11 | 119–130 | **P7 uncertainty weights, 3 seeds** — *only if* step 0g showed per-term gradient norms spanning >2 orders of magnitude | **+0.00 to +0.02** | Kurin et al. **[V]** say the plain sum usually wins. Run only on evidence, not on principle. |
| 12 | 130–141 | **P11 interface + rSASA auxiliary, 3 seeds** | **+0.00 to +0.02**, with a real chance of **negative** on non-interfacial sites | Strongest *paper* in the list (PeSTo, *Nat. Commun.* 2023 **[V]**) but the weakest argument for *our* metric, and the overlap confound (interfaces vs pockets) could trade one subset for another. Last of the model-side arms. |
| — | 141–168 | **Reserve**: 5-fold confirmation of every arm that won, and a repeat of the OOF run if P5/P7/P11 changed the stack | — | Every number in this document that gets published needs ≥3 seeds and the paired cluster bootstrap that `collect_ablations.py` already computes. Budget for it, or the week produces unpublishable single-seed deltas. |
| — | — | **Not scheduled**: P10b side-chain jitter, l = 3, conservation, triangle updates, GradNorm | — | Declined above, with reasons. |

### Why this order, in one paragraph

The order is: **(i) get a baseline at all**, because nothing is interpretable without one and the network has
never been trained; **(ii) the free wins first** — P1 is cache-level, P6 is a schedule, P12 is 128 parameters, and
all three have primary-source support or our own measurement behind them; **(iii) the ordering objective next**,
because ordering is the bottleneck but the loss is sparse and benefits from a converged trunk; **(iv) the two
paper-critical ablation-validity arms (P4, P9) in the middle of the week**, not at the end, because if the
equivariance or degree-2 claim fails we want time to re-frame; **(v) the OOF run late but not last**, because it
is the only step that produces the number that ships and it must run on a settled configuration; **(vi) the
speculative and weakly-evidenced arms last**, where cutting them costs least. The deliberate consequence is that
the week's most likely outcome is **+0.02 to +0.05 ranker top-1 plus the first controlled equivariance and
degree-2 ablation for pocket detection** — and if the accuracy gain is nil, the ablation table is still a genuine
contribution, because the survey establishes **[V]** that it does not exist in the literature.

---

## Sources

- [Rank & Sort Loss for Object Detection and Instance Segmentation (ICCV 2021)](https://arxiv.org/abs/2107.11669)
- [The LambdaLoss Framework for Ranking Metric Optimization (CIKM 2018)](http://bibtex.github.io/CIKM-2018-WangLGBN.html)
- [In Defense of the Unitary Scalarization for Deep Multi-Task Learning (NeurIPS 2022)](https://papers.nips.cc/paper_files/paper/2022/hash/4f301ae934f396086bfefd1139039dbd-Abstract-Conference.html)
- [Multi-Task Learning Using Uncertainty to Weigh Losses (CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)
- [Frame Averaging for Invariant and Equivariant Network Design (ICLR 2022)](https://openreview.net/forum?id=zIUyj55nXR)
- [Boosting CNN Protein Binding Site Prediction with SE(3)-invariant transformers, transfer learning and homology-based augmentation (Lee, Byun & Shin)](https://arxiv.org/html/2303.08818v2)
- [Positive-Unlabeled Learning with Non-Negative Risk Estimator (NIPS 2017)](https://arxiv.org/abs/1703.00593)
- [End-to-End Object Detection with Transformers (DETR, ECCV 2020)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460205.pdf)
- [On the Expressive Power of Geometric Graph Neural Networks (ICML 2023)](https://arxiv.org/abs/2301.09308)
- [Learning from Protein Structure with Geometric Vector Perceptrons (ICLR 2021)](https://openreview.net/forum?id=1YLJDvSx6J4)
- [Protein ligand binding site prediction using graph transformer neural network (PLOS ONE 2024)](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0308425)
- [CryptoBench: cryptic protein-ligand binding sites dataset and benchmark (Bioinformatics 2025)](https://pubmed.ncbi.nlm.nih.gov/39693053/)
- [Predicting locations of cryptic pockets with PocketMiner (Nat. Commun. 14:1177, 2023)](https://www.nature.com/articles/s41467-023-36699-3)
- [PrankWeb 3 (Nucleic Acids Research 2022)](https://academic.oup.com/nar/article/50/W1/W593/6591527)
- [Using Multiple Vector Channels Improves E(n)-Equivariant GNNs](https://arxiv.org/abs/2309.03139)
- All EquiPocket / VN-EGNN / GDEGAN / LIGYSIS / GrASP / P2Rank / PharmacoNet / PeSTo / HEGNN citations are as
  recorded and tagged in `docs/LITERATURE_SURVEY.md`; they were not re-read in this session and keep that
  document's tags.
