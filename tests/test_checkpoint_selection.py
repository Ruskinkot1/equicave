"""Checkpoint selection has to be able to see the task the model is for.

Taken from the first full GPU run (fold 0, seed 0, `runs/training/ablations/full_fold0_seed0/history.json`). The
staged schedule holds the centre, confidence and listwise terms at zero for twenty epochs; when they switch on,
site top-1 goes 0.040 -> 0.868 and the median centre error 16.4 A -> 1.1 A within four epochs. The selection score
was the mean of occ_ap and res_ap, which over those same epochs went 0.790 -> 0.790 and 0.711 -> 0.706 -- so it
fell slightly across the only change that mattered. The consequences were both of the ones available: the kept
checkpoint was epoch 14 (metrics.json: top1 0.022, ceiling 0.118) and patience stopped the run at 24 while top-1
was still climbing. Every ablation arm would have been scored at that same blind checkpoint, which would have made
the whole grid read as noise around a broken model.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from training.pockets.net_task import SELECT_WEIGHTS, select_score  # noqa: E402

# epoch -> the three numbers the score can be built from, as that run recorded them
RUN = {
    14: dict(occ_ap=0.7957, res_ap=0.7148, top1=0.0221),      # was selected
    20: dict(occ_ap=0.7897, res_ap=0.7106, top1=0.0404),      # last stage-1 epoch
    22: dict(occ_ap=0.7940, res_ap=0.7099, top1=0.8456),      # the centre head starts working
    24: dict(occ_ap=0.7903, res_ap=0.7057, top1=0.8676),      # best site detection; run stopped here
}


def score(e, cfg=None):
    r = dict(occ_ap=e["occ_ap"], res_ap=e["res_ap"], net_sites={"top1": e["top1"]})
    return select_score(r, cfg)


def test_the_old_score_preferred_the_blind_checkpoint():
    # The regression this guards, stated as the arithmetic that caused it.
    old = {ep: (e["occ_ap"] + e["res_ap"]) / 2 for ep, e in RUN.items()}
    assert max(old, key=old.get) == 14
    assert old[24] < old[14]


def test_selection_now_prefers_the_epoch_that_finds_pockets():
    now = {ep: score(e) for ep, e in RUN.items()}
    assert max(now, key=now.get) == 24, now
    assert now[22] > now[20], now
    # and by a margin no amount of drift in the per-node heads can close
    assert now[24] - now[14] > 0.3, now


def test_site_detection_carries_real_weight():
    assert SELECT_WEIGHTS["net_top1"] >= 0.5
    assert abs(sum(SELECT_WEIGHTS.values()) - 1.0) < 1e-9


def test_an_arm_that_predicts_no_sites_is_not_scored_zero():
    # no_probes has no site metric rather than a bad one; dropping the weight is not the same as scoring it 0.
    no_sites = select_score(dict(occ_ap=0.80, res_ap=0.70), None)
    assert abs(no_sites - 0.75) < 1e-9, no_sites
    assert no_sites > score(RUN[14]), "an arm without sites must not be punished below a blind checkpoint"


def test_weights_can_be_overridden_from_the_config():
    cfg = dict(optim=dict(select_weights={"net_top1": 1.0, "occ_ap": 0.0, "res_ap": 0.0}))
    assert abs(score(RUN[24], cfg) - 0.8676) < 1e-9


def test_missing_everything_is_nan_not_an_exception():
    import math
    assert math.isnan(select_score({}, None))
