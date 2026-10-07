"""pockets-net: multi-task training of EquiCave-Net (segmentation, centres, properties, hotspot field).

Usage: python -m training pockets-net --config training/configs/pockets_net.yaml --out runs/training/pockets-net \
           [--device cuda] [--set split.val_fold=1 optim.seed=2 ablation=no_esm]
The task featurises the manifest into a cache once (data.build_cache), trains on all folds but `split.val_fold`,
evaluates on that fold every epoch, keeps the best EMA weights, and writes model_card.json + metrics.json.
Out-of-fold network features for the ranker: `--set mode=oof` trains one model per fold and writes
data/processed/net_features_<tag>.csv (net_seg, net_center_conf, net_hot_mean per native candidate).
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from equicave import progress
from training.common.config import load_config, pick_device, run_dir, seed_all, write_model_card
from training.pockets import data as D
from training.pockets import model as M

DEFAULTS = Path(__file__).resolve().parents[1] / "configs/pockets_net.yaml"


def _cfg(a):
    cfg = load_config(a.config or DEFAULTS, [])
    import yaml
    abl = yaml.safe_load((DEFAULTS.parent / "ablations.yaml").read_text())["ablations"]
    over = list(a.set)
    name = next((kv.split("=", 1)[1] for kv in over if kv.startswith("ablation=")), "full")
    over = [kv for kv in over if not kv.startswith("ablation=")]
    for k, v in abl.get(name, {}).items():
        over.insert(0, f"{k}={json.dumps(v)}")
    cfg = load_config(a.config or DEFAULTS, over)
    cfg["ablation"] = name
    return cfg


class EMA:
    def __init__(self, model, decay):
        self.decay = decay; self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model):
        import torch
        with torch.no_grad():
            for s, p in zip(self.shadow.parameters(), model.parameters()):
                s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)


def losses(out, b, w, res_mask=None, pot_mask=None, pot_target=None):
    """Weighted multi-task loss. Every pass of a recycled forward is supervised (deep supervision), the last one fully."""
    import torch
    passes = out.get("passes", [out])
    L, total = {}, 0.0
    for i, o in enumerate(passes):
        last = i == len(passes) - 1
        scale = 1.0 if last else w.get("recycle_weight", 0.5)
        part = {}
        part["res"] = M.seg_loss(o["res_logit"], b["y_res"], w["res_pos_weight"])
        part["occ"] = M.seg_loss(o["occ_logit"], b["y_occ"], w["occ_pos_weight"])
        reg, conf = M.center_set_loss(o["center"], o["conf_logit"], o.get("probe_pos", b["pos"][b["slices"]["probe"]]),
                                      b["site_centers"], hungarian=w.get("hungarian", True))
        part["center"], part["conf"] = reg / 4.0, conf
        part["hot"] = M.focal_bce(o["hot_logit"], b["y_hot"])
        if "dir" in o and "probe_dir" in b:            # dense geometric signal: every probe near exactly one site
            part["dir"] = M.direction_loss(o["dir"], b["probe_dir"], b["probe_dir_mask"])
        if w.get("w_listwise", 0.0) and "site_centers" in b:   # orders probes within a structure
            part["listwise"] = M.listwise_site_loss(o["conf_logit"], o["center"], b["site_centers"],
                                                    o.get("probe_pos", b["pos"][b["slices"]["probe"]]))
        if last and "site_logit" in o and "site_centers" in b:
            # The ranking objective on the list the metric is computed from, plus the single comparison top-1
            # actually decides: best correct site against best incorrect one.
            part["site_rank"] = M.site_rank_loss(o["site_logit"], o["site_center"], b["site_centers"])
            part["site_margin"] = M.site_margin_loss(o["site_logit"], o["site_center"], b["site_centers"],
                                                     margin=w.get("site_margin", 1.0))
        if "prop_logit" in o:
            part["prop"] = torch.nn.functional.binary_cross_entropy_with_logits(o["prop_logit"], b["y_prop"])
        if last and res_mask is not None and "seq_logit" in o and "res_type" in b:
            part["seq"] = M.masked_residue_loss(o["seq_logit"], b["res_type"], res_mask)
        if last and "size_pred" in o and "site_n_atoms" in b:
            part["size"] = M.size_loss(o["size_pred"], b["site_n_atoms"])
        if last and pot_mask is not None and "pot_logit" in o and pot_target is not None:
            part["potential"] = M.potential_loss(o["pot_logit"], pot_target, pot_mask)
        total = total + scale * sum(w.get(f"w_{k}", 0.0) * v for k, v in part.items())
        if last:
            L = part
        else:
            L.update({f"{k}@{i}": v for k, v in part.items()})
    return total, {k: float(v.detach()) if hasattr(v, "detach") else float(v) for k, v in L.items()}


def predict_sites(out, probe_pos, nms: float = 6.0, max_sites: int = 30):
    """Network-only site list, ranked.

    With the site decoder the list is its own: the tokens were formed by this same NMS rule inside the model and
    then scored against one another, so its ordering is the model's answer and re-deriving one from per-probe
    confidence would discard exactly the comparison the decoder exists to make. Without it (the `no_site_decoder`
    arm, and any older checkpoint) the list is the predicted centres in confidence order under NMS.
    """
    import torch
    if "site_logit" in out and len(out["site_logit"]):
        sl = out["site_logit"].detach().cpu().numpy()
        sc = out["site_center"].detach().cpu().numpy()
        order = np.argsort(-sl)[:max_sites]
        return sc[order], sl[order]
    conf = torch.sigmoid(out["conf_logit"]).detach().cpu().numpy(); c = out["center"].detach().cpu().numpy()
    order = np.argsort(-conf); keep = []
    for i in order:
        if all(np.linalg.norm(c[i] - c[j]) > nms for j in keep):
            keep.append(i)
        if len(keep) >= max_sites:
            break
    return c[keep], conf[keep]


def net_features(out, probe_pos, centers: np.ndarray, r: float = 4.0, radii=(4.0, 8.0)) -> list[dict]:
    """Per native candidate centre: the network's view of that site, as features a ranker can use.

    Three scalars threw away most of what the network knows: the seven hotspot classes were collapsed by a max, so a
    strongly hydrophobic site and a strongly polar one produced the same number. Here each class is kept, at two
    radii, with mean and max, together with occupancy, confidence, how many predicted centres fall nearby and how far
    the nearest one is. Everything is a rotation-invariant scalar: no vector component is exported, or the ranker
    would become frame-dependent and nothing downstream would notice.
    """
    import torch
    from scipy.spatial import cKDTree
    occ = torch.sigmoid(out["occ_logit"]).detach().cpu().numpy()
    conf = torch.sigmoid(out["conf_logit"]).detach().cpu().numpy()
    hot = torch.sigmoid(out["hot_logit"]).detach().cpu().numpy()
    pc = out["center"].detach().cpu().numpy()
    offset_norm = np.linalg.norm(out["offset"].detach().cpu().numpy(), axis=1)
    tp, tc = cKDTree(probe_pos), cKDTree(pc)
    n_hot = hot.shape[1]
    # The site decoder's own score, carried to whichever native candidate each of its sites belongs to. This is the
    # one column that holds a comparison between pockets rather than a summary of one pocket, so it is the column
    # the `network only` ranking uses.
    site_sc = site_ct = None
    if "site_logit" in out and len(out["site_logit"]):
        site_sc = out["site_logit"].detach().cpu().numpy()
        site_ct = out["site_center"].detach().cpu().numpy()
        site_tree = cKDTree(site_ct)
    rows = []
    for ctr in centers:
        row = {}
        for rad in radii:
            nb = tp.query_ball_point(ctr, rad)
            tag = int(rad)
            row[f"net_occ_mean_{tag}"] = float(occ[nb].mean()) if nb else 0.0
            row[f"net_occ_max_{tag}"] = float(occ[nb].max()) if nb else 0.0
            row[f"net_conf_mean_{tag}"] = float(conf[nb].mean()) if nb else 0.0
            row[f"net_conf_max_{tag}"] = float(conf[nb].max()) if nb else 0.0
            row[f"net_offset_mean_{tag}"] = float(offset_norm[nb].mean()) if nb else 0.0
            for j in range(n_hot):
                row[f"net_hot{j}_mean_{tag}"] = float(hot[nb, j].mean()) if nb else 0.0
                row[f"net_hot{j}_max_{tag}"] = float(hot[nb, j].max()) if nb else 0.0
            row[f"net_n_probes_{tag}"] = float(len(nb))
        nc = tc.query_ball_point(ctr, r)
        row["net_center_conf"] = float(conf[nc].max()) if nc else 0.0
        row["net_n_centers"] = float(len(nc))
        row["net_center_dist"] = float(np.linalg.norm(pc - ctr, axis=1).min()) if len(pc) else 99.0
        row["net_seg"] = row["net_occ_mean_4"]                      # kept so older models and tables still work
        row["net_hot_mean"] = float(np.mean([row[f"net_hot{j}_mean_4"] for j in range(n_hot)]))
        if site_sc is not None:
            near = site_tree.query_ball_point(ctr, 8.0)
            row["net_site_score"] = float(site_sc[near].max()) if near else float(site_sc.min() - 1.0)
            row["net_site_dist"] = float(np.linalg.norm(site_ct - ctr, axis=1).min())
        rows.append(row)
    return rows


def evaluate(model, files, device, cfg, log=print) -> dict:
    import torch
    from equicave import metrics as MT
    ys = {k: [] for k in ("res", "occ")}; ps = {k: [] for k in ("res", "occ")}
    yh, ph, yp, pp, yprox = [], [], [], [], []
    rank_rows = []
    model.eval()
    with torch.no_grad():
        for f in files:
            d = D.load(f); b = D.to_torch(d, device); out = model(b)
            ys["res"].append(d["y_res"]); ps["res"].append(torch.sigmoid(out["res_logit"]).cpu().numpy())
            ys["occ"].append(d["y_occ"]); ps["occ"].append(torch.sigmoid(out["occ_logit"]).cpu().numpy())
            yh.append(d["y_hot"]); ph.append(torch.sigmoid(out["hot_logit"]).detach().cpu().numpy())
            if "y_hot_proximity" in d:
                yprox.append(d["y_hot_proximity"])
            if "prop_logit" in out:
                yp.append(d["y_prop"]); pp.append(torch.sigmoid(out["prop_logit"]).cpu().numpy())
            c, conf = predict_sites(out, d["pos"][d["n_res"]:d["n_res"] + d["n_probe"]])
            L = d["lig_xyz"]; sites = d["site_centers"]; n_sites = len(sites)
            for k, ctr in enumerate(c, 1):
                dca = float(np.linalg.norm(L - ctr, axis=1).min())
                dcc = float(np.linalg.norm(sites - ctr, axis=1).min()) if len(sites) else float("inf")
                rank_rows.append(dict(pdb=d["pdb"], cluster30=str(d.get("cluster30", d["pdb"])), n_sites=n_sites, rank=k,
                                      score=float(conf[k - 1]), label=int(dca <= 4.0), label_dcc=int(dcc <= 4.0),
                                      label_dcc10=int(dcc <= 10.0), dca=dca, dcc=dcc))
    res = {}
    for k in ("res", "occ"):
        y, p = np.concatenate(ys[k]), np.concatenate(ps[k])
        res[f"{k}_auroc"] = MT.auroc(y, p); res[f"{k}_ap"] = MT.average_precision(y, p); res[f"{k}_ece"] = MT.ece(y, p)
    Y, P = np.concatenate(yh), np.concatenate(ph)
    res["hot_ap_per_class"] = [MT.average_precision(Y[:, j], P[:, j]) for j in range(Y.shape[1])]
    res["hot_auroc_per_class"] = [MT.auroc(Y[:, j], P[:, j]) for j in range(Y.shape[1])]
    res["hot_enrichment_top10_per_class"] = [MT.enrichment_at(Y[:, j], P[:, j], 0.1) for j in range(Y.shape[1])]
    res["hot_ece_per_class"] = [MT.ece(Y[:, j], P[:, j]) for j in range(Y.shape[1])]
    res["hot_permutation_p_per_class"] = [MT.permutation_control(Y[:, j], P[:, j], MT.average_precision, n=50)[1]
                                          if Y[:, j].any() else float("nan") for j in range(Y.shape[1])]
    if yprox:                      # the same predictions scored against the plain proximity target, for the ablation
        Yp = np.concatenate(yprox)
        res["hot_ap_per_class_proximity_target"] = [MT.average_precision(Yp[:, j], P[:, j]) for j in range(Yp.shape[1])]
        res["interaction_validated_fraction"] = (Y.sum(0) / np.maximum(Yp.sum(0), 1)).round(3).tolist()
    if yp:
        Y, P = np.concatenate(yp), np.concatenate(pp)
        res["prop_auroc_per_class"] = [MT.auroc(Y[:, j], P[:, j]) for j in range(Y.shape[1])]
        res["prop_ap_per_class"] = [MT.average_precision(Y[:, j], P[:, j]) for j in range(Y.shape[1])]
    if rank_rows:
        rr = pd.DataFrame(rank_rows)
        per = MT.per_structure(rr, "score")
        res["net_sites"] = {k: float(per[k].mean()) for k in ("top1", "top3", "topN", "topN2", "ceiling")}
        # DCC, the criterion the literature's equivariant gains live in, at 4 A and at the 10 A LIGYSIS recommends
        for name, col in (("net_sites_dcc4", "label_dcc"), ("net_sites_dcc10", "label_dcc10")):
            p_ = MT.per_structure(rr, "score", label_col=col)
            res[name] = {k: float(p_[k].mean()) for k in ("top1", "top3", "topN", "topN2", "ceiling")}
        res["net_center_error_median"] = float(rr.groupby("pdb")["dcc"].min().median())
    res["val_score"] = select_score(res, cfg)
    return res


# How much of the selection score each validation metric carries. Site top-1 is the task the project is judged on,
# so it must be in here: a score built only from the per-node classification heads is blind to whether the model
# can place a pocket at all. Measured on the first full GPU run (fold 0, seed 0): between epochs 20 and 24, as the
# staged schedule switched the centre and ranking terms on, site top-1 went 0.040 -> 0.868 and the median centre
# error 16.4 A -> 1.1 A, while occ_ap drifted 0.790 -> 0.790 and res_ap 0.711 -> 0.706. The old mean of those two
# therefore *fell* across the only improvement that mattered, so checkpoint selection kept epoch 14 -- whose
# metrics.json reports top1 0.022 and a ceiling of 0.118 -- and patience 10 stopped the run at 24 while top-1 was
# still climbing. Every ablation arm would have been scored at the same blind checkpoint.
SELECT_WEIGHTS = {"net_top1": 0.5, "occ_ap": 0.25, "res_ap": 0.25}


def select_score(res: dict, cfg: dict | None = None) -> float:
    """The scalar that checkpoint selection and early stopping use.

    An arm that predicts no sites at all (`no_probes`) has no site metric rather than a bad one, so its weight is
    dropped and the rest renormalised instead of scoring it zero -- the comparison between arms is made on the
    ablation table, not by starving one arm's checkpoint selection.
    """
    w = dict(SELECT_WEIGHTS)
    w.update(((cfg or {}).get("optim", {}) or {}).get("select_weights", {}) or {})
    vals = {"net_top1": res.get("net_sites", {}).get("top1"), "occ_ap": res.get("occ_ap"), "res_ap": res.get("res_ap")}
    use = {k: v for k, v in vals.items() if v is not None and not (isinstance(v, float) and math.isnan(v)) and w.get(k)}
    if not use:
        return float("nan")
    tot = sum(w[k] for k in use)
    return float(sum(w[k] * use[k] for k in use) / tot)


def _match_dim(in_dims, mc, kw, cfg, reference: str, log=print, tol: float = 0.02) -> int:
    """Widen a reduced arm until its parameter count is within `tol` of the reference arm's.

    Turning off the tensor or vector channels also removes their projections, their contribution to the edge input
    and their gates, so the arm is strictly smaller than `full` and the difference between them measures the degree
    *plus* the capacity. Bisecting on `dim` removes the capacity half of that, which is what makes the degree-2 and
    equivariance ablations able to conclude anything.
    """
    import yaml
    abl = yaml.safe_load((DEFAULTS.parent / "ablations.yaml").read_text())["ablations"]
    ref_over = {k.split(".", 1)[1]: v for k, v in abl.get(reference, {}).items() if k.startswith("model.")}
    ref_mc = {k: v for k, v in dict(cfg["model"], **ref_over).items() if k not in ("backbone", "lmax", "match_params")}
    count = lambda m: sum(p.numel() for p in M.EquiCaveNet(in_dims, **m, **kw).parameters())
    target = count(dict(ref_mc, n_edge_scalar=mc.get("n_edge_scalar", 0), n_init_vec=mc.get("n_init_vec", 3)))
    # dim must stay a multiple of the head count, so the search is over valid widths rather than every integer
    base = dict(mc)
    step = max(1, int(base.get("heads", 1)))
    widths = [d for d in range(step, base["dim"] * 32 + step, step)]
    # The step in dim is the head count, and one step moves the parameter count by several per cent, so an exact
    # match does not exist. We take the smallest valid width that reaches the reference count, never a smaller one:
    # the reduced arm then has at least as many parameters as `full`, so if it still loses, the degree it lacks is
    # doing the work and the result is conclusive in the direction that matters. The achieved excess is logged and
    # stored in the model card so the paper reports it rather than implying an exact match.
    best, got = None, None
    for d in widths:                                   # the count is monotone in dim
        c = count(dict(base, dim=d))
        if c >= target:
            best, got = d, c
            break
    if best is None:
        best, got = widths[-1], count(dict(base, dim=widths[-1]))
    log(f"match_params={reference}: dim {base['dim']} -> {best}, parameters {got} against {target} "
        f"({100 * (got - target) / target:+.1f} %)")
    if got < target:
        log(f"  WARNING: the arm is SMALLER than {reference}; a positive result for {reference} would be confounded "
            f"by capacity and cannot be published as a degree effect")
    elif got > (1 + tol) * target:
        log(f"  note: {100 * (got - target) / target:.1f} % more parameters than {reference} (the step in dim is "
            f"{step}, so no closer width exists). The arm is over-provisioned, which only strengthens a null result.")
    return best


def train_one(cfg, files_tr, files_va, device, out_dir: Path, log=print) -> dict:
    import torch
    seed_all(cfg["optim"]["seed"])
    d0 = D.load(files_tr[0])
    in_dims = dict(res=d0["feat_res"].shape[1], probe=d0["feat_probe"].shape[1], surf=d0["feat_surf"].shape[1])
    mc = dict(cfg["model"])
    backbone = mc.pop("backbone", "cartesian")
    match_to = mc.pop("match_params", None)
    mc["n_edge_scalar"] = d0["edge_scalar"].shape[1] if "edge_scalar" in d0 else 0
    mc["n_init_vec"] = int(d0["vec0"].shape[1])
    mc.setdefault("invariant_mode", cfg["model"].get("invariant_mode", "frames"))
    kw = dict(n_hot=d0["y_hot"].shape[1], n_props=d0["y_prop"].shape[1])
    if backbone == "e3nn":
        from training.pockets.model_e3nn import EquiCaveNetE3
        model = EquiCaveNetE3(in_dims, **{k: v for k, v in mc.items() if k in
                                          ("dim", "layers", "lmax", "n_rbf", "cutoff", "dropout", "n_init_vec")}, **kw).to(device)
    else:
        if match_to:
            mc["dim"] = _match_dim(in_dims, mc, kw, cfg, match_to, log)
        model = M.EquiCaveNet(in_dims, **mc, **kw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
    E, acc = cfg["optim"]["epochs"], cfg["optim"]["accumulate"]
    steps = E * math.ceil(len(files_tr) / acc); warm = cfg["optim"]["warmup_epochs"] * math.ceil(len(files_tr) / acc)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps)))))
    ema = EMA(model, cfg["optim"]["ema"]); rng = np.random.default_rng(cfg["optim"]["seed"])
    use_amp = cfg["optim"].get("amp", "none") == "bf16" and device.type == "cuda"
    best, best_state, bad, history = -1.0, None, 0, []
    # A cache built before the direction target existed has no `probe_dir`, and `losses` skips the term when it is
    # absent -- so without this check a GPU run would train for hours with its headline new mechanism silently off.
    if cfg["loss"].get("w_dir", 0.0) > 0 and "probe_dir" not in d0:
        raise RuntimeError(
            f"loss.w_dir is {cfg['loss']['w_dir']} but the feature cache at {cfg['data']['cache_dir']} has no "
            "'probe_dir': it predates the direction target. Delete the cache directory and rerun so build_cache "
            "regenerates it, or set loss.w_dir=0.0 (ablation arm no_direction_loss) to train without it.")
    st = cfg.get("stages", {}) or {}
    stage1 = int(st.get("warmup_epochs", 0)) if st.get("enabled") else 0
    if stage1:
        log(f"staged training: epochs 1-{stage1} with {','.join(st.get('stage1_zero', []))} at zero, then all terms")

    # Resume. A multi-hour run that dies -- a restarted container, a pre-empted node, a full disk -- otherwise loses
    # every epoch, and this trainer is meant to be driven unattended. The checkpoint is written after each epoch and
    # holds everything the loop needs to continue identically: weights, the EMA shadow, the optimiser and schedule
    # state, the RNG stream that decides the structure order and the masks, and the early-stopping counters.
    ck_path = out_dir / "checkpoint.pt"
    start_ep = 0
    if ck_path.exists():
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); ema.shadow.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        rng.bit_generator.state = ck["rng"]
        best, bad, history, start_ep = ck["best"], ck["bad"], ck["history"], ck["epoch"]
        best_state = ck.get("best_state")
        log(f"resumed from {ck_path.name} at epoch {start_ep + 1}/{E}, best val score {best:.4f}")

    for ep in range(start_ep, E):
        # P6: stage 1 converges the dense per-node heads before the sparse ranking terms are switched on, so the
        # sparse gradients do not fight a dense signal through an untrained trunk. A schedule, not a parameter.
        lw = dict(cfg["loss"])
        if ep < stage1:
            for k in st.get("stage1_zero", []):
                lw[k] = 0.0
        elif stage1 and ep == stage1:
            # The objective changes here, so a score from before it is not a score to beat. Stage 1 trains with the
            # centre, confidence and listwise terms at zero, which is why the first run to reach this point stopped
            # on patience measured against a stage-1 best while site top-1 was still climbing steeply. Selection
            # restarts from this epoch, and a stage-1 checkpoint is never the answer: it was chosen without the
            # terms the model exists to optimise.
            best, bad = -float("inf"), 0
            log(f"stage 2 begins at epoch {ep + 1}: early stopping restarts, {cfg['optim']['patience']} epochs of "
                f"patience from here")
        model.train(); t0 = time.time(); order = rng.permutation(len(files_tr)); agg = {}; run = 0.0
        # An epoch is one pass over every cached structure — hours at full scale, so it reports inside the epoch
        # rather than only in the one line at the end of it.
        with progress.Bar(f"epoch {ep + 1}/{E}", len(order), unit="struct") as bar:
            for i, j in enumerate(order):
                b = D.to_torch(D.load(files_tr[j]), device)
                if cfg["optim"]["rotate"]:
                    b = D.random_rotation(b, rng)
                aug = cfg["optim"].get("augment", {}) or {}
                b = D.jitter(b, rng, pos=aug.get("pos", 0.0), feat=aug.get("feat", 0.0), drop=aug.get("drop", 0.0))
                b, res_mask = D.mask_residues(b, cfg["loss"].get("mask_frac", 0.15), rng)
                b, pot_mask, pot_target = D.mask_probe_potential(b, cfg["loss"].get("pot_mask_frac", 0.2), rng)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    out = model(b); total, parts = losses(out, b, lw, res_mask, pot_mask, pot_target)
                (total / acc).backward()
                for k, v in parts.items():
                    agg[k] = agg.get(k, 0) + v / len(order)
                if (i + 1) % acc == 0 or i == len(order) - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(); sched.step(); ema.update(model)
                run += float(sum(parts.values()))    # parts hold floats already, so no device sync here
                bar.update(1, postfix=f"loss {run / (i + 1):.3f}")
        ev = evaluate(ema.shadow, files_va, device, cfg) if files_va else dict(val_score=float("nan"))
        history.append(dict(epoch=ep + 1, train=agg, val=ev, seconds=round(time.time() - t0, 1)))
        log(f"epoch {ep + 1}/{E} loss {sum(agg.values()):.3f} {json.dumps({k: round(v, 3) for k, v in agg.items()})} | val occ_ap {ev.get('occ_ap', float('nan')):.3f} res_ap {ev.get('res_ap', float('nan')):.3f} "
            f"net top1 {ev.get('net_sites', {}).get('top1', float('nan')):.3f} | {time.time() - t0:.0f}s")
        if ev["val_score"] > best or math.isnan(ev["val_score"]):
            best, bad = ev["val_score"], 0; best_state = copy.deepcopy(ema.shadow.state_dict())
            torch.save(dict(state=best_state, in_dims=in_dims, cfg=cfg, n_hot=d0["y_hot"].shape[1], n_props=d0["y_prop"].shape[1],
                            n_edge_scalar=mc.get("n_edge_scalar", 0), n_init_vec=mc.get("n_init_vec", 3)), out_dir / "model.pt")
        else:
            bad += 1
        torch.save(dict(model=model.state_dict(), ema=ema.shadow.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), rng=rng.bit_generator.state, best=best, bad=bad,
                        history=history, epoch=ep + 1, best_state=best_state), ck_path)
        if bad >= cfg["optim"]["patience"]:
            log("early stop"); break
    if best_state is not None:
        ema.shadow.load_state_dict(best_state)
    else:
        log("no epoch improved on the initial score; returning the EMA weights as they stand")
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    return dict(model=ema.shadow, best=best, history=history, in_dims=in_dims)


def load_model(path, device):
    import torch
    ck = torch.load(path, map_location=device, weights_only=False)
    mc = dict(ck["cfg"]["model"]); backbone = mc.pop("backbone", "cartesian")
    mc.setdefault("n_edge_scalar", ck.get("n_edge_scalar", 0))
    mc.setdefault("n_init_vec", ck.get("n_init_vec", 3))
    if backbone == "e3nn":
        from training.pockets.model_e3nn import EquiCaveNetE3
        m = EquiCaveNetE3(ck["in_dims"], n_hot=ck["n_hot"], n_props=ck["n_props"],
                          **{k: v for k, v in mc.items() if k in ("dim", "layers", "lmax", "n_rbf", "cutoff", "dropout", "n_init_vec")}).to(device)
        m.load_state_dict(ck["state"]); m.eval()
        return m, ck["cfg"]
    m = M.EquiCaveNet(ck["in_dims"], n_hot=ck["n_hot"], n_props=ck["n_props"], **mc).to(device)
    m.load_state_dict(ck["state"]); m.eval()
    return m, ck["cfg"]


def run(a) -> int:
    cfg = _cfg(a); device = pick_device(a.device)
    dc = cfg["data"]; cache = Path(dc["cache_dir"]); mode = cfg.get("mode", "train")
    tag = cfg.get("tag", cfg["ablation"])
    out_dir = run_dir(a.out, f"{tag}_fold{cfg['split']['val_fold']}_seed{cfg['optim']['seed']}")
    log = lambda s: (print(s, flush=True), open(out_dir / "log.txt", "a").write(s + "\n"))
    log(f"config: {json.dumps(cfg)}")
    ids = D.build_cache(Path(dc["manifest"]), Path(dc["pdb_dir"]), cache, dc.get("esm"), dc.get("limit", 0), dc["n_probe"], dc["n_surf"], str(device), log, dc.get("k_scale", 1.0), dc.get("druglike_only", False), dc.get("require_interaction", True),
                       dc.get("residue_chemistry", True), dc.get("probe_potential", True),
                       dc.get("probe_sampling", "tiered"), dc.get("point_model_tag", ""),
                       dc.get("probe_ligandable_frac", 0.5))
    files = [cache / f"{i}.npz" for i in ids]
    folds = {f: int(D.load(f)["fold"]) for f in files}
    if mode == "oof":                                   # one model per fold, features for the ranker on the held-out fold
        from equicave import pocket_features as pf
        rows = []
        # find_table, not a hardcoded .csv: every candidate table in this project is gzipped, and reading the wrong
        # name here failed *after* the five folds had trained, which is the most expensive place to discover it.
        from equicave import tables
        cand = tables.read_table(Path("data/processed"), f"candidates_{cfg.get('cand_tag', 'native')}")
        for k in sorted(set(folds.values())):
            tr = [f for f in files if folds[f] != k]; va = [f for f in files if folds[f] == k]
            fd = run_dir(out_dir, f"fold{k}"); r = train_one(cfg, tr, va, device, fd, log)
            import torch
            with torch.no_grad():
                for f in va:
                    d = D.load(f); b = D.to_torch(d, device); out = r["model"](b)
                    sub = cand[cand.pdb == d["pdb"]]
                    centers = np.array([[float(x) for x in c.split(";")] for c in sub["center"]])
                    for (idx, _), nf in zip(sub.iterrows(), net_features(out, d["pos"][d["n_res"]:d["n_res"] + d["n_probe"]], centers)):
                        rows.append(dict(pdb=d["pdb"], center=sub.loc[idx, "center"], **nf))
        pd.DataFrame(rows).to_csv(Path("data/processed") / f"net_features_{tag}.csv", index=False)
        log(f"wrote net features for {len(rows)} candidates")
        return 0
    k = cfg["split"]["val_fold"]
    tr = [f for f in files if folds[f] != k]; va = [f for f in files if folds[f] == k]
    log(f"{len(tr)} train / {len(va)} val structures (val fold {k}), device {device}")
    r = train_one(cfg, tr, va, device, out_dir, log)
    ev = evaluate(r["model"], va, device, cfg) if va else {}
    (out_dir / "metrics.json").write_text(json.dumps(ev, indent=1))
    from equicave import pocket_features as pf
    write_model_card(out_dir, dict(name="EquiCave-Net", task="pockets-net", ablation=cfg["ablation"], config=cfg, geometry=pf.geometry(),
                                   n_train=len(tr), n_val=len(va), val_fold=k, metrics=ev, device=str(device),
                                   scope="non-commercial scope if trained on non-commercial data; RCSB-only training is CC0",
                                   note="numbers are validation-fold estimates; publish only converged runs with >= 3 seeds"))
    log(f"done: best val score {r['best']:.3f}; metrics in {out_dir / 'metrics.json'}")
    return 0
