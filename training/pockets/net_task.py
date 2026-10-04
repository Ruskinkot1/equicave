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


def losses(out, b, w, res_mask=None):
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
        if "prop_logit" in o:
            part["prop"] = torch.nn.functional.binary_cross_entropy_with_logits(o["prop_logit"], b["y_prop"])
        if last and res_mask is not None and "seq_logit" in o and "res_type" in b:
            part["seq"] = M.masked_residue_loss(o["seq_logit"], b["res_type"], res_mask)
        if last and "size_pred" in o and "site_n_atoms" in b:
            part["size"] = M.size_loss(o["size_pred"], b["site_n_atoms"])
        total = total + scale * sum(w.get(f"w_{k}", 0.0) * v for k, v in part.items())
        if last:
            L = part
        else:
            L.update({f"{k}@{i}": v for k, v in part.items()})
    return total, {k: float(v.detach()) if hasattr(v, "detach") else float(v) for k, v in L.items()}


def predict_sites(out, probe_pos, nms: float = 6.0, max_sites: int = 30):
    """Network-only site list: predicted centres ranked by confidence with NMS (for evaluation of the network alone)."""
    import torch
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
    res["val_score"] = float(np.nanmean([res["occ_ap"], res["res_ap"]]))
    return res


def train_one(cfg, files_tr, files_va, device, out_dir: Path, log=print) -> dict:
    import torch
    seed_all(cfg["optim"]["seed"])
    d0 = D.load(files_tr[0])
    in_dims = dict(res=d0["feat_res"].shape[1], probe=d0["feat_probe"].shape[1], surf=d0["feat_surf"].shape[1])
    mc = dict(cfg["model"])
    backbone = mc.pop("backbone", "cartesian")
    mc["n_edge_scalar"] = d0["edge_scalar"].shape[1] if "edge_scalar" in d0 else 0
    mc["n_init_vec"] = int(d0["vec0"].shape[1])
    mc.setdefault("invariant_mode", cfg["model"].get("invariant_mode", "frames"))
    kw = dict(n_hot=d0["y_hot"].shape[1], n_props=d0["y_prop"].shape[1])
    if backbone == "e3nn":
        from training.pockets.model_e3nn import EquiCaveNetE3
        model = EquiCaveNetE3(in_dims, **{k: v for k, v in mc.items() if k in
                                          ("dim", "layers", "lmax", "n_rbf", "cutoff", "dropout", "n_init_vec")}, **kw).to(device)
    else:
        model = M.EquiCaveNet(in_dims, **mc, **kw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
    E, acc = cfg["optim"]["epochs"], cfg["optim"]["accumulate"]
    steps = E * math.ceil(len(files_tr) / acc); warm = cfg["optim"]["warmup_epochs"] * math.ceil(len(files_tr) / acc)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps)))))
    ema = EMA(model, cfg["optim"]["ema"]); rng = np.random.default_rng(cfg["optim"]["seed"])
    use_amp = cfg["optim"].get("amp", "none") == "bf16" and device.type == "cuda"
    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(E):
        model.train(); t0 = time.time(); order = rng.permutation(len(files_tr)); agg = {}
        for i, j in enumerate(order):
            b = D.to_torch(D.load(files_tr[j]), device)
            if cfg["optim"]["rotate"]:
                b = D.random_rotation(b, rng)
            b, res_mask = D.mask_residues(b, cfg["loss"].get("mask_frac", 0.15), rng)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(b); total, parts = losses(out, b, cfg["loss"], res_mask)
            (total / acc).backward()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0) + v / len(order)
            if (i + 1) % acc == 0 or i == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(); sched.step(); ema.update(model)
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
            if bad >= cfg["optim"]["patience"]:
                log("early stop"); break
    ema.shadow.load_state_dict(best_state)
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
                       dc.get("residue_chemistry", True))
    files = [cache / f"{i}.npz" for i in ids]
    folds = {f: int(D.load(f)["fold"]) for f in files}
    if mode == "oof":                                   # one model per fold, features for the ranker on the held-out fold
        from equicave import pocket_features as pf
        rows = []
        cand = pd.read_csv(Path("data/processed") / f"candidates_{cfg.get('cand_tag', 'native')}.csv")
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
