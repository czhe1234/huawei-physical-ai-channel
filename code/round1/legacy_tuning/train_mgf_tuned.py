# （本改动由Claude fable5）
"""Tuning copy of train_marginal_gain_field.py.

Adds: --loss-weights (beam,delay,T), train-set shape-cosine diagnostics
(overfit gap), grad clipping. Everything else identical to the original.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import PATHS
from powerfield_common import (ASSETS, TINY, assemble_matrix, build_collapsed_caches,
                               build_features, regime_of)
from power_calibration_lab import metrics_batch, summarize


class MarginalField(nn.Module):
    def __init__(self, in_dim, width=512, depth=3, dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.LayerNorm(width), nn.SiLU(),
                       nn.Dropout(dropout)]
            d = width
        self.trunk = nn.Sequential(*layers)
        self.head_beam = nn.Linear(d, 128)
        self.head_delay = nn.Linear(d, 192)
        self.head_total = nn.Linear(d, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.head_beam(h), self.head_delay(h), self.head_total(h)


def cos_loss(logits, target_shape):
    p = torch.softmax(logits, dim=1)
    num = (p * target_shape).sum(1)
    den = p.norm(dim=1) * target_shape.norm(dim=1) + 1e-20
    return (1.0 - num / den).mean()


def shape_cos(pred_shape, true_shape):
    num = (pred_shape * true_shape).sum(1)
    den = np.linalg.norm(pred_shape, axis=1) * np.linalg.norm(true_shape, axis=1) + 1e-20
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--loss-weights", default="0.45,0.45,0.10",
                    help="beam,delay,logT loss weights")
    ap.add_argument("--feature-keys", default="ff,blocked,geo,agree,base_beam,base_delay,base_total,knn_beam,knn_delay,knn_total,nn1_beam,nn1_delay,nn1_total,sharp_beam,sharp_delay")
    ap.add_argument("--etas", default="0.4,0.5,0.6,0.75")
    ap.add_argument("--gain-clip", type=float, default=8.0)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--train-all", action="store_true")
    ap.add_argument("--save-model", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--feats-cache", default=ASSETS + "/powerfield_train_feats_v2.npz")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    coll, bcoll = build_collapsed_caches()
    pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy")
    blocked = np.load(ASSETS + "/height_geometry_cache.npz")["train_blocked"].astype(np.float64)
    val = np.load(ASSETS + "/test_regime_analysis/test_matched_val_sets.npz")
    eval_idx = val["spatial_matched"][: args.n_eval]
    excl_train = set(val["spatial_matched"].tolist())

    B_t = coll.sum(-1).reshape(2000, 128)
    D_t = coll.sum((1, 2))
    T_t = coll.sum((1, 2, 3))
    nz = T_t > 0
    b_shape = B_t / np.maximum(T_t[:, None], TINY)
    d_shape = D_t / np.maximum(T_t[:, None], TINY)

    B_b = bcoll.sum(-1).reshape(2000, 128)
    D_b = bcoll.sum((1, 2))
    T_b = bcoll.sum((1, 2, 3))

    if os.path.exists(args.feats_cache):
        zz = np.load(args.feats_cache)
        feats = {k: zz[k] for k in zz.files}
    else:
        feats = build_features(pos, blocked, coll, pos, blocked,
                               B_b.reshape(2000, 16, 8), D_b, T_b,
                               exclude_self=True, k=args.k)
        np.savez(args.feats_cache, **feats)
    keys = args.feature_keys.split(",")
    X = assemble_matrix(feats, keys)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd

    if args.train_all:
        train_ids = np.where(nz)[0]
    else:
        train_ids = np.array([i for i in range(2000) if i not in excl_train and nz[i]])
    print(f"train n={len(train_ids)}, eval n={len(eval_idx)}, feat dim={X.shape[1]}")

    Xtr = torch.tensor(Xn[train_ids], device=dev)
    Btr = torch.tensor(b_shape[train_ids], dtype=torch.float32, device=dev)
    Dtr = torch.tensor(d_shape[train_ids], dtype=torch.float32, device=dev)
    Ttr = torch.tensor(np.log(T_t[train_ids] + TINY), dtype=torch.float32, device=dev)
    Tbr = torch.tensor(np.log(T_b[train_ids] + TINY), dtype=torch.float32, device=dev)

    Xev = torch.tensor(Xn[eval_idx], device=dev)

    wb, wd, wt = [float(v) for v in args.loss_weights.split(",")]

    pred_beam_acc = np.zeros((len(eval_idx), 128), dtype=np.float64)
    pred_delay_acc = np.zeros((len(eval_idx), 192), dtype=np.float64)
    pred_logT_acc = np.zeros((len(eval_idx),), dtype=np.float64)
    tr_beam_acc = np.zeros((len(train_ids), 128), dtype=np.float64)
    tr_delay_acc = np.zeros((len(train_ids), 192), dtype=np.float64)

    models = []
    diverged = False
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        model = MarginalField(X.shape[1], args.width, args.depth, args.dropout).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        for ep in range(args.epochs):
            model.train()
            opt.zero_grad()
            lb, ld, lt = model(Xtr)
            loss_b = cos_loss(lb, Btr)
            loss_d = cos_loss(ld, Dtr)
            loss_t = nn.functional.huber_loss(lt.squeeze(1) + Tbr, Ttr, delta=1.0)
            loss = wb * loss_b + wd * loss_d + wt * loss_t
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            if ep % 500 == 0 or ep == args.epochs - 1:
                print(f"seed{seed} ep{ep} loss={loss.item():.4f} b={loss_b.item():.4f} d={loss_d.item():.4f} t={loss_t.item():.4f}", flush=True)
            if not np.isfinite(loss.item()):
                print(f"seed{seed} ep{ep} DIVERGED")
                diverged = True
                break
        model.eval()
        with torch.no_grad():
            lb, ld, lt = model(Xev)
            pred_beam_acc += torch.softmax(lb, 1).cpu().numpy()
            pred_delay_acc += torch.softmax(ld, 1).cpu().numpy()
            pred_logT_acc += (lt.squeeze(1).cpu().numpy()
                              + np.log(T_b[eval_idx] + TINY))
            lb2, ld2, _ = model(Xtr)
            tr_beam_acc += torch.softmax(lb2, 1).cpu().numpy()
            tr_delay_acc += torch.softmax(ld2, 1).cpu().numpy()
        models.append(model)

    pred_beam = (pred_beam_acc / args.seeds).astype(np.float32)
    pred_delay = (pred_delay_acc / args.seeds).astype(np.float32)
    pred_logT = (pred_logT_acc / args.seeds).astype(np.float32)
    tr_beam = (tr_beam_acc / args.seeds).astype(np.float32)
    tr_delay = (tr_delay_acc / args.seeds).astype(np.float32)

    bs_true = b_shape[eval_idx]
    ds_true = d_shape[eval_idx]
    bs_base = B_b[eval_idx] / np.maximum(T_b[eval_idx, None], TINY)
    ds_base = D_b[eval_idx] / np.maximum(T_b[eval_idx, None], TINY)
    kb = np.exp(feats["knn_beam"][eval_idx])
    kd = np.exp(feats["knn_delay"][eval_idx])
    sb = np.exp(feats["sharp_beam"][eval_idx])
    sdl = np.exp(feats["sharp_delay"][eval_idx])
    diag = {
        "diverged": diverged,
        "beam_cos_model": float(np.nanmean(shape_cos(pred_beam, bs_true))),
        "beam_cos_base": float(np.nanmean(shape_cos(bs_base, bs_true))),
        "beam_cos_knn": float(np.nanmean(shape_cos(kb, bs_true))),
        "beam_cos_sharp": float(np.nanmean(shape_cos(sb, bs_true))),
        "delay_cos_model": float(np.nanmean(shape_cos(pred_delay, ds_true))),
        "delay_cos_base": float(np.nanmean(shape_cos(ds_base, ds_true))),
        "delay_cos_knn": float(np.nanmean(shape_cos(kd, ds_true))),
        "delay_cos_sharp": float(np.nanmean(shape_cos(sdl, ds_true))),
        "train_beam_cos_model": float(np.nanmean(shape_cos(tr_beam, b_shape[train_ids]))),
        "train_delay_cos_model": float(np.nanmean(shape_cos(tr_delay, d_shape[train_ids]))),
        "logT_mae_model": float(np.nanmean(np.abs(pred_logT - np.log(T_t[eval_idx] + TINY)))),
        "logT_mae_base": float(np.nanmean(np.abs(np.log(T_b[eval_idx] + TINY) - np.log(T_t[eval_idx] + TINY)))),
        "model_shape_cos_to_sharp_beam": float(np.nanmean(shape_cos(pred_beam, sb))),
        "model_shape_cos_to_sharp_delay": float(np.nanmean(shape_cos(pred_delay, sdl))),
    }
    print("shape diagnostics:", json.dumps(diag, indent=1), flush=True)

    base_mm = np.load(ASSETS + "/train_v36_loo_base_derot_ad.npy", mmap_mode="r")
    res_mm = np.load(ASSETS + "/train_v36_loo_residual_derot_ad.npy", mmap_mode="r")
    regimes = regime_of(blocked)[eval_idx]
    out = {"config": vars(args), "diagnostics": diag}

    etas = [float(e) for e in args.etas.split(",")]
    variant_names = ["base"] + [f"field_eta{e:g}" for e in etas]
    acc = {v: {kk: [] for kk in ["pas", "pdp", "e2", "t2", "p2", "dot"]}
           for v in variant_names}
    batch = 50
    for bs in range(0, len(eval_idx), batch):
        be = min(bs + batch, len(eval_idx))
        ids = eval_idx[bs:be]
        base_np = np.asarray(base_mm[ids])
        truth_np = base_np + np.asarray(res_mm[ids])
        base_t = torch.from_numpy(base_np).to(dev)
        truth_t = torch.from_numpy(truth_np).to(dev)
        m = metrics_batch(base_t, truth_t)
        for kk in m:
            acc["base"][kk].append(m[kk])

        Pb = (base_t.real**2 + base_t.imag**2).sum(dim=(1, 4))
        Tb_pt = Pb.sum(dim=(1, 2, 3), keepdim=True)
        bb = Pb.sum(-1, keepdim=True) / (Tb_pt + TINY)
        db = Pb.sum(dim=(1, 2), keepdim=True) / (Tb_pt + TINY)
        bp = torch.from_numpy(pred_beam[bs:be]).to(dev).reshape(-1, 16, 8, 1)
        dp = torch.from_numpy(pred_delay[bs:be]).to(dev).reshape(-1, 1, 1, 192)
        delta_b = 1e-4 / 128.0
        delta_d = 1e-4 / 192.0
        g2 = ((bp + delta_b) / (bb + delta_b)) * ((dp + delta_d) / (db + delta_d))
        for e in etas:
            g = torch.sqrt(g2) ** e
            g = g.clamp(1.0 / args.gain_clip, args.gain_clip)
            pred = base_t * g[:, None, :, :, None, :]
            m = metrics_batch(pred, truth_t)
            for kk in m:
                acc[f"field_eta{e:g}"][kk].append(m[kk])
        print(f"eval batch {bs}-{be}", flush=True)

    for v in variant_names:
        summarize(v, acc[v], regimes, out)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", args.out)

    if args.save_model:
        torch.save({
            "state_dicts": [m.state_dict() for m in models],
            "mu": mu, "sd": sd, "keys": keys,
            "in_dim": X.shape[1],
            "width": args.width, "depth": args.depth, "dropout": args.dropout,
        }, args.save_model)
        print("saved model", args.save_model)


if __name__ == "__main__":
    main()
