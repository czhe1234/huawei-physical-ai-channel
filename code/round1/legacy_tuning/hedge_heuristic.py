# （本改动由Claude fable5）
"""No-NN hedged shape predictor.

Candidates per point (all LOO exclude-self, derotated AD collapsed power):
  sharp4  : sharp_marginals k=4 p=3 (IDW of normalized shapes, sharp weights)
  sharp8  : sharp_marginals k=8 p=3
  knn32   : IDW k=32 (p=2, floor 0.25) of NORMALIZED shapes
  base    : LOO base collapsed power marginals, normalized
  uniform : flat shape

Hedge: shape = normalize(w*sharp + (1-w)*knn32), w picked per agreement decile
(agree_beam / agree_delay from powerfield_train_feats_v2.npz), fitted on TRAIN
points only (excluding all 500 spatial_matched + zero-power points), separately
for beam and delay.  hedge3 additionally allows a base-shape component.

Full metric: rank-1 gains on LOO base AD (delta_b=1e-4/128, delta_d=1e-4/192,
clip 8.0), eta grid.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import PATHS
from powerfield_common import (ASSETS, TINY, build_collapsed_caches,
                               geom_effective_dist, regime_of)
from power_calibration_lab import metrics_batch, summarize

TUNING_RESULTS = PATHS.analysis / "results" / "tuning"


def shape_cos(pred, true):
    num = (pred * true).sum(1)
    den = np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1) + 1e-20
    return num / den


def compute_candidates(P_ids, pos, blocked, Bn, Dn, T_train):
    """For each point id (a train index, LOO exclude-self), compute candidate
    beam/delay shapes: sharp4, sharp8, knn32."""
    n = len(P_ids)
    out = {name: (np.empty((n, 128), np.float32), np.empty((n, 192), np.float32))
           for name in ["sharp4", "sharp8", "knn32"]}
    for j, q in enumerate(P_ids):
        eff = geom_effective_dist(pos, blocked, pos[q, :2], blocked[q]).copy()
        eff[T_train <= 0] = np.inf
        eff[q] = np.inf
        for name, k, p, floor in [("sharp4", 4, 3, 0.3), ("sharp8", 8, 3, 0.3),
                                  ("knn32", 32, 2, 0.25)]:
            idx = np.argpartition(eff, k)[:k]
            idx = idx[np.argsort(eff[idx])]
            w = 1.0 / np.maximum(eff[idx], floor) ** p
            w = w / w.sum()
            out[name][0][j] = (w[:, None] * Bn[idx]).sum(0)
            out[name][1][j] = (w[:, None] * Dn[idx]).sum(0)
    return out


def fit_decile_w(agree_tr, cand_tr, truth_tr, w_grid, edges):
    """Per agreement decile, pick w maximizing mean cos(w*sharp+(1-w)*knn, truth)."""
    sharp, knn = cand_tr
    bins = np.clip(np.searchsorted(edges, agree_tr, side="right"), 0, len(edges))
    w_of_bin = np.zeros(len(edges) + 1)
    table = []
    for b in range(len(edges) + 1):
        m = bins == b
        best_w, best_c = 1.0, -np.inf
        row = {}
        for w in w_grid:
            c = float(np.mean(shape_cos(w * sharp[m] + (1 - w) * knn[m], truth_tr[m])))
            row[w] = c
            if c > best_c:
                best_c, best_w = c, w
        w_of_bin[b] = best_w
        table.append({"bin": b, "n": int(m.sum()), "best_w": best_w,
                      "best_cos": best_c, "cos_by_w": row})
    return w_of_bin, table


def fit_decile_w3(agree_tr, sharp, knn, base, truth_tr, edges):
    """3-component simplex grid per decile: ws in 0..1 step .25, wb in {0,.25,.5}."""
    bins = np.clip(np.searchsorted(edges, agree_tr, side="right"), 0, len(edges))
    combos = [(ws, wb) for ws in (0, 0.25, 0.5, 0.75, 1.0)
              for wb in (0, 0.25, 0.5) if ws + wb <= 1.0]
    w_of_bin = np.zeros((len(edges) + 1, 2))
    table = []
    for b in range(len(edges) + 1):
        m = bins == b
        best, best_c = (1.0, 0.0), -np.inf
        for ws, wb in combos:
            wk = 1.0 - ws - wb
            c = float(np.mean(shape_cos(
                ws * sharp[m] + wk * knn[m] + wb * base[m], truth_tr[m])))
            if c > best_c:
                best_c, best = c, (ws, wb)
        w_of_bin[b] = best
        table.append({"bin": b, "n": int(m.sum()), "best_ws_wb": best,
                      "best_cos": best_c})
    return w_of_bin, table


def apply_hedge(agree, sharp, knn, w_of_bin, edges):
    bins = np.clip(np.searchsorted(edges, agree, side="right"), 0, len(edges))
    w = w_of_bin[bins][:, None]
    s = w * sharp + (1 - w) * knn
    return s / np.maximum(s.sum(1, keepdims=True), TINY)


def apply_hedge3(agree, sharp, knn, base, w_of_bin, edges):
    bins = np.clip(np.searchsorted(edges, agree, side="right"), 0, len(edges))
    ws = w_of_bin[bins, 0][:, None]
    wb = w_of_bin[bins, 1][:, None]
    s = ws * sharp + (1 - ws - wb) * knn + wb * base
    return s / np.maximum(s.sum(1, keepdims=True), TINY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--etas", default="0.4,0.5,0.6,0.75,1.0")
    ap.add_argument("--gain-variants",
                    default="sharp8,knn32,hedge_sharp4,hedge_sharp8,hedge3_sharp8")
    ap.add_argument("--gain-clip", type=float, default=8.0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default=str(TUNING_RESULTS / "hedge_heuristic_results.json"))
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    coll, bcoll = build_collapsed_caches()
    pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy")
    blocked = np.load(ASSETS + "/height_geometry_cache.npz")["train_blocked"].astype(np.float64)
    val = np.load(ASSETS + "/test_regime_analysis/test_matched_val_sets.npz")
    eval_idx = val["spatial_matched"][: args.n_eval]
    excl = set(val["spatial_matched"].tolist())

    T_t = coll.sum((1, 2, 3))
    Bn = (coll.sum(-1).reshape(2000, 128) / np.maximum(T_t[:, None], TINY)).astype(np.float32)
    Dn = (coll.sum((1, 2)) / np.maximum(T_t[:, None], TINY)).astype(np.float32)
    T_b = bcoll.sum((1, 2, 3))
    Bb = (bcoll.sum(-1).reshape(2000, 128) / np.maximum(T_b[:, None], TINY)).astype(np.float32)
    Db = (bcoll.sum((1, 2)) / np.maximum(T_b[:, None], TINY)).astype(np.float32)

    train_ids = np.array([i for i in range(2000) if i not in excl and T_t[i] > 0])
    print(f"train n={len(train_ids)} eval n={len(eval_idx)}")

    feats = np.load(ASSETS + "/powerfield_train_feats_v2.npz")
    agree_beam = feats["agree"][:, 0]
    agree_delay = feats["agree"][:, 1]

    print("computing candidate shapes (train)...")
    cand_tr = compute_candidates(train_ids, pos, blocked, Bn, Dn, T_t)
    print("computing candidate shapes (eval)...")
    cand_ev = compute_candidates(eval_idx, pos, blocked, Bn, Dn, T_t)

    # decile edges from train agreement
    edges_b = np.quantile(agree_beam[train_ids], np.linspace(0.1, 0.9, 9))
    edges_d = np.quantile(agree_delay[train_ids], np.linspace(0.1, 0.9, 9))
    w_grid = [0.0, 0.25, 0.5, 0.75, 1.0]

    out = {"config": vars(args)}
    fit_tables = {}
    hedges_ev = {}
    for sharp_name in ["sharp4", "sharp8"]:
        wb_beam, tb = fit_decile_w(
            agree_beam[train_ids],
            (cand_tr[sharp_name][0], cand_tr["knn32"][0]), Bn[train_ids],
            w_grid, edges_b)
        wb_delay, td = fit_decile_w(
            agree_delay[train_ids],
            (cand_tr[sharp_name][1], cand_tr["knn32"][1]), Dn[train_ids],
            w_grid, edges_d)
        fit_tables[f"hedge_{sharp_name}"] = {"beam": tb, "delay": td,
                                             "w_beam": wb_beam.tolist(),
                                             "w_delay": wb_delay.tolist()}
        hb = apply_hedge(agree_beam[eval_idx], cand_ev[sharp_name][0],
                         cand_ev["knn32"][0], wb_beam, edges_b)
        hd = apply_hedge(agree_delay[eval_idx], cand_ev[sharp_name][1],
                         cand_ev["knn32"][1], wb_delay, edges_d)
        hedges_ev[f"hedge_{sharp_name}"] = (hb, hd)
        print(f"hedge_{sharp_name}: w_beam per decile={wb_beam.tolist()}")
        print(f"hedge_{sharp_name}: w_delay per decile={wb_delay.tolist()}")

    # 3-component hedge with base shape (sharp8)
    wb3_beam, t3b = fit_decile_w3(agree_beam[train_ids], cand_tr["sharp8"][0],
                                  cand_tr["knn32"][0], Bb[train_ids],
                                  Bn[train_ids], edges_b)
    wb3_delay, t3d = fit_decile_w3(agree_delay[train_ids], cand_tr["sharp8"][1],
                                   cand_tr["knn32"][1], Db[train_ids],
                                   Dn[train_ids], edges_d)
    fit_tables["hedge3_sharp8"] = {"beam": t3b, "delay": t3d,
                                   "w_beam": wb3_beam.tolist(),
                                   "w_delay": wb3_delay.tolist()}
    hedges_ev["hedge3_sharp8"] = (
        apply_hedge3(agree_beam[eval_idx], cand_ev["sharp8"][0],
                     cand_ev["knn32"][0], Bb[eval_idx], wb3_beam, edges_b),
        apply_hedge3(agree_delay[eval_idx], cand_ev["sharp8"][1],
                     cand_ev["knn32"][1], Db[eval_idx], wb3_delay, edges_d))
    print(f"hedge3_sharp8: (ws,wb) beam per decile={wb3_beam.tolist()}")
    print(f"hedge3_sharp8: (ws,wb) delay per decile={wb3_delay.tolist()}")
    out["fit_tables"] = fit_tables

    # --- shape variants for eval ---
    ones_b = np.full((len(eval_idx), 128), 1.0 / 128, np.float32)
    ones_d = np.full((len(eval_idx), 192), 1.0 / 192, np.float32)
    shape_variants = {
        "sharp4": cand_ev["sharp4"],
        "sharp8": cand_ev["sharp8"],
        "knn32": cand_ev["knn32"],
        "baseshape": (Bb[eval_idx], Db[eval_idx]),
        "uniform": (ones_b, ones_d),
        **hedges_ev,
    }
    # mixed marginal combos: hedge one marginal, keep sharp8 for the other
    shape_variants["mix_hbeam_sdelay"] = (hedges_ev["hedge_sharp8"][0],
                                          cand_ev["sharp8"][1])
    shape_variants["mix_sbeam_hdelay"] = (cand_ev["sharp8"][0],
                                          hedges_ev["hedge_sharp8"][1])
    shape_variants["mix_sbeam_kdelay"] = (cand_ev["sharp8"][0],
                                          cand_ev["knn32"][1])
    bs_true, ds_true = Bn[eval_idx], Dn[eval_idx]
    out["shape_cosines"] = {}
    for name, (hb, hd) in shape_variants.items():
        cb = float(np.nanmean(shape_cos(hb, bs_true)))
        cd = float(np.nanmean(shape_cos(hd, ds_true)))
        out["shape_cosines"][name] = {"beam": cb, "delay": cd}
        print(f"shape[{name}] beam={cb:.4f} delay={cd:.4f}")

    # --- full metric via rank-1 gains ---
    etas = [float(e) for e in args.etas.split(",")]
    gain_variants = args.gain_variants.split(",")
    vnames = ["base"] + [f"{v}_eta{e:g}" for v in gain_variants for e in etas]
    acc = {v: {kk: [] for kk in ["pas", "pdp", "e2", "t2", "p2", "dot"]}
           for v in vnames}
    base_mm = np.load(ASSETS + "/train_v36_loo_base_derot_ad.npy", mmap_mode="r")
    res_mm = np.load(ASSETS + "/train_v36_loo_residual_derot_ad.npy", mmap_mode="r")
    regimes = regime_of(blocked)[eval_idx]
    delta_b = 1e-4 / 128.0
    delta_d = 1e-4 / 192.0
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
        for v in gain_variants:
            hb, hd = shape_variants[v]
            bp = torch.from_numpy(np.ascontiguousarray(hb[bs:be])).to(dev).reshape(-1, 16, 8, 1)
            dp = torch.from_numpy(np.ascontiguousarray(hd[bs:be])).to(dev).reshape(-1, 1, 1, 192)
            g2 = ((bp + delta_b) / (bb + delta_b)) * ((dp + delta_d) / (db + delta_d))
            for e in etas:
                g = (torch.sqrt(g2) ** e).clamp(1.0 / args.gain_clip, args.gain_clip)
                m = metrics_batch(base_t * g[:, None, :, :, None, :], truth_t)
                for kk in m:
                    acc[f"{v}_eta{e:g}"][kk].append(m[kk])
        print(f"eval batch {bs}-{be}")

    for v in vnames:
        summarize(v, acc[v], regimes, out)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
