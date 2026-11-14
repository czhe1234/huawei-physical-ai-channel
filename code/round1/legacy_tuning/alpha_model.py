# （本改动由Claude fable5）
"""Per-point amplitude scaling (alpha) + power floor for the NMSE term.

Prediction = LOO base AD with sharp8 (k=8, p=3) rank-1 marginal gains,
eta=0.5, gain clip 8 (recreates sharp8_eta0.5 in sharp_marginal_gain_results.json).

Steps:
 1. Recompute sharp8-eta0.5 predictions for all 2000 points, collect per-point
    (pas, pdp, e2, t2, p2, dot) on GPU. Verify C1/C2 vs reference JSON.
 2. Oracle alpha* = dot/p2 on TRAIN (non-spatial_matched, T>0). Fit
    HistGradientBoostingRegressor(alpha* clip [0,2]) on scalar features.
    5-fold CV -> R^2, MAE.
 3. On eval 300: apply alpha_hat clip [0.3, 1.5]; report C3 conventions
    before/after; numerically verify C1/C2 invariance (<=1e-6).
 4. Global alpha grid {0.5..1.2} chosen on train C3g -> eval C3g.
 5. Power floor: q10 (on train) of ratio pred_total/knn_total; eval points
    below the floor are rescaled up to it. Report all conventions; also
    combined with alpha_hat.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import PATHS
from powerfield_common import ASSETS, TINY, build_collapsed_caches, sharp_marginals, regime_of
from power_calibration_lab import metrics_batch

DEV = torch.device("cuda:1")
OUT = PATHS.analysis / "results" / "tuning" / "alpha_model_results.json"
REF_C1, REF_C2 = 0.665781557559967, 0.7354739308357239  # sharp8_eta0.5
ETA, GAIN_CLIP = 0.5, 8.0


def c_scores(pas, pdp, e2, t2):
    nz = t2 > 0
    nmse_i = np.where(nz, e2 / np.where(nz, t2, 1.0), np.nan)
    c1, c2 = float(np.nanmean(pas)), float(np.nanmean(pdp))
    c3g = float(e2.sum() / t2.sum())
    c3pp = float(np.nanmean(nmse_i))
    c3inv = float(np.nanmean(1.0 / (1.0 + nmse_i)))
    return dict(C1=c1, C2=c2, C3_global=c3g, C3_pp_mean=c3pp, C3_inv_mean=c3inv,
                C_global=0.4 * c1 + 0.4 * c2 + 0.2 / (1 + c3g),
                C_ppmean=0.4 * c1 + 0.4 * c2 + 0.2 / (1 + c3pp),
                C_invmean=0.4 * c1 + 0.4 * c2 + 0.2 * c3inv)


def scaled_e2(c, p2, dot, t2):
    """e2 of c*pred: c^2 p2 - 2 c dot + t2 (c per-point or scalar)."""
    return c ** 2 * p2 - 2.0 * c * dot + t2


def main():
    coll, bcoll = build_collapsed_caches()
    pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy")
    blocked = np.load(ASSETS + "/height_geometry_cache.npz")["train_blocked"].astype(np.float64)
    val = np.load(ASSETS + "/test_regime_analysis/test_matched_val_sets.npz")
    matched = val["spatial_matched"]
    eval_idx = matched[:300]

    T_t = coll.sum((1, 2, 3))
    nz = T_t > 0
    Bn = coll.sum(-1).reshape(2000, 128) / np.maximum(T_t[:, None], TINY)
    Dn = coll.sum((1, 2)) / np.maximum(T_t[:, None], TINY)

    # --- sharp8 (k=8, p=3) marginal shapes for all points (exclude self) ---
    print("computing sharp8 marginals for 2000 points ...")
    sb8 = np.empty((2000, 128), dtype=np.float32)
    sd8 = np.empty((2000, 192), dtype=np.float32)
    for i in range(2000):
        excl = np.hypot(pos[:, 0] - pos[i, 0], pos[:, 1] - pos[i, 1]) < 1e-6
        b, d, *_ = sharp_marginals(Bn, Dn, T_t, pos, blocked, pos[i, :2],
                                   blocked[i], k=8, p=3, exclude_idx=excl)
        sb8[i], sd8[i] = b, d
    print("sharp8 marginals done")

    # --- GPU pass: sharp8-eta0.5 predictions + per-point metric parts ---
    base_mm = np.load(ASSETS + "/train_v36_loo_base_derot_ad.npy", mmap_mode="r")
    res_mm = np.load(ASSETS + "/train_v36_loo_residual_derot_ad.npy", mmap_mode="r")
    keys = ["pas", "pdp", "e2", "t2", "p2", "dot"]
    st = {k: np.empty(2000, dtype=np.float64) for k in keys}
    st_base = {k: np.empty(2000, dtype=np.float64) for k in keys}
    delta_b, delta_d = 1e-4 / 128.0, 1e-4 / 192.0
    batch = 50
    for bs in range(0, 2000, batch):
        be = min(bs + batch, 2000)
        ids = np.arange(bs, be)
        base_np = np.asarray(base_mm[bs:be])
        truth_t = torch.from_numpy(base_np + np.asarray(res_mm[bs:be])).to(DEV)
        base_t = torch.from_numpy(base_np).to(DEV)
        mb = metrics_batch(base_t, truth_t)
        for k in keys:
            st_base[k][ids] = mb[k]
        Pb = (base_t.real ** 2 + base_t.imag ** 2).sum(dim=(1, 4))
        Tb_pt = Pb.sum(dim=(1, 2, 3), keepdim=True)
        bb = Pb.sum(-1, keepdim=True) / (Tb_pt + TINY)
        db = Pb.sum(dim=(1, 2), keepdim=True) / (Tb_pt + TINY)
        bp = torch.from_numpy(sb8[bs:be]).to(DEV).reshape(-1, 16, 8, 1)
        dp = torch.from_numpy(sd8[bs:be]).to(DEV).reshape(-1, 1, 1, 192)
        g2 = ((bp + delta_b) / (bb + delta_b)) * ((dp + delta_d) / (db + delta_d))
        g = (torch.sqrt(g2) ** ETA).clamp(1.0 / GAIN_CLIP, GAIN_CLIP)
        pred = base_t * g[:, None, :, :, None, :]
        m = metrics_batch(pred, truth_t)
        for k in keys:
            st[k][ids] = m[k]
        if bs % 400 == 0:
            print(f"  batch {bs}-{be}")

    out = {"reference": {"C1": REF_C1, "C2": REF_C2}, "variants": {}}
    ev = {k: st[k][eval_idx] for k in keys}
    row = c_scores(ev["pas"], ev["pdp"], ev["e2"], ev["t2"])
    print("recreated sharp8_eta0.5 eval:", json.dumps(row))
    print(f"C1 diff vs ref: {row['C1'] - REF_C1:.2e}, C2 diff: {row['C2'] - REF_C2:.2e}")
    out["variants"]["sharp8_eta0.5"] = row
    out["variants"]["base"] = c_scores(st_base["pas"][eval_idx], st_base["pdp"][eval_idx],
                                       st_base["e2"][eval_idx], st_base["t2"][eval_idx])

    # --- train/eval split ---
    excl = set(matched.tolist())
    train_ids = np.array([i for i in range(2000) if i not in excl and nz[i]])
    print(f"train n={len(train_ids)}, eval n={len(eval_idx)}")

    # --- oracle alpha* + features ---
    alpha_star = np.where(st["p2"] > 0, st["dot"] / np.where(st["p2"] > 0, st["p2"], 1.0), 0.0)
    y_all = np.clip(alpha_star, 0.0, 2.0)

    z = np.load(ASSETS + "/powerfield_train_feats_v2.npz")
    logT_ratio = 0.5 * (np.log(st["p2"] + TINY) - np.log(st_base["p2"] + TINY))
    X_all = np.concatenate([z["agree"], z["geo"], z["blocked"], z["knn_total"],
                            z["nn1_total"], z["base_total"],
                            logT_ratio[:, None].astype(np.float32)], axis=1)
    print("feature dim:", X_all.shape[1])

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score, mean_absolute_error

    Xtr, ytr = X_all[train_ids], y_all[train_ids]
    cv_pred = np.empty_like(ytr)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(Xtr):
        hgb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                            max_depth=None, min_samples_leaf=20,
                                            l2_regularization=1.0, random_state=0)
        hgb.fit(Xtr[tr], ytr[tr])
        cv_pred[te] = hgb.predict(Xtr[te])
    cv = {"R2": float(r2_score(ytr, cv_pred)),
          "MAE": float(mean_absolute_error(ytr, cv_pred)),
          "R2_clipped_pred": float(r2_score(ytr, np.clip(cv_pred, 0.3, 1.5))),
          "MAE_clipped_pred": float(mean_absolute_error(ytr, np.clip(cv_pred, 0.3, 1.5))),
          "target_mean": float(ytr.mean()), "target_median": float(np.median(ytr))}
    print("5-fold CV on train:", json.dumps(cv))
    out["alpha_hat_cv"] = cv

    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                          max_depth=None, min_samples_leaf=20,
                                          l2_regularization=1.0, random_state=0)
    model.fit(Xtr, ytr)
    a_hat_ev = np.clip(model.predict(X_all[eval_idx]), 0.3, 1.5)
    out["alpha_hat_eval_stats"] = {
        "min": float(a_hat_ev.min()), "max": float(a_hat_ev.max()),
        "mean": float(a_hat_ev.mean()), "median": float(np.median(a_hat_ev)),
        "frac_at_lower_clip": float((a_hat_ev <= 0.3 + 1e-9).mean()),
        "frac_at_upper_clip": float((a_hat_ev >= 1.5 - 1e-9).mean()),
    }
    print("alpha_hat eval stats:", json.dumps(out["alpha_hat_eval_stats"]))

    # --- global alpha grid, chosen on TRAIN C3g ---
    tr = {k: st[k][train_ids] for k in keys}
    grid = {}
    for a in np.arange(0.5, 1.2001, 0.1):
        a = round(float(a), 2)
        c3_tr = float(scaled_e2(a, tr["p2"], tr["dot"], tr["t2"]).sum() / tr["t2"].sum())
        c3_ev = float(scaled_e2(a, ev["p2"], ev["dot"], ev["t2"]).sum() / ev["t2"].sum())
        grid[a] = {"train_C3g": c3_tr, "eval_C3g": c3_ev}
    best_a = min(grid, key=lambda a: grid[a]["train_C3g"])
    a_closed = float(tr["dot"].sum() / tr["p2"].sum())
    out["global_alpha_grid"] = {str(a): grid[a] for a in grid}
    out["global_alpha_best_on_train"] = best_a
    out["global_alpha_closed_form_train"] = a_closed
    print("global alpha grid:", json.dumps(out["global_alpha_grid"], indent=1))
    print(f"best global alpha on train: {best_a} (closed-form {a_closed:.4f})")

    # --- power floor ---
    knnT = np.exp(z["knn_total"][:, 0].astype(np.float64))  # IDW neighbor total power
    ratio_tr = st["p2"][train_ids] / np.maximum(knnT[train_ids], TINY)
    q10 = float(np.quantile(ratio_tr, 0.10))
    out["floor_q10_ratio"] = q10
    print(f"floor: q10 of pred_total/knn_total on train = {q10:.4f}")

    def floor_scale(c_prev):
        """Per-point scale to lift (c_prev*pred) power up to q10*knnT."""
        p2s = c_prev ** 2 * ev["p2"]
        thr = q10 * knnT[eval_idx]
        return np.where(p2s < thr, np.sqrt(thr / np.maximum(p2s, TINY)), 1.0)

    ones = np.ones(len(eval_idx))
    s_only = floor_scale(ones)
    s_after = floor_scale(a_hat_ev)
    out["floor_stats"] = {"frac_raised_no_alpha": float((s_only > 1).mean()),
                          "max_scale_no_alpha": float(s_only.max()),
                          "frac_raised_after_alpha_hat": float((s_after > 1).mean())}

    # --- power-guarded alpha: alpha=1 on low-predicted-power points -------
    # Downscaling tiny-power points barely helps global NMSE but measurably
    # hurts local C2 (per-pair PDP norms drop to the eps=1e-30 floor of the
    # scorer). Guard: keep alpha=1 below a train-quantile of pred power p2.
    guards = {}
    for q in (0.2, 0.3, 0.5):
        thr = float(np.quantile(st["p2"][train_ids], q))
        a_g = np.where(st["p2"][eval_idx] < thr, 1.0, a_hat_ev)
        guards[q] = a_g
        out.setdefault("guard_thresholds", {})[str(q)] = thr

    variants = {
        "alpha_hat_clip0.3_1.5": a_hat_ev,
        "alpha_oracle_unclipped": alpha_star[eval_idx],
        "alpha_oracle_clip0.3_1.5": np.clip(alpha_star[eval_idx], 0.3, 1.5),
        f"global_alpha_{best_a}": np.full(len(eval_idx), best_a),
        "global_alpha_closed_form": np.full(len(eval_idx), a_closed),
        "floor_only": s_only,
        "alpha_hat_plus_floor": a_hat_ev * s_after,
        f"global_alpha_{best_a}_plus_floor": best_a * floor_scale(np.full(len(eval_idx), best_a)),
        "alpha_hat_guard_q20": guards[0.2],
        "alpha_hat_guard_q30": guards[0.3],
        "alpha_hat_guard_q50": guards[0.5],
        "alpha_hat_guard_q30_plus_floor": guards[0.3] * floor_scale(guards[0.3]),
    }

    # --- numeric C1/C2 per variant (scaling is NOT metric-invariant under
    # the local eps convention at low power); e2 taken analytically (exact).
    print("numeric C1/C2 pass over eval variants ...")
    pas_num = {v: np.empty(len(eval_idx)) for v in variants}
    pdp_num = {v: np.empty(len(eval_idx)) for v in variants}
    e2_chk = {v: [] for v in variants}
    for bs in range(0, len(eval_idx), batch):
        be = min(bs + batch, len(eval_idx))
        ids = eval_idx[bs:be]
        base_np = np.asarray(base_mm[ids])
        truth_t = torch.from_numpy(base_np + np.asarray(res_mm[ids])).to(DEV)
        base_t = torch.from_numpy(base_np).to(DEV)
        Pb = (base_t.real ** 2 + base_t.imag ** 2).sum(dim=(1, 4))
        Tb_pt = Pb.sum(dim=(1, 2, 3), keepdim=True)
        bb = Pb.sum(-1, keepdim=True) / (Tb_pt + TINY)
        db = Pb.sum(dim=(1, 2), keepdim=True) / (Tb_pt + TINY)
        bp = torch.from_numpy(sb8[ids]).to(DEV).reshape(-1, 16, 8, 1)
        dp = torch.from_numpy(sd8[ids]).to(DEV).reshape(-1, 1, 1, 192)
        g2 = ((bp + delta_b) / (bb + delta_b)) * ((dp + delta_d) / (db + delta_d))
        g = (torch.sqrt(g2) ** ETA).clamp(1.0 / GAIN_CLIP, GAIN_CLIP)
        pred0 = base_t * g[:, None, :, :, None, :]
        for v, c in variants.items():
            a = torch.from_numpy(c[bs:be].astype(np.float32)).to(DEV)
            m = metrics_batch(pred0 * a[:, None, None, None, None, None], truth_t)
            pas_num[v][bs:be] = m["pas"]
            pdp_num[v][bs:be] = m["pdp"]
            e2_an = scaled_e2(c[bs:be], ev["p2"][bs:be], ev["dot"][bs:be], ev["t2"][bs:be])
            e2_chk[v].append(np.abs(m["e2"] - e2_an) / np.maximum(np.abs(e2_an), 1e-30))
        print(f"  eval batch {bs}-{be}")

    out["e2_analytic_max_rel_err"] = float(max(np.concatenate(e2_chk[v]).max()
                                               for v in variants))
    for v, c in variants.items():
        e2n = scaled_e2(c, ev["p2"], ev["dot"], ev["t2"])
        r = c_scores(pas_num[v], pdp_num[v], e2n, ev["t2"])
        r["dC1_vs_unscaled"] = r["C1"] - out["variants"]["sharp8_eta0.5"]["C1"]
        r["dC2_vs_unscaled"] = r["C2"] - out["variants"]["sharp8_eta0.5"]["C2"]
        out["variants"][v] = r
        print(f"[{v}] C1={r['C1']:.4f} C2={r['C2']:.4f} C3g={r['C3_global']:.4f} "
              f"Cg={r['C_global']:.4f} Cpp={r['C_ppmean']:.4f} Cinv={r['C_invmean']:.4f} "
              f"dC1={r['dC1_vs_unscaled']:+.5f} dC2={r['dC2_vs_unscaled']:+.5f}")

    # analytic (scale-invariant C1/C2) view for the headline variants
    for v in ["alpha_hat_clip0.3_1.5", f"global_alpha_{best_a}", "floor_only",
              "alpha_hat_plus_floor", "alpha_hat_guard_q30"]:
        c = variants[v]
        e2n = scaled_e2(c, ev["p2"], ev["dot"], ev["t2"])
        out["variants"][v + "__analyticC1C2"] = c_scores(ev["pas"], ev["pdp"], e2n, ev["t2"])

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", OUT)


if __name__ == "__main__":
    main()
