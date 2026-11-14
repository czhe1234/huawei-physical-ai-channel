from project_paths import PATHS
# （本改动由Claude fable5）
"""Power-calibration lab.

Tests the hypothesis that C1/C2 (0.8 of the score) can be raised by predicting
the angle-delay POWER of each point and imposing it on the base (v36-LOO)
prediction's phases, instead of predicting the complex channel directly.

Works entirely in the derotated angle-delay domain using the LOO caches:
  train_v36_loo_base_derot_ad.npy      (2000, 2,16,8,4,192) complex64  base
  train_v36_loo_residual_derot_ad.npy  (2000, 2,16,8,4,192) complex64  truth-base

Both pred and truth are compared in the same domain; PAS/PDP cosines and NMSE
are identical to H-domain values because the transforms are unitary up to a
constant factor and the derotation is a common per-(point, subcarrier) phase.

Variants:
  base            : v36 LOO prediction as-is (reference)
  oracle_full     : |ad| = truth magnitude, phase = base phase (upper bound)
  oracle_coll     : gain from true pol/UE-collapsed power (16,8,192)
  knn_coll        : gain from geometry-gated kNN IDW of neighbor collapsed powers
  knn_coll_plain  : same without geometry gating
Each variant is also reported with oracle per-point scale (alpha) and best
global scale, to bound the NMSE term.
"""

import argparse
import json
import math

import numpy as np
import torch

ASSETS = PATHS.assets
BS_POS = np.array([50.0, 0.0, 25.0])


def load_all():
    base = np.load(ASSETS + "/train_v36_loo_base_derot_ad.npy", mmap_mode="r")
    res = np.load(ASSETS + "/train_v36_loo_residual_derot_ad.npy", mmap_mode="r")
    pos = np.load(str(PATHS.round1 / "Round1_Train_Pos.npy"))
    geo = np.load(ASSETS + "/height_geometry_cache.npz")
    blocked = geo["train_blocked"].astype(np.float64)
    val = np.load(ASSETS + "/test_regime_analysis/test_matched_val_sets.npz")
    return base, res, pos, blocked, val


def regime_of(blocked):
    # thresholds from generate_regime_s1style_geom_blend.py
    r = np.full(blocked.shape, "medium", dtype=object)
    r[blocked <= 0] = "clear"
    r[(blocked >= 1) & (blocked <= 25)] = "light"
    r[(blocked > 25) & (blocked <= 100)] = "medium"
    r[blocked > 100] = "heavy"
    return r


def metrics_batch(pred, truth):
    """pred, truth: torch complex64 (B, 2,16,8,4,192) on GPU.

    Returns dict of per-point arrays: pas, pdp, nmse (B,).
    """
    eps = 1e-30
    # --- C1: per-subcarrier beamspace power cosine ---
    # delay -> subcarrier axis
    Xp = torch.fft.fft(pred, dim=-1)  # (B,2,16,8,4,S)
    Xt = torch.fft.fft(truth, dim=-1)
    # beam power over (pol,h,v) for each (ue, s)
    Pp = (Xp.real**2 + Xp.imag**2).reshape(Xp.shape[0], 256, 4, 192)
    Pt = (Xt.real**2 + Xt.imag**2).reshape(Xt.shape[0], 256, 4, 192)
    num = (Pp * Pt).sum(dim=1)
    den = Pp.norm(dim=1) * Pt.norm(dim=1)
    pas = (num / (den + eps)).mean(dim=(1, 2))  # avg over ue, s
    # --- C2: per-antenna-pair PDP cosine ---
    ht_p = torch.fft.ifft2(pred, dim=(2, 3))  # back to antenna domain
    ht_t = torch.fft.ifft2(truth, dim=(2, 3))
    Dp = (ht_p.real**2 + ht_p.imag**2).reshape(ht_p.shape[0], 256, 4, 192)
    Dt = (ht_t.real**2 + ht_t.imag**2).reshape(ht_t.shape[0], 256, 4, 192)
    num2 = (Dp * Dt).sum(dim=-1)
    den2 = Dp.norm(dim=-1) * Dt.norm(dim=-1)
    pdp = (num2 / (den2 + eps)).mean(dim=(1, 2))  # avg over 256x4 pairs
    # --- NMSE per point + inner products for alpha ---
    err = pred - truth
    e2 = (err.real**2 + err.imag**2).sum(dim=(1, 2, 3, 4, 5))
    t2 = (truth.real**2 + truth.imag**2).sum(dim=(1, 2, 3, 4, 5))
    p2 = (pred.real**2 + pred.imag**2).sum(dim=(1, 2, 3, 4, 5))
    dot = (pred.real * truth.real + pred.imag * truth.imag).sum(dim=(1, 2, 3, 4, 5))
    return dict(
        pas=pas.cpu().numpy(),
        pdp=pdp.cpu().numpy(),
        e2=e2.cpu().numpy(),
        t2=t2.cpu().numpy(),
        p2=p2.cpu().numpy(),
        dot=dot.cpu().numpy(),
    )


def summarize(name, acc, regimes, out):
    pas = np.concatenate(acc["pas"])
    pdp = np.concatenate(acc["pdp"])
    e2 = np.concatenate(acc["e2"])
    t2 = np.concatenate(acc["t2"])
    p2 = np.concatenate(acc["p2"])
    dot = np.concatenate(acc["dot"])
    nz = t2 > 0
    nmse_i = np.where(nz, e2 / np.where(nz, t2, 1.0), np.nan)
    # C3 conventions
    c3_global = e2.sum() / t2.sum()
    c3_pp_mean = np.nanmean(nmse_i)
    c3_inv_mean = np.nanmean(1.0 / (1.0 + nmse_i))
    # oracle per-point alpha
    alpha = np.where(p2 > 0, dot / np.where(p2 > 0, p2, 1.0), 0.0)
    e2_alpha = e2 - np.where(p2 > 0, dot**2 / np.where(p2 > 0, p2, 1.0), 0.0)
    c3_global_alpha = e2_alpha.sum() / t2.sum()
    # best global alpha
    a_g = dot.sum() / p2.sum()
    e2_ag = (a_g**2 * p2 - 2 * a_g * dot + t2)
    c3_global_ag = e2_ag.sum() / t2.sum()
    C1, C2 = np.nanmean(pas), np.nanmean(pdp)
    res = {
        "C1_pas": float(C1),
        "C2_pdp": float(C2),
        "C3_global": float(c3_global),
        "C3_pp_mean": float(c3_pp_mean),
        "C3_invterm_mean": float(c3_inv_mean),
        "C_global": float(0.4 * C1 + 0.4 * C2 + 0.2 / (1 + c3_global)),
        "C_ppmean": float(0.4 * C1 + 0.4 * C2 + 0.2 / (1 + c3_pp_mean)),
        "C_invmean": float(0.4 * C1 + 0.4 * C2 + 0.2 * c3_inv_mean),
        "C3_global_oracle_alpha": float(c3_global_alpha),
        "C3_global_best_global_alpha": float(c3_global_ag),
        "best_global_alpha": float(a_g),
        "nmse_median": float(np.nanmedian(nmse_i)),
        "frac_nmse_gt1": float(np.nanmean(nmse_i > 1)),
        "alpha_median": float(np.median(alpha)),
    }
    # per-regime C1/C2
    for reg in ["clear", "light", "medium", "heavy"]:
        m = regimes == reg
        if m.sum() > 0:
            res[f"C1_{reg}"] = float(np.nanmean(pas[m]))
            res[f"C2_{reg}"] = float(np.nanmean(pdp[m]))
            res[f"n_{reg}"] = int(m.sum())
    out[name] = res
    print(f"[{name}] C1={C1:.4f} C2={C2:.4f} C3g={c3_global:.4f} "
          f"Cg={res['C_global']:.4f} Cpp={res['C_ppmean']:.4f} Cinv={res['C_invmean']:.4f}")
    return res


def knn_weights(pos, blocked, q_idx, k, geom_gate, exclude):
    """IDW weights over train points for query q_idx (a train point, LOO)."""
    dq = pos[:, :2] - pos[q_idx, :2]
    d = np.hypot(dq[:, 0], dq[:, 1])
    if geom_gate:
        los_q = blocked[q_idx] <= 0
        los_i = blocked <= 0
        eff = d * (1.0 + 1.0 * (los_i != los_q) + 0.5 * np.abs(blocked - blocked[q_idx]) / 80.0)
    else:
        eff = d.copy()
    eff[exclude] = np.inf
    idx = np.argpartition(eff, k)[:k]
    w = 1.0 / np.maximum(eff[idx], 0.25) ** 2
    w = w / w.sum()
    return idx, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-set", default="spatial_matched")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--gain-clip", type=float, default=1000.0)
    ap.add_argument("--gain-eta", type=float, default=1.0)
    ap.add_argument("--log-avg", action="store_true")
    ap.add_argument("--out", default=ASSETS + "/power_calibration_lab_results.json")
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    base_mm, res_mm, pos, blocked, val = load_all()
    eval_idx = val[args.val_set][: args.n]
    regimes_all = regime_of(blocked)
    regimes = regimes_all[eval_idx]

    # collapsed truth power for all train points, in chunks
    print("building collapsed truth power [2000,16,8,192] ...")
    coll = np.empty((2000, 16, 8, 192), dtype=np.float32)
    for s in range(0, 2000, 100):
        e = min(s + 100, 2000)
        t = np.asarray(base_mm[s:e]) + np.asarray(res_mm[s:e])
        coll[s:e] = (np.abs(t) ** 2).sum(axis=(1, 4))
        del t
    print("collapsed power built. total:", float(coll.sum()))

    variants = [
        "base", "oracle_full", "oracle_coll",
        "oracle_beam",      # true beam map (16,8,1), delay structure from base
        "oracle_delay",     # true delay profile (1,1,192), beam structure from base
        "oracle_margins",   # composed beam-gain x delay-gain (rank-1 correction)
        "knn_coll", "knn_coll_plain",
        "nn1_coll",         # single geometry-matched neighbor power transfer
        "knn_trim",         # trimmed mean (drop top/bottom neighbor per bin)
        "knn_blend_e50",    # gain = (P_knn/P_base)^0.5 (partial calibration)
        "knn_blend_e25",    # gain = (P_knn/P_base)^0.25
        "knn_blend_heavy",  # eta: clear/light 0, medium 0.25, heavy 0.5
    ]
    acc = {v: {kk: [] for kk in ["pas", "pdp", "e2", "t2", "p2", "dot"]} for v in variants}
    eta_regime = {"clear": 0.0, "light": 0.0, "medium": 0.25, "heavy": 0.5}

    eps_t = 1e-30
    for bs in range(0, len(eval_idx), args.batch):
        be = min(bs + args.batch, len(eval_idx))
        ids = eval_idx[bs:be]
        base_np = np.asarray(base_mm[ids])
        truth_np = base_np + np.asarray(res_mm[ids])
        base_t = torch.from_numpy(base_np).to(dev)
        truth_t = torch.from_numpy(truth_np).to(dev)

        # --- base ---
        m = metrics_batch(base_t, truth_t)
        for kk in m:
            acc["base"][kk].append(m[kk])

        # --- oracle_full: truth magnitude, base phase ---
        mag_t = truth_t.abs()
        ph = base_t / (base_t.abs() + eps_t)
        # where base is ~zero, keep truth magnitude with unit phase (real)
        pred = mag_t * torch.where(base_t.abs() > eps_t, ph, torch.ones_like(ph))
        m = metrics_batch(pred, truth_t)
        for kk in m:
            acc["oracle_full"][kk].append(m[kk])

        # --- oracle gains ---
        Pb = (base_t.real**2 + base_t.imag**2).sum(dim=(1, 4))  # (B,16,8,192)
        Pt = torch.from_numpy(coll[ids]).to(dev)
        delta = 1e-4 * Pb.mean(dim=(1, 2, 3), keepdim=True) + 1e-30

        def apply_gain(Ptarget, Pbase, eta):
            g = torch.sqrt((Ptarget + delta) / (Pbase + delta)) ** eta
            g = g.clamp(1.0 / args.gain_clip, args.gain_clip)
            return base_t * g[:, None, :, :, None, :]

        m = metrics_batch(apply_gain(Pt, Pb, args.gain_eta), truth_t)
        for kk in m:
            acc["oracle_coll"][kk].append(m[kk])

        # beam-only oracle: match true beam map, keep base delay structure per beam
        Pt_beam = Pt.sum(dim=-1, keepdim=True)
        Pb_beam = Pb.sum(dim=-1, keepdim=True)
        m = metrics_batch(apply_gain(Pt_beam, Pb_beam, args.gain_eta), truth_t)
        for kk in m:
            acc["oracle_beam"][kk].append(m[kk])

        # delay-only oracle
        Pt_del = Pt.sum(dim=(1, 2), keepdim=True)
        Pb_del = Pb.sum(dim=(1, 2), keepdim=True)
        m = metrics_batch(apply_gain(Pt_del, Pb_del, args.gain_eta), truth_t)
        for kk in m:
            acc["oracle_delay"][kk].append(m[kk])

        # composed marginal oracle: beam gain x delay gain (rank-1 correction)
        g_b = torch.sqrt((Pt_beam + delta) / (Pb_beam + delta))
        g_d = torch.sqrt((Pt_del + delta) / (Pb_del + delta))
        g = (g_b * g_d).clamp(1.0 / args.gain_clip, args.gain_clip)
        m = metrics_batch(base_t * g[:, None, :, :, None, :], truth_t)
        for kk in m:
            acc["oracle_margins"][kk].append(m[kk])

        # --- knn power predictions (geometry-gated unless plain) ---
        preds_np = {v: np.empty((len(ids), 16, 8, 192), dtype=np.float32)
                    for v in ["knn_coll", "knn_coll_plain", "nn1_coll", "knn_trim"]}
        for j, q in enumerate(ids):
            idx, w = knn_weights(pos, blocked, q, args.k, True, exclude=[q])
            nbr = coll[idx]
            if args.log_avg:
                preds_np["knn_coll"][j] = np.exp(
                    (w[:, None, None, None] * np.log(nbr + 1e-20)).sum(axis=0))
            else:
                preds_np["knn_coll"][j] = (w[:, None, None, None] * nbr).sum(axis=0)
            order = np.argsort(-w)
            preds_np["nn1_coll"][j] = nbr[order[0]]
            # trimmed mean over top-8 neighbors per bin
            top = nbr[order[:8]]
            top_sorted = np.sort(top, axis=0)
            preds_np["knn_trim"][j] = top_sorted[1:-1].mean(axis=0)
            idxp, wp = knn_weights(pos, blocked, q, args.k, False, exclude=[q])
            preds_np["knn_coll_plain"][j] = (wp[:, None, None, None] * coll[idxp]).sum(axis=0)

        for vname in ["knn_coll", "knn_coll_plain", "nn1_coll", "knn_trim"]:
            Ptp = torch.from_numpy(preds_np[vname]).to(dev)
            m = metrics_batch(apply_gain(Ptp, Pb, args.gain_eta), truth_t)
            for kk in m:
                acc[vname][kk].append(m[kk])

        # --- partial-calibration blends of knn_coll ---
        Ptp = torch.from_numpy(preds_np["knn_coll"]).to(dev)
        for vname, eta in [("knn_blend_e50", 0.5), ("knn_blend_e25", 0.25)]:
            m = metrics_batch(apply_gain(Ptp, Pb, eta), truth_t)
            for kk in m:
                acc[vname][kk].append(m[kk])
        # per-regime eta
        etas = torch.tensor([eta_regime[r] for r in regimes_all[ids]],
                            dtype=torch.float32, device=dev)[:, None, None, None]
        g = torch.sqrt((Ptp + delta) / (Pb + delta)) ** etas
        g = g.clamp(1.0 / args.gain_clip, args.gain_clip)
        m = metrics_batch(base_t * g[:, None, :, :, None, :], truth_t)
        for kk in m:
            acc["knn_blend_heavy"][kk].append(m[kk])

        print(f"  batch {bs}-{be} done")

    out = {"config": vars(args), "val_set": args.val_set}
    for v in variants:
        summarize(v, acc[v], regimes, out)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
