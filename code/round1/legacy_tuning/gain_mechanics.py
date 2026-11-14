# （本改动由Claude fable5）
"""Rank-1 gain mechanics tuning with the sharp8 (k=8,p=3) shape source fixed.

Sweeps, on the eval-300 subset (spatial_matched[:300]) with full metrics:
  - per-regime eta (clear/light/medium/heavy)
  - gain clip {4, 8, 16}
  - separate eta for beam-gain vs delay-gain
  - delta regularizer scale {1e-3, 1e-4, 1e-5}
  - per-polarization delay gain (neighbor pol-resolved delay shapes)

Gain formula (as in train_marginal_gain_field.py):
  g2_beam  = (bp + db_)/(bb + db_)         bp,bb normalized 128-beam shapes
  g2_delay = (dp + dd_)/(db + dd_)         dp,db normalized 192-delay shapes
  g = g2_beam^(eta_b/2) * g2_delay^(eta_d/2), clamped to [1/clip, clip]
  pred = base * g   (broadcast over pol, ue)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import PATHS
from powerfield_common import ASSETS, TINY, build_collapsed_caches, regime_of, geom_effective_dist
from power_calibration_lab import metrics_batch, summarize

REGS = ["clear", "light", "medium", "heavy"]
TUNING_RESULTS = PATHS.analysis / "results" / "tuning"


def sharp_shapes(coll, dpol_n, T_t, pos, blocked, eval_idx, k=8, p=3):
    """Sharp local weighted average of neighbor normalized shapes (k=8,p=3).

    Returns beam (E,128), delay (E,192), pol-delay (E,2,192).
    """
    n = len(coll)
    Bn = coll.sum(-1).reshape(n, 128) / np.maximum(T_t[:, None], TINY)
    Dn = coll.sum((1, 2)) / np.maximum(T_t[:, None], TINY)
    E = len(eval_idx)
    sb = np.empty((E, 128), dtype=np.float64)
    sd = np.empty((E, 192), dtype=np.float64)
    sdp = np.empty((E, 2, 192), dtype=np.float64)
    for j, q in enumerate(eval_idx):
        eff = geom_effective_dist(pos, blocked, pos[q, :2], blocked[q]).copy()
        eff[T_t <= 0] = np.inf
        d_all = np.hypot(pos[:, 0] - pos[q, 0], pos[:, 1] - pos[q, 1])
        eff[d_all < 1e-6] = np.inf  # LOO: exclude self / co-located
        idx = np.argpartition(eff, k)[:k]
        idx = idx[np.argsort(eff[idx])]
        w = 1.0 / np.maximum(eff[idx], 0.3) ** p
        w = w / w.sum()
        sb[j] = (w[:, None] * Bn[idx]).sum(0)
        sd[j] = (w[:, None] * Dn[idx]).sum(0)
        sdp[j] = (w[:, None, None] * dpol_n[idx]).sum(0)
    return sb.astype(np.float32), sd.astype(np.float32), sdp.astype(np.float32)


def build_pol_delay_cache():
    """Pol-resolved truth delay power (2000,2,192), cached."""
    import os
    f = TUNING_RESULTS / "truth_pol_delay_power.npy"
    if os.path.exists(f):
        return np.load(f)
    base_mm = np.load(ASSETS + "/train_v36_loo_base_derot_ad.npy", mmap_mode="r")
    res_mm = np.load(ASSETS + "/train_v36_loo_residual_derot_ad.npy", mmap_mode="r")
    dpol = np.empty((2000, 2, 192), dtype=np.float32)
    for s in range(0, 2000, 100):
        e = min(s + 100, 2000)
        t = np.asarray(base_mm[s:e]) + np.asarray(res_mm[s:e])
        dpol[s:e] = (np.abs(t) ** 2).sum(axis=(2, 3, 4))
        del t
    np.save(f, dpol)
    return dpol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--phase", default="1")
    ap.add_argument("--out", default=str(TUNING_RESULTS / "gain_mechanics_results.json"))
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    coll, bcoll = build_collapsed_caches()
    pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy")
    blocked = np.load(ASSETS + "/height_geometry_cache.npz")["train_blocked"].astype(np.float64)
    val = np.load(ASSETS + "/test_regime_analysis/test_matched_val_sets.npz")
    eval_idx = val["spatial_matched"][: args.n_eval]
    regimes = regime_of(blocked)[eval_idx]
    T_t = coll.sum((1, 2, 3))

    print("building pol-resolved delay cache ...")
    dpol = build_pol_delay_cache()
    dpol_n = dpol / np.maximum(T_t[:, None, None], TINY)

    print("computing sharp8 shapes for eval points ...")
    sb, sd, sdp = sharp_shapes(coll, dpol_n, T_t, pos, blocked, eval_idx, k=8, p=3)

    # shape-cosine diagnostics
    bs_true = coll[eval_idx].sum(-1).reshape(-1, 128) / np.maximum(T_t[eval_idx, None], TINY)
    ds_true = coll[eval_idx].sum((1, 2)) / np.maximum(T_t[eval_idx, None], TINY)

    def mcos(a, b):
        return float(np.nanmean((a * b).sum(1) /
                     (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-20)))
    diag = {"beam_cos_sharp8": mcos(sb, bs_true), "delay_cos_sharp8": mcos(sd, ds_true)}
    print("shape diagnostics:", diag)

    # ---- variant definitions ----
    # each: name -> dict(eta_b={reg:...}, eta_d={reg:...}, clip, delta, pol=False)
    def V(eb, ed=None, clip=8.0, delta=1e-4, pol=False):
        if not isinstance(eb, dict):
            eb = {r: eb for r in REGS}
        if ed is None:
            ed = eb
        elif not isinstance(ed, dict):
            ed = {r: ed for r in REGS}
        return dict(eta_b=eb, eta_d=ed, clip=clip, delta=delta, pol=pol)

    def RG(c, l, m, h):
        return dict(zip(REGS, [c, l, m, h]))

    variants = {"ref_e0.5_c8_d1e-4": V(0.5)}
    if args.phase == "1":
        # per-regime eta (same beam/delay), clip8 d1e-4
        for tup in [(0.25, 0.25, 0.5, 0.5), (0.25, 0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5, 0.75), (0.5, 0.5, 0.75, 0.5),
                    (0.5, 0.5, 0.75, 0.75), (0.25, 0.5, 0.75, 0.75),
                    (0.5, 0.75, 0.75, 0.75), (0.25, 0.25, 0.75, 0.75),
                    (0.25, 0.5, 0.5, 0.75), (0.5, 0.25, 0.5, 0.5),
                    (0.25, 0.25, 0.25, 0.5), (0.5, 0.5, 0.5, 0.25),
                    (0.5, 0.5, 0.25, 0.25), (0.75, 0.75, 0.75, 0.75)]:
            variants[f"reg_c{tup[0]:g}_l{tup[1]:g}_m{tup[2]:g}_h{tup[3]:g}"] = V(RG(*tup))
        # clip sweep at uniform 0.5
        for c in [4.0, 16.0]:
            variants[f"clip{c:g}_e0.5"] = V(0.5, clip=c)
        # beam vs delay eta
        for eb, ed in [(0.5, 0.75), (0.75, 0.5), (0.25, 0.5), (0.5, 0.25),
                       (0.75, 0.25), (0.25, 0.75)]:
            variants[f"bd_eb{eb:g}_ed{ed:g}"] = V(eb, ed)
        # delta sweep
        for dl in [1e-3, 1e-5]:
            variants[f"delta{dl:g}_e0.5"] = V(0.5, delta=dl)
        # per-pol delay gain
        variants["pol_e0.5"] = V(0.5, pol=True)
    else:
        # phase 2: combinations of phase-1 winners (edited after phase 1)
        for name, v in json.load(open(TUNING_RESULTS / "gain_mechanics_phase2.json")).items():
            variants[name] = V(v["eta_b"] if isinstance(v["eta_b"], (int, float))
                               else v["eta_b"],
                               v.get("eta_d"), v.get("clip", 8.0),
                               v.get("delta", 1e-4), v.get("pol", False))

    names = ["base"] + list(variants)
    acc = {v: {kk: [] for kk in ["pas", "pdp", "e2", "t2", "p2", "dot"]} for v in names}

    base_mm = np.load(ASSETS + "/train_v36_loo_base_derot_ad.npy", mmap_mode="r")
    res_mm = np.load(ASSETS + "/train_v36_loo_residual_derot_ad.npy", mmap_mode="r")

    for bs_i in range(0, len(eval_idx), args.batch):
        be = min(bs_i + args.batch, len(eval_idx))
        ids = eval_idx[bs_i:be]
        regs = regimes[bs_i:be]
        base_np = np.asarray(base_mm[ids])
        truth_np = base_np + np.asarray(res_mm[ids])
        base_t = torch.from_numpy(base_np).to(dev)
        truth_t = torch.from_numpy(truth_np).to(dev)
        m = metrics_batch(base_t, truth_t)
        for kk in m:
            acc["base"][kk].append(m[kk])

        Pb = (base_t.real**2 + base_t.imag**2).sum(dim=(1, 4))  # (B,16,8,192)
        Tb_pt = Pb.sum(dim=(1, 2, 3), keepdim=True)
        bb = Pb.sum(-1, keepdim=True) / (Tb_pt + TINY)          # base beam shape (B,16,8,1)
        db = Pb.sum(dim=(1, 2), keepdim=True) / (Tb_pt + TINY)  # base delay shape (B,1,1,192)
        # pol-resolved base delay shape (B,2,192)
        Pb_pol = (base_t.real**2 + base_t.imag**2).sum(dim=(2, 3, 4))
        db_pol = Pb_pol / (Pb_pol.sum(dim=(1, 2), keepdim=True) + TINY)

        bp = torch.from_numpy(sb[bs_i:be]).to(dev).reshape(-1, 16, 8, 1)
        dp = torch.from_numpy(sd[bs_i:be]).to(dev).reshape(-1, 1, 1, 192)
        dpp = torch.from_numpy(sdp[bs_i:be]).to(dev)            # (B,2,192)

        for name, v in variants.items():
            delta_b = v["delta"] / 128.0
            delta_d = v["delta"] / 192.0
            eb = torch.tensor([v["eta_b"][r] for r in regs], dtype=torch.float32,
                              device=dev).reshape(-1, 1, 1, 1)
            ed = torch.tensor([v["eta_d"][r] for r in regs], dtype=torch.float32,
                              device=dev).reshape(-1, 1, 1, 1)
            g2b = (bp + delta_b) / (bb + delta_b)
            if v["pol"]:
                delta_dp = v["delta"] / (2 * 192.0)
                g2d_pol = (dpp + delta_dp) / (db_pol + delta_dp)     # (B,2,192)
                gb = g2b ** (eb / 2.0)                               # (B,16,8,1)
                gd = g2d_pol ** (ed.reshape(-1, 1, 1) / 2.0)         # (B,2,192)
                g = gb[:, None, :, :, None, :] * gd[:, :, None, None, None, :]
                g = g.clamp(1.0 / v["clip"], v["clip"])
                pred = base_t * g
            else:
                g2d = (dp + delta_d) / (db + delta_d)
                g = g2b ** (eb / 2.0) * g2d ** (ed / 2.0)
                g = g.clamp(1.0 / v["clip"], v["clip"])
                pred = base_t * g[:, None, :, :, None, :]
            m = metrics_batch(pred, truth_t)
            for kk in m:
                acc[name][kk].append(m[kk])
        print(f"batch {bs_i}-{be} done ({len(variants)} variants)")

    out = {"config": vars(args), "diagnostics": diag}
    for v in names:
        summarize(v, acc[v], regimes, out)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
