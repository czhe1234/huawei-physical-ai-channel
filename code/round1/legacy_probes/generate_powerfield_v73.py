from project_paths import PATHS
"""v73: marginal power-field calibration on top of the v55 base prediction.

Method (data-driven, no ray tracing):
 1. Base complex prediction = v55 pipeline output (geometry-gated physical
    interpolation + residual component gate). It supplies phase structure.
 2. For each test point, predict the true pol/UE-collapsed beam-power shape
    (16x8), delay-power shape (192), and per-polarization delay shapes (2x192)
    by sharp geometry-gated kNN shape transfer from the training channels
    (k=8, IDW p=3 on a blockage-aware effective distance). Power *shapes*
    are spatially smooth (measured beam-shape cosine 0.96 at 1-2 m even in
    heavy-blockage regions), unlike complex coefficients.
 3. Impose the predicted shapes on the base prediction as a rank-1
    multiplicative gain in the derotated angle-delay domain, with per-regime
    partial-calibration exponents (LOO-tuned):
      eta_beam : clear 0.5 / light 0.5  / medium 0.875 / heavy 0.75
      eta_delay: clear 0.5 / light 0.25 / medium 0.75  / heavy 0.25
    gain clip 16, per-polarization delay gains.
 4. Optional per-point amplitude calibration (--alpha-mode guard) and power
    floor for the NMSE term; off by default in v73.

LOO-validated on 300 spatially-matched holdout points:
  base C=0.6401 -> v73 recipe C=0.6761 (C1 0.610->0.679, C2 0.711->0.734).
"""

import argparse
import json

import numpy as np

ASSETS = PATHS.assets
TINY = 1e-30
MP, MH, MV, N_ANT, S_SC = 2, 16, 8, 4, 192
BS_POS = np.array([50.0, 0.0, 25.0])
ETA_BEAM = {"clear": 0.5, "light": 0.5, "medium": 0.875, "heavy": 0.75}
ETA_DELAY = {"clear": 0.5, "light": 0.25, "medium": 0.75, "heavy": 0.25}
GAIN_CLIP = 16.0
DELTA = 1e-4


def regime_of(blocked):
    r = np.full(blocked.shape, "medium", dtype=object)
    r[blocked <= 0] = "clear"
    r[(blocked >= 1) & (blocked <= 25)] = "light"
    r[(blocked > 25) & (blocked <= 100)] = "medium"
    r[blocked > 100] = "heavy"
    return r


def beta_subcarriers(beta=140.3, gamma=0.10):
    sub = np.linspace(-0.5, 0.5, S_SC, dtype=np.float32)
    return (beta + gamma * sub).astype(np.float32)


def ad_transform(h):
    cir = np.fft.ifft(h, axis=-1)
    arr = cir.reshape(h.shape[0], MP, MH, MV, N_ANT, S_SC)
    return np.fft.fft2(arr, axes=(2, 3)).astype(np.complex64)


def inverse_ad(ad):
    arr = np.fft.ifft2(ad, axes=(2, 3))
    cir = arr.reshape(ad.shape[0], MP * MH * MV, N_ANT, S_SC)
    return np.fft.fft(cir, axis=-1).astype(np.complex64)


def geom_effective_dist(pos_train, blocked_train, q_xy, q_blocked):
    d = np.hypot(pos_train[:, 0] - q_xy[0], pos_train[:, 1] - q_xy[1])
    los_q = q_blocked <= 0
    los_i = blocked_train <= 0
    return d * (1.0 + 1.0 * (los_i != los_q)
                + 0.5 * np.abs(blocked_train - q_blocked) / 80.0)


def build_train_power_bank(train_channel_path, pos_train, beta_s, batch=50):
    """Collapsed AD power of the (derotated) training channels.

    Returns coll (P,16,8,192), pol-delay (P,2,192), totals (P,).
    """
    train_mm = np.load(train_channel_path, mmap_mode="r")
    n = train_mm.shape[0]
    d_tr = np.linalg.norm(pos_train - BS_POS, axis=1)
    coll = np.empty((n, MH, MV, S_SC), dtype=np.float32)
    dpol = np.empty((n, MP, S_SC), dtype=np.float32)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        h = np.asarray(train_mm[s:e], dtype=np.complex64)
        h = h * np.exp(1j * d_tr[s:e, None, None, None] * beta_s).astype(np.complex64)
        ad = ad_transform(h)
        p = (np.abs(ad) ** 2)
        coll[s:e] = p.sum(axis=(1, 4))
        dpol[s:e] = p.sum(axis=(2, 3, 4))
    total = coll.sum((1, 2, 3))
    return coll, dpol, total


def sharp_shapes(coll, dpol, total, pos_train, blocked_train, q_pos, q_blocked,
                 k=8, p=3.0):
    Bn = coll.sum(-1).reshape(len(coll), 128) / np.maximum(total[:, None], TINY)
    Dn = coll.sum((1, 2)) / np.maximum(total[:, None], TINY)
    Dp = dpol / np.maximum(total[:, None, None], TINY)
    E = len(q_pos)
    sb = np.empty((E, 128), dtype=np.float32)
    sd = np.empty((E, 192), dtype=np.float32)
    sdp = np.empty((E, 2, 192), dtype=np.float32)
    ktot = np.empty(E, dtype=np.float32)
    for j in range(E):
        eff = geom_effective_dist(pos_train, blocked_train, q_pos[j, :2],
                                  q_blocked[j]).copy()
        eff[total <= 0] = np.inf
        idx = np.argpartition(eff, k)[:k]
        idx = idx[np.argsort(eff[idx])]
        w = 1.0 / np.maximum(eff[idx], 0.3) ** p
        w = w / w.sum()
        sb[j] = (w[:, None] * Bn[idx]).sum(0)
        sd[j] = (w[:, None] * Dn[idx]).sum(0)
        sdp[j] = (w[:, None, None] * Dp[idx]).sum(0)
        ktot[j] = (w * total[idx]).sum()
    return sb, sd, sdp, ktot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(PATHS.analysis / "probes/Round1_Test_Channel.npy"),
                    help="base complex prediction (v55 pipeline output)")
    ap.add_argument("--data", default=str(PATHS.round1))
    ap.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    ap.add_argument("--output", default=str(PATHS.cache_dir(73) / "v73_Round1_Test_Channel.npy"))
    ap.add_argument("--alpha-mode", choices=["off", "global", "file"], default="off")
    ap.add_argument("--alpha-global", type=float, default=0.8)
    ap.add_argument("--alpha-file", default="")
    ap.add_argument("--power-floor-rel", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--report", default=ASSETS + "/powerfield_v73_report.json")
    args = ap.parse_args()

    beta_s = beta_subcarriers()
    pos_train = np.load(args.data + "/Round1_Train_Pos.npy")
    pos_test = np.load(args.data + "/Round1_Test_Pos.npy")
    geo = np.load(args.geometry_cache)
    blocked_tr = geo["train_blocked"].astype(np.float64)
    blocked_te = geo["test_blocked"].astype(np.float64)
    regimes = regime_of(blocked_te)

    print("building train power bank ...")
    coll, dpol, total = build_train_power_bank(
        args.data + "/Round1_Train_Channel.npy", pos_train, beta_s, args.batch)

    print("predicting sharp shapes for test points ...")
    sb, sd, sdp, ktot = sharp_shapes(coll, dpol, total, pos_train, blocked_tr,
                                     pos_test, blocked_te)

    print("loading base prediction ...")
    base_h = np.load(args.base)
    assert base_h.shape == (500, 256, 4, 192) and base_h.dtype == np.complex64
    d_te = np.linalg.norm(pos_test - BS_POS, axis=1)

    out = np.empty_like(base_h)
    eta_b = np.array([ETA_BEAM[r] for r in regimes], dtype=np.float32)
    eta_d = np.array([ETA_DELAY[r] for r in regimes], dtype=np.float32)

    print("applying rank-1 marginal gains ...")
    for s in range(0, 500, args.batch):
        e = min(s + args.batch, 500)
        hb = base_h[s:e] * np.exp(
            1j * d_te[s:e, None, None, None] * beta_s).astype(np.complex64)
        ad = ad_transform(hb)
        P = np.abs(ad) ** 2
        Pb = P.sum(axis=(1, 4))                       # (B,16,8,192)
        Tb = Pb.sum(axis=(1, 2, 3), keepdims=True)
        bb = Pb.sum(-1, keepdims=True) / (Tb + TINY)              # (B,16,8,1)
        db_pol = P.sum(axis=(2, 3, 4))                             # (B,2,192)
        db_pol = db_pol / (db_pol.sum(axis=(1, 2), keepdims=True) + TINY)

        delta_b = DELTA / 128.0
        delta_dp = DELTA / (2 * 192.0)
        bp = sb[s:e].reshape(-1, 16, 8, 1)
        dpp = sdp[s:e]
        ebv = eta_b[s:e].reshape(-1, 1, 1, 1)
        edv = eta_d[s:e].reshape(-1, 1, 1)
        g2b = (bp + delta_b) / (bb + delta_b)
        g2d = (dpp + delta_dp) / (db_pol + delta_dp)
        gb = g2b ** (ebv / 2.0)                        # (B,16,8,1)
        gd = g2d ** (edv / 2.0)                        # (B,2,192)
        g = gb[:, None, :, :, None, :] * gd[:, :, None, None, None, :]
        g = np.clip(g, 1.0 / GAIN_CLIP, GAIN_CLIP).astype(np.float32)
        ad_out = (ad * g).astype(np.complex64)
        h = inverse_ad(ad_out).reshape(e - s, 256, 4, 192)
        out[s:e] = h * np.exp(
            -1j * d_te[s:e, None, None, None] * beta_s).astype(np.complex64)

    scale = np.ones(500, dtype=np.float32)
    if args.alpha_mode == "global":
        scale[:] = args.alpha_global
    elif args.alpha_mode == "file":
        scale = np.load(args.alpha_file).astype(np.float32)
    if args.power_floor_rel > 0:
        p_now = (np.abs(out) ** 2).sum(axis=(1, 2, 3))
        target_min = args.power_floor_rel * ktot
        lift = np.sqrt(np.maximum(target_min, TINY)
                       / np.maximum(p_now * scale ** 2, TINY))
        scale = scale * np.maximum(lift, 1.0).astype(np.float32)
        print(f"power floor lifted {int((lift > 1.0001).sum())} points")
    if not np.allclose(scale, 1.0):
        out = (out * scale[:, None, None, None]).astype(np.complex64)

    assert np.isfinite(out.view(np.float32)).all()
    np.save(args.output, out)

    p_out = (np.abs(out) ** 2).sum(axis=(1, 2, 3))
    p_base = (np.abs(base_h) ** 2).sum(axis=(1, 2, 3))
    report = {
        "regime_counts": {r: int((regimes == r).sum())
                          for r in ["clear", "light", "medium", "heavy"]},
        "total_power_ratio": float(p_out.sum() / p_base.sum()),
        "per_point_power_ratio_p10_p50_p90": [
            float(np.percentile(p_out / np.maximum(p_base, TINY), q))
            for q in (10, 50, 90)],
        "min_point_power": float(p_out.min()),
        "alpha_mode": args.alpha_mode,
        "power_floor_rel": args.power_floor_rel,
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print("saved", args.output)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
