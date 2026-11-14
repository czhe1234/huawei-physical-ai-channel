# （本改动由Claude fable5）
"""Generate Round1_Test_Channel.npy with marginal power-field calibration.

Pipeline:
 1. Start from a base complex prediction (default: the v55 pipeline output),
    which supplies the phase structure.
 2. Derotate each test point by exp(+j*beta_s*d_BS) and transform to the
    angle-delay (AD) domain (P,H,V,N,tau layout).
 3. Predict each test point's true beam-power shape (16x8) and delay-power
    shape (192) from training data: sharp geometry-gated kNN shape transfer
    and/or the trained neural marginal field.
 4. Impose the predicted shapes on the base prediction as a rank-1
    multiplicative gain in the AD domain (partial calibration exponent eta).
 5. Optional per-point amplitude scale (NMSE calibration) and power floor.
 6. Inverse-transform, re-rotate, save complex64 (500,256,4,192).
"""

import argparse
import json

import numpy as np

from project_paths import PATHS
from powerfield_common import (ASSETS, TINY, assemble_matrix,
                               build_collapsed_caches, build_features,
                               regime_of, sharp_marginals)

MP, MH, MV, N_ANT, S_SC = 2, 16, 8, 4, 192


def beta_subcarriers(beta: float, gamma: float) -> np.ndarray:
    sub = np.linspace(-0.5, 0.5, S_SC, dtype=np.float32)
    return (float(beta) + float(gamma) * sub).astype(np.float32)


def ad_transform(h):
    cir = np.fft.ifft(h, axis=-1)
    arr = cir.reshape(h.shape[0], MP, MH, MV, N_ANT, S_SC)
    return np.fft.fft2(arr, axes=(2, 3)).astype(np.complex64)


def inverse_ad(ad):
    arr = np.fft.ifft2(ad, axes=(2, 3))
    cir = arr.reshape(ad.shape[0], MP * MH * MV, N_ANT, S_SC)
    return np.fft.fft(cir, axis=-1).astype(np.complex64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(PATHS.analysis / "probes/Round1_Test_Channel.npy"),
                    help="base complex prediction (v55 pipeline output)")
    ap.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    ap.add_argument("--shape-source", choices=["model", "sharp", "blend"],
                    default="model")
    ap.add_argument("--model", default=ASSETS + "/marginal_field_final.pt")
    ap.add_argument("--sharp-k", type=int, default=8)
    ap.add_argument("--sharp-p", type=float, default=3.0)
    ap.add_argument("--blend-w", type=float, default=0.5,
                    help="weight of model shape in blend mode")
    ap.add_argument("--eta-beam", type=float, default=0.5)
    ap.add_argument("--eta-delay", type=float, default=0.5)
    ap.add_argument("--eta-regime", default="",
                    help="optional per-regime eta override 'clear:0.4,light:0.4,medium:0.5,heavy:0.6' applied to both beam and delay")
    ap.add_argument("--gain-clip", type=float, default=8.0)
    ap.add_argument("--alpha-global", type=float, default=1.0)
    ap.add_argument("--alpha-per-point", default="",
                    help="npy file with 500 per-point scales (optional)")
    ap.add_argument("--power-floor-rel", type=float, default=0.0,
                    help="floor each point's total power at rel * knn-interpolated total power")
    ap.add_argument("--beta", type=float, default=140.3)
    ap.add_argument("--gamma", type=float, default=0.10)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--report", default=ASSETS + "/powerfield_submission_report.json")
    args = ap.parse_args()

    bs_pos = np.array([50.0, 0.0, 25.0])
    pos_train = np.load(str(PATHS.round1 / "Round1_Train_Pos.npy"))
    pos_test = np.load(str(PATHS.round1 / "Round1_Test_Pos.npy"))
    geo = np.load(ASSETS + "/height_geometry_cache.npz")
    blocked_tr = geo["train_blocked"].astype(np.float64)
    blocked_te = geo["test_blocked"].astype(np.float64)

    coll, _ = build_collapsed_caches()
    T_tr = coll.sum((1, 2, 3))
    Bn_tr = coll.sum(-1).reshape(2000, 128) / np.maximum(T_tr[:, None], TINY)
    Dn_tr = coll.sum((1, 2)) / np.maximum(T_tr[:, None], TINY)

    base_h = np.load(args.base)
    assert base_h.shape == (500, 256, 4, 192) and base_h.dtype == np.complex64
    beta_s = beta_subcarriers(args.beta, args.gamma)
    d_te = np.linalg.norm(pos_test - bs_pos, axis=1)

    # --- base AD + collapsed marginals ---
    print("transforming base to AD domain ...")
    base_ad = np.empty((500, MP, MH, MV, N_ANT, S_SC), dtype=np.complex64)
    for s in range(0, 500, args.batch):
        e = min(s + args.batch, 500)
        hb = base_h[s:e] * np.exp(1j * d_te[s:e, None, None, None] * beta_s)[
            :, :, :, :].astype(np.complex64)
        base_ad[s:e] = ad_transform(hb)
    bcoll_te = (np.abs(base_ad) ** 2).sum(axis=(1, 4)).astype(np.float32)
    B_b = bcoll_te.sum(-1).reshape(500, 16, 8)
    D_b = bcoll_te.sum((1, 2))
    T_b = bcoll_te.sum((1, 2, 3))

    # --- predicted shapes ---
    print("predicting marginal shapes ...")
    sharp_b = np.empty((500, 128), dtype=np.float32)
    sharp_d = np.empty((500, 192), dtype=np.float32)
    for i in range(500):
        sb, sd, *_ = sharp_marginals(Bn_tr, Dn_tr, T_tr, pos_train, blocked_tr,
                                     pos_test[i, :2], blocked_te[i],
                                     k=args.sharp_k, p=args.sharp_p)
        sharp_b[i], sharp_d[i] = sb, sd

    pred_beam, pred_delay = sharp_b, sharp_d
    knn_total = None
    if args.shape_source in ("model", "blend"):
        import torch
        ck = torch.load(args.model, map_location="cpu", weights_only=False)
        from train_marginal_gain_field import MarginalField
        feats = build_features(pos_test, blocked_te, coll, pos_train, blocked_tr,
                               B_b, D_b, T_b, exclude_self=False, k=32)
        knn_total = np.exp(feats["knn_total"][:, 0])
        X = assemble_matrix(feats, ck["keys"])
        Xn = (X - ck["mu"]) / ck["sd"]
        xb = torch.tensor(Xn, dtype=torch.float32)
        acc_b = np.zeros((500, 128))
        acc_d = np.zeros((500, 192))
        for sd_ in ck["state_dicts"]:
            model = MarginalField(ck["in_dim"], ck["width"], ck["depth"],
                                  ck["dropout"])
            model.load_state_dict(sd_)
            model.eval()
            with torch.no_grad():
                lb, ld, lt = model(xb)
                acc_b += torch.softmax(lb, 1).numpy()
                acc_d += torch.softmax(ld, 1).numpy()
        mb = (acc_b / len(ck["state_dicts"])).astype(np.float32)
        md = (acc_d / len(ck["state_dicts"])).astype(np.float32)
        if args.shape_source == "model":
            pred_beam, pred_delay = mb, md
        else:
            w = args.blend_w
            pred_beam = (w * mb + (1 - w) * sharp_b).astype(np.float32)
            pred_delay = (w * md + (1 - w) * sharp_d).astype(np.float32)

    # --- per-regime eta ---
    regimes = regime_of(blocked_te)
    eta_b = np.full(500, args.eta_beam, dtype=np.float32)
    eta_d = np.full(500, args.eta_delay, dtype=np.float32)
    if args.eta_regime:
        table = dict(kv.split(":") for kv in args.eta_regime.split(","))
        for i, r in enumerate(regimes):
            if r in table:
                eta_b[i] = eta_d[i] = float(table[r])

    # --- apply rank-1 gains ---
    print("applying gains ...")
    delta_b = 1e-4 / 128.0
    delta_d = 1e-4 / 192.0
    bb = B_b.reshape(500, 128) / np.maximum(T_b[:, None], TINY)
    db = D_b / np.maximum(T_b[:, None], TINY)
    g_b = ((pred_beam + delta_b) / (bb + delta_b)) ** (0.5 * eta_b[:, None])
    g_d = ((pred_delay + delta_d) / (db + delta_d)) ** (0.5 * eta_d[:, None])
    g = (g_b.reshape(500, 16, 8, 1) * g_d.reshape(500, 1, 1, 192))
    g = np.clip(g, 1.0 / args.gain_clip, args.gain_clip).astype(np.float32)
    out_ad = base_ad * g[:, None, :, :, None, :]

    # --- per-point amplitude policy ---
    scale = np.full(500, args.alpha_global, dtype=np.float32)
    if args.alpha_per_point:
        scale = scale * np.load(args.alpha_per_point).astype(np.float32)
    p_now = (np.abs(out_ad) ** 2).sum(axis=(1, 2, 3, 4, 5))
    if args.power_floor_rel > 0 and knn_total is None:
        feats_total = np.empty(500, dtype=np.float32)
        from powerfield_common import knn_power_marginals
        for i in range(500):
            _, _, t, *_ = knn_power_marginals(coll, pos_train, blocked_tr,
                                              pos_test[i, :2], blocked_te[i], k=32)
            feats_total[i] = t
        knn_total = feats_total
    if args.power_floor_rel > 0:
        target_min = args.power_floor_rel * knn_total
        lift = np.sqrt(np.maximum(target_min, TINY) / np.maximum(p_now * scale**2, TINY))
        lift = np.maximum(lift, 1.0)
        scale = scale * lift.astype(np.float32)
        print(f"power floor lifted {int((lift > 1.0001).sum())} points")
    out_ad *= scale[:, None, None, None, None, None]

    # --- back to H domain ---
    print("inverse transform ...")
    out = np.empty_like(base_h)
    for s in range(0, 500, args.batch):
        e = min(s + args.batch, 500)
        h = inverse_ad(out_ad[s:e])
        h = h.reshape(e - s, 256, 4, 192)
        out[s:e] = h * np.exp(-1j * d_te[s:e, None, None, None] * beta_s).astype(np.complex64)

    assert np.isfinite(out.view(np.float32)).all()
    np.save(args.output, out)
    report = {
        "config": {k: v for k, v in vars(args).items()},
        "regime_counts": {r: int((regimes == r).sum()) for r in
                          ["clear", "light", "medium", "heavy"]},
        "total_power_ratio_vs_base": float(
            (np.abs(out) ** 2).sum() / (np.abs(base_h) ** 2).sum()),
        "per_point_power_p10_ratio": float(np.percentile(
            (np.abs(out) ** 2).sum(axis=(1, 2, 3))
            / np.maximum((np.abs(base_h) ** 2).sum(axis=(1, 2, 3)), TINY), 10)),
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print("saved", args.output)
    print(json.dumps(report["regime_counts"]))


if __name__ == "__main__":
    main()
