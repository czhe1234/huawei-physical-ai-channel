#!/usr/bin/env python3
"""Generate v61 candidate: learned correction over train residual prior.

This is a safer learned residual block than ``generate_learned_ad_residual_block``.
The failed v60/v61-base branch tried to predict residual as a scalar times the
base component.  That often produces destructive interference with the base.

Here the model starts from the already useful v58 train-derived residual prior:

    R_prior(component) = IDW_geometry(H_train - v36_LOO) in AD space

and learns a small complex correction to that residual prior:

    R_final = ((1 - blend) * 1 + blend * MLP(token)) * R_prior
    H_pred  = v36 + gain(regime) * inverse_AD(R_final)

If the MLP is unhelpful and ``blend`` is small, this degrades toward v58 rather
than toward v36.
"""

from __future__ import annotations
from project_paths import PATHS

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from build_map_features import BS_POS
from evaluate_local_ad_paths import MP, MH, MV, N_ANT, S_SC, idw_weights, inverse_ad
from evaluate_s1style_geom_blend import geometry_cache, select_geom_neighbors
from generate_train_residual_ad_prior import beta_subcarriers, radial_phase


REGIMES = ("clear", "light", "medium", "heavy")


def regime_id(blocked: int) -> int:
    if blocked <= 0:
        return 0
    if blocked <= 25:
        return 1
    if blocked <= 100:
        return 2
    return 3


def regime_label(blocked: int) -> str:
    return REGIMES[regime_id(int(blocked))]


def gain_for_regime(blocked: int, args) -> tuple[float, str]:
    label = regime_label(int(blocked))
    gains = {
        "clear": float(args.clear_gain),
        "light": float(args.light_gain),
        "medium": float(args.medium_gain),
        "heavy": float(args.heavy_gain),
    }
    return gains[label], label


def flat_to_hvs(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(ids, dtype=np.int64)
    h = ids // (MV * S_SC)
    rem = ids % (MV * S_SC)
    v = rem // S_SC
    s = rem % S_SC
    return h.astype(np.int64), v.astype(np.int64), s.astype(np.int64)


def top_ids(score: np.ndarray, count: int) -> np.ndarray:
    flat = np.asarray(score, dtype=np.float64).reshape(-1)
    k = min(int(count), flat.size)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    ids = np.argpartition(flat, -k)[-k:]
    ids = ids[np.argsort(flat[ids])[::-1]]
    return ids.astype(np.int64)


def make_geom_cfg(args, k: int | None = None) -> dict:
    return {
        "k": int(k if k is not None else args.residual_k),
        "los_penalty": float(args.los_penalty),
        "occ_penalty": float(args.occ_penalty),
        "r_penalty": float(args.r_penalty),
        "theta_penalty": float(args.theta_penalty),
        "occ_scale": float(args.occ_scale),
        "r_scale": float(args.r_scale),
        "theta_scale_deg": float(args.theta_scale_deg),
    }


def clip_complex(z: np.ndarray, cap: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.complex64).copy()
    mag = np.abs(z)
    mask = mag > float(cap)
    if np.any(mask):
        z[mask] *= np.float32(float(cap)) / mag[mask].astype(np.float32)
    return z.astype(np.complex64)


def component_features(ids: np.ndarray) -> np.ndarray:
    h, v, s = flat_to_hvs(ids)
    hf = h.astype(np.float32)
    vf = v.astype(np.float32)
    sf = s.astype(np.float32)
    return np.column_stack(
        [
            hf / max(MH - 1, 1),
            vf / max(MV - 1, 1),
            sf / max(S_SC - 1, 1),
            np.sin(2.0 * np.pi * hf / MH),
            np.cos(2.0 * np.pi * hf / MH),
            np.sin(2.0 * np.pi * vf / MV),
            np.cos(2.0 * np.pi * vf / MV),
            np.sin(2.0 * np.pi * sf / S_SC),
            np.cos(2.0 * np.pi * sf / S_SC),
        ]
    ).astype(np.float32)


def sample_features(pos: np.ndarray, blocked: int, dist: float) -> np.ndarray:
    p = np.asarray(pos, dtype=np.float32)
    d = p - BS_POS.astype(np.float32)
    norm = max(float(np.linalg.norm(d)), 1e-6)
    u = d / norm
    one = np.zeros(4, dtype=np.float32)
    one[regime_id(int(blocked))] = 1.0
    return np.asarray(
        [
            p[0] / 300.0,
            p[1] / 300.0,
            p[2] / 50.0,
            float(dist) / 300.0,
            u[0],
            u[1],
            u[2],
            float(blocked) / 300.0,
            *one,
        ],
        dtype=np.float32,
    )


def scalar_fit(src: np.ndarray, dst: np.ndarray, ridge: float, cap: float) -> tuple[complex, float]:
    sp = float(np.sum(np.abs(src) ** 2, dtype=np.float64))
    dp = float(np.sum(np.abs(dst) ** 2, dtype=np.float64))
    if sp <= 1e-30:
        return 1.0 + 0.0j, 0.0
    num = np.sum(np.conj(src) * dst, dtype=np.complex128)
    sc = complex(num / (sp * (1.0 + float(ridge)) + 1e-30))
    sc = clip_complex(np.asarray([sc], dtype=np.complex64), cap)[0]
    pred = sc * src
    denom = float(np.sqrt(max(np.sum(np.abs(pred) ** 2, dtype=np.float64), 1e-30) * max(dp, 1e-30)))
    coh = float(np.real(np.sum(np.conj(pred) * dst, dtype=np.complex128)) / denom) if denom > 0 else 0.0
    return complex(sc), float(np.clip(coh, -1.0, 1.0))


def select_prior_components(prior_power: np.ndarray, base_power: np.ndarray, args) -> np.ndarray:
    ids = []
    ids.extend(top_ids(prior_power, args.prior_components).tolist())
    if args.base_components > 0:
        ids.extend(top_ids(base_power, args.base_components).tolist())
    ids = np.unique(np.asarray(ids, dtype=np.int64))
    if ids.size > args.components:
        pp = np.asarray(prior_power).reshape(-1)
        bp = np.asarray(base_power).reshape(-1)
        ppn = pp / max(float(np.max(pp)), 1e-30)
        bpn = bp / max(float(np.max(bp)), 1e-30)
        score = ppn + float(args.base_select_weight) * bpn
        ids = ids[np.argsort(score[ids])[::-1][: args.components]]
    return ids.astype(np.int64)


def local_prior(
    residual_cache,
    residual_power_cache,
    base_cache,
    pos_all: np.ndarray,
    blocked_all: np.ndarray,
    q_pos: np.ndarray,
    q_blocked: int,
    q_index: int | None,
    args,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = make_geom_cfg(args, args.residual_k)
    idx, _raw_d, eff_d = select_geom_neighbors(pos_all, blocked_all, q_pos, int(q_blocked), q_index, cfg)
    weights = idw_weights(eff_d, args.residual_power)
    prior_power = np.tensordot(weights.astype(np.float32), np.asarray(residual_power_cache[idx]), axes=(0, 0))
    # A weighted base power is useful only for component selection/features.
    base_power = np.zeros((MH, MV, S_SC), dtype=np.float32)
    for j, wi in zip(idx, weights):
        base_power += float(wi) * np.sum(np.abs(np.asarray(base_cache[j])) ** 2, axis=(0, 3), dtype=np.float64).astype(np.float32)
    return idx, weights, prior_power.astype(np.float32), base_power.astype(np.float32)


def build_tokens_for_point(
    q_pos: np.ndarray,
    q_blocked: int,
    q_dist: float,
    q_index: int | None,
    pos_all: np.ndarray,
    blocked_all: np.ndarray,
    base_cache,
    residual_cache,
    residual_power_cache,
    args,
    true_residual_ad: np.ndarray | None,
    test_base_ad: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, dict]:
    idx, weights, prior_power, weighted_base_power = local_prior(
        residual_cache,
        residual_power_cache,
        base_cache,
        pos_all,
        blocked_all,
        q_pos,
        int(q_blocked),
        q_index,
        args,
    )
    if q_index is not None:
        q_base_ad = np.asarray(base_cache[q_index], dtype=np.complex64)
    else:
        q_base_ad = np.asarray(test_base_ad, dtype=np.complex64)
    q_base_power = np.sum(np.abs(q_base_ad) ** 2, axis=(0, 3), dtype=np.float64)
    comp_ids = select_prior_components(prior_power, q_base_power + weighted_base_power, args)
    h, v, s = flat_to_hvs(comp_ids)
    feat = []
    y = []
    w = []
    prior_coeffs = np.zeros((len(comp_ids), MP, N_ANT), dtype=np.complex64)
    point_sample_feat = sample_features(q_pos, int(q_blocked), q_dist)
    comp_feat = component_features(comp_ids)
    for row, (hh, vv, ss) in enumerate(zip(h, v, s)):
        neigh_res = np.asarray(residual_cache[idx, :, int(hh), int(vv), :, int(ss)], dtype=np.complex64)
        prior = np.tensordot(weights.astype(np.float32), neigh_res, axes=(0, 0)).astype(np.complex64)
        prior_coeffs[row] = prior
        base_coeff = q_base_ad[:, int(hh), int(vv), :, int(ss)]
        prior_p = float(np.sum(np.abs(prior) ** 2, dtype=np.float64))
        base_p = float(np.sum(np.abs(base_coeff) ** 2, dtype=np.float64))
        phase = float(np.angle(np.sum(prior, dtype=np.complex128))) if prior_p > 1e-30 else 0.0
        base_phase = float(np.angle(np.sum(base_coeff, dtype=np.complex128))) if base_p > 1e-30 else 0.0
        prior_to_base, prior_base_coh = scalar_fit(base_coeff, prior, args.scalar_ridge, args.feature_scalar_cap)
        local = [
            np.log1p(prior_p * 1e6),
            np.log1p(float(prior_power[int(hh), int(vv), int(ss)]) * 1e6),
            np.log1p(base_p * 1e6),
            np.sqrt(prior_p / max(base_p, 1e-30)),
            np.cos(phase),
            np.sin(phase),
            np.cos(base_phase),
            np.sin(base_phase),
            float(prior_to_base.real),
            float(prior_to_base.imag),
            float(np.abs(prior_to_base)),
            float(np.angle(prior_to_base) / np.pi),
            float(prior_base_coh),
        ]
        feat.append(np.concatenate([comp_feat[row], point_sample_feat, np.asarray(local, dtype=np.float32)]))
        if true_residual_ad is not None:
            truth = true_residual_ad[:, int(hh), int(vv), :, int(ss)]
            sc, coh = scalar_fit(prior, truth, args.scalar_ridge, args.target_scalar_cap)
            y.append([float(np.real(sc)), float(np.imag(sc))])
            truth_p = float(np.sum(np.abs(truth) ** 2, dtype=np.float64))
            w.append(max(prior_p, truth_p, 1e-30) * max(coh, 0.05))
    x = np.asarray(feat, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0).astype(np.float32)
    y_arr = np.asarray(y, dtype=np.float32) if true_residual_ad is not None else None
    w_arr = np.asarray(w, dtype=np.float32) if true_residual_ad is not None else np.ones(len(comp_ids), dtype=np.float32)
    meta = {
        "component_ids": comp_ids,
        "prior_power": float(np.sum(np.abs(prior_coeffs) ** 2, dtype=np.float64)),
        "selected_prior_score": float(np.sum(prior_power.reshape(-1)[comp_ids], dtype=np.float64)),
    }
    return x, y_arr, w_arr, comp_ids, prior_coeffs, meta


class ResidualPriorMLP(torch.nn.Module):
    def __init__(self, in_dim: int, width: int, depth: int, dropout: float):
        super().__init__()
        layers = []
        dim = int(in_dim)
        for _ in range(int(depth)):
            layers.append(torch.nn.Linear(dim, int(width)))
            layers.append(torch.nn.SiLU())
            layers.append(torch.nn.LayerNorm(int(width)))
            if dropout > 0:
                layers.append(torch.nn.Dropout(float(dropout)))
            dim = int(width)
        layers.append(torch.nn.Linear(dim, 2))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def normalize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=20.0, neginf=-20.0)
    mean = x.mean(axis=0).astype(np.float32)
    std = (x.std(axis=0) + 1e-6).astype(np.float32)
    return ((x - mean[None]) / std[None]).astype(np.float32), mean, std


def train_model(x: np.ndarray, y: np.ndarray, w: np.ndarray, args):
    x_norm, mean, std = normalize(x)
    w = w.astype(np.float32)
    w = w / max(float(np.mean(w)), 1e-12)
    w = np.clip(w, 0.05, float(args.max_weight)).astype(np.float32)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = ResidualPriorMLP(x.shape[1], args.width, args.depth, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    xt = torch.from_numpy(x_norm)
    yt = torch.from_numpy(y.astype(np.float32))
    wt = torch.from_numpy(w.astype(np.float32))
    rng = np.random.default_rng(args.seed)
    losses = []
    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(len(x))
        total = 0.0
        model.train()
        for start in range(0, len(x), args.batch_size):
            ids = torch.from_numpy(perm[start:start + args.batch_size]).long()
            xb = xt[ids].to(device)
            yb = yt[ids].to(device)
            wb = wt[ids].to(device)
            pred = model(xb)
            loss_vec = torch.nn.functional.huber_loss(pred, yb, reduction="none", delta=args.huber_delta).sum(dim=1)
            loss = (loss_vec * wb).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            total += float(loss.detach().cpu()) * len(ids)
        losses.append(total / max(len(x), 1))
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(f"epoch {epoch}/{args.epochs} weighted_huber={losses[-1]:.6f}", flush=True)
    return model, mean, std, {"losses": losses, "device": str(device)}


def predict_scalars(model, mean, std, feat, args) -> np.ndarray:
    x = ((np.nan_to_num(feat.astype(np.float32), nan=0.0, posinf=20.0, neginf=-20.0) - mean[None]) / std[None]).astype(np.float32)
    device = next(model.parameters()).device
    outs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), args.infer_batch_size):
            xb = torch.from_numpy(x[start:start + args.infer_batch_size]).to(device)
            outs.append(model(xb).detach().cpu().numpy().astype(np.float32))
    out = np.concatenate(outs, axis=0) if outs else np.empty((0, 2), dtype=np.float32)
    sc = out[:, 0].astype(np.float32) + 1j * out[:, 1].astype(np.float32)
    sc = clip_complex(sc.astype(np.complex64), args.pred_scalar_cap)
    # Residual stack interpolation: blend learned correction with identity
    # scalar=1, so the block stays close to v58 when confidence is low.
    return ((1.0 - float(args.model_blend)) + float(args.model_blend) * sc).astype(np.complex64)


def guarded_correction(base_h: np.ndarray, corr: np.ndarray, args) -> tuple[np.ndarray, bool, float]:
    base_power = float(np.sum(np.abs(base_h) ** 2, dtype=np.float64))
    pred_power = float(np.sum(np.abs(base_h + corr) ** 2, dtype=np.float64))
    if args.min_pred_power_ratio <= 0 or pred_power >= args.min_pred_power_ratio * base_power:
        return corr, False, pred_power
    target = args.min_pred_power_ratio * base_power
    lo, hi = 0.0, 1.0
    best = 0.0
    best_power = base_power
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        p = float(np.sum(np.abs(base_h + np.float32(mid) * corr) ** 2, dtype=np.float64))
        if p >= target:
            best = mid
            best_power = p
            lo = mid
        else:
            hi = mid
    return (np.float32(best) * corr).astype(np.complex64), True, best_power


def build_dataset(args, train_pos, train_blocked, dist_train, base_cache, residual_cache, residual_power_cache):
    rng = np.random.default_rng(args.seed)
    ids = np.arange(len(train_pos), dtype=np.int64)
    if 0 < args.train_points < len(ids):
        ids = rng.choice(ids, size=args.train_points, replace=False).astype(np.int64)
    feats, ys, ws = [], [], []
    for c, i in enumerate(ids, start=1):
        x, y, w, _comp, _prior, _meta = build_tokens_for_point(
            train_pos[i],
            int(train_blocked[i]),
            float(dist_train[i]),
            int(i),
            train_pos,
            train_blocked,
            base_cache,
            residual_cache,
            residual_power_cache,
            args,
            np.asarray(residual_cache[i], dtype=np.complex64),
        )
        feats.append(x)
        ys.append(y)
        ws.append(w)
        if c % args.progress_every == 0 or c == len(ids):
            print(f"train prior tokens {c}/{len(ids)} examples={sum(len(f) for f in feats)}", flush=True)
    return (
        np.concatenate(feats, axis=0).astype(np.float32),
        np.concatenate(ys, axis=0).astype(np.float32),
        np.concatenate(ws, axis=0).astype(np.float32),
        ids,
    )


def generate(args):
    data = Path(args.data)
    train_pos = np.load(data / "Round1_Train_Pos.npy")
    test_pos = np.load(data / "Round1_Test_Pos.npy")
    v36 = np.load(args.v36, mmap_mode="r")
    if v36.shape != (len(test_pos), 256, 4, 192):
        raise ValueError(f"bad v36 shape {v36.shape}")
    train_blocked, test_blocked = geometry_cache(args.data, args.cache, False)
    base_cache = np.load(args.base_ad_cache, mmap_mode="r")
    residual_cache = np.load(args.residual_ad_cache, mmap_mode="r")
    residual_power_cache = np.load(args.residual_power_cache, mmap_mode="r")
    beta_s = beta_subcarriers(args.beta, args.gamma)
    dist_train = np.linalg.norm(train_pos - BS_POS[None, :], axis=1)
    dist_test = np.linalg.norm(test_pos - BS_POS[None, :], axis=1)

    x, y, w, train_ids = build_dataset(args, train_pos, train_blocked, dist_train, base_cache, residual_cache, residual_power_cache)
    print(f"dataset x={x.shape} y={y.shape} weight_mean={float(np.mean(w)):.6e}", flush=True)
    model, mean, std, train_stats = train_model(x, y, w, args)

    out = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.complex64, shape=v36.shape)
    stats = {
        "regime_counts": {k: 0 for k in REGIMES},
        "gain_sum": 0.0,
        "base_power_sum": 0.0,
        "corr_power_sum": 0.0,
        "pred_power_sum": 0.0,
        "scalar_abs_sum": 0.0,
        "scalar_count": 0,
        "point_scaled": 0,
        "power_guarded": 0,
        "selected_prior_power_sum": 0.0,
    }
    for i in range(len(test_pos)):
        base_h = np.asarray(v36[i], dtype=np.complex64)
        phase = radial_phase(float(dist_test[i]), beta_s, +1.0)
        base_derot = (base_h * phase[None, None, :]).astype(np.complex64)
        base_ad = np.fft.fft2(
            np.fft.ifft(base_derot[None], axis=-1).reshape(1, MP, MH, MV, N_ANT, S_SC),
            axes=(2, 3),
        ).astype(np.complex64)[0]
        feat, _y, _w, comp_ids, prior_coeffs, meta = build_tokens_for_point(
            test_pos[i],
            int(test_blocked[i]),
            float(dist_test[i]),
            None,
            train_pos,
            train_blocked,
            base_cache,
            residual_cache,
            residual_power_cache,
            args,
            None,
            test_base_ad=base_ad,
        )
        scalars = predict_scalars(model, mean, std, feat, args)
        gain, label = gain_for_regime(int(test_blocked[i]), args)
        corr_ad = np.zeros((1, MP, MH, MV, N_ANT, S_SC), dtype=np.complex64)
        h, v, s = flat_to_hvs(comp_ids)
        for j, (hh, vv, ss) in enumerate(zip(h, v, s)):
            corr_ad[0, :, int(hh), int(vv), :, int(ss)] = np.complex64(gain) * scalars[j] * prior_coeffs[j]
        corr_derot = inverse_ad(corr_ad)[0]
        base_power = float(np.sum(np.abs(base_h) ** 2, dtype=np.float64))
        corr_power = float(np.sum(np.abs(corr_derot) ** 2, dtype=np.float64))
        max_corr = float(args.max_corr_power_ratio) * max(base_power, 1e-30)
        if args.max_corr_power_ratio > 0 and corr_power > max_corr:
            corr_derot *= np.float32(np.sqrt(max_corr / corr_power))
            stats["point_scaled"] += 1
        phase_back = radial_phase(float(dist_test[i]), beta_s, -1.0)
        corr = (corr_derot * phase_back[None, None, :]).astype(np.complex64)
        corr, guarded, pred_power = guarded_correction(base_h, corr, args)
        pred = (base_h + corr).astype(np.complex64)
        if not (np.isfinite(pred.real).all() and np.isfinite(pred.imag).all()):
            raise FloatingPointError(f"non-finite at test point {i}")
        out[i] = pred
        stats["regime_counts"][label] += 1
        stats["gain_sum"] += float(gain)
        stats["base_power_sum"] += base_power
        stats["corr_power_sum"] += float(np.sum(np.abs(corr) ** 2, dtype=np.float64))
        stats["pred_power_sum"] += float(pred_power)
        stats["scalar_abs_sum"] += float(np.sum(np.abs(scalars), dtype=np.float64))
        stats["scalar_count"] += int(len(scalars))
        stats["selected_prior_power_sum"] += float(meta["selected_prior_score"])
        if guarded:
            stats["power_guarded"] += 1
        if (i + 1) % args.test_progress_every == 0 or i + 1 == len(test_pos):
            out.flush()
            print(f"generated {i + 1}/{len(test_pos)}", flush=True)

    stats["gain_mean"] = stats["gain_sum"] / max(len(test_pos), 1)
    stats["corr_to_base_power"] = stats["corr_power_sum"] / max(stats["base_power_sum"], 1e-30)
    stats["pred_to_base_power"] = stats["pred_power_sum"] / max(stats["base_power_sum"], 1e-30)
    stats["scalar_abs_mean"] = stats["scalar_abs_sum"] / max(stats["scalar_count"], 1)
    payload = {
        "method": "learned_prior_residual_block",
        "version": str(args.version),
        "uses_external_s1": False,
        "formula": "v36 + gain(regime) * inverse_AD(((1-blend)+blend*MLP(token))*R_prior)",
        "params": {
            "components": int(args.components),
            "prior_components": int(args.prior_components),
            "base_components": int(args.base_components),
            "train_points": int(len(train_ids)),
            "examples": int(len(x)),
            "feature_dim": int(x.shape[1]),
            "model_blend": float(args.model_blend),
            "max_corr_power_ratio": float(args.max_corr_power_ratio),
            "min_pred_power_ratio": float(args.min_pred_power_ratio),
            "gains": {
                "clear": float(args.clear_gain),
                "light": float(args.light_gain),
                "medium": float(args.medium_gain),
                "heavy": float(args.heavy_gain),
            },
        },
        "train": train_stats,
        "stats": stats,
        "output": args.output,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.model_out:
        torch.save({"model": model.state_dict(), "mean": mean, "std": std, "args": vars(args), "payload": payload}, args.model_out)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--cache", default=str(PATHS.artifacts / "shared_features/height_geometry_cache.npz"))
    parser.add_argument("--v36", default=str(PATHS.artifacts / "external/tmp_blend_inspect/v36.npy"))
    parser.add_argument("--base-ad-cache", default=str(PATHS.artifacts / "shared_features/train_v36_loo_base_derot_ad.npy"))
    parser.add_argument("--residual-ad-cache", default=str(PATHS.artifacts / "shared_features/train_v36_loo_residual_derot_ad.npy"))
    parser.add_argument("--residual-power-cache", default=str(PATHS.artifacts / "shared_features/train_v36_loo_residual_derot_ad_power.npy"))
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--out", default=str(PATHS.version_report_dir(61) / "learned_prior_residual_block_v61_generate.json"))
    parser.add_argument("--model-out", default=str(PATHS.artifacts / "checkpoints/learned_prior_residual_block_v61.pt"))
    parser.add_argument("--version", default="v61")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--prior-components", type=int, default=16)
    parser.add_argument("--base-components", type=int, default=0)
    parser.add_argument("--base-select-weight", type=float, default=0.25)
    parser.add_argument("--train-points", type=int, default=2000)
    parser.add_argument("--residual-k", type=int, default=32)
    parser.add_argument("--residual-power", type=float, default=2.0)
    parser.add_argument("--scalar-ridge", type=float, default=0.05)
    parser.add_argument("--feature-scalar-cap", type=float, default=2.0)
    parser.add_argument("--target-scalar-cap", type=float, default=2.0)
    parser.add_argument("--pred-scalar-cap", type=float, default=1.5)
    parser.add_argument("--model-blend", type=float, default=0.35)
    parser.add_argument("--max-corr-power-ratio", type=float, default=0.22)
    parser.add_argument("--min-pred-power-ratio", type=float, default=1.0)
    parser.add_argument("--clear-gain", type=float, default=0.25)
    parser.add_argument("--light-gain", type=float, default=0.25)
    parser.add_argument("--medium-gain", type=float, default=0.75)
    parser.add_argument("--heavy-gain", type=float, default=1.50)
    parser.add_argument("--los-penalty", type=float, default=1.0)
    parser.add_argument("--occ-penalty", type=float, default=0.5)
    parser.add_argument("--r-penalty", type=float, default=0.0)
    parser.add_argument("--theta-penalty", type=float, default=0.0)
    parser.add_argument("--occ-scale", type=float, default=80.0)
    parser.add_argument("--r-scale", type=float, default=80.0)
    parser.add_argument("--theta-scale-deg", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--infer-batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-delta", type=float, default=0.15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-weight", type=float, default=30.0)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--test-progress-every", type=int, default=5)
    return parser.parse_args()


def main():
    generate(parse_args())


if __name__ == "__main__":
    main()
