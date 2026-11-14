#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_frequency_dependent_phase import S_SC
from evaluate_local_ad_paths import BS_POS, idw_weights
from evaluate_physical_local_linear import affine_coefficients, predict_affine, steering_phase_query
from generate_geometric_steering_blend import unit_vectors


EPS = 1e-30


def power_mean(h):
    return float(np.mean(np.abs(h) ** 2, dtype=np.float64))


def complex_cosine(a, b):
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    num = abs(np.vdot(a_flat, b_flat))
    den = np.sqrt(float(np.vdot(a_flat, a_flat).real) * float(np.vdot(b_flat, b_flat).real))
    return float(num / max(den, EPS))


def coeff_stats(coeff):
    coeff64 = coeff.astype(np.float64)
    neg_mass = float(np.sum(np.maximum(-coeff64, 0.0)))
    l1 = float(np.sum(np.abs(coeff64)))
    return neg_mass, l1, float(neg_mass / max(l1, EPS))


def predict_steered_idw(neigh, distances, dist_delta, u_all, idx, qi, k, power, beta, gamma, alpha, steering_scale):
    idx = idx[:k]
    distances = distances[:k]
    neigh = neigh[:k]
    dist_delta = dist_delta[:k]
    weights = idw_weights(distances, power)
    sub = np.linspace(-0.5, 0.5, S_SC, dtype=np.float32)
    beta_s = (beta + gamma * sub).astype(np.float32)
    radial = np.exp(1j * dist_delta[:, None] * beta_s[None, :]).astype(np.complex64)
    base_terms = radial[:, None, None, :] * neigh
    base = np.sum(weights[:, None, None, None] * base_terms, axis=0)
    steer = steering_phase_query(u_all, idx, qi, steering_scale)
    geo = np.sum(weights[:, None, None, None] * steer[:, :, None, None] * base_terms, axis=0)
    return ((1.0 - alpha) * base + alpha * geo).astype(np.complex64)


def predict_derotated_idw(neigh, distances, dist_delta, k, power, beta, gamma):
    neigh = neigh[:k]
    distances = distances[:k]
    dist_delta = dist_delta[:k]
    weights = idw_weights(distances, power)
    sub = np.linspace(-0.5, 0.5, S_SC, dtype=np.float32)
    beta_s = (beta + gamma * sub).astype(np.float32)
    radial = np.exp(1j * dist_delta[:, None] * beta_s[None, :]).astype(np.complex64)
    return np.sum(weights[:, None, None, None] * radial[:, None, None, :] * neigh, axis=0).astype(np.complex64)


def diagnose(pred_aff, pred_small, pred_idw, coeff):
    p_aff = power_mean(pred_aff)
    p_small = power_mean(pred_small)
    p_idw = power_mean(pred_idw)
    neg_mass, l1, neg_frac = coeff_stats(coeff)
    return {
        "p_aff": p_aff,
        "p_small": p_small,
        "p_idw": p_idw,
        "ratio_small": float(p_aff / max(p_small, EPS)),
        "ratio_idw": float(p_aff / max(p_idw, EPS)),
        "neg_mass": neg_mass,
        "l1": l1,
        "neg_frac": neg_frac,
        "cos_small": complex_cosine(pred_aff, pred_small),
    }


def apply_policy(pred_aff, pred_small, pred_idw, diag, args):
    flagged = (
        diag["cos_small"] >= args.cos_thresh
        and diag["ratio_small"] <= args.ratio_small_thresh
        and diag["ratio_idw"] <= args.ratio_idw_thresh
        and diag["neg_frac"] >= args.neg_frac_thresh
    )
    if not flagged:
        return pred_aff, "affine", 1.0

    severe = (
        diag["ratio_small"] <= args.severe_ratio_small_thresh
        or diag["neg_frac"] >= args.severe_neg_frac_thresh
    )
    if severe and args.mode == "gain_then_small":
        return pred_small, "fallback_small", 1.0
    if severe and args.mode == "blend_small":
        pred = args.blend_affine * pred_aff + (1.0 - args.blend_affine) * pred_small
        return pred.astype(np.complex64), "blend_small", 1.0

    ref_power = max(diag["p_small"], diag["p_idw"])
    gain = float(min(args.gain_cap, np.sqrt(ref_power / max(diag["p_aff"], EPS))))
    return (gain * pred_aff).astype(np.complex64), "gain", gain


def predict_v11(neigh, distances, rel_xy, dist_delta, u_all, idx, qi, args):
    coeff = affine_coefficients(rel_xy, distances[:args.k], args.power, args.ridge)
    pred_aff = predict_affine(
        neigh[:args.k],
        coeff,
        dist_delta[:args.k],
        u_all,
        idx[:args.k],
        qi,
        args.beta,
        args.gamma,
        args.alpha,
        args.steering_scale,
    )
    pred_small = predict_steered_idw(
        neigh,
        distances,
        dist_delta,
        u_all,
        idx,
        qi,
        args.fallback_k,
        args.power,
        args.beta,
        args.gamma,
        args.alpha,
        args.steering_scale,
    )
    pred_idw = predict_derotated_idw(neigh, distances, dist_delta, args.idw_k, args.power, args.beta, args.gamma)
    diag = diagnose(pred_aff, pred_small, pred_idw, coeff)
    pred, branch, gain = apply_policy(pred_aff, pred_small, pred_idw, diag, args)
    diag["branch"] = branch
    diag["gain"] = float(gain)
    return pred.astype(np.complex64), diag


def summarize_diags(diags):
    branches = {}
    for diag in diags:
        branches[diag["branch"]] = branches.get(diag["branch"], 0) + 1
    summary = {"count": len(diags), "branches": branches}
    for key in ["ratio_small", "ratio_idw", "neg_frac", "cos_small", "gain", "p_aff", "p_small", "p_idw"]:
        values = np.asarray([d[key] for d in diags], dtype=np.float64)
        summary[key] = {
            "min": float(np.min(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q50": float(np.quantile(values, 0.50)),
            "q95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values)),
        }
    return summary


def validate_shapes(pos, test_pos, train_h):
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"unexpected train position shape {pos.shape}")
    if test_pos.ndim != 2 or test_pos.shape[1] != 3:
        raise ValueError(f"unexpected test position shape {test_pos.shape}")
    if train_h.ndim != 4 or train_h.shape[1:] != (256, 4, 192):
        raise ValueError(f"unexpected train channel shape {train_h.shape}")


def generate(args):
    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))
    train_h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    validate_shapes(pos, test_pos, train_h)
    max_k = max(args.k, args.fallback_k, args.idw_k)

    pos_all = np.concatenate([pos, test_pos], axis=0)
    dist_all = np.linalg.norm(pos_all - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos_all)
    test_offset = len(pos)

    tree = cKDTree(pos[:, :2])
    distances, neighbor_idx = tree.query(test_pos[:, :2], k=max_k, workers=-1)
    if max_k == 1:
        distances = distances[:, None]
        neighbor_idx = neighbor_idx[:, None]

    out = np.lib.format.open_memmap(
        args.output,
        mode="w+",
        dtype=np.complex64,
        shape=(len(test_pos), train_h.shape[1], train_h.shape[2], train_h.shape[3]),
    )
    diags = []
    for start in range(0, len(test_pos), args.batch):
        end = min(start + args.batch, len(test_pos))
        for i in range(start, end):
            idx = neighbor_idx[i]
            rel_xy = pos[idx[:args.k], :2] - test_pos[i, :2]
            neigh = np.asarray(train_h[idx])
            qi = test_offset + i
            pred, diag = predict_v11(neigh, distances[i], rel_xy, dist_all[idx] - dist_all[qi], u_all, idx, qi, args)
            out[i] = pred
            diags.append(diag)
        block = np.asarray(out[start:end])
        if not (np.isfinite(block.real).all() and np.isfinite(block.imag).all()):
            raise FloatingPointError(f"non-finite values in generated block {start}:{end}")
        out.flush()
        print(f"generated {end}/{len(test_pos)}", flush=True)

    return {
        "method": "physical_local_linear_v11_safe_gain_fallback",
        "k": int(args.k),
        "power": float(args.power),
        "beta": float(args.beta),
        "gamma": float(args.gamma),
        "alpha": float(args.alpha),
        "ridge": float(args.ridge),
        "steering_scale": float(args.steering_scale),
        "fallback_k": int(args.fallback_k),
        "idw_k": int(args.idw_k),
        "ratio_small_thresh": float(args.ratio_small_thresh),
        "ratio_idw_thresh": float(args.ratio_idw_thresh),
        "neg_frac_thresh": float(args.neg_frac_thresh),
        "cos_thresh": float(args.cos_thresh),
        "gain_cap": float(args.gain_cap),
        "mode": args.mode,
        "output": args.output,
        "summary": summarize_diags(diags),
    }


def add_common_args(parser):
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=0.3)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--fallback-k", type=int, default=4)
    parser.add_argument("--idw-k", type=int, default=16)
    parser.add_argument("--ratio-small-thresh", type=float, default=0.50)
    parser.add_argument("--ratio-idw-thresh", type=float, default=0.60)
    parser.add_argument("--neg-frac-thresh", type=float, default=0.15)
    parser.add_argument("--cos-thresh", type=float, default=0.99)
    parser.add_argument("--gain-cap", type=float, default=1.25)
    parser.add_argument("--severe-ratio-small-thresh", type=float, default=0.30)
    parser.add_argument("--severe-neg-frac-thresh", type=float, default=0.30)
    parser.add_argument("--mode", choices=["gain_only", "gain_then_small", "blend_small"], default="gain_only")
    parser.add_argument("--blend-affine", type=float, default=0.25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--output", default=str(PATHS.temporary / "Round1_Test_Channel.npy"))
    parser.add_argument("--out", default=str(PATHS.version_report_dir(11) / "physical_local_linear_v11_generate.json"))
    parser.add_argument("--batch", type=int, default=10)
    add_common_args(parser)
    args = parser.parse_args()
    result = generate(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
