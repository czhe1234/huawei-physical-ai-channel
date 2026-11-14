#!/usr/bin/env python3
from project_paths import PATHS
"""Train a conservative scalar expert for geometry-assigned AD components."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from evaluate_assigned_component_oracle import (
    assign_dynamic,
    assignment_mask,
    scalar_fit,
    top_components_from_ad,
)
from evaluate_derotated_loo_grid import choose_val_indices
from evaluate_local_ad_paths import (
    BS_POS,
    MH,
    MV,
    S_SC,
    ad_transform,
    flat_components,
    idw_weights,
    inverse_ad,
    parse_ints,
)
from evaluate_physical_local_linear import affine_coefficients, predict_affine
from generate_geometric_steering_blend import unit_vectors
from spectrum_projection import MetricAccumulator


def parse_floats(text):
    return [float(x) for x in str(text).split(",") if x.strip()]


def parse_words(text):
    return [x for x in str(text).split(",") if x.strip()]


def id_features(hbin, vbin, sbin):
    h = hbin.astype(np.float32)
    v = vbin.astype(np.float32)
    s = sbin.astype(np.float32)
    return np.column_stack([
        h / max(MH - 1, 1),
        v / max(MV - 1, 1),
        s / max(S_SC - 1, 1),
        np.sin(2.0 * np.pi * h / MH),
        np.cos(2.0 * np.pi * h / MH),
        np.sin(2.0 * np.pi * v / MV),
        np.cos(2.0 * np.pi * v / MV),
        np.sin(2.0 * np.pi * s / S_SC),
        np.cos(2.0 * np.pi * s / S_SC),
    ]).astype(np.float32)


def component_coeff(ad_one, component_id):
    h, v, s = flat_components(np.asarray([component_id], dtype=np.int64))
    return ad_one[:, h[0], v[0], :, s[0]]


def set_component_coeff(ad_one, component_id, coeff):
    h, v, s = flat_components(np.asarray([component_id], dtype=np.int64))
    ad_one[:, h[0], v[0], :, s[0]] = coeff


def clip_scalar_log_phase(scalar, gain_cap, phase_cap):
    log_gain = np.log(np.maximum(np.abs(scalar), 1e-12))
    phase = np.angle(scalar)
    log_gain = np.clip(log_gain, -np.log(gain_cap), np.log(gain_cap))
    phase = np.clip(phase, -phase_cap, phase_cap)
    return log_gain.astype(np.float32), phase.astype(np.float32)


def scalar_from_log_phase(log_gain, phase, gain_cap, phase_cap):
    log_gain = np.clip(log_gain, -np.log(gain_cap), np.log(gain_cap))
    phase = np.clip(phase, -phase_cap, phase_cap)
    return (np.exp(log_gain) * np.exp(1j * phase)).astype(np.complex64)


def clip_complex_scalar(scalar, gain_cap):
    scalar = scalar.astype(np.complex64, copy=True)
    mag = np.abs(scalar)
    good = mag > 1e-12
    scalar[good] = (np.minimum(mag[good], gain_cap) * scalar[good] / mag[good]).astype(np.complex64)
    scalar[~good] = np.complex64(1.0 + 0.0j)
    return scalar.astype(np.complex64)


def baseline_for_queries(pos, channels, query_idx, args, train_pool=None):
    if train_pool is None:
        train_pool = np.arange(len(pos), dtype=np.int64)
    tree = cKDTree(pos[train_pool, :2])
    d_all, local_all = tree.query(pos[query_idx, :2], k=args.k + 1, workers=-1)
    if d_all.ndim == 1:
        d_all = d_all[None, :]
        local_all = local_all[None, :]
    neighbor_idx = np.empty((len(query_idx), args.k), dtype=np.int64)
    distances = np.empty((len(query_idx), args.k), dtype=np.float64)
    for i, qi in enumerate(query_idx):
        cand = train_pool[local_all[i]]
        keep = cand != qi
        neighbor_idx[i] = cand[keep][:args.k]
        distances[i] = d_all[i, keep][:args.k]

    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)
    preds = []
    coeffs = []
    for row, qi in enumerate(query_idx):
        idx = neighbor_idx[row]
        neigh = np.asarray(channels[idx])
        rel_xy = pos[idx, :2] - pos[qi, :2]
        coeff = affine_coefficients(rel_xy, distances[row], args.power, args.ridge)
        pred = predict_affine(
            neigh,
            coeff,
            dist_all[idx] - dist_all[qi],
            u_all,
            idx,
            qi,
            args.beta,
            args.gamma,
            args.alpha,
            args.steering_scale,
        )
        preds.append(pred.astype(np.complex64))
        coeffs.append(coeff.astype(np.float32))
        if (row + 1) % 50 == 0 or row + 1 == len(query_idx):
            print(f"baseline {row + 1}/{len(query_idx)}", flush=True)
    return np.asarray(preds), neighbor_idx, distances, np.asarray(coeffs)


def build_features_for_row(
    pos,
    path_cache,
    assign_cache,
    qi,
    base_ad_one,
    truth_ad_one,
    neigh_ad,
    neigh_distances,
    component_ids,
    assign,
    mask,
    args,
):
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if not np.any(mask):
        return None
    comp_ids = component_ids[mask]
    hbin, vbin, sbin = flat_components(comp_ids.astype(np.int64))
    best_path = assign["best_path"][0][mask].astype(np.int64)
    row = np.arange(comp_ids.shape[0])

    path_prefix = "train"
    path_type = path_cache[f"{path_prefix}_path_type"][qi, best_path].astype(np.float32)
    plane_id = path_cache[f"{path_prefix}_plane_id"][qi, best_path].astype(np.float32)
    path_length = path_cache[f"{path_prefix}_path_length"][qi, best_path].astype(np.float32)
    delay_bin = path_cache[f"{path_prefix}_delay_bin"][qi, best_path].astype(np.float32)
    path_score = path_cache[f"{path_prefix}_score"][qi, best_path].astype(np.float32)
    tx_dir = path_cache[f"{path_prefix}_tx_dir"][qi, best_path].astype(np.float32)
    rx_dir = path_cache[f"{path_prefix}_rx_dir"][qi, best_path].astype(np.float32)
    excess = path_cache[f"{path_prefix}_excess_length"][qi, best_path].astype(np.float32)
    edge = path_cache[f"{path_prefix}_edge_margin"][qi, best_path].astype(np.float32)
    incidence = path_cache[f"{path_prefix}_incidence_cos"][qi, best_path].astype(np.float32)
    reflect = path_cache[f"{path_prefix}_reflect_point"][qi, best_path].astype(np.float32)
    virtual = path_cache[f"{path_prefix}_virtual_source"][qi, best_path].astype(np.float32)

    q = pos[qi].astype(np.float32)
    qdist = float(np.linalg.norm(q - BS_POS.astype(np.float32)))
    qdir = (q - BS_POS.astype(np.float32)) / max(qdist, 1e-6)
    qfeat = np.repeat(np.asarray([q[0] / 300.0, q[1] / 300.0, q[2] / 50.0, qdist / 300.0, *qdir], dtype=np.float32)[None], len(comp_ids), axis=0)

    base_coeff = np.stack([component_coeff(base_ad_one, int(c)) for c in comp_ids], axis=0)
    truth_coeff = np.stack([component_coeff(truth_ad_one, int(c)) for c in comp_ids], axis=0)
    scalar = np.asarray([scalar_fit(base_coeff[i], truth_coeff[i]) for i in range(len(comp_ids))], dtype=np.complex64)
    scalar_clipped = clip_complex_scalar(scalar, args.clip_gain)
    log_gain, phase = clip_scalar_log_phase(scalar_clipped, args.clip_gain, args.clip_phase)

    base_flat = base_coeff.reshape(len(comp_ids), -1)
    truth_flat = truth_coeff.reshape(len(comp_ids), -1)
    base_power = np.sum(np.abs(base_flat) ** 2, axis=1).astype(np.float32)
    truth_power = np.sum(np.abs(truth_flat) ** 2, axis=1).astype(np.float32)
    coeff_sum = np.sum(base_flat, axis=1)
    coeff_phase = np.angle(coeff_sum).astype(np.float32)
    coeff_feat = np.column_stack([
        np.log1p(base_power),
        np.sqrt(base_power / np.maximum(np.mean(base_power), 1e-12)),
        np.cos(coeff_phase),
        np.sin(coeff_phase),
    ]).astype(np.float32)

    local_feat = []
    weights = idw_weights(neigh_distances, args.power)
    for j, comp in enumerate(comp_ids):
        neigh_coeff = np.stack([component_coeff(neigh_ad[n], int(comp)) for n in range(neigh_ad.shape[0])], axis=0)
        near = neigh_coeff[0]
        mixed = np.tensordot(weights.astype(np.float32), neigh_coeff, axes=(0, 0)).astype(np.complex64)
        near_scalar = scalar_fit(base_coeff[j], near)
        mixed_scalar = scalar_fit(base_coeff[j], mixed)
        near_power = float(np.sum(np.abs(near) ** 2, dtype=np.float64))
        mixed_power = float(np.sum(np.abs(mixed) ** 2, dtype=np.float64))
        bp = float(base_power[j])
        local_feat.append([
            mixed_scalar.real,
            mixed_scalar.imag,
            np.clip(np.abs(mixed_scalar), 0.0, 4.0),
            np.angle(mixed_scalar) / np.pi,
            near_scalar.real,
            near_scalar.imag,
            np.clip(np.abs(near_scalar), 0.0, 4.0),
            np.angle(near_scalar) / np.pi,
            np.log1p(mixed_power),
            np.log1p(near_power),
            np.clip(mixed_power / max(bp, 1e-12), 0.0, 8.0),
            np.clip(near_power / max(bp, 1e-12), 0.0, 8.0),
        ])
    local_feat = np.asarray(local_feat, dtype=np.float32)

    residual_feat = np.column_stack([
        assign["delay_err"][0][mask] / 4.0,
        assign["h_err"][0][mask] / 4.0,
        assign["v_err"][0][mask] / 4.0,
        np.sqrt(np.maximum(assign["score"][0][mask], 0.0)) / 4.0,
    ]).astype(np.float32)

    geom_phase = np.column_stack([
        np.sin(args.beta * path_length),
        np.cos(args.beta * path_length),
        np.sin(args.beta * excess),
        np.cos(args.beta * excess),
    ]).astype(np.float32)

    path_feat = np.column_stack([
        path_type,
        np.clip(plane_id, -1, 200) / 200.0,
        path_length / 300.0,
        delay_bin / S_SC,
        path_score / 10.0,
        excess / 100.0,
        edge / 50.0,
        incidence,
        tx_dir,
        rx_dir,
        reflect[:, 0] / 300.0,
        reflect[:, 1] / 300.0,
        reflect[:, 2] / 50.0,
        virtual[:, 0] / 300.0,
        virtual[:, 1] / 300.0,
        virtual[:, 2] / 50.0,
    ]).astype(np.float32)

    feat = np.concatenate([
        id_features(hbin, vbin, sbin),
        qfeat,
        residual_feat,
        geom_phase,
        coeff_feat,
        local_feat,
        path_feat,
    ], axis=1).astype(np.float32)
    feat = np.nan_to_num(feat, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
    if args.target_mode == "realimag":
        target = np.column_stack([scalar_clipped.real, scalar_clipped.imag]).astype(np.float32)
    elif args.target_mode == "residual_realimag":
        target = np.column_stack([scalar_clipped.real - 1.0, scalar_clipped.imag]).astype(np.float32)
    else:
        target = np.column_stack([log_gain, phase / args.clip_phase]).astype(np.float32)
    target = np.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
    if args.weight_mode == "base":
        weight = np.maximum(base_power, 1e-12).astype(np.float32)
    elif args.weight_mode == "truth":
        weight = np.maximum(truth_power, 1e-12).astype(np.float32)
    elif args.weight_mode == "sqrt_truth":
        weight = np.sqrt(np.maximum(truth_power, 1e-12)).astype(np.float32)
    else:
        weight = np.ones_like(base_power, dtype=np.float32)
    weight = np.nan_to_num(weight, nan=1e-6, posinf=1.0, neginf=1e-6).astype(np.float32)
    return feat, target, weight, comp_ids, scalar, base_power


def make_component_assignment(base_ad, path_cache, assign_cache, query_idx, count):
    ids, _ = top_components_from_ad(base_ad, count)
    path_delay = assign_cache["path_delay_bin"][query_idx].astype(np.float32)
    path_h = assign_cache["path_h_bin"][query_idx].astype(np.float32)
    path_v = assign_cache["path_v_bin"][query_idx].astype(np.float32)
    path_valid = assign_cache["path_valid"][query_idx]
    return ids, assign_dynamic(ids, path_delay, path_h, path_v, path_valid, assign_cache["weights"].astype(np.float32))


class ScalarExpert(torch.nn.Module):
    def __init__(self, in_dim, width, depth, dropout):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(depth):
            layers.append(torch.nn.Linear(dim, width))
            layers.append(torch.nn.SiLU())
            layers.append(torch.nn.LayerNorm(width))
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            dim = width
        layers.append(torch.nn.Linear(dim, 2))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def normalize(x):
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=10.0, neginf=-10.0)
    mean = x.mean(axis=0).astype(np.float32)
    std = (x.std(axis=0) + 1e-6).astype(np.float32)
    return ((x - mean) / std).astype(np.float32), mean, std


def train_model(x, y, w, args):
    x_norm, mean, std = normalize(x)
    w = w / max(float(np.mean(w)), 1e-12)
    w = np.clip(w, 0.05, args.max_weight).astype(np.float32)
    if args.learner == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        models = []
        for j in range(2):
            model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=args.hgb_learning_rate,
                max_iter=args.hgb_max_iter,
                max_leaf_nodes=args.hgb_max_leaf_nodes,
                l2_regularization=args.hgb_l2,
                early_stopping=True,
                random_state=args.seed + j,
            )
            model.fit(x_norm, y[:, j].astype(np.float32), sample_weight=w)
            models.append(model)
            print(f"hgb target={j} train_score={model.score(x_norm, y[:, j]):.6f}", flush=True)
        return {"kind": "hgb", "models": models}, mean, std
    if args.learner == "extratrees":
        from sklearn.ensemble import ExtraTreesRegressor

        model = ExtraTreesRegressor(
            n_estimators=args.et_trees,
            max_depth=args.et_max_depth if args.et_max_depth > 0 else None,
            min_samples_leaf=args.et_min_samples_leaf,
            max_features=args.et_max_features,
            random_state=args.seed,
            n_jobs=-1,
        )
        model.fit(x_norm, y.astype(np.float32), sample_weight=w)
        print(f"extratrees train_score={model.score(x_norm, y):.6f}", flush=True)
        return {"kind": "extratrees", "model": model}, mean, std

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = ScalarExpert(x.shape[1], args.width, args.depth, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    xt = torch.from_numpy(x_norm)
    yt = torch.from_numpy(y.astype(np.float32))
    wt = torch.from_numpy(w.astype(np.float32))
    rng = np.random.default_rng(args.seed)
    n = len(x)
    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(n)
        total = 0.0
        model.train()
        for start in range(0, n, args.batch_size):
            idx = torch.from_numpy(perm[start:start + args.batch_size]).long()
            xb = xt[idx].to(device)
            yb = yt[idx].to(device)
            wb = wt[idx].to(device)
            pred = model(xb)
            loss_vec = torch.nn.functional.huber_loss(pred, yb, reduction="none", delta=args.huber_delta).sum(dim=1)
            loss = (loss_vec * wb).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(idx)
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(f"epoch {epoch}/{args.epochs} loss={total / max(n, 1):.6f}", flush=True)
    return model, mean, std


def predict_scalar(model, mean, std, feat, args):
    feat = np.nan_to_num(feat.astype(np.float32), nan=0.0, posinf=10.0, neginf=-10.0)
    x = ((feat - mean[None]) / std[None]).astype(np.float32)
    if isinstance(model, dict):
        if model["kind"] == "hgb":
            out = np.column_stack([m.predict(x) for m in model["models"]]).astype(np.float32)
        elif model["kind"] == "extratrees":
            out = model["model"].predict(x).astype(np.float32)
        else:
            raise ValueError(model["kind"])
        if args.target_mode == "realimag":
            return clip_complex_scalar(out[:, 0].astype(np.float32) + 1j * out[:, 1].astype(np.float32), args.pred_gain_cap)
        if args.target_mode == "residual_realimag":
            return clip_complex_scalar((1.0 + out[:, 0].astype(np.float32)) + 1j * out[:, 1].astype(np.float32), args.pred_gain_cap)
        log_gain = out[:, 0]
        phase = out[:, 1] * args.clip_phase
        return scalar_from_log_phase(log_gain, phase, args.pred_gain_cap, args.pred_phase_cap)
    device = next(model.parameters()).device
    outs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), args.infer_batch_size):
            xb = torch.from_numpy(x[start:start + args.infer_batch_size]).to(device)
            outs.append(model(xb).detach().cpu().numpy())
    out = np.concatenate(outs, axis=0)
    if args.target_mode == "realimag":
        return clip_complex_scalar(out[:, 0].astype(np.float32) + 1j * out[:, 1].astype(np.float32), args.pred_gain_cap)
    if args.target_mode == "residual_realimag":
        return clip_complex_scalar((1.0 + out[:, 0].astype(np.float32)) + 1j * out[:, 1].astype(np.float32), args.pred_gain_cap)
    log_gain = out[:, 0]
    phase = out[:, 1] * args.clip_phase
    return scalar_from_log_phase(log_gain, phase, args.pred_gain_cap, args.pred_phase_cap)


def patch_ad(base_ad_one, comp_ids, scalar, rho):
    out = np.array(base_ad_one, copy=True)
    for comp, sc in zip(comp_ids, scalar):
        coeff = component_coeff(out, int(comp))
        set_component_coeff(out, int(comp), (1.0 + rho * (sc - 1.0)) * coeff)
    return out


def build_dataset(pos, channels, query_idx, train_pool, path_cache, assign_cache, args, label):
    baseline, neighbor_idx, distances, _ = baseline_for_queries(pos, channels, query_idx, args, train_pool)
    truth = np.asarray(channels[query_idx])
    base_ad = ad_transform(baseline)
    truth_ad = ad_transform(truth)
    component_ids, dyn_assign = make_component_assignment(base_ad, path_cache, assign_cache, query_idx, args.component_count)
    mask = assignment_mask(dyn_assign, np.asarray(args.thresholds, dtype=np.float32))
    feats = []
    targets = []
    weights = []
    meta = []
    for i, qi in enumerate(query_idx):
        neigh_ad = ad_transform(np.asarray(channels[neighbor_idx[i]]))
        built = build_features_for_row(
            pos,
            path_cache,
            assign_cache,
            int(qi),
            base_ad[i],
            truth_ad[i],
            neigh_ad,
            distances[i],
            component_ids[i],
            {k: v[i:i + 1] for k, v in dyn_assign.items() if k != "component_ids"},
            mask[i:i + 1],
            args,
        )
        if built is not None:
            feat, target, weight, comp_ids, scalar, base_power = built
            feats.append(feat)
            targets.append(target)
            weights.append(weight)
            meta.append({
                "row": i,
                "query": int(qi),
                "component_ids": comp_ids,
                "scalar_oracle": scalar,
            })
        if (i + 1) % 50 == 0 or i + 1 == len(query_idx):
            print(f"{label} feature rows {i + 1}/{len(query_idx)}", flush=True)
    if feats:
        x = np.concatenate(feats, axis=0).astype(np.float32)
        y = np.concatenate(targets, axis=0).astype(np.float32)
        w = np.concatenate(weights, axis=0).astype(np.float32)
    else:
        x = np.empty((0, 1), dtype=np.float32)
        y = np.empty((0, 2), dtype=np.float32)
        w = np.empty((0,), dtype=np.float32)
    return {
        "baseline": baseline,
        "truth": truth,
        "base_ad": base_ad,
        "truth_ad": truth_ad,
        "component_ids": component_ids,
        "assign": dyn_assign,
        "mask": mask,
        "features": x,
        "targets": y,
        "weights": w,
        "meta": meta,
    }


def evaluate_dataset(dataset, model, mean, std, args):
    names = ["baseline", "scalar_oracle"]
    names.extend([f"scalar_pred_rho{rho:g}" for rho in args.rhos])
    names.extend([f"local_idw_rho{rho:g}" for rho in args.rhos])
    names.extend([f"local_near_rho{rho:g}" for rho in args.rhos])
    stats = {name: [0.0, 0.0] for name in names}
    accs = {name: MetricAccumulator("sumpol") for name in names}
    cursor = 0
    for i in range(len(dataset["truth"])):
        truth = dataset["truth"][i:i + 1]
        pred_base = dataset["baseline"][i:i + 1]
        num = float(np.sum(np.abs(pred_base - truth) ** 2, dtype=np.float64))
        den = float(np.sum(np.abs(truth) ** 2, dtype=np.float64))
        stats["baseline"][0] += num
        stats["baseline"][1] += den
        accs["baseline"].add(pred_base, truth)

        mask = dataset["mask"][i]
        n = int(np.sum(mask))
        if n > 0:
            feat = dataset["features"][cursor:cursor + n]
            oracle_y = dataset["targets"][cursor:cursor + n]
            cursor += n
            if args.target_mode == "realimag":
                oracle_scalar = clip_complex_scalar(oracle_y[:, 0].astype(np.float32) + 1j * oracle_y[:, 1].astype(np.float32), args.clip_gain)
            elif args.target_mode == "residual_realimag":
                oracle_scalar = clip_complex_scalar((1.0 + oracle_y[:, 0].astype(np.float32)) + 1j * oracle_y[:, 1].astype(np.float32), args.clip_gain)
            else:
                oracle_scalar = scalar_from_log_phase(
                    oracle_y[:, 0],
                    oracle_y[:, 1] * args.clip_phase,
                    args.clip_gain,
                    args.clip_phase,
                )
            pred_scalar = predict_scalar(model, mean, std, feat, args)
            comp_ids = dataset["component_ids"][i][mask]
            oracle_ad = patch_ad(dataset["base_ad"][i], comp_ids, oracle_scalar, 1.0)
            pred_oracle = inverse_ad(oracle_ad[None])
            local_start = 9 + 7 + 4 + 4 + 4
            local_idw_scalar = clip_complex_scalar(
                feat[:, local_start].astype(np.float32) + 1j * feat[:, local_start + 1].astype(np.float32),
                args.pred_gain_cap,
            )
            local_near_scalar = clip_complex_scalar(
                feat[:, local_start + 4].astype(np.float32) + 1j * feat[:, local_start + 5].astype(np.float32),
                args.pred_gain_cap,
            )
            pred_ad_by_rho = {
                rho: patch_ad(dataset["base_ad"][i], comp_ids, pred_scalar, rho)
                for rho in args.rhos
            }
            local_idw_ad_by_rho = {
                rho: patch_ad(dataset["base_ad"][i], comp_ids, local_idw_scalar, rho)
                for rho in args.rhos
            }
            local_near_ad_by_rho = {
                rho: patch_ad(dataset["base_ad"][i], comp_ids, local_near_scalar, rho)
                for rho in args.rhos
            }
        else:
            pred_oracle = pred_base
            pred_ad_by_rho = {}
            local_idw_ad_by_rho = {}
            local_near_ad_by_rho = {}

        num = float(np.sum(np.abs(pred_oracle - truth) ** 2, dtype=np.float64))
        stats["scalar_oracle"][0] += num
        stats["scalar_oracle"][1] += den
        accs["scalar_oracle"].add(pred_oracle, truth)

        for rho in args.rhos:
            name = f"scalar_pred_rho{rho:g}"
            if n > 0:
                pred = inverse_ad(pred_ad_by_rho[rho][None])
            else:
                pred = pred_base
            num = float(np.sum(np.abs(pred - truth) ** 2, dtype=np.float64))
            stats[name][0] += num
            stats[name][1] += den
            accs[name].add(pred, truth)
            name = f"local_idw_rho{rho:g}"
            if n > 0:
                pred = inverse_ad(local_idw_ad_by_rho[rho][None])
            else:
                pred = pred_base
            num = float(np.sum(np.abs(pred - truth) ** 2, dtype=np.float64))
            stats[name][0] += num
            stats[name][1] += den
            accs[name].add(pred, truth)
            name = f"local_near_rho{rho:g}"
            if n > 0:
                pred = inverse_ad(local_near_ad_by_rho[rho][None])
            else:
                pred = pred_base
            num = float(np.sum(np.abs(pred - truth) ** 2, dtype=np.float64))
            stats[name][0] += num
            stats[name][1] += den
            accs[name].add(pred, truth)
        if (i + 1) % 10 == 0 or i + 1 == len(dataset["truth"]):
            print(f"eval {i + 1}/{len(dataset['truth'])}", flush=True)
    results = []
    for name in names:
        results.append({
            "method": name,
            "global_nmse_direct": float(stats[name][0] / max(stats[name][1], 1e-30)),
            **accs[name].result(),
        })
    results.sort(key=lambda r: r["global_nmse_direct"])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--paths", default=str(PATHS.artifacts / "shared_features/path_candidates.npz"))
    parser.add_argument("--assignments", default=str(PATHS.artifacts / "shared_features/component_path_assignments.npz"))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/assigned_scalar_expert.json"))
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--val-mode", choices=["matched-test", "random"], default="matched-test")
    parser.add_argument("--train-query-count", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--component-count", type=int, default=64)
    parser.add_argument("--thresholds", type=parse_floats, default="2.0,3.0,2.0")
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=0.3)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--clip-gain", type=float, default=3.0)
    parser.add_argument("--clip-phase", type=float, default=3.14159265359)
    parser.add_argument("--pred-gain-cap", type=float, default=1.5)
    parser.add_argument("--pred-phase-cap", type=float, default=1.0)
    parser.add_argument("--target-mode", choices=["logphase", "realimag", "residual_realimag"], default="logphase")
    parser.add_argument("--weight-mode", choices=["base", "truth", "sqrt_truth", "uniform"], default="base")
    parser.add_argument("--rhos", type=parse_floats, default="0.05,0.1,0.2,0.3,0.5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learner", choices=["mlp", "hgb", "extratrees"], default="mlp")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--infer-batch-size", type=int, default=65536)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-delta", type=float, default=0.25)
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--hgb-max-iter", type=int, default=400)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--hgb-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--hgb-l2", type=float, default=0.01)
    parser.add_argument("--et-trees", type=int, default=300)
    parser.add_argument("--et-max-depth", type=int, default=24)
    parser.add_argument("--et-min-samples-leaf", type=int, default=3)
    parser.add_argument("--et-max-features", type=float, default=0.7)
    args = parser.parse_args()

    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))
    channels = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    path_cache = np.load(args.paths)
    assign_cache = np.load(args.assignments)
    val_idx = choose_val_indices(pos[:, :2], test_pos[:, :2], args.val_count, args.val_mode, args.seed)
    train_pool = np.setdiff1d(np.arange(len(pos), dtype=np.int64), val_idx, assume_unique=False)
    rng = np.random.default_rng(args.seed)
    train_queries = np.array(train_pool, copy=True)
    if 0 < args.train_query_count < len(train_queries):
        train_queries = rng.choice(train_queries, size=args.train_query_count, replace=False)
    train_queries = np.asarray(train_queries, dtype=np.int64)

    print(f"building train dataset queries={len(train_queries)} val={len(val_idx)}", flush=True)
    train_data = build_dataset(pos, channels, train_queries, train_pool, path_cache, assign_cache, args, "train")
    print(
        f"train examples={train_data['features'].shape[0]} "
        f"feature_dim={train_data['features'].shape[1] if train_data['features'].size else 0}",
        flush=True,
    )
    model, mean, std = train_model(train_data["features"], train_data["targets"], train_data["weights"], args)

    print("building validation dataset", flush=True)
    val_data = build_dataset(pos, channels, val_idx, train_pool, path_cache, assign_cache, args, "val")
    results = evaluate_dataset(val_data, model, mean, std, args)
    payload = {
        "args": vars(args),
        "train_examples": int(train_data["features"].shape[0]),
        "val_examples": int(val_data["features"].shape[0]),
        "feature_dim": int(train_data["features"].shape[1] if train_data["features"].size else 0),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
