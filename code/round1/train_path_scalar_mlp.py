#!/usr/bin/env python3
from project_paths import PATHS
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from evaluate_ad_coeff_methods import component_scalar, eval_one
from evaluate_derotated_loo_grid import choose_val_indices
from evaluate_local_ad_paths import BS_POS, MH, MV, MP, idw_weights
from evaluate_physical_local_linear import affine_coefficients, predict_affine
from evaluate_ue_angle_delay import (
    NP,
    NV,
    S_SC,
    ad3_to_flat,
    ad3_transform,
    flat_components3,
    inverse_ad3,
    gather_coeff3,
    scatter_coeff3,
    select_components3,
)
from generate_geometric_steering_blend import steering_phase, unit_vectors
from spectrum_projection import MetricAccumulator


def parse_ints(text):
    return [int(x) for x in text.split(",") if x.strip()]


def parse_floats(text):
    return [float(x) for x in text.split(",") if x.strip()]


def load_selected_map_features(path, indices_text, split):
    if not path:
        return None
    z = np.load(path)
    key = f"{split}_features"
    feats = z[key].astype(np.float32)
    indices = parse_ints(indices_text)
    if indices:
        feats = feats[:, indices]
    return feats.astype(np.float32)


def load_virtual_source_model(path, max_sources, delay_fit_path):
    if not path:
        return None
    with open(path, "r") as f:
        payload = json.load(f)
    planes = payload.get("planes", [])[:max_sources]
    bs_xy = BS_POS[:2].astype(np.float64)
    sources = [BS_POS.astype(np.float64)]
    strengths = [1.0]
    max_count = max([float(p.get("count", 1.0)) for p in planes] + [1.0])
    for plane in planes:
        normal = np.asarray(plane["normal"], dtype=np.float64)
        normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
        c = float(plane["c"])
        signed = float(np.dot(normal, bs_xy) - c)
        mirror_xy = bs_xy - 2.0 * signed * normal
        sources.append(np.asarray([mirror_xy[0], mirror_xy[1], BS_POS[2]], dtype=np.float64))
        strengths.append(np.sqrt(float(plane.get("count", 1.0)) / max_count))
    fit = np.load(delay_fit_path)
    return {
        "sources": np.asarray(sources, dtype=np.float64),
        "strengths": np.asarray(strengths, dtype=np.float32),
        "delay_slope": float(fit["slope"]),
        "delay_intercept": float(fit["intercept"]),
    }


def circular_abs_delta(values, centers, period):
    delta = np.abs(values - centers)
    return np.minimum(delta, period - delta)


def virtual_source_feature_block(pos, qi, hbin, vbin, sbin, virtual_model, steering_scale):
    if virtual_model is None:
        return np.zeros(len(hbin), dtype=np.int64), None
    q = pos[qi].astype(np.float64)
    sources = virtual_model["sources"]
    vec = q[None, :] - sources
    lengths = np.linalg.norm(vec, axis=1)
    dirs = vec / np.maximum(lengths[:, None], 1e-12)
    delay_pred = virtual_model["delay_slope"] * lengths + virtual_model["delay_intercept"]
    h_pred = (steering_scale * dirs[:, 1] / (2.0 * np.pi) * MH) % MH
    v_pred = (steering_scale * (-dirs[:, 2]) / (2.0 * np.pi) * MV) % MV

    delay_res = circular_abs_delta(sbin[:, None].astype(np.float64), delay_pred[None, :], S_SC)
    h_res = circular_abs_delta(hbin[:, None].astype(np.float64), h_pred[None, :], MH)
    v_res = circular_abs_delta(vbin[:, None].astype(np.float64), v_pred[None, :], MV)
    score = (delay_res / 4.0) ** 2 + (h_res / 4.0) ** 2 + (v_res / 2.0) ** 2
    best = np.argmin(score, axis=1)
    row = np.arange(len(best))
    best_len = lengths[best]
    best_strength = virtual_model["strengths"][best]
    return best.astype(np.int64), np.column_stack([
        np.clip(delay_res[row, best] / 8.0, 0.0, 4.0),
        np.clip(h_res[row, best] / 8.0, 0.0, 4.0),
        np.clip(v_res[row, best] / 4.0, 0.0, 4.0),
        np.clip(np.sqrt(score[row, best]), 0.0, 8.0),
        best_len / 300.0,
        best_strength,
        (best == 0).astype(np.float32),
        np.log1p(best.astype(np.float32)) / 5.0,
    ]).astype(np.float32)


def make_neighbors(pos, query_idx, train_pool, k):
    tree = cKDTree(pos[train_pool, :2])
    d, local = tree.query(pos[query_idx, :2], k=k + 1, workers=-1)
    if np.ndim(query_idx) == 0:
        query_idx = np.asarray([int(query_idx)], dtype=np.int64)
        d = d[None, :]
        local = local[None, :]
    out_idx = np.empty((len(query_idx), k), dtype=np.int64)
    out_d = np.empty((len(query_idx), k), dtype=np.float64)
    for i, qi in enumerate(query_idx):
        cand = train_pool[local[i]]
        keep = cand != qi
        out_idx[i] = cand[keep][:k]
        out_d[i] = d[i, keep][:k]
    return out_d, out_idx


def radial_phase(dist_delta, beta, gamma):
    sub = np.linspace(-0.5, 0.5, S_SC, dtype=np.float32)
    beta_s = (beta + gamma * sub).astype(np.float32)
    return np.exp(1j * dist_delta[:, None] * beta_s[None, :]).astype(np.complex64)


def clip_scalar(scalar, clip_abs):
    if clip_abs <= 0:
        return scalar.astype(np.complex64)
    mag = np.abs(scalar)
    good = mag > 1e-12
    out = scalar.copy()
    clipped = np.minimum(mag[good], clip_abs)
    out[good] = (clipped * scalar[good] / mag[good]).astype(np.complex64)
    out[~good] = 1.0 + 0.0j
    return out.astype(np.complex64)


def unit_complex(x):
    mag = np.abs(x)
    out = np.ones_like(x, dtype=np.complex64)
    good = mag > 1e-12
    out[good] = (x[good] / mag[good]).astype(np.complex64)
    return out


def local_loo_scalar_summary(coeff, weights, neigh_xy, power):
    k = coeff.shape[0]
    comp_count = coeff.shape[1]
    if k < 3:
        ones = np.ones(comp_count, dtype=np.complex64)
        return np.column_stack([
            ones.real,
            ones.imag,
            ones.real,
            ones.imag,
            np.ones(comp_count, dtype=np.float32),
            np.zeros(comp_count, dtype=np.float32),
        ]).astype(np.float32)
    scalars = np.empty((k, comp_count), dtype=np.complex64)
    all_ids = np.arange(k)
    for j in range(k):
        mask = all_ids != j
        d = np.linalg.norm(neigh_xy[mask] - neigh_xy[j], axis=1)
        wj = idw_weights(d, power)
        pred_j = np.tensordot(wj.astype(np.float32), coeff[mask], axes=(0, 0)).astype(np.complex64)
        scalars[j] = component_scalar(pred_j, coeff[j])
    scalar_idw = np.tensordot(weights.astype(np.float32), scalars, axes=(0, 0)).astype(np.complex64)
    scalar_near = scalars[0]
    scalar_phase = unit_complex(scalar_idw)
    return np.column_stack([
        scalar_idw.real,
        scalar_idw.imag,
        scalar_near.real,
        scalar_near.imag,
        np.clip(np.abs(scalar_idw), 0.0, 3.0),
        np.angle(scalar_phase) / np.pi,
    ]).astype(np.float32)


def component_feature_block(
    pos,
    dist_all,
    u_all,
    qi,
    neigh_idx,
    distances,
    weights,
    power,
    component_ids,
    pred_coeff,
    coeff,
    map_features=None,
    virtual_model=None,
    steering_scale=4.71238898038,
):
    hbin, vbin, uvbin, sbin = flat_components3(component_ids)
    count = component_ids.shape[0]
    q = pos[qi]
    qdist = dist_all[qi]
    qu = u_all[qi]
    d_sorted = np.sort(distances)
    d1 = d_sorted[0]
    d4 = d_sorted[min(3, len(d_sorted) - 1)]
    d8 = d_sorted[min(7, len(d_sorted) - 1)]
    dmean = float(np.mean(d_sorted))
    qfeat = np.asarray([
        q[0] / 300.0,
        q[1] / 300.0,
        qdist / 300.0,
        qu[0],
        qu[1],
        qu[2],
        d1 / 20.0,
        d4 / 20.0,
        d8 / 20.0,
        dmean / 20.0,
    ], dtype=np.float32)
    qfeat = np.repeat(qfeat[None, :], count, axis=0)

    source_id, virtual_feat = virtual_source_feature_block(
        pos, qi, hbin, vbin, sbin, virtual_model, steering_scale
    )

    comp_phase = np.column_stack([
        hbin,
        vbin,
        uvbin,
        sbin,
        source_id,
        np.sin(2 * np.pi * hbin / MH),
        np.cos(2 * np.pi * hbin / MH),
        np.sin(2 * np.pi * vbin / MV),
        np.cos(2 * np.pi * vbin / MV),
        np.sin(2 * np.pi * uvbin / NV),
        np.cos(2 * np.pi * uvbin / NV),
        np.sin(2 * np.pi * sbin / S_SC),
        np.cos(2 * np.pi * sbin / S_SC),
        (hbin - (MH - 1) * 0.5) / MH,
        (vbin - (MV - 1) * 0.5) / MV,
        (uvbin - (NV - 1) * 0.5) / max(NV, 1),
        sbin / S_SC,
    ]).astype(np.float32)

    pred_flat = pred_coeff.reshape(count, -1)
    pred_power = np.sum(np.abs(pred_flat) ** 2, axis=1).astype(np.float64)
    neigh_power = np.sum(np.abs(coeff) ** 2, axis=(2, 3), dtype=np.float64)
    weighted_power = np.tensordot(weights.astype(np.float64), neigh_power, axes=(0, 0))
    nearest_power = neigh_power[0]
    coherence = pred_power / np.maximum(weighted_power, 1e-30)
    phase_ref = np.sum(pred_flat, axis=1)
    phase = np.angle(phase_ref)
    dyn = np.column_stack([
        np.log1p(pred_power),
        np.log1p(weighted_power),
        np.log1p(nearest_power),
        np.clip(coherence, 0.0, 4.0),
        np.cos(phase),
        np.sin(phase),
    ]).astype(np.float32)
    loo = local_loo_scalar_summary(coeff, weights, pos[neigh_idx, :2], power)
    blocks = [comp_phase, qfeat, dyn]
    if map_features is not None:
        mf = np.repeat(map_features[qi][None, :].astype(np.float32), count, axis=0)
        blocks.append(mf)
    if virtual_feat is not None:
        blocks.append(virtual_feat)
    blocks.append(loo)
    return np.concatenate(blocks, axis=1), pred_power.astype(np.float32)


def query_components(
    h,
    pos,
    dist_all,
    u_all,
    qi,
    neigh_idx,
    distances,
    args,
    map_features=None,
    virtual_model=None,
):
    neigh_h = np.asarray(h[neigh_idx])
    weights_idw = idw_weights(distances, args.power)
    dist_delta = dist_all[neigh_idx] - dist_all[qi]
    radial = radial_phase(dist_delta, args.beta, args.gamma)
    steer = steering_phase(u_all, neigh_idx, qi, args.steering_scale)
    corrected = neigh_h * radial[:, None, None, :] * steer[:, :, None, None]
    corrected_ad = ad3_transform(corrected, args.n_order)
    corrected_power = np.sum(np.abs(corrected_ad) ** 2, axis=(1, 5), dtype=np.float64)
    component_ids = select_components3(
        corrected_power, weights_idw, args.component_mode, args.per_neighbor, args.component_count
    )
    coeff = gather_coeff3(corrected_ad, component_ids)
    rel_xy = pos[neigh_idx, :2] - pos[qi, :2]
    affine_w = affine_coefficients(rel_xy, distances, args.power, args.ridge)
    pred_coeff = np.tensordot(affine_w.astype(np.float32), coeff, axes=(0, 0)).astype(np.complex64)
    truth = np.asarray(h[qi:qi + 1])
    truth_flat = ad3_to_flat(ad3_transform(truth, args.n_order))
    truth_coeff = truth_flat[0, component_ids]
    scalar = component_scalar(pred_coeff, truth_coeff)
    scalar = clip_scalar(scalar, args.clip_scalar)
    feat, pred_power = component_feature_block(
        pos,
        dist_all,
        u_all,
        qi,
        neigh_idx,
        distances,
        weights_idw,
        args.power,
        component_ids,
        pred_coeff,
        coeff,
        map_features,
        virtual_model,
        args.steering_scale,
    )
    return {
        "truth": truth,
        "neigh_h": neigh_h,
        "dist_delta": dist_delta,
        "component_ids": component_ids,
        "pred_coeff": pred_coeff,
        "truth_coeff": truth_coeff,
        "scalar": scalar,
        "features": feat,
        "pred_power": pred_power,
        "affine_w": affine_w,
    }


def build_examples(
    h,
    pos,
    dist_all,
    u_all,
    query_idx,
    train_pool,
    args,
    label,
    map_features=None,
    virtual_model=None,
):
    d, neigh = make_neighbors(pos, query_idx, train_pool, args.k)
    feats = []
    targets = []
    weights = []
    for i, qi in enumerate(query_idx):
        item = query_components(
            h, pos, dist_all, u_all, int(qi), neigh[i], d[i], args, map_features, virtual_model
        )
        feats.append(item["features"])
        targets.append(np.column_stack([item["scalar"].real, item["scalar"].imag]).astype(np.float32))
        weights.append(np.maximum(item["pred_power"], 1e-12).astype(np.float32))
        if (i + 1) % 50 == 0 or i + 1 == len(query_idx):
            print(f"{label} examples {i + 1}/{len(query_idx)}", flush=True)
    x = np.concatenate(feats, axis=0).astype(np.float32)
    y = np.concatenate(targets, axis=0).astype(np.float32)
    w = np.concatenate(weights, axis=0).astype(np.float32)
    w = w / max(float(np.mean(w)), 1e-12)
    w = np.clip(w, 0.05, args.max_weight).astype(np.float32)
    return x, y, w


class ScalarMLP(torch.nn.Module):
    def __init__(self, in_dim, width, depth):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(depth):
            layers.append(torch.nn.Linear(dim, width))
            layers.append(torch.nn.SiLU())
            layers.append(torch.nn.LayerNorm(width))
            dim = width
        layers.append(torch.nn.Linear(dim, 2))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ScalarEmbedMLP(torch.nn.Module):
    def __init__(self, cont_dim, width, depth, emb_h, emb_v, emb_uv, emb_s):
        super().__init__()
        self.h_emb = torch.nn.Embedding(MH, emb_h)
        self.v_emb = torch.nn.Embedding(MV, emb_v)
        self.uv_emb = torch.nn.Embedding(NV, emb_uv)
        self.s_emb = torch.nn.Embedding(S_SC, emb_s)
        in_dim = cont_dim + emb_h + emb_v + emb_uv + emb_s
        layers = []
        dim = in_dim
        for _ in range(depth):
            layers.append(torch.nn.Linear(dim, width))
            layers.append(torch.nn.SiLU())
            layers.append(torch.nn.LayerNorm(width))
            dim = width
        layers.append(torch.nn.Linear(dim, 2))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        ids = x[:, :4].long()
        cont = x[:, 4:]
        h = ids[:, 0].clamp(0, MH - 1)
        v = ids[:, 1].clamp(0, MV - 1)
        uv = ids[:, 2].clamp(0, NV - 1)
        s = ids[:, 3].clamp(0, S_SC - 1)
        emb = torch.cat([self.h_emb(h), self.v_emb(v), self.uv_emb(uv), self.s_emb(s)], dim=1)
        return self.net(torch.cat([cont, emb], dim=1))


class ScalarDelayMoE(torch.nn.Module):
    def __init__(self, cont_dim, width, depth, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.experts = torch.nn.ModuleList([ScalarMLP(cont_dim, width, depth) for _ in range(num_experts)])

    def forward(self, x):
        ids = x[:, :4].long()
        cont = x[:, 4:]
        delay = ids[:, 3].clamp(0, S_SC - 1)
        group = torch.div(delay * self.num_experts, S_SC, rounding_mode="floor").clamp(0, self.num_experts - 1)
        out = torch.empty((x.shape[0], 2), dtype=cont.dtype, device=cont.device)
        for g in range(self.num_experts):
            mask = group == g
            if torch.any(mask):
                out[mask] = self.experts[g](cont[mask])
        return out


class ScalarSourceMoE(torch.nn.Module):
    def __init__(self, cont_dim, width, depth, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.experts = torch.nn.ModuleList([ScalarMLP(cont_dim, width, depth) for _ in range(num_experts)])

    def forward(self, x):
        ids = x[:, :5].long()
        cont = x[:, 5:]
        source = ids[:, 4].clamp(0, self.num_experts - 1)
        out = torch.empty((x.shape[0], 2), dtype=cont.dtype, device=cont.device)
        for g in range(self.num_experts):
            mask = source == g
            if torch.any(mask):
                out[mask] = self.experts[g](cont[mask])
        return out


def normalize_features(x, args):
    if args.model_kind in {"embed_mlp", "delay_moe", "source_moe"}:
        id_count = 5 if args.model_kind == "source_moe" else 4
        ids = x[:, :id_count].astype(np.float32)
        cont = x[:, id_count:].astype(np.float32)
        mean = cont.mean(axis=0).astype(np.float32)
        std = (cont.std(axis=0) + 1e-6).astype(np.float32)
        return np.concatenate([ids, (cont - mean) / std], axis=1).astype(np.float32), mean, std
    mean = x.mean(axis=0).astype(np.float32)
    std = (x.std(axis=0) + 1e-6).astype(np.float32)
    return ((x - mean) / std).astype(np.float32), mean, std


def apply_normalize_features(features, mean, std, args):
    if args.model_kind in {"embed_mlp", "delay_moe", "source_moe"}:
        id_count = 5 if args.model_kind == "source_moe" else 4
        ids = features[:, :id_count].astype(np.float32)
        cont = features[:, id_count:].astype(np.float32)
        return np.concatenate([ids, (cont - mean[None, :]) / std[None, :]], axis=1).astype(np.float32)
    return ((features.astype(np.float32) - mean[None, :]) / std[None, :]).astype(np.float32)


def train_model(x, y, w, args):
    rng = np.random.default_rng(args.seed)
    y_train = target_for_mode(x, y, args.target_mode)
    x, mean, std = normalize_features(x, args)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.model_kind == "embed_mlp":
        model = ScalarEmbedMLP(
            x.shape[1] - 4,
            args.width,
            args.depth,
            args.emb_h,
            args.emb_v,
            args.emb_uv,
            args.emb_s,
        ).to(device)
    elif args.model_kind == "delay_moe":
        model = ScalarDelayMoE(x.shape[1] - 4, args.width, args.depth, args.num_experts).to(device)
    elif args.model_kind == "source_moe":
        model = ScalarSourceMoE(x.shape[1] - 5, args.width, args.depth, args.num_experts).to(device)
    else:
        model = ScalarMLP(x.shape[1], args.width, args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y_train)
    wt = torch.from_numpy(w)
    n = x.shape[0]
    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(n)
        total = 0.0
        seen = 0
        model.train()
        for start in range(0, n, args.batch_size):
            idx = torch.from_numpy(perm[start:start + args.batch_size]).long()
            xb = xt[idx].to(device)
            yb = yt[idx].to(device)
            wb = wt[idx].to(device)
            pred = model(xb)
            err = pred - yb
            if args.loss == "huber":
                elem = torch.nn.functional.huber_loss(pred, yb, reduction="none", delta=args.huber_delta).sum(dim=1)
                loss = (elem * wb).mean()
            else:
                loss = ((err ** 2).sum(dim=1) * wb).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(idx)
            seen += len(idx)
        if epoch == 1 or epoch % max(args.print_every, 1) == 0 or epoch == args.epochs:
            print(f"epoch {epoch}/{args.epochs} weighted_mse={total / max(seen, 1):.6f}", flush=True)
    return model, mean.astype(np.float32), std.astype(np.float32)


def scalar_base_from_features(features, target_mode):
    if target_mode == "direct":
        return np.zeros((features.shape[0], 2), dtype=np.float32)
    if target_mode == "residual_identity":
        base = np.zeros((features.shape[0], 2), dtype=np.float32)
        base[:, 0] = 1.0
        return base
    if target_mode == "residual_loo_idw":
        return features[:, -6:-4].astype(np.float32)
    if target_mode == "residual_loo_nearest":
        return features[:, -4:-2].astype(np.float32)
    raise ValueError(target_mode)


def target_for_mode(features, target, target_mode):
    return (target.astype(np.float32) - scalar_base_from_features(features, target_mode)).astype(np.float32)


def predict_scalars(model, mean, std, features, args):
    base = scalar_base_from_features(features, args.target_mode)
    x = apply_normalize_features(features, mean, std, args)
    device = next(model.parameters()).device
    outs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], args.infer_batch_size):
            xb = torch.from_numpy(x[start:start + args.infer_batch_size]).to(device)
            outs.append(model(xb).detach().cpu().numpy())
    out = np.concatenate(outs, axis=0)
    if args.target_mode != "direct":
        out = out + base
    scalar = out[:, 0].astype(np.float32) + 1j * out[:, 1].astype(np.float32)
    return clip_scalar(scalar.astype(np.complex64), args.pred_clip_scalar)


def sparse_pred(component_ids, coeff, n_order):
    return inverse_ad3(scatter_coeff3(component_ids, coeff.astype(np.complex64)), n_order)


def evaluate_model(
    h,
    pos,
    dist_all,
    u_all,
    val_idx,
    train_pool,
    model,
    mean,
    std,
    args,
    map_features=None,
    virtual_model=None,
):
    d, neigh = make_neighbors(pos, val_idx, train_pool, args.k)
    names = ["physical_full", "sparse_base", "scalar_pred", "scalar_oracle"]
    names.extend([f"blend_{rho:g}" for rho in args.blend_rhos])
    stats = {name: {"num": 0.0, "den": 0.0} for name in names}
    accs = {name: MetricAccumulator("sumpol") for name in names}
    for i, qi in enumerate(val_idx):
        item = query_components(
            h, pos, dist_all, u_all, int(qi), neigh[i], d[i], args, map_features, virtual_model
        )
        pred_scalar = predict_scalars(model, mean, std, item["features"], args)
        coeff_base = item["pred_coeff"]
        coeff_pred = (coeff_base * pred_scalar[:, None, None]).astype(np.complex64)
        coeff_oracle = (coeff_base * item["scalar"][:, None, None]).astype(np.complex64)
        pred_sparse_base = sparse_pred(item["component_ids"], coeff_base, args.n_order)
        pred_scalar_sparse = sparse_pred(item["component_ids"], coeff_pred, args.n_order)
        pred_oracle_sparse = sparse_pred(item["component_ids"], coeff_oracle, args.n_order)
        pred_full = predict_affine(
            item["neigh_h"],
            item["affine_w"],
            item["dist_delta"],
            u_all,
            neigh[i],
            int(qi),
            args.beta,
            args.gamma,
            args.alpha,
            args.steering_scale,
        )
        truth = item["truth"]
        preds = {
            "physical_full": pred_full[None],
            "sparse_base": pred_sparse_base,
            "scalar_pred": pred_scalar_sparse,
            "scalar_oracle": pred_oracle_sparse,
        }
        for rho in args.blend_rhos:
            preds[f"blend_{rho:g}"] = ((1.0 - rho) * pred_full[None] + rho * pred_scalar_sparse).astype(np.complex64)
        for name, pred in preds.items():
            num, den = eval_one(truth, pred)
            stats[name]["num"] += num
            stats[name]["den"] += den
            accs[name].add(pred, truth)
        if (i + 1) % 10 == 0 or i + 1 == len(val_idx):
            print(f"val reconstruct {i + 1}/{len(val_idx)}", flush=True)
    results = []
    for name in names:
        nmse = stats[name]["num"] / max(stats[name]["den"], 1e-30)
        results.append({
            "method": name,
            "global_nmse_direct": float(nmse),
            **accs[name].result(),
        })
    results.sort(key=lambda r: r["global_nmse_direct"])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PATHS.round1))
    parser.add_argument("--out", default=str(PATHS.analysis / "results/json/path_scalar_mlp.json"))
    parser.add_argument("--map-features", default="")
    parser.add_argument(
        "--map-feature-indices",
        default="0,1,2,3,5,9,13,17,21,25,28,32,33,34,35,39,40,41",
    )
    parser.add_argument("--virtual-sources", default="")
    parser.add_argument("--max-virtual-sources", type=int, default=80)
    parser.add_argument(
        "--delay-fit",
        default=str(PATHS.artifacts / "shared_features/delay_length_fit.npz"),
    )
    parser.add_argument("--val-count", type=int, default=50)
    parser.add_argument("--val-mode", choices=["matched-test", "random"], default="matched-test")
    parser.add_argument("--train-query-count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--component-count", type=int, default=256)
    parser.add_argument("--component-mode", default="weighted")
    parser.add_argument("--per-neighbor", type=int, default=256)
    parser.add_argument("--n-order", default="vp")
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=140.3)
    parser.add_argument("--gamma", type=float, default=0.15)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=0.3)
    parser.add_argument("--steering-scale", type=float, default=4.71238898038)
    parser.add_argument("--clip-scalar", type=float, default=2.0)
    parser.add_argument("--pred-clip-scalar", type=float, default=2.0)
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--blend-rhos", type=parse_floats, default="0.25,0.5,0.75,1.0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--infer-batch-size", type=int, default=65536)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--model-kind", choices=["mlp", "embed_mlp", "delay_moe", "source_moe"], default="mlp")
    parser.add_argument("--emb-h", type=int, default=8)
    parser.add_argument("--emb-v", type=int, default=8)
    parser.add_argument("--emb-uv", type=int, default=4)
    parser.add_argument("--emb-s", type=int, default=16)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-mode", choices=["direct", "residual_identity", "residual_loo_idw", "residual_loo_nearest"], default="direct")
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse")
    parser.add_argument("--huber-delta", type=float, default=0.25)
    parser.add_argument("--print-every", type=int, default=5)
    args = parser.parse_args()

    pos = np.load(os.path.join(args.data, "Round1_Train_Pos.npy"))
    test_pos = np.load(os.path.join(args.data, "Round1_Test_Pos.npy"))
    h = np.load(os.path.join(args.data, "Round1_Train_Channel.npy"), mmap_mode="r")
    val_idx = choose_val_indices(pos[:, :2], test_pos[:, :2], args.val_count, args.val_mode, args.seed)
    train_pool = np.setdiff1d(np.arange(len(pos), dtype=np.int64), val_idx, assume_unique=False)
    rng = np.random.default_rng(args.seed)
    train_queries = np.array(train_pool, copy=True)
    if 0 < args.train_query_count < len(train_queries):
        train_queries = rng.choice(train_queries, size=args.train_query_count, replace=False)
    train_queries = np.asarray(train_queries, dtype=np.int64)
    dist_all = np.linalg.norm(pos - BS_POS[None, :], axis=1)
    u_all = unit_vectors(pos)
    map_train = load_selected_map_features(args.map_features, args.map_feature_indices, "train")
    virtual_model = load_virtual_source_model(args.virtual_sources, args.max_virtual_sources, args.delay_fit)

    print(
        f"train_queries={len(train_queries)} val={len(val_idx)} k={args.k} comp={args.component_count} "
        f"n_order={args.n_order} virtual_sources={0 if virtual_model is None else len(virtual_model['sources'])}",
        flush=True,
    )
    x, y, w = build_examples(
        h, pos, dist_all, u_all, train_queries, train_pool, args, "train", map_train, virtual_model
    )
    print(f"examples x={x.shape} y={y.shape} weight_mean={w.mean():.4f}", flush=True)
    model, mean, std = train_model(x, y, w, args)
    results = evaluate_model(
        h, pos, dist_all, u_all, val_idx, train_pool, model, mean, std, args, map_train, virtual_model
    )
    payload = {
        "args": vars(args),
        "train_examples": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
