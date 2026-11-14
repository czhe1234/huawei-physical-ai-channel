#!/usr/bin/env python3
"""Generate v99: learned component scalar gate on top of v98.

This version trains a small component-level model from train LOO residuals.
For each strong PHV angle-delay component in the train LOO base, it estimates
the oracle complex scalar

    r* = <base_component, truth_component> / <base_component, base_component>

and learns to predict a conservative residual scalar `(r* - 1)` plus a
reliability score from geometry, Sionna path-candidate features, and component
identity/power features.

At inference, the model is applied to dominant AD components of the v98 output:

    component <- component * (1 + rho * gate * predicted_residual)

The total AD-domain power of every point is restored to the v98 value.  This
keeps v98's validated power/support structure and only tests whether learned
component-wise phase/scalar correction can help high-risk regions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from generate_v87_easy_field_projection import (
    BS_POS,
    MH,
    MP,
    MV,
    N_ANT,
    S_SC,
    ad_transform,
    beta_subcarriers,
    inverse_ad,
    summarize,
)
from project_paths import PATHS
from powerfield_common import ASSETS, TINY, regime_of


REGIMES = ("clear", "light", "medium", "heavy")
MAT_COUNT = 6


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_regime_values(text: str, default: float) -> dict[str, float]:
    out = {r: float(default) for r in REGIMES}
    if not text:
        return out
    for item in text.split(","):
        if not item.strip():
            continue
        k, v = item.split(":")
        k = k.strip()
        if k not in out:
            raise ValueError(f"unknown regime {k!r}")
        out[k] = float(v)
    return out


def top_ids_from_power(power_hvt: np.ndarray, k: int) -> np.ndarray:
    flat = power_hvt.reshape(power_hvt.shape[0], -1)
    kk = min(int(k), flat.shape[1])
    ids = np.argpartition(flat, -kk, axis=1)[:, -kk:]
    vals = np.take_along_axis(flat, ids, axis=1)
    order = np.argsort(-vals, axis=1)
    return np.take_along_axis(ids, order, axis=1).astype(np.int64)


def flat_to_hvt(flat_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(flat_ids, dtype=np.int64)
    h = ids // (MV * S_SC)
    rem = ids % (MV * S_SC)
    v = rem // S_SC
    t = rem % S_SC
    return h.astype(np.int64), v.astype(np.int64), t.astype(np.int64)


def component_vectors(ad: np.ndarray, row: int, ids: np.ndarray) -> np.ndarray:
    h, v, t = flat_to_hvt(ids)
    # ad: B,MP,MH,MV,N,S -> C,MP,N
    vec = ad[row, :, h, v, :, t]
    return np.moveaxis(vec, 0, 1).reshape(len(ids), -1).astype(np.complex64)


def component_vectors_one(ad_one: np.ndarray, ids: np.ndarray) -> np.ndarray:
    h, v, t = flat_to_hvt(ids)
    vec = ad_one[:, h, v, :, t]
    return np.moveaxis(vec, 0, 1).reshape(len(ids), -1).astype(np.complex64)


def compute_oracle_scalar(base_vec: np.ndarray, truth_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ip = np.sum(np.conj(base_vec) * truth_vec, axis=1)
    bp = np.sum(np.abs(base_vec) ** 2, axis=1).real
    tp = np.sum(np.abs(truth_vec) ** 2, axis=1).real
    scalar = ip / np.maximum(bp, 1e-30)
    coherence = np.abs(ip) / np.maximum(np.sqrt(bp * tp), 1e-30)
    power = bp.astype(np.float32)
    return scalar.astype(np.complex64), coherence.astype(np.float32), power


def clip_scalar_residual(scalar: np.ndarray, gain_cap: float, phase_cap: float) -> np.ndarray:
    mag = np.clip(np.abs(scalar), 1.0 / float(gain_cap), float(gain_cap))
    phase = np.clip(np.angle(scalar), -float(phase_cap), float(phase_cap))
    clipped = mag * np.exp(1j * phase)
    return (clipped - 1.0).astype(np.complex64)


def path_angle_bins(tx_dir: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    # Same sign convention used in prior calibrated path-component diagnostics.
    h = (-float(scale) * tx_dir[..., 1] / (2.0 * np.pi) * MH) % MH
    v = (float(scale) * (-tx_dir[..., 2]) / (2.0 * np.pi) * MV) % MV
    return h.astype(np.float32), v.astype(np.float32)


def circular_bin_diff(a: np.ndarray, b: np.ndarray, period: float) -> np.ndarray:
    return np.abs(((a - b + period / 2.0) % period) - period / 2.0)


def point_features(pos: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    rel = pos - BS_POS[None, :]
    dist = np.linalg.norm(rel, axis=1)
    az = np.arctan2(rel[:, 1], rel[:, 0])
    el = np.arctan2(rel[:, 2], np.maximum(np.hypot(rel[:, 0], rel[:, 1]), 1e-6))
    reg = regime_of(blocked)
    one = np.stack([(reg == r).astype(np.float32) for r in REGIMES], axis=1)
    x = (pos[:, 0] - 120.0) / 120.0
    y = (pos[:, 1] + 80.0) / 120.0
    feats = [
        x,
        y,
        pos[:, 2] / 30.0,
        dist / 250.0,
        np.log1p(dist) / 6.0,
        np.sin(az),
        np.cos(az),
        np.sin(el),
        np.cos(el),
        blocked / 160.0,
        np.log1p(np.maximum(blocked, 0.0)) / 6.0,
        (blocked <= 0).astype(np.float32),
    ]
    return np.concatenate([np.stack(feats, axis=1).astype(np.float32), one], axis=1)


def component_base_features(ids: np.ndarray, ranks: np.ndarray, frac: np.ndarray, comp_power: np.ndarray) -> np.ndarray:
    h, v, t = flat_to_hvt(ids)
    h_f = h.astype(np.float32)
    v_f = v.astype(np.float32)
    t_f = t.astype(np.float32)
    return np.column_stack(
        [
            h_f / (MH - 1),
            v_f / (MV - 1),
            t_f / (S_SC - 1),
            np.sin(2.0 * np.pi * h_f / MH),
            np.cos(2.0 * np.pi * h_f / MH),
            np.sin(2.0 * np.pi * v_f / MV),
            np.cos(2.0 * np.pi * v_f / MV),
            np.sin(2.0 * np.pi * t_f / S_SC),
            np.cos(2.0 * np.pi * t_f / S_SC),
            ranks.astype(np.float32) / 96.0,
            np.log1p(np.maximum(comp_power, 0.0)) / 8.0,
            np.sqrt(np.maximum(frac, 0.0)),
            np.log(np.maximum(frac, 1e-12)) / 12.0,
        ]
    ).astype(np.float32)


def path_component_features(
    path_npz: np.lib.npyio.NpzFile,
    prefix: str,
    point_ids: np.ndarray,
    comp_ids: np.ndarray,
    scale: float,
) -> np.ndarray:
    valid = path_npz[f"{prefix}_valid"][point_ids].astype(bool)
    ptype = path_npz[f"{prefix}_path_type"][point_ids]
    material = np.asarray(path_npz[f"{prefix}_material_id"][point_ids], dtype=np.int32)
    delay = np.nan_to_num(path_npz[f"{prefix}_delay_bin"][point_ids].astype(np.float32), nan=999.0, posinf=999.0, neginf=999.0)
    score = np.nan_to_num(path_npz[f"{prefix}_score"][point_ids].astype(np.float32), nan=-8.0, posinf=4.0, neginf=-8.0)
    excess = np.nan_to_num(path_npz[f"{prefix}_excess_length"][point_ids].astype(np.float32), nan=0.0, posinf=900.0, neginf=0.0)
    edge = np.nan_to_num(path_npz[f"{prefix}_edge_margin"][point_ids].astype(np.float32), nan=0.0, posinf=0.35, neginf=0.0)
    inc = np.nan_to_num(path_npz[f"{prefix}_incidence_cos"][point_ids].astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    tx = np.nan_to_num(path_npz[f"{prefix}_tx_dir"][point_ids].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    rx = np.nan_to_num(path_npz[f"{prefix}_rx_dir"][point_ids].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    los_clear = path_npz[f"{prefix}_los_clear"][point_ids].astype(np.float32)
    los_hit = np.asarray(path_npz[f"{prefix}_los_hit_count"][point_ids], dtype=np.float32)
    los_t = np.nan_to_num(path_npz[f"{prefix}_los_first_t"][point_ids].astype(np.float32), nan=1.0, posinf=1.0, neginf=0.0)

    h, v, t = flat_to_hvt(comp_ids)
    ph, pv = path_angle_bins(tx, scale)
    hdiff = circular_bin_diff(ph, h[:, None].astype(np.float32), float(MH))
    vdiff = circular_bin_diff(pv, v[:, None].astype(np.float32), float(MV))
    ddiff = np.abs(delay - t[:, None].astype(np.float32))
    valid_f = valid.astype(np.float32)
    # Lower is better.  Path score is only a weak prior and never overrides invalid paths.
    cost = ddiff / 2.6 + hdiff / 3.5 + vdiff / 2.0 - 0.08 * np.clip(score, -4.0, 4.0)
    cost = np.where(valid, cost, 1e6)
    best = np.argmin(cost, axis=1)
    rows = np.arange(len(point_ids))
    has = valid[rows, best]
    best_cost = np.where(has, cost[rows, best], 99.0).astype(np.float32)
    best_delay_diff = np.where(has, ddiff[rows, best], 99.0).astype(np.float32)
    best_h_diff = np.where(has, hdiff[rows, best], 99.0).astype(np.float32)
    best_v_diff = np.where(has, vdiff[rows, best], 99.0).astype(np.float32)
    best_score = np.where(has, score[rows, best], -8.0).astype(np.float32)
    best_type = np.where(has, ptype[rows, best], -1).astype(np.int32)
    best_mat = np.where(has, material[rows, best], -1).astype(np.int32)
    best_excess = np.where(has, excess[rows, best], 0.0).astype(np.float32)
    best_edge = np.where(has, edge[rows, best], 0.0).astype(np.float32)
    best_inc = np.where(has, inc[rows, best], 0.0).astype(np.float32)
    best_tx = np.where(has[:, None], tx[rows, best], 0.0).astype(np.float32)
    best_rx = np.where(has[:, None], rx[rows, best], 0.0).astype(np.float32)

    # Also expose LOS and best reflection delay proximity separately.
    los_valid = valid[:, 0].astype(np.float32)
    los_delay_diff = np.abs(delay[:, 0] - t.astype(np.float32))
    ref_mask = valid & (ptype == 1)
    ref_cost = np.where(ref_mask, cost, 1e6)
    ref_best = np.argmin(ref_cost, axis=1)
    has_ref = ref_mask[rows, ref_best]
    ref_delay_diff = np.where(has_ref, ddiff[rows, ref_best], 99.0).astype(np.float32)
    ref_h_diff = np.where(has_ref, hdiff[rows, ref_best], 99.0).astype(np.float32)
    ref_v_diff = np.where(has_ref, vdiff[rows, ref_best], 99.0).astype(np.float32)
    ref_score = np.where(has_ref, score[rows, ref_best], -8.0).astype(np.float32)

    mat_oh = np.zeros((len(point_ids), MAT_COUNT), dtype=np.float32)
    mat_idx = np.clip(best_mat + 1, 0, MAT_COUNT - 1)
    mat_oh[np.arange(len(point_ids)), mat_idx] = 1.0
    type_oh = np.zeros((len(point_ids), 4), dtype=np.float32)
    type_idx = np.clip(best_type + 1, 0, 3)
    type_oh[np.arange(len(point_ids)), type_idx] = 1.0

    feats = np.concatenate(
        [
            np.column_stack(
                [
                    has.astype(np.float32),
                    best_cost / 12.0,
                    best_delay_diff / 12.0,
                    best_h_diff / 8.0,
                    best_v_diff / 4.0,
                    best_score / 4.0,
                    np.log1p(np.maximum(best_excess, 0.0)) / 7.0,
                    best_edge / 0.35,
                    best_inc,
                    los_clear,
                    los_valid,
                    los_hit / 20.0,
                    np.clip(los_t, 0.0, 1.0),
                    los_delay_diff / 24.0,
                    has_ref.astype(np.float32),
                    ref_delay_diff / 12.0,
                    ref_h_diff / 8.0,
                    ref_v_diff / 4.0,
                    ref_score / 4.0,
                    valid_f.sum(axis=1) / valid_f.shape[1],
                ]
            ).astype(np.float32),
            type_oh,
            mat_oh,
            best_tx,
            best_rx,
        ],
        axis=1,
    )
    return np.nan_to_num(feats, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)


def build_component_dataset(
    base_ad: np.ndarray,
    residual_ad: np.ndarray,
    pos: np.ndarray,
    blocked: np.ndarray,
    path_npz: np.lib.npyio.NpzFile,
    args: argparse.Namespace,
    indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    n = base_ad.shape[0]
    if indices is None:
        indices = np.arange(n, dtype=np.int64)
    pf = point_features(pos, blocked)
    feats: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    coherences: list[np.ndarray] = []
    powers: list[np.ndarray] = []
    for block_start in range(0, len(indices), int(args.train_batch)):
        ids_pt = indices[block_start : block_start + int(args.train_batch)]
        b = np.asarray(base_ad[ids_pt], dtype=np.complex64)
        t = b + np.asarray(residual_ad[ids_pt], dtype=np.complex64)
        comp_power_hvt = np.sum(np.abs(b) ** 2, axis=(1, 4), dtype=np.float64).astype(np.float32)
        total = np.maximum(comp_power_hvt.sum(axis=(1, 2, 3)), TINY).astype(np.float32)
        top = top_ids_from_power(comp_power_hvt, int(args.train_top_components))
        rows_feat = []
        rows_target = []
        rows_weight = []
        rows_coh = []
        rows_power = []
        for r, qid in enumerate(ids_pt):
            comp_ids = top[r]
            ranks = np.arange(len(comp_ids), dtype=np.float32)
            bv = component_vectors(b, r, comp_ids)
            tv = component_vectors(t, r, comp_ids)
            scalar, coh, cpow = compute_oracle_scalar(bv, tv)
            residual = clip_scalar_residual(scalar, float(args.scalar_gain_cap), float(args.scalar_phase_cap))
            frac = cpow / max(float(total[r]), TINY)
            cfeat = component_base_features(comp_ids, ranks, frac, cpow)
            pfeat = np.repeat(pf[qid][None, :], len(comp_ids), axis=0)
            path_feat = path_component_features(path_npz, "train", np.full(len(comp_ids), int(qid), dtype=np.int64), comp_ids, float(args.path_steering_scale))
            rows_feat.append(np.concatenate([pfeat, cfeat, path_feat], axis=1))
            rows_target.append(
                np.column_stack(
                    [
                        residual.real.astype(np.float32),
                        residual.imag.astype(np.float32),
                        np.clip(coh, 0.0, 1.0).astype(np.float32),
                    ]
                )
            )
            # Power and coherence both matter.  Low-coherence labels are not
            # dropped, but they receive a smaller scalar-regression weight.
            w = np.sqrt(np.maximum(frac, 1e-12)) * (0.10 + np.clip(coh, 0.0, 1.0) ** 2)
            rows_weight.append(w.astype(np.float32))
            rows_coh.append(coh.astype(np.float32))
            rows_power.append(frac.astype(np.float32))
        feats.append(np.concatenate(rows_feat, axis=0).astype(np.float32))
        targets.append(np.concatenate(rows_target, axis=0).astype(np.float32))
        weights.append(np.concatenate(rows_weight, axis=0).astype(np.float32))
        coherences.append(np.concatenate(rows_coh, axis=0).astype(np.float32))
        powers.append(np.concatenate(rows_power, axis=0).astype(np.float32))
        print(f"component dataset {block_start + len(ids_pt)}/{len(indices)}", flush=True)
    x = np.concatenate(feats, axis=0).astype(np.float32)
    y = np.concatenate(targets, axis=0).astype(np.float32)
    w = np.concatenate(weights, axis=0).astype(np.float32)
    coh_all = np.concatenate(coherences, axis=0)
    pow_all = np.concatenate(powers, axis=0)
    stats = {
        "examples": int(len(x)),
        "feature_dim": int(x.shape[1]),
        "coherence": summarize(coh_all),
        "power_fraction": summarize(pow_all),
        "residual_abs": summarize(np.abs(y[:, 0] + 1j * y[:, 1])),
        "target_weight": summarize(w),
    }
    return x, y, w, stats


class ComponentScalarNet(nn.Module):
    def __init__(self, in_dim: int, width: int, depth: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers.extend([nn.Linear(d, width), nn.SiLU(), nn.LayerNorm(width)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = width
        layers.append(nn.Linear(d, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        # Keep residual scalar in a conservative interval; reliability is logit.
        out12 = torch.tanh(out[:, :2])
        return torch.cat([out12, out[:, 2:3]], dim=1)


def train_model(x: np.ndarray, y: np.ndarray, w: np.ndarray, args: argparse.Namespace) -> tuple[ComponentScalarNet, np.ndarray, np.ndarray, dict[str, object]]:
    mean = x.mean(axis=0).astype(np.float32)
    std = np.maximum(x.std(axis=0), 1e-3).astype(np.float32)
    x_n = np.nan_to_num((x - mean[None, :]) / std[None, :], nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)
    w_n = w / max(float(np.mean(w)), 1e-12)
    w_n = np.clip(w_n, 0.05, float(args.max_weight)).astype(np.float32)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ComponentScalarNet(x.shape[1], int(args.width), int(args.depth), float(args.dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    ds = TensorDataset(torch.from_numpy(x_n), torch.from_numpy(y.astype(np.float32)), torch.from_numpy(w_n))
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=True, drop_last=False)
    hist: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        scalar_losses = []
        rel_losses = []
        for xb, yb, wb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            pred = model(xb)
            scalar_loss = nn.functional.huber_loss(pred[:, :2], yb[:, :2], reduction="none", delta=float(args.huber_delta)).sum(dim=1)
            rel_target = yb[:, 2].clamp(0, 1)
            rel_loss = nn.functional.binary_cross_entropy_with_logits(pred[:, 2], rel_target, reduction="none")
            loss = (scalar_loss * wb).mean() + float(args.rel_loss_weight) * rel_loss.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            scalar_losses.append(float((scalar_loss * wb).mean().detach().cpu()))
            rel_losses.append(float(rel_loss.mean().detach().cpu()))
        if epoch == 1 or epoch % int(args.log_every) == 0 or epoch == int(args.epochs):
            row = {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "scalar_loss": float(np.mean(scalar_losses)),
                "rel_loss": float(np.mean(rel_losses)),
            }
            hist.append(row)
            print("component train", json.dumps(row, ensure_ascii=False), flush=True)
    return model, mean, std, {"history": hist, "device": str(device)}


@torch.no_grad()
def predict_model(
    model: ComponentScalarNet,
    mean: np.ndarray,
    std: np.ndarray,
    feat: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.nan_to_num((feat.astype(np.float32) - mean[None, :]) / std[None, :], nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)
    device = next(model.parameters()).device
    outs = []
    model.eval()
    for s in range(0, len(x), int(args.infer_batch_size)):
        xb = torch.from_numpy(x[s : s + int(args.infer_batch_size)]).to(device)
        outs.append(model(xb).detach().cpu().numpy())
    pred = np.concatenate(outs, axis=0).astype(np.float32)
    residual = (pred[:, 0] + 1j * pred[:, 1]).astype(np.complex64)
    mag = np.minimum(np.abs(residual), float(args.pred_residual_cap))
    good = np.abs(residual) > 1e-12
    residual[good] = mag[good] * residual[good] / np.abs(residual[good])
    residual[~good] = np.complex64(0.0 + 0.0j)
    reliability = 1.0 / (1.0 + np.exp(-pred[:, 2]))
    return residual.astype(np.complex64), reliability.astype(np.float32)


def build_inference_features(
    ad: np.ndarray,
    pos: np.ndarray,
    blocked: np.ndarray,
    path_npz: np.lib.npyio.NpzFile,
    args: argparse.Namespace,
    prefix: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], dict[str, object]]:
    pf = point_features(pos, blocked)
    ids_all: list[np.ndarray] = []
    feats_all: list[np.ndarray] = []
    frac_all: list[np.ndarray] = []
    comp_power_all: list[np.ndarray] = []
    stats = {"top_frac": [], "component_power": []}
    for s in range(0, ad.shape[0], int(args.project_batch)):
        e = min(s + int(args.project_batch), ad.shape[0])
        a = np.asarray(ad[s:e], dtype=np.complex64)
        power_hvt = np.sum(np.abs(a) ** 2, axis=(1, 4), dtype=np.float64).astype(np.float32)
        total = np.maximum(power_hvt.sum(axis=(1, 2, 3)), TINY).astype(np.float32)
        top = top_ids_from_power(power_hvt, int(args.apply_top_components))
        for r in range(e - s):
            comp_ids = top[r]
            h, v, t = flat_to_hvt(comp_ids)
            cpow = power_hvt[r, h, v, t].astype(np.float32)
            frac = cpow / max(float(total[r]), TINY)
            ranks = np.arange(len(comp_ids), dtype=np.float32)
            cfeat = component_base_features(comp_ids, ranks, frac, cpow)
            pfeat = np.repeat(pf[s + r][None, :], len(comp_ids), axis=0)
            path_feat = path_component_features(path_npz, prefix, np.full(len(comp_ids), s + r, dtype=np.int64), comp_ids, float(args.path_steering_scale))
            feats = np.concatenate([pfeat, cfeat, path_feat], axis=1).astype(np.float32)
            ids_all.append(comp_ids.astype(np.int64))
            feats_all.append(feats)
            frac_all.append(frac.astype(np.float32))
            comp_power_all.append(cpow.astype(np.float32))
            stats["top_frac"].append(float(frac.sum()))
            stats["component_power"].extend(cpow.tolist())
    return ids_all, feats_all, frac_all, comp_power_all, stats


def apply_component_scalar_gate(
    model: ComponentScalarNet,
    mean: np.ndarray,
    std: np.ndarray,
    args: argparse.Namespace,
    pos_test: np.ndarray,
    blocked_test: np.ndarray,
    path_npz: np.lib.npyio.NpzFile,
) -> dict[str, object]:
    base = np.load(args.base, mmap_mode="r")
    beta_s = beta_subcarriers(float(args.beta), float(args.gamma))
    d_test = np.linalg.norm(pos_test - BS_POS[None, :], axis=1)
    regimes = regime_of(blocked_test)
    rho_by_reg = parse_regime_values(args.rho, 0.0)
    out = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.complex64, shape=base.shape)
    all_stats = {
        "applied_count": [],
        "applied_energy_frac": [],
        "mean_gate": [],
        "mean_reliability": [],
        "mean_abs_residual": [],
        "power_ratio": [],
        "delta_num": 0.0,
        "delta_den": 0.0,
    }
    by_regime = {r: {"count": 0, "applied": 0, "energy": 0.0, "applied_energy": 0.0, "gate": []} for r in REGIMES}
    for s in range(0, base.shape[0], int(args.project_batch)):
        e = min(s + int(args.project_batch), base.shape[0])
        n = e - s
        h0 = np.asarray(base[s:e], dtype=np.complex64)
        derot = h0 * np.exp(1j * d_test[s:e, None, None, None] * beta_s).astype(np.complex64)
        ad0 = ad_transform(derot)
        original_power = np.maximum(np.sum(np.abs(ad0) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64), TINY).astype(np.float32)
        power_hvt = np.sum(np.abs(ad0) ** 2, axis=(1, 4), dtype=np.float64).astype(np.float32)
        total_hvt = np.maximum(power_hvt.sum(axis=(1, 2, 3)), TINY).astype(np.float32)
        top = top_ids_from_power(power_hvt, int(args.apply_top_components))
        for r in range(n):
            idx_global = s + r
            comp_ids = top[r]
            h, v, t = flat_to_hvt(comp_ids)
            cpow = power_hvt[r, h, v, t].astype(np.float32)
            frac = cpow / max(float(total_hvt[r]), TINY)
            ranks = np.arange(len(comp_ids), dtype=np.float32)
            cfeat = component_base_features(comp_ids, ranks, frac, cpow)
            pf = point_features(pos_test[idx_global : idx_global + 1], blocked_test[idx_global : idx_global + 1])
            pfeat = np.repeat(pf, len(comp_ids), axis=0)
            path_feat = path_component_features(path_npz, "test", np.full(len(comp_ids), idx_global, dtype=np.int64), comp_ids, float(args.path_steering_scale))
            feat = np.concatenate([pfeat, cfeat, path_feat], axis=1).astype(np.float32)
            residual, rel = predict_model(model, mean, std, feat, args)
            reg = str(regimes[idx_global])
            rho = float(rho_by_reg[reg])
            # Gate is learned reliability, path/component energy, and user-set regime strength.
            energy_gate = np.clip((frac - float(args.frac_floor)) / max(float(args.frac_scale), 1e-8), 0.0, 1.0)
            rel_gate = np.clip((rel - float(args.rel_floor)) / max(float(args.rel_scale), 1e-8), 0.0, 1.0)
            gate = rho * rel_gate * energy_gate
            # Top ranks should be corrected more cautiously because they dominate NMSE if wrong.
            rank_damp = np.where(ranks < 4, float(args.top4_damp), np.where(ranks < 16, float(args.top16_damp), 1.0)).astype(np.float32)
            gate = (gate * rank_damp).astype(np.float32)
            if float(args.max_gate) > 0:
                gate = np.minimum(gate, float(args.max_gate)).astype(np.float32)
            scalar = (1.0 + gate.astype(np.complex64) * residual).astype(np.complex64)
            mag = np.abs(scalar)
            good = mag > 1e-12
            scalar[good] = np.clip(mag[good], 1.0 / float(args.apply_gain_cap), float(args.apply_gain_cap)) * scalar[good] / mag[good]
            scalar[~good] = np.complex64(1.0 + 0.0j)
            phase = np.clip(np.angle(scalar), -float(args.apply_phase_cap), float(args.apply_phase_cap))
            scalar = (np.abs(scalar) * np.exp(1j * phase)).astype(np.complex64)
            for j, cid in enumerate(comp_ids):
                if gate[j] <= 0:
                    continue
                hh, vv, tt = int(h[j]), int(v[j]), int(t[j])
                ad0[r, :, hh, vv, :, tt] *= scalar[j]
            applied = gate > 1e-7
            all_stats["applied_count"].append(int(applied.sum()))
            all_stats["applied_energy_frac"].append(float(frac[applied].sum() if np.any(applied) else 0.0))
            all_stats["mean_gate"].append(float(np.mean(gate)))
            all_stats["mean_reliability"].append(float(np.mean(rel)))
            all_stats["mean_abs_residual"].append(float(np.mean(np.abs(residual))))
            by = by_regime[reg]
            by["count"] += 1
            by["applied"] += int(applied.sum())
            by["energy"] += float(total_hvt[r])
            by["applied_energy"] += float(cpow[applied].sum() if np.any(applied) else 0.0)
            by["gate"].extend(gate.tolist())

        new_power = np.maximum(np.sum(np.abs(ad0) ** 2, axis=(1, 2, 3, 4, 5), dtype=np.float64), TINY).astype(np.float32)
        ad0 *= np.sqrt(original_power / new_power)[:, None, None, None, None, None].astype(np.float32)
        h1 = inverse_ad(ad0).reshape(n, 256, 4, 192)
        h1 = h1 * np.exp(-1j * d_test[s:e, None, None, None] * beta_s).astype(np.complex64)
        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite output block {s}:{e}")
        out[s:e] = h1.astype(np.complex64)
        out.flush()
        all_stats["power_ratio"].extend((np.sum(np.abs(h1) ** 2, axis=(1, 2, 3)) / np.maximum(np.sum(np.abs(h0) ** 2, axis=(1, 2, 3)), TINY)).tolist())
        diff = h1 - h0
        all_stats["delta_num"] += float(np.sum(np.abs(diff) ** 2, dtype=np.float64))
        all_stats["delta_den"] += float(np.sum(np.abs(h0) ** 2, dtype=np.float64))
        print(f"v99 projected {e}/500", flush=True)
    return {
        "overall": {k: summarize(v) for k, v in all_stats.items() if isinstance(v, list)},
        "delta_nmse_to_base": float(all_stats["delta_num"] / max(all_stats["delta_den"], TINY)),
        "by_regime": {
            r: {
                "count": vals["count"],
                "applied": vals["applied"],
                "applied_energy_frac": vals["applied_energy"] / max(vals["energy"], TINY),
                "gate": summarize(vals["gate"]),
            }
            for r, vals in by_regime.items()
        },
    }


def write_zip(args: argparse.Namespace) -> None:
    if not args.zip_out:
        return
    zpath = Path(args.zip_out)
    zpath.parent.mkdir(parents=True, exist_ok=True)
    files = [
        (args.output, "Round1_Test_Channel.npy"),
        (__file__, "generate_v99_learned_component_scalar_gate.py"),
        (ASSETS + "/generate_v87_easy_field_projection.py", "generate_v87_easy_field_projection.py"),
        (ASSETS + "/powerfield_common.py", "powerfield_common.py"),
        (PATHS.code_root / "project_paths.py", "project_paths.py"),
    ]
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as zf:
        for src, name in files:
            zf.write(src, name)
    print(f"wrote {zpath}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PATHS.round1))
    ap.add_argument("--base", default=str(PATHS.cache_dir(99) / "v99_Round1_Test_Channel.npy"))
    ap.add_argument("--output", default=str(PATHS.cache_dir(99) / "v99_Round1_Test_Channel.npy"))
    ap.add_argument("--report", default=ASSETS + "/v99_learned_component_scalar_gate_generate.json")
    ap.add_argument("--zip-out", default=str(PATHS.round1_versions / "Round1_Submission_candidate_v99_learned_component_scalar_gate.zip"))
    ap.add_argument("--train-base-ad", default=ASSETS + "/train_v36_loo_base_derot_ad.npy")
    ap.add_argument("--train-residual-ad", default=ASSETS + "/train_v36_loo_residual_derot_ad.npy")
    ap.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    ap.add_argument("--path-candidates", default=ASSETS + "/sionna_path_candidates.npz")
    ap.add_argument("--seed", type=int, default=9901)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--beta", type=float, default=140.3)
    ap.add_argument("--gamma", type=float, default=0.10)
    ap.add_argument("--path-steering-scale", type=float, default=4.71238898038)
    ap.add_argument("--train-top-components", type=int, default=64)
    ap.add_argument("--apply-top-components", type=int, default=96)
    ap.add_argument("--train-batch", type=int, default=40)
    ap.add_argument("--project-batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--infer-batch-size", type=int, default=65536)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.04)
    ap.add_argument("--lr", type=float, default=9e-4)
    ap.add_argument("--weight-decay", type=float, default=2e-4)
    ap.add_argument("--huber-delta", type=float, default=0.12)
    ap.add_argument("--rel-loss-weight", type=float, default=0.18)
    ap.add_argument("--max-weight", type=float, default=25.0)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--scalar-gain-cap", type=float, default=1.85)
    ap.add_argument("--scalar-phase-cap", type=float, default=1.35)
    ap.add_argument("--pred-residual-cap", type=float, default=0.42)
    ap.add_argument("--rho", default="clear:0.00,light:0.025,medium:0.070,heavy:0.095")
    ap.add_argument("--rel-floor", type=float, default=0.42)
    ap.add_argument("--rel-scale", type=float, default=0.34)
    ap.add_argument("--frac-floor", type=float, default=0.0004)
    ap.add_argument("--frac-scale", type=float, default=0.010)
    ap.add_argument("--top4-damp", type=float, default=0.35)
    ap.add_argument("--top16-damp", type=float, default=0.65)
    ap.add_argument("--max-gate", type=float, default=0.12)
    ap.add_argument("--apply-gain-cap", type=float, default=1.08)
    ap.add_argument("--apply-phase-cap", type=float, default=0.16)
    args = ap.parse_args()

    set_seed(int(args.seed))
    data_dir = Path(args.data)
    pos_train = np.load(data_dir / "Round1_Train_Pos.npy")
    pos_test = np.load(data_dir / "Round1_Test_Pos.npy")
    geo = np.load(args.geometry_cache)
    blocked_train = geo["train_blocked"].astype(np.float64)
    blocked_test = geo["test_blocked"].astype(np.float64)
    base_ad = np.load(args.train_base_ad, mmap_mode="r")
    residual_ad = np.load(args.train_residual_ad, mmap_mode="r")
    path_npz = np.load(args.path_candidates)

    x, y, w, dataset_stats = build_component_dataset(base_ad, residual_ad, pos_train, blocked_train, path_npz, args)
    model, mean, std, train_info = train_model(x, y, w, args)
    proj = apply_component_scalar_gate(model, mean, std, args, pos_test, blocked_test, path_npz)

    out = np.load(args.output, mmap_mode="r")
    payload = {
        "method": "v99_learned_component_scalar_gate",
        "base": args.base,
        "operation": "learned component-level residual complex scalar on top AD components; gated by learned reliability, regime, and component energy; per-point total power restored",
        "args": vars(args),
        "dataset": dataset_stats,
        "train": train_info,
        "projection": proj,
        "output_check": {
            "shape": list(out.shape),
            "dtype": str(out.dtype),
            "finite_real": bool(np.isfinite(out.real).all()),
            "finite_imag": bool(np.isfinite(out.imag).all()),
        },
    }
    Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_zip(args)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
