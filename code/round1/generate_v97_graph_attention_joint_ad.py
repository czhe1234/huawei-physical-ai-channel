#!/usr/bin/env python3
"""Generate v97: graph-attention joint AD target on top of v93.

v95 showed that pushing the geometry-kNN joint AD target too hard over-shoots.
v96 showed that a hand-written agreement gate recovers most of that loss, but
still does not beat v93.  This version replaces the hand-written neighbor
weights with a small learned attention model.

The model is trained on train-point leave-one-out support completion:

    query geometry + candidate neighbor geometry/support summaries
        -> attention weights over candidate neighbors
        -> log-domain weighted joint AD support target

The submission still keeps the v93 complex phase and per-point total power.
Only the normalized collapsed joint AD support P(angle_h, angle_v, delay) is
nudged toward the learned target.
"""

from __future__ import annotations

import argparse
import json
import math
import random
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
    row_cos,
    summarize,
)
from project_paths import PATHS
from powerfield_common import ASSETS, TINY, geom_effective_dist, regime_of


REGIMES = ("clear", "light", "medium", "heavy")


def parse_regime_values(text: str, default: float) -> dict[str, float]:
    out = {r: float(default) for r in REGIMES}
    if not text:
        return out
    for item in text.split(","):
        if not item:
            continue
        key, value = item.split(":")
        key = key.strip()
        if key not in out:
            raise ValueError(f"unknown regime {key!r}")
        out[key] = float(value)
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def normalize_support(coll: np.ndarray, floor: float) -> np.ndarray:
    total = np.maximum(coll.sum(axis=(1, 2, 3), keepdims=True), TINY)
    out = np.asarray(coll / total, dtype=np.float32)
    if floor > 0:
        out = np.maximum(out, floor).astype(np.float32)
        out /= np.maximum(out.sum(axis=(1, 2, 3), keepdims=True), TINY)
    return out


def support_summary(supp: np.ndarray) -> np.ndarray:
    n = supp.shape[0]
    flat = supp.reshape(n, -1)
    beam = supp.sum(axis=-1).reshape(n, -1)
    delay = supp.sum(axis=(1, 2))
    top = np.sort(flat, axis=1)[:, ::-1]
    top1 = top[:, 0]
    top4 = top[:, :4].sum(axis=1)
    top16 = top[:, :16].sum(axis=1)
    top64 = top[:, :64].sum(axis=1)
    ent = -(flat * np.log(flat + 1e-30)).sum(axis=1) / math.log(flat.shape[1])
    beam_ent = -(beam * np.log(beam + 1e-30)).sum(axis=1) / math.log(beam.shape[1])
    delay_ent = -(delay * np.log(delay + 1e-30)).sum(axis=1) / math.log(delay.shape[1])
    beam_max = beam.max(axis=1)
    delay_max = delay.max(axis=1)
    return np.stack(
        [top1, top4, top16, top64, ent, beam_ent, delay_ent, beam_max, delay_max],
        axis=1,
    ).astype(np.float32)


def path_summary(path_npz: np.lib.npyio.NpzFile, prefix: str) -> np.ndarray:
    """Compact per-point Sionna path-candidate summary.

    The attention model should not perform ray tracing at inference time inside
    the zip; it consumes these deterministic candidate features that were
    already generated from the Sionna etoile mesh.
    """
    valid = path_npz[f"{prefix}_valid"].astype(bool)
    ptype = path_npz[f"{prefix}_path_type"]
    delay = np.asarray(path_npz[f"{prefix}_delay_bin"], dtype=np.float32)
    score = np.asarray(path_npz[f"{prefix}_score"], dtype=np.float32)
    excess = np.asarray(path_npz[f"{prefix}_excess_length"], dtype=np.float32)
    edge_margin = np.asarray(path_npz[f"{prefix}_edge_margin"], dtype=np.float32)
    incidence = np.asarray(path_npz[f"{prefix}_incidence_cos"], dtype=np.float32)
    tx_dir = np.asarray(path_npz[f"{prefix}_tx_dir"], dtype=np.float32)
    rx_dir = np.asarray(path_npz[f"{prefix}_rx_dir"], dtype=np.float32)
    los_clear = path_npz[f"{prefix}_los_clear"].astype(np.float32)
    los_hit = np.asarray(path_npz[f"{prefix}_los_hit_count"], dtype=np.float32)
    los_t = np.asarray(path_npz[f"{prefix}_los_first_t"], dtype=np.float32)

    n = valid.shape[0]
    los_valid = (valid[:, 0] & (ptype[:, 0] == 0)).astype(np.float32)
    los_delay = np.where(los_valid > 0, delay[:, 0], 0.0)
    ref_mask = valid & (ptype == 1)
    ref_count = ref_mask.sum(axis=1).astype(np.float32)
    safe_score = np.where(ref_mask, score, -1e9)
    top_ref = np.argmax(safe_score, axis=1)
    rows = np.arange(n)
    has_ref = ref_count > 0
    top_score = np.where(has_ref, safe_score[rows, top_ref], -8.0)
    top_delay = np.where(has_ref, delay[rows, top_ref], 0.0)
    top_excess = np.where(has_ref, excess[rows, top_ref], 0.0)
    top_margin = np.where(has_ref, edge_margin[rows, top_ref], 0.0)
    top_inc = np.where(has_ref, incidence[rows, top_ref], 0.0)
    top_tx = np.where(has_ref[:, None], tx_dir[rows, top_ref], 0.0)
    top_rx = np.where(has_ref[:, None], rx_dir[rows, top_ref], 0.0)

    # First few path delays/scores preserve coarse multi-path structure.
    d = np.where(valid, delay, 0.0)
    s = np.where(valid, np.clip(score, -8.0, 4.0), -8.0)
    first_delays = d[:, :4] / 24.0
    first_scores = s[:, :4] / 4.0

    feats = [
        los_clear,
        los_valid,
        los_hit / 20.0,
        np.clip(los_t, 0.0, 1.0),
        los_delay / 24.0,
        ref_count / 12.0,
        top_score / 4.0,
        top_delay / 24.0,
        np.log1p(np.maximum(top_excess, 0.0)) / 7.0,
        top_margin / 0.35,
        top_inc,
        has_ref.astype(np.float32),
    ]
    out = np.concatenate(
        [np.stack(feats, axis=1), top_tx, top_rx, first_delays, first_scores],
        axis=1,
    ).astype(np.float32)
    return out


def build_neighbor_indices(
    pos_train: np.ndarray,
    blocked_train: np.ndarray,
    pos_query: np.ndarray,
    blocked_query: np.ndarray,
    k: int,
    exclude_self: bool,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(pos_query)
    idx = np.empty((n, k), dtype=np.int64)
    eff_out = np.empty((n, k), dtype=np.float32)
    for i in range(n):
        eff = geom_effective_dist(pos_train, blocked_train, pos_query[i, :2], blocked_query[i]).astype(np.float64)
        if exclude_self:
            same = np.hypot(pos_train[:, 0] - pos_query[i, 0], pos_train[:, 1] - pos_query[i, 1]) < 1e-8
            eff[same] = np.inf
        ids = np.argpartition(eff, k)[:k]
        ids = ids[np.argsort(eff[ids])]
        idx[i] = ids
        eff_out[i] = eff[ids].astype(np.float32)
        if (i + 1) % 250 == 0:
            print(f"neighbors {i+1}/{n}", flush=True)
    return idx, eff_out


def edge_features(
    pos_query: np.ndarray,
    blocked_query: np.ndarray,
    pos_train: np.ndarray,
    blocked_train: np.ndarray,
    neighbor_idx: np.ndarray,
    eff: np.ndarray,
    train_summary: np.ndarray,
    query_path: np.ndarray | None = None,
    train_path: np.ndarray | None = None,
) -> np.ndarray:
    q = pos_query[:, None, :]
    nb_pos = pos_train[neighbor_idx]
    nb_blocked = blocked_train[neighbor_idx]
    dx = nb_pos[..., 0] - q[..., 0]
    dy = nb_pos[..., 1] - q[..., 1]
    d2 = np.hypot(dx, dy)
    q_bs = pos_query - BS_POS[None, :]
    nb_bs = nb_pos - BS_POS[None, None, :]
    q_az = np.arctan2(q_bs[:, 1], q_bs[:, 0])[:, None]
    nb_az = np.arctan2(nb_bs[..., 1], nb_bs[..., 0])
    daz = np.arctan2(np.sin(nb_az - q_az), np.cos(nb_az - q_az))
    q_reg = regime_of(blocked_query)
    nb_reg = regime_of(nb_blocked)
    same_reg = (nb_reg == q_reg[:, None]).astype(np.float32)
    k = neighbor_idx.shape[1]
    q_blocked = np.repeat(blocked_query[:, None], k, axis=1)
    los_q = (blocked_query <= 0).astype(np.float32)[:, None]
    los_nb = (nb_blocked <= 0).astype(np.float32)
    reg_onehot = np.stack([(q_reg == r).astype(np.float32) for r in REGIMES], axis=1)
    reg_edge = np.repeat(reg_onehot[:, None, :], k, axis=1)
    summ = train_summary[neighbor_idx]
    feats = [
        dx / 100.0,
        dy / 100.0,
        d2 / 20.0,
        np.log1p(d2),
        eff / 20.0,
        1.0 / np.maximum(eff, 0.25),
        np.sin(daz),
        np.cos(daz),
        q_blocked / 100.0,
        nb_blocked / 100.0,
        np.abs(nb_blocked - q_blocked) / 100.0,
        same_reg,
        los_q.repeat(k, axis=1),
        los_nb,
    ]
    blocks = [x[..., None].astype(np.float32) for x in feats] + [reg_edge, summ]
    if query_path is not None and train_path is not None:
        qp = np.repeat(query_path[:, None, :], k, axis=1)
        npth = train_path[neighbor_idx]
        blocks.extend([qp.astype(np.float32), npth.astype(np.float32), np.abs(qp - npth).astype(np.float32)])
        # Direction and delay consistency of best reflection candidates.
        q_tx = qp[..., 12:15]
        n_tx = npth[..., 12:15]
        q_rx = qp[..., 15:18]
        n_rx = npth[..., 15:18]
        tx_dot = np.sum(q_tx * n_tx, axis=2, keepdims=True)
        rx_dot = np.sum(q_rx * n_rx, axis=2, keepdims=True)
        delay_diff = np.abs(qp[..., 7:8] - npth[..., 7:8])
        los_delay_diff = np.abs(qp[..., 4:5] - npth[..., 4:5])
        blocks.extend([tx_dot.astype(np.float32), rx_dot.astype(np.float32), delay_diff.astype(np.float32), los_delay_diff.astype(np.float32)])
    out = np.concatenate(blocks, axis=2)
    return out.astype(np.float32)


class AttentionNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, edge: torch.Tensor) -> torch.Tensor:
        return self.net(edge).squeeze(-1)


def cosine_rows(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    num = torch.sum(a * b, dim=1)
    den = torch.sqrt(torch.sum(a * a, dim=1) * torch.sum(b * b, dim=1)).clamp_min(eps)
    return num / den


def support_loss(pred: torch.Tensor, truth: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    b = pred.shape[0]
    joint = cosine_rows(pred, truth)
    pred_hvt = pred.reshape(b, MH, MV, S_SC)
    true_hvt = truth.reshape(b, MH, MV, S_SC)
    pred_beam = pred_hvt.sum(dim=-1).reshape(b, -1)
    true_beam = true_hvt.sum(dim=-1).reshape(b, -1)
    pred_delay = pred_hvt.sum(dim=(1, 2))
    true_delay = true_hvt.sum(dim=(1, 2))
    beam = cosine_rows(pred_beam, true_beam)
    delay = cosine_rows(pred_delay, true_delay)
    score = 0.50 * joint.mean() + 0.25 * beam.mean() + 0.25 * delay.mean()
    loss = -score
    return loss, {
        "joint": float(joint.mean().detach().cpu()),
        "beam": float(beam.mean().detach().cpu()),
        "delay": float(delay.mean().detach().cpu()),
        "score": float(score.detach().cpu()),
    }


def train_attention(
    edge: np.ndarray,
    neighbor_idx: np.ndarray,
    support: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[AttentionNet, dict[str, object]]:
    n, k, f = edge.shape
    model = AttentionNet(f, int(args.hidden), float(args.dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    edge_t = torch.from_numpy(edge.astype(np.float32))
    idx_t = torch.from_numpy(neighbor_idx.astype(np.int64))
    supp = torch.from_numpy(support.reshape(n, -1).astype(np.float32)).to(device)
    log_supp = torch.log(supp.clamp_min(float(args.target_floor)))
    ds = TensorDataset(torch.arange(n, dtype=torch.long))
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=True, drop_last=False)
    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        vals = []
        for (rows_cpu,) in loader:
            rows = rows_cpu.to(device)
            e = edge_t[rows_cpu].to(device)
            nb = idx_t[rows_cpu].to(device)
            scores = model(e)
            w = torch.softmax(scores / float(args.temperature), dim=1)
            pred_log = torch.sum(w[..., None] * log_supp[nb], dim=1)
            pred = torch.exp(pred_log).clamp_min(float(args.target_floor))
            pred = pred / pred.sum(dim=1, keepdim=True).clamp_min(1e-30)
            truth = supp[rows]
            loss, m = support_loss(pred, truth)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            vals.append(m)
        if epoch == 1 or epoch % int(args.log_every) == 0 or epoch == int(args.epochs):
            avg = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
            avg["epoch"] = epoch
            history.append(avg)
            print("train", json.dumps(avg, ensure_ascii=False), flush=True)
    return model, {"history": history, "in_dim": f, "k": k}


@torch.no_grad()
def predict_support(
    model: AttentionNet,
    edge: np.ndarray,
    neighbor_idx: np.ndarray,
    support: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    n = edge.shape[0]
    supp = torch.from_numpy(support.reshape(support.shape[0], -1).astype(np.float32)).to(device)
    log_supp = torch.log(supp.clamp_min(float(args.target_floor)))
    out = np.empty((n, MH, MV, S_SC), dtype=np.float32)
    weights = np.empty((n, neighbor_idx.shape[1]), dtype=np.float32)
    edge_t = torch.from_numpy(edge.astype(np.float32))
    idx_t = torch.from_numpy(neighbor_idx.astype(np.int64))
    for start in range(0, n, int(args.predict_batch)):
        end = min(start + int(args.predict_batch), n)
        e = edge_t[start:end].to(device)
        nb = idx_t[start:end].to(device)
        scores = model(e)
        w = torch.softmax(scores / float(args.temperature), dim=1)
        pred_log = torch.sum(w[..., None] * log_supp[nb], dim=1)
        pred = torch.exp(pred_log).clamp_min(float(args.target_floor))
        pred = pred / pred.sum(dim=1, keepdim=True).clamp_min(1e-30)
        out[start:end] = pred.reshape(end - start, MH, MV, S_SC).cpu().numpy().astype(np.float32)
        weights[start:end] = w.cpu().numpy().astype(np.float32)
    return out, weights


def projection(
    target: np.ndarray,
    weights: np.ndarray,
    regimes: np.ndarray,
    pos_test: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    base = np.load(args.base, mmap_mode="r")
    beta_s = beta_subcarriers(args.beta, args.gamma)
    d_test = np.linalg.norm(pos_test - BS_POS[None, :], axis=1)
    out = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.complex64, shape=base.shape)
    eta_base = parse_regime_values(args.eta, 0.0)
    gain_clip = tuple(float(x) for x in args.gain_clip.split(","))
    top_gain_clip = tuple(float(x) for x in args.top_gain_clip.split(","))
    maxw = weights.max(axis=1)
    entropy = -(weights * np.log(np.maximum(weights, 1e-12))).sum(axis=1) / math.log(weights.shape[1])
    conf = np.clip((maxw - float(args.conf_center)) / max(float(args.conf_width), 1e-6), 0.0, 1.0)
    conf *= np.clip((float(args.entropy_high) - entropy) / max(float(args.entropy_high) - float(args.entropy_low), 1e-6), 0.0, 1.0)
    eta = np.array([eta_base[str(r)] for r in regimes], dtype=np.float32)
    eta *= (float(args.min_gate) + (float(args.max_gate) - float(args.min_gate)) * conf).astype(np.float32)
    stats: dict[str, object] = {
        "eta": eta.tolist(),
        "attention_max": maxw.tolist(),
        "attention_entropy": entropy.tolist(),
        "confidence": conf.tolist(),
        "power_ratio": [],
        "joint_cos_before": [],
        "joint_cos_after": [],
        "pas_cos_to_base": [],
        "pdp_cos_to_base": [],
        "gain_min": [],
        "gain_max": [],
        "delta_num": 0.0,
        "delta_den": 0.0,
    }
    by_regime = {r: {"count": 0, "eta": [], "conf": [], "joint_before": [], "joint_after": []} for r in REGIMES}
    for start in range(0, base.shape[0], int(args.project_batch)):
        end = min(start + int(args.project_batch), base.shape[0])
        n = end - start
        h0 = np.asarray(base[start:end], dtype=np.complex64)
        derot = h0 * np.exp(1j * d_test[start:end, None, None, None] * beta_s).astype(np.complex64)
        ad0 = ad_transform(derot)
        p0 = (np.abs(ad0) ** 2).astype(np.float32)
        coll0 = p0.sum(axis=(1, 4))
        total0 = np.maximum(coll0.sum(axis=(1, 2, 3)), TINY).astype(np.float32)
        cur = coll0 / total0[:, None, None, None]
        tgt = target[start:end].astype(np.float32)
        exponent = eta[start:end].astype(np.float32)
        ratio_hvt = ((tgt + args.target_floor) / (cur + args.target_floor)) ** exponent[:, None, None, None]
        p = p0 * ratio_hvt[:, None, :, :, None, :]
        gain = np.sqrt(p / np.maximum(p0, 1e-30)).astype(np.float32)
        gain = np.clip(gain, gain_clip[0], gain_clip[1])

        flat_power = coll0.reshape(n, -1)
        ktop = min(int(args.top_components), flat_power.shape[1])
        top_ids = np.argpartition(flat_power, -ktop, axis=1)[:, -ktop:]
        top_mask = np.zeros_like(flat_power, dtype=bool)
        top_mask[np.arange(n)[:, None], top_ids] = True
        top_mask = top_mask.reshape(n, MH, MV, S_SC)[:, None, :, :, None, :]
        gain = np.where(top_mask, np.clip(gain, top_gain_clip[0], top_gain_clip[1]), gain)

        ad1 = (ad0 * gain).astype(np.complex64)
        p1 = (np.abs(ad1) ** 2).astype(np.float32)
        total1 = np.maximum(p1.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
        ad1 *= np.sqrt(total0 / total1)[:, None, None, None, None, None].astype(np.float32)
        h1 = inverse_ad(ad1).reshape(n, 256, 4, 192)
        h1 = h1 * np.exp(-1j * d_test[start:end, None, None, None] * beta_s).astype(np.complex64)
        if not (np.isfinite(h1.real).all() and np.isfinite(h1.imag).all()):
            raise FloatingPointError(f"non-finite block {start}:{end}")
        out[start:end] = h1.astype(np.complex64)
        out.flush()

        p1 = (np.abs(ad1) ** 2).astype(np.float32)
        coll1 = p1.sum(axis=(1, 4))
        total_after = np.maximum(coll1.sum(axis=(1, 2, 3)), TINY)
        cur1 = coll1 / total_after[:, None, None, None]
        beam0 = cur.sum(axis=-1)
        beam1 = cur1.sum(axis=-1)
        delay0 = cur.sum(axis=(1, 2))
        delay1 = cur1.sum(axis=(1, 2))
        before = row_cos(cur, tgt)
        after = row_cos(cur1, tgt)
        stats["power_ratio"].extend((total_after / total0).tolist())
        stats["joint_cos_before"].extend(before.tolist())
        stats["joint_cos_after"].extend(after.tolist())
        stats["pas_cos_to_base"].extend(row_cos(beam1, beam0).tolist())
        stats["pdp_cos_to_base"].extend(row_cos(delay1, delay0).tolist())
        stats["gain_min"].extend(np.min(gain.reshape(n, -1), axis=1).tolist())
        stats["gain_max"].extend(np.max(gain.reshape(n, -1), axis=1).tolist())
        diff = h1 - h0
        stats["delta_num"] += float(np.sum(np.abs(diff) ** 2, dtype=np.float64))
        stats["delta_den"] += float(np.sum(np.abs(h0) ** 2, dtype=np.float64))
        for j, reg in enumerate(regimes[start:end]):
            b = by_regime[str(reg)]
            b["count"] += 1
            b["eta"].append(float(exponent[j]))
            b["conf"].append(float(conf[start + j]))
            b["joint_before"].append(float(before[j]))
            b["joint_after"].append(float(after[j]))
        print(f"projected {end}/500", flush=True)
    payload = {
        "eta": summarize(stats["eta"]),
        "attention_max": summarize(stats["attention_max"]),
        "attention_entropy": summarize(stats["attention_entropy"]),
        "confidence": summarize(stats["confidence"]),
        "power_ratio": summarize(stats["power_ratio"]),
        "joint_cos_before": summarize(stats["joint_cos_before"]),
        "joint_cos_after": summarize(stats["joint_cos_after"]),
        "pas_cos_to_base": summarize(stats["pas_cos_to_base"]),
        "pdp_cos_to_base": summarize(stats["pdp_cos_to_base"]),
        "gain_min": summarize(stats["gain_min"]),
        "gain_max": summarize(stats["gain_max"]),
        "delta_nmse_to_base": float(stats["delta_num"] / max(stats["delta_den"], TINY)),
        "by_regime": {r: {k: summarize(v) if isinstance(v, list) else v for k, v in vals.items()} for r, vals in by_regime.items()},
    }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=ASSETS + "/cache_v93/v93_Round1_Test_Channel.npy")
    ap.add_argument("--output", default=str(PATHS.cache_dir(97) / "v97_Round1_Test_Channel.npy"))
    ap.add_argument("--data", default=str(PATHS.round1))
    ap.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    ap.add_argument("--coll", default=ASSETS + "/truth_coll_power.npy")
    ap.add_argument("--report", default=ASSETS + "/v97_graph_attention_joint_ad_generate.json")
    ap.add_argument("--target-out", default=ASSETS + "/cache_v97/v97_graph_attention_target.npy")
    ap.add_argument("--weights-out", default=ASSETS + "/cache_v97/v97_graph_attention_weights.npy")
    ap.add_argument("--seed", type=int, default=9701)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--predict-batch", type=int, default=128)
    ap.add_argument("--project-batch", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--log-every", type=int, default=15)
    ap.add_argument("--target-floor", type=float, default=1e-12)
    ap.add_argument("--eta", default="clear:0.02,light:0.04,medium:0.22,heavy:0.26")
    ap.add_argument("--min-gate", type=float, default=0.10)
    ap.add_argument("--max-gate", type=float, default=1.50)
    ap.add_argument("--conf-center", type=float, default=0.42)
    ap.add_argument("--conf-width", type=float, default=0.28)
    ap.add_argument("--entropy-low", type=float, default=0.25)
    ap.add_argument("--entropy-high", type=float, default=0.88)
    ap.add_argument("--gain-clip", default="0.58,1.82")
    ap.add_argument("--top-gain-clip", default="0.72,1.50")
    ap.add_argument("--top-components", type=int, default=256)
    ap.add_argument("--beta", type=float, default=140.3)
    ap.add_argument("--gamma", type=float, default=0.10)
    args = ap.parse_args()

    set_seed(int(args.seed))
    device = choose_device(args.device)
    print(f"device={device}", flush=True)

    data_dir = Path(args.data)
    pos_train = np.load(data_dir / "Round1_Train_Pos.npy")
    pos_test = np.load(data_dir / "Round1_Test_Pos.npy")
    geo = np.load(args.geometry_cache)
    blocked_train = geo["train_blocked"].astype(np.float64)
    blocked_test = geo["test_blocked"].astype(np.float64)
    regimes = regime_of(blocked_test)
    coll = np.asarray(np.load(args.coll, mmap_mode="r"), dtype=np.float32)
    support = normalize_support(coll, float(args.target_floor))
    train_summary = support_summary(support)

    print("build train neighbors", flush=True)
    train_idx, train_eff = build_neighbor_indices(pos_train, blocked_train, pos_train, blocked_train, int(args.k), True)
    print("build test neighbors", flush=True)
    test_idx, test_eff = build_neighbor_indices(pos_train, blocked_train, pos_test, blocked_test, int(args.k), False)
    train_edge = edge_features(pos_train, blocked_train, pos_train, blocked_train, train_idx, train_eff, train_summary)
    test_edge = edge_features(pos_test, blocked_test, pos_train, blocked_train, test_idx, test_eff, train_summary)

    edge_mean = train_edge.reshape(-1, train_edge.shape[-1]).mean(axis=0, keepdims=True).astype(np.float32)
    edge_std = np.maximum(train_edge.reshape(-1, train_edge.shape[-1]).std(axis=0, keepdims=True), 1e-3).astype(np.float32)
    train_edge_n = (train_edge - edge_mean[None, :, :]) / edge_std[None, :, :]
    test_edge_n = (test_edge - edge_mean[None, :, :]) / edge_std[None, :, :]

    model, train_info = train_attention(train_edge_n.astype(np.float32), train_idx, support, args, device)
    target, weights = predict_support(model, test_edge_n.astype(np.float32), test_idx, support, args, device)

    Path(args.target_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.target_out, target)
    np.save(args.weights_out, weights)

    proj = projection(target, weights, regimes, pos_test, args)
    payload = {
        "method": "v97_graph_attention_joint_ad",
        "base": args.base,
        "operation": "learned graph-attention neighbor weights -> log-domain joint AD target; phase and total power inherited from v93",
        "neighbor_k": int(args.k),
        "train": train_info,
        "projection": {
            "eta_base": parse_regime_values(args.eta, 0.0),
            "gates": {
                "min": float(args.min_gate),
                "max": float(args.max_gate),
                "conf_center": float(args.conf_center),
                "conf_width": float(args.conf_width),
                "entropy_low": float(args.entropy_low),
                "entropy_high": float(args.entropy_high),
            },
            "stats": proj,
        },
        "outputs": {
            "submission_npy": args.output,
            "target": args.target_out,
            "weights": args.weights_out,
        },
    }
    Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
