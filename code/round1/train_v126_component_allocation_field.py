#!/usr/bin/env python3
"""Train and generate v126: point/component allocation field.

v125 assigns one average 2-polarization x 4-UE distribution to each support
template.  This version replaces that coarse average with a learned field.  A
training point contributes one supervised example for every strong H/V/delay
component.  The network receives official point-cloud geometry, receiver
position, current channel summaries, component coordinates/power and the
current eight-way allocation, and predicts the truth eight-way allocation.

The model changes positive magnitudes only.  The energy of every collapsed
H/V/delay component is preserved exactly, which also preserves point total
power and all angle-delay coefficient phases.
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
from torch.utils.data import DataLoader, Dataset

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
)
from power_calibration_lab import metrics_batch
from project_paths import PATHS
from powerfield_common import ASSETS, TINY, regime_of


HVT = MH * MV * S_SC
PORTS = MP * N_ANT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summarize(values: np.ndarray | list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "p10": float(np.quantile(x, 0.10)),
        "median": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def entropy_rows(x: np.ndarray, axis: int = -1) -> np.ndarray:
    p = x / np.maximum(x.sum(axis=axis, keepdims=True), TINY)
    n = x.shape[axis]
    return (-np.sum(p * np.log(np.maximum(p, 1e-30)), axis=axis) / math.log(max(n, 2))).astype(np.float32)


def coordinate_bank() -> np.ndarray:
    ids = np.arange(HVT, dtype=np.int64)
    h = ids // (MV * S_SC)
    rem = ids % (MV * S_SC)
    v = rem // S_SC
    t = rem % S_SC
    feats: list[np.ndarray] = []
    for val, period, frequencies in ((h, MH, (1, 2, 4)), (v, MV, (1, 2, 4)), (t, S_SC, (1, 2, 4, 8))):
        angle = 2.0 * np.pi * val.astype(np.float32) / float(period)
        for freq in frequencies:
            feats.append(np.sin(freq * angle))
            feats.append(np.cos(freq * angle))
    feats.extend([
        h.astype(np.float32) / float(MH - 1),
        v.astype(np.float32) / float(MV - 1),
        t.astype(np.float32) / float(S_SC - 1),
    ])
    return np.stack(feats, axis=1).astype(np.float32)


def static_point_features(pos: np.ndarray, blocked: np.ndarray, scene: np.ndarray) -> np.ndarray:
    rel = pos - BS_POS[None, :]
    dist = np.linalg.norm(rel, axis=1)
    az = np.arctan2(rel[:, 1], rel[:, 0])
    el = np.arcsin(np.clip(rel[:, 2] / np.maximum(dist, 1e-6), -1.0, 1.0))
    x = (pos[:, 0] - 150.0) / 100.0
    y = (pos[:, 1] + 75.0) / 100.0
    feats: list[np.ndarray] = [
        x,
        y,
        rel[:, 2] / 50.0,
        dist / 150.0,
        np.log1p(dist) / 6.0,
        np.sin(az),
        np.cos(az),
        np.sin(el),
        np.cos(el),
        blocked / 150.0,
        np.log1p(np.maximum(blocked, 0.0)) / 6.0,
    ]
    for freq in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        for val in (x, y):
            feats.append(np.sin(np.pi * freq * val))
            feats.append(np.cos(np.pi * freq * val))
    regimes = regime_of(blocked)
    for name in ("clear", "light", "medium", "heavy"):
        feats.append((regimes == name).astype(np.float32))
    return np.concatenate([np.stack(feats, axis=1), scene.astype(np.float32)], axis=1).astype(np.float32)


def robust_standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    med = np.median(train, axis=0).astype(np.float32)
    lo = np.quantile(train, 0.10, axis=0).astype(np.float32)
    hi = np.quantile(train, 0.90, axis=0).astype(np.float32)
    scale = np.maximum(hi - lo, 1e-3).astype(np.float32)
    tr = np.clip((train - med) / scale, -8.0, 8.0).astype(np.float32)
    te = np.clip((test - med) / scale, -8.0, 8.0).astype(np.float32)
    return tr, te, med, scale


def ad_power8(ad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = (ad.real**2 + ad.imag**2).astype(np.float32)
    p8 = p.transpose(0, 2, 3, 5, 1, 4).reshape(len(ad), HVT, PORTS)
    coll = p8.sum(axis=2)
    return p8, coll


def point_channel_summary(p8: np.ndarray, coll: np.ndarray) -> np.ndarray:
    total = np.maximum(coll.sum(axis=1), TINY)
    b = len(coll)
    coll4 = coll.reshape(b, MH, MV, S_SC)
    beam = coll4.sum(axis=3).reshape(b, MH * MV)
    delay = coll4.sum(axis=(1, 2))
    beam /= total[:, None]
    delay /= total[:, None]
    port = p8.sum(axis=1) / total[:, None]
    top = np.sort(coll / total[:, None], axis=1)[:, -16:]
    h_grid = np.arange(MH, dtype=np.float32)[None, :, None]
    v_grid = np.arange(MV, dtype=np.float32)[None, None, :]
    beam2 = beam.reshape(b, MH, MV)
    h_mean = (beam2 * h_grid).sum(axis=(1, 2)) / float(MH - 1)
    v_mean = (beam2 * v_grid).sum(axis=(1, 2)) / float(MV - 1)
    t_grid = np.arange(S_SC, dtype=np.float32)[None, :]
    t_mean = (delay * t_grid).sum(axis=1) / float(S_SC - 1)
    t_var = (delay * (t_grid / float(S_SC - 1) - t_mean[:, None]) ** 2).sum(axis=1)
    return np.concatenate([
        np.log(np.maximum(total, 1e-30))[:, None] / 20.0,
        port,
        entropy_rows(beam)[:, None],
        entropy_rows(delay)[:, None],
        h_mean[:, None],
        v_mean[:, None],
        t_mean[:, None],
        np.sqrt(np.maximum(t_var, 0.0))[:, None],
        top[:, -1:].astype(np.float32),
        top[:, -4:].sum(axis=1, keepdims=True).astype(np.float32),
        top.sum(axis=1, keepdims=True).astype(np.float32),
    ], axis=1).astype(np.float32)


def component_features(
    p8: np.ndarray,
    coll: np.ndarray,
    ids: np.ndarray,
    ranks: np.ndarray,
    coord: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    b, k = ids.shape
    selected_p = np.take_along_axis(p8, ids[:, :, None], axis=1)
    selected_c = np.take_along_axis(coll, ids, axis=1)
    alloc = selected_p / np.maximum(selected_c[:, :, None], 1e-30)
    total = np.maximum(coll.sum(axis=1), 1e-30)
    frac = selected_c / total[:, None]
    max_frac = np.maximum(frac[:, :1], 1e-30)
    coll4 = coll.reshape(b, MH, MV, S_SC)
    beam = coll4.sum(axis=3).reshape(b, MH * MV) / total[:, None]
    delay = coll4.sum(axis=(1, 2)) / total[:, None]
    h = ids // (MV * S_SC)
    rem = ids % (MV * S_SC)
    v = rem // S_SC
    t = rem % S_SC
    hv = h * MV + v
    beam_sel = np.take_along_axis(beam, hv, axis=1)
    delay_sel = np.take_along_axis(delay, t, axis=1)
    scalars = np.stack([
        np.clip(np.log10(np.maximum(frac, 1e-12)) / 12.0 + 1.0, 0.0, 1.0),
        np.clip(np.log10(np.maximum(frac / max_frac, 1e-12)) / 12.0 + 1.0, 0.0, 1.0),
        np.sqrt(np.maximum(frac, 0.0)),
        np.sqrt(np.maximum(beam_sel, 0.0)),
        np.sqrt(np.maximum(delay_sel, 0.0)),
        ranks.astype(np.float32) / max(float(k - 1), 1.0),
    ], axis=2).astype(np.float32)
    cf = np.concatenate([
        coord[ids],
        scalars,
        alloc,
        np.log(np.maximum(alloc, 1e-5)) / 12.0,
    ], axis=2).astype(np.float32)
    return cf, alloc.astype(np.float32)


def build_train_cache(args: argparse.Namespace) -> dict[str, np.ndarray]:
    cache = Path(args.component_cache)
    required = [cache / name for name in (
        "component_features.npy", "targets.npy", "base_alloc.npy", "component_ids.npy",
        "sample_weights.npy", "channel_summary.npy"
    )]
    if all(path.exists() for path in required) and not args.rebuild_component_cache:
        return {path.stem: np.load(path, mmap_mode="r") for path in required}

    cache.mkdir(parents=True, exist_ok=True)
    base_mm = np.load(args.train_base_ad, mmap_mode="r")
    residual_mm = np.load(args.train_residual_ad, mmap_mode="r")
    n = len(base_mm)
    k = int(args.top_components)
    coord = coordinate_bank()
    # Determine feature width from a tiny batch.
    probe = np.array(base_mm[:1], dtype=np.complex64, copy=True)
    p8_probe, coll_probe = ad_power8(probe)
    ids_probe = np.argsort(-coll_probe, axis=1)[:, :k]
    cf_probe, _ = component_features(p8_probe, coll_probe, ids_probe, np.arange(k)[None, :], coord)
    d_comp = cf_probe.shape[2]

    comp = np.lib.format.open_memmap(required[0], mode="w+", dtype=np.float32, shape=(n, k, d_comp))
    targets = np.lib.format.open_memmap(required[1], mode="w+", dtype=np.float32, shape=(n, k, PORTS))
    base_alloc = np.lib.format.open_memmap(required[2], mode="w+", dtype=np.float32, shape=(n, k, PORTS))
    component_ids = np.lib.format.open_memmap(required[3], mode="w+", dtype=np.int32, shape=(n, k))
    weights = np.lib.format.open_memmap(required[4], mode="w+", dtype=np.float32, shape=(n, k))
    summary_dim = point_channel_summary(p8_probe, coll_probe).shape[1]
    summary = np.lib.format.open_memmap(required[5], mode="w+", dtype=np.float32, shape=(n, summary_dim))

    for start in range(0, n, int(args.prep_batch)):
        end = min(start + int(args.prep_batch), n)
        base = np.array(base_mm[start:end], dtype=np.complex64, copy=True)
        truth = base + np.asarray(residual_mm[start:end], dtype=np.complex64)
        pb, cb = ad_power8(base)
        pt, ct = ad_power8(truth)
        ids = np.argpartition(cb, -k, axis=1)[:, -k:]
        vals = np.take_along_axis(cb, ids, axis=1)
        order = np.argsort(-vals, axis=1)
        ids = np.take_along_axis(ids, order, axis=1)
        ranks = np.broadcast_to(np.arange(k, dtype=np.float32)[None, :], ids.shape)
        cf, ba = component_features(pb, cb, ids, ranks, coord)
        tc = np.take_along_axis(ct, ids, axis=1)
        tp = np.take_along_axis(pt, ids[:, :, None], axis=1)
        ta = tp / np.maximum(tc[:, :, None], 1e-30)
        truth_total = np.maximum(ct.sum(axis=1), 1e-30)
        w = np.sqrt(np.maximum(tc / truth_total[:, None] * float(k), 1e-5))
        # Downweight components whose base/truth collapsed energies disagree
        # strongly; allocation is ill-defined when support identity is wrong.
        bc = np.take_along_axis(cb, ids, axis=1)
        support_agree = np.exp(-0.25 * np.abs(np.log((tc + 1e-20) / (bc + 1e-20))))
        w = np.clip(w * support_agree, 0.10, 4.0).astype(np.float32)

        comp[start:end] = cf
        targets[start:end] = ta.astype(np.float32)
        base_alloc[start:end] = ba
        component_ids[start:end] = ids.astype(np.int32)
        weights[start:end] = w
        summary[start:end] = point_channel_summary(pb, cb)
        for arr in (comp, targets, base_alloc, component_ids, weights, summary):
            arr.flush()
        print(f"component cache {end}/{n}", flush=True)

    return {
        "component_features": comp,
        "targets": targets,
        "base_alloc": base_alloc,
        "component_ids": component_ids,
        "sample_weights": weights,
        "channel_summary": summary,
    }


class ComponentDataset(Dataset):
    def __init__(
        self,
        point_features: np.ndarray,
        component_features_arr: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
        flat_indices: np.ndarray,
    ) -> None:
        self.point_features = point_features
        self.component_features = component_features_arr
        self.targets = targets
        self.weights = weights
        self.flat_indices = flat_indices.astype(np.int64)
        self.k = component_features_arr.shape[1]

    def __len__(self) -> int:
        return len(self.flat_indices)

    def __getitem__(self, index: int):
        flat = int(self.flat_indices[index])
        point = flat // self.k
        comp = flat % self.k
        return (
            self.point_features[point],
            self.component_features[point, comp],
            self.targets[point, comp],
            self.weights[point, comp],
        )


class AllocationField(nn.Module):
    def __init__(self, point_dim: int, comp_dim: int, hidden: int, dropout: float, delta_cap: float) -> None:
        super().__init__()
        self.delta_cap = float(delta_cap)
        self.point = nn.Sequential(
            nn.Linear(point_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.comp = nn.Sequential(
            nn.Linear(comp_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, PORTS),
        )

    def forward(self, point: torch.Tensor, comp: torch.Tensor) -> torch.Tensor:
        p = self.point(point)
        c = self.comp(comp)
        raw = self.fuse(torch.cat([p, c, p * c], dim=1))
        delta = self.delta_cap * torch.tanh(raw / self.delta_cap)
        base = torch.clamp(comp[:, -PORTS:], min=-1.0, max=0.0) * 12.0
        return torch.softmax(base + delta, dim=1)


def train_model(
    point_features: np.ndarray,
    cache: dict[str, np.ndarray],
    point_ids: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    epochs: int,
    device: torch.device,
) -> AllocationField:
    set_seed(seed)
    k = cache["component_features"].shape[1]
    flat = (point_ids[:, None] * k + np.arange(k)[None, :]).reshape(-1)
    ds = ComponentDataset(
        point_features,
        cache["component_features"],
        cache["targets"],
        cache["sample_weights"],
        flat,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.workers),
        pin_memory=True,
        persistent_workers=int(args.workers) > 0,
        generator=generator,
    )
    model = AllocationField(
        point_features.shape[1], cache["component_features"].shape[2],
        int(args.hidden), float(args.dropout), float(args.delta_cap),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(epochs), 1), eta_min=float(args.lr) * 0.05)

    for epoch in range(int(epochs)):
        model.train()
        loss_sum = weight_sum = 0.0
        for point, comp, target, weight in loader:
            point = point.to(device, non_blocking=True)
            comp = comp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            pred = model(point, comp)
            ce = -torch.sum(target * torch.log(pred.clamp_min(1e-8)), dim=1)
            cosine = 1.0 - torch.sum(pred * target, dim=1) / (
                torch.linalg.vector_norm(pred, dim=1) * torch.linalg.vector_norm(target, dim=1)
            ).clamp_min(1e-8)
            loss = torch.sum(weight * (ce + float(args.cosine_weight) * cosine)) / weight.sum().clamp_min(1e-8)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += float(loss.detach()) * float(weight.sum())
            weight_sum += float(weight.sum())
        scheduler.step()
        print(f"seed={seed} epoch={epoch + 1}/{epochs} loss={loss_sum / max(weight_sum, 1e-8):.6f}", flush=True)
    return model


@torch.no_grad()
def predict_points(
    model: AllocationField,
    point_features: np.ndarray,
    component_features_arr: np.ndarray,
    point_ids: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    k = component_features_arr.shape[1]
    out = np.empty((len(point_ids), k, PORTS), dtype=np.float32)
    for out_start in range(0, len(point_ids), 8):
        out_end = min(out_start + 8, len(point_ids))
        ids = point_ids[out_start:out_end]
        point = np.repeat(point_features[ids], k, axis=0)
        comp = np.asarray(component_features_arr[ids]).reshape(-1, component_features_arr.shape[2])
        pred_chunks = []
        for start in range(0, len(point), int(batch_size)):
            end = min(start + int(batch_size), len(point))
            pred = model(
                torch.from_numpy(point[start:end]).to(device),
                torch.from_numpy(comp[start:end]).to(device),
            )
            pred_chunks.append(pred.cpu().numpy())
        out[out_start:out_end] = np.concatenate(pred_chunks).reshape(out_end - out_start, k, PORTS)
    return out


def mix_allocation(base: np.ndarray, pred: np.ndarray, strength: float) -> np.ndarray:
    log_mix = (1.0 - float(strength)) * np.log(np.maximum(base, 1e-6))
    log_mix += float(strength) * np.log(np.maximum(pred, 1e-6))
    out = np.exp(log_mix).astype(np.float32)
    out /= np.maximum(out.sum(axis=2, keepdims=True), TINY)
    return out


def apply_to_ad(ad: np.ndarray, component_ids: np.ndarray, allocation: np.ndarray) -> np.ndarray:
    p8, coll = ad_power8(ad)
    b, k = component_ids.shape
    base_p = np.take_along_axis(p8, component_ids[:, :, None], axis=1)
    base_c = np.take_along_axis(coll, component_ids, axis=1)
    base_alloc = base_p / np.maximum(base_c[:, :, None], 1e-30)
    gain = np.sqrt((allocation + 1e-8) / (base_alloc + 1e-8)).astype(np.float32)
    ad8 = ad.transpose(0, 2, 3, 5, 1, 4).reshape(b, HVT, PORTS).copy()
    rows = np.arange(b)[:, None]
    ad8[rows, component_ids] *= gain
    # Exact component-energy restoration after finite floors/clipping.
    changed = ad8[rows, component_ids]
    changed_power = np.maximum((changed.real**2 + changed.imag**2).sum(axis=2), 1e-30)
    changed *= np.sqrt(base_c / changed_power)[:, :, None].astype(np.float32)
    ad8[rows, component_ids] = changed
    return ad8.reshape(b, MH, MV, S_SC, MP, N_ANT).transpose(0, 4, 1, 2, 5, 3).astype(np.complex64)


def score_ad_variants(
    base_mm: np.ndarray,
    residual_mm: np.ndarray,
    point_ids: np.ndarray,
    component_ids: np.ndarray,
    base_alloc: np.ndarray,
    predicted: np.ndarray,
    strengths: list[float],
    device: torch.device,
    batch: int,
) -> dict[str, dict[str, float]]:
    accum = {s: {k: [] for k in ("pas", "pdp", "e2", "t2")} for s in strengths}
    for start in range(0, len(point_ids), int(batch)):
        end = min(start + int(batch), len(point_ids))
        ids = point_ids[start:end]
        base = np.array(base_mm[ids], dtype=np.complex64, copy=True)
        truth = base + np.asarray(residual_mm[ids], dtype=np.complex64)
        for strength in strengths:
            if strength == 0.0:
                one = base
            else:
                alloc = mix_allocation(base_alloc[ids], predicted[start:end], strength)
                one = apply_to_ad(base, component_ids[ids], alloc)
            m = metrics_batch(torch.from_numpy(one).to(device), torch.from_numpy(truth).to(device))
            for key in accum[strength]:
                accum[strength][key].append(m[key])
        print(f"validate projection {end}/{len(point_ids)}", flush=True)
    out: dict[str, dict[str, float]] = {}
    for strength in strengths:
        vals = {k: np.concatenate(v) for k, v in accum[strength].items()}
        nmse = float(vals["e2"].sum() / np.maximum(vals["t2"].sum(), TINY))
        pas = float(vals["pas"].mean())
        pdp = float(vals["pdp"].mean())
        out[str(strength)] = {
            "score": float(0.4 * pas + 0.4 * pdp + 0.2 / (1.0 + nmse)),
            "pas": pas,
            "pdp": pdp,
            "nmse": nmse,
        }
    return out


def build_test_components(
    base_path: str,
    pos_test: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = np.load(base_path, mmap_mode="r")
    n = len(base)
    k = int(args.top_components)
    beta_s = beta_subcarriers(float(args.beta), float(args.gamma))
    d = np.linalg.norm(pos_test - BS_POS[None, :], axis=1)
    coord = coordinate_bank()
    comp = None
    ids_out = np.empty((n, k), dtype=np.int32)
    alloc_out = np.empty((n, k, PORTS), dtype=np.float32)
    summary_out = None
    for start in range(0, n, int(args.prep_batch)):
        end = min(start + int(args.prep_batch), n)
        h = np.asarray(base[start:end], dtype=np.complex64)
        derot = h * np.exp(1j * d[start:end, None, None, None] * beta_s).astype(np.complex64)
        ad = ad_transform(derot)
        p8, coll = ad_power8(ad)
        ids = np.argpartition(coll, -k, axis=1)[:, -k:]
        vals = np.take_along_axis(coll, ids, axis=1)
        order = np.argsort(-vals, axis=1)
        ids = np.take_along_axis(ids, order, axis=1)
        ranks = np.broadcast_to(np.arange(k, dtype=np.float32)[None, :], ids.shape)
        cf, ba = component_features(p8, coll, ids, ranks, coord)
        if comp is None:
            comp = np.empty((n, k, cf.shape[2]), dtype=np.float32)
        comp[start:end] = cf
        ids_out[start:end] = ids.astype(np.int32)
        alloc_out[start:end] = ba
        one_summary = point_channel_summary(p8, coll)
        if summary_out is None:
            summary_out = np.empty((n, one_summary.shape[1]), dtype=np.float32)
        summary_out[start:end] = one_summary
    assert comp is not None and summary_out is not None
    return comp, ids_out, alloc_out, summary_out


def generate_submission(
    models: list[AllocationField],
    point_train: np.ndarray,
    point_test: np.ndarray,
    test_comp: np.ndarray,
    test_ids: np.ndarray,
    test_base_alloc: np.ndarray,
    pos_test: np.ndarray,
    strength: float,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    model_predictions = []
    test_point_ids = np.arange(len(point_test), dtype=np.int64)
    for i, model in enumerate(models):
        pred = predict_points(model, point_test, test_comp, test_point_ids, int(args.predict_batch), device)
        model_predictions.append(pred)
        print(f"predicted test allocation model {i + 1}/{len(models)}", flush=True)
    log_pred = np.mean([np.log(np.maximum(p, 1e-6)) for p in model_predictions], axis=0)
    pred = np.exp(log_pred).astype(np.float32)
    pred /= np.maximum(pred.sum(axis=2, keepdims=True), TINY)
    allocation = mix_allocation(test_base_alloc, pred, strength)

    base = np.load(args.base, mmap_mode="r")
    out = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.complex64, shape=base.shape)
    beta_s = beta_subcarriers(float(args.beta), float(args.gamma))
    d = np.linalg.norm(pos_test - BS_POS[None, :], axis=1)
    delta_num = delta_den = 0.0
    power_ratio: list[float] = []
    for start in range(0, len(base), int(args.project_batch)):
        end = min(start + int(args.project_batch), len(base))
        h0 = np.asarray(base[start:end], dtype=np.complex64)
        derot = h0 * np.exp(1j * d[start:end, None, None, None] * beta_s).astype(np.complex64)
        ad0 = ad_transform(derot)
        ad1 = apply_to_ad(ad0, test_ids[start:end], allocation[start:end])
        h1 = inverse_ad(ad1).reshape(end - start, 256, 4, 192)
        h1 *= np.exp(-1j * d[start:end, None, None, None] * beta_s).astype(np.complex64)
        out[start:end] = h1.astype(np.complex64)
        out.flush()
        diff = h1 - h0
        delta_num += float(np.sum(np.abs(diff) ** 2, dtype=np.float64))
        delta_den += float(np.sum(np.abs(h0) ** 2, dtype=np.float64))
        power_ratio.extend((np.sum(np.abs(h1) ** 2, axis=(1, 2, 3)) / np.maximum(np.sum(np.abs(h0) ** 2, axis=(1, 2, 3)), TINY)).tolist())
        print(f"generate {args.method_name} {end}/{len(base)}", flush=True)
    return {
        "ensemble_models": len(models),
        "strength": float(strength),
        "delta_nmse_to_v125": float(delta_num / max(delta_den, TINY)),
        "power_ratio": summarize(power_ratio),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PATHS.round1))
    ap.add_argument("--method-name", default="v126_component_allocation_field")
    ap.add_argument("--scene-cache", default=ASSETS + "/official_pointcloud_scene_cache.npz")
    ap.add_argument("--geometry-cache", default=ASSETS + "/height_geometry_cache.npz")
    ap.add_argument("--selection", default=ASSETS + "/feature_matched_candidates_500.json")
    ap.add_argument("--train-base-ad", default=ASSETS + "/train_v98like_loo_base_ad.npy")
    ap.add_argument("--train-residual-ad", default=ASSETS + "/train_v98like_loo_residual_ad.npy")
    ap.add_argument("--component-cache", default=ASSETS + "/cache_v126/component_training")
    ap.add_argument("--base", default=ASSETS + "/cache_v125/v125_Round1_Test_Channel.npy")
    ap.add_argument("--output", default=ASSETS + "/cache_v126/v126_Round1_Test_Channel.npy")
    ap.add_argument("--report", default=ASSETS + "/v126_component_allocation_field_generate.json")
    ap.add_argument("--model-out", default=ASSETS + "/cache_v126/v126_component_allocation_field.pt")
    ap.add_argument("--zip-out", default=str(PATHS.round1_versions / "v126_component_allocation_field_candidate_pending.zip"))
    ap.add_argument("--rebuild-component-cache", action="store_true")
    ap.add_argument("--top-components", type=int, default=512)
    ap.add_argument("--prep-batch", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--dropout", type=float, default=0.04)
    ap.add_argument("--delta-cap", type=float, default=3.0)
    ap.add_argument("--cosine-weight", type=float, default=0.50)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--final-epochs", type=int, default=18)
    ap.add_argument("--ensemble", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--predict-batch", type=int, default=32768)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=2e-4)
    ap.add_argument("--strengths", default="0,0.25,0.50,0.75,1.0,1.25")
    ap.add_argument("--min-submit-strength", type=float, default=0.65)
    ap.add_argument("--aggressive-multiplier", type=float, default=1.20)
    ap.add_argument("--beta", type=float, default=140.3)
    ap.add_argument("--gamma", type=float, default=0.10)
    ap.add_argument("--project-batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12601)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    data = Path(args.data)
    pos_train = np.load(data / "Round1_Train_Pos.npy")
    pos_test = np.load(data / "Round1_Test_Pos.npy")
    scene = np.load(args.scene_cache, allow_pickle=True)
    geo = np.load(args.geometry_cache)
    blocked_train = geo["train_blocked"].astype(np.float32)
    blocked_test = geo["test_blocked"].astype(np.float32)

    cache = build_train_cache(args)
    test_comp, test_ids, test_base_alloc, test_summary = build_test_components(args.base, pos_test, args)
    static_train = static_point_features(pos_train, blocked_train, scene["train_features"])
    static_test = static_point_features(pos_test, blocked_test, scene["test_features"])
    point_train_raw = np.concatenate([static_train, np.asarray(cache["channel_summary"])], axis=1)
    point_test_raw = np.concatenate([static_test, test_summary], axis=1)
    point_train, point_test, point_med, point_scale = robust_standardize(point_train_raw, point_test_raw)

    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    val_ids = np.asarray(selection["val_indices"], dtype=np.int64)
    fit_mask = np.ones(len(pos_train), dtype=bool)
    fit_mask[val_ids] = False
    fit_ids = np.flatnonzero(fit_mask)
    cv_model = train_model(point_train, cache, fit_ids, args, int(args.seed), int(args.epochs), device)
    val_pred = predict_points(cv_model, point_train, cache["component_features"], val_ids, int(args.predict_batch), device)
    strengths = [float(x) for x in args.strengths.split(",")]
    base_mm = np.load(args.train_base_ad, mmap_mode="r")
    residual_mm = np.load(args.train_residual_ad, mmap_mode="r")
    validation = score_ad_variants(
        base_mm, residual_mm, val_ids, cache["component_ids"], cache["base_alloc"],
        val_pred, strengths, device, int(args.project_batch),
    )
    best_strength = max(strengths, key=lambda x: validation[str(x)]["score"])
    # This is intentionally a breakthrough candidate rather than a tiny CV
    # shrink.  Use the validated direction and push one controlled step past
    # the local optimum, while capping at the tested 1.25 endpoint.
    submit_strength = min(1.25, max(float(args.min_submit_strength), best_strength * float(args.aggressive_multiplier)))
    print(json.dumps({"validation": validation, "best_strength": best_strength, "submit_strength": submit_strength}, indent=2), flush=True)

    all_ids = np.arange(len(pos_train), dtype=np.int64)
    models: list[AllocationField] = []
    states = []
    for i in range(int(args.ensemble)):
        model = train_model(
            point_train, cache, all_ids, args, int(args.seed) + 97 * (i + 1),
            int(args.final_epochs), device,
        )
        models.append(model)
        states.append({k: v.detach().cpu() for k, v in model.state_dict().items()})
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "states": states,
        "point_dim": point_train.shape[1],
        "comp_dim": cache["component_features"].shape[2],
        "point_median": point_med,
        "point_scale": point_scale,
        "args": vars(args),
    }, args.model_out)

    generation = generate_submission(
        models, point_train, point_test, test_comp, test_ids, test_base_alloc,
        pos_test, submit_strength, args, device,
    )
    payload = {
        "method": args.method_name,
        "base": args.base,
        "operation": (
            "point/component neural field predicts the 2-pol x 4-UE allocation of each top H/V/delay component "
            "from official point-cloud geometry, position, current channel summary and component state; preserve "
            "collapsed support, total power and AD phase"
        ),
        "args": vars(args),
        "train_points": int(len(pos_train)),
        "samples_per_point": int(args.top_components),
        "validation": validation,
        "validation_best_strength": float(best_strength),
        "submit_strength": float(submit_strength),
        "generation": generation,
        "outputs": {"submission_npy": args.output, "model": args.model_out},
    }
    Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    files = [
        (args.output, "Round1_Test_Channel.npy"),
        (__file__, f"train_{args.method_name}.py"),
        (ASSETS + "/build_v125_equivalent_crossfit_base.py", "build_v125_equivalent_crossfit_base.py"),
        (ASSETS + "/generate_v87_easy_field_projection.py", "generate_v87_easy_field_projection.py"),
        (ASSETS + "/power_calibration_lab.py", "power_calibration_lab.py"),
        (ASSETS + "/powerfield_common.py", "powerfield_common.py"),
        (PATHS.code_root / "project_paths.py", "project_paths.py"),
    ]
    Path(args.zip_out).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_out, "w", compression=zipfile.ZIP_STORED) as zf:
        for src, arc in files:
            zf.write(src, arc)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {args.zip_out}", flush=True)


if __name__ == "__main__":
    main()
