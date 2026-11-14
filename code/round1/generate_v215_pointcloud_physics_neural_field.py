#!/usr/bin/env python3
"""Train and generate v215, a point-cloud Physical-AI correction field.

The Sionna-backed v212 result is used only as experimental motivation: it
showed that delay/PDP structure is a major error source.  This implementation
does not read v210-v214 outputs, Sionna channels, restored meshes, object IDs,
or material labels.

The neural model is supervised by the released training channels.  It encodes
official point-cloud scene features and official-point-cloud LOS/reflection
candidate tokens with a Transformer, then predicts denoised PAS/PDP marginal
fields and total power.  A deterministic differentiable-style decoder imposes
the learned fields on the pre-Sionna v209 AI backbone while retaining its fine
complex phase texture.
"""

from __future__ import annotations

import json
import math
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

import generate_v178_test_track_complex_consensus as track
import generate_v196_train_guided_delay_pdp_mouth as v196
from evaluate_derotated_loo_grid import choose_val_indices
from generate_v87_easy_field_projection import MH, MP, MV, N_ANT, S_SC, summarize
from generate_v159_frontier_consensus_support_soup import digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION = "v215_pointcloud_physics_neural_field"
VERSION_NUMBER = 215

OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v215_Round1_Test_Channel.npy"
ACTION_OUT = OUT_DIR / "v215_pointcloud_physics_neural_actions.npz"
CHECKPOINT = PATHS.checkpoints / "v215_pointcloud_physics_neural_field.pt"
REPORT = REPORT_DIR / "v215_pointcloud_physics_neural_field_generate.json"
ZIP_OUT = PATHS.round1_versions / "v215_pointcloud_physics_neural_field_candidate_pending.zip"

BASE = PATHS.cache_dir(209) / "v209_Round1_Test_Channel.npy"
TRAIN_MARGINALS = PATHS.cache_dir(196) / "v196_train_delay_marginals.npz"
TRAIN_COLL = PATHS.shared_features / "truth_coll_power.npy"
SCENE_CACHE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"
GEODESIC_CACHE = PATHS.shared_features / "geodesic_support_features.npz"
PATH_CACHE = PATHS.shared_features / "multibounce_path_candidates.npz"
GEOM_CACHE = PATHS.shared_features / "height_geometry_cache.npz"

FLOOR = 1e-10
PATH_COUNT = 15
ANCHOR_K = 72
VAL_COUNT = 300
SEEDS = (21501, 21517, 21533)
PCA_DIMS = {"beam": 32, "d": 32, "du": 48, "dh": 64}
TRAIN_EPOCHS = 180
FINAL_EPOCHS = 180
BATCH_SIZE = 128
IPF_ITERS = 5
OUTER_ITERS = 2


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize(x: np.ndarray, axis: tuple[int, ...] | int) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), FLOOR)
    out /= np.maximum(out.sum(axis=axis, keepdims=True), TINY)
    return out.astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = a.reshape(len(a), -1).astype(np.float64)
    y = b.reshape(len(b), -1).astype(np.float64)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def subtract_uniform_floor(x: np.ndarray, strength: float = 0.90) -> tuple[np.ndarray, np.ndarray]:
    """Remove a robust white-noise floor along delay and renormalize."""
    arr = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    floor = np.median(arr, axis=-1, keepdims=True).astype(np.float32)
    clean = np.maximum(arr - strength * floor, 0.0).astype(np.float32)
    axes = tuple(range(1, clean.ndim))
    total = clean.sum(axis=axes, keepdims=True)
    fallback = total.reshape(len(clean)) <= 1e-20
    if np.any(fallback):
        clean[fallback] = arr[fallback]
    return normalize(clean, axis=axes), floor.squeeze(-1).astype(np.float32)


def load_targets() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    delay = np.load(TRAIN_MARGINALS)
    raw = {
        "d": normalize(delay["d"], axis=1),
        "du": normalize(delay["du"], axis=(1, 2)),
        "dh": normalize(delay["dh"], axis=(1, 2)),
    }
    clean_d, d_floor = subtract_uniform_floor(raw["d"], 0.90)
    clean_du, _ = subtract_uniform_floor(raw["du"], 0.90)
    clean_dh, _ = subtract_uniform_floor(raw["dh"], 0.90)

    coll = np.load(TRAIN_COLL, mmap_mode="r")
    beam = np.empty((len(coll), MH * MV), dtype=np.float32)
    beam_raw = np.empty_like(beam)
    for start in range(0, len(coll), 50):
        end = min(start + 50, len(coll))
        block = np.asarray(coll[start:end], dtype=np.float32)
        raw_total = np.maximum(block.sum(axis=(1, 2, 3)), TINY)
        beam_raw[start:end] = normalize(block.sum(axis=3).reshape(end - start, -1), axis=1)
        noise = np.median(block.reshape(end - start, -1), axis=1).astype(np.float32)
        clean = np.maximum(block - 0.90 * noise[:, None, None, None], 0.0)
        clean_total = clean.sum(axis=(1, 2, 3))
        bad = clean_total <= 1e-20
        if np.any(bad):
            clean[bad] = block[bad]
        beam[start:end] = normalize(clean.sum(axis=3).reshape(end - start, -1), axis=1)
        del raw_total

    valid = delay["valid"].astype(bool) & (delay["total_power"].astype(np.float64) > 1e-20)
    removed_fraction = np.clip(0.90 * d_floor * S_SC, 0.0, 0.97).astype(np.float32)
    clean_power = np.maximum(delay["total_power"].astype(np.float64) * (1.0 - removed_fraction), 1e-30)
    log_power = np.log(clean_power).astype(np.float32)
    clean = {"beam": beam, "d": clean_d, "du": clean_du, "dh": clean_dh}
    raw["beam"] = beam_raw
    return clean, raw, log_power, valid


def effective_distance(
    query_pos: np.ndarray,
    query_blocked: np.ndarray,
    bank_pos: np.ndarray,
    bank_blocked: np.ndarray,
) -> np.ndarray:
    dx = query_pos[:, None, 0] - bank_pos[None, :, 0]
    dy = query_pos[:, None, 1] - bank_pos[None, :, 1]
    dist = np.hypot(dx, dy)
    q_los = query_blocked[:, None] <= 0
    b_los = bank_blocked[None, :] <= 0
    factor = 1.0 + 0.90 * (q_los != b_los) + 0.45 * np.abs(
        query_blocked[:, None] - bank_blocked[None, :]
    ) / 80.0
    return (dist * factor).astype(np.float32)


@dataclass
class Anchors:
    beam: np.ndarray
    d: np.ndarray
    du: np.ndarray
    dh: np.ndarray
    log_power: np.ndarray
    agreement: np.ndarray
    effective_count: np.ndarray


def build_anchors(
    query_pos: np.ndarray,
    query_blocked: np.ndarray,
    bank_pos: np.ndarray,
    bank_blocked: np.ndarray,
    bank_targets: dict[str, np.ndarray],
    bank_log_power: np.ndarray,
    query_ids: np.ndarray | None = None,
    bank_ids: np.ndarray | None = None,
) -> Anchors:
    dist = effective_distance(query_pos, query_blocked, bank_pos, bank_blocked)
    if query_ids is not None and bank_ids is not None:
        same = query_ids[:, None] == bank_ids[None, :]
        dist[same] = np.inf
    k = min(ANCHOR_K, len(bank_pos) - 1 if query_ids is not None else len(bank_pos))
    idx = np.argpartition(dist, k - 1, axis=1)[:, :k]
    dsel = np.take_along_axis(dist, idx, axis=1)
    order = np.argsort(dsel, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    dsel = np.take_along_axis(dsel, order, axis=1)
    weights = 1.0 / np.maximum(dsel, 0.30) ** 2
    same_regime = regime_of(query_blocked)[:, None] == regime_of(bank_blocked)[idx]
    weights *= np.where(same_regime, 1.25, 0.72).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), TINY)

    values: dict[str, np.ndarray] = {}
    for key, target in bank_targets.items():
        selected = target[idx]
        shape = (len(query_pos), k) + (1,) * (selected.ndim - 2)
        values[key] = normalize(
            np.sum(selected * weights.reshape(shape), axis=1),
            axis=tuple(range(1, target.ndim)),
        )
    log_power = np.sum(bank_log_power[idx] * weights, axis=1).astype(np.float32)
    agreement = np.sum(
        weights * np.exp(-np.abs(bank_log_power[idx] - log_power[:, None])), axis=1
    ).astype(np.float32)
    effective_count = (1.0 / np.maximum(np.sum(weights**2, axis=1), TINY)).astype(np.float32)
    return Anchors(
        beam=values["beam"],
        d=values["d"],
        du=values["du"],
        dh=values["dh"],
        log_power=log_power,
        agreement=agreement,
        effective_count=effective_count,
    )


def d_stats(d: np.ndarray) -> np.ndarray:
    idx = np.arange(S_SC, dtype=np.float32)
    mean_raw = np.sum(d * idx[None, :], axis=1)
    var = np.sum(d * (idx[None, :] - mean_raw[:, None]) ** 2, axis=1)
    entropy = -np.sum(d * np.log(np.maximum(d, FLOOR)), axis=1) / math.log(S_SC)
    return np.stack(
        [
            np.argmax(d, axis=1).astype(np.float32) / (S_SC - 1),
            mean_raw / (S_SC - 1),
            np.sqrt(np.maximum(var, 0.0)) / (S_SC - 1),
            entropy,
            np.max(d, axis=1),
            d[:, :24].sum(axis=1),
            d[:, 24:72].sum(axis=1),
            d[:, 72:].sum(axis=1),
        ],
        axis=1,
    ).astype(np.float32)


def position_features(pos: np.ndarray) -> np.ndarray:
    bs = np.asarray([50.0, 0.0, 25.0], dtype=np.float32)
    rel = pos.astype(np.float32) - bs[None, :]
    dist = np.linalg.norm(rel, axis=1)
    az = np.arctan2(rel[:, 1], rel[:, 0])
    el = np.arcsin(np.clip(rel[:, 2] / np.maximum(dist, 1e-6), -1.0, 1.0))
    x = (pos[:, 0].astype(np.float32) - 150.0) / 120.0
    y = (pos[:, 1].astype(np.float32) + 75.0) / 120.0
    blocks = [
        pos.astype(np.float32),
        rel / 250.0,
        (dist / 250.0)[:, None],
        np.stack([np.sin(az), np.cos(az), np.sin(el), np.cos(el)], axis=1),
    ]
    fourier = []
    for freq in (1.0, 2.0, 4.0, 8.0, 16.0):
        for value in (x, y):
            fourier.extend([np.sin(np.pi * freq * value), np.cos(np.pi * freq * value)])
    blocks.append(np.stack(fourier, axis=1).astype(np.float32))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def make_global_features(
    prefix: str,
    indices: np.ndarray | None,
    pos: np.ndarray,
    blocked: np.ndarray,
    anchors: Anchors,
) -> np.ndarray:
    scene = np.load(SCENE_CACHE)[f"{prefix}_features"].astype(np.float32)
    geodesic = np.load(GEODESIC_CACHE)[f"{prefix}_features"].astype(np.float32)
    if indices is not None:
        scene = scene[indices]
        geodesic = geodesic[indices]
    regimes = regime_of(blocked)
    onehot = np.stack(
        [(regimes.astype(str) == name).astype(np.float32) for name in ("clear", "light", "medium", "heavy")],
        axis=1,
    )
    features = np.concatenate(
        [
            position_features(pos),
            blocked[:, None].astype(np.float32) / 160.0,
            onehot,
            np.nan_to_num(scene, nan=0.0, posinf=8.0, neginf=-8.0),
            np.nan_to_num(geodesic, nan=0.0, posinf=8.0, neginf=-8.0),
            d_stats(anchors.d),
            anchors.log_power[:, None],
            anchors.agreement[:, None],
            (anchors.effective_count / ANCHOR_K)[:, None],
        ],
        axis=1,
    )
    return np.nan_to_num(features, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)


def make_path_tokens(
    prefix: str,
    pos: np.ndarray,
    indices: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = np.load(PATH_CACHE)
    def take(name: str) -> np.ndarray:
        value = paths[f"{prefix}_{name}"]
        return value if indices is None else value[indices]

    valid = take("valid").astype(bool)
    ptype = take("path_type").astype(np.int64)
    length = np.nan_to_num(take("path_length").astype(np.float32), nan=0.0, posinf=500.0, neginf=0.0)
    delay = np.nan_to_num(take("delay_bin").astype(np.float32), nan=0.0, posinf=32.0, neginf=0.0)
    score = np.nan_to_num(take("score").astype(np.float32), nan=-4.0, posinf=4.0, neginf=-4.0)
    excess = np.nan_to_num(take("excess_length").astype(np.float32), nan=0.0, posinf=500.0, neginf=0.0)
    edge = np.nan_to_num(take("edge_margin").astype(np.float32), nan=0.0, posinf=0.5, neginf=0.0)
    inc = np.nan_to_num(take("incidence_cos").astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    tx = np.nan_to_num(take("tx_dir").astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    rx = np.nan_to_num(take("rx_dir").astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ref1 = np.nan_to_num(take("reflect_point").astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ref2 = np.nan_to_num(take("reflect_point_2").astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    rel1 = np.where(valid[..., None], (ref1 - pos[:, None, :]) / 350.0, 0.0)
    rel2 = np.where(valid[..., None], (ref2 - pos[:, None, :]) / 350.0, 0.0)
    numeric = np.concatenate(
        [
            valid[..., None].astype(np.float32),
            (length / 400.0)[..., None],
            (delay / 24.0)[..., None],
            (score / 4.0)[..., None],
            (np.log1p(excess) / 7.0)[..., None],
            (np.clip(edge, 0.0, 0.5) / 0.5)[..., None],
            inc[..., None],
            tx,
            rx,
            rel1,
            rel2,
        ],
        axis=2,
    ).astype(np.float32)
    type_id = np.where(valid, np.clip(ptype + 1, 1, 3), 0).astype(np.int64)
    return np.nan_to_num(numeric, nan=0.0, posinf=4.0, neginf=-4.0), type_id, valid


@dataclass
class Basis:
    mean: np.ndarray
    components: np.ndarray
    coeff_scale: np.ndarray
    shape: tuple[int, ...]


def fit_basis(anchor: np.ndarray, target: np.ndarray, components: int) -> tuple[Basis, np.ndarray]:
    n = len(anchor)
    residual = np.clip(
        np.log(np.maximum(target.reshape(n, -1), FLOOR))
        - np.log(np.maximum(anchor.reshape(n, -1), FLOOR)),
        -8.0,
        8.0,
    ).astype(np.float32)
    pca = PCA(n_components=min(components, n - 1, residual.shape[1]), svd_solver="randomized", random_state=215)
    coeff = pca.fit_transform(residual).astype(np.float32)
    scale = np.maximum(coeff.std(axis=0), 1e-4).astype(np.float32)
    return Basis(
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        coeff_scale=scale,
        shape=target.shape[1:],
    ), (coeff / scale[None, :]).astype(np.float32)


class PhysicsPathField(nn.Module):
    def __init__(self, global_dim: int, path_dim: int, output_dims: dict[str, int]):
        super().__init__()
        d_model = 128
        self.type_emb = nn.Embedding(4, 16)
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, 256), nn.SiLU(), nn.LayerNorm(256), nn.Linear(256, d_model)
        )
        self.path_proj = nn.Sequential(
            nn.Linear(path_dim + 16, d_model), nn.SiLU(), nn.LayerNorm(d_model), nn.Linear(d_model, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=384,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.trunk = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 256), nn.SiLU(), nn.Dropout(0.08), nn.Linear(256, 192), nn.SiLU()
        )
        self.heads = nn.ModuleDict({key: nn.Linear(192, dim) for key, dim in output_dims.items()})
        self.power_head = nn.Linear(192, 1)

    def forward(
        self,
        global_features: torch.Tensor,
        path_numeric: torch.Tensor,
        path_type: torch.Tensor,
        path_valid: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        query = self.global_proj(global_features).unsqueeze(1)
        path = self.path_proj(torch.cat([path_numeric, self.type_emb(path_type)], dim=2))
        tokens = torch.cat([query, path], dim=1)
        mask = torch.cat(
            [torch.zeros((len(path_valid), 1), dtype=torch.bool, device=path_valid.device), ~path_valid], dim=1
        )
        encoded = self.encoder(tokens, src_key_padding_mask=mask)
        latent = self.trunk(encoded[:, 0])
        return {key: head(latent) for key, head in self.heads.items()}, self.power_head(latent).squeeze(1)


@dataclass
class Prepared:
    global_features: np.ndarray
    path_numeric: np.ndarray
    path_type: np.ndarray
    path_valid: np.ndarray
    anchors: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]
    target_coeff: dict[str, np.ndarray]
    anchor_log_power: np.ndarray
    target_log_power: np.ndarray


def standardize_features(
    train_global: np.ndarray,
    train_path: np.ndarray,
    train_valid: np.ndarray,
    *others: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    g_mean = train_global.mean(axis=0).astype(np.float32)
    g_std = np.maximum(train_global.std(axis=0), 1e-5).astype(np.float32)
    valid_values = train_path[train_valid]
    p_mean = valid_values.mean(axis=0).astype(np.float32)
    p_std = np.maximum(valid_values.std(axis=0), 1e-5).astype(np.float32)

    def transform(g: np.ndarray, p: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gn = np.clip((g - g_mean) / g_std, -8.0, 8.0).astype(np.float32)
        pn = np.clip((p - p_mean[None, None, :]) / p_std[None, None, :], -8.0, 8.0).astype(np.float32)
        pn[~valid] = 0.0
        return gn, pn, valid

    train_g, train_p, _ = transform(train_global, train_path, train_valid)
    converted = [transform(g, p, valid) for g, p, valid in others]
    stats = {"global_mean": g_mean, "global_std": g_std, "path_mean": p_mean, "path_std": p_std}
    return train_g, train_p, converted, stats


def torch_basis(basis: Basis, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(basis.mean).to(device),
        torch.from_numpy(basis.components).to(device),
        torch.from_numpy(basis.coeff_scale).to(device),
    )


def reconstruct_shape(
    coeff_norm: torch.Tensor,
    anchor: torch.Tensor,
    basis_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    mean, components, scale = basis_tensors
    delta = (coeff_norm * scale[None, :]) @ components + mean[None, :]
    logits = torch.log(anchor.reshape(len(anchor), -1).clamp_min(FLOOR)) + delta
    return torch.softmax(logits, dim=1)


def cosine_loss(pred: torch.Tensor, truth: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = 1.0 - F.cosine_similarity(pred, truth.reshape(len(truth), -1), dim=1)
    return torch.sum(loss * weight) / weight.sum().clamp_min(1e-6)


def train_model(
    prepared: Prepared,
    bases: dict[str, Basis],
    epochs: int,
    seed: int,
    device: torch.device,
    validation: Prepared | None = None,
) -> tuple[PhysicsPathField, dict[str, object]]:
    set_seed(seed)
    model = PhysicsPathField(
        prepared.global_features.shape[1],
        prepared.path_numeric.shape[2],
        {key: len(bases[key].coeff_scale) for key in bases},
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    basis_gpu = {key: torch_basis(value, device) for key, value in bases.items()}
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    best_metric = -np.inf
    best_epoch = epochs
    best_state: dict[str, torch.Tensor] | None = None
    n = len(prepared.global_features)

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(n)
        running = []
        for start in range(0, n, BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            g = torch.from_numpy(prepared.global_features[ids]).to(device)
            p = torch.from_numpy(prepared.path_numeric[ids]).to(device)
            t = torch.from_numpy(prepared.path_type[ids]).to(device)
            v = torch.from_numpy(prepared.path_valid[ids]).to(device)
            coeff_pred, power_delta = model(g, p, t, v)
            losses: dict[str, torch.Tensor] = {}
            for key in bases:
                anchor = torch.from_numpy(prepared.anchors[key][ids]).to(device)
                truth = torch.from_numpy(prepared.targets[key][ids]).to(device)
                coeff_truth = torch.from_numpy(prepared.target_coeff[key][ids]).to(device)
                pred = reconstruct_shape(coeff_pred[key], anchor, basis_gpu[key])
                weight = torch.ones(len(ids), dtype=torch.float32, device=device)
                losses[key] = cosine_loss(pred, truth, weight)
                losses[f"{key}_coeff"] = F.smooth_l1_loss(coeff_pred[key], coeff_truth, beta=0.5)
            target_power_delta = torch.from_numpy(
                prepared.target_log_power[ids] - prepared.anchor_log_power[ids]
            ).to(device)
            losses["power"] = F.smooth_l1_loss(power_delta, target_power_delta, beta=0.6)
            coeff_loss = sum(losses[f"{key}_coeff"] for key in bases) / len(bases)
            loss = (
                0.30 * losses["beam"]
                + 0.30 * losses["d"]
                + 0.12 * losses["du"]
                + 0.12 * losses["dh"]
                + 0.08 * losses["power"]
                + 0.08 * coeff_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running.append(float(loss.detach().cpu()))
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            item = {"epoch": float(epoch), "train_loss": float(np.mean(running))}
            if validation is not None:
                val_pred = predict_model(model, validation, bases, device)
                item["val_composite_cos"] = float(
                    0.30 * np.mean(row_cos(val_pred["beam"], validation.targets["beam"]))
                    + 0.30 * np.mean(row_cos(val_pred["d"], validation.targets["d"]))
                    + 0.12 * np.mean(row_cos(val_pred["du"], validation.targets["du"]))
                    + 0.12 * np.mean(row_cos(val_pred["dh"], validation.targets["dh"]))
                )
                calibrated = []
                for alpha in np.linspace(0.0, 1.0, 21, dtype=np.float32):
                    score = 0.0
                    for key, weight in (("beam", 0.36), ("d", 0.36), ("du", 0.14), ("dh", 0.14)):
                        blended = normalize(
                            (1.0 - alpha) * validation.anchors[key] + alpha * val_pred[key],
                            axis=tuple(range(1, validation.anchors[key].ndim)),
                        )
                        score += weight * float(np.mean(row_cos(blended, validation.targets[key])))
                    calibrated.append(score)
                item["val_calibrated_support"] = float(np.max(calibrated))
                item["val_calibrated_alpha"] = float(np.linspace(0.0, 1.0, 21)[int(np.argmax(calibrated))])
                if item["val_calibrated_support"] > best_metric:
                    best_metric = item["val_calibrated_support"]
                    best_epoch = epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            history.append(item)
            print(f"v215 seed={seed} epoch={epoch}/{epochs} {json.dumps(item)}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "history": history,
        "best_epoch": int(best_epoch),
        "best_calibrated_support": None if not np.isfinite(best_metric) else float(best_metric),
    }


@torch.no_grad()
def predict_model(
    model: PhysicsPathField,
    prepared: Prepared,
    bases: dict[str, Basis],
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    basis_gpu = {key: torch_basis(value, device) for key, value in bases.items()}
    out = {key: [] for key in bases}
    power = []
    for start in range(0, len(prepared.global_features), 256):
        end = min(start + 256, len(prepared.global_features))
        coeff, power_delta = model(
            torch.from_numpy(prepared.global_features[start:end]).to(device),
            torch.from_numpy(prepared.path_numeric[start:end]).to(device),
            torch.from_numpy(prepared.path_type[start:end]).to(device),
            torch.from_numpy(prepared.path_valid[start:end]).to(device),
        )
        for key in bases:
            anchor = torch.from_numpy(prepared.anchors[key][start:end]).to(device)
            shape = reconstruct_shape(coeff[key], anchor, basis_gpu[key])
            out[key].append(shape.cpu().numpy().reshape((end - start,) + bases[key].shape).astype(np.float32))
        power.append((power_delta.cpu().numpy() + prepared.anchor_log_power[start:end]).astype(np.float32))
    result = {key: np.concatenate(value).astype(np.float32) for key, value in out.items()}
    result["log_power"] = np.concatenate(power).astype(np.float32)
    return result


def prepare_split(
    prefix: str,
    indices: np.ndarray | None,
    pos: np.ndarray,
    blocked: np.ndarray,
    anchors: Anchors,
    targets: dict[str, np.ndarray],
    log_power: np.ndarray,
    bases: dict[str, Basis],
    feature_stats: dict[str, np.ndarray] | None,
) -> tuple[Prepared, dict[str, np.ndarray] | None]:
    global_raw = make_global_features(prefix, indices, pos, blocked, anchors)
    path_raw, path_type, path_valid = make_path_tokens(prefix, pos, indices)
    if feature_stats is None:
        valid_values = path_raw[path_valid]
        feature_stats = {
            "global_mean": global_raw.mean(axis=0).astype(np.float32),
            "global_std": np.maximum(global_raw.std(axis=0), 1e-5).astype(np.float32),
            "path_mean": valid_values.mean(axis=0).astype(np.float32),
            "path_std": np.maximum(valid_values.std(axis=0), 1e-5).astype(np.float32),
        }
    global_features = np.clip(
        (global_raw - feature_stats["global_mean"]) / feature_stats["global_std"], -8.0, 8.0
    ).astype(np.float32)
    path_numeric = np.clip(
        (path_raw - feature_stats["path_mean"][None, None, :]) / feature_stats["path_std"][None, None, :],
        -8.0,
        8.0,
    ).astype(np.float32)
    path_numeric[~path_valid] = 0.0
    anchor_dict = {key: getattr(anchors, key) for key in ("beam", "d", "du", "dh")}
    coeff = {}
    for key, basis in bases.items():
        residual = np.clip(
            np.log(np.maximum(targets[key].reshape(len(pos), -1), FLOOR))
            - np.log(np.maximum(anchor_dict[key].reshape(len(pos), -1), FLOOR)),
            -8.0,
            8.0,
        )
        raw_coeff = (residual - basis.mean[None, :]) @ basis.components.T
        coeff[key] = (raw_coeff / basis.coeff_scale[None, :]).astype(np.float32)
    return Prepared(
        global_features=global_features,
        path_numeric=path_numeric,
        path_type=path_type,
        path_valid=path_valid,
        anchors=anchor_dict,
        targets=targets,
        target_coeff=coeff,
        anchor_log_power=anchors.log_power,
        target_log_power=log_power,
    ), feature_stats


def validation_calibration(
    pred: dict[str, np.ndarray],
    val: Prepared,
    regimes: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, object]]:
    alpha_grid = np.linspace(0.0, 1.25, 26, dtype=np.float32)
    support_alpha: dict[str, float] = {}
    power_alpha: dict[str, float] = {}
    support_action_cap: dict[str, float] = {}
    report: dict[str, object] = {}
    for regime in ("clear", "light", "medium", "heavy"):
        mask = regimes.astype(str) == regime
        if not np.any(mask):
            support_alpha[regime] = 0.0
            power_alpha[regime] = 0.0
            support_action_cap[regime] = 0.0
            continue
        scores = []
        for alpha in alpha_grid:
            score = 0.0
            for key, weight in (("beam", 0.36), ("d", 0.36), ("du", 0.14), ("dh", 0.14)):
                blended = normalize(
                    (1.0 - alpha) * val.anchors[key][mask] + alpha * pred[key][mask],
                    axis=tuple(range(1, val.anchors[key].ndim)),
                )
                score += weight * float(np.mean(row_cos(blended, val.targets[key][mask])))
            scores.append(score)
        best = int(np.argmax(scores))
        support_alpha[regime] = float(alpha_grid[best])
        calibrated_d = normalize(
            (1.0 - alpha_grid[best]) * val.anchors["d"][mask]
            + alpha_grid[best] * pred["d"][mask],
            axis=1,
        )
        d_action = np.sum(np.abs(calibrated_d - val.anchors["d"][mask]), axis=1)
        support_action_cap[regime] = float(np.quantile(d_action, 0.90))

        power_mae = []
        for alpha in alpha_grid:
            value = val.anchor_log_power[mask] + alpha * (pred["log_power"][mask] - val.anchor_log_power[mask])
            power_mae.append(float(np.mean(np.abs(value - val.target_log_power[mask]))))
        pbest = int(np.argmin(power_mae))
        power_alpha[regime] = float(alpha_grid[pbest])
        report[regime] = {
            "count": int(np.sum(mask)),
            "support_alpha": support_alpha[regime],
            "support_score_anchor": float(scores[0]),
            "support_score_best": float(scores[best]),
            "d_action_cap_p90": support_action_cap[regime],
            "power_alpha": power_alpha[regime],
            "power_mae_anchor": float(power_mae[0]),
            "power_mae_best": float(power_mae[pbest]),
        }
    return support_alpha, power_alpha, support_action_cap, report


def safe_ratio(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    return (np.maximum(desired, FLOOR) / np.maximum(current, FLOOR)).astype(np.float32)


def support_marginals(support: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = support.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32)
    du = support.sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
    dh = support.sum(axis=(1, 3, 4), dtype=np.float64).astype(np.float32)
    return normalize(d, axis=1), normalize(du, axis=(1, 2)), normalize(dh, axis=(1, 2))


def apply_neural_fields(
    prediction: dict[str, np.ndarray],
    regimes: np.ndarray,
    support_alpha: dict[str, float],
    power_alpha: dict[str, float],
    support_action_cap: dict[str, float],
    power_clip: tuple[float, float],
) -> dict[str, np.ndarray]:
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape)
    diagnostics = {key: [] for key in ("power_factor", "d_l1", "du_l1", "dh_l1", "beam_l1")}
    alpha_s = np.asarray([support_alpha[str(r)] for r in regimes], dtype=np.float32)
    alpha_p = np.asarray([power_alpha[str(r)] for r in regimes], dtype=np.float32)

    for start in range(0, len(source), 5):
        end = min(start + 5, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        cir0 = np.fft.ifft(h0, axis=-1)
        arr0 = cir0.reshape(end - start, MP, MH, MV, N_ANT, S_SC)
        power0 = (np.abs(arr0) ** 2).astype(np.float32)
        total0 = np.maximum(power0.sum(axis=(1, 2, 3, 4, 5)), TINY).astype(np.float32)
        support0 = power0 / total0[:, None, None, None, None, None]
        current_d, current_du, current_dh = support_marginals(support0)
        local_alpha = alpha_s[start:end].copy()
        raw_d_action = np.sum(np.abs(prediction["d"][start:end] - current_d), axis=1)
        cap = np.asarray([support_action_cap[str(r)] for r in regimes[start:end]], dtype=np.float32)
        proposed_action = local_alpha * raw_d_action
        local_alpha *= np.minimum(1.0, cap / np.maximum(proposed_action, 1e-8)).astype(np.float32)
        desired_d = normalize(
            (1.0 - local_alpha[:, None]) * current_d + local_alpha[:, None] * prediction["d"][start:end], axis=1
        )
        desired_du = normalize(
            (1.0 - local_alpha[:, None, None]) * current_du
            + local_alpha[:, None, None] * prediction["du"][start:end],
            axis=(1, 2),
        )
        desired_dh = normalize(
            (1.0 - local_alpha[:, None, None]) * current_dh
            + local_alpha[:, None, None] * prediction["dh"][start:end],
            axis=(1, 2),
        )

        work_h = h0.copy()
        for _ in range(OUTER_ITERS):
            cir = np.fft.ifft(work_h, axis=-1)
            arr = cir.reshape(end - start, MP, MH, MV, N_ANT, S_SC)
            p = (np.abs(arr) ** 2).astype(np.float32)
            support = p / np.maximum(p.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)
            for _ in range(IPF_ITERS):
                v196.apply_delay_ratio(
                    support,
                    safe_ratio(desired_d, support.sum(axis=(1, 2, 3, 4), dtype=np.float64).astype(np.float32)),
                    "d",
                    0.96,
                )
                v196.apply_delay_ratio(
                    support,
                    safe_ratio(desired_du, support.sum(axis=(1, 2, 3), dtype=np.float64).astype(np.float32)),
                    "du",
                    0.34,
                )
                v196.apply_delay_ratio(
                    support,
                    safe_ratio(desired_dh, support.sum(axis=(1, 3, 4), dtype=np.float64).astype(np.float32)),
                    "dh",
                    0.30,
                )
                support = normalize(support.reshape(end - start, -1), axis=1).reshape(p.shape)
            gain = np.sqrt((support * total0[:, None, None, None, None, None]) / np.maximum(p, 1e-30))
            gain = np.clip(gain, 0.20, 5.0).astype(np.float32)
            work_h = np.fft.fft((arr * gain).reshape(end - start, 256, N_ANT, S_SC), axis=-1).astype(np.complex64)

            ad = np.fft.fft2(
                np.fft.ifft(work_h, axis=-1).reshape(end - start, MP, MH, MV, N_ANT, S_SC), axes=(2, 3)
            ).astype(np.complex64)
            ad_power = (np.abs(ad) ** 2).astype(np.float32)
            beam_current = normalize(ad_power.sum(axis=(1, 4, 5)).reshape(end - start, -1), axis=1)
            beam_desired = normalize(
                (1.0 - local_alpha[:, None]) * beam_current
                + local_alpha[:, None] * prediction["beam"][start:end],
                axis=1,
            )
            beam_gain = np.sqrt(safe_ratio(beam_desired, beam_current)).reshape(end - start, 1, MH, MV, 1, 1)
            beam_gain = np.clip(beam_gain, 0.35, 2.85).astype(np.float32)
            ad *= beam_gain
            arr = np.fft.ifft2(ad, axes=(2, 3))
            work_h = np.fft.fft(arr.reshape(end - start, 256, N_ANT, S_SC), axis=-1).astype(np.complex64)

        current_log = np.log(total0)
        desired_log = current_log + alpha_p[start:end] * (prediction["log_power"][start:end] - current_log)
        log_delta = np.clip(desired_log - current_log, power_clip[0], power_clip[1])
        current_h_power = np.maximum(np.sum(np.abs(work_h) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY)
        desired_h_power = np.exp(current_log + log_delta) * S_SC
        work_h *= np.sqrt(desired_h_power / current_h_power)[:, None, None, None].astype(np.float32)
        output[start:end] = work_h.astype(np.complex64)
        output.flush()

        final_power = v196.delay_power_from_h(work_h.astype(np.complex64))
        final_support = final_power / np.maximum(final_power.sum(axis=(1, 2, 3, 4, 5), keepdims=True), TINY)
        final_d, final_du, final_dh = support_marginals(final_support)
        final_ad = np.fft.fft2(
            np.fft.ifft(work_h, axis=-1).reshape(end - start, MP, MH, MV, N_ANT, S_SC), axes=(2, 3)
        )
        final_beam = normalize((np.abs(final_ad) ** 2).sum(axis=(1, 4, 5)).reshape(end - start, -1), axis=1)
        diagnostics["power_factor"].append(np.exp(log_delta).astype(np.float32))
        diagnostics["d_l1"].append(np.sum(np.abs(final_d - current_d), axis=1).astype(np.float32))
        diagnostics["du_l1"].append(np.sum(np.abs(final_du - current_du), axis=(1, 2)).astype(np.float32))
        diagnostics["dh_l1"].append(np.sum(np.abs(final_dh - current_dh), axis=(1, 2)).astype(np.float32))
        diagnostics["beam_l1"].append(np.sum(np.abs(final_beam - prediction["beam"][start:end]), axis=1).astype(np.float32))
        print(f"v215 physical neural projection {end}/{len(source)}", flush=True)
    return {key: np.concatenate(value).astype(np.float32) for key, value in diagnostics.items()}


def write_zip(report: dict[str, object]) -> dict[str, object]:
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        archive.write(CHECKPOINT, CHECKPOINT.name)
        for src in [
            PATHS.code / "generate_v215_pointcloud_physics_neural_field.py",
            PATHS.code / "build_official_pointcloud_scene.py",
            PATHS.code / "build_multibounce_path_candidates.py",
            PATHS.code / "analyze_geodesic_support_source.py",
            PATHS.code / "generate_v209_adaptive_metric_projection_controller.py",
            PATHS.code / "generate_v196_train_guided_delay_pdp_mouth.py",
            PATHS.code_root / "project_paths.py",
        ]:
            archive.write(src, src.name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    output = np.load(OUTPUT, mmap_mode="r")
    finite = True
    for start in range(0, len(output), 20):
        block = np.asarray(output[start : start + 20], dtype=np.complex64)
        finite = finite and bool(np.isfinite(block.real).all() and np.isfinite(block.imag).all())
    report["integrity"] = {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "zip_crc": True,
        "zip_members": members,
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    for path in [BASE, TRAIN_MARGINALS, TRAIN_COLL, SCENE_CACHE, GEODESIC_CACHE, PATH_CACHE, GEOM_CACHE]:
        require_file(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_train = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    pos_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geom = np.load(GEOM_CACHE)
    blocked_train = geom["train_blocked"].astype(np.float32)
    blocked_test = geom["test_blocked"].astype(np.float32)
    regimes_train = regime_of(blocked_train)
    regimes_test = regime_of(blocked_test)
    clean, raw, log_power, valid = load_targets()
    valid_ids = np.flatnonzero(valid)
    matched = choose_val_indices(pos_train[:, :2], pos_test[:, :2], VAL_COUNT, "matched-test", 215)
    val_ids = np.intersect1d(matched, valid_ids)
    train_ids = np.setdiff1d(valid_ids, val_ids)

    train_targets = {key: value[train_ids] for key, value in clean.items()}
    val_targets = {key: value[val_ids] for key, value in clean.items()}
    train_anchor = build_anchors(
        pos_train[train_ids], blocked_train[train_ids], pos_train[train_ids], blocked_train[train_ids],
        train_targets, log_power[train_ids], query_ids=train_ids, bank_ids=train_ids,
    )
    val_anchor = build_anchors(
        pos_train[val_ids], blocked_train[val_ids], pos_train[train_ids], blocked_train[train_ids],
        train_targets, log_power[train_ids], query_ids=val_ids, bank_ids=train_ids,
    )
    bases: dict[str, Basis] = {}
    train_coeff: dict[str, np.ndarray] = {}
    for key in PCA_DIMS:
        basis, coeff = fit_basis(getattr(train_anchor, key), train_targets[key], PCA_DIMS[key])
        bases[key] = basis
        train_coeff[key] = coeff

    train_prepared, feature_stats = prepare_split(
        "train", train_ids, pos_train[train_ids], blocked_train[train_ids], train_anchor,
        train_targets, log_power[train_ids], bases, None,
    )
    train_prepared.target_coeff = train_coeff
    val_prepared, _ = prepare_split(
        "train", val_ids, pos_train[val_ids], blocked_train[val_ids], val_anchor,
        val_targets, log_power[val_ids], bases, feature_stats,
    )

    validation_model, validation_history = train_model(
        train_prepared, bases, TRAIN_EPOCHS, SEEDS[0], device, validation=val_prepared
    )
    val_pred = predict_model(validation_model, val_prepared, bases, device)
    support_alpha, learned_power_alpha, support_action_cap, calibration = validation_calibration(
        val_pred, val_prepared, regimes_train[val_ids]
    )
    power_alpha = {name: 0.0 for name in ("clear", "light", "medium", "heavy")}
    validation_diagnostics = {
        "beam_cos_anchor": summarize(row_cos(val_prepared.anchors["beam"], val_targets["beam"])),
        "beam_cos_model": summarize(row_cos(val_pred["beam"], val_targets["beam"])),
        "d_cos_anchor": summarize(row_cos(val_prepared.anchors["d"], val_targets["d"])),
        "d_cos_model": summarize(row_cos(val_pred["d"], val_targets["d"])),
        "du_cos_anchor": summarize(row_cos(val_prepared.anchors["du"], val_targets["du"])),
        "du_cos_model": summarize(row_cos(val_pred["du"], val_targets["du"])),
        "dh_cos_anchor": summarize(row_cos(val_prepared.anchors["dh"], val_targets["dh"])),
        "dh_cos_model": summarize(row_cos(val_pred["dh"], val_targets["dh"])),
        "log_power_mae_anchor": float(np.mean(np.abs(val_prepared.anchor_log_power - val_prepared.target_log_power))),
        "log_power_mae_model": float(np.mean(np.abs(val_pred["log_power"] - val_prepared.target_log_power))),
        "calibration": calibration,
    }
    print(json.dumps(validation_diagnostics, indent=2), flush=True)
    selected_final_epochs = max(1, int(validation_history["best_epoch"]))
    del validation_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_targets = {key: value[valid_ids] for key, value in clean.items()}
    all_anchor = build_anchors(
        pos_train[valid_ids], blocked_train[valid_ids], pos_train[valid_ids], blocked_train[valid_ids],
        all_targets, log_power[valid_ids], query_ids=valid_ids, bank_ids=valid_ids,
    )
    final_bases: dict[str, Basis] = {}
    final_coeff: dict[str, np.ndarray] = {}
    for key in PCA_DIMS:
        basis, coeff = fit_basis(getattr(all_anchor, key), all_targets[key], PCA_DIMS[key])
        final_bases[key] = basis
        final_coeff[key] = coeff
    all_prepared, final_feature_stats = prepare_split(
        "train", valid_ids, pos_train[valid_ids], blocked_train[valid_ids], all_anchor,
        all_targets, log_power[valid_ids], final_bases, None,
    )
    all_prepared.target_coeff = final_coeff
    test_anchor = build_anchors(
        pos_test, blocked_test, pos_train[valid_ids], blocked_train[valid_ids],
        all_targets, log_power[valid_ids],
    )
    dummy_targets = {key: getattr(test_anchor, key) for key in PCA_DIMS}
    test_prepared, _ = prepare_split(
        "test", None, pos_test, blocked_test, test_anchor,
        dummy_targets, test_anchor.log_power, final_bases, final_feature_stats,
    )

    state_dicts = []
    final_histories = []
    test_predictions = []
    for seed in SEEDS:
        model, history = train_model(all_prepared, final_bases, selected_final_epochs, seed, device)
        test_predictions.append(predict_model(model, test_prepared, final_bases, device))
        state_dicts.append({key: value.detach().cpu() for key, value in model.state_dict().items()})
        final_histories.append(history)
    ensemble = {
        key: np.mean(np.stack([pred[key] for pred in test_predictions], axis=0), axis=0).astype(np.float32)
        for key in (*PCA_DIMS.keys(), "log_power")
    }
    power_residual = log_power[valid_ids] - all_anchor.log_power
    power_clip = (float(np.quantile(power_residual, 0.03)), float(np.quantile(power_residual, 0.97)))
    apply_diag = apply_neural_fields(
        ensemble, regimes_test, support_alpha, power_alpha, support_action_cap, power_clip
    )
    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes_test.astype("U8"),
        support_alpha=np.asarray([support_alpha[str(r)] for r in regimes_test], dtype=np.float32),
        power_alpha=np.asarray([power_alpha[str(r)] for r in regimes_test], dtype=np.float32),
        support_action_cap=np.asarray([support_action_cap[str(r)] for r in regimes_test], dtype=np.float32),
        desired_beam=ensemble["beam"],
        desired_d=ensemble["d"],
        desired_du=ensemble["du"],
        desired_dh=ensemble["dh"],
        desired_log_power=ensemble["log_power"],
        **apply_diag,
    )

    checkpoint_payload = {
        "method": VERSION,
        "seeds": SEEDS,
        "state_dicts": state_dicts,
        "global_dim": all_prepared.global_features.shape[1],
        "path_dim": all_prepared.path_numeric.shape[2],
        "pca_dims": PCA_DIMS,
        "feature_stats": final_feature_stats,
        "bases": {
            key: {
                "mean": basis.mean,
                "components": basis.components,
                "coeff_scale": basis.coeff_scale,
                "shape": basis.shape,
            }
            for key, basis in final_bases.items()
        },
        "support_alpha": support_alpha,
        "power_alpha": power_alpha,
        "learned_power_alpha_not_deployed": learned_power_alpha,
        "support_action_cap": support_action_cap,
        "power_clip": power_clip,
        "neural_head_official_inputs_only": True,
        "backbone": PATHS.relative(BASE),
        "full_pipeline_one_click_from_raw_inputs": False,
    }
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload, CHECKPOINT)

    comparison_to_base = track.compare_output_to(OUTPUT, BASE, regimes_test)
    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "The v212 gain identifies normalized delay structure as a bottleneck, but the same invariant can be learned "
            "from released noisy train H when a point-cloud path Transformer predicts denoised PAS/PDP fields."
        ),
        "compliance_boundary": {
            "neural_model": "3-layer point-cloud path-token Transformer with learned PAS/PDP/power heads",
            "official_supervision": "released Round1 train H only",
            "official_geometry": "Round1_Map.ply derived scene, multibounce and geodesic features only",
            "excluded": "no v212 teacher, no Sionna H, no restored mesh/object/material labels, no ray-traced test target",
            "physics_decoder": "alternating delay-marginal and angle-beam amplitude projection; phase is not hand-authored",
        },
        "parameters": {
            "base": PATHS.relative(BASE),
            "validation_count": int(len(val_ids)),
            "valid_train_count": int(len(valid_ids)),
            "exact_zero_excluded_from_supervision": int(np.sum(~valid)),
            "anchor_k": ANCHOR_K,
            "pca_dims": PCA_DIMS,
            "seeds": SEEDS,
            "train_epochs": TRAIN_EPOCHS,
            "final_epochs": selected_final_epochs,
            "support_alpha": support_alpha,
            "power_alpha": power_alpha,
            "learned_power_alpha_not_deployed": learned_power_alpha,
            "support_action_cap": support_action_cap,
            "power_clip": power_clip,
            "power_policy": "preserve v209 until a v209-aligned OOF power base exists",
        },
        "validation": validation_diagnostics,
        "validation_history": validation_history,
        "final_histories": final_histories,
        "diagnostics": {
            "overall": {key: summarize(value) for key, value in apply_diag.items()},
            "by_regime": by_regime_summary(regimes_test, apply_diag),
        },
        "comparison_to_v209": comparison_to_base,
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "checkpoint": PATHS.relative(CHECKPOINT),
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report = write_zip(report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
