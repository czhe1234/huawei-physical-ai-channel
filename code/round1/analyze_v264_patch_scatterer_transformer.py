#!/usr/bin/env python3
"""v264: patch-ID-aware radio scatterer Transformer for PDP rendering."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import nnls
from scipy.spatial import cKDTree

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v254_route_topology_spectral_field as v254
import analyze_v257_neural_pairwise_spectral_transport as v257
import build_official_pointcloud_scene as pointcloud_scene
from generate_v87_easy_field_projection import S_SC, summarize
from project_paths import PATHS


VERSION_NUMBER = 264
VERSION = "v264_patch_scatterer_transformer"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = PATHS.artifacts / "checkpoints" / VERSION
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v264_Round1_Test_Channel.npy"
TARGETS = OUT_DIR / "v264_scatterer_pdp_targets.npz"
ACTIONS = OUT_DIR / "v264_scatterer_actions.npz"
NNLS_CACHE = OUT_DIR / "v264_train_scatterer_nnls.npz"
TEST_CANDIDATES = OUT_DIR / "v264_test_wall_reflection_candidates.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
TRAIN_CANDIDATES = PATHS.cache_dir(263) / "v263_full_wall_reflection_candidates.npz"
SCENE_CACHE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"

LOCAL_GATE = 0.645604
VAL_COUNT = 320
VAL_SEED = 254
REFLECTIONS = 32
PATHS_PER_POINT = REFLECTIONS + 1
SHIFTS = np.asarray([-0.85, 0.0, 0.85], dtype=np.float32)
TAIL_COUNT = 6
RIDGE = 0.022
SUM_PENALTY = 0.55
TRAIN_EPOCHS = 260
BATCH_SIZE = 80
SEEDS = (26401, 26431, 26461)
LIGHT_SPEED = 299792458.0
SUBCARRIER_SPACING = 30000.0
MIN_OFFICIAL_GAIN = 2e-4


@dataclass
class FeatureSet:
    global_features: np.ndarray
    path_numeric: np.ndarray
    patch_id: np.ndarray
    path_type: np.ndarray
    path_valid: np.ndarray
    delay: np.ndarray

    def take(self, ids: np.ndarray) -> "FeatureSet":
        return FeatureSet(
            self.global_features[ids],
            self.path_numeric[ids],
            self.patch_id[ids],
            self.path_type[ids],
            self.path_valid[ids],
            self.delay[ids],
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_d(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), 1e-10)
    out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-20)
    return out.astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    y = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def fixed_numpy_atoms(delay: np.ndarray, path_type: np.ndarray, valid: np.ndarray) -> np.ndarray:
    bins = np.arange(S_SC, dtype=np.float32)[None, None, None, :]
    center = delay[:, :, None, None] + SHIFTS[None, None, :, None]
    sigma = np.where(path_type == 0, 0.62, 0.96).astype(np.float32)
    atoms = np.exp(-0.5 * ((bins - center) / sigma[:, :, None, None]) ** 2)
    atoms *= valid[:, :, None, None]
    atoms /= np.maximum(atoms.sum(axis=3, keepdims=True), 1e-30)
    path_atoms = atoms.reshape(len(delay), -1, S_SC)
    x = np.arange(S_SC, dtype=np.float32)[None, :]
    max_delay = np.max(np.where(valid, delay, 0.0), axis=1)[:, None]
    delta = np.maximum(x - max_delay, 0.0)
    causal = (x >= max_delay).astype(np.float32)
    fixed = np.stack(
        [
            np.repeat(np.exp(-0.5 * (x / 0.55) ** 2), len(delay), axis=0),
            np.repeat(np.exp(-0.5 * ((x - 1.0) / 0.75) ** 2), len(delay), axis=0),
            np.repeat(np.exp(-0.5 * ((x - 2.0) / 1.05) ** 2), len(delay), axis=0),
            np.repeat(np.exp(-0.5 * ((x - 10.0) / 4.5) ** 2), len(delay), axis=0),
            np.exp(-delta / 8.0) * causal,
            np.exp(-delta / 22.0) * causal,
        ],
        axis=1,
    ).astype(np.float32)
    fixed /= np.maximum(fixed.sum(axis=2, keepdims=True), 1e-30)
    return np.concatenate([path_atoms, fixed], axis=1).astype(np.float32)


def build_nnls_labels(
    target_d: np.ndarray,
    features: FeatureSet,
    valid_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if NNLS_CACHE.is_file():
        data = np.load(NNLS_CACHE)
        return data["weights"], data["reconstruction"], json.loads(str(data["report_json"].item()))
    atoms = fixed_numpy_atoms(features.delay, features.path_type, features.path_valid)
    atom_count = atoms.shape[1]
    weights = np.zeros((len(target_d), atom_count), dtype=np.float32)
    reconstruction = np.zeros_like(target_d, dtype=np.float32)
    eye = math.sqrt(RIDGE) * np.eye(atom_count, dtype=np.float64)
    sum_row = math.sqrt(SUM_PENALTY) * np.ones((1, atom_count), dtype=np.float64)
    zeros = np.zeros(atom_count, dtype=np.float64)
    for count, point_id in enumerate(valid_ids, start=1):
        matrix = np.vstack([atoms[point_id].T.astype(np.float64), eye, sum_row])
        rhs = np.concatenate(
            [
                target_d[point_id].astype(np.float64),
                zeros,
                [math.sqrt(SUM_PENALTY)],
            ]
        )
        weight, _ = nnls(matrix, rhs, maxiter=atom_count * 10)
        if weight.sum() <= 1e-20:
            weight[-1] = 1.0
        weight /= weight.sum()
        weights[point_id] = weight.astype(np.float32)
        reconstruction[point_id] = normalize_d(
            (weight[None] @ atoms[point_id]).astype(np.float32)
        )[0]
        if count % 100 == 0 or count == len(valid_ids):
            print(f"v264 NNLS {count}/{len(valid_ids)}", flush=True)
    report = {
        "atom_count": int(atom_count),
        "oracle_cos": summarize(row_cos(reconstruction[valid_ids], target_d[valid_ids])),
        "path_mass": summarize(weights[valid_ids, :-TAIL_COUNT].sum(axis=1)),
    }
    np.savez_compressed(
        NNLS_CACHE,
        weights=weights,
        reconstruction=reconstruction,
        report_json=np.asarray(json.dumps(report)),
    )
    return weights, reconstruction, report


def build_features(
    pos: np.ndarray,
    candidates: dict[str, np.ndarray],
    patches: np.ndarray,
    global_features: np.ndarray,
) -> FeatureSet:
    count = len(pos)
    patch_id = np.zeros((count, PATHS_PER_POINT), dtype=np.int64)
    path_type = np.ones((count, PATHS_PER_POINT), dtype=np.int64)
    path_type[:, 0] = 0
    path_valid = np.zeros((count, PATHS_PER_POINT), dtype=bool)
    path_valid[:, 0] = True
    path_length = np.zeros((count, PATHS_PER_POINT), dtype=np.float32)
    excess = np.zeros_like(path_length)
    score = np.zeros_like(path_length)
    reflection_point = np.zeros((count, PATHS_PER_POINT, 3), dtype=np.float32)

    bs = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
    direct = np.linalg.norm(pos - bs[None, :], axis=1).astype(np.float32)
    path_length[:, 0] = direct
    score[:, 0] = 1.0
    reflection_point[:, 0] = pos.astype(np.float32)
    ids = candidates["patch_id"][:, :REFLECTIONS].astype(np.int64)
    candidate_score = candidates["score"][:, :REFLECTIONS].astype(np.float32)
    valid = candidate_score > 0.0
    patch_id[:, 1:] = np.where(valid, ids + 1, 0)
    path_valid[:, 1:] = valid
    path_length[:, 1:] = candidates["path_length"][:, :REFLECTIONS]
    excess[:, 1:] = candidates["excess_length"][:, :REFLECTIONS]
    score[:, 1:] = candidate_score
    reflection_point[:, 1:] = candidates["reflection_point"][:, :REFLECTIONS]
    delay = path_length / LIGHT_SPEED * S_SC * SUBCARRIER_SPACING

    patch_numeric = np.zeros((count, PATHS_PER_POINT, 10), dtype=np.float32)
    safe_id = np.clip(ids, 0, len(patches) - 1)
    selected_patch = patches[safe_id]
    patch_numeric[:, 1:, 0:2] = selected_patch[:, :, 0:2]
    patch_numeric[:, 1:, 2] = selected_patch[:, :, 2] / 500.0
    patch_numeric[:, 1:, 3] = (selected_patch[:, :, 4] - selected_patch[:, :, 3]) / 250.0
    patch_numeric[:, 1:, 4] = (selected_patch[:, :, 5] + selected_patch[:, :, 6]) / 60.0
    patch_numeric[:, 1:, 5] = (selected_patch[:, :, 6] - selected_patch[:, :, 5]) / 40.0
    patch_numeric[:, 1:, 6] = np.log1p(selected_patch[:, :, 7]) / 10.0
    patch_numeric[:, 1:, 7] = selected_patch[:, :, 8]
    patch_numeric[:, 1:, 8] = selected_patch[:, :, 9]
    patch_numeric[:, 1:, 9] = selected_patch[:, :, 11] / 160.0
    patch_numeric[:, 1:] *= valid[:, :, None]

    bs_to_point = reflection_point.astype(np.float64) - bs[None, None, :]
    point_to_ue = pos[:, None, :].astype(np.float64) - reflection_point.astype(np.float64)
    bs_norm = np.maximum(np.linalg.norm(bs_to_point, axis=2, keepdims=True), 1e-8)
    ue_norm = np.maximum(np.linalg.norm(point_to_ue, axis=2, keepdims=True), 1e-8)
    path_numeric = np.concatenate(
        [
            (delay / 24.0)[:, :, None],
            (path_length / 500.0)[:, :, None],
            (excess / 300.0)[:, :, None],
            score[:, :, None],
            (reflection_point - bs[None, None, :]) / 500.0,
            (reflection_point - pos[:, None, :]) / 500.0,
            (bs_to_point / bs_norm).astype(np.float32),
            (point_to_ue / ue_norm).astype(np.float32),
            patch_numeric,
        ],
        axis=2,
    ).astype(np.float32)
    path_numeric[~path_valid] = 0.0
    return FeatureSet(
        global_features=np.asarray(global_features, dtype=np.float32),
        path_numeric=path_numeric,
        patch_id=patch_id,
        path_type=path_type,
        path_valid=path_valid,
        delay=delay.astype(np.float32),
    )


def standardize_features(
    train: FeatureSet,
    full_train: FeatureSet,
    test: FeatureSet,
) -> tuple[FeatureSet, FeatureSet, FeatureSet, dict[str, np.ndarray]]:
    global_mean = train.global_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    global_std = train.global_features.std(axis=0, dtype=np.float64).astype(np.float32)
    global_std[global_std < 1e-5] = 1.0
    values = train.path_numeric[train.path_valid]
    path_mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    path_std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    path_std[path_std < 1e-5] = 1.0

    def apply(feature: FeatureSet) -> FeatureSet:
        numeric = ((feature.path_numeric - path_mean) / path_std).astype(np.float32)
        numeric[~feature.path_valid] = 0.0
        return FeatureSet(
            ((feature.global_features - global_mean) / global_std).astype(np.float32),
            numeric,
            feature.patch_id,
            feature.path_type,
            feature.path_valid,
            feature.delay,
        )

    return apply(train), apply(full_train), apply(test), {
        "global_mean": global_mean,
        "global_std": global_std,
        "path_mean": path_mean,
        "path_std": path_std,
    }


def torch_atoms(delay: torch.Tensor, path_type: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    bins = torch.arange(S_SC, dtype=torch.float32, device=delay.device)[None, None, None, :]
    shifts = torch.tensor(SHIFTS, dtype=torch.float32, device=delay.device)[None, None, :, None]
    sigma = torch.where(path_type == 0, 0.62, 0.96).float()
    path = torch.exp(-0.5 * ((bins - delay[:, :, None, None] - shifts) / sigma[:, :, None, None]) ** 2)
    path = path * valid[:, :, None, None]
    path = path / path.sum(dim=3, keepdim=True).clamp_min(1e-30)
    path = path.flatten(1, 2)
    x = torch.arange(S_SC, dtype=torch.float32, device=delay.device)[None, :]
    max_delay = torch.max(delay.masked_fill(~valid, -1e9), dim=1).values[:, None]
    delta = torch.clamp(x - max_delay, min=0.0)
    causal = (x >= max_delay).float()
    fixed = torch.stack(
        [
            torch.exp(-0.5 * (x / 0.55) ** 2).expand(len(delay), -1),
            torch.exp(-0.5 * ((x - 1.0) / 0.75) ** 2).expand(len(delay), -1),
            torch.exp(-0.5 * ((x - 2.0) / 1.05) ** 2).expand(len(delay), -1),
            torch.exp(-0.5 * ((x - 10.0) / 4.5) ** 2).expand(len(delay), -1),
            torch.exp(-delta / 8.0) * causal,
            torch.exp(-delta / 22.0) * causal,
        ],
        dim=1,
    )
    fixed = fixed / fixed.sum(dim=2, keepdim=True).clamp_min(1e-30)
    return torch.cat([path, fixed], dim=1)


class ScattererTransformer(nn.Module):
    def __init__(self, global_dim: int, path_dim: int, patch_count: int) -> None:
        super().__init__()
        width = 224
        self.patch_embedding = nn.Embedding(patch_count + 1, 40)
        self.type_embedding = nn.Embedding(2, 12)
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, 384), nn.SiLU(), nn.LayerNorm(384), nn.Linear(384, width)
        )
        self.path_proj = nn.Sequential(
            nn.Linear(path_dim + 52, 320), nn.SiLU(), nn.LayerNorm(320), nn.Linear(320, width)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=7,
            dim_feedforward=896,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=5)
        self.path_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 224), nn.SiLU())
        self.global_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 224), nn.SiLU())
        self.path_energy = nn.Linear(224, 1)
        self.shift_weight = nn.Linear(224, len(SHIFTS))
        self.tail_weight = nn.Linear(224, TAIL_COUNT)

    def forward(self, feature: FeatureSet, ids: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        global_feature = torch.from_numpy(feature.global_features[ids]).to(device)
        numeric = torch.from_numpy(feature.path_numeric[ids]).to(device)
        patch_id = torch.from_numpy(feature.patch_id[ids]).to(device)
        path_type = torch.from_numpy(feature.path_type[ids]).to(device)
        valid = torch.from_numpy(feature.path_valid[ids]).to(device)
        delay = torch.from_numpy(feature.delay[ids]).to(device)
        query = self.global_proj(global_feature).unsqueeze(1)
        path = self.path_proj(
            torch.cat([numeric, self.patch_embedding(patch_id), self.type_embedding(path_type)], dim=2)
        )
        token = torch.cat([query, path], dim=1)
        mask = torch.cat(
            [torch.zeros((len(ids), 1), dtype=torch.bool, device=device), ~valid], dim=1
        )
        encoded = self.encoder(token, src_key_padding_mask=mask)
        global_latent = self.global_head(encoded[:, 0])
        path_latent = self.path_head(encoded[:, 1:])
        path_logits = self.path_energy(path_latent) + self.shift_weight(path_latent)
        path_logits = path_logits.masked_fill(~valid[..., None], -1e9).flatten(1)
        logits = torch.cat([path_logits, self.tail_weight(global_latent)], dim=1)
        weights = torch.softmax(logits, dim=1)
        atoms = torch_atoms(delay, path_type, valid)
        pred = torch.sum(weights[:, :, None] * atoms, dim=1)
        pred /= pred.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return pred, weights


def loss_fn(
    pred: torch.Tensor,
    weights: torch.Tensor,
    truth_d: torch.Tensor,
    truth_weights: torch.Tensor,
) -> torch.Tensor:
    cos = 1.0 - F.cosine_similarity(pred, truth_d, dim=1).mean()
    cdf = torch.mean(torch.abs(torch.cumsum(pred, dim=1) - torch.cumsum(truth_d, dim=1)))
    label = truth_weights + 0.0015 / truth_weights.shape[1]
    label /= label.sum(dim=1, keepdim=True)
    weight_kl = torch.sum(
        label * (torch.log(label.clamp_min(1e-10)) - torch.log(weights.clamp_min(1e-10))),
        dim=1,
    ).mean()
    log_shape = F.smooth_l1_loss(
        torch.log(pred.clamp_min(1e-10)), torch.log(truth_d.clamp_min(1e-10)), beta=1.0
    )
    return 0.57 * cos + 0.13 * cdf + 0.24 * weight_kl + 0.06 * log_shape


@torch.no_grad()
def predict(
    model: ScattererTransformer,
    feature: FeatureSet,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    atom_count = PATHS_PER_POINT * len(SHIFTS) + TAIL_COUNT
    pred = np.empty((len(feature.delay), S_SC), dtype=np.float32)
    weight = np.empty((len(feature.delay), atom_count), dtype=np.float32)
    for start in range(0, len(feature.delay), 256):
        end = min(start + 256, len(feature.delay))
        ids = np.arange(start, end, dtype=np.int64)
        value, current_weight = model(feature, ids, device)
        pred[start:end] = value.cpu().numpy()
        weight[start:end] = current_weight.cpu().numpy()
    return pred, weight


def train_model(
    feature: FeatureSet,
    target_d: np.ndarray,
    target_weights: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
    patch_count: int,
    validation: tuple[FeatureSet, np.ndarray, np.ndarray] | None,
) -> tuple[ScattererTransformer, dict[str, object]]:
    set_seed(seed)
    model = ScattererTransformer(
        feature.global_features.shape[1], feature.path_numeric.shape[2], patch_count
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=2.5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    rng = np.random.default_rng(seed)
    best_cos = -np.inf
    best_epoch = epochs
    best_state = None
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(target_d))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            pred, weight = model(feature, ids, device)
            loss = loss_fn(
                pred,
                weight,
                torch.from_numpy(target_d[ids]).to(device),
                torch.from_numpy(target_weights[ids]).to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            item: dict[str, float] = {"epoch": epoch, "train_loss": float(np.mean(losses))}
            if validation is not None:
                val_feature, val_target, val_weights = validation
                val_pred, val_weight = predict(model, val_feature, device)
                item["val_cos"] = float(np.mean(row_cos(val_pred, val_target)))
                item["val_weight_cos"] = float(np.mean(row_cos(val_weight, val_weights)))
                if item["val_cos"] > best_cos:
                    best_cos = item["val_cos"]
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                    }
            history.append(item)
            print(f"v264 seed={seed} epoch={epoch}/{epochs} {json.dumps(item)}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_cos": None if not np.isfinite(best_cos) else float(best_cos),
        "history": history,
    }


def knn_anchor(
    fit_pos: np.ndarray,
    fit_d: np.ndarray,
    query_pos: np.ndarray,
) -> np.ndarray:
    distance, local = cKDTree(fit_pos).query(query_pos, k=16)
    weight = 1.0 / np.maximum(distance, 0.05) ** 2.0
    weight /= weight.sum(axis=1, keepdims=True)
    return normalize_d(np.sum(weight[:, :, None] * fit_d[local], axis=1))


def main() -> None:
    for path in (TARGET_CACHE, TRAIN_CANDIDATES, SCENE_CACHE, BASE):
        if not path.is_file():
            raise FileNotFoundError(path)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    scene = np.load(SCENE_CACHE)
    patches = scene["wall_patches"].astype(np.float32)
    train_node, test_node = v257.build_node_features(train_pos[:, :2], test_pos[:, :2])
    train_candidates = dict(np.load(TRAIN_CANDIDATES))
    train_features_raw = build_features(train_pos, train_candidates, patches, train_node)

    if TEST_CANDIDATES.is_file():
        test_candidates = dict(np.load(TEST_CANDIDATES))
    else:
        test_candidates = pointcloud_scene.build_reflection_candidates(
            test_pos, patches, top_k=REFLECTIONS, batch=64
        )
        np.savez_compressed(TEST_CANDIDATES, **test_candidates)
    test_features_raw = build_features(test_pos, test_candidates, patches, test_node)

    target_data = np.load(TARGET_CACHE)
    target_d = normalize_d(target_data["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target_data["valid"].astype(bool)).astype(np.int64)
    matched = v254.choose_seeded_matched_indices(
        train_pos[:, :2], test_pos[:, :2], VAL_COUNT, VAL_SEED
    )
    val_ids = np.intersect1d(matched, valid_ids)
    fit_ids = np.setdiff1d(valid_ids, val_ids)
    nnls_weights, nnls_reconstruction, nnls_report = build_nnls_labels(
        target_d, train_features_raw, valid_ids
    )
    fit_raw = train_features_raw.take(fit_ids)
    val_raw = train_features_raw.take(val_ids)
    fit_feature, full_feature, test_feature, scaling = standardize_features(
        fit_raw, train_features_raw, test_features_raw
    )
    val_feature = full_feature.take(val_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_model, validation_training = train_model(
        fit_feature,
        target_d[fit_ids],
        nnls_weights[fit_ids],
        SEEDS[0],
        TRAIN_EPOCHS,
        device,
        len(patches),
        (val_feature, target_d[val_ids], nnls_weights[val_ids]),
    )
    val_pred, val_weight = predict(validation_model, val_feature, device)
    anchor = knn_anchor(train_pos[fit_ids, :2], target_d[fit_ids], train_pos[val_ids, :2])
    alpha_grid = np.linspace(0.0, 1.25, 26, dtype=np.float32)
    alpha_scores = []
    for alpha in alpha_grid:
        blend = normalize_d((1.0 - alpha) * anchor + alpha * val_pred)
        alpha_scores.append(float(np.mean(row_cos(blend, target_d[val_ids]))))
    best_alpha_index = int(np.argmax(alpha_scores))
    best_alpha = float(alpha_grid[best_alpha_index])
    official_gain = float(alpha_scores[best_alpha_index] - alpha_scores[0])
    official_selected = bool(best_alpha > 0.0 and official_gain >= MIN_OFFICIAL_GAIN)

    local_score = None
    actions = None
    final_training: list[dict[str, object]] = []
    checkpoints: list[str] = []
    if official_selected:
        best_epoch = int(validation_training["best_epoch"])
        all_raw = train_features_raw.take(valid_ids)
        all_feature, _, final_test_feature, final_scaling = standardize_features(
            all_raw, train_features_raw, test_features_raw
        )
        predictions = []
        for seed in SEEDS:
            model, training = train_model(
                all_feature,
                target_d[valid_ids],
                nnls_weights[valid_ids],
                seed,
                best_epoch,
                device,
                len(patches),
                None,
            )
            pred, _ = predict(model, final_test_feature, device)
            predictions.append(pred)
            checkpoint = CHECKPOINT_DIR / f"seed_{seed}.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "scaling": final_scaling,
                    "best_alpha": best_alpha,
                    "epochs": best_epoch,
                },
                checkpoint,
            )
            checkpoints.append(PATHS.relative(checkpoint))
            final_training.append(training)
        test_pred = normalize_d(np.mean(predictions, axis=0))
        base = np.load(BASE, mmap_mode="r")
        current_sh, current_d = v247.extract_current(np.asarray(base, dtype=np.complex64))
        target_test_d = normalize_d((1.0 - best_alpha) * current_d + best_alpha * test_pred)
        np.savez_compressed(
            TARGETS,
            d=target_test_d,
            predicted_d=test_pred,
            current_d=current_d,
            alpha=np.asarray(best_alpha),
        )
        actions = v247.project(current_sh, target_test_d, OUTPUT)
        np.savez_compressed(ACTIONS, **actions)
        local_score = v208.score_prediction(OUTPUT)

    report = {
        "method": VERSION,
        "status": (
            "gate_passed_needs_packaging"
            if local_score is not None
            and float(local_score["score_traceback_leaked"]) > LOCAL_GATE
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "Stable official wall-patch identities let a path-set Transformer learn the higher "
            "expressiveness revealed by the v263 scatterer NNLS oracle."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H NNLS scatterer labels",
                "official train/test coordinates",
                "official point-cloud wall patches and analytic reflections",
                "PyTorch patch-ID-aware Transformer",
                "strict v239 carrier after official selection",
            ],
            "does_not_use": [
                "Sionna or synthetic channel labels",
                "teacher output",
                "local replay truth for training or selection",
                "hidden labels",
                "raw RT H",
            ],
        },
        "parameters": {
            "reflection_tokens": REFLECTIONS,
            "paths_per_point": PATHS_PER_POINT,
            "patch_count": int(len(patches)),
            "train_epochs": TRAIN_EPOCHS,
            "seeds": list(SEEDS),
            "device": str(device),
        },
        "nnls": nnls_report,
        "validation": {
            "fit_count": int(len(fit_ids)),
            "validation_count": int(len(val_ids)),
            "training": validation_training,
            "anchor_cos": float(alpha_scores[0]),
            "model_cos": float(np.mean(row_cos(val_pred, target_d[val_ids]))),
            "model_weight_cos": float(np.mean(row_cos(val_weight, nnls_weights[val_ids]))),
            "oracle_cos": float(np.mean(row_cos(nnls_reconstruction[val_ids], target_d[val_ids]))),
            "best_alpha": best_alpha,
            "best_cos": float(alpha_scores[best_alpha_index]),
            "official_gain": official_gain,
            "official_selected": official_selected,
            "alpha_curve": [
                {"alpha": float(alpha), "cos": score}
                for alpha, score in zip(alpha_grid, alpha_scores)
            ],
        },
        "final_training": final_training,
        "projection_diagnostics": (
            {key: summarize(value) for key, value in actions.items()}
            if actions is not None
            else None
        ),
        "local_gate": (
            {
                "required": LOCAL_GATE,
                "score": local_score,
                "passed": bool(float(local_score["score_traceback_leaked"]) > LOCAL_GATE),
                "gap": float(local_score["score_traceback_leaked"]) - LOCAL_GATE,
            }
            if local_score is not None
            else None
        ),
        "outputs": {
            "prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None,
            "targets": PATHS.relative(TARGETS) if TARGETS.is_file() else None,
            "actions": PATHS.relative(ACTIONS) if ACTIONS.is_file() else None,
            "nnls": PATHS.relative(NNLS_CACHE),
            "test_candidates": PATHS.relative(TEST_CANDIDATES),
            "checkpoints": checkpoints,
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
