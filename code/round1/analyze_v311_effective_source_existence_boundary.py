#!/usr/bin/env python3
"""Audit whether effective-source existence has a predictable physical boundary."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.metrics import average_precision_score, roc_auc_score

import analyze_v254_route_topology_spectral_field as v254
import analyze_v273_planar_barrier_kernel as v273
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v300_path_gain_fingerprint_oracle as v300
import analyze_v304_inverse_scatterer_identifiability as v304
import analyze_v305_pointcloud_path_identity_match as v305
import analyze_v309_unmatched_effective_scatterers as v309
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 311
VERSION = "v311_effective_source_existence_boundary"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"

OUTER_SEEDS = (311, 1311, 2311, 3311, 4311)
OUTER_COUNT = 320
INNER_COUNT = 240
CLUSTER_EPS = 3.0
CLUSTER_MIN_SAMPLES = 5
MIN_UNIQUE_UE = 5
MAX_EPOCHS = 40
EVAL_EVERY = 5
NEGATIVE_RATIO = 12
MIN_LENGTH_ERROR = 0.50
MIN_H_ERROR = 1.00


@dataclass
class Observations:
    point: np.ndarray
    point_id: np.ndarray
    component_id: np.ndarray
    mass: np.ndarray
    length: np.ndarray
    h: np.ndarray


class PairMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.SiLU(),
        )
        self.presence = nn.Linear(64, 1)
        self.mass = nn.Linear(64, 1)

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(feature)
        return self.presence(hidden).squeeze(1), self.mass(hidden).squeeze(1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_observations() -> Observations:
    raw = np.load(v309.CLUSTER_CACHE)
    component = np.load(v297.COMPONENT_CACHE)
    fingerprint = np.load(v300.FINGERPRINT_CACHE)
    calibration = json.loads(v304.REPORT.read_text(encoding="utf-8"))["los_calibration"]
    point_id = raw["point_id"].astype(np.int64)
    component_id = raw["component_id"].astype(np.int64)
    length = (
        component["delay"][point_id, component_id] - float(calibration["intercept_bins"])
    ) / float(calibration["slope_bins_per_meter"])
    h = v304.horizontal_center(fingerprint["horizontal"])[point_id, component_id]
    return Observations(
        point=raw["point"].astype(np.float32),
        point_id=point_id,
        component_id=component_id,
        mass=raw["mass"].astype(np.float32),
        length=length.astype(np.float32),
        h=h.astype(np.float32),
    )


def discover_sources(observation: Observations, fit_ids: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    selected = np.isin(observation.point_id, fit_ids)
    point = observation.point[selected]
    point_id = observation.point_id[selected]
    labels = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES).fit_predict(point)
    centers: list[np.ndarray] = []
    occurrence: list[int] = []
    unique_ue: list[int] = []
    for label in np.unique(labels[labels >= 0]):
        take = labels == label
        diversity = len(np.unique(point_id[take]))
        if diversity < MIN_UNIQUE_UE:
            continue
        centers.append(np.median(point[take], axis=0))
        occurrence.append(int(np.sum(take)))
        unique_ue.append(int(diversity))
    if not centers:
        raise RuntimeError("No fold-local effective sources survived the diversity rule")
    return np.asarray(centers, dtype=np.float32), {
        "count": len(centers),
        "occurrence": summarize(np.asarray(occurrence)),
        "unique_ue": summarize(np.asarray(unique_ue)),
        "fit_observation_count": int(np.sum(selected)),
    }


def assign_targets(
    observation: Observations,
    query_ids: np.ndarray,
    centers: np.ndarray,
    train_pos: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    target = np.zeros((len(query_ids), len(centers)), dtype=np.float32)
    row_lookup = {int(point_id): row for row, point_id in enumerate(query_ids)}
    selected = np.isin(observation.point_id, query_ids)
    selected_rows = np.flatnonzero(selected)
    by_point: dict[int, list[int]] = {}
    for observation_row in selected_rows:
        by_point.setdefault(int(observation.point_id[observation_row]), []).append(int(observation_row))
    source_first_length = np.linalg.norm(centers - v304.BS[:2], axis=1)
    source_h = v304.expected_h_bin(
        np.arctan2(centers[:, 1] - v304.BS[1], centers[:, 0] - v304.BS[0])
    )
    represented_mass = 0.0
    total_mass = float(np.sum(observation.mass[selected], dtype=np.float64))
    matched_count = 0
    for point_id, rows in by_point.items():
        rows_array = np.asarray(rows, dtype=np.int64)
        predicted_length = source_first_length + np.linalg.norm(
            centers - train_pos[point_id, :2], axis=1
        )
        length_error = np.abs(observation.length[rows_array, None] - predicted_length[None, :])
        length_error /= v304.METERS_PER_BIN
        h_error = v305.circular_distance(
            observation.h[rows_array, None], source_h[None, :], 16.0
        )
        cost = (length_error / MIN_LENGTH_ERROR) ** 2 + (h_error / MIN_H_ERROR) ** 2
        observation_index, source_index = linear_sum_assignment(cost)
        keep = (
            (length_error[observation_index, source_index] <= MIN_LENGTH_ERROR)
            & (h_error[observation_index, source_index] <= MIN_H_ERROR)
        )
        if not np.any(keep):
            continue
        observation_index = observation_index[keep]
        source_index = source_index[keep]
        confidence = np.exp(-0.5 * cost[observation_index, source_index]).astype(np.float32)
        value = observation.mass[rows_array[observation_index]] * confidence
        np.add.at(target[row_lookup[point_id]], source_index, value)
        represented_mass += float(np.sum(observation.mass[rows_array[observation_index]]))
        matched_count += len(source_index)
    return target, {
        "observation_count": int(np.sum(selected)),
        "matched_count": int(matched_count),
        "represented_unmatched_mass": represented_mass / max(total_mass, 1e-30),
    }


def route_features(
    query_xy: np.ndarray,
    centers: np.ndarray,
    scene: dict[str, np.ndarray],
) -> np.ndarray:
    t = np.linspace(0.04, 0.96, 24, dtype=np.float32)
    point = centers[None, :, None, :] + t[None, None, :, None] * (
        query_xy[:, None, None, :] - centers[None, :, None, :]
    )
    ix = np.floor((point[..., 0] - float(scene["x_min"])) / float(scene["resolution"])).astype(np.int64)
    iy = np.floor((point[..., 1] - float(scene["y_min"])) / float(scene["resolution"])).astype(np.int64)
    shape = scene["component_labels"].shape
    valid = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
    safe_x = np.clip(ix, 0, shape[1] - 1)
    safe_y = np.clip(iy, 0, shape[0] - 1)
    labels = scene["component_labels"][safe_y, safe_x]
    occupied = valid & (labels > 0)
    block_fraction = np.mean(occupied, axis=2)
    run_count = np.sum(occupied & ~np.concatenate([np.zeros_like(occupied[:, :, :1]), occupied[:, :, :-1]], axis=2), axis=2)
    component_change = np.sum(
        occupied[:, :, 1:]
        & occupied[:, :, :-1]
        & (labels[:, :, 1:] != labels[:, :, :-1]),
        axis=2,
    )
    first = np.where(np.any(occupied, axis=2), np.argmax(occupied, axis=2) / 23.0, 1.0)
    last = np.where(
        np.any(occupied, axis=2),
        (23 - np.argmax(occupied[:, :, ::-1], axis=2)) / 23.0,
        0.0,
    )
    z_max = np.where(occupied, scene["grid_z_max"][safe_y, safe_x], 0.0)
    z_max = np.nan_to_num(z_max, nan=0.0, posinf=0.0, neginf=0.0)
    vertical = np.where(occupied, scene["grid_vertical_count"][safe_y, safe_x], 0.0)
    return np.stack(
        [
            block_fraction,
            run_count / 8.0,
            component_change / 8.0,
            first,
            last,
            np.max(z_max, axis=2) / 40.0,
            np.mean(np.log1p(vertical), axis=2) / 8.0,
        ],
        axis=2,
    ).astype(np.float32)


def support_statistics(
    bank_ids: np.ndarray,
    bank_target: np.ndarray,
    train_pos: np.ndarray,
    scene_feature: np.ndarray,
    signature: np.ndarray,
    centers: np.ndarray,
) -> dict[str, np.ndarray | list[cKDTree | None]]:
    source_count = bank_target.shape[1]
    centroid = centers.copy().astype(np.float32)
    radius = np.full(source_count, 150.0, dtype=np.float32)
    covariance_inv = np.broadcast_to(np.eye(2, dtype=np.float32)[None], (source_count, 2, 2)).copy()
    count = np.zeros(source_count, dtype=np.float32)
    mass_mean = np.zeros(source_count, dtype=np.float32)
    mass_p90 = np.zeros(source_count, dtype=np.float32)
    scene_mean = np.zeros((source_count, scene_feature.shape[1]), dtype=np.float32)
    signature_mean = np.zeros((source_count, signature.shape[1]), dtype=np.float32)
    trees: list[cKDTree | None] = []
    bank_xy = train_pos[bank_ids, :2]
    for source in range(source_count):
        positive = bank_target[:, source] > 0.0
        if not np.any(positive):
            trees.append(None)
            continue
        point = bank_xy[positive]
        mass = bank_target[positive, source].astype(np.float64)
        weight = mass / np.maximum(np.sum(mass), 1e-30)
        centroid[source] = np.sum(point * weight[:, None], axis=0)
        delta = point - centroid[source]
        covariance = np.einsum("n,ni,nj->ij", weight, delta, delta)
        covariance += np.eye(2) * 25.0
        covariance_inv[source] = np.linalg.inv(covariance).astype(np.float32)
        radius[source] = max(float(np.quantile(np.linalg.norm(delta, axis=1), 0.90)), 3.0)
        count[source] = float(np.sum(positive))
        mass_mean[source] = float(np.mean(mass))
        mass_p90[source] = float(np.quantile(mass, 0.90))
        scene_mean[source] = np.sum(scene_feature[bank_ids[positive]] * weight[:, None], axis=0)
        signature_mean[source] = np.sum(signature[bank_ids[positive]] * weight[:, None], axis=0)
        trees.append(cKDTree(point))
    return {
        "centroid": centroid,
        "radius": radius,
        "covariance_inv": covariance_inv,
        "count": count,
        "mass_mean": mass_mean,
        "mass_p90": mass_p90,
        "scene_mean": scene_mean,
        "signature_mean": signature_mean,
        "trees": trees,
    }


def pair_features(
    query_ids: np.ndarray,
    centers: np.ndarray,
    support: dict[str, np.ndarray | list[cKDTree | None]],
    train_pos: np.ndarray,
    scene_feature: np.ndarray,
    signature: np.ndarray,
    scene: dict[str, np.ndarray],
) -> tuple[np.ndarray, int]:
    query_xy = train_pos[query_ids, :2].astype(np.float32)
    source_count = len(centers)
    delta = query_xy[:, None, :] - centers[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    direction = delta / np.maximum(distance[:, :, None], 1e-6)
    source_bs = centers - v304.BS[:2]
    source_bs_distance = np.linalg.norm(source_bs, axis=1)
    direct = np.linalg.norm(query_xy - v304.BS[:2], axis=1)
    path_length = source_bs_distance[None, :] + distance
    support_delta = query_xy[:, None, :] - np.asarray(support["centroid"])[None, :, :]
    support_distance = np.linalg.norm(support_delta, axis=2)
    covariance_inv = np.asarray(support["covariance_inv"])
    mahalanobis = np.sqrt(
        np.maximum(np.einsum("qsi,sij,qsj->qs", support_delta, covariance_inv, support_delta), 0.0)
    )
    nearest = np.full_like(distance, 500.0)
    third = np.full_like(distance, 500.0)
    trees = support["trees"]
    assert isinstance(trees, list)
    for source, tree in enumerate(trees):
        if tree is None:
            continue
        take = min(3, int(np.asarray(support["count"])[source]))
        values = tree.query(query_xy, k=take)[0]
        if take == 1:
            nearest[:, source] = values
            third[:, source] = values
        else:
            nearest[:, source] = values[:, 0]
            third[:, source] = values[:, -1]
    departure = np.arctan2(source_bs[:, 1], source_bs[:, 0])
    geometry = np.stack(
        [
            np.broadcast_to(centers[None, :, 0] / 400.0, distance.shape),
            np.broadcast_to(centers[None, :, 1] / 400.0, distance.shape),
            np.broadcast_to(source_bs_distance[None, :] / 500.0, distance.shape),
            delta[:, :, 0] / 500.0,
            delta[:, :, 1] / 500.0,
            distance / 500.0,
            direction[:, :, 0],
            direction[:, :, 1],
            np.broadcast_to(np.sin(departure)[None, :], distance.shape),
            np.broadcast_to(np.cos(departure)[None, :], distance.shape),
            path_length / 800.0,
            (path_length - direct[:, None]) / 500.0,
            support_delta[:, :, 0] / 300.0,
            support_delta[:, :, 1] / 300.0,
            support_distance / 300.0,
            support_distance / np.maximum(np.asarray(support["radius"])[None, :], 3.0),
            np.clip(mahalanobis, 0.0, 20.0) / 10.0,
            nearest / 300.0,
            third / 300.0,
            np.broadcast_to(np.log1p(np.asarray(support["count"]))[None, :] / 8.0, distance.shape),
            np.broadcast_to(np.asarray(support["mass_mean"])[None, :], distance.shape),
            np.broadcast_to(np.asarray(support["mass_p90"])[None, :], distance.shape),
        ],
        axis=2,
    ).astype(np.float32)
    geometry_dim = geometry.shape[2]
    query_scene = scene_feature[query_ids, None, :]
    support_scene = np.asarray(support["scene_mean"])[None, :, :]
    query_scene = np.broadcast_to(query_scene, (len(query_ids), source_count, query_scene.shape[2]))
    support_scene = np.broadcast_to(support_scene, query_scene.shape)
    query_signature = signature[query_ids, None, :].astype(np.float32)
    source_signature = np.asarray(support["signature_mean"])[None, :, :]
    expected_mismatch = np.mean(
        query_signature * (1.0 - source_signature)
        + (1.0 - query_signature) * source_signature,
        axis=2,
    )
    intersection = np.sum(query_signature * source_signature, axis=2)
    union = np.sum(np.maximum(query_signature, source_signature), axis=2)
    signature_cos = intersection / np.maximum(
        np.sqrt(np.sum(query_signature, axis=2) * np.sum(source_signature**2, axis=2)), 1e-6
    )
    signature_feature = np.stack(
        [expected_mismatch, intersection / 10.0, union / 10.0, signature_cos], axis=2
    ).astype(np.float32)
    route = route_features(query_xy, centers, scene)
    full = np.concatenate(
        [geometry, query_scene, query_scene - support_scene, signature_feature, route], axis=2
    )
    return full.astype(np.float32), geometry_dim


def knn_prediction(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    bank_target: np.ndarray,
    train_pos: np.ndarray,
    k: int = 16,
) -> np.ndarray:
    take = min(k, len(bank_ids))
    distance, local = cKDTree(train_pos[bank_ids, :2]).query(train_pos[query_ids, :2], k=take)
    if take == 1:
        distance = distance[:, None]
        local = local[:, None]
    weight = 1.0 / np.maximum(distance, 0.05) ** 2
    weight /= np.sum(weight, axis=1, keepdims=True)
    return np.sum(weight[:, :, None] * bank_target[local], axis=1).astype(np.float32)


def standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = train.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std[std < 1e-5] = 1.0
    return tuple(((value - mean) / std).astype(np.float32) for value in (train,) + others)


def sampled_pairs(feature: np.ndarray, target: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    flat_target = target.reshape(-1)
    positive = np.flatnonzero(flat_target > 0.0)
    negative = np.flatnonzero(flat_target <= 0.0)
    rng = np.random.default_rng(seed)
    count = min(len(negative), max(len(positive) * NEGATIVE_RATIO, 2048))
    selected = np.concatenate([positive, rng.choice(negative, size=count, replace=False)])
    rng.shuffle(selected)
    return feature.reshape(-1, feature.shape[2])[selected], flat_target[selected]


def train_model(
    train_feature: np.ndarray,
    train_target: np.ndarray,
    validation_feature: np.ndarray | None,
    validation_target: np.ndarray | None,
    seed: int,
    epochs: int,
    device: torch.device,
) -> tuple[PairMLP, int, list[dict[str, float]]]:
    set_seed(seed)
    x, y = sampled_pairs(train_feature, train_target, seed)
    x_tensor = torch.from_numpy(x).to(device)
    y_tensor = torch.from_numpy(y).to(device)
    presence = (y_tensor > 0.0).float()
    positive_count = max(float(torch.sum(presence).item()), 1.0)
    negative_count = max(float(len(presence) - positive_count), 1.0)
    positive_weight = torch.tensor(min(negative_count / positive_count, 20.0), device=device)
    model = PairMLP(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)
    batch_size = 4096
    best_epoch = epochs
    best_ap = -math.inf
    best_state = None
    history: list[dict[str, float]] = []
    rng = np.random.default_rng(seed + 17)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(x))
        for start in range(0, len(order), batch_size):
            ids = torch.from_numpy(order[start : start + batch_size]).to(device)
            logits, mass_logits = model(x_tensor[ids])
            batch_presence = presence[ids]
            classification = F.binary_cross_entropy_with_logits(
                logits, batch_presence, pos_weight=positive_weight
            )
            positive = batch_presence > 0.5
            if torch.any(positive):
                regression = F.smooth_l1_loss(
                    torch.sigmoid(mass_logits[positive]), y_tensor[ids][positive], beta=0.05
                )
            else:
                regression = logits.sum() * 0.0
            loss = classification + 3.0 * regression
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if validation_feature is not None and (epoch % EVAL_EVERY == 0 or epoch == epochs):
            prediction = predict_model(model, validation_feature, device)
            metric = evaluate_prediction(prediction, validation_target)
            row = {"epoch": epoch, "pair_ap": metric["pair_ap"], "top5_mass_recall": metric["top5_mass_recall"]}
            history.append(row)
            score = metric["pair_ap"] + 0.25 * metric["top5_mass_recall"]
            if score > best_ap:
                best_ap = score
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, history


@torch.no_grad()
def predict_model(model: PairMLP, feature: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    flat = feature.reshape(-1, feature.shape[2])
    output = np.empty(len(flat), dtype=np.float32)
    for start in range(0, len(flat), 8192):
        value = torch.from_numpy(flat[start : start + 8192]).to(device)
        presence, mass = model(value)
        output[start : start + len(value)] = (
            torch.sigmoid(presence) * torch.sigmoid(mass)
        ).cpu().numpy()
    return output.reshape(feature.shape[:2])


def evaluate_prediction(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    truth = target > 0.0
    flat_truth = truth.reshape(-1)
    flat_prediction = prediction.reshape(-1)
    pair_ap = float(average_precision_score(flat_truth, flat_prediction))
    pair_auc = float(roc_auc_score(flat_truth, flat_prediction)) if np.any(flat_truth) and np.any(~flat_truth) else 0.5
    valid = np.sum(target, axis=1) > 0.0
    if not np.any(valid):
        return {
            "pair_ap": pair_ap,
            "pair_auc": pair_auc,
            "mass_cosine": 0.0,
            "top3_mass_recall": 0.0,
            "top5_mass_recall": 0.0,
            "positive_query_fraction": 0.0,
        }
    pred = prediction[valid]
    value = target[valid]
    cosine = np.sum(pred * value, axis=1) / np.maximum(
        np.linalg.norm(pred, axis=1) * np.linalg.norm(value, axis=1), 1e-20
    )
    recalls = {}
    for k in (3, 5):
        take = min(k, pred.shape[1])
        top = np.argpartition(pred, -take, axis=1)[:, -take:]
        selected = np.take_along_axis(value, top, axis=1).sum(axis=1)
        recalls[k] = float(np.sum(selected) / np.maximum(np.sum(value), 1e-20))
    return {
        "pair_ap": pair_ap,
        "pair_auc": pair_auc,
        "mass_cosine": float(np.mean(cosine)),
        "top3_mass_recall": recalls[3],
        "top5_mass_recall": recalls[5],
        "positive_query_fraction": float(np.mean(valid)),
    }


def main() -> None:
    observation = load_observations()
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    targets = np.load(v273.TARGET_CACHE)
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    scene_data = np.load(v273.SCENE_CACHE)
    scene = {key: scene_data[key] for key in scene_data.files}
    scene_feature = scene["train_features"].astype(np.float32)
    if not v273.FEATURE_CACHE.is_file():
        raise FileNotFoundError(v273.FEATURE_CACHE)
    blocker = np.load(v273.FEATURE_CACHE)
    signature = blocker["train_signature"].astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_rows: list[dict[str, object]] = []
    metric_names = ("prevalence", "knn", "geometry_mlp", "scene_mlp", "shuffled_scene_mlp")
    pooled_prediction: dict[str, list[np.ndarray]] = {name: [] for name in metric_names}
    pooled_target: list[np.ndarray] = []
    for fold, seed in enumerate(OUTER_SEEDS):
        outer_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], OUTER_COUNT, seed
            ),
            valid_ids,
        )
        outer_fit = np.setdiff1d(valid_ids, outer_ids)
        centers, source_report = discover_sources(observation, outer_fit)
        fit_target, fit_assignment = assign_targets(observation, outer_fit, centers, train_pos)
        outer_target, outer_assignment = assign_targets(observation, outer_ids, centers, train_pos)
        inner_ids = v273.matched_subset(
            outer_fit, train_pos[:, :2], test_pos[:, :2], INNER_COUNT, seed + 10_000
        )
        inner_fit = np.setdiff1d(outer_fit, inner_ids)
        inner_fit_rows = np.searchsorted(outer_fit, inner_fit)
        inner_rows = np.searchsorted(outer_fit, inner_ids)

        inner_support = support_statistics(
            inner_fit,
            fit_target[inner_fit_rows],
            train_pos,
            scene_feature,
            signature,
            centers,
        )
        inner_fit_feature, geometry_dim = pair_features(
            inner_fit, centers, inner_support, train_pos, scene_feature, signature, scene
        )
        inner_val_feature, _ = pair_features(
            inner_ids, centers, inner_support, train_pos, scene_feature, signature, scene
        )
        inner_fit_feature, inner_val_feature = standardize(inner_fit_feature, inner_val_feature)

        selected_epochs: dict[str, int] = {}
        inner_history: dict[str, list[dict[str, float]]] = {}
        for name, take in (("geometry_mlp", geometry_dim), ("scene_mlp", inner_fit_feature.shape[2])):
            _model, best_epoch, history = train_model(
                inner_fit_feature[:, :, :take],
                fit_target[inner_fit_rows],
                inner_val_feature[:, :, :take],
                fit_target[inner_rows],
                seed + (0 if name == "geometry_mlp" else 100),
                MAX_EPOCHS,
                device,
            )
            selected_epochs[name] = best_epoch
            inner_history[name] = history

        full_support = support_statistics(
            outer_fit, fit_target, train_pos, scene_feature, signature, centers
        )
        fit_feature, geometry_dim = pair_features(
            outer_fit, centers, full_support, train_pos, scene_feature, signature, scene
        )
        outer_feature, _ = pair_features(
            outer_ids, centers, full_support, train_pos, scene_feature, signature, scene
        )
        fit_feature, outer_feature = standardize(fit_feature, outer_feature)
        predictions: dict[str, np.ndarray] = {}
        prevalence = np.mean(fit_target > 0.0, axis=0, keepdims=True).astype(np.float32)
        predictions["prevalence"] = np.broadcast_to(prevalence, outer_target.shape).copy()
        predictions["knn"] = knn_prediction(outer_ids, outer_fit, fit_target, train_pos)
        for name, take in (("geometry_mlp", geometry_dim), ("scene_mlp", fit_feature.shape[2])):
            model, _epoch, _history = train_model(
                fit_feature[:, :, :take],
                fit_target,
                None,
                None,
                seed + 500 + (0 if name == "geometry_mlp" else 100),
                selected_epochs[name],
                device,
            )
            predictions[name] = predict_model(model, outer_feature[:, :, :take], device)

        rng = np.random.default_rng(seed + 31_100)
        shuffled_target = fit_target.copy()
        for source in range(shuffled_target.shape[1]):
            shuffled_target[:, source] = shuffled_target[rng.permutation(len(shuffled_target)), source]
        shuffled_support = support_statistics(
            outer_fit, shuffled_target, train_pos, scene_feature, signature, centers
        )
        shuffled_fit_feature, _ = pair_features(
            outer_fit, centers, shuffled_support, train_pos, scene_feature, signature, scene
        )
        shuffled_outer_feature, _ = pair_features(
            outer_ids, centers, shuffled_support, train_pos, scene_feature, signature, scene
        )
        shuffled_fit_feature, shuffled_outer_feature = standardize(
            shuffled_fit_feature, shuffled_outer_feature
        )
        shuffled_model, _epoch, _history = train_model(
            shuffled_fit_feature,
            shuffled_target,
            None,
            None,
            seed + 900,
            selected_epochs["scene_mlp"],
            device,
        )
        predictions["shuffled_scene_mlp"] = predict_model(
            shuffled_model, shuffled_outer_feature, device
        )

        metrics = {name: evaluate_prediction(value, outer_target) for name, value in predictions.items()}
        fold_rows.append(
            {
                "fold": fold,
                "seed": seed,
                "source_discovery": source_report,
                "fit_assignment": fit_assignment,
                "outer_assignment": outer_assignment,
                "selected_epochs": selected_epochs,
                "inner_history": inner_history,
                "metrics": metrics,
            }
        )
        for name in metric_names:
            pooled_prediction[name].append(predictions[name])
        pooled_target.append(outer_target)
        print(
            f"v311 fold={fold} sources={len(centers)} "
            f"knn_ap={metrics['knn']['pair_ap']:.4f} "
            f"geometry_ap={metrics['geometry_mlp']['pair_ap']:.4f} "
            f"scene_ap={metrics['scene_mlp']['pair_ap']:.4f} "
            f"scene_top5={metrics['scene_mlp']['top5_mass_recall']:.4f}",
            flush=True,
        )

    pooled: dict[str, dict[str, float]] = {}
    flat_truth = np.concatenate([(value > 0.0).reshape(-1) for value in pooled_target])
    for name in metric_names:
        flat_prediction = np.concatenate([value.reshape(-1) for value in pooled_prediction[name]])
        pooled[name] = {
            "pair_ap": float(average_precision_score(flat_truth, flat_prediction)),
            "pair_auc": float(roc_auc_score(flat_truth, flat_prediction)),
        }
        for metric in (
            "mass_cosine",
            "top3_mass_recall",
            "top5_mass_recall",
            "positive_query_fraction",
        ):
            pooled[name][metric] = float(
                np.mean([row["metrics"][name][metric] for row in fold_rows])
            )
    fold_scene_wins = sum(
        row["metrics"]["scene_mlp"]["top5_mass_recall"]
        > row["metrics"]["knn"]["top5_mass_recall"]
        for row in fold_rows
    )
    passed = (
        pooled["scene_mlp"]["pair_ap"] >= 4.0 * pooled["shuffled_scene_mlp"]["pair_ap"]
        and pooled["scene_mlp"]["mass_cosine"] >= pooled["knn"]["mass_cosine"] + 0.05
        and fold_scene_wins >= 4
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "The v309 effective sources are useful only inside source-specific spatial/scene "
            "existence regions that can be learned from official point-cloud conditions."
        ),
        "leakage_control": (
            "Every outer fold rediscovers DBSCAN sources using outer-fit UE observations only; "
            "outer UE components are used only to score source existence and mass."
        ),
        "fold_rows": fold_rows,
        "pooled_metrics": pooled,
        "decision_rule": {
            "scene_pair_ap_vs_shuffle": "at least 4x",
            "scene_mass_cosine_vs_knn": "at least +0.05",
            "scene_top5_recall_fold_wins_vs_knn": "at least 4/5",
            "passed": bool(passed),
        },
        "next_step": (
            "Train a support-masked joint renderer only if the decision rule passes; otherwise "
            "stop the effective-source deployment branch and retain v308."
        ),
        "outputs": {"report": PATHS.relative(REPORT), "zip": None},
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"pooled_metrics": pooled, "decision_rule": report["decision_rule"]}, indent=2))


if __name__ == "__main__":
    main()
