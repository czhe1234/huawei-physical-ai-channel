#!/usr/bin/env python3
"""Transfer the pre-Sionna regime/component lifecycle chain onto Round 2 V4."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
import torch

from analyze_v4_exact_joint_projection import (
    exact_ad_projection,
    exact_pas_projection,
    exact_pdp_projection,
    pas_distribution,
    pdp_distribution,
    preserve_point_power,
)
from build_geodesic_corridor_features import CACHE as GEODESIC_CACHE
from build_geodesic_corridor_features import build_features as build_geodesic_features
from build_v3_anchored_neural_rbf import TRAIN_AD
from data import load_positions
from geometry_features import build_geometry_features
from project_paths import PATHS
from run_rbf_spectral_field import OOF_BASE, OOF_META, statistics
from run_v4_calibrated_exact_projection import (
    EXACT_PATTERN,
    OOF_CALIBRATED_PAS,
    OOF_CALIBRATED_PDP,
    TARGET_AD,
    TARGET_PAS,
    TARGET_PDP,
    action_by_name,
    initialize_trajectory,
    mode_targets,
    continue_trajectory,
)
from spatial_hole_cv import SpatialHoleFold, build_spatial_hole_folds


VERSION = "Round2-E41-regime-component-lifecycle"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "regime_lifecycle_support"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_regime_lifecycle_support"
VERSION_DIR = PATHS.analysis / "results" / "version_reports" / "V6"

V4_SELECTION = (
    PATHS.artifacts
    / "round2"
    / "v4_calibrated_exact_projection"
    / "calibrated_exact_projection_selection.json"
)
V4_TEST = (
    PATHS.artifacts
    / "round2"
    / "v4_calibrated_exact_projection"
    / "Round2_V4_Calibrated_Exact_Projection_Test_Channel.npy"
)
ROUTE_CACHE = (
    PATHS.artifacts / "round2" / "route_topology" / "round2_route_topology_features.npz"
)

OOF_V4 = ARTIFACT_DIR / "oof_v4_channel.npy"
OOF_CURRENT_AD = ARTIFACT_DIR / "oof_v4_current_ad_f16.npy"
TEST_CURRENT_AD = ARTIFACT_DIR / "test_v4_current_ad_f16.npy"
OOF_TEMPLATE = ARTIFACT_DIR / "oof_regime_template_f16.npy"
TEST_TEMPLATE = ARTIFACT_DIR / "test_regime_template_f16.npy"
OOF_LIFECYCLE_PLAIN = ARTIFACT_DIR / "oof_lifecycle_plain_f16.npy"
OOF_LIFECYCLE_VISIBILITY = ARTIFACT_DIR / "oof_lifecycle_visibility_f16.npy"
OOF_LIFECYCLE_SHIFT = ARTIFACT_DIR / "oof_lifecycle_shift_f16.npy"
TEST_LIFECYCLE_PLAIN = ARTIFACT_DIR / "test_lifecycle_plain_f16.npy"
TEST_LIFECYCLE_VISIBILITY = ARTIFACT_DIR / "test_lifecycle_visibility_f16.npy"
TEST_LIFECYCLE_SHIFT = ARTIFACT_DIR / "test_lifecycle_shift_f16.npy"
REGIME_CACHE = ARTIFACT_DIR / "regime_assignments.npz"
SUPPORT_REPORT = ANALYSIS_DIR / "support_stage_ablation.json"
METRICS = ARTIFACT_DIR / "v6_projection_metrics.npz"
SELECTION = ARTIFACT_DIR / "v6_selection.json"
OUTPUT = ARTIFACT_DIR / "Round2_V6_Regime_Component_Lifecycle_Test_Channel.npy"
REPORT = ANALYSIS_DIR / "regime_component_lifecycle.json"
VERSION_REPORT = VERSION_DIR / "V6_regime_component_lifecycle.json"

REGIME_NAMES = ("clear", "light", "medium", "heavy")
TARGET_PATHS = {
    "template": (OOF_TEMPLATE, TEST_TEMPLATE),
    "lifecycle_plain": (OOF_LIFECYCLE_PLAIN, TEST_LIFECYCLE_PLAIN),
    "lifecycle_visibility": (
        OOF_LIFECYCLE_VISIBILITY,
        TEST_LIFECYCLE_VISIBILITY,
    ),
    "lifecycle_shift": (OOF_LIFECYCLE_SHIFT, TEST_LIFECYCLE_SHIFT),
}
PROFILE_ALPHA = {
    "soft": np.asarray((0.04, 0.08, 0.16, 0.24), dtype=np.float32),
    "balanced": np.asarray((0.06, 0.15, 0.30, 0.42), dtype=np.float32),
    "focused": np.asarray((0.00, 0.08, 0.32, 0.55), dtype=np.float32),
}
PROJECTION_MODES = ("ad", "metric", "joint")
PROJECTION_CYCLES = 2
TINY = 1e-30


@dataclass(frozen=True)
class Candidate:
    target: str
    profile: str
    mode: str

    @property
    def name(self) -> str:
        return f"{self.target}_{self.profile}_{self.mode}"


def normalize_support(value: np.ndarray) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = array.sum(axis=(1, 2, 3), keepdims=True, dtype=np.float64)
    return np.divide(
        array,
        np.maximum(total, 1e-20),
        out=np.zeros_like(array),
        where=total > 0.0,
    )


def support_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_root = np.sqrt(np.maximum(left, 0.0)).reshape(len(left), -1)
    right_root = np.sqrt(np.maximum(right, 0.0)).reshape(len(right), -1)
    return np.sum(left_root * right_root, axis=1, dtype=np.float64).astype(
        np.float32
    )


def channel_ad_support(
    channel: np.ndarray, output_path, device: torch.device, batch_size: int
) -> np.ndarray:
    if output_path.is_file():
        cached = np.load(output_path, mmap_mode="r")
        if cached.shape == (len(channel), 16, 8, 192):
            return cached
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float16, shape=(len(channel), 16, 8, 192)
    )
    for start in range(0, len(channel), batch_size):
        end = min(start + batch_size, len(channel))
        value = torch.as_tensor(
            np.array(channel[start:end], dtype=np.complex64, copy=True), device=device
        )
        with torch.inference_mode():
            shaped = value.reshape(len(value), 2, 16, 8, 4, 192)
            transformed = torch.fft.ifft(
                torch.fft.fft2(shaped, dim=(2, 3)), dim=-1
            )
            power = transformed.abs().square().sum(dim=(1, 4)).float()
            power /= power.sum(dim=(1, 2, 3), keepdim=True).clamp_min(TINY)
        output[start:end] = power.cpu().numpy().astype(np.float16)
        output.flush()
    return np.load(output_path, mmap_mode="r")


def _parse_v4_candidate(
    base: torch.Tensor,
    ad: torch.Tensor,
    raw_pas: torch.Tensor,
    raw_pdp: torch.Tensor,
    calibrated_pas: torch.Tensor,
    calibrated_pdp: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if name.startswith("legacy_"):
        from pre_sionna_spectral import project_channel

        action = action_by_name(name.removeprefix("legacy_"))
        return project_channel(
            base,
            ad,
            raw_pas,
            raw_pdp,
            action.alpha_ad,
            action.alpha_pas,
            action.alpha_pdp,
        )
    match = EXACT_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(name)
    mode, order, action_name, cycles_text = match.groups()
    from run_rbf_spectral_field import ACTIONS

    action_index = next(
        index for index, action in enumerate(ACTIONS) if action.name == action_name
    )
    pas, pdp = mode_targets(
        mode, raw_pas, raw_pdp, calibrated_pas, calibrated_pdp
    )
    candidate, desired_pas, desired_pdp, reference_power = initialize_trajectory(
        base, ad, pas, pdp, action_index, order
    )
    for _ in range(2, int(cycles_text) + 1):
        candidate = continue_trajectory(
            candidate, desired_pas, desired_pdp, reference_power, order
        )
    return candidate


def replay_oof_v4(device: torch.device, batch_size: int) -> np.ndarray:
    if OOF_V4.is_file():
        cached = np.load(OOF_V4, mmap_mode="r")
        if cached.shape == (500, 256, 4, 192):
            return cached
    metadata = np.load(OOF_META)
    fold_id = metadata["fold_id"].astype(np.int8)
    query_ids = metadata["query_ids"].astype(np.int64)
    _, _, _, train_site, _ = load_positions()
    query_site = train_site[query_ids]
    selection = json.loads(V4_SELECTION.read_text(encoding="utf-8"))
    decisions = {
        (int(row["held_fold"]), int(row["site"])): row
        for row in selection["nested_decisions"]
    }
    base = np.load(OOF_BASE, mmap_mode="r")
    ad_cache = np.load(TARGET_AD, mmap_mode="r")
    pas_cache = np.load(TARGET_PAS, mmap_mode="r")
    pdp_cache = np.load(TARGET_PDP, mmap_mode="r")
    calibrated_pas_cache = np.load(OOF_CALIBRATED_PAS, mmap_mode="r")
    calibrated_pdp_cache = np.load(OOF_CALIBRATED_PDP, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OOF_V4, mode="w+", dtype=np.complex64, shape=base.shape
    )
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        base_tensor = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        ad = torch.as_tensor(np.asarray(ad_cache[start:end], dtype=np.float32), device=device)
        raw_pas = torch.as_tensor(
            np.asarray(pas_cache[start:end], dtype=np.float32), device=device
        )
        raw_pdp = torch.as_tensor(
            np.asarray(pdp_cache[start:end], dtype=np.float32), device=device
        )
        calibrated_pas = torch.as_tensor(
            np.asarray(calibrated_pas_cache[start:end], dtype=np.float32), device=device
        )
        calibrated_pdp = torch.as_tensor(
            np.asarray(calibrated_pdp_cache[start:end], dtype=np.float32), device=device
        )
        block = np.empty_like(np.asarray(base[start:end]))
        block_fold = fold_id[start:end]
        block_site = query_site[start:end]
        with torch.inference_mode():
            for fold_value in np.unique(block_fold):
                for site_value in np.unique(block_site):
                    local = np.flatnonzero(
                        (block_fold == fold_value) & (block_site == site_value)
                    )
                    if len(local) == 0:
                        continue
                    row = decisions[(int(fold_value), int(site_value))]
                    candidate = _parse_v4_candidate(
                        base_tensor[local],
                        ad[local],
                        raw_pas[local],
                        raw_pdp[local],
                        calibrated_pas[local],
                        calibrated_pdp[local],
                        str(row["candidate"]),
                    )
                    block[local] = (
                        candidate.cpu().numpy() * np.float32(row["scale"])
                    ).astype(np.complex64)
        output[start:end] = block
        output.flush()
        print(f"V4 OOF replay {end}/{len(base)}", flush=True)
    return np.load(OOF_V4, mmap_mode="r")


def support_descriptor(value: np.ndarray) -> np.ndarray:
    support = normalize_support(value)
    horizontal_vertical = support.sum(axis=3, dtype=np.float64).astype(np.float32)
    delay = support.sum(axis=(1, 2), dtype=np.float64).astype(np.float32)
    horizontal_delay = support.sum(axis=2, dtype=np.float64).astype(np.float32)
    vertical_delay = support.sum(axis=1, dtype=np.float64).astype(np.float32)
    horizontal_delay = horizontal_delay.reshape(len(support), 16, 24, 8).sum(axis=3)
    vertical_delay = vertical_delay.reshape(len(support), 8, 24, 8).sum(axis=3)
    parts = [horizontal_vertical, delay, horizontal_delay, vertical_delay]
    output = []
    for part in parts:
        root = np.sqrt(np.maximum(part.reshape(len(part), -1), 0.0))
        root /= np.maximum(np.linalg.norm(root, axis=1, keepdims=True), 1e-20)
        output.append(root.astype(np.float32))
    return np.concatenate(output, axis=1).astype(np.float32)


def _risk_columns(names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    index = {name: slot for slot, name in enumerate(names)}
    columns = np.asarray(
        [
            index["path_wall_fraction"],
            index["mean_path_cost"],
            index["geodesic_ratio"],
            index["route_occupancy_fraction"],
            index["panorama_blocked_fraction"],
            index["mean_clearance"],
        ],
        dtype=np.int64,
    )
    weights = np.asarray((0.25, 0.23, 0.20, 0.14, 0.10, -0.08), dtype=np.float32)
    return columns, weights


def physical_risk_and_regime(
    train_features: np.ndarray,
    query_features: np.ndarray,
    train_site: np.ndarray,
    query_site: np.ndarray,
    fit_indices: np.ndarray,
    names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    columns, weights = _risk_columns(names)
    train_risk = np.zeros(len(train_features), dtype=np.float32)
    query_risk = np.zeros(len(query_features), dtype=np.float32)
    train_regime = np.zeros(len(train_features), dtype=np.int8)
    query_regime = np.zeros(len(query_features), dtype=np.int8)
    diagnostics: dict[str, object] = {"sites": {}}
    for site_value in np.unique(train_site):
        fit = fit_indices[train_site[fit_indices] == site_value]
        train_rows = np.flatnonzero(train_site == site_value)
        query_rows = np.flatnonzero(query_site == site_value)
        reference = train_features[fit][:, columns]
        median = np.median(reference, axis=0)
        q25, q75 = np.quantile(reference, (0.25, 0.75), axis=0)
        scale = np.maximum(q75 - q25, 1e-6)
        train_standard = np.clip(
            (train_features[train_rows][:, columns] - median) / scale, -2.5, 3.5
        )
        query_standard = np.clip(
            (query_features[query_rows][:, columns] - median) / scale, -2.5, 3.5
        )
        train_risk[train_rows] = train_standard @ weights
        query_risk[query_rows] = query_standard @ weights
        thresholds = np.quantile(train_risk[fit], (0.25, 0.50, 0.75))
        train_regime[train_rows] = np.digitize(
            train_risk[train_rows], thresholds
        ).astype(np.int8)
        query_regime[query_rows] = np.digitize(
            query_risk[query_rows], thresholds
        ).astype(np.int8)
        diagnostics["sites"][str(int(site_value))] = {
            "thresholds": thresholds.tolist(),
            "fit_count": int(len(fit)),
            "query_counts": {
                REGIME_NAMES[regime]: int(np.sum(query_regime[query_rows] == regime))
                for regime in range(4)
            },
        }
    return train_risk, query_risk, train_regime, query_regime, diagnostics


def _classifier_probability(
    classifier: ExtraTreesClassifier, features: np.ndarray, classes: int
) -> np.ndarray:
    raw = classifier.predict_proba(features)
    output = np.zeros((len(features), classes), dtype=np.float32)
    output[:, classifier.classes_.astype(np.int64)] = raw.astype(np.float32)
    return output


def build_regime_template(
    truth_support: np.ndarray,
    truth_descriptor: np.ndarray,
    current_support: np.ndarray,
    current_descriptor: np.ndarray,
    train_features: np.ndarray,
    query_features: np.ndarray,
    train_site: np.ndarray,
    query_site: np.ndarray,
    train_regime: np.ndarray,
    query_regime: np.ndarray,
    fit_indices: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    output = np.zeros_like(current_support, dtype=np.float32)
    confidence = np.zeros(len(current_support), dtype=np.float32)
    records: list[dict[str, object]] = []
    valid = np.asarray(truth_support).sum(axis=(1, 2, 3), dtype=np.float64) > 0.0
    for site_value in np.unique(query_site):
        for regime in range(4):
            rows = np.flatnonzero(
                (query_site == site_value) & (query_regime == regime)
            )
            fit = fit_indices[
                (train_site[fit_indices] == site_value)
                & (train_regime[fit_indices] == regime)
                & valid[fit_indices]
            ]
            if len(rows) == 0:
                continue
            if len(fit) < 20:
                fit = fit_indices[
                    (train_site[fit_indices] == site_value) & valid[fit_indices]
                ]
            cluster_count = min(6, max(1, len(fit) // 100))
            if cluster_count == 1:
                labels = np.zeros(len(fit), dtype=np.int32)
            else:
                labels = KMeans(
                    n_clusters=cluster_count,
                    random_state=seed + 97 * int(site_value) + 11 * regime,
                    n_init=10,
                    max_iter=200,
                ).fit_predict(truth_descriptor[fit])
            prototypes = np.empty(
                (cluster_count, 16, 8, 192), dtype=np.float32
            )
            for cluster in range(cluster_count):
                members = fit[labels == cluster]
                root = np.sqrt(
                    np.maximum(np.asarray(truth_support[members], dtype=np.float32), 0.0)
                )
                prototypes[cluster] = np.square(np.mean(root, axis=0))
            prototypes = normalize_support(prototypes)
            prototype_descriptor = support_descriptor(prototypes)
            if cluster_count == 1:
                geometry_probability = np.ones((len(rows), 1), dtype=np.float32)
            else:
                classifier = ExtraTreesClassifier(
                    n_estimators=320,
                    min_samples_leaf=8,
                    max_features=0.78,
                    class_weight="balanced",
                    random_state=seed + 1009 * int(site_value) + 31 * regime,
                    n_jobs=-1,
                )
                classifier.fit(train_features[fit], labels)
                geometry_probability = _classifier_probability(
                    classifier, query_features[rows], cluster_count
                )
            similarity = current_descriptor[rows] @ prototype_descriptor.T
            logits = (similarity - np.max(similarity, axis=1, keepdims=True)) / 0.055
            support_probability = np.exp(np.clip(logits, -50.0, 0.0)).astype(
                np.float32
            )
            support_probability /= np.maximum(
                support_probability.sum(axis=1, keepdims=True), 1e-20
            )
            probability = 0.45 * geometry_probability + 0.55 * support_probability
            probability /= np.maximum(probability.sum(axis=1, keepdims=True), 1e-20)
            root_prototype = np.sqrt(np.maximum(prototypes, 0.0)).reshape(
                cluster_count, -1
            )
            template = np.square(probability @ root_prototype).reshape(
                len(rows), 16, 8, 192
            )
            output[rows] = normalize_support(template)
            confidence[rows] = np.max(probability, axis=1)
            records.append(
                {
                    "site": int(site_value),
                    "regime": REGIME_NAMES[regime],
                    "fit_count": int(len(fit)),
                    "query_count": int(len(rows)),
                    "clusters": int(cluster_count),
                    "confidence_mean": float(np.mean(confidence[rows])),
                }
            )
    missing = output.sum(axis=(1, 2, 3)) <= 0.0
    output[missing] = current_support[missing]
    return normalize_support(output), confidence, records


def extract_peaks(
    support: np.ndarray,
    peak_count: int = 20,
    candidate_count: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = normalize_support(support)
    flat = value.reshape(len(value), -1)
    take = min(candidate_count, flat.shape[1])
    candidate = np.argpartition(flat, -take, axis=1)[:, -take:]
    candidate_value = np.take_along_axis(flat, candidate, axis=1)
    order = np.argsort(candidate_value, axis=1)[:, ::-1]
    candidate = np.take_along_axis(candidate, order, axis=1)
    candidate_value = np.take_along_axis(candidate_value, order, axis=1)
    coordinates = np.zeros((len(value), peak_count, 3), dtype=np.int16)
    energy = np.zeros((len(value), peak_count), dtype=np.float32)
    valid = np.zeros((len(value), peak_count), dtype=bool)
    for row in range(len(value)):
        selected: list[tuple[int, int, int]] = []
        for flat_index, candidate_energy in zip(candidate[row], candidate_value[row]):
            if candidate_energy <= 1e-7:
                break
            horizontal = int(flat_index // (8 * 192))
            remainder = int(flat_index % (8 * 192))
            vertical = int(remainder // 192)
            delay = int(remainder % 192)
            keep = True
            for old_h, old_v, old_d in selected:
                dh = min(abs(horizontal - old_h), 16 - abs(horizontal - old_h))
                dv = min(abs(vertical - old_v), 8 - abs(vertical - old_v))
                dd = min(abs(delay - old_d), 192 - abs(delay - old_d))
                if dh <= 1 and dv <= 1 and dd <= 3:
                    keep = False
                    break
            if not keep:
                continue
            slot = len(selected)
            selected.append((horizontal, vertical, delay))
            coordinates[row, slot] = (horizontal, vertical, delay)
            energy[row, slot] = float(candidate_energy)
            valid[row, slot] = True
            if len(selected) == peak_count:
                break
    return coordinates, energy, valid


def circular_delta(value: int, reference: int, size: int) -> int:
    return int((value - reference + size // 2) % size - size // 2)


def spectral_cost(coordinates: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    dh = np.abs(coordinates[:, 0].astype(np.int32) - int(candidate[0]))
    dv = np.abs(coordinates[:, 1].astype(np.int32) - int(candidate[1]))
    dd = np.abs(coordinates[:, 2].astype(np.int32) - int(candidate[2]))
    dh = np.minimum(dh, 16 - dh)
    dv = np.minimum(dv, 8 - dv)
    dd = np.minimum(dd, 192 - dd)
    return (dh / 1.5) ** 2 + (dv / 1.0) ** 2 + (dd / 3.0) ** 2


def nearest_peak(
    coordinates: np.ndarray,
    energy: np.ndarray,
    valid: np.ndarray,
    candidate: np.ndarray,
) -> tuple[bool, np.ndarray, float]:
    rows = np.flatnonzero(valid)
    if len(rows) == 0:
        return False, candidate, 0.0
    cost = spectral_cost(coordinates[rows], candidate)
    best = int(np.argmin(cost))
    if float(cost[best]) > 1.0:
        return False, candidate, 0.0
    slot = rows[best]
    return True, coordinates[slot], float(energy[slot])


def deduplicate_candidates(
    candidates: list[tuple[np.ndarray, float]], maximum: int = 44
) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for coordinate, _ in sorted(candidates, key=lambda item: item[1], reverse=True):
        duplicate = False
        for old in output:
            if float(spectral_cost(old[None, :], coordinate)[0]) <= 0.65:
                duplicate = True
                break
        if not duplicate:
            output.append(np.asarray(coordinate, dtype=np.int16))
        if len(output) == maximum:
            break
    return output


def _weighted_coordinate(
    coordinates: list[np.ndarray], weights: list[float], reference: np.ndarray
) -> np.ndarray:
    if not coordinates:
        return reference.astype(np.int16)
    weight = np.asarray(weights, dtype=np.float64)
    weight /= np.maximum(weight.sum(), 1e-20)
    result = []
    for axis, size in enumerate((16, 8, 192)):
        angle = 2.0 * np.pi * np.asarray(
            [coordinate[axis] for coordinate in coordinates], dtype=np.float64
        ) / size
        mean = np.sum(weight * np.exp(1j * angle))
        if abs(mean) < 1e-8:
            result.append(int(reference[axis]))
        else:
            result.append(int(round((np.angle(mean) % (2.0 * np.pi)) * size / (2.0 * np.pi))) % size)
    return np.asarray(result, dtype=np.int16)


def _add_kernel(grid: np.ndarray, coordinate: np.ndarray, mass: float) -> None:
    entries: list[tuple[int, int, int, float]] = []
    for dh in range(-1, 2):
        for dv in range(-1, 2):
            for dd in range(-3, 4):
                weight = math.exp(
                    -0.5
                    * (
                        (dh / 0.75) ** 2
                        + (dv / 0.65) ** 2
                        + (dd / 1.35) ** 2
                    )
                )
                entries.append(
                    (
                        (int(coordinate[0]) + dh) % 16,
                        (int(coordinate[1]) + dv) % 8,
                        (int(coordinate[2]) + dd) % 192,
                        weight,
                    )
                )
    total = sum(entry[3] for entry in entries)
    for horizontal, vertical, delay, weight in entries:
        grid[horizontal, vertical, delay] += np.float32(mass * weight / total)


def _route_signatures(prefix: str) -> tuple[np.ndarray, np.ndarray]:
    route = np.load(ROUTE_CACHE, allow_pickle=False)
    occupancy = np.asarray(route[f"{prefix}_route_occupancy"], dtype=np.float32)
    panorama = np.asarray(route[f"{prefix}_panorama"], dtype=np.float32)
    return occupancy.reshape(len(occupancy), -1), panorama.reshape(len(panorama), -1)


def _edge_multiplier(
    query_global: int,
    neighbor_global: int,
    query_geo: np.ndarray,
    neighbor_geo: np.ndarray,
    query_occupancy: np.ndarray,
    neighbor_occupancy: np.ndarray,
    query_panorama: np.ndarray,
    neighbor_panorama: np.ndarray,
    query_regime: int,
    neighbor_regime: int,
) -> float:
    geo_delta = query_geo - neighbor_geo
    geo_factor = math.exp(-0.20 * float(np.mean(geo_delta * geo_delta)))
    occupancy_delta = float(
        np.mean(np.abs(query_occupancy - neighbor_occupancy))
    )
    panorama_delta = float(np.mean(np.abs(query_panorama - neighbor_panorama)))
    topology = math.exp(-2.2 * occupancy_delta - 1.4 * panorama_delta)
    regime = 1.0 if query_regime == neighbor_regime else 0.55
    del query_global, neighbor_global
    return geo_factor * topology * regime


def build_lifecycle_targets(
    current_support: np.ndarray,
    template_support: np.ndarray,
    query_positions: np.ndarray,
    query_site: np.ndarray,
    query_global_ids: np.ndarray,
    fit_indices: np.ndarray,
    train_positions: np.ndarray,
    train_site: np.ndarray,
    truth_peaks: tuple[np.ndarray, np.ndarray, np.ndarray],
    current_peaks: tuple[np.ndarray, np.ndarray, np.ndarray],
    template_peaks: tuple[np.ndarray, np.ndarray, np.ndarray],
    train_regime: np.ndarray,
    query_regime: np.ndarray,
    train_risk: np.ndarray,
    query_risk: np.ndarray,
    train_geodesic: np.ndarray,
    query_geodesic: np.ndarray,
    train_route: tuple[np.ndarray, np.ndarray],
    query_route: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    truth_coord, truth_energy, truth_valid = truth_peaks
    current_coord, current_energy, current_valid = current_peaks
    template_coord, template_energy, template_valid = template_peaks
    plain = np.empty_like(current_support, dtype=np.float32)
    visibility = np.empty_like(current_support, dtype=np.float32)
    shifted = np.empty_like(current_support, dtype=np.float32)
    train_occupancy, train_panorama = train_route
    query_occupancy, query_panorama = query_route
    selected_geo = np.asarray((2, 3, 4, 5, 10, 11, 12, 13, 16, 17, 18, 25), dtype=np.int64)
    reference = train_geodesic[fit_indices][:, selected_geo]
    median = np.median(reference, axis=0)
    q25, q75 = np.quantile(reference, (0.25, 0.75), axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    standardized_train = np.clip(
        (train_geodesic[:, selected_geo] - median) / scale, -3.0, 4.0
    ).astype(np.float32)
    standardized_query = np.clip(
        (query_geodesic[:, selected_geo] - median) / scale, -3.0, 4.0
    ).astype(np.float32)
    diagnostics = {
        "birth_components": 0,
        "stable_components": 0,
        "death_components": 0,
        "shifted_components": 0,
        "candidate_components": 0,
    }

    for site_value in np.unique(query_site):
        query_rows = np.flatnonzero(query_site == site_value)
        fit = fit_indices[train_site[fit_indices] == site_value]
        fit_tree = cKDTree(train_positions[fit, :2])
        query_tree = cKDTree(query_positions[query_rows, :2])
        boundary_k = min(16, len(fit))
        query_k = min(9, len(query_rows))
        boundary_distance, boundary_local = fit_tree.query(
            query_positions[query_rows, :2], k=boundary_k, workers=-1
        )
        query_distance, query_local = query_tree.query(
            query_positions[query_rows, :2], k=query_k, workers=-1
        )
        if boundary_k == 1:
            boundary_distance = boundary_distance[:, None]
            boundary_local = boundary_local[:, None]
        if query_k == 1:
            query_distance = query_distance[:, None]
            query_local = query_local[:, None]

        for local_index, row in enumerate(query_rows):
            boundary_ids = fit[boundary_local[local_index]].astype(np.int64)
            boundary_d = np.asarray(boundary_distance[local_index], dtype=np.float32)
            neighbor_rows = query_rows[query_local[local_index]].astype(np.int64)
            neighbor_d = np.asarray(query_distance[local_index], dtype=np.float32)
            nonself = neighbor_rows != row
            neighbor_rows = neighbor_rows[nonself]
            neighbor_d = neighbor_d[nonself]
            candidates: list[tuple[np.ndarray, float]] = []
            for slot in np.flatnonzero(current_valid[row]):
                candidates.append((current_coord[row, slot], 2.0 * current_energy[row, slot]))
            for slot in np.flatnonzero(template_valid[row])[:12]:
                candidates.append((template_coord[row, slot], 1.2 * template_energy[row, slot]))
            for boundary_id in boundary_ids[:4]:
                for slot in np.flatnonzero(truth_valid[boundary_id])[:6]:
                    candidates.append(
                        (truth_coord[boundary_id, slot], truth_energy[boundary_id, slot])
                    )
            candidates_unique = deduplicate_candidates(candidates)
            diagnostics["candidate_components"] += len(candidates_unique)

            plain_query_weight = np.exp(-neighbor_d / 7.0).astype(np.float64)
            plain_boundary_weight = np.exp(-boundary_d / 10.0).astype(np.float64)
            visible_query_weight = plain_query_weight.copy()
            visible_boundary_weight = plain_boundary_weight.copy()
            for index, neighbor_row in enumerate(neighbor_rows):
                visible_query_weight[index] *= _edge_multiplier(
                    int(query_global_ids[row]),
                    int(query_global_ids[neighbor_row]),
                    standardized_query[row],
                    standardized_query[neighbor_row],
                    query_occupancy[row],
                    query_occupancy[neighbor_row],
                    query_panorama[row],
                    query_panorama[neighbor_row],
                    int(query_regime[row]),
                    int(query_regime[neighbor_row]),
                )
            for index, boundary_id in enumerate(boundary_ids):
                visible_boundary_weight[index] *= _edge_multiplier(
                    int(query_global_ids[row]),
                    int(boundary_id),
                    standardized_query[row],
                    standardized_train[boundary_id],
                    query_occupancy[row],
                    train_occupancy[boundary_id],
                    query_panorama[row],
                    train_panorama[boundary_id],
                    int(query_regime[row]),
                    int(train_regime[boundary_id]),
                )

            def normalized_group(weight: np.ndarray, share: float) -> np.ndarray:
                if len(weight) == 0 or float(np.sum(weight)) <= 0.0:
                    return np.zeros_like(weight, dtype=np.float64)
                return share * weight / np.sum(weight)

            weight_sets = {
                "plain": (
                    normalized_group(plain_query_weight, 0.45),
                    normalized_group(plain_boundary_weight, 0.55),
                ),
                "visibility": (
                    normalized_group(visible_query_weight, 0.45),
                    normalized_group(visible_boundary_weight, 0.55),
                ),
                "shift": (
                    normalized_group(visible_query_weight, 0.45),
                    normalized_group(visible_boundary_weight, 0.55),
                ),
            }
            sparse_outputs: dict[str, np.ndarray] = {}
            coordinate_outputs: dict[str, list[tuple[np.ndarray, float]]] = {}
            for mode, (query_weight, boundary_weight) in weight_sets.items():
                component_rows: list[tuple[np.ndarray, float]] = []
                for candidate in candidates_unique:
                    base_found, _, base_energy = nearest_peak(
                        current_coord[row], current_energy[row], current_valid[row], candidate
                    )
                    template_found, _, prior_energy = nearest_peak(
                        template_coord[row], template_energy[row], template_valid[row], candidate
                    )
                    matched_coordinates: list[np.ndarray] = []
                    matched_weights: list[float] = []
                    weighted_energy = 0.0
                    matched_total = 0.0
                    query_presence = 0.0
                    boundary_presence = 0.0
                    for index, neighbor_row in enumerate(neighbor_rows):
                        found, coordinate, energy_value = nearest_peak(
                            current_coord[neighbor_row],
                            current_energy[neighbor_row],
                            current_valid[neighbor_row],
                            candidate,
                        )
                        if found:
                            weight = float(query_weight[index])
                            query_presence += weight / 0.45 if 0.45 > 0 else 0.0
                            weighted_energy += weight * energy_value
                            matched_total += weight
                            matched_coordinates.append(coordinate)
                            matched_weights.append(weight * math.sqrt(max(energy_value, 1e-8)))
                    for index, boundary_id in enumerate(boundary_ids):
                        found, coordinate, energy_value = nearest_peak(
                            truth_coord[boundary_id],
                            truth_energy[boundary_id],
                            truth_valid[boundary_id],
                            candidate,
                        )
                        if found:
                            weight = float(boundary_weight[index])
                            boundary_presence += weight / 0.55 if 0.55 > 0 else 0.0
                            weighted_energy += weight * energy_value
                            matched_total += weight
                            matched_coordinates.append(coordinate)
                            matched_weights.append(weight * math.sqrt(max(energy_value, 1e-8)))
                    consensus_energy = (
                        weighted_energy / matched_total if matched_total > 1e-12 else 0.0
                    )
                    template_presence = 1.0 if template_found and prior_energy > 2e-5 else 0.0
                    if base_found:
                        reliability = np.clip(
                            0.40 * query_presence
                            + 0.40 * boundary_presence
                            + 0.20 * template_presence,
                            0.0,
                            1.0,
                        )
                        source_energy = 0.62 * consensus_energy + 0.38 * prior_energy
                        mass = (1.0 - reliability) * base_energy + reliability * source_energy
                        if query_presence + boundary_presence < 0.20 and not template_found:
                            mass *= 0.70
                            diagnostics["death_components"] += 1
                        else:
                            diagnostics["stable_components"] += 1
                    else:
                        reliability = np.clip(
                            0.48 * boundary_presence
                            + 0.32 * query_presence
                            + 0.20 * template_presence,
                            0.0,
                            1.0,
                        )
                        mass = reliability * (
                            0.65 * consensus_energy + 0.35 * prior_energy
                        ) * 0.78
                        if mass > 2e-5:
                            diagnostics["birth_components"] += 1
                    coordinate = np.asarray(candidate, dtype=np.int16)
                    if mode == "shift" and matched_coordinates:
                        proposed = _weighted_coordinate(
                            matched_coordinates, matched_weights, coordinate
                        )
                        regime = int(query_regime[row])
                        maximum = ((0, 0, 0), (1, 0, 1), (1, 1, 2), (2, 1, 4))[regime]
                        moved = coordinate.copy()
                        for axis, size in enumerate((16, 8, 192)):
                            delta = circular_delta(int(proposed[axis]), int(coordinate[axis]), size)
                            delta = int(np.clip(delta, -maximum[axis], maximum[axis]))
                            moved[axis] = (int(coordinate[axis]) + delta) % size
                        if regime >= 2 and np.any(moved != coordinate) and boundary_presence > 0.20:
                            diagnostics["shifted_components"] += 1
                            coordinate = moved
                    if mass > 1e-7:
                        component_rows.append((coordinate, float(mass)))
                coordinate_outputs[mode] = component_rows
                grid = np.zeros((16, 8, 192), dtype=np.float32)
                for coordinate, mass in sorted(
                    component_rows, key=lambda item: item[1], reverse=True
                )[:24]:
                    _add_kernel(grid, coordinate, mass)
                if float(grid.sum()) <= 0.0:
                    grid = np.asarray(current_support[row], dtype=np.float32).copy()
                grid /= np.maximum(grid.sum(dtype=np.float64), 1e-20)
                sparse_outputs[mode] = grid
            plain[row] = normalize_support(
                (
                    0.45 * np.asarray(current_support[row : row + 1], dtype=np.float32)
                    + 0.40 * sparse_outputs["plain"][None]
                    + 0.15 * np.asarray(template_support[row : row + 1], dtype=np.float32)
                )
            )[0]
            visibility[row] = normalize_support(
                (
                    0.45 * np.asarray(current_support[row : row + 1], dtype=np.float32)
                    + 0.40 * sparse_outputs["visibility"][None]
                    + 0.15 * np.asarray(template_support[row : row + 1], dtype=np.float32)
                )
            )[0]
            shift_strength = float(
                np.clip(
                    (query_risk[row] - np.quantile(train_risk[fit], 0.50))
                    / max(
                        np.quantile(train_risk[fit], 0.90)
                        - np.quantile(train_risk[fit], 0.50),
                        1e-6,
                    ),
                    0.0,
                    1.0,
                )
            )
            shifted[row] = normalize_support(
                (
                    (0.55 - 0.10 * shift_strength)
                    * np.asarray(current_support[row : row + 1], dtype=np.float32)
                    + (0.30 + 0.10 * shift_strength)
                    * sparse_outputs["shift"][None]
                    + 0.15 * np.asarray(template_support[row : row + 1], dtype=np.float32)
                )
            )[0]
    return normalize_support(plain), normalize_support(visibility), normalize_support(shifted), diagnostics


def _save_support(path, value: np.ndarray) -> np.ndarray:
    output = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float16, shape=value.shape
    )
    output[:] = normalize_support(value).astype(np.float16)
    output.flush()
    return np.load(path, mmap_mode="r")


def build_support_targets(
    folds: list[SpatialHoleFold],
    oof_current: np.ndarray,
    test_current: np.ndarray,
) -> dict[str, object]:
    setup, train_positions, test_positions, train_site, test_site = load_positions()
    geometry = build_geometry_features(
        train_positions,
        test_positions,
        train_site,
        test_site,
        setup.bs_positions,
    )
    train_geodesic, test_geodesic, geodesic_names = build_geodesic_features()
    model_train_features = np.concatenate(
        [geometry.train, train_geodesic], axis=1
    ).astype(np.float32)
    model_test_features = np.concatenate(
        [geometry.test, test_geodesic], axis=1
    ).astype(np.float32)
    truth_support = normalize_support(np.load(TRAIN_AD, mmap_mode="r"))
    truth_descriptor = support_descriptor(truth_support)
    oof_current_value = normalize_support(oof_current)
    test_current_value = normalize_support(test_current)
    oof_descriptor = support_descriptor(oof_current_value)
    test_descriptor = support_descriptor(test_current_value)
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    global_to_oof = np.full(len(train_positions), -1, dtype=np.int64)
    global_to_oof[query_ids] = np.arange(len(query_ids), dtype=np.int64)

    oof_template = np.zeros_like(oof_current_value, dtype=np.float32)
    oof_plain = np.zeros_like(oof_current_value, dtype=np.float32)
    oof_visibility = np.zeros_like(oof_current_value, dtype=np.float32)
    oof_shift = np.zeros_like(oof_current_value, dtype=np.float32)
    oof_regime = np.zeros(len(query_ids), dtype=np.int8)
    oof_risk = np.zeros(len(query_ids), dtype=np.float32)
    template_records: list[dict[str, object]] = []
    lifecycle_records: list[dict[str, object]] = []
    train_route = _route_signatures("train")
    truth_peaks = extract_peaks(truth_support)

    for fold in folds:
        rows = global_to_oof[fold.query_indices]
        train_risk, query_risk, train_regime, query_regime, regime_report = (
            physical_risk_and_regime(
                train_geodesic,
                train_geodesic[fold.query_indices],
                train_site,
                train_site[fold.query_indices],
                fold.fit_indices,
                geodesic_names,
            )
        )
        template, confidence, records = build_regime_template(
            truth_support,
            truth_descriptor,
            oof_current_value[rows],
            oof_descriptor[rows],
            model_train_features,
            model_train_features[fold.query_indices],
            train_site,
            train_site[fold.query_indices],
            train_regime,
            query_regime,
            fold.fit_indices,
            20_260_806 + fold.fold * 101,
        )
        current_peaks = extract_peaks(oof_current_value[rows])
        template_peaks = extract_peaks(template)
        query_route = (
            train_route[0][fold.query_indices],
            train_route[1][fold.query_indices],
        )
        plain, visibility, shift, lifecycle_report = build_lifecycle_targets(
            oof_current_value[rows],
            template,
            train_positions[fold.query_indices],
            train_site[fold.query_indices],
            fold.query_indices,
            fold.fit_indices,
            train_positions,
            train_site,
            truth_peaks,
            current_peaks,
            template_peaks,
            train_regime,
            query_regime,
            train_risk,
            query_risk,
            train_geodesic,
            train_geodesic[fold.query_indices],
            train_route,
            query_route,
        )
        oof_template[rows] = template
        oof_plain[rows] = plain
        oof_visibility[rows] = visibility
        oof_shift[rows] = shift
        oof_regime[rows] = query_regime
        oof_risk[rows] = query_risk
        template_records.append(
            {
                "fold": int(fold.fold),
                "confidence_mean": float(np.mean(confidence)),
                "regime": regime_report,
                "groups": records,
            }
        )
        lifecycle_records.append({"fold": int(fold.fold), **lifecycle_report})
        print(f"support targets fold={fold.fold} complete", flush=True)

    full_fit = np.arange(len(train_positions), dtype=np.int64)
    train_risk, test_risk, train_regime, test_regime, test_regime_report = (
        physical_risk_and_regime(
            train_geodesic,
            test_geodesic,
            train_site,
            test_site,
            full_fit,
            geodesic_names,
        )
    )
    test_template, test_confidence, test_records = build_regime_template(
        truth_support,
        truth_descriptor,
        test_current_value,
        test_descriptor,
        model_train_features,
        model_test_features,
        train_site,
        test_site,
        train_regime,
        test_regime,
        full_fit,
        20_260_906,
    )
    test_peaks = extract_peaks(test_current_value)
    test_template_peaks = extract_peaks(test_template)
    test_route = _route_signatures("test")
    test_plain, test_visibility, test_shift, test_lifecycle_report = (
        build_lifecycle_targets(
            test_current_value,
            test_template,
            test_positions,
            test_site,
            np.arange(len(test_positions), dtype=np.int64),
            full_fit,
            train_positions,
            train_site,
            truth_peaks,
            test_peaks,
            test_template_peaks,
            train_regime,
            test_regime,
            train_risk,
            test_risk,
            train_geodesic,
            test_geodesic,
            train_route,
            test_route,
        )
    )

    _save_support(OOF_TEMPLATE, oof_template)
    _save_support(OOF_LIFECYCLE_PLAIN, oof_plain)
    _save_support(OOF_LIFECYCLE_VISIBILITY, oof_visibility)
    _save_support(OOF_LIFECYCLE_SHIFT, oof_shift)
    _save_support(TEST_TEMPLATE, test_template)
    _save_support(TEST_LIFECYCLE_PLAIN, test_plain)
    _save_support(TEST_LIFECYCLE_VISIBILITY, test_visibility)
    _save_support(TEST_LIFECYCLE_SHIFT, test_shift)
    np.savez_compressed(
        REGIME_CACHE,
        oof_regime=oof_regime,
        test_regime=test_regime,
        oof_risk=oof_risk,
        test_risk=test_risk,
        regime_names=np.asarray(REGIME_NAMES, dtype="U12"),
    )

    truth_query = truth_support[query_ids]
    fold_id = metadata["fold_id"].astype(np.int8)
    support_rows: dict[str, object] = {}
    for name, value in {
        "current": oof_current_value,
        "template": oof_template,
        "lifecycle_plain": oof_plain,
        "lifecycle_visibility": oof_visibility,
        "lifecycle_shift": oof_shift,
    }.items():
        cosine = support_cosine(value, truth_query)
        support_rows[name] = {
            "mean": float(np.mean(cosine)),
            "fold": [
                float(np.mean(cosine[fold_id == fold]))
                for fold in np.unique(fold_id)
            ],
            "regime": {
                REGIME_NAMES[regime]: float(np.mean(cosine[oof_regime == regime]))
                for regime in range(4)
                if np.any(oof_regime == regime)
            },
        }
    payload = {
        "experiment": VERSION,
        "status": "support_targets_complete",
        "official_inputs_only": True,
        "sionna_used": False,
        "support_cosine": support_rows,
        "template_records": template_records,
        "lifecycle_records": lifecycle_records,
        "test": {
            "regime": test_regime_report,
            "template_confidence_mean": float(np.mean(test_confidence)),
            "template_groups": test_records,
            "lifecycle": test_lifecycle_report,
        },
    }
    SUPPORT_REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _fixed_blend_support(
    current: torch.Tensor, target: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    desired = (1.0 - alpha[:, None, None, None]) * current + alpha[
        :, None, None, None
    ] * target
    return desired / desired.sum(dim=(1, 2, 3), keepdim=True).clamp_min(TINY)


def _marginal_targets(
    base: torch.Tensor, desired_ad: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    current_pas = pas_distribution(base)
    current_pdp = pdp_distribution(base)
    current_hv = current_pas.sum(dim=1).mean(dim=(3, 4))
    current_hv /= current_hv.sum(dim=(1, 2), keepdim=True).clamp_min(TINY)
    target_hv = desired_ad.sum(dim=3)
    target_hv /= target_hv.sum(dim=(1, 2), keepdim=True).clamp_min(TINY)
    ratio_hv = (target_hv + 1e-8) / (current_hv + 1e-8)
    ratio_hv = ratio_hv.clamp(0.20, 5.0)
    desired_pas = current_pas * ratio_hv[:, None, :, :, None, None]
    desired_pas /= desired_pas.sum(dim=(1, 2, 3), keepdim=True).clamp_min(TINY)

    current_delay = current_pdp.mean(dim=(1, 2))
    current_delay /= current_delay.sum(dim=1, keepdim=True).clamp_min(TINY)
    target_delay = desired_ad.sum(dim=(1, 2))
    target_delay /= target_delay.sum(dim=1, keepdim=True).clamp_min(TINY)
    ratio_delay = (target_delay + 1e-8) / (current_delay + 1e-8)
    ratio_delay = ratio_delay.clamp(0.20, 5.0)
    desired_pdp = current_pdp * ratio_delay[:, None, None, :]
    desired_pdp /= desired_pdp.sum(dim=3, keepdim=True).clamp_min(TINY)
    return desired_pas, desired_pdp


def apply_candidate(
    base: torch.Tensor,
    current_ad: torch.Tensor,
    target: torch.Tensor,
    regime: np.ndarray,
    site: np.ndarray,
    candidate: Candidate,
) -> torch.Tensor:
    alpha = torch.as_tensor(
        PROFILE_ALPHA[candidate.profile][regime], dtype=torch.float32, device=base.device
    )
    desired_ad = _fixed_blend_support(current_ad, target, alpha)
    reference_power = base.abs().square().sum(dim=(1, 2, 3))
    output = base
    if candidate.mode in ("ad", "joint"):
        output = preserve_point_power(
            exact_ad_projection(output, desired_ad), reference_power
        )
    if candidate.mode in ("metric", "joint"):
        desired_pas, desired_pdp = _marginal_targets(base, desired_ad)
        for _ in range(PROJECTION_CYCLES):
            for site_value in np.unique(site):
                rows = np.flatnonzero(site == site_value)
                local = output[rows]
                if int(site_value) == 0:
                    local = exact_pdp_projection(local, desired_pdp[rows])
                    local = exact_pas_projection(local, desired_pas[rows])
                else:
                    local = exact_pas_projection(local, desired_pas[rows])
                    local = exact_pdp_projection(local, desired_pdp[rows])
                output[rows] = preserve_point_power(local, reference_power[rows])
    return preserve_point_power(output, reference_power)


def point_metrics(
    prediction: torch.Tensor, truth: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shaped_prediction = prediction.reshape(len(prediction), 2, 16, 8, 4, 192)
    shaped_truth = truth.reshape(len(truth), 2, 16, 8, 4, 192)
    prediction_pas = torch.fft.fft2(shaped_prediction, dim=(2, 3)).abs().square().float()
    truth_pas = torch.fft.fft2(shaped_truth, dim=(2, 3)).abs().square().float()
    prediction_pdp = torch.fft.ifft(prediction, dim=-1).abs().square().float()
    truth_pdp = torch.fft.ifft(truth, dim=-1).abs().square().float()
    pas_num = (prediction_pas * truth_pas).sum(dim=(1, 2, 3), dtype=torch.float64)
    pas_den = torch.sqrt(
        prediction_pas.square().sum(dim=(1, 2, 3), dtype=torch.float64)
        * truth_pas.square().sum(dim=(1, 2, 3), dtype=torch.float64)
    )
    pas_den = torch.where(pas_den == 0.0, torch.full_like(pas_den, 1e-12), pas_den)
    pas = (pas_num / pas_den).mean(dim=(1, 2), dtype=torch.float64)
    pdp_num = (prediction_pdp * truth_pdp).sum(dim=3, dtype=torch.float64)
    pdp_den = torch.sqrt(
        prediction_pdp.square().sum(dim=3, dtype=torch.float64)
        * truth_pdp.square().sum(dim=3, dtype=torch.float64)
    )
    pdp_den = torch.where(pdp_den == 0.0, torch.full_like(pdp_den, 1e-12), pdp_den)
    pdp = (pdp_num / pdp_den).mean(dim=(1, 2), dtype=torch.float64)
    error = (prediction - truth).abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
    energy = truth.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
    return (
        pas.cpu().numpy(),
        pdp.cpu().numpy(),
        error.cpu().numpy(),
        energy.cpu().numpy(),
    )


def candidate_list() -> list[Candidate]:
    return [
        Candidate(target, profile, mode)
        for target in TARGET_PATHS
        for profile in PROFILE_ALPHA
        for mode in PROJECTION_MODES
    ]


def evaluate_projection_grid(
    oof_v4: np.ndarray,
    oof_current: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    if METRICS.is_file():
        cached = np.load(METRICS, allow_pickle=False)
        expected = ["identity"] + [candidate.name for candidate in candidate_list()]
        if cached["candidate_names"].astype(str).tolist() == expected:
            return {key: cached[key] for key in cached.files}
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    regimes = np.load(REGIME_CACHE)["oof_regime"].astype(np.int8)
    _, _, _, train_site, _ = load_positions()
    site = train_site[query_ids].astype(np.int8)
    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    target_arrays = {
        name: np.load(paths[0], mmap_mode="r") for name, paths in TARGET_PATHS.items()
    }
    candidates = candidate_list()
    names = ["identity"] + [candidate.name for candidate in candidates]
    result = {
        key: np.empty((len(names), len(query_ids)), dtype=np.float64)
        for key in ("pas", "pdp", "error", "energy")
    }
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        base = torch.as_tensor(
            np.array(oof_v4[start:end], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True),
            device=device,
        )
        current = torch.as_tensor(
            np.asarray(oof_current[start:end], dtype=np.float32), device=device
        )
        with torch.inference_mode():
            values = point_metrics(base, truth)
            for key, value in zip(("pas", "pdp", "error", "energy"), values):
                result[key][0, start:end] = value
            for candidate_index, candidate in enumerate(candidates, start=1):
                target = torch.as_tensor(
                    np.asarray(target_arrays[candidate.target][start:end], dtype=np.float32),
                    device=device,
                )
                prediction = apply_candidate(
                    base.clone(),
                    current,
                    target,
                    regimes[start:end],
                    site[start:end],
                    candidate,
                )
                values = point_metrics(prediction, truth)
                for key, value in zip(("pas", "pdp", "error", "energy"), values):
                    result[key][candidate_index, start:end] = value
        print(f"V6 projection grid {end}/{len(query_ids)}", flush=True)
    result["candidate_names"] = np.asarray(names, dtype="U72")
    result["query_ids"] = query_ids
    result["fold_id"] = fold_id
    result["site"] = site
    result["regime"] = regimes
    np.savez_compressed(METRICS, **result)
    return result


def aggregate_candidate(
    metrics: dict[str, np.ndarray], candidate_index: int, rows: np.ndarray
) -> dict[str, float | int]:
    return statistics(
        metrics["pas"][candidate_index, rows],
        metrics["pdp"][candidate_index, rows],
        metrics["error"][candidate_index, rows],
        metrics["energy"][candidate_index, rows],
    )


def selected_statistics(
    metrics: dict[str, np.ndarray], selected: np.ndarray
) -> dict[str, float | int]:
    point = np.arange(len(selected))
    return statistics(
        metrics["pas"][selected, point],
        metrics["pdp"][selected, point],
        metrics["error"][selected, point],
        metrics["energy"][selected, point],
    )


def stage_selection(
    metrics: dict[str, np.ndarray], stage: str
) -> dict[str, object]:
    names = metrics["candidate_names"].astype(str)
    fold_id = metrics["fold_id"].astype(np.int8)
    site = metrics["site"].astype(np.int8)
    regime = metrics["regime"].astype(np.int8)
    allowed = np.flatnonzero(
        (names == "identity") | np.char.startswith(names, stage + "_")
    )
    selected = np.zeros(len(fold_id), dtype=np.int32)
    decisions: list[dict[str, object]] = []
    for held_fold in np.unique(fold_id):
        for site_value in np.unique(site):
            for regime_value in range(4):
                held = np.flatnonzero(
                    (fold_id == held_fold)
                    & (site == site_value)
                    & (regime == regime_value)
                )
                if len(held) == 0:
                    continue
                train = np.flatnonzero(
                    (fold_id != held_fold)
                    & (site == site_value)
                    & (regime == regime_value)
                )
                if len(train) < 8:
                    choice = 0
                else:
                    choice = max(
                        allowed,
                        key=lambda candidate_index: float(
                            aggregate_candidate(metrics, int(candidate_index), train)["score"]
                        ),
                    )
                selected[held] = int(choice)
                decisions.append(
                    {
                        "held_fold": int(held_fold),
                        "site": int(site_value),
                        "regime": REGIME_NAMES[regime_value],
                        "candidate": str(names[choice]),
                        "train_count": int(len(train)),
                        "held_count": int(len(held)),
                    }
                )
    overall = selected_statistics(metrics, selected)
    baseline = selected_statistics(metrics, np.zeros(len(selected), dtype=np.int32))
    folds = []
    for fold in np.unique(fold_id):
        rows = np.flatnonzero(fold_id == fold)
        value = selected_statistics(
            {key: metrics[key][:, rows] if key in ("pas", "pdp", "error", "energy") else metrics[key] for key in metrics},
            selected[rows],
        )
        base = aggregate_candidate(metrics, 0, rows)
        folds.append(
            {
                "fold": int(fold),
                **value,
                "delta_score": float(value["score"] - base["score"]),
            }
        )

    policy: list[dict[str, object]] = []
    for site_value in np.unique(site):
        for regime_value in range(4):
            rows = np.flatnonzero((site == site_value) & (regime == regime_value))
            if len(rows) == 0:
                continue
            eligible: list[int] = [0]
            for candidate_index in allowed:
                if int(candidate_index) == 0:
                    continue
                deltas = []
                for fold in np.unique(fold_id):
                    fold_rows = rows[fold_id[rows] == fold]
                    if len(fold_rows) == 0:
                        continue
                    candidate_score = float(
                        aggregate_candidate(metrics, int(candidate_index), fold_rows)["score"]
                    )
                    base_score = float(aggregate_candidate(metrics, 0, fold_rows)["score"])
                    deltas.append(candidate_score - base_score)
                if deltas and np.sum(np.asarray(deltas) > 0.0) >= min(3, len(deltas)):
                    eligible.append(int(candidate_index))
            choice = max(
                eligible,
                key=lambda candidate_index: float(
                    aggregate_candidate(metrics, candidate_index, rows)["score"]
                ),
            )
            policy.append(
                {
                    "site": int(site_value),
                    "regime": REGIME_NAMES[regime_value],
                    "regime_index": int(regime_value),
                    "candidate": str(names[choice]),
                    "candidate_index": int(choice),
                    "count": int(len(rows)),
                    "oof_score": float(aggregate_candidate(metrics, choice, rows)["score"]),
                }
            )
    policy_selected = np.zeros(len(fold_id), dtype=np.int32)
    for row in policy:
        mask = (site == int(row["site"])) & (regime == int(row["regime_index"]))
        policy_selected[mask] = int(row["candidate_index"])
    return {
        "stage": stage,
        "baseline": baseline,
        "nested": overall,
        "nested_folds": folds,
        "nested_decisions": decisions,
        "positive_folds": int(sum(float(row["delta_score"]) > 0.0 for row in folds)),
        "final_policy": policy,
        "final_policy_oof": selected_statistics(metrics, policy_selected),
    }


def select_stage(metrics: dict[str, np.ndarray]) -> dict[str, object]:
    stages = [stage_selection(metrics, stage) for stage in TARGET_PATHS]
    passed = [
        row
        for row in stages
        if row["positive_folds"] >= 4
        and float(row["nested"]["score"]) > float(row["baseline"]["score"])
    ]
    winner = max(passed, key=lambda row: float(row["nested"]["score"])) if passed else None
    payload = {
        "experiment": VERSION,
        "baseline": stages[0]["baseline"],
        "stages": stages,
        "winner": winner,
    }
    SELECTION.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _candidate_from_name(name: str) -> Candidate:
    for candidate in candidate_list():
        if candidate.name == name:
            return candidate
    raise KeyError(name)


def generate_test_output(
    selection: dict[str, object],
    test_current: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, object] | None:
    winner = selection["winner"]
    if winner is None:
        return None
    base = np.load(V4_TEST, mmap_mode="r")
    regimes = np.load(REGIME_CACHE)["test_regime"].astype(np.int8)
    _, _, _, _, test_site = load_positions()
    target_name = str(winner["stage"])
    target = np.load(TARGET_PATHS[target_name][1], mmap_mode="r")
    policy = {
        (int(row["site"]), int(row["regime_index"])): str(row["candidate"])
        for row in winner["final_policy"]
    }
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        block = np.array(base[start:end], dtype=np.complex64, copy=True)
        block_site = test_site[start:end]
        block_regime = regimes[start:end]
        for site_value in np.unique(block_site):
            for regime_value in np.unique(block_regime):
                rows = np.flatnonzero(
                    (block_site == site_value) & (block_regime == regime_value)
                )
                if len(rows) == 0:
                    continue
                name = policy[(int(site_value), int(regime_value))]
                if name == "identity":
                    continue
                candidate = _candidate_from_name(name)
                base_tensor = torch.as_tensor(block[rows], device=device)
                current_tensor = torch.as_tensor(
                    np.asarray(test_current[start:end][rows], dtype=np.float32),
                    device=device,
                )
                target_tensor = torch.as_tensor(
                    np.asarray(target[start:end][rows], dtype=np.float32), device=device
                )
                with torch.inference_mode():
                    prediction = apply_candidate(
                        base_tensor,
                        current_tensor,
                        target_tensor,
                        block_regime[rows],
                        block_site[rows],
                        candidate,
                    )
                block[rows] = prediction.cpu().numpy().astype(np.complex64)
        output[start:end] = block
        output.flush()
        print(f"V6 test output {end}/{len(base)}", flush=True)
    finite = True
    for start in range(0, len(output), 8):
        block = np.asarray(output[start : start + 8])
        finite = finite and bool(
            np.isfinite(block.real).all() and np.isfinite(block.imag).all()
        )
    return {
        "path": PATHS.relative(OUTPUT),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "bytes": int(OUTPUT.stat().st_size),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("all", "prepare", "evaluate", "test"), default="all"
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    VERSION_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    setup, train_positions, test_positions, train_site, test_site = load_positions()
    folds = build_spatial_hole_folds(
        train_positions, test_positions, train_site, test_site
    )
    oof_v4 = replay_oof_v4(device, args.batch_size)
    oof_current = channel_ad_support(
        oof_v4, OOF_CURRENT_AD, device, args.batch_size
    )
    test_v4 = np.load(V4_TEST, mmap_mode="r")
    test_current = channel_ad_support(
        test_v4, TEST_CURRENT_AD, device, args.batch_size
    )

    support_payload = None
    if args.phase in ("all", "prepare"):
        support_payload = build_support_targets(folds, oof_current, test_current)
        if args.phase == "prepare":
            print(json.dumps(support_payload, ensure_ascii=False, indent=2), flush=True)
            return
    metrics = None
    selection = None
    if args.phase in ("all", "evaluate"):
        metrics = evaluate_projection_grid(
            oof_v4, oof_current, device, args.batch_size
        )
        selection = select_stage(metrics)
        if args.phase == "evaluate":
            print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)
            return
    if selection is None:
        selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    output = generate_test_output(
        selection, test_current, device, args.batch_size
    )
    status = "candidate_ready" if output is not None else "analysis_only"
    report = {
        "experiment": VERSION,
        "status": status,
        "primary_hypothesis": (
            "Explicit propagation regimes, sparse component templates, query-query "
            "birth/stable/death evidence and point-cloud geodesic visibility can improve "
            "V4 without using Sionna or replacing its complex carrier."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "validation": (
            "five complete dual-site spatial holes; held labels are excluded from every "
            "prototype, boundary component, regime threshold and policy decision"
        ),
        "mechanism_chain": [
            "V4 carrier/array-aligned complex backbone",
            "clear/light/medium/heavy physical-risk partition",
            "regime-conditioned sparse support templates",
            "query-query component birth/stable/death consensus",
            "official-map visibility/geodesic edge control and bounded support motion",
            "official PAS/PDP mouth-aligned exact projection",
        ],
        "support_ablation": support_payload
        if support_payload is not None
        else json.loads(SUPPORT_REPORT.read_text(encoding="utf-8")),
        "selection": selection,
        "output": output,
        "outputs": {
            "geodesic_features": PATHS.relative(GEODESIC_CACHE),
            "regimes": PATHS.relative(REGIME_CACHE),
            "support_report": PATHS.relative(SUPPORT_REPORT),
            "metrics": PATHS.relative(METRICS),
            "selection": PATHS.relative(SELECTION),
            "test_channel": PATHS.relative(OUTPUT) if output is not None else None,
            "report": PATHS.relative(REPORT),
        },
        "seconds": float(time.time() - started),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    REPORT.write_text(text, encoding="utf-8")
    VERSION_REPORT.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "winner": selection["winner"],
                "output": output,
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
