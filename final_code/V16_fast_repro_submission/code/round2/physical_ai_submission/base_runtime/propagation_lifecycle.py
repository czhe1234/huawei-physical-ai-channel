"""Frozen runtime primitives for propagation lifecycle."""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
import torch
from .exact_projection import (
    exact_ad_projection,
    exact_pas_projection,
    exact_pdp_projection,
    pas_distribution,
    pdp_distribution,
    preserve_point_power,
)
from project_paths import PATHS

ARTIFACT_DIR = PATHS.artifacts / "round2" / "regime_lifecycle_support"
OOF_TEMPLATE = ARTIFACT_DIR / "oof_regime_template_f16.npy"
TEST_TEMPLATE = ARTIFACT_DIR / "test_regime_template_f16.npy"
OOF_LIFECYCLE_PLAIN = ARTIFACT_DIR / "oof_lifecycle_plain_f16.npy"
OOF_LIFECYCLE_VISIBILITY = ARTIFACT_DIR / "oof_lifecycle_visibility_f16.npy"
OOF_LIFECYCLE_SHIFT = ARTIFACT_DIR / "oof_lifecycle_shift_f16.npy"
TEST_LIFECYCLE_PLAIN = ARTIFACT_DIR / "test_lifecycle_plain_f16.npy"
TEST_LIFECYCLE_VISIBILITY = ARTIFACT_DIR / "test_lifecycle_visibility_f16.npy"
TEST_LIFECYCLE_SHIFT = ARTIFACT_DIR / "test_lifecycle_shift_f16.npy"
REGIME_NAMES = ("clear", "light", "medium", "heavy")
TARGET_PATHS = {
    "template": (OOF_TEMPLATE, TEST_TEMPLATE),
    "lifecycle_plain": (OOF_LIFECYCLE_PLAIN, TEST_LIFECYCLE_PLAIN),
    "lifecycle_visibility": (OOF_LIFECYCLE_VISIBILITY, TEST_LIFECYCLE_VISIBILITY),
    "lifecycle_shift": (OOF_LIFECYCLE_SHIFT, TEST_LIFECYCLE_SHIFT),
}
PROFILE_ALPHA = {
    "soft": np.asarray((0.04, 0.08, 0.16, 0.24), dtype=np.float32),
    "balanced": np.asarray((0.06, 0.15, 0.3, 0.42), dtype=np.float32),
    "focused": np.asarray((0.0, 0.08, 0.32, 0.55), dtype=np.float32),
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
        array, np.maximum(total, 1e-20), out=np.zeros_like(array), where=total > 0.0
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
            transformed = torch.fft.ifft(torch.fft.fft2(shaped, dim=(2, 3)), dim=-1)
            power = transformed.abs().square().sum(dim=(1, 4)).float()
            power /= power.sum(dim=(1, 2, 3), keepdim=True).clamp_min(TINY)
        output[start:end] = power.cpu().numpy().astype(np.float16)
        output.flush()
    return np.load(output_path, mmap_mode="r")


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
    weights = np.asarray((0.25, 0.23, 0.2, 0.14, 0.1, -0.08), dtype=np.float32)
    return (columns, weights)


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
        scale = np.maximum(q75 - q25, 1e-06)
        train_standard = np.clip(
            (train_features[train_rows][:, columns] - median) / scale, -2.5, 3.5
        )
        query_standard = np.clip(
            (query_features[query_rows][:, columns] - median) / scale, -2.5, 3.5
        )
        train_risk[train_rows] = train_standard @ weights
        query_risk[query_rows] = query_standard @ weights
        thresholds = np.quantile(train_risk[fit], (0.25, 0.5, 0.75))
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
    return (train_risk, query_risk, train_regime, query_regime, diagnostics)


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
            rows = np.flatnonzero((query_site == site_value) & (query_regime == regime))
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
            prototypes = np.empty((cluster_count, 16, 8, 192), dtype=np.float32)
            for cluster in range(cluster_count):
                members = fit[labels == cluster]
                root = np.sqrt(
                    np.maximum(
                        np.asarray(truth_support[members], dtype=np.float32), 0.0
                    )
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
            support_probability = np.exp(np.clip(logits, -50.0, 0.0)).astype(np.float32)
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
    return (normalize_support(output), confidence, records)


def extract_peaks(
    support: np.ndarray, peak_count: int = 20, candidate_count: int = 128
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
            if candidate_energy <= 1e-07:
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
                if dh <= 1 and dv <= 1 and (dd <= 3):
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
    return (coordinates, energy, valid)


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
        return (False, candidate, 0.0)
    cost = spectral_cost(coordinates[rows], candidate)
    best = int(np.argmin(cost))
    if float(cost[best]) > 1.0:
        return (False, candidate, 0.0)
    slot = rows[best]
    return (True, coordinates[slot], float(energy[slot]))


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
        angle = (
            2.0
            * np.pi
            * np.asarray(
                [coordinate[axis] for coordinate in coordinates], dtype=np.float64
            )
            / size
        )
        mean = np.sum(weight * np.exp(1j * angle))
        if abs(mean) < 1e-08:
            result.append(int(reference[axis]))
        else:
            result.append(
                int(round(np.angle(mean) % (2.0 * np.pi) * size / (2.0 * np.pi))) % size
            )
    return np.asarray(result, dtype=np.int16)


def _add_kernel(grid: np.ndarray, coordinate: np.ndarray, mass: float) -> None:
    entries: list[tuple[int, int, int, float]] = []
    for dh in range(-1, 2):
        for dv in range(-1, 2):
            for dd in range(-3, 4):
                weight = math.exp(
                    -0.5 * ((dh / 0.75) ** 2 + (dv / 0.65) ** 2 + (dd / 1.35) ** 2)
                )
                entries.append(
                    (
                        (int(coordinate[0]) + dh) % 16,
                        (int(coordinate[1]) + dv) % 8,
                        (int(coordinate[2]) + dd) % 192,
                        weight,
                    )
                )
    total = sum((entry[3] for entry in entries))
    for horizontal, vertical, delay, weight in entries:
        grid[horizontal, vertical, delay] += np.float32(mass * weight / total)


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
    geo_factor = math.exp(-0.2 * float(np.mean(geo_delta * geo_delta)))
    occupancy_delta = float(np.mean(np.abs(query_occupancy - neighbor_occupancy)))
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
    selected_geo = np.asarray(
        (2, 3, 4, 5, 10, 11, 12, 13, 16, 17, 18, 25), dtype=np.int64
    )
    reference = train_geodesic[fit_indices][:, selected_geo]
    median = np.median(reference, axis=0)
    q25, q75 = np.quantile(reference, (0.25, 0.75), axis=0)
    scale = np.maximum(q75 - q25, 1e-06)
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
                candidates.append(
                    (current_coord[row, slot], 2.0 * current_energy[row, slot])
                )
            for slot in np.flatnonzero(template_valid[row])[:12]:
                candidates.append(
                    (template_coord[row, slot], 1.2 * template_energy[row, slot])
                )
            for boundary_id in boundary_ids[:4]:
                for slot in np.flatnonzero(truth_valid[boundary_id])[:6]:
                    candidates.append(
                        (
                            truth_coord[boundary_id, slot],
                            truth_energy[boundary_id, slot],
                        )
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
                        current_coord[row],
                        current_energy[row],
                        current_valid[row],
                        candidate,
                    )
                    template_found, _, prior_energy = nearest_peak(
                        template_coord[row],
                        template_energy[row],
                        template_valid[row],
                        candidate,
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
                            matched_weights.append(
                                weight * math.sqrt(max(energy_value, 1e-08))
                            )
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
                            matched_weights.append(
                                weight * math.sqrt(max(energy_value, 1e-08))
                            )
                    consensus_energy = (
                        weighted_energy / matched_total
                        if matched_total > 1e-12
                        else 0.0
                    )
                    template_presence = (
                        1.0 if template_found and prior_energy > 2e-05 else 0.0
                    )
                    if base_found:
                        reliability = np.clip(
                            0.4 * query_presence
                            + 0.4 * boundary_presence
                            + 0.2 * template_presence,
                            0.0,
                            1.0,
                        )
                        source_energy = 0.62 * consensus_energy + 0.38 * prior_energy
                        mass = (
                            1.0 - reliability
                        ) * base_energy + reliability * source_energy
                        if query_presence + boundary_presence < 0.2 and (
                            not template_found
                        ):
                            mass *= 0.7
                            diagnostics["death_components"] += 1
                        else:
                            diagnostics["stable_components"] += 1
                    else:
                        reliability = np.clip(
                            0.48 * boundary_presence
                            + 0.32 * query_presence
                            + 0.2 * template_presence,
                            0.0,
                            1.0,
                        )
                        mass = (
                            reliability
                            * (0.65 * consensus_energy + 0.35 * prior_energy)
                            * 0.78
                        )
                        if mass > 2e-05:
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
                            delta = circular_delta(
                                int(proposed[axis]), int(coordinate[axis]), size
                            )
                            delta = int(np.clip(delta, -maximum[axis], maximum[axis]))
                            moved[axis] = (int(coordinate[axis]) + delta) % size
                        if (
                            regime >= 2
                            and np.any(moved != coordinate)
                            and (boundary_presence > 0.2)
                        ):
                            diagnostics["shifted_components"] += 1
                            coordinate = moved
                    if mass > 1e-07:
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
                0.45 * np.asarray(current_support[row : row + 1], dtype=np.float32)
                + 0.4 * sparse_outputs["plain"][None]
                + 0.15 * np.asarray(template_support[row : row + 1], dtype=np.float32)
            )[0]
            visibility[row] = normalize_support(
                0.45 * np.asarray(current_support[row : row + 1], dtype=np.float32)
                + 0.4 * sparse_outputs["visibility"][None]
                + 0.15 * np.asarray(template_support[row : row + 1], dtype=np.float32)
            )[0]
            shift_strength = float(
                np.clip(
                    (query_risk[row] - np.quantile(train_risk[fit], 0.5))
                    / max(
                        np.quantile(train_risk[fit], 0.9)
                        - np.quantile(train_risk[fit], 0.5),
                        1e-06,
                    ),
                    0.0,
                    1.0,
                )
            )
            shifted[row] = normalize_support(
                (0.55 - 0.1 * shift_strength)
                * np.asarray(current_support[row : row + 1], dtype=np.float32)
                + (0.3 + 0.1 * shift_strength) * sparse_outputs["shift"][None]
                + 0.15 * np.asarray(template_support[row : row + 1], dtype=np.float32)
            )[0]
    return (
        normalize_support(plain),
        normalize_support(visibility),
        normalize_support(shifted),
        diagnostics,
    )


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
    ratio_hv = (target_hv + 1e-08) / (current_hv + 1e-08)
    ratio_hv = ratio_hv.clamp(0.2, 5.0)
    desired_pas = current_pas * ratio_hv[:, None, :, :, None, None]
    desired_pas /= desired_pas.sum(dim=(1, 2, 3), keepdim=True).clamp_min(TINY)
    current_delay = current_pdp.mean(dim=(1, 2))
    current_delay /= current_delay.sum(dim=1, keepdim=True).clamp_min(TINY)
    target_delay = desired_ad.sum(dim=(1, 2))
    target_delay /= target_delay.sum(dim=1, keepdim=True).clamp_min(TINY)
    ratio_delay = (target_delay + 1e-08) / (current_delay + 1e-08)
    ratio_delay = ratio_delay.clamp(0.2, 5.0)
    desired_pdp = current_pdp * ratio_delay[:, None, None, :]
    desired_pdp /= desired_pdp.sum(dim=3, keepdim=True).clamp_min(TINY)
    return (desired_pas, desired_pdp)


def apply_candidate(
    base: torch.Tensor,
    current_ad: torch.Tensor,
    target: torch.Tensor,
    regime: np.ndarray,
    site: np.ndarray,
    candidate: Candidate,
) -> torch.Tensor:
    alpha = torch.as_tensor(
        PROFILE_ALPHA[candidate.profile][regime],
        dtype=torch.float32,
        device=base.device,
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


def candidate_list() -> list[Candidate]:
    return [
        Candidate(target, profile, mode)
        for target in TARGET_PATHS
        for profile in PROFILE_ALPHA
        for mode in PROJECTION_MODES
    ]


def _candidate_from_name(name: str) -> Candidate:
    for candidate in candidate_list():
        if candidate.name == name:
            return candidate
    raise KeyError(name)
