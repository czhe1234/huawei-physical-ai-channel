#!/usr/bin/env python3
"""Audit whether multi-configuration geometry paths support transferable path events."""

from __future__ import annotations

import json

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

import analyze_v305_pointcloud_path_identity_match as v305
import analyze_v312_highdepth_path_identity_match as v312
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 464
VERSION = "v464_multiconfig_path_event_feasibility"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
DATA_CACHE = OUT_DIR / "v464_path_event_dataset.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"

PATH_A = PATHS.cache_dir(291) / "v291_highdepth_pointcloud_paths.npz"
PATH_B = PATHS.cache_dir(321) / "v321_highdepth_diffraction_paths.npz"
GEODESIC = PATHS.shared_features / "geodesic_support_features.npz"
GEOMETRY = PATHS.shared_features / "height_geometry_cache.npz"

PATH_LIMIT = 64
MIN_DEPTH = 2
NEGATIVE_COST_MIN = 16.0
STABILITY_LENGTH_SCALE = 0.25
STABILITY_H_SCALE = 0.50
CALIBRATION_FRACTION = 0.20
TARGET_CALIBRATION_PRECISION = 0.80
MAX_NEGATIVE_RATIO = 30


def require_file(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def circular_distance(left: np.ndarray, right: np.ndarray, size: float) -> np.ndarray:
    delta = np.abs(np.asarray(left) - np.asarray(right))
    return np.minimum(delta, size - delta)


def candidate_h(path_data: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    vertices = path_data[f"{prefix}_vertices"].astype(np.float32)
    first = vertices[:, :, 0].copy()
    direct = path_data[f"{prefix}_interaction_count"] == 0
    positions = path_data[f"{prefix}_pos"].astype(np.float32)
    first[direct] = np.broadcast_to(positions[:, None, :], first.shape)[direct]
    return v305.expected_h_bin(first)


def build_positive_labels() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    match = np.load(v312.MATCH_CACHE)
    good = (
        (match["length_error_bins"] <= v312.GOOD_LENGTH_ERROR)
        & (match["h_error_bins"] <= v312.GOOD_H_ERROR)
        & (match["candidate_id"] < PATH_LIMIT)
    )
    positive = np.zeros((2000, PATH_LIMIT), dtype=bool)
    mass = np.zeros((2000, PATH_LIMIT), dtype=np.float32)
    confidence = np.zeros((2000, PATH_LIMIT), dtype=np.float32)
    point = match["point_id"][good].astype(np.int64)
    candidate = match["candidate_id"][good].astype(np.int64)
    current_mass = match["mass"][good].astype(np.float32)
    current_confidence = np.exp(-0.5 * match["cost"][good]).astype(np.float32)
    for index in range(len(point)):
        p = point[index]
        c = candidate[index]
        positive[p, c] = True
        if current_mass[index] >= mass[p, c]:
            mass[p, c] = current_mass[index]
            confidence[p, c] = current_confidence[index]
    return positive, mass, confidence


def clean_label_mask(
    observations: dict[str, np.ndarray],
    paths: dict[str, np.ndarray],
    h_center: np.ndarray,
    positive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = (
        paths["train_valid"][:, :PATH_LIMIT].astype(bool)
        & (paths["train_interaction_count"][:, :PATH_LIMIT] >= MIN_DEPTH)
    )
    one_bounce = v312.one_bounce_good_mask(
        len(observations["delay"]), observations["delay"].shape[1]
    )
    clean = np.zeros_like(valid)
    minimum_cost = np.full(valid.shape, np.inf, dtype=np.float32)
    for point_id in range(len(valid)):
        active = np.flatnonzero(
            observations["valid"][point_id]
            & (observations["mass"][point_id] >= v305.MIN_COMPONENT_MASS)
            & (~one_bounce[point_id])
        )
        candidates = np.flatnonzero(valid[point_id])
        if len(active) == 0 or len(candidates) == 0:
            continue
        length_error = np.abs(
            observations["path_length"][point_id, active, None]
            - paths["train_path_length"][point_id, candidates][None]
        ) / v312.v304.METERS_PER_BIN
        h_error = circular_distance(
            observations["h_center"][point_id, active, None],
            h_center[point_id, candidates][None],
            16.0,
        )
        cost = (length_error / v312.LENGTH_SCALE_BINS) ** 2 + (
            h_error / v312.H_SCALE_BINS
        ) ** 2
        minimum_cost[point_id, candidates] = np.min(cost, axis=0).astype(np.float32)
        clean[point_id, candidates] = minimum_cost[point_id, candidates] >= NEGATIVE_COST_MIN
    clean |= positive
    clean &= valid
    return clean, minimum_cost


def multiconfig_stability(
    path_a: dict[str, np.ndarray],
    path_b: dict[str, np.ndarray],
    h_a: np.ndarray,
    h_b: np.ndarray,
) -> dict[str, np.ndarray]:
    score = np.zeros((2000, PATH_LIMIT), dtype=np.float32)
    length_error = np.full((2000, PATH_LIMIT), 99.0, dtype=np.float32)
    h_error = np.full((2000, PATH_LIMIT), 8.0, dtype=np.float32)
    depth_error = np.full((2000, PATH_LIMIT), 16.0, dtype=np.float32)
    mask_agreement = np.zeros((2000, PATH_LIMIT), dtype=np.float32)
    for point_id in range(2000):
        source = np.flatnonzero(
            path_a["train_valid"][point_id, :PATH_LIMIT]
            & (path_a["train_interaction_count"][point_id, :PATH_LIMIT] >= MIN_DEPTH)
        )
        reference = np.flatnonzero(
            path_b["train_valid"][point_id]
            & (path_b["train_interaction_count"][point_id] >= MIN_DEPTH)
        )
        if len(source) == 0 or len(reference) == 0:
            continue
        current_length = np.abs(
            path_a["train_path_length"][point_id, source, None]
            - path_b["train_path_length"][point_id, reference][None]
        ) / v312.v304.METERS_PER_BIN
        current_h = circular_distance(
            h_a[point_id, source, None], h_b[point_id, reference][None], 16.0
        )
        current_depth = np.abs(
            path_a["train_interaction_count"][point_id, source, None].astype(np.float32)
            - path_b["train_interaction_count"][point_id, reference][None].astype(np.float32)
        )
        current_mask = (
            path_a["train_interaction_mask"][point_id, source, None]
            == path_b["train_interaction_mask"][point_id, reference][None]
        ).astype(np.float32)
        cost = (
            (current_length / STABILITY_LENGTH_SCALE) ** 2
            + (current_h / STABILITY_H_SCALE) ** 2
            + 0.20 * current_depth**2
            + 0.35 * (1.0 - current_mask)
        )
        nearest = np.argmin(cost, axis=1)
        rows = np.arange(len(source))
        length_error[point_id, source] = current_length[rows, nearest]
        h_error[point_id, source] = current_h[rows, nearest]
        depth_error[point_id, source] = current_depth[rows, nearest]
        mask_agreement[point_id, source] = current_mask[rows, nearest]
        score[point_id, source] = np.exp(
            -0.5 * np.minimum(cost[rows, nearest], 30.0)
        ).astype(np.float32)
    return {
        "stability_score": score,
        "stability_length_error": length_error,
        "stability_h_error": h_error,
        "stability_depth_error": depth_error,
        "stability_mask_agreement": mask_agreement,
    }


def path_features(
    paths: dict[str, np.ndarray],
    stability: dict[str, np.ndarray],
    clean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    point_id, candidate_id = np.nonzero(clean)
    positions = paths["train_pos"].astype(np.float32)
    bs = paths["bs_pos"].astype(np.float32)
    geometry = np.load(GEOMETRY)
    blocked = geometry["train_blocked"].astype(np.float32)
    geodesic = np.load(GEODESIC)["train_features"].astype(np.float32)
    depth = paths["train_interaction_count"][point_id, candidate_id].astype(np.int64)
    interactions = paths["train_interactions"][point_id, candidate_id].astype(np.int64)
    vertices = np.nan_to_num(
        paths["train_vertices"][point_id, candidate_id].astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    first = vertices[:, 0]
    last = vertices[np.arange(len(vertices)), np.maximum(depth - 1, 0)]
    first_vector = first - bs[None]
    last_vector = positions[point_id] - last
    first_unit = first_vector / np.maximum(
        np.linalg.norm(first_vector, axis=1, keepdims=True), 1e-6
    )
    last_unit = last_vector / np.maximum(
        np.linalg.norm(last_vector, axis=1, keepdims=True), 1e-6
    )
    relative = positions[point_id] - bs[None]
    direct = np.linalg.norm(relative, axis=1)
    length = paths["train_path_length"][point_id, candidate_id].astype(np.float32)
    score = np.nan_to_num(
        paths["train_score"][point_id, candidate_id].astype(np.float32),
        nan=-20.0,
        posinf=8.0,
        neginf=-20.0,
    )
    theta_t = paths["train_theta_t"][point_id, candidate_id].astype(np.float32)
    phi_t = paths["train_phi_t"][point_id, candidate_id].astype(np.float32)
    theta_r = paths["train_theta_r"][point_id, candidate_id].astype(np.float32)
    phi_r = paths["train_phi_r"][point_id, candidate_id].astype(np.float32)
    interaction_count = np.stack(
        [np.sum(interactions == value, axis=1) for value in (1, 4, 8)], axis=1
    ).astype(np.float32)
    first_interactions = interactions[:, :6].astype(np.float32) / 8.0
    point_numeric = np.concatenate(
        [
            positions[point_id, :2] / 500.0,
            relative[:, :2] / 500.0,
            (direct / 500.0)[:, None],
            (blocked[point_id] / 250.0)[:, None],
            geodesic[point_id],
        ],
        axis=1,
    )
    path_numeric = np.concatenate(
        [
            (length / 800.0)[:, None],
            ((length - direct) / 500.0)[:, None],
            np.clip(score, -20.0, 8.0)[:, None] / 8.0,
            (depth.astype(np.float32) / 16.0)[:, None],
            interaction_count / 16.0,
            np.stack(
                [
                    np.sin(theta_t),
                    np.cos(theta_t),
                    np.sin(phi_t),
                    np.cos(phi_t),
                    np.sin(theta_r),
                    np.cos(theta_r),
                    np.sin(phi_r),
                    np.cos(phi_r),
                ],
                axis=1,
            ),
            first_vector / 500.0,
            last_vector / 500.0,
            first_unit,
            last_unit,
            first_interactions,
        ],
        axis=1,
    ).astype(np.float32)
    base = np.concatenate([point_numeric, path_numeric], axis=1).astype(np.float32)
    stability_numeric = np.stack(
        [
            stability["stability_score"][point_id, candidate_id],
            np.clip(
                stability["stability_length_error"][point_id, candidate_id], 0.0, 8.0
            )
            / 8.0,
            stability["stability_h_error"][point_id, candidate_id] / 8.0,
            stability["stability_depth_error"][point_id, candidate_id] / 16.0,
            stability["stability_mask_agreement"][point_id, candidate_id],
        ],
        axis=1,
    ).astype(np.float32)
    feature = np.concatenate([base, stability_numeric], axis=1)
    feature_names = [f"base_{index}" for index in range(base.shape[1])] + [
        "stability_score",
        "stability_length_error",
        "stability_h_error",
        "stability_depth_error",
        "stability_mask_agreement",
    ]
    return point_id.astype(np.int32), candidate_id.astype(np.int16), feature, feature_names, base.shape[1]


def point_fold_labels(
    paths: dict[str, np.ndarray], stability: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    positions = paths["train_pos"].astype(np.float32)
    geometry = np.load(GEOMETRY)
    blocked = geometry["train_blocked"].astype(np.float32)
    geodesic = np.load(GEODESIC)["train_features"].astype(np.float32)
    valid = (
        paths["train_valid"][:, :PATH_LIMIT].astype(bool)
        & (paths["train_interaction_count"][:, :PATH_LIMIT] >= MIN_DEPTH)
    )
    count = valid.sum(axis=1).astype(np.float32)
    mean_depth = np.sum(
        paths["train_interaction_count"][:, :PATH_LIMIT] * valid, axis=1
    ) / np.maximum(count, 1.0)
    mean_score = np.sum(
        np.nan_to_num(paths["train_score"][:, :PATH_LIMIT], nan=-20.0) * valid,
        axis=1,
    ) / np.maximum(count, 1.0)
    mean_stability = np.sum(stability["stability_score"] * valid, axis=1) / np.maximum(
        count, 1.0
    )
    region_x = StandardScaler().fit_transform(positions[:, :2])
    mode_x = StandardScaler().fit_transform(
        np.concatenate(
            [
                geodesic,
                blocked[:, None],
                count[:, None],
                mean_depth[:, None],
                mean_score[:, None],
                mean_stability[:, None],
            ],
            axis=1,
        )
    )
    return {
        "region": KMeans(n_clusters=5, random_state=464, n_init=20).fit_predict(region_x),
        "mode": KMeans(n_clusters=5, random_state=1464, n_init=20).fit_predict(mode_x),
    }


def choose_training_rows(
    mask: np.ndarray, label: np.ndarray, seed: int
) -> np.ndarray:
    positives = np.flatnonzero(mask & label)
    negatives = np.flatnonzero(mask & (~label))
    limit = min(len(negatives), MAX_NEGATIVE_RATIO * max(len(positives), 1))
    rng = np.random.default_rng(seed)
    if len(negatives) > limit:
        negatives = rng.choice(negatives, size=limit, replace=False)
    selected = np.concatenate([positives, negatives])
    rng.shuffle(selected)
    return selected


def fit_model(
    feature: np.ndarray,
    label: np.ndarray,
    mass: np.ndarray,
    rows: np.ndarray,
    seed: int,
) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        learning_rate=0.055,
        max_iter=260,
        max_leaf_nodes=31,
        min_samples_leaf=24,
        l2_regularization=0.7,
        class_weight="balanced",
        random_state=seed,
    )
    weight = np.ones(len(rows), dtype=np.float32)
    positive = label[rows]
    if np.any(positive):
        scale = np.sqrt(
            np.maximum(mass[rows][positive], 1e-4)
            / np.maximum(np.median(mass[rows][positive]), 1e-4)
        )
        weight[positive] = np.clip(scale, 0.5, 4.0)
    model.fit(feature[rows], label[rows].astype(np.int8), sample_weight=weight)
    return model


def precision_threshold(
    probability: np.ndarray,
    label: np.ndarray,
    mass: np.ndarray,
) -> float:
    order = np.argsort(probability)[::-1]
    truth = label[order].astype(np.float64)
    cumulative_positive = np.cumsum(truth)
    count = np.arange(1, len(order) + 1)
    precision = cumulative_positive / count
    positive_mass = mass[order] * truth
    cumulative_mass = np.cumsum(positive_mass)
    eligible = (precision >= TARGET_CALIBRATION_PRECISION) & (cumulative_positive >= 8)
    if not np.any(eligible):
        return float(np.quantile(probability, 0.995))
    index = int(np.flatnonzero(eligible)[np.argmax(cumulative_mass[eligible])])
    return float(probability[order[index]])


def evaluate_prediction(
    probability: np.ndarray,
    label: np.ndarray,
    mass: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    selected = probability >= threshold
    true_selected = selected & label
    precision = float(np.sum(true_selected) / max(np.sum(selected), 1))
    recall = float(np.sum(true_selected) / max(np.sum(label), 1))
    mass_recall = float(
        np.sum(mass[true_selected], dtype=np.float64)
        / max(np.sum(mass[label], dtype=np.float64), 1e-30)
    )
    return {
        "average_precision": float(average_precision_score(label, probability)),
        "threshold": threshold,
        "selected_count": int(np.sum(selected)),
        "precision": precision,
        "count_recall": recall,
        "mass_recall": mass_recall,
        "prevalence": float(np.mean(label)),
    }


def evaluate_scheme(
    name: str,
    fold_label: np.ndarray,
    point_id: np.ndarray,
    feature: np.ndarray,
    base_dim: int,
    label: np.ndarray,
    mass: np.ndarray,
) -> dict[str, object]:
    rows = []
    pooled = {
        "base_probability": [],
        "full_probability": [],
        "label": [],
        "mass": [],
    }
    for fold in range(5):
        query = fold_label[point_id] == fold
        fit_point = fold_label != fold
        point_hash = (
            np.arange(len(fold_label), dtype=np.int64) * 2_654_435_761
            + fold * 97_531
        ) % 100
        calibration_point = fit_point & (point_hash < int(100 * CALIBRATION_FRACTION))
        train_point = fit_point & (~calibration_point)
        train_mask = train_point[point_id]
        calibration = calibration_point[point_id]
        train_rows = choose_training_rows(train_mask, label, 46_400 + fold)
        base_model = fit_model(
            feature[:, :base_dim], label, mass, train_rows, 46_410 + fold
        )
        full_model = fit_model(feature, label, mass, train_rows, 46_420 + fold)
        calibration_base = base_model.predict_proba(feature[calibration, :base_dim])[:, 1]
        calibration_full = full_model.predict_proba(feature[calibration])[:, 1]
        base_threshold = precision_threshold(
            calibration_base, label[calibration], mass[calibration]
        )
        full_threshold = precision_threshold(
            calibration_full, label[calibration], mass[calibration]
        )
        query_base = base_model.predict_proba(feature[query, :base_dim])[:, 1]
        query_full = full_model.predict_proba(feature[query])[:, 1]
        base_metrics = evaluate_prediction(
            query_base, label[query], mass[query], base_threshold
        )
        full_metrics = evaluate_prediction(
            query_full, label[query], mass[query], full_threshold
        )
        rows.append(
            {
                "fold": fold,
                "query_point_count": int(np.sum(fold_label == fold)),
                "query_candidate_count": int(np.sum(query)),
                "query_positive_count": int(np.sum(label[query])),
                "base_geometry": base_metrics,
                "with_multiconfig_stability": full_metrics,
                "ap_gain": full_metrics["average_precision"]
                - base_metrics["average_precision"],
            }
        )
        pooled["base_probability"].append(query_base)
        pooled["full_probability"].append(query_full)
        pooled["label"].append(label[query])
        pooled["mass"].append(mass[query])
        print(
            f"v464 {name} fold={fold} AP={base_metrics['average_precision']:.4f}"
            f"->{full_metrics['average_precision']:.4f} precision={full_metrics['precision']:.3f}"
            f" mass_recall={full_metrics['mass_recall']:.3f}",
            flush=True,
        )
    pooled_label = np.concatenate(pooled["label"])
    pooled_mass = np.concatenate(pooled["mass"])
    pooled_base = np.concatenate(pooled["base_probability"])
    pooled_full = np.concatenate(pooled["full_probability"])
    return {
        "name": name,
        "folds": rows,
        "pooled": {
            "prevalence": float(np.mean(pooled_label)),
            "base_average_precision": float(
                average_precision_score(pooled_label, pooled_base)
            ),
            "full_average_precision": float(
                average_precision_score(pooled_label, pooled_full)
            ),
            "ap_gain": float(
                average_precision_score(pooled_label, pooled_full)
                - average_precision_score(pooled_label, pooled_base)
            ),
            "mean_precision": float(
                np.mean([row["with_multiconfig_stability"]["precision"] for row in rows])
            ),
            "mean_mass_recall": float(
                np.mean([row["with_multiconfig_stability"]["mass_recall"] for row in rows])
            ),
            "stability_fold_wins": int(sum(row["ap_gain"] > 0.0 for row in rows)),
            "positive_mass": float(np.sum(pooled_mass[pooled_label], dtype=np.float64)),
        },
    }


def main() -> None:
    for path in (
        PATH_A,
        PATH_B,
        GEODESIC,
        GEOMETRY,
        v312.MATCH_CACHE,
        v305.MATCH_CACHE,
        v305.v297.COMPONENT_CACHE,
        v305.v300.FINGERPRINT_CACHE,
        v312.v304.REPORT,
    ):
        require_file(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data_a = np.load(PATH_A)
    data_b = np.load(PATH_B)
    paths_a = {key: data_a[key] for key in data_a.files}
    paths_b = {key: data_b[key] for key in data_b.files}
    observations = v305.load_observations()
    h_a = candidate_h(paths_a, "train")
    h_b = candidate_h(paths_b, "train")
    positive, positive_mass, positive_confidence = build_positive_labels()
    clean, minimum_cost = clean_label_mask(
        observations, paths_a, h_a, positive
    )
    stability = multiconfig_stability(paths_a, paths_b, h_a, h_b)
    point_id, candidate_id, feature, feature_names, base_dim = path_features(
        paths_a, stability, clean
    )
    label = positive[point_id, candidate_id]
    mass = positive_mass[point_id, candidate_id]
    confidence = positive_confidence[point_id, candidate_id]
    negative_weight = np.full(len(label), 1.0, dtype=np.float32)
    np.savez_compressed(
        DATA_CACHE,
        point_id=point_id,
        candidate_id=candidate_id,
        feature=feature,
        feature_names=np.asarray(feature_names),
        base_dim=np.int32(base_dim),
        label=label,
        mass=mass,
        confidence=confidence,
        minimum_cost=minimum_cost[point_id, candidate_id],
        negative_weight=negative_weight,
    )
    fold_labels = point_fold_labels(paths_a, stability)
    region = evaluate_scheme(
        "leave_region_out",
        fold_labels["region"],
        point_id,
        feature,
        base_dim,
        label,
        mass,
    )
    mode = evaluate_scheme(
        "leave_mode_out",
        fold_labels["mode"],
        point_id,
        feature,
        base_dim,
        label,
        mass,
    )
    pooled_positive_mass = float(np.sum(positive_mass, dtype=np.float64))
    matched = np.load(v312.MATCH_CACHE)
    good = (
        (matched["length_error_bins"] <= v312.GOOD_LENGTH_ERROR)
        & (matched["h_error_bins"] <= v312.GOOD_H_ERROR)
        & (matched["candidate_id"] < PATH_LIMIT)
    )
    positive_stability = stability["stability_score"][
        matched["point_id"][good].astype(np.int64),
        matched["candidate_id"][good].astype(np.int64),
    ]
    region_pass = (
        region["pooled"]["full_average_precision"] >= 0.20
        and region["pooled"]["mean_precision"] >= 0.70
        and region["pooled"]["mean_mass_recall"] >= 0.30
    )
    mode_pass = (
        mode["pooled"]["full_average_precision"] >= 0.25
        and mode["pooled"]["mean_precision"] >= 0.70
        and mode["pooled"]["mean_mass_recall"] >= 0.30
    )
    stability_wins = (
        region["pooled"]["ap_gain"] >= 0.01
        and mode["pooled"]["ap_gain"] >= 0.01
        and region["pooled"]["stability_fold_wins"]
        + mode["pooled"]["stability_fold_wins"]
        >= 7
    )
    passed = bool(region_pass and mode_pass and stability_wins)
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "A path's recurrence across independent depth-16 solver configurations adds "
            "transferable existence information beyond single-configuration geometry."
        ),
        "evidence_boundary": {
            "uses": [
                "official train H path components and fingerprints",
                "official point-cloud geometry-only v291/v321 paths",
                "official train coordinates and geometry features",
            ],
            "does_not_use": [
                "test truth",
                "Sionna CFR/path coefficients",
                "v207 replay truth",
                "online score for model selection",
            ],
        },
        "dataset": {
            "candidate_count": int(len(label)),
            "positive_count": int(np.sum(label)),
            "positive_prevalence": float(np.mean(label)),
            "positive_mass": pooled_positive_mass,
            "positive_stability": summarize(positive_stability),
            "path_limit": PATH_LIMIT,
            "minimum_depth": MIN_DEPTH,
            "clean_negative_cost_min": NEGATIVE_COST_MIN,
        },
        "leave_region_out": region,
        "leave_mode_out": mode,
        "decision_rule": {
            "region": "AP>=0.20, precision>=0.70, mass recall>=0.30",
            "mode": "AP>=0.25, precision>=0.70, mass recall>=0.30",
            "multiconfig_increment": "AP gain>=0.01 in both splits and >=7/10 fold wins",
            "region_pass": bool(region_pass),
            "mode_pass": bool(mode_pass),
            "stability_pass": bool(stability_wins),
            "passed": passed,
        },
        "outputs": {
            "dataset": PATHS.relative(DATA_CACHE),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
