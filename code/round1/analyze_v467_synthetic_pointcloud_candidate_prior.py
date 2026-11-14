#!/usr/bin/env python3
"""Test a synthetic-to-official path-event prior with aligned candidate sets."""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

import analyze_v305_pointcloud_path_identity_match as v305
import analyze_v312_highdepth_path_identity_match as v312
import analyze_v464_multiconfig_path_event_feasibility as v464
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 467
VERSION = "v467_synthetic_pointcloud_candidate_prior"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_CACHE = OUT_DIR / "v467_synthetic_pointcloud_candidates.npz"
LABEL_CACHE = OUT_DIR / "v467_synthetic_candidate_labels.npz"
V442_SHARDS = PATHS.cache_dir(442) / "shards"
V464_DATA = PATHS.cache_dir(464) / "v464_path_event_dataset.npz"
OFFICIAL_PATHS = PATHS.cache_dir(291) / "v291_highdepth_pointcloud_paths.npz"

COUNT = 5_000
PATH_LIMIT = 64
MIN_DEPTH = 2
LIGHT_SPEED = 299_792_458.0
METERS_PER_BIN = LIGHT_SPEED / (192.0 * 30_000.0)
LENGTH_SCALE_BINS = 0.50
H_SCALE_BINS = 0.75
GOOD_LENGTH_ERROR = 0.50
GOOD_H_ERROR = 1.00
PRECISION_TARGET = 0.70
CALIBRATION_FRACTION = 0.20
MIN_SYNTHETIC_AP = 0.20
MIN_OFFICIAL_AP_GAIN = 0.01
MIN_OFFICIAL_FOLD_WINS = 7


def require_file(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def circular_distance(left: np.ndarray, right: np.ndarray, period: float) -> np.ndarray:
    delta = np.abs(np.asarray(left) - np.asarray(right))
    return np.minimum(delta, period - delta)


def true_h_bin(phi_t: np.ndarray) -> np.ndarray:
    relative = np.asarray(phi_t, dtype=np.float32) - math.radians(-30.0)
    return np.mod(np.sin(relative) * 8.0, 16.0).astype(np.float32)


def candidate_h(paths: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    vertices = paths[f"{prefix}vertices"].astype(np.float32)
    first = vertices[:, :, 0].copy()
    direct = paths[f"{prefix}interaction_count"] == 0
    positions = paths[f"{prefix}pos"].astype(np.float32)
    first[direct] = np.broadcast_to(positions[:, None, :], first.shape)[direct]
    return v305.expected_h_bin(first)


def load_true_paths() -> dict[str, np.ndarray]:
    names = (
        "valid",
        "tau",
        "phi_t",
        "normalized_power",
        "interaction_count",
    )
    parts = {name: [] for name in names}
    for shard_id in range(COUNT // 1_000):
        path = V442_SHARDS / f"shard_{shard_id:05d}" / "path_targets.npz"
        require_file(path)
        data = np.load(path)
        for name in names:
            parts[name].append(data[name])
    return {name: np.concatenate(value, axis=0) for name, value in parts.items()}


def build_labels(paths: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if LABEL_CACHE.is_file():
        data = np.load(LABEL_CACHE)
        return {key: data[key] for key in data.files}

    truth = load_true_paths()
    candidate_valid = paths["valid"] & (paths["interaction_count"] >= MIN_DEPTH)
    true_valid = truth["valid"] & (truth["interaction_count"] >= MIN_DEPTH)
    path_h = candidate_h(paths, "")
    truth_h = true_h_bin(truth["phi_t"])
    truth_length = truth["tau"].astype(np.float64) * LIGHT_SPEED
    label = np.zeros(paths["valid"].shape, dtype=np.float32)
    length_error_out = np.full(paths["valid"].shape, np.inf, dtype=np.float32)
    h_error_out = np.full(paths["valid"].shape, np.inf, dtype=np.float32)
    matched_true_mass = np.zeros(COUNT, dtype=np.float32)
    available_true_mass = np.sum(
        truth["normalized_power"].astype(np.float32) * true_valid,
        axis=1,
        dtype=np.float64,
    ).astype(np.float32)

    for point_id in range(COUNT):
        candidate_id = np.flatnonzero(candidate_valid[point_id, :PATH_LIMIT])
        true_id = np.flatnonzero(true_valid[point_id])
        if len(candidate_id) == 0 or len(true_id) == 0:
            continue
        length_error = np.abs(
            paths["path_length"][point_id, candidate_id, None]
            - truth_length[point_id, true_id][None]
        ) / METERS_PER_BIN
        h_error = circular_distance(
            path_h[point_id, candidate_id, None],
            truth_h[point_id, true_id][None],
            16.0,
        )
        cost = (length_error / LENGTH_SCALE_BINS) ** 2 + (
            h_error / H_SCALE_BINS
        ) ** 2
        row, column = linear_sum_assignment(cost)
        for left, right in zip(row, column):
            candidate_index = int(candidate_id[left])
            true_index = int(true_id[right])
            current_length_error = float(length_error[left, right])
            current_h_error = float(h_error[left, right])
            length_error_out[point_id, candidate_index] = current_length_error
            h_error_out[point_id, candidate_index] = current_h_error
            if (
                current_length_error <= GOOD_LENGTH_ERROR
                and current_h_error <= GOOD_H_ERROR
            ):
                mass = float(truth["normalized_power"][point_id, true_index])
                label[point_id, candidate_index] = mass
                matched_true_mass[point_id] += mass
        if (point_id + 1) % 500 == 0 or point_id + 1 == COUNT:
            print(f"v467 synthetic labels {point_id + 1}/{COUNT}", flush=True)

    np.savez_compressed(
        LABEL_CACHE,
        label=label,
        candidate_valid=candidate_valid,
        matched_true_mass=matched_true_mass,
        available_true_mass=available_true_mass,
        length_error=length_error_out,
        h_error=h_error_out,
    )
    data = np.load(LABEL_CACHE)
    return {key: data[key] for key in data.files}


def path_features(
    paths: dict[str, np.ndarray], prefix: str, limit: int = PATH_LIMIT
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = paths[f"{prefix}valid"][:, :limit] & (
        paths[f"{prefix}interaction_count"][:, :limit] >= MIN_DEPTH
    )
    point_id, candidate_id = np.nonzero(valid)
    positions = paths[f"{prefix}pos"].astype(np.float32)
    bs = paths.get(f"{prefix}bs_pos", paths.get("bs_pos"))
    bs = np.asarray(bs, dtype=np.float32)
    if bs.ndim > 1:
        bs = bs[0]
    depth = paths[f"{prefix}interaction_count"][point_id, candidate_id].astype(np.int64)
    interactions = paths[f"{prefix}interactions"][point_id, candidate_id].astype(np.int64)
    vertices = np.nan_to_num(
        paths[f"{prefix}vertices"][point_id, candidate_id].astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    first = vertices[:, 0]
    last = vertices[np.arange(len(vertices)), np.maximum(depth - 1, 0)]
    relative = positions[point_id] - bs[None]
    direct = np.linalg.norm(relative, axis=1)
    first_vector = first - bs[None]
    last_vector = positions[point_id] - last
    first_unit = first_vector / np.maximum(
        np.linalg.norm(first_vector, axis=1, keepdims=True), 1e-6
    )
    last_unit = last_vector / np.maximum(
        np.linalg.norm(last_vector, axis=1, keepdims=True), 1e-6
    )
    length = paths[f"{prefix}path_length"][point_id, candidate_id].astype(np.float32)
    score = np.nan_to_num(
        paths[f"{prefix}score"][point_id, candidate_id].astype(np.float32),
        nan=-20.0,
        posinf=8.0,
        neginf=-20.0,
    )
    angles = [
        paths[f"{prefix}{name}"][point_id, candidate_id].astype(np.float32)
        for name in ("theta_t", "phi_t", "theta_r", "phi_r")
    ]
    angle_features = np.column_stack(
        [value for angle in angles for value in (np.sin(angle), np.cos(angle))]
    )
    mask = paths[f"{prefix}interaction_mask"][point_id, candidate_id].astype(np.int16)
    interaction_count = np.stack(
        [np.sum(interactions == value, axis=1) for value in (1, 4, 8)], axis=1
    ).astype(np.float32)
    features = np.concatenate(
        [
            positions[point_id, :2] / 500.0,
            relative[:, :2] / 500.0,
            (direct / 500.0)[:, None],
            (length / 800.0)[:, None],
            ((length - direct) / 500.0)[:, None],
            np.clip(score, -20.0, 8.0)[:, None] / 8.0,
            (depth.astype(np.float32) / 16.0)[:, None],
            np.stack([((mask & bit) != 0).astype(np.float32) for bit in (1, 2, 4, 8)], axis=1),
            interaction_count / 16.0,
            angle_features,
            first_vector / 500.0,
            last_vector / 500.0,
            first_unit,
            last_unit,
            interactions[:, :6].astype(np.float32) / 8.0,
        ],
        axis=1,
    ).astype(np.float32)
    return point_id.astype(np.int32), candidate_id.astype(np.int16), features


def fit_model(
    feature: np.ndarray,
    label: np.ndarray,
    mass: np.ndarray,
    rows: np.ndarray,
    seed: int,
) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        learning_rate=0.055,
        max_iter=240,
        max_leaf_nodes=31,
        min_samples_leaf=28,
        l2_regularization=0.8,
        class_weight="balanced",
        random_state=seed,
    )
    sample_weight = np.ones(len(rows), dtype=np.float32)
    positive = label[rows]
    if np.any(positive):
        scale = np.sqrt(
            np.maximum(mass[rows][positive], 1e-5)
            / np.maximum(np.median(mass[rows][positive]), 1e-5)
        )
        sample_weight[positive] = np.clip(scale, 0.5, 4.0)
    model.fit(feature[rows], label[rows].astype(np.int8), sample_weight=sample_weight)
    return model


def synthetic_validation(
    point_id: np.ndarray,
    feature: np.ndarray,
    label: np.ndarray,
    mass: np.ndarray,
    positions: np.ndarray,
) -> tuple[list[dict[str, float | int]], float]:
    groups = KMeans(n_clusters=5, random_state=467, n_init=20).fit_predict(
        positions[:, :2]
    )
    rows = []
    prediction = np.zeros(len(label), dtype=np.float32)
    for fold in range(5):
        query = groups[point_id] == fold
        fit = ~query
        train_rows = v464.choose_training_rows(fit, label, 46_700 + fold)
        model = fit_model(feature, label, mass, train_rows, 46_700 + fold)
        prediction[query] = model.predict_proba(feature[query])[:, 1]
        ap = float(average_precision_score(label[query], prediction[query]))
        rows.append(
            {
                "fold": fold,
                "query_count": int(np.sum(query)),
                "positive_count": int(np.sum(label[query])),
                "prevalence": float(np.mean(label[query])),
                "average_precision": ap,
            }
        )
        print(f"v467 synthetic fold={fold} AP={ap:.4f}", flush=True)
    return rows, float(average_precision_score(label, prediction))


def calibration_points(point_ids: np.ndarray, fold: int) -> np.ndarray:
    unique = np.unique(point_ids)
    rng = np.random.default_rng(467_000 + fold)
    count = max(int(round(CALIBRATION_FRACTION * len(unique))), 1)
    return rng.choice(unique, size=count, replace=False)


def official_validation(
    base_feature: np.ndarray,
    prior: np.ndarray,
    point_id: np.ndarray,
    label: np.ndarray,
    mass: np.ndarray,
    point_groups: dict[str, np.ndarray],
) -> tuple[list[dict[str, object]], dict[str, float | int]]:
    rows: list[dict[str, object]] = []
    ap_gain = []
    fold_wins = 0
    for split_name, groups in point_groups.items():
        for fold in range(5):
            query = groups[point_id] == fold
            fit = ~query
            calibration_id = calibration_points(point_id[fit], fold + (0 if split_name == "region" else 10))
            calibration = fit & np.isin(point_id, calibration_id)
            train = fit & (~calibration)
            train_rows = v464.choose_training_rows(train, label, 467_100 + fold)
            variants = {
                "baseline": base_feature,
                "synthetic_prior": np.column_stack([base_feature, prior]),
            }
            variant_rows = {}
            for variant_name, feature in variants.items():
                model = fit_model(
                    feature,
                    label,
                    mass,
                    train_rows,
                    467_100 + fold + (100 if variant_name == "synthetic_prior" else 0),
                )
                calibration_probability = model.predict_proba(feature[calibration])[:, 1]
                threshold = v464.precision_threshold(
                    calibration_probability, label[calibration], mass[calibration]
                )
                probability = model.predict_proba(feature[query])[:, 1]
                metrics = v464.evaluate_prediction(
                    probability, label[query], mass[query], threshold
                )
                variant_rows[variant_name] = metrics
            gain = (
                variant_rows["synthetic_prior"]["average_precision"]
                - variant_rows["baseline"]["average_precision"]
            )
            ap_gain.append(gain)
            fold_wins += int(gain > 0.0)
            rows.append(
                {
                    "split": split_name,
                    "fold": fold,
                    "query_count": int(np.sum(query)),
                    "baseline": variant_rows["baseline"],
                    "synthetic_prior": variant_rows["synthetic_prior"],
                    "average_precision_gain": gain,
                }
            )
            print(
                f"v467 official {split_name} fold={fold} AP gain={gain:+.5f}",
                flush=True,
            )
    split_gain = {
        name: float(
            np.mean([row["average_precision_gain"] for row in rows if row["split"] == name])
        )
        for name in ("region", "mode")
    }
    summary = {
        "mean_average_precision_gain": float(np.mean(ap_gain)),
        "region_average_precision_gain": split_gain["region"],
        "mode_average_precision_gain": split_gain["mode"],
        "fold_wins": int(fold_wins),
    }
    return rows, summary


def main() -> None:
    for path in (CANDIDATE_CACHE, V464_DATA, OFFICIAL_PATHS):
        require_file(path)
    synthetic_data = np.load(CANDIDATE_CACHE)
    synthetic_paths = {key: synthetic_data[key] for key in synthetic_data.files}
    labels = build_labels(synthetic_paths)
    synthetic_point, synthetic_candidate, synthetic_feature = path_features(
        synthetic_paths, ""
    )
    synthetic_mass = labels["label"][synthetic_point, synthetic_candidate].astype(np.float32)
    synthetic_label = synthetic_mass > 0.0
    synthetic_rows, synthetic_ap = synthetic_validation(
        synthetic_point,
        synthetic_feature,
        synthetic_label,
        synthetic_mass,
        synthetic_paths["pos"],
    )

    final_rows = v464.choose_training_rows(
        np.ones(len(synthetic_label), dtype=bool), synthetic_label, 467_999
    )
    synthetic_model = fit_model(
        synthetic_feature,
        synthetic_label,
        synthetic_mass,
        final_rows,
        467_999,
    )

    official_path_data = np.load(OFFICIAL_PATHS)
    official_paths = {key: official_path_data[key] for key in official_path_data.files}
    official_point, official_candidate, official_feature = path_features(
        official_paths, "train_"
    )
    official_prior_matrix = np.zeros((2000, PATH_LIMIT), dtype=np.float32)
    official_prior_matrix[official_point, official_candidate] = synthetic_model.predict_proba(
        official_feature
    )[:, 1]

    official_data = np.load(V464_DATA)
    point_id = official_data["point_id"].astype(np.int32)
    candidate_id = official_data["candidate_id"].astype(np.int16)
    base_dim = int(official_data["base_dim"])
    base_feature = official_data["feature"][:, :base_dim].astype(np.float32)
    official_label = official_data["label"].astype(bool)
    official_mass = official_data["mass"].astype(np.float32)
    prior = official_prior_matrix[point_id, candidate_id]

    path_a = np.load(v464.PATH_A)
    path_b = np.load(v464.PATH_B)
    h_a = v464.candidate_h(path_a, "train")
    h_b = v464.candidate_h(path_b, "train")
    stability = v464.multiconfig_stability(path_a, path_b, h_a, h_b)
    point_groups = v464.point_fold_labels(path_a, stability)
    official_rows, official_summary = official_validation(
        base_feature,
        prior,
        point_id,
        official_label,
        official_mass,
        point_groups,
    )

    matched_fraction = labels["matched_true_mass"] / np.maximum(
        labels["available_true_mass"], 1e-30
    )
    passed = (
        synthetic_ap >= MIN_SYNTHETIC_AP
        and official_summary["region_average_precision_gain"] >= MIN_OFFICIAL_AP_GAIN
        and official_summary["mode_average_precision_gain"] >= MIN_OFFICIAL_AP_GAIN
        and official_summary["fold_wins"] >= MIN_OFFICIAL_FOLD_WINS
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Synthetic true-scene paths can supervise the same official-point-cloud candidate "
            "interface used at deployment, removing v445 power-ranked membership leakage and "
            "providing a transferable path-existence prior."
        ),
        "compliance_boundary": {
            "uses": [
                "v442 auxiliary Sionna paths at random non-test positions",
                "official point-cloud geometry candidates at those positions",
                "official train-H path-event labels only inside each validation fold",
            ],
            "does_not_use": [
                "Sionna channels or paths at official test coordinates",
                "hidden test H or online score for selection",
                "v212/v458 test teacher outputs",
            ],
        },
        "synthetic_labels": {
            "point_count": COUNT,
            "candidate_row_count": int(len(synthetic_label)),
            "positive_count": int(np.sum(synthetic_label)),
            "positive_prevalence": float(np.mean(synthetic_label)),
            "matched_true_mass_fraction": summarize(matched_fraction),
        },
        "synthetic_validation": {
            "fold_rows": synthetic_rows,
            "pooled_average_precision": synthetic_ap,
        },
        "official_validation": {
            "fold_rows": official_rows,
            "summary": official_summary,
            "prior": summarize(prior),
        },
        "decision_rule": {
            "synthetic_average_precision": f">={MIN_SYNTHETIC_AP}",
            "official_region_average_precision_gain": f">={MIN_OFFICIAL_AP_GAIN}",
            "official_mode_average_precision_gain": f">={MIN_OFFICIAL_AP_GAIN}",
            "official_fold_wins": f">={MIN_OFFICIAL_FOLD_WINS}/10",
            "passed": bool(passed),
        },
        "next_action": (
            "Scale the aligned synthetic candidate dataset only if the prior improves both "
            "official split families; otherwise stop synthetic path-event pretraining."
        ),
        "outputs": {
            "candidate_cache": PATHS.relative(CANDIDATE_CACHE),
            "label_cache": PATHS.relative(LABEL_CACHE),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "synthetic_ap": synthetic_ap,
                "matched_mass": summarize(matched_fraction),
                "official": official_summary,
                "passed": passed,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
