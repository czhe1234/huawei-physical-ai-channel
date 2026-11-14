#!/usr/bin/env python3
"""Test whether v305-unmatched paths form reusable effective scatterers."""

from __future__ import annotations

import json

import numpy as np
from sklearn.cluster import DBSCAN

import analyze_v297_continuous_delay_component_transport as v297
import analyze_v300_path_gain_fingerprint_oracle as v300
import analyze_v304_inverse_scatterer_identifiability as v304
import analyze_v305_pointcloud_path_identity_match as v305
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 309
VERSION = "v309_unmatched_effective_scatterers"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CLUSTER_CACHE = OUT_DIR / "v309_effective_scatterer_clusters.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"

GOOD_LENGTH_ERROR = 0.50
GOOD_H_ERROR = 1.00
CLUSTER_SETTINGS = ((2.0, 5), (3.0, 5), (5.0, 5), (5.0, 3))
SELECTED_EPS = 3.0
SELECTED_MIN_SAMPLES = 5
SHUFFLE_REPEATS = 20


def expected_h_from_point(point: np.ndarray) -> np.ndarray:
    vector = point - v304.BS[:2]
    azimuth = np.arctan2(vector[:, 1], vector[:, 0])
    return v304.expected_h_bin(azimuth)


def weighted_mean(value: np.ndarray, weight: np.ndarray) -> float:
    return float(
        np.sum(value.astype(np.float64) * weight.astype(np.float64))
        / np.maximum(np.sum(weight, dtype=np.float64), 1e-30)
    )


def residuals(
    source_point: np.ndarray,
    ue_point: np.ndarray,
    observed_length: np.ndarray,
    observed_h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predicted_length = (
        np.linalg.norm(source_point - v304.BS[:2], axis=1)
        + np.linalg.norm(source_point - ue_point, axis=1)
    )
    predicted_h = expected_h_from_point(source_point)
    length_error = np.abs(predicted_length - observed_length) / v304.METERS_PER_BIN
    h_error = v305.circular_distance(predicted_h, observed_h, 16.0)
    return length_error.astype(np.float32), h_error.astype(np.float32)


def main() -> None:
    inverse = np.load(v304.SCATTERER_CACHE)
    matches = np.load(v305.MATCH_CACHE)
    component = np.load(v297.COMPONENT_CACHE)
    fingerprint = np.load(v300.FINGERPRINT_CACHE)
    calibration = json.loads(v304.REPORT.read_text(encoding="utf-8"))["los_calibration"]
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    good = np.zeros(component["valid"].shape, dtype=bool)
    high_quality = (
        (matches["length_error_bins"] <= GOOD_LENGTH_ERROR)
        & (matches["h_error_bins"] <= GOOD_H_ERROR)
    )
    good[
        matches["point_id"][high_quality],
        matches["component_id"][high_quality].astype(np.int64),
    ] = True
    point = inverse["absolute_point"].astype(np.float32)
    point_id = inverse["absolute_point_id"].astype(np.int64)
    component_id = inverse["absolute_component_id"].astype(np.int64)
    mass = component["weight"][point_id, component_id].astype(np.float32)
    unmatched = ~good[point_id, component_id]
    point = point[unmatched]
    point_id = point_id[unmatched]
    component_id = component_id[unmatched]
    mass = mass[unmatched]
    observed_length = (
        component["delay"][point_id, component_id]
        - float(calibration["intercept_bins"])
    ) / float(calibration["slope_bins_per_meter"])
    observed_h = v304.horizontal_center(fingerprint["horizontal"])[point_id, component_id]

    setting_rows = []
    selected_labels = None
    for eps, min_samples in CLUSTER_SETTINGS:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(point)
        clustered = labels >= 0
        unique = np.unique(labels[clustered])
        point_diversity = []
        for label in unique:
            point_diversity.append(len(np.unique(point_id[labels == label])))
        row = {
            "eps": eps,
            "min_samples": min_samples,
            "cluster_count": int(len(unique)),
            "point_fraction": float(np.mean(clustered)),
            "mass_fraction": weighted_mean(clustered.astype(np.float32), mass),
            "unique_ue_per_cluster": summarize(np.asarray(point_diversity)),
        }
        setting_rows.append(row)
        if eps == SELECTED_EPS and min_samples == SELECTED_MIN_SAMPLES:
            selected_labels = labels
    assert selected_labels is not None
    clustered = selected_labels >= 0
    centers: dict[int, np.ndarray] = {}
    for label in np.unique(selected_labels[clustered]):
        centers[int(label)] = np.median(point[selected_labels == label], axis=0)
    predicted_source = np.asarray(
        [centers[int(label)] for label in selected_labels[clustered]], dtype=np.float32
    )
    length_error, h_error = residuals(
        predicted_source,
        train_pos[point_id[clustered], :2],
        observed_length[clustered],
        observed_h[clustered],
    )
    good_reconstruction = (length_error <= 0.50) & (h_error <= 1.0)
    actual = {
        "clustered_count": int(np.sum(clustered)),
        "cluster_count": int(len(centers)),
        "length_error_bins": summarize(length_error),
        "h_error_bins": summarize(h_error),
        "mass_weighted_length_error": weighted_mean(length_error, mass[clustered]),
        "mass_weighted_h_error": weighted_mean(h_error, mass[clustered]),
        "mass_good_fraction": weighted_mean(good_reconstruction.astype(np.float32), mass[clustered]),
    }

    rng = np.random.default_rng(309)
    control_good = []
    control_length = []
    control_h = []
    clustered_ids = np.flatnonzero(clustered)
    for _repeat in range(SHUFFLE_REPEATS):
        permutation = rng.permutation(clustered_ids)
        shuffled_length, shuffled_h = residuals(
            predicted_source,
            train_pos[point_id[permutation], :2],
            observed_length[permutation],
            observed_h[permutation],
        )
        weight = mass[permutation]
        control_length.append(weighted_mean(shuffled_length, weight))
        control_h.append(weighted_mean(shuffled_h, weight))
        control_good.append(
            weighted_mean(
                ((shuffled_length <= 0.50) & (shuffled_h <= 1.0)).astype(np.float32),
                weight,
            )
        )
    control = {
        "mass_weighted_length_error": summarize(np.asarray(control_length)),
        "mass_weighted_h_error": summarize(np.asarray(control_h)),
        "mass_good_fraction": summarize(np.asarray(control_good)),
        "actual_good_minus_control": float(actual["mass_good_fraction"] - np.mean(control_good)),
    }
    cluster_ids = np.asarray(sorted(centers), dtype=np.int32)
    cluster_centers = np.asarray([centers[int(value)] for value in cluster_ids], dtype=np.float32)
    np.savez_compressed(
        CLUSTER_CACHE,
        point=point,
        point_id=point_id,
        component_id=component_id,
        mass=mass,
        labels=selected_labels,
        cluster_ids=cluster_ids,
        cluster_centers=cluster_centers,
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "unmatched": {
            "count": int(len(point)),
            "mass": summarize(mass),
        },
        "cluster_settings": setting_rows,
        "selected": {
            "eps": SELECTED_EPS,
            "min_samples": SELECTED_MIN_SAMPLES,
            "actual": actual,
            "shuffle_control": control,
        },
        "decision_rule": (
            "Add effective-source candidates only if clustered unmatched mass is substantial "
            "and cluster-center length/H reconstruction clearly beats shuffled UE controls."
        ),
        "outputs": {
            "clusters": PATHS.relative(CLUSTER_CACHE),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
