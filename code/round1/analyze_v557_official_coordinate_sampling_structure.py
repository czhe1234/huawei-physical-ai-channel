#!/usr/bin/env python3
"""Audit whether the official coordinates expose a recoverable route/order state."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree

from project_paths import PATHS


VERSION_NUMBER = 557
VERSION = "v557_official_coordinate_sampling_structure"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
QUANTILES = (0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0)


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        f"q{int(round(100 * quantile)):02d}": float(np.quantile(values, quantile))
        for quantile in QUANTILES
    }


def nearest_summary(points: np.ndarray, neighbors: int = 5) -> dict[str, object]:
    distance, _ = cKDTree(points).query(points, k=neighbors + 1)
    return {
        f"neighbor_{index}": summarize(distance[:, index])
        for index in range(1, neighbors + 1)
    }


def main() -> None:
    train = np.load(PATHS.round1 / "Round1_Train_Pos.npy")
    test = np.load(PATHS.round1 / "Round1_Test_Pos.npy")
    all_points = np.vstack([train, test])[:, :2]
    labels = np.concatenate(
        [np.zeros(len(train), dtype=np.int8), np.ones(len(test), dtype=np.int8)]
    )

    all_distance, all_index = cKDTree(all_points).query(all_points, k=4)
    test_train_distance, _ = cKDTree(train[:, :2]).query(test[:, :2], k=8)
    order_distance = {
        "train": summarize(np.linalg.norm(np.diff(train[:, :2], axis=0), axis=1)),
        "test": summarize(np.linalg.norm(np.diff(test[:, :2], axis=0), axis=1)),
    }
    nearest_label = {
        f"neighbor_{index}_same_split_fraction": float(
            np.mean(labels[all_index[:, index]] == labels)
        )
        for index in range(1, 4)
    }

    route_like = bool(
        order_distance["train"]["q50"]
        < 3.0 * nearest_summary(all_points, 1)["neighbor_1"]["q50"]
    )
    random_mixing_expected = float(
        (len(train) / len(all_points)) ** 2 + (len(test) / len(all_points)) ** 2
    )
    report = {
        "method": VERSION,
        "status": "analysis_rejected",
        "primary_hypothesis": (
            "The 2000 train and 500 test coordinates may be an interleaved sample of "
            "continuous receiver routes, exposing a latent one-dimensional propagation state."
        ),
        "inputs": {
            "train_positions": "datasets/round1/Round1_Train_Pos.npy",
            "test_positions": "datasets/round1/Round1_Test_Pos.npy",
        },
        "population": {
            "train": int(len(train)),
            "test": int(len(test)),
            "all": int(len(all_points)),
            "constant_z": bool(np.unique(np.vstack([train, test])[:, 2]).size == 1),
        },
        "coordinate_ranges": {
            "minimum": all_points.min(axis=0).tolist(),
            "maximum": all_points.max(axis=0).tolist(),
        },
        "nearest_neighbor_distance_m": {
            "train": nearest_summary(train[:, :2]),
            "test": nearest_summary(test[:, :2]),
            "all": nearest_summary(all_points),
            "test_to_train": {
                f"neighbor_{index + 1}": summarize(test_train_distance[:, index])
                for index in range(test_train_distance.shape[1])
            },
        },
        "raw_array_adjacent_distance_m": order_distance,
        "train_test_local_mixing": {
            **nearest_label,
            "random_independent_split_expectation": random_mixing_expected,
        },
        "decision_gate": {
            "raw_order_median_within_3x_spatial_nn": route_like,
            "pass": route_like,
        },
        "conclusion": (
            "The raw train/test order is not a receiver trajectory: adjacent train rows "
            "are about two orders of magnitude farther apart than spatial nearest neighbors. "
            "Train and test labels are locally mixed close to an independent 80/20 split. "
            "Use the coordinates as a two-dimensional transductive point set, not as a "
            "recoverable one-dimensional generation sequence."
        ),
        "outputs": {"report": PATHS.relative(REPORT), "submission_zip": None},
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
