#!/usr/bin/env python3
"""Learn a high-precision spatial existence field for matched wall paths."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score

import analyze_v297_continuous_delay_component_transport as v297
import analyze_v305_pointcloud_path_identity_match as v305
import analyze_v306_matched_patch_energy_transformer as v306
import analyze_v309_unmatched_effective_scatterers as v309
import analyze_v315_full_sequence_confirmatory as v315
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 328
VERSION = "v328_patch_existence_support_field"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"

K_SUPPORT = 16
DISTANCE_FLOOR = 0.25
PRECISION_TARGET = 0.70
MIN_RECALL_AT_PRECISION = 0.15
MIN_AP = 0.15
SHUFFLE_REPEATS = 10


def build_labels(candidate: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    valid = candidate["score"] > 0.0
    label = np.zeros(valid.shape, dtype=np.float32)
    match = np.load(v305.MATCH_CACHE)
    good = (
        (match["length_error_bins"] <= v309.GOOD_LENGTH_ERROR)
        & (match["h_error_bins"] <= v309.GOOD_H_ERROR)
    )
    point = match["point_id"][good].astype(np.int64)
    candidate_id = match["candidate_id"][good].astype(np.int64)
    mass = match["mass"][good].astype(np.float32)
    np.maximum.at(label, (point, candidate_id), mass)
    return valid, label


def patch_rows(
    candidate: dict[str, np.ndarray], valid: np.ndarray, point_ids: np.ndarray
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    rows: dict[int, list[tuple[int, int]]] = {}
    for point_id in point_ids:
        for candidate_id in np.flatnonzero(valid[point_id]):
            patch_id = int(candidate["patch_id"][point_id, candidate_id])
            rows.setdefault(patch_id, []).append((int(point_id), int(candidate_id)))
    return {
        patch_id: (
            np.asarray([item[0] for item in values], dtype=np.int64),
            np.asarray([item[1] for item in values], dtype=np.int64),
        )
        for patch_id, values in rows.items()
    }


def predict(
    candidate: dict[str, np.ndarray],
    valid: np.ndarray,
    label: np.ndarray,
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
    xy: np.ndarray,
    shuffled_label: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_label = label if shuffled_label is None else shuffled_label
    fit_patch = patch_rows(candidate, valid, fit_ids)
    query_point: list[int] = []
    query_candidate: list[int] = []
    prediction: list[float] = []
    truth: list[float] = []
    tree_cache: dict[int, tuple[np.ndarray, np.ndarray, cKDTree]] = {}
    for point_id in query_ids:
        for candidate_id in np.flatnonzero(valid[point_id]):
            patch_id = int(candidate["patch_id"][point_id, candidate_id])
            if patch_id not in fit_patch:
                score = 0.0
            else:
                if patch_id not in tree_cache:
                    fit_point, fit_candidate = fit_patch[patch_id]
                    tree_cache[patch_id] = (
                        fit_point,
                        fit_candidate,
                        cKDTree(xy[fit_point]),
                    )
                fit_point, fit_candidate, tree = tree_cache[patch_id]
                count = min(K_SUPPORT, len(fit_point))
                distance, local = tree.query(xy[point_id], k=count)
                distance = np.atleast_1d(distance)
                local = np.atleast_1d(local)
                weight = 1.0 / np.maximum(distance, DISTANCE_FLOOR) ** 2
                neighbor_label = (
                    source_label[fit_point[local], fit_candidate[local]] > 0.0
                ).astype(np.float64)
                score = float(np.sum(weight * neighbor_label) / np.sum(weight))
            query_point.append(int(point_id))
            query_candidate.append(int(candidate_id))
            prediction.append(score)
            truth.append(float(label[point_id, candidate_id]))
    return (
        np.asarray(query_point, dtype=np.int32),
        np.asarray(query_candidate, dtype=np.int16),
        np.asarray(prediction, dtype=np.float32),
        np.asarray(truth, dtype=np.float32),
    )


def precision_recall_at_target(
    score: np.ndarray, label: np.ndarray, target: float
) -> dict[str, float]:
    binary = label > 0.0
    order = np.argsort(score)[::-1]
    positive = binary[order].astype(np.float64)
    mass = label[order].astype(np.float64)
    count_precision = np.cumsum(positive) / np.arange(1, len(order) + 1)
    mass_precision = np.cumsum(mass) / np.maximum(
        np.cumsum((score[order] > 0.0).astype(np.float64)), 1.0
    )
    valid = np.flatnonzero(count_precision >= target)
    if len(valid) == 0:
        return {
            "selected_count": 0,
            "pair_fraction": 0.0,
            "count_precision": 0.0,
            "count_recall": 0.0,
            "mass_recall": 0.0,
            "score_threshold": float("inf"),
        }
    end = int(valid[-1]) + 1
    selected_positive = float(np.sum(positive[:end]))
    return {
        "selected_count": end,
        "pair_fraction": end / max(len(order), 1),
        "count_precision": selected_positive / end,
        "count_recall": selected_positive / max(float(np.sum(binary)), 1.0),
        "mass_recall": float(np.sum(mass[:end]) / np.maximum(np.sum(label), 1e-30)),
        "score_threshold": float(score[order[end - 1]]),
        "mean_selected_mass": float(np.mean(mass[:end])),
        "mass_per_selected_pair": float(np.sum(mass[:end]) / end),
    }


def shuffle_within_patch(
    candidate: dict[str, np.ndarray],
    valid: np.ndarray,
    label: np.ndarray,
    fit_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    shuffled = label.copy()
    for _patch_id, (point, candidate_id) in patch_rows(candidate, valid, fit_ids).items():
        values = shuffled[point, candidate_id].copy()
        rng.shuffle(values)
        shuffled[point, candidate_id] = values
    return shuffled


def main() -> None:
    if not v306.TRAIN_CANDIDATES.is_file():
        raise FileNotFoundError(v306.TRAIN_CANDIDATES)
    data = np.load(v306.TRAIN_CANDIDATES)
    candidate = {key: data[key] for key in data.files}
    valid, label = build_labels(candidate)
    component = np.load(v297.COMPONENT_CACHE)
    valid_ids = np.flatnonzero(component["valid"].any(axis=1)).astype(np.int64)
    xy = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)[:, :2]
    folds = v315.disjoint_folds(valid_ids, xy)
    rng = np.random.default_rng(328)
    fold_rows = []

    for fold, query_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, query_ids)
        _point, _candidate, score, truth = predict(
            candidate, valid, label, fit_ids, query_ids, xy
        )
        binary = truth > 0.0
        ap = float(average_precision_score(binary, score))
        high_precision = precision_recall_at_target(score, truth, PRECISION_TARGET)
        shuffle_ap = []
        for _ in range(SHUFFLE_REPEATS):
            shuffled_label = shuffle_within_patch(
                candidate, valid, label, fit_ids, rng
            )
            _p, _c, shuffled_score, _truth = predict(
                candidate,
                valid,
                label,
                fit_ids,
                query_ids,
                xy,
                shuffled_label=shuffled_label,
            )
            shuffle_ap.append(float(average_precision_score(binary, shuffled_score)))
        fold_rows.append(
            {
                "fold": fold,
                "fit_point_count": int(len(fit_ids)),
                "query_point_count": int(len(query_ids)),
                "query_pair_count": int(len(score)),
                "positive_pair_count": int(np.sum(binary)),
                "positive_prevalence": float(np.mean(binary)),
                "average_precision": ap,
                "high_precision": high_precision,
                "shuffle_average_precision": summarize(np.asarray(shuffle_ap)),
                "ap_minus_shuffle": ap - float(np.mean(shuffle_ap)),
            }
        )
        print(
            f"v328 fold={fold} AP={ap:.4f} precision={high_precision['count_precision']:.3f} "
            f"recall={high_precision['count_recall']:.3f}",
            flush=True,
        )

    mean_ap = float(np.mean([row["average_precision"] for row in fold_rows]))
    mean_recall = float(
        np.mean([row["high_precision"]["count_recall"] for row in fold_rows])
    )
    mean_mass_recall = float(
        np.mean([row["high_precision"]["mass_recall"] for row in fold_rows])
    )
    fold_wins = int(
        sum(
            row["average_precision"] >= MIN_AP
            and row["high_precision"]["count_recall"] >= MIN_RECALL_AT_PRECISION
            for row in fold_rows
        )
    )
    passed = mean_ap >= MIN_AP and mean_recall >= MIN_RECALL_AT_PRECISION and fold_wins >= 4
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "A patch-specific spatial KNN over candidate exposures can identify a high-precision "
            "subset of one-bounce path existence for v325 phase transport."
        ),
        "parameters": {
            "support_k": K_SUPPORT,
            "distance_floor_m": DISTANCE_FLOOR,
            "precision_target": PRECISION_TARGET,
            "minimum_recall_at_precision": MIN_RECALL_AT_PRECISION,
            "minimum_average_precision": MIN_AP,
            "shuffle_repeats": SHUFFLE_REPEATS,
        },
        "label": {
            "positive_pair_count": int(np.sum(label > 0.0)),
            "positive_mass": float(np.sum(label, dtype=np.float64)),
            "valid_candidate_pair_count": int(np.sum(valid)),
        },
        "fold_rows": fold_rows,
        "summary": {
            "average_precision": mean_ap,
            "count_recall_at_70pct_precision": mean_recall,
            "mass_recall_at_70pct_precision": mean_mass_recall,
            "fold_wins": fold_wins,
        },
        "decision_rule": {
            "average_precision": f">={MIN_AP}",
            "count_recall_at_70pct_precision": f">={MIN_RECALL_AT_PRECISION}",
            "fold_wins": ">=4/5",
            "passed": bool(passed),
        },
        "outputs": {"report": PATHS.relative(REPORT), "zip": None},
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
