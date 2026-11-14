#!/usr/bin/env python3
"""Score the frozen v334 pathwise KNN with the historical v208 traceback metric."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree

import analyze_v208_strict_historical_replay as v208
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v304_inverse_scatterer_identifiability as v304
import analyze_v306_matched_patch_energy_transformer as v306
import analyze_v325_path_phase_transport_identifiability as v325
import analyze_v326_complex_path_subspace_transport as v326
import analyze_v328_patch_existence_support_field as v328
import analyze_v329_highprecision_complex_path_transport as v329
import analyze_v334_pathwise_derotated_knn as v334
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 336
VERSION = "v336_v334_strict_replay"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = OUT_DIR / "v336_v334_Round1_Test_Channel.npy"
REPORT = REPORT_DIR / f"{VERSION}.json"


def predict_test(
    test_pos: np.ndarray,
    fit_ids: np.ndarray,
    train_pos: np.ndarray,
    channel: np.ndarray,
    candidate: dict[str, np.ndarray],
    selector: np.ndarray,
    threshold: float,
    rows: dict[str, np.ndarray],
    gain: np.ndarray,
    observed_delay: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, dict[str, object]]:
    tree = cKDTree(train_pos[fit_ids, :2])
    distance, local = tree.query(test_pos[:, :2], k=min(v326.K_BASE, len(fit_ids)))
    if distance.ndim == 1:
        distance = distance[:, None]
        local = local[:, None]
    neighbor_ids = fit_ids[local]
    weight = 1.0 / np.maximum(distance, 0.05) ** v326.BASE_POWER
    weight /= weight.sum(axis=1, keepdims=True)
    bs = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
    train_radial = np.linalg.norm(train_pos - bs[None, :], axis=1)
    test_radial = np.linalg.norm(test_pos - bs[None, :], axis=1)
    lookup = v334.row_lookup(rows)
    calibration = json.loads(v304.REPORT.read_text(encoding="utf-8"))["los_calibration"]
    output = np.empty(
        (len(test_pos), channel.shape[1], channel.shape[2], channel.shape[3]),
        dtype=np.complex64,
    )
    corrected_paths = np.zeros(len(test_pos), dtype=np.float32)
    supporting_neighbors = np.zeros(len(test_pos), dtype=np.float32)

    for query_row in range(len(test_pos)):
        neighbors = neighbor_ids[query_row]
        radial_phase = np.exp(
            1j * v326.BASE_BETA * (train_radial[neighbors] - test_radial[query_row])
        ).astype(np.complex64)
        coefficient = (weight[query_row] * radial_phase).astype(np.complex64)
        neighbor_h = np.asarray(channel[neighbors], dtype=np.complex64)
        prediction = np.sum(
            coefficient[:, None, None, None] * neighbor_h,
            axis=0,
            dtype=np.complex64,
        )
        active_candidates = np.flatnonzero(
            (candidate["score"][query_row] > 0.0)
            & (selector[query_row] >= threshold)
        )
        used_patch: set[int] = set()
        for candidate_id in active_candidates:
            patch_id = int(candidate["patch_id"][query_row, candidate_id])
            if patch_id in used_patch:
                continue
            support = [
                (neighbor_row, lookup[(int(point_id), patch_id)])
                for neighbor_row, point_id in enumerate(neighbors)
                if (int(point_id), patch_id) in lookup
            ]
            if not support:
                continue
            used_patch.add(patch_id)
            query_length = float(candidate["path_length"][query_row, candidate_id])
            geometry_delay = (
                float(calibration["intercept_bins"])
                + float(calibration["slope_bins_per_meter"]) * query_length
            )
            for neighbor_row, bank_row in support:
                bank_gain = gain[bank_row]
                bank_delay = float(observed_delay[bank_row])
                bank_path = v334.render_path(bank_gain, bank_delay)
                prediction -= coefficient[neighbor_row] * bank_path
                query_delay = geometry_delay + (
                    bank_delay
                    - float(calibration["intercept_bins"])
                    - float(calibration["slope_bins_per_meter"])
                    * float(rows["length"][bank_row])
                )
                transported_gain = bank_gain * np.complex64(
                    np.exp(-1j * beta * (query_length - float(rows["length"][bank_row])))
                )
                query_path = v334.render_path(transported_gain, query_delay)
                prediction += np.float32(weight[query_row, neighbor_row]) * query_path
                supporting_neighbors[query_row] += 1.0
            corrected_paths[query_row] += 1.0
        output[query_row] = prediction
        if (query_row + 1) % 100 == 0 or query_row + 1 == len(test_pos):
            print(f"v336 test prediction {query_row + 1}/{len(test_pos)}", flush=True)
    return output, {
        "corrected_paths": summarize(corrected_paths),
        "supporting_neighbors": summarize(supporting_neighbors),
        "active_fraction": float(np.mean(corrected_paths > 0.0)),
    }


def main() -> None:
    v329_report = json.loads(v329.REPORT.read_text(encoding="utf-8"))
    gain_data = v325.recover_center_gains()
    rows = v325.matched_rows(gain_data)
    gain = v326.recover_matched_gain_tensors(rows)
    component = np.load(v297.COMPONENT_CACHE)
    observed_delay = component["delay"][rows["point"], rows["component"]].astype(np.float32)
    valid_ids = np.flatnonzero(component["valid"].any(axis=1)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    channel = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    train_data = np.load(v306.TRAIN_CANDIDATES)
    train_candidate = {key: train_data[key] for key in train_data.files}
    candidate_valid, label = v328.build_labels(train_candidate)
    test_data = np.load(v306.TEST_CANDIDATES)
    test_candidate = {key: test_data[key] for key in test_data.files}
    selector = v329.cross_dataset_selector(
        train_candidate,
        candidate_valid,
        label,
        valid_ids,
        train_pos[:, :2],
        test_candidate,
        test_pos[:, :2],
    )
    threshold = float(
        np.median([row["threshold"] for row in v329_report["fold_rows"]])
    )
    beta = v326.estimate_beta(rows, valid_ids, train_pos[:, :2])
    prediction, diagnostics = predict_test(
        test_pos,
        valid_ids,
        train_pos,
        channel,
        test_candidate,
        selector,
        threshold,
        rows,
        gain,
        observed_delay,
        beta,
    )
    np.save(OUTPUT, prediction.astype(np.complex64))
    local_score = v208.score_prediction(OUTPUT)
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "diagnostic_truth_policy": (
            "The v208 synthetic test truth is read only after the frozen v334 method produces "
            "its complete test prediction; it does not select any method parameter."
        ),
        "frozen_config": {
            "threshold": threshold,
            "threshold_source": "median of v329 inner-calibrated fold thresholds",
            "beta": beta,
            "operator": "v334 aggregation-before pathwise de-rotation",
        },
        "diagnostics": diagnostics,
        "prediction": {
            "path": PATHS.relative(OUTPUT),
            "shape": list(prediction.shape),
            "dtype": str(prediction.dtype),
            "finite": bool(np.isfinite(prediction).all()),
        },
        "strict_replay": local_score,
        "references": {
            "v212_score_traceback_leaked": 0.645603643,
            "v284_score_traceback_leaked": 0.592511228,
        },
        "outputs": {
            "prediction": PATHS.relative(OUTPUT),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["strict_replay"], indent=2), flush=True)


if __name__ == "__main__":
    main()
