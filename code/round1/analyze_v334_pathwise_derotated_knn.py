#!/usr/bin/env python3
"""Align matched reflection paths before KNN aggregation instead of replacing them after."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree

import analyze_v297_continuous_delay_component_transport as v297
import analyze_v304_inverse_scatterer_identifiability as v304
import analyze_v306_matched_patch_energy_transformer as v306
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v325_path_phase_transport_identifiability as v325
import analyze_v326_complex_path_subspace_transport as v326
import analyze_v328_patch_existence_support_field as v328
import analyze_v329_highprecision_complex_path_transport as v329
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 334
VERSION = "v334_pathwise_derotated_knn"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"


def render_path(gain: np.ndarray, delay: float) -> np.ndarray:
    frequency = np.arange(v297.N_DELAY, dtype=np.float32)
    steering = np.exp(
        -2j * np.pi * frequency * float(delay) / v297.N_DELAY
    ).astype(np.complex64)
    return (gain.reshape(-1, 1) * steering[None, :]).reshape(256, 4, v297.N_DELAY)


def row_lookup(rows: dict[str, np.ndarray]) -> dict[tuple[int, int], int]:
    return {
        (int(point_id), int(patch_id)): row
        for row, (point_id, patch_id) in enumerate(zip(rows["point"], rows["patch"]))
    }


def predict(
    query_ids: np.ndarray,
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
    distance, local = tree.query(train_pos[query_ids, :2], k=min(v326.K_BASE, len(fit_ids)))
    if distance.ndim == 1:
        distance = distance[:, None]
        local = local[:, None]
    neighbor_ids = fit_ids[local]
    weight = 1.0 / np.maximum(distance, 0.05) ** v326.BASE_POWER
    weight /= weight.sum(axis=1, keepdims=True)
    bs = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
    radial_distance = np.linalg.norm(train_pos - bs[None, :], axis=1)
    lookup = row_lookup(rows)
    calibration = json.loads(v304.REPORT.read_text(encoding="utf-8"))["los_calibration"]
    output = np.empty(
        (len(query_ids), channel.shape[1], channel.shape[2], channel.shape[3]),
        dtype=np.complex64,
    )
    corrected_paths = np.zeros(len(query_ids), dtype=np.float32)
    supporting_neighbors = np.zeros(len(query_ids), dtype=np.float32)

    for query_row, query_id in enumerate(query_ids):
        neighbors = neighbor_ids[query_row]
        radial_phase = np.exp(
            1j * v326.BASE_BETA * (radial_distance[neighbors] - radial_distance[query_id])
        ).astype(np.complex64)
        coefficient = (weight[query_row] * radial_phase).astype(np.complex64)
        neighbor_h = np.asarray(channel[neighbors], dtype=np.complex64)
        prediction = np.sum(
            coefficient[:, None, None, None] * neighbor_h,
            axis=0,
            dtype=np.complex64,
        )
        active_candidates = np.flatnonzero(
            (candidate["score"][query_id] > 0.0)
            & (selector[query_row] >= threshold)
        )
        used_patch: set[int] = set()
        for candidate_id in active_candidates:
            patch_id = int(candidate["patch_id"][query_id, candidate_id])
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
            query_length = float(candidate["path_length"][query_id, candidate_id])
            geometry_delay = (
                float(calibration["intercept_bins"])
                + float(calibration["slope_bins_per_meter"]) * query_length
            )
            for neighbor_row, bank_row in support:
                bank_gain = gain[bank_row]
                bank_delay = float(observed_delay[bank_row])
                bank_path = render_path(bank_gain, bank_delay)
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
                query_path = render_path(transported_gain, query_delay)
                prediction += np.float32(weight[query_row, neighbor_row]) * query_path
                supporting_neighbors[query_row] += 1.0
            corrected_paths[query_row] += 1.0
        output[query_row] = prediction
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
    channel = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])
    data = np.load(v306.TRAIN_CANDIDATES)
    candidate = {key: data[key] for key in data.files}
    candidate_valid, label = v328.build_labels(candidate)
    fold_rows = []

    for fold, query_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, query_ids)
        threshold = float(v329_report["fold_rows"][fold]["threshold"])
        beta = float(v329_report["fold_rows"][fold]["outer_beta"])
        selector, _truth = v329.selector_matrix(
            candidate,
            candidate_valid,
            label,
            fit_ids,
            query_ids,
            train_pos[:, :2],
        )
        base = v326.base_predictions(train_pos[query_ids], fit_ids, train_pos, channel)
        prediction, diagnostics = predict(
            query_ids,
            fit_ids,
            train_pos,
            channel,
            candidate,
            selector,
            threshold,
            rows,
            gain,
            observed_delay,
            beta,
        )
        truth = np.asarray(channel[query_ids])
        base_metrics = v326.score(base, truth)
        metrics = v326.score(prediction, truth)
        fold_rows.append(
            {
                "fold": fold,
                "threshold": threshold,
                "beta": beta,
                "diagnostics": diagnostics,
                "base": base_metrics,
                "prediction": metrics,
                "score_delta": metrics["score"] - base_metrics["score"],
                "pas_delta": metrics["pas"] - base_metrics["pas"],
                "pdp_delta": metrics["pdp"] - base_metrics["pdp"],
                "inv_nmse_delta": metrics["inv_nmse"] - base_metrics["inv_nmse"],
            }
        )
        print(
            f"v334 fold={fold} active={diagnostics['active_fraction']:.3f} "
            f"score={fold_rows[-1]['score_delta']:+.6f}",
            flush=True,
        )

    summary = {
        key: float(np.mean([row[key] for row in fold_rows]))
        for key in ("score_delta", "pas_delta", "pdp_delta", "inv_nmse_delta")
    }
    summary["fold_wins"] = int(sum(row["score_delta"] > 0.0 for row in fold_rows))
    passed = (
        summary["score_delta"] >= v326.MIN_SCORE_DELTA
        and summary["fold_wins"] >= 4
        and summary["pas_delta"] >= -v326.MAX_SPECTRAL_DEGRADATION
        and summary["pdp_delta"] >= -v326.MAX_SPECTRAL_DEGRADATION
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Removing each identified neighbor path before radial KNN and re-adding it after "
            "path-specific length/delay transport avoids post-hoc base competition."
        ),
        "parameters": {
            "base_k": v326.K_BASE,
            "base_power": v326.BASE_POWER,
            "background_beta": v326.BASE_BETA,
            "path_beta_and_selector_threshold": "frozen from v329 outer folds",
            "correction_alpha": 1.0,
        },
        "fold_rows": fold_rows,
        "summary": summary,
        "decision_rule": {
            "score_delta": f">={v326.MIN_SCORE_DELTA}",
            "fold_wins": ">=4/5",
            "pas_delta": f">=-{v326.MAX_SPECTRAL_DEGRADATION}",
            "pdp_delta": f">=-{v326.MAX_SPECTRAL_DEGRADATION}",
            "passed": bool(passed),
        },
        "outputs": {"report": PATHS.relative(REPORT), "zip": None},
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "decision": report["decision_rule"]}, indent=2))


if __name__ == "__main__":
    main()
