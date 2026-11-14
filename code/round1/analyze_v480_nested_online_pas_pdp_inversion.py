#!/usr/bin/env python3
"""Nested-CV online inversion with jointly released PAS and linkwise PDP targets."""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import pearsonr, spearmanr

import analyze_v208_strict_historical_replay as v208
import analyze_v479_online_spectral_target_inversion as v479
import generate_v426_sionna_linkwise_pdp_completion_diagnostic as v426
import generate_v427_sionna_conventional_pas_dual_lock_diagnostic as v427
import train_v469_neural_sionna_calibrator as v469
from project_paths import PATHS


VERSION_NUMBER = 480
VERSION = "v480_nested_online_pas_pdp_inversion"
CACHE = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
PAS_TARGET = CACHE / "v480_inverted_pas_target.npy"
PDP_TARGET = CACHE / "v480_inverted_linkwise_pdp_target.npy"
OUTPUT = CACHE / "v480_Round1_Test_Channel.npy"
ETA_VALUES = np.asarray([0.0, 0.20, 0.40, 0.60, 0.80, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0], dtype=np.float64)
PAS_WEIGHTS = np.linspace(0.0, 1.0, 11)
RIDGES = np.logspace(-8, -1, 15)


def nested_loo(
    pas_gram: np.ndarray,
    pdp_gram: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    outer_prediction = np.empty(len(scores), dtype=np.float64)
    rows = []
    for outer in range(len(scores)):
        outer_fit = np.arange(len(scores)) != outer
        train_ids = np.flatnonzero(outer_fit)
        train_scores = scores[train_ids]
        best = None
        for pas_weight in PAS_WEIGHTS:
            full_kernel = pas_weight * pas_gram + (1.0 - pas_weight) * pdp_gram
            train_kernel = full_kernel[np.ix_(train_ids, train_ids)]
            for ridge in RIDGES:
                inner_prediction = v479.loo_predictions(
                    train_kernel, train_scores, float(ridge)
                )
                rmse = float(np.sqrt(np.mean((inner_prediction - train_scores) ** 2)))
                key = (rmse, -float(pearsonr(inner_prediction, train_scores).statistic))
                if best is None or key < best[0]:
                    best = (key, float(pas_weight), float(ridge), full_kernel)
        assert best is not None
        _, pas_weight, ridge, full_kernel = best
        alpha, intercept = v479.kernel_fit(
            full_kernel[np.ix_(train_ids, train_ids)], train_scores, ridge
        )
        prediction = float(full_kernel[outer, train_ids] @ alpha + intercept)
        outer_prediction[outer] = prediction
        rows.append(
            {
                "held_out_version": int(v479.VERSIONS[outer][0]),
                "truth": float(scores[outer]),
                "prediction": prediction,
                "selected_pas_weight": pas_weight,
                "selected_ridge": ridge,
                "inner_rmse": float(best[0][0]),
            }
        )
    return outer_prediction, rows


def grid_similarities(
    arrays: list[np.ndarray],
    anchor: np.ndarray,
    alpha: np.ndarray,
    batch_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(arrays)
    pas_dot = np.zeros((len(ETA_VALUES), count), dtype=np.float64)
    pdp_dot = np.zeros((len(ETA_VALUES), count), dtype=np.float64)
    pas_vectors = 0
    pdp_vectors = 0
    for start in range(0, 500, batch_size):
        end = min(start + batch_size, 500)
        source_pas = []
        source_pdp = []
        for value in arrays:
            pas, pdp, _ = v479.metric_features(
                np.asarray(value[start:end], dtype=np.complex64)
            )
            source_pas.append(pas)
            source_pdp.append(pdp)
        source_pas = np.stack(source_pas, axis=0)
        source_pdp = np.stack(source_pdp, axis=0)
        anchor_pas, anchor_pdp, _ = v479.metric_features(
            np.asarray(anchor[start:end], dtype=np.complex64)
        )
        pas_direction = np.tensordot(alpha, source_pas, axes=(0, 0))
        pdp_direction = np.tensordot(alpha, source_pdp, axes=(0, 0))
        pas_direction /= np.maximum(
            np.linalg.norm(pas_direction, axis=-1, keepdims=True), 1e-20
        )
        pdp_direction /= np.maximum(
            np.linalg.norm(pdp_direction, axis=-1, keepdims=True), 1e-20
        )
        for eta_id, eta in enumerate(ETA_VALUES):
            candidate_pas = v479.normalize_l2(
                np.maximum(anchor_pas + float(eta) * pas_direction, 0.0), -1
            )
            candidate_pdp = v479.normalize_l2(
                np.maximum(anchor_pdp + float(eta) * pdp_direction, 0.0), -1
            )
            pas_dot[eta_id] += np.einsum(
                "bqfd,kbqfd->k", candidate_pas, source_pas, optimize=True
            )
            pdp_dot[eta_id] += np.einsum(
                "bld,kbld->k", candidate_pdp, source_pdp, optimize=True
            )
        pas_vectors += int(np.prod(anchor_pas.shape[:-1]))
        pdp_vectors += int(np.prod(anchor_pdp.shape[:-1]))
        if end % 50 == 0:
            print(f"v480 grid {end}/500", flush=True)
    return pas_dot / pas_vectors, pdp_dot / pdp_vectors


def write_targets(
    arrays: list[np.ndarray],
    anchor: np.ndarray,
    alpha: np.ndarray,
    eta_pas: float,
    eta_pdp: float,
    batch_size: int = 1,
) -> None:
    pas_output = np.lib.format.open_memmap(
        PAS_TARGET,
        mode="w+",
        dtype=np.float16,
        shape=(500, 2, 16, 8, 4, 192),
    )
    pdp_output = np.lib.format.open_memmap(
        PDP_TARGET,
        mode="w+",
        dtype=np.float16,
        shape=(500, 256, 4, 192),
    )
    for start in range(0, 500, batch_size):
        end = min(start + batch_size, 500)
        source_pas = []
        source_pdp = []
        for value in arrays:
            pas, pdp, _ = v479.metric_features(
                np.asarray(value[start:end], dtype=np.complex64)
            )
            source_pas.append(pas)
            source_pdp.append(pdp)
        source_pas = np.stack(source_pas, axis=0)
        source_pdp = np.stack(source_pdp, axis=0)
        anchor_pas, anchor_pdp, _ = v479.metric_features(
            np.asarray(anchor[start:end], dtype=np.complex64)
        )
        pas_direction = np.tensordot(alpha, source_pas, axes=(0, 0))
        pdp_direction = np.tensordot(alpha, source_pdp, axes=(0, 0))
        pas_direction /= np.maximum(
            np.linalg.norm(pas_direction, axis=-1, keepdims=True), 1e-20
        )
        pdp_direction /= np.maximum(
            np.linalg.norm(pdp_direction, axis=-1, keepdims=True), 1e-20
        )
        desired_pas = np.maximum(anchor_pas + eta_pas * pas_direction, 0.0)
        desired_pas /= np.maximum(np.sum(desired_pas, axis=-1, keepdims=True), 1e-20)
        desired_pdp = np.maximum(anchor_pdp + eta_pdp * pdp_direction, 0.0)
        desired_pdp /= np.maximum(np.sum(desired_pdp, axis=-1, keepdims=True), 1e-20)
        pas_output[start:end] = desired_pas.reshape(
            end - start, 4, 192, 2, 16, 8
        ).transpose(0, 3, 4, 5, 1, 2).astype(np.float16)
        pdp_output[start:end] = desired_pdp.reshape(
            end - start, 256, 4, 192
        ).astype(np.float16)
    pas_output.flush()
    pdp_output.flush()


def generate_channel(batch_size: int = 1) -> None:
    anchor = np.load(v469.OUTPUT, mmap_mode="r")
    pas_target = np.load(PAS_TARGET, mmap_mode="r")
    pdp_target = np.load(PDP_TARGET, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=anchor.shape
    )
    for start in range(0, 500, batch_size):
        end = min(start + batch_size, 500)
        base_h = np.asarray(anchor[start:end], dtype=np.complex64)
        desired_pas = np.asarray(pas_target[start:end], dtype=np.float32)
        desired_pdp = np.asarray(pdp_target[start:end], dtype=np.float32)
        candidate = base_h.copy()
        for _ in range(5):
            candidate = v427.project_conventional_pas(candidate, desired_pas)
            candidate = v426.restore_desired_pdp(candidate, base_h, desired_pdp)
        output[start:end] = candidate
        if end % 50 == 0:
            output.flush()
            print(f"v480 projection {end}/500", flush=True)
    output.flush()


def similarities_to_versions(
    candidate: np.ndarray,
    arrays: list[np.ndarray],
    batch_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    pas = np.zeros(len(arrays), dtype=np.float64)
    pdp = np.zeros(len(arrays), dtype=np.float64)
    pas_count = 0
    pdp_count = 0
    for start in range(0, 500, batch_size):
        end = min(start + batch_size, 500)
        candidate_pas, candidate_pdp, _ = v479.metric_features(
            np.asarray(candidate[start:end], dtype=np.complex64)
        )
        for index, value in enumerate(arrays):
            source_pas, source_pdp, _ = v479.metric_features(
                np.asarray(value[start:end], dtype=np.complex64)
            )
            pas[index] += np.sum(candidate_pas * source_pas, dtype=np.float64)
            pdp[index] += np.sum(candidate_pdp * source_pdp, dtype=np.float64)
        pas_count += int(np.prod(candidate_pas.shape[:-1]))
        pdp_count += int(np.prod(candidate_pdp.shape[:-1]))
    return pas / pas_count, pdp / pdp_count


def main() -> None:
    source_report = json.loads(v479.REPORT.read_text(encoding="utf-8"))
    pas_gram = np.asarray(source_report["pas_gram"], dtype=np.float64)
    pdp_gram = np.asarray(source_report["pdp_gram"], dtype=np.float64)
    scores = np.asarray([item[1] for item in v479.VERSIONS], dtype=np.float64)
    nested_prediction, nested_rows = nested_loo(pas_gram, pdp_gram, scores)
    nested_metrics = {
        "rmse": float(np.sqrt(np.mean((nested_prediction - scores) ** 2))),
        "pearson": float(pearsonr(nested_prediction, scores).statistic),
        "spearman": float(spearmanr(nested_prediction, scores).statistic),
        "rows": nested_rows,
    }
    selected = source_report["selected_kernel"]
    pas_weight = float(selected["pas_weight"])
    kernel = pas_weight * pas_gram + (1.0 - pas_weight) * pdp_gram
    alpha, intercept = v479.kernel_fit(kernel, scores, float(selected["ridge"]))
    arrays = [np.load(path, mmap_mode="r") for _, _, path in v479.VERSIONS]
    anchor = np.load(v469.OUTPUT, mmap_mode="r")
    pas_sim, pdp_sim = grid_similarities(arrays, anchor, alpha)
    grid_rows = []
    best = None
    for pas_id, eta_pas in enumerate(ETA_VALUES):
        for pdp_id, eta_pdp in enumerate(ETA_VALUES):
            similarity = pas_weight * pas_sim[pas_id] + (1.0 - pas_weight) * pdp_sim[pdp_id]
            prediction = float(similarity @ alpha + intercept)
            row = {
                "eta_pas": float(eta_pas),
                "eta_pdp": float(eta_pdp),
                "predicted_online_score": prediction,
            }
            grid_rows.append(row)
            if best is None or prediction > best[0]:
                best = (prediction, float(eta_pas), float(eta_pdp))
    assert best is not None
    predicted_target_score, eta_pas, eta_pdp = best
    CACHE.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_targets(arrays, anchor, alpha, eta_pas, eta_pdp)
    generate_channel()
    candidate = np.load(OUTPUT, mmap_mode="r")
    actual_pas_sim, actual_pdp_sim = similarities_to_versions(candidate, arrays)
    actual_similarity = pas_weight * actual_pas_sim + (1.0 - pas_weight) * actual_pdp_sim
    actual_predicted_score = float(actual_similarity @ alpha + intercept)
    replay = v208.score_prediction(OUTPUT)
    baseline_replay = v208.score_prediction(v469.OUTPUT)
    nested_passed = (
        nested_metrics["pearson"] >= 0.75
        and nested_metrics["spearman"] >= 0.70
        and nested_metrics["rmse"] <= 0.025
    )
    candidate_passed = nested_passed and actual_predicted_score >= 0.75
    report = {
        "method": VERSION,
        "status": "candidate_signal_passed" if candidate_passed else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "The nested-validated hidden spectral projection can jointly release conventional "
            "PAS and linkwise PDP while preserving v469 complex phase and link energy."
        ),
        "nested_loo": nested_metrics,
        "inherited_full_kernel": selected,
        "full_fit": {"alpha": alpha.tolist(), "intercept": intercept},
        "candidate_grid": grid_rows,
        "selected": {
            "eta_pas": eta_pas,
            "eta_pdp": eta_pdp,
            "ideal_target_predicted_online_score": predicted_target_score,
            "realized_output_predicted_online_score": actual_predicted_score,
            "pas_similarity_to_versions": actual_pas_sim.tolist(),
            "pdp_similarity_to_versions": actual_pdp_sim.tolist(),
        },
        "replay": replay,
        "baseline_v469_replay": baseline_replay,
        "replay_delta_score": float(
            replay["score_traceback_leaked"] - baseline_replay["score_traceback_leaked"]
        ),
        "decision": {
            "nested_passed": nested_passed,
            "candidate_passed": candidate_passed,
        },
        "outputs": {
            "pas_target": PATHS.relative(PAS_TARGET),
            "pdp_target": PATHS.relative(PDP_TARGET),
            "channel": PATHS.relative(OUTPUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "nested_loo": nested_metrics,
        "selected": report["selected"],
        "replay_delta_score": report["replay_delta_score"],
        "decision": report["decision"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
