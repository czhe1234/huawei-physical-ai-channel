#!/usr/bin/env python3
"""Audit whether v364 adds an independent, nested-valid online-kernel measurement."""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import pearsonr, spearmanr

import analyze_v479_online_spectral_target_inversion as v479
import analyze_v480_nested_online_pas_pdp_inversion as v480
import train_v485_direct_online_kernel_channel as v485
import train_v486_direct_online_kernel_continuation as v486
from project_paths import PATHS


VERSION_NUMBER = 487
VERSION = "v487_augmented_online_kernel"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
V364_CHANNEL = PATHS.cache_dir(364) / "Round1_Test_Channel.npy"
VERSIONS = (*v479.VERSIONS, (364, 0.514695, V364_CHANNEL))
PAS_WEIGHTS = np.linspace(0.0, 1.0, 11)
RIDGES = np.logspace(-8, -1, 15)


def nested_loo(
    pas_gram: np.ndarray,
    pdp_gram: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    predictions = np.empty(len(scores), dtype=np.float64)
    rows: list[dict[str, object]] = []
    for outer in range(len(scores)):
        train_ids = np.flatnonzero(np.arange(len(scores)) != outer)
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
                correlation = float(pearsonr(inner_prediction, train_scores).statistic)
                key = (rmse, -correlation)
                if best is None or key < best[0]:
                    best = (key, float(pas_weight), float(ridge), full_kernel)
        assert best is not None
        _, pas_weight, ridge, full_kernel = best
        alpha, intercept = v479.kernel_fit(
            full_kernel[np.ix_(train_ids, train_ids)], train_scores, ridge
        )
        prediction = float(full_kernel[outer, train_ids] @ alpha + intercept)
        predictions[outer] = prediction
        rows.append(
            {
                "held_out_version": int(VERSIONS[outer][0]),
                "truth": float(scores[outer]),
                "prediction": prediction,
                "selected_pas_weight": pas_weight,
                "selected_ridge": ridge,
                "inner_rmse": float(best[0][0]),
            }
        )
    return predictions, rows


def candidate_prediction(
    candidate: np.ndarray,
    arrays: list[np.ndarray],
    pas_weight: float,
    alpha: np.ndarray,
    intercept: float,
) -> dict[str, object]:
    pas_similarity, pdp_similarity = v480.similarities_to_versions(
        candidate, arrays, batch_size=1
    )
    similarity = pas_weight * pas_similarity + (1.0 - pas_weight) * pdp_similarity
    return {
        "predicted_online_score": float(similarity @ alpha + intercept),
        "pas_similarity": pas_similarity.tolist(),
        "pdp_similarity": pdp_similarity.tolist(),
    }


def main() -> None:
    old_report = json.loads(v479.REPORT.read_text(encoding="utf-8"))
    old_pas = np.asarray(old_report["pas_gram"], dtype=np.float64)
    old_pdp = np.asarray(old_report["pdp_gram"], dtype=np.float64)
    arrays = [np.load(path, mmap_mode="r") for _, _, path in VERSIONS]
    v364_pas, v364_pdp = v480.similarities_to_versions(
        arrays[-1], arrays, batch_size=1
    )
    count = len(VERSIONS)
    pas_gram = np.empty((count, count), dtype=np.float64)
    pdp_gram = np.empty((count, count), dtype=np.float64)
    pas_gram[:-1, :-1] = old_pas
    pdp_gram[:-1, :-1] = old_pdp
    pas_gram[-1] = v364_pas
    pas_gram[:, -1] = v364_pas
    pdp_gram[-1] = v364_pdp
    pdp_gram[:, -1] = v364_pdp

    scores = np.asarray([score for _, score, _ in VERSIONS], dtype=np.float64)
    nested_prediction, nested_rows = nested_loo(pas_gram, pdp_gram, scores)
    nested_metrics = {
        "rmse": float(np.sqrt(np.mean((nested_prediction - scores) ** 2))),
        "pearson": float(pearsonr(nested_prediction, scores).statistic),
        "spearman": float(spearmanr(nested_prediction, scores).statistic),
        "rows": nested_rows,
    }
    search_result = v479.choose_kernel(pas_gram, pdp_gram, scores)
    selected = search_result["best"]
    pas_weight = float(selected["pas_weight"])
    ridge = float(selected["ridge"])
    kernel = np.asarray(search_result["kernel"], dtype=np.float64)
    alpha, intercept = v479.kernel_fit(kernel, scores, ridge)
    eigenvalues = np.linalg.eigvalsh(kernel)
    effective_rank = float(
        np.sum(eigenvalues) ** 2 / np.maximum(np.sum(eigenvalues**2), 1e-20)
    )
    v485_prediction = candidate_prediction(
        np.load(v485.OUTPUT, mmap_mode="r"), arrays, pas_weight, alpha, intercept
    )
    v486_prediction = candidate_prediction(
        np.load(v486.OUTPUT, mmap_mode="r"), arrays, pas_weight, alpha, intercept
    )
    old_nested = json.loads(v480.REPORT.read_text(encoding="utf-8"))["nested_loo"]
    passed = (
        nested_metrics["rmse"] <= 0.025
        and nested_metrics["pearson"] >= 0.90
        and nested_metrics["spearman"] >= 0.85
    )
    report = {
        "method": VERSION,
        "status": "signal_passed" if passed else "analysis_only_rejected",
        "primary_hypothesis": (
            "The spectrally distinct new-metric v364 result adds an independent hidden-target "
            "measurement that improves the identifiability of the online kernel."
        ),
        "versions": [
            {"version": int(version), "score": float(score), "path": PATHS.relative(path)}
            for version, score, path in VERSIONS
        ],
        "old_nested_loo": {
            "rmse": float(old_nested["rmse"]),
            "pearson": float(old_nested["pearson"]),
            "spearman": float(old_nested["spearman"]),
        },
        "augmented_nested_loo": nested_metrics,
        "selected_kernel": selected,
        "full_fit": {"alpha": alpha.tolist(), "intercept": intercept},
        "kernel_diagnostics": {
            "condition_number": float(np.linalg.cond(kernel)),
            "effective_rank": effective_rank,
            "minimum_eigenvalue": float(eigenvalues[0]),
            "maximum_eigenvalue": float(eigenvalues[-1]),
        },
        "v364_similarity_row": {
            "pas": v364_pas.tolist(),
            "pdp": v364_pdp.tolist(),
        },
        "candidate_predictions": {"v485": v485_prediction, "v486": v486_prediction},
        "decision": passed,
        "report": PATHS.relative(REPORT),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "old_nested_loo": report["old_nested_loo"],
                "augmented_nested_loo": nested_metrics,
                "selected_kernel": {
                    "pas_weight": pas_weight,
                    "ridge": ridge,
                    "rmse": selected["rmse"],
                    "pearson": selected["pearson"],
                    "spearman": selected["spearman"],
                },
                "kernel_diagnostics": report["kernel_diagnostics"],
                "candidate_predictions": {
                    key: value["predicted_online_score"]
                    for key, value in report["candidate_predictions"].items()
                },
                "passed": passed,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
