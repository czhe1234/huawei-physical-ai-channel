#!/usr/bin/env python3
"""Strict nested audit of multi-configuration Sionna joint-S-H stacking."""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize

import analyze_v432_sionna_link_energy_calibration as v432
import analyze_v434_sionna_joint_sh_calibration as v434
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 494
VERSION = "v494_sionna_configuration_stacking"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
OOF_CACHE = CACHE_DIR / f"{VERSION}_oof.npz"

OFFICIAL = PATHS.round1 / "Round1_Train_Channel.npy"
POSITIONS = PATHS.round1 / "Round1_Train_Pos.npy"
ALL_INTERACTION = PATHS.cache_dir(461) / "train_joint_sh.npy"
REGIMES = ("clear", "light", "medium", "heavy")
ALPHA_GRID = np.linspace(0.0, 1.5, 31, dtype=np.float32)
SOURCE_NAMES = (
    "robust_v434",
    "depth16_f6p0",
    "depth5_f2p6",
    "depth5_f6p0",
    "depth5_f6p7",
    "depth5_all_f6p0",
)


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    return value / np.maximum(value.sum(axis=1, keepdims=True), TINY)


def load_sources() -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    truth, _ = v434.load_sh(OFFICIAL)
    high, high_total = v434.load_sh(v434.HIGH)
    low_loaded = [v434.load_sh(path) for path in v434.LOWS]
    lows = [item[0] for item in low_loaded]
    low_totals = [item[1] for item in low_loaded]
    robust, point_quality, _ = v434.robust_source(
        high, high_total, lows, low_totals
    )
    all_interaction = normalize_rows(
        np.load(ALL_INTERACTION, mmap_mode="r").reshape(len(truth), -1)
    )
    raw = [robust, high, *lows, all_interaction]
    cleaned: list[np.ndarray] = []
    for source in raw:
        source = normalize_rows(source)
        valid = source.sum(axis=1) > 0.5
        source = source.copy()
        source[~valid] = robust[~valid]
        cleaned.append(normalize_rows(source))
    return truth.astype(np.float32), cleaned, point_quality.astype(np.float32)


def fit_convex_weights(
    truth: np.ndarray,
    candidates: np.ndarray,
    ids: np.ndarray,
) -> np.ndarray:
    count = candidates.shape[0]
    if len(ids) == 0:
        return np.full(count, 1.0 / count, dtype=np.float32)
    gram = np.empty((count, count), dtype=np.float64)
    target = np.empty(count, dtype=np.float64)
    truth_rows = np.asarray(truth[ids], dtype=np.float32)
    for left in range(count):
        left_rows = np.asarray(candidates[left, ids], dtype=np.float32)
        target[left] = np.sum(left_rows * truth_rows, dtype=np.float64)
        for right in range(left, count):
            value = np.sum(
                left_rows * np.asarray(candidates[right, ids], dtype=np.float32),
                dtype=np.float64,
            )
            gram[left, right] = value
            gram[right, left] = value
    scale = max(float(np.trace(gram)), 1.0)
    gram /= scale
    target /= scale

    def objective(weight: np.ndarray) -> float:
        return float(weight @ gram @ weight - 2.0 * target @ weight)

    def gradient(weight: np.ndarray) -> np.ndarray:
        return 2.0 * (gram @ weight - target)

    result = minimize(
        objective,
        np.full(count, 1.0 / count, dtype=np.float64),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda value: np.sum(value) - 1.0},
        options={"maxiter": 300, "ftol": 1e-12},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"convex weight fit failed: {result.message}")
    weight = np.maximum(result.x, 0.0)
    weight /= np.maximum(weight.sum(), 1e-30)
    return weight.astype(np.float32)


def weighted_target(
    candidates: np.ndarray,
    ids: np.ndarray,
    regimes: np.ndarray,
    weights: dict[str, np.ndarray],
) -> np.ndarray:
    output = np.empty((len(ids), candidates.shape[-1]), dtype=np.float32)
    for regime in REGIMES:
        local = regimes[ids] == regime
        if not np.any(local):
            continue
        rows = ids[local]
        output[local] = np.tensordot(
            weights[regime], candidates[:, rows], axes=(0, 0)
        ).astype(np.float32)
    return normalize_rows(output)


def select_alpha(
    truth: np.ndarray,
    baseline: np.ndarray,
    target: np.ndarray,
    quality: np.ndarray,
    ids: np.ndarray,
    regimes: np.ndarray,
) -> dict[str, float]:
    selected: dict[str, float] = {}
    for regime in REGIMES:
        local_ids = ids[regimes[ids] == regime]
        if len(local_ids) == 0:
            selected[regime] = 0.0
            continue
        base_rows = baseline[local_ids]
        target_rows = target[local_ids]
        truth_rows = truth[local_ids]
        local_quality = quality[local_ids]
        best_alpha = 0.0
        best_score = -np.inf
        for alpha in ALPHA_GRID:
            effective = np.clip(alpha * local_quality, 0.0, 1.0)
            candidate = normalize_rows(
                base_rows * (1.0 - effective[:, None])
                + target_rows * effective[:, None]
            )
            score = float(np.mean(v432.row_cosine(truth_rows, candidate)))
            if score > best_score:
                best_score = score
                best_alpha = float(alpha)
        selected[regime] = best_alpha
    return selected


def blend_target(
    baseline: np.ndarray,
    target: np.ndarray,
    quality: np.ndarray,
    ids: np.ndarray,
    regimes: np.ndarray,
    alpha: dict[str, float],
) -> np.ndarray:
    effective = np.asarray(
        [alpha[str(regimes[index])] for index in ids], dtype=np.float32
    ) * quality[ids]
    effective = np.clip(effective, 0.0, 1.0)
    return normalize_rows(
        baseline * (1.0 - effective[:, None]) + target * effective[:, None]
    )


def calibrated_sources(
    truth: np.ndarray,
    sources: list[np.ndarray],
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
    regimes: np.ndarray,
) -> np.ndarray:
    output = []
    for source in sources:
        ratio = v432.fit_ratio(truth, source, fit_ids, regimes)
        output.append(v432.apply_ratio(source, query_ids, regimes, ratio))
    return np.stack(output, axis=0).astype(np.float32)


def summarize(value: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, dtype=np.float64)
    return {
        "mean": float(np.mean(value)),
        "p10": float(np.quantile(value, 0.10)),
        "median": float(np.median(value)),
        "p90": float(np.quantile(value, 0.90)),
    }


def main() -> None:
    for path in (OFFICIAL, POSITIONS, ALL_INTERACTION, v434.OOF_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    truth, sources, quality = load_sources()
    positions = np.load(POSITIONS).astype(np.float32)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked = geometry["train_blocked"].astype(np.float32)
    regimes = regime_of(blocked).astype("U8")
    fold_id = np.load(v434.OOF_CACHE)["fold_id"].astype(np.int16)
    all_ids = np.arange(len(truth), dtype=np.int64)

    stacked_prediction = np.zeros_like(truth)
    old_prediction = np.zeros_like(truth)
    baseline_prediction = np.zeros_like(truth)
    medoid_prediction = np.zeros_like(truth)
    source_score = np.zeros((len(sources), len(truth)), dtype=np.float32)
    fold_rows: list[dict[str, object]] = []
    fold_weights: list[dict[str, list[float]]] = []
    fold_alphas: list[dict[str, float]] = []

    for outer in range(5):
        query_ids = np.flatnonzero(fold_id == outer)
        fit_ids = np.flatnonzero(fold_id != outer)
        inner_baseline = np.zeros_like(truth)
        inner_candidates = np.zeros(
            (len(sources), len(truth), truth.shape[1]), dtype=np.float32
        )
        for inner in range(5):
            if inner == outer:
                continue
            inner_query = np.flatnonzero(fold_id == inner)
            inner_fit = np.flatnonzero((fold_id != outer) & (fold_id != inner))
            inner_baseline[inner_query] = v432.neighbor_prediction(
                inner_fit,
                inner_query,
                positions,
                blocked,
                regimes,
                truth,
                k=48,
            )
            inner_candidates[:, inner_query] = calibrated_sources(
                truth, sources, inner_fit, inner_query, regimes
            )

        query_baseline = v432.neighbor_prediction(
            fit_ids, query_ids, positions, blocked, regimes, truth, k=48
        )
        query_candidates = calibrated_sources(
            truth, sources, fit_ids, query_ids, regimes
        )
        weights: dict[str, np.ndarray] = {}
        for regime in REGIMES:
            regime_fit = fit_ids[regimes[fit_ids] == regime]
            weights[regime] = fit_convex_weights(
                truth, inner_candidates, regime_fit
            )
        inner_target = np.zeros_like(truth)
        inner_target[fit_ids] = weighted_target(
            inner_candidates, fit_ids, regimes, weights
        )
        query_target = weighted_target(
            query_candidates, np.arange(len(query_ids)), regimes[query_ids], weights
        )
        stacked_alpha = select_alpha(
            truth,
            inner_baseline,
            inner_target,
            quality,
            fit_ids,
            regimes,
        )
        old_alpha = select_alpha(
            truth,
            inner_baseline,
            inner_candidates[0],
            quality,
            fit_ids,
            regimes,
        )
        stacked_prediction[query_ids] = blend_target(
            query_baseline,
            query_target,
            quality,
            np.arange(len(query_ids)),
            regimes[query_ids],
            stacked_alpha,
        )
        old_prediction[query_ids] = blend_target(
            query_baseline,
            query_candidates[0],
            quality,
            np.arange(len(query_ids)),
            regimes[query_ids],
            old_alpha,
        )
        query_source_predictions = np.empty(
            (len(sources), len(query_ids), truth.shape[1]), dtype=np.float32
        )
        source_alphas: dict[str, dict[str, float]] = {}
        for source_index, source_name in enumerate(SOURCE_NAMES):
            source_alpha = select_alpha(
                truth,
                inner_baseline,
                inner_candidates[source_index],
                quality,
                fit_ids,
                regimes,
            )
            source_alphas[source_name] = source_alpha
            query_source_predictions[source_index] = blend_target(
                query_baseline,
                query_candidates[source_index],
                quality,
                np.arange(len(query_ids)),
                regimes[query_ids],
                source_alpha,
            )
            source_score[source_index, query_ids] = v432.row_cosine(
                truth[query_ids], query_source_predictions[source_index]
            )
        target_norm = np.sqrt(
            np.sum(query_candidates * query_candidates, axis=2, dtype=np.float64)
        )
        agreement = np.zeros((len(sources), len(query_ids)), dtype=np.float64)
        for left in range(len(sources)):
            for right in range(len(sources)):
                if left == right:
                    continue
                numerator = np.sum(
                    query_candidates[left] * query_candidates[right],
                    axis=1,
                    dtype=np.float64,
                )
                agreement[left] += numerator / np.maximum(
                    target_norm[left] * target_norm[right], 1e-30
                )
        medoid_index = np.argmax(agreement, axis=0)
        medoid_prediction[query_ids] = query_source_predictions[
            medoid_index, np.arange(len(query_ids))
        ]
        baseline_prediction[query_ids] = normalize_rows(query_baseline)
        base_score = v432.row_cosine(truth[query_ids], baseline_prediction[query_ids])
        old_score = v432.row_cosine(truth[query_ids], old_prediction[query_ids])
        stacked_score = v432.row_cosine(
            truth[query_ids], stacked_prediction[query_ids]
        )
        fold_rows.append(
            {
                "fold": outer,
                "count": int(len(query_ids)),
                "baseline": float(np.mean(base_score)),
                "old_robust": float(np.mean(old_score)),
                "stacked": float(np.mean(stacked_score)),
                "gain_vs_old": float(np.mean(stacked_score - old_score)),
                "gain_vs_baseline": float(np.mean(stacked_score - base_score)),
                "stacked_alpha": stacked_alpha,
                "old_alpha": old_alpha,
                "source_alphas": source_alphas,
                "medoid_source_counts": {
                    SOURCE_NAMES[index]: int(np.sum(medoid_index == index))
                    for index in range(len(sources))
                },
            }
        )
        fold_weights.append(
            {regime: weights[regime].tolist() for regime in REGIMES}
        )
        fold_alphas.append(stacked_alpha)
        print(
            f"v494 outer {outer + 1}/5 gain={fold_rows[-1]['gain_vs_old']:+.6f}",
            flush=True,
        )

    baseline_score = v432.row_cosine(truth, baseline_prediction)
    old_score = v432.row_cosine(truth, old_prediction)
    stacked_score = v432.row_cosine(truth, stacked_prediction)
    medoid_score = v432.row_cosine(truth, medoid_prediction)
    oracle_score = np.max(source_score, axis=0)
    oracle_source = np.argmax(source_score, axis=0).astype(np.int16)
    gain = stacked_score - old_score
    mean_weights = {
        regime: np.mean(
            np.asarray([row[regime] for row in fold_weights], dtype=np.float32),
            axis=0,
        ).tolist()
        for regime in REGIMES
    }
    mean_alpha = {
        regime: float(np.mean([row[regime] for row in fold_alphas]))
        for regime in REGIMES
    }
    pass_gate = bool(
        float(np.mean(gain)) >= 0.005
        and sum(float(row["gain_vs_old"]) > 0.0 for row in fold_rows) >= 4
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "Separate official-train calibration and strict nested nonnegative stacking of "
            "six Sionna configurations may recover propagation support that the fixed v434 "
            "robust average discards."
        ),
        "evidence_boundary": {
            "outer_split": "five mutually exclusive spatial folds",
            "inner_selection": "remaining four folds only",
            "test_sionna_used": False,
            "online_feedback_used_for_parameters": False,
        },
        "sources": list(SOURCE_NAMES),
        "scores": {
            "baseline": summarize(baseline_score),
            "old_robust": summarize(old_score),
            "stacked": summarize(stacked_score),
            "observable_medoid": summarize(medoid_score),
            "per_point_source_oracle": summarize(oracle_score),
            "gain_vs_old": summarize(gain),
            "mean_gain_vs_old": float(np.mean(gain)),
            "medoid_gain_vs_old": float(np.mean(medoid_score - old_score)),
            "oracle_gain_vs_old": float(np.mean(oracle_score - old_score)),
            "fold_wins_vs_old": int(
                sum(float(row["gain_vs_old"]) > 0.0 for row in fold_rows)
            ),
        },
        "folds": fold_rows,
        "mean_weights": mean_weights,
        "mean_alpha": mean_alpha,
        "deployment_gate": {
            "required_gain": 0.005,
            "required_fold_wins": 4,
            "pass": pass_gate,
        },
        "outputs": {
            "oof": PATHS.relative(OOF_CACHE),
            "report": PATHS.relative(REPORT),
        },
    }
    np.savez_compressed(
        OOF_CACHE,
        fold_id=fold_id,
        truth=truth,
        baseline=baseline_prediction,
        old_robust=old_prediction,
        stacked=stacked_prediction,
        observable_medoid=medoid_prediction,
        baseline_score=baseline_score,
        old_score=old_score,
        stacked_score=stacked_score,
        medoid_score=medoid_score,
        source_score=source_score,
        oracle_score=oracle_score,
        oracle_source=oracle_source,
        gain_vs_old=gain,
        source_names=np.asarray(SOURCE_NAMES),
        mean_weights=np.asarray(
            [mean_weights[regime] for regime in REGIMES], dtype=np.float32
        ),
        mean_alpha=np.asarray([mean_alpha[regime] for regime in REGIMES], dtype=np.float32),
    )
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
