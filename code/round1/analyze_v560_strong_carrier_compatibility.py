#!/usr/bin/env python3
"""Test whether the v445 path residual improves the stronger v543 OOF carrier."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import generate_v543_disagreement_conditioned_cell_fusion as v543
from project_paths import PATHS


VERSION_NUMBER = 560
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER) / "official"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
TRANSFER_SOURCE = CACHE_DIR / "v560_official_transfer_oof.npz"
STRONG_SOURCE = PATHS.cache_dir(543) / "v543_cellwise_fusion_oof.npz"
TRUTH_SOURCE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
OUTPUT = CACHE_DIR / "v560_strong_carrier_compatibility.npz"
REPORT = REPORT_DIR / "v560_strong_carrier_compatibility.json"

FLOOR = 1e-7
ACTION_CLIP = 6.0
PRIMARY_PROTOCOL = "spatial"
PRIMARY_METHOD = "legacy_full_k128"
METHODS = (
    "raw_geometry",
    "random_encoder",
    "legacy_full_k128",
    "clean_candidate_shuffle",
)
PROTOCOLS = ("random", "spatial")
MIN_WEIGHTED_GAIN = 1e-3
MIN_POSITIVE_FOLDS = 4
MIN_CONTROL_MARGIN = 5e-4


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), FLOOR)
    value /= np.maximum(
        np.sum(value, axis=tuple(range(1, value.ndim)), keepdims=True),
        1e-30,
    )
    return value.astype(np.float32)


def transfer_action(
    strong: np.ndarray, weak_base: np.ndarray, weak_model: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    log_action = np.clip(
        np.log(np.maximum(weak_model, FLOOR))
        - np.log(np.maximum(weak_base, FLOOR)),
        -ACTION_CLIP,
        ACTION_CLIP,
    ).astype(np.float32)
    candidate = normalize_rows(
        strong * np.exp(log_action).astype(np.float32)
    )
    return candidate, log_action


def means(metric: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    return {key: float(np.mean(value[mask])) for key, value in metric.items()}


def difference(
    model: dict[str, float], base: dict[str, float]
) -> dict[str, float]:
    return {key: float(model[key] - base[key]) for key in model}


def evaluate(
    truth: np.ndarray,
    strong: np.ndarray,
    candidate: np.ndarray,
    folds: np.ndarray,
    regimes: np.ndarray,
) -> dict[str, object]:
    base_metric = v543.spectrum_scores(truth, strong)
    model_metric = v543.spectrum_scores(truth, candidate)
    all_rows = np.ones(len(truth), dtype=bool)
    base = means(base_metric, all_rows)
    model = means(model_metric, all_rows)
    point_gain = model_metric["weighted"] - base_metric["weighted"]
    fold_rows = []
    for fold in np.unique(folds):
        local = folds == fold
        fold_base = means(base_metric, local)
        fold_model = means(model_metric, local)
        fold_rows.append(
            {
                "fold": int(fold),
                "count": int(np.sum(local)),
                "base": fold_base,
                "model": fold_model,
                "delta": difference(fold_model, fold_base),
            }
        )
    regime_rows = {}
    for regime in np.unique(regimes):
        local = regimes == regime
        regime_base = means(base_metric, local)
        regime_model = means(model_metric, local)
        regime_rows[str(regime)] = {
            "count": int(np.sum(local)),
            "base": regime_base,
            "model": regime_model,
            "delta": difference(regime_model, regime_base),
            "positive_point_fraction": float(np.mean(point_gain[local] > 0.0)),
        }
    return {
        "base": base,
        "model": model,
        "delta": difference(model, base),
        "positive_point_fraction": float(np.mean(point_gain > 0.0)),
        "positive_folds": int(
            np.sum([row["delta"]["weighted"] > 0.0 for row in fold_rows])
        ),
        "folds": fold_rows,
        "regimes": regime_rows,
    }


def action_summary(log_action: np.ndarray) -> dict[str, float]:
    absolute = np.abs(log_action).reshape(len(log_action), -1)
    return {
        "mean_abs": float(np.mean(absolute)),
        "point_median_abs_mean": float(np.mean(np.median(absolute, axis=1))),
        "p90_abs": float(np.quantile(absolute, 0.90)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "clipped_fraction": float(np.mean(absolute >= ACTION_CLIP - 1e-6)),
    }


def main() -> None:
    for path in (TRANSFER_SOURCE, STRONG_SOURCE, TRUTH_SOURCE):
        require(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    transfer = np.load(TRANSFER_SOURCE)
    strong_cache = np.load(STRONG_SOURCE)
    target_cache = np.load(TRUTH_SOURCE)
    valid_ids = transfer["valid_ids"].astype(np.int64)
    truth = strong_cache["truth"][valid_ids].reshape(-1, v543.HEIGHT, v543.WIDTH)
    independent_truth = target_cache["sh"][valid_ids].astype(np.float32)
    truth_absolute_error = np.abs(truth - independent_truth)
    truth_error = float(np.max(truth_absolute_error))
    truth_mean_error = float(np.mean(truth_absolute_error))
    truth_cosine = v543.row_cosine(truth, independent_truth)
    truth_min_cosine = float(np.min(truth_cosine))
    if truth_mean_error > 1e-6 or truth_min_cosine < 0.999999:
        raise ValueError(
            "v543 and v218 truth mismatch: "
            f"mean_abs={truth_mean_error}, min_cosine={truth_min_cosine}"
        )
    strong = normalize_rows(
        strong_cache["calibrated"][valid_ids].reshape(
            -1, v543.HEIGHT, v543.WIDTH
        )
    )
    folds = strong_cache["fold_id"][valid_ids].astype(np.int16)
    regimes = strong_cache["regimes"][valid_ids].astype("U8")

    results: dict[str, object] = {}
    candidates: dict[str, np.ndarray] = {}
    actions: dict[str, np.ndarray] = {}
    for protocol in PROTOCOLS:
        weak_base = normalize_rows(transfer[f"{protocol}_base"].astype(np.float32))
        protocol_result = {}
        for method in METHODS:
            weak_model = normalize_rows(
                transfer[f"{protocol}_{method}"].astype(np.float32)
            )
            candidate, log_action = transfer_action(strong, weak_base, weak_model)
            key = f"{protocol}_{method}"
            candidates[key] = candidate
            actions[key] = log_action
            protocol_result[method] = {
                **evaluate(truth, strong, candidate, folds, regimes),
                "action": action_summary(log_action),
            }
        results[protocol] = protocol_result

    primary = results[PRIMARY_PROTOCOL][PRIMARY_METHOD]
    control_names = ("raw_geometry", "random_encoder", "clean_candidate_shuffle")
    strongest_control_gain = max(
        float(results[PRIMARY_PROTOCOL][name]["delta"]["weighted"])
        for name in control_names
    )
    primary_gain = float(primary["delta"]["weighted"])
    control_margin = primary_gain - strongest_control_gain
    random_replication_gain = float(
        results["random"][PRIMARY_METHOD]["delta"]["weighted"]
    )
    passed = bool(
        primary_gain >= MIN_WEIGHTED_GAIN
        and int(primary["positive_folds"]) >= MIN_POSITIVE_FOLDS
        and control_margin >= MIN_CONTROL_MARGIN
        and random_replication_gain > 0.0
    )

    primary_key = f"{PRIMARY_PROTOCOL}_{PRIMARY_METHOD}"
    np.savez_compressed(
        OUTPUT,
        valid_ids=valid_ids,
        fold_id=folds,
        regimes=regimes,
        truth=truth.astype(np.float16),
        strong=strong.astype(np.float16),
        primary_candidate=candidates[primary_key].astype(np.float16),
        primary_log_action=actions[primary_key].astype(np.float16),
    )
    report = {
        "method": "v560_strong_carrier_compatibility",
        "status": "passed_generate_v561" if passed else "failed_stop_v445_deployment",
        "primary_hypothesis": (
            "The path residual learned by the full v445 Sionna-pretrained encoder remains "
            "useful after transfer from its weak KNN anchor to the stronger v543 OOF carrier."
        ),
        "protocol": {
            "primary": f"{PRIMARY_PROTOCOL}_{PRIMARY_METHOD}",
            "action": (
                "candidate=normalize(v543_calibrated * "
                "clip_ratio(v560_model/v560_base)); no truth-selected transfer strength"
            ),
            "evaluated_rows": int(len(valid_ids)),
            "saturated_and_invalid_rows": int(2000 - len(valid_ids)),
            "truth_source_max_abs_error": truth_error,
            "truth_source_mean_abs_error": truth_mean_error,
            "truth_source_min_point_cosine": truth_min_cosine,
        },
        "results": results,
        "decision": {
            "minimum_weighted_gain": MIN_WEIGHTED_GAIN,
            "minimum_positive_folds": MIN_POSITIVE_FOLDS,
            "minimum_control_margin": MIN_CONTROL_MARGIN,
            "primary_weighted_gain": primary_gain,
            "primary_positive_folds": int(primary["positive_folds"]),
            "strongest_control_gain": strongest_control_gain,
            "control_margin": float(control_margin),
            "random_protocol_replication_gain": random_replication_gain,
            "passed": passed,
            "next_step": (
                "train the full official-H adapter and create v561 with v559 fallback"
                if passed
                else "do not deploy the old v445 residual; redesign path-existence supervision"
            ),
        },
        "outputs": {
            "cache": PATHS.relative(OUTPUT),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
