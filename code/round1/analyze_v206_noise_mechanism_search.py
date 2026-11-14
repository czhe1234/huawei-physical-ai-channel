#!/usr/bin/env python3
"""Compare fixed, relative, hybrid, and power-law noise-floor hypotheses."""

from __future__ import annotations

import csv
import json

import numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import HuberRegressor

from project_paths import PATHS


VERSION = "v206_noise_mechanism_search"
REPORT_DIR = PATHS.version_report_dir(206)
REPORT = REPORT_DIR / "v206_noise_mechanism_search.json"
POINT_CSV = PATHS.version_report_dir(168) / "v168_channel_awgn_point_metrics.csv"
SPLIT_CACHE = PATHS.cache_dir(206) / "v206_spatial_calibration_split.npz"
TINY = 1e-300


def load_metrics() -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(POINT_CSV.open("r", encoding="utf-8", newline="")))
    return {
        "power": np.asarray([float(row["raw_total_power"]) for row in rows], dtype=np.float64),
        "tail": np.asarray([float(row["tail_coeff_mean_power"]) for row in rows], dtype=np.float64),
        "fraction": np.asarray([float(row["floor_fraction_tail_mean"]) for row in rows], dtype=np.float64),
        "regime": np.asarray([row["regime"] for row in rows]),
    }


def error_summary(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    residual = prediction - target
    return {
        "log_rmse": float(np.sqrt(np.mean(residual**2))),
        "log_mae": float(np.mean(np.abs(residual))),
        "log_median_absolute_error": float(np.median(np.abs(residual))),
        "log_bias": float(np.mean(residual)),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    split = np.load(SPLIT_CACHE)
    power = metrics["power"]
    tail = metrics["tail"]
    valid = (power > 0.0) & (tail > 0.0) & np.isfinite(power) & np.isfinite(tail)
    calibration = np.intersect1d(split["calibration_indices"], np.flatnonzero(valid))
    validation = np.intersect1d(split["validation_indices"], np.flatnonzero(valid))
    log_power = np.full_like(power, np.nan)
    log_tail = np.full_like(tail, np.nan)
    log_power[valid] = np.log(power[valid])
    log_tail[valid] = np.log(tail[valid])

    cal_x = log_power[calibration]
    cal_y = log_tail[calibration]
    models: dict[str, dict[str, object]] = {}

    fixed_intercept = float(np.median(cal_y))
    relative_intercept = float(np.median(cal_y - cal_x))
    huber = HuberRegressor(epsilon=1.5, alpha=0.0, max_iter=1000).fit(cal_x[:, None], cal_y)

    initial_b = float(np.quantile(tail[calibration], 0.02))
    initial = np.asarray([relative_intercept, np.log(max(initial_b, TINY))])

    def hybrid_residual(parameters: np.ndarray) -> np.ndarray:
        log_relative, log_fixed = parameters
        prediction = np.logaddexp(log_relative + cal_x, log_fixed)
        return prediction - cal_y

    hybrid_fit = least_squares(hybrid_residual, initial, loss="soft_l1", f_scale=0.5, max_nfev=2000)
    hybrid_relative, hybrid_fixed = [float(value) for value in hybrid_fit.x]

    predictors = {
        "fixed_variance": {
            "parameters": {"log_fixed_floor": fixed_intercept},
            "predict": lambda x: np.full_like(x, fixed_intercept),
        },
        "relative_to_power": {
            "parameters": {"slope": 1.0, "log_relative_coefficient": relative_intercept},
            "predict": lambda x: x + relative_intercept,
        },
        "hybrid_relative_plus_floor": {
            "parameters": {
                "log_relative_coefficient": hybrid_relative,
                "relative_coefficient": float(np.exp(hybrid_relative)),
                "log_fixed_floor": hybrid_fixed,
                "fixed_floor": float(np.exp(hybrid_fixed)),
            },
            "predict": lambda x: np.logaddexp(hybrid_relative + x, hybrid_fixed),
        },
        "free_power_law": {
            "parameters": {
                "slope": float(huber.coef_[0]),
                "intercept": float(huber.intercept_),
            },
            "predict": lambda x: huber.predict(x[:, None]),
        },
    }

    for name, model in predictors.items():
        predict = model.pop("predict")
        models[name] = {
            **model,
            "calibration": error_summary(predict(log_power[calibration]), log_tail[calibration]),
            "validation": error_summary(predict(log_power[validation]), log_tail[validation]),
        }

    ranking = sorted(models, key=lambda name: models[name]["validation"]["log_mae"])
    fraction = metrics["fraction"][valid]
    fraction = fraction[np.isfinite(fraction) & (fraction > 0.0)]
    fraction_quantiles = np.quantile(fraction, [0.1, 0.5, 0.9])
    implied_snr = -10.0 * np.log10(fraction_quantiles)
    hybrid_fixed_fraction_at_power = {
        name: float(np.exp(hybrid_fixed) / (np.exp(hybrid_relative) * level + np.exp(hybrid_fixed)))
        for name, level in {
            "power_p10": float(np.quantile(power[valid], 0.1)),
            "power_p50": float(np.quantile(power[valid], 0.5)),
            "power_p90": float(np.quantile(power[valid], 0.9)),
        }.items()
    }

    slope = float(huber.coef_[0])
    conclusion = {
        "best_validation_model": ranking[0],
        "power_law_slope": slope,
        "relative_noise_supported": bool(abs(slope - 1.0) < 0.15),
        "fixed_awgn_only_rejected": bool(
            models["fixed_variance"]["validation"]["log_mae"]
            > 1.5 * models["relative_to_power"]["validation"]["log_mae"]
        ),
        "interpretation": (
            "A slope near one means the observable circular tail scales approximately with channel power. "
            "Because true diffuse paths remain in the tail, the inferred SNR is a lower bound and random noise "
            "must not be added to predictions."
        ),
    }
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "If official complex Gaussian noise is mainly relative to channel power, a unit-slope tail-floor "
            "model should generalize across disjoint spatial tiles much better than a fixed-variance model."
        ),
        "inputs": {
            "point_metrics": PATHS.relative(POINT_CSV),
            "spatial_split": PATHS.relative(SPLIT_CACHE),
        },
        "counts": {
            "calibration": int(len(calibration)),
            "validation": int(len(validation)),
            "exact_zero_excluded": int(np.sum(power == 0.0)),
        },
        "models": models,
        "validation_ranking": ranking,
        "hybrid_fixed_fraction_at_power": hybrid_fixed_fraction_at_power,
        "tail_floor_fraction_quantiles_p10_p50_p90": fraction_quantiles.tolist(),
        "implied_snr_lower_bound_db_p10_p50_p90": implied_snr.tolist(),
        "limitations": [
            "The AD tail includes true weak multipath, so it is only an upper bound on Gaussian noise power.",
            "No repeated clean/noisy observation exists at the same official point.",
            "The 16 exact-zero rows are a separate phenomenon and are excluded from Gaussian fitting.",
        ],
        "conclusion": conclusion,
        "outputs": {"report": PATHS.relative(REPORT)},
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
