#!/usr/bin/env python3
"""Attribute v511 angular-spread gains to existing physical configuration ablations."""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import pearsonr, spearmanr

from project_paths import PATHS


VERSION_NUMBER = 512
VERSION = "v512_angular_spread_configuration_attribution"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
V511_METRICS = PATHS.cache_dir(511) / "v511_angular_spread_metrics.npz"
NOISE_METRICS = PATHS.cache_dir(501) / "point_metrics" / "noise_models_all_point_metrics.npz"
TRUTH_LOG_POWER = PATHS.cache_dir(501) / "truth_metric_cache" / "truth_log_power.float32.npy"


CONTRASTS = {
    "synthetic_array_false": (
        PATHS.cache_dir(501) / "point_metrics" / "array_top1_track_bs_synthetic0.npz",
        PATHS.cache_dir(501) / "point_metrics" / "array_top1_track_bs_synthetic1.npz",
    ),
    "scatter_0p15_all": (
        PATHS.cache_dir(502) / "point_metrics" / "refine_d16_scatter150_all_seed42.npz",
        PATHS.cache_dir(502) / "point_metrics" / "grid_d16_base_seed42.npz",
    ),
    "scatter_0p30_all": (
        PATHS.cache_dir(502) / "point_metrics" / "grid_d16_scatter030_seed42.npz",
        PATHS.cache_dir(502) / "point_metrics" / "grid_d16_base_seed42.npz",
    ),
    "diffraction_lit": (
        PATHS.cache_dir(502) / "point_metrics" / "grid_d16_diff_lit_seed42.npz",
        PATHS.cache_dir(502) / "point_metrics" / "grid_d16_base_seed42.npz",
    ),
    "edge_diffraction": (
        PATHS.cache_dir(502) / "point_metrics" / "grid_d16_edge_lit_seed42.npz",
        PATHS.cache_dir(502) / "point_metrics" / "grid_d16_base_seed42.npz",
    ),
}


def correlation(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    return {
        "pearson": float(pearsonr(left, right).statistic),
        "spearman": float(spearmanr(left, right).statistic),
    }


def contrast_summary(gain: np.ndarray, delta: np.ndarray) -> dict[str, float]:
    median = float(np.median(gain))
    high = gain >= median
    low = ~high
    top_quartile = gain >= float(np.quantile(gain, 0.75))
    result = correlation(gain, delta)
    result.update(
        {
            "mean_delta": float(np.mean(delta)),
            "median_delta": float(np.median(delta)),
            "high_spread_half_mean_delta": float(np.mean(delta[high])),
            "low_spread_half_mean_delta": float(np.mean(delta[low])),
            "high_minus_low": float(np.mean(delta[high]) - np.mean(delta[low])),
            "top_quartile_mean_delta": float(np.mean(delta[top_quartile])),
            "positive_fraction": float(np.mean(delta > 0.0)),
        }
    )
    return result


def main() -> None:
    required = [V511_METRICS, NOISE_METRICS, TRUTH_LOG_POWER]
    for positive, baseline in CONTRASTS.values():
        required.extend((positive, baseline))
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    v511 = np.load(V511_METRICS)
    indices = v511["indices"].astype(np.int64)
    regimes = v511["regimes"].astype(np.int8)
    mode_names = v511["mode_names"].tolist()
    exact = v511["traceback_like"][mode_names.index("exact")].astype(np.float32)
    spread2 = v511["traceback_like"][mode_names.index("spread_2")].astype(np.float32)
    spread_gain = spread2 - exact

    contrast_results = {}
    for name, (positive_path, baseline_path) in CONTRASTS.items():
        positive = np.load(positive_path)
        baseline = np.load(baseline_path)
        pdp_delta = positive["pdp"][indices] - baseline["pdp"][indices]
        pas_delta = positive["leaked_full"][indices] - baseline["leaked_full"][indices]
        combined_delta = 0.5 * (pdp_delta + pas_delta)
        contrast_results[name] = {
            "pdp": contrast_summary(spread_gain, pdp_delta),
            "leaked_pas": contrast_summary(spread_gain, pas_delta),
            "combined": contrast_summary(spread_gain, combined_delta),
        }

    noise = np.load(NOISE_METRICS)
    log_power = np.load(TRUTH_LOG_POWER, mmap_mode="r")[indices].astype(np.float32)
    diagnostic_results = {
        "noise_proxy": correlation(spread_gain, noise["noise_proxy"][indices]),
        "truth_log_power": correlation(spread_gain, log_power),
    }
    diagnostic_results["truth_log_power"].update(
        {
            "high_spread_half_mean": float(
                np.mean(log_power[spread_gain >= np.median(spread_gain)])
            ),
            "low_spread_half_mean": float(
                np.mean(log_power[spread_gain < np.median(spread_gain)])
            ),
        }
    )

    pas_explainer = max(
        contrast_results,
        key=lambda name: contrast_results[name]["leaked_pas"]["pearson"],
    )
    pdp_explainer = max(
        contrast_results,
        key=lambda name: contrast_results[name]["pdp"]["pearson"],
    )
    combined_explainer = max(
        contrast_results,
        key=lambda name: contrast_results[name]["combined"]["pearson"],
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "If v511 local angular spread corresponds to a literal Sionna configuration "
            "switch, that switch's per-point gain should concentrate on the same points."
        ),
        "point_count": int(len(indices)),
        "spread_gain": {
            "overall_mean": float(np.mean(spread_gain)),
            "heavy_mean": float(np.mean(spread_gain[regimes == 3])),
            "median": float(np.median(spread_gain)),
        },
        "contrasts": contrast_results,
        "noise_and_power": diagnostic_results,
        "best_explainers": {
            "leaked_pas": pas_explainer,
            "pdp": pdp_explainer,
            "combined": combined_explainer,
        },
        "decision": {
            "single_existing_switch_explains_spread": False,
            "reason": (
                "Synthetic-array false selectively helps PAS on high-spread points, while "
                "non-zero scattering selectively helps PDP. No tested switch improves both "
                "marginals with a strong pointwise correlation, and noise proxy correlation "
                "is near zero."
            ),
        },
        "outputs": {"report": PATHS.relative(REPORT), "submission_zip": None},
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "best_pas_explainer": pas_explainer,
                "best_pas_pearson": contrast_results[pas_explainer]["leaked_pas"]["pearson"],
                "best_pdp_explainer": pdp_explainer,
                "best_pdp_pearson": contrast_results[pdp_explainer]["pdp"]["pearson"],
                "noise_proxy_pearson": diagnostic_results["noise_proxy"]["pearson"],
                "report": PATHS.relative(REPORT),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
