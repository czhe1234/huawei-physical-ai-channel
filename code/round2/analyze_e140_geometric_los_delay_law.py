#!/usr/bin/env python3
"""Verify the geometric LOS distance-over-c law in official training channels."""

from __future__ import annotations

import json
import time

import numpy as np

from analyze_v4_map_virtual_source_phase import (
    LIGHT_SPEED,
    METERS_PER_DELAY_BIN,
    S_SC,
    SUBCARRIER_SPACING,
)
from data import load_positions
from project_paths import PATHS
from run_path_component_transport import circular_delta, component_energy


VERSION = "Round2-E140-geometric-los-delay-law"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "geometric_los_delay_law"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_geometric_los_delay_law"
METRICS = ARTIFACT_DIR / "geometric_los_delay_metrics.npz"
REPORT = ANALYSIS_DIR / "geometric_los_delay_law.json"

LOS_LABELS = (
    PATHS.artifacts / "round2" / "geometric_los_inventory" / "geometric_los_labels.npz"
)
COMPONENT_CACHE = (
    PATHS.artifacts
    / "round2"
    / "v4_path_component_transport"
    / "train_components_k24_os2_ref2.npz"
)

SIGNIFICANT_ENERGY_FRACTION = 0.05
STRICT_DELAY_ERROR_BINS = 0.35
THRESHOLDS_BINS = (0.10, 0.20, 0.35, 0.50, 1.00)
TINY = 1e-30


def summarize(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def threshold_counts(error: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    return {
        f"within_{threshold:.2f}_bin": int(np.sum(mask & (error <= threshold)))
        for threshold in THRESHOLDS_BINS
    }


def main() -> None:
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    setup, positions, _, site, _ = load_positions()
    labels = np.load(LOS_LABELS, allow_pickle=False)
    geometric_los = labels["train_center_clear"].astype(bool)
    component = np.load(COMPONENT_CACHE, allow_pickle=False)
    coordinates = component["coordinates"].astype(np.float64)
    coefficients = component["coefficients"].astype(np.complex64)
    point_power = component["point_power"].astype(np.float64)

    energy = component_energy(coefficients)
    energy_fraction = energy / np.maximum(
        np.sum(energy, axis=1, keepdims=True), TINY
    )
    distance_m = np.linalg.norm(positions - setup.bs_positions[site], axis=1)
    expected_delay_s = distance_m / LIGHT_SPEED
    expected_delay_bin = distance_m / METERS_PER_DELAY_BIN
    delay_error = np.abs(
        circular_delta(coordinates[:, :, 2], expected_delay_bin[:, None], S_SC)
    )

    strongest_error = delay_error[:, 0]
    strongest_delta = circular_delta(
        coordinates[:, 0, 2], expected_delay_bin, S_SC
    )
    strongest_observed_unwrapped = expected_delay_bin + strongest_delta
    significant = energy_fraction >= SIGNIFICANT_ENERGY_FRACTION
    significant_error = np.where(significant, delay_error, np.inf)
    has_significant = np.any(significant, axis=1)
    best_significant_component_raw = np.argmin(significant_error, axis=1)
    best_significant_error = significant_error[
        np.arange(len(positions)), best_significant_component_raw
    ]
    best_significant_component = np.where(
        has_significant, best_significant_component_raw, -1
    )
    best_significant_error = np.where(
        has_significant, best_significant_error, S_SC / 2.0
    )

    nonzero = point_power > TINY
    observable_los = geometric_los & nonzero
    strongest_match = observable_los & (strongest_error <= STRICT_DELAY_ERROR_BINS)
    significant_match = observable_los & (
        best_significant_error <= STRICT_DELAY_ERROR_BINS
    )
    fit = np.polyfit(
        expected_delay_bin[strongest_match],
        strongest_observed_unwrapped[strongest_match],
        1,
    )
    correlation = np.corrcoef(
        expected_delay_bin[strongest_match],
        strongest_observed_unwrapped[strongest_match],
    )[0, 1]

    strongest_error_s = strongest_error / (S_SC * SUBCARRIER_SPACING)
    strongest_error_m = strongest_error * METERS_PER_DELAY_BIN
    exceptions = np.flatnonzero(observable_los & ~strongest_match)

    np.savez_compressed(
        METRICS,
        geometric_los=geometric_los,
        nonzero=nonzero,
        site=site.astype(np.int8),
        distance_m=distance_m.astype(np.float32),
        expected_delay_s=expected_delay_s.astype(np.float64),
        expected_delay_bin=expected_delay_bin.astype(np.float32),
        strongest_observed_delay_bin=coordinates[:, 0, 2].astype(np.float32),
        strongest_error_bin=strongest_error.astype(np.float32),
        strongest_energy_fraction=energy_fraction[:, 0].astype(np.float32),
        best_significant_component=best_significant_component.astype(np.int8),
        best_significant_error_bin=best_significant_error.astype(np.float32),
    )

    payload = {
        "experiment": VERSION,
        "uses_official_h": True,
        "uses_sionna": False,
        "method": {
            "channel_representation": "24 off-grid delay-angle components extracted from official train H",
            "expected_los_delay": "Euclidean serving-BS-to-UE distance divided by c",
            "significant_energy_fraction": SIGNIFICANT_ENERGY_FRACTION,
            "strict_delay_error_bins": STRICT_DELAY_ERROR_BINS,
            "meters_per_delay_bin": METERS_PER_DELAY_BIN,
            "nanoseconds_per_delay_bin": 1e9 / (S_SC * SUBCARRIER_SPACING),
        },
        "counts": {
            "geometric_los": int(np.sum(geometric_los)),
            "zero_channel_los": int(np.sum(geometric_los & ~nonzero)),
            "observable_los": int(np.sum(observable_los)),
            "strongest_path_match": int(np.sum(strongest_match)),
            "significant_path_match": int(np.sum(significant_match)),
            "strongest_exception_ids": exceptions.tolist(),
            "strongest_thresholds": threshold_counts(strongest_error, observable_los),
            "nlos_strongest_thresholds": threshold_counts(
                strongest_error, (~geometric_los) & nonzero
            ),
            "by_site": {
                str(site_value): {
                    "geometric_los": int(
                        np.sum(geometric_los & (site == site_value))
                    ),
                    "observable_los": int(
                        np.sum(observable_los & (site == site_value))
                    ),
                    "strongest_path_match": int(
                        np.sum(strongest_match & (site == site_value))
                    ),
                    "significant_path_match": int(
                        np.sum(significant_match & (site == site_value))
                    ),
                }
                for site_value in (0, 1)
            },
        },
        "strongest_path_fit": {
            "slope": float(fit[0]),
            "intercept_bins": float(fit[1]),
            "pearson": float(correlation),
            "absolute_error_bins": summarize(strongest_error[strongest_match]),
            "absolute_error_m": summarize(strongest_error_m[strongest_match]),
            "absolute_error_ns": summarize(1e9 * strongest_error_s[strongest_match]),
        },
        "exception_recovery": [
            {
                "point_id": int(point_id),
                "strongest_error_bins": float(strongest_error[point_id]),
                "best_significant_component": int(
                    best_significant_component[point_id]
                ),
                "best_significant_error_bins": float(
                    best_significant_error[point_id]
                ),
            }
            for point_id in exceptions
        ],
        "decision": {
            "distance_over_c_supported_for_all_observable_geometric_los": bool(
                np.all(significant_match[observable_los])
            ),
            "test_points_empirically_verifiable": False,
        },
        "outputs": {
            "metrics": PATHS.relative(METRICS),
            "report": PATHS.relative(REPORT),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
