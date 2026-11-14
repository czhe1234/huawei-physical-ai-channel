#!/usr/bin/env python3
"""Identify path-subspace perturbation fingerprints in official train H."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.optimize import nnls
from scipy.spatial import cKDTree

import analyze_v297_continuous_delay_component_transport as v297
import analyze_v300_path_gain_fingerprint_oracle as v300
import analyze_v304_inverse_scatterer_identifiability as v304
import analyze_v305_pointcloud_path_identity_match as v305
import analyze_v309_unmatched_effective_scatterers as v309
import analyze_v325_path_phase_transport_identifiability as v325
from project_paths import PATHS


VERSION = "v569_path_tangent_noise_inversion"
OUT_DIR = PATHS.cache_dir(569)
REPORT_DIR = PATHS.version_report_dir(569)
FIG_DIR = REPORT_DIR / "figures"
OBSERVABLE_CACHE = OUT_DIR / "v569_path_observables.npz"
REPORT = REPORT_DIR / "v569_path_tangent_noise_inversion.json"

CANDIDATE_CACHE = PATHS.cache_dir(263) / "v263_full_wall_reflection_candidates.npz"
GEOMETRY = PATHS.shared_features / "height_geometry_cache.npz"
TRAIN_POS = PATHS.round1 / "Round1_Train_Pos.npy"

BETA = 140.28
MAX_PAIR_DISTANCE = 20.0
VARIOGRAM_FIT_DISTANCE = 7.5
DISTANCE_EDGES = np.asarray(
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0001],
    dtype=np.float64,
)
POWER_GRID = (0.5, 1.0, 1.5, 2.0)
BOOTSTRAP_REPEATS = 300
INJECTION_REPEATS = 40
RNG_SEED = 569_001
TINY = 1e-30
UNIFORM_PHASE_SEMIVARIANCE = math.pi**2 / 6.0

QUALITY_RULES = {
    "core": {
        "mass": 0.002,
        "length_error": 0.50,
        "h_error": 1.00,
        "median_coherence": 0.65,
        "valid_channels": 4,
    },
    "strict": {
        "mass": 0.005,
        "length_error": 0.35,
        "h_error": 0.75,
        "median_coherence": 0.80,
        "valid_channels": 4,
    },
    "ultra": {
        "mass": 0.010,
        "length_error": 0.20,
        "h_error": 0.50,
        "median_coherence": 0.90,
        "valid_channels": 4,
    },
}

METRIC_LABELS = {
    "delay": "时延残差",
    "h_angle": "水平出射角残差",
    "log_amplitude": "路径对数幅度",
    "phase_geometry": "几何长度校正后相位",
    "phase_observed": "观测时延校正后相位",
}


def configure_chinese_font() -> None:
    families = {font.name for font in font_manager.fontManager.ttflist}
    selected = "DejaVu Sans"
    for family in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"):
        if family in families:
            selected = family
            break
    plt.rcParams.update(
        {
            "font.sans-serif": [selected, "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )


def summarize(values: np.ndarray) -> dict[str, float | int]:
    value = np.asarray(values, dtype=np.float64)
    value = value[np.isfinite(value)]
    if len(value) == 0:
        return {"count": 0}
    return {
        "count": int(len(value)),
        "mean": float(np.mean(value)),
        "std": float(np.std(value)),
        "min": float(np.min(value)),
        "p05": float(np.quantile(value, 0.05)),
        "p10": float(np.quantile(value, 0.10)),
        "median": float(np.median(value)),
        "p90": float(np.quantile(value, 0.90)),
        "p95": float(np.quantile(value, 0.95)),
        "max": float(np.max(value)),
    }


def circular_delta(left: np.ndarray, right: np.ndarray, period: float) -> np.ndarray:
    return (np.asarray(left) - np.asarray(right) + period / 2.0) % period - period / 2.0


def build_observables() -> dict[str, np.ndarray]:
    if OBSERVABLE_CACHE.is_file():
        cached = np.load(OBSERVABLE_CACHE, allow_pickle=False)
        return {key: cached[key] for key in cached.files}

    gain_data = v325.recover_center_gains()
    match = np.load(v305.MATCH_CACHE)
    candidates = np.load(CANDIDATE_CACHE)
    component = np.load(v297.COMPONENT_CACHE)
    fingerprint = np.load(v300.FINGERPRINT_CACHE)
    calibration = json.loads(v304.REPORT.read_text(encoding="utf-8"))["los_calibration"]

    good = (
        (match["length_error_bins"] <= v309.GOOD_LENGTH_ERROR)
        & (match["h_error_bins"] <= v309.GOOD_H_ERROR)
    )
    source_index = np.flatnonzero(good)
    point = match["point_id"][good].astype(np.int64)
    component_id = match["component_id"][good].astype(np.int64)
    candidate_id = match["candidate_id"][good].astype(np.int64)
    patch = match["patch_id"][good].astype(np.int64)
    mass = match["mass"][good].astype(np.float64)

    selected: dict[tuple[int, int], int] = {}
    for row, key in enumerate(zip(point.tolist(), patch.tolist())):
        previous = selected.get(key)
        if previous is None or mass[row] > mass[previous]:
            selected[key] = row
    keep = np.asarray(sorted(selected.values()), dtype=np.int64)
    source_index = source_index[keep]
    point = point[keep]
    component_id = component_id[keep]
    candidate_id = candidate_id[keep]
    patch = patch[keep]
    mass = mass[keep]

    candidate_length = candidates["path_length"][point, candidate_id].astype(np.float64)
    reflection_point = candidates["reflection_point"][point, candidate_id]
    expected_h = v305.expected_h_bin(reflection_point).astype(np.float64)

    delay = component["delay"][point, component_id].astype(np.float64)
    observed_length = (
        delay - float(calibration["intercept_bins"])
    ) / float(calibration["slope_bins_per_meter"])
    delay_residual_bins = (
        observed_length - candidate_length
    ) / float(v304.METERS_PER_BIN)

    observed_h_all = v304.horizontal_center(
        fingerprint["horizontal"].astype(np.float32)
    ).astype(np.float64)
    observed_h = observed_h_all[point, component_id]
    h_residual_bins = circular_delta(observed_h, expected_h, 16.0)

    gain = gain_data["center_gain"][point, component_id].reshape(len(point), -1)
    coherence = gain_data["center_coherence"][point, component_id].reshape(
        len(point), -1
    ).astype(np.float64)
    amplitude = np.abs(gain).astype(np.float64)
    log_amplitude = np.log(np.maximum(amplitude, 1e-30))
    phase_geometry = np.angle(
        gain * np.exp(1j * BETA * candidate_length[:, None])
    ).astype(np.float64)
    phase_observed = np.angle(
        gain * np.exp(1j * BETA * observed_length[:, None])
    ).astype(np.float64)
    channel_valid = (coherence >= 0.65) & (amplitude > 0.0)

    payload = {
        "source_index": source_index.astype(np.int32),
        "point": point.astype(np.int32),
        "component": component_id.astype(np.int16),
        "candidate": candidate_id.astype(np.int16),
        "patch": patch.astype(np.int32),
        "mass": mass.astype(np.float32),
        "length_error_bins": match["length_error_bins"][source_index].astype(np.float32),
        "h_error_bins": match["h_error_bins"][source_index].astype(np.float32),
        "candidate_length": candidate_length.astype(np.float32),
        "observed_length": observed_length.astype(np.float32),
        "delay_residual_bins": delay_residual_bins.astype(np.float32),
        "expected_h": expected_h.astype(np.float32),
        "observed_h": observed_h.astype(np.float32),
        "h_residual_bins": h_residual_bins.astype(np.float32),
        "log_amplitude": log_amplitude.astype(np.float32),
        "phase_geometry": phase_geometry.astype(np.float32),
        "phase_observed": phase_observed.astype(np.float32),
        "coherence": coherence.astype(np.float32),
        "channel_valid": channel_valid,
    }
    np.savez_compressed(OBSERVABLE_CACHE, **payload)
    return payload


def quality_masks(observables: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    coherence = observables["coherence"].astype(np.float64)
    channel_valid = observables["channel_valid"].astype(bool)
    median_coherence = np.median(coherence, axis=1)
    valid_count = np.sum(channel_valid, axis=1)
    result: dict[str, np.ndarray] = {}
    for name, rule in QUALITY_RULES.items():
        result[name] = (
            (observables["mass"] >= rule["mass"])
            & (observables["length_error_bins"] <= rule["length_error"])
            & (observables["h_error_bins"] <= rule["h_error"])
            & (median_coherence >= rule["median_coherence"])
            & (valid_count >= rule["valid_channels"])
        )
    return result


def build_pairs(
    observables: dict[str, np.ndarray],
    row_mask: np.ndarray,
    positions: np.ndarray,
) -> dict[str, np.ndarray]:
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    distance: list[np.ndarray] = []
    patch_out: list[np.ndarray] = []
    for patch_id in np.unique(observables["patch"][row_mask]):
        rows = np.flatnonzero(row_mask & (observables["patch"] == patch_id))
        if len(rows) < 2:
            continue
        local_positions = positions[observables["point"][rows]]
        local_pairs = np.asarray(
            sorted(cKDTree(local_positions).query_pairs(MAX_PAIR_DISTANCE)),
            dtype=np.int64,
        )
        if local_pairs.size == 0:
            continue
        local_pairs = local_pairs.reshape(-1, 2)
        left_rows = rows[local_pairs[:, 0]]
        right_rows = rows[local_pairs[:, 1]]
        left.append(left_rows)
        right.append(right_rows)
        distance.append(
            np.linalg.norm(
                positions[observables["point"][left_rows]]
                - positions[observables["point"][right_rows]],
                axis=1,
            )
        )
        patch_out.append(np.full(len(left_rows), patch_id, dtype=np.int32))
    if not left:
        empty = np.empty(0, dtype=np.float64)
        return {
            "left": np.empty(0, dtype=np.int64),
            "right": np.empty(0, dtype=np.int64),
            "distance": empty,
            "patch": np.empty(0, dtype=np.int32),
        }
    return {
        "left": np.concatenate(left),
        "right": np.concatenate(right),
        "distance": np.concatenate(distance).astype(np.float64),
        "patch": np.concatenate(patch_out),
    }


def pair_metric(
    name: str,
    observables: dict[str, np.ndarray],
    pairs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    left = pairs["left"]
    right = pairs["right"]
    row_weight = np.sqrt(
        observables["mass"][left].astype(np.float64)
        * observables["mass"][right].astype(np.float64)
    )
    if name == "delay":
        delta = (
            observables["delay_residual_bins"][left]
            - observables["delay_residual_bins"][right]
        ).astype(np.float64)
        return {
            "distance": pairs["distance"],
            "patch": pairs["patch"],
            "semivariance": 0.5 * delta**2,
            "weight": row_weight,
        }
    if name == "h_angle":
        delta = circular_delta(
            observables["h_residual_bins"][left],
            observables["h_residual_bins"][right],
            16.0,
        )
        return {
            "distance": pairs["distance"],
            "patch": pairs["patch"],
            "semivariance": 0.5 * delta**2,
            "weight": row_weight,
        }

    channel_valid = (
        observables["channel_valid"][left]
        & observables["channel_valid"][right]
    )
    channel_weight = row_weight[:, None] * np.sqrt(
        observables["coherence"][left].astype(np.float64)
        * observables["coherence"][right].astype(np.float64)
    )
    channel_weight[~channel_valid] = 0.0
    if name == "log_amplitude":
        delta = (
            observables["log_amplitude"][left]
            - observables["log_amplitude"][right]
        ).astype(np.float64)
    elif name in ("phase_geometry", "phase_observed"):
        delta = circular_delta(
            observables[name][left], observables[name][right], 2.0 * math.pi
        ).astype(np.float64)
    else:
        raise KeyError(name)
    distance = np.broadcast_to(pairs["distance"][:, None], delta.shape)
    patch = np.broadcast_to(pairs["patch"][:, None], delta.shape)
    valid = channel_weight > 0.0
    return {
        "distance": distance[valid],
        "patch": patch[valid],
        "semivariance": 0.5 * delta[valid] ** 2,
        "weight": channel_weight[valid],
    }


def binned_variogram(
    metric: dict[str, np.ndarray],
    weight_multiplier: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    distance = metric["distance"].astype(np.float64)
    value = metric["semivariance"].astype(np.float64)
    weight = metric["weight"].astype(np.float64).copy()
    if weight_multiplier is not None:
        weight *= weight_multiplier
    centers: list[float] = []
    means: list[float] = []
    counts: list[int] = []
    weight_sums: list[float] = []
    for lower, upper in zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:]):
        mask = (
            (distance >= lower)
            & (distance < upper)
            & np.isfinite(value)
            & np.isfinite(weight)
            & (weight > 0.0)
        )
        if not np.any(mask):
            centers.append((lower + upper) / 2.0)
            means.append(float("nan"))
            counts.append(0)
            weight_sums.append(0.0)
            continue
        local_value = value[mask]
        local_weight = weight[mask]
        upper_clip = float(np.quantile(local_value, 0.95))
        local_value = np.minimum(local_value, upper_clip)
        centers.append(float(np.average(distance[mask], weights=local_weight)))
        means.append(float(np.average(local_value, weights=local_weight)))
        counts.append(int(np.sum(mask)))
        weight_sums.append(float(np.sum(local_weight)))
    return {
        "distance": np.asarray(centers, dtype=np.float64),
        "semivariance": np.asarray(means, dtype=np.float64),
        "count": np.asarray(counts, dtype=np.int64),
        "weight_sum": np.asarray(weight_sums, dtype=np.float64),
    }


def fit_nugget_from_binned(binned: dict[str, np.ndarray]) -> dict[str, float]:
    valid = (
        np.isfinite(binned["semivariance"])
        & (binned["count"] >= 20)
        & (binned["distance"] <= VARIOGRAM_FIT_DISTANCE)
    )
    if np.sum(valid) < 3:
        return {
            "nugget": float("nan"),
            "slope": float("nan"),
            "power": float("nan"),
            "rmse": float("nan"),
            "fit_bins": int(np.sum(valid)),
        }
    distance = binned["distance"][valid]
    target = binned["semivariance"][valid]
    fit_weight = np.sqrt(binned["count"][valid].astype(np.float64))
    fit_weight /= max(float(np.mean(fit_weight)), TINY)
    best: dict[str, float] | None = None
    for power in POWER_GRID:
        design = np.column_stack([np.ones(len(distance)), distance**power])
        coefficient = nnls(design * fit_weight[:, None], target * fit_weight)[0]
        predicted = design @ coefficient
        rmse = float(
            np.sqrt(np.average((target - predicted) ** 2, weights=fit_weight**2))
        )
        row = {
            "nugget": float(coefficient[0]),
            "slope": float(coefficient[1]),
            "power": float(power),
            "rmse": rmse,
            "fit_bins": int(np.sum(valid)),
        }
        if best is None or row["rmse"] < best["rmse"]:
            best = row
    assert best is not None
    return best


def bootstrap_nugget(
    metric: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    patches = np.unique(metric["patch"])
    output = np.full(BOOTSTRAP_REPEATS, np.nan, dtype=np.float64)
    for repeat in range(BOOTSTRAP_REPEATS):
        sampled = rng.choice(patches, size=len(patches), replace=True)
        unique, count = np.unique(sampled, return_counts=True)
        multiplicity = dict(zip(unique.tolist(), count.tolist()))
        multiplier = np.asarray(
            [multiplicity.get(int(patch), 0) for patch in metric["patch"]],
            dtype=np.float64,
        )
        fit = fit_nugget_from_binned(binned_variogram(metric, multiplier))
        output[repeat] = fit["nugget"]
    return output


def metric_units(name: str, nugget: float) -> dict[str, float | str]:
    sigma = math.sqrt(max(float(nugget), 0.0))
    if name == "delay":
        return {
            "sigma": sigma,
            "unit": "delay_bin",
            "sigma_meters": sigma * float(v304.METERS_PER_BIN),
        }
    if name == "h_angle":
        return {
            "sigma": sigma,
            "unit": "horizontal_angle_bin",
            "broadside_equivalent_degrees": math.degrees(sigma / 8.0),
        }
    if name == "log_amplitude":
        return {
            "sigma": sigma,
            "unit": "log_amplitude",
            "equivalent_relative_std": math.sqrt(max(math.exp(sigma**2) - 1.0, 0.0)),
        }
    return {
        "sigma": sigma,
        "unit": "radian",
        "sigma_degrees": math.degrees(sigma),
    }


def estimate_metrics(
    observables: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    positions: np.ndarray,
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
]:
    rng = np.random.default_rng(RNG_SEED)
    report: dict[str, dict[str, object]] = {}
    pair_cache: dict[str, dict[str, np.ndarray]] = {}
    variograms: dict[str, dict[str, np.ndarray]] = {}
    for quality, mask in masks.items():
        pairs = build_pairs(observables, mask, positions)
        pair_cache[quality] = pairs
        quality_report: dict[str, object] = {
            "row_count": int(np.sum(mask)),
            "point_count": int(len(np.unique(observables["point"][mask]))),
            "patch_count": int(len(np.unique(observables["patch"][mask]))),
            "pair_count": int(len(pairs["left"])),
            "pair_distance": summarize(pairs["distance"]),
            "metrics": {},
        }
        for name in METRIC_LABELS:
            metric = pair_metric(name, observables, pairs)
            binned = binned_variogram(metric)
            fit = fit_nugget_from_binned(binned)
            bootstrap = bootstrap_nugget(metric, rng)
            finite_bootstrap = bootstrap[np.isfinite(bootstrap)]
            row = {
                "pair_observation_count": int(len(metric["semivariance"])),
                "fit": fit,
                "bootstrap": summarize(finite_bootstrap),
                "physical_scale": metric_units(name, fit["nugget"]),
                "positive_bootstrap_fraction": float(
                    np.mean(finite_bootstrap > 0.0)
                )
                if len(finite_bootstrap)
                else float("nan"),
            }
            quality_report["metrics"][name] = row
            variograms[f"{quality}:{name}"] = binned
        report[quality] = quality_report
    return report, pair_cache, variograms


def inject_metric_values(
    name: str,
    observables: dict[str, np.ndarray],
    rng: np.random.Generator,
    sigma: float,
) -> dict[str, np.ndarray]:
    output = {key: value for key, value in observables.items()}
    if name == "delay":
        output[name + "_residual_bins"] = (
            observables["delay_residual_bins"]
            + rng.normal(0.0, sigma, len(observables["point"]))
        ).astype(np.float32)
    elif name == "h_angle":
        output["h_residual_bins"] = circular_delta(
            observables["h_residual_bins"]
            + rng.normal(0.0, sigma, len(observables["point"])),
            0.0,
            16.0,
        ).astype(np.float32)
    elif name == "log_amplitude":
        common = rng.normal(0.0, sigma, (len(observables["point"]), 1))
        output["log_amplitude"] = (
            observables["log_amplitude"] + common
        ).astype(np.float32)
    elif name in ("phase_geometry", "phase_observed"):
        common = rng.normal(0.0, sigma, (len(observables["point"]), 1))
        output[name] = np.angle(
            np.exp(1j * (observables[name] + common))
        ).astype(np.float32)
    else:
        raise KeyError(name)
    return output


def smooth_control_observables(
    observables: dict[str, np.ndarray],
    positions: np.ndarray,
) -> dict[str, np.ndarray]:
    output = {key: np.array(value, copy=True) for key, value in observables.items()}
    row_xy = positions[observables["point"]]
    centered = np.empty_like(row_xy, dtype=np.float64)
    for patch_id in np.unique(observables["patch"]):
        mask = observables["patch"] == patch_id
        centered[mask] = row_xy[mask] - np.mean(row_xy[mask], axis=0)
    x = centered[:, 0]
    y = centered[:, 1]
    smooth = 0.01 * x + 0.006 * y + 0.0001 * (x**2 + y**2)
    channel_offset = np.linspace(-0.03, 0.03, 8, dtype=np.float64)[None, :]

    output["delay_residual_bins"] = (0.2 * smooth).astype(np.float32)
    output["h_residual_bins"] = (0.4 * smooth).astype(np.float32)
    output["log_amplitude"] = (smooth[:, None] + channel_offset).astype(np.float32)
    phase = 2.0 * smooth[:, None] + channel_offset
    output["phase_geometry"] = np.angle(np.exp(1j * phase)).astype(np.float32)
    output["phase_observed"] = output["phase_geometry"].copy()
    return output


def injection_controls(
    observables: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    positions: np.ndarray,
    base_report: dict[str, dict[str, object]],
    *,
    use_smooth_control: bool,
) -> dict[str, object]:
    rng = np.random.default_rng(RNG_SEED + 1000)
    quality = "strict"
    mask = masks[quality]
    pairs = build_pairs(observables, mask, positions)
    baseline = (
        smooth_control_observables(observables, positions)
        if use_smooth_control
        else observables
    )
    sigma_grid = {
        "delay": (0.01, 0.03, 0.10),
        "h_angle": (0.02, 0.05, 0.15),
        "log_amplitude": (0.02, 0.05, 0.10),
        "phase_geometry": (0.02, 0.05, 0.10),
        "phase_observed": (0.02, 0.05, 0.10),
    }
    result: dict[str, object] = {}
    for name, sigmas in sigma_grid.items():
        if use_smooth_control:
            base_nugget = float(
                fit_nugget_from_binned(
                    binned_variogram(pair_metric(name, baseline, pairs))
                )["nugget"]
            )
        else:
            base_nugget = float(
                base_report[quality]["metrics"][name]["fit"]["nugget"]
            )
        rows = []
        for sigma in sigmas:
            recovered = []
            for _ in range(INJECTION_REPEATS):
                injected = inject_metric_values(name, baseline, rng, sigma)
                fit = fit_nugget_from_binned(
                    binned_variogram(pair_metric(name, injected, pairs))
                )
                recovered.append((fit["nugget"] - base_nugget) / (sigma**2))
            rows.append(
                {
                    "sigma": float(sigma),
                    "target_variance": float(sigma**2),
                    "baseline_nugget": base_nugget,
                    "increment_recovery_ratio": summarize(np.asarray(recovered)),
                }
            )
        result[name] = rows
    return result


def injection_detection_limits(injection: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, rows in injection.items():
        accepted = []
        for row in rows:
            summary = row["increment_recovery_ratio"]
            median = float(summary.get("median", np.nan))
            p10 = float(summary.get("p10", np.nan))
            if np.isfinite(median) and 0.6 <= median <= 1.3 and p10 > 0.0:
                accepted.append(float(row["sigma"]))
        sigma = min(accepted) if accepted else float("nan")
        result[name] = {
            "minimum_reliably_detected_sigma": sigma,
            "physical_scale": metric_units(name, sigma**2)
            if np.isfinite(sigma)
            else None,
            "criterion": "median recovery in [0.6, 1.3] and p10 > 0",
        }
    return result


def channel_structure(
    observables: dict[str, np.ndarray],
    pairs: dict[str, np.ndarray],
) -> dict[str, object]:
    rng = np.random.default_rng(RNG_SEED + 2000)
    close = pairs["distance"] <= 3.0
    left = pairs["left"][close]
    right = pairs["right"][close]
    result: dict[str, object] = {}
    for name in ("log_amplitude", "phase_geometry", "phase_observed"):
        valid = (
            observables["channel_valid"][left]
            & observables["channel_valid"][right]
        )
        complete = np.all(valid, axis=1)
        if name == "log_amplitude":
            delta = (
                observables[name][left] - observables[name][right]
            ).astype(np.float64)
        else:
            delta = circular_delta(
                observables[name][left], observables[name][right], 2.0 * math.pi
            ).astype(np.float64)
        delta = delta[complete]
        if len(delta) < 10:
            result[name] = {"count": int(len(delta))}
            continue
        delta -= np.mean(delta, axis=0, keepdims=True)
        covariance = np.cov(delta, rowvar=False)
        eigenvalue = np.linalg.eigvalsh(covariance)
        correlation = np.corrcoef(delta, rowvar=False)
        off_diagonal = correlation[~np.eye(correlation.shape[0], dtype=bool)]
        total_energy = np.mean(delta**2, axis=1)
        common_energy = np.mean(delta, axis=1) ** 2
        shuffled_pc1 = []
        for _ in range(100):
            shuffled = np.column_stack(
                [delta[rng.permutation(len(delta)), column] for column in range(delta.shape[1])]
            )
            shuffled_eigenvalue = np.linalg.eigvalsh(np.cov(shuffled, rowvar=False))
            shuffled_pc1.append(
                float(shuffled_eigenvalue[-1] / np.maximum(np.sum(shuffled_eigenvalue), TINY))
            )
        result[name] = {
            "count": int(len(delta)),
            "pc1_fraction": float(eigenvalue[-1] / np.maximum(np.sum(eigenvalue), TINY)),
            "mean_off_diagonal_correlation": float(np.mean(off_diagonal)),
            "common_mode_energy_fraction": float(
                np.mean(common_energy / np.maximum(total_energy, TINY))
            ),
            "column_shuffle_pc1_fraction": summarize(np.asarray(shuffled_pc1)),
        }
    return result


def mechanism_assessment(
    metric_report: dict[str, dict[str, object]],
    official_injection: dict[str, object],
    smooth_injection: dict[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in METRIC_LABELS:
        nuggets = np.asarray(
            [metric_report[q]["metrics"][name]["fit"]["nugget"] for q in QUALITY_RULES],
            dtype=np.float64,
        )
        bootstrap_p05 = np.asarray(
            [metric_report[q]["metrics"][name]["bootstrap"].get("p05", np.nan) for q in QUALITY_RULES],
            dtype=np.float64,
        )
        official_recovery = np.asarray(
            [
                row["increment_recovery_ratio"].get("median", np.nan)
                for row in official_injection[name]
            ],
            dtype=np.float64,
        )
        smooth_recovery = np.asarray(
            [
                row["increment_recovery_ratio"].get("median", np.nan)
                for row in smooth_injection[name]
            ],
            dtype=np.float64,
        )
        finite = nuggets[np.isfinite(nuggets) & (nuggets > 0.0)]
        stability_ratio = (
            float(np.max(finite) / np.min(finite)) if len(finite) >= 2 else None
        )
        positive_all = bool(np.all(bootstrap_p05 > 0.0))
        calibrated = bool(
            np.all(np.isfinite(smooth_recovery[-2:]))
            and np.all((smooth_recovery[-2:] >= 0.6) & (smooth_recovery[-2:] <= 1.3))
        )
        stable = bool(stability_ratio is not None and stability_ratio <= 2.0)
        observable_nugget = positive_all and calibrated and stable
        result[name] = {
            "nuggets_core_strict_ultra": nuggets.tolist(),
            "bootstrap_p05_core_strict_ultra": bootstrap_p05.tolist(),
            "quality_stability_ratio": stability_ratio,
            "official_baseline_injection_recovery_medians": official_recovery.tolist(),
            "smooth_control_injection_recovery_medians": smooth_recovery.tolist(),
            "positive_in_all_quality_bootstraps": positive_all,
            "estimator_calibrated_on_smooth_control": calibrated,
            "quality_stable": stable,
            "observable_zero_distance_nugget": observable_nugget,
            "official_noise_uniquely_identified": False,
            "strict_interpretation": (
                "A stable zero-distance nugget is observable, but without repeated official "
                "clean/noisy realizations it remains an upper bound mixing official path-state "
                "randomness, path-identity errors, and unresolved deterministic path transitions."
                if observable_nugget
                else "No path-randomness claim passes all stability and calibration controls."
            ),
        }
    return result


def phase_diagnostics(
    metric_report: dict[str, dict[str, object]],
) -> dict[str, object]:
    observed = np.asarray(
        [
            metric_report[quality]["metrics"]["phase_observed"]["fit"]["nugget"]
            for quality in QUALITY_RULES
        ],
        dtype=np.float64,
    )
    geometry = np.asarray(
        [
            metric_report[quality]["metrics"]["phase_geometry"]["fit"]["nugget"]
            for quality in QUALITY_RULES
        ],
        dtype=np.float64,
    )
    return {
        "uniform_random_phase_semivariance": UNIFORM_PHASE_SEMIVARIANCE,
        "phase_observed_nugget_to_uniform_ratio": (
            observed / UNIFORM_PHASE_SEMIVARIANCE
        ).tolist(),
        "phase_geometry_nugget_to_uniform_ratio": (
            geometry / UNIFORM_PHASE_SEMIVARIANCE
        ).tolist(),
        "interpretation": (
            "The observed-delay correction leaves an almost distance-independent phase "
            "semivariance near pi^2/6, the value for two independent uniform phases. The "
            "finite-bandwidth delay estimate therefore randomizes carrier phase and cannot be "
            "used as evidence of official phase noise. Geometry-length correction preserves "
            "substantially more phase structure."
        ),
    }


def make_figures(
    metric_report: dict[str, dict[str, object]],
    variograms: dict[str, dict[str, np.ndarray]],
    official_injection: dict[str, object],
    smooth_injection: dict[str, object],
    channel: dict[str, object],
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    quality_labels = {"core": "核心", "strict": "严格", "ultra": "极严格"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for quality, label in quality_labels.items():
        distance = metric_report[quality]["pair_distance"]
        ax.bar(
            label,
            metric_report[quality]["pair_count"],
            label=f"中位距离 {distance.get('median', float('nan')):.1f}m",
        )
    ax.set_ylabel("同墙面路径配对数量")
    ax.set_title("三档路径质量门槛仍保留足够空间配对")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "00_路径质量门槛与配对数量.png", dpi=180)
    plt.close(fig)

    for group, names, filename, title in (
        (
            "geometry",
            ("delay", "h_angle"),
            "01_时延与角度半方差曲线.png",
            "几何路径参数的空间半方差",
        ),
        (
            "gain",
            ("log_amplitude", "phase_geometry", "phase_observed"),
            "02_幅度与相位半方差曲线.png",
            "路径复增益的空间半方差",
        ),
    ):
        del group
        fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5), squeeze=False)
        for axis, name in zip(axes[0], names):
            for quality, label in quality_labels.items():
                binned = variograms[f"{quality}:{name}"]
                axis.plot(
                    binned["distance"],
                    binned["semivariance"],
                    "o-",
                    label=label,
                )
                nugget = metric_report[quality]["metrics"][name]["fit"]["nugget"]
                axis.axhline(nugget, linestyle="--", alpha=0.35)
            axis.set_title(METRIC_LABELS[name])
            axis.set_xlabel("同路径族位置距离（米）")
            axis.set_ylabel("半方差")
            axis.grid(True, alpha=0.2)
        axes[0, 0].legend()
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, dpi=180)
        plt.close(fig)

    names = list(METRIC_LABELS)
    x = np.arange(len(names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(13, 6))
    for offset, quality in enumerate(quality_labels):
        value = [
            metric_report[quality]["metrics"][name]["fit"]["nugget"] for name in names
        ]
        ax.bar(x + (offset - 1) * width, value, width=width, label=quality_labels[quality])
    ax.set_yscale("log")
    ax.set_xticks(x, [METRIC_LABELS[name] for name in names], rotation=15, ha="right")
    ax.set_ylabel("零距离半方差截距")
    ax.set_title("质量门槛收紧后的随机截距稳定性")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_四类路径扰动截距对比.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.8), squeeze=False)
    for axis, name in zip(axes[0], names):
        sigma = [row["sigma"] for row in official_injection[name]]
        official_recovery = [
            row["increment_recovery_ratio"]["median"]
            for row in official_injection[name]
        ]
        smooth_recovery = [
            row["increment_recovery_ratio"]["median"]
            for row in smooth_injection[name]
        ]
        axis.plot(sigma, official_recovery, "o-", label="官方观测基线")
        axis.plot(sigma, smooth_recovery, "s--", label="平滑无噪声控制")
        axis.axhline(1.0, color="black", linestyle="--")
        axis.axhspan(0.6, 1.3, color="green", alpha=0.1)
        axis.set_title(METRIC_LABELS[name])
        axis.set_xlabel("注入标准差")
        axis.set_ylabel("新增方差恢复比")
        axis.grid(True, alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("人工路径扰动的零距离截距恢复校准")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_人工路径扰动恢复校准.png", dpi=180)
    plt.close(fig)

    channel_names = ("log_amplitude", "phase_geometry", "phase_observed")
    fig, ax = plt.subplots(figsize=(10, 6))
    observed = [channel[name].get("pc1_fraction", np.nan) for name in channel_names]
    shuffled = [
        channel[name].get("column_shuffle_pc1_fraction", {}).get("median", np.nan)
        for name in channel_names
    ]
    index = np.arange(len(channel_names))
    ax.bar(index - 0.18, observed, width=0.36, label="真实通道对应")
    ax.bar(index + 0.18, shuffled, width=0.36, label="逐通道打乱控制")
    ax.set_xticks(index, [METRIC_LABELS[name] for name in channel_names])
    ax.set_ylabel("8通道残差第一主成分占比")
    ax.set_title("近距离路径残差是否属于公共路径模态")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_路径扰动跨端口公共模态.png", dpi=180)
    plt.close(fig)


def main() -> None:
    configure_chinese_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    observables = build_observables()
    masks = quality_masks(observables)
    positions = np.load(TRAIN_POS).astype(np.float64)[:, :2]
    metric_report, pair_cache, variograms = estimate_metrics(observables, masks, positions)
    official_injection = injection_controls(
        observables,
        masks,
        positions,
        metric_report,
        use_smooth_control=False,
    )
    smooth_injection = injection_controls(
        observables,
        masks,
        positions,
        metric_report,
        use_smooth_control=True,
    )
    detection_limits = injection_detection_limits(smooth_injection)
    channel = channel_structure(observables, pair_cache["strict"])
    assessment = mechanism_assessment(
        metric_report,
        official_injection,
        smooth_injection,
    )
    phase = phase_diagnostics(metric_report)
    make_figures(
        metric_report,
        variograms,
        official_injection,
        smooth_injection,
        channel,
    )

    geometry = np.load(GEOMETRY)
    blocked = np.asarray(geometry["train_blocked"], dtype=np.int32)
    row_regime = np.where(
        blocked[observables["point"]] <= 0,
        "clear",
        np.where(
            blocked[observables["point"]] <= 25,
            "light",
            np.where(blocked[observables["point"]] <= 100, "medium", "heavy"),
        ),
    )

    report = {
        "method": VERSION,
        "status": "analysis_complete_no_submission",
        "primary_hypothesis": (
            "For a stable physical path family, deterministic propagation parameters vary "
            "smoothly with UE position. A nonzero semivariogram intercept as spatial distance "
            "approaches zero is therefore an upper bound on path-subspace randomness. Delay, "
            "departure angle, log amplitude, and carrier-corrected phase have distinct tangent "
            "fingerprints."
        ),
        "identifiability_boundary": {
            "identifiable": [
                "zero-distance random-nugget upper bounds for stable matched paths",
                "whether delay correction removes a material part of carrier-corrected phase variation",
                "whether close-pair gain residuals are common across polarization/UE channels",
            ],
            "not_identifiable": [
                "the exact clean path state or exact random perturbation at one point",
                "a unique separation between organizer noise, path-identity errors, and unresolved deterministic path transitions",
                "noise on paths that are born or disappear between neighboring points",
            ],
        },
        "inputs": {
            "official_train_h": PATHS.relative(PATHS.round1 / "Round1_Train_Channel.npy"),
            "positions": PATHS.relative(TRAIN_POS),
            "continuous_components": PATHS.relative(v297.COMPONENT_CACHE),
            "path_fingerprints": PATHS.relative(v300.FINGERPRINT_CACHE),
            "pointcloud_matches": PATHS.relative(v305.MATCH_CACHE),
            "center_gains": PATHS.relative(v325.CENTER_GAIN_CACHE),
            "reflection_candidates": PATHS.relative(CANDIDATE_CACHE),
        },
        "parameters": {
            "carrier_beta_rad_per_meter": BETA,
            "maximum_pair_distance_m": MAX_PAIR_DISTANCE,
            "variogram_fit_distance_m": VARIOGRAM_FIT_DISTANCE,
            "distance_edges_m": DISTANCE_EDGES.tolist(),
            "quality_rules": QUALITY_RULES,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "injection_repeats": INJECTION_REPEATS,
        },
        "observable_population": {
            "row_count": int(len(observables["point"])),
            "point_count": int(len(np.unique(observables["point"]))),
            "patch_count": int(len(np.unique(observables["patch"]))),
            "regime_counts": {
                regime: int(np.sum(row_regime == regime))
                for regime in ("clear", "light", "medium", "heavy")
            },
        },
        "variogram_results": metric_report,
        "injection_controls": {
            "official_observation_stress_test": official_injection,
            "smooth_noiseless_calibration": smooth_injection,
            "detection_limits": detection_limits,
        },
        "phase_diagnostics": phase,
        "cross_channel_structure": channel,
        "mechanism_assessment": assessment,
        "strict_conclusion": (
            "Any positive path-level nugget remains an upper bound rather than a uniquely "
            "identified organizer noise variance. A mechanism is promoted only when it is "
            "positive across patch bootstraps, stable as match quality tightens, and its added "
            "variance is recovered on a smooth noiseless control. Even then, a single official "
            "realization cannot uniquely separate noise from path mismatch or unresolved "
            "deterministic transitions."
        ),
        "outputs": {
            "observable_cache": PATHS.relative(OBSERVABLE_CACHE),
            "report": PATHS.relative(REPORT),
            "figures": [PATHS.relative(path) for path in sorted(FIG_DIR.glob("*.png"))],
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["mechanism_assessment"], ensure_ascii=False, indent=2))
    print(f"wrote {PATHS.relative(REPORT)}", flush=True)


if __name__ == "__main__":
    main()
