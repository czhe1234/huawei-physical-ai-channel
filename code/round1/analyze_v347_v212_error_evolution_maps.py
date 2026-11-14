#!/usr/bin/env python3
"""Map how the v200 -> v212 online-improvement chain changes local errors.

The hidden online truth is unavailable. All pointwise errors in this analysis
use the frozen v207 synthetic clean-test channel and the v296 traceback scorer.
Online scores are shown only as scalar version-level evidence.
"""

from __future__ import annotations

import csv
import gc
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import analyze_v205_sionna_config_calibration as calibration
from powerfield_common import regime_of
from project_paths import PATHS


VERSION = 347
NAME = "v347_v212_error_evolution_maps"
REPORT_DIR = PATHS.version_report_dir(VERSION)
REPORT = REPORT_DIR / f"{NAME}.json"
POINT_CSV = REPORT_DIR / "v347_point_metrics.csv"
TRANSITION_CSV = REPORT_DIR / "v347_transition_point_gains.csv"
WORST_CSV = REPORT_DIR / "v347_v212_worst_points.csv"
SCORE_FIGURE = REPORT_DIR / "v347_score_evolution.png"
ERROR_FIGURE = REPORT_DIR / "v347_chain_composite_error_maps.png"
GAIN_FIGURE = REPORT_DIR / "v347_transition_gain_maps.png"
REGIME_FIGURE = REPORT_DIR / "v347_transition_regime_gains.png"
FINAL_FIGURE = REPORT_DIR / "v347_v212_final_residual_maps.png"
STRICT_FIGURE = REPORT_DIR / "v347_strict_reference_maps.png"

TRUTH = PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"
TRUTH_DELAY = PATHS.cache_dir(296) / "v296_truth_delay_power.npy"
TRUTH_LEAKED = PATHS.cache_dir(296) / "v296_truth_leaked_power.npy"
TRUTH_ENERGY = PATHS.cache_dir(296) / "v296_truth_energy.npy"
GEOMETRY = PATHS.shared_features / "height_geometry_cache.npz"
MAP_CACHE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"
HISTORICAL_LOCAL = (
    PATHS.version_report_dir(296) / "v296_actual_submission_local_scores.csv"
)

BATCH = 4
EXPECTED_SHAPE = (500, 256, 4, 192)
TINY = 1e-30
REGIMES = ("clear", "light", "medium", "heavy")

PRIMARY = (200, 209, 210, 211, 212)
STRICT_REFERENCES = (218, 284)
ONLINE = {
    200: 0.511854,
    209: 0.511934,
    210: 0.534149,
    211: 0.551384,
    212: 0.559160,
    218: 0.507201,
    284: 0.513129,
}
METHODS = {
    200: "selective power gate",
    209: "metric projection",
    210: "Sionna PDP prior",
    211: "PDP completion",
    212: "multi-domain PDP",
    218: "strict Physical-AI",
    284: "strict planar residual",
}
PREDICTIONS = {
    version: PATHS.cache_dir(version) / f"v{version}_Round1_Test_Channel.npy"
    for version in (*PRIMARY, *STRICT_REFERENCES)
}


def summarize(values: np.ndarray) -> dict[str, float | int]:
    value = np.asarray(values, dtype=np.float64)
    value = value[np.isfinite(value)]
    if value.size == 0:
        return {"count": 0}
    return {
        "count": int(value.size),
        "mean": float(np.mean(value)),
        "std": float(np.std(value)),
        "p10": float(np.quantile(value, 0.10)),
        "p50": float(np.quantile(value, 0.50)),
        "p90": float(np.quantile(value, 0.90)),
        "min": float(np.min(value)),
        "max": float(np.max(value)),
    }


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3 or np.std(x[valid]) <= 0 or np.std(y[valid]) <= 0:
        return float("nan")
    return float(spearmanr(x[valid], y[valid]).statistic)


def pas_power(value: np.ndarray) -> np.ndarray:
    value6 = value.reshape(len(value), 2, 16, 8, 4, 192)
    power = np.abs(np.fft.fft2(value6, axes=(2, 3))).astype(np.float32) ** 2
    return power.reshape(len(value), 256, 4, 192)


def leaked_power(value: np.ndarray) -> np.ndarray:
    value6 = value.reshape(len(value), 2, 16, 8, 4, 192)
    leaked = value6.transpose(0, 1, 5, 2, 4, 3)
    leaked = np.abs(np.fft.fft2(leaked, axes=(2, 3))).astype(np.float32) ** 2
    return leaked.transpose(0, 1, 4, 5, 2, 3).reshape(
        len(value), 2, 4, 8, 3072
    )


def apply_zero_policy(
    score: np.ndarray,
    truth_energy: np.ndarray,
    prediction_energy: np.ndarray,
) -> np.ndarray:
    output = np.asarray(score, dtype=np.float32).copy()
    truth_zero = truth_energy <= 1e-25
    prediction_zero = prediction_energy <= 1e-25
    output[truth_zero & prediction_zero] = 1.0
    output[truth_zero & ~prediction_zero] = 0.0
    return output


def score_version(version: int) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    path = PREDICTIONS[version]
    prediction = np.load(path, mmap_mode="r")
    truth = np.load(TRUTH, mmap_mode="r")
    truth_delay = np.load(TRUTH_DELAY, mmap_mode="r")
    truth_leaked = np.load(TRUTH_LEAKED, mmap_mode="r")
    truth_energy_cache = np.load(TRUTH_ENERGY, mmap_mode="r")
    if prediction.shape != EXPECTED_SHAPE or prediction.dtype != np.complex64:
        raise ValueError(f"invalid v{version} prediction: {prediction.shape} {prediction.dtype}")
    if truth.shape != EXPECTED_SHAPE or truth.dtype != np.complex64:
        raise ValueError(f"invalid truth: {truth.shape} {truth.dtype}")

    parts: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "pas",
            "pdp",
            "leaked_sh",
            "nmse_score",
            "point_nmse",
            "log_power_error",
            "prediction_power",
            "truth_power",
        )
    }
    total_error = 0.0
    total_energy = 0.0
    finite = True
    for start in range(0, len(prediction), BATCH):
        end = min(start + BATCH, len(prediction))
        pred = np.asarray(prediction[start:end], dtype=np.complex64)
        target = np.asarray(truth[start:end], dtype=np.complex64)
        finite = finite and bool(np.isfinite(pred).all())
        prediction_energy = np.sum(
            np.abs(pred) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
        truth_energy = np.asarray(truth_energy_cache[start:end], dtype=np.float64)
        error = np.sum(np.abs(pred - target) ** 2, axis=(1, 2, 3), dtype=np.float64)

        pred_delay = np.abs(np.fft.ifft(pred, axis=-1)).astype(np.float32) ** 2
        pdp = calibration.cosine(
            pred_delay, np.asarray(truth_delay[start:end]), (3,)
        ).mean(axis=(1, 2))
        pred_leaked = leaked_power(pred)
        leaked = calibration.cosine(
            pred_leaked, np.asarray(truth_leaked[start:end]), (4,)
        ).mean(axis=(1, 2, 3))
        pas = calibration.cosine(pas_power(pred), pas_power(target), (1,)).mean(
            axis=(1, 2)
        )
        pdp = apply_zero_policy(pdp, truth_energy, prediction_energy)
        leaked = apply_zero_policy(leaked, truth_energy, prediction_energy)
        pas = apply_zero_policy(pas, truth_energy, prediction_energy)

        truth_zero = truth_energy <= 1e-25
        prediction_zero = prediction_energy <= 1e-25
        point_nmse = np.full(len(pred), np.nan, dtype=np.float64)
        nmse_score = np.zeros(len(pred), dtype=np.float64)
        nonzero = ~truth_zero
        point_nmse[nonzero] = error[nonzero] / np.maximum(truth_energy[nonzero], TINY)
        nmse_score[nonzero] = 1.0 / (1.0 + point_nmse[nonzero])
        nmse_score[truth_zero & prediction_zero] = 1.0

        log_power_error = np.full(len(pred), np.nan, dtype=np.float64)
        log_power_error[nonzero] = np.log10(
            np.maximum(prediction_energy[nonzero], TINY)
            / np.maximum(truth_energy[nonzero], TINY)
        )
        for key, value in (
            ("pas", pas),
            ("pdp", pdp),
            ("leaked_sh", leaked),
            ("nmse_score", nmse_score),
            ("point_nmse", point_nmse),
            ("log_power_error", log_power_error),
            ("prediction_power", prediction_energy),
            ("truth_power", truth_energy),
        ):
            parts[key].append(np.asarray(value))
        total_error += float(np.sum(error))
        total_energy += float(np.sum(truth_energy))
        del pred, target, pred_delay, pred_leaked

    point = {key: np.concatenate(value) for key, value in parts.items()}
    point["composite"] = (
        0.4 * point["leaked_sh"]
        + 0.4 * point["pdp"]
        + 0.2 * point["nmse_score"]
    )
    point["composite_error"] = 1.0 - point["composite"]
    point["pas_error"] = 1.0 - point["pas"]
    point["pdp_error"] = 1.0 - point["pdp"]
    point["leaked_sh_error"] = 1.0 - point["leaked_sh"]
    point["nmse_error"] = 1.0 - point["nmse_score"]

    global_nmse = total_error / max(total_energy, TINY)
    global_nmse_score = 1.0 / (1.0 + global_nmse)
    summary = {
        "version": version,
        "method": METHODS[version],
        "prediction": PATHS.relative(path),
        "online_score": ONLINE[version],
        "finite": finite,
        "pas": float(np.mean(point["pas"])),
        "pdp": float(np.mean(point["pdp"])),
        "leaked_sh": float(np.mean(point["leaked_sh"])),
        "global_nmse": float(global_nmse),
        "nmse_score": float(global_nmse_score),
        "local_score": float(
            0.4 * np.mean(point["leaked_sh"])
            + 0.4 * np.mean(point["pdp"])
            + 0.2 * global_nmse_score
        ),
        "point_composite_mean": float(np.mean(point["composite"])),
        "point_imperfect_count": int(np.sum(point["composite_error"] > 1e-8)),
    }
    return summary, point


def load_historical_local() -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with HISTORICAL_LOCAL.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            version = int(row["version"])
            rows[version] = {
                "local_score": float(row["local_score"]),
                "pdp": float(row["pdp"]),
                "leaked_sh": float(row["leaked_sh"]),
                "nmse_score": float(row["nmse_score"]),
            }
    return rows


def validate_against_v296(summaries: dict[int, dict[str, object]]) -> None:
    expected = load_historical_local()
    for version, summary in summaries.items():
        if version not in expected:
            continue
        for key in ("local_score", "pdp", "leaked_sh", "nmse_score"):
            delta = abs(float(summary[key]) - expected[version][key])
            if delta > 2e-6:
                raise RuntimeError(f"v{version} {key} mismatch vs v296: {delta}")


def action_change(old_version: int, new_version: int) -> np.ndarray:
    old = np.load(PREDICTIONS[old_version], mmap_mode="r")
    new = np.load(PREDICTIONS[new_version], mmap_mode="r")
    output = np.empty(len(old), dtype=np.float64)
    for start in range(0, len(old), BATCH):
        end = min(start + BATCH, len(old))
        source = np.asarray(old[start:end], dtype=np.complex64)
        target = np.asarray(new[start:end], dtype=np.complex64)
        denominator = np.sum(np.abs(source) ** 2, axis=(1, 2, 3), dtype=np.float64)
        numerator = np.sum(
            np.abs(target - source) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
        output[start:end] = numerator / np.maximum(denominator, TINY)
    return output


def transition_results(
    summaries: dict[int, dict[str, object]],
    points: dict[int, dict[str, np.ndarray]],
    regimes: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, dict[str, np.ndarray]]]:
    results: list[dict[str, object]] = []
    raw: dict[str, dict[str, np.ndarray]] = {}
    metric_keys = ("composite", "pdp", "leaked_sh", "nmse_score", "pas")
    for old_version, new_version in zip(PRIMARY[:-1], PRIMARY[1:]):
        name = f"v{old_version}->v{new_version}"
        gain = {
            key: points[new_version][key] - points[old_version][key]
            for key in metric_keys
        }
        change = action_change(old_version, new_version)
        gain["action_rel_nmse"] = change
        raw[name] = gain
        result: dict[str, object] = {
            "transition": name,
            "old_version": old_version,
            "new_version": new_version,
            "online_delta": ONLINE[new_version] - ONLINE[old_version],
            "local_score_delta": float(summaries[new_version]["local_score"])
            - float(summaries[old_version]["local_score"]),
            "action_rel_nmse": summarize(change),
            "action_to_composite_gain_spearman": safe_spearman(
                np.log1p(change), gain["composite"]
            ),
            "overall": {},
            "by_regime": {},
        }
        for key in metric_keys:
            result["overall"][key] = {
                "mean_gain": float(np.mean(gain[key])),
                "median_gain": float(np.median(gain[key])),
                "improved_fraction": float(np.mean(gain[key] > 0)),
                "degraded_fraction": float(np.mean(gain[key] < 0)),
            }
        for regime in REGIMES:
            mask = regimes == regime
            result["by_regime"][regime] = {
                key: {
                    "mean_gain": float(np.mean(gain[key][mask])),
                    "improved_fraction": float(np.mean(gain[key][mask] > 0)),
                }
                for key in metric_keys
            }
            result["by_regime"][regime]["count"] = int(np.sum(mask))
            result["by_regime"][regime]["action_rel_nmse_mean"] = float(
                np.mean(change[mask])
            )
        results.append(result)
    return results, raw


def load_map_background() -> dict[str, object]:
    cache = np.load(MAP_CACHE)
    vertical = np.asarray(cache["grid_vertical_count"], dtype=np.float32)
    x_min = float(cache["x_min"])
    y_min = float(cache["y_min"])
    resolution = float(cache["resolution"])
    ny, nx = vertical.shape
    image = np.log1p(vertical)
    positive = image[image > 0]
    vmax = float(np.quantile(positive, 0.98)) if positive.size else 1.0
    return {
        "image": image,
        "extent": (x_min, x_min + nx * resolution, y_min, y_min + ny * resolution),
        "vmax": vmax,
    }


def map_base(
    axis: plt.Axes,
    positions: np.ndarray,
    background: dict[str, object],
) -> None:
    axis.imshow(
        background["image"],
        extent=background["extent"],
        origin="lower",
        cmap="Greys",
        vmin=0,
        vmax=background["vmax"],
        alpha=0.28,
        interpolation="nearest",
        rasterized=True,
    )
    margin = 18.0
    axis.set_xlim(float(np.min(positions[:, 0]) - margin), float(np.max(positions[:, 0]) + margin))
    axis.set_ylim(float(np.min(positions[:, 1]) - margin), float(np.max(positions[:, 1]) + margin))
    axis.scatter([50.0], [0.0], marker="*", s=80, c="#111111", edgecolors="white", linewidths=0.7, zorder=4)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")


def robust_limit(values: list[np.ndarray], quantile: float = 0.98) -> float:
    merged = np.concatenate([np.ravel(value[np.isfinite(value)]) for value in values])
    if merged.size == 0:
        return 1.0
    return max(float(np.quantile(np.abs(merged), quantile)), 1e-5)


def plot_score_evolution(summaries: dict[int, dict[str, object]]) -> None:
    labels = [f"v{version}" for version in PRIMARY]
    x = np.arange(len(PRIMARY))
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 9.0), constrained_layout=True)
    axis = axes[0]
    for key, label, color in (
        ("local_score", "local traceback", "#111111"),
        ("pdp", "local PDP", "#0072B2"),
        ("leaked_sh", "local traceback PAS", "#009E73"),
        ("nmse_score", "local NMSE score", "#D55E00"),
        ("online_score", "online score", "#CC79A7"),
    ):
        value = [float(summaries[version][key]) for version in PRIMARY]
        axis.plot(x, value, marker="o", linewidth=2.0, label=label, color=color)
    axis.set_xticks(x, labels)
    axis.set_ylabel("score")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, loc="best")
    axis.set_title("v200-v212 score evolution (local truth is synthetic, not official hidden truth)")

    transitions = [f"v{old}->v{new}" for old, new in zip(PRIMARY[:-1], PRIMARY[1:])]
    tx = np.arange(len(transitions))
    width = 0.18
    for index, (key, label, color) in enumerate(
        (
            ("online_score", "online", "#CC79A7"),
            ("local_score", "local traceback", "#111111"),
            ("pdp", "local PDP", "#0072B2"),
            ("leaked_sh", "local traceback PAS", "#009E73"),
        )
    ):
        delta = [
            float(summaries[new][key]) - float(summaries[old][key])
            for old, new in zip(PRIMARY[:-1], PRIMARY[1:])
        ]
        axes[1].bar(tx + (index - 1.5) * width, delta, width=width, label=label, color=color)
    axes[1].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1].set_xticks(tx, transitions)
    axes[1].set_ylabel("delta score")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(ncol=4, loc="best")
    axes[1].set_title("Gain at each transition")
    figure.savefig(SCORE_FIGURE, dpi=180)
    plt.close(figure)


def plot_chain_errors(
    summaries: dict[int, dict[str, object]],
    points: dict[int, dict[str, np.ndarray]],
    positions: np.ndarray,
    background: dict[str, object],
) -> None:
    values = [points[version]["composite_error"] for version in PRIMARY]
    vmax = robust_limit(values)
    figure, axes = plt.subplots(2, 3, figsize=(16.0, 10.2), constrained_layout=True)
    scatter = None
    for axis, version, value in zip(axes.flat, PRIMARY, values):
        map_base(axis, positions, background)
        scatter = axis.scatter(
            positions[:, 0], positions[:, 1], c=value, s=26, cmap="magma", vmin=0, vmax=vmax,
            edgecolors="white", linewidths=0.25, zorder=3,
        )
        axis.set_title(
            f"v{version} online={ONLINE[version]:.6f}\nlocal={float(summaries[version]['local_score']):.6f}"
        )
    axes.flat[-1].axis("off")
    if scatter is not None:
        figure.colorbar(scatter, ax=axes[:, :], shrink=0.82, label="point composite error (lower is better)")
    figure.suptitle("Spatial evolution of pointwise traceback error")
    figure.savefig(ERROR_FIGURE, dpi=180)
    plt.close(figure)


def plot_transition_gains(
    raw: dict[str, dict[str, np.ndarray]],
    positions: np.ndarray,
    background: dict[str, object],
) -> None:
    transitions = list(raw)
    metrics = (
        ("composite", "composite gain"),
        ("pdp", "PDP gain"),
        ("leaked_sh", "traceback PAS gain"),
        ("nmse_score", "point NMSE gain"),
    )
    limits = {key: robust_limit([raw[name][key] for name in transitions]) for key, _ in metrics}
    figure, axes = plt.subplots(len(transitions), len(metrics), figsize=(19.0, 17.5), constrained_layout=True)
    column_scatters: dict[str, object] = {}
    for row, transition in enumerate(transitions):
        for column, (key, label) in enumerate(metrics):
            axis = axes[row, column]
            map_base(axis, positions, background)
            limit = limits[key]
            scatter = axis.scatter(
                positions[:, 0], positions[:, 1], c=raw[transition][key], s=20,
                cmap="RdBu_r", vmin=-limit, vmax=limit, edgecolors="none", zorder=3,
            )
            column_scatters[key] = scatter
            axis.set_title(f"{transition}: {label}")
    for column, (key, label) in enumerate(metrics):
        figure.colorbar(
            column_scatters[key], ax=axes[:, column], shrink=0.72,
            label=f"{label} (>0 is better)",
        )
    figure.suptitle("Where each online-improving transition helps or hurts on synthetic local truth")
    figure.savefig(GAIN_FIGURE, dpi=180)
    plt.close(figure)


def plot_regime_gains(
    raw: dict[str, dict[str, np.ndarray]], regimes: np.ndarray
) -> None:
    transitions = list(raw)
    metrics = (
        ("composite", "composite"),
        ("pdp", "PDP"),
        ("leaked_sh", "traceback PAS"),
        ("nmse_score", "point NMSE"),
    )
    matrices = {}
    for key, _ in metrics:
        matrices[key] = np.asarray(
            [
                [float(np.mean(raw[name][key][regimes == regime])) for regime in REGIMES]
                for name in transitions
            ],
            dtype=np.float64,
        )
    limit = robust_limit(list(matrices.values()), quantile=1.0)
    figure, axes = plt.subplots(1, 4, figsize=(18.0, 5.4), constrained_layout=True)
    image = None
    for axis, (key, label) in zip(axes, metrics):
        matrix = matrices[key]
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        axis.set_xticks(np.arange(len(REGIMES)), REGIMES)
        axis.set_yticks(np.arange(len(transitions)), transitions)
        axis.set_title(f"mean {label} gain")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, f"{matrix[row, column]:+.4f}", ha="center", va="center", fontsize=8)
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.78, label="mean point gain (>0 is better)")
    figure.suptitle("Which propagation regimes change at each step")
    figure.savefig(REGIME_FIGURE, dpi=180)
    plt.close(figure)


def plot_final_residuals(
    points: dict[int, dict[str, np.ndarray]],
    positions: np.ndarray,
    background: dict[str, object],
) -> None:
    metrics = (
        ("pdp_error", "v212 PDP error", "magma", 0.0),
        ("leaked_sh_error", "v212 traceback PAS error", "magma", 0.0),
        ("nmse_error", "v212 point NMSE error", "magma", 0.0),
        ("log_power_error", "v212 log10 power ratio error", "RdBu_r", None),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 11.0), constrained_layout=True)
    for axis, (key, label, cmap, lower) in zip(axes.flat, metrics):
        value = points[212][key]
        map_base(axis, positions, background)
        if lower is None:
            limit = robust_limit([value])
            vmin, vmax = -limit, limit
        else:
            vmin, vmax = lower, robust_limit([value])
        scatter = axis.scatter(
            positions[:, 0], positions[:, 1], c=value, s=27, cmap=cmap, vmin=vmin, vmax=vmax,
            edgecolors="white", linewidths=0.25, zorder=3,
        )
        axis.set_title(label)
        figure.colorbar(scatter, ax=axis, shrink=0.78)
    figure.suptitle("What remains wrong in v212 under the synthetic local reference")
    figure.savefig(FINAL_FIGURE, dpi=180)
    plt.close(figure)


def plot_strict_references(
    summaries: dict[int, dict[str, object]],
    points: dict[int, dict[str, np.ndarray]],
    positions: np.ndarray,
    background: dict[str, object],
) -> None:
    versions = (218, 284, 212)
    metrics = (
        ("composite_error", "composite error"),
        ("pdp_error", "PDP error"),
        ("leaked_sh_error", "traceback PAS error"),
    )
    limits = {
        key: robust_limit([points[version][key] for version in versions])
        for key, _ in metrics
    }
    figure, axes = plt.subplots(3, 3, figsize=(16.5, 15.5), constrained_layout=True)
    column_scatters: dict[str, object] = {}
    for row, version in enumerate(versions):
        for column, (key, label) in enumerate(metrics):
            axis = axes[row, column]
            map_base(axis, positions, background)
            scatter = axis.scatter(
                positions[:, 0], positions[:, 1], c=points[version][key], s=22,
                cmap="magma", vmin=0, vmax=limits[key], edgecolors="none", zorder=3,
            )
            column_scatters[key] = scatter
            axis.set_title(
                f"v{version} {label}\nonline={ONLINE[version]:.6f}, local={float(summaries[version]['local_score']):.6f}"
            )
    for column, (key, label) in enumerate(metrics):
        figure.colorbar(column_scatters[key], ax=axes[:, column], shrink=0.72, label=label)
    figure.suptitle("Strict references versus v212 (v212 is diagnostic, not a compliant final method)")
    figure.savefig(STRICT_FIGURE, dpi=180)
    plt.close(figure)


def write_point_csv(
    points: dict[int, dict[str, np.ndarray]],
    positions: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
) -> None:
    fields = (
        "version", "online_score", "point_id", "x", "y", "z", "blocked", "regime",
        "composite", "composite_error", "pas", "pdp", "leaked_sh", "nmse_score",
        "point_nmse", "log_power_error",
    )
    with POINT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for version in (*PRIMARY, *STRICT_REFERENCES):
            for index in range(len(positions)):
                writer.writerow(
                    {
                        "version": version,
                        "online_score": ONLINE[version],
                        "point_id": index,
                        "x": float(positions[index, 0]),
                        "y": float(positions[index, 1]),
                        "z": float(positions[index, 2]),
                        "blocked": float(blocked[index]),
                        "regime": str(regimes[index]),
                        "composite": float(points[version]["composite"][index]),
                        "composite_error": float(points[version]["composite_error"][index]),
                        "pas": float(points[version]["pas"][index]),
                        "pdp": float(points[version]["pdp"][index]),
                        "leaked_sh": float(points[version]["leaked_sh"][index]),
                        "nmse_score": float(points[version]["nmse_score"][index]),
                        "point_nmse": float(points[version]["point_nmse"][index]),
                        "log_power_error": float(points[version]["log_power_error"][index]),
                    }
                )


def write_transition_csv(
    raw: dict[str, dict[str, np.ndarray]],
    positions: np.ndarray,
    regimes: np.ndarray,
) -> None:
    fields = (
        "transition", "point_id", "x", "y", "regime", "online_delta",
        "composite_gain", "pdp_gain", "leaked_sh_gain", "pas_gain", "nmse_score_gain",
        "action_rel_nmse",
    )
    with TRANSITION_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for transition, values in raw.items():
            old_version, new_version = [int(value[1:]) for value in transition.split("->")]
            for index in range(len(positions)):
                writer.writerow(
                    {
                        "transition": transition,
                        "point_id": index,
                        "x": float(positions[index, 0]),
                        "y": float(positions[index, 1]),
                        "regime": str(regimes[index]),
                        "online_delta": ONLINE[new_version] - ONLINE[old_version],
                        "composite_gain": float(values["composite"][index]),
                        "pdp_gain": float(values["pdp"][index]),
                        "leaked_sh_gain": float(values["leaked_sh"][index]),
                        "pas_gain": float(values["pas"][index]),
                        "nmse_score_gain": float(values["nmse_score"][index]),
                        "action_rel_nmse": float(values["action_rel_nmse"][index]),
                    }
                )


def write_worst_csv(
    points: dict[int, dict[str, np.ndarray]],
    raw: dict[str, dict[str, np.ndarray]],
    positions: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
) -> None:
    order = np.argsort(points[212]["composite_error"])[::-1][:50]
    transitions = list(raw)
    fields = [
        "rank", "point_id", "x", "y", "blocked", "regime", "v212_composite_error",
        "v212_pdp_error", "v212_leaked_sh_error", "v212_nmse_error", "v212_log_power_error",
    ] + [f"{name}_composite_gain" for name in transitions]
    with WORST_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, index in enumerate(order, start=1):
            row = {
                "rank": rank,
                "point_id": int(index),
                "x": float(positions[index, 0]),
                "y": float(positions[index, 1]),
                "blocked": float(blocked[index]),
                "regime": str(regimes[index]),
                "v212_composite_error": float(points[212]["composite_error"][index]),
                "v212_pdp_error": float(points[212]["pdp_error"][index]),
                "v212_leaked_sh_error": float(points[212]["leaked_sh_error"][index]),
                "v212_nmse_error": float(points[212]["nmse_error"][index]),
                "v212_log_power_error": float(points[212]["log_power_error"][index]),
            }
            for name in transitions:
                row[f"{name}_composite_gain"] = float(raw[name]["composite"][index])
            writer.writerow(row)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    required = (
        TRUTH, TRUTH_DELAY, TRUTH_LEAKED, TRUTH_ENERGY, GEOMETRY, MAP_CACHE,
        HISTORICAL_LOCAL, *PREDICTIONS.values(),
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    positions = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    geometry = np.load(GEOMETRY)
    blocked = geometry["test_blocked"].astype(np.float64)
    regimes = regime_of(blocked).astype("U8")
    background = load_map_background()

    summaries: dict[int, dict[str, object]] = {}
    points: dict[int, dict[str, np.ndarray]] = {}
    for version in (*PRIMARY, *STRICT_REFERENCES):
        print(f"score v{version}", flush=True)
        summaries[version], points[version] = score_version(version)
        print(
            f"v{version} online={ONLINE[version]:.6f} local={float(summaries[version]['local_score']):.6f} "
            f"pdp={float(summaries[version]['pdp']):.6f}",
            flush=True,
        )
        gc.collect()
    validate_against_v296(summaries)

    transitions, raw_transitions = transition_results(summaries, points, regimes)
    write_point_csv(points, positions, blocked, regimes)
    write_transition_csv(raw_transitions, positions, regimes)
    write_worst_csv(points, raw_transitions, positions, blocked, regimes)
    plot_score_evolution(summaries)
    plot_chain_errors(summaries, points, positions, background)
    plot_transition_gains(raw_transitions, positions, background)
    plot_regime_gains(raw_transitions, regimes)
    plot_final_residuals(points, positions, background)
    plot_strict_references(summaries, points, positions, background)

    report = {
        "analysis": NAME,
        "status": "analysis_only_no_candidate",
        "truth_boundary": {
            "pointwise_truth": PATHS.relative(TRUTH),
            "meaning": "v207 synthetic clean test reference, not official hidden online truth",
            "online_evidence": "scalar score per version only; official pointwise errors are unavailable",
            "point_composite": "0.4*traceback_leaked_SH + 0.4*PDP + 0.2*pointwise_NMSE_score",
            "global_local_score": "v296 score_traceback_leaked with global NMSE",
        },
        "primary_chain": list(PRIMARY),
        "strict_references": list(STRICT_REFERENCES),
        "versions": {str(version): summaries[version] for version in summaries},
        "transitions": transitions,
        "regime_counts": {regime: int(np.sum(regimes == regime)) for regime in REGIMES},
        "v212_residual": {
            "composite_error": summarize(points[212]["composite_error"]),
            "pdp_error": summarize(points[212]["pdp_error"]),
            "traceback_pas_error": summarize(points[212]["leaked_sh_error"]),
            "point_nmse_error": summarize(points[212]["nmse_error"]),
            "abs_log10_power_error": summarize(np.abs(points[212]["log_power_error"])),
            "worst_50_regime_counts": {
                regime: int(
                    np.sum(
                        regimes[np.argsort(points[212]["composite_error"])[::-1][:50]] == regime
                    )
                )
                for regime in REGIMES
            },
        },
        "outputs": [
            PATHS.relative(path)
            for path in (
                REPORT, POINT_CSV, TRANSITION_CSV, WORST_CSV, SCORE_FIGURE,
                ERROR_FIGURE, GAIN_FIGURE, REGIME_FIGURE, FINAL_FIGURE, STRICT_FIGURE,
            )
        ],
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "report": PATHS.relative(REPORT),
        "v212": summaries[212],
        "transitions": transitions,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
