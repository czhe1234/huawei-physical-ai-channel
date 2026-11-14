#!/usr/bin/env python3
"""Map how the v189-v212 online-improvement chain changed local errors."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

import analyze_v205_sionna_config_calibration as calibration
import analyze_v208_strict_historical_replay as v208
import analyze_v296_actual_submission_local_online_relation as v296
from powerfield_common import BS_POS, regime_of
from project_paths import PATHS


VERSION_NUMBER = 347
VERSION = "v347_v212_error_evolution"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
VERSION_CSV = REPORT_DIR / "v347_version_metrics.csv"
TRANSITION_CSV = REPORT_DIR / "v347_transition_metrics.csv"
POINT_VERSION_CSV = REPORT_DIR / "v347_point_version_metrics.csv"
POINT_TRANSITION_CSV = REPORT_DIR / "v347_point_transition_metrics.csv"
POINT_METRICS = REPORT_DIR / "v347_point_metrics.npz"
ABSOLUTE_FIGURE = REPORT_DIR / "v347_absolute_error_maps.png"
TRANSITION_FIGURE = REPORT_DIR / "v347_transition_delta_maps.png"
SUMMARY_FIGURE = REPORT_DIR / "v347_metric_evolution_summary.png"
V212_FIGURE = REPORT_DIR / "v347_v212_remaining_error_map.png"
GEOMETRY = PATHS.shared_features / "height_geometry_cache.npz"
SCENE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"
LOCAL_SCORE_CSV = PATHS.version_report_dir(296) / "v296_actual_submission_local_scores.csv"
BATCH = 8
TINY = 1e-30
EXPECTED_SHAPE = (500, 256, 4, 192)
REGIMES = ("clear", "light", "medium", "heavy")


@dataclass(frozen=True)
class VersionSpec:
    version: int
    name: str
    online_score: float

    @property
    def prediction(self) -> Path:
        return PATHS.cache_dir(self.version) / f"v{self.version}_Round1_Test_Channel.npy"


VERSIONS = (
    VersionSpec(189, "leaked multi-marginal", 0.510038),
    VersionSpec(196, "train-guided delay", 0.511432),
    VersionSpec(200, "selective power", 0.511854),
    VersionSpec(209, "metric projection", 0.511934),
    VersionSpec(210, "Sionna PDP prior", 0.534149),
    VersionSpec(211, "PDP completion", 0.551384),
    VersionSpec(212, "multi-domain PDP", 0.559160),
)

POINT_FIELDS = (
    "pdp",
    "leaked_sh",
    "nmse_score",
    "point_proxy",
    "point_nmse",
    "log10_point_nmse",
    "log_power_error",
    "abs_log_power_error",
)


def quantile_limits(values: list[np.ndarray], *, symmetric: bool = False) -> tuple[float, float]:
    merged = np.concatenate([np.asarray(value, dtype=np.float64) for value in values])
    merged = merged[np.isfinite(merged)]
    if symmetric:
        bound = max(float(np.quantile(np.abs(merged), 0.98)), 1e-6)
        return -bound, bound
    low, high = np.quantile(merged, (0.02, 0.98))
    if high <= low:
        high = low + 1e-6
    return float(low), float(high)


def summarize(values: np.ndarray) -> dict[str, float | int]:
    value = np.asarray(values, dtype=np.float64)
    value = value[np.isfinite(value)]
    return {
        "count": int(len(value)),
        "mean": float(np.mean(value)),
        "std": float(np.std(value)),
        "p10": float(np.quantile(value, 0.10)),
        "p50": float(np.quantile(value, 0.50)),
        "p90": float(np.quantile(value, 0.90)),
        "min": float(np.min(value)),
        "max": float(np.max(value)),
    }


def load_expected_local_scores() -> dict[int, dict[str, float]]:
    expected: dict[int, dict[str, float]] = {}
    with LOCAL_SCORE_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            version = int(row["version"])
            if version in {spec.version for spec in VERSIONS}:
                expected[version] = {
                    "score_traceback": float(row["local_score"]),
                    "pdp": float(row["pdp"]),
                    "leaked_sh": float(row["leaked_sh"]),
                    "nmse_score": float(row["nmse_score"]),
                }
    missing = sorted({spec.version for spec in VERSIONS} - set(expected))
    if missing:
        raise FileNotFoundError(f"v296 local scores missing versions: {missing}")
    return expected


def validate_inputs() -> None:
    required = [v208.TEST_TRUTH, GEOMETRY, SCENE, LOCAL_SCORE_CSV]
    required.extend(spec.prediction for spec in VERSIONS)
    missing = [PATHS.relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing v347 inputs: {missing}")
    for spec in VERSIONS:
        value = np.load(spec.prediction, mmap_mode="r")
        if value.shape != EXPECTED_SHAPE or value.dtype != np.complex64:
            raise ValueError(
                f"invalid v{spec.version} prediction: {value.shape} {value.dtype}"
            )


def evaluate_chain() -> tuple[
    dict[int, dict[str, np.ndarray]],
    dict[tuple[int, int], dict[str, np.ndarray]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    v296.build_truth_cache()
    truth = np.load(v208.TEST_TRUTH, mmap_mode="r")
    truth_delay = np.load(v296.TRUTH_DELAY, mmap_mode="r")
    truth_leaked = np.load(v296.TRUTH_LEAKED, mmap_mode="r")
    truth_energy = np.load(v296.TRUTH_ENERGY, mmap_mode="r")
    predictions = {
        spec.version: np.load(spec.prediction, mmap_mode="r") for spec in VERSIONS
    }
    point = {
        spec.version: {
            field: np.empty(EXPECTED_SHAPE[0], dtype=np.float64) for field in POINT_FIELDS
        }
        for spec in VERSIONS
    }
    transitions = {
        (base.version, current.version): {
            "change_nmse": np.empty(EXPECTED_SHAPE[0], dtype=np.float64),
            "log10_change_nmse": np.empty(EXPECTED_SHAPE[0], dtype=np.float64),
            "log_power_ratio": np.empty(EXPECTED_SHAPE[0], dtype=np.float64),
        }
        for base, current in zip(VERSIONS[:-1], VERSIONS[1:])
    }
    global_error = {spec.version: 0.0 for spec in VERSIONS}
    transition_error = {key: 0.0 for key in transitions}
    transition_base_energy = {key: 0.0 for key in transitions}
    total_truth_energy = float(np.sum(truth_energy, dtype=np.float64))

    for start in range(0, EXPECTED_SHAPE[0], BATCH):
        end = min(start + BATCH, EXPECTED_SHAPE[0])
        target = np.asarray(truth[start:end], dtype=np.complex64)
        target_energy = np.asarray(truth_energy[start:end], dtype=np.float64)
        true_zero = target_energy <= 1e-25
        previous: np.ndarray | None = None
        previous_spec: VersionSpec | None = None
        previous_energy: np.ndarray | None = None

        for spec in VERSIONS:
            prediction = np.asarray(predictions[spec.version][start:end], dtype=np.complex64)
            if not np.isfinite(prediction).all():
                raise FloatingPointError(f"non-finite prediction in v{spec.version}")
            pred_delay = np.abs(np.fft.ifft(prediction, axis=-1)).astype(np.float32) ** 2
            pred_leaked = v296.leaked_power(prediction)
            pdp = calibration.cosine(
                pred_delay, np.asarray(truth_delay[start:end]), (3,)
            ).mean(axis=(1, 2))
            leaked = calibration.cosine(
                pred_leaked, np.asarray(truth_leaked[start:end]), (4,)
            ).mean(axis=(1, 2, 3))
            pred_energy = np.sum(
                np.abs(prediction) ** 2, axis=(1, 2, 3), dtype=np.float64
            )
            pred_zero = pred_energy <= 1e-25
            pdp[true_zero & pred_zero] = 1.0
            pdp[true_zero & ~pred_zero] = 0.0
            leaked[true_zero & pred_zero] = 1.0
            leaked[true_zero & ~pred_zero] = 0.0
            squared_error = np.sum(
                np.abs(prediction - target) ** 2,
                axis=(1, 2, 3),
                dtype=np.float64,
            )
            point_nmse = squared_error / np.maximum(target_energy, TINY)
            nmse_score = 1.0 / (1.0 + point_nmse)
            log_power_error = np.log(np.maximum(pred_energy, TINY)) - np.log(
                np.maximum(target_energy, TINY)
            )
            values = point[spec.version]
            values["pdp"][start:end] = pdp
            values["leaked_sh"][start:end] = leaked
            values["nmse_score"][start:end] = nmse_score
            values["point_proxy"][start:end] = (
                0.4 * leaked + 0.4 * pdp + 0.2 * nmse_score
            )
            values["point_nmse"][start:end] = point_nmse
            values["log10_point_nmse"][start:end] = np.log10(
                np.maximum(point_nmse, TINY)
            )
            values["log_power_error"][start:end] = log_power_error
            values["abs_log_power_error"][start:end] = np.abs(log_power_error)
            global_error[spec.version] += float(np.sum(squared_error, dtype=np.float64))

            if previous is not None and previous_spec is not None and previous_energy is not None:
                key = (previous_spec.version, spec.version)
                change_error = np.sum(
                    np.abs(prediction - previous) ** 2,
                    axis=(1, 2, 3),
                    dtype=np.float64,
                )
                change_nmse = change_error / np.maximum(previous_energy, TINY)
                transitions[key]["change_nmse"][start:end] = change_nmse
                transitions[key]["log10_change_nmse"][start:end] = np.log10(
                    np.maximum(change_nmse, TINY)
                )
                transitions[key]["log_power_ratio"][start:end] = np.log(
                    np.maximum(pred_energy, TINY)
                ) - np.log(np.maximum(previous_energy, TINY))
                transition_error[key] += float(np.sum(change_error, dtype=np.float64))
                transition_base_energy[key] += float(
                    np.sum(previous_energy, dtype=np.float64)
                )
            previous = prediction
            previous_spec = spec
            previous_energy = pred_energy
        print(f"v347 metrics {end}/{EXPECTED_SHAPE[0]}", flush=True)

    expected = load_expected_local_scores()
    version_rows: list[dict[str, object]] = []
    for spec in VERSIONS:
        values = point[spec.version]
        global_nmse = global_error[spec.version] / max(total_truth_energy, TINY)
        nmse_score = 1.0 / (1.0 + global_nmse)
        pdp = float(np.mean(values["pdp"]))
        leaked = float(np.mean(values["leaked_sh"]))
        score = 0.4 * leaked + 0.4 * pdp + 0.2 * nmse_score
        row = {
            "version": spec.version,
            "name": spec.name,
            "online_score": spec.online_score,
            "local_score_traceback": score,
            "pdp": pdp,
            "leaked_sh": leaked,
            "global_nmse": global_nmse,
            "nmse_score": nmse_score,
            "mean_point_proxy": float(np.mean(values["point_proxy"])),
            "mean_abs_log_power_error": float(
                np.mean(values["abs_log_power_error"])
            ),
        }
        for key in ("score_traceback", "pdp", "leaked_sh", "nmse_score"):
            actual_key = "local_score_traceback" if key == "score_traceback" else key
            if abs(float(row[actual_key]) - expected[spec.version][key]) > 2e-6:
                raise RuntimeError(
                    f"v{spec.version} {key} mismatch: "
                    f"{row[actual_key]} != {expected[spec.version][key]}"
                )
        version_rows.append(row)

    geometry = np.load(GEOMETRY)
    regimes = regime_of(geometry["test_blocked"])
    transition_rows: list[dict[str, object]] = []
    for base, current in zip(VERSIONS[:-1], VERSIONS[1:]):
        key = (base.version, current.version)
        base_values = point[base.version]
        current_values = point[current.version]
        values = transitions[key]
        for metric in ("pdp", "leaked_sh", "nmse_score", "point_proxy"):
            values[f"delta_{metric}"] = current_values[metric] - base_values[metric]
        row: dict[str, object] = {
            "transition": f"v{base.version}->v{current.version}",
            "base_version": base.version,
            "version": current.version,
            "online_delta": current.online_score - base.online_score,
            "local_score_delta": (
                next(item for item in version_rows if item["version"] == current.version)[
                    "local_score_traceback"
                ]
                - next(item for item in version_rows if item["version"] == base.version)[
                    "local_score_traceback"
                ]
            ),
            "global_output_change_nmse": transition_error[key]
            / max(transition_base_energy[key], TINY),
            "mean_abs_log_power_change": float(
                np.mean(np.abs(values["log_power_ratio"]))
            ),
        }
        for metric in ("pdp", "leaked_sh", "nmse_score", "point_proxy"):
            delta = values[f"delta_{metric}"]
            row[f"mean_delta_{metric}"] = float(np.mean(delta))
            row[f"improved_fraction_{metric}"] = float(np.mean(delta > 0.0))
            row[f"worsened_fraction_{metric}"] = float(np.mean(delta < 0.0))
        row["by_regime"] = {
            regime: {
                metric: float(np.mean(values[f"delta_{metric}"][regimes == regime]))
                for metric in ("pdp", "leaked_sh", "nmse_score", "point_proxy")
            }
            for regime in REGIMES
        }
        order = np.argsort(values["delta_point_proxy"])
        row["top_improved_indices"] = order[-10:][::-1].astype(int).tolist()
        row["top_worsened_indices"] = order[:10].astype(int).tolist()
        transition_rows.append(row)
    return point, transitions, version_rows, transition_rows


def map_background() -> dict[str, object]:
    scene = np.load(SCENE)
    wall = np.log1p(scene["grid_vertical_count"].astype(np.float32))
    x_min = float(scene["x_min"])
    y_min = float(scene["y_min"])
    resolution = float(scene["resolution"])
    return {
        "wall": wall,
        "extent": (
            x_min,
            x_min + wall.shape[1] * resolution,
            y_min,
            y_min + wall.shape[0] * resolution,
        ),
        "vmax": max(float(np.quantile(wall[wall > 0], 0.98)), 1.0),
    }


def decorate_map(axis: plt.Axes, background: dict[str, object]) -> None:
    axis.imshow(
        background["wall"],
        origin="lower",
        extent=background["extent"],
        cmap="Greys",
        vmin=0.0,
        vmax=background["vmax"],
        alpha=0.20,
        interpolation="nearest",
        zorder=0,
    )
    axis.scatter(
        [BS_POS[0]], [BS_POS[1]], marker="*", s=42, color="black", zorder=3
    )
    axis.set_aspect("equal", adjustable="box")


def plot_absolute_maps(
    point: dict[int, dict[str, np.ndarray]], positions: np.ndarray
) -> None:
    background = map_background()
    fields = (
        ("pdp_deficit", "PDP deficit", "magma"),
        ("leaked_deficit", "Leaked-PAS deficit", "magma"),
        ("log10_point_nmse", "log10 point NMSE", "magma"),
        ("abs_log_power_error", "|log power error|", "magma"),
    )
    derived = {
        spec.version: {
            "pdp_deficit": 1.0 - point[spec.version]["pdp"],
            "leaked_deficit": 1.0 - point[spec.version]["leaked_sh"],
            "log10_point_nmse": point[spec.version]["log10_point_nmse"],
            "abs_log_power_error": point[spec.version]["abs_log_power_error"],
        }
        for spec in VERSIONS
    }
    limits = {
        key: quantile_limits([derived[spec.version][key] for spec in VERSIONS])
        for key, _, _ in fields
    }
    fig, axes = plt.subplots(
        len(VERSIONS), len(fields), figsize=(17, 25), sharex=True, sharey=True,
        constrained_layout=True,
    )
    for row, spec in enumerate(VERSIONS):
        for col, (key, title, cmap) in enumerate(fields):
            axis = axes[row, col]
            decorate_map(axis, background)
            vmin, vmax = limits[key]
            axis.scatter(
                positions[:, 0], positions[:, 1], c=derived[spec.version][key],
                s=18, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0, zorder=2,
            )
            if row == 0:
                axis.set_title(title)
            if col == 0:
                axis.set_ylabel(f"v{spec.version}  online={spec.online_score:.6f}\ny (m)")
            if row == len(VERSIONS) - 1:
                axis.set_xlabel("x (m)")
    for col, (key, _, cmap) in enumerate(fields):
        color = plt.cm.ScalarMappable(norm=Normalize(*limits[key]), cmap=cmap)
        fig.colorbar(color, ax=axes[:, col], fraction=0.015, pad=0.01)
    fig.suptitle(
        "v189-v212 absolute error against the v207 Sionna reference\n"
        "Diagnostic only: this is not hidden online truth",
        fontsize=15,
    )
    fig.savefig(ABSOLUTE_FIGURE, dpi=180)
    plt.close(fig)


def plot_transition_maps(
    transitions: dict[tuple[int, int], dict[str, np.ndarray]],
    positions: np.ndarray,
) -> None:
    background = map_background()
    pairs = list(zip(VERSIONS[:-1], VERSIONS[1:]))
    fields = (
        ("log10_change_nmse", "log10 output change", "viridis", False),
        ("delta_pdp", "PDP gain", "RdYlGn", True),
        ("delta_leaked_sh", "Leaked-PAS gain", "RdYlGn", True),
        ("delta_nmse_score", "Point NMSE-score gain", "RdYlGn", True),
        ("delta_point_proxy", "Point diagnostic gain", "RdYlGn", True),
    )
    limits = {
        key: quantile_limits(
            [transitions[(base.version, current.version)][key] for base, current in pairs],
            symmetric=symmetric,
        )
        for key, _, _, symmetric in fields
    }
    fig, axes = plt.subplots(
        len(pairs), len(fields), figsize=(21, 23), sharex=True, sharey=True,
        constrained_layout=True,
    )
    for row, (base, current) in enumerate(pairs):
        values = transitions[(base.version, current.version)]
        for col, (key, title, cmap, _) in enumerate(fields):
            axis = axes[row, col]
            decorate_map(axis, background)
            vmin, vmax = limits[key]
            axis.scatter(
                positions[:, 0], positions[:, 1], c=values[key], s=18,
                cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0, zorder=2,
            )
            if row == 0:
                axis.set_title(title)
            if col == 0:
                axis.set_ylabel(
                    f"v{base.version}->v{current.version}\n"
                    f"online {current.online_score - base.online_score:+.6f}\ny (m)"
                )
            if row == len(pairs) - 1:
                axis.set_xlabel("x (m)")
    for col, (key, _, cmap, _) in enumerate(fields):
        color = plt.cm.ScalarMappable(norm=Normalize(*limits[key]), cmap=cmap)
        fig.colorbar(color, ax=axes[:, col], fraction=0.013, pad=0.008)
    fig.suptitle(
        "What changed at each online improvement\n"
        "Green means lower local reference error; first column shows action magnitude",
        fontsize=15,
    )
    fig.savefig(TRANSITION_FIGURE, dpi=180)
    plt.close(fig)


def plot_summary(
    version_rows: list[dict[str, object]],
    transition_rows: list[dict[str, object]],
) -> None:
    labels = [f"v{row['version']}" for row in version_rows]
    online = np.asarray([row["online_score"] for row in version_rows], dtype=float)
    local = np.asarray([row["local_score_traceback"] for row in version_rows], dtype=float)
    x = np.arange(len(labels))
    transition_labels = [str(row["transition"]) for row in transition_rows]
    tx = np.arange(len(transition_labels))

    fig, axes = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
    axis = axes[0, 0]
    online_line = axis.plot(x, online, marker="o", linewidth=2, color="#b83a2d", label="online")[0]
    axis.set_ylabel("Online score", color="#b83a2d")
    axis.tick_params(axis="y", labelcolor="#b83a2d")
    axis.set_xticks(x, labels)
    twin = axis.twinx()
    local_line = twin.plot(x, local, marker="s", linewidth=2, color="#2b67a5", label="local reference")[0]
    twin.set_ylabel("Local v207-reference score", color="#2b67a5")
    twin.tick_params(axis="y", labelcolor="#2b67a5")
    axis.legend([online_line, local_line], ["online", "local reference"], loc="upper left")
    axis.set_title("Online and local-reference evolution")
    axis.grid(alpha=0.25)

    axis = axes[0, 1]
    for key, label, color in (
        ("pdp", "PDP", "#337a4f"),
        ("leaked_sh", "Leaked PAS", "#7c4a9e"),
        ("nmse_score", "NMSE score", "#c47d20"),
    ):
        axis.plot(
            x, [row[key] for row in version_rows], marker="o", linewidth=2,
            label=label, color=color,
        )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Local metric")
    axis.set_title("Which local metric moved")
    axis.legend()
    axis.grid(alpha=0.25)

    axis = axes[1, 0]
    width = 0.20
    bars = (
        ("online_delta", "Online", "#b83a2d"),
        ("mean_delta_pdp", "PDP", "#337a4f"),
        ("mean_delta_leaked_sh", "Leaked PAS", "#7c4a9e"),
        ("mean_delta_nmse_score", "Point NMSE score", "#c47d20"),
    )
    for index, (key, label, color) in enumerate(bars):
        axis.bar(
            tx + (index - 1.5) * width,
            [row[key] for row in transition_rows],
            width=width,
            label=label,
            color=color,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(tx, transition_labels, rotation=25, ha="right")
    axis.set_ylabel("Delta")
    axis.set_title("Per-transition gains")
    axis.legend(ncols=2)
    axis.grid(axis="y", alpha=0.25)

    matrix = np.asarray(
        [
            [row["by_regime"][regime]["point_proxy"] for regime in REGIMES]
            for row in transition_rows
        ],
        dtype=np.float64,
    )
    bound = max(float(np.quantile(np.abs(matrix), 0.98)), 1e-6)
    axis = axes[1, 1]
    image = axis.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-bound, vmax=bound)
    axis.set_xticks(range(len(REGIMES)), REGIMES)
    axis.set_yticks(range(len(transition_labels)), transition_labels)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, f"{matrix[row, col]:+.3f}", ha="center", va="center", fontsize=8)
    axis.set_title("Point diagnostic gain by regime")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle(
        "v189-v212 error evolution summary (v207-reference diagnostic)", fontsize=15
    )
    fig.savefig(SUMMARY_FIGURE, dpi=180)
    plt.close(fig)


def plot_v212_remaining(
    point: dict[int, dict[str, np.ndarray]], positions: np.ndarray
) -> None:
    background = map_background()
    fields = (
        (1.0 - point[212]["pdp"], "v212 PDP deficit", "magma", False),
        (1.0 - point[212]["leaked_sh"], "v212 leaked-PAS deficit", "magma", False),
        (point[212]["log10_point_nmse"], "v212 log10 point NMSE", "magma", False),
        (point[212]["abs_log_power_error"], "v212 |log power error|", "magma", False),
        (point[212]["pdp"] - point[189]["pdp"], "v189->v212 PDP gain", "RdYlGn", True),
        (
            point[212]["point_proxy"] - point[189]["point_proxy"],
            "v189->v212 diagnostic gain",
            "RdYlGn",
            True,
        ),
    )
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharex=True, sharey=True, constrained_layout=True)
    for axis, (values, title, cmap, symmetric) in zip(axes.flat, fields):
        decorate_map(axis, background)
        vmin, vmax = quantile_limits([values], symmetric=symmetric)
        scatter = axis.scatter(
            positions[:, 0], positions[:, 1], c=values, s=22, cmap=cmap,
            vmin=vmin, vmax=vmax, linewidths=0, zorder=2,
        )
        axis.set_title(title)
        fig.colorbar(scatter, ax=axis, fraction=0.046, pad=0.03)
    for axis in axes[-1]:
        axis.set_xlabel("x (m)")
    for axis in axes[:, 0]:
        axis.set_ylabel("y (m)")
    fig.suptitle(
        "v212 remaining local-reference errors and total change from v189\n"
        "Not hidden online error",
        fontsize=15,
    )
    fig.savefig(V212_FIGURE, dpi=180)
    plt.close(fig)


def write_outputs(
    point: dict[int, dict[str, np.ndarray]],
    transitions: dict[tuple[int, int], dict[str, np.ndarray]],
    version_rows: list[dict[str, object]],
    transition_rows: list[dict[str, object]],
    positions: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
) -> None:
    truth_energy = np.load(v296.TRUTH_ENERGY, mmap_mode="r")
    nonzero_truth = np.asarray(truth_energy) > 1e-25
    zero_truth_indices = np.flatnonzero(~nonzero_truth).astype(int).tolist()
    version_fields = list(version_rows[0].keys())
    with VERSION_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=version_fields)
        writer.writeheader()
        writer.writerows(version_rows)

    flat_transition_rows = [
        {key: value for key, value in row.items() if key not in ("by_regime", "top_improved_indices", "top_worsened_indices")}
        for row in transition_rows
    ]
    with TRANSITION_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_transition_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_transition_rows)

    point_version_fields = [
        "version", "index", "x", "y", "blocked", "regime", *POINT_FIELDS
    ]
    with POINT_VERSION_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=point_version_fields)
        writer.writeheader()
        for spec in VERSIONS:
            for index in range(EXPECTED_SHAPE[0]):
                row: dict[str, object] = {
                    "version": spec.version,
                    "index": index,
                    "x": float(positions[index, 0]),
                    "y": float(positions[index, 1]),
                    "blocked": float(blocked[index]),
                    "regime": str(regimes[index]),
                }
                row.update({field: float(point[spec.version][field][index]) for field in POINT_FIELDS})
                writer.writerow(row)

    transition_fields = [
        "transition", "index", "x", "y", "blocked", "regime",
        "change_nmse", "log10_change_nmse", "log_power_ratio",
        "delta_pdp", "delta_leaked_sh", "delta_nmse_score", "delta_point_proxy",
    ]
    with POINT_TRANSITION_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=transition_fields)
        writer.writeheader()
        for base, current in zip(VERSIONS[:-1], VERSIONS[1:]):
            key = (base.version, current.version)
            values = transitions[key]
            for index in range(EXPECTED_SHAPE[0]):
                writer.writerow(
                    {
                        "transition": f"v{base.version}->v{current.version}",
                        "index": index,
                        "x": float(positions[index, 0]),
                        "y": float(positions[index, 1]),
                        "blocked": float(blocked[index]),
                        "regime": str(regimes[index]),
                        **{field: float(values[field][index]) for field in transition_fields[6:]},
                    }
                )

    arrays: dict[str, np.ndarray] = {
        "positions": positions.astype(np.float32),
        "blocked": blocked.astype(np.float32),
        "regimes": regimes.astype(str),
    }
    for spec in VERSIONS:
        arrays.update(
            {f"v{spec.version}_{field}": value.astype(np.float32) for field, value in point[spec.version].items()}
        )
    for key, values in transitions.items():
        prefix = f"v{key[0]}_to_v{key[1]}"
        arrays.update({f"{prefix}_{field}": value.astype(np.float32) for field, value in values.items()})
    np.savez_compressed(POINT_METRICS, **arrays)

    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "A common per-point reference-error decomposition can show which spatial regions and "
            "which PDP/leaked-PAS/NMSE factors changed at every online improvement from v189 to v212."
        ),
        "interpretation_boundary": {
            "local_truth": PATHS.relative(v208.TEST_TRUTH),
            "warning": (
                "The local truth is the same v207 Sionna reference used directly as normalized "
                "PDP guidance by v210-v212. Their local improvement is therefore reference alignment, "
                "not an independent estimate of hidden online point errors."
            ),
            "valid_use": [
                "locate which test coordinates each version changed",
                "separate PDP, leaked-PAS, NMSE and power effects",
                "compare spatial concentration with observed online score deltas",
            ],
            "invalid_use": [
                "claim these are official hidden-test point errors",
                "select a new strict model solely from this local reference",
            ],
        },
        "versions": version_rows,
        "transitions": transition_rows,
        "regime_counts": {
            regime: int(np.sum(regimes == regime)) for regime in REGIMES
        },
        "local_truth_zero_indices": zero_truth_indices,
        "v212_remaining_error": {
            "pdp_deficit": summarize(1.0 - point[212]["pdp"]),
            "leaked_pas_deficit": summarize(1.0 - point[212]["leaked_sh"]),
            "point_nmse_nonzero_truth": summarize(
                point[212]["point_nmse"][nonzero_truth]
            ),
            "abs_log_power_error_nonzero_truth": summarize(
                point[212]["abs_log_power_error"][nonzero_truth]
            ),
            "truth_zero_point_nmse": point[212]["point_nmse"][~nonzero_truth].tolist(),
            "by_regime": {
                regime: {
                    "pdp": float(np.mean(point[212]["pdp"][regimes == regime])),
                    "leaked_sh": float(np.mean(point[212]["leaked_sh"][regimes == regime])),
                    "point_proxy": float(np.mean(point[212]["point_proxy"][regimes == regime])),
                }
                for regime in REGIMES
            },
        },
        "outputs": {
            "report": PATHS.relative(REPORT),
            "version_csv": PATHS.relative(VERSION_CSV),
            "transition_csv": PATHS.relative(TRANSITION_CSV),
            "point_version_csv": PATHS.relative(POINT_VERSION_CSV),
            "point_transition_csv": PATHS.relative(POINT_TRANSITION_CSV),
            "point_metrics": PATHS.relative(POINT_METRICS),
            "absolute_error_maps": PATHS.relative(ABSOLUTE_FIGURE),
            "transition_delta_maps": PATHS.relative(TRANSITION_FIGURE),
            "summary_figure": PATHS.relative(SUMMARY_FIGURE),
            "v212_remaining_map": PATHS.relative(V212_FIGURE),
        },
        "submission_zip": None,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    validate_inputs()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    positions = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    geometry = np.load(GEOMETRY)
    blocked = geometry["test_blocked"].astype(np.float64)
    regimes = regime_of(blocked)
    point, transitions, version_rows, transition_rows = evaluate_chain()
    plot_absolute_maps(point, positions)
    plot_transition_maps(transitions, positions)
    plot_summary(version_rows, transition_rows)
    plot_v212_remaining(point, positions)
    write_outputs(
        point, transitions, version_rows, transition_rows, positions, blocked, regimes
    )
    print(json.dumps({
        "report": PATHS.relative(REPORT),
        "figures": [
            PATHS.relative(ABSOLUTE_FIGURE),
            PATHS.relative(TRANSITION_FIGURE),
            PATHS.relative(SUMMARY_FIGURE),
            PATHS.relative(V212_FIGURE),
        ],
        "v212_local_score": next(
            row["local_score_traceback"] for row in version_rows if row["version"] == 212
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
