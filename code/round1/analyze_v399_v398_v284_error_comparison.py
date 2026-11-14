#!/usr/bin/env python3
"""Compare v398 with the stronger strict v284 using OOF truth and test actions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from powerfield_common import BS_POS, regime_of
from project_paths import PATHS


VERSION_NUMBER = 399
VERSION = "v399_v398_v284_error_comparison"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
FIGURE_DIR = REPORT_DIR / "版本三九八与版本二八四误差对照图"
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CSV = REPORT_DIR / "v399_oof_point_metrics.csv"
TEST_CSV = REPORT_DIR / "v399_test_action_metrics.csv"

OOF_CACHE = (
    PATHS.cache_dir(397) / "v397_confocal_v239_carrier_transfer_oof.npz"
)
TEST_TARGETS = PATHS.cache_dir(398) / "v398_relative_confocal_targets.npz"
V284_H = PATHS.cache_dir(284) / "v284_Round1_Test_Channel.npy"
V398_H = PATHS.cache_dir(398) / "v398_Round1_Test_Channel.npy"
GEOMETRY = PATHS.shared_features / "height_geometry_cache.npz"
SCENE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"

ACTIVE_BINS = 24
CIRCULAR_SEAM = 96
TINY = 1e-30
REGIMES = ("clear", "light", "medium", "heavy")
REGIME_NAMES = {
    "clear": "无遮挡",
    "light": "轻遮挡",
    "medium": "中遮挡",
    "heavy": "重遮挡",
}
REGIME_COLORS = {
    "clear": "#2a9d8f",
    "light": "#e9c46a",
    "medium": "#f4a261",
    "heavy": "#c44536",
}
ONLINE_SCORES = {"v284": 0.513129, "v398": 0.510891}


def configure_chinese_font() -> None:
    families = ("Noto Sans CJK JP", "Droid Sans Fallback")
    selected = "DejaVu Sans"
    for family in families:
        try:
            font_path = font_manager.findfont(
                font_manager.FontProperties(family=family),
                fallback_to_default=False,
            )
            font_manager.fontManager.addfont(font_path)
            selected = font_manager.FontProperties(fname=font_path).get_name()
            break
        except ValueError:
            continue
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
        }
    )


def normalize_rows(value: np.ndarray) -> np.ndarray:
    result = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    return result / np.maximum(result.sum(axis=1, keepdims=True), TINY)


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(left * left, axis=1, dtype=np.float64)
        * np.sum(right * right, axis=1, dtype=np.float64)
    )
    return numerator / np.maximum(denominator, TINY)


def circular_emd(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.roll(left, -CIRCULAR_SEAM, axis=1)
    right = np.roll(right, -CIRCULAR_SEAM, axis=1)
    return np.mean(
        np.abs(np.cumsum(left, axis=1) - np.cumsum(right, axis=1)), axis=1
    )


def active_centroid(value: np.ndarray) -> np.ndarray:
    active = value[:, :ACTIVE_BINS]
    bins = np.arange(ACTIVE_BINS, dtype=np.float64)
    return np.sum(active * bins[None, :], axis=1) / np.maximum(
        active.sum(axis=1), TINY
    )


def summarize(value: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(value, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p10": float(np.quantile(finite, 0.10)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.90)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if np.sum(valid) < 3 or np.std(left[valid]) <= 0 or np.std(right[valid]) <= 0:
        return 0.0
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def validate_inputs() -> None:
    required = (
        OOF_CACHE,
        TEST_TARGETS,
        V284_H,
        V398_H,
        GEOMETRY,
        SCENE,
        PATHS.round1 / "Round1_Train_Pos.npy",
        PATHS.round1 / "Round1_Test_Pos.npy",
    )
    missing = [PATHS.relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing v399 inputs: {missing}")


def map_background() -> dict[str, object]:
    scene = np.load(SCENE)
    wall = np.log1p(scene["grid_vertical_count"].astype(np.float32))
    x_min = float(scene["x_min"])
    y_min = float(scene["y_min"])
    resolution = float(scene["resolution"])
    positive = wall[wall > 0]
    return {
        "wall": wall,
        "extent": (
            x_min,
            x_min + wall.shape[1] * resolution,
            y_min,
            y_min + wall.shape[0] * resolution,
        ),
        "vmax": max(float(np.quantile(positive, 0.98)), 1.0),
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
        [BS_POS[0]], [BS_POS[1]], marker="*", s=52, color="black", zorder=4
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("横坐标（米）")
    axis.set_ylabel("纵坐标（米）")


def robust_limits(values: np.ndarray, symmetric: bool = False) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if symmetric:
        bound = max(float(np.quantile(np.abs(values), 0.98)), 1e-8)
        return -bound, bound
    low, high = np.quantile(values, (0.02, 0.98))
    return float(low), float(max(high, low + 1e-8))


def scatter_metric(
    axis: plt.Axes,
    positions: np.ndarray,
    values: np.ndarray,
    background: dict[str, object],
    title: str,
    cmap: str,
    *,
    symmetric: bool = False,
    fixed_limits: tuple[float, float] | None = None,
) -> None:
    decorate_map(axis, background)
    vmin, vmax = (
        fixed_limits
        if fixed_limits is not None
        else robust_limits(values, symmetric=symmetric)
    )
    points = axis.scatter(
        positions[:, 0],
        positions[:, 1],
        c=values,
        s=12,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        zorder=2,
    )
    axis.set_title(title)
    plt.colorbar(points, ax=axis, fraction=0.046, pad=0.03)


def load_oof_metrics() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    cache = np.load(OOF_CACHE)
    ids = cache["ids"].astype(np.int64)
    truth = normalize_rows(cache["truth"])
    carrier = normalize_rows(cache["carrier"])
    full = normalize_rows(cache["full"])
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    geometry = np.load(GEOMETRY)
    blocked = geometry["train_blocked"][ids].astype(np.float64)
    regimes = regime_of(blocked).astype("U8")

    carrier_cos = row_cosine(truth, carrier)
    full_cos = row_cosine(truth, full)
    carrier_emd = circular_emd(truth, carrier)
    full_emd = circular_emd(truth, full)
    carrier_l1 = np.sum(np.abs(truth - carrier), axis=1)
    full_l1 = np.sum(np.abs(truth - full), axis=1)
    truth_center = active_centroid(truth)
    carrier_center_error = np.abs(active_centroid(carrier) - truth_center)
    full_center_error = np.abs(active_centroid(full) - truth_center)
    tail_error_difference = np.sum(
        np.abs(full[:, ACTIVE_BINS:] - carrier[:, ACTIVE_BINS:]), axis=1
    )

    metrics = {
        "ids": ids,
        "fold": cache["fold"].astype(np.int8),
        "positions": train_pos[ids],
        "blocked": blocked,
        "regime": regimes,
        "carrier_cosine": carrier_cos,
        "full_cosine": full_cos,
        "cosine_gain": full_cos - carrier_cos,
        "carrier_error": 1.0 - carrier_cos,
        "full_error": 1.0 - full_cos,
        "carrier_emd": carrier_emd,
        "full_emd": full_emd,
        "emd_gain": carrier_emd - full_emd,
        "carrier_l1": carrier_l1,
        "full_l1": full_l1,
        "l1_gain": carrier_l1 - full_l1,
        "carrier_center_error": carrier_center_error,
        "full_center_error": full_center_error,
        "center_error_gain": carrier_center_error - full_center_error,
        "action_l1": np.sum(np.abs(full - carrier), axis=1),
        "tail_change_l1": tail_error_difference,
    }
    summary: dict[str, object] = {
        "point_count": int(len(ids)),
        "unique_point_count": int(len(np.unique(ids))),
        "cosine": {
            "carrier_mean": float(np.mean(carrier_cos)),
            "full_mean": float(np.mean(full_cos)),
            "mean_gain": float(np.mean(full_cos - carrier_cos)),
            "point_win_rate": float(np.mean(full_cos > carrier_cos)),
        },
        "circular_emd": {
            "carrier_mean": float(np.mean(carrier_emd)),
            "full_mean": float(np.mean(full_emd)),
            "mean_absolute_gain": float(np.mean(carrier_emd - full_emd)),
            "relative_reduction": float(
                (np.mean(carrier_emd) - np.mean(full_emd))
                / max(np.mean(carrier_emd), TINY)
            ),
            "point_win_rate": float(np.mean(full_emd < carrier_emd)),
        },
        "l1": {
            "carrier_mean": float(np.mean(carrier_l1)),
            "full_mean": float(np.mean(full_l1)),
            "mean_gain": float(np.mean(carrier_l1 - full_l1)),
            "point_win_rate": float(np.mean(full_l1 < carrier_l1)),
        },
        "active_centroid_error": {
            "carrier_mean": float(np.mean(carrier_center_error)),
            "full_mean": float(np.mean(full_center_error)),
            "mean_gain": float(np.mean(carrier_center_error - full_center_error)),
            "point_win_rate": float(np.mean(full_center_error < carrier_center_error)),
        },
        "action_l1": summarize(metrics["action_l1"]),
        "action_gain_pearson": safe_corr(metrics["action_l1"], metrics["cosine_gain"]),
        "base_error_gain_pearson": safe_corr(
            metrics["carrier_error"], metrics["cosine_gain"]
        ),
        "tail_change_l1_max": float(np.max(tail_error_difference)),
        "by_regime": {},
    }
    for regime in REGIMES:
        mask = regimes == regime
        summary["by_regime"][regime] = {
            "count": int(np.sum(mask)),
            "cosine_gain": float(np.mean(metrics["cosine_gain"][mask])),
            "cosine_win_rate": float(np.mean(metrics["cosine_gain"][mask] > 0)),
            "emd_gain": float(np.mean(metrics["emd_gain"][mask])),
            "action_l1": float(np.mean(metrics["action_l1"][mask])),
        }
    return metrics, summary


def unique_point_audit(
    metrics: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    ids = metrics["ids"].astype(np.int64)
    unique_ids, first, inverse, counts = np.unique(
        ids, return_index=True, return_inverse=True, return_counts=True
    )

    def aggregate(field: str) -> np.ndarray:
        values = np.asarray(metrics[field], dtype=np.float64)
        return np.bincount(inverse, weights=values) / counts

    gain = aggregate("cosine_gain")
    action = aggregate("action_l1")
    gain_second = np.bincount(
        inverse, weights=np.asarray(metrics["cosine_gain"], dtype=np.float64) ** 2
    ) / counts
    gain_std = np.sqrt(np.maximum(gain_second - gain**2, 0.0))
    positive_rate = np.bincount(
        inverse, weights=(metrics["cosine_gain"] > 0).astype(np.float64)
    ) / counts
    sign_consistency = np.maximum(positive_rate, 1.0 - positive_rate)
    audit = {
        "ids": unique_ids,
        "positions": metrics["positions"][first],
        "blocked": metrics["blocked"][first],
        "regime": metrics["regime"][first],
        "repeat_count": counts.astype(np.float64),
        "mean_cosine_gain": gain,
        "mean_action_l1": action,
        "cosine_gain_std": gain_std,
        "sign_consistency": sign_consistency,
    }
    repeated = counts > 1
    summary: dict[str, object] = {
        "occurrence_count": int(len(ids)),
        "unique_point_count": int(len(unique_ids)),
        "duplicate_occurrence_fraction": float(1.0 - len(unique_ids) / len(ids)),
        "repeated_unique_point_fraction": float(np.mean(repeated)),
        "max_repeat_count": int(np.max(counts)),
        "repeat_count_histogram": {
            str(value): int(np.sum(counts == value))
            for value in range(1, int(np.max(counts)) + 1)
        },
        "unique_weighted_cosine_gain": float(np.mean(gain)),
        "unique_weighted_point_win_rate": float(np.mean(gain > 0)),
        "repeated_point_mean_sign_consistency": float(
            np.mean(sign_consistency[repeated])
        ),
        "validation_design": (
            "five independently seeded test-matched holdouts; validation point sets "
            "overlap and are not five mutually exclusive folds"
        ),
    }
    return audit, summary


def full_h_action_metrics() -> dict[str, np.ndarray | float]:
    base = np.load(V284_H, mmap_mode="r")
    current = np.load(V398_H, mmap_mode="r")
    count = len(base)
    action_nmse = np.empty(count, dtype=np.float64)
    power_ratio = np.empty(count, dtype=np.float64)
    pas_h_l1 = np.empty(count, dtype=np.float64)
    for start in range(0, count, 8):
        end = min(start + 8, count)
        left = np.asarray(base[start:end], dtype=np.complex64)
        right = np.asarray(current[start:end], dtype=np.complex64)
        left_power = np.sum(np.abs(left) ** 2, axis=(1, 2, 3), dtype=np.float64)
        right_power = np.sum(np.abs(right) ** 2, axis=(1, 2, 3), dtype=np.float64)
        action_nmse[start:end] = np.sum(
            np.abs(right - left) ** 2, axis=(1, 2, 3), dtype=np.float64
        ) / np.maximum(left_power, TINY)
        power_ratio[start:end] = right_power / np.maximum(left_power, TINY)

        left_cir = np.fft.ifft(left, axis=-1).reshape(-1, 2, 16, 8, 4, 192)
        right_cir = np.fft.ifft(right, axis=-1).reshape(-1, 2, 16, 8, 4, 192)
        left_ad = np.fft.fft(left_cir, axis=2)
        right_ad = np.fft.fft(right_cir, axis=2)
        left_pas = np.sum(
            np.abs(left_ad) ** 2, axis=(1, 3, 4, 5), dtype=np.float64
        )
        right_pas = np.sum(
            np.abs(right_ad) ** 2, axis=(1, 3, 4, 5), dtype=np.float64
        )
        left_pas /= np.maximum(left_pas.sum(axis=1, keepdims=True), TINY)
        right_pas /= np.maximum(right_pas.sum(axis=1, keepdims=True), TINY)
        pas_h_l1[start:end] = np.sum(np.abs(right_pas - left_pas), axis=1)
    return {
        "action_nmse": action_nmse,
        "power_ratio": power_ratio,
        "pas_h_l1": pas_h_l1,
        "max_abs_power_ratio_error": float(np.max(np.abs(power_ratio - 1.0))),
    }


def load_test_metrics(
    oof_action: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    targets = np.load(TEST_TARGETS)
    current_d = normalize_rows(targets["current_d"])
    target_d = normalize_rows(targets["target_d"])
    positions = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    geometry = np.load(GEOMETRY)
    blocked = geometry["test_blocked"].astype(np.float64)
    regimes = regime_of(blocked).astype("U8")
    h_action = full_h_action_metrics()

    action_l1 = np.sum(np.abs(target_d - current_d), axis=1)
    action_cosine_distance = 1.0 - row_cosine(target_d, current_d)
    centroid_shift = active_centroid(target_d) - active_centroid(current_d)
    peak_shift = (
        np.argmax(target_d[:, :ACTIVE_BINS], axis=1)
        - np.argmax(current_d[:, :ACTIVE_BINS], axis=1)
    ).astype(np.float64)
    tail_change = np.sum(
        np.abs(target_d[:, ACTIVE_BINS:] - current_d[:, ACTIVE_BINS:]), axis=1
    )

    metrics = {
        "ids": np.arange(len(positions), dtype=np.int64),
        "positions": positions,
        "blocked": blocked,
        "regime": regimes,
        "action_l1": action_l1,
        "action_cosine_distance": action_cosine_distance,
        "centroid_shift": centroid_shift,
        "peak_shift": peak_shift,
        "tail_change_l1": tail_change,
        "h_action_nmse": np.asarray(h_action["action_nmse"]),
        "power_ratio": np.asarray(h_action["power_ratio"]),
        "pas_h_l1": np.asarray(h_action["pas_h_l1"]),
    }
    oof_p90 = float(np.quantile(oof_action, 0.90))
    summary: dict[str, object] = {
        "point_count": int(len(positions)),
        "action_l1": summarize(action_l1),
        "action_cosine_distance": summarize(action_cosine_distance),
        "centroid_shift": summarize(centroid_shift),
        "absolute_centroid_shift": summarize(np.abs(centroid_shift)),
        "peak_shift_nonzero_rate": float(np.mean(peak_shift != 0)),
        "tail_change_l1_max": float(np.max(tail_change)),
        "h_action_nmse": summarize(metrics["h_action_nmse"]),
        "pas_h_l1": summarize(metrics["pas_h_l1"]),
        "max_abs_power_ratio_error": h_action["max_abs_power_ratio_error"],
        "oof_action_p90": oof_p90,
        "fraction_above_oof_action_p90": float(np.mean(action_l1 > oof_p90)),
        "mean_action_ratio_test_to_oof": float(
            np.mean(action_l1) / max(np.mean(oof_action), TINY)
        ),
        "by_regime": {},
    }
    for regime in REGIMES:
        mask = regimes == regime
        summary["by_regime"][regime] = {
            "count": int(np.sum(mask)),
            "action_l1": float(np.mean(action_l1[mask])),
            "h_action_nmse": float(np.mean(metrics["h_action_nmse"][mask])),
            "pas_h_l1": float(np.mean(metrics["pas_h_l1"][mask])),
        }
    return metrics, summary


def write_point_csv(path: Path, metrics: dict[str, np.ndarray], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("point_id", "x", "y", "z", "blocked", "regime", *fields))
        for row, point_id in enumerate(metrics["ids"]):
            position = metrics["positions"][row]
            writer.writerow(
                (
                    int(point_id),
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                    float(metrics["blocked"][row]),
                    str(metrics["regime"][row]),
                    *(float(metrics[field][row]) for field in fields),
                )
            )


def plot_evidence_summary(
    oof: dict[str, np.ndarray],
    oof_summary: dict[str, object],
    test: dict[str, np.ndarray],
) -> Path:
    output = FIGURE_DIR / "一、线上结果与本地证据对照.png"
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    axes[0, 0].bar(
        ["版本二八四", "版本三九八"],
        [ONLINE_SCORES["v284"], ONLINE_SCORES["v398"]],
        color=["#2a9d8f", "#c44536"],
        width=0.58,
    )
    axes[0, 0].set_ylim(0.505, 0.516)
    axes[0, 0].set_ylabel("线上分数")
    axes[0, 0].set_title("线上：版本三九八低于版本二八四")
    for index, score in enumerate(ONLINE_SCORES.values()):
        axes[0, 0].text(index, score + 0.00025, f"{score:.6f}", ha="center")

    metrics = oof_summary
    labels = ["余弦相似度", "一减环形搬运误差"]
    carrier = [
        metrics["cosine"]["carrier_mean"],
        1.0 - metrics["circular_emd"]["carrier_mean"],
    ]
    full = [
        metrics["cosine"]["full_mean"],
        1.0 - metrics["circular_emd"]["full_mean"],
    ]
    x = np.arange(len(labels))
    axes[0, 1].bar(x - 0.18, carrier, 0.36, label="原载体", color="#7a8793")
    axes[0, 1].bar(x + 0.18, full, 0.36, label="共焦修正", color="#3a86a8")
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylim(min(carrier + full) - 0.01, 1.0)
    axes[0, 1].set_title("训练OOF：平均PDP指标确有改善")
    axes[0, 1].legend()

    gain = oof["cosine_gain"]
    bound = max(float(np.quantile(np.abs(gain), 0.99)), 1e-5)
    axes[1, 0].hist(gain, bins=55, color="#3a86a8", alpha=0.85)
    axes[1, 0].axvline(0.0, color="black", linewidth=1)
    axes[1, 0].set_xlim(-bound, bound)
    axes[1, 0].set_xlabel("逐点PDP余弦增益（正数表示改善）")
    axes[1, 0].set_ylabel("点数")
    axes[1, 0].set_title(
        f"OOF逐点胜率仅 {np.mean(gain > 0):.1%}，并非所有点改善"
    )

    max_action = max(
        float(np.quantile(oof["action_l1"], 0.99)),
        float(np.quantile(test["action_l1"], 0.99)),
    )
    bins = np.linspace(0.0, max_action, 55)
    axes[1, 1].hist(
        oof["action_l1"], bins=bins, density=True, alpha=0.65, label="训练OOF"
    )
    axes[1, 1].hist(
        test["action_l1"], bins=bins, density=True, alpha=0.65, label="测试部署"
    )
    axes[1, 1].set_xlabel("PDP改动强度（L1距离）")
    axes[1, 1].set_ylabel("密度")
    axes[1, 1].set_title("OOF动作与测试动作的分布对照")
    axes[1, 1].legend()

    figure.suptitle("版本三九八为何本地证据为正、线上却下降", fontsize=15)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_oof_maps(oof: dict[str, np.ndarray], background: dict[str, object]) -> Path:
    output = FIGURE_DIR / "二、训练OOF逐点误差改善空间分布.png"
    figure, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    positions = oof["positions"]
    scatter_metric(
        axes[0, 0], positions, oof["carrier_error"], background, "修正前PDP余弦误差", "magma"
    )
    scatter_metric(
        axes[0, 1], positions, oof["full_error"], background, "修正后PDP余弦误差", "magma"
    )
    scatter_metric(
        axes[1, 0],
        positions,
        oof["cosine_gain"],
        background,
        "逐点PDP余弦增益（红色改善、蓝色变差）",
        "RdBu_r",
        symmetric=True,
    )
    scatter_metric(
        axes[1, 1], positions, oof["action_l1"], background, "共焦模块实际改动强度", "viridis"
    )
    figure.suptitle("训练OOF：共焦修正在哪里改善、在哪里变差", fontsize=15)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_test_maps(test: dict[str, np.ndarray], background: dict[str, object]) -> Path:
    output = FIGURE_DIR / "三、测试集相对版本二八四的实际改动.png"
    figure, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    positions = test["positions"]
    scatter_metric(
        axes[0, 0], positions, test["action_l1"], background, "PDP改动强度（L1距离）", "viridis"
    )
    scatter_metric(
        axes[0, 1],
        positions,
        test["centroid_shift"],
        background,
        "前二十四个时延格的能量中心位移",
        "RdBu_r",
        symmetric=True,
    )
    scatter_metric(
        axes[1, 0], positions, test["h_action_nmse"], background, "完整复信道相对改动量", "plasma"
    )
    scatter_metric(
        axes[1, 1], positions, test["pas_h_l1"], background, "水平角功率谱的间接改动量", "cividis"
    )
    figure.suptitle(
        "测试集只有改动图，没有隐藏真值，颜色不能解释为线上逐点误差",
        fontsize=14,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_regime_comparison(
    oof_summary: dict[str, object], test_summary: dict[str, object]
) -> Path:
    output = FIGURE_DIR / "四、不同遮挡分区的改善与部署动作.png"
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    labels = [REGIME_NAMES[value] for value in REGIMES]
    colors = [REGIME_COLORS[value] for value in REGIMES]
    oof_by = oof_summary["by_regime"]
    test_by = test_summary["by_regime"]

    axes[0, 0].bar(
        labels, [oof_by[value]["cosine_gain"] for value in REGIMES], color=colors
    )
    axes[0, 0].axhline(0.0, color="black", linewidth=1)
    axes[0, 0].set_ylabel("平均PDP余弦增益")
    axes[0, 0].set_title("训练OOF平均改善")

    axes[0, 1].bar(
        labels, [oof_by[value]["cosine_win_rate"] for value in REGIMES], color=colors
    )
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].set_ylabel("逐点胜率")
    axes[0, 1].set_title("训练OOF并非各分区都稳定")

    axes[1, 0].bar(
        labels, [oof_by[value]["action_l1"] for value in REGIMES], color=colors
    )
    axes[1, 0].set_ylabel("PDP改动强度")
    axes[1, 0].set_title("训练OOF动作强度")

    axes[1, 1].bar(
        labels, [test_by[value]["action_l1"] for value in REGIMES], color=colors
    )
    axes[1, 1].set_ylabel("PDP改动强度")
    axes[1, 1].set_title("测试部署动作强度")

    figure.suptitle("按遮挡分区检查本地收益能否迁移到测试", fontsize=15)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_scope_summary(
    oof_summary: dict[str, object], test_summary: dict[str, object]
) -> Path:
    output = FIGURE_DIR / "五、改善项与未改善项总览.png"
    figure, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)

    labels = ["PDP余弦", "环形搬运误差", "PDP一范数", "早期能量中心"]
    values = [
        oof_summary["cosine"]["point_win_rate"],
        oof_summary["circular_emd"]["point_win_rate"],
        oof_summary["l1"]["point_win_rate"],
        oof_summary["active_centroid_error"]["point_win_rate"],
    ]
    axes[0].barh(labels, values, color="#3a86a8")
    axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_xlabel("训练OOF逐点改善比例")
    axes[0].set_title("真正得到监督并出现改善的项目")
    for index, value in enumerate(values):
        axes[0].text(value + 0.015, index, f"{value:.1%}", va="center")

    axes[1].axis("off")
    lines = [
        "没有直接改善或没有真值可验证的项目",
        "",
        f"总功率：保持不变，最大相对误差 {test_summary['max_abs_power_ratio_error']:.2e}",
        f"后168个时延格：完全不改，最大L1变化 {test_summary['tail_change_l1_max']:.2e}",
        "路径相位：每个时延格只乘正实数，不学习新的相位",
        "极化与用户端口分配：每个时延格内保持版本二八四结构",
        (
            "水平角功率谱：没有独立目标，却被间接改变，"
            f"平均L1 {test_summary['pas_h_l1']['mean']:.4f}、最大 {test_summary['pas_h_l1']['max']:.4f}"
        ),
        "隐藏测试误差：没有真值，不能从空间改动图直接判断",
        "",
        f"最终线上：{ONLINE_SCORES['v284']:.6f} → {ONLINE_SCORES['v398']:.6f}",
        f"线上变化：{ONLINE_SCORES['v398'] - ONLINE_SCORES['v284']:+.6f}",
    ]
    axes[1].text(
        0.02,
        0.96,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=12,
        linespacing=1.55,
        bbox={"facecolor": "#f5f5f2", "edgecolor": "#b8b8b0", "boxstyle": "square,pad=0.8"},
    )
    axes[1].set_title("该模块的能力边界", pad=16)
    figure.suptitle("版本三九八改善了什么，没有改善什么", fontsize=15)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_oof_overlap_audit(
    audit: dict[str, np.ndarray],
    summary: dict[str, object],
    background: dict[str, object],
) -> Path:
    output = FIGURE_DIR / "六、OOF重复验证点与独立性审计.png"
    figure, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    positions = audit["positions"]
    scatter_metric(
        axes[0, 0],
        positions,
        audit["repeat_count"],
        background,
        "每个训练点进入验证集的次数",
        "viridis",
        fixed_limits=(1.0, float(summary["max_repeat_count"])),
    )
    scatter_metric(
        axes[0, 1],
        positions,
        audit["mean_cosine_gain"],
        background,
        "按唯一训练点平均后的PDP余弦增益",
        "RdBu_r",
        symmetric=True,
    )

    histogram = summary["repeat_count_histogram"]
    counts = np.asarray([int(key) for key in histogram], dtype=np.int64)
    point_counts = np.asarray([histogram[str(value)] for value in counts], dtype=np.int64)
    axes[1, 0].bar(counts, point_counts, color="#3a86a8")
    axes[1, 0].set_xticks(counts)
    axes[1, 0].set_xlabel("同一点被验证的次数")
    axes[1, 0].set_ylabel("唯一训练点数量")
    axes[1, 0].set_title(
        f"1587条记录仅覆盖 {summary['unique_point_count']} 个唯一点"
    )

    repeated = audit["repeat_count"] > 1
    axes[1, 1].hist(
        audit["sign_consistency"][repeated],
        bins=np.linspace(0.5, 1.0, 11),
        color="#e76f51",
        alpha=0.85,
    )
    axes[1, 1].set_xlim(0.5, 1.0)
    axes[1, 1].set_xlabel("重复验证时增益方向一致率")
    axes[1, 1].set_ylabel("重复训练点数量")
    axes[1, 1].set_title("重复点在不同随机留出中是否稳定")

    figure.suptitle(
        "所谓五折是五个有重叠的独立留出集，不是五个互斥折",
        fontsize=15,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def main() -> None:
    validate_inputs()
    configure_chinese_font()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    oof, oof_summary = load_oof_metrics()
    unique_audit, unique_summary = unique_point_audit(oof)
    test, test_summary = load_test_metrics(oof["action_l1"])
    background = map_background()

    write_point_csv(
        OOF_CSV,
        oof,
        (
            "carrier_cosine",
            "full_cosine",
            "cosine_gain",
            "carrier_emd",
            "full_emd",
            "emd_gain",
            "carrier_l1",
            "full_l1",
            "l1_gain",
            "carrier_center_error",
            "full_center_error",
            "center_error_gain",
            "action_l1",
            "tail_change_l1",
        ),
    )
    write_point_csv(
        TEST_CSV,
        test,
        (
            "action_l1",
            "action_cosine_distance",
            "centroid_shift",
            "peak_shift",
            "tail_change_l1",
            "h_action_nmse",
            "power_ratio",
            "pas_h_l1",
        ),
    )

    figures = [
        plot_evidence_summary(oof, oof_summary, test),
        plot_oof_maps(oof, background),
        plot_test_maps(test, background),
        plot_regime_comparison(oof_summary, test_summary),
        plot_scope_summary(oof_summary, test_summary),
        plot_oof_overlap_audit(unique_audit, unique_summary, background),
    ]
    report = {
        "method": VERSION,
        "status": "analysis_complete_no_candidate",
        "primary_hypothesis": (
            "The confocal OOF gain may be spatially selective and fail online because "
            "v398 changes only early relative D while preserving other v284 errors."
        ),
        "truth_boundary": {
            "oof_error_maps": "official train-H held-out truth from v397",
            "test_maps": "v398-v284 action only; hidden online truth is unavailable",
            "forbidden_interpretation": "test action intensity is not per-point online error",
        },
        "online_comparison": {
            "v284": ONLINE_SCORES["v284"],
            "v398": ONLINE_SCORES["v398"],
            "delta": ONLINE_SCORES["v398"] - ONLINE_SCORES["v284"],
        },
        "oof": oof_summary,
        "oof_unique_point_audit": unique_summary,
        "test_action": test_summary,
        "unchanged_by_design": {
            "point_total_power": True,
            "delay_bins_24_to_191": True,
            "complex_phase_within_each_delay_bin": True,
            "per_delay_bin_antenna_polarization_ue_allocation": True,
            "explicit_pas_target": False,
        },
        "identified_selection_gaps": [
            "v397 selected only on D/PDP metrics, while deployment also changed PAS indirectly",
            "v397 did not evaluate full-H NMSE or PAS collateral effects",
            "the five validation sets overlap, so 5/5 wins are not five mutually exclusive folds",
        ],
        "outputs": {
            "figures": [PATHS.relative(path) for path in figures],
            "oof_point_csv": PATHS.relative(OOF_CSV),
            "test_action_csv": PATHS.relative(TEST_CSV),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
