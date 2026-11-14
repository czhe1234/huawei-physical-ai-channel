#!/usr/bin/env python3
"""Visualize train-domain gaps between official spectra and physical predictions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from powerfield_common import BS_POS
from project_paths import PATHS


VERSION_NUMBER = 544
VERSION = "v544_train_physical_spectrum_gap_visualization"
OUTPUT_DIR = PATHS.analysis / "visualizations" / "训练集Sionna物理谱差异"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
DETAILS = OUTPUT_DIR / "训练集逐点谱差异.csv"

OOF = PATHS.cache_dir(543) / "v543_cellwise_fusion_oof.npz"
TRAIN_POS = PATHS.round1 / "Round1_Train_Pos.npy"
TRAIN_POWER = PATHS.shared_features / "train_total_power.npy"
SCENE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"
PATH_SPAN = PATHS.cache_dir(505) / "v505_path_span_point_metrics.npz"
DELAY_CONTROL = PATHS.cache_dir(508) / "v508_frequency_control_metrics.npz"
ANGLE_CONTROL = PATHS.cache_dir(510) / "v510_joint_delay_h_metrics.npz"
SPREAD_CONTROL = PATHS.cache_dir(511) / "v511_angular_spread_metrics.npz"

FIGURES = {
    "space": OUTPUT_DIR / "01_训练集逐点误差空间分布.png",
    "components": OUTPUT_DIR / "02_四类传播区域的误差来源.png",
    "spectra": OUTPUT_DIR / "03_官方真实谱与物理预测平均谱.png",
    "correction": OUTPUT_DIR / "04_模型修正与理想修正方向.png",
    "points": OUTPUT_DIR / "05_典型困难点逐点谱对照.png",
    "marginals": OUTPUT_DIR / "06_典型困难点PDP与PAS对照.png",
    "path_evidence": OUTPUT_DIR / "07_路径候选覆盖与角度分配证据.png",
}

REGIMES = ("clear", "light", "medium", "heavy")
REGIME_CN = {
    "clear": "无遮挡",
    "light": "轻遮挡",
    "medium": "中遮挡",
    "heavy": "重遮挡",
}
REGIME_INDEX_CN = {0: "无遮挡", 1: "轻遮挡", 2: "中遮挡", 3: "重遮挡"}
HEIGHT = 192
WIDTH = 16
TINY = 1e-30


def configure_chinese_font() -> None:
    selected = "DejaVu Sans"
    for family in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"):
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=family),
                fallback_to_default=False,
            )
            font_manager.fontManager.addfont(path)
            selected = font_manager.FontProperties(fname=path).get_name()
            break
        except ValueError:
            continue
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 260,
        }
    )


def validate_inputs() -> None:
    required = (
        OOF,
        TRAIN_POS,
        TRAIN_POWER,
        SCENE,
        PATH_SPAN,
        DELAY_CONTROL,
        ANGLE_CONTROL,
        SPREAD_CONTROL,
    )
    missing = [PATHS.relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing visualization inputs: {missing}")


def normalize_rows(value: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    return output / np.maximum(output.sum(axis=1, keepdims=True), TINY)


def summarize(value: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(value, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "p10": float(np.quantile(finite, 0.10)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.90)),
    }


def load_inputs() -> dict[str, np.ndarray]:
    oof = np.load(OOF)
    output = {key: np.asarray(oof[key]) for key in oof.files}
    output["positions"] = np.load(TRAIN_POS).astype(np.float32)
    output["power"] = np.load(TRAIN_POWER).astype(np.float64)
    output["zero"] = output["power"] == 0.0
    output["valid"] = ~output["zero"]
    if len(output["positions"]) != 2000 or int(np.sum(output["zero"])) != 16:
        raise ValueError("unexpected train point count or exact-zero count")
    for key in ("truth", "anchor", "calibrated", "cell_oracle"):
        if output[key].shape != (2000, HEIGHT * WIDTH):
            raise ValueError(f"unexpected {key} shape: {output[key].shape}")
    return output


def load_background() -> dict[str, object]:
    scene = np.load(SCENE)
    wall = np.log1p(scene["grid_vertical_count"].astype(np.float32))
    positive = wall[wall > 0]
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
        [BS_POS[0]],
        [BS_POS[1]],
        marker="*",
        s=110,
        color="#111827",
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("横坐标（米）")
    axis.set_ylabel("纵坐标（米）")


def robust_bounds(value: np.ndarray, symmetric: bool = False) -> tuple[float, float]:
    finite = np.asarray(value, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if symmetric:
        bound = max(float(np.quantile(np.abs(finite), 0.97)), 1e-5)
        return -bound, bound
    low, high = np.quantile(finite, (0.02, 0.98))
    return float(low), float(max(high, low + 1e-5))


def scatter_map(
    axis: plt.Axes,
    positions: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    zero: np.ndarray,
    background: dict[str, object],
    title: str,
    *,
    cmap: str,
    symmetric: bool = False,
) -> None:
    decorate_map(axis, background)
    vmin, vmax = robust_bounds(values[valid], symmetric=symmetric)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax) if symmetric else None
    points = axis.scatter(
        positions[valid, 0],
        positions[valid, 1],
        c=values[valid],
        s=18,
        cmap=cmap,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
        norm=norm,
        linewidth=0,
        alpha=0.88,
        rasterized=True,
        zorder=2,
    )
    if np.any(zero):
        axis.scatter(
            positions[zero, 0],
            positions[zero, 1],
            marker="x",
            s=30,
            color="#111827",
            linewidth=1.1,
            zorder=5,
            label="16个零功率点",
        )
    axis.set_title(title, fontweight="bold")
    plt.colorbar(points, ax=axis, fraction=0.045, pad=0.02)


def plot_spatial_maps(
    data: dict[str, np.ndarray], background: dict[str, object]
) -> None:
    valid = data["valid"]
    zero = data["zero"]
    position = data["positions"]
    metrics = (
        (data["anchor_weighted"], "18源物理锚点综合相似度", "viridis", False),
        (data["anchor_delay"], "18源物理锚点时延相似度", "viridis", False),
        (data["anchor_angle"], "18源物理锚点角度相似度", "viridis", False),
        (
            data["calibrated_weighted"] - data["anchor_weighted"],
            "逐格模型OOF增量",
            "RdBu_r",
            True,
        ),
        (
            data["oracle_weighted"] - data["anchor_weighted"],
            "候选逐格上限增量",
            "magma",
            False,
        ),
        (
            data["oracle_weighted"] - data["calibrated_weighted"],
            "模型仍未释放的候选余量",
            "magma",
            False,
        ),
    )
    figure, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    for axis, (values, title, cmap, symmetric) in zip(axes.flat, metrics):
        scatter_map(
            axis,
            position,
            values,
            valid,
            zero,
            background,
            title,
            cmap=cmap,
            symmetric=symmetric,
        )
    figure.suptitle(
        "全部2000个训练点：官方真实谱与当前物理预测的空间差异\n"
        "黑色叉号为16个零功率点，仅标记位置，不参与谱误差均值",
        fontsize=17,
        fontweight="bold",
    )
    figure.savefig(FIGURES["space"], bbox_inches="tight")
    plt.close(figure)


def plot_component_scores(data: dict[str, np.ndarray]) -> None:
    valid = data["valid"]
    methods = (
        ("anchor", "18源物理锚点", "#64748b"),
        ("calibrated", "逐格模型OOF", "#2563eb"),
        ("oracle", "候选逐格上限", "#dc2626"),
    )
    components = (
        ("joint", "联合时延-角度谱"),
        ("delay", "PDP时延边际"),
        ("angle", "PAS角度边际"),
        ("weighted", "综合谱指标"),
    )
    x = np.arange(len(REGIMES), dtype=np.float64)
    width = 0.24
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for axis, (component, title) in zip(axes.flat, components):
        for method_index, (prefix, label, color) in enumerate(methods):
            values = []
            for regime in REGIMES:
                mask = valid & (data["regimes"] == regime)
                values.append(float(np.mean(data[f"{prefix}_{component}"][mask])))
            bars = axis.bar(
                x + (method_index - 1) * width,
                values,
                width,
                label=label,
                color=color,
                alpha=0.90,
            )
            axis.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
        axis.set_xticks(x, [REGIME_CN[value] for value in REGIMES])
        axis.set_ylim(0.45, 1.015)
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylabel("余弦相似度（越高越好）")
        axis.set_title(title, fontweight="bold")
    axes[0, 0].legend(loc="lower left", ncol=1)
    figure.suptitle(
        "误差来源分解：当前物理锚点、逐格模型与候选上限",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(FIGURES["components"], bbox_inches="tight")
    plt.close(figure)


def spectrum_db(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float64), TINY)
    return 10.0 * np.log10(value / np.max(value))


def plot_average_spectra(data: dict[str, np.ndarray]) -> None:
    valid = data["valid"]
    figure, axes = plt.subplots(
        len(REGIMES), 4, figsize=(16, 14), constrained_layout=True
    )
    column_titles = (
        "官方真实平均谱",
        "18源物理锚点平均谱",
        "真实谱 - 物理锚点",
        "真实谱 - 候选逐格上限",
    )
    for row, regime in enumerate(REGIMES):
        mask = valid & (data["regimes"] == regime)
        truth = data["truth"][mask].reshape(-1, HEIGHT, WIDTH).mean(axis=0)
        anchor = data["anchor"][mask].reshape(-1, HEIGHT, WIDTH).mean(axis=0)
        oracle = data["cell_oracle"][mask].reshape(-1, HEIGHT, WIDTH).mean(axis=0)
        truth /= max(float(truth.sum()), TINY)
        anchor /= max(float(anchor.sum()), TINY)
        oracle /= max(float(oracle.sum()), TINY)
        panels = (
            spectrum_db(truth),
            spectrum_db(anchor),
            truth - anchor,
            truth - oracle,
        )
        for column, (axis, panel) in enumerate(zip(axes[row], panels)):
            if column < 2:
                image = axis.imshow(
                    panel,
                    origin="lower",
                    aspect="auto",
                    cmap="magma",
                    vmin=-35.0,
                    vmax=0.0,
                    interpolation="nearest",
                )
            else:
                bound = max(float(np.quantile(np.abs(panel), 0.995)), 1e-8)
                image = axis.imshow(
                    panel,
                    origin="lower",
                    aspect="auto",
                    cmap="RdBu_r",
                    norm=TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound),
                    interpolation="nearest",
                )
            if row == 0:
                axis.set_title(column_titles[column], fontweight="bold")
            if column == 0:
                axis.set_ylabel(f"{REGIME_CN[regime]}\n时延bin")
            if row == len(REGIMES) - 1:
                axis.set_xlabel("水平角度bin")
            plt.colorbar(image, ax=axis, fraction=0.045, pad=0.02)
    figure.suptitle(
        "四类传播区域的平均联合时延-角度谱\n"
        "差值图中红色表示物理预测缺少能量，蓝色表示物理预测能量过多",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(FIGURES["spectra"], bbox_inches="tight")
    plt.close(figure)


def plot_correction_direction(data: dict[str, np.ndarray]) -> None:
    valid = data["valid"]
    figure, axes = plt.subplots(
        len(REGIMES), 3, figsize=(14, 14), constrained_layout=True
    )
    column_titles = (
        "候选上限要求的修正",
        "逐格模型实际修正",
        "模型修正 - 候选上限修正",
    )
    for row, regime in enumerate(REGIMES):
        mask = valid & (data["regimes"] == regime)
        anchor = data["anchor"][mask].reshape(-1, HEIGHT, WIDTH)
        calibrated = data["calibrated"][mask].reshape(-1, HEIGHT, WIDTH)
        oracle = data["cell_oracle"][mask].reshape(-1, HEIGHT, WIDTH)
        desired = np.mean(oracle - anchor, axis=0)
        learned = np.mean(calibrated - anchor, axis=0)
        panels = (desired, learned, learned - desired)
        shared = max(
            float(
                np.quantile(
                    np.abs(np.concatenate([desired.ravel(), learned.ravel()])), 0.995
                )
            ),
            1e-8,
        )
        for column, (axis, panel) in enumerate(zip(axes[row], panels)):
            bound = (
                shared
                if column < 2
                else max(float(np.quantile(np.abs(panel), 0.995)), 1e-8)
            )
            image = axis.imshow(
                panel,
                origin="lower",
                aspect="auto",
                cmap="RdBu_r",
                norm=TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound),
                interpolation="nearest",
            )
            if row == 0:
                axis.set_title(column_titles[column], fontweight="bold")
            if column == 0:
                axis.set_ylabel(f"{REGIME_CN[regime]}\n时延bin")
            if row == len(REGIMES) - 1:
                axis.set_xlabel("水平角度bin")
            plt.colorbar(image, ax=axis, fraction=0.045, pad=0.02)
    figure.suptitle(
        "训练集上模型学到的修正方向与候选可实现方向",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(FIGURES["correction"], bbox_inches="tight")
    plt.close(figure)


def select_representatives(data: dict[str, np.ndarray]) -> dict[str, int]:
    selected: dict[str, int] = {}
    valid = data["valid"]
    model_gain = data["calibrated_weighted"] - data["anchor_weighted"]
    oracle_gain = data["oracle_weighted"] - data["anchor_weighted"]
    for regime in REGIMES:
        ids = np.flatnonzero(valid & (data["regimes"] == regime))
        missed = oracle_gain[ids] - np.maximum(model_gain[ids], 0.0)
        negative_bonus = (model_gain[ids] <= 0.0).astype(np.float32) * 0.05
        selected[regime] = int(ids[np.argmax(missed + negative_bonus)])
    return selected


def plot_representative_spectra(
    data: dict[str, np.ndarray], selected: dict[str, int]
) -> None:
    keys = ("truth", "anchor", "calibrated", "cell_oracle")
    titles = ("官方真实谱", "18源物理锚点", "逐格模型OOF", "候选逐格上限")
    figure, axes = plt.subplots(4, 4, figsize=(15, 13), constrained_layout=True)
    for row, regime in enumerate(REGIMES):
        point = selected[regime]
        for column, (key, title) in enumerate(zip(keys, titles)):
            panel = data[key][point].reshape(HEIGHT, WIDTH)
            image = axes[row, column].imshow(
                spectrum_db(panel),
                origin="lower",
                aspect="auto",
                cmap="magma",
                vmin=-35.0,
                vmax=0.0,
                interpolation="nearest",
            )
            if row == 0:
                axes[row, column].set_title(title, fontweight="bold")
            if row == 3:
                axes[row, column].set_xlabel("水平角度bin")
            if column == 0:
                position = data["positions"][point]
                axes[row, column].set_ylabel(
                    f"{REGIME_CN[regime]} 点{point}\n"
                    f"({position[0]:.1f}, {position[1]:.1f})\n时延bin"
                )
            plt.colorbar(image, ax=axes[row, column], fraction=0.045, pad=0.02)
        axes[row, 1].text(
            0.02,
            0.98,
            f"综合={data['anchor_weighted'][point]:.3f}",
            transform=axes[row, 1].transAxes,
            va="top",
            color="white",
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
        )
        axes[row, 2].text(
            0.02,
            0.98,
            f"综合={data['calibrated_weighted'][point]:.3f}",
            transform=axes[row, 2].transAxes,
            va="top",
            color="white",
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
        )
        axes[row, 3].text(
            0.02,
            0.98,
            f"综合={data['oracle_weighted'][point]:.3f}",
            transform=axes[row, 3].transAxes,
            va="top",
            color="white",
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
        )
    figure.suptitle(
        "四类传播区域中“候选有答案但模型没有充分选对”的典型训练点",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(FIGURES["points"], bbox_inches="tight")
    plt.close(figure)


def plot_representative_marginals(
    data: dict[str, np.ndarray], selected: dict[str, int]
) -> None:
    keys = ("truth", "anchor", "calibrated", "cell_oracle")
    labels = ("官方真实", "18源锚点", "逐格模型", "候选上限")
    colors = ("#111827", "#64748b", "#2563eb", "#dc2626")
    figure, axes = plt.subplots(4, 2, figsize=(15, 13), constrained_layout=True)
    for row, regime in enumerate(REGIMES):
        point = selected[regime]
        for key, label, color in zip(keys, labels, colors):
            grid = data[key][point].reshape(HEIGHT, WIDTH)
            delay = grid.sum(axis=1)
            angle = grid.sum(axis=0)
            delay /= max(float(delay.sum()), TINY)
            angle /= max(float(angle.sum()), TINY)
            axes[row, 0].plot(delay, label=label, color=color, linewidth=1.5)
            axes[row, 1].plot(
                angle,
                label=label,
                color=color,
                linewidth=1.7,
                marker="o",
                markersize=2.5,
            )
        axes[row, 0].set_ylabel(f"{REGIME_CN[regime]} 点{point}\n归一化能量")
        axes[row, 0].set_title("PDP：能量落在哪个时延", fontweight="bold")
        axes[row, 1].set_title("PAS：能量落在哪个水平角度", fontweight="bold")
        axes[row, 0].grid(alpha=0.22)
        axes[row, 1].grid(alpha=0.22)
        axes[row, 1].set_xticks(np.arange(WIDTH))
    axes[-1, 0].set_xlabel("时延bin")
    axes[-1, 1].set_xlabel("水平角度bin")
    axes[0, 0].legend(ncol=2)
    figure.suptitle(
        "典型困难训练点的PDP与PAS边际对照",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(FIGURES["marginals"], bbox_inches="tight")
    plt.close(figure)


def regime_means(values: np.ndarray, regimes: np.ndarray) -> list[float]:
    return [float(np.mean(values[regimes == index])) for index in range(4)]


def plot_path_evidence() -> None:
    span = np.load(PATH_SPAN)
    delay = np.load(DELAY_CONTROL)
    angle = np.load(ANGLE_CONTROL)
    spread = np.load(SPREAD_CONTROL)
    x = np.arange(4, dtype=np.float64)
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    for offset, (key, label, color) in enumerate(
        (
            ("raw_pdp_cosine", "原始Sionna", "#64748b"),
            ("oracle_pdp_cosine", "只重估路径能量", "#2563eb"),
            ("pdp_support_coverage", "候选支撑覆盖", "#dc2626"),
        )
    ):
        axes[0, 0].bar(
            x + (offset - 1) * 0.24,
            regime_means(span[key], span["regimes"]),
            0.24,
            label=label,
            color=color,
        )
    axes[0, 0].set_title("PDP：候选路径位置与路径能量的差距", fontweight="bold")
    axes[0, 0].set_xticks(x, [REGIME_INDEX_CN[index] for index in range(4)])
    axes[0, 0].set_ylim(0.35, 1.02)
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.22)

    for offset, (key, label, color) in enumerate(
        (
            ("raw_joint_sh_cosine", "原始Sionna", "#64748b"),
            ("oracle_joint_sh_cosine", "只重估路径能量", "#2563eb"),
            ("joint_sh_support_coverage", "候选支撑覆盖", "#dc2626"),
        )
    ):
        axes[0, 1].bar(
            x + (offset - 1) * 0.24,
            regime_means(span[key], span["regimes"]),
            0.24,
            label=label,
            color=color,
        )
    axes[0, 1].set_title("联合时延-角度谱：角度/端口仍是主要缺口", fontweight="bold")
    axes[0, 1].set_xticks(x, [REGIME_INDEX_CN[index] for index in range(4)])
    axes[0, 1].set_ylim(0.25, 1.02)
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.22)

    heavy = delay["regimes"] == 3
    path_index = int(np.flatnonzero(delay["path_limits"] == 64)[0])
    cv_index = int(np.flatnonzero(delay["cv_names"] == "contiguous_4fold")[0])
    delay_values = [
        float(np.mean(delay["traceback_like"][index, path_index, cv_index, heavy]))
        for index in range(len(delay["source_names"]))
    ]
    delay_labels = ("真实候选", "同类点打乱", "+3bin", "均匀网格", "随机网格")
    axes[1, 0].bar(
        np.arange(len(delay_values)),
        delay_values,
        color=("#16a34a", "#94a3b8", "#dc2626", "#f59e0b", "#a855f7"),
    )
    axes[1, 0].set_xticks(np.arange(len(delay_values)), delay_labels, rotation=18)
    axes[1, 0].set_ylim(0.35, 1.02)
    axes[1, 0].set_ylabel("连续频段留出综合分")
    axes[1, 0].set_title("重遮挡点：真实时延候选显著优于错误候选", fontweight="bold")
    axes[1, 0].grid(axis="y", alpha=0.22)

    heavy_angle = angle["regimes"] == 3
    angle_values = [
        float(np.mean(value[heavy_angle])) for value in angle["traceback_like"]
    ]
    angle_labels = ("真实角度K32", "真实角度K64", "角度+3bin", "时延角度打乱")
    heavy_spread = spread["regimes"] == 3
    spread_values = [
        float(np.mean(value[heavy_spread])) for value in spread["traceback_like"]
    ]
    combined_values = angle_values + spread_values[:4]
    combined_labels = angle_labels + ("精确角度", "扩散0.5", "扩散1", "扩散2")
    axes[1, 1].bar(
        np.arange(len(combined_values)),
        combined_values,
        color=["#2563eb"] * 2 + ["#dc2626", "#94a3b8"] + ["#64748b"] * 3 + ["#16a34a"],
    )
    axes[1, 1].set_xticks(np.arange(len(combined_values)), combined_labels, rotation=24)
    axes[1, 1].set_ylim(0.35, 1.02)
    axes[1, 1].set_ylabel("连续频段留出综合分")
    axes[1, 1].set_title(
        "重遮挡点：角度需局部扩散，不能照搬单一射线", fontweight="bold"
    )
    axes[1, 1].grid(axis="y", alpha=0.22)

    figure.suptitle(
        "历史训练集机制实验汇总：候选路径大体存在，但能量与角度分配不准",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(FIGURES["path_evidence"], bbox_inches="tight")
    plt.close(figure)


def write_details(data: dict[str, np.ndarray]) -> None:
    model_gain = data["calibrated_weighted"] - data["anchor_weighted"]
    oracle_gain = data["oracle_weighted"] - data["anchor_weighted"]
    fields = (
        "训练点索引",
        "横坐标",
        "纵坐标",
        "高度",
        "传播区域",
        "是否零功率",
        "物理锚点联合谱",
        "物理锚点PDP",
        "物理锚点PAS",
        "物理锚点综合",
        "逐格模型综合",
        "逐格模型增量",
        "候选上限综合",
        "候选上限增量",
        "仍未释放余量",
    )
    with DETAILS.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(data["positions"])):
            position = data["positions"][index]
            writer.writerow(
                {
                    "训练点索引": index,
                    "横坐标": f"{position[0]:.6f}",
                    "纵坐标": f"{position[1]:.6f}",
                    "高度": f"{position[2]:.6f}",
                    "传播区域": REGIME_CN[str(data["regimes"][index])],
                    "是否零功率": int(data["zero"][index]),
                    "物理锚点联合谱": f"{data['anchor_joint'][index]:.8f}",
                    "物理锚点PDP": f"{data['anchor_delay'][index]:.8f}",
                    "物理锚点PAS": f"{data['anchor_angle'][index]:.8f}",
                    "物理锚点综合": f"{data['anchor_weighted'][index]:.8f}",
                    "逐格模型综合": f"{data['calibrated_weighted'][index]:.8f}",
                    "逐格模型增量": f"{model_gain[index]:.8f}",
                    "候选上限综合": f"{data['oracle_weighted'][index]:.8f}",
                    "候选上限增量": f"{oracle_gain[index]:.8f}",
                    "仍未释放余量": f"{oracle_gain[index] - model_gain[index]:.8f}",
                }
            )


def build_report(
    data: dict[str, np.ndarray], selected: dict[str, int]
) -> dict[str, object]:
    valid = data["valid"]
    model_gain = data["calibrated_weighted"] - data["anchor_weighted"]
    oracle_gain = data["oracle_weighted"] - data["anchor_weighted"]
    overall = {
        "anchor": {
            key: summarize(data[f"anchor_{key}"][valid])
            for key in ("joint", "delay", "angle", "weighted")
        },
        "cell_model": {
            key: summarize(data[f"calibrated_{key}"][valid])
            for key in ("joint", "delay", "angle", "weighted")
        },
        "cell_oracle": {
            key: summarize(data[f"oracle_{key}"][valid])
            for key in ("joint", "delay", "angle", "weighted")
        },
        "cell_model_gain": summarize(model_gain[valid]),
        "cell_oracle_gain": summarize(oracle_gain[valid]),
        "model_positive_fraction": float(np.mean(model_gain[valid] > 0.0)),
        "oracle_positive_fraction": float(np.mean(oracle_gain[valid] > 0.0)),
        "released_oracle_fraction": float(
            np.mean(model_gain[valid]) / np.mean(oracle_gain[valid])
        ),
    }
    by_regime: dict[str, object] = {}
    for regime in REGIMES:
        mask = valid & (data["regimes"] == regime)
        by_regime[regime] = {
            "count": int(np.sum(mask)),
            "anchor_weighted": float(np.mean(data["anchor_weighted"][mask])),
            "anchor_joint": float(np.mean(data["anchor_joint"][mask])),
            "anchor_delay": float(np.mean(data["anchor_delay"][mask])),
            "anchor_angle": float(np.mean(data["anchor_angle"][mask])),
            "cell_model_gain": float(np.mean(model_gain[mask])),
            "cell_oracle_gain": float(np.mean(oracle_gain[mask])),
            "remaining_gap": float(np.mean(oracle_gain[mask] - model_gain[mask])),
            "model_positive_fraction": float(np.mean(model_gain[mask] > 0.0)),
        }
    return {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "The current ceiling is caused by incorrect path energy and angular allocation, "
            "not simply by missing delay support. Visualize this directly on official train spectra."
        ),
        "scope": {
            "all_train_points": 2000,
            "spectral_metrics_points": int(np.sum(valid)),
            "exact_zero_points_marked_not_normalized": int(np.sum(data["zero"])),
            "oof": True,
            "test_points_used": False,
        },
        "overall": overall,
        "by_regime": by_regime,
        "representative_points": selected,
        "findings": {
            "largest_anchor_error_component": "joint_delay_angle_allocation",
            "candidate_bank_has_large_remaining_margin": True,
            "cell_model_releases_only_part_of_candidate_margin": True,
            "recommended_next_model": (
                "Track path families across space and learn existence, complex energy, "
                "angular spread, and port response from official train H."
            ),
        },
        "outputs": {
            "directory": PATHS.relative(OUTPUT_DIR),
            "figures": {key: PATHS.relative(path) for key, path in FIGURES.items()},
            "details": PATHS.relative(DETAILS),
            "report": PATHS.relative(REPORT),
        },
    }


def main() -> None:
    configure_chinese_font()
    validate_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    background = load_background()
    selected = select_representatives(data)
    plot_spatial_maps(data, background)
    plot_component_scores(data)
    plot_average_spectra(data)
    plot_correction_direction(data)
    plot_representative_spectra(data, selected)
    plot_representative_marginals(data, selected)
    plot_path_evidence()
    write_details(data)
    report = build_report(data, selected)
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["overall"], indent=2, ensure_ascii=False))
    for path in FIGURES.values():
        print(PATHS.relative(path))


if __name__ == "__main__":
    main()
