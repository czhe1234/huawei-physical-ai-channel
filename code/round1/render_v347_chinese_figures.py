#!/usr/bin/env python3
"""Render the complete v347 figure set with Chinese names and labels."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
import numpy as np

from powerfield_common import BS_POS
from project_paths import PATHS


REPORT_DIR = PATHS.version_report_dir(347)
PRIMARY_DATA = REPORT_DIR / "v347_point_metrics.npz"
PRIMARY_VERSION_CSV = REPORT_DIR / "v347_version_metrics.csv"
PRIMARY_TRANSITION_CSV = REPORT_DIR / "v347_transition_metrics.csv"
PRIMARY_REPORT = REPORT_DIR / "v347_v212_error_evolution.json"
SUPPLEMENT_POINT_CSV = REPORT_DIR / "v347_point_metrics.csv"
SUPPLEMENT_TRANSITION_CSV = REPORT_DIR / "v347_transition_point_gains.csv"
SUPPLEMENT_REPORT = REPORT_DIR / "v347_v212_error_evolution_maps.json"
SCENE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"

CHINESE_ROOT = REPORT_DIR / "版本二一二逐步误差分析图"
OVERVIEW_DIR = CHINESE_ROOT / "一、总体演变"
CHANGE_DIR = CHINESE_ROOT / "二、逐步变化"
FINAL_DIR = CHINESE_ROOT / "三、最终误差"
COMPARE_DIR = CHINESE_ROOT / "四、分区与对照"
INDEX_FILE = CHINESE_ROOT / "图表清单.json"

SUMMARY_FIGURE = OVERVIEW_DIR / "线上分数与本地指标演变总览.png"
SCORE_FIGURE = OVERVIEW_DIR / "版本链分数演变.png"
ABSOLUTE_FIGURE = OVERVIEW_DIR / "逐版本分项绝对误差空间分布.png"
CHAIN_FIGURE = OVERVIEW_DIR / "主提升链综合误差空间演变.png"
TRANSITION_FIGURE = CHANGE_DIR / "相邻版本实际改动与分项增益空间分布.png"
GAIN_FIGURE = CHANGE_DIR / "主提升链逐点增益空间分布.png"
REGIME_FIGURE = COMPARE_DIR / "不同遮挡区域分项增益对比.png"
STRICT_FIGURE = COMPARE_DIR / "严格合规版本与版本二一二对照.png"
REMAINING_FIGURE = FINAL_DIR / "版本二一二剩余误差空间分布.png"
FINAL_FIGURE = FINAL_DIR / "版本二一二最终误差分解.png"

PRIMARY_VERSIONS = (189, 196, 200, 209, 210, 211, 212)
MAIN_VERSIONS = (200, 209, 210, 211, 212)
MAIN_TRANSITIONS = ((200, 209), (209, 210), (210, 211), (211, 212))
STRICT_VERSIONS = (218, 284, 212)
REGIMES = ("clear", "light", "medium", "heavy")
REGIME_NAMES = {
    "clear": "无遮挡",
    "light": "轻遮挡",
    "medium": "中遮挡",
    "heavy": "重遮挡",
}
CHINESE_DIGITS = "零一二三四五六七八九"


def configure_chinese_font() -> None:
    try:
        font_path = font_manager.findfont(
            font_manager.FontProperties(family="Noto Sans CJK JP"),
            fallback_to_default=False,
        )
        font_manager.fontManager.addfont(font_path)
        chinese_family = font_manager.FontProperties(fname=font_path).get_name()
    except ValueError:
        chinese_family = "Droid Sans Fallback"

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [chinese_family, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
        }
    )


def chinese_digits(value: int) -> str:
    return "".join(CHINESE_DIGITS[int(digit)] for digit in str(value))


def version_name(value: int) -> str:
    return f"版本{chinese_digits(value)}"


def transition_name(start: int, end: int) -> str:
    return f"{version_name(start)}至{version_name(end)}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_primary_version_rows() -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    for row in read_csv(PRIMARY_VERSION_CSV):
        version = int(row["version"])
        rows[version] = {
            key: float(value)
            for key, value in row.items()
            if key not in ("version", "name")
        }
    return rows


def load_primary_transition_rows() -> dict[tuple[int, int], dict[str, float]]:
    rows: dict[tuple[int, int], dict[str, float]] = {}
    for row in read_csv(PRIMARY_TRANSITION_CSV):
        key = (int(row["base_version"]), int(row["version"]))
        rows[key] = {
            field: float(value)
            for field, value in row.items()
            if field not in ("transition", "base_version", "version")
        }
    return rows


def load_supplement_points() -> dict[int, dict[str, np.ndarray]]:
    rows = read_csv(SUPPLEMENT_POINT_CSV)
    fields = (
        "composite",
        "composite_error",
        "pas",
        "pdp",
        "leaked_sh",
        "nmse_score",
        "point_nmse",
        "log_power_error",
    )
    versions = sorted({int(row["version"]) for row in rows})
    output = {
        version: {field: np.full(500, np.nan, dtype=np.float64) for field in fields}
        for version in versions
    }
    for row in rows:
        version = int(row["version"])
        index = int(row["point_id"])
        for field in fields:
            output[version][field][index] = float(row[field])
    return output


def load_supplement_transitions() -> dict[tuple[int, int], dict[str, np.ndarray]]:
    rows = read_csv(SUPPLEMENT_TRANSITION_CSV)
    fields = (
        "composite_gain",
        "pdp_gain",
        "leaked_sh_gain",
        "pas_gain",
        "nmse_score_gain",
        "action_rel_nmse",
    )
    keys = sorted(
        {
            tuple(int(part.removeprefix("v")) for part in row["transition"].split("->"))
            for row in rows
        }
    )
    output = {
        key: {field: np.full(500, np.nan, dtype=np.float64) for field in fields}
        for key in keys
    }
    for row in rows:
        key = tuple(
            int(part.removeprefix("v")) for part in row["transition"].split("->")
        )
        index = int(row["point_id"])
        for field in fields:
            output[key][field][index] = float(row[field])
    return output


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
        [BS_POS[0]],
        [BS_POS[1]],
        marker="*",
        s=44,
        color="black",
        zorder=3,
    )
    axis.set_aspect("equal", adjustable="box")


def finite_values(values: list[np.ndarray]) -> np.ndarray:
    merged = np.concatenate([np.asarray(value, dtype=np.float64).ravel() for value in values])
    return merged[np.isfinite(merged)]


def limits(values: list[np.ndarray], *, symmetric: bool = False) -> tuple[float, float]:
    merged = finite_values(values)
    if symmetric:
        bound = max(float(np.quantile(np.abs(merged), 0.98)), 1e-6)
        return -bound, bound
    low, high = np.quantile(merged, (0.02, 0.98))
    return float(low), float(max(high, low + 1e-6))


def primary_value(data: np.lib.npyio.NpzFile, version: int, field: str) -> np.ndarray:
    return np.asarray(data[f"v{version}_{field}"], dtype=np.float64)


def transition_value(
    data: np.lib.npyio.NpzFile, start: int, end: int, field: str
) -> np.ndarray:
    return np.asarray(data[f"v{start}_to_v{end}_{field}"], dtype=np.float64)


def plot_summary(
    data: np.lib.npyio.NpzFile,
    version_rows: dict[int, dict[str, float]],
    transition_rows: dict[tuple[int, int], dict[str, float]],
    primary_report: dict[str, object],
) -> None:
    labels = [version_name(version) for version in PRIMARY_VERSIONS]
    x = np.arange(len(PRIMARY_VERSIONS))
    figure, axes = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)

    axis = axes[0, 0]
    online = [version_rows[version]["online_score"] for version in PRIMARY_VERSIONS]
    local = [version_rows[version]["local_score_traceback"] for version in PRIMARY_VERSIONS]
    online_line = axis.plot(x, online, marker="o", linewidth=2, color="#b83a2d")[0]
    axis.set_ylabel("线上分数", color="#b83a2d")
    axis.tick_params(axis="y", labelcolor="#b83a2d")
    axis.set_xticks(x, labels)
    twin = axis.twinx()
    local_line = twin.plot(x, local, marker="s", linewidth=2, color="#2b67a5")[0]
    twin.set_ylabel("本地参考分数", color="#2b67a5")
    twin.tick_params(axis="y", labelcolor="#2b67a5")
    axis.legend(
        [online_line, local_line], ["线上分数", "本地参考分数"], loc="upper left"
    )
    axis.set_title("线上分数与本地参考分数演变")
    axis.grid(alpha=0.25)

    axis = axes[0, 1]
    for key, label, color in (
        ("pdp", "时延功率谱", "#337a4f"),
        ("leaked_sh", "泄露角度功率谱", "#7c4a9e"),
        ("nmse_score", "归一化均方误差分数", "#c47d20"),
    ):
        axis.plot(
            x,
            [version_rows[version][key] for version in PRIMARY_VERSIONS],
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )
    axis.set_xticks(x, labels)
    axis.set_ylabel("本地指标")
    axis.set_title("本地分项指标变化")
    axis.legend()
    axis.grid(alpha=0.25)

    pairs = list(zip(PRIMARY_VERSIONS[:-1], PRIMARY_VERSIONS[1:]))
    pair_labels = [transition_name(start, end) for start, end in pairs]
    tx = np.arange(len(pairs))
    axis = axes[1, 0]
    width = 0.20
    bars = (
        ("online_delta", "线上分数", "#b83a2d"),
        ("mean_delta_pdp", "时延功率谱", "#337a4f"),
        ("mean_delta_leaked_sh", "泄露角度功率谱", "#7c4a9e"),
        ("mean_delta_nmse_score", "逐点归一化误差分数", "#c47d20"),
    )
    for index, (key, label, color) in enumerate(bars):
        axis.bar(
            tx + (index - 1.5) * width,
            [transition_rows[pair][key] for pair in pairs],
            width=width,
            label=label,
            color=color,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(tx, pair_labels, rotation=28, ha="right")
    axis.set_ylabel("增量")
    axis.set_title("相邻版本分项增益")
    axis.legend(ncols=2)
    axis.grid(axis="y", alpha=0.25)

    report_transitions = {
        (int(row["base_version"]), int(row["version"])): row
        for row in primary_report["transitions"]
    }
    matrix = np.asarray(
        [
            [
                report_transitions[pair]["by_regime"][regime]["point_proxy"]
                for regime in REGIMES
            ]
            for pair in pairs
        ],
        dtype=np.float64,
    )
    bound = max(float(np.quantile(np.abs(matrix), 0.98)), 1e-6)
    axis = axes[1, 1]
    image = axis.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-bound, vmax=bound)
    axis.set_xticks(range(len(REGIMES)), [REGIME_NAMES[name] for name in REGIMES])
    axis.set_yticks(range(len(pair_labels)), pair_labels)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    axis.set_title("不同遮挡区域的逐点诊断增益")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="平均增益")
    figure.suptitle(
        "版本一八九至版本二一二误差演变总览（本地参考诊断）",
        fontsize=16,
    )
    figure.savefig(SUMMARY_FIGURE, dpi=180)
    plt.close(figure)


def plot_score_evolution(version_rows: dict[int, dict[str, float]]) -> None:
    labels = [version_name(version) for version in MAIN_VERSIONS]
    x = np.arange(len(MAIN_VERSIONS))
    figure, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    for key, label, color in (
        ("local_score_traceback", "本地回溯口径分数", "#111111"),
        ("pdp", "本地时延功率谱分数", "#0072B2"),
        ("leaked_sh", "本地泄露角度功率谱分数", "#009E73"),
        ("nmse_score", "本地归一化均方误差分数", "#D55E00"),
        ("online_score", "线上分数", "#CC79A7"),
    ):
        axes[0].plot(
            x,
            [version_rows[version][key] for version in MAIN_VERSIONS],
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("分数")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncols=3)
    axes[0].set_title("版本二百至版本二一二分数演变")

    pairs = list(zip(MAIN_VERSIONS[:-1], MAIN_VERSIONS[1:]))
    pair_labels = [transition_name(start, end) for start, end in pairs]
    tx = np.arange(len(pairs))
    width = 0.20
    for index, (key, label, color) in enumerate(
        (
            ("online_score", "线上分数", "#CC79A7"),
            ("local_score_traceback", "本地回溯口径分数", "#111111"),
            ("pdp", "时延功率谱分数", "#0072B2"),
            ("leaked_sh", "泄露角度功率谱分数", "#009E73"),
        )
    ):
        delta = [
            version_rows[end][key] - version_rows[start][key] for start, end in pairs
        ]
        axes[1].bar(
            tx + (index - 1.5) * width,
            delta,
            width=width,
            label=label,
            color=color,
        )
    axes[1].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1].set_xticks(tx, pair_labels, rotation=20, ha="right")
    axes[1].set_ylabel("分数增量")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(ncols=4)
    axes[1].set_title("相邻版本分数增量")
    figure.suptitle("主提升链分数演变（本地参考不代表线上隐藏真值）", fontsize=15)
    figure.savefig(SCORE_FIGURE, dpi=180)
    plt.close(figure)


def plot_absolute_maps(
    data: np.lib.npyio.NpzFile,
    version_rows: dict[int, dict[str, float]],
    positions: np.ndarray,
    background: dict[str, object],
) -> None:
    fields = (
        ("pdp", "时延功率谱缺口", "magma", lambda value: 1.0 - value),
        ("leaked_sh", "泄露角度功率谱缺口", "magma", lambda value: 1.0 - value),
        ("log10_point_nmse", "逐点归一化误差的十进对数", "magma", lambda value: value),
        ("abs_log_power_error", "总功率对数误差绝对值", "magma", lambda value: value),
    )
    values = {
        (version, field): transform(primary_value(data, version, field))
        for version in PRIMARY_VERSIONS
        for field, _, _, transform in fields
    }
    field_limits = {
        field: limits([values[(version, field)] for version in PRIMARY_VERSIONS])
        for field, _, _, _ in fields
    }
    figure, axes = plt.subplots(
        len(PRIMARY_VERSIONS),
        len(fields),
        figsize=(18, 25),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for row, version in enumerate(PRIMARY_VERSIONS):
        for column, (field, title, cmap, _) in enumerate(fields):
            axis = axes[row, column]
            decorate_map(axis, background)
            vmin, vmax = field_limits[field]
            axis.scatter(
                positions[:, 0],
                positions[:, 1],
                c=values[(version, field)],
                s=18,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidths=0,
                zorder=2,
            )
            if row == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(
                    f"{version_name(version)}，线上分数={version_rows[version]['online_score']:.6f}\n纵坐标（米）"
                )
            if row == len(PRIMARY_VERSIONS) - 1:
                axis.set_xlabel("横坐标（米）")
    for column, (field, _, cmap, _) in enumerate(fields):
        color = plt.cm.ScalarMappable(norm=Normalize(*field_limits[field]), cmap=cmap)
        figure.colorbar(color, ax=axes[:, column], fraction=0.014, pad=0.008, label="误差值")
    figure.suptitle(
        "逐版本分项绝对误差空间分布\n仅针对本地参考，不代表线上隐藏真值",
        fontsize=16,
    )
    figure.savefig(ABSOLUTE_FIGURE, dpi=180)
    plt.close(figure)


def plot_chain_composite(
    supplement_points: dict[int, dict[str, np.ndarray]],
    supplement_report: dict[str, object],
    positions: np.ndarray,
    background: dict[str, object],
) -> None:
    summaries = {int(key): value for key, value in supplement_report["versions"].items()}
    values = [supplement_points[version]["composite_error"] for version in MAIN_VERSIONS]
    _, vmax = limits(values)
    figure, axes = plt.subplots(2, 3, figsize=(16, 10.5), constrained_layout=True)
    scatter = None
    for axis, version, value in zip(axes.flat, MAIN_VERSIONS, values):
        decorate_map(axis, background)
        scatter = axis.scatter(
            positions[:, 0],
            positions[:, 1],
            c=value,
            s=26,
            cmap="magma",
            vmin=0,
            vmax=vmax,
            edgecolors="white",
            linewidths=0.25,
            zorder=3,
        )
        axis.set_title(
            f"{version_name(version)}\n线上分数={summaries[version]['online_score']:.6f}，本地分数={summaries[version]['local_score']:.6f}"
        )
        axis.set_xlabel("横坐标（米）")
        axis.set_ylabel("纵坐标（米）")
    axes.flat[-1].axis("off")
    if scatter is not None:
        figure.colorbar(scatter, ax=axes, shrink=0.82, label="逐点综合误差，越低越好")
    figure.suptitle("主提升链逐点综合误差空间演变", fontsize=16)
    figure.savefig(CHAIN_FIGURE, dpi=180)
    plt.close(figure)


def plot_transition_maps(
    data: np.lib.npyio.NpzFile,
    positions: np.ndarray,
    background: dict[str, object],
) -> None:
    pairs = list(zip(PRIMARY_VERSIONS[:-1], PRIMARY_VERSIONS[1:]))
    fields = (
        ("log10_change_nmse", "输出改动强度的十进对数", "viridis", False),
        ("delta_pdp", "时延功率谱增益", "RdYlGn", True),
        ("delta_leaked_sh", "泄露角度功率谱增益", "RdYlGn", True),
        ("delta_nmse_score", "逐点归一化误差分数增益", "RdYlGn", True),
        ("delta_point_proxy", "逐点诊断分数增益", "RdYlGn", True),
    )
    field_limits = {
        field: limits(
            [transition_value(data, start, end, field) for start, end in pairs],
            symmetric=symmetric,
        )
        for field, _, _, symmetric in fields
    }
    figure, axes = plt.subplots(
        len(pairs),
        len(fields),
        figsize=(22, 23),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for row, (start, end) in enumerate(pairs):
        for column, (field, title, cmap, _) in enumerate(fields):
            axis = axes[row, column]
            decorate_map(axis, background)
            vmin, vmax = field_limits[field]
            axis.scatter(
                positions[:, 0],
                positions[:, 1],
                c=transition_value(data, start, end, field),
                s=18,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidths=0,
                zorder=2,
            )
            if row == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(f"{transition_name(start, end)}\n纵坐标（米）")
            if row == len(pairs) - 1:
                axis.set_xlabel("横坐标（米）")
    for column, (field, title, cmap, _) in enumerate(fields):
        color = plt.cm.ScalarMappable(norm=Normalize(*field_limits[field]), cmap=cmap)
        label = "改动强度" if field == "log10_change_nmse" else "增益，正值表示改善"
        figure.colorbar(color, ax=axes[:, column], fraction=0.012, pad=0.007, label=label)
    figure.suptitle(
        "相邻版本实际改动与分项增益空间分布\n绿色表示改善，红色表示变差",
        fontsize=16,
    )
    figure.savefig(TRANSITION_FIGURE, dpi=180)
    plt.close(figure)


def plot_main_gain_maps(
    transitions: dict[tuple[int, int], dict[str, np.ndarray]],
    positions: np.ndarray,
    background: dict[str, object],
) -> None:
    fields = (
        ("composite_gain", "逐点综合分数增益"),
        ("pdp_gain", "时延功率谱增益"),
        ("leaked_sh_gain", "泄露角度功率谱增益"),
        ("nmse_score_gain", "逐点归一化误差分数增益"),
    )
    field_limits = {
        field: limits([transitions[pair][field] for pair in MAIN_TRANSITIONS], symmetric=True)
        for field, _ in fields
    }
    figure, axes = plt.subplots(4, 4, figsize=(19, 17.5), constrained_layout=True)
    for row, pair in enumerate(MAIN_TRANSITIONS):
        for column, (field, title) in enumerate(fields):
            axis = axes[row, column]
            decorate_map(axis, background)
            vmin, vmax = field_limits[field]
            axis.scatter(
                positions[:, 0],
                positions[:, 1],
                c=transitions[pair][field],
                s=20,
                cmap="RdYlGn",
                vmin=vmin,
                vmax=vmax,
                edgecolors="none",
                zorder=3,
            )
            axis.set_title(f"{transition_name(*pair)}：{title}")
            axis.set_xlabel("横坐标（米）")
            axis.set_ylabel("纵坐标（米）")
    for column, (field, title) in enumerate(fields):
        color = plt.cm.ScalarMappable(norm=Normalize(*field_limits[field]), cmap="RdYlGn")
        figure.colorbar(
            color,
            ax=axes[:, column],
            shrink=0.72,
            label=f"{title}，正值表示改善",
        )
    figure.suptitle("主提升链逐点增益空间分布", fontsize=16)
    figure.savefig(GAIN_FIGURE, dpi=180)
    plt.close(figure)


def plot_regime_gains(
    supplement_report: dict[str, object],
) -> None:
    transitions = {
        (int(row["old_version"]), int(row["new_version"])): row
        for row in supplement_report["transitions"]
    }
    metrics = (
        ("composite", "逐点综合分数"),
        ("pdp", "时延功率谱"),
        ("leaked_sh", "泄露角度功率谱"),
        ("nmse_score", "逐点归一化误差分数"),
    )
    matrices = {
        metric: np.asarray(
            [
                [
                    transitions[pair]["by_regime"][regime][metric]["mean_gain"]
                    for regime in REGIMES
                ]
                for pair in MAIN_TRANSITIONS
            ],
            dtype=np.float64,
        )
        for metric, _ in metrics
    }
    bound = max(float(np.max(np.abs(np.concatenate(list(matrices.values()))))), 1e-6)
    figure, axes = plt.subplots(1, 4, figsize=(19, 5.6), constrained_layout=True)
    image = None
    for axis, (metric, title) in zip(axes, metrics):
        matrix = matrices[metric]
        image = axis.imshow(matrix, cmap="RdYlGn", vmin=-bound, vmax=bound, aspect="auto")
        axis.set_xticks(range(len(REGIMES)), [REGIME_NAMES[name] for name in REGIMES])
        axis.set_yticks(
            range(len(MAIN_TRANSITIONS)),
            [transition_name(*pair) for pair in MAIN_TRANSITIONS],
        )
        axis.set_title(f"{title}平均增益")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:+.4f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.78, label="平均增益，正值表示改善")
    figure.suptitle("不同遮挡区域在各提升步骤中的分项增益", fontsize=16)
    figure.savefig(REGIME_FIGURE, dpi=180)
    plt.close(figure)


def plot_remaining(
    data: np.lib.npyio.NpzFile,
    positions: np.ndarray,
    background: dict[str, object],
) -> None:
    fields = (
        (1.0 - primary_value(data, 212, "pdp"), "时延功率谱缺口", "magma", False),
        (
            1.0 - primary_value(data, 212, "leaked_sh"),
            "泄露角度功率谱缺口",
            "magma",
            False,
        ),
        (
            primary_value(data, 212, "log10_point_nmse"),
            "逐点归一化误差的十进对数",
            "magma",
            False,
        ),
        (
            primary_value(data, 212, "abs_log_power_error"),
            "总功率对数误差绝对值",
            "magma",
            False,
        ),
        (
            primary_value(data, 212, "pdp") - primary_value(data, 189, "pdp"),
            "版本一八九至版本二一二的时延谱增益",
            "RdYlGn",
            True,
        ),
        (
            primary_value(data, 212, "point_proxy")
            - primary_value(data, 189, "point_proxy"),
            "版本一八九至版本二一二的逐点诊断增益",
            "RdYlGn",
            True,
        ),
    )
    figure, axes = plt.subplots(2, 3, figsize=(18, 10.5), sharex=True, sharey=True, constrained_layout=True)
    for axis, (value, title, cmap, symmetric) in zip(axes.flat, fields):
        decorate_map(axis, background)
        vmin, vmax = limits([value], symmetric=symmetric)
        scatter = axis.scatter(
            positions[:, 0],
            positions[:, 1],
            c=value,
            s=22,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
            zorder=2,
        )
        axis.set_title(title)
        figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.03)
    for axis in axes[-1]:
        axis.set_xlabel("横坐标（米）")
    for axis in axes[:, 0]:
        axis.set_ylabel("纵坐标（米）")
    figure.suptitle(
        "版本二一二剩余误差与相对版本一八九的总体改善\n不代表线上隐藏逐点误差",
        fontsize=16,
    )
    figure.savefig(REMAINING_FIGURE, dpi=180)
    plt.close(figure)


def plot_final_residual(
    points: dict[int, dict[str, np.ndarray]],
    positions: np.ndarray,
    background: dict[str, object],
) -> None:
    fields = (
        (1.0 - points[212]["pdp"], "时延功率谱误差", "magma", False),
        (1.0 - points[212]["leaked_sh"], "泄露角度功率谱误差", "magma", False),
        (1.0 - points[212]["nmse_score"], "逐点归一化误差分数缺口", "magma", False),
        (points[212]["log_power_error"], "总功率十进对数比值误差", "RdYlGn", True),
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 11.5), constrained_layout=True)
    for axis, (value, title, cmap, symmetric) in zip(axes.flat, fields):
        decorate_map(axis, background)
        vmin, vmax = limits([value], symmetric=symmetric)
        if not symmetric:
            vmin = 0.0
        scatter = axis.scatter(
            positions[:, 0],
            positions[:, 1],
            c=value,
            s=27,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolors="white",
            linewidths=0.25,
            zorder=3,
        )
        axis.set_title(title)
        axis.set_xlabel("横坐标（米）")
        axis.set_ylabel("纵坐标（米）")
        figure.colorbar(scatter, ax=axis, shrink=0.78)
    figure.suptitle("版本二一二最终剩余误差分解（本地参考诊断）", fontsize=16)
    figure.savefig(FINAL_FIGURE, dpi=180)
    plt.close(figure)


def plot_strict_compare(
    points: dict[int, dict[str, np.ndarray]],
    supplement_report: dict[str, object],
    positions: np.ndarray,
    background: dict[str, object],
) -> None:
    summaries = {int(key): value for key, value in supplement_report["versions"].items()}
    fields = (
        ("composite_error", "逐点综合误差"),
        ("pdp", "时延功率谱误差"),
        ("leaked_sh", "泄露角度功率谱误差"),
    )
    values = {
        (version, field): (
            points[version][field]
            if field == "composite_error"
            else 1.0 - points[version][field]
        )
        for version in STRICT_VERSIONS
        for field, _ in fields
    }
    field_limits = {
        field: limits([values[(version, field)] for version in STRICT_VERSIONS])
        for field, _ in fields
    }
    figure, axes = plt.subplots(3, 3, figsize=(17, 15.5), constrained_layout=True)
    for row, version in enumerate(STRICT_VERSIONS):
        for column, (field, title) in enumerate(fields):
            axis = axes[row, column]
            decorate_map(axis, background)
            vmin, vmax = field_limits[field]
            axis.scatter(
                positions[:, 0],
                positions[:, 1],
                c=values[(version, field)],
                s=22,
                cmap="magma",
                vmin=max(0.0, vmin),
                vmax=vmax,
                edgecolors="none",
                zorder=3,
            )
            axis.set_title(
                f"{version_name(version)}：{title}\n线上分数={summaries[version]['online_score']:.6f}，本地分数={summaries[version]['local_score']:.6f}"
            )
            axis.set_xlabel("横坐标（米）")
            axis.set_ylabel("纵坐标（米）")
    for column, (field, title) in enumerate(fields):
        color = plt.cm.ScalarMappable(norm=Normalize(*field_limits[field]), cmap="magma")
        figure.colorbar(color, ax=axes[:, column], shrink=0.72, label=title)
    figure.suptitle(
        "严格合规版本与版本二一二的本地参考误差对照\n版本二一二仅作诊断，不是严格最终方法",
        fontsize=16,
    )
    figure.savefig(STRICT_FIGURE, dpi=180)
    plt.close(figure)


def write_index() -> None:
    groups = {
        "一、总体演变": [
            (SUMMARY_FIGURE, "线上分数、本地指标、相邻版本增益和遮挡区域增益总览"),
            (SCORE_FIGURE, "主提升链的绝对分数与相邻版本增量"),
            (ABSOLUTE_FIGURE, "七个版本在统一色标下的分项绝对误差"),
            (CHAIN_FIGURE, "版本二百至版本二一二的逐点综合误差演变"),
        ],
        "二、逐步变化": [
            (TRANSITION_FIGURE, "六次相邻升级的实际改动和四类分项增益"),
            (GAIN_FIGURE, "四次主提升步骤的逐点增益空间分布"),
        ],
        "三、最终误差": [
            (REMAINING_FIGURE, "版本二一二的剩余误差及相对版本一八九的总体改善"),
            (FINAL_FIGURE, "版本二一二的时延谱、角度谱、归一化误差和总功率误差分解"),
        ],
        "四、分区与对照": [
            (REGIME_FIGURE, "无遮挡、轻遮挡、中遮挡、重遮挡区域的分项增益"),
            (STRICT_FIGURE, "严格合规版本二一八、二八四与诊断版本二一二的对照"),
        ],
    }
    payload = {
        "目录说明": "本目录汇总版本二一二提升链的全部中文分析图。",
        "解释边界": (
            "逐点误差来自本地参考，不是官方隐藏线上逐点真值；版本二一零至二一二还直接使用过该参考的时延谱信息。"
        ),
        "阅读顺序": ["一、总体演变", "二、逐步变化", "三、最终误差", "四、分区与对照"],
        "图表分组": {
            group: [
                {
                    "文件": str(path.relative_to(CHINESE_ROOT)),
                    "内容": description,
                }
                for path, description in files
            ]
            for group, files in groups.items()
        },
    }
    INDEX_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    configure_chinese_font()
    required = (
        PRIMARY_DATA,
        PRIMARY_VERSION_CSV,
        PRIMARY_TRANSITION_CSV,
        PRIMARY_REPORT,
        SUPPLEMENT_POINT_CSV,
        SUPPLEMENT_TRANSITION_CSV,
        SUPPLEMENT_REPORT,
        SCENE,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    for directory in (OVERVIEW_DIR, CHANGE_DIR, FINAL_DIR, COMPARE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    data = np.load(PRIMARY_DATA)
    version_rows = load_primary_version_rows()
    transition_rows = load_primary_transition_rows()
    primary_report = json.loads(PRIMARY_REPORT.read_text(encoding="utf-8"))
    supplement_points = load_supplement_points()
    supplement_transitions = load_supplement_transitions()
    supplement_report = json.loads(SUPPLEMENT_REPORT.read_text(encoding="utf-8"))
    positions = np.asarray(data["positions"], dtype=np.float64)
    background = map_background()

    plot_summary(data, version_rows, transition_rows, primary_report)
    plot_score_evolution(version_rows)
    plot_absolute_maps(data, version_rows, positions, background)
    plot_chain_composite(supplement_points, supplement_report, positions, background)
    plot_transition_maps(data, positions, background)
    plot_main_gain_maps(supplement_transitions, positions, background)
    plot_regime_gains(supplement_report)
    plot_remaining(data, positions, background)
    plot_final_residual(supplement_points, positions, background)
    plot_strict_compare(supplement_points, supplement_report, positions, background)
    write_index()

    output_files = sorted(CHINESE_ROOT.rglob("*.png"))
    if len(output_files) != 10:
        raise RuntimeError(f"expected 10 Chinese figures, found {len(output_files)}")
    print(
        json.dumps(
            {
                "中文图表目录": PATHS.relative(CHINESE_ROOT),
                "图表数量": len(output_files),
                "图表": [PATHS.relative(path) for path in output_files],
                "图表清单": PATHS.relative(INDEX_FILE),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
