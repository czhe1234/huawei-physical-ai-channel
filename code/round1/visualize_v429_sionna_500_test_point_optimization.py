#!/usr/bin/env python3
"""Visualize where Sionna-assisted stages changed all 500 test points."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import analyze_v205_sionna_config_calibration as calibration
from powerfield_common import BS_POS, regime_of
from project_paths import PATHS


VERSION_NUMBER = 429
VERSION = "v429_sionna_500_test_point_optimization_visualization"
OUTPUT_DIR = PATHS.analysis / "visualizations" / "Sionna辅助优化的500个测试点"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
METRICS = REPORT_DIR / "v429_500个测试点逐点指标.npz"

REFERENCE = PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"
CHANNELS = {
    "v209": PATHS.cache_dir(209) / "v209_Round1_Test_Channel.npy",
    "v212": PATHS.cache_dir(212) / "v212_Round1_Test_Channel.npy",
    "v424": PATHS.cache_dir(424) / "v424_Round1_Test_Channel.npy",
    "v426": PATHS.cache_dir(426) / "v426_Round1_Test_Channel.npy",
    "v427": PATHS.cache_dir(427) / "v427_Round1_Test_Channel.npy",
    "v428": PATHS.cache_dir(428) / "v428_Round1_Test_Channel.npy",
}
ONLINE = {
    "v209": 0.511934,
    "v212": 0.559160,
    "v424": 0.567791,
    "v426": 0.573416,
    "v427": 0.565229,
    "v428": 0.572969,
}
SCENE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"
GEOMETRY = PATHS.shared_features / "height_geometry_cache.npz"
TEST_POS = PATHS.round1 / "Round1_Test_Pos.npy"

PNG_OVERVIEW = OUTPUT_DIR / "01_500个测试点的主要受益阶段总览.png"
PNG_STAGES = OUTPUT_DIR / "02_三个阶段分别改善了哪些位置.png"
PNG_TOTAL = OUTPUT_DIR / "03_从v209到v426的累计改善与分区统计.png"
PNG_REMAINING = OUTPUT_DIR / "04_v426仍然困难的测试点在哪里.png"
PNG_EVIDENCE = OUTPUT_DIR / "05_哪些Sionna信息在线上有效.png"
HTML_INTERACTIVE = OUTPUT_DIR / "06_500个测试点交互式优化地图.html"
CSV_DETAILS = OUTPUT_DIR / "500个测试点优化明细.csv"
JSON_SUMMARY = OUTPUT_DIR / "500个测试点优化摘要.json"

REGIMES = ("clear", "light", "medium", "heavy")
REGIME_CN = {"clear": "无遮挡", "light": "轻遮挡", "medium": "中遮挡", "heavy": "重遮挡"}
STAGE_CN = {
    0: "时延/PDP补全",
    1: "leaked-PAS角度-时延",
    2: "完整逐链路PDP",
    3: "累计未明显改善",
}
STAGE_COLORS = {0: "#2563eb", 1: "#db2777", 2: "#059669", 3: "#7b8794"}
NORTH_MASK_NAME = "主城区（y≥-110米）"
SOUTH_MASK_NAME = "南部片区（y<-110米）"
BATCH = 2


def configure_chinese_font() -> None:
    selected = "DejaVu Sans"
    for family in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"):
        try:
            font_path = font_manager.findfont(
                font_manager.FontProperties(family=family), fallback_to_default=False
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
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 280,
        }
    )


def validate_inputs() -> None:
    required = (REFERENCE, SCENE, GEOMETRY, TEST_POS, *CHANNELS.values())
    missing = [PATHS.relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing visualization inputs: {missing}")


def compute_point_metrics() -> dict[str, np.ndarray]:
    reference = np.load(REFERENCE, mmap_mode="r")
    channels = {name: np.load(path, mmap_mode="r") for name, path in CHANNELS.items()}
    count = len(reference)
    keys = ("pas", "pdp", "leaked_sh", "ad_support", "complex_coherence", "nmse_score", "traceback_score")
    output = {f"{name}_{key}": np.zeros(count, dtype=np.float32) for name in channels for key in keys}

    for start in range(0, count, BATCH):
        end = min(start + BATCH, count)
        target = np.asarray(reference[start:end], dtype=np.complex64)
        target_views = calibration.spectral_views(target)
        target_energy = target_views["energy"]
        true_zero = target_energy <= 1e-25
        for name, channel in channels.items():
            prediction = np.asarray(channel[start:end], dtype=np.complex64)
            scores = calibration.spectral_scores(prediction, target_views)
            pred_energy = np.sum(np.abs(prediction) ** 2, axis=(1, 2, 3), dtype=np.float64)
            pred_zero = pred_energy <= 1e-25
            for key in ("pas", "pdp", "leaked_sh", "ad_support"):
                value = scores[key].astype(np.float32).copy()
                value[true_zero & pred_zero] = 1.0
                value[true_zero & ~pred_zero] = 0.0
                output[f"{name}_{key}"][start:end] = value
            output[f"{name}_complex_coherence"][start:end] = scores[
                "complex_coherence"
            ].astype(np.float32)
            error = np.sum(
                np.abs(prediction - target) ** 2,
                axis=(1, 2, 3),
                dtype=np.float64,
            )
            nmse = error / np.maximum(target_energy, 1e-30)
            nmse_score = (1.0 / (1.0 + nmse)).astype(np.float32)
            nmse_score[true_zero & pred_zero] = 1.0
            nmse_score[true_zero & ~pred_zero] = 0.0
            output[f"{name}_nmse_score"][start:end] = nmse_score
            output[f"{name}_traceback_score"][start:end] = (
                0.4 * output[f"{name}_leaked_sh"][start:end]
                + 0.4 * output[f"{name}_pdp"][start:end]
                + 0.2 * nmse_score
            )
        print(f"v429 逐点指标 {end}/{count}", flush=True)
    return output


def add_derived_metrics(data: dict[str, np.ndarray]) -> None:
    data["阶段1_PDP增量"] = data["v212_pdp"] - data["v209_pdp"]
    data["阶段2_leakedPAS增量"] = data["v424_leaked_sh"] - data["v212_leaked_sh"]
    data["阶段3_逐链路PDP增量"] = data["v426_pdp"] - data["v424_pdp"]
    data["阶段1_诊断分增量"] = data["v212_traceback_score"] - data["v209_traceback_score"]
    data["阶段2_诊断分增量"] = data["v424_traceback_score"] - data["v212_traceback_score"]
    data["阶段3_诊断分增量"] = data["v426_traceback_score"] - data["v424_traceback_score"]
    data["累计诊断分增量"] = data["v426_traceback_score"] - data["v209_traceback_score"]
    data["v426剩余PDP误差"] = 1.0 - data["v426_pdp"]
    data["v426剩余leakedPAS误差"] = 1.0 - data["v426_leaked_sh"]
    contributions = np.stack(
        [data["阶段1_诊断分增量"], data["阶段2_诊断分增量"], data["阶段3_诊断分增量"]],
        axis=1,
    )
    dominant = np.argmax(contributions, axis=1).astype(np.int16)
    weak = (data["累计诊断分增量"] <= 0.002) | (np.max(contributions, axis=1) <= 0.001)
    dominant[weak] = 3
    data["主要受益阶段"] = dominant


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
        "vmax": max(float(np.quantile(positive, 0.975)), 1.0),
    }


def regional_masks(test_pos: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [
        (NORTH_MASK_NAME, test_pos[:, 1] >= -110.0),
        (SOUTH_MASK_NAME, test_pos[:, 1] < -110.0),
    ]


def region_limits(test_pos: np.ndarray, mask: np.ndarray, pad: float = 7.0) -> tuple[float, float, float, float]:
    points = test_pos[mask, :2]
    return (
        float(points[:, 0].min() - pad),
        float(points[:, 0].max() + pad),
        float(points[:, 1].min() - pad),
        float(points[:, 1].max() + pad),
    )


def decorate_map(
    axis: plt.Axes,
    background: dict[str, object],
    limits: tuple[float, float, float, float],
    title: str,
) -> None:
    axis.set_facecolor("#f8fafc")
    axis.imshow(
        background["wall"],
        origin="lower",
        extent=background["extent"],
        cmap="Greys",
        vmin=0.0,
        vmax=background["vmax"],
        alpha=0.27,
        interpolation="nearest",
        zorder=0,
    )
    if limits[0] <= BS_POS[0] <= limits[1] and limits[2] <= BS_POS[1] <= limits[3]:
        axis.scatter(
            [BS_POS[0]], [BS_POS[1]], marker="*", s=240, color="#111827",
            edgecolor="white", linewidth=1.1, zorder=8,
        )
        axis.text(BS_POS[0] + 2.0, BS_POS[1] + 2.0, "基站", fontsize=10, fontweight="bold")
    axis.set_xlim(limits[0], limits[1])
    axis.set_ylim(limits[2], limits[3])
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title, fontweight="bold")
    axis.set_xlabel("x（米）")
    axis.set_ylabel("y（米）")
    axis.grid(color="#64748b", alpha=0.12, linewidth=0.55)


def annotate_indices(
    axis: plt.Axes,
    test_pos: np.ndarray,
    indices: np.ndarray,
    color: str = "#111827",
) -> None:
    offsets = ((5, 5), (5, -11), (-20, 5), (-20, -11))
    for rank, index in enumerate(indices):
        dx, dy = offsets[rank % len(offsets)]
        axis.annotate(
            f"T{int(index):03d}",
            (test_pos[index, 0], test_pos[index, 1]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.5,
            color=color,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": color, "alpha": 0.88, "lw": 0.55},
            zorder=10,
        )


def save_figure(fig: plt.Figure, png_path: Path) -> list[Path]:
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png_path, pdf_path]


def plot_dominant_overview(
    test_pos: np.ndarray,
    data: dict[str, np.ndarray],
    background: dict[str, object],
) -> list[Path]:
    fig = plt.figure(figsize=(21, 15))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.28, 0.54])
    map_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    total = data["累计诊断分增量"]
    dominant = data["主要受益阶段"]
    size_scale = np.quantile(np.abs(total), 0.95)
    sizes = 42.0 + 145.0 * np.clip(np.abs(total) / max(float(size_scale), 1e-6), 0.0, 1.0)
    for axis, (region_name, mask) in zip(map_axes, regional_masks(test_pos)):
        decorate_map(axis, background, region_limits(test_pos, mask), f"{region_name}：主要由哪一步改善")
        for stage in range(4):
            active = mask & (dominant == stage)
            marker = "X" if stage == 3 else "o"
            axis.scatter(
                test_pos[active, 0], test_pos[active, 1], s=sizes[active], marker=marker,
                c=STAGE_COLORS[stage], edgecolor="white" if stage != 3 else "#475569",
                linewidth=0.75, alpha=0.91, zorder=4 + stage,
            )
        region_indices = np.where(mask)[0]
        strongest = region_indices[np.argsort(total[region_indices])[-7:][::-1]]
        annotate_indices(axis, test_pos, strongest)

    counts = {stage: int(np.sum(dominant == stage)) for stage in range(4)}
    count_axis = fig.add_subplot(grid[1, 0])
    stages = np.arange(4)
    bars = count_axis.barh(
        [STAGE_CN[stage] for stage in stages],
        [counts[stage] for stage in stages],
        color=[STAGE_COLORS[stage] for stage in stages],
    )
    count_axis.invert_yaxis()
    count_axis.set_title("500个测试点的主要受益阶段计数")
    count_axis.set_xlabel("测试点数量")
    count_axis.grid(axis="x", alpha=0.18)
    for bar, stage in zip(bars, stages):
        count_axis.text(
            bar.get_width() + 3,
            bar.get_y() + bar.get_height() / 2,
            f"{counts[stage]}点（{counts[stage] / len(test_pos):.1%}）",
            va="center",
            fontweight="bold",
        )

    summary_axis = fig.add_subplot(grid[1, 1])
    summary_axis.axis("off")
    summary_axis.text(
        0.02,
        0.96,
        "如何读图\n"
        "• 颜色：该点贡献最大的正向阶段\n"
        "• 圆点越大：v209→v426累计改善越大\n"
        "• 灰色叉号：累计改善≤0.002或没有明确主导阶段\n"
        "• Txxx：累计改善最明显的代表点\n\n"
        "真实线上证据\n"
        "v209→v212：+0.047226\n"
        "v212→v424：+0.008631\n"
        "v424→v426：+0.005625\n\n"
        "注意：逐点颜色依据同坐标Sionna诊断参考，\n"
        "用于解释我们注入了什么，不是官方隐藏测试真值。",
        va="top",
        fontsize=13,
        linespacing=1.48,
        bbox={"boxstyle": "round,pad=0.7", "fc": "#f8fafc", "ec": "#94a3b8"},
    )
    fig.suptitle("500个测试点：Sionna信息主要优化了什么", fontsize=22, fontweight="bold")
    fig.subplots_adjust(left=0.055, right=0.975, top=0.93, bottom=0.055, wspace=0.18, hspace=0.24)
    return save_figure(fig, PNG_OVERVIEW)


def symmetric_norm(values: np.ndarray) -> TwoSlopeNorm:
    limit = max(float(np.quantile(np.abs(values), 0.975)), 0.005)
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def plot_stage_maps(
    test_pos: np.ndarray,
    data: dict[str, np.ndarray],
    background: dict[str, object],
) -> list[Path]:
    rows = [
        ("阶段一：v209→v212，归一化时延/PDP补全", "阶段1_PDP增量", "+0.047226线上"),
        ("阶段二：v212→v424，leaked-PAS角度-时延支撑", "阶段2_leakedPAS增量", "+0.008631线上"),
        ("阶段三：v424→v426，完整逐链路PDP", "阶段3_逐链路PDP增量", "+0.005625线上"),
    ]
    regions = regional_masks(test_pos)
    fig, axes = plt.subplots(3, 2, figsize=(20, 24), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("阶段增量", ["#b91c1c", "#fef2f2", "#eff6ff", "#1d4ed8"])
    for row, (title, key, online_text) in enumerate(rows):
        values = data[key]
        norm = symmetric_norm(values)
        last_scatter = None
        for col, (region_name, mask) in enumerate(regions):
            axis = axes[row, col]
            decorate_map(axis, background, region_limits(test_pos, mask), f"{title}\n{region_name}，{online_text}")
            last_scatter = axis.scatter(
                test_pos[mask, 0], test_pos[mask, 1], c=values[mask], cmap=cmap, norm=norm,
                s=54, edgecolor="white", linewidth=0.62, alpha=0.95, zorder=5,
            )
            indices = np.where(mask)[0]
            positive = indices[np.argsort(values[indices])[-4:][::-1]]
            negative = indices[np.argsort(values[indices])[:2]]
            annotate_indices(axis, test_pos, np.concatenate([positive, negative]))
        colorbar = fig.colorbar(last_scatter, ax=axes[row, :], fraction=0.018, pad=0.012)
        colorbar.set_label("接近Sionna诊断参考的增量（蓝色改善，红色下降）")
    fig.suptitle("三个Sionna阶段分别改善了哪些测试点", fontsize=22, fontweight="bold")
    return save_figure(fig, PNG_STAGES)


def plot_total_and_regime(
    test_pos: np.ndarray,
    regimes: np.ndarray,
    data: dict[str, np.ndarray],
    background: dict[str, object],
) -> list[Path]:
    fig = plt.figure(figsize=(21, 17), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.18, 0.82])
    total = data["累计诊断分增量"]
    norm = symmetric_norm(total)
    cmap = LinearSegmentedColormap.from_list("累计改善", ["#991b1b", "#fff7ed", "#ecfdf5", "#047857"])
    for col, (region_name, mask) in enumerate(regional_masks(test_pos)):
        axis = fig.add_subplot(grid[0, col])
        decorate_map(axis, background, region_limits(test_pos, mask), f"{region_name}：v209→v426累计诊断改善")
        scatter = axis.scatter(
            test_pos[mask, 0], test_pos[mask, 1], c=total[mask], cmap=cmap, norm=norm,
            s=62, edgecolor="white", linewidth=0.68, alpha=0.96, zorder=5,
        )
        indices = np.where(mask)[0]
        annotate_indices(axis, test_pos, indices[np.argsort(total[indices])[-7:][::-1]])
    colorbar = fig.colorbar(scatter, ax=fig.axes[:2], fraction=0.022, pad=0.012)
    colorbar.set_label("v426诊断分 - v209诊断分")

    box_axis = fig.add_subplot(grid[1, 0])
    groups = [total[regimes == regime] for regime in REGIMES]
    box = box_axis.boxplot(groups, patch_artist=True, tick_labels=[REGIME_CN[r] for r in REGIMES], showfliers=False)
    colors = ("#dbeafe", "#fef3c7", "#fed7aa", "#fecaca")
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
    box_axis.axhline(0.0, color="#334155", linewidth=1.0)
    box_axis.set_title("累计改善集中在中度和重度传播点")
    box_axis.set_ylabel("逐点累计诊断分增量")
    box_axis.grid(axis="y", alpha=0.2)
    for index, regime in enumerate(REGIMES, start=1):
        values = total[regimes == regime]
        box_axis.text(index, box_axis.get_ylim()[1] * 0.91, f"正向{np.mean(values > 0):.0%}", ha="center", fontsize=10)

    bar_axis = fig.add_subplot(grid[1, 1])
    stage_keys = ("阶段1_诊断分增量", "阶段2_诊断分增量", "阶段3_诊断分增量")
    means = [float(np.mean(data[key])) for key in stage_keys]
    positive_rates = [float(np.mean(data[key] > 0.0)) for key in stage_keys]
    labels = ("时延/PDP补全", "leaked-PAS", "完整逐链路PDP")
    bars = bar_axis.barh(labels, means, color=(STAGE_COLORS[0], STAGE_COLORS[1], STAGE_COLORS[2]))
    bar_axis.axvline(0.0, color="#334155", linewidth=1.0)
    bar_axis.set_title("每个阶段的平均逐点贡献")
    bar_axis.set_xlabel("平均诊断分增量")
    bar_axis.grid(axis="x", alpha=0.2)
    for bar, mean, rate in zip(bars, means, positive_rates):
        bar_axis.text(mean + 0.001, bar.get_y() + bar.get_height() / 2, f"{mean:+.4f}；正向点{rate:.0%}", va="center", fontsize=11)
    fig.suptitle("从v209到v426：500个测试点累计改善在哪里", fontsize=22, fontweight="bold")
    return save_figure(fig, PNG_TOTAL)


def plot_remaining_errors(
    test_pos: np.ndarray,
    data: dict[str, np.ndarray],
    background: dict[str, object],
) -> list[Path]:
    rows = [
        ("v426剩余PDP误差", "v426剩余PDP误差", "时延能量仍不准确"),
        ("v426剩余leaked-PAS误差", "v426剩余leakedPAS误差", "角度-时延支撑仍不准确"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(20, 16.5), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("剩余误差", ["#fff7ed", "#fb923c", "#991b1b"])
    for row, (title, key, explanation) in enumerate(rows):
        values = data[key]
        vmax = max(float(np.quantile(values, 0.98)), 0.05)
        norm = Normalize(vmin=0.0, vmax=vmax)
        last_scatter = None
        for col, (region_name, mask) in enumerate(regional_masks(test_pos)):
            axis = axes[row, col]
            decorate_map(axis, background, region_limits(test_pos, mask), f"{title}\n{region_name}：颜色越深表示{explanation}")
            sizes = 42.0 + 115.0 * np.clip(values[mask] / vmax, 0.0, 1.0)
            last_scatter = axis.scatter(
                test_pos[mask, 0], test_pos[mask, 1], c=values[mask], cmap=cmap, norm=norm,
                s=sizes, edgecolor="white", linewidth=0.65, alpha=0.96, zorder=5,
            )
            indices = np.where(mask)[0]
            annotate_indices(axis, test_pos, indices[np.argsort(values[indices])[-7:][::-1]], color="#7f1d1d")
        colorbar = fig.colorbar(last_scatter, ax=axes[row, :], fraction=0.02, pad=0.012)
        colorbar.set_label("1 - 与Sionna诊断参考的余弦接近度")
    fig.suptitle("v426以后仍然困难的测试点在哪里", fontsize=22, fontweight="bold")
    return save_figure(fig, PNG_REMAINING)


def plot_online_evidence(data: dict[str, np.ndarray]) -> list[Path]:
    transitions = [
        ("v209→v212\n归一化PDP/时延", ONLINE["v212"] - ONLINE["v209"], np.mean(data["阶段1_PDP增量"])),
        ("v212→v424\nleaked-PAS+PDP锁", ONLINE["v424"] - ONLINE["v212"], np.mean(data["阶段2_leakedPAS增量"])),
        ("v424→v426\n完整逐链路PDP", ONLINE["v426"] - ONLINE["v424"], np.mean(data["阶段3_逐链路PDP增量"])),
        ("v426→v427\nconventional PAS", ONLINE["v427"] - ONLINE["v426"], np.mean(data["v427_pas"] - data["v426_pas"])),
        ("v426→v428\nfull leaked slice", ONLINE["v428"] - ONLINE["v426"], np.mean(data["v428_leaked_sh"] - data["v426_leaked_sh"])),
    ]
    labels = [row[0] for row in transitions]
    online_delta = np.asarray([row[1] for row in transitions])
    local_delta = np.asarray([row[2] for row in transitions])
    colors = ["#059669" if value > 0 else "#dc2626" for value in online_delta]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8.5), constrained_layout=True)
    bars = axes[0].barh(labels, online_delta, color=colors)
    axes[0].axvline(0.0, color="#334155", linewidth=1.0)
    axes[0].invert_yaxis()
    axes[0].set_title("真实线上裁决：哪些Sionna信息有效")
    axes[0].set_xlabel("线上分数增量")
    axes[0].grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, online_delta):
        axes[0].text(value + (0.001 if value >= 0 else -0.001), bar.get_y() + bar.get_height() / 2, f"{value:+.6f}", va="center", ha="left" if value >= 0 else "right", fontweight="bold")

    axes[1].scatter(local_delta, online_delta, s=150, c=colors, edgecolor="white", linewidth=1.0)
    for index, label in enumerate(labels):
        axes[1].annotate(label.replace("\n", " "), (local_delta[index], online_delta[index]), xytext=(6, 7), textcoords="offset points", fontsize=10)
    axes[1].axhline(0.0, color="#334155", linewidth=1.0)
    axes[1].axvline(0.0, color="#334155", linewidth=1.0)
    axes[1].set_title("本地接近Sionna并不保证线上提升")
    axes[1].set_xlabel("对应本地参考指标的平均增量")
    axes[1].set_ylabel("真实线上分数增量")
    axes[1].grid(alpha=0.2)
    axes[1].text(
        0.02, 0.03,
        "v427和v428说明：继续逼近同坐标Sionna参考，\n可能只是在拟合错误口径，而不是逼近隐藏真值。",
        transform=axes[1].transAxes, color="#7f1d1d",
        bbox={"boxstyle": "round,pad=0.45", "fc": "#fff1f2", "ec": "#fda4af"},
    )
    fig.suptitle("Sionna真正帮到我们的信息边界", fontsize=22, fontweight="bold")
    return save_figure(fig, PNG_EVIDENCE)


def build_hover_text(
    test_pos: np.ndarray,
    regimes: np.ndarray,
    blocked: np.ndarray,
    data: dict[str, np.ndarray],
) -> np.ndarray:
    return np.asarray(
        [
            (
                f"测试点 T{index:03d}<br>x={test_pos[index,0]:.2f}米，y={test_pos[index,1]:.2f}米"
                f"<br>传播难度：{REGIME_CN[str(regimes[index])]}，遮挡采样数={blocked[index]:.0f}"
                f"<br>阶段一 ΔPDP={data['阶段1_PDP增量'][index]:+.4f}"
                f"<br>阶段二 Δleaked-PAS={data['阶段2_leakedPAS增量'][index]:+.4f}"
                f"<br>阶段三 Δ逐链路PDP={data['阶段3_逐链路PDP增量'][index]:+.4f}"
                f"<br>v209→v426累计诊断分={data['累计诊断分增量'][index]:+.4f}"
                f"<br>主要受益：{STAGE_CN[int(data['主要受益阶段'][index])]}"
                f"<br>v426剩余PDP误差={data['v426剩余PDP误差'][index]:.4f}"
                f"<br>v426剩余leaked-PAS误差={data['v426剩余leakedPAS误差'][index]:.4f}"
            )
            for index in range(len(test_pos))
        ],
        dtype=object,
    )


def write_interactive_map(
    test_pos: np.ndarray,
    regimes: np.ndarray,
    blocked: np.ndarray,
    data: dict[str, np.ndarray],
    background: dict[str, object],
) -> Path:
    wall = background["wall"][::2, ::2]
    x0, x1, y0, y1 = background["extent"]
    x = np.linspace(x0, x1, wall.shape[1])
    y = np.linspace(y0, y1, wall.shape[0])
    hover = build_hover_text(test_pos, regimes, blocked, data)
    metrics = [
        ("累计改善", data["累计诊断分增量"], "v209→v426累计诊断分", "RdBu"),
        ("阶段一：PDP", data["阶段1_PDP增量"], "v209→v212 ΔPDP", "RdBu"),
        ("阶段二：leaked-PAS", data["阶段2_leakedPAS增量"], "v212→v424 Δleaked-PAS", "RdBu"),
        ("阶段三：逐链路PDP", data["阶段3_逐链路PDP增量"], "v424→v426 Δ逐链路PDP", "RdBu"),
        ("v426剩余PDP误差", data["v426剩余PDP误差"], "剩余PDP误差", "YlOrRd"),
        ("v426剩余leaked-PAS误差", data["v426剩余leakedPAS误差"], "剩余leaked-PAS误差", "YlOrRd"),
    ]
    figure = go.Figure()
    figure.add_trace(
        go.Heatmap(
            x=x, y=y, z=wall, colorscale=[[0, "rgba(255,255,255,0)"], [1, "rgba(71,85,105,0.38)"]],
            showscale=False, hoverinfo="skip", name="建筑物背景",
        )
    )
    for metric_index, (name, values, colorbar_title, colorscale) in enumerate(metrics):
        if "剩余" in name:
            cmin, cmax, cmid = 0.0, float(np.quantile(values, 0.98)), None
        else:
            limit = max(float(np.quantile(np.abs(values), 0.975)), 0.005)
            cmin, cmax, cmid = -limit, limit, 0.0
        marker = {
            "size": 11,
            "color": values,
            "colorscale": colorscale,
            "cmin": cmin,
            "cmax": cmax,
            "line": {"color": "white", "width": 0.6},
            "colorbar": {"title": colorbar_title, "thickness": 18},
        }
        if cmid is not None:
            marker["cmid"] = cmid
        figure.add_trace(
            go.Scattergl(
                x=test_pos[:, 0], y=test_pos[:, 1], mode="markers", marker=marker,
                text=hover, hovertemplate="%{text}<extra></extra>", name=name,
                visible=metric_index == 0,
            )
        )
    figure.add_trace(
        go.Scatter(
            x=[BS_POS[0]], y=[BS_POS[1]], mode="markers+text", text=["基站"], textposition="top right",
            marker={"symbol": "star", "size": 18, "color": "#111827", "line": {"color": "white", "width": 1}},
            hovertemplate="基站<extra></extra>", name="基站",
        )
    )
    buttons = []
    for metric_index, (name, _, _, _) in enumerate(metrics):
        visibility = [True] + [index == metric_index for index in range(len(metrics))] + [True]
        buttons.append(
            {
                "label": name,
                "method": "update",
                "args": [
                    {"visible": visibility},
                    {"title": f"500个测试点：{name}<br><sup>颜色基于同坐标Sionna诊断参考，不是官方隐藏测试真值</sup>"},
                ],
            }
        )
    figure.update_layout(
        title="500个测试点：累计改善<br><sup>可缩放、拖动并悬停查看每个测试点</sup>",
        width=1450,
        height=1050,
        template="plotly_white",
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 0.01, "y": 1.08, "xanchor": "left"}],
        margin={"l": 70, "r": 110, "t": 120, "b": 70},
        xaxis={"title": "x（米）", "scaleanchor": "y", "scaleratio": 1},
        yaxis={"title": "y（米）"},
        legend={"orientation": "h", "y": -0.08},
    )
    figure.write_html(HTML_INTERACTIVE, include_plotlyjs=True, full_html=True)
    return HTML_INTERACTIVE


def write_details(
    test_pos: np.ndarray,
    regimes: np.ndarray,
    blocked: np.ndarray,
    data: dict[str, np.ndarray],
) -> tuple[Path, dict[str, object]]:
    frame = pd.DataFrame(
        {
            "测试点编号": [f"T{index:03d}" for index in range(len(test_pos))],
            "数组索引": np.arange(len(test_pos)),
            "x坐标_米": test_pos[:, 0],
            "y坐标_米": test_pos[:, 1],
            "传播难度": [REGIME_CN[str(value)] for value in regimes],
            "遮挡采样数": blocked,
            "阶段1_PDP增量_v209到v212": data["阶段1_PDP增量"],
            "阶段2_leakedPAS增量_v212到v424": data["阶段2_leakedPAS增量"],
            "阶段3_逐链路PDP增量_v424到v426": data["阶段3_逐链路PDP增量"],
            "累计诊断分增量_v209到v426": data["累计诊断分增量"],
            "主要受益阶段": [STAGE_CN[int(value)] for value in data["主要受益阶段"]],
            "v426_PDP接近度": data["v426_pdp"],
            "v426_leakedPAS接近度": data["v426_leaked_sh"],
            "v426剩余PDP误差": data["v426剩余PDP误差"],
            "v426剩余leakedPAS误差": data["v426剩余leakedPAS误差"],
            "v428相对v426_leakedPAS变化": data["v428_leaked_sh"] - data["v426_leaked_sh"],
        }
    )
    frame.to_csv(CSV_DETAILS, index=False, encoding="utf-8-sig")
    dominant = data["主要受益阶段"]
    summary = {
        "测试点总数": int(len(test_pos)),
        "区域计数": {
            NORTH_MASK_NAME: int(np.sum(test_pos[:, 1] >= -110.0)),
            SOUTH_MASK_NAME: int(np.sum(test_pos[:, 1] < -110.0)),
        },
        "主要受益阶段计数": {STAGE_CN[stage]: int(np.sum(dominant == stage)) for stage in range(4)},
        "阶段平均增量": {
            "阶段1_PDP": float(np.mean(data["阶段1_PDP增量"])),
            "阶段2_leakedPAS": float(np.mean(data["阶段2_leakedPAS增量"])),
            "阶段3_逐链路PDP": float(np.mean(data["阶段3_逐链路PDP增量"])),
            "累计诊断分": float(np.mean(data["累计诊断分增量"])),
        },
        "累计正向点比例": float(np.mean(data["累计诊断分增量"] > 0.0)),
        "v426剩余误差均值": {
            "PDP": float(np.mean(data["v426剩余PDP误差"])),
            "leakedPAS": float(np.mean(data["v426剩余leakedPAS误差"])),
        },
        "警告": "逐点颜色和增量使用同坐标v207 Sionna reference，仅用于解释注入机制，不是官方隐藏测试真值。",
    }
    JSON_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return CSV_DETAILS, summary


def main() -> None:
    configure_chinese_font()
    validate_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    test_pos = np.load(TEST_POS).astype(np.float64)
    geometry = np.load(GEOMETRY)
    blocked = geometry["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked).astype("U8")
    background = load_background()

    if METRICS.is_file():
        cached = np.load(METRICS)
        data = {
            key: cached[key]
            for key in cached.files
            if key not in {"test_pos", "blocked", "regimes"}
        }
    else:
        data = compute_point_metrics()
        add_derived_metrics(data)
        np.savez_compressed(
            METRICS,
            test_pos=test_pos.astype(np.float32),
            blocked=blocked,
            regimes=regimes,
            **data,
        )

    outputs: list[Path] = []
    outputs.extend(plot_dominant_overview(test_pos, data, background))
    outputs.extend(plot_stage_maps(test_pos, data, background))
    outputs.extend(plot_total_and_regime(test_pos, regimes, data, background))
    outputs.extend(plot_remaining_errors(test_pos, data, background))
    outputs.extend(plot_online_evidence(data))
    outputs.append(write_interactive_map(test_pos, regimes, blocked, data, background))
    csv_path, summary = write_details(test_pos, regimes, blocked, data)
    outputs.extend([csv_path, JSON_SUMMARY, METRICS])

    report = {
        "method": VERSION,
        "status": "analysis_complete_no_candidate",
        "primary_question": "Across all 500 official test coordinates, what did the Sionna-assisted v209-v426 stages change, and where are the remaining errors?",
        "reference_boundary": {
            "reference": PATHS.relative(REFERENCE),
            "warning": "same-coordinate Sionna diagnostic reference, not official hidden truth",
        },
        "versions": {name: PATHS.relative(path) for name, path in CHANNELS.items()},
        "online_scores": ONLINE,
        "stage_definitions": {
            "stage_1": "v209->v212 normalized PDP/delay completion",
            "stage_2": "v212->v424 leaked PAS angle-delay support with full PDP lock",
            "stage_3": "v424->v426 complete 256x4 linkwise PDP completion",
        },
        "summary": summary,
        "new_dependency": {
            "name": "plotly",
            "purpose": "offline interactive 500-point map with zoom and per-point hover details",
            "environment_files_updated": ["environment.yml", "environment.cuda.yml"],
        },
        "outputs": [PATHS.relative(path) for path in outputs],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
