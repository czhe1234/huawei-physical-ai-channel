#!/usr/bin/env python3
"""Plot official unseen medium/heavy test locations on the point-cloud map."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from build_map_features import read_ply_memmap
from powerfield_common import BS_POS, regime_of
from project_paths import PATHS


GEOMETRY = PATHS.shared_features / "height_geometry_cache.npz"
SCENE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"
OUTPUT_DIR = PATHS.analysis / "visualizations" / "未见中重度测试点位置"
PNG_PATH = OUTPUT_DIR / "未见中重度测试点位置总览.png"
PDF_PATH = OUTPUT_DIR / "未见中重度测试点位置总览.pdf"
CSV_PATH = OUTPUT_DIR / "中重度测试点坐标与编号.csv"
SUMMARY_PATH = OUTPUT_DIR / "中重度测试点位置摘要.json"
POINT_CLOUD_PNG = OUTPUT_DIR / "官方原始点云与中重度测试点.png"
POINT_CLOUD_PDF = OUTPUT_DIR / "官方原始点云与中重度测试点.pdf"

REGIME_COLORS = {
    "clear": "#9aa5b1",
    "light": "#d9a441",
    "medium": "#ef8a27",
    "heavy": "#c92f3b",
}
REGIME_NAMES = {
    "clear": "无遮挡",
    "light": "轻遮挡",
    "medium": "中遮挡",
    "heavy": "重遮挡",
}


def configure_chinese_font() -> None:
    selected = "DejaVu Sans"
    for family in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"):
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
            "font.size": 11,
        }
    )


def validate_inputs() -> None:
    required = (
        PATHS.round1 / "Round1_Train_Pos.npy",
        PATHS.round1 / "Round1_Test_Pos.npy",
        GEOMETRY,
        SCENE,
    )
    missing = [PATHS.relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing visualization inputs: {missing}")


def load_background() -> dict[str, object]:
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
        "vmax": max(float(np.quantile(positive, 0.97)), 1.0),
    }


def sample_point_cloud(max_points: int = 360_000, seed: int = 20260719) -> dict[str, np.ndarray]:
    vertices = read_ply_memmap(PATHS.round1 / "Round1_Map.ply")
    rng = np.random.default_rng(seed)
    count = min(max_points, len(vertices))
    indices = np.sort(rng.choice(len(vertices), size=count, replace=False))
    return {
        name: np.asarray(vertices[name][indices], dtype=np.float32)
        for name in ("x", "y", "z", "nz")
    }


def map_limits(train_pos: np.ndarray, test_pos: np.ndarray) -> tuple[float, float, float, float]:
    points = np.concatenate([train_pos[:, :2], test_pos[:, :2]], axis=0)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    return x_min - 10.0, x_max + 10.0, y_min - 10.0, y_max + 10.0


def decorate_map(
    axis: plt.Axes,
    background: dict[str, object],
    limits: tuple[float, float, float, float],
) -> None:
    axis.set_facecolor("#f7f8fa")
    axis.imshow(
        background["wall"],
        origin="lower",
        extent=background["extent"],
        cmap="Greys",
        vmin=0.0,
        vmax=background["vmax"],
        alpha=0.34,
        interpolation="nearest",
        zorder=0,
    )
    axis.scatter(
        [BS_POS[0]],
        [BS_POS[1]],
        marker="*",
        s=220,
        color="#17191c",
        edgecolor="white",
        linewidth=1.2,
        zorder=8,
        label="基站",
    )
    axis.set_xlim(limits[0], limits[1])
    axis.set_ylim(limits[2], limits[3])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("横坐标 x（米）")
    axis.set_ylabel("纵坐标 y（米）")
    axis.grid(color="#6d7782", alpha=0.15, linewidth=0.6)


def plot_overview(
    axis: plt.Axes,
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    regimes: np.ndarray,
    background: dict[str, object],
    limits: tuple[float, float, float, float],
) -> None:
    decorate_map(axis, background, limits)
    axis.scatter(
        train_pos[:, 0],
        train_pos[:, 1],
        s=7,
        marker=".",
        color="#7790a6",
        alpha=0.22,
        linewidths=0,
        zorder=1,
        label="已知训练位置",
    )
    background_mask = np.isin(regimes, ("clear", "light"))
    axis.scatter(
        test_pos[background_mask, 0],
        test_pos[background_mask, 1],
        s=20,
        marker="o",
        facecolor="#f5f6f7",
        edgecolor="#7f8994",
        linewidth=0.65,
        alpha=0.82,
        zorder=3,
        label="测试点：无遮挡/轻遮挡",
    )
    medium = regimes == "medium"
    axis.scatter(
        test_pos[medium, 0],
        test_pos[medium, 1],
        s=52,
        marker="o",
        facecolor=REGIME_COLORS["medium"],
        edgecolor="white",
        linewidth=0.8,
        alpha=0.96,
        zorder=5,
        label=f"测试点：中遮挡（{int(np.sum(medium))}）",
    )
    heavy = regimes == "heavy"
    axis.scatter(
        test_pos[heavy, 0],
        test_pos[heavy, 1],
        s=48,
        marker="D",
        facecolor=REGIME_COLORS["heavy"],
        edgecolor="white",
        linewidth=0.7,
        alpha=0.92,
        zorder=4,
        label=f"测试点：重遮挡（{int(np.sum(heavy))}）",
    )
    axis.set_title(
        "总览：官方测试集中未提供信道真值的中/重遮挡位置",
        fontsize=15,
        fontweight="bold",
        pad=10,
    )
    handles, labels = axis.get_legend_handles_labels()
    order = [1, 2, 3, 4, 0]
    axis.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=0.94,
        edgecolor="#c6ccd2",
        fontsize=10,
    )


def plot_regime_detail(
    axis: plt.Axes,
    test_pos: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
    regime: str,
    background: dict[str, object],
    limits: tuple[float, float, float, float],
) -> None:
    decorate_map(axis, background, limits)
    other = regimes != regime
    axis.scatter(
        test_pos[other, 0],
        test_pos[other, 1],
        s=9,
        marker="o",
        facecolor="#d8dde2",
        edgecolor="none",
        alpha=0.34,
        zorder=2,
    )
    mask = regimes == regime
    values = blocked[mask].astype(np.float64)
    if regime == "medium":
        cmap = LinearSegmentedColormap.from_list(
            "medium_blocked", ("#ffd777", "#ef8a27", "#a94b13")
        )
        marker = "o"
        limits_color = (25.0, 100.0)
    else:
        cmap = LinearSegmentedColormap.from_list(
            "heavy_blocked", ("#f6a4a0", "#cf3f46", "#741724")
        )
        marker = "D"
        limits_color = (100.0, 205.0)
    scatter = axis.scatter(
        test_pos[mask, 0],
        test_pos[mask, 1],
        c=values,
        s=46 if regime == "medium" else 38,
        marker=marker,
        cmap=cmap,
        vmin=limits_color[0],
        vmax=limits_color[1],
        edgecolor="white",
        linewidth=0.6,
        alpha=0.96,
        zorder=5,
    )
    axis.set_title(
        f"{REGIME_NAMES[regime]}测试点：{int(np.sum(mask))} 个",
        fontsize=13,
        fontweight="bold",
    )
    colorbar = axis.figure.colorbar(scatter, ax=axis, fraction=0.035, pad=0.02)
    colorbar.set_label("BS-UE 直线路径遮挡采样数")


def style_3d_axis(axis: plt.Axes, *, title: str, elev: float, azim: float) -> None:
    axis.view_init(elev=elev, azim=azim)
    axis.set_title(title, fontsize=13, fontweight="bold", pad=8)
    axis.set_xlabel("x（米）", labelpad=7)
    axis.set_ylabel("y（米）", labelpad=7)
    axis.set_zlabel("高度 z（米）", labelpad=5)
    axis.grid(False)
    axis.xaxis.pane.set_facecolor((0.96, 0.97, 0.98, 1.0))
    axis.yaxis.pane.set_facecolor((0.96, 0.97, 0.98, 1.0))
    axis.zaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))


def scatter_point_cloud_3d(
    axis: plt.Axes,
    points: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    point_size: float,
    alpha: float,
) -> None:
    axis.scatter(
        points["x"][mask],
        points["y"][mask],
        points["z"][mask],
        c=points["z"][mask],
        s=point_size,
        cmap="terrain",
        vmin=0.0,
        vmax=50.0,
        alpha=alpha,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )


def overlay_targets_3d(
    axis: plt.Axes,
    test_pos: np.ndarray,
    regimes: np.ndarray,
) -> None:
    for regime, marker, size in (("medium", "o", 24), ("heavy", "D", 20)):
        mask = regimes == regime
        axis.scatter(
            test_pos[mask, 0],
            test_pos[mask, 1],
            test_pos[mask, 2] + 0.35,
            s=size,
            marker=marker,
            color=REGIME_COLORS[regime],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.94,
            depthshade=False,
        )
    axis.scatter(
        [BS_POS[0]],
        [BS_POS[1]],
        [BS_POS[2]],
        marker="*",
        s=130,
        color="#14171a",
        edgecolor="white",
        linewidth=0.8,
        depthshade=False,
    )


def plot_raw_point_cloud(
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    regimes: np.ndarray,
    limits: tuple[float, float, float, float],
) -> None:
    points = sample_point_cloud()
    finite = np.isfinite(points["x"]) & np.isfinite(points["y"]) & np.isfinite(points["z"])
    zoom = (
        finite
        & (points["x"] >= limits[0])
        & (points["x"] <= limits[1])
        & (points["y"] >= limits[2])
        & (points["y"] <= limits[3])
    )
    structures = zoom & (points["z"] > 0.5)
    full_structures = finite & (points["z"] > 0.5)

    figure = plt.figure(figsize=(20, 16), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)

    full_axis = figure.add_subplot(grid[0, 0], projection="3d")
    scatter_point_cloud_3d(
        full_axis, points, full_structures, point_size=0.55, alpha=0.36
    )
    style_3d_axis(
        full_axis,
        title="完整官方点云：三维鸟瞰",
        elev=43,
        azim=-57,
    )
    full_axis.set_box_aspect((1.25, 1.0, 0.22))

    top_axis = figure.add_subplot(grid[0, 1])
    scatter = top_axis.scatter(
        points["x"][zoom],
        points["y"][zoom],
        c=points["z"][zoom],
        s=0.55,
        cmap="terrain",
        vmin=0.0,
        vmax=50.0,
        alpha=0.34,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    top_axis.scatter(
        train_pos[:, 0],
        train_pos[:, 1],
        s=5,
        color="#71879a",
        alpha=0.16,
        linewidths=0,
        label="训练位置",
        zorder=2,
    )
    for regime, marker, size in (("medium", "o", 36), ("heavy", "D", 30)):
        mask = regimes == regime
        top_axis.scatter(
            test_pos[mask, 0],
            test_pos[mask, 1],
            s=size,
            marker=marker,
            color=REGIME_COLORS[regime],
            edgecolor="white",
            linewidth=0.55,
            alpha=0.94,
            label=f"{REGIME_NAMES[regime]}测试点",
            zorder=4,
        )
    top_axis.scatter(
        [BS_POS[0]],
        [BS_POS[1]],
        marker="*",
        s=180,
        color="#14171a",
        edgecolor="white",
        linewidth=0.9,
        label="基站",
        zorder=5,
    )
    top_axis.set_xlim(limits[0], limits[1])
    top_axis.set_ylim(limits[2], limits[3])
    top_axis.set_aspect("equal", adjustable="box")
    top_axis.set_title("测试区域：原始点云俯视与目标点", fontsize=13, fontweight="bold")
    top_axis.set_xlabel("x（米）")
    top_axis.set_ylabel("y（米）")
    top_axis.grid(alpha=0.14, linewidth=0.6)
    top_axis.legend(loc="upper right", fontsize=9, framealpha=0.93)
    colorbar = figure.colorbar(scatter, ax=top_axis, fraction=0.036, pad=0.02)
    colorbar.set_label("原始点云高度 z（米）")

    for subplot, elev, azim, title in (
        (grid[1, 0], 28, -68, "测试区域三维点云：西南视角"),
        (grid[1, 1], 24, 118, "测试区域三维点云：东北视角"),
    ):
        axis = figure.add_subplot(subplot, projection="3d")
        scatter_point_cloud_3d(axis, points, structures, point_size=0.85, alpha=0.34)
        overlay_targets_3d(axis, test_pos, regimes)
        axis.set_xlim(limits[0], limits[1])
        axis.set_ylim(limits[2], limits[3])
        axis.set_zlim(0.0, 55.0)
        axis.set_box_aspect((0.9, 1.15, 0.42))
        style_3d_axis(axis, title=title, elev=elev, azim=azim)

    figure.suptitle(
        "Round1_Map.ply 原始三维点云与未见中/重遮挡测试点",
        fontsize=19,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.003,
        "点云按真实高度着色；三维面板为突出建筑隐藏 z<=0.5 米地面层，俯视面板保留全部原始点。"
        "测试点位于官方 UE 高度 z=1.5 米。",
        ha="center",
        fontsize=10,
        color="#4b535b",
    )
    figure.savefig(POINT_CLOUD_PNG, dpi=200, facecolor="white")
    figure.savefig(POINT_CLOUD_PDF, dpi=160, facecolor="white")
    plt.close(figure)


def write_point_table(
    test_pos: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
) -> None:
    target = np.isin(regimes, ("medium", "heavy"))
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(("测试点编号", "横坐标x_米", "纵坐标y_米", "高度z_米", "遮挡采样数", "分区"))
        for index in np.flatnonzero(target):
            writer.writerow(
                (
                    int(index),
                    f"{test_pos[index, 0]:.6f}",
                    f"{test_pos[index, 1]:.6f}",
                    f"{test_pos[index, 2]:.6f}",
                    int(blocked[index]),
                    REGIME_NAMES[str(regimes[index])],
                )
            )


def main() -> None:
    validate_inputs()
    configure_chinese_font()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    geometry = np.load(GEOMETRY)
    blocked = geometry["test_blocked"].astype(np.int32)
    regimes = regime_of(blocked).astype(str)
    background = load_background()
    limits = map_limits(train_pos, test_pos)

    figure = plt.figure(figsize=(18, 20), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.18, 1.0))
    overview = figure.add_subplot(grid[0, :])
    medium_axis = figure.add_subplot(grid[1, 0])
    heavy_axis = figure.add_subplot(grid[1, 1])

    plot_overview(overview, train_pos, test_pos, regimes, background, limits)
    plot_regime_detail(
        medium_axis, test_pos, blocked, regimes, "medium", background, limits
    )
    plot_regime_detail(
        heavy_axis, test_pos, blocked, regimes, "heavy", background, limits
    )
    target_count = int(np.sum(np.isin(regimes, ("medium", "heavy"))))
    figure.suptitle(
        f"未见中/重遮挡测试点空间分布（{target_count}/500）",
        fontsize=20,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.002,
        "定义：中遮挡为 25 < 遮挡采样数 <= 100，重遮挡为遮挡采样数 > 100。"
        "坐标来自官方测试集；灰色建筑背景来自 Round1_Map.ply。",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#4b535b",
    )
    figure.savefig(PNG_PATH, dpi=220, facecolor="white")
    figure.savefig(PDF_PATH, facecolor="white")
    plt.close(figure)

    plot_raw_point_cloud(train_pos, test_pos, regimes, limits)

    write_point_table(test_pos, blocked, regimes)
    counts = {name: int(np.sum(regimes == name)) for name in REGIME_NAMES}
    summary = {
        "definition": {
            "unseen": "official test position without provided channel ground truth",
            "medium": "25 < test_blocked <= 100",
            "heavy": "test_blocked > 100",
        },
        "counts": counts,
        "medium_heavy_count": target_count,
        "medium_heavy_fraction": target_count / len(test_pos),
        "outputs": {
            "png": PATHS.relative(PNG_PATH),
            "pdf": PATHS.relative(PDF_PATH),
            "point_table": PATHS.relative(CSV_PATH),
            "raw_point_cloud_png": PATHS.relative(POINT_CLOUD_PNG),
            "raw_point_cloud_pdf": PATHS.relative(POINT_CLOUD_PDF),
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
