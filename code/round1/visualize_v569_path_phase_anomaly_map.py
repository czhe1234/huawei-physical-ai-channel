#!/usr/bin/env python3
"""Visualize local path-phase anomalies on the official point cloud."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

import analyze_v569_path_tangent_noise_inversion as v569
from build_map_features import BS_POS, read_ply_memmap
from project_paths import PATHS


FIG_DIR = v569.FIG_DIR
CSV_PATH = v569.REPORT_DIR / "v569_path_phase_spatial_anomalies.csv"
SUMMARY_PATH = v569.REPORT_DIR / "v569_path_phase_spatial_anomaly_summary.json"

NEIGHBOR_RADIUS_M = 7.5
KERNEL_SCALE_M = 3.0
MAX_NEIGHBORS = 8
MIN_NEIGHBORS = 2
MIN_VALID_CHANNELS = 4
POINT_CLOUD_LIMIT = 320_000
TOP_PATH_LIMIT = 40
RNG_SEED = 569_071
TINY = 1e-30

REGIME_NAMES = {
    "clear": "无遮挡",
    "light": "轻遮挡",
    "medium": "中遮挡",
    "heavy": "重遮挡",
}
REGIME_COLORS = {
    "clear": "#3b82b8",
    "light": "#2a9d78",
    "medium": "#e09f3e",
    "heavy": "#c44536",
}


def summarize(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0}
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def regime_labels(blocked: np.ndarray) -> np.ndarray:
    blocked = np.asarray(blocked, dtype=np.int32)
    return np.where(
        blocked <= 0,
        "clear",
        np.where(blocked <= 25, "light", np.where(blocked <= 100, "medium", "heavy")),
    )


def compute_local_phase_anomaly(
    observables: dict[str, np.ndarray],
    quality_mask: np.ndarray,
    positions_xy: np.ndarray,
) -> dict[str, np.ndarray]:
    row_count = len(observables["point"])
    anomaly = np.full(row_count, np.nan, dtype=np.float64)
    neighbor_count = np.zeros(row_count, dtype=np.int16)
    mean_neighbor_distance = np.full(row_count, np.nan, dtype=np.float64)
    local_concentration = np.full(row_count, np.nan, dtype=np.float64)
    valid_channel_count = np.zeros(row_count, dtype=np.int8)

    for patch_id in np.unique(observables["patch"][quality_mask]):
        rows = np.flatnonzero(quality_mask & (observables["patch"] == patch_id))
        if len(rows) < MIN_NEIGHBORS + 1:
            continue
        local_xy = positions_xy[observables["point"][rows]]
        tree = cKDTree(local_xy)
        for local_index, row in enumerate(rows):
            local_neighbors = np.asarray(
                tree.query_ball_point(local_xy[local_index], NEIGHBOR_RADIUS_M),
                dtype=np.int64,
            )
            local_neighbors = local_neighbors[local_neighbors != local_index]
            if len(local_neighbors) < MIN_NEIGHBORS:
                continue
            distance = np.linalg.norm(
                local_xy[local_neighbors] - local_xy[local_index], axis=1
            )
            order = np.argsort(distance)[:MAX_NEIGHBORS]
            local_neighbors = local_neighbors[order]
            distance = distance[order]
            neighbor_rows = rows[local_neighbors]

            channel_delta: list[float] = []
            channel_weight: list[float] = []
            channel_concentration: list[float] = []
            for channel in range(observables["phase_geometry"].shape[1]):
                valid = (
                    observables["channel_valid"][row, channel]
                    & observables["channel_valid"][neighbor_rows, channel]
                )
                if int(np.sum(valid)) < MIN_NEIGHBORS:
                    continue
                local_rows = neighbor_rows[valid]
                local_distance = distance[valid]
                weight = np.exp(-0.5 * (local_distance / KERNEL_SCALE_M) ** 2)
                weight *= np.sqrt(
                    np.maximum(
                        observables["mass"][row]
                        * observables["mass"][local_rows],
                        TINY,
                    )
                )
                weight *= np.sqrt(
                    np.maximum(
                        observables["coherence"][row, channel]
                        * observables["coherence"][local_rows, channel],
                        0.0,
                    )
                )
                weight_sum = float(np.sum(weight))
                if weight_sum <= 0.0:
                    continue
                phasor = np.sum(
                    weight
                    * np.exp(1j * observables["phase_geometry"][local_rows, channel])
                )
                if abs(phasor) <= TINY:
                    continue
                prediction = float(np.angle(phasor))
                delta = float(
                    np.angle(
                        np.exp(
                            1j
                            * (
                                observables["phase_geometry"][row, channel]
                                - prediction
                            )
                        )
                    )
                )
                channel_delta.append(delta)
                channel_weight.append(float(observables["coherence"][row, channel]))
                channel_concentration.append(float(abs(phasor) / weight_sum))

            if len(channel_delta) < MIN_VALID_CHANNELS:
                continue
            delta = np.asarray(channel_delta, dtype=np.float64)
            weight = np.asarray(channel_weight, dtype=np.float64)
            anomaly[row] = float(
                np.sqrt(
                    np.average(2.0 * (1.0 - np.cos(delta)), weights=weight)
                )
            )
            neighbor_count[row] = len(neighbor_rows)
            mean_neighbor_distance[row] = float(np.mean(distance))
            local_concentration[row] = float(np.mean(channel_concentration))
            valid_channel_count[row] = len(channel_delta)

    return {
        "anomaly_rad": anomaly,
        "anomaly_deg": np.degrees(anomaly),
        "neighbor_count": neighbor_count,
        "mean_neighbor_distance_m": mean_neighbor_distance,
        "local_concentration": local_concentration,
        "valid_channel_count": valid_channel_count,
    }


def aggregate_points(
    observables: dict[str, np.ndarray], anomaly_deg: np.ndarray, point_count: int
) -> dict[str, np.ndarray]:
    point_anomaly = np.full(point_count, np.nan, dtype=np.float64)
    point_row = np.full(point_count, -1, dtype=np.int32)
    point_path_count = np.zeros(point_count, dtype=np.int16)
    valid_rows = np.flatnonzero(np.isfinite(anomaly_deg))
    for row in valid_rows:
        point_id = int(observables["point"][row])
        point_path_count[point_id] += 1
        if not np.isfinite(point_anomaly[point_id]) or anomaly_deg[row] > point_anomaly[point_id]:
            point_anomaly[point_id] = anomaly_deg[row]
            point_row[point_id] = row
    return {
        "anomaly_deg": point_anomaly,
        "row": point_row,
        "path_count": point_path_count,
    }


def sample_point_cloud(
    limits: tuple[float, float, float, float]
) -> dict[str, np.ndarray]:
    vertices = read_ply_memmap(PATHS.round1 / "Round1_Map.ply")
    x = np.asarray(vertices["x"])
    y = np.asarray(vertices["y"])
    z = np.asarray(vertices["z"])
    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
        & (x >= limits[0])
        & (x <= limits[1])
        & (y >= limits[2])
        & (y <= limits[3])
        & (z > 0.35)
    )
    indices = np.flatnonzero(mask)
    if len(indices) > POINT_CLOUD_LIMIT:
        rng = np.random.default_rng(RNG_SEED)
        indices = np.sort(rng.choice(indices, size=POINT_CLOUD_LIMIT, replace=False))
    return {
        "x": np.asarray(x[indices], dtype=np.float32),
        "y": np.asarray(y[indices], dtype=np.float32),
        "z": np.asarray(z[indices], dtype=np.float32),
    }


def draw_point_cloud_2d(
    axis: plt.Axes,
    cloud: dict[str, np.ndarray],
    limits: tuple[float, float, float, float],
) -> None:
    axis.scatter(
        cloud["x"],
        cloud["y"],
        c=np.clip(cloud["z"], 0.0, 35.0),
        s=0.42,
        cmap="Greys",
        vmin=0.0,
        vmax=35.0,
        alpha=0.24,
        linewidths=0,
        rasterized=True,
        zorder=0,
    )
    axis.scatter(
        [BS_POS[0]],
        [BS_POS[1]],
        marker="*",
        s=230,
        color="#111827",
        edgecolor="white",
        linewidth=1.0,
        zorder=8,
        label="基站",
    )
    axis.set_xlim(limits[0], limits[1])
    axis.set_ylim(limits[2], limits[3])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("横坐标 x（米）")
    axis.set_ylabel("纵坐标 y（米）")
    axis.grid(alpha=0.13, linewidth=0.6)


def draw_point_anomaly_map(
    axis: plt.Axes,
    cloud: dict[str, np.ndarray],
    limits: tuple[float, float, float, float],
    positions: np.ndarray,
    point_data: dict[str, np.ndarray],
    norm: Normalize,
    cmap: matplotlib.colors.Colormap,
    title: str,
) -> plt.Collection:
    draw_point_cloud_2d(axis, cloud, limits)
    valid = np.isfinite(point_data["anomaly_deg"])
    point_ids = np.flatnonzero(valid)
    size = 22.0 + 11.0 * np.minimum(point_data["path_count"][point_ids], 4)
    scatter = axis.scatter(
        positions[point_ids, 0],
        positions[point_ids, 1],
        c=point_data["anomaly_deg"][point_ids],
        s=size,
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.94,
        zorder=4,
    )
    threshold = float(np.quantile(point_data["anomaly_deg"][valid], 0.90))
    top = valid & (point_data["anomaly_deg"] >= threshold)
    axis.scatter(
        positions[top, 0],
        positions[top, 1],
        s=78,
        facecolor="none",
        edgecolor="#111827",
        linewidth=1.0,
        zorder=5,
        label="异常最高10%",
    )
    axis.set_title(title, fontsize=13, fontweight="bold")
    axis.legend(loc="upper right", fontsize=9, framealpha=0.92)
    return scatter


def top_path_rows(anomaly_deg: np.ndarray, limit: int = TOP_PATH_LIMIT) -> np.ndarray:
    valid_rows = np.flatnonzero(np.isfinite(anomaly_deg))
    if len(valid_rows) == 0:
        return valid_rows
    threshold = float(np.quantile(anomaly_deg[valid_rows], 0.90))
    rows = valid_rows[anomaly_deg[valid_rows] >= threshold]
    order = np.argsort(anomaly_deg[rows])[::-1]
    return rows[order[:limit]]


def draw_candidate_paths_2d(
    axis: plt.Axes,
    rows: np.ndarray,
    observables: dict[str, np.ndarray],
    positions: np.ndarray,
    reflection_points: np.ndarray,
    anomaly_deg: np.ndarray,
    norm: Normalize,
    cmap: matplotlib.colors.Colormap,
    label_count: int = 10,
) -> None:
    for rank, row in enumerate(rows):
        point_id = int(observables["point"][row])
        reflection = reflection_points[row]
        receiver = positions[point_id]
        color = cmap(norm(anomaly_deg[row]))
        axis.plot(
            [BS_POS[0], reflection[0], receiver[0]],
            [BS_POS[1], reflection[1], receiver[1]],
            color=color,
            linewidth=1.0 if rank >= 10 else 1.6,
            alpha=0.58 if rank >= 10 else 0.86,
            linestyle="--",
            zorder=2,
        )
        axis.scatter(
            [reflection[0]],
            [reflection[1]],
            marker="x",
            s=30,
            color=color,
            linewidth=1.0,
            zorder=5,
        )
        axis.scatter(
            [receiver[0]],
            [receiver[1]],
            marker="o",
            s=42,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=6,
        )
        if rank < label_count:
            label = axis.text(
                receiver[0] + 1.2,
                receiver[1] + 1.2,
                f"点{point_id}",
                fontsize=8,
                color="#111827",
                zorder=9,
            )
            label.set_path_effects(
                [path_effects.withStroke(linewidth=2.2, foreground="white")]
            )


def draw_candidate_paths_3d(
    axis: plt.Axes,
    cloud: dict[str, np.ndarray],
    rows: np.ndarray,
    observables: dict[str, np.ndarray],
    positions: np.ndarray,
    reflection_points: np.ndarray,
    anomaly_deg: np.ndarray,
    norm: Normalize,
    cmap: matplotlib.colors.Colormap,
    limits: tuple[float, float, float, float],
) -> None:
    stride = max(len(cloud["x"]) // 90_000, 1)
    axis.scatter(
        cloud["x"][::stride],
        cloud["y"][::stride],
        cloud["z"][::stride],
        s=0.35,
        c="#606a73",
        alpha=0.075,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )
    for row in rows[:30]:
        point_id = int(observables["point"][row])
        reflection = reflection_points[row]
        receiver = positions[point_id]
        color = cmap(norm(anomaly_deg[row]))
        axis.plot(
            [BS_POS[0], reflection[0], receiver[0]],
            [BS_POS[1], reflection[1], receiver[1]],
            [BS_POS[2], reflection[2], receiver[2]],
            color=color,
            linewidth=1.25,
            alpha=0.78,
        )
        axis.scatter(
            [reflection[0]],
            [reflection[1]],
            [reflection[2]],
            marker="x",
            s=16,
            color=color,
            depthshade=False,
        )
    axis.scatter(
        [BS_POS[0]],
        [BS_POS[1]],
        [BS_POS[2]],
        marker="*",
        s=125,
        color="#111827",
        edgecolor="white",
        linewidth=0.8,
        depthshade=False,
    )
    axis.set_xlim(limits[0], limits[1])
    axis.set_ylim(limits[2], limits[3])
    axis.set_zlim(0.0, 52.0)
    axis.set_box_aspect((limits[1] - limits[0], limits[3] - limits[2], 105.0))
    axis.view_init(elev=29, azim=-61)
    axis.set_xlabel("x（米）")
    axis.set_ylabel("y（米）")
    axis.set_zlabel("z（米）")
    axis.set_title("最高异常路径候选的三维位置", fontsize=13, fontweight="bold")


def correlation_summary(
    observables: dict[str, np.ndarray],
    result: dict[str, np.ndarray],
    blocked: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    anomaly = result["anomaly_deg"]
    row_blocked = blocked[observables["point"]]
    variables = {
        "遮挡计数": row_blocked,
        "路径质量mass": observables["mass"],
        "路径长度匹配误差": observables["length_error_bins"],
        "水平角匹配误差": observables["h_error_bins"],
        "中位阵列相干度": np.median(observables["coherence"], axis=1),
        "可用近邻数量": result["neighbor_count"],
        "邻域相位集中度": result["local_concentration"],
    }
    output: dict[str, dict[str, float | int]] = {}
    for name, values in variables.items():
        valid = np.isfinite(anomaly) & np.isfinite(values)
        statistic, pvalue = spearmanr(values[valid], anomaly[valid])
        output[name] = {
            "count": int(np.sum(valid)),
            "spearman": float(statistic),
            "pvalue": float(pvalue),
        }
    return output


def write_csv(
    observables: dict[str, np.ndarray],
    positions: np.ndarray,
    reflection_points: np.ndarray,
    masks: dict[str, np.ndarray],
    blocked: np.ndarray,
    regimes: np.ndarray,
    strict_result: dict[str, np.ndarray],
    ultra_result: dict[str, np.ndarray],
) -> None:
    fields = (
        "row_id",
        "point_id",
        "component_id",
        "candidate_id",
        "patch_id",
        "regime",
        "blocked_count",
        "strict_quality",
        "ultra_quality",
        "ue_x",
        "ue_y",
        "ue_z",
        "reflection_x",
        "reflection_y",
        "reflection_z",
        "strict_anomaly_deg",
        "ultra_anomaly_deg",
        "strict_neighbor_count",
        "ultra_neighbor_count",
        "strict_local_concentration",
        "ultra_local_concentration",
        "mass",
        "length_error_bins",
        "h_error_bins",
        "median_coherence",
    )
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in np.flatnonzero(masks["strict"]):
            point_id = int(observables["point"][row])

            def value_or_blank(value: float) -> float | str:
                return float(value) if np.isfinite(value) else ""

            writer.writerow(
                {
                    "row_id": int(row),
                    "point_id": point_id,
                    "component_id": int(observables["component"][row]),
                    "candidate_id": int(observables["candidate"][row]),
                    "patch_id": int(observables["patch"][row]),
                    "regime": REGIME_NAMES[str(regimes[row])],
                    "blocked_count": int(blocked[point_id]),
                    "strict_quality": 1,
                    "ultra_quality": int(masks["ultra"][row]),
                    "ue_x": float(positions[point_id, 0]),
                    "ue_y": float(positions[point_id, 1]),
                    "ue_z": float(positions[point_id, 2]),
                    "reflection_x": float(reflection_points[row, 0]),
                    "reflection_y": float(reflection_points[row, 1]),
                    "reflection_z": float(reflection_points[row, 2]),
                    "strict_anomaly_deg": value_or_blank(
                        strict_result["anomaly_deg"][row]
                    ),
                    "ultra_anomaly_deg": value_or_blank(
                        ultra_result["anomaly_deg"][row]
                    ),
                    "strict_neighbor_count": int(
                        strict_result["neighbor_count"][row]
                    ),
                    "ultra_neighbor_count": int(ultra_result["neighbor_count"][row]),
                    "strict_local_concentration": value_or_blank(
                        strict_result["local_concentration"][row]
                    ),
                    "ultra_local_concentration": value_or_blank(
                        ultra_result["local_concentration"][row]
                    ),
                    "mass": float(observables["mass"][row]),
                    "length_error_bins": float(
                        observables["length_error_bins"][row]
                    ),
                    "h_error_bins": float(observables["h_error_bins"][row]),
                    "median_coherence": float(
                        np.median(observables["coherence"][row])
                    ),
                }
            )


def make_spatial_figure(
    observables: dict[str, np.ndarray],
    positions: np.ndarray,
    reflection_points: np.ndarray,
    cloud: dict[str, np.ndarray],
    limits: tuple[float, float, float, float],
    strict_points: dict[str, np.ndarray],
    ultra_points: dict[str, np.ndarray],
    strict_result: dict[str, np.ndarray],
    ultra_result: dict[str, np.ndarray],
) -> tuple[Path, Path]:
    valid_strict = np.isfinite(strict_result["anomaly_deg"])
    color_max = float(np.quantile(strict_result["anomaly_deg"][valid_strict], 0.95))
    norm = Normalize(vmin=0.0, vmax=color_max)
    cmap = plt.get_cmap("turbo")
    top_rows = top_path_rows(ultra_result["anomaly_deg"])

    figure = plt.figure(figsize=(22, 16), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    strict_axis = figure.add_subplot(grid[0, 0])
    strict_scatter = draw_point_anomaly_map(
        strict_axis,
        cloud,
        limits,
        positions,
        strict_points,
        norm,
        cmap,
        "严格路径：每个训练点的最大局部相位异常",
    )
    colorbar = figure.colorbar(strict_scatter, ax=strict_axis, fraction=0.035, pad=0.02)
    colorbar.set_label("局部相位异常（等效度数）")

    path_axis = figure.add_subplot(grid[0, 1])
    draw_point_cloud_2d(path_axis, cloud, limits)
    draw_candidate_paths_2d(
        path_axis,
        top_rows,
        observables,
        positions,
        reflection_points,
        ultra_result["anomaly_deg"],
        norm,
        cmap,
    )
    path_axis.set_title(
        "极严格路径中异常最高10%的候选反射链路",
        fontsize=13,
        fontweight="bold",
    )
    path_axis.text(
        0.01,
        0.01,
        "虚线仅表示已匹配的一次反射几何候选，不等同于官方真实射线路径",
        transform=path_axis.transAxes,
        fontsize=9,
        color="#374151",
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "none"},
        zorder=10,
    )

    three_dimensional_axis = figure.add_subplot(grid[1, 0], projection="3d")
    draw_candidate_paths_3d(
        three_dimensional_axis,
        cloud,
        top_rows,
        observables,
        positions,
        reflection_points,
        ultra_result["anomaly_deg"],
        norm,
        cmap,
        limits,
    )

    reflection_axis = figure.add_subplot(grid[1, 1])
    draw_point_cloud_2d(reflection_axis, cloud, limits)
    valid_ultra = np.isfinite(ultra_result["anomaly_deg"])
    rows = np.flatnonzero(valid_ultra)
    reflection_scatter = reflection_axis.scatter(
        reflection_points[rows, 0],
        reflection_points[rows, 1],
        c=ultra_result["anomaly_deg"][rows],
        s=34,
        cmap=cmap,
        norm=norm,
        marker="X",
        edgecolor="white",
        linewidth=0.35,
        alpha=0.94,
        zorder=5,
    )
    threshold = float(np.quantile(ultra_result["anomaly_deg"][rows], 0.90))
    top = rows[ultra_result["anomaly_deg"][rows] >= threshold]
    reflection_axis.scatter(
        reflection_points[top, 0],
        reflection_points[top, 1],
        s=92,
        facecolor="none",
        edgecolor="#111827",
        linewidth=1.0,
        zorder=6,
        label="异常最高10%反射点",
    )
    reflection_axis.set_title(
        "极严格路径：异常对应的墙面反射位置",
        fontsize=13,
        fontweight="bold",
    )
    reflection_axis.legend(loc="upper right", fontsize=9, framealpha=0.92)
    colorbar = figure.colorbar(
        reflection_scatter, ax=reflection_axis, fraction=0.035, pad=0.02
    )
    colorbar.set_label("局部相位异常（等效度数）")

    figure.suptitle(
        "官方点云上的路径相位异常空间分布",
        fontsize=20,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.003,
        "相位异常是几何长度校正后，与同墙面7.5米内近邻路径的8端口局部相位不一致程度；它是未解释路径状态，不等同于官方噪声。",
        ha="center",
        fontsize=10,
        color="#374151",
    )
    png = FIG_DIR / "06_点云上的路径相位异常空间分布.png"
    pdf = FIG_DIR / "06_点云上的路径相位异常空间分布.pdf"
    figure.savefig(png, dpi=190, facecolor="white")
    figure.savefig(pdf, dpi=160, facecolor="white")
    plt.close(figure)
    return png, pdf


def make_diagnostic_figure(
    observables: dict[str, np.ndarray],
    strict_result: dict[str, np.ndarray],
    ultra_result: dict[str, np.ndarray],
    regimes: np.ndarray,
    correlations: dict[str, dict[str, float | int]],
) -> tuple[Path, Path]:
    strict_valid = np.isfinite(strict_result["anomaly_deg"])
    ultra_valid = np.isfinite(ultra_result["anomaly_deg"])
    figure, axes = plt.subplots(2, 2, figsize=(18, 13), constrained_layout=True)

    bins = np.linspace(0.0, 115.0, 24)
    axes[0, 0].hist(
        strict_result["anomaly_deg"][strict_valid],
        bins=bins,
        alpha=0.62,
        color="#277da1",
        label=f"严格路径（n={int(np.sum(strict_valid))}）",
    )
    axes[0, 0].hist(
        ultra_result["anomaly_deg"][ultra_valid],
        bins=bins,
        alpha=0.58,
        color="#f8961e",
        label=f"极严格路径（n={int(np.sum(ultra_valid))}）",
    )
    axes[0, 0].set_xlabel("局部相位异常（等效度数）")
    axes[0, 0].set_ylabel("路径数量")
    axes[0, 0].set_title("相位异常总体分布", fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.18)

    regime_order = ("clear", "light", "medium", "heavy")
    box_values = [
        strict_result["anomaly_deg"][strict_valid & (regimes == regime)]
        for regime in regime_order
    ]
    box = axes[0, 1].boxplot(
        box_values,
        tick_labels=[
            f"{REGIME_NAMES[regime]}\n(n={len(values)})"
            for regime, values in zip(regime_order, box_values)
        ],
        patch_artist=True,
        showfliers=False,
    )
    for patch, regime in zip(box["boxes"], regime_order):
        patch.set_facecolor(REGIME_COLORS[regime])
        patch.set_alpha(0.72)
    axes[0, 1].set_ylabel("局部相位异常（等效度数）")
    axes[0, 1].set_title("不同遮挡等级的异常分布", fontweight="bold")
    axes[0, 1].grid(axis="y", alpha=0.18)

    for regime in regime_order:
        mask = strict_valid & (regimes == regime)
        axes[1, 0].scatter(
            observables["length_error_bins"][mask],
            strict_result["anomaly_deg"][mask],
            s=18 + 850 * observables["mass"][mask],
            color=REGIME_COLORS[regime],
            alpha=0.68,
            edgecolor="white",
            linewidth=0.25,
            label=REGIME_NAMES[regime],
        )
    axes[1, 0].set_xlabel("候选路径长度匹配误差（delay bin）")
    axes[1, 0].set_ylabel("局部相位异常（等效度数）")
    axes[1, 0].set_title("相位异常与路径匹配误差", fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(alpha=0.18)

    names = list(correlations)
    values = [float(correlations[name]["spearman"]) for name in names]
    colors = ["#c44536" if value > 0 else "#277da1" for value in values]
    y = np.arange(len(names))
    axes[1, 1].barh(y, values, color=colors, alpha=0.86)
    axes[1, 1].axvline(0.0, color="#111827", linewidth=0.8)
    axes[1, 1].set_yticks(y, names)
    axes[1, 1].set_xlim(-0.62, 0.62)
    axes[1, 1].set_xlabel("与相位异常的 Spearman 相关系数")
    axes[1, 1].set_title("异常更像环境信号还是匹配质量问题", fontweight="bold")
    axes[1, 1].grid(axis="x", alpha=0.18)
    for index, value in enumerate(values):
        axes[1, 1].text(
            value + (0.018 if value >= 0 else -0.018),
            index,
            f"{value:+.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )

    figure.suptitle(
        "路径相位异常的质量与环境诊断",
        fontsize=19,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.003,
        "正相关表示该变量越大、异常越强。异常与长度匹配误差正相关、与遮挡计数弱负相关，提示高异常热点不能直接解释为官方噪声。",
        ha="center",
        fontsize=10,
        color="#374151",
    )
    png = FIG_DIR / "07_路径相位异常与匹配质量诊断.png"
    pdf = FIG_DIR / "07_路径相位异常与匹配质量诊断.pdf"
    figure.savefig(png, dpi=190, facecolor="white")
    figure.savefig(pdf, dpi=160, facecolor="white")
    plt.close(figure)
    return png, pdf


def main() -> None:
    v569.configure_chinese_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    observables = v569.build_observables()
    masks = v569.quality_masks(observables)
    positions = np.load(v569.TRAIN_POS).astype(np.float64)
    geometry = np.load(v569.GEOMETRY)
    blocked = np.asarray(geometry["train_blocked"], dtype=np.int32)
    row_regimes = regime_labels(blocked[observables["point"]])

    candidates = np.load(v569.CANDIDATE_CACHE)
    reflection_points = candidates["reflection_point"][
        observables["point"], observables["candidate"]
    ].astype(np.float64)
    strict_result = compute_local_phase_anomaly(
        observables, masks["strict"], positions[:, :2]
    )
    ultra_result = compute_local_phase_anomaly(
        observables, masks["ultra"], positions[:, :2]
    )
    strict_points = aggregate_points(observables, strict_result["anomaly_deg"], len(positions))
    ultra_points = aggregate_points(observables, ultra_result["anomaly_deg"], len(positions))

    strict_rows = np.flatnonzero(masks["strict"])
    scene_xy = np.vstack(
        [
            positions[observables["point"][strict_rows], :2],
            reflection_points[strict_rows, :2],
            BS_POS[None, :2],
        ]
    )
    limits = (
        float(np.min(scene_xy[:, 0]) - 12.0),
        float(np.max(scene_xy[:, 0]) + 12.0),
        float(np.min(scene_xy[:, 1]) - 12.0),
        float(np.max(scene_xy[:, 1]) + 12.0),
    )
    cloud = sample_point_cloud(limits)
    correlations = correlation_summary(observables, strict_result, blocked)

    spatial_png, spatial_pdf = make_spatial_figure(
        observables,
        positions,
        reflection_points,
        cloud,
        limits,
        strict_points,
        ultra_points,
        strict_result,
        ultra_result,
    )
    diagnostic_png, diagnostic_pdf = make_diagnostic_figure(
        observables,
        strict_result,
        ultra_result,
        row_regimes,
        correlations,
    )
    write_csv(
        observables,
        positions,
        reflection_points,
        masks,
        blocked,
        row_regimes,
        strict_result,
        ultra_result,
    )

    valid_strict = np.isfinite(strict_result["anomaly_deg"])
    valid_ultra = np.isfinite(ultra_result["anomaly_deg"])
    regime_stats = {
        regime: summarize(
            strict_result["anomaly_deg"][valid_strict & (row_regimes == regime)]
        )
        for regime in ("clear", "light", "medium", "heavy")
    }
    top_rows = top_path_rows(ultra_result["anomaly_deg"], limit=20)
    summary = {
        "method": "v569_path_phase_spatial_anomaly_visualization",
        "status": "analysis_visualization_complete",
        "definition": (
            "Geometry-length-corrected path phase disagreement relative to up to eight "
            "same-wall neighbors within 7.5 meters, aggregated over valid polarization-UE "
            "channels with circular chord distance."
        ),
        "parameters": {
            "neighbor_radius_m": NEIGHBOR_RADIUS_M,
            "kernel_scale_m": KERNEL_SCALE_M,
            "maximum_neighbors": MAX_NEIGHBORS,
            "minimum_neighbors": MIN_NEIGHBORS,
            "minimum_valid_channels": MIN_VALID_CHANNELS,
        },
        "strict": {
            "quality_row_count": int(np.sum(masks["strict"])),
            "anomaly_row_count": int(np.sum(valid_strict)),
            "anomaly_degrees": summarize(strict_result["anomaly_deg"]),
            "point_count": int(np.sum(np.isfinite(strict_points["anomaly_deg"]))),
            "regime_degrees": regime_stats,
            "correlations": correlations,
        },
        "ultra": {
            "quality_row_count": int(np.sum(masks["ultra"])),
            "anomaly_row_count": int(np.sum(valid_ultra)),
            "anomaly_degrees": summarize(ultra_result["anomaly_deg"]),
            "point_count": int(np.sum(np.isfinite(ultra_points["anomaly_deg"]))),
        },
        "top_ultra_paths": [
            {
                "row_id": int(row),
                "point_id": int(observables["point"][row]),
                "patch_id": int(observables["patch"][row]),
                "anomaly_deg": float(ultra_result["anomaly_deg"][row]),
                "length_error_bins": float(observables["length_error_bins"][row]),
                "h_error_bins": float(observables["h_error_bins"][row]),
                "mass": float(observables["mass"][row]),
            }
            for row in top_rows
        ],
        "interpretation": (
            "High local phase disagreement is positively associated with path-length match "
            "error and negatively associated with path mass and neighbor support. It is not "
            "concentrated in more heavily blocked rows. The map therefore localizes unresolved "
            "path states and match ambiguity, not uniquely identified official noise."
        ),
        "outputs": {
            "spatial_png": PATHS.relative(spatial_png),
            "spatial_pdf": PATHS.relative(spatial_pdf),
            "diagnostic_png": PATHS.relative(diagnostic_png),
            "diagnostic_pdf": PATHS.relative(diagnostic_pdf),
            "path_table": PATHS.relative(CSV_PATH),
            "summary": PATHS.relative(SUMMARY_PATH),
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    report = json.loads(v569.REPORT.read_text(encoding="utf-8"))
    report["spatial_phase_anomaly_visualization"] = summary
    report["outputs"]["figures"] = [
        PATHS.relative(path) for path in sorted(FIG_DIR.glob("*.png"))
    ]
    report["outputs"]["phase_anomaly_csv"] = PATHS.relative(CSV_PATH)
    report["outputs"]["phase_anomaly_summary"] = PATHS.relative(SUMMARY_PATH)
    v569.REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
