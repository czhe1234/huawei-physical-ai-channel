#!/usr/bin/env python3
"""Audit the released Round2 geometry, positions, and training channels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.cluster import DBSCAN
from sklearn.metrics import average_precision_score, roc_auc_score

from project_paths import PATHS


OUTPUT_DIR = PATHS.analysis / "results" / "round2_initial_data_audit"
FIGURE_DIR = PATHS.analysis / "visualizations" / "复赛初始数据审计"
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a torch device")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-round1-comparison", action="store_true")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def finite_summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0}
    quantiles = np.quantile(array, [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
    return {
        "count": int(len(array)),
        "min": float(quantiles[0]),
        "q01": float(quantiles[1]),
        "q10": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q90": float(quantiles[4]),
        "q99": float(quantiles[5]),
        "max": float(quantiles[6]),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def load_setup() -> dict[str, Any]:
    path = PATHS.round2 / "Round2_Setup.json"
    return json.loads(path.read_text(encoding="utf-8"))


def infer_site_ids(
    train_pos: np.ndarray, test_pos: np.ndarray, bs_positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if len(bs_positions) != 2:
        raise ValueError("The initial Round2 audit currently expects exactly two BS sites")
    train_y = np.sort(train_pos[:, 1])
    gaps = np.diff(train_y)
    gap_index = int(np.argmax(gaps))
    boundary = float((train_y[gap_index] + train_y[gap_index + 1]) / 2.0)
    site_order = np.argsort(bs_positions[:, 1])
    low_site, high_site = int(site_order[0]), int(site_order[1])

    def assign(position: np.ndarray) -> np.ndarray:
        return np.where(position[:, 1] < boundary, low_site, high_site).astype(np.int8)

    train_site = assign(train_pos)
    test_site = assign(test_pos)
    report = {
        "method": "largest empty interval on the Y axis, mapped by BS Y ordering",
        "boundary_y": boundary,
        "empty_interval": [float(train_y[gap_index]), float(train_y[gap_index + 1])],
        "train_counts": np.bincount(train_site, minlength=2).astype(int).tolist(),
        "test_counts": np.bincount(test_site, minlength=2).astype(int).tolist(),
    }
    return train_site, test_site, report


def position_analysis(
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    train_site: np.ndarray,
    test_site: np.ndarray,
    bs_positions: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    nearest_train = np.zeros(len(test_pos), dtype=np.float64)
    eighth_train = np.zeros(len(test_pos), dtype=np.float64)
    test_cluster = np.full(len(test_pos), -1, dtype=np.int16)
    cluster_rows: list[dict[str, Any]] = []
    site_reports: list[dict[str, Any]] = []

    for site in range(len(bs_positions)):
        train_indices = np.flatnonzero(train_site == site)
        test_indices = np.flatnonzero(test_site == site)
        train_xy = train_pos[train_indices, :2]
        test_xy = test_pos[test_indices, :2]
        distances, _ = cKDTree(train_xy).query(test_xy, k=8)
        nearest_train[test_indices] = distances[:, 0]
        eighth_train[test_indices] = distances[:, -1]

        local_clusters = DBSCAN(eps=15.0, min_samples=3).fit_predict(test_xy)
        if np.any(local_clusters < 0):
            raise RuntimeError(f"site {site} has unassigned test points at DBSCAN eps=15m")
        test_cluster[test_indices] = local_clusters + site * 10
        for local_id in sorted(np.unique(local_clusters)):
            local_mask = local_clusters == local_id
            points = test_xy[local_mask]
            cluster_rows.append(
                {
                    "site": site,
                    "cluster": int(local_id),
                    "global_cluster": int(site * 10 + local_id),
                    "count": int(np.sum(local_mask)),
                    "x_mean": float(np.mean(points[:, 0])),
                    "y_mean": float(np.mean(points[:, 1])),
                    "x_min": float(np.min(points[:, 0])),
                    "x_max": float(np.max(points[:, 0])),
                    "y_min": float(np.min(points[:, 1])),
                    "y_max": float(np.max(points[:, 1])),
                }
            )
        train_tree = cKDTree(train_xy)
        train_nn = train_tree.query(train_xy, k=2)[0][:, 1]
        site_reports.append(
            {
                "site": site,
                "bs_position": bs_positions[site].tolist(),
                "train_count": int(len(train_indices)),
                "test_count": int(len(test_indices)),
                "train_bounds": [train_pos[train_indices].min(0).tolist(), train_pos[train_indices].max(0).tolist()],
                "test_bounds": [test_pos[test_indices].min(0).tolist(), test_pos[test_indices].max(0).tolist()],
                "train_nearest_train_xy_m": finite_summary(train_nn),
                "test_nearest_train_xy_m": finite_summary(distances[:, 0]),
                "test_eighth_train_xy_m": finite_summary(distances[:, -1]),
                "test_cluster_count_eps15": int(len(np.unique(local_clusters))),
                "test_cluster_sizes_eps15": [
                    int(np.sum(local_clusters == value))
                    for value in sorted(np.unique(local_clusters))
                ],
            }
        )

    report = {
        "train_shape": list(train_pos.shape),
        "test_shape": list(test_pos.shape),
        "train_bounds": [train_pos.min(0).tolist(), train_pos.max(0).tolist()],
        "test_bounds": [test_pos.min(0).tolist(), test_pos.max(0).tolist()],
        "train_unique_z": np.unique(train_pos[:, 2]).tolist(),
        "test_unique_z": np.unique(test_pos[:, 2]).tolist(),
        "duplicate_train_positions": int(len(train_pos) - len(np.unique(train_pos, axis=0))),
        "duplicate_test_positions": int(len(test_pos) - len(np.unique(test_pos, axis=0))),
        "site_reports": site_reports,
        "test_cluster_definition": "DBSCAN on XY independently per site, eps=15m, min_samples=3",
    }
    return report, nearest_train, eighth_train, test_cluster, cluster_rows


def map_analysis(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    mesh = trimesh.load(path, process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a Trimesh, got {type(mesh).__name__}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    abs_z = np.abs(normals[:, 2])
    report = {
        "format": "ASCII PLY triangle mesh",
        "vertex_count": int(len(vertices)),
        "face_count": int(len(mesh.faces)),
        "bounds": mesh.bounds.tolist(),
        "extents": mesh.extents.tolist(),
        "surface_area": float(mesh.area),
        "is_watertight": bool(mesh.is_watertight),
        "vertex_z": finite_summary(vertices[:, 2]),
        "face_orientation_fraction": {
            "near_vertical_abs_nz_lt_0.25": float(np.mean(abs_z < 0.25)),
            "slanted_abs_nz_0.25_to_0.75": float(np.mean((abs_z >= 0.25) & (abs_z <= 0.75))),
            "near_horizontal_abs_nz_gt_0.75": float(np.mean(abs_z > 0.75)),
        },
    }
    return report, vertices


def normalized_spectrum_metrics(power: torch.Tensor, topk: int) -> tuple[torch.Tensor, ...]:
    total = power.sum(dim=1, keepdim=True)
    normalized = torch.where(total > 0, power / total.clamp_min(TINY), torch.zeros_like(power))
    top1 = normalized.max(dim=1).values
    top = torch.topk(normalized, k=topk, dim=1).values.sum(dim=1)
    entropy = -(normalized * torch.log(normalized.clamp_min(TINY))).sum(dim=1)
    entropy = entropy / math.log(power.shape[1])
    return top1, top, entropy


def scan_round2_channel(
    path: Path, setup: dict[str, Any], device: torch.device, batch_size: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    channel = np.load(path, mmap_mode="r")
    expected = (
        int(setup["P_Train"]),
        int(setup["M"]),
        int(setup["N"]),
        int(setup["S"]),
    )
    if channel.shape != expected or channel.dtype != np.complex64:
        raise ValueError(f"Unexpected channel array: {channel.shape} {channel.dtype}")

    count = len(channel)
    metrics = {
        "power": np.empty(count, dtype=np.float64),
        "frequency_coherence": np.empty(count, dtype=np.float32),
        "frequency_phase_step": np.empty(count, dtype=np.float32),
        "pdp_top1": np.empty(count, dtype=np.float32),
        "pdp_top8": np.empty(count, dtype=np.float32),
        "pdp_entropy": np.empty(count, dtype=np.float32),
        "pas_h_top1": np.empty(count, dtype=np.float32),
        "pas_h_top4": np.empty(count, dtype=np.float32),
        "pas_h_entropy": np.empty(count, dtype=np.float32),
        "pas_v_top1": np.empty(count, dtype=np.float32),
        "pas_v_top4": np.empty(count, dtype=np.float32),
        "pas_v_entropy": np.empty(count, dtype=np.float32),
        "zero_coefficient_count": np.empty(count, dtype=np.int32),
    }
    all_finite = True
    maximum_magnitude = 0.0

    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        host = np.array(channel[start:stop], copy=True)
        value = torch.from_numpy(host).to(device=device)
        magnitude_power = value.real.square() + value.imag.square()
        point_power = magnitude_power.sum(dim=(1, 2, 3), dtype=torch.float64)
        metrics["power"][start:stop] = point_power.cpu().numpy()
        metrics["zero_coefficient_count"][start:stop] = (
            (value == 0).reshape(len(value), -1).sum(dim=1).cpu().numpy()
        )
        all_finite = all_finite and bool(torch.isfinite(value).all().item())
        maximum_magnitude = max(maximum_magnitude, float(torch.abs(value).max().item()))

        adjacent = torch.sum(
            torch.conj(value[..., :-1]) * value[..., 1:], dim=(1, 2, 3)
        )
        left = magnitude_power[..., :-1].sum(dim=(1, 2, 3), dtype=torch.float64)
        right = magnitude_power[..., 1:].sum(dim=(1, 2, 3), dtype=torch.float64)
        denominator = torch.sqrt(left * right)
        coherence = torch.where(
            denominator > 0,
            torch.abs(adjacent).to(torch.float64) / denominator.clamp_min(TINY),
            torch.zeros_like(denominator),
        )
        metrics["frequency_coherence"][start:stop] = coherence.cpu().numpy()
        metrics["frequency_phase_step"][start:stop] = torch.angle(adjacent).cpu().numpy()

        cir = torch.fft.ifft(value, dim=-1)
        pdp = cir.abs().square().sum(dim=(1, 2))
        values = normalized_spectrum_metrics(pdp, topk=8)
        for key, item in zip(("pdp_top1", "pdp_top8", "pdp_entropy"), values):
            metrics[key][start:stop] = item.cpu().numpy()

        array = value.reshape(
            len(value),
            int(setup["M_P"]),
            int(setup["M_H"]),
            int(setup["M_V"]),
            int(setup["N"]),
            int(setup["S"]),
        )
        pas_h = torch.fft.fft(array, dim=2).abs().square().sum(dim=(1, 3, 4, 5))
        values = normalized_spectrum_metrics(pas_h, topk=4)
        for key, item in zip(("pas_h_top1", "pas_h_top4", "pas_h_entropy"), values):
            metrics[key][start:stop] = item.cpu().numpy()

        pas_v = torch.fft.fft(array, dim=3).abs().square().sum(dim=(1, 2, 4, 5))
        values = normalized_spectrum_metrics(pas_v, topk=4)
        for key, item in zip(("pas_v_top1", "pas_v_top4", "pas_v_entropy"), values):
            metrics[key][start:stop] = item.cpu().numpy()

        del host, value, magnitude_power, cir, pdp, array, pas_h, pas_v

    coefficients_per_point = int(np.prod(channel.shape[1:]))
    zero_points = metrics["power"] == 0
    partial_zero_points = (
        (metrics["zero_coefficient_count"] > 0)
        & (metrics["zero_coefficient_count"] < coefficients_per_point)
    )
    nonzero = ~zero_points
    report = {
        "path": PATHS.relative(path),
        "shape": list(channel.shape),
        "dtype": str(channel.dtype),
        "file_bytes": int(path.stat().st_size),
        "all_finite": all_finite,
        "maximum_coefficient_magnitude": maximum_magnitude,
        "coefficients_per_point": coefficients_per_point,
        "exact_zero_point_count": int(np.sum(zero_points)),
        "exact_zero_point_fraction": float(np.mean(zero_points)),
        "partial_zero_point_count": int(np.sum(partial_zero_points)),
        "zero_coefficient_count": int(np.sum(metrics["zero_coefficient_count"], dtype=np.int64)),
        "power": finite_summary(metrics["power"]),
        "nonzero_log10_power": finite_summary(np.log10(metrics["power"][nonzero])),
        "frequency_coherence": finite_summary(metrics["frequency_coherence"]),
        "nonzero_frequency_coherence": finite_summary(metrics["frequency_coherence"][nonzero]),
        "frequency_phase_step_rad": finite_summary(metrics["frequency_phase_step"][nonzero]),
        "pdp_top1": finite_summary(metrics["pdp_top1"][nonzero]),
        "pdp_top8": finite_summary(metrics["pdp_top8"][nonzero]),
        "pdp_entropy": finite_summary(metrics["pdp_entropy"][nonzero]),
        "pas_h_top1": finite_summary(metrics["pas_h_top1"][nonzero]),
        "pas_h_top4": finite_summary(metrics["pas_h_top4"][nonzero]),
        "pas_h_entropy": finite_summary(metrics["pas_h_entropy"][nonzero]),
        "pas_v_top1": finite_summary(metrics["pas_v_top1"][nonzero]),
        "pas_v_top4": finite_summary(metrics["pas_v_top4"][nonzero]),
        "pas_v_entropy": finite_summary(metrics["pas_v_entropy"][nonzero]),
        "array_order_used_for_diagnostic_pas": "[M_P, M_H, M_V, N, S]",
    }
    return report, metrics


def scan_power_only(path: Path, device: torch.device, batch_size: int) -> dict[str, Any]:
    channel = np.load(path, mmap_mode="r")
    power = np.empty(len(channel), dtype=np.float64)
    all_finite = True
    for start in range(0, len(channel), batch_size):
        stop = min(start + batch_size, len(channel))
        value = torch.from_numpy(np.array(channel[start:stop], copy=True)).to(device=device)
        point_power = (value.real.square() + value.imag.square()).sum(
            dim=(1, 2, 3), dtype=torch.float64
        )
        power[start:stop] = point_power.cpu().numpy()
        all_finite = all_finite and bool(torch.isfinite(value).all().item())
    nonzero = power > 0
    return {
        "path": PATHS.relative(path),
        "shape": list(channel.shape),
        "dtype": str(channel.dtype),
        "all_finite": all_finite,
        "exact_zero_point_count": int(np.sum(~nonzero)),
        "exact_zero_point_fraction": float(np.mean(~nonzero)),
        "power": finite_summary(power),
        "nonzero_log10_power": finite_summary(np.log10(power[nonzero])),
    }


def zero_spatial_analysis(
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    train_site: np.ndarray,
    test_site: np.ndarray,
    zero_label: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    train_neighbor_fraction = np.zeros(len(train_pos), dtype=np.float32)
    test_neighbor_fraction = np.zeros(len(test_pos), dtype=np.float32)
    test_neighbor_idw = np.zeros(len(test_pos), dtype=np.float32)
    reports = []
    for site in sorted(np.unique(train_site)):
        train_indices = np.flatnonzero(train_site == site)
        test_indices = np.flatnonzero(test_site == site)
        train_xy = train_pos[train_indices, :2]
        tree = cKDTree(train_xy)
        train_dist, train_neighbor = tree.query(train_xy, k=9)
        local_label = zero_label[train_indices].astype(np.float64)
        fraction = np.mean(local_label[train_neighbor[:, 1:]], axis=1)
        train_neighbor_fraction[train_indices] = fraction

        test_dist, test_neighbor = tree.query(test_pos[test_indices, :2], k=8)
        test_fraction = np.mean(local_label[test_neighbor], axis=1)
        weight = 1.0 / np.maximum(test_dist, 0.5) ** 2
        test_idw = np.sum(weight * local_label[test_neighbor], axis=1) / np.sum(weight, axis=1)
        test_neighbor_fraction[test_indices] = test_fraction
        test_neighbor_idw[test_indices] = test_idw
        reports.append(
            {
                "site": int(site),
                "zero_count": int(np.sum(local_label)),
                "zero_fraction": float(np.mean(local_label)),
                "nearest_8_zero_fraction_auc": float(roc_auc_score(local_label, fraction)),
                "nearest_8_zero_fraction_ap": float(average_precision_score(local_label, fraction)),
                "nearest_neighbor_same_label_fraction": float(
                    np.mean(local_label[train_neighbor[:, 1]] == local_label)
                ),
                "test_zero_risk_unweighted": finite_summary(test_fraction),
                "test_zero_risk_idw": finite_summary(test_idw),
                "test_idw_risk_ge_0_5": int(np.sum(test_idw >= 0.5)),
                "test_idw_risk_ge_0_8": int(np.sum(test_idw >= 0.8)),
            }
        )
    return {"site_reports": reports}, train_neighbor_fraction, test_neighbor_idw


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def configure_chinese_font() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )


def plot_locations(
    vertices: np.ndarray,
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    train_site: np.ndarray,
    test_site: np.ndarray,
    test_cluster: np.ndarray,
    bs_positions: np.ndarray,
) -> None:
    configure_chinese_font()
    figure, axis = plt.subplots(figsize=(11, 9))
    position_min = np.minimum(train_pos[:, :2].min(0), test_pos[:, :2].min(0)) - 30
    position_max = np.maximum(train_pos[:, :2].max(0), test_pos[:, :2].max(0)) + 30
    map_mask = np.all(
        (vertices[:, :2] >= position_min[None, :])
        & (vertices[:, :2] <= position_max[None, :]),
        axis=1,
    )
    axis.scatter(vertices[map_mask, 0], vertices[map_mask, 1], s=1.0, color="#8A8A8A", alpha=0.22)
    site_colors = ("#2C6EAA", "#D97706")
    for site, color in enumerate(site_colors):
        mask = train_site == site
        axis.scatter(
            train_pos[mask, 0],
            train_pos[mask, 1],
            s=6,
            color=color,
            alpha=0.28,
            linewidths=0,
            label=f"基站{site}训练点",
        )
    cluster_colors = plt.get_cmap("tab10")
    for cluster in sorted(np.unique(test_cluster)):
        mask = test_cluster == cluster
        site = int(np.unique(test_site[mask])[0])
        axis.scatter(
            test_pos[mask, 0],
            test_pos[mask, 1],
            s=24,
            facecolor=cluster_colors(cluster % 10),
            edgecolor="white",
            linewidth=0.35,
            alpha=0.92,
        )
        center = test_pos[mask, :2].mean(0)
        axis.text(center[0], center[1], f"{site}-{cluster % 10}", fontsize=8, ha="center", va="center")
    for site, position in enumerate(bs_positions):
        axis.scatter(position[0], position[1], marker="*", s=260, color="#F2C94C", edgecolor="#111111", linewidth=1.0, zorder=8)
        axis.text(position[0] + 4, position[1] + 4, f"基站{site}", fontsize=10, weight="bold")
    axis.set_xlim(position_min[0], position_max[0])
    axis.set_ylim(position_min[1], position_max[1])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("X 坐标（米）")
    axis.set_ylabel("Y 坐标（米）")
    axis.set_title("复赛地图、两基站训练点与十个测试空间簇")
    axis.grid(alpha=0.15)
    axis.legend(loc="upper left", frameon=True)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "01_复赛地图与训练测试点分布.png", dpi=220)
    plt.close(figure)


def plot_power_map(
    train_pos: np.ndarray, train_site: np.ndarray, power: np.ndarray, zero: np.ndarray
) -> None:
    configure_chinese_font()
    nonzero_log = np.log10(power[~zero])
    lower, upper = np.quantile(nonzero_log, [0.02, 0.98])
    normalization = Normalize(vmin=lower, vmax=upper)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.4), constrained_layout=True)
    scatter = None
    for site, axis in enumerate(axes):
        mask = (train_site == site) & ~zero
        scatter = axis.scatter(
            train_pos[mask, 0],
            train_pos[mask, 1],
            c=np.log10(power[mask]),
            cmap="viridis",
            norm=normalization,
            s=10,
            linewidths=0,
        )
        zero_mask = (train_site == site) & zero
        axis.scatter(
            train_pos[zero_mask, 0],
            train_pos[zero_mask, 1],
            marker="x",
            s=28,
            color="#D62728",
            linewidth=1.0,
            label=f"全零信道：{int(np.sum(zero_mask))}点",
        )
        axis.set_title(f"基站{site}训练信道总功率")
        axis.set_xlabel("X 坐标（米）")
        axis.set_ylabel("Y 坐标（米）")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.15)
        axis.legend(loc="best")
    assert scatter is not None
    figure.colorbar(scatter, ax=axes, label="log10（逐点信道总功率）", shrink=0.88)
    figure.savefig(FIGURE_DIR / "02_训练信道总功率与全零点空间分布.png", dpi=220)
    plt.close(figure)


def plot_distributions(
    power: np.ndarray,
    zero: np.ndarray,
    train_site: np.ndarray,
    nearest_train: np.ndarray,
    test_site: np.ndarray,
    test_zero_risk: np.ndarray,
) -> None:
    configure_chinese_font()
    colors = ("#2C6EAA", "#D97706")
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for site, color in enumerate(colors):
        mask = (train_site == site) & ~zero
        axes[0].hist(np.log10(power[mask]), bins=45, density=True, alpha=0.52, color=color, label=f"基站{site}")
        test_mask = test_site == site
        axes[1].hist(nearest_train[test_mask], bins=30, alpha=0.52, color=color, label=f"基站{site}")
        axes[2].hist(test_zero_risk[test_mask], bins=np.linspace(0, 1, 21), alpha=0.52, color=color, label=f"基站{site}")
    axes[0].set_title("非零训练点总功率分布")
    axes[0].set_xlabel("log10（总功率）")
    axes[0].set_ylabel("密度")
    axes[1].set_title("测试点到同基站训练集的最近距离")
    axes[1].set_xlabel("距离（米）")
    axes[1].set_ylabel("点数")
    axes[2].set_title("测试点局部全零信道风险")
    axes[2].set_xlabel("8近邻反距离加权全零比例")
    axes[2].set_ylabel("点数")
    for axis in axes:
        axis.grid(alpha=0.18)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "03_功率距离与全零风险分布.png", dpi=220)
    plt.close(figure)


def plot_spectral_metrics(metrics: dict[str, np.ndarray], zero: np.ndarray, train_site: np.ndarray) -> None:
    configure_chinese_font()
    names = [
        ("pdp_top8", "PDP最强8个时延格能量占比"),
        ("pas_h_top4", "水平PAS最强4个角度格能量占比"),
        ("pdp_entropy", "PDP归一化熵"),
        ("pas_h_entropy", "水平PAS归一化熵"),
    ]
    colors = ("#2C6EAA", "#D97706")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (key, title) in zip(axes.flat, names):
        for site, color in enumerate(colors):
            mask = (train_site == site) & ~zero
            axis.hist(metrics[key][mask], bins=40, density=True, alpha=0.52, color=color, label=f"基站{site}")
        axis.set_title(title)
        axis.set_xlabel("数值")
        axis.set_ylabel("密度")
        axis.grid(alpha=0.18)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "04_角度时延谱稀疏性分布.png", dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    setup = load_setup()
    train_pos = np.load(PATHS.round2 / "Round2_Train_Pos.npy")
    test_pos = np.load(PATHS.round2 / "Round2_Test_Pos.npy")
    bs_positions = np.asarray(setup["X"], dtype=np.float64)
    train_site, test_site, site_inference = infer_site_ids(train_pos, test_pos, bs_positions)
    position_report, nearest_train, eighth_train, test_cluster, cluster_rows = position_analysis(
        train_pos, test_pos, train_site, test_site, bs_positions
    )
    map_report, map_vertices = map_analysis(PATHS.round2 / "Round2_Map.ply")
    channel_report, metrics = scan_round2_channel(
        PATHS.round2 / "Round2_Train_Channel.npy", setup, device, args.batch_size
    )
    zero = metrics["power"] == 0
    zero_report, zero_neighbor_fraction, test_zero_risk = zero_spatial_analysis(
        train_pos, test_pos, train_site, test_site, zero
    )

    site_channel_reports = []
    for site in range(len(bs_positions)):
        mask = train_site == site
        nonzero = mask & ~zero
        site_channel_reports.append(
            {
                "site": site,
                "point_count": int(np.sum(mask)),
                "exact_zero_count": int(np.sum(mask & zero)),
                "exact_zero_fraction": float(np.mean(zero[mask])),
                "power": finite_summary(metrics["power"][mask]),
                "nonzero_log10_power": finite_summary(np.log10(metrics["power"][nonzero])),
                "nonzero_frequency_coherence": finite_summary(metrics["frequency_coherence"][nonzero]),
                "nonzero_pdp_top8": finite_summary(metrics["pdp_top8"][nonzero]),
                "nonzero_pas_h_top4": finite_summary(metrics["pas_h_top4"][nonzero]),
            }
        )
    channel_report["site_reports"] = site_channel_reports
    nonzero = ~zero
    log_power = np.log10(metrics["power"][nonzero])
    coherence = metrics["frequency_coherence"][nonzero].astype(np.float64)
    correlation = spearmanr(log_power, coherence)
    decile_edges = np.quantile(log_power, np.linspace(0.0, 1.0, 11))
    coherence_deciles = []
    for index in range(10):
        upper_operator = np.less_equal if index == 9 else np.less
        mask = (log_power >= decile_edges[index]) & upper_operator(
            log_power, decile_edges[index + 1]
        )
        coherence_deciles.append(
            {
                "decile": index,
                "log10_power_min": float(decile_edges[index]),
                "log10_power_max": float(decile_edges[index + 1]),
                "count": int(np.sum(mask)),
                "frequency_coherence_median": float(np.median(coherence[mask])),
                "one_minus_coherence_median": float(np.median(1.0 - coherence[mask])),
            }
        )
    channel_report["power_frequency_coherence_relation"] = {
        "interpretation": "Adjacent-subcarrier coherence is only a noise proxy; multipath dispersion is a confounder.",
        "spearman_log10_power_vs_coherence": float(correlation.statistic),
        "p_value": float(correlation.pvalue),
        "deciles": coherence_deciles,
    }

    round1_comparison = None
    round1_path = PATHS.round1 / "Round1_Train_Channel.npy"
    if not args.skip_round1_comparison and round1_path.is_file():
        round1_comparison = scan_power_only(round1_path, device, args.batch_size)

    report = {
        "analysis": "round2_initial_data_audit",
        "status": "analysis_only",
        "device": str(device),
        "setup": setup,
        "site_inference": site_inference,
        "positions": position_report,
        "map": map_report,
        "channel": channel_report,
        "zero_channel_spatial_analysis": zero_report,
        "round1_power_comparison": round1_comparison,
        "key_facts": {
            "two_disjoint_bs_user_regions": True,
            "test_is_ten_spatial_clusters_not_iid_random_points": True,
            "test_cluster_count": int(len(np.unique(test_cluster))),
            "round2_exact_zero_points": int(np.sum(zero)),
            "round2_exact_zero_fraction": float(np.mean(zero)),
            "all_partial_zero_points_absent": bool(channel_report["partial_zero_point_count"] == 0),
        },
    }
    report_path = OUTPUT_DIR / "round2_initial_data_audit.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    train_rows = []
    for index in range(len(train_pos)):
        train_rows.append(
            {
                "index": index,
                "site": int(train_site[index]),
                "x": float(train_pos[index, 0]),
                "y": float(train_pos[index, 1]),
                "z": float(train_pos[index, 2]),
                "power": float(metrics["power"][index]),
                "log10_power": float(np.log10(metrics["power"][index])) if not zero[index] else "",
                "exact_zero": int(zero[index]),
                "zero_neighbor_fraction_k8": float(zero_neighbor_fraction[index]),
                "frequency_coherence": float(metrics["frequency_coherence"][index]),
                "frequency_phase_step": float(metrics["frequency_phase_step"][index]),
                "pdp_top1": float(metrics["pdp_top1"][index]),
                "pdp_top8": float(metrics["pdp_top8"][index]),
                "pdp_entropy": float(metrics["pdp_entropy"][index]),
                "pas_h_top1": float(metrics["pas_h_top1"][index]),
                "pas_h_top4": float(metrics["pas_h_top4"][index]),
                "pas_h_entropy": float(metrics["pas_h_entropy"][index]),
            }
        )
    train_fields = list(train_rows[0])
    write_csv(OUTPUT_DIR / "round2_train_point_summary.csv", train_fields, train_rows)

    test_rows = []
    for index in range(len(test_pos)):
        test_rows.append(
            {
                "index": index,
                "site": int(test_site[index]),
                "cluster": int(test_cluster[index]),
                "x": float(test_pos[index, 0]),
                "y": float(test_pos[index, 1]),
                "z": float(test_pos[index, 2]),
                "nearest_train_xy_m": float(nearest_train[index]),
                "eighth_train_xy_m": float(eighth_train[index]),
                "zero_risk_idw_k8": float(test_zero_risk[index]),
            }
        )
    write_csv(OUTPUT_DIR / "round2_test_point_summary.csv", list(test_rows[0]), test_rows)
    write_csv(OUTPUT_DIR / "round2_test_clusters.csv", list(cluster_rows[0]), cluster_rows)

    plot_locations(map_vertices, train_pos, test_pos, train_site, test_site, test_cluster, bs_positions)
    plot_power_map(train_pos, train_site, metrics["power"], zero)
    plot_distributions(metrics["power"], zero, train_site, nearest_train, test_site, test_zero_risk)
    plot_spectral_metrics(metrics, zero, train_site)

    print(json.dumps({
        "report": PATHS.relative(report_path),
        "figures": PATHS.relative(FIGURE_DIR),
        "device": str(device),
        "zero_points": int(np.sum(zero)),
        "test_clusters": int(len(np.unique(test_cluster))),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
