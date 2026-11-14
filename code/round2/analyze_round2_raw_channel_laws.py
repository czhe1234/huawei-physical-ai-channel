#!/usr/bin/env python3
"""Analyze spatial, path, phase, and MIMO laws in the official Round 2 data."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from matplotlib.colors import Normalize
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp, spearmanr

from project_paths import PATHS
from round2.data import load_positions
from round2.run_path_component_transport import Config, circular_delta, match_components


RESULT_DIR = PATHS.analysis / "results" / "round2_raw_channel_law_audit"
FIGURE_DIR = PATHS.analysis / "visualizations" / "复赛原始信道传播规律"
CACHE_DIR = PATHS.artifacts / "round2" / "raw_channel_law_audit"

TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
SPECTRAL_DESCRIPTORS = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_train_spectral_descriptors.npz"
)
ANGLE_DELAY_CACHE = (
    PATHS.artifacts / "round2" / "pre_sionna" / "round2_train_ad_support_f16.npy"
)
COMPONENT_CACHE = (
    PATHS.artifacts
    / "round2"
    / "v4_path_component_transport"
    / "train_components_k24_os2_ref2.npz"
)
GEOMETRY_CACHE = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_geometry_visibility_features.npz"
)
GEODESIC_CACHE = (
    PATHS.artifacts
    / "round2"
    / "regime_lifecycle_support"
    / "geodesic_corridor_features.npz"
)
DERIVED_CACHE = CACHE_DIR / "原始信道点级派生量.npz"
PAIR_CACHE = CACHE_DIR / "原始信道空间对派生量.npz"

MP = 2
MH = 16
MV = 8
M_TX = MP * MH * MV
N_UE = 4
S_SC = 192
PATH_COUNT = 24
TINY = 1e-30
RANDOM_SEED = 20_260_807
DISTANCE_EDGES = np.asarray(
    [0.0, 1.0, 2.0, 3.0, 4.5, 6.5, 9.0, 13.0, 20.0, 35.0, 60.0, 100.0, 170.0, np.inf],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PointData:
    positions: np.ndarray
    site: np.ndarray
    bs_distance: np.ndarray
    power: np.ndarray
    valid: np.ndarray
    global_pas: np.ndarray
    global_pdp: np.ndarray
    main_h: np.ndarray
    main_v: np.ndarray
    main_delay: np.ndarray
    main_h_angle: np.ndarray
    main_v_angle: np.ndarray
    discrete_coordinates: np.ndarray
    discrete_energy: np.ndarray
    component_coordinates: np.ndarray
    component_coefficients: np.ndarray
    component_energy: np.ndarray
    component_share: np.ndarray
    geometry: np.ndarray
    geometry_names: tuple[str, ...]
    geodesic: np.ndarray
    geodesic_names: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--path-pair-limit", type=int, default=24000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.titleweight": "semibold",
            "axes.grid": True,
            "grid.alpha": 0.18,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def finite_summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0}
    quantiles = np.quantile(array, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {
        "count": int(len(array)),
        "min": float(quantiles[0]),
        "q10": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q90": float(quantiles[5]),
        "max": float(quantiles[6]),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def scalar(value: Any) -> float | int | bool | str:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: scalar(row.get(key, "")) for key in fields})


def cosine_pairwise(values: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    first = np.asarray(values[left], dtype=np.float64)
    second = np.asarray(values[right], dtype=np.float64)
    numerator = np.sum(first * second, axis=1)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > TINY,
    )


def signed_fft_bin(index: np.ndarray, period: int) -> np.ndarray:
    return (np.asarray(index, dtype=np.float64) + period / 2.0) % period - period / 2.0


def equivalent_array_angle(index: np.ndarray, period: int) -> np.ndarray:
    spatial_frequency = 2.0 * signed_fft_bin(index, period) / period
    return np.rad2deg(np.arcsin(np.clip(spatial_frequency, -1.0, 1.0)))


def distance_bin_indices(distance: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(distance), DISTANCE_EDGES[1:-1], right=True)


def distance_bin_label(index: int) -> str:
    lower = DISTANCE_EDGES[index]
    upper = DISTANCE_EDGES[index + 1]
    return f"{lower:g}–{upper:g}米" if np.isfinite(upper) else f">{lower:g}米"


def binned_curve(
    distance: np.ndarray,
    values: np.ndarray,
    *,
    minimum_count: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bins = distance_bin_indices(distance)
    centers: list[float] = []
    medians: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    counts: list[int] = []
    for bin_index in range(len(DISTANCE_EDGES) - 1):
        mask = (bins == bin_index) & np.isfinite(values)
        if np.sum(mask) < minimum_count:
            continue
        finite_distance = np.asarray(distance[mask], dtype=np.float64)
        finite_values = np.asarray(values[mask], dtype=np.float64)
        centers.append(float(np.median(finite_distance)))
        medians.append(float(np.median(finite_values)))
        lower.append(float(np.quantile(finite_values, 0.25)))
        upper.append(float(np.quantile(finite_values, 0.75)))
        counts.append(int(len(finite_values)))
    return (
        np.asarray(centers),
        np.asarray(medians),
        np.asarray(lower),
        np.asarray(upper),
        np.asarray(counts),
    )


def plot_binned(
    axis: plt.Axes,
    distance: np.ndarray,
    values: np.ndarray,
    *,
    label: str,
    color: str,
    interval: bool = True,
    linestyle: str = "-",
) -> None:
    x, median, lower, upper, _ = binned_curve(distance, values)
    axis.plot(x, median, marker="o", ms=4.0, lw=2.0, color=color, label=label, linestyle=linestyle)
    if interval:
        axis.fill_between(x, lower, upper, color=color, alpha=0.13, linewidth=0)


def map_background(axis: plt.Axes, vertices: np.ndarray, positions: np.ndarray) -> None:
    lower = positions[:, :2].min(axis=0) - 20.0
    upper = positions[:, :2].max(axis=0) + 20.0
    mask = np.all((vertices[:, :2] >= lower) & (vertices[:, :2] <= upper), axis=1)
    axis.scatter(
        vertices[mask, 0],
        vertices[mask, 1],
        s=0.7,
        c="#777777",
        alpha=0.14,
        linewidths=0,
        rasterized=True,
    )
    axis.set_xlim(lower[0], upper[0])
    axis.set_ylim(lower[1], upper[1])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("X坐标（米）")
    axis.set_ylabel("Y坐标（米）")


def add_base_stations(
    axis: plt.Axes,
    bs_positions: np.ndarray,
    site_values: Iterable[int] | None = None,
) -> None:
    selected = range(len(bs_positions)) if site_values is None else site_values
    for site in selected:
        position = bs_positions[site]
        axis.scatter(
            position[0],
            position[1],
            marker="*",
            s=180,
            color="#F2C94C",
            edgecolor="#222222",
            linewidth=0.8,
            zorder=8,
        )
        axis.text(position[0] + 3.0, position[1] + 3.0, f"基站{site}", fontsize=9, weight="bold")


def extract_discrete_peaks(
    cache: np.ndarray,
    valid: np.ndarray,
    topk: int = PATH_COUNT,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(cache)
    coordinates = np.zeros((count, topk, 3), dtype=np.int16)
    energy = np.zeros((count, topk), dtype=np.float32)
    cumulative = np.zeros((count, topk), dtype=np.float32)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        values = np.asarray(cache[start:end], dtype=np.float32).reshape(end - start, -1)
        indices = np.argpartition(values, -topk, axis=1)[:, -topk:]
        selected = np.take_along_axis(values, indices, axis=1)
        order = np.argsort(selected, axis=1)[:, ::-1]
        indices = np.take_along_axis(indices, order, axis=1)
        selected = np.take_along_axis(selected, order, axis=1)
        delay = indices % S_SC
        partial = indices // S_SC
        vertical = partial % MV
        horizontal = partial // MV
        coordinates[start:end] = np.stack([horizontal, vertical, delay], axis=2).astype(np.int16)
        energy[start:end] = selected
        total = np.sum(values, axis=1, keepdims=True)
        normalized = np.divide(selected, total, out=np.zeros_like(selected), where=total > TINY)
        cumulative[start:end] = np.cumsum(normalized, axis=1)
    coordinates[~valid] = 0
    energy[~valid] = 0.0
    cumulative[~valid] = 0.0
    return coordinates, energy, cumulative


def compute_mimo_point_metrics(
    channel: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(channel)
    rank_share = np.zeros((count, N_UE), dtype=np.float32)
    port_share = np.zeros((count, MP * N_UE), dtype=np.float32)
    phase_delay = np.zeros(count, dtype=np.float32)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        value = torch.as_tensor(
            np.array(channel[start:end], dtype=np.complex64, copy=True),
            device=device,
        )
        energy = value.abs().square().sum(dim=(1, 2, 3))
        adjacent = torch.sum(torch.conj(value[..., :-1]) * value[..., 1:], dim=(1, 2, 3))
        delay = torch.remainder(-torch.angle(adjacent) * S_SC / (2.0 * math.pi), S_SC)
        phase_delay[start:end] = delay.cpu().numpy().astype(np.float32)

        by_frequency = value.permute(0, 3, 1, 2)
        gram = torch.einsum("bsmi,bsmj->bsij", torch.conj(by_frequency), by_frequency)
        eigenvalues = torch.linalg.eigvalsh(gram).real.flip(dims=(-1,)).clamp_min(0.0)
        rank_energy = eigenvalues.sum(dim=1)
        rank_share[start:end] = torch.where(
            energy[:, None] > 0,
            rank_energy / energy[:, None].clamp_min(1e-30),
            torch.zeros_like(rank_energy),
        ).cpu().numpy().astype(np.float32)

        shaped = value.reshape(-1, MP, MH, MV, N_UE, S_SC)
        port_energy = shaped.abs().square().sum(dim=(2, 3, 5)).reshape(-1, MP * N_UE)
        port_share[start:end] = torch.where(
            energy[:, None] > 0,
            port_energy / energy[:, None].clamp_min(1e-30),
            torch.zeros_like(port_energy),
        ).cpu().numpy().astype(np.float32)
        print(f"原始H点级相位与MIMO统计 {end}/{count}", flush=True)
    return rank_share, port_share, phase_delay


def build_point_data(device: torch.device, batch_size: int, force: bool) -> tuple[PointData, dict[str, np.ndarray]]:
    setup, positions, _, site, _ = load_positions()
    descriptors = np.load(SPECTRAL_DESCRIPTORS, allow_pickle=False)
    global_pas = descriptors["global_pas"].astype(np.float32)
    global_pdp = descriptors["global_pdp"].astype(np.float32)
    power = descriptors["total_power"].astype(np.float64)
    valid = power > 0.0
    serving = setup.bs_positions[site.astype(np.int64)]
    bs_distance = np.linalg.norm(positions - serving, axis=1)

    pas_array = global_pas.reshape(-1, MP, MH, MV)
    horizontal_pas = pas_array.sum(axis=(1, 3))
    vertical_pas = pas_array.sum(axis=(1, 2))
    main_h = np.argmax(horizontal_pas, axis=1).astype(np.int16)
    main_v = np.argmax(vertical_pas, axis=1).astype(np.int16)
    main_delay = np.argmax(global_pdp, axis=1).astype(np.int16)
    main_h_angle = equivalent_array_angle(main_h, MH).astype(np.float32)
    main_v_angle = equivalent_array_angle(main_v, MV).astype(np.float32)

    angle_delay = np.load(ANGLE_DELAY_CACHE, mmap_mode="r")
    discrete_coordinates, discrete_energy, discrete_cumulative = extract_discrete_peaks(
        angle_delay, valid
    )

    component = np.load(COMPONENT_CACHE, allow_pickle=False)
    component_coordinates = component["coordinates"].astype(np.float32)
    component_coefficients = component["coefficients"].astype(np.complex64)
    component_energy = np.sum(np.abs(component_coefficients) ** 2, axis=2, dtype=np.float64)
    component_total = component_energy.sum(axis=1, keepdims=True)
    component_share = np.divide(
        component_energy,
        component_total,
        out=np.zeros_like(component_energy),
        where=component_total > TINY,
    ).astype(np.float32)

    geometry_file = np.load(GEOMETRY_CACHE, allow_pickle=False)
    geometry = geometry_file["train"].astype(np.float32)
    geometry_names = tuple(geometry_file["names"].astype(str).tolist())
    geodesic_file = np.load(GEODESIC_CACHE, allow_pickle=False)
    geodesic = geodesic_file["train_features"].astype(np.float32)
    geodesic_names = tuple(geodesic_file["feature_names"].astype(str).tolist())

    if DERIVED_CACHE.is_file() and not force:
        derived = np.load(DERIVED_CACHE, allow_pickle=False)
        rank_share = derived["rank_share"].astype(np.float32)
        port_share = derived["port_share"].astype(np.float32)
        phase_delay = derived["phase_delay"].astype(np.float32)
    else:
        channel = np.load(TRAIN_CHANNEL, mmap_mode="r")
        rank_share, port_share, phase_delay = compute_mimo_point_metrics(
            channel, device, batch_size
        )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            DERIVED_CACHE,
            rank_share=rank_share,
            port_share=port_share,
            phase_delay=phase_delay,
            discrete_coordinates=discrete_coordinates,
            discrete_energy=discrete_energy,
            discrete_cumulative=discrete_cumulative,
        )

    point = PointData(
        positions=positions,
        site=site,
        bs_distance=bs_distance,
        power=power,
        valid=valid,
        global_pas=global_pas,
        global_pdp=global_pdp,
        main_h=main_h,
        main_v=main_v,
        main_delay=main_delay,
        main_h_angle=main_h_angle,
        main_v_angle=main_v_angle,
        discrete_coordinates=discrete_coordinates,
        discrete_energy=discrete_energy,
        component_coordinates=component_coordinates,
        component_coefficients=component_coefficients,
        component_energy=component_energy,
        component_share=component_share,
        geometry=geometry,
        geometry_names=geometry_names,
        geodesic=geodesic,
        geodesic_names=geodesic_names,
    )
    derived_values = {
        "horizontal_pas": horizontal_pas.astype(np.float32),
        "vertical_pas": vertical_pas.astype(np.float32),
        "discrete_cumulative": discrete_cumulative,
        "rank_share": rank_share,
        "port_share": port_share,
        "phase_delay": phase_delay,
    }
    return point, derived_values


def build_stratified_pairs(point: PointData, limit_per_bin: int = 2600) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    candidates: set[tuple[int, int]] = set()
    for site_value in np.unique(point.site):
        ids = np.flatnonzero((point.site == site_value) & point.valid)
        xy = point.positions[ids, :2]
        _, neighbor = cKDTree(xy).query(xy, k=min(25, len(ids)), workers=-1)
        for row, source in enumerate(ids):
            for local in np.atleast_1d(neighbor[row])[1:]:
                target = int(ids[int(local)])
                candidates.add((min(int(source), target), max(int(source), target)))
        random_left = rng.choice(ids, size=450_000, replace=True)
        random_right = rng.choice(ids, size=450_000, replace=True)
        for left, right in zip(random_left, random_right):
            if left != right:
                candidates.add((min(int(left), int(right)), max(int(left), int(right))))

    pair_array = np.asarray(sorted(candidates), dtype=np.int32)
    distance = np.linalg.norm(
        point.positions[pair_array[:, 0], :2] - point.positions[pair_array[:, 1], :2],
        axis=1,
    )
    bins = distance_bin_indices(distance)
    selected: list[np.ndarray] = []
    for bin_index in range(len(DISTANCE_EDGES) - 1):
        local = np.flatnonzero(bins == bin_index)
        if len(local) > limit_per_bin:
            local = np.sort(rng.choice(local, size=limit_per_bin, replace=False))
        selected.append(local)
    chosen = np.concatenate(selected)
    order = np.argsort(distance[chosen], kind="stable")
    chosen = chosen[order]
    return pair_array[chosen, 0], pair_array[chosen, 1], distance[chosen]


def match_discrete_paths(
    left_coordinates: np.ndarray,
    left_energy: np.ndarray,
    right_coordinates: np.ndarray,
    right_energy: np.ndarray,
) -> tuple[float, float, float, float]:
    delta_h = circular_delta(
        right_coordinates[None, :, 0], left_coordinates[:, None, 0], MH
    )
    delta_v = circular_delta(
        right_coordinates[None, :, 1], left_coordinates[:, None, 1], MV
    )
    delta_d = circular_delta(
        right_coordinates[None, :, 2], left_coordinates[:, None, 2], S_SC
    )
    cost = (delta_h / 1.5) ** 2 + (delta_v / 1.0) ** 2 + (delta_d / 2.0) ** 2
    rows, columns = linear_sum_assignment(cost)
    accepted = (
        (np.abs(delta_h[rows, columns]) <= 2.0)
        & (np.abs(delta_v[rows, columns]) <= 1.5)
        & (np.abs(delta_d[rows, columns]) <= 4.0)
        & (cost[rows, columns] <= 8.0)
    )
    rows = rows[accepted]
    columns = columns[accepted]
    left_share = left_energy / max(float(np.sum(left_energy)), TINY)
    right_share = right_energy / max(float(np.sum(right_energy)), TINY)
    common = float(np.sum(np.minimum(left_share[rows], right_share[columns])))
    if not len(rows):
        return common, math.nan, math.nan, math.nan
    weight = np.minimum(left_share[rows], right_share[columns])
    weight /= max(float(np.sum(weight)), TINY)
    return (
        common,
        float(np.sum(weight * np.abs(delta_h[rows, columns]))),
        float(np.sum(weight * np.abs(delta_v[rows, columns]))),
        float(np.sum(weight * np.abs(delta_d[rows, columns]))),
    )


def continuous_match_metrics(
    point: PointData,
    left: int,
    right: int,
    config: Config,
    *,
    shuffled_identity: bool = False,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    left_coord = point.component_coordinates[left, :PATH_COUNT]
    left_coeff = point.component_coefficients[left, :PATH_COUNT]
    right_coord = point.component_coordinates[right, :PATH_COUNT]
    right_coeff = point.component_coefficients[right, :PATH_COUNT]
    if shuffled_identity:
        if rng is None:
            raise ValueError("rng is required for identity shuffle")
        right_coeff = right_coeff[rng.permutation(PATH_COUNT)]
    matched, _, coherence = match_components(
        left_coord,
        left_coeff,
        right_coord,
        right_coeff,
        config,
    )
    accepted = matched >= 0
    left_share = point.component_share[left, :PATH_COUNT].astype(np.float64)
    right_raw_energy = np.sum(np.abs(right_coeff) ** 2, axis=1, dtype=np.float64)
    right_share = right_raw_energy / max(float(np.sum(right_raw_energy)), TINY)
    if not np.any(accepted):
        return {
            "common": 0.0,
            "coverage": 0.0,
            "h_delta": math.nan,
            "v_delta": math.nan,
            "d_delta": math.nan,
            "coherence": math.nan,
            "log_energy_delta": math.nan,
        }
    right_indices = matched[accepted].astype(np.int64)
    left_indices = np.flatnonzero(accepted)
    common_weight = np.minimum(left_share[left_indices], right_share[right_indices])
    common = float(np.sum(common_weight))
    coverage = 0.5 * (
        float(np.sum(left_share[left_indices])) + float(np.sum(right_share[right_indices]))
    )
    normalized_weight = common_weight / max(float(np.sum(common_weight)), TINY)
    delta_h = np.abs(
        circular_delta(right_coord[right_indices, 0], left_coord[left_indices, 0], MH)
    )
    delta_v = np.abs(
        circular_delta(right_coord[right_indices, 1], left_coord[left_indices, 1], MV)
    )
    delta_d = np.abs(
        circular_delta(right_coord[right_indices, 2], left_coord[left_indices, 2], S_SC)
    )
    log_delta = np.abs(
        np.log(np.maximum(right_share[right_indices], TINY))
        - np.log(np.maximum(left_share[left_indices], TINY))
    )
    return {
        "common": common,
        "coverage": coverage,
        "h_delta": float(np.sum(normalized_weight * delta_h)),
        "v_delta": float(np.sum(normalized_weight * delta_v)),
        "d_delta": float(np.sum(normalized_weight * delta_d)),
        "coherence": float(np.sum(normalized_weight * coherence[accepted])),
        "log_energy_delta": float(np.sum(normalized_weight * log_delta)),
    }


def nearest_path_discontinuity(point: PointData, config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    continuous_unmatched = np.full(len(point.positions), np.nan, dtype=np.float32)
    discrete_unmatched = np.full(len(point.positions), np.nan, dtype=np.float32)
    spectral_jump = np.full(len(point.positions), np.nan, dtype=np.float32)
    for site_value in np.unique(point.site):
        ids = np.flatnonzero((point.site == site_value) & point.valid)
        _, local = cKDTree(point.positions[ids, :2]).query(
            point.positions[ids, :2], k=2, workers=-1
        )
        for row, source in enumerate(ids):
            target = int(ids[local[row, 1]])
            continuous = continuous_match_metrics(point, int(source), target, config)
            discrete = match_discrete_paths(
                point.discrete_coordinates[source],
                point.discrete_energy[source],
                point.discrete_coordinates[target],
                point.discrete_energy[target],
            )
            continuous_unmatched[source] = np.float32(1.0 - continuous["common"])
            discrete_unmatched[source] = np.float32(1.0 - discrete[0])
            pas_cosine = cosine_pairwise(
                point.global_pas, np.asarray([source]), np.asarray([target])
            )[0]
            pdp_cosine = cosine_pairwise(
                point.global_pdp, np.asarray([source]), np.asarray([target])
            )[0]
            spectral_jump[source] = np.float32(1.0 - 0.5 * (pas_cosine + pdp_cosine))
    return continuous_unmatched, discrete_unmatched, spectral_jump


def compute_path_pair_metrics(
    point: PointData,
    left: np.ndarray,
    right: np.ndarray,
    distance: np.ndarray,
    limit: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED + 11)
    if len(left) > limit:
        bins = distance_bin_indices(distance)
        selections: list[np.ndarray] = []
        per_bin = max(200, limit // (len(DISTANCE_EDGES) - 1))
        for bin_index in range(len(DISTANCE_EDGES) - 1):
            local = np.flatnonzero(bins == bin_index)
            if len(local) > per_bin:
                local = np.sort(rng.choice(local, size=per_bin, replace=False))
            selections.append(local)
        selected = np.concatenate(selections)
        if len(selected) > limit:
            selected = np.sort(rng.choice(selected, size=limit, replace=False))
    else:
        selected = np.arange(len(left))
    pair_left = left[selected]
    pair_right = right[selected]
    pair_distance = distance[selected]
    output = {
        "left": pair_left.astype(np.int32),
        "right": pair_right.astype(np.int32),
        "distance": pair_distance.astype(np.float32),
        "continuous_common": np.zeros(len(selected), dtype=np.float32),
        "continuous_coverage": np.zeros(len(selected), dtype=np.float32),
        "continuous_h_delta": np.full(len(selected), np.nan, dtype=np.float32),
        "continuous_v_delta": np.full(len(selected), np.nan, dtype=np.float32),
        "continuous_d_delta": np.full(len(selected), np.nan, dtype=np.float32),
        "continuous_coherence": np.full(len(selected), np.nan, dtype=np.float32),
        "continuous_log_energy_delta": np.full(len(selected), np.nan, dtype=np.float32),
        "discrete_common": np.zeros(len(selected), dtype=np.float32),
        "discrete_h_delta": np.full(len(selected), np.nan, dtype=np.float32),
        "discrete_v_delta": np.full(len(selected), np.nan, dtype=np.float32),
        "discrete_d_delta": np.full(len(selected), np.nan, dtype=np.float32),
        "identity_shuffle_common": np.full(len(selected), np.nan, dtype=np.float32),
    }
    config = Config(components=PATH_COUNT, transported_components=PATH_COUNT)
    shuffle_mask = rng.random(len(selected)) < min(1.0, 5000.0 / max(len(selected), 1))
    for row, (source, target) in enumerate(zip(pair_left, pair_right)):
        continuous = continuous_match_metrics(point, int(source), int(target), config)
        discrete = match_discrete_paths(
            point.discrete_coordinates[source],
            point.discrete_energy[source],
            point.discrete_coordinates[target],
            point.discrete_energy[target],
        )
        for key in (
            "common",
            "coverage",
            "h_delta",
            "v_delta",
            "d_delta",
            "coherence",
            "log_energy_delta",
        ):
            output[f"continuous_{key}"][row] = continuous[key]
        output["discrete_common"][row] = discrete[0]
        output["discrete_h_delta"][row] = discrete[1]
        output["discrete_v_delta"][row] = discrete[2]
        output["discrete_d_delta"][row] = discrete[3]
        if shuffle_mask[row]:
            shuffled = continuous_match_metrics(
                point,
                int(source),
                int(target),
                config,
                shuffled_identity=True,
                rng=rng,
            )
            output["identity_shuffle_common"][row] = shuffled["common"]
        if (row + 1) % 1000 == 0 or row + 1 == len(selected):
            print(f"连续/离散路径配对 {row + 1}/{len(selected)}", flush=True)
    return output


def fit_carrier_beta(
    normalized_cross: np.ndarray,
    radial_delta: np.ndarray,
    site: np.ndarray,
    raw_abs: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    beta = np.full(2, 140.2, dtype=np.float64)
    rows: list[dict[str, float | int]] = []
    candidates = np.arange(130.0, 150.0001, 0.01, dtype=np.float64)
    pair_id = np.arange(len(radial_delta))
    for site_value in (0, 1):
        fit = (
            (site == site_value)
            & (pair_id % 2 == 0)
            & (raw_abs >= np.quantile(raw_abs[site == site_value], 0.55))
            & (np.abs(radial_delta) > 0.05)
        )
        evaluate = (site == site_value) & (pair_id % 2 == 1)
        value = normalized_cross[fit]
        delta = radial_delta[fit]
        score = np.empty(len(candidates), dtype=np.float64)
        for start in range(0, len(candidates), 256):
            end = min(start + 256, len(candidates))
            phase = np.exp(1j * delta[:, None] * candidates[None, start:end])
            score[start:end] = np.mean(np.real(value[:, None] * phase), axis=0)
        selected = float(candidates[int(np.argmax(score))])
        beta[site_value] = selected
        held_correct = np.real(
            normalized_cross[evaluate]
            * np.exp(1j * selected * radial_delta[evaluate])
        )
        held_wrong = np.real(
            normalized_cross[evaluate]
            * np.exp(1j * (selected - 20.0) * radial_delta[evaluate])
        )
        rows.append(
            {
                "site": site_value,
                "fit_pair_count": int(np.sum(fit)),
                "held_pair_count": int(np.sum(evaluate)),
                "beta_rad_per_m": selected,
                "equivalent_frequency_ghz": selected * 299_792_458.0 / (2.0 * math.pi) / 1e9,
                "fit_objective": float(np.max(score)),
                "held_corrected_signed_coherence": float(np.mean(held_correct)),
                "held_wrong_beta_signed_coherence": float(np.mean(held_wrong)),
            }
        )
    return beta, rows


def compute_phase_pair_metrics(
    point: PointData,
    derived: dict[str, np.ndarray],
    path_pairs: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, float | int]]]:
    left = path_pairs["left"].astype(np.int64)
    right = path_pairs["right"].astype(np.int64)
    count = len(left)
    channel = np.load(TRAIN_CHANNEL, mmap_mode="r")
    resident_channel: torch.Tensor | None = None
    if device.type == "cuda":
        required = int(np.prod(channel.shape) * np.dtype(np.complex64).itemsize)
        free_memory, _ = torch.cuda.mem_get_info(device)
        if free_memory > required * 1.35:
            resident_channel = torch.empty(channel.shape, dtype=torch.complex64, device=device)
            load_batch = max(4, batch_size)
            for load_start in range(0, len(channel), load_batch):
                load_end = min(load_start + load_batch, len(channel))
                resident_channel[load_start:load_end].copy_(
                    torch.as_tensor(
                        np.array(channel[load_start:load_end], dtype=np.complex64, copy=True),
                        device=device,
                    )
                )
                if (load_end % 500 == 0) or load_end == len(channel):
                    print(f"原始H载入单张GPU {load_end}/{len(channel)}", flush=True)
    raw_cross = np.zeros(count, dtype=np.complex64)
    raw_abs = np.zeros(count, dtype=np.float32)
    delay_aligned_cross = np.zeros(count, dtype=np.complex64)
    pair_batch = max(8, min(128, batch_size * 4))
    frequency_index = torch.arange(S_SC, dtype=torch.float32, device=device)
    for start in range(0, count, pair_batch):
        end = min(start + pair_batch, count)
        if resident_channel is not None:
            left_index = torch.as_tensor(left[start:end], dtype=torch.long, device=device)
            right_index = torch.as_tensor(right[start:end], dtype=torch.long, device=device)
            first = resident_channel[left_index]
            second = resident_channel[right_index]
        else:
            first = torch.as_tensor(
                np.array(channel[left[start:end]], dtype=np.complex64, copy=True), device=device
            )
            second = torch.as_tensor(
                np.array(channel[right[start:end]], dtype=np.complex64, copy=True), device=device
            )
        cross_frequency = torch.sum(torch.conj(first) * second, dim=(1, 2))
        denominator = torch.sqrt(
            first.abs().square().sum(dim=(1, 2, 3))
            * second.abs().square().sum(dim=(1, 2, 3))
        ).clamp_min(1e-30)
        cross = cross_frequency.sum(dim=1) / denominator
        raw_cross[start:end] = cross.cpu().numpy().astype(np.complex64)
        raw_abs[start:end] = cross.abs().cpu().numpy().astype(np.float32)
        delay_delta = torch.as_tensor(
            circular_delta(
                point.main_delay[right[start:end]],
                point.main_delay[left[start:end]],
                S_SC,
            ),
            dtype=torch.float32,
            device=device,
        )
        correction = torch.exp(
            1j * 2.0 * math.pi * delay_delta[:, None] * frequency_index[None] / S_SC
        )
        aligned = torch.sum(cross_frequency * correction, dim=1) / denominator
        delay_aligned_cross[start:end] = aligned.cpu().numpy().astype(np.complex64)
        if (end % 2000 == 0) or end == count:
            print(f"原始H空间对相干性 {end}/{count}", flush=True)
    del resident_channel

    radial_delta = point.bs_distance[right] - point.bs_distance[left]
    pair_site = point.site[left].astype(np.int8)
    beta, beta_rows = fit_carrier_beta(raw_cross, radial_delta, pair_site, raw_abs)
    carrier = np.exp(1j * beta[pair_site] * radial_delta)
    wrong_beta = np.exp(1j * (beta[pair_site] - 20.0) * radial_delta)
    rng = np.random.default_rng(RANDOM_SEED + 29)
    shuffled_radial = np.empty_like(radial_delta)
    for site_value in (0, 1):
        mask = pair_site == site_value
        shuffled_radial[mask] = rng.permutation(radial_delta[mask])
    wrong_map = np.exp(1j * beta[pair_site] * shuffled_radial)
    output = {
        "raw_abs": raw_abs,
        "raw_signed": np.real(raw_cross).astype(np.float32),
        "carrier_signed": np.real(raw_cross * carrier).astype(np.float32),
        "carrier_delay_signed": np.real(delay_aligned_cross * carrier).astype(np.float32),
        "delay_aligned_abs": np.abs(delay_aligned_cross).astype(np.float32),
        "wrong_beta_signed": np.real(raw_cross * wrong_beta).astype(np.float32),
        "wrong_map_signed": np.real(raw_cross * wrong_map).astype(np.float32),
        "radial_delta": radial_delta.astype(np.float32),
        "site": pair_site,
        "port_cosine": cosine_pairwise(derived["port_share"], left, right).astype(np.float32),
    }
    return output, beta_rows


def spectrum_pair_metrics(
    point: PointData,
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED + 3)
    shuffled = np.arange(len(point.positions), dtype=np.int64)
    for site_value in (0, 1):
        ids = np.flatnonzero((point.site == site_value) & point.valid)
        shuffled[ids] = rng.permutation(ids)
    main_discrete_left = point.discrete_coordinates[left, 0]
    main_discrete_right = point.discrete_coordinates[right, 0]
    discrete_consistency = (
        (np.abs(circular_delta(main_discrete_right[:, 0], main_discrete_left[:, 0], MH)) <= 1.0)
        & (np.abs(circular_delta(main_discrete_right[:, 1], main_discrete_left[:, 1], MV)) <= 1.0)
        & (np.abs(circular_delta(main_discrete_right[:, 2], main_discrete_left[:, 2], S_SC)) <= 2.0)
    )
    main_cont_left = point.component_coordinates[left, 0]
    main_cont_right = point.component_coordinates[right, 0]
    continuous_consistency = (
        (np.abs(circular_delta(main_cont_right[:, 0], main_cont_left[:, 0], MH)) <= 1.0)
        & (np.abs(circular_delta(main_cont_right[:, 1], main_cont_left[:, 1], MV)) <= 1.0)
        & (np.abs(circular_delta(main_cont_right[:, 2], main_cont_left[:, 2], S_SC)) <= 2.0)
    )
    return {
        "log_power_delta": np.abs(np.log10(point.power[left]) - np.log10(point.power[right])).astype(np.float32),
        "pas_distance": (1.0 - cosine_pairwise(point.global_pas, left, right)).astype(np.float32),
        "pdp_distance": (1.0 - cosine_pairwise(point.global_pdp, left, right)).astype(np.float32),
        "shuffled_pas_distance": (
            1.0 - cosine_pairwise(point.global_pas, shuffled[left], shuffled[right])
        ).astype(np.float32),
        "shuffled_pdp_distance": (
            1.0 - cosine_pairwise(point.global_pdp, shuffled[left], shuffled[right])
        ).astype(np.float32),
        "discrete_main_consistency": discrete_consistency.astype(np.float32),
        "continuous_main_consistency": continuous_consistency.astype(np.float32),
    }


def geometry_controls(
    point: PointData,
    continuous_unmatched: np.ndarray,
    spectral_jump: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    geometry_index = {name: index for index, name in enumerate(point.geometry_names)}
    geodesic_index = {name: index for index, name in enumerate(point.geodesic_names)}
    features = {
        "距最近墙面（米）": point.geometry[:, geometry_index["wall_nearest_xy"]].astype(np.float64),
        "直达扇区净空比例": point.geometry[:, geometry_index["fan_clear_fraction"]].astype(np.float64),
        "传播通路平均宽度（米）": (
            2.0 * point.geodesic[:, geodesic_index["mean_clearance"]]
        ).astype(np.float64),
        "相对直线绕行距离（米）": point.geodesic[:, geodesic_index["geodesic_excess"]].astype(np.float64),
    }
    outcomes = {
        "是否全零": (~point.valid).astype(np.float64),
        "log10总功率": np.where(point.valid, np.log10(np.maximum(point.power, TINY)), np.nan),
        "连续路径未匹配能量": continuous_unmatched.astype(np.float64),
        "最近邻谱突变": spectral_jump.astype(np.float64),
    }
    rng = np.random.default_rng(RANDOM_SEED + 41)
    shuffled_index = np.arange(len(point.positions))
    for site_value in (0, 1):
        ids = np.flatnonzero(point.site == site_value)
        shuffled_index[ids] = rng.permutation(ids)
    rows: list[dict[str, Any]] = []
    for feature_name, feature in features.items():
        for outcome_name, outcome in outcomes.items():
            mask = np.isfinite(feature) & np.isfinite(outcome)
            true_rho = spearmanr(feature[mask], outcome[mask]).statistic
            shuffled_feature = feature[shuffled_index]
            shuffled_rho = spearmanr(shuffled_feature[mask], outcome[mask]).statistic
            rows.append(
                {
                    "control": "地图错位",
                    "feature": feature_name,
                    "outcome": outcome_name,
                    "count": int(np.sum(mask)),
                    "true_spearman": float(true_rho),
                    "wrong_map_spearman": float(shuffled_rho),
                    "absolute_margin": float(abs(true_rho) - abs(shuffled_rho)),
                }
            )
    return rows, features


def geometry_binned_rows(
    point: PointData,
    features: dict[str, np.ndarray],
    continuous_unmatched: np.ndarray,
    spectral_jump: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    log_power = np.where(point.valid, np.log10(np.maximum(point.power, TINY)), np.nan)
    for feature_name, values in features.items():
        finite = np.isfinite(values)
        edges = np.unique(np.quantile(values[finite], np.linspace(0.0, 1.0, 11)))
        if len(edges) < 4:
            lower, upper = np.nanmin(values), np.nanmax(values)
            edges = np.linspace(lower, upper + 1e-9, 6)
        bins = np.clip(np.digitize(values, edges[1:-1], right=True), 0, len(edges) - 2)
        for bin_index in range(len(edges) - 1):
            mask = finite & (bins == bin_index)
            valid_mask = mask & point.valid
            rows.append(
                {
                    "feature": feature_name,
                    "bin": bin_index,
                    "lower": float(edges[bin_index]),
                    "upper": float(edges[bin_index + 1]),
                    "center": float(np.median(values[mask])) if np.any(mask) else math.nan,
                    "count": int(np.sum(mask)),
                    "zero_rate": float(np.mean(~point.valid[mask])) if np.any(mask) else math.nan,
                    "median_log_power": float(np.nanmedian(log_power[valid_mask])) if np.any(valid_mask) else math.nan,
                    "median_unmatched_energy": float(np.nanmedian(continuous_unmatched[valid_mask])) if np.any(valid_mask) else math.nan,
                    "median_spectral_jump": float(np.nanmedian(spectral_jump[valid_mask])) if np.any(valid_mask) else math.nan,
                }
            )
    return rows


def save_figure(figure: plt.Figure, name: str) -> None:
    figure.savefig(FIGURE_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(figure)


def binned_mean(distance: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins = distance_bin_indices(distance)
    x: list[float] = []
    mean: list[float] = []
    count: list[int] = []
    for bin_index in range(len(DISTANCE_EDGES) - 1):
        mask = (bins == bin_index) & np.isfinite(values)
        if np.sum(mask) < 15:
            continue
        x.append(float(np.median(distance[mask])))
        mean.append(float(np.mean(values[mask])))
        count.append(int(np.sum(mask)))
    return np.asarray(x), np.asarray(mean), np.asarray(count)


def plot_figures_01_to_04(
    point: PointData,
    setup: Any,
    vertices: np.ndarray,
) -> None:
    valid_log = np.log10(point.power[point.valid])
    lower, upper = np.quantile(valid_log, [0.02, 0.98])
    normalization = Normalize(vmin=lower, vmax=upper)
    figure, axes = plt.subplots(1, 2, figsize=(15.5, 6.5), constrained_layout=True)
    scatter = None
    for site_value, axis in enumerate(axes):
        map_background(axis, vertices, point.positions[point.site == site_value])
        mask = (point.site == site_value) & point.valid
        scatter = axis.scatter(
            point.positions[mask, 0],
            point.positions[mask, 1],
            c=np.log10(point.power[mask]),
            cmap="viridis",
            norm=normalization,
            s=13,
            linewidths=0,
            zorder=3,
        )
        add_base_stations(axis, setup.bs_positions, (site_value,))
        axis.set_title(f"基站{site_value}：非零训练点总功率")
    assert scatter is not None
    figure.colorbar(scatter, ax=axes, label="log10（信道总功率）", shrink=0.88)
    save_figure(figure, "01_训练点总功率空间分布.png")

    figure, axes = plt.subplots(1, 2, figsize=(15.5, 6.5), constrained_layout=True)
    for site_value, axis in enumerate(axes):
        local = point.site == site_value
        map_background(axis, vertices, point.positions[local])
        valid = local & point.valid
        zero = local & ~point.valid
        axis.scatter(
            point.positions[valid, 0],
            point.positions[valid, 1],
            s=8,
            color="#B8B8B8",
            alpha=0.45,
            linewidths=0,
            label=f"非零点 {int(np.sum(valid))}",
        )
        axis.scatter(
            point.positions[zero, 0],
            point.positions[zero, 1],
            s=24,
            marker="x",
            color="#111111",
            linewidths=1.0,
            label=f"全零点 {int(np.sum(zero))}",
        )
        add_base_stations(axis, setup.bs_positions, (site_value,))
        axis.set_title(f"基站{site_value}：全零信道覆盖状态")
        axis.legend(loc="best")
    save_figure(figure, "02_全零信道点空间聚集.png")

    colors = ("#2C6EAA", "#D97706")
    figure, axis = plt.subplots(figsize=(9.5, 6.2))
    for site_value, color in enumerate(colors):
        mask = (point.site == site_value) & point.valid
        x = point.bs_distance[mask]
        y = np.log10(point.power[mask])
        axis.scatter(x, y, s=7, alpha=0.13, color=color, linewidths=0)
        edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, 17)))
        centers: list[float] = []
        median: list[float] = []
        q25: list[float] = []
        q75: list[float] = []
        for lower_edge, upper_edge in zip(edges[:-1], edges[1:]):
            local = (x >= lower_edge) & (x <= upper_edge)
            if np.sum(local) < 8:
                continue
            centers.append(float(np.median(x[local])))
            median.append(float(np.median(y[local])))
            q25.append(float(np.quantile(y[local], 0.25)))
            q75.append(float(np.quantile(y[local], 0.75)))
        axis.plot(centers, median, color=color, lw=2.2, marker="o", ms=4, label=f"基站{site_value}分箱中位数")
        axis.fill_between(centers, q25, q75, color=color, alpha=0.12)
    axis.set_xlabel("训练点到服务基站的三维距离（米）")
    axis.set_ylabel("log10（信道总功率）")
    axis.set_title("总功率随基站距离的变化")
    axis.legend()
    save_figure(figure, "03_基站距离与总功率关系.png")

    figure, axis = plt.subplots(figsize=(9.5, 6.2))
    for site_value, color in enumerate(colors):
        mask = point.site == site_value
        x = point.bs_distance[mask]
        y = (~point.valid[mask]).astype(np.float64)
        edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, 13)))
        centers: list[float] = []
        rate: list[float] = []
        count: list[int] = []
        for lower_edge, upper_edge in zip(edges[:-1], edges[1:]):
            local = (x >= lower_edge) & (x <= upper_edge)
            centers.append(float(np.median(x[local])))
            rate.append(float(np.mean(y[local])))
            count.append(int(np.sum(local)))
        axis.plot(centers, rate, color=color, lw=2.2, marker="o", ms=4, label=f"基站{site_value}")
    axis.set_xlabel("训练点到服务基站的三维距离（米）")
    axis.set_ylabel("全零信道比例")
    axis.set_ylim(-0.02, 1.02)
    axis.set_title("全零覆盖状态与基站距离")
    axis.legend()
    save_figure(figure, "04_基站距离分箱与全零比例.png")


def plot_figures_05_to_06(
    point: PointData,
    derived: dict[str, np.ndarray],
    pair_distance: np.ndarray,
    spectrum_pairs: dict[str, np.ndarray],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17.0, 5.2), constrained_layout=True)
    labels = (
        ("log_power_delta", "|Δlog10总功率|", "#2C6EAA"),
        ("pas_distance", "1 − PAS余弦相似度", "#00876C"),
        ("pdp_distance", "1 − PDP余弦相似度", "#D97706"),
    )
    for axis, (key, ylabel, color) in zip(axes, labels):
        plot_binned(axis, pair_distance, spectrum_pairs[key], label="真实空间对应", color=color)
        if key in ("pas_distance", "pdp_distance"):
            plot_binned(
                axis,
                pair_distance,
                spectrum_pairs[f"shuffled_{key}"],
                label="同站空间打乱控制",
                color="#777777",
                interval=False,
                linestyle="--",
            )
        axis.set_xscale("log")
        axis.set_xlabel("同基站两点间距离（米，对数轴）")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8)
    figure.suptitle("空间距离对功率、PAS和PDP连续性的影响", fontsize=14)
    save_figure(figure, "05_空间距离与功率及谱差异.png")

    continuous = np.cumsum(point.component_share, axis=1)
    discrete = derived["discrete_cumulative"]
    k = np.arange(1, PATH_COUNT + 1)
    figure, axis = plt.subplots(figsize=(9.4, 6.2))
    for values, label, color in (
        (continuous[point.valid], "24分量连续离网格分解", "#2C6EAA"),
        (discrete[point.valid], "离散angle-delay网格峰", "#D97706"),
    ):
        median = np.median(values, axis=0)
        q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
        axis.plot(k, median, marker="o", ms=3.5, lw=2.2, color=color, label=label)
        axis.fill_between(k, q25, q75, color=color, alpha=0.13)
    axis.axhline(0.90, color="#555555", linestyle="--", lw=1.0)
    axis.axhline(0.95, color="#555555", linestyle=":", lw=1.0)
    axis.set_xlim(1, PATH_COUNT)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("保留的最强路径/峰数量K")
    axis.set_ylabel("累计能量比例")
    axis.set_title("信道能量是否集中在少量delay-angle路径")
    axis.legend()
    save_figure(figure, "06_最强K条路径累计能量.png")


def plot_figures_07_to_09(
    point: PointData,
    setup: Any,
    vertices: np.ndarray,
    pair_distance: np.ndarray,
    spectrum_pairs: dict[str, np.ndarray],
) -> None:
    delay_max = max(1, int(np.max(point.main_delay[point.valid])))
    figure, axes = plt.subplots(1, 2, figsize=(15.5, 6.5), constrained_layout=True)
    scatter = None
    for site_value, axis in enumerate(axes):
        local = point.site == site_value
        map_background(axis, vertices, point.positions[local])
        mask = local & point.valid
        scatter = axis.scatter(
            point.positions[mask, 0],
            point.positions[mask, 1],
            c=point.main_delay[mask],
            cmap="turbo",
            vmin=0,
            vmax=delay_max,
            s=13,
            linewidths=0,
        )
        zero = local & ~point.valid
        axis.scatter(point.positions[zero, 0], point.positions[zero, 1], marker="x", s=18, color="black")
        add_base_stations(axis, setup.bs_positions, (site_value,))
        axis.set_title(f"基站{site_value}：主PDP时延格")
    assert scatter is not None
    figure.colorbar(scatter, ax=axes, label=f"主PDP时延格（数据范围0–{delay_max}）", shrink=0.88)
    save_figure(figure, "07_主PDP时延空间分布.png")

    figure, axes = plt.subplots(2, 2, figsize=(15.5, 12.0), constrained_layout=True)
    for site_value in (0, 1):
        local = point.site == site_value
        for row, (values, title, limits, cmap) in enumerate(
            (
                (point.main_h_angle, "水平阵列等效角", (-90.0, 90.0), "twilight_shifted"),
                (point.main_v_angle, "垂直阵列等效角", (-90.0, 90.0), "coolwarm"),
            )
        ):
            axis = axes[row, site_value]
            map_background(axis, vertices, point.positions[local])
            mask = local & point.valid
            scatter = axis.scatter(
                point.positions[mask, 0],
                point.positions[mask, 1],
                c=values[mask],
                cmap=cmap,
                vmin=limits[0],
                vmax=limits[1],
                s=13,
                linewidths=0,
            )
            add_base_stations(axis, setup.bs_positions, (site_value,))
            axis.set_title(f"基站{site_value}：{title}")
            figure.colorbar(scatter, ax=axis, label="等效角（度）", shrink=0.82)
    figure.suptitle("主PAS阵列角空间分布（半波长等效角，仅表示FFT空间频率）", fontsize=14)
    save_figure(figure, "08_主水平与垂直PAS等效角空间分布.png")

    figure, axis = plt.subplots(figsize=(9.5, 6.2))
    for key, label, color in (
        ("discrete_main_consistency", "离散angle-delay主峰", "#D97706"),
        ("continuous_main_consistency", "连续离网格主分量", "#2C6EAA"),
    ):
        x, mean, _ = binned_mean(pair_distance, spectrum_pairs[key])
        axis.plot(x, mean, marker="o", ms=4, lw=2.2, color=color, label=label)
    axis.set_xscale("log")
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("同基站两点间距离（米，对数轴）")
    axis.set_ylabel("主峰位置一致率")
    axis.set_title("主路径delay-angle身份随空间距离的保持率")
    axis.legend()
    save_figure(figure, "09_空间距离与主路径一致率.png")


def plot_figures_10_to_13(
    point: PointData,
    setup: Any,
    vertices: np.ndarray,
    path_pairs: dict[str, np.ndarray],
    continuous_unmatched: np.ndarray,
    discrete_unmatched: np.ndarray,
) -> None:
    distance = path_pairs["distance"]
    figure, axis = plt.subplots(figsize=(9.5, 6.2))
    plot_binned(
        axis,
        distance,
        path_pairs["continuous_common"],
        label="连续24分量共同能量",
        color="#2C6EAA",
    )
    plot_binned(
        axis,
        distance,
        path_pairs["discrete_common"],
        label="离散24峰共同能量",
        color="#D97706",
    )
    shuffled = np.isfinite(path_pairs["identity_shuffle_common"])
    if np.any(shuffled):
        plot_binned(
            axis,
            distance[shuffled],
            path_pairs["identity_shuffle_common"][shuffled],
            label="路径身份打乱控制",
            color="#777777",
            interval=False,
            linestyle="--",
        )
    axis.set_xscale("log")
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("同基站两点间距离（米，对数轴）")
    axis.set_ylabel("可匹配路径的共同能量比例")
    axis.set_title("两种路径表示下的路径持续性")
    axis.legend()
    save_figure(figure, "10_空间距离与匹配路径解释能量.png")

    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), constrained_layout=True)
    for axis, key, ylabel, color in (
        (axes[0], "continuous_d_delta", "|Δ时延格| / 米", "#D97706"),
        (axes[1], "continuous_h_delta", "|Δ水平角格| / 米", "#2C6EAA"),
        (axes[2], "continuous_v_delta", "|Δ垂直角格| / 米", "#00876C"),
    ):
        rate = np.divide(
            path_pairs[key],
            distance,
            out=np.full_like(path_pairs[key], np.nan),
            where=distance > 0.2,
        )
        plot_binned(axis, distance, rate, label="连续24分量", color=color)
        axis.set_xscale("log")
        axis.set_xlabel("两点距离（米，对数轴）")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8)
    figure.suptitle("匹配路径的delay-angle位置变化速度", fontsize=14)
    save_figure(figure, "11_匹配路径时延与角度变化速度.png")

    figure, axes = plt.subplots(2, 2, figsize=(15.5, 12.0), constrained_layout=True)
    for site_value in (0, 1):
        local = point.site == site_value
        for row, (values, title) in enumerate(
            (
                (continuous_unmatched, "连续24分量未匹配能量"),
                (discrete_unmatched, "离散24峰未匹配能量"),
            )
        ):
            axis = axes[row, site_value]
            map_background(axis, vertices, point.positions[local])
            mask = local & point.valid & np.isfinite(values)
            scatter = axis.scatter(
                point.positions[mask, 0],
                point.positions[mask, 1],
                c=values[mask],
                cmap="magma",
                vmin=0.0,
                vmax=float(np.nanquantile(values, 0.98)),
                s=13,
                linewidths=0,
            )
            add_base_stations(axis, setup.bs_positions, (site_value,))
            axis.set_title(f"基站{site_value}：{title}")
            figure.colorbar(scatter, ax=axis, label="未匹配能量比例", shrink=0.82)
    figure.suptitle("相对最近同站非零邻点的路径出生/消失风险", fontsize=14)
    save_figure(figure, "12_路径出生消失未匹配能量空间分布.png")

    figure, axis = plt.subplots(figsize=(9.5, 6.2))
    plot_binned(
        axis,
        distance,
        path_pairs["continuous_log_energy_delta"],
        label="匹配路径能量变化",
        color="#8E5EA2",
    )
    axis.set_xscale("log")
    axis.set_xlabel("同基站两点间距离（米，对数轴）")
    axis.set_ylabel("能量加权 |Δln（路径能量占比）|")
    axis.set_title("路径身份保持时，路径能量仍如何变化")
    axis.legend()
    save_figure(figure, "13_空间距离与匹配路径能量变化.png")


def plot_figure_14(geometry_rows: list[dict[str, Any]]) -> None:
    feature_order = (
        "距最近墙面（米）",
        "直达扇区净空比例",
        "传播通路平均宽度（米）",
        "相对直线绕行距离（米）",
    )
    outcomes = (
        ("zero_rate", "全零比例", "#111111"),
        ("median_log_power", "非零点log10总功率", "#2C6EAA"),
        ("median_unmatched_energy", "路径未匹配能量", "#D97706"),
        ("median_spectral_jump", "最近邻谱突变", "#8E5EA2"),
    )
    figure, axes = plt.subplots(4, 4, figsize=(19.0, 15.5), constrained_layout=True)
    for row_index, feature_name in enumerate(feature_order):
        selected = [row for row in geometry_rows if row["feature"] == feature_name]
        selected.sort(key=lambda row: int(row["bin"]))
        x = np.asarray([row["center"] for row in selected], dtype=np.float64)
        for column_index, (key, ylabel, color) in enumerate(outcomes):
            axis = axes[row_index, column_index]
            y = np.asarray([row[key] for row in selected], dtype=np.float64)
            axis.plot(x, y, marker="o", ms=4, lw=2.0, color=color)
            axis.set_xlabel(feature_name)
            axis.set_ylabel(ylabel)
            if column_index == 0:
                axis.set_ylim(-0.02, min(1.02, max(0.10, float(np.nanmax(y) * 1.15))))
    figure.suptitle("官方地图描述量与覆盖、功率、路径事件和谱突变的分箱关系", fontsize=15)
    save_figure(figure, "14_地图结构与覆盖及路径突变关系.png")


def plot_figures_15_to_17(
    point: PointData,
    derived: dict[str, np.ndarray],
    path_pairs: dict[str, np.ndarray],
    phase_pairs: dict[str, np.ndarray],
) -> None:
    distance = path_pairs["distance"]
    figure, axis = plt.subplots(figsize=(9.5, 6.2))
    plot_binned(
        axis,
        distance,
        phase_pairs["raw_abs"],
        label="原始复信道绝对相干性",
        color="#2C6EAA",
    )
    plot_binned(
        axis,
        distance,
        phase_pairs["delay_aligned_abs"],
        label="主时延差修正后绝对相干性",
        color="#D97706",
    )
    axis.set_xscale("log")
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("同基站两点间距离（米，对数轴）")
    axis.set_ylabel("|归一化复内积|")
    axis.set_title("原始复信道形状的空间相干距离")
    axis.legend()
    save_figure(figure, "15_空间距离与原始复信道相干性.png")

    figure, axis = plt.subplots(figsize=(10.0, 6.4))
    for key, label, color, style in (
        ("raw_signed", "未修正", "#777777", "--"),
        ("carrier_signed", "官方H拟合载波距离相位", "#2C6EAA", "-"),
        ("carrier_delay_signed", "载波距离相位 + 主时延差", "#00876C", "-"),
        ("wrong_beta_signed", "错误波数控制", "#C0392B", ":"),
        ("wrong_map_signed", "错位距离控制", "#8E5EA2", ":"),
    ):
        plot_binned(
            axis,
            distance,
            phase_pairs[key],
            label=label,
            color=color,
            interval=False,
            linestyle=style,
        )
    axis.axhline(0.0, color="#222222", lw=0.8)
    axis.set_xscale("log")
    axis.set_xlabel("同基站两点间距离（米，对数轴）")
    axis.set_ylabel("归一化复内积实部（有符号相干性）")
    axis.set_title("载波相位和路径时延修正是否恢复可预测相位")
    axis.legend(fontsize=8, ncol=2)
    save_figure(figure, "16_载波与路径时延修正后的相位相干性.png")

    phase_delay = derived["phase_delay"]
    dominant = np.max(point.global_pdp, axis=1)
    valid = point.valid & np.isfinite(phase_delay)
    strong = valid & (dominant >= np.quantile(dominant[valid], 0.60))
    display_max = float(
        math.ceil(
            max(
                np.max(point.main_delay[valid]),
                np.max(phase_delay[valid]),
            )
            + 1.0
        )
    )
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.0), constrained_layout=True)
    for axis, mask, title in (
        (axes[0], valid, "全部非零点"),
        (axes[1], strong, "PDP主峰占比较高的40%点"),
    ):
        plot = axis.hexbin(
            point.main_delay[mask],
            phase_delay[mask],
            gridsize=48,
            bins="log",
            cmap="viridis",
            mincnt=1,
        )
        axis.plot([0, display_max], [0, display_max], color="#D62728", lw=1.2, linestyle="--")
        axis.set_xlim(0, display_max)
        axis.set_ylim(0, display_max)
        axis.set_xlabel("PDP主峰时延格")
        axis.set_ylabel("频率相位斜率估计时延格")
        axis.set_title(title)
        figure.colorbar(plot, ax=axis, label="点数（对数）")
    figure.suptitle("PDP能量峰与复相位频率斜率是否指向同一时延", fontsize=14)
    save_figure(figure, "17_PDP主时延与频率相位斜率对照.png")


def plot_figures_18_to_20(
    point: PointData,
    derived: dict[str, np.ndarray],
    path_pairs: dict[str, np.ndarray],
    phase_pairs: dict[str, np.ndarray],
    continuous_unmatched: np.ndarray,
) -> list[dict[str, Any]]:
    colors = ("#2C6EAA", "#D97706")
    rank_share = derived["rank_share"]
    figure, axis = plt.subplots(figsize=(9.5, 6.2))
    rank_index = np.arange(1, N_UE + 1)
    for site_value, color in enumerate(colors):
        mask = (point.site == site_value) & point.valid
        cumulative = np.cumsum(rank_share[mask], axis=1)
        median = np.median(cumulative, axis=0)
        q25, q75 = np.quantile(cumulative, [0.25, 0.75], axis=0)
        axis.plot(rank_index, median, marker="o", ms=5, lw=2.2, color=color, label=f"基站{site_value}")
        axis.fill_between(rank_index, q25, q75, color=color, alpha=0.13)
    axis.set_xticks(rank_index)
    axis.set_xlim(1, N_UE)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("保留的MIMO奇异模态数量")
    axis.set_ylabel("累计信道能量比例")
    axis.set_title("256×4 MIMO信道的有效秩")
    axis.legend()
    save_figure(figure, "18_MIMO奇异值累计解释能量.png")

    distance = path_pairs["distance"]
    figure, axis = plt.subplots(figsize=(9.5, 6.2))
    plot_binned(
        axis,
        distance,
        phase_pairs["port_cosine"],
        label="真实极化×UE功率组织",
        color="#00876C",
    )
    rng = np.random.default_rng(RANDOM_SEED + 53)
    shuffled = phase_pairs["port_cosine"].copy()
    for site_value in (0, 1):
        mask = phase_pairs["site"] == site_value
        shuffled[mask] = rng.permutation(shuffled[mask])
    plot_binned(
        axis,
        distance,
        shuffled,
        label="同站空间打乱控制",
        color="#777777",
        interval=False,
        linestyle="--",
    )
    axis.set_xscale("log")
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("同基站两点间距离（米，对数轴）")
    axis.set_ylabel("8维极化×UE功率向量余弦相似度")
    axis.set_title("极化与UE端口能量分配的空间连续性")
    axis.legend()
    save_figure(figure, "19_极化与UE功率组织空间连续性.png")

    component_k90 = np.argmax(np.cumsum(point.component_share, axis=1) >= 0.90, axis=1) + 1
    port_entropy = -np.sum(
        derived["port_share"] * np.log(np.maximum(derived["port_share"], 1e-30)), axis=1
    ) / math.log(MP * N_UE)
    descriptors = {
        "log10总功率": np.where(point.valid, np.log10(np.maximum(point.power, TINY)), np.nan),
        "PDP熵": -np.sum(
            point.global_pdp * np.log(np.maximum(point.global_pdp, 1e-30)), axis=1
        ) / math.log(S_SC),
        "90%能量所需路径数": component_k90.astype(np.float64),
        "最近邻路径未匹配能量": continuous_unmatched.astype(np.float64),
        "第一MIMO模态能量": rank_share[:, 0].astype(np.float64),
        "极化×UE功率熵": port_entropy.astype(np.float64),
    }
    quantiles = np.linspace(0.02, 0.98, 49)
    figure, axes = plt.subplots(2, 3, figsize=(16.5, 10.0), constrained_layout=True)
    site_rows: list[dict[str, Any]] = []
    for axis, (name, values) in zip(axes.ravel(), descriptors.items()):
        samples: list[np.ndarray] = []
        for site_value, color in enumerate(colors):
            mask = (point.site == site_value) & point.valid & np.isfinite(values)
            sample = values[mask]
            samples.append(sample)
            axis.plot(
                quantiles,
                np.quantile(sample, quantiles),
                lw=2.2,
                color=color,
                label=f"基站{site_value}",
            )
        ks = ks_2samp(samples[0], samples[1])
        site_rows.append(
            {
                "descriptor": name,
                "site0_count": int(len(samples[0])),
                "site1_count": int(len(samples[1])),
                "site0_median": float(np.median(samples[0])),
                "site1_median": float(np.median(samples[1])),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
            }
        )
        axis.set_xlabel("分位数")
        axis.set_ylabel(name)
        axis.set_title(f"{name}（KS={ks.statistic:.3f}）")
        axis.legend(fontsize=8)
    figure.suptitle("基站0与基站1的传播描述量分布是否可共享", fontsize=14)
    save_figure(figure, "20_双基站传播描述量分位数对照.png")
    return site_rows


def distance_statistics_rows(
    distance: np.ndarray,
    spectrum_pairs: dict[str, np.ndarray],
    path_pairs: dict[str, np.ndarray],
    phase_pairs: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    full_bins = distance_bin_indices(distance)
    path_distance = path_pairs["distance"]
    path_bins = distance_bin_indices(path_distance)
    for bin_index in range(len(DISTANCE_EDGES) - 1):
        full = full_bins == bin_index
        path = path_bins == bin_index
        if not np.any(full) and not np.any(path):
            continue
        row: dict[str, Any] = {
            "distance_bin": distance_bin_label(bin_index),
            "pair_count": int(np.sum(full)),
            "path_pair_count": int(np.sum(path)),
            "median_distance_m": float(np.median(distance[full])) if np.any(full) else math.nan,
        }
        for key in (
            "log_power_delta",
            "pas_distance",
            "pdp_distance",
            "shuffled_pas_distance",
            "shuffled_pdp_distance",
            "discrete_main_consistency",
            "continuous_main_consistency",
        ):
            row[key] = float(np.mean(spectrum_pairs[key][full])) if np.any(full) else math.nan
        for key in (
            "continuous_common",
            "discrete_common",
            "continuous_h_delta",
            "continuous_v_delta",
            "continuous_d_delta",
            "continuous_log_energy_delta",
        ):
            row[key] = float(np.nanmedian(path_pairs[key][path])) if np.any(path) else math.nan
        for key in (
            "raw_abs",
            "delay_aligned_abs",
            "raw_signed",
            "carrier_signed",
            "carrier_delay_signed",
            "wrong_beta_signed",
            "wrong_map_signed",
            "port_cosine",
        ):
            row[key] = float(np.nanmedian(phase_pairs[key][path])) if np.any(path) else math.nan
        rows.append(row)
    return rows


def control_rows(
    distance: np.ndarray,
    spectrum_pairs: dict[str, np.ndarray],
    path_pairs: dict[str, np.ndarray],
    phase_pairs: dict[str, np.ndarray],
    geometry_control_rows: list[dict[str, Any]],
    beta_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    near = distance <= 3.0
    rows.extend(
        [
            {
                "control": "空间打乱",
                "quantity": "3米内PAS距离",
                "true_value": float(np.mean(spectrum_pairs["pas_distance"][near])),
                "control_value": float(np.mean(spectrum_pairs["shuffled_pas_distance"][near])),
            },
            {
                "control": "空间打乱",
                "quantity": "3米内PDP距离",
                "true_value": float(np.mean(spectrum_pairs["pdp_distance"][near])),
                "control_value": float(np.mean(spectrum_pairs["shuffled_pdp_distance"][near])),
            },
        ]
    )
    path_near = path_pairs["distance"] <= 3.0
    shuffled = path_near & np.isfinite(path_pairs["identity_shuffle_common"])
    rows.append(
        {
            "control": "路径身份打乱",
            "quantity": "3米内共同路径能量",
            "true_value": float(np.mean(path_pairs["continuous_common"][shuffled])),
            "control_value": float(np.mean(path_pairs["identity_shuffle_common"][shuffled])),
        }
    )
    rows.extend(geometry_control_rows)
    phase_near = path_pairs["distance"] <= 6.5
    rows.extend(
        [
            {
                "control": "错误波数",
                "quantity": "6.5米内载波修正有符号相干性",
                "true_value": float(np.mean(phase_pairs["carrier_signed"][phase_near])),
                "control_value": float(np.mean(phase_pairs["wrong_beta_signed"][phase_near])),
            },
            {
                "control": "错位距离",
                "quantity": "6.5米内载波修正有符号相干性",
                "true_value": float(np.mean(phase_pairs["carrier_signed"][phase_near])),
                "control_value": float(np.mean(phase_pairs["wrong_map_signed"][phase_near])),
            },
        ]
    )
    for beta_row in beta_rows:
        rows.append(
            {
                "control": "错误波数独立留出",
                "quantity": f"基站{beta_row['site']}有符号相干性",
                "true_value": beta_row["held_corrected_signed_coherence"],
                "control_value": beta_row["held_wrong_beta_signed_coherence"],
            }
        )
    return rows


def save_pair_cache(
    left: np.ndarray,
    right: np.ndarray,
    distance: np.ndarray,
    spectrum: dict[str, np.ndarray],
    path: dict[str, np.ndarray],
    phase: dict[str, np.ndarray],
    continuous_unmatched: np.ndarray,
    discrete_unmatched: np.ndarray,
    spectral_jump: np.ndarray,
    beta_rows: list[dict[str, Any]],
) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "pair_left": left,
        "pair_right": right,
        "pair_distance": distance,
        "continuous_unmatched": continuous_unmatched,
        "discrete_unmatched": discrete_unmatched,
        "spectral_jump": spectral_jump,
        "beta_rows_json": np.asarray(json.dumps(beta_rows, ensure_ascii=False)),
    }
    payload.update({f"spectrum__{key}": value for key, value in spectrum.items()})
    payload.update({f"path__{key}": value for key, value in path.items()})
    payload.update({f"phase__{key}": value for key, value in phase.items()})
    np.savez_compressed(PAIR_CACHE, **payload)


def load_pair_cache() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
]:
    cached = np.load(PAIR_CACHE, allow_pickle=False)
    spectrum = {
        key.removeprefix("spectrum__"): cached[key]
        for key in cached.files
        if key.startswith("spectrum__")
    }
    path = {
        key.removeprefix("path__"): cached[key]
        for key in cached.files
        if key.startswith("path__")
    }
    phase = {
        key.removeprefix("phase__"): cached[key]
        for key in cached.files
        if key.startswith("phase__")
    }
    beta_rows = json.loads(str(cached["beta_rows_json"]))
    return (
        cached["pair_left"],
        cached["pair_right"],
        cached["pair_distance"],
        spectrum,
        path,
        phase,
        cached["continuous_unmatched"],
        cached["discrete_unmatched"],
        cached["spectral_jump"],
        beta_rows,
    )


def point_table_rows(
    point: PointData,
    derived: dict[str, np.ndarray],
    continuous_unmatched: np.ndarray,
    discrete_unmatched: np.ndarray,
    spectral_jump: np.ndarray,
) -> Iterable[dict[str, Any]]:
    cumulative = np.cumsum(point.component_share, axis=1)
    k90 = np.argmax(cumulative >= 0.90, axis=1) + 1
    port_entropy = -np.sum(
        derived["port_share"] * np.log(np.maximum(derived["port_share"], 1e-30)), axis=1
    ) / math.log(MP * N_UE)
    for index in range(len(point.positions)):
        yield {
            "point_id": index,
            "site": int(point.site[index]),
            "x_m": point.positions[index, 0],
            "y_m": point.positions[index, 1],
            "z_m": point.positions[index, 2],
            "bs_distance_m": point.bs_distance[index],
            "is_zero": int(not point.valid[index]),
            "total_power": point.power[index],
            "log10_power": math.log10(point.power[index]) if point.valid[index] else math.nan,
            "main_delay_bin": int(point.main_delay[index]) if point.valid[index] else -1,
            "main_h_bin": int(point.main_h[index]) if point.valid[index] else -1,
            "main_v_bin": int(point.main_v[index]) if point.valid[index] else -1,
            "main_h_equivalent_angle_deg": point.main_h_angle[index] if point.valid[index] else math.nan,
            "main_v_equivalent_angle_deg": point.main_v_angle[index] if point.valid[index] else math.nan,
            "component_top1_share": point.component_share[index, 0] if point.valid[index] else math.nan,
            "component_k90": int(k90[index]) if point.valid[index] else 0,
            "continuous_unmatched_energy": continuous_unmatched[index],
            "discrete_unmatched_energy": discrete_unmatched[index],
            "nearest_neighbor_spectral_jump": spectral_jump[index],
            "frequency_phase_delay_bin": derived["phase_delay"][index] if point.valid[index] else math.nan,
            "mimo_rank1_share": derived["rank_share"][index, 0] if point.valid[index] else math.nan,
            "mimo_rank2_cumulative_share": np.sum(derived["rank_share"][index, :2]) if point.valid[index] else math.nan,
            "port_power_entropy": port_entropy[index] if point.valid[index] else math.nan,
        }


def spectrum_pair_table_rows(
    left: np.ndarray,
    right: np.ndarray,
    distance: np.ndarray,
    spectrum: dict[str, np.ndarray],
) -> Iterable[dict[str, Any]]:
    for row in range(len(left)):
        yield {
            "left_point": int(left[row]),
            "right_point": int(right[row]),
            "distance_m": distance[row],
            **{key: values[row] for key, values in spectrum.items()},
        }


def path_pair_table_rows(
    path: dict[str, np.ndarray],
    phase: dict[str, np.ndarray],
) -> Iterable[dict[str, Any]]:
    for row in range(len(path["left"])):
        yield {
            **{key: values[row] for key, values in path.items()},
            **{key: values[row] for key, values in phase.items()},
        }


def build_summary(
    point: PointData,
    derived: dict[str, np.ndarray],
    pair_distance: np.ndarray,
    spectrum: dict[str, np.ndarray],
    path: dict[str, np.ndarray],
    phase: dict[str, np.ndarray],
    continuous_unmatched: np.ndarray,
    discrete_unmatched: np.ndarray,
    spectral_jump: np.ndarray,
    geometry_control: list[dict[str, Any]],
    beta_rows: list[dict[str, Any]],
    site_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    component_file = np.load(COMPONENT_CACHE, allow_pickle=False)
    residual = component_file["residual_fraction"].astype(np.float64)
    component_cumulative = np.cumsum(point.component_share, axis=1)
    k90 = np.argmax(component_cumulative >= 0.90, axis=1) + 1
    discrete_cumulative = derived["discrete_cumulative"]

    nearest_same = []
    for site_value in (0, 1):
        ids = np.flatnonzero(point.site == site_value)
        _, local = cKDTree(point.positions[ids, :2]).query(point.positions[ids, :2], k=2)
        nearest_same.extend((point.valid[ids] == point.valid[ids[local[:, 1]]]).tolist())
    rng = np.random.default_rng(RANDOM_SEED + 71)
    shuffled_valid = point.valid.copy()
    for site_value in (0, 1):
        ids = np.flatnonzero(point.site == site_value)
        shuffled_valid[ids] = rng.permutation(shuffled_valid[ids])
    shuffled_same = []
    for site_value in (0, 1):
        ids = np.flatnonzero(point.site == site_value)
        _, local = cKDTree(point.positions[ids, :2]).query(point.positions[ids, :2], k=2)
        shuffled_same.extend((shuffled_valid[ids] == shuffled_valid[ids[local[:, 1]]]).tolist())

    power_rho = []
    zero_rho = []
    for site_value in (0, 1):
        valid = (point.site == site_value) & point.valid
        all_site = point.site == site_value
        power_rho.append(
            float(spearmanr(point.bs_distance[valid], np.log10(point.power[valid])).statistic)
        )
        zero_rho.append(
            float(spearmanr(point.bs_distance[all_site], (~point.valid[all_site]).astype(float)).statistic)
        )

    near = pair_distance <= 3.0
    far = pair_distance >= 35.0
    path_near = path["distance"] <= 3.0
    path_far = path["distance"] >= 35.0
    identity = path_near & np.isfinite(path["identity_shuffle_common"])
    phase_near = path["distance"] <= 6.5
    phase_far = path["distance"] >= 35.0

    dominant = np.max(point.global_pdp, axis=1)
    strong = point.valid & (dominant >= np.quantile(dominant[point.valid], 0.60))
    phase_delay_error = np.abs(
        circular_delta(derived["phase_delay"], point.main_delay, S_SC)
    )
    representation_mask = np.isfinite(continuous_unmatched) & np.isfinite(discrete_unmatched)
    representation_rho = spearmanr(
        continuous_unmatched[representation_mask], discrete_unmatched[representation_mask]
    ).statistic

    geometry_sorted = sorted(
        geometry_control,
        key=lambda row: abs(float(row["true_spearman"])),
        reverse=True,
    )
    rank_cumulative = np.cumsum(derived["rank_share"], axis=1)
    summary: dict[str, Any] = {
        "scope": {
            "train_points": int(len(point.positions)),
            "nonzero_points": int(np.sum(point.valid)),
            "zero_points": int(np.sum(~point.valid)),
            "zero_by_site": [int(np.sum((point.site == site) & ~point.valid)) for site in (0, 1)],
            "pair_count": int(len(pair_distance)),
            "path_pair_count": int(len(path["distance"])),
        },
        "coverage_power": {
            "nearest_neighbor_same_zero_state_rate": float(np.mean(nearest_same)),
            "same_site_shuffle_same_zero_state_rate": float(np.mean(shuffled_same)),
            "distance_log_power_spearman_by_site": power_rho,
            "distance_zero_state_spearman_by_site": zero_rho,
            "nonzero_log_power": finite_summary(np.log10(point.power[point.valid])),
        },
        "spatial_continuity": {
            "near_definition_m": 3.0,
            "far_definition_m": 35.0,
            "near_pair_count": int(np.sum(near)),
            "far_pair_count": int(np.sum(far)),
            "near_pas_distance": float(np.mean(spectrum["pas_distance"][near])),
            "far_pas_distance": float(np.mean(spectrum["pas_distance"][far])),
            "near_pdp_distance": float(np.mean(spectrum["pdp_distance"][near])),
            "far_pdp_distance": float(np.mean(spectrum["pdp_distance"][far])),
            "near_shuffled_pas_distance": float(np.mean(spectrum["shuffled_pas_distance"][near])),
            "near_shuffled_pdp_distance": float(np.mean(spectrum["shuffled_pdp_distance"][near])),
            "near_discrete_main_consistency": float(np.mean(spectrum["discrete_main_consistency"][near])),
            "far_discrete_main_consistency": float(np.mean(spectrum["discrete_main_consistency"][far])),
            "near_continuous_main_consistency": float(np.mean(spectrum["continuous_main_consistency"][near])),
            "far_continuous_main_consistency": float(np.mean(spectrum["continuous_main_consistency"][far])),
        },
        "path_sparsity": {
            "component_reconstruction_explained_mean": float(np.mean(1.0 - residual[point.valid])),
            "component_top1_share_median": float(np.median(point.component_share[point.valid, 0])),
            "component_top4_share_median": float(np.median(component_cumulative[point.valid, 3])),
            "component_top8_share_median": float(np.median(component_cumulative[point.valid, 7])),
            "component_k90_median": float(np.median(k90[point.valid])),
            "discrete_top1_share_median": float(np.median(discrete_cumulative[point.valid, 0])),
            "discrete_top4_share_median": float(np.median(discrete_cumulative[point.valid, 3])),
            "discrete_top8_share_median": float(np.median(discrete_cumulative[point.valid, 7])),
        },
        "path_lifecycle": {
            "near_continuous_common_energy": float(np.mean(path["continuous_common"][path_near])),
            "far_continuous_common_energy": float(np.mean(path["continuous_common"][path_far])),
            "near_discrete_common_energy": float(np.mean(path["discrete_common"][path_near])),
            "far_discrete_common_energy": float(np.mean(path["discrete_common"][path_far])),
            "near_identity_shuffle_common_energy": float(np.mean(path["identity_shuffle_common"][identity])),
            "continuous_discrete_unmatched_spearman": float(representation_rho),
            "nearest_continuous_unmatched": finite_summary(continuous_unmatched[point.valid]),
            "nearest_discrete_unmatched": finite_summary(discrete_unmatched[point.valid]),
            "nearest_spectral_jump": finite_summary(spectral_jump[point.valid]),
        },
        "geometry": {
            "strongest_relations": geometry_sorted[:8],
        },
        "phase": {
            "beta_by_site": beta_rows,
            "near_raw_absolute_coherence": float(np.mean(phase["raw_abs"][phase_near])),
            "far_raw_absolute_coherence": float(np.mean(phase["raw_abs"][phase_far])),
            "near_raw_signed_coherence": float(np.mean(phase["raw_signed"][phase_near])),
            "near_carrier_signed_coherence": float(np.mean(phase["carrier_signed"][phase_near])),
            "near_carrier_delay_signed_coherence": float(np.mean(phase["carrier_delay_signed"][phase_near])),
            "near_wrong_beta_signed_coherence": float(np.mean(phase["wrong_beta_signed"][phase_near])),
            "near_wrong_map_signed_coherence": float(np.mean(phase["wrong_map_signed"][phase_near])),
            "phase_delay_circular_error_all": finite_summary(phase_delay_error[point.valid]),
            "phase_delay_circular_error_strong_pdp": finite_summary(phase_delay_error[strong]),
        },
        "mimo": {
            "rank1_share_median": float(np.median(derived["rank_share"][point.valid, 0])),
            "rank2_cumulative_median": float(np.median(rank_cumulative[point.valid, 1])),
            "rank3_cumulative_median": float(np.median(rank_cumulative[point.valid, 2])),
            "near_port_cosine": float(np.mean(phase["port_cosine"][phase_near])),
            "far_port_cosine": float(np.mean(phase["port_cosine"][phase_far])),
        },
        "site_comparison": site_rows,
    }
    return summary


def render_report(summary: dict[str, Any]) -> str:
    scope = summary["scope"]
    coverage = summary["coverage_power"]
    continuity = summary["spatial_continuity"]
    sparsity = summary["path_sparsity"]
    lifecycle = summary["path_lifecycle"]
    phase = summary["phase"]
    mimo = summary["mimo"]
    geometry = summary["geometry"]
    strongest_geometry = geometry["strongest_relations"][0]
    beta0, beta1 = phase["beta_by_site"]
    site_largest = max(summary["site_comparison"], key=lambda row: row["ks_statistic"])
    return f"""# 复赛官方原始信道传播规律分析

本报告只使用官方提供的 `{scope['train_points']}` 个训练信道、训练坐标、两基站位置和官方地图。没有使用测试真值、历史版本预测或 Sionna 辅助信道。`{scope['zero_points']}` 个全零信道点全部保留，并被视为覆盖状态，而不是坏点删除。连续路径结果来自24分量离网格分解，离散结果来自原始 angle-delay 网格；只有二者方向一致时才形成算法结论。

## 功率与全零覆盖状态

### 结果数据、图

- 全零点共 `{scope['zero_points']}/{scope['train_points']}`，基站0/1分别为 `{scope['zero_by_site'][0]}/{scope['zero_by_site'][1]}`。
- 最近同站邻点具有相同零/非零状态的比例为 `{coverage['nearest_neighbor_same_zero_state_rate']:.4f}`，同站随机打乱后为 `{coverage['same_site_shuffle_same_zero_state_rate']:.4f}`。
- 距离与非零点log功率的Spearman相关：基站0 `{coverage['distance_log_power_spearman_by_site'][0]:+.4f}`，基站1 `{coverage['distance_log_power_spearman_by_site'][1]:+.4f}`。
![图01：训练点总功率空间分布](../../visualizations/复赛原始信道传播规律/01_训练点总功率空间分布.png)
![图02：全零信道点空间聚集](../../visualizations/复赛原始信道传播规律/02_全零信道点空间聚集.png)
![图03：基站距离与总功率关系](../../visualizations/复赛原始信道传播规律/03_基站距离与总功率关系.png)
![图04：基站距离分箱与全零比例](../../visualizations/复赛原始信道传播规律/04_基站距离分箱与全零比例.png)

### 现象

全零点不是均匀散落，而是形成连续空间区域；最近邻覆盖状态一致率显著高于同站打乱控制。距离与功率有关，但两站相关强度不同，且同一距离上仍存在很宽的功率分布。

### 能说明什么问题

零信道首先是可学习的“覆盖状态”，不能作为独立坏点直接删除。总功率也不能只用路径损耗公式预测：基站距离是基础项，地图遮挡和传播区域必须共同决定剩余量。

## 局部功率、PAS与PDP连续性

### 结果数据、图

- 3米内真实PAS/PDP距离分别为 `{continuity['near_pas_distance']:.4f}/{continuity['near_pdp_distance']:.4f}`，同站空间打乱控制为 `{continuity['near_shuffled_pas_distance']:.4f}/{continuity['near_shuffled_pdp_distance']:.4f}`。
- 35米以上PAS/PDP距离分别为 `{continuity['far_pas_distance']:.4f}/{continuity['far_pdp_distance']:.4f}`。
![图05：空间距离与功率及谱差异](../../visualizations/复赛原始信道传播规律/05_空间距离与功率及谱差异.png)

### 现象

近距离点的PAS和PDP明显比远距离点相似，而且真实近邻显著优于保持站点分布不变的空间打乱控制；但距离增大后差异并非简单线性增长，说明存在区域边界和路径事件。

### 能说明什么问题

测试点之间的局部图关系是真信息源，可以约束谱随空间连续变化；但普通距离KNN只能描述平滑部分，不能处理跨墙或路径出生/消失边界。

## 路径稀疏性

### 结果数据、图

- 24分量连续分解平均解释原始H能量 `{sparsity['component_reconstruction_explained_mean']:.4%}`。
- 连续分量最强1/4/8条累计能量中位数为 `{sparsity['component_top1_share_median']:.4f}/{sparsity['component_top4_share_median']:.4f}/{sparsity['component_top8_share_median']:.4f}`，达到90%能量所需分量数中位数为 `{sparsity['component_k90_median']:.1f}`。
- 离散网格最强1/4/8峰累计能量中位数为 `{sparsity['discrete_top1_share_median']:.4f}/{sparsity['discrete_top4_share_median']:.4f}/{sparsity['discrete_top8_share_median']:.4f}`。
![图06：最强K条路径累计能量](../../visualizations/复赛原始信道传播规律/06_最强K条路径累计能量.png)

### 现象

连续离网格分量比离散单格更快集中能量，这是离网格泄漏造成的正常差异；两种表示都确认主要能量由少数路径结构承担。

### 能说明什么问题

预测完整78万多个复系数不是最自然的任务。更合适的中间表示是少量路径token：每条路径预测delay、水平/垂直角、能量、8维端口响应和相位状态。

## 主路径的空间组织

### 结果数据、图

- 3米内离散/连续主路径一致率为 `{continuity['near_discrete_main_consistency']:.4f}/{continuity['near_continuous_main_consistency']:.4f}`，35米以上为 `{continuity['far_discrete_main_consistency']:.4f}/{continuity['far_continuous_main_consistency']:.4f}`。
![图07：主PDP时延空间分布](../../visualizations/复赛原始信道传播规律/07_主PDP时延空间分布.png)
![图08：主水平与垂直PAS等效角空间分布](../../visualizations/复赛原始信道传播规律/08_主水平与垂直PAS等效角空间分布.png)
![图09：空间距离与主路径一致率](../../visualizations/复赛原始信道传播规律/09_空间距离与主路径一致率.png)

### 现象

主时延和阵列角形成连续空间片区，并在部分地图边界发生突变。离散峰与连续分量给出的距离衰减方向一致，不是某一种分解算法独有的现象。

### 能说明什么问题

模型应同时看全局场景和局部邻域：全局地图决定所在传播区，局部图负责在区内追踪主路径。只看坐标或只看最近邻都会遗漏其中一半信息。

## 路径持续、出生与消失

### 结果数据、图

- 3米内连续/离散表示的共同路径能量为 `{lifecycle['near_continuous_common_energy']:.4f}/{lifecycle['near_discrete_common_energy']:.4f}`，35米以上为 `{lifecycle['far_continuous_common_energy']:.4f}/{lifecycle['far_discrete_common_energy']:.4f}`。
- 打乱路径support与8端口身份对应后，3米内共同能量为 `{lifecycle['near_identity_shuffle_common_energy']:.4f}`。
- 最近邻连续/离散未匹配能量的Spearman相关为 `{lifecycle['continuous_discrete_unmatched_spearman']:+.4f}`。
![图10：空间距离与匹配路径解释能量](../../visualizations/复赛原始信道传播规律/10_空间距离与匹配路径解释能量.png)
![图11：匹配路径时延与角度变化速度](../../visualizations/复赛原始信道传播规律/11_匹配路径时延与角度变化速度.png)
![图12：路径出生消失未匹配能量空间分布](../../visualizations/复赛原始信道传播规律/12_路径出生消失未匹配能量空间分布.png)
![图13：空间距离与匹配路径能量变化](../../visualizations/复赛原始信道传播规律/13_空间距离与匹配路径能量变化.png)

### 现象

近邻通常共享大部分强路径，但并非全部；未匹配能量在空间上集中成边界带。离散与连续表示对“哪里容易发生路径事件”的排序呈同向相关，路径身份打乱后共同能量下降。

### 能说明什么问题

瓶颈不是单纯把已有峰平滑得更准，而是判断哪些路径在目标点存在。后续模型需要显式的birth/stable/death门控，并让门控依赖地图和周围多个点，而不是对所有点统一投影。

## 地图结构与传播事件

### 结果数据、图

- 绝对相关最强的地图关系是“{strongest_geometry['feature']} → {strongest_geometry['outcome']}”：真实Spearman `{strongest_geometry['true_spearman']:+.4f}`，同站地图错位控制 `{strongest_geometry['wrong_map_spearman']:+.4f}`。
- 完整16组关系见 [地图分箱统计.csv](地图分箱统计.csv) 与 [严格控制实验.csv](严格控制实验.csv)。
![图14：地图结构与覆盖及路径突变关系](../../visualizations/复赛原始信道传播规律/14_地图结构与覆盖及路径突变关系.png)

### 现象

墙距、直达扇区净空、传播通路平均宽度和绕行距离对四类目标的关联强弱不同；真实地图关系整体强于同站错位控制，但没有任何单一描述量解释全部变化。

### 能说明什么问题

地图特征有增量信息，但应作为路径存在概率和不确定度的条件，而不是直接把某个几何量硬映射成PAS/PDP。需要多尺度地图编码器联合学习局部墙面与BS到UE的全局通路。

## 复相位与载波规律

### 结果数据、图

- 从偶数空间对拟合、奇数空间对留出的波数为基站0 `{beta0['beta_rad_per_m']:.3f}` rad/m、基站1 `{beta1['beta_rad_per_m']:.3f}` rad/m，对应 `{beta0['equivalent_frequency_ghz']:.3f}/{beta1['equivalent_frequency_ghz']:.3f}` GHz。
- 6.5米内原始绝对相干性 `{phase['near_raw_absolute_coherence']:.4f}`；有符号相干性从未修正 `{phase['near_raw_signed_coherence']:.4f}` 变为载波修正 `{phase['near_carrier_signed_coherence']:.4f}`，再加入主时延差后为 `{phase['near_carrier_delay_signed_coherence']:.4f}`。
- 错误波数/错位距离控制分别为 `{phase['near_wrong_beta_signed_coherence']:.4f}/{phase['near_wrong_map_signed_coherence']:.4f}`。
- 强PDP主峰点的“PDP峰—频率相位斜率”循环误差中位数为 `{phase['phase_delay_circular_error_strong_pdp']['median']:.3f}` 格。
![图15：空间距离与原始复信道相干性](../../visualizations/复赛原始信道传播规律/15_空间距离与原始复信道相干性.png)
![图16：载波与路径时延修正后的相位相干性](../../visualizations/复赛原始信道传播规律/16_载波与路径时延修正后的相位相干性.png)
![图17：PDP主时延与频率相位斜率对照](../../visualizations/复赛原始信道传播规律/17_PDP主时延与频率相位斜率对照.png)

### 现象

复信道绝对形状只在短距离内保持相干；利用从官方H独立拟合的距离载波相位可恢复一部分有符号相干性，错误波数和错位距离不能得到同样效果。主时延差修正是否继续改善取决于多径是否由单一峰主导。

### 能说明什么问题

相位不是不可学，但必须按路径分解：公共距离载波只能校准一个粗相位，频率斜率由路径时延决定，剩余绝对相位仍需要路径身份或虚拟源。对低置信多径点强制相位运输会破坏NMSE。

## MIMO秩与极化、UE端口组织

### 结果数据、图

- 第一MIMO奇异模态能量中位数 `{mimo['rank1_share_median']:.4f}`，前2/3模态累计能量中位数 `{mimo['rank2_cumulative_median']:.4f}/{mimo['rank3_cumulative_median']:.4f}`。
- 6.5米内8维极化×UE功率余弦 `{mimo['near_port_cosine']:.4f}`，35米以上为 `{mimo['far_port_cosine']:.4f}`。
![图18：MIMO奇异值累计解释能量](../../visualizations/复赛原始信道传播规律/18_MIMO奇异值累计解释能量.png)
![图19：极化与UE功率组织空间连续性](../../visualizations/复赛原始信道传播规律/19_极化与UE功率组织空间连续性.png)

### 现象

256×4信道在每个子载波上具有明显低秩结构，极化×UE能量分配在近邻位置也比远距离更连续，但其连续性并不等于逐链路复相位连续。

### 能说明什么问题

端口维不应独立预测。更合适的是共享路径token后，用低秩端口响应头生成4个UE端口和两极化的幅相；这能减少样本需求，同时保留逐链路差异。

## 两基站规律能否共享

### 结果数据、图

- 两站差异最大的描述量为“{site_largest['descriptor']}”，KS统计量 `{site_largest['ks_statistic']:.4f}`。
- 完整数值见 [双基站分布对照.csv](双基站分布对照.csv)。
![图20：双基站传播描述量分位数对照](../../visualizations/复赛原始信道传播规律/20_双基站传播描述量分位数对照.png)

### 现象

两站共享“局部连续、少量强路径、低秩MIMO”这些机制，但功率、路径复杂度和突变风险的具体分布并不完全相同。

### 能说明什么问题

模型主体可以共享，归一化、覆盖头、功率头和路径门控需要基站条件化；直接把两站混成一个无站点标识的回归问题会引入系统偏差。

## 综合结论

### 结果数据、图

所有点级、空间对、路径配对、分箱和控制数据分别保存在 [点级统计.csv](点级统计.csv)、[空间对谱统计.csv](空间对谱统计.csv)、[路径配对统计.csv](路径配对统计.csv)、[距离分箱统计.csv](距离分箱统计.csv) 和 [严格控制实验.csv](严格控制实验.csv)。

### 现象

官方数据同时表现出五个稳定事实：覆盖状态空间聚集、谱在传播区内连续、能量集中于少量路径、路径会在地图边界出生/消失、MIMO端口响应具有低秩共享结构。相位存在可恢复的物理规律，但有效范围和置信度明显小于功率谱规律。

### 能说明什么问题

下一阶段最有依据的模型不是继续调统一PAS/PDP投影强度，而是“全局地图编码 + 测试点局部图 + 稀疏路径token + birth/stable/death门控 + 低秩端口响应 + 置信度相位头”。训练目标应同时监督零状态、总功率、delay-angle路径集合、PAS/PDP和复信道NMSE，并对连续区与边界区使用不同损失权重。
"""


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    configure_plotting()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"分析设备: {device}", flush=True)

    setup, _, _, _, _ = load_positions()
    point, derived = build_point_data(device, args.batch_size, args.force)
    mesh = trimesh.load(PATHS.round2 / "Round2_Map.ply", process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh, got {type(mesh).__name__}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)

    if PAIR_CACHE.is_file() and not args.force:
        (
            pair_left,
            pair_right,
            pair_distance,
            spectrum,
            path,
            phase,
            continuous_unmatched,
            discrete_unmatched,
            spectral_jump,
            beta_rows,
        ) = load_pair_cache()
        print(f"复用空间对缓存: {PATHS.relative(PAIR_CACHE)}", flush=True)
    else:
        pair_left, pair_right, pair_distance = build_stratified_pairs(point)
        print(f"分层空间对数量: {len(pair_distance)}", flush=True)
        spectrum = spectrum_pair_metrics(point, pair_left, pair_right)
        config = Config(components=PATH_COUNT, transported_components=PATH_COUNT)
        continuous_unmatched, discrete_unmatched, spectral_jump = nearest_path_discontinuity(
            point, config
        )
        path = compute_path_pair_metrics(
            point,
            pair_left,
            pair_right,
            pair_distance,
            args.path_pair_limit,
        )
        phase, beta_rows = compute_phase_pair_metrics(
            point, derived, path, device, args.batch_size
        )
        save_pair_cache(
            pair_left,
            pair_right,
            pair_distance,
            spectrum,
            path,
            phase,
            continuous_unmatched,
            discrete_unmatched,
            spectral_jump,
            beta_rows,
        )

    geometry_control, geometry_features = geometry_controls(
        point, continuous_unmatched, spectral_jump
    )
    geometry_rows = geometry_binned_rows(
        point, geometry_features, continuous_unmatched, spectral_jump
    )

    plot_figures_01_to_04(point, setup, vertices)
    plot_figures_05_to_06(point, derived, pair_distance, spectrum)
    plot_figures_07_to_09(point, setup, vertices, pair_distance, spectrum)
    plot_figures_10_to_13(
        point, setup, vertices, path, continuous_unmatched, discrete_unmatched
    )
    plot_figure_14(geometry_rows)
    plot_figures_15_to_17(point, derived, path, phase)
    site_rows = plot_figures_18_to_20(
        point, derived, path, phase, continuous_unmatched
    )

    distance_rows = distance_statistics_rows(
        pair_distance, spectrum, path, phase
    )
    controls = control_rows(
        pair_distance,
        spectrum,
        path,
        phase,
        geometry_control,
        beta_rows,
    )
    write_csv(
        RESULT_DIR / "点级统计.csv",
        point_table_rows(
            point,
            derived,
            continuous_unmatched,
            discrete_unmatched,
            spectral_jump,
        ),
    )
    write_csv(
        RESULT_DIR / "空间对谱统计.csv",
        spectrum_pair_table_rows(pair_left, pair_right, pair_distance, spectrum),
    )
    write_csv(RESULT_DIR / "路径配对统计.csv", path_pair_table_rows(path, phase))
    write_csv(RESULT_DIR / "距离分箱统计.csv", distance_rows)
    write_csv(RESULT_DIR / "地图分箱统计.csv", geometry_rows)
    write_csv(RESULT_DIR / "严格控制实验.csv", controls)
    write_csv(RESULT_DIR / "双基站分布对照.csv", site_rows)
    write_csv(RESULT_DIR / "载波波数拟合.csv", beta_rows)

    summary = build_summary(
        point,
        derived,
        pair_distance,
        spectrum,
        path,
        phase,
        continuous_unmatched,
        discrete_unmatched,
        spectral_jump,
        geometry_control,
        beta_rows,
        site_rows,
    )
    (RESULT_DIR / "原始信道传播规律统计.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULT_DIR / "原始信道传播规律分析.md").write_text(
        render_report(summary), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": PATHS.relative(RESULT_DIR / "原始信道传播规律分析.md"),
                "figures": len(list(FIGURE_DIR.glob("*.png"))),
                "zero_points": summary["scope"]["zero_points"],
                "near_pas_distance": summary["spatial_continuity"]["near_pas_distance"],
                "near_path_common": summary["path_lifecycle"]["near_continuous_common_energy"],
                "near_carrier_signed": summary["phase"]["near_carrier_signed_coherence"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
