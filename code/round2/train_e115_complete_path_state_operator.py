#!/usr/bin/env python3
"""Train a complete path-state operator from official H and map observables."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F

import analyze_global_local_path_lifecycle as e79
import train_path_conditioned_scene_gnn as e83
from common.official_metrics import official_cosine
from data import load_positions
from project_paths import PATHS
from run_path_component_transport import _component_atom, component_energy
from run_path_track_spectral_field import valid_components
from train_surface_interaction_energy_model import (
    TARGET_PAS,
    TARGET_PDP,
    normalized_direction,
    render_spectra,
)


VERSION = "Round2-E115-complete-path-state-operator"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "complete_path_state_operator"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_complete_path_state_operator"
SCREEN_REPORT = ANALYSIS_DIR / "complete_path_state_outer0_screen.json"

COMPONENTS = (
    PATHS.artifacts
    / "round2"
    / "v4_path_component_transport"
    / "train_components_k24_os2_ref2.npz"
)
FULL_FOLD = (
    PATHS.artifacts / "round2" / "full_path_pdp_link_gate" / "train_fold_id.npy"
)
GEODESIC_CACHE = (
    PATHS.artifacts
    / "round2"
    / "regime_lifecycle_support"
    / "geodesic_corridor_features.npz"
)
RASTER_CACHE = (
    PATHS.artifacts
    / "round2"
    / "offline_sionna_geometry_student"
    / "official_map_geometry_rasters.npz"
)
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"

MP = 2
MH = 16
MV = 8
N_UE = 4
S_SC = 192
PORTS = MP * N_UE
PERIODS = np.asarray((MH, MV, S_SC), dtype=np.float32)
TARGET_PATHS = 24
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    donor_count: int = 32
    donor_components: int = 20
    candidate_count: int = 176
    graph_neighbors: int = 4
    minimum_fraction: float = 1e-5
    spatial_scale_m: float = 12.0
    beta: float = 140.30
    soft_cost_limit: float = 0.25
    soft_bandwidth: float = 0.25
    hidden: int = 128
    heads: int = 8
    candidate_layers: int = 3
    graph_layers: int = 1
    dropout: float = 0.08
    epochs: int = 40
    minimum_epochs: int = 12
    validation_interval: int = 2
    batch_size: int = 12
    learning_rate: float = 5e-4
    weight_decay: float = 2e-4
    score_limit: float = 4.0
    coordinate_limits: tuple[float, float, float] = (1.5, 1.0, 3.0)
    port_limit: float = 2.0
    phase_limit: float = math.pi
    spectral_weight: float = 1.0
    listwise_weight: float = 0.45
    mass_hellinger_weight: float = 0.25
    candidate_bce_weight: float = 0.04
    coordinate_weight: float = 0.16
    port_weight: float = 0.20
    phase_weight: float = 0.04
    zero_weight: float = 0.10
    power_weight: float = 0.04
    residual_weight: float = 0.003
    random_state: int = 20260808115


@dataclass(frozen=True)
class Variant:
    name: str
    use_scene: bool
    use_graph: bool


VARIANTS = (
    Variant("candidate_only", False, False),
    Variant("candidate_scene", True, False),
    Variant("complete", True, True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare-screen", "screen"), default="prepare-screen"
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def cache_dir(tag: str) -> Path:
    return ARTIFACT_DIR / tag


def array_path(directory: Path, name: str) -> Path:
    return directory / f"{name}.npy"


def safe_unit_complex(value: np.ndarray) -> np.ndarray:
    magnitude = np.abs(value)
    return np.divide(
        value,
        np.maximum(magnitude, 1e-12),
        out=np.ones_like(value, dtype=np.complex64),
    ).astype(np.complex64)


def normalize_rows(value: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    return np.divide(
        output,
        np.maximum(output.sum(axis=1, keepdims=True), TINY),
        out=np.zeros_like(output),
    ).astype(np.float32)


def soft_path_targets(
    candidate_coordinate: np.ndarray,
    truth_coordinate: np.ndarray,
    truth_coefficient: np.ndarray,
    point_power: float,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(candidate_coordinate)
    mass = np.zeros(count, dtype=np.float32)
    delta = np.zeros((count, 3), dtype=np.float32)
    port = np.full((count, PORTS), 1.0 / PORTS, dtype=np.float32)
    phase = np.zeros((count, 2 * PORTS), dtype=np.float32)
    phase[:, :PORTS] = 1.0
    if count == 0 or point_power <= TINY:
        return mass, delta, port, phase

    truth_energy = np.sum(np.abs(truth_coefficient) ** 2, axis=1, dtype=np.float64)
    truth_fraction = truth_energy / max(float(point_power), TINY)
    valid = truth_fraction >= config.minimum_fraction
    if not np.any(valid):
        return mass, delta, port, phase
    truth_coordinate = truth_coordinate[valid]
    truth_coefficient = truth_coefficient[valid]
    truth_fraction = truth_fraction[valid]

    cost = e79.support_cost(candidate_coordinate, truth_coordinate)
    kernel = np.exp(-0.5 * cost / config.soft_bandwidth)
    kernel[cost > config.soft_cost_limit] = 0.0
    kernel /= np.maximum(kernel.sum(axis=0, keepdims=True), TINY)
    contribution = kernel * truth_fraction[None]
    mass = contribution.sum(axis=1).astype(np.float32)
    responsibility = np.divide(
        contribution,
        np.maximum(contribution.sum(axis=1, keepdims=True), TINY),
        out=np.zeros_like(contribution),
    )
    coordinate_delta = np.stack(
        tuple(
            e79.periodic_delta(
                truth_coordinate[None, :, axis],
                candidate_coordinate[:, None, axis],
                float(PERIODS[axis]),
            )
            for axis in range(3)
        ),
        axis=2,
    )
    delta = np.einsum("ct,ctd->cd", responsibility, coordinate_delta).astype(
        np.float32
    )
    truth_port = normalize_rows(np.abs(truth_coefficient) ** 2)
    local_port = responsibility @ truth_port
    truth_phase = safe_unit_complex(truth_coefficient)
    local_phase = responsibility @ truth_phase
    local_phase = safe_unit_complex(local_phase.astype(np.complex64))
    positive = mass > 0.0
    port[positive] = local_port[positive]
    phase[positive] = np.concatenate(
        (local_phase[positive].real, local_phase[positive].imag), axis=1
    ).astype(np.float32)
    return mass, delta, port, phase


def context_for_group(
    query_ids: np.ndarray,
    bank: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    point_power: np.ndarray,
    geodesic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(query_ids)
    output = np.zeros((count, e79.POINT_CONTEXT_DIM + geodesic.shape[1]), np.float32)
    base_log_power = np.zeros(count, np.float32)
    take = min(64, len(bank))
    distance, local = cKDTree(positions[bank, :2]).query(
        positions[query_ids, :2], k=take, workers=-1
    )
    distance = np.asarray(distance)
    local = np.asarray(local)
    if take == 1:
        distance = distance[:, None]
        local = local[:, None]
    donors = bank[local.astype(np.int64)]
    for row, point_id in enumerate(query_ids):
        local_distance = np.asarray(distance[row], dtype=np.float64)
        local_donors = donors[row]
        valid = point_power[local_donors] > TINY
        log10_power = np.log10(np.maximum(point_power[local_donors], 1e-30))
        natural_log_power = np.log(np.maximum(point_power[local_donors], 1e-30))
        weight = np.exp(-local_distance / 12.0)
        weight /= max(float(np.sum(weight)), TINY)
        positive_weight = weight * valid
        positive_total = max(float(np.sum(positive_weight)), TINY)
        power_mean = float(np.sum(positive_weight * log10_power) / positive_total)
        power_std = float(
            np.sqrt(
                np.sum(positive_weight * np.square(log10_power - power_mean))
                / positive_total
            )
        )
        base_log_power[row] = np.float32(
            np.sum(positive_weight * natural_log_power) / positive_total
        )
        relative = positions[point_id, :2] - bs_positions[int(site[point_id]), :2]
        radial = max(float(np.linalg.norm(relative)), 1e-9)
        angle = math.atan2(float(relative[1]), float(relative[0]))
        offsets = positions[local_donors, :2] - positions[point_id, :2]
        covariance = offsets.T @ (weight[:, None] * offsets)
        eigenvalues = np.linalg.eigvalsh(covariance)
        isotropy = float(eigenvalues[0] / max(eigenvalues[-1], 1e-9))
        quadrant = (
            (offsets[:, 0] >= 0).astype(np.int8)
            + 2 * (offsets[:, 1] >= 0).astype(np.int8)
        )
        coverage = float(len(np.unique(quadrant))) / 4.0
        quantiles = np.quantile(local_distance, (0.0, 0.25, 0.50, 0.90)) / 30.0
        zero_rates = [
            float(np.mean(~valid[: min(limit, len(valid))]))
            for limit in (8, 24, 64)
        ]
        local_context = np.asarray(
            [
                relative[0] / 150.0,
                relative[1] / 150.0,
                radial / 200.0,
                math.sin(angle),
                math.cos(angle),
                float(site[point_id] == 0),
                float(site[point_id] == 1),
                *quantiles.tolist(),
                *zero_rates,
                power_mean / 20.0,
                power_std / 10.0,
                isotropy,
                coverage,
            ],
            dtype=np.float32,
        )
        map_context = np.sign(geodesic[point_id]) * np.log1p(
            np.abs(geodesic[point_id])
        )
        output[row] = np.concatenate((local_context, map_context)).astype(np.float32)
    return output, base_log_power


def candidate_features(
    query_ids: np.ndarray,
    donor_ids: np.ndarray,
    component_ids: np.ndarray,
    donor_distance: np.ndarray,
    token_score: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    coordinate: np.ndarray,
    coefficient: np.ndarray,
    fraction: np.ndarray,
    point_power: np.ndarray,
    residual_fraction: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    query = positions[query_ids]
    donor = positions[donor_ids]
    source = bs_positions[site[query_ids]]
    query_range = np.linalg.norm(query - source, axis=1)
    donor_range = np.linalg.norm(donor - source[:, None], axis=2)
    selected_coordinate = coordinate[donor_ids, component_ids]
    selected_coefficient = coefficient[donor_ids, component_ids]
    path_energy = np.sum(np.abs(selected_coefficient) ** 2, axis=2, dtype=np.float64)
    base_port = np.divide(
        np.abs(selected_coefficient) ** 2,
        np.maximum(path_energy[..., None], TINY),
    ).astype(np.float32)
    radial_phase = np.exp(
        1j * config.beta * (donor_range - query_range[:, None])
    ).astype(np.complex64)
    transported = selected_coefficient * radial_phase[..., None]
    base_phase_complex = safe_unit_complex(transported)
    base_phase = np.concatenate(
        (base_phase_complex.real, base_phase_complex.imag), axis=2
    ).astype(np.float32)

    period_phase = 2.0 * np.pi * selected_coordinate / PERIODS
    coordinate_cycle = np.concatenate((np.sin(period_phase), np.cos(period_phase)), axis=2)
    donor_offset = donor[:, :, :2] - query[:, None, :2]
    donor_angle = np.arctan2(donor_offset[..., 1], donor_offset[..., 0])
    query_relative = query[:, :2] - source[:, :2]
    query_angle = np.arctan2(query_relative[:, 1], query_relative[:, 0])
    donor_rank = np.broadcast_to(
        np.arange(config.candidate_count, dtype=np.float32)[None], donor_ids.shape
    )
    selected_fraction = fraction[donor_ids, component_ids]
    feature = np.concatenate(
        (
            coordinate_cycle,
            selected_coordinate / PERIODS,
            donor_offset / 30.0,
            donor_distance[..., None] / 30.0,
            np.sin(donor_angle)[..., None],
            np.cos(donor_angle)[..., None],
            ((donor_range - query_range[:, None]) / 30.0)[..., None],
            (donor_range / 200.0)[..., None],
            np.broadcast_to((query_range / 200.0)[:, None, None], (*donor_ids.shape, 1)),
            np.log(np.maximum(selected_fraction, 1e-12))[..., None],
            np.sqrt(np.maximum(selected_fraction, 0.0))[..., None],
            np.log(np.maximum(token_score, 1e-12))[..., None],
            (donor_rank / max(config.candidate_count - 1, 1))[..., None],
            (component_ids.astype(np.float32) / max(config.donor_components - 1, 1))[
                ..., None
            ],
            np.log(np.maximum(point_power[donor_ids], 1e-30))[..., None],
            residual_fraction[donor_ids][..., None],
            np.sqrt(np.maximum(base_port, 0.0)),
            base_phase,
            np.broadcast_to(np.sin(query_angle)[:, None, None], (*donor_ids.shape, 1)),
            np.broadcast_to(np.cos(query_angle)[:, None, None], (*donor_ids.shape, 1)),
            np.broadcast_to((site[query_ids] == 0)[:, None, None], (*donor_ids.shape, 1)),
            np.broadcast_to((site[query_ids] == 1)[:, None, None], (*donor_ids.shape, 1)),
        ),
        axis=2,
    ).astype(np.float32)
    return feature, selected_coordinate, base_port, base_phase


def build_graph_features(
    directory: Path,
    positions: np.ndarray,
    site: np.ndarray,
    fold_id: np.ndarray,
    config: Config,
) -> dict[str, object]:
    coordinate = np.load(array_path(directory, "coordinate"), mmap_mode="r")
    base_log_mass = np.load(array_path(directory, "base_log_mass"), mmap_mode="r")
    mask = np.load(array_path(directory, "mask"), mmap_mode="r")
    feature = np.load(array_path(directory, "candidate_feature"), mmap_mode="r")
    output = np.lib.format.open_memmap(
        array_path(directory, "graph_feature"),
        mode="w+",
        dtype=np.float16,
        shape=(len(positions), config.candidate_count, config.graph_neighbors, 10),
    )
    neighbor_ids = np.full(
        (len(positions), config.graph_neighbors), -1, dtype=np.int32
    )
    for fold in range(5):
        for site_value in np.unique(site):
            rows = np.flatnonzero((fold_id == fold) & (site == site_value))
            if len(rows) <= 1:
                continue
            take = min(config.graph_neighbors + 1, len(rows))
            distance, local = cKDTree(positions[rows, :2]).query(
                positions[rows, :2], k=take, workers=-1
            )
            distance = np.asarray(distance)
            local = np.asarray(local)
            if take == 1:
                distance = distance[:, None]
                local = local[:, None]
            for local_row, query_id in enumerate(rows):
                candidates = rows[local[local_row, 1:take].astype(np.int64)]
                if len(candidates) < config.graph_neighbors:
                    candidates = np.pad(
                        candidates,
                        (0, config.graph_neighbors - len(candidates)),
                        mode="edge",
                    )
                neighbor_ids[query_id] = candidates[: config.graph_neighbors]
                query_mask = np.asarray(mask[query_id], dtype=bool)
                query_coordinate = np.asarray(coordinate[query_id], dtype=np.float32)
                for graph_rank, neighbor_id in enumerate(neighbor_ids[query_id]):
                    neighbor_mask = np.asarray(mask[neighbor_id], dtype=bool)
                    local_feature = np.zeros((config.candidate_count, 10), np.float32)
                    if np.any(query_mask) and np.any(neighbor_mask):
                        query_valid = np.flatnonzero(query_mask)
                        neighbor_valid = np.flatnonzero(neighbor_mask)
                        cost = e79.support_cost(
                            query_coordinate[query_valid],
                            np.asarray(coordinate[neighbor_id, neighbor_valid], np.float32),
                        )
                        best_column = np.argmin(cost, axis=1)
                        matched = neighbor_valid[best_column]
                        best_cost = cost[np.arange(len(query_valid)), best_column]
                        delta = np.stack(
                            tuple(
                                e79.periodic_delta(
                                    np.asarray(coordinate[neighbor_id, matched, axis]),
                                    query_coordinate[query_valid, axis],
                                    float(PERIODS[axis]),
                                )
                                for axis in range(3)
                            ),
                            axis=1,
                        )
                        offset = positions[neighbor_id, :2] - positions[query_id, :2]
                        separation = float(np.linalg.norm(offset))
                        local_feature[query_valid, :2] = offset / 30.0
                        local_feature[query_valid, 2] = separation / 30.0
                        local_feature[query_valid, 3:6] = delta / np.asarray(
                            (1.5, 1.0, 2.0), dtype=np.float32
                        )
                        local_feature[query_valid, 6] = np.log1p(best_cost) / 4.0
                        local_feature[query_valid, 7] = (
                            np.asarray(base_log_mass[neighbor_id, matched], np.float32)
                            / 20.0
                        )
                        local_feature[query_valid, 8] = np.asarray(
                            feature[neighbor_id, matched, 11], np.float32
                        )
                        local_feature[query_valid, 9] = 1.0
                    output[query_id, :, graph_rank] = local_feature.astype(np.float16)
                if (local_row + 1) % 250 == 0:
                    print(
                        f"graph fold={fold} site={site_value} "
                        f"{local_row + 1}/{len(rows)}",
                        flush=True,
                    )
    output.flush()
    np.save(array_path(directory, "graph_neighbor_ids"), neighbor_ids)
    return {
        "graph_feature_shape": list(output.shape),
        "graph_neighbor_shape": list(neighbor_ids.shape),
    }


def build_cache(
    tag: str,
    globally_blocked_folds: tuple[int, ...],
    config: Config,
    force: bool,
) -> dict[str, object]:
    directory = cache_dir(tag)
    state_path = directory / "state.json"
    if state_path.is_file() and not force:
        return json.loads(state_path.read_text(encoding="utf-8"))
    directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    setup, positions, _, site, _ = load_positions()
    fold_id = np.load(FULL_FOLD).astype(np.int8)
    component = np.load(COMPONENTS, allow_pickle=False)
    coordinate = component["coordinates"].astype(np.float32)
    coefficient = component["coefficients"].astype(np.complex64)
    point_power = component["point_power"].astype(np.float64)
    residual_fraction = component["residual_fraction"].astype(np.float32)
    valid, fraction = valid_components(
        component_energy(coefficient),
        point_power,
        config.donor_components,
        config.minimum_fraction,
    )
    geodesic = np.load(GEODESIC_CACHE, allow_pickle=False)["train_features"].astype(
        np.float32
    )

    probe_query = np.asarray([0], dtype=np.int64)
    probe_donor = np.zeros((1, config.candidate_count), dtype=np.int64)
    probe_component = np.zeros_like(probe_donor)
    probe_distance = np.ones_like(probe_donor, dtype=np.float32)
    probe_score = np.ones_like(probe_distance)
    probe_feature = candidate_features(
        probe_query,
        probe_donor,
        probe_component,
        probe_distance,
        probe_score,
        positions,
        site,
        setup.bs_positions,
        coordinate,
        coefficient,
        fraction,
        point_power,
        residual_fraction,
        config,
    )[0]
    feature_dim = int(probe_feature.shape[-1])
    arrays = {
        "candidate_feature": np.lib.format.open_memmap(
            array_path(directory, "candidate_feature"),
            mode="w+",
            dtype=np.float16,
            shape=(len(positions), config.candidate_count, feature_dim),
        ),
        "coordinate": np.lib.format.open_memmap(
            array_path(directory, "coordinate"),
            mode="w+",
            dtype=np.float16,
            shape=(len(positions), config.candidate_count, 3),
        ),
        "base_log_mass": np.lib.format.open_memmap(
            array_path(directory, "base_log_mass"),
            mode="w+",
            dtype=np.float16,
            shape=(len(positions), config.candidate_count),
        ),
        "base_log_port": np.lib.format.open_memmap(
            array_path(directory, "base_log_port"),
            mode="w+",
            dtype=np.float16,
            shape=(len(positions), config.candidate_count, PORTS),
        ),
        "base_phase": np.lib.format.open_memmap(
            array_path(directory, "base_phase"),
            mode="w+",
            dtype=np.float16,
            shape=(len(positions), config.candidate_count, 2 * PORTS),
        ),
        "target_mass": np.lib.format.open_memmap(
            array_path(directory, "target_mass"),
            mode="w+",
            dtype=np.float16,
            shape=(len(positions), config.candidate_count),
        ),
        "target_delta": np.lib.format.open_memmap(
            array_path(directory, "target_delta"),
            mode="w+",
            dtype=np.float16,
            shape=(len(positions), config.candidate_count, 3),
        ),
        "target_port": np.lib.format.open_memmap(
            array_path(directory, "target_port"),
            mode="w+",
            dtype=np.float16,
            shape=(len(positions), config.candidate_count, PORTS),
        ),
        "target_phase": np.lib.format.open_memmap(
            array_path(directory, "target_phase"),
            mode="w+",
            dtype=np.float16,
            shape=(len(positions), config.candidate_count, 2 * PORTS),
        ),
        "mask": np.lib.format.open_memmap(
            array_path(directory, "mask"),
            mode="w+",
            dtype=bool,
            shape=(len(positions), config.candidate_count),
        ),
        "context": np.lib.format.open_memmap(
            array_path(directory, "context"),
            mode="w+",
            dtype=np.float32,
            shape=(len(positions), e79.POINT_CONTEXT_DIM + geodesic.shape[1]),
        ),
        "base_log_power": np.lib.format.open_memmap(
            array_path(directory, "base_log_power"),
            mode="w+",
            dtype=np.float32,
            shape=(len(positions),),
        ),
    }

    blocked = np.asarray(globally_blocked_folds, dtype=np.int8)
    candidate_count_values: list[int] = []
    for query_fold in range(5):
        excluded = np.unique(np.concatenate((blocked, [query_fold]))).astype(np.int8)
        for site_value in np.unique(site):
            query_ids = np.flatnonzero((fold_id == query_fold) & (site == site_value))
            if len(query_ids) == 0:
                continue
            all_bank = np.flatnonzero(
                (site == site_value) & ~np.isin(fold_id, excluded)
            )
            nonzero_bank = all_bank[point_power[all_bank] > TINY]
            take = min(config.donor_count, len(nonzero_bank))
            distance, local = cKDTree(positions[nonzero_bank, :2]).query(
                positions[query_ids, :2], k=take, workers=-1
            )
            distance = np.asarray(distance)
            local = np.asarray(local)
            if take == 1:
                distance = distance[:, None]
                local = local[:, None]
            donors = nonzero_bank[local.astype(np.int64)]
            point_weight = np.exp(
                -0.5 * np.square(distance / config.spatial_scale_m)
            ) / np.sqrt(np.maximum(distance, 1.0))
            point_weight /= np.maximum(point_weight.sum(axis=1, keepdims=True), TINY)
            donor_table = np.repeat(donors, config.donor_components, axis=1)
            component_table = np.broadcast_to(
                np.arange(config.donor_components, dtype=np.int64)[None, None],
                (len(query_ids), take, config.donor_components),
            ).reshape(len(query_ids), -1)
            token_weight = np.repeat(point_weight, config.donor_components, axis=1)
            token_valid = valid[donor_table, component_table]
            score = fraction[donor_table, component_table] * np.sqrt(
                np.maximum(token_weight, 1e-12)
            )
            score[~token_valid] = -np.inf
            available = token_valid.sum(axis=1)
            if np.min(available) < config.candidate_count:
                raise RuntimeError(
                    f"insufficient candidates fold={query_fold} site={site_value}: "
                    f"minimum={int(np.min(available))}"
                )
            selected = np.argpartition(
                -score, config.candidate_count - 1, axis=1
            )[:, : config.candidate_count]
            selected_score = np.take_along_axis(score, selected, axis=1)
            order = np.argsort(-selected_score, axis=1, kind="stable")
            selected = np.take_along_axis(selected, order, axis=1)
            selected_score = np.take_along_axis(selected_score, order, axis=1)
            selected_donor = np.take_along_axis(donor_table, selected, axis=1)
            selected_component = np.take_along_axis(component_table, selected, axis=1)
            donor_rank = selected // config.donor_components
            selected_distance = np.take_along_axis(distance, donor_rank, axis=1)

            feature, selected_coordinate, base_port, base_phase = candidate_features(
                query_ids,
                selected_donor,
                selected_component,
                selected_distance.astype(np.float32),
                selected_score.astype(np.float32),
                positions,
                site,
                setup.bs_positions,
                coordinate,
                coefficient,
                fraction,
                point_power,
                residual_fraction,
                config,
            )
            arrays["candidate_feature"][query_ids] = feature.astype(np.float16)
            arrays["coordinate"][query_ids] = selected_coordinate.astype(np.float16)
            arrays["base_log_mass"][query_ids] = np.log(
                np.maximum(selected_score, 1e-12)
            ).astype(np.float16)
            arrays["base_log_port"][query_ids] = np.log(
                np.maximum(base_port, 1e-8)
            ).astype(np.float16)
            arrays["base_phase"][query_ids] = base_phase.astype(np.float16)
            arrays["mask"][query_ids] = True
            context, base_power = context_for_group(
                query_ids,
                all_bank,
                positions,
                site,
                setup.bs_positions,
                point_power,
                geodesic,
            )
            arrays["context"][query_ids] = context
            arrays["base_log_power"][query_ids] = base_power
            for local_row, point_id in enumerate(query_ids):
                target = soft_path_targets(
                    selected_coordinate[local_row],
                    coordinate[point_id, :TARGET_PATHS],
                    coefficient[point_id, :TARGET_PATHS],
                    float(point_power[point_id]),
                    config,
                )
                arrays["target_mass"][point_id] = target[0].astype(np.float16)
                arrays["target_delta"][point_id] = target[1].astype(np.float16)
                arrays["target_port"][point_id] = target[2].astype(np.float16)
                arrays["target_phase"][point_id] = target[3].astype(np.float16)
            candidate_count_values.extend(available.tolist())
            print(
                f"cache={tag} query_fold={query_fold} site={site_value} "
                f"queries={len(query_ids)} bank={len(all_bank)}",
                flush=True,
            )

    for value in arrays.values():
        value.flush()
    np.save(array_path(directory, "fold_id"), fold_id)
    np.save(array_path(directory, "point_power"), point_power)
    graph = build_graph_features(directory, positions, site, fold_id, config)
    state = {
        "experiment": VERSION,
        "tag": tag,
        "globally_blocked_folds": list(globally_blocked_folds),
        "config": asdict(config),
        "candidate_feature_dim": feature_dim,
        "context_dim": int(arrays["context"].shape[1]),
        "candidate_available_min": int(np.min(candidate_count_values)),
        "candidate_available_mean": float(np.mean(candidate_count_values)),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
        **graph,
        "seconds": float(time.time() - started),
    }
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return state


def convolutional_scene_branch(
    input_channels: int,
    hidden: int,
    output_size: tuple[int, int],
) -> nn.Sequential:
    middle = hidden // 2
    return nn.Sequential(
        nn.Conv2d(input_channels, middle, 3, padding=1),
        nn.GroupNorm(8, middle),
        nn.GELU(),
        nn.Conv2d(middle, hidden, 3, stride=2, padding=1),
        nn.GroupNorm(8, hidden),
        nn.GELU(),
        nn.Conv2d(hidden, hidden, 3, padding=1),
        nn.GroupNorm(8, hidden),
        nn.GELU(),
        nn.AdaptiveAvgPool2d(output_size),
    )


class SceneEncoder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.route = convolutional_scene_branch(11, hidden, (4, 12))
        self.panorama = convolutional_scene_branch(9, hidden, (2, 16))
        self.polar = convolutional_scene_branch(4, hidden, (4, 16))
        self.corridor = convolutional_scene_branch(4, hidden, (4, 16))
        self.type_embedding = nn.Parameter(torch.zeros(4, hidden))
        nn.init.normal_(self.type_embedding, std=0.02)

    @staticmethod
    def tokens(value: torch.Tensor) -> torch.Tensor:
        return value.flatten(2).transpose(1, 2)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        blocks = (
            self.tokens(self.route(batch["route"])),
            self.tokens(self.panorama(batch["panorama"])),
            self.tokens(self.polar(batch["polar"])),
            self.tokens(self.corridor(batch["corridor"])),
        )
        return torch.cat(
            tuple(value + self.type_embedding[index] for index, value in enumerate(blocks)),
            dim=1,
        )


class CompletePathStateOperator(nn.Module):
    def __init__(
        self,
        candidate_dim: int,
        context_dim: int,
        graph_dim: int,
        config: Config,
        variant: Variant,
    ) -> None:
        super().__init__()
        self.config = config
        self.variant = variant
        hidden = config.hidden
        self.candidate_input = nn.Sequential(
            nn.Linear(candidate_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        self.context_input = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        self.graph_input = nn.Sequential(
            nn.Linear(graph_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        graph_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.heads,
            dim_feedforward=2 * hidden,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.graph_encoder = nn.TransformerEncoder(
            graph_layer, num_layers=config.graph_layers
        )
        self.graph_fusion = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.scene_encoder = SceneEncoder(hidden)
        self.scene_attention = nn.MultiheadAttention(
            hidden,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.scene_norm = nn.LayerNorm(hidden)
        candidate_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.heads,
            dim_feedforward=3 * hidden,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_encoder = nn.TransformerEncoder(
            candidate_layer, num_layers=config.candidate_layers
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.score_head = nn.Linear(hidden, 1)
        self.coordinate_head = nn.Linear(hidden, 3)
        self.port_head = nn.Linear(hidden, PORTS)
        self.phase_head = nn.Linear(hidden, PORTS)
        self.zero_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.power_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        for layer in (
            self.score_head,
            self.coordinate_head,
            self.port_head,
            self.phase_head,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.zero_head[-1].weight)
        nn.init.zeros_(self.zero_head[-1].bias)
        nn.init.zeros_(self.power_head[-1].weight)
        nn.init.zeros_(self.power_head[-1].bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mask = batch["mask"]
        hidden = self.candidate_input(batch["candidate_feature"])
        context = self.context_input(batch["context"])
        hidden = hidden + context[:, None]
        if self.variant.use_graph:
            batch_size, candidates, neighbors, _ = batch["graph_feature"].shape
            graph = self.graph_input(batch["graph_feature"])
            graph = graph.reshape(
                batch_size * candidates, neighbors, self.config.hidden
            )
            graph = self.graph_encoder(graph)
            graph = graph.mean(dim=1).reshape(
                batch_size, candidates, self.config.hidden
            )
            hidden = self.graph_fusion(torch.cat((hidden, graph), dim=2))
        if self.variant.use_scene:
            scene = self.scene_encoder(batch)
            attended, _ = self.scene_attention(
                hidden, scene, scene, need_weights=False
            )
            hidden = self.scene_norm(hidden + attended)
        hidden = self.candidate_encoder(hidden, src_key_padding_mask=~mask)
        hidden = self.output_norm(hidden)
        score_residual = self.config.score_limit * torch.tanh(
            self.score_head(hidden)[..., 0]
        )
        mass_logits = batch["base_log_mass"] + score_residual
        mass_logits = mass_logits.masked_fill(~mask, -1e4)
        mass = torch.softmax(mass_logits, dim=1)
        coordinate_scale = torch.tensor(
            self.config.coordinate_limits,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        coordinate_residual = torch.tanh(self.coordinate_head(hidden)) * coordinate_scale
        coordinate = batch["coordinate"] + coordinate_residual
        port_residual = self.config.port_limit * torch.tanh(self.port_head(hidden))
        port = torch.softmax(batch["base_log_port"] + port_residual, dim=2)
        base_complex = torch.complex(
            batch["base_phase"][..., :PORTS],
            batch["base_phase"][..., PORTS:],
        )
        base_angle = torch.angle(base_complex)
        phase_residual = self.config.phase_limit * torch.tanh(self.phase_head(hidden))
        phase = base_angle + phase_residual
        weight = mask.to(hidden.dtype)
        pooled = (hidden * weight[..., None]).sum(dim=1) / weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        point_hidden = torch.cat((pooled, context), dim=1)
        zero_logit = self.zero_head(point_hidden)[:, 0]
        log_power = batch["base_log_power"] + 2.0 * torch.tanh(
            self.power_head(point_hidden)[:, 0]
        )
        return {
            "mass": mass,
            "mass_logits": mass_logits,
            "coordinate": coordinate,
            "coordinate_residual": coordinate_residual,
            "port": port,
            "port_residual": port_residual,
            "phase": phase,
            "phase_residual": phase_residual,
            "score_residual": score_residual,
            "zero_logit": zero_logit,
            "log_power": log_power,
        }


class TensorStore:
    def __init__(self, directory: Path, device: torch.device) -> None:
        self.device = device
        self.directory = directory
        self.state = json.loads((directory / "state.json").read_text(encoding="utf-8"))

        def tensor(name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
            value = np.load(array_path(directory, name), mmap_mode="r")
            return torch.tensor(np.asarray(value), dtype=dtype, device=device)

        self.candidate_feature = tensor("candidate_feature")
        self.coordinate = tensor("coordinate")
        self.base_log_mass = tensor("base_log_mass")
        self.base_log_port = tensor("base_log_port")
        self.base_phase = tensor("base_phase")
        self.target_mass = tensor("target_mass")
        self.target_delta = tensor("target_delta")
        self.target_port = tensor("target_port")
        self.target_phase = tensor("target_phase")
        self.mask = tensor("mask", torch.bool)
        self.context = tensor("context")
        self.base_log_power = tensor("base_log_power")
        self.graph_feature = tensor("graph_feature")
        self.fold_id = tensor("fold_id", torch.int64)
        self.point_power = tensor("point_power", torch.float64)
        self.target_pas = torch.tensor(
            np.asarray(np.load(TARGET_PAS, mmap_mode="r")), device=device
        )
        self.target_pdp = torch.tensor(
            np.asarray(np.load(TARGET_PDP, mmap_mode="r")), device=device
        )
        setup, positions, _, site, _ = load_positions()
        ids = np.arange(len(positions), dtype=np.int64)
        route, panorama = e83.route_node_features(ids, positions, site, setup.bs_positions)
        self.route = torch.as_tensor(route, device=device)
        self.panorama = torch.as_tensor(panorama, device=device)
        raster = np.load(RASTER_CACHE, allow_pickle=False)
        self.polar = torch.as_tensor(
            raster["train_polar"].astype(np.float32), device=device
        )
        self.corridor = torch.as_tensor(
            raster["train_corridor"].astype(np.float32), device=device
        )

    def normalization(self, fit_rows: np.ndarray) -> dict[str, torch.Tensor]:
        rows = torch.as_tensor(fit_rows, device=self.device)
        selected = self.candidate_feature[rows]
        selected_mask = self.mask[rows]
        values = selected[selected_mask]
        candidate_mean = values.mean(dim=0)
        candidate_std = values.std(dim=0)
        candidate_std = torch.where(
            candidate_std < 1e-4, torch.ones_like(candidate_std), candidate_std
        )
        context_mean = self.context[rows].mean(dim=0)
        context_std = self.context[rows].std(dim=0)
        context_std = torch.where(
            context_std < 1e-4, torch.ones_like(context_std), context_std
        )
        graph_values = self.graph_feature[rows]
        graph_valid = graph_values[..., -1] > 0.5
        graph_selected = graph_values[graph_valid]
        graph_mean = graph_selected.mean(dim=0)
        graph_std = graph_selected.std(dim=0)
        graph_std = torch.where(
            graph_std < 1e-4, torch.ones_like(graph_std), graph_std
        )
        return {
            "candidate_mean": candidate_mean,
            "candidate_std": candidate_std,
            "context_mean": context_mean,
            "context_std": context_std,
            "graph_mean": graph_mean,
            "graph_std": graph_std,
        }

    def batch(
        self, rows: torch.Tensor, normalization: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        candidate = (
            self.candidate_feature[rows] - normalization["candidate_mean"]
        ) / normalization["candidate_std"]
        context = (self.context[rows] - normalization["context_mean"]) / normalization[
            "context_std"
        ]
        graph = (self.graph_feature[rows] - normalization["graph_mean"]) / normalization[
            "graph_std"
        ]
        graph = torch.where(
            self.graph_feature[rows][..., -1:] > 0.5, graph, torch.zeros_like(graph)
        )
        return {
            "candidate_feature": candidate.clamp(-8.0, 8.0),
            "context": context.clamp(-8.0, 8.0),
            "graph_feature": graph.clamp(-8.0, 8.0),
            "coordinate": self.coordinate[rows],
            "base_log_mass": self.base_log_mass[rows],
            "base_log_port": self.base_log_port[rows],
            "base_phase": self.base_phase[rows],
            "base_log_power": self.base_log_power[rows],
            "mask": self.mask[rows],
            "target_mass": self.target_mass[rows],
            "target_delta": self.target_delta[rows],
            "target_port": self.target_port[rows],
            "target_phase": self.target_phase[rows],
            "target_pas": self.target_pas[rows],
            "target_pdp": self.target_pdp[rows],
            "point_power": self.point_power[rows],
            "route": self.route[rows].permute(0, 3, 1, 2),
            "panorama": self.panorama[rows].permute(0, 3, 1, 2),
            "polar": self.polar[rows],
            "corridor": self.corridor[rows],
        }


def raw_output(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mass_logits = batch["base_log_mass"].masked_fill(~batch["mask"], -1e4)
    base_complex = torch.complex(
        batch["base_phase"][..., :PORTS], batch["base_phase"][..., PORTS:]
    )
    zeros_score = torch.zeros_like(mass_logits)
    zeros_coordinate = torch.zeros_like(batch["coordinate"])
    zeros_port = torch.zeros_like(batch["base_log_port"])
    zeros_phase = torch.zeros_like(batch["base_log_port"])
    return {
        "mass": torch.softmax(mass_logits, dim=1),
        "mass_logits": mass_logits,
        "coordinate": batch["coordinate"],
        "coordinate_residual": zeros_coordinate,
        "port": torch.softmax(batch["base_log_port"], dim=2),
        "port_residual": zeros_port,
        "phase": torch.angle(base_complex),
        "phase_residual": zeros_phase,
        "score_residual": zeros_score,
        "zero_logit": torch.zeros(len(mass_logits), device=mass_logits.device),
        "log_power": batch["base_log_power"],
    }


def sufficient_scores(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rendered = {
        "mass": output["mass"],
        "coordinate": output["coordinate"],
        "port": output["port"].reshape(
            len(output["mass"]), output["mass"].shape[1], MP, N_UE
        ),
    }
    pas, pdp = render_spectra(rendered)
    pas_norm = torch.sqrt(
        pas.square().sum(dim=(1, 2, 3), dtype=torch.float64)
    )
    pas_dot = (pas * batch["target_pas"]).sum(
        dim=(1, 2, 3), dtype=torch.float64
    )
    pas_slice = pas_dot / torch.where(
        pas_norm > 0.0, pas_norm, torch.ones_like(pas_norm)
    )
    pas_score = pas_slice.mean(dim=1, dtype=torch.float64)
    pdp_norm = torch.sqrt(pdp.square().sum(dim=3, dtype=torch.float64))
    pdp_dot = (pdp * batch["target_pdp"]).sum(dim=3, dtype=torch.float64)
    pdp_slice = pdp_dot / torch.where(
        pdp_norm > 0.0, pdp_norm, torch.ones_like(pdp_norm)
    )
    pdp_score = pdp_slice.mean(dim=(1, 2), dtype=torch.float64)
    return pas_score, pdp_score, pas, pdp


def state_objective(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Config,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_mass = batch["target_mass"]
    target_total = target_mass.sum(dim=1)
    nonzero = target_total > 0.0
    target_distribution = target_mass / target_total[:, None].clamp_min(TINY)
    pas_score, pdp_score, _, _ = sufficient_scores(output, batch)
    spectral_score = 0.5 * (pas_score + pdp_score)
    spectral_loss = (
        1.0 - spectral_score[nonzero].mean()
        if nonzero.any()
        else output["mass"].sum() * 0.0
    )
    listwise_per_point = -(
        target_distribution * torch.log_softmax(output["mass_logits"], dim=1)
    ).sum(dim=1)
    listwise = (
        listwise_per_point[nonzero].mean()
        if nonzero.any()
        else output["mass"].sum() * 0.0
    )
    mass_cosine = (
        torch.sqrt(output["mass"].clamp_min(1e-12))
        * torch.sqrt(target_distribution.clamp_min(0.0))
    ).sum(dim=1)
    mass_hellinger = (
        (1.0 - mass_cosine[nonzero]).mean()
        if nonzero.any()
        else output["mass"].sum() * 0.0
    )
    positive = target_mass > 0.0
    candidate_bce = F.binary_cross_entropy_with_logits(
        output["mass_logits"][batch["mask"]],
        positive[batch["mask"]].to(output["mass_logits"].dtype),
    )
    if positive.any():
        weight = target_mass[positive]
        weight = weight / weight.mean().clamp_min(TINY)
        coordinate = F.smooth_l1_loss(
            output["coordinate_residual"][positive],
            batch["target_delta"][positive],
            reduction="none",
        ).mean(dim=1)
        coordinate_loss = (coordinate * weight).mean()
        port_cosine = (
            torch.sqrt(output["port"][positive].clamp_min(1e-12))
            * torch.sqrt(batch["target_port"][positive].clamp_min(0.0))
        ).sum(dim=1)
        port_loss = ((1.0 - port_cosine) * weight).mean()
        selected_target_phase = batch["target_phase"][positive]
        target_phase = torch.complex(
            selected_target_phase[..., :PORTS],
            selected_target_phase[..., PORTS:],
        )
        predicted_phase = torch.exp(1j * output["phase"][positive])
        phase_cosine = torch.real(predicted_phase * torch.conj(target_phase))
        phase_weight = batch["target_port"][positive]
        phase_score = (phase_cosine * phase_weight).sum(dim=1)
        phase_loss = ((1.0 - phase_score) * weight).mean()
    else:
        coordinate_loss = output["mass"].sum() * 0.0
        port_loss = output["mass"].sum() * 0.0
        phase_loss = output["mass"].sum() * 0.0
        phase_score = torch.zeros(1, device=output["mass"].device)
    zero_target = (batch["point_power"] <= 0.0).to(output["zero_logit"].dtype)
    zero_loss = F.binary_cross_entropy_with_logits(output["zero_logit"], zero_target)
    power_valid = batch["point_power"] > 0.0
    if power_valid.any():
        target_log_power = torch.log(batch["point_power"][power_valid]).to(
            output["log_power"].dtype
        )
        power_loss = F.smooth_l1_loss(
            output["log_power"][power_valid], target_log_power
        )
    else:
        power_loss = output["log_power"].sum() * 0.0
    residual = (
        output["score_residual"].square().mean()
        + output["coordinate_residual"].square().mean()
        + output["port_residual"].square().mean()
        + output["phase_residual"].square().mean()
    )
    loss = (
        config.spectral_weight * spectral_loss
        + config.listwise_weight * listwise
        + config.mass_hellinger_weight * mass_hellinger
        + config.candidate_bce_weight * candidate_bce
        + config.coordinate_weight * coordinate_loss
        + config.port_weight * port_loss
        + config.phase_weight * phase_loss
        + config.zero_weight * zero_loss
        + config.power_weight * power_loss
        + config.residual_weight * residual
    )
    return loss, {
        "loss": loss,
        "pas": pas_score,
        "pdp": pdp_score,
        "spectral": spectral_score,
        "listwise": listwise.detach(),
        "mass_hellinger": mass_hellinger.detach(),
        "coordinate": coordinate_loss.detach(),
        "port": port_loss.detach(),
        "phase": phase_loss.detach(),
        "phase_score": phase_score.detach().mean(),
        "zero": zero_loss.detach(),
        "power": power_loss.detach(),
    }


def render_complex_channel(output: dict[str, torch.Tensor]) -> torch.Tensor:
    count = min(TARGET_PATHS, output["mass"].shape[1])
    selected_mass, selected = torch.topk(output["mass"], count, dim=1)
    coordinate = torch.gather(
        output["coordinate"], 1, selected[..., None].expand(-1, -1, 3)
    )
    port = torch.gather(
        output["port"], 1, selected[..., None].expand(-1, -1, PORTS)
    )
    phase = torch.gather(
        output["phase"], 1, selected[..., None].expand(-1, -1, PORTS)
    )
    coefficient_power = selected_mass[..., None].clamp_min(0.0) * port.clamp_min(0.0)
    positive_power = coefficient_power > 0.0
    safe_power = torch.where(
        positive_power, coefficient_power, torch.ones_like(coefficient_power)
    )
    coefficient_amplitude = torch.where(
        positive_power, torch.sqrt(safe_power), torch.zeros_like(coefficient_power)
    )
    coefficient = coefficient_amplitude * torch.exp(1j * phase)
    atom = _component_atom(coordinate, torch.complex64, coordinate.device)
    coefficient = coefficient.reshape(-1, count, MP, N_UE)
    channel = torch.einsum("bkpu,bkhvd->bpuhvd", coefficient, atom)
    channel = channel.permute(0, 1, 3, 4, 2, 5).reshape(
        -1, MP * MH * MV, N_UE, S_SC
    )
    current_power = channel.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
    target_power = torch.exp(output["log_power"].double())
    scale = torch.sqrt(target_power / current_power.clamp_min(TINY)).to(channel.dtype)
    return channel * scale[:, None, None, None]


def torch_official_metrics(
    prediction: torch.Tensor, truth: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_shaped = prediction.reshape(-1, MP, MH, MV, N_UE, S_SC)
    truth_shaped = truth.reshape(-1, MP, MH, MV, N_UE, S_SC)
    pred_pas = torch.fft.fft2(pred_shaped, dim=(2, 3)).abs().square().float()
    truth_pas = torch.fft.fft2(truth_shaped, dim=(2, 3)).abs().square().float()
    pas_dot = (pred_pas * truth_pas).sum(dim=(1, 2, 3), dtype=torch.float64)
    pas_norm_product = (
        pred_pas.square().sum(dim=(1, 2, 3), dtype=torch.float64)
        * truth_pas.square().sum(dim=(1, 2, 3), dtype=torch.float64)
    )
    # Replacing the zero product before sqrt preserves the official 1e-12
    # denominator while avoiding an infinite sqrt gradient at zero.
    pas_denominator = torch.sqrt(
        torch.where(
            pas_norm_product == 0.0,
            torch.full_like(pas_norm_product, 1e-24),
            pas_norm_product,
        )
    )
    pas = pas_dot / pas_denominator
    pas = pas.mean(dim=(1, 2), dtype=torch.float64)
    pred_pdp = torch.fft.ifft(prediction, dim=3).abs().square().float()
    truth_pdp = torch.fft.ifft(truth, dim=3).abs().square().float()
    pdp_dot = (pred_pdp * truth_pdp).sum(dim=3, dtype=torch.float64)
    pdp_norm_product = (
        pred_pdp.square().sum(dim=3, dtype=torch.float64)
        * truth_pdp.square().sum(dim=3, dtype=torch.float64)
    )
    pdp_denominator = torch.sqrt(
        torch.where(
            pdp_norm_product == 0.0,
            torch.full_like(pdp_norm_product, 1e-24),
            pdp_norm_product,
        )
    )
    pdp = pdp_dot / pdp_denominator
    pdp = pdp.mean(dim=(1, 2), dtype=torch.float64)
    squared_error = (prediction - truth).abs().square().sum(
        dim=(1, 2, 3), dtype=torch.float64
    )
    truth_energy = truth.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
    return pas, pdp, squared_error, truth_energy


@torch.inference_mode()
def evaluate_rows(
    model: CompletePathStateOperator | None,
    store: TensorStore,
    rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    config: Config,
    batch_size: int,
    include_complex: bool,
) -> dict[str, object]:
    if model is not None:
        model.eval()
    train_h = np.load(TRAIN_CHANNEL, mmap_mode="r") if include_complex else None
    pas_parts: list[np.ndarray] = []
    pdp_parts: list[np.ndarray] = []
    mass_parts: list[np.ndarray] = []
    phase_parts: list[np.ndarray] = []
    zero_prediction: list[np.ndarray] = []
    zero_target: list[np.ndarray] = []
    power_errors: list[np.ndarray] = []
    complex_pas: list[np.ndarray] = []
    complex_pdp: list[np.ndarray] = []
    complex_error = 0.0
    complex_energy = 0.0
    losses = []
    for start in range(0, len(rows), batch_size):
        local = rows[start : start + batch_size]
        row_tensor = torch.as_tensor(local, device=store.device)
        batch = store.batch(row_tensor, normalization)
        output = raw_output(batch) if model is None else model(batch)
        loss, values = state_objective(output, batch, config)
        losses.append(float(loss.item()) * len(local))
        pas_parts.append(values["pas"].cpu().numpy())
        pdp_parts.append(values["pdp"].cpu().numpy())
        target_distribution = batch["target_mass"] / batch["target_mass"].sum(
            dim=1, keepdim=True
        ).clamp_min(TINY)
        nonzero = batch["target_mass"].sum(dim=1) > 0.0
        mass_cosine = (
            torch.sqrt(output["mass"].clamp_min(1e-12))
            * torch.sqrt(target_distribution.clamp_min(0.0))
        ).sum(dim=1)
        mass_parts.append(mass_cosine.cpu().numpy())
        positive = batch["target_mass"] > 0.0
        local_phase = torch.zeros(len(local), dtype=torch.float64, device=store.device)
        if positive.any():
            target_phase = torch.complex(
                batch["target_phase"][..., :PORTS],
                batch["target_phase"][..., PORTS:],
            )
            phase_cosine = torch.real(
                torch.exp(1j * output["phase"]) * torch.conj(target_phase)
            )
            phase_cosine = (phase_cosine * batch["target_port"]).sum(dim=2)
            numerator = (phase_cosine * batch["target_mass"]).sum(dim=1)
            denominator = batch["target_mass"].sum(dim=1).clamp_min(TINY)
            local_phase = numerator.double() / denominator.double()
        phase_parts.append(local_phase.cpu().numpy())
        zero_prediction.append(torch.sigmoid(output["zero_logit"]).cpu().numpy())
        zero_target.append((batch["point_power"] <= 0.0).cpu().numpy())
        valid_power = batch["point_power"] > 0.0
        local_power_error = torch.full(
            (len(local),), np.nan, dtype=torch.float64, device=store.device
        )
        local_power_error[valid_power] = (
            output["log_power"][valid_power].double()
            - torch.log(batch["point_power"][valid_power])
        ).abs()
        power_errors.append(local_power_error.cpu().numpy())
        if include_complex and train_h is not None:
            prediction = render_complex_channel(output)
            truth = torch.as_tensor(
                np.asarray(train_h[local], dtype=np.complex64), device=store.device
            )
            cp, cd, error, energy = torch_official_metrics(prediction, truth)
            complex_pas.append(cp.cpu().numpy())
            complex_pdp.append(cd.cpu().numpy())
            complex_error += float(error.sum().item())
            complex_energy += float(energy.sum().item())
    pas = np.concatenate(pas_parts)
    pdp = np.concatenate(pdp_parts)
    mass = np.concatenate(mass_parts)
    phase = np.concatenate(phase_parts)
    zero_probability = np.concatenate(zero_prediction)
    is_zero = np.concatenate(zero_target).astype(bool)
    power_error = np.concatenate(power_errors)
    output: dict[str, object] = {
        "count": int(len(rows)),
        "loss": float(np.sum(losses) / max(len(rows), 1)),
        "path_endpoint_pas": float(np.mean(pas)),
        "path_endpoint_pdp": float(np.mean(pdp)),
        "path_endpoint_mean": float(np.mean(0.5 * (pas + pdp))),
        "path_endpoint_fold": [
            float(np.mean(0.5 * (pas[store.fold_id[torch.as_tensor(rows, device=store.device)].cpu().numpy() == fold] + pdp[store.fold_id[torch.as_tensor(rows, device=store.device)].cpu().numpy() == fold])))
            if np.any(store.fold_id[torch.as_tensor(rows, device=store.device)].cpu().numpy() == fold)
            else None
            for fold in range(5)
        ],
        "mass_hellinger_cosine": float(np.mean(mass)),
        "path_phase_cosine": float(np.mean(phase[~is_zero])) if np.any(~is_zero) else 0.0,
        "zero_probability_zero_mean": float(np.mean(zero_probability[is_zero]))
        if np.any(is_zero)
        else None,
        "zero_probability_nonzero_mean": float(np.mean(zero_probability[~is_zero]))
        if np.any(~is_zero)
        else None,
        "log_power_mae": float(np.nanmean(power_error)),
        "per_point_pas": pas,
        "per_point_pdp": pdp,
    }
    if include_complex:
        cp = np.concatenate(complex_pas)
        cd = np.concatenate(complex_pdp)
        nmse = complex_error / max(complex_energy, TINY)
        output["complex_pas"] = float(np.mean(cp))
        output["complex_pdp"] = float(np.mean(cd))
        output["complex_nmse"] = float(nmse)
        output["complex_nmse_score"] = float(1.0 / (1.0 + nmse))
        output["complex_score"] = float(
            0.4 * np.mean(cp) + 0.4 * np.mean(cd) + 0.2 / (1.0 + nmse)
        )
    return output


def compact_metrics(values: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in values.items()
        if not key.startswith("per_point_")
    }


def train_variant(
    store: TensorStore,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    held_rows: np.ndarray,
    config: Config,
    variant: Variant,
    seed: int,
) -> tuple[CompletePathStateOperator, dict[str, object], dict[str, torch.Tensor]]:
    effective_seed = seed % (2**32)
    torch.manual_seed(effective_seed)
    np.random.seed(effective_seed)
    normalization = store.normalization(fit_rows)
    model = CompletePathStateOperator(
        store.candidate_feature.shape[-1],
        store.context.shape[-1],
        store.graph_feature.shape[-1],
        config,
        variant,
    ).to(store.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.epochs, 1),
        eta_min=config.learning_rate * 0.05,
    )
    rng = np.random.default_rng(effective_seed)
    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = config.minimum_epochs
    best_validation = -math.inf
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(fit_rows)
        losses = []
        for start in range(0, len(order), config.batch_size):
            rows = torch.as_tensor(
                order[start : start + config.batch_size], device=store.device
            )
            batch = store.batch(rows, normalization)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss, _ = state_objective(output, batch, config)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss variant={variant.name} epoch={epoch}"
                )
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            if not torch.isfinite(gradient):
                raise RuntimeError(
                    f"non-finite gradient variant={variant.name} epoch={epoch}"
                )
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        row = {"epoch": float(epoch), "train_loss": float(np.mean(losses))}
        if (
            epoch >= config.minimum_epochs
            and epoch % config.validation_interval == 0
        ):
            validation = evaluate_rows(
                model,
                store,
                validation_rows,
                normalization,
                config,
                config.batch_size,
                False,
            )
            score = float(validation["path_endpoint_mean"])
            row["validation_path_endpoint"] = score
            row["validation_mass_cosine"] = float(
                validation["mass_hellinger_cosine"]
            )
            if score > best_validation:
                best_validation = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
        history.append(row)
        if epoch % 4 == 0 or epoch == config.epochs:
            message = (
                f"variant={variant.name} epoch={epoch}/{config.epochs} "
                f"train={row['train_loss']:.6f}"
            )
            if "validation_path_endpoint" in row:
                message += f" validation={row['validation_path_endpoint']:.6f}"
            print(message, flush=True)
    model.load_state_dict(best_state)
    validation = evaluate_rows(
        model,
        store,
        validation_rows,
        normalization,
        config,
        config.batch_size,
        True,
    )
    held = evaluate_rows(
        model,
        store,
        held_rows,
        normalization,
        config,
        config.batch_size,
        True,
    )
    checkpoint = ARTIFACT_DIR / f"outer0_screen_{variant.name}.pt"
    torch.save(
        {
            "experiment": VERSION,
            "variant": asdict(variant),
            "config": asdict(config),
            "effective_seed": effective_seed,
            "state_dict": model.state_dict(),
            "normalization": {key: value.cpu() for key, value in normalization.items()},
            "best_epoch": best_epoch,
            "official_h_only": True,
            "sionna_used": False,
        },
        checkpoint,
    )
    result = {
        "variant": asdict(variant),
        "effective_seed": effective_seed,
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "best_epoch": int(best_epoch),
        "history": history,
        "validation": compact_metrics(validation),
        "held": compact_metrics(held),
        "checkpoint": PATHS.relative(checkpoint),
    }
    return model, result, normalization


def run_screen(config: Config, device: torch.device) -> dict[str, object]:
    directory = cache_dir("screen_block01")
    if not (directory / "state.json").is_file():
        raise FileNotFoundError(
            "screen cache is missing; run --stage prepare-screen first"
        )
    started = time.time()
    store = TensorStore(directory, device)
    fold_id = store.fold_id.cpu().numpy()
    held_rows = np.flatnonzero(fold_id == 0)
    validation_rows = np.flatnonzero(fold_id == 1)
    fit_rows = np.flatnonzero(np.isin(fold_id, (2, 3, 4)))
    baseline_normalization = store.normalization(fit_rows)
    baseline_validation = evaluate_rows(
        None,
        store,
        validation_rows,
        baseline_normalization,
        config,
        config.batch_size,
        True,
    )
    baseline_held = evaluate_rows(
        None,
        store,
        held_rows,
        baseline_normalization,
        config,
        config.batch_size,
        True,
    )
    rows = []
    for index, variant in enumerate(VARIANTS):
        _, result, _ = train_variant(
            store,
            fit_rows,
            validation_rows,
            held_rows,
            config,
            variant,
            config.random_state + 100 * index,
        )
        rows.append(result)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    ranking = sorted(
        rows,
        key=lambda row: float(row["validation"]["path_endpoint_mean"]),
        reverse=True,
    )
    selected = ranking[0]
    complete = next(row for row in rows if row["variant"]["name"] == "complete")
    candidate_only = next(
        row for row in rows if row["variant"]["name"] == "candidate_only"
    )
    gate = {
        "minimum_held_gain_over_raw": 0.03,
        "minimum_complete_gain_over_candidate_only": 0.003,
        "held_gain_over_raw": float(
            complete["held"]["path_endpoint_mean"]
            - baseline_held["path_endpoint_mean"]
        ),
        "complete_gain_over_candidate_only": float(
            complete["held"]["path_endpoint_mean"]
            - candidate_only["held"]["path_endpoint_mean"]
        ),
        "validation_selected_variant": selected["variant"]["name"],
    }
    gate["passed"] = bool(
        gate["held_gain_over_raw"] >= gate["minimum_held_gain_over_raw"]
        and gate["complete_gain_over_candidate_only"]
        >= gate["minimum_complete_gain_over_candidate_only"]
        and gate["validation_selected_variant"] == "complete"
    )
    report = {
        "experiment": VERSION,
        "status": "outer0_screen_complete",
        "primary_hypothesis": (
            "Uncompressed official-neighbor complex path tokens contain sufficient "
            "support, while full map boundaries and query-query path continuity add "
            "deployable information for per-point path competition and state recovery."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "protocol": {
            "globally_blocked_folds": [0, 1],
            "fit_folds": [2, 3, 4],
            "validation_fold": 1,
            "held_fold": 0,
            "held_labels_seen": False,
            "held_h_used_as_donor_anywhere": False,
            "validation_h_used_as_donor_anywhere": False,
        },
        "config": asdict(config),
        "cache": store.state,
        "baseline": {
            "validation": compact_metrics(baseline_validation),
            "held": compact_metrics(baseline_held),
        },
        "variants": rows,
        "validation_ranking": [row["variant"]["name"] for row in ranking],
        "deployment_gate": gate,
        "seconds": float(time.time() - started),
    }
    SCREEN_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "baseline": report["baseline"],
                "variants": [
                    {
                        "name": row["variant"]["name"],
                        "best_epoch": row["best_epoch"],
                        "validation": row["validation"],
                        "held": row["held"],
                    }
                    for row in rows
                ],
                "gate": gate,
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return report


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    config = Config(epochs=args.epochs, batch_size=args.batch_size)
    if args.stage == "prepare-screen":
        report = build_cache("screen_block01", (0, 1), config, args.force)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    else:
        run_screen(config, torch.device(args.device))


if __name__ == "__main__":
    main()
