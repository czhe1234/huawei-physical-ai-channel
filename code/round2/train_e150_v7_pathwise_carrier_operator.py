#!/usr/bin/env python3
"""Train a pathwise, geometry-conditioned carrier operator around the V7 channel."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F

import analyze_e114_path_state_variable_ceiling as e114
import analyze_e119_v12_unique_path_complement as e119
import analyze_e123_complete_path_variable_system as e123
import analyze_e124_one_to_one_path_identity_ceiling as e124
import analyze_e146_map_propagation_graph_coverage as e146
import analyze_e147_height_representation_contrast as e147
import analyze_e148_v7_pathwise_neural_residual as e148
import build_v7_complex_carrier_bank as carrier_bank
from carrier_calibration import estimate_carrier
from complex_baselines import affine_coefficients, idw_weights, _steering_phase
from data import load_positions
from project_paths import PATHS
from run_path_component_transport import _component_atom
from run_rbf_spectral_field import statistics
import train_e115_complete_path_state_operator as e115
import train_e118_hierarchical_unique_path_operator as e118


VERSION = "Round2-E150-v7-pathwise-carrier-operator"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v7_pathwise_carrier_operator"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v7_pathwise_carrier_operator"
REPORT = ANALYSIS_DIR / "v7_pathwise_carrier_operator.json"
METRICS = ARTIFACT_DIR / "v7_pathwise_carrier_operator_metrics.npz"
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"

LOCAL_CANDIDATES = 64
TOTAL_CANDIDATES = 96
CARRIERS = 17
PORTS = e115.PORTS
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    hidden: int = 192
    heads: int = 8
    layers: int = 3
    dropout: float = 0.06
    carrier_hidden: int = 64
    pretrain_epochs: int = 18
    finetune_epochs: int = 18
    validation_interval: int = 2
    batch_size: int = 8
    metric_batch_size: int = 8
    learning_rate: float = 3e-4
    finetune_learning_rate: float = 8e-5
    weight_decay: float = 3e-4
    action_limit: float = 0.35
    coordinate_limits: tuple[float, float, float] = (0.45, 0.30, 1.25)
    path_gain_limit: float = 0.80
    port_gain_limit: float = 0.35
    phase_limit: float = 0.50
    power_limit: float = 0.25
    identity_weight: float = 0.30
    coordinate_weight: float = 0.12
    coefficient_weight: float = 0.28
    power_weight: float = 0.06
    official_weight: float = 1.0
    component_barrier_weight: float = 0.35
    trust_weight: float = 0.01
    geometry_control_strata: int = 4
    required_score: float = 0.677521
    required_v7_gain: float = 0.005
    required_positive_folds: int = 4
    required_geometry_margin: float = 0.002
    seed: int = 20_260_810_150


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--phase", choices=("cache", "oracle", "train", "all"), default="all"
    )
    parser.add_argument("--outer", default="all")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=18)
    parser.add_argument("--finetune-epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--metric-batch-size", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def cache_dir(outer: int) -> Path:
    return ARTIFACT_DIR / f"outer{outer}_pathwise_cache"


def cache_path(outer: int, name: str) -> Path:
    return cache_dir(outer) / f"{name}.npy"


def checkpoint_path(outer: int) -> Path:
    return ARTIFACT_DIR / f"outer{outer}_pathwise_carrier_operator.pt"


def selected_outers(value: str) -> list[int]:
    if value == "all":
        return list(range(5))
    outer = int(value)
    if outer not in range(5):
        raise ValueError(f"invalid outer fold: {value}")
    return [outer]


def periodic_delta(value: np.ndarray, target: np.ndarray, period: float) -> np.ndarray:
    return np.remainder(value - target + period / 2.0, period) - period / 2.0


def strict_neighbors(
    positions: np.ndarray,
    site: np.ndarray,
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    neighbors = np.empty((len(query_ids), count), dtype=np.int64)
    distances = np.empty((len(query_ids), count), dtype=np.float64)
    fit_membership = np.zeros(len(positions), dtype=bool)
    fit_membership[fit_ids] = True
    for site_value in np.unique(site[query_ids]):
        site_value = int(site_value)
        fit_site = fit_ids[site[fit_ids] == site_value]
        rows = np.flatnonzero(site[query_ids] == site_value)
        tree = cKDTree(positions[fit_site, :2])
        raw_distance, raw_local = tree.query(
            positions[query_ids[rows], :2], k=count + 1, workers=-1
        )
        raw_ids = fit_site[raw_local]
        for local_row, destination in enumerate(rows):
            query_id = int(query_ids[destination])
            valid = raw_ids[local_row] != query_id if fit_membership[query_id] else np.ones(
                count + 1, dtype=bool
            )
            chosen = raw_ids[local_row][valid][:count]
            chosen_distance = raw_distance[local_row][valid][:count]
            if len(chosen) != count:
                raise RuntimeError(f"insufficient strict neighbors for point {query_id}")
            neighbors[destination] = chosen
            distances[destination] = chosen_distance
    return distances, neighbors


def fast_carrier_predictions(
    donor_channel: np.ndarray,
    donor_positions: np.ndarray,
    query_position: np.ndarray,
    distances: np.ndarray,
    bs_position: np.ndarray,
    configs: list,
) -> np.ndarray:
    """Generate the exact 17 linear V7 carriers while sharing common work."""

    channel = np.asarray(donor_channel, dtype=np.complex64)
    output = np.empty((len(configs), *channel.shape[1:]), dtype=np.complex64)
    query_range = float(np.linalg.norm(query_position - bs_position))
    donor_range = np.linalg.norm(donor_positions - bs_position[None], axis=1)
    subcarrier = np.linspace(-0.5, 0.5, channel.shape[-1], dtype=np.float32)
    beta = float(configs[0].beta) + float(configs[0].gamma) * subcarrier
    radial = np.exp(
        1j * (donor_range - query_range)[:, None] * beta[None]
    ).astype(np.complex64)
    aligned = channel * radial[:, None, None, :]

    for index, config in enumerate(configs):
        k = min(int(config.k), len(distances))
        if config.method == "nearest":
            output[index] = channel[0]
            continue
        local_distance = distances[:k]
        if config.method in ("raw_idw", "radial_idw"):
            coefficient = idw_weights(local_distance, config.distance_power)
        elif config.method == "radial_affine":
            coefficient = affine_coefficients(
                donor_positions[:k, :2] - query_position[None, :2],
                local_distance,
                config.distance_power,
                config.ridge,
            )
        else:
            raise ValueError(config.method)
        source = channel[:k] if config.method == "raw_idw" else aligned[:k]
        if config.steering_alpha > 0.0:
            steering = _steering_phase(
                donor_positions[:k], query_position, bs_position, config
            )
            prediction = np.einsum(
                "d,daus,da->aus", coefficient, source, steering, optimize=True
            )
        else:
            prediction = np.einsum("d,daus->aus", coefficient, source, optimize=True)
        output[index] = prediction.astype(np.complex64)
    return output


def batched_carrier_predictions(
    train_channel: torch.Tensor,
    neighbor_ids: np.ndarray,
    neighbor_positions: np.ndarray,
    query_positions: np.ndarray,
    distances: np.ndarray,
    bs_positions: np.ndarray,
    query_site: np.ndarray,
    configs_by_site: dict[int, list],
) -> torch.Tensor:
    """Generate the exact carrier family for several queries on one GPU."""

    batch_size, donor_count = neighbor_ids.shape
    antenna_count = train_channel.shape[1]
    raw_indices = (0, 1, 3, 5, 7)
    radial_indices = (2, 4, 6, 8, 9, 10, 11, 12)
    steering_indices = (13, 14, 15, 16)
    raw_weight = np.zeros(
        (batch_size, len(raw_indices), donor_count), dtype=np.complex64
    )
    radial_weight = np.zeros(
        (batch_size, len(radial_indices), donor_count), dtype=np.complex64
    )
    steering_weight = np.zeros(
        (batch_size, len(steering_indices), donor_count, antenna_count),
        dtype=np.complex64,
    )
    radial_phase = np.empty(
        (batch_size, donor_count, train_channel.shape[-1]), dtype=np.complex64
    )
    subcarrier = np.linspace(
        -0.5, 0.5, train_channel.shape[-1], dtype=np.float32
    )
    raw_slot = {index: slot for slot, index in enumerate(raw_indices)}
    radial_slot = {index: slot for slot, index in enumerate(radial_indices)}
    steering_slot = {index: slot for slot, index in enumerate(steering_indices)}
    for row in range(batch_size):
        site_value = int(query_site[row])
        configs = configs_by_site[site_value]
        donor_position = neighbor_positions[row]
        query_position = query_positions[row]
        query_range = float(np.linalg.norm(query_position - bs_positions[site_value]))
        donor_range = np.linalg.norm(
            donor_position - bs_positions[site_value][None], axis=1
        )
        beta = float(configs[0].beta) + float(configs[0].gamma) * subcarrier
        radial_phase[row] = np.exp(
            1j * (donor_range - query_range)[:, None] * beta[None]
        ).astype(np.complex64)
        for index, config in enumerate(configs):
            k = min(int(config.k), donor_count)
            if config.method == "nearest":
                raw_weight[row, raw_slot[index], 0] = 1.0
                continue
            local_distance = distances[row, :k]
            if config.method in ("raw_idw", "radial_idw"):
                coefficient = idw_weights(local_distance, config.distance_power)
            elif config.method == "radial_affine":
                coefficient = affine_coefficients(
                    donor_position[:k, :2] - query_position[None, :2],
                    local_distance,
                    config.distance_power,
                    config.ridge,
                )
            else:
                raise ValueError(config.method)
            if index in raw_slot:
                raw_weight[row, raw_slot[index], :k] = coefficient
            elif index in radial_slot:
                radial_weight[row, radial_slot[index], :k] = coefficient
            elif index in steering_slot:
                steering = _steering_phase(
                    donor_position[:k],
                    query_position,
                    bs_positions[site_value],
                    config,
                )
                steering_weight[row, steering_slot[index], :k] = (
                    coefficient[:, None] * steering
                )
            else:
                raise RuntimeError(f"unassigned carrier index {index}")

    neighbor_tensor = torch.as_tensor(
        neighbor_ids, dtype=torch.long, device=train_channel.device
    )
    donor = train_channel[neighbor_tensor]
    phase = torch.as_tensor(radial_phase, device=train_channel.device)
    aligned = donor * phase[:, :, None, None]
    raw_prediction = torch.einsum(
        "bcd,bdaus->bcaus",
        torch.as_tensor(raw_weight, device=train_channel.device),
        donor,
    )
    radial_prediction = torch.einsum(
        "bcd,bdaus->bcaus",
        torch.as_tensor(radial_weight, device=train_channel.device),
        aligned,
    )
    steering_prediction = torch.einsum(
        "bcda,bdaus->bcaus",
        torch.as_tensor(steering_weight, device=train_channel.device),
        aligned,
    )
    output = torch.empty(
        (
            batch_size,
            CARRIERS,
            train_channel.shape[1],
            train_channel.shape[2],
            train_channel.shape[3],
        ),
        dtype=torch.complex64,
        device=train_channel.device,
    )
    output[:, raw_indices] = raw_prediction
    output[:, radial_indices] = radial_prediction
    output[:, steering_indices] = steering_prediction
    return output


@torch.inference_mode()
def batched_carrier_samples(
    train_channel: torch.Tensor,
    neighbor_ids: np.ndarray,
    neighbor_positions: np.ndarray,
    query_positions: np.ndarray,
    distances: np.ndarray,
    bs_positions: np.ndarray,
    query_site: np.ndarray,
    configs_by_site: dict[int, list],
    coordinates: np.ndarray,
    scale: float,
    reference_channel: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    predicted = batched_carrier_predictions(
        train_channel,
        neighbor_ids,
        neighbor_positions,
        query_positions,
        distances,
        bs_positions,
        query_site,
        configs_by_site,
    )
    power = predicted.abs().square().sum(
        dim=(2, 3, 4), dtype=torch.float64
    )
    batch_size, carrier_count = predicted.shape[:2]
    field = complex_angle_delay(
        predicted.reshape(-1, *predicted.shape[2:])
    )
    coordinate = torch.as_tensor(coordinates, device=train_channel.device)
    repeated = (
        coordinate[:, None]
        .expand(-1, carrier_count, -1, -1)
        .reshape(batch_size * carrier_count, coordinate.shape[1], 3)
    )
    sampled = periodic_trilinear_sample(field, repeated)
    sampled = (
        sampled.reshape(batch_size, carrier_count, coordinate.shape[1], PORTS)
        .permute(0, 2, 1, 3)
        .mul(scale)
    )
    reference_sample = None
    if reference_channel is not None:
        reference_field = complex_angle_delay(reference_channel)
        reference_sample = periodic_trilinear_sample(
            reference_field, coordinate
        ).mul(scale)
    return (
        sampled.cpu().numpy(),
        power.cpu().numpy(),
        reference_sample.cpu().numpy() if reference_sample is not None else None,
    )


def complex_angle_delay(channel: torch.Tensor) -> torch.Tensor:
    shaped = (
        channel.reshape(-1, e115.MP, e115.MH, e115.MV, e115.N_UE, e115.S_SC)
        .permute(0, 1, 4, 2, 3, 5)
        .reshape(-1, PORTS, e115.MH, e115.MV, e115.S_SC)
    )
    return torch.fft.ifft(torch.fft.fft2(shaped, dim=(-3, -2)), dim=-1)


def periodic_trilinear_sample(
    field: torch.Tensor, coordinate: torch.Tensor
) -> torch.Tensor:
    value = field.permute(0, 2, 3, 4, 1)
    lower = torch.floor(coordinate).to(torch.int64)
    fraction = coordinate - lower.to(coordinate.dtype)
    batch = torch.arange(len(field), device=field.device)[:, None]
    output = torch.zeros(
        (*coordinate.shape[:2], field.shape[1]),
        dtype=field.dtype,
        device=field.device,
    )
    periods = (e115.MH, e115.MV, e115.S_SC)
    for dh in (0, 1):
        wh = fraction[..., 0] if dh else 1.0 - fraction[..., 0]
        ih = (lower[..., 0] + dh) % periods[0]
        for dv in (0, 1):
            wv = fraction[..., 1] if dv else 1.0 - fraction[..., 1]
            iv = (lower[..., 1] + dv) % periods[1]
            for dd in (0, 1):
                wd = fraction[..., 2] if dd else 1.0 - fraction[..., 2]
                index_d = (lower[..., 2] + dd) % periods[2]
                output += (wh * wv * wd)[..., None] * value[
                    batch, ih, iv, index_d
                ]
    return output


def fit_strict_array_axes(
    outer: int,
    fit_ids: np.ndarray,
    observations: dict[str, np.ndarray],
    site: np.ndarray,
    config: e146.Config,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    valid_fit = observations["valid"].astype(bool)
    fit_mask = np.zeros(len(site), dtype=bool)
    fit_mask[fit_ids] = True
    valid_fit &= fit_mask
    site_values = np.unique(site).astype(np.int64)
    horizontal = np.zeros((int(site_values.max()) + 1, 4), dtype=np.float64)
    vertical = np.zeros_like(horizontal)
    record: dict[str, object] = {"fit_los_count": int(np.sum(valid_fit)), "sites": {}}
    axes = (
        ("horizontal", e115.MH, 12.0, horizontal),
        ("vertical", e115.MV, 40.0, vertical),
    )
    counts = {int(value): int(np.sum(valid_fit & (site == value))) for value in site_values}
    anchor_site = max(counts, key=counts.get)
    record["anchor_site"] = int(anchor_site)
    for axis_index, (name, period, bound, destination) in enumerate(axes):
        anchor_rows = np.flatnonzero(valid_fit & (site == anchor_site))
        unconstrained = e146.fit_unconstrained_axis(
            observations["unit"][anchor_rows],
            observations[name][anchor_rows],
            observations["weight"][anchor_rows],
            float(period),
            bound,
            config.seed + outer * 1000 + axis_index * 100,
            config,
        )
        norm = float(np.linalg.norm(unconstrained[:3]))
        record[f"{name}_shared_vector_norm"] = norm
        for site_value in site_values:
            site_value = int(site_value)
            rows = np.flatnonzero(valid_fit & (site == site_value))
            if len(rows) < 12:
                raise RuntimeError(
                    f"insufficient LOS rows outer={outer} site={site_value}: {len(rows)}"
                )
            parameter = e146.fit_fixed_norm_axis(
                observations["unit"][rows],
                observations[name][rows],
                observations["weight"][rows],
                float(period),
                norm,
                config.seed + outer * 10_000 + site_value * 1000 + axis_index * 10,
                config,
            )
            destination[site_value] = parameter
            error = e146.axis_error(
                parameter,
                observations["unit"][rows],
                observations[name][rows],
                float(period),
            )
            site_record = record["sites"].setdefault(str(site_value), {})
            site_record[f"{name}_parameter"] = parameter.tolist()
            site_record[f"{name}_fit_error_p50_p90"] = np.quantile(
                error, (0.5, 0.9)
            ).tolist()
    return horizontal, vertical, record


def multiheight_map_bank(
    base: dict[str, np.ndarray],
    current: np.ndarray,
    metadata: dict[str, np.ndarray],
    site: np.ndarray,
    bs_positions: np.ndarray,
    vertical_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    config = e147.Config()
    coordinate = np.array(current, copy=True)
    mask = np.array(base["mask"], copy=True)
    parent = np.broadcast_to(
        np.arange(mask.shape[1], dtype=np.int16)[None], mask.shape
    ).copy()
    alternative_count = np.zeros(mask.shape, dtype=np.int16)
    interaction = base["path_type"] >= e146.TYPE_REFLECTION
    for row in range(len(mask)):
        free = list(np.flatnonzero(~mask[row]))
        parents = np.flatnonzero(mask[row] & interaction[row])
        if not free or len(parents) == 0:
            continue
        parents = parents[np.argsort(base["score"][row, parents])]
        proposals: dict[int, list[float]] = {}
        for source in parents:
            low = float(metadata["z_min"][row, source])
            high = float(metadata["z_max"][row, source])
            interior_low = min(low + config.height_margin_m, high)
            interior_high = max(high - config.height_margin_m, interior_low)
            heights = [config.planar_height_m]
            heights.extend(
                interior_low + q * (interior_high - interior_low)
                for q in config.height_quantiles
            )
            values = e147.vertical_bin(
                np.repeat(
                    metadata["first_xy"][row, source][None].astype(np.float64),
                    len(heights),
                    axis=0,
                ),
                np.asarray(heights, dtype=np.float64),
                bs_positions[int(site[row])],
                vertical_axis[row],
            )
            unique: list[float] = []
            current_v = float(coordinate[row, source, 1])
            for value in values:
                distance = [
                    abs(float(periodic_delta(value, old, e115.MV)))
                    for old in (current_v, *unique)
                ]
                if min(distance) >= config.minimum_distinct_v_bin:
                    unique.append(float(value))
            proposals[int(source)] = unique
            alternative_count[row, source] = len(unique)
        round_index = 0
        while free:
            progressed = False
            for source in parents:
                values = proposals[int(source)]
                if round_index >= len(values) or not free:
                    continue
                slot = free.pop(0)
                coordinate[row, slot] = coordinate[row, source]
                coordinate[row, slot, 1] = values[round_index]
                mask[row, slot] = True
                parent[row, slot] = source
                alternative_count[row, slot] = len(values)
                progressed = True
            if not progressed:
                break
            round_index += 1
    return coordinate, mask, parent, alternative_count


def map_geometry_features(
    candidate_coordinate: np.ndarray,
    candidate_mask: np.ndarray,
    map_coordinate: np.ndarray,
    map_mask: np.ndarray,
    map_parent: np.ndarray,
    map_alternatives: np.ndarray,
    map_base: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    count, candidates, _ = candidate_coordinate.shape
    feature_parts: list[np.ndarray] = []
    nearest_index = np.full((count, candidates), -1, dtype=np.int16)
    names = [
        "log1p_map_match_cost",
        "map_match_confidence",
        "map_delta_h",
        "map_delta_v",
        "map_delta_delay",
        *(f"map_type_{name}" for name in e146.TYPE_NAMES),
        "map_depth",
        "map_log_path_length",
        "map_excess_length",
        "map_log_physical_score",
        "first_interaction_forward",
        "first_interaction_lateral",
        "first_interaction_to_ue",
        "first_interaction_z",
        "surface_z_min",
        "surface_z_max",
        "surface_z_span",
        "multiheight_alternative_count",
    ]
    output = np.zeros((count, candidates, len(names)), dtype=np.float32)
    for row in range(count):
        valid_map = np.flatnonzero(map_mask[row])
        valid_candidate = np.flatnonzero(candidate_mask[row])
        if len(valid_map) == 0 or len(valid_candidate) == 0:
            continue
        query = candidate_coordinate[row, valid_candidate]
        bank = map_coordinate[row, valid_map]
        dh = periodic_delta(query[:, None, 0], bank[None, :, 0], e115.MH)
        dv = periodic_delta(query[:, None, 1], bank[None, :, 1], e115.MV)
        dd = periodic_delta(query[:, None, 2], bank[None, :, 2], e115.S_SC)
        cost = (dh / 1.5) ** 2 + dv**2 + (dd / 2.0) ** 2
        local = np.argmin(cost, axis=1)
        selected_map = valid_map[local]
        selected_parent = map_parent[row, selected_map]
        nearest_index[row, valid_candidate] = selected_map.astype(np.int16)
        selected_cost = cost[np.arange(len(local)), local]
        path_type = map_base["path_type"][row, selected_parent]
        one_hot = np.eye(len(e146.TYPE_NAMES), dtype=np.float32)[path_type]
        path_length = map_base["path_length"][row, selected_parent].astype(np.float64)
        direct = float(np.linalg.norm(positions[row] - bs_positions[int(site[row])]))
        physical_score = map_base["score"][row, selected_parent].astype(np.float64)
        first_xy = metadata["first_xy"][row, selected_parent].astype(np.float64)
        first_z = metadata["first_z"][row, selected_parent].astype(np.float64)
        z_min = metadata["z_min"][row, selected_parent].astype(np.float64)
        z_max = metadata["z_max"][row, selected_parent].astype(np.float64)
        bs = bs_positions[int(site[row])]
        ue = positions[row]
        forward = ue[:2] - bs[:2]
        forward /= max(float(np.linalg.norm(forward)), 1e-12)
        lateral = np.asarray((-forward[1], forward[0]))
        first_offset = first_xy - bs[:2]
        to_ue = np.linalg.norm(first_xy - ue[:2], axis=1)
        local_feature = np.column_stack(
            (
                np.log1p(selected_cost),
                np.exp(-selected_cost),
                dh[np.arange(len(local)), local] / 1.5,
                dv[np.arange(len(local)), local],
                dd[np.arange(len(local)), local] / 2.0,
                one_hot,
                map_base["depth"][row, selected_parent] / 3.0,
                np.log1p(path_length),
                (path_length - direct) / 100.0,
                np.log1p(np.maximum(physical_score, 0.0)),
                (first_offset @ forward) / 100.0,
                (first_offset @ lateral) / 100.0,
                to_ue / 100.0,
                first_z / 30.0,
                z_min / 30.0,
                z_max / 30.0,
                (z_max - z_min) / 30.0,
                map_alternatives[row, selected_map] / 8.0,
            )
        ).astype(np.float32)
        output[row, valid_candidate] = local_feature
    return output, nearest_index, names


def point_features(
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    v7_power: np.ndarray,
    carrier_power: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    source = bs_positions[site]
    offset = positions - source
    distance = np.linalg.norm(offset, axis=1)
    unit = offset / np.maximum(distance[:, None], 1e-12)
    log_carrier = np.log(np.maximum(carrier_power, TINY))
    values = np.column_stack(
        (
            offset / 100.0,
            distance / 100.0,
            np.log1p(distance),
            unit,
            site == 0,
            site == 1,
            np.log(np.maximum(v7_power, TINY)),
            log_carrier.mean(axis=1),
            log_carrier.std(axis=1),
            log_carrier.max(axis=1) - log_carrier.min(axis=1),
        )
    ).astype(np.float32)
    names = [
        "offset_x",
        "offset_y",
        "offset_z",
        "distance",
        "log_distance",
        "unit_x",
        "unit_y",
        "unit_z",
        "site_0",
        "site_1",
        "log_v7_power",
        "mean_log_carrier_power",
        "std_log_carrier_power",
        "range_log_carrier_power",
    ]
    return values, names


def save_array(directory: Path, name: str, value: np.ndarray) -> None:
    np.save(directory / f"{name}.npy", value)


def load_cache(outer: int, device: torch.device | None = None) -> dict[str, object]:
    directory = cache_dir(outer)
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    arrays = {
        name: np.load(directory / f"{name}.npy", mmap_mode="r")
        for name in state["arrays"]
    }
    if device is None:
        return {"state": state, **arrays}
    return {
        "state": state,
        **{
            name: torch.as_tensor(np.asarray(value), device=device)
            for name, value in arrays.items()
        },
    }


def build_cache(
    outer: int,
    config: Config,
    device: torch.device,
    force: bool,
    *,
    query_ids_override: np.ndarray | None = None,
    directory_override: Path | None = None,
    reference_carrier_index: int | None = None,
    reference_name: str = "v7",
    carrier_batch_size: int = 1,
) -> dict[str, object]:
    directory = cache_dir(outer) if directory_override is None else directory_override
    state_path = directory / "state.json"
    if state_path.is_file() and not force:
        return json.loads(state_path.read_text(encoding="utf-8"))
    directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    inner = (outer + 1) % 5
    setup, positions_all, _, site_all, _ = load_positions()
    fold = np.load(e115.FULL_FOLD).astype(np.int8)
    if query_ids_override is None:
        metadata = np.load(e119.OOF_META, allow_pickle=False)
        query_ids = metadata["query_ids"].astype(np.int64)
    else:
        query_ids = np.asarray(query_ids_override, dtype=np.int64)
    query_positions = positions_all[query_ids]
    query_site = site_all[query_ids].astype(np.int8)
    query_fold = fold[query_ids]
    fit_ids = np.flatnonzero((fold != outer) & (fold != inner))
    train_channel = np.load(TRAIN_CHANNEL, mmap_mode="r")
    component = np.load(e114.COMPONENTS, allow_pickle=False)
    point_power_all = component["point_power"].astype(np.float64)
    los = np.load(e146.LOS_LABELS, allow_pickle=False)["train_center_clear"].astype(bool)
    observations = e146.select_los_observations(
        positions_all,
        site_all,
        setup.bs_positions,
        component["coordinates"].astype(np.float32),
        component["coefficients"].astype(np.complex64),
        component["point_power"].astype(np.float64),
        los,
    )
    map_config = e146.Config()
    horizontal_site, vertical_site, axis_record = fit_strict_array_axes(
        outer, fit_ids, observations, site_all, map_config
    )

    map_loaded = np.load(e146.CANDIDATE_CACHE, allow_pickle=False)
    map_base = {key: map_loaded[key][query_ids] for key in map_loaded.files if key != "config_json"}
    metadata_loaded = np.load(e147.METADATA_CACHE, allow_pickle=False)
    interaction_metadata = {
        key: metadata_loaded[key][query_ids]
        for key in ("first_xy", "first_z", "z_min", "z_max")
    }
    horizontal = horizontal_site[query_site]
    vertical = vertical_site[query_site]
    map_current = e146.map_coordinates_linear_array(map_base, horizontal, vertical)
    map_coordinate, map_mask, map_parent, map_alternatives = multiheight_map_bank(
        map_base,
        map_current,
        interaction_metadata,
        query_site,
        setup.bs_positions,
        vertical,
    )

    source_directory = e118.cache_dir(e123.cache_tag(outer))
    local_coordinate_all = np.load(source_directory / "coordinate.npy", mmap_mode="r")
    local_mask_all = np.load(source_directory / "mask.npy", mmap_mode="r")
    local_cluster_all = np.load(source_directory / "cluster_feature.npy", mmap_mode="r")
    local_mass_all = np.load(source_directory / "base_log_mass.npy", mmap_mode="r")
    local_port_all = np.load(source_directory / "base_log_port.npy", mmap_mode="r")
    local_phase_all = np.load(source_directory / "base_phase.npy", mmap_mode="r")
    local_member_all = np.load(source_directory / "member_count.npy", mmap_mode="r")
    local_coordinate = np.asarray(local_coordinate_all[query_ids], dtype=np.float32)
    local_mask = np.asarray(local_mask_all[query_ids], dtype=bool)

    distances, neighbors = strict_neighbors(
        positions_all, site_all, fit_ids, query_ids, 32
    )
    calibration = estimate_carrier(
        train_channel,
        positions_all,
        site_all,
        setup.bs_positions,
        fit_ids,
        point_power_all,
        max_pairs_per_site=500,
        seed=config.seed + outer * 101,
    )
    base_configs = carrier_bank.carrier_configs()
    configs_by_site = {
        int(site_value): [
            replace(
                template,
                beta=calibration[int(site_value)].beta,
                gamma=calibration[int(site_value)].gamma,
            )
            for template in base_configs
        ]
        for site_value in np.unique(query_site)
    }
    v7_all = (
        np.load(e148.V7_CHANNEL, mmap_mode="r")
        if reference_carrier_index is None
        else None
    )

    coordinate = np.zeros((len(query_ids), TOTAL_CANDIDATES, 3), dtype=np.float32)
    mask = np.zeros((len(query_ids), TOTAL_CANDIDATES), dtype=bool)
    source_type = np.full((len(query_ids), TOTAL_CANDIDATES), -1, dtype=np.int8)
    local_index = np.full((len(query_ids), TOTAL_CANDIDATES), -1, dtype=np.int16)
    map_index = np.full((len(query_ids), TOTAL_CANDIDATES), -1, dtype=np.int16)
    carrier_coefficient = np.zeros(
        (len(query_ids), TOTAL_CANDIDATES, CARRIERS, PORTS), dtype=np.complex64
    )
    v7_coefficient = np.zeros(
        (len(query_ids), TOTAL_CANDIDATES, PORTS), dtype=np.complex64
    )
    carrier_power = np.zeros((len(query_ids), CARRIERS), dtype=np.float64)
    v7_power = (
        np.sum(np.abs(v7_all) ** 2, axis=(1, 2, 3), dtype=np.float64)
        if v7_all is not None
        else np.zeros(len(query_ids), dtype=np.float64)
    )
    scale = math.sqrt(e115.S_SC / (e115.MH * e115.MV))

    def consume_candidate_row(
        row: int,
        sampled: np.ndarray,
        v7_sample: np.ndarray,
    ) -> None:
        destination = 0
        valid_local = np.flatnonzero(local_mask[row])
        for source in valid_local:
            coordinate[row, destination] = local_coordinate[row, source]
            mask[row, destination] = True
            source_type[row, destination] = 0
            local_index[row, destination] = source
            carrier_coefficient[row, destination] = sampled[source]
            v7_coefficient[row, destination] = v7_sample[source]
            destination += 1

        valid_map = np.flatnonzero(map_mask[row])
        if len(valid_local):
            h = periodic_delta(
                map_coordinate[row, valid_map, 0][:, None],
                local_coordinate[row, valid_local, 0][None, :],
                e115.MH,
            )
            v = periodic_delta(
                map_coordinate[row, valid_map, 1][:, None],
                local_coordinate[row, valid_local, 1][None, :],
                e115.MV,
            )
            d = periodic_delta(
                map_coordinate[row, valid_map, 2][:, None],
                local_coordinate[row, valid_local, 2][None, :],
                e115.S_SC,
            )
            duplicate = np.min((h / 1.5) ** 2 + v**2 + (d / 2.0) ** 2, axis=1) <= 0.16
            valid_map = valid_map[~duplicate]
        if len(valid_map):
            map_sample = sampled[LOCAL_CANDIDATES + valid_map]
            energy = np.sum(np.abs(map_sample) ** 2, axis=2, dtype=np.float64)
            fraction = energy / np.maximum(carrier_power[row][None], TINY)
            physical = map_base["score"][row, map_parent[row, valid_map]].astype(
                np.float64
            )
            ranking_score = (
                np.median(np.log(np.maximum(fraction, 1e-12)), axis=1)
                + 0.35 * np.max(np.log(np.maximum(fraction, 1e-12)), axis=1)
                - 0.03 * np.log1p(np.maximum(physical, 0.0))
            )
            order = valid_map[np.argsort(-ranking_score)]
            for source in order[: TOTAL_CANDIDATES - destination]:
                coordinate[row, destination] = map_coordinate[row, source]
                mask[row, destination] = True
                source_type[row, destination] = 1
                map_index[row, destination] = source
                carrier_coefficient[row, destination] = sampled[
                    LOCAL_CANDIDATES + source
                ]
                v7_coefficient[row, destination] = v7_sample[
                    LOCAL_CANDIDATES + source
                ]
                destination += 1

    if carrier_batch_size <= 1:
        for row, query_id in enumerate(query_ids):
            donor_ids = neighbors[row]
            donor_channel = np.asarray(
                train_channel[donor_ids], dtype=np.complex64
            )
            predicted = fast_carrier_predictions(
                donor_channel,
                positions_all[donor_ids],
                positions_all[query_id],
                distances[row],
                setup.bs_positions[int(query_site[row])],
                configs_by_site[int(query_site[row])],
            )
            carrier_power[row] = np.sum(
                np.abs(predicted) ** 2, axis=(1, 2, 3), dtype=np.float64
            )
            prediction_tensor = torch.as_tensor(predicted, device=device)
            carrier_field = complex_angle_delay(prediction_tensor)
            preliminary_coordinate = np.concatenate(
                (local_coordinate[row], map_coordinate[row]), axis=0
            ).astype(np.float32)
            repeated = torch.as_tensor(
                np.broadcast_to(
                    preliminary_coordinate[None],
                    (CARRIERS, len(preliminary_coordinate), 3),
                ).copy(),
                device=device,
            )
            sampled = periodic_trilinear_sample(carrier_field, repeated)
            sampled = (sampled * scale).permute(1, 0, 2).cpu().numpy()
            del prediction_tensor, carrier_field, repeated

            if reference_carrier_index is None:
                assert v7_all is not None
                v7_tensor = torch.as_tensor(
                    np.array(v7_all[row : row + 1], dtype=np.complex64, copy=True),
                    device=device,
                )
                v7_field = complex_angle_delay(v7_tensor)
                v7_sample = periodic_trilinear_sample(
                    v7_field,
                    torch.as_tensor(preliminary_coordinate[None], device=device),
                )[0]
                v7_sample = (v7_sample * scale).cpu().numpy()
                del v7_tensor, v7_field
            else:
                if reference_carrier_index not in range(CARRIERS):
                    raise ValueError(
                        "invalid reference carrier index: "
                        f"{reference_carrier_index}"
                    )
                v7_sample = sampled[:, reference_carrier_index]
                v7_power[row] = carrier_power[row, reference_carrier_index]
            consume_candidate_row(row, sampled, v7_sample)
            if (row + 1) % 10 == 0 or row + 1 == len(query_ids):
                print(
                    f"E150 cache outer={outer} carriers {row + 1}/{len(query_ids)}",
                    flush=True,
                )
    else:
        if reference_carrier_index is not None and reference_carrier_index not in range(
            CARRIERS
        ):
            raise ValueError(
                f"invalid reference carrier index: {reference_carrier_index}"
            )
        train_channel_device = torch.as_tensor(
            np.asarray(train_channel), device=device
        )
        reference_device = (
            torch.as_tensor(np.asarray(v7_all), device=device)
            if v7_all is not None
            else None
        )
        for start in range(0, len(query_ids), carrier_batch_size):
            stop = min(start + carrier_batch_size, len(query_ids))
            preliminary_coordinate = np.concatenate(
                (local_coordinate[start:stop], map_coordinate[start:stop]), axis=1
            ).astype(np.float32)
            sampled_batch, power_batch, reference_batch = batched_carrier_samples(
                train_channel_device,
                neighbors[start:stop],
                positions_all[neighbors[start:stop]],
                query_positions[start:stop],
                distances[start:stop],
                setup.bs_positions,
                query_site[start:stop],
                configs_by_site,
                preliminary_coordinate,
                scale,
                reference_device[start:stop]
                if reference_device is not None
                else None,
            )
            carrier_power[start:stop] = power_batch
            for local_row, row in enumerate(range(start, stop)):
                sampled = sampled_batch[local_row]
                if reference_carrier_index is None:
                    assert reference_batch is not None
                    v7_sample = reference_batch[local_row]
                else:
                    v7_sample = sampled[:, reference_carrier_index]
                    v7_power[row] = carrier_power[row, reference_carrier_index]
                consume_candidate_row(row, sampled, v7_sample)
            print(
                f"E150 cache outer={outer} carriers {stop}/{len(query_ids)}",
                flush=True,
            )
        del train_channel_device, reference_device

    geometry, nearest_map, geometry_names = map_geometry_features(
        coordinate,
        mask,
        map_coordinate,
        map_mask,
        map_parent,
        map_alternatives,
        map_base,
        interaction_metadata,
        query_positions,
        query_site,
        setup.bs_positions,
    )

    local_feature = np.zeros((len(query_ids), TOTAL_CANDIDATES, 66), dtype=np.float32)
    for row, query_id in enumerate(query_ids):
        destinations = np.flatnonzero(local_index[row] >= 0)
        if len(destinations) == 0:
            continue
        source = local_index[row, destinations].astype(np.int64)
        local_feature[row, destinations] = np.concatenate(
            (
                np.asarray(local_cluster_all[query_id, source], dtype=np.float32),
                np.asarray(local_mass_all[query_id, source], dtype=np.float32)[:, None],
                np.asarray(local_port_all[query_id, source], dtype=np.float32),
                np.asarray(local_phase_all[query_id, source], dtype=np.float32),
                np.log1p(
                    np.asarray(local_member_all[query_id, source], dtype=np.float32)
                )[:, None],
            ),
            axis=1,
        )
    phase = 2.0 * np.pi * coordinate / e115.PERIODS[None, None]
    candidate_feature = np.concatenate(
        (
            np.sin(phase),
            np.cos(phase),
            np.eye(2, dtype=np.float32)[np.clip(source_type, 0, 1)]
            * mask[..., None],
            local_feature,
            geometry,
        ),
        axis=2,
    ).astype(np.float32)
    candidate_feature[~mask] = 0.0
    candidate_names = [
        "sin_h",
        "sin_v",
        "sin_delay",
        "cos_h",
        "cos_v",
        "cos_delay",
        "source_local",
        "source_map",
        *(f"local_feature_{index}" for index in range(local_feature.shape[2])),
        *geometry_names,
    ]
    point_feature, point_names = point_features(
        query_positions,
        query_site,
        setup.bs_positions,
        v7_power,
        carrier_power,
    )

    arrays = {
        "query_ids": query_ids.astype(np.int32),
        "query_fold": query_fold.astype(np.int8),
        "coordinate": coordinate.astype(np.float16),
        "mask": mask,
        "source_type": source_type,
        "local_index": local_index,
        "map_index": map_index,
        "nearest_map_index": nearest_map,
        "candidate_feature": candidate_feature.astype(np.float16),
        "point_feature": point_feature,
        "carrier_coefficient": carrier_coefficient,
        "v7_coefficient": v7_coefficient,
        "carrier_log_power": np.log(np.maximum(carrier_power, TINY)).astype(np.float32),
        "v7_log_power": np.log(np.maximum(v7_power, TINY)).astype(np.float32),
    }
    for name, value in arrays.items():
        save_array(directory, name, value)
    state = {
        "experiment": VERSION,
        "outer": outer,
        "inner": inner,
        "official_h_only": True,
        "sionna_used": False,
        "v12_used": False,
        "reference_name": reference_name,
        "reference_carrier_index": reference_carrier_index,
        "carrier_batch_size": int(carrier_batch_size),
        "candidate_count": TOTAL_CANDIDATES,
        "carrier_count": CARRIERS,
        "candidate_feature_names": candidate_names,
        "point_feature_names": point_names,
        "array_axis_fit": axis_record,
        "carrier_calibration": {
            str(key): value.to_dict() for key, value in calibration.items()
        },
        "candidate_count_mean": float(np.mean(mask.sum(axis=1))),
        "local_candidate_count_mean": float(np.mean(np.sum(source_type == 0, axis=1))),
        "map_candidate_count_mean": float(np.mean(np.sum(source_type == 1, axis=1))),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
        "seconds": float(time.time() - started),
    }
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return state


def target_state(cache: dict[str, object]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    query_ids = np.asarray(cache["query_ids"], dtype=np.int64)
    coordinate = np.asarray(cache["coordinate"], dtype=np.float32)
    mask = np.asarray(cache["mask"], dtype=bool)
    component = np.load(e114.COMPONENTS, allow_pickle=False)
    truth_power = component["point_power"][query_ids].astype(np.float64)
    targets, audit = e124.build_one_to_one_targets(
        coordinate,
        mask,
        component["coordinates"][query_ids].astype(np.float32),
        component["coefficients"][query_ids].astype(np.complex64),
        truth_power,
    )
    coefficient_power = (
        targets["mass_absolute"][..., None].astype(np.float64)
        * truth_power[:, None, None]
        * targets["port"].astype(np.float64)
    )
    coefficient = np.sqrt(np.maximum(coefficient_power, 0.0)).astype(np.float32)
    coefficient = coefficient * targets["phase"].astype(np.complex64)
    return {
        "positive": targets["mass_absolute"] > 0.0,
        "mass_absolute": targets["mass_absolute"].astype(np.float32),
        "coordinate": targets["coordinate"].astype(np.float32),
        "coefficient": coefficient.astype(np.complex64),
        "point_power": truth_power,
    }, audit


def render_coefficients(
    coordinate: torch.Tensor,
    coefficient: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    local_coefficient = torch.where(
        mask[..., None], coefficient, torch.zeros_like(coefficient)
    )
    atom = _component_atom(coordinate, torch.complex64, coordinate.device)
    shaped = local_coefficient.reshape(
        len(coordinate), coordinate.shape[1], e115.MP, e115.N_UE
    )
    channel = torch.einsum("bkpu,bkhvd->bpuhvd", shaped, atom)
    return channel.permute(0, 1, 3, 4, 2, 5).reshape(
        -1, e115.MP * e115.MH * e115.MV, e115.N_UE, e115.S_SC
    )


def aggregate_metric_arrays(
    values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> dict[str, float | int]:
    return statistics(*values)


@torch.inference_mode()
def score_prediction_rows(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(np.empty(len(prediction), dtype=np.float64) for _ in range(4))
    for start in range(0, len(prediction), batch_size):
        end = min(start + batch_size, len(prediction))
        values = e115.torch_official_metrics(prediction[start:end], truth[start:end])
        for destination, value in zip(arrays, values, strict=True):
            destination[start:end] = value.cpu().numpy()
    return arrays


@torch.inference_mode()
def oracle_residual(
    cache: dict[str, object],
    targets: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    count = len(targets["positive"])
    output = torch.empty(
        (count, e115.MP * e115.MH * e115.MV, e115.N_UE, e115.S_SC),
        dtype=torch.complex64,
        device=device,
    )
    source_type = np.asarray(cache["source_type"], dtype=np.int8)
    base_mask = np.asarray(cache["mask"], dtype=bool) & (source_type == 0)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        coordinate = torch.as_tensor(
            np.asarray(cache["coordinate"][start:end], dtype=np.float32), device=device
        )
        mask = torch.as_tensor(
            np.asarray(cache["mask"][start:end], dtype=bool), device=device
        )
        target_coordinate = torch.as_tensor(
            targets["coordinate"][start:end], device=device
        )
        target_coefficient = torch.as_tensor(
            targets["coefficient"][start:end], device=device
        )
        base_coefficient = torch.as_tensor(
            np.asarray(cache["v7_coefficient"][start:end], dtype=np.complex64),
            device=device,
        )
        local_mask = torch.as_tensor(base_mask[start:end], device=device)
        target_channel = render_coefficients(
            target_coordinate, target_coefficient, mask
        )
        base_channel = render_coefficients(coordinate, base_coefficient, local_mask)
        output[start:end] = target_channel - base_channel
    return output


def run_oracle_fold(
    outer: int,
    config: Config,
    device: torch.device,
) -> dict[str, object]:
    cache = load_cache(outer)
    targets, assignment = target_state(cache)
    query_ids = np.asarray(cache["query_ids"], dtype=np.int64)
    query_fold = np.asarray(cache["query_fold"], dtype=np.int8)
    inner = (outer + 1) % 5
    validation = np.flatnonzero(query_fold == inner)
    held = np.flatnonzero(query_fold == outer)
    v7_np = np.asarray(np.load(e148.V7_CHANNEL, mmap_mode="r"), dtype=np.complex64)
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    truth_np = np.asarray(truth_all[query_ids], dtype=np.complex64)
    v7 = torch.as_tensor(v7_np, device=device)
    truth = torch.as_tensor(truth_np, device=device)
    residual = oracle_residual(cache, targets, device, config.metric_batch_size)
    alpha_grid = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
    validation_rows = []
    for alpha in alpha_grid:
        values = score_prediction_rows(
            v7[validation] + float(alpha) * residual[validation],
            truth[validation],
            config.metric_batch_size,
        )
        validation_rows.append(
            {"alpha": float(alpha), "metrics": aggregate_metric_arrays(values)}
        )
    winner = max(validation_rows, key=lambda item: float(item["metrics"]["score"]))
    alpha = float(winner["alpha"])
    held_values = score_prediction_rows(
        v7[held] + alpha * residual[held],
        truth[held],
        config.metric_batch_size,
    )
    held_baseline_values = score_prediction_rows(
        v7[held], truth[held], config.metric_batch_size
    )
    held_metrics = aggregate_metric_arrays(held_values)
    held_baseline = aggregate_metric_arrays(held_baseline_values)
    return {
        "outer": outer,
        "inner": inner,
        "assignment": assignment,
        "validation_curve": validation_rows,
        "selected_alpha": alpha,
        "held_baseline": held_baseline,
        "held_oracle": held_metrics,
        "held_delta": float(held_metrics["score"] - held_baseline["score"]),
        "interface_ceiling_passed": bool(held_metrics["score"] > config.required_score),
    }


class FoldStore:
    def __init__(self, outer: int, device: torch.device) -> None:
        raw = load_cache(outer)
        self.outer = outer
        self.device = device
        self.state = raw["state"]
        self.query_ids_np = np.asarray(raw["query_ids"], dtype=np.int64)
        self.query_fold_np = np.asarray(raw["query_fold"], dtype=np.int8)
        self.mask = torch.as_tensor(np.asarray(raw["mask"], bool), device=device)
        self.coordinate = torch.as_tensor(
            np.asarray(raw["coordinate"], np.float32), device=device
        )
        self.source_type = torch.as_tensor(
            np.asarray(raw["source_type"], np.int64), device=device
        )
        self.candidate_feature = torch.as_tensor(
            np.asarray(raw["candidate_feature"], np.float32), device=device
        )
        self.point_feature = torch.as_tensor(
            np.asarray(raw["point_feature"], np.float32), device=device
        )
        self.carrier_coefficient = torch.as_tensor(
            np.asarray(raw["carrier_coefficient"], np.complex64), device=device
        )
        self.v7_coefficient = torch.as_tensor(
            np.asarray(raw["v7_coefficient"], np.complex64), device=device
        )
        self.carrier_log_power = torch.as_tensor(
            np.asarray(raw["carrier_log_power"], np.float32), device=device
        )
        self.v7_log_power = torch.as_tensor(
            np.asarray(raw["v7_log_power"], np.float32), device=device
        )
        targets, self.assignment = target_state(raw)
        self.target_positive = torch.as_tensor(targets["positive"], device=device)
        self.target_mass = torch.as_tensor(targets["mass_absolute"], device=device)
        self.target_coordinate = torch.as_tensor(targets["coordinate"], device=device)
        self.target_coefficient = torch.as_tensor(targets["coefficient"], device=device)
        self.target_point_power = torch.as_tensor(
            targets["point_power"], dtype=torch.float64, device=device
        )
        v7_np = np.asarray(np.load(e148.V7_CHANNEL, mmap_mode="r"), np.complex64)
        truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
        truth_np = np.asarray(truth_all[self.query_ids_np], np.complex64)
        self.v7 = torch.as_tensor(v7_np, device=device)
        self.truth = torch.as_tensor(truth_np, device=device)
        with torch.inference_mode():
            base_values = e115.torch_official_metrics(self.v7, self.truth)
        self.base_pas, self.base_pdp, self.base_error, self.base_energy = base_values

    def normalization(self, fit_rows: np.ndarray) -> dict[str, torch.Tensor]:
        rows = torch.as_tensor(fit_rows, device=self.device)
        valid = self.mask[rows]
        candidate_values = self.candidate_feature[rows][valid]
        candidate_mean = candidate_values.mean(dim=0)
        candidate_std = candidate_values.std(dim=0)
        candidate_std = torch.where(
            candidate_std < 1e-4, torch.ones_like(candidate_std), candidate_std
        )
        point_mean = self.point_feature[rows].mean(dim=0)
        point_std = self.point_feature[rows].std(dim=0)
        point_std = torch.where(point_std < 1e-4, torch.ones_like(point_std), point_std)
        return {
            "candidate_mean": candidate_mean,
            "candidate_std": candidate_std,
            "point_mean": point_mean,
            "point_std": point_std,
        }

    def batch(
        self,
        rows: torch.Tensor,
        normalization: dict[str, torch.Tensor],
        candidate_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        candidate = (
            self.candidate_feature[rows]
            if candidate_override is None
            else candidate_override
        )
        candidate = (candidate - normalization["candidate_mean"]) / normalization[
            "candidate_std"
        ]
        candidate = torch.where(
            self.mask[rows, :, None], candidate.clamp(-8.0, 8.0), torch.zeros_like(candidate)
        )
        point = (self.point_feature[rows] - normalization["point_mean"]) / normalization[
            "point_std"
        ]
        return {
            "rows": rows,
            "mask": self.mask[rows],
            "coordinate": self.coordinate[rows],
            "source_type": self.source_type[rows],
            "candidate_feature": candidate,
            "point_feature": point.clamp(-8.0, 8.0),
            "carrier_coefficient": self.carrier_coefficient[rows],
            "carrier_log_power": self.carrier_log_power[rows],
            "v7_coefficient": self.v7_coefficient[rows],
            "v7_log_power": self.v7_log_power[rows],
            "target_positive": self.target_positive[rows],
            "target_mass": self.target_mass[rows],
            "target_coordinate": self.target_coordinate[rows],
            "target_coefficient": self.target_coefficient[rows],
            "target_point_power": self.target_point_power[rows],
            "v7": self.v7[rows],
            "truth": self.truth[rows],
            "base_pas": self.base_pas[rows],
            "base_pdp": self.base_pdp[rows],
            "base_error": self.base_error[rows],
            "base_energy": self.base_energy[rows],
        }


class PathwiseCarrierOperator(nn.Module):
    def __init__(
        self,
        candidate_dim: int,
        point_dim: int,
        config: Config,
    ) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden
        carrier_hidden = config.carrier_hidden
        self.candidate_input = nn.Sequential(
            nn.Linear(candidate_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.point_input = nn.Sequential(
            nn.Linear(point_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        carrier_feature_dim = 1 + PORTS + 2 * PORTS + 1
        self.carrier_input = nn.Sequential(
            nn.Linear(carrier_feature_dim, carrier_hidden),
            nn.GELU(),
            nn.LayerNorm(carrier_hidden),
            nn.Linear(carrier_hidden, carrier_hidden),
        )
        self.carrier_embedding = nn.Parameter(
            torch.randn(CARRIERS, carrier_hidden) * 0.02
        )
        self.carrier_query = nn.Linear(hidden, carrier_hidden)
        self.carrier_key = nn.Linear(carrier_hidden, carrier_hidden)
        self.carrier_value = nn.Linear(carrier_hidden, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.heads,
            dim_feedforward=3 * hidden,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.path_encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.output_norm = nn.LayerNorm(hidden)
        self.existence_head = nn.Linear(hidden, 1)
        self.mix_logit_head = nn.Linear(hidden, CARRIERS)
        self.mix_phase_head = nn.Linear(hidden, CARRIERS)
        self.coordinate_head = nn.Linear(hidden, 3)
        self.path_gain_head = nn.Linear(hidden, 1)
        self.port_gain_head = nn.Linear(hidden, PORTS)
        self.port_phase_head = nn.Linear(hidden, PORTS)
        self.state_power_head = nn.Linear(hidden, 1)
        self.action_head = nn.Linear(hidden, 1)
        self.power_action_head = nn.Linear(hidden, 1)
        for layer_to_zero in (self.action_head, self.power_action_head):
            nn.init.zeros_(layer_to_zero.weight)
            nn.init.zeros_(layer_to_zero.bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mask = batch["mask"]
        point = self.point_input(batch["point_feature"])
        candidate = self.candidate_input(batch["candidate_feature"]) + point[:, None]
        coefficient = batch["carrier_coefficient"]
        energy = coefficient.abs().square().sum(dim=3).clamp_min(TINY)
        log_fraction = torch.log(energy) - batch["carrier_log_power"][:, None]
        port_amplitude = coefficient.abs() / torch.sqrt(energy[..., None])
        reference = batch.get("reference_coefficient", batch["v7_coefficient"])
        reference_unit = reference / reference.abs().clamp_min(1e-8)
        relative = coefficient / coefficient.abs().clamp_min(1e-8)
        relative = relative * torch.conj(reference_unit[:, :, None])
        power_center = batch["carrier_log_power"] - batch["carrier_log_power"].mean(
            dim=1, keepdim=True
        )
        carrier_feature = torch.cat(
            (
                log_fraction.clamp(-20.0, 5.0)[..., None],
                port_amplitude,
                relative.real,
                relative.imag,
                power_center[:, None, :, None].expand(-1, coefficient.shape[1], -1, 1),
            ),
            dim=3,
        )
        carrier = self.carrier_input(carrier_feature)
        carrier = carrier + self.carrier_embedding[None, None]
        query = self.carrier_query(candidate)
        key = self.carrier_key(carrier)
        attention = torch.einsum("bkh,bkch->bkc", query, key) / math.sqrt(
            self.config.carrier_hidden
        )
        attention = torch.softmax(attention, dim=2)
        pooled = torch.einsum(
            "bkc,bkch->bkh", attention, self.carrier_value(carrier)
        )
        hidden = self.path_encoder(candidate + pooled, src_key_padding_mask=~mask)
        hidden = self.output_norm(hidden)

        existence_logit = self.existence_head(hidden)[..., 0]
        mix_logit = self.mix_logit_head(hidden)
        mix_phase = math.pi * torch.tanh(self.mix_phase_head(hidden))
        mix_weight = torch.softmax(mix_logit, dim=2) * torch.exp(1j * mix_phase)
        mixed = torch.einsum("bkc,bkcp->bkp", mix_weight, coefficient)
        path_gain = self.config.path_gain_limit * torch.tanh(
            self.path_gain_head(hidden)[..., 0]
        )
        port_gain = self.config.port_gain_limit * torch.tanh(
            self.port_gain_head(hidden)
        )
        port_phase = self.config.phase_limit * torch.tanh(
            self.port_phase_head(hidden)
        )
        existence = torch.sigmoid(existence_logit)
        state_coefficient = (
            existence[..., None]
            * mixed
            * torch.exp(path_gain[..., None] + port_gain)
            * torch.exp(1j * port_phase)
        )
        coordinate_scale = torch.as_tensor(
            self.config.coordinate_limits, device=hidden.device, dtype=hidden.dtype
        )
        coordinate_delta = torch.tanh(self.coordinate_head(hidden)) * coordinate_scale
        state_coordinate = batch["coordinate"] + coordinate_delta
        action = self.config.action_limit * torch.tanh(self.action_head(hidden)[..., 0])
        action = action * mask.to(action.dtype)
        valid = mask.to(hidden.dtype)
        pooled_hidden = (hidden * valid[..., None]).sum(dim=1) / valid.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        state_power_delta = 2.0 * torch.tanh(self.state_power_head(pooled_hidden)[:, 0])
        power_action = self.config.power_limit * torch.tanh(
            self.power_action_head(pooled_hidden)[:, 0]
        )
        return {
            "existence_logit": existence_logit,
            "existence": existence,
            "mix_weight": mix_weight,
            "state_coefficient": state_coefficient,
            "state_coordinate": state_coordinate,
            "coordinate_delta": coordinate_delta,
            "path_gain": path_gain,
            "port_gain": port_gain,
            "port_phase": port_phase,
            "state_log_power": batch["v7_log_power"] + state_power_delta,
            "action": action,
            "power_action": power_action,
        }

    def render(
        self,
        batch: dict[str, torch.Tensor],
        output: dict[str, torch.Tensor],
        alpha: float,
    ) -> torch.Tensor:
        local_base = batch["source_type"] == 0
        base_coefficient = torch.where(
            local_base[..., None],
            batch["v7_coefficient"],
            torch.zeros_like(batch["v7_coefficient"]),
        )
        scale = float(alpha) * output["action"]
        modified_channel = render_coefficients(
            output["state_coordinate"],
            output["state_coefficient"] * scale[..., None],
            batch["mask"],
        )
        base_channel = render_coefficients(
            batch["coordinate"],
            base_coefficient * scale[..., None],
            batch["mask"],
        )
        prediction = batch["v7"] + modified_channel - base_channel
        power_scale = torch.exp(0.5 * float(alpha) * output["power_action"])
        active = batch["v7"].abs().square().sum(dim=(1, 2, 3)) > 0.0
        prediction = prediction * power_scale[:, None, None, None]
        return torch.where(
            active[:, None, None, None], prediction, batch["v7"]
        )


def state_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["mask"]
    positive = batch["target_positive"] & mask
    negative = (~batch["target_positive"]) & mask
    positive_count = positive.sum().clamp_min(1)
    negative_count = negative.sum().clamp_min(1)
    existence = (
        F.binary_cross_entropy_with_logits(
            output["existence_logit"][positive],
            torch.ones_like(output["existence_logit"][positive]),
        )
        if positive.any()
        else output["existence_logit"].sum() * 0.0
    )
    if negative.any():
        existence = existence + 0.35 * F.binary_cross_entropy_with_logits(
            output["existence_logit"][negative],
            torch.zeros_like(output["existence_logit"][negative]),
        )
    if positive.any():
        periods = output["state_coordinate"].new_tensor(e115.PERIODS)
        coordinate_error = torch.remainder(
            output["state_coordinate"][positive]
            - batch["target_coordinate"][positive]
            + periods / 2.0,
            periods,
        ) - periods / 2.0
        coordinate_scale = output["state_coordinate"].new_tensor((1.5, 1.0, 2.0))
        coordinate = F.smooth_l1_loss(
            coordinate_error / coordinate_scale, torch.zeros_like(coordinate_error)
        )
        predicted = output["state_coefficient"][positive]
        target = batch["target_coefficient"][positive]
        predicted_energy = predicted.abs().square().sum(dim=1).clamp_min(TINY)
        target_energy = target.abs().square().sum(dim=1).clamp_min(TINY)
        amplitude = F.smooth_l1_loss(
            torch.log(predicted_energy), torch.log(target_energy)
        )
        predicted_port = predicted.abs().square() / predicted_energy[:, None]
        target_port = target.abs().square() / target_energy[:, None]
        port = 1.0 - (
            torch.sqrt(predicted_port.clamp_min(1e-12))
            * torch.sqrt(target_port.clamp_min(1e-12))
        ).sum(dim=1).mean()
        phase = torch.real(
            predicted
            / predicted.abs().clamp_min(1e-8)
            * torch.conj(target / target.abs().clamp_min(1e-8))
        )
        phase = 1.0 - (phase * target_port).sum(dim=1).mean()
        coefficient = 0.35 * amplitude + 0.25 * port + 0.40 * phase
    else:
        coordinate = output["state_coordinate"].sum() * 0.0
        coefficient = output["state_coefficient"].real.sum() * 0.0
    valid_power = batch["target_point_power"] > 0.0
    if valid_power.any():
        power = F.smooth_l1_loss(
            output["state_log_power"][valid_power].double(),
            torch.log(batch["target_point_power"][valid_power]),
        )
    else:
        power = output["state_log_power"].sum() * 0.0
    loss = (
        config.identity_weight * existence
        + config.coordinate_weight * coordinate
        + config.coefficient_weight * coefficient
        + config.power_weight * power
    )
    return loss, {
        "state_loss": float(loss.detach().cpu()),
        "existence": float(existence.detach().cpu()),
        "coordinate": float(coordinate.detach().cpu()),
        "coefficient": float(coefficient.detach().cpu()),
        "power": float(power.detach().cpu()),
        "positive_count": int(positive_count.detach().cpu()),
        "negative_count": int(negative_count.detach().cpu()),
    }


def stable_clip_gradients(
    parameters,
    maximum_norm: float,
) -> tuple[float, bool]:
    gradients = [
        parameter.grad
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return 0.0, False
    sanitized = False
    norm_squared = torch.zeros((), dtype=torch.float64, device=gradients[0].device)
    for gradient in gradients:
        if not torch.isfinite(gradient).all():
            torch.nan_to_num_(gradient, nan=0.0, posinf=0.0, neginf=0.0)
            sanitized = True
        norm_squared += gradient.double().square().sum()
    norm = torch.sqrt(norm_squared)
    scale = min(1.0, maximum_norm / max(float(norm.detach().cpu()), 1e-30))
    if scale < 1.0:
        for gradient in gradients:
            gradient.mul_(scale)
    return float(norm.detach().cpu()), sanitized


def official_loss(
    model: PathwiseCarrierOperator,
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = model.render(batch, output, 1.0)
    pas, pdp, error, energy = e115.torch_official_metrics(prediction, batch["truth"])
    pas_mean = pas.mean(dtype=torch.float64)
    pdp_mean = pdp.mean(dtype=torch.float64)
    nmse = error.sum(dtype=torch.float64) / energy.sum(dtype=torch.float64).clamp_min(TINY)
    score = 0.4 * pas_mean + 0.4 * pdp_mean + 0.2 / (1.0 + nmse)
    base_pas = batch["base_pas"].mean(dtype=torch.float64)
    base_pdp = batch["base_pdp"].mean(dtype=torch.float64)
    base_nmse = batch["base_error"].sum(dtype=torch.float64) / batch[
        "base_energy"
    ].sum(dtype=torch.float64).clamp_min(TINY)
    barrier = (
        F.relu(base_pas - pas_mean)
        + F.relu(base_pdp - pdp_mean)
        + 0.2 * F.relu(nmse - base_nmse)
    )
    trust = output["action"].square().mean() + output["power_action"].square().mean()
    loss = (
        -config.official_weight * score
        + config.component_barrier_weight * barrier
        + config.trust_weight * trust.double()
    )
    return loss, {
        "score": float(score.detach().cpu()),
        "pas": float(pas_mean.detach().cpu()),
        "pdp": float(pdp_mean.detach().cpu()),
        "nmse": float(nmse.detach().cpu()),
        "barrier": float(barrier.detach().cpu()),
        "trust": float(trust.detach().cpu()),
    }


@torch.inference_mode()
def evaluate_state_rows(
    model: PathwiseCarrierOperator,
    store: FoldStore,
    rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    config: Config,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, list[float]] = {
        "state_loss": [],
        "existence": [],
        "coordinate": [],
        "coefficient": [],
        "power": [],
    }
    for start in range(0, len(rows), config.batch_size):
        local = torch.as_tensor(rows[start : start + config.batch_size], device=store.device)
        batch = store.batch(local, normalization)
        output = model(batch)
        _, diagnostics = state_loss(output, batch, config)
        for name in totals:
            totals[name].append(float(diagnostics[name]))
    return {name: float(np.mean(value)) for name, value in totals.items()}


@torch.inference_mode()
def evaluate_model_rows(
    model: PathwiseCarrierOperator,
    store: FoldStore,
    rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    alpha: float,
    config: Config,
    candidate_override: torch.Tensor | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], dict[str, float]]:
    model.eval()
    arrays = tuple(np.empty(len(rows), dtype=np.float64) for _ in range(4))
    action_parts: list[np.ndarray] = []
    power_parts: list[np.ndarray] = []
    existence_parts: list[np.ndarray] = []
    for start in range(0, len(rows), config.metric_batch_size):
        end = min(start + config.metric_batch_size, len(rows))
        local = torch.as_tensor(rows[start:end], device=store.device)
        override = None
        if candidate_override is not None:
            override = candidate_override[start:end]
        batch = store.batch(local, normalization, override)
        output = model(batch)
        prediction = model.render(batch, output, alpha)
        values = e115.torch_official_metrics(prediction, batch["truth"])
        for destination, value in zip(arrays, values, strict=True):
            destination[start:end] = value.cpu().numpy()
        valid = batch["mask"]
        action_parts.append(output["action"][valid].cpu().numpy())
        power_parts.append(output["power_action"].cpu().numpy())
        existence_parts.append(output["existence"][valid].cpu().numpy())
    diagnostics = {
        "action_abs_mean": float(np.mean(np.abs(np.concatenate(action_parts))))
        if action_parts
        else 0.0,
        "power_action_abs_mean": float(np.mean(np.abs(np.concatenate(power_parts))))
        if power_parts
        else 0.0,
        "existence_mean": float(np.mean(np.concatenate(existence_parts)))
        if existence_parts
        else 0.0,
    }
    return arrays, diagnostics


def train_pretrain(
    model: PathwiseCarrierOperator,
    store: FoldStore,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    config: Config,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    for parameter in model.action_head.parameters():
        parameter.requires_grad = False
    for parameter in model.power_action_head.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.pretrain_epochs, 1), eta_min=config.learning_rate * 0.05
    )
    rng = np.random.default_rng(seed)
    best = math.inf
    best_epoch = 0
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    history: list[dict[str, object]] = []
    for epoch in range(1, config.pretrain_epochs + 1):
        model.train()
        order = rng.permutation(fit_rows)
        losses = []
        for start in range(0, len(order), config.batch_size):
            local = torch.as_tensor(order[start : start + config.batch_size], device=store.device)
            batch = store.batch(local, normalization)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss, _ = state_loss(output, batch, config)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite E150 state loss epoch={epoch}")
            loss.backward()
            stable_clip_gradients(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        row: dict[str, object] = {
            "epoch": epoch,
            "train_state_loss": float(np.mean(losses)),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        if epoch == 1 or epoch % config.validation_interval == 0 or epoch == config.pretrain_epochs:
            validation = evaluate_state_rows(
                model, store, validation_rows, normalization, config
            )
            row["validation"] = validation
            if validation["state_loss"] < best - 1e-9:
                best = validation["state_loss"]
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        history.append(row)
        if epoch == 1 or epoch % 2 == 0 or epoch == config.pretrain_epochs:
            message = (
                f"E150 outer={store.outer} pretrain={epoch}/{config.pretrain_epochs} "
                f"train={row['train_state_loss']:.6f}"
            )
            if "validation" in row:
                message += f" inner={row['validation']['state_loss']:.6f}"
            print(message, flush=True)
    model.load_state_dict(best_state)
    for parameter in model.action_head.parameters():
        parameter.requires_grad = True
    for parameter in model.power_action_head.parameters():
        parameter.requires_grad = True
    return best_state, {
        "best_epoch": best_epoch,
        "best_validation_state_loss": best,
        "history": history,
    }


def train_finetune(
    model: PathwiseCarrierOperator,
    store: FoldStore,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    config: Config,
    seed: int,
) -> tuple[dict[str, torch.Tensor], float, dict[str, object]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.finetune_learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.finetune_epochs, 1),
        eta_min=config.finetune_learning_rate * 0.05,
    )
    alpha_grid = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
    baseline_values = tuple(
        value[validation_rows].cpu().numpy()
        for value in (store.base_pas, store.base_pdp, store.base_error, store.base_energy)
    )
    baseline = statistics(*baseline_values)
    best_score = float(baseline["score"])
    best_epoch = 0
    best_alpha = 0.0
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    history: list[dict[str, object]] = [
        {
            "epoch": 0,
            "selected_alpha": 0.0,
            "selected_validation": baseline,
        }
    ]
    rng = np.random.default_rng(seed)
    sanitized_gradient_steps = 0
    for epoch in range(1, config.finetune_epochs + 1):
        model.train()
        order = rng.permutation(fit_rows)
        losses = []
        scores = []
        for start in range(0, len(order), config.batch_size):
            local = torch.as_tensor(order[start : start + config.batch_size], device=store.device)
            batch = store.batch(local, normalization)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            supervised, _ = state_loss(output, batch, config)
            score_loss, diagnostics = official_loss(model, output, batch, config)
            loss = score_loss + 0.20 * supervised
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite E150 official loss epoch={epoch}")
            loss.backward()
            _, sanitized = stable_clip_gradients(model.parameters(), 1.0)
            if sanitized:
                sanitized_gradient_steps += 1
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            scores.append(diagnostics["score"])
        scheduler.step()
        row: dict[str, object] = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_batch_score": float(np.mean(scores)),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        if epoch == 1 or epoch % config.validation_interval == 0 or epoch == config.finetune_epochs:
            curve = []
            for alpha in alpha_grid:
                values, diagnostics = evaluate_model_rows(
                    model,
                    store,
                    validation_rows,
                    normalization,
                    alpha,
                    config,
                )
                curve.append(
                    {
                        "alpha": float(alpha),
                        "metrics": statistics(*values),
                        "diagnostics": diagnostics,
                    }
                )
            winner = max(curve, key=lambda item: float(item["metrics"]["score"]))
            row["alpha_curve"] = curve
            row["selected_alpha"] = float(winner["alpha"])
            row["selected_validation"] = winner["metrics"]
            score = float(winner["metrics"]["score"])
            if score > best_score + 1e-10:
                best_score = score
                best_epoch = epoch
                best_alpha = float(winner["alpha"])
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        history.append(row)
        if epoch == 1 or epoch % 2 == 0 or epoch == config.finetune_epochs:
            message = (
                f"E150 outer={store.outer} finetune={epoch}/{config.finetune_epochs} "
                f"train={row['train_batch_score']:.6f}"
            )
            if "selected_validation" in row:
                message += (
                    f" inner={row['selected_validation']['score']:.6f} "
                    f"alpha={row['selected_alpha']:.2f}"
                )
            print(message, flush=True)
    model.load_state_dict(best_state)
    return best_state, best_alpha, {
        "best_epoch": best_epoch,
        "best_alpha": best_alpha,
        "best_validation_score": best_score,
        "sanitized_gradient_steps": sanitized_gradient_steps,
        "history": history,
    }


def geometry_control_override(
    store: FoldStore,
    held_rows: np.ndarray,
    seed: int,
    strata: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    setup, positions, _, site, _ = load_positions()
    query_positions = positions[store.query_ids_np[held_rows]]
    query_site = site[store.query_ids_np[held_rows]].astype(np.int8)
    permutation, diagnostics = e148.mismatch_permutation(
        query_positions,
        query_site,
        setup.bs_positions,
        seed,
        strata,
    )
    names = store.state["candidate_feature_names"]
    geometry_start = names.index("log1p_map_match_cost")
    source = store.candidate_feature[
        torch.as_tensor(held_rows[permutation], device=store.device)
    ]
    target = store.candidate_feature[
        torch.as_tensor(held_rows, device=store.device)
    ].clone()
    target[..., geometry_start:] = source[..., geometry_start:]
    return target, diagnostics


def train_fold(
    outer: int,
    config: Config,
    device: torch.device,
) -> tuple[
    dict[str, object],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    inner = (outer + 1) % 5
    store = FoldStore(outer, device)
    fit_rows = np.flatnonzero(
        (store.query_fold_np != outer) & (store.query_fold_np != inner)
    )
    validation_rows = np.flatnonzero(store.query_fold_np == inner)
    held_rows = np.flatnonzero(store.query_fold_np == outer)
    normalization = store.normalization(fit_rows)
    torch.manual_seed(config.seed + outer * 1009)
    model = PathwiseCarrierOperator(
        store.candidate_feature.shape[-1], store.point_feature.shape[-1], config
    ).to(device)
    zero_values, _ = evaluate_model_rows(
        model, store, held_rows, normalization, 0.0, config
    )
    zero_summary = statistics(*zero_values)
    expected = statistics(
        *(value[held_rows].cpu().numpy() for value in (
            store.base_pas,
            store.base_pdp,
            store.base_error,
            store.base_energy,
        ))
    )
    if abs(float(zero_summary["score"]) - float(expected["score"])) > 1e-10:
        raise RuntimeError(
            f"E150 zero initialization mismatch outer={outer}: "
            f"{zero_summary['score']} != {expected['score']}"
        )
    _, pretrain_record = train_pretrain(
        model,
        store,
        fit_rows,
        validation_rows,
        normalization,
        config,
        config.seed + outer * 2027,
    )
    best_state, alpha, finetune_record = train_finetune(
        model,
        store,
        fit_rows,
        validation_rows,
        normalization,
        config,
        config.seed + outer * 4099,
    )
    selected_values, selected_diagnostics = evaluate_model_rows(
        model, store, held_rows, normalization, alpha, config
    )
    selected = statistics(*selected_values)
    baseline = expected
    override, control_record = geometry_control_override(
        store,
        held_rows,
        config.seed + outer * 8191,
        config.geometry_control_strata,
    )
    control_values, control_diagnostics = evaluate_model_rows(
        model,
        store,
        held_rows,
        normalization,
        alpha,
        config,
        override,
    )
    control = statistics(*control_values)
    checkpoint = checkpoint_path(outer)
    torch.save(
        {
            "experiment": VERSION,
            "outer": outer,
            "inner": inner,
            "config": asdict(config),
            "state_dict": best_state,
            "normalization": {
                key: value.detach().cpu() for key, value in normalization.items()
            },
            "selected_alpha": alpha,
            "candidate_feature_names": store.state["candidate_feature_names"],
            "point_feature_names": store.state["point_feature_names"],
            "official_h_only": True,
            "sionna_used": False,
            "v12_used": False,
        },
        checkpoint,
    )
    record = {
        "outer": outer,
        "inner": inner,
        "fit_count": int(len(fit_rows)),
        "validation_count": int(len(validation_rows)),
        "held_count": int(len(held_rows)),
        "assignment": store.assignment,
        "pretrain": pretrain_record,
        "finetune": finetune_record,
        "selected_alpha": alpha,
        "held_baseline": baseline,
        "held_selected": selected,
        "held_delta": float(selected["score"] - baseline["score"]),
        "selected_diagnostics": selected_diagnostics,
        "geometry_control": {
            **control_record,
            "metrics": control,
            "diagnostics": control_diagnostics,
        },
        "geometry_margin": float(selected["score"] - control["score"]),
        "checkpoint": PATHS.relative(checkpoint),
    }
    print(
        f"E150 outer={outer} held={selected['score']:.6f} "
        f"delta={record['held_delta']:+.6f} "
        f"geometry_margin={record['geometry_margin']:+.6f}",
        flush=True,
    )
    del model, store
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record, selected_values, control_values


def full_baseline(config: Config, device: torch.device) -> dict[str, object]:
    metadata = np.load(e119.OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    v7 = np.asarray(np.load(e148.V7_CHANNEL, mmap_mode="r"), np.complex64)
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    truth = np.asarray(truth_all[query_ids], np.complex64)
    values = e148.metric_arrays(v7, truth, device, config.metric_batch_size)
    summary = statistics(*values)
    expected = 0.644456345600551
    if abs(float(summary["score"]) - expected) > 1e-9:
        raise RuntimeError(f"V7 baseline mismatch: {summary['score']} != {expected}")
    return {"query_ids": query_ids, "values": values, "metrics": summary}


def smoke_test(config: Config, device: torch.device) -> None:
    model = PathwiseCarrierOperator(97, 14, config).to(device)
    batch = {
        "mask": torch.ones(2, TOTAL_CANDIDATES, dtype=torch.bool, device=device),
        "coordinate": torch.zeros(2, TOTAL_CANDIDATES, 3, device=device),
        "source_type": torch.zeros(2, TOTAL_CANDIDATES, dtype=torch.long, device=device),
        "candidate_feature": torch.randn(2, TOTAL_CANDIDATES, 97, device=device),
        "point_feature": torch.randn(2, 14, device=device),
        "carrier_coefficient": torch.randn(
            2, TOTAL_CANDIDATES, CARRIERS, PORTS, device=device
        ).to(torch.complex64),
        "carrier_log_power": torch.zeros(2, CARRIERS, device=device),
        "v7_coefficient": torch.randn(2, TOTAL_CANDIDATES, PORTS, device=device).to(
            torch.complex64
        ),
        "v7_log_power": torch.zeros(2, device=device),
        "v7": torch.randn(2, 256, 4, 192, device=device).to(torch.complex64),
    }
    output = model(batch)
    prediction = model.render(batch, output, 0.0)
    maximum = float((prediction - batch["v7"]).abs().max().cpu())
    if maximum != 0.0:
        raise RuntimeError(f"zero initialization smoke mismatch: {maximum}")
    prediction = model.render(batch, output, 1.0)
    if prediction.shape != (2, 256, 4, 192) or not torch.isfinite(prediction).all():
        raise RuntimeError("invalid E150 smoke prediction")
    print(
        json.dumps(
            {
                "status": "smoke_passed",
                "zero_initialization_max_error": maximum,
                "prediction_shape": list(prediction.shape),
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    config = Config(
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        metric_batch_size=args.metric_batch_size,
    )
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed % (2**32 - 1))
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    required = (
        TRAIN_CHANNEL,
        e148.V7_CHANNEL,
        e114.COMPONENTS,
        e146.CANDIDATE_CACHE,
        e147.METADATA_CACHE,
        e146.LOS_LABELS,
        e115.FULL_FOLD,
        e119.OOF_META,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.smoke:
        smoke_test(config, device)
        return

    outers = selected_outers(args.outer)
    cache_records = []
    if args.phase in ("cache", "all"):
        for outer in outers:
            cache_records.append(
                build_cache(outer, config, device, args.force_cache)
            )

    oracle_records = []
    if args.phase in ("oracle", "all"):
        for outer in outers:
            if not (cache_dir(outer) / "state.json").is_file():
                build_cache(outer, config, device, args.force_cache)
            record = run_oracle_fold(outer, config, device)
            oracle_records.append(record)
            print(
                f"E150 oracle outer={outer} held={record['held_oracle']['score']:.6f} "
                f"delta={record['held_delta']:+.6f}",
                flush=True,
            )

    fold_records = []
    selected_values = tuple(np.empty(500, dtype=np.float64) for _ in range(4))
    control_values = tuple(np.empty(500, dtype=np.float64) for _ in range(4))
    baseline = full_baseline(config, device)
    if args.phase in ("train", "all"):
        fold_all = np.load(e115.FULL_FOLD).astype(np.int8)
        query_fold = fold_all[baseline["query_ids"]]
        for outer in outers:
            if not (cache_dir(outer) / "state.json").is_file():
                build_cache(outer, config, device, args.force_cache)
            record, values, control = train_fold(outer, config, device)
            held = np.flatnonzero(query_fold == outer)
            for destination, value in zip(selected_values, values, strict=True):
                destination[held] = value
            for destination, value in zip(control_values, control, strict=True):
                destination[held] = value
            fold_records.append(record)

    complete = len(outers) == 5 and len(fold_records) == 5
    selected = statistics(*selected_values) if complete else None
    control = statistics(*control_values) if complete else None
    gate = None
    if complete and selected is not None and control is not None:
        fold_delta = [float(record["held_delta"]) for record in fold_records]
        gate = {
            "required_fixed500_score": config.required_score,
            "required_v7_gain": config.required_v7_gain,
            "required_positive_folds": config.required_positive_folds,
            "required_geometry_margin": config.required_geometry_margin,
            "fixed500_score": float(selected["score"]),
            "v7_gain": float(selected["score"] - baseline["metrics"]["score"]),
            "positive_folds": int(sum(value > 0.0 for value in fold_delta)),
            "minimum_fold_delta": float(min(fold_delta)),
            "geometry_margin": float(selected["score"] - control["score"]),
            "pas_non_worse": bool(selected["pas"] >= baseline["metrics"]["pas"]),
            "pdp_non_worse": bool(selected["pdp"] >= baseline["metrics"]["pdp"]),
            "nmse_non_worse": bool(selected["nmse"] <= baseline["metrics"]["nmse"]),
        }
        gate["passed"] = bool(
            gate["fixed500_score"] > gate["required_fixed500_score"]
            and gate["v7_gain"] >= gate["required_v7_gain"]
            and gate["positive_folds"] >= gate["required_positive_folds"]
            and gate["geometry_margin"] >= gate["required_geometry_margin"]
            and gate["pas_non_worse"]
            and gate["pdp_non_worse"]
            and gate["nmse_non_worse"]
        )
        np.savez_compressed(
            METRICS,
            query_ids=baseline["query_ids"].astype(np.int32),
            baseline_pas=baseline["values"][0],
            baseline_pdp=baseline["values"][1],
            baseline_error=baseline["values"][2],
            baseline_energy=baseline["values"][3],
            selected_pas=selected_values[0],
            selected_pdp=selected_values[1],
            selected_error=selected_values[2],
            selected_energy=selected_values[3],
            control_pas=control_values[0],
            control_pdp=control_values[1],
            control_error=control_values[2],
            control_energy=control_values[3],
        )

    report = {
        "experiment": VERSION,
        "status": (
            "analysis_only_positive"
            if gate is not None and gate["passed"]
            else "analysis_only_negative"
            if gate is not None
            else "partial_screen_complete"
        ),
        "final_goal": (
            "Official-input-only end-to-end test H, fixed500 above 0.677521 and "
            "Round 2 online above 0.70"
        ),
        "single_experimental_goal": (
            "Upgrade V7 from whole-channel interpolation to pathwise transport using "
            "strict 17-carrier evidence, multiheight map candidates, one-to-one path "
            "state supervision, analytic H synthesis, and direct official-score training."
        ),
        "protocol": {
            "canonical_fixed500": True,
            "outer_inner_excluded_from_carrier_donors": True,
            "fit_queries_exclude_self_neighbor": True,
            "inner_selects_state_epoch_official_epoch_and_alpha": True,
            "outer_used_once_for_final_scoring": True,
            "strict_zero_initialization_to_v7": True,
            "official_h_only": True,
            "sionna_used": False,
            "v12_used": False,
            "submission_generated": False,
        },
        "config": asdict(config),
        "cache_records": cache_records,
        "oracle_interface": oracle_records,
        "baseline_v7": baseline["metrics"],
        "folds": fold_records,
        "selected": selected,
        "geometry_control": control,
        "gate": gate,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "baseline_v7": baseline["metrics"],
                "oracle_interface": oracle_records,
                "selected": selected,
                "geometry_control": control,
                "gate": gate,
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
