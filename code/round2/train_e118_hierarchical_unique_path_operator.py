#!/usr/bin/env python3
"""Train a hierarchical operator from raw evidence tokens to unique paths."""

from __future__ import annotations

import argparse
import copy
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

import analyze_e117_unique_path_cluster_ceiling as e117
import analyze_global_local_path_lifecycle as e79
import train_e115_complete_path_state_operator as e115
import train_e116_nested_path_state_operator as e116
from data import load_positions
from project_paths import PATHS


VERSION = "Round2-E118-hierarchical-unique-path-operator"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "hierarchical_unique_path_operator"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_hierarchical_unique_path_operator"
SCREEN_REPORT = ANALYSIS_DIR / "hierarchical_unique_path_outer0_screen.json"
STATE_NAMES = ("mass", "coordinate", "port", "phase")
GRAPH_DIM = 34


@dataclass(frozen=True)
class Config(e115.Config):
    candidate_count: int = 64
    token_layers: int = 2
    cluster_threshold: float = 0.15
    graph_neighbors: int = 4
    candidate_layers: int = 2
    zero_weight: float = 0.0
    power_weight: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare-screen", "screen"), default="prepare-screen"
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--core-epochs", type=int, default=28)
    parser.add_argument("--scene-epochs", type=int, default=20)
    parser.add_argument("--graph-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def cache_dir(tag: str = "screen_block01") -> Path:
    return ARTIFACT_DIR / tag


def array_path(directory: Path, name: str) -> Path:
    return directory / f"{name}.npy"


def cluster_groups(
    coordinate: np.ndarray,
    base_mass: np.ndarray,
    threshold: float,
    max_clusters: int,
) -> tuple[list[np.ndarray], np.ndarray, int]:
    order = np.argsort(-base_mass, kind="stable")
    cost = e79.support_cost(coordinate, coordinate)
    assigned = np.zeros(len(coordinate), dtype=bool)
    groups: list[np.ndarray] = []
    for leader in order:
        if assigned[leader]:
            continue
        group = np.flatnonzero((cost[leader] <= threshold) & ~assigned)
        assigned[group] = True
        groups.append(group)
    mass = np.asarray(
        [np.sum(base_mass[group], dtype=np.float64) for group in groups]
    )
    selected = np.argsort(-mass, kind="stable")[:max_clusters]
    return [groups[int(index)] for index in selected], mass[selected], len(groups)


def normalized_complex_average(
    value: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    summed = np.sum(weight * value, axis=0, dtype=np.complex128)
    magnitude = np.abs(summed)
    return np.divide(
        summed,
        np.maximum(magnitude, 1e-12),
        out=np.ones_like(summed, dtype=np.complex128),
    ).astype(np.complex64)


def cluster_feature(
    coordinate: np.ndarray,
    mass: float,
    port: np.ndarray,
    phase: np.ndarray,
    member_coordinate: np.ndarray,
    member_mass: np.ndarray,
    raw_token_count: int,
) -> np.ndarray:
    angle = 2.0 * np.pi * coordinate / e115.PERIODS
    delta = np.stack(
        tuple(
            e79.periodic_delta(
                member_coordinate[:, axis],
                np.full(len(member_coordinate), coordinate[axis], np.float32),
                float(e115.PERIODS[axis]),
            )
            for axis in range(3)
        ),
        axis=1,
    )
    total = max(float(np.sum(member_mass, dtype=np.float64)), e115.TINY)
    spread = np.sqrt(
        np.sum(member_mass[:, None] * np.square(delta), axis=0, dtype=np.float64)
        / total
    ) / np.asarray((1.5, 1.0, 3.0), np.float64)
    concentration = float(np.max(member_mass, initial=0.0) / total)
    return np.concatenate(
        (
            np.sin(angle),
            np.cos(angle),
            coordinate / e115.PERIODS,
            np.asarray(
                (
                    math.log(max(mass, 1e-12)),
                    math.sqrt(max(mass, 0.0)),
                    len(member_coordinate) / max(raw_token_count, 1),
                ),
                np.float32,
            ),
            spread.astype(np.float32),
            np.asarray((concentration,), np.float32),
            np.sqrt(np.maximum(port, 0.0)),
            phase.real,
            phase.imag,
        )
    ).astype(np.float32)


def build_graph_features(
    directory: Path,
    positions: np.ndarray,
    site: np.ndarray,
    fold_id: np.ndarray,
    config: Config,
) -> dict[str, object]:
    coordinate = np.load(array_path(directory, "coordinate"), mmap_mode="r")
    base_log_mass = np.load(array_path(directory, "base_log_mass"), mmap_mode="r")
    base_log_port = np.load(array_path(directory, "base_log_port"), mmap_mode="r")
    base_phase = np.load(array_path(directory, "base_phase"), mmap_mode="r")
    member_count = np.load(array_path(directory, "member_count"), mmap_mode="r")
    mask = np.load(array_path(directory, "mask"), mmap_mode="r")
    output = np.lib.format.open_memmap(
        array_path(directory, "graph_feature"),
        mode="w+",
        dtype=np.float16,
        shape=(
            len(positions),
            config.candidate_count,
            config.graph_neighbors,
            GRAPH_DIM,
        ),
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
                neighbors = rows[local[local_row, 1:take].astype(np.int64)]
                if len(neighbors) < config.graph_neighbors:
                    neighbors = np.pad(
                        neighbors,
                        (0, config.graph_neighbors - len(neighbors)),
                        mode="edge",
                    )
                neighbor_ids[query_id] = neighbors[: config.graph_neighbors]
                query_valid = np.flatnonzero(mask[query_id])
                query_coordinate = np.asarray(coordinate[query_id], np.float32)
                for graph_rank, neighbor_id in enumerate(neighbor_ids[query_id]):
                    local_feature = np.zeros(
                        (config.candidate_count, GRAPH_DIM), np.float32
                    )
                    neighbor_valid = np.flatnonzero(mask[neighbor_id])
                    if len(query_valid) > 0 and len(neighbor_valid) > 0:
                        cost = e79.support_cost(
                            query_coordinate[query_valid],
                            np.asarray(coordinate[neighbor_id, neighbor_valid], np.float32),
                        )
                        column = np.argmin(cost, axis=1)
                        matched = neighbor_valid[column]
                        best_cost = cost[np.arange(len(query_valid)), column]
                        delta = np.stack(
                            tuple(
                                e79.periodic_delta(
                                    np.asarray(coordinate[neighbor_id, matched, axis]),
                                    query_coordinate[query_valid, axis],
                                    float(e115.PERIODS[axis]),
                                )
                                for axis in range(3)
                            ),
                            axis=1,
                        )
                        offset = positions[neighbor_id, :2] - positions[query_id, :2]
                        separation = float(np.linalg.norm(offset))
                        local_feature[query_valid, 0:2] = offset / 30.0
                        local_feature[query_valid, 2] = separation / 30.0
                        local_feature[query_valid, 3:6] = delta / np.asarray(
                            (1.5, 1.0, 3.0), np.float32
                        )
                        local_feature[query_valid, 6] = np.log1p(best_cost) / 4.0
                        local_feature[query_valid, 7] = np.asarray(
                            base_log_mass[neighbor_id, matched], np.float32
                        ) / 20.0
                        local_feature[query_valid, 8] = np.asarray(
                            member_count[neighbor_id, matched], np.float32
                        ) / 176.0
                        local_feature[query_valid, 9:17] = np.sqrt(
                            np.exp(
                                np.asarray(
                                    base_log_port[neighbor_id, matched], np.float32
                                )
                            )
                        )
                        local_feature[query_valid, 17:33] = np.asarray(
                            base_phase[neighbor_id, matched], np.float32
                        )
                        local_feature[query_valid, 33] = 1.0
                    output[query_id, :, graph_rank] = local_feature.astype(np.float16)
    output.flush()
    np.save(array_path(directory, "graph_neighbor_ids"), neighbor_ids)
    return {
        "graph_feature_shape": list(output.shape),
        "graph_neighbor_shape": list(neighbor_ids.shape),
    }


def build_cache(
    config: Config,
    force: bool,
    tag: str = "screen_block01",
    source_tag: str = "screen_block01",
) -> dict[str, object]:
    directory = cache_dir(tag)
    state_path = directory / "state.json"
    if state_path.is_file() and not force:
        return json.loads(state_path.read_text(encoding="utf-8"))
    directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    source = e115.cache_dir(source_tag)
    coordinate = np.asarray(
        np.load(e115.array_path(source, "coordinate"), mmap_mode="r"), np.float32
    )
    raw_log_mass = np.asarray(
        np.load(e115.array_path(source, "base_log_mass"), mmap_mode="r"), np.float32
    )
    raw_mass = np.exp(raw_log_mass)
    raw_log_port = np.asarray(
        np.load(e115.array_path(source, "base_log_port"), mmap_mode="r"), np.float32
    )
    raw_port = np.exp(raw_log_port)
    raw_phase_parts = np.asarray(
        np.load(e115.array_path(source, "base_phase"), mmap_mode="r"), np.float32
    )
    raw_phase = raw_phase_parts[..., : e115.PORTS] + 1j * raw_phase_parts[
        ..., e115.PORTS :
    ]
    target_mass = np.asarray(
        np.load(e115.array_path(source, "target_mass"), mmap_mode="r"), np.float32
    )
    target_delta = np.asarray(
        np.load(e115.array_path(source, "target_delta"), mmap_mode="r"), np.float32
    )
    target_port = np.asarray(
        np.load(e115.array_path(source, "target_port"), mmap_mode="r"), np.float32
    )
    target_phase_parts = np.asarray(
        np.load(e115.array_path(source, "target_phase"), mmap_mode="r"), np.float32
    )
    target_phase = target_phase_parts[..., : e115.PORTS] + 1j * target_phase_parts[
        ..., e115.PORTS :
    ]
    fold_id = np.load(e115.array_path(source, "fold_id"))
    point_power = np.load(e115.array_path(source, "point_power"))
    count, raw_tokens = raw_mass.shape
    feature_dim = 6 + 3 + 3 + 3 + 1 + e115.PORTS + 2 * e115.PORTS
    arrays = {
        "token_cluster": np.lib.format.open_memmap(
            array_path(directory, "token_cluster"),
            mode="w+",
            dtype=np.int16,
            shape=(count, raw_tokens),
        ),
        "mask": np.lib.format.open_memmap(
            array_path(directory, "mask"),
            mode="w+",
            dtype=bool,
            shape=(count, config.candidate_count),
        ),
        "coordinate": np.lib.format.open_memmap(
            array_path(directory, "coordinate"),
            mode="w+",
            dtype=np.float16,
            shape=(count, config.candidate_count, 3),
        ),
        "base_log_mass": np.lib.format.open_memmap(
            array_path(directory, "base_log_mass"),
            mode="w+",
            dtype=np.float16,
            shape=(count, config.candidate_count),
        ),
        "base_log_port": np.lib.format.open_memmap(
            array_path(directory, "base_log_port"),
            mode="w+",
            dtype=np.float16,
            shape=(count, config.candidate_count, e115.PORTS),
        ),
        "base_phase": np.lib.format.open_memmap(
            array_path(directory, "base_phase"),
            mode="w+",
            dtype=np.float16,
            shape=(count, config.candidate_count, 2 * e115.PORTS),
        ),
        "target_mass": np.lib.format.open_memmap(
            array_path(directory, "target_mass"),
            mode="w+",
            dtype=np.float16,
            shape=(count, config.candidate_count),
        ),
        "target_delta": np.lib.format.open_memmap(
            array_path(directory, "target_delta"),
            mode="w+",
            dtype=np.float16,
            shape=(count, config.candidate_count, 3),
        ),
        "target_port": np.lib.format.open_memmap(
            array_path(directory, "target_port"),
            mode="w+",
            dtype=np.float16,
            shape=(count, config.candidate_count, e115.PORTS),
        ),
        "target_phase": np.lib.format.open_memmap(
            array_path(directory, "target_phase"),
            mode="w+",
            dtype=np.float16,
            shape=(count, config.candidate_count, 2 * e115.PORTS),
        ),
        "cluster_feature": np.lib.format.open_memmap(
            array_path(directory, "cluster_feature"),
            mode="w+",
            dtype=np.float16,
            shape=(count, config.candidate_count, feature_dim),
        ),
        "member_count": np.lib.format.open_memmap(
            array_path(directory, "member_count"),
            mode="w+",
            dtype=np.int16,
            shape=(count, config.candidate_count),
        ),
    }
    arrays["token_cluster"][:] = -1
    arrays["mask"][:] = False
    arrays["coordinate"][:] = 0.0
    arrays["base_log_mass"][:] = -30.0
    arrays["base_log_port"][:] = math.log(1.0 / e115.PORTS)
    arrays["base_phase"][:] = 0.0
    arrays["base_phase"][:, :, : e115.PORTS] = 1.0
    arrays["target_mass"][:] = 0.0
    arrays["target_delta"][:] = 0.0
    arrays["target_port"][:] = 1.0 / e115.PORTS
    arrays["target_phase"][:] = 0.0
    arrays["target_phase"][:, :, : e115.PORTS] = 1.0
    arrays["cluster_feature"][:] = 0.0
    arrays["member_count"][:] = 0
    corrected_target = np.mod(coordinate + target_delta, e115.PERIODS)
    cluster_counts = []
    retained_mass = []
    for point_id in range(count):
        groups, _, total_cluster_count = cluster_groups(
            coordinate[point_id],
            raw_mass[point_id],
            config.cluster_threshold,
            config.candidate_count,
        )
        cluster_counts.append(total_cluster_count)
        for cluster_id, group in enumerate(groups):
            arrays["token_cluster"][point_id, group] = cluster_id
            arrays["mask"][point_id, cluster_id] = True
            arrays["member_count"][point_id, cluster_id] = len(group)
            local_base_mass = raw_mass[point_id, group]
            base_total = float(np.sum(local_base_mass, dtype=np.float64))
            local_target_mass = target_mass[point_id, group]
            target_total = float(np.sum(local_target_mass, dtype=np.float64))
            local_coordinate = coordinate[point_id, group]
            base_coordinate = e117.circular_mean(local_coordinate, local_base_mass)
            base_port = np.sum(
                local_base_mass[:, None] * raw_port[point_id, group],
                axis=0,
                dtype=np.float64,
            ) / max(base_total, e115.TINY)
            base_port = base_port / max(float(np.sum(base_port)), e115.TINY)
            base_direction = normalized_complex_average(
                raw_phase[point_id, group],
                local_base_mass[:, None] * raw_port[point_id, group],
            )
            arrays["coordinate"][point_id, cluster_id] = base_coordinate
            arrays["base_log_mass"][point_id, cluster_id] = math.log(
                max(base_total, 1e-12)
            )
            arrays["base_log_port"][point_id, cluster_id] = np.log(
                np.maximum(base_port, 1e-8)
            )
            arrays["base_phase"][point_id, cluster_id] = np.concatenate(
                (base_direction.real, base_direction.imag)
            )
            arrays["target_mass"][point_id, cluster_id] = target_total
            if target_total > 0.0:
                target_coordinate = e117.circular_mean(
                    corrected_target[point_id, group], local_target_mass
                )
                delta = np.asarray(
                    [
                        e79.periodic_delta(
                            target_coordinate[axis],
                            base_coordinate[axis],
                            float(e115.PERIODS[axis]),
                        )
                        for axis in range(3)
                    ],
                    np.float32,
                )
                local_target_port = np.sum(
                    local_target_mass[:, None] * target_port[point_id, group],
                    axis=0,
                    dtype=np.float64,
                ) / max(target_total, e115.TINY)
                local_target_port /= max(
                    float(np.sum(local_target_port)), e115.TINY
                )
                target_direction = normalized_complex_average(
                    target_phase[point_id, group],
                    local_target_mass[:, None] * target_port[point_id, group],
                )
                arrays["target_delta"][point_id, cluster_id] = delta
                arrays["target_port"][point_id, cluster_id] = local_target_port
                arrays["target_phase"][point_id, cluster_id] = np.concatenate(
                    (target_direction.real, target_direction.imag)
                )
            arrays["cluster_feature"][point_id, cluster_id] = cluster_feature(
                base_coordinate,
                base_total,
                base_port,
                base_direction,
                local_coordinate,
                local_base_mass,
                raw_tokens,
            )
        retained_mass.append(
            float(np.sum(arrays["target_mass"][point_id], dtype=np.float64))
            / max(float(np.sum(target_mass[point_id], dtype=np.float64)), e115.TINY)
        )
        if (point_id + 1) % 500 == 0:
            print(f"clustered {point_id + 1}/{count}", flush=True)
    for value in arrays.values():
        value.flush()
    np.save(array_path(directory, "fold_id"), fold_id)
    np.save(array_path(directory, "point_power"), point_power)
    setup, positions, _, site, _ = load_positions()
    graph = build_graph_features(directory, positions, site, fold_id, config)
    state = {
        "experiment": VERSION,
        "status": "screen_cache_complete",
        "source_cache": PATHS.relative(source),
        "config": asdict(config),
        "cluster_feature_dim": feature_dim,
        "cluster_count_median": float(np.median(cluster_counts)),
        "cluster_count_p90": float(np.quantile(cluster_counts, 0.90)),
        "retained_target_mass_median": float(np.median(retained_mass)),
        "retained_target_mass_mean": float(np.mean(retained_mass)),
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


class ClusterTensorStore:
    def __init__(self, directory: Path, device: torch.device) -> None:
        self.device = device
        self.directory = directory
        self.state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
        source = PATHS.root / self.state["source_cache"]
        self.raw = e115.TensorStore(source, device)

        def tensor(name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
            value = np.load(array_path(directory, name), mmap_mode="r")
            return torch.tensor(np.asarray(value), dtype=dtype, device=device)

        self.token_cluster = tensor("token_cluster", torch.int64)
        self.mask = tensor("mask", torch.bool)
        self.coordinate = tensor("coordinate")
        self.base_log_mass = tensor("base_log_mass")
        self.base_log_port = tensor("base_log_port")
        self.base_phase = tensor("base_phase")
        self.target_mass = tensor("target_mass")
        self.target_delta = tensor("target_delta")
        self.target_port = tensor("target_port")
        self.target_phase = tensor("target_phase")
        self.cluster_feature = tensor("cluster_feature")
        self.graph_feature = tensor("graph_feature")
        self.fold_id = tensor("fold_id", torch.int64)
        self.point_power = tensor("point_power", torch.float64)

    def normalization(self, fit_rows: np.ndarray) -> dict[str, torch.Tensor]:
        normalization = self.raw.normalization(fit_rows)
        rows = torch.as_tensor(fit_rows, device=self.device)
        mask = self.mask[rows]
        cluster_values = self.cluster_feature[rows][mask]
        cluster_mean = cluster_values.mean(dim=0)
        cluster_std = cluster_values.std(dim=0)
        cluster_std = torch.where(
            cluster_std < 1e-4, torch.ones_like(cluster_std), cluster_std
        )
        graph = self.graph_feature[rows]
        valid = graph[..., -1] > 0.5
        graph_values = graph[valid]
        graph_mean = graph_values.mean(dim=0)
        graph_std = graph_values.std(dim=0)
        graph_std = torch.where(graph_std < 1e-4, torch.ones_like(graph_std), graph_std)
        graph_mean[-1] = 0.0
        graph_std[-1] = 1.0
        return {
            **normalization,
            "cluster_mean": cluster_mean,
            "cluster_std": cluster_std,
            "unique_graph_mean": graph_mean,
            "unique_graph_std": graph_std,
        }

    def batch(
        self, rows: torch.Tensor, normalization: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        raw = self.raw.batch(rows, normalization)
        cluster = (
            self.cluster_feature[rows] - normalization["cluster_mean"]
        ) / normalization["cluster_std"]
        graph = (
            self.graph_feature[rows] - normalization["unique_graph_mean"]
        ) / normalization["unique_graph_std"]
        graph = torch.where(
            self.graph_feature[rows][..., -1:] > 0.5,
            graph,
            torch.zeros_like(graph),
        )
        raw.update(
            {
                "token_feature": raw.pop("candidate_feature"),
                "token_mask": raw.pop("mask"),
                "token_base_log_mass": raw.pop("base_log_mass"),
                "token_cluster": self.token_cluster[rows],
                "cluster_feature": cluster.clamp(-8.0, 8.0),
                "graph_feature": graph.clamp(-8.0, 8.0),
                "coordinate": self.coordinate[rows],
                "base_log_mass": self.base_log_mass[rows],
                "base_log_port": self.base_log_port[rows],
                "base_phase": self.base_phase[rows],
                "mask": self.mask[rows],
                "target_mass": self.target_mass[rows],
                "target_delta": self.target_delta[rows],
                "target_port": self.target_port[rows],
                "target_phase": self.target_phase[rows],
                "point_power": self.point_power[rows],
            }
        )
        return raw


class HierarchicalUniquePathOperator(nn.Module):
    def __init__(
        self,
        token_dim: int,
        cluster_dim: int,
        context_dim: int,
        graph_dim: int,
        config: Config,
        variant: e115.Variant,
    ) -> None:
        super().__init__()
        self.config = config
        self.variant = variant
        hidden = config.hidden
        self.token_input = nn.Sequential(
            nn.Linear(token_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.context_input = nn.Sequential(
            nn.Linear(context_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        token_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.heads,
            dim_feedforward=3 * hidden,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_encoder = nn.TransformerEncoder(
            token_layer, num_layers=config.token_layers
        )
        self.cluster_input = nn.Sequential(
            nn.Linear(cluster_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.pool_fusion = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        cluster_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.heads,
            dim_feedforward=3 * hidden,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cluster_encoder = nn.TransformerEncoder(
            cluster_layer, num_layers=config.candidate_layers
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.scene_encoder = e116.PositionalSceneEncoder(hidden)
        self.scene_attention = nn.MultiheadAttention(
            hidden, config.heads, dropout=config.dropout, batch_first=True
        )
        self.scene_context_norm = nn.LayerNorm(hidden)
        self.graph_input = nn.Sequential(
            nn.Linear(graph_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)
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
        self.graph_context_norm = nn.LayerNorm(hidden)
        self.scene_gates = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(hidden)) for name in STATE_NAMES}
        )
        self.graph_gates = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(hidden)) for name in STATE_NAMES}
        )
        self.score_head = nn.Linear(hidden, 1)
        self.coordinate_head = nn.Linear(hidden, 3)
        self.port_head = nn.Linear(hidden, e115.PORTS)
        self.phase_head = nn.Linear(hidden, e115.PORTS)
        for layer in (
            self.score_head,
            self.coordinate_head,
            self.port_head,
            self.phase_head,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def state_hidden(
        self,
        core: torch.Tensor,
        state: str,
        scene: torch.Tensor | None,
        graph: torch.Tensor | None,
    ) -> torch.Tensor:
        output = core
        if scene is not None:
            output = output + torch.tanh(self.scene_gates[state])[None, None] * scene
        if graph is not None:
            output = output + torch.tanh(self.graph_gates[state])[None, None] * graph
        return output

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        context = self.context_input(batch["context"])
        token = self.token_input(batch["token_feature"]) + context[:, None]
        token = self.token_encoder(token, src_key_padding_mask=~batch["token_mask"])
        token_cluster = batch["token_cluster"]
        valid_token = token_cluster >= 0
        membership = F.one_hot(
            token_cluster.clamp_min(0), num_classes=self.config.candidate_count
        ).to(token.dtype)
        membership = membership * valid_token[..., None].to(token.dtype)
        stabilized_log_mass = batch["token_base_log_mass"] - batch[
            "token_base_log_mass"
        ].max(dim=1, keepdim=True).values
        token_weight = torch.exp(stabilized_log_mass) * valid_token.to(token.dtype)
        weighted_membership = membership * token_weight[..., None]
        denominator = weighted_membership.sum(dim=1).clamp_min(1e-12)
        mean = torch.einsum("btk,bth->bkh", weighted_membership, token)
        mean = mean / denominator[..., None]
        second = torch.einsum("btk,bth->bkh", weighted_membership, token.square())
        second = second / denominator[..., None]
        deviation = torch.sqrt((second - mean.square()).clamp_min(0.0) + 1e-6)
        cluster = self.pool_fusion(torch.cat((mean, deviation), dim=2))
        cluster = cluster + self.cluster_input(batch["cluster_feature"]) + context[:, None]
        mask = batch["mask"]
        cluster = self.cluster_encoder(cluster, src_key_padding_mask=~mask)
        core = self.output_norm(cluster)

        scene_context = None
        if self.variant.use_scene:
            scene = self.scene_encoder(batch)
            attended, _ = self.scene_attention(core, scene, scene, need_weights=False)
            scene_context = self.scene_context_norm(attended)
        graph_context = None
        if self.variant.use_graph:
            batch_size, clusters, neighbors, _ = batch["graph_feature"].shape
            graph = self.graph_input(batch["graph_feature"])
            graph = graph.reshape(batch_size * clusters, neighbors, self.config.hidden)
            graph = self.graph_encoder(graph)
            graph = graph.mean(dim=1).reshape(
                batch_size, clusters, self.config.hidden
            )
            graph_context = self.graph_context_norm(graph)

        mass_hidden = self.state_hidden(core, "mass", scene_context, graph_context)
        coordinate_hidden = self.state_hidden(
            core, "coordinate", scene_context, graph_context
        )
        port_hidden = self.state_hidden(core, "port", scene_context, graph_context)
        phase_hidden = self.state_hidden(core, "phase", scene_context, graph_context)
        score_residual = self.config.score_limit * torch.tanh(
            self.score_head(mass_hidden)[..., 0]
        )
        mass_logits = (batch["base_log_mass"] + score_residual).masked_fill(
            ~mask, -1e4
        )
        coordinate_scale = torch.as_tensor(
            self.config.coordinate_limits, device=core.device, dtype=core.dtype
        )
        coordinate_residual = (
            torch.tanh(self.coordinate_head(coordinate_hidden)) * coordinate_scale
        )
        port_residual = self.config.port_limit * torch.tanh(self.port_head(port_hidden))
        base_complex = torch.complex(
            batch["base_phase"][..., : e115.PORTS],
            batch["base_phase"][..., e115.PORTS :],
        )
        phase_residual = self.config.phase_limit * torch.tanh(
            self.phase_head(phase_hidden)
        )
        return {
            "mass": torch.softmax(mass_logits, dim=1),
            "mass_logits": mass_logits,
            "coordinate": batch["coordinate"] + coordinate_residual,
            "coordinate_residual": coordinate_residual,
            "port": torch.softmax(batch["base_log_port"] + port_residual, dim=2),
            "port_residual": port_residual,
            "phase": torch.angle(base_complex) + phase_residual,
            "phase_residual": phase_residual,
            "score_residual": score_residual,
            "zero_logit": torch.zeros(len(core), device=core.device, dtype=core.dtype),
            "log_power": batch["base_log_power"],
        }


def stage_prefixes(stage: str) -> tuple[str, ...]:
    if stage == "scene":
        return (
            "scene_encoder.",
            "scene_attention.",
            "scene_context_norm.",
            "scene_gates.",
        )
    if stage == "graph":
        return (
            "graph_input.",
            "graph_encoder.",
            "graph_context_norm.",
            "graph_gates.",
        )
    return ()


def configure_stage(
    model: HierarchicalUniquePathOperator, stage: str
) -> list[nn.Parameter]:
    scene = stage_prefixes("scene")
    graph = stage_prefixes("graph")
    selected = []
    for name, parameter in model.named_parameters():
        if stage == "core":
            trainable = not name.startswith(scene) and not name.startswith(graph)
        else:
            trainable = name.startswith(stage_prefixes(stage))
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(parameter)
    model.eval()
    if stage == "core":
        for name, module in model.named_modules():
            if name and not name.startswith(scene) and not name.startswith(graph):
                module.train()
    elif stage == "scene":
        model.scene_encoder.train()
        model.scene_attention.train()
        model.scene_context_norm.train()
    else:
        model.graph_input.train()
        model.graph_encoder.train()
        model.graph_context_norm.train()
    return selected


def gate_summary(
    model: HierarchicalUniquePathOperator,
) -> dict[str, dict[str, float]]:
    return {
        branch: {
            state: float(torch.tanh(gates[state]).abs().mean().detach().cpu().item())
            for state in STATE_NAMES
        }
        for branch, gates in (
            ("scene", model.scene_gates),
            ("graph", model.graph_gates),
        )
    }


def train_stage(
    model: HierarchicalUniquePathOperator,
    store: ClusterTensorStore,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    held_rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    config: Config,
    stage: str,
    variant: e115.Variant,
    seed: int,
    checkpoint_prefix: str = "outer0_hierarchical",
) -> dict[str, object]:
    effective_seed = seed % (2**32)
    torch.manual_seed(effective_seed)
    np.random.seed(effective_seed)
    model.variant = variant
    trainable = configure_stage(model, stage)
    learning_rate = config.learning_rate if stage == "core" else 3e-4
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.epochs, 1), eta_min=learning_rate * 0.05
    )
    initial = e115.evaluate_rows(
        model,
        store,
        validation_rows,
        normalization,
        config,
        config.batch_size,
        False,
    )
    best_validation = float(initial["path_endpoint_mean"])
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float]] = [
        {
            "epoch": 0.0,
            "train_loss": float("nan"),
            "validation_path_endpoint": best_validation,
            "validation_mass_cosine": float(initial["mass_hellinger_cosine"]),
        }
    ]
    rng = np.random.default_rng(effective_seed)
    for epoch in range(1, config.epochs + 1):
        configure_stage(model, stage)
        order = rng.permutation(fit_rows)
        losses = []
        for start in range(0, len(order), config.batch_size):
            rows = torch.as_tensor(
                order[start : start + config.batch_size], device=store.device
            )
            batch = store.batch(rows, normalization)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss, _ = e115.state_objective(output, batch, config)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss stage={stage} epoch={epoch}")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(trainable, 2.0)
            if not torch.isfinite(gradient):
                raise RuntimeError(f"non-finite gradient stage={stage} epoch={epoch}")
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        row = {"epoch": float(epoch), "train_loss": float(np.mean(losses))}
        if epoch % config.validation_interval == 0:
            validation = e115.evaluate_rows(
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
                f"stage={stage} epoch={epoch}/{config.epochs} "
                f"train={row['train_loss']:.6f}"
            )
            if "validation_path_endpoint" in row:
                message += f" validation={row['validation_path_endpoint']:.6f}"
            print(message, flush=True)
    model.load_state_dict(best_state)
    validation = e115.evaluate_rows(
        model,
        store,
        validation_rows,
        normalization,
        config,
        config.batch_size,
        True,
    )
    held = e115.evaluate_rows(
        model,
        store,
        held_rows,
        normalization,
        config,
        config.batch_size,
        True,
    )
    checkpoint = ARTIFACT_DIR / f"{checkpoint_prefix}_{stage}.pt"
    torch.save(
        {
            "experiment": VERSION,
            "stage": stage,
            "variant": asdict(variant),
            "config": asdict(config),
            "effective_seed": effective_seed,
            "state_dict": model.state_dict(),
            "normalization": {key: value.cpu() for key, value in normalization.items()},
            "best_epoch": best_epoch,
            "official_h_only": True,
            "sionna_used": False,
            "v12_power_anchored": True,
        },
        checkpoint,
    )
    return {
        "stage": stage,
        "variant": asdict(variant),
        "effective_seed": effective_seed,
        "trainable_parameter_count": int(sum(value.numel() for value in trainable)),
        "best_epoch": int(best_epoch),
        "history": history,
        "validation": e115.compact_metrics(validation),
        "held": e115.compact_metrics(held),
        "gates": gate_summary(model),
        "checkpoint": PATHS.relative(checkpoint),
    }


def run_screen(config: Config, device: torch.device, args: argparse.Namespace) -> None:
    started = time.time()
    store = ClusterTensorStore(cache_dir(), device)
    fold_id = store.fold_id.cpu().numpy()
    held_rows = np.flatnonzero(fold_id == 0)
    validation_rows = np.flatnonzero(fold_id == 1)
    fit_rows = np.flatnonzero(np.isin(fold_id, (2, 3, 4)))
    normalization = store.normalization(fit_rows)
    baseline_validation = e115.evaluate_rows(
        None,
        store,
        validation_rows,
        normalization,
        config,
        args.batch_size,
        True,
    )
    baseline_held = e115.evaluate_rows(
        None,
        store,
        held_rows,
        normalization,
        config,
        args.batch_size,
        True,
    )
    seed = config.random_state % (2**32)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HierarchicalUniquePathOperator(
        store.raw.candidate_feature.shape[-1],
        store.cluster_feature.shape[-1],
        store.raw.context.shape[-1],
        store.graph_feature.shape[-1],
        config,
        e115.Variant("hierarchical_core", False, False),
    ).to(device)
    stages = []
    stages.append(
        train_stage(
            model,
            store,
            fit_rows,
            validation_rows,
            held_rows,
            normalization,
            replace(config, epochs=args.core_epochs),
            "core",
            e115.Variant("hierarchical_core", False, False),
            seed,
        )
    )
    stages.append(
        train_stage(
            model,
            store,
            fit_rows,
            validation_rows,
            held_rows,
            normalization,
            replace(config, epochs=args.scene_epochs),
            "scene",
            e115.Variant("hierarchical_scene", True, False),
            seed + 1,
        )
    )
    stages.append(
        train_stage(
            model,
            store,
            fit_rows,
            validation_rows,
            held_rows,
            normalization,
            replace(config, epochs=args.graph_epochs),
            "graph",
            e115.Variant("hierarchical_complete", True, True),
            seed + 2,
        )
    )
    core = stages[0]
    complete = stages[-1]
    ranking = sorted(
        stages,
        key=lambda row: float(row["validation"]["path_endpoint_mean"]),
        reverse=True,
    )
    gate = {
        "minimum_held_gain_over_raw": 0.03,
        "minimum_complete_gain_over_core": 0.003,
        "held_gain_over_raw": float(
            complete["held"]["path_endpoint_mean"]
            - baseline_held["path_endpoint_mean"]
        ),
        "complete_gain_over_core": float(
            complete["held"]["path_endpoint_mean"]
            - core["held"]["path_endpoint_mean"]
        ),
        "validation_selected_stage": ranking[0]["stage"],
    }
    gate["passed"] = bool(
        gate["held_gain_over_raw"] >= gate["minimum_held_gain_over_raw"]
        and gate["complete_gain_over_core"]
        >= gate["minimum_complete_gain_over_core"]
        and gate["validation_selected_stage"] == "graph"
    )
    report = {
        "experiment": VERSION,
        "status": "outer0_screen_complete",
        "primary_hypothesis": (
            "A hierarchical raw-token evidence encoder followed by unique-path state "
            "competition removes duplicate-label entropy while retaining every official "
            "neighbor path observation."
        ),
        "official_h_only": True,
        "sionna_used": False,
        "protocol": {
            "globally_blocked_folds": [0, 1],
            "fit_folds": [2, 3, 4],
            "validation_fold": 1,
            "held_fold": 0,
            "cluster_threshold_selected_on_validation": config.cluster_threshold,
            "held_h_used_as_donor_anywhere": False,
            "validation_h_used_as_donor_anywhere": False,
        },
        "config": asdict(config),
        "cache": store.state,
        "baseline": {
            "validation": e115.compact_metrics(baseline_validation),
            "held": e115.compact_metrics(baseline_held),
        },
        "stages": stages,
        "validation_ranking": [row["stage"] for row in ranking],
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
                "stages": [
                    {
                        "stage": row["stage"],
                        "best_epoch": row["best_epoch"],
                        "validation": row["validation"],
                        "held": row["held"],
                        "gates": row["gates"],
                    }
                    for row in stages
                ],
                "gate": gate,
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    config = Config(batch_size=args.batch_size)
    if args.stage == "prepare-screen":
        print(json.dumps(build_cache(config, args.force), ensure_ascii=False, indent=2))
    else:
        run_screen(config, torch.device(args.device), args)


if __name__ == "__main__":
    main()
