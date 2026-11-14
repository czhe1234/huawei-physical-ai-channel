#!/usr/bin/env python3
"""Train and score coefficient-free geometry path models for E214."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F

from common.official_metrics import (
    PointMetricStatistics,
    aggregate_statistics,
    point_metric_statistics_torch,
)
from data import load_positions
from project_paths import PATHS
import build_e214_geometry_path_topology as topology
import train_e213_no_rt_spectral_residual as e213


VERSION = "Round2-E214-geometry-path-topology"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e214_geometry_path_topology"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e214_geometry_path_topology"
PREPARE_REPORT = ANALYSIS_DIR / "prepare_report.json"
TRUTH_COMPONENTS = ARTIFACT_DIR / "official_h_top128_components.npz"
VARIANTS = (
    "base_s1",
    "base_s24",
    "base_scatter_s1",
    "base_scatter_s24",
)
VARIANT_INFO = {
    "base_s1": {"seed_mode": "s1", "scatter": False},
    "base_s24": {"seed_mode": "s24", "scatter": False},
    "base_scatter_s1": {"seed_mode": "s1", "scatter": True},
    "base_scatter_s24": {"seed_mode": "s24", "scatter": True},
}

MP = 2
MH = 16
MV = 8
N_UE = 4
S_SC = 192
PATHS_PER_POINT = 128
BASE_RESERVE = 48
SCATTER_RESERVE = PATHS_PER_POINT - BASE_RESERVE
STEPS = 16
PORTS = MP * N_UE
PERIODS = np.asarray((MH, MV, S_SC), dtype=np.float32)
SUPPORT_SCALES = np.asarray((2.0, 1.5, 3.0), dtype=np.float32)
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    hidden: int = 128
    heads: int = 4
    sequence_layers: int = 1
    set_layers: int = 2
    feedforward: int = 384
    dropout: float = 0.06
    epochs: int = 48
    minimum_epochs: int = 8
    validation_interval: int = 4
    batch_size: int = 32
    prediction_batch_size: int = 64
    channel_batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 3e-4
    coordinate_limits: tuple[float, float, float] = (2.0, 2.0, 4.0)
    mass_residual_limit: float = 5.0
    spectral_weight: float = 1.0
    mass_weight: float = 0.35
    coordinate_weight: float = 0.08
    port_weight: float = 0.08
    existence_weight: float = 0.03
    residual_weight: float = 0.002
    pas_action_scale: float = 0.20
    pdp_action_scale: float = 0.20
    projection_cycles: int = 4
    top_truth_components: int = 128
    assignment_cost_limit: float = 9.0
    minimum_truth_fraction: float = 1e-6
    random_state: int = 20260816214


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "aggregate", "status"), required=True)
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    parser.add_argument("--outer", type=int)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def variant_artifact_dir(variant: str) -> Path:
    return ARTIFACT_DIR / "variants" / variant


def variant_analysis_dir(variant: str) -> Path:
    return ANALYSIS_DIR / variant


def labels_path(variant: str) -> Path:
    return variant_artifact_dir(variant) / "official_h_path_labels.npz"


def checkpoint_path(variant: str, outer: int) -> Path:
    return variant_artifact_dir(variant) / f"outer{outer}_model.pt"


def endpoint_path(variant: str, outer: int) -> Path:
    return variant_artifact_dir(variant) / f"outer{outer}_endpoints.npz"


def fold_report_path(variant: str, outer: int) -> Path:
    return variant_analysis_dir(variant) / f"outer{outer}.json"


def final_report_path(variant: str) -> Path:
    return variant_analysis_dir(variant) / "fixed500_report.json"


def output_channel_path(variant: str) -> Path:
    return variant_artifact_dir(variant) / "fixed500_channel.npy"


def failure_report_path(stage: str, variant: str, outer: int | None) -> Path:
    suffix = variant if outer is None else f"{variant}_outer{outer}"
    return ANALYSIS_DIR / f"failure_{stage}_{suffix}.json"


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def atomic_torch(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    value = seed % (2**32)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def config_payload(config: Config) -> dict[str, object]:
    """Return the JSON-stable representation used in reports and resume checks."""
    return json.loads(json.dumps(asdict(config)))


def topology_dir(source: str, seed_mode: str) -> Path:
    return topology.merged_dir(source, seed_mode)


def validate_variant(variant: str) -> None:
    if variant not in VARIANT_INFO:
        raise ValueError(f"unsupported variant: {variant}")


def periodic_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return ((left - right + 0.5 * PERIODS) % PERIODS) - 0.5 * PERIODS


def point_features() -> np.ndarray:
    setup, positions, _, site, _ = load_positions()
    positions = positions.astype(np.float32)
    bs = setup.bs_positions[site].astype(np.float32)
    delta = positions - bs
    distance = np.linalg.norm(delta, axis=1)
    bearing = np.arctan2(delta[:, 1], delta[:, 0])
    elevation = np.arctan2(delta[:, 2], np.maximum(np.linalg.norm(delta[:, :2], axis=1), 1e-6))
    output = np.column_stack(
        (
            positions,
            delta,
            distance,
            np.log1p(distance),
            np.sin(bearing),
            np.cos(bearing),
            np.sin(elevation),
            np.cos(elevation),
            site == 0,
            site == 1,
        )
    )
    return output.astype(np.float32)


def build_truth_components(
    config: Config, device: torch.device, force: bool
) -> dict[str, object]:
    if TRUTH_COMPONENTS.is_file() and not force:
        with np.load(TRUTH_COMPONENTS, allow_pickle=False) as payload:
            if (
                payload["coordinate"].shape == (4000, config.top_truth_components, 3)
                and payload["port"].shape == (4000, config.top_truth_components, PORTS)
            ):
                return {
                    "status": "reused",
                    "path": PATHS.relative(TRUTH_COMPONENTS),
                    "count": 4000,
                }
    started = time.time()
    channel = np.load(e213.TRAIN_H, mmap_mode="r")
    count = len(channel)
    coordinate = np.zeros((count, config.top_truth_components, 3), dtype=np.float32)
    mass = np.zeros((count, config.top_truth_components), dtype=np.float32)
    port = np.zeros((count, config.top_truth_components, PORTS), dtype=np.float32)
    valid = np.zeros((count, config.top_truth_components), dtype=bool)
    for start in range(0, count, config.channel_batch_size):
        end = min(start + config.channel_batch_size, count)
        local = torch.as_tensor(
            np.array(channel[start:end], dtype=np.complex64, copy=True), device=device
        ).reshape(-1, MP, MH, MV, N_UE, S_SC)
        with torch.inference_mode():
            transformed = torch.fft.ifft(torch.fft.fft2(local, dim=(2, 3)), dim=5)
            state_power = transformed.abs().square().float()
            collapsed = state_power.sum(dim=(1, 4))
            flat = collapsed.reshape(end - start, -1)
            top_energy, top_index = torch.topk(flat, config.top_truth_components, dim=1)
            total = flat.sum(dim=1)
            port_flat = state_power.permute(0, 2, 3, 5, 1, 4).reshape(
                end - start, -1, PORTS
            )
            selected_port = torch.gather(
                port_flat,
                1,
                top_index[..., None].expand(-1, -1, PORTS),
            )
            local_mass = top_energy / total[:, None].clamp_min(TINY)
            local_port = selected_port / selected_port.sum(dim=2, keepdim=True).clamp_min(TINY)
        index = top_index.cpu().numpy().astype(np.int64)
        horizontal = index // (MV * S_SC)
        vertical = (index // S_SC) % MV
        delay = index % S_SC
        coordinate[start:end] = np.stack((horizontal, vertical, delay), axis=2).astype(np.float32)
        mass[start:end] = local_mass.cpu().numpy().astype(np.float32)
        port[start:end] = local_port.cpu().numpy().astype(np.float32)
        valid[start:end] = mass[start:end] >= config.minimum_truth_fraction
        if end == count or end % 500 == 0:
            print(f"E214 official-H components {end}/{count}", flush=True)
        del local, transformed, state_power, collapsed, flat, selected_port
    atomic_npz(
        TRUTH_COMPONENTS,
        coordinate=coordinate,
        mass=mass,
        port=port,
        valid=valid,
    )
    return {
        "status": "created_from_official_h",
        "path": PATHS.relative(TRUTH_COMPONENTS),
        "count": count,
        "top_components": config.top_truth_components,
        "mean_retained_mass": float(np.mean(np.sum(mass, axis=1))),
        "seconds": float(time.time() - started),
    }


class TopologyArrays:
    def __init__(self, variant: str) -> None:
        validate_variant(variant)
        info = VARIANT_INFO[variant]
        seed_mode = str(info["seed_mode"])
        self.variant = variant
        self.scatter = bool(info["scatter"])
        self.base = {
            name: np.load(topology_dir("base", seed_mode) / f"{name}.npy", mmap_mode="r")
            for name in topology.MERGED_SHAPES
        }
        self.scatter_arrays = (
            {
                name: np.load(
                    topology_dir("scatter", seed_mode) / f"{name}.npy", mmap_mode="r"
                )
                for name in topology.MERGED_SHAPES
            }
            if self.scatter
            else None
        )

    def get(self, name: str, rows: np.ndarray) -> np.ndarray:
        base = np.asarray(self.base[name][rows])
        if self.scatter_arrays is None:
            return np.array(base, copy=True)
        scatter = np.asarray(self.scatter_arrays[name][rows])
        return np.concatenate(
            (base[:, :BASE_RESERVE], scatter[:, :SCATTER_RESERVE]), axis=1
        )

    def full(self, name: str) -> np.ndarray:
        rows = np.arange(4000, dtype=np.int64)
        return self.get(name, rows)


def support_cost(candidates: np.ndarray, truth: np.ndarray) -> np.ndarray:
    delta = periodic_delta(truth[None, :, :] - candidates[:, None, :], np.zeros(3, dtype=np.float32))
    return np.square(delta / SUPPORT_SCALES, dtype=np.float32).sum(axis=2)


def build_labels(variant: str, config: Config, force: bool) -> dict[str, object]:
    output = labels_path(variant)
    if output.is_file() and not force:
        with np.load(output, allow_pickle=False) as payload:
            if payload["target_mass"].shape == (4000, PATHS_PER_POINT):
                return {
                    "status": "reused",
                    "path": PATHS.relative(output),
                    "matched_points": int(np.sum(payload["valid"])),
                }
    started = time.time()
    topology_arrays = TopologyArrays(variant)
    candidate_coordinate = topology_arrays.full("coordinate").astype(np.float32)
    candidate_mask = topology_arrays.full("mask").astype(bool)
    with np.load(TRUTH_COMPONENTS, allow_pickle=False) as truth:
        truth_coordinate = truth["coordinate"].astype(np.float32)
        truth_mass = truth["mass"].astype(np.float32)
        truth_port = truth["port"].astype(np.float32)
        truth_valid = truth["valid"].astype(bool)
    target_mass = np.zeros((4000, PATHS_PER_POINT), dtype=np.float32)
    target_delta = np.zeros((4000, PATHS_PER_POINT, 3), dtype=np.float32)
    target_port = np.zeros((4000, PATHS_PER_POINT, PORTS), dtype=np.float32)
    assigned = np.zeros((4000, PATHS_PER_POINT), dtype=bool)
    matched_count = np.zeros(4000, dtype=np.int16)
    matched_fraction = np.zeros(4000, dtype=np.float32)
    accepted_cost: list[np.ndarray] = []
    for row in range(4000):
        candidate_ids = np.flatnonzero(candidate_mask[row])
        truth_ids = np.flatnonzero(truth_valid[row])
        if len(candidate_ids) == 0 or len(truth_ids) == 0:
            continue
        cost = support_cost(
            candidate_coordinate[row, candidate_ids], truth_coordinate[row, truth_ids]
        )
        local_candidate, local_truth = linear_sum_assignment(cost)
        keep = cost[local_candidate, local_truth] <= config.assignment_cost_limit
        candidate_selected = candidate_ids[local_candidate[keep]]
        truth_selected = truth_ids[local_truth[keep]]
        if len(candidate_selected) == 0:
            continue
        target_mass[row, candidate_selected] = truth_mass[row, truth_selected]
        target_delta[row, candidate_selected] = periodic_delta(
            truth_coordinate[row, truth_selected],
            candidate_coordinate[row, candidate_selected],
        )
        target_port[row, candidate_selected] = truth_port[row, truth_selected]
        assigned[row, candidate_selected] = True
        matched_count[row] = len(candidate_selected)
        matched_fraction[row] = float(np.sum(truth_mass[row, truth_selected]))
        accepted_cost.append(cost[local_candidate[keep], local_truth[keep]])
        if (row + 1) % 500 == 0 or row == 3999:
            print(f"E214 labels variant={variant} {row + 1}/4000", flush=True)
    valid = matched_fraction > 0.0
    cost_values = np.concatenate(accepted_cost) if accepted_cost else np.empty(0, dtype=np.float32)
    atomic_npz(
        output,
        candidate_mask=candidate_mask,
        target_mass=target_mass,
        target_delta=target_delta,
        target_port=target_port,
        assigned=assigned,
        valid=valid,
        matched_count=matched_count,
        matched_fraction=matched_fraction,
    )
    return {
        "status": "created_from_official_h",
        "path": PATHS.relative(output),
        "matched_points": int(np.sum(valid)),
        "matched_count_mean": float(np.mean(matched_count[valid])) if np.any(valid) else 0.0,
        "matched_fraction_mean": float(np.mean(matched_fraction[valid])) if np.any(valid) else 0.0,
        "accepted_cost_mean": float(np.mean(cost_values)) if len(cost_values) else 0.0,
        "seconds": float(time.time() - started),
    }


def topology_audit() -> dict[str, object]:
    report_path = topology.ANALYSIS_DIR / "topology_audit.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or report.get("forbidden_fields_accessed"):
        raise RuntimeError("sanitized topology audit did not pass")
    if set(report.get("allowed_fields", ())) != set(topology.ALLOWED_FIELDS):
        raise RuntimeError("sanitized topology allowlist changed")
    return report


def zero_action_identity(config: Config, device: torch.device) -> float:
    base_h = np.load(e213.CANONICAL_BASE_H, mmap_mode="r")
    base_pas = np.load(e213.CANONICAL_BASE_PAS, mmap_mode="r")
    base_pdp = np.load(e213.CANONICAL_BASE_PDP, mmap_mode="r")
    rows = np.asarray((0, 499), dtype=np.int64)
    channel = torch.as_tensor(np.array(base_h[rows], copy=True), device=device)
    pas = torch.as_tensor(np.asarray(base_pas[rows]), device=device)
    pdp = torch.as_tensor(np.asarray(base_pdp[rows]), device=device)
    output = e213.project_channel(channel, pas, pdp, pas, pdp, config)
    return float((output - channel).abs().max().item())


def prepare(config: Config, device: torch.device, force: bool) -> dict[str, object]:
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    audit = topology_audit()
    required = (
        e213.TRAIN_H,
        e213.TRAIN_BASE_H,
        e213.CANONICAL_BASE_H,
        e213.OOF_META,
        e213.FULL_FOLD,
        e213.TRAIN_BASE_PAS,
        e213.TRAIN_BASE_PDP,
        e213.CANONICAL_BASE_PAS,
        e213.CANONICAL_BASE_PDP,
        e213.OFFICIAL_PAS,
        e213.OFFICIAL_PDP,
        e213.OFFICIAL_POWER,
    )
    missing = [PATHS.relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing E214 required inputs: {missing}")
    truth = build_truth_components(config, device, force)
    labels = {variant: build_labels(variant, config, force) for variant in VARIANTS}
    lineage = e213.fold_lineage()
    if lineage["matched_fold_count"] != 124 or lineage["label_access_risk_count"] != 376:
        raise RuntimeError(f"historical fold lineage changed: {lineage}")
    identity_error = zero_action_identity(config, device)
    if identity_error != 0.0:
        raise RuntimeError(f"zero-action V7 identity failed: {identity_error}")
    gradient = finite_gradient_check(config, device)
    report = {
        "experiment": VERSION,
        "status": "complete",
        "config": config_payload(config),
        "topology_audit": {
            "path": PATHS.relative(topology.ANALYSIS_DIR / "topology_audit.json"),
            "forbidden_fields_accessed": audit.get("forbidden_fields_accessed", []),
        },
        "official_h_components": truth,
        "labels": labels,
        "lineage": lineage,
        "zero_action_v7_max_abs": identity_error,
        "finite_gradient_check": gradient,
        "wall_seconds": float(time.time() - started),
        "device": str(device),
        "cuda_peak_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "exit_code": 0,
    }
    atomic_json(PREPARE_REPORT, report)
    return report


class FeatureStore:
    def __init__(self, variant: str, device: torch.device) -> None:
        self.variant = variant
        self.device = device
        self.topology = TopologyArrays(variant)
        with np.load(labels_path(variant), allow_pickle=False) as payload:
            self.labels = {name: np.array(payload[name], copy=True) for name in payload.files}
        topology_mask = self.topology.full("mask").astype(bool)
        if not np.array_equal(topology_mask, self.labels["candidate_mask"]):
            raise RuntimeError(f"candidate identity changed for {variant}")
        self.mask_count = topology_mask.sum(axis=1).astype(np.int16)
        self.point_raw = point_features()
        self.point = torch.as_tensor(self.point_raw, device=device)
        self.base_pas = torch.as_tensor(
            np.asarray(np.load(e213.TRAIN_BASE_PAS, mmap_mode="r")), device=device
        )
        self.base_pdp = torch.as_tensor(
            np.asarray(np.load(e213.TRAIN_BASE_PDP, mmap_mode="r")), device=device
        )
        self.target_pas = torch.as_tensor(
            np.asarray(np.load(e213.OFFICIAL_PAS, mmap_mode="r")), device=device
        )
        self.target_pdp = torch.as_tensor(
            np.asarray(np.load(e213.OFFICIAL_PDP, mmap_mode="r")), device=device
        )
        self.power = np.asarray(np.load(e213.OFFICIAL_POWER, mmap_mode="r"), dtype=np.float64)

    def normalization(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        point = self.point_raw[rows].astype(np.float64)
        point_mean = point.mean(axis=0)
        point_std = point.std(axis=0)
        point_std[point_std < 1e-5] = 1.0
        total = np.zeros(19, dtype=np.float64)
        square = np.zeros(19, dtype=np.float64)
        count = 0
        for start in range(0, len(rows), 256):
            local_rows = rows[start : start + 256]
            continuous = self.topology.get("continuous", local_rows).astype(np.float64)
            mask = self.topology.get("mask", local_rows).astype(bool)
            values = continuous[mask]
            if len(values):
                total += values.sum(axis=0)
                square += np.square(values).sum(axis=0)
                count += len(values)
        if count == 0:
            raise RuntimeError(f"no valid topology paths in normalization for {self.variant}")
        continuous_mean = total / count
        variance = np.maximum(square / count - continuous_mean**2, 1e-8)
        continuous_std = np.sqrt(variance)
        continuous_std[continuous_std < 1e-4] = 1.0
        return {
            "point_mean": point_mean.astype(np.float32),
            "point_std": point_std.astype(np.float32),
            "continuous_mean": continuous_mean.astype(np.float32),
            "continuous_std": continuous_std.astype(np.float32),
        }

    def active_rows(self, rows: np.ndarray) -> np.ndarray:
        return rows[(self.power[rows] > 0.0) & (self.mask_count[rows] > 0)]

    def batch(
        self,
        rows: np.ndarray,
        normalization: dict[str, np.ndarray],
        base_pas: torch.Tensor | np.ndarray | None = None,
        base_pdp: torch.Tensor | np.ndarray | None = None,
        include_targets: bool = True,
    ) -> dict[str, torch.Tensor]:
        rows = np.asarray(rows, dtype=np.int64)
        mask = self.topology.get("mask", rows).astype(bool)
        has_paths = mask.any(axis=1)
        if np.any(~has_paths):
            mask[~has_paths, 0] = True
        continuous = self.topology.get("continuous", rows).astype(np.float32)
        continuous = (
            continuous - normalization["continuous_mean"][None, None]
        ) / normalization["continuous_std"][None, None]
        coordinate = self.topology.get("coordinate", rows).astype(np.float32)
        presence = self.topology.get("presence", rows).astype(np.float32)
        presence[~has_paths, 0] = 1.0
        output: dict[str, torch.Tensor] = {
            "rows": torch.as_tensor(rows, dtype=torch.long, device=self.device),
            "mask": torch.as_tensor(mask, device=self.device),
            "has_paths": torch.as_tensor(has_paths, device=self.device),
            "continuous": torch.as_tensor(continuous, device=self.device),
            "coordinate": torch.as_tensor(coordinate, device=self.device),
            "presence": torch.as_tensor(presence, device=self.device),
            "depth": torch.as_tensor(
                self.topology.get("depth", rows).astype(np.int64), device=self.device
            ),
            "interactions": torch.as_tensor(
                self.topology.get("interactions", rows).astype(np.int64), device=self.device
            ),
            "objects": torch.as_tensor(
                self.topology.get("objects", rows).astype(np.int64), device=self.device
            ),
            "primitives": torch.as_tensor(
                self.topology.get("primitives", rows).astype(np.int64), device=self.device
            ),
            "sequence_geometry": torch.as_tensor(
                self.topology.get("sequence_geometry", rows).astype(np.float32),
                device=self.device,
            ),
            "family_hash": torch.as_tensor(
                self.topology.get("family_hash", rows).astype(np.uint64).view(np.int64),
                device=self.device,
            ),
            "source": torch.as_tensor(
                self.topology.get("source", rows).astype(np.int64), device=self.device
            ),
            "point": (
                self.point[torch.as_tensor(rows, device=self.device)]
                - torch.as_tensor(normalization["point_mean"], device=self.device)
            )
            / torch.as_tensor(normalization["point_std"], device=self.device),
        }
        if base_pas is None:
            output["base_pas"] = self.base_pas[output["rows"]]
        else:
            output["base_pas"] = torch.as_tensor(base_pas, device=self.device)
        if base_pdp is None:
            output["base_pdp"] = self.base_pdp[output["rows"]]
        else:
            output["base_pdp"] = torch.as_tensor(base_pdp, device=self.device)
        if include_targets:
            output.update(
                {
                    "target_pas": self.target_pas[output["rows"]],
                    "target_pdp": self.target_pdp[output["rows"]],
                    "target_mass": torch.as_tensor(
                        self.labels["target_mass"][rows], device=self.device
                    ),
                    "target_delta": torch.as_tensor(
                        self.labels["target_delta"][rows], device=self.device
                    ),
                    "target_port": torch.as_tensor(
                        self.labels["target_port"][rows], device=self.device
                    ),
                    "assigned": torch.as_tensor(
                        self.labels["assigned"][rows], device=self.device
                    ),
                    "label_valid": torch.as_tensor(
                        self.labels["valid"][rows], device=self.device
                    ),
                }
            )
        return output


def masked_mean(value: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight.unsqueeze(-1)).sum(dim=dim) / weight.sum(
        dim=dim, keepdim=True
    ).clamp_min(1.0)


class GeometryPathModel(nn.Module):
    def __init__(self, point_dim: int, config: Config) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden
        self.spectrum_encoder = nn.Sequential(
            nn.Linear(MP * MH * MV * N_UE + MP * N_UE * S_SC, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, hidden),
            nn.SiLU(),
        )
        self.point_encoder = nn.Sequential(
            nn.Linear(point_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        self.path_encoder = nn.Sequential(
            nn.Linear(19 + 2 * PORTS, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        self.sequence_input = nn.Linear(16, hidden)
        self.interaction_embedding = nn.Embedding(8, hidden)
        self.object_embedding = nn.Embedding(1025, hidden)
        self.primitive_embedding = nn.Embedding(8193, hidden)
        self.step_embedding = nn.Embedding(STEPS, hidden)
        self.family_embedding_a = nn.Embedding(32749, hidden)
        self.family_embedding_b = nn.Embedding(32749, hidden)
        self.source_embedding = nn.Embedding(2, hidden)
        sequence_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.heads,
            dim_feedforward=config.feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(
            sequence_layer,
            num_layers=config.sequence_layers,
            enable_nested_tensor=False,
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        set_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.heads,
            dim_feedforward=config.feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            set_layer,
            num_layers=config.set_layers,
            enable_nested_tensor=False,
        )
        self.mass_head = nn.Linear(hidden, 1)
        self.coordinate_head = nn.Linear(hidden, 3)
        self.port_head = nn.Linear(hidden, PORTS)
        self.gate_head = nn.Linear(2 * hidden, 2)
        for layer in (self.mass_head, self.coordinate_head, self.port_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -1.5)

    def sampled_base(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        coordinate = batch["coordinate"]
        horizontal = torch.remainder(torch.round(coordinate[..., 0]).long(), MH)
        vertical = torch.remainder(torch.round(coordinate[..., 1]).long(), MV)
        delay = torch.remainder(torch.round(coordinate[..., 2]).long(), S_SC)
        batch_index = torch.arange(len(coordinate), device=coordinate.device)[:, None]
        pas_grid = batch["base_pas"].permute(0, 2, 3, 1, 4)
        pdp_grid = batch["base_pdp"].permute(0, 3, 1, 2)
        pas = pas_grid[batch_index, horizontal, vertical].reshape(
            len(coordinate), PATHS_PER_POINT, PORTS
        )
        pdp = pdp_grid[batch_index, delay].reshape(
            len(coordinate), PATHS_PER_POINT, PORTS
        )
        return torch.log1p(1000.0 * torch.cat((pas, pdp), dim=2).clamp_min(0.0))

    def sequence_state(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        geometry = batch["sequence_geometry"]
        count, paths, steps, _ = geometry.shape
        step = torch.arange(steps, device=geometry.device)
        depth = batch["depth"]
        active = step[None, None] < depth[..., None]
        token_mask = step[None, None] < depth.clamp_min(1)[..., None]
        objects = torch.where(
            active,
            torch.remainder(batch["objects"], 1024) + 1,
            torch.zeros_like(batch["objects"]),
        )
        primitives = torch.where(
            active,
            torch.remainder(batch["primitives"], 8192) + 1,
            torch.zeros_like(batch["primitives"]),
        )
        interactions = torch.where(active, batch["interactions"].clamp(0, 7), 0)
        hidden = (
            self.sequence_input(geometry)
            + self.interaction_embedding(interactions)
            + self.object_embedding(objects)
            + self.primitive_embedding(primitives)
            + self.step_embedding(step)[None, None]
        )
        flat_hidden = hidden.reshape(count * paths, steps, self.config.hidden)
        flat_mask = token_mask.reshape(count * paths, steps)
        encoded = self.sequence_encoder(flat_hidden, src_key_padding_mask=~flat_mask)
        pooled = masked_mean(encoded, flat_mask, dim=1).reshape(
            count, paths, self.config.hidden
        )
        family = batch["family_hash"]
        family_a = torch.remainder(family, 32749)
        family_b = torch.remainder(family // 32749, 32749)
        return (
            pooled
            + self.family_embedding_a(family_a)
            + self.family_embedding_b(family_b)
            + self.source_embedding(batch["source"].clamp(0, 1))
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        base_pas = e213.normalize_compact_pas(batch["base_pas"].float())
        base_pdp = e213.normalize_compact_pdp(batch["base_pdp"].float())
        spectrum = torch.cat(
            (torch.sqrt(base_pas.clamp_min(0.0)).flatten(1), torch.sqrt(base_pdp.clamp_min(0.0)).flatten(1)),
            dim=1,
        )
        point = self.point_encoder(batch["point"])
        point = point + self.spectrum_encoder(spectrum)
        path = self.path_encoder(
            torch.cat((batch["continuous"], self.sampled_base(batch)), dim=2)
        )
        sequence = self.sequence_state(batch)
        hidden = self.fusion(
            torch.cat((path, sequence, point[:, None].expand(-1, PATHS_PER_POINT, -1)), dim=2)
        )
        hidden = self.set_encoder(hidden, src_key_padding_mask=~batch["mask"])
        mass_residual = self.config.mass_residual_limit * torch.tanh(
            self.mass_head(hidden)[..., 0]
        )
        prior = (
            torch.log(batch["presence"].clamp_min(1e-4))
            - 0.12 * batch["depth"].float()
            - 0.002 * batch["coordinate"][..., 2]
        )
        mass_logits = (prior + mass_residual).masked_fill(~batch["mask"], -1e4)
        mass = torch.softmax(mass_logits, dim=1)
        coordinate_scale = torch.as_tensor(
            self.config.coordinate_limits, device=hidden.device, dtype=hidden.dtype
        )
        coordinate_residual = torch.tanh(self.coordinate_head(hidden)) * coordinate_scale
        coordinate = torch.remainder(
            batch["coordinate"] + coordinate_residual,
            torch.as_tensor(PERIODS, device=hidden.device),
        )
        port_logits = self.port_head(hidden)
        port = torch.softmax(port_logits, dim=2).reshape(
            len(hidden), PATHS_PER_POINT, MP, N_UE
        )
        rendered_pas, rendered_pdp = render_spectra(mass, coordinate, port)
        pooled = masked_mean(hidden, batch["mask"], dim=1)
        gate = torch.sigmoid(self.gate_head(torch.cat((pooled, point), dim=1)))
        pas = blend_pas(base_pas, rendered_pas, gate[:, 0])
        pdp = blend_pdp(base_pdp, rendered_pdp, gate[:, 1])
        usable = batch["has_paths"] & (base_pas.sum(dim=(1, 2, 3, 4)) > 0.0)
        pas = torch.where(usable[:, None, None, None, None], pas, batch["base_pas"].float())
        pdp = torch.where(usable[:, None, None, None], pdp, batch["base_pdp"].float())
        return {
            "pas": pas.float(),
            "pdp": pdp.float(),
            "mass": mass.float(),
            "mass_logits": mass_logits.float(),
            "mass_residual": mass_residual.float(),
            "coordinate": coordinate.float(),
            "coordinate_residual": coordinate_residual.float(),
            "port": port.float(),
            "port_logits": port_logits.float(),
            "gate": gate.float(),
        }


def spectral_kernels(coordinate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coordinate = coordinate.float()
    horizontal_index = torch.arange(MH, device=coordinate.device, dtype=torch.float32)
    vertical_index = torch.arange(MV, device=coordinate.device, dtype=torch.float32)
    delay_index = torch.arange(S_SC, device=coordinate.device, dtype=torch.float32)
    horizontal_wave = torch.exp(
        2j * torch.pi * coordinate[..., 0, None] * horizontal_index / float(MH)
    )
    vertical_wave = torch.exp(
        2j * torch.pi * coordinate[..., 1, None] * vertical_index / float(MV)
    )
    delay_wave = torch.exp(
        -2j * torch.pi * coordinate[..., 2, None] * delay_index / float(S_SC)
    )
    horizontal = torch.fft.fft(horizontal_wave, dim=-1).abs().square().float()
    vertical = torch.fft.fft(vertical_wave, dim=-1).abs().square().float()
    delay = torch.fft.ifft(delay_wave, dim=-1).abs().square().float()
    horizontal /= horizontal.sum(dim=-1, keepdim=True).clamp_min(TINY)
    vertical /= vertical.sum(dim=-1, keepdim=True).clamp_min(TINY)
    delay /= delay.sum(dim=-1, keepdim=True).clamp_min(TINY)
    return horizontal, vertical, delay


def render_spectra(
    mass: torch.Tensor, coordinate: torch.Tensor, port: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    horizontal, vertical, delay = spectral_kernels(coordinate)
    energy = mass[..., None, None] * port
    pas = torch.einsum("bkpu,bkh,bkv->bphvu", energy, horizontal, vertical)
    pas = e213.normalize_compact_pas(pas)
    pdp = torch.einsum("bkpu,bkd->bpud", energy, delay)
    pdp = e213.normalize_compact_pdp(pdp)
    return pas, pdp


def blend_pas(base: torch.Tensor, rendered: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    weight = gate[:, None, None, None, None]
    mixed = ((1.0 - weight) * torch.sqrt(base.clamp_min(0.0)) + weight * torch.sqrt(rendered.clamp_min(0.0))).square()
    return e213.normalize_compact_pas(mixed)


def blend_pdp(base: torch.Tensor, rendered: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    weight = gate[:, None, None, None]
    mixed = ((1.0 - weight) * torch.sqrt(base.clamp_min(0.0)) + weight * torch.sqrt(rendered.clamp_min(0.0))).square()
    return e213.normalize_compact_pdp(mixed)


def training_objective(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    pas_score, pdp_score = e213.spectral_cosines(
        output["pas"], output["pdp"], batch["target_pas"], batch["target_pdp"]
    )
    spectral_loss = 1.0 - 0.5 * (pas_score.mean() + pdp_score.mean())
    target_mass = batch["target_mass"].clamp_min(0.0)
    target_total = target_mass.sum(dim=1, keepdim=True)
    target_normalized = target_mass / target_total.clamp_min(TINY)
    valid = (target_total[:, 0] > 0.0).float()
    mass_per_point = -(target_normalized * torch.log(output["mass"].clamp_min(1e-8))).sum(dim=1)
    mass_loss = (mass_per_point * valid).sum() / valid.sum().clamp_min(1.0)
    coordinate_error = F.smooth_l1_loss(
        output["coordinate_residual"], batch["target_delta"], reduction="none"
    ).mean(dim=2)
    coordinate_loss = (
        (coordinate_error * target_normalized).sum(dim=1) * valid
    ).sum() / valid.sum().clamp_min(1.0)
    port_error = -(
        batch["target_port"] * torch.log(output["port"].reshape_as(batch["target_port"]).clamp_min(1e-8))
    ).sum(dim=2)
    port_loss = ((port_error * target_normalized).sum(dim=1) * valid).sum() / valid.sum().clamp_min(1.0)
    active_logits = output["mass_logits"][batch["mask"]]
    active_target = batch["assigned"][batch["mask"]].float()
    positive_weight = torch.as_tensor(3.0, device=active_logits.device)
    existence_loss = F.binary_cross_entropy_with_logits(
        active_logits, active_target, pos_weight=positive_weight
    )
    residual = (
        output["mass_residual"].square().mean()
        + (output["coordinate_residual"] / torch.as_tensor(config.coordinate_limits, device=active_logits.device)).square().mean()
        + output["port_logits"].square().mean()
        + output["gate"].square().mean()
    )
    loss = (
        config.spectral_weight * spectral_loss
        + config.mass_weight * mass_loss
        + config.coordinate_weight * coordinate_loss
        + config.port_weight * port_loss
        + config.existence_weight * existence_loss
        + config.residual_weight * residual
    )
    return loss, {
        "loss": float(loss.detach().item()),
        "pas": float(pas_score.mean().detach().item()),
        "pdp": float(pdp_score.mean().detach().item()),
        "mass": float(mass_loss.detach().item()),
        "coordinate": float(coordinate_loss.detach().item()),
        "port": float(port_loss.detach().item()),
        "existence": float(existence_loss.detach().item()),
        "gate_pas": float(output["gate"][:, 0].mean().detach().item()),
        "gate_pdp": float(output["gate"][:, 1].mean().detach().item()),
    }


class CompleteValidationBank:
    def __init__(self, rows: np.ndarray, device: torch.device) -> None:
        self.rows = np.asarray(rows, dtype=np.int64)
        base = np.load(e213.TRAIN_BASE_H, mmap_mode="r")
        truth = np.load(e213.TRAIN_H, mmap_mode="r")
        self.base = torch.as_tensor(
            np.array(base[self.rows], dtype=np.complex64, copy=True), device=device
        )
        self.truth = torch.as_tensor(
            np.array(truth[self.rows], dtype=np.complex64, copy=True), device=device
        )
        self.baseline = aggregate_statistics(
            point_metric_statistics_torch(self.base, self.truth)
        )


@torch.inference_mode()
def predict_endpoints(
    model: GeometryPathModel,
    store: FeatureStore,
    rows: np.ndarray,
    normalization: dict[str, np.ndarray],
    config: Config,
    base_pas: np.ndarray | None = None,
    base_pdp: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    output_pas = np.zeros((len(rows), MP, MH, MV, N_UE), dtype=np.float32)
    output_pdp = np.zeros((len(rows), MP, N_UE, S_SC), dtype=np.float32)
    for start in range(0, len(rows), config.prediction_batch_size):
        end = min(start + config.prediction_batch_size, len(rows))
        local_pas = None if base_pas is None else base_pas[start:end]
        local_pdp = None if base_pdp is None else base_pdp[start:end]
        batch = store.batch(
            rows[start:end],
            normalization,
            base_pas=local_pas,
            base_pdp=local_pdp,
            include_targets=False,
        )
        with torch.autocast(
            device_type=store.device.type,
            dtype=torch.bfloat16,
            enabled=store.device.type == "cuda",
        ):
            prediction = model(batch)
        output_pas[start:end] = prediction["pas"].cpu().numpy().astype(np.float32)
        output_pdp[start:end] = prediction["pdp"].cpu().numpy().astype(np.float32)
    return output_pas, output_pdp


@torch.inference_mode()
def evaluate_complete(
    model: GeometryPathModel,
    store: FeatureStore,
    bank: CompleteValidationBank,
    normalization: dict[str, np.ndarray],
    config: Config,
) -> dict[str, float | int]:
    prediction_pas, prediction_pdp = predict_endpoints(
        model, store, bank.rows, normalization, config
    )
    base_pas = store.base_pas[torch.as_tensor(bank.rows, device=store.device)]
    base_pdp = store.base_pdp[torch.as_tensor(bank.rows, device=store.device)]
    prediction_pas_tensor = torch.as_tensor(prediction_pas, device=store.device)
    prediction_pdp_tensor = torch.as_tensor(prediction_pdp, device=store.device)
    parts: list[PointMetricStatistics] = []
    for start in range(0, len(bank.rows), config.channel_batch_size):
        end = min(start + config.channel_batch_size, len(bank.rows))
        candidate = e213.project_channel(
            bank.base[start:end],
            base_pas[start:end],
            base_pdp[start:end],
            prediction_pas_tensor[start:end],
            prediction_pdp_tensor[start:end],
            config,
        )
        parts.append(point_metric_statistics_torch(candidate, bank.truth[start:end]))
    return aggregate_statistics(e213.concatenate_statistics(parts))


def average_epoch_metrics(values: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: float(np.mean([value[name] for value in values]))
        for name in values[0]
    }


def train_phase(
    model: GeometryPathModel,
    store: FeatureStore,
    train_rows: np.ndarray,
    normalization: dict[str, np.ndarray],
    config: Config,
    epochs: int,
    seed: int,
    validation_bank: CompleteValidationBank | None,
) -> tuple[dict[str, torch.Tensor] | None, int, list[dict[str, object]], dict[str, float]]:
    if epochs == 0:
        return None, 0, [], {"training_seconds": 0.0, "validation_seconds": 0.0}
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=store.device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=config.learning_rate * 0.06
    )
    rng = np.random.default_rng(seed)
    active_rows = store.active_rows(train_rows)
    if len(active_rows) == 0:
        raise RuntimeError(f"no active training rows for {store.variant}")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_score = -math.inf
    history: list[dict[str, object]] = []
    training_seconds = 0.0
    validation_seconds = 0.0
    if validation_bank is not None:
        best_score = float(validation_bank.baseline["score"])
        history.append(
            {"epoch": 0, "train": None, "validation": validation_bank.baseline}
        )
        print(
            f"E214 variant={store.variant} epoch=0 validation={best_score:.9f}",
            flush=True,
        )
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(active_rows)
        epoch_values: list[dict[str, float]] = []
        phase_started = time.time()
        for start in range(0, len(order), config.batch_size):
            rows = order[start : start + config.batch_size]
            batch = store.batch(rows, normalization)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=store.device.type,
                dtype=torch.bfloat16,
                enabled=store.device.type == "cuda",
            ):
                output = model(batch)
                loss, metrics = training_objective(output, batch, config)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite loss variant={store.variant} epoch={epoch}"
                )
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            if not gradients or not all(bool(torch.all(torch.isfinite(item))) for item in gradients):
                raise FloatingPointError(
                    f"non-finite gradient variant={store.variant} epoch={epoch}"
                )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            epoch_values.append(metrics)
        training_seconds += time.time() - phase_started
        scheduler.step()
        train_metrics = average_epoch_metrics(epoch_values)
        record: dict[str, object] = {"epoch": epoch, "train": train_metrics}
        if validation_bank is not None and epoch >= config.minimum_epochs and (
            epoch % config.validation_interval == 0 or epoch == epochs
        ):
            validation_started = time.time()
            validation = evaluate_complete(
                model, store, validation_bank, normalization, config
            )
            validation_seconds += time.time() - validation_started
            score = float(validation["score"])
            record["validation"] = validation
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
        history.append(record)
        if epoch % config.validation_interval == 0 or epoch == epochs:
            score_text = (
                f"{float(record['validation']['score']):.9f}"
                if "validation" in record
                else "pending"
            )
            print(
                f"E214 variant={store.variant} epoch={epoch}/{epochs} "
                f"loss={train_metrics['loss']:.6f} validation={score_text} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
    return best_state, best_epoch, history, {
        "training_seconds": float(training_seconds),
        "validation_seconds": float(validation_seconds),
    }


def finite_gradient_check(config: Config, device: torch.device) -> dict[str, object]:
    set_seed(config.random_state)
    store = FeatureStore("base_s1", device)
    rows = np.arange(256, dtype=np.int64)
    normalization = store.normalization(rows)
    active = store.active_rows(rows)[: min(4, config.batch_size)]
    if len(active) == 0:
        raise RuntimeError("finite-gradient check found no active rows")
    model = GeometryPathModel(store.point_raw.shape[1], config).to(device)
    batch = store.batch(active, normalization)
    output = model(batch)
    loss, _ = training_objective(output, batch, config)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    finite = bool(torch.isfinite(loss)) and bool(gradients) and all(
        bool(torch.all(torch.isfinite(item))) for item in gradients
    )
    result = {
        "finite": finite,
        "loss": float(loss.detach().item()),
        "batch": int(len(active)),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    del model, store, batch, output, loss, gradients
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not finite:
        raise FloatingPointError("E214 finite-gradient check failed")
    return result


def train_outer(
    variant: str,
    outer: int,
    config: Config,
    device: torch.device,
    force: bool,
) -> dict[str, object]:
    validate_variant(variant)
    if outer not in range(5):
        raise ValueError("outer must be in [0, 4]")
    report_path = fold_report_path(variant, outer)
    if report_path.is_file() and endpoint_path(variant, outer).is_file() and not force:
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("status") == "complete" and previous.get("config") == config_payload(config):
            return {"status": "reused", **previous}
    if not PREPARE_REPORT.is_file():
        raise FileNotFoundError("E214 prepare stage has not completed")
    started = time.time()
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    store = FeatureStore(variant, device)
    metadata = np.load(e213.OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    canonical_fold = metadata["fold_id"].astype(np.int8)
    broad_fold = np.load(e213.FULL_FOLD, allow_pickle=False).astype(np.int8)
    validation_fold = (outer + 1) % 5
    screen_fit = np.flatnonzero((broad_fold != outer) & (broad_fold != validation_fold))
    validation_rows = np.flatnonzero(broad_fold == validation_fold)
    refit_rows = np.flatnonzero(broad_fold != outer)
    canonical_rows = np.flatnonzero(canonical_fold == outer)
    held_ids = query_ids[canonical_rows]

    screen_normalization = store.normalization(screen_fit)
    validation_bank = CompleteValidationBank(validation_rows, device)
    set_seed(config.random_state + outer * 10000 + 11)
    screen_model = GeometryPathModel(store.point_raw.shape[1], config).to(device)
    best_state, selected_epoch, screen_history, screen_timing = train_phase(
        screen_model,
        store,
        screen_fit,
        screen_normalization,
        config,
        config.epochs,
        config.random_state + outer * 10000 + 101,
        validation_bank,
    )
    del validation_bank, screen_model, screen_normalization
    if device.type == "cuda":
        torch.cuda.empty_cache()

    canonical_base_pas_all = np.load(e213.CANONICAL_BASE_PAS, mmap_mode="r")
    canonical_base_pdp_all = np.load(e213.CANONICAL_BASE_PDP, mmap_mode="r")
    held_base_pas = np.asarray(canonical_base_pas_all[canonical_rows], dtype=np.float32)
    held_base_pdp = np.asarray(canonical_base_pdp_all[canonical_rows], dtype=np.float32)
    refit_timing = {"training_seconds": 0.0, "validation_seconds": 0.0}
    if selected_epoch == 0 or best_state is None:
        prediction_pas = held_base_pas.copy()
        prediction_pdp = held_base_pdp.copy()
        checkpoint_payload: dict[str, object] = {
            "experiment": VERSION,
            "variant": variant,
            "outer": outer,
            "selected_epoch": 0,
            "config": config_payload(config),
            "state_dict": None,
            "fallback": "V7",
        }
    else:
        refit_normalization = store.normalization(refit_rows)
        set_seed(config.random_state + outer * 10000 + 29)
        model = GeometryPathModel(store.point_raw.shape[1], config).to(device)
        _, _, _, refit_timing = train_phase(
            model,
            store,
            refit_rows,
            refit_normalization,
            config,
            selected_epoch,
            config.random_state + outer * 10000 + 303,
            None,
        )
        prediction_pas, prediction_pdp = predict_endpoints(
            model,
            store,
            held_ids,
            refit_normalization,
            config,
            base_pas=held_base_pas,
            base_pdp=held_base_pdp,
        )
        checkpoint_payload = {
            "experiment": VERSION,
            "variant": variant,
            "outer": outer,
            "validation_fold": validation_fold,
            "selected_epoch": selected_epoch,
            "config": config_payload(config),
            "state_dict": model.state_dict(),
            "normalization": refit_normalization,
            "forbidden_rt_fields_used": [],
        }
        del model
    atomic_torch(checkpoint_path(variant, outer), checkpoint_payload)
    atomic_npz(
        endpoint_path(variant, outer),
        query_ids=held_ids.astype(np.int32),
        canonical_rows=canonical_rows.astype(np.int32),
        canonical_fold=np.full(len(held_ids), outer, dtype=np.int8),
        broad_fold=broad_fold[held_ids].astype(np.int8),
        pas=prediction_pas.astype(np.float32),
        pdp=prediction_pdp.astype(np.float32),
    )
    endpoint = e213.endpoint_summary(
        prediction_pas,
        prediction_pdp,
        store.target_pas[torch.as_tensor(held_ids, device=device)],
        store.target_pdp[torch.as_tensor(held_ids, device=device)],
    )
    label_access_risk = broad_fold[held_ids] != outer
    report = {
        "experiment": VERSION,
        "status": "complete",
        "variant": variant,
        "outer": outer,
        "validation_fold": validation_fold,
        "config": config_payload(config),
        "selected_epoch": selected_epoch,
        "counts": {
            "screen_fit": int(len(screen_fit)),
            "validation": int(len(validation_rows)),
            "refit": int(len(refit_rows)),
            "held": int(len(held_ids)),
            "held_label_access_risk": int(np.sum(label_access_risk)),
            "held_lineage_match": int(np.sum(~label_access_risk)),
            "active_screen": int(len(store.active_rows(screen_fit))),
            "active_refit": int(len(store.active_rows(refit_rows))),
        },
        "screen_history": screen_history,
        "endpoint": endpoint,
        "timing": {
            "screen": screen_timing,
            "refit": refit_timing,
            "wall_seconds": float(time.time() - started),
        },
        "resources": {
            "device": str(device),
            "cuda_peak_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0,
            "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "data_lineage": {
            "topology_variant": variant,
            "official_h_labels": True,
            "v7_base": True,
            "allowed_rt_fields": list(topology.ALLOWED_FIELDS),
            "forbidden_rt_fields_used": [],
        },
        "checkpoint": PATHS.relative(checkpoint_path(variant, outer)),
        "endpoints": PATHS.relative(endpoint_path(variant, outer)),
        "exit_code": 0,
    }
    atomic_json(report_path, report)
    return report


def aggregate_variant(
    variant: str, config: Config, device: torch.device
) -> dict[str, object]:
    validate_variant(variant)
    started = time.time()
    metadata = np.load(e213.OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    canonical_fold = metadata["fold_id"].astype(np.int8)
    prediction_pas = np.zeros((len(query_ids), MP, MH, MV, N_UE), dtype=np.float32)
    prediction_pdp = np.zeros((len(query_ids), MP, N_UE, S_SC), dtype=np.float32)
    seen = np.zeros(len(query_ids), dtype=np.uint8)
    fold_reports: list[dict[str, Any]] = []
    for outer in range(5):
        report = json.loads(fold_report_path(variant, outer).read_text(encoding="utf-8"))
        if report.get("status") != "complete" or report.get("config") != config_payload(config):
            raise RuntimeError(f"invalid completed fold {variant}/outer{outer}")
        with np.load(endpoint_path(variant, outer), allow_pickle=False) as payload:
            rows = payload["canonical_rows"].astype(np.int64)
            if not np.array_equal(query_ids[rows], payload["query_ids"].astype(np.int64)):
                raise RuntimeError(f"query identity mismatch {variant}/outer{outer}")
            if np.any(seen[rows]) or not np.all(canonical_fold[rows] == outer):
                raise RuntimeError(f"fold coverage mismatch {variant}/outer{outer}")
            prediction_pas[rows] = payload["pas"].astype(np.float32)
            prediction_pdp[rows] = payload["pdp"].astype(np.float32)
            seen[rows] = 1
        fold_reports.append(report)
    if not np.all(seen == 1):
        raise RuntimeError(f"incomplete canonical coverage for {variant}")
    label_access_risk = sum(
        int(report["counts"]["held_label_access_risk"]) for report in fold_reports
    )
    if label_access_risk != 376:
        raise RuntimeError(f"unexpected label access risk for {variant}: {label_access_risk}")

    base_source = np.load(e213.CANONICAL_BASE_H, mmap_mode="r")
    truth_source = np.load(e213.TRAIN_H, mmap_mode="r")
    base_pas_source = np.load(e213.CANONICAL_BASE_PAS, mmap_mode="r")
    base_pdp_source = np.load(e213.CANONICAL_BASE_PDP, mmap_mode="r")
    output_path = output_channel_path(variant)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f".{output_path.name}.partial-{os.getpid()}")
    output = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.complex64,
        shape=(len(query_ids), 256, N_UE, S_SC),
    )
    candidate_parts: list[PointMetricStatistics] = []
    baseline_parts: list[PointMetricStatistics] = []
    for start in range(0, len(query_ids), config.channel_batch_size):
        end = min(start + config.channel_batch_size, len(query_ids))
        base = torch.as_tensor(
            np.array(base_source[start:end], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(truth_source[query_ids[start:end]], dtype=np.complex64, copy=True),
            device=device,
        )
        candidate = e213.project_channel(
            base,
            torch.as_tensor(np.asarray(base_pas_source[start:end]), device=device),
            torch.as_tensor(np.asarray(base_pdp_source[start:end]), device=device),
            torch.as_tensor(prediction_pas[start:end], device=device),
            torch.as_tensor(prediction_pdp[start:end], device=device),
            config,
        )
        output[start:end] = candidate.cpu().numpy().astype(np.complex64)
        candidate_parts.append(point_metric_statistics_torch(candidate, truth))
        baseline_parts.append(point_metric_statistics_torch(base, truth))
        if end == len(query_ids) or end % 100 == 0:
            output.flush()
            print(f"E214 aggregate variant={variant} {end}/{len(query_ids)}", flush=True)
    output.flush()
    del output
    os.replace(partial, output_path)
    candidate_statistics = e213.concatenate_statistics(candidate_parts)
    baseline_statistics = e213.concatenate_statistics(baseline_parts)
    candidate = aggregate_statistics(candidate_statistics)
    baseline = aggregate_statistics(baseline_statistics)
    folds = []
    for outer in range(5):
        mask = canonical_fold == outer
        fold_candidate = aggregate_statistics(candidate_statistics, mask)
        fold_baseline = aggregate_statistics(baseline_statistics, mask)
        folds.append(
            {
                "outer": outer,
                "candidate": fold_candidate,
                "baseline": fold_baseline,
                "delta": float(fold_candidate["score"] - fold_baseline["score"]),
            }
        )
    target_pas = torch.as_tensor(
        np.asarray(np.load(e213.OFFICIAL_PAS, mmap_mode="r")[query_ids]), device=device
    )
    target_pdp = torch.as_tensor(
        np.asarray(np.load(e213.OFFICIAL_PDP, mmap_mode="r")[query_ids]), device=device
    )
    endpoint = e213.endpoint_summary(
        prediction_pas, prediction_pdp, target_pas, target_pdp
    )
    output_check = np.load(output_path, mmap_mode="r")
    v16 = {
        "pas": 0.6769415758539015,
        "pdp": 0.7803473980573273,
        "nmse": 0.9331251955178452,
        "nmse_score": 0.5172970702149067,
        "score": 0.6863750036074728,
    }
    report = {
        "experiment": VERSION,
        "status": "complete",
        "variant": variant,
        "protocol": {
            "name": "historical_v16_compatibility",
            "query_count": 500,
            "fold_lineage_match": 124,
            "label_access_risk": label_access_risk,
            "strict_score_used_for_gate": False,
        },
        "config": config_payload(config),
        "baseline_v7": baseline,
        "candidate": candidate,
        "delta_vs_v7": {
            key: float(candidate[key] - baseline[key])
            for key in ("pas", "pdp", "nmse_score", "score")
        },
        "v16_reference": v16,
        "delta_vs_v16": {
            key: float(candidate[key] - v16[key])
            for key in ("pas", "pdp", "nmse_score", "score")
        },
        "success_threshold": 0.680000,
        "passed": bool(candidate["score"] >= 0.680000),
        "endpoint": endpoint,
        "folds": folds,
        "selected_epochs": [int(report["selected_epoch"]) for report in fold_reports],
        "output": {
            "path": PATHS.relative(output_path),
            "shape": list(output_check.shape),
            "dtype": str(output_check.dtype),
            "finite": bool(np.all(np.isfinite(output_check))),
        },
        "data_lineage": {
            "official_h_labels": True,
            "v7_base": True,
            "topology_variant": variant,
            "allowed_rt_fields": list(topology.ALLOWED_FIELDS),
            "forbidden_rt_fields_used": [],
        },
        "runtime": {
            "aggregate_wall_seconds": float(time.time() - started),
            "fold_wall_seconds_max": float(
                max(report["timing"]["wall_seconds"] for report in fold_reports)
            ),
            "fold_wall_seconds_sum": float(
                sum(report["timing"]["wall_seconds"] for report in fold_reports)
            ),
            "measured_bottleneck": "path sequence/set encoding and complete-H validation",
        },
        "resources": {
            "aggregate_device": str(device),
            "aggregate_cuda_peak_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0,
            "aggregate_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "exit_code": 0,
    }
    atomic_json(final_report_path(variant), report)
    return report


def status(config: Config) -> dict[str, object]:
    return {
        "experiment": VERSION,
        "prepare": PREPARE_REPORT.is_file(),
        "variants": {
            variant: {
                "folds": [fold_report_path(variant, outer).is_file() for outer in range(5)],
                "aggregate": final_report_path(variant).is_file(),
            }
            for variant in VARIANTS
        },
        "config": config_payload(config),
    }


def main() -> None:
    args = parse_args()
    config = Config(epochs=args.epochs, batch_size=args.batch_size)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if args.stage == "prepare":
            result: object = prepare(config, device, args.force)
        elif args.stage == "train":
            if args.variant == "all":
                raise ValueError("train requires one concrete --variant")
            if args.outer is None:
                raise ValueError("train requires --outer")
            result = train_outer(args.variant, args.outer, config, device, args.force)
        elif args.stage == "aggregate":
            variants = VARIANTS if args.variant == "all" else (args.variant,)
            result = {
                variant: aggregate_variant(variant, config, device) for variant in variants
            }
        else:
            result = status(config)
        print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False), flush=True)
    except Exception as error:
        payload = {
            "experiment": VERSION,
            "status": "failed",
            "stage": args.stage,
            "variant": args.variant,
            "outer": args.outer,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "exit_code": 1,
        }
        atomic_json(failure_report_path(args.stage, args.variant, args.outer), payload)
        raise


if __name__ == "__main__":
    main()
