#!/usr/bin/env python3
"""Compare compressed and full multiseed histories against official path states."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
import os
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

import analyze_global_local_path_lifecycle as e79
from common.official_metrics import (
    PointMetricStatistics,
    aggregate_statistics,
    point_metric_statistics_torch,
)
from data import load_positions
from project_paths import PATHS
from run_path_component_transport import component_energy, synthesize_batch


VERSION = "Round2-E192-multiseed-relation-probe"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e192_multiseed_path_generator"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e192_multiseed_path_generator"
ANCHOR_DIR = ARTIFACT_DIR / "event_anchors_train"
HISTORY_DIR = ARTIFACT_DIR / "event_history_train"
PROBE_DIR = ARTIFACT_DIR / "relation_probe"
FEATURE_CACHE = PROBE_DIR / "member_feature_f16.npy"
FEATURE_COMPLETE = PROBE_DIR / "member_feature_complete.json"
LABEL_CACHE = PROBE_DIR / "official_path_labels.npz"
REPORT = ANALYSIS_DIR / "relation_probe.json"
CAPABILITY_REPORT = ANALYSIS_DIR / "relation_probe_capability.json"
PREDICTIONS = PROBE_DIR / "relation_probe_predictions.npz"
DIRECT_REPORT = ANALYSIS_DIR / "direct_history_fixed500.json"
DIRECT_CHANNEL = PROBE_DIR / "direct_history_fixed500_channel.npy"
DIRECT_FOLD_DIR = PROBE_DIR / "direct_history_folds"

PATH_CACHE = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_oracle"
    / "n4000_s24_k128_b096_d064_scatter1_sel106"
)
COMPONENTS = (
    PATHS.artifacts
    / "round2"
    / "v4_path_component_transport"
    / "train_components_k24_os2_ref2.npz"
)
BASE_COORDINATE = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "sionna_aligned_coordinates_f32.npy"
)
POINT_POWER = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "official_point_power_f64.npy"
)
FULL_FOLD = (
    PATHS.artifacts / "round2" / "full_path_pdp_link_gate" / "train_fold_id.npy"
)
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"

SEEDS = 24
EVENTS = 128
PORTS = 8
PERIODS = np.asarray((16.0, 8.0, 192.0), dtype=np.float32)
COORDINATE_LIMITS = np.asarray((1.5, 1.0, 3.0), dtype=np.float32)
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    hidden: int = 96
    heads: int = 4
    seed_layers: int = 1
    event_layers: int = 2
    feedforward: int = 256
    dropout: float = 0.08
    epochs: int = 36
    minimum_epochs: int = 12
    validation_interval: int = 4
    batch_size: int = 96
    learning_rate: float = 5e-4
    weight_decay: float = 2e-4
    minimum_fraction: float = 1e-5
    match_cost_limit: float = 0.25
    match_bandwidth: float = 0.25
    random_state: int = 20260813192
    minimum_positive_folds: int = 4
    minimum_shuffle_error_reduction: float = 0.10
    minimum_subspace_score: float = 0.710000
    capability_points: int = 32
    capability_nonzero_points: int = 24
    capability_steps: int = 160
    capability_minimum_composite: float = 0.80
    capability_minimum_improvement: float = 0.15
    capability_minimum_loss_reduction: float = 0.50
    official_epochs: int = 12
    official_batch_size: int = 16
    official_learning_rate: float = 1e-4
    official_validation_interval: int = 2


@dataclass
class Labels:
    mass_absolute: np.ndarray
    mass_normalized: np.ndarray
    delta: np.ndarray
    port: np.ndarray
    phase: np.ndarray
    active: np.ndarray
    point_power: np.ndarray
    zero: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "prepare",
            "capability",
            "screen",
            "full",
            "direct",
            "direct-finalize",
        ),
        default="screen",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=Config().batch_size)
    parser.add_argument("--epochs", type=int, default=Config().epochs)
    parser.add_argument(
        "--official-epochs", type=int, default=Config().official_epochs
    )
    parser.add_argument(
        "--official-batch-size", type=int, default=Config().official_batch_size
    )
    parser.add_argument(
        "--direct-outer",
        type=int,
        choices=range(5),
        action="append",
        help="Run only the selected direct outer fold; repeat for multiple folds.",
    )
    parser.add_argument("--force-prepare", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npy")
    np.save(temporary, value)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def direct_fold_paths(outer: int) -> dict[str, Path]:
    return {
        "checkpoint": DIRECT_FOLD_DIR / f"outer{outer}_checkpoint.pt",
        "prediction": DIRECT_FOLD_DIR / f"outer{outer}_prediction.npy",
        "rows": DIRECT_FOLD_DIR / f"outer{outer}_rows.npy",
        "report": DIRECT_FOLD_DIR / f"outer{outer}_report.json",
        "error": DIRECT_FOLD_DIR / f"outer{outer}_error.json",
    }


def required_inputs() -> tuple[Path, ...]:
    return (
        ANCHOR_DIR / "mask.npy",
        ANCHOR_DIR / "representative_slot.npy",
        HISTORY_DIR / "complete.json",
        HISTORY_DIR / "mask.npy",
        HISTORY_DIR / "continuous.npy",
        HISTORY_DIR / "reference_coefficient.npy",
        HISTORY_DIR / "port_energy.npy",
        HISTORY_DIR / "interaction_count.npy",
        HISTORY_DIR / "interactions.npy",
        HISTORY_DIR / "objects.npy",
        HISTORY_DIR / "primitives.npy",
        HISTORY_DIR / "vertices.npy",
        BASE_COORDINATE,
        COMPONENTS,
        POINT_POWER,
        FULL_FOLD,
        OOF_META,
        TRAIN_CHANNEL,
    )


def unit_complex(value: np.ndarray) -> np.ndarray:
    magnitude = np.abs(value)
    return np.divide(
        value,
        np.maximum(magnitude, 1e-12),
        out=np.ones_like(value, dtype=np.complex64),
    ).astype(np.complex64)


def normalized_rows(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    positive = np.maximum(np.asarray(value, dtype=np.float64), 0.0) * mask
    total = positive.sum(axis=1, keepdims=True, dtype=np.float64)
    return np.divide(
        positive,
        np.maximum(total, TINY),
        out=np.zeros_like(positive),
    ).astype(np.float32)


def sequence_hash_features(
    interactions: np.ndarray,
    objects: np.ndarray,
    primitives: np.ndarray,
    depth: np.ndarray,
) -> np.ndarray:
    shape = depth.shape
    first = np.full(shape, np.uint64(1469598103934665603), dtype=np.uint64)
    second = np.full(shape, np.uint64(1099511628211), dtype=np.uint64)
    for step in range(interactions.shape[-1]):
        active = step < depth
        value = (
            interactions[..., step].astype(np.uint64)
            ^ (objects[..., step].astype(np.uint64) * np.uint64(0x9E3779B185EBCA87))
            ^ (
                primitives[..., step].astype(np.uint64)
                * np.uint64(0xC2B2AE3D27D4EB4F)
            )
            ^ np.uint64(step + 1)
        )
        first = np.where(active, (first ^ value) * np.uint64(1099511628211), first)
        second = np.where(
            active,
            (second ^ (value >> np.uint64(1))) * np.uint64(0x9E3779B185EBCA87),
            second,
        )
    scale = float(2**32)
    return np.stack(
        (
            (first & np.uint64(0xFFFFFFFF)).astype(np.float32) / scale,
            (first >> np.uint64(32)).astype(np.float32) / scale,
            (second & np.uint64(0xFFFFFFFF)).astype(np.float32) / scale,
            (second >> np.uint64(32)).astype(np.float32) / scale,
        ),
        axis=-1,
    )


def vertex_features(vertices: np.ndarray, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    step = np.arange(vertices.shape[-2], dtype=np.int64)
    valid = step < depth[..., None]
    weight = valid.astype(np.float32)
    denominator = weight.sum(axis=-1, keepdims=True).clip(min=1.0)
    mean = (vertices * weight[..., None]).sum(axis=-2) / denominator
    first = vertices[..., 0, :]
    last_index = np.maximum(depth.astype(np.int64) - 1, 0)
    last = np.take_along_axis(
        vertices, last_index[..., None, None].repeat(3, axis=-1), axis=-2
    )[..., 0, :]
    segment = np.diff(vertices, axis=-2)
    segment_valid = step[1:] < depth[..., None]
    length = (
        np.linalg.norm(segment, axis=-1) * segment_valid.astype(np.float32)
    ).sum(axis=-1, keepdims=True)
    output = np.concatenate((first, last, mean, np.log1p(length)), axis=-1)
    output *= mask[..., None]
    return output.astype(np.float32)


def build_member_features(force: bool) -> dict[str, object]:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    if FEATURE_CACHE.is_file() and FEATURE_COMPLETE.is_file() and not force:
        value = np.load(FEATURE_CACHE, mmap_mode="r")
        marker = json.loads(FEATURE_COMPLETE.read_text(encoding="utf-8"))
        expected = (SEEDS, 4000, EVENTS, 69)
        if value.shape != expected or str(value.dtype) != "float16":
            raise RuntimeError("invalid completed E192 member feature cache")
        if marker.get("shape") != list(expected) or marker.get("dtype") != "float16":
            raise RuntimeError("E192 member feature marker does not match cache")
        return {
            "reused": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    started = time.time()
    FEATURE_COMPLETE.unlink(missing_ok=True)
    temporary_cache = FEATURE_CACHE.with_suffix(".tmp.npy")
    temporary_cache.unlink(missing_ok=True)
    mask = np.load(HISTORY_DIR / "mask.npy", mmap_mode="r")
    continuous = np.load(HISTORY_DIR / "continuous.npy", mmap_mode="r")
    coefficient = np.load(
        HISTORY_DIR / "reference_coefficient.npy", mmap_mode="r"
    )
    port_energy = np.load(HISTORY_DIR / "port_energy.npy", mmap_mode="r")
    depth = np.load(HISTORY_DIR / "interaction_count.npy", mmap_mode="r")
    interactions = np.load(HISTORY_DIR / "interactions.npy", mmap_mode="r")
    objects = np.load(HISTORY_DIR / "objects.npy", mmap_mode="r")
    primitives = np.load(HISTORY_DIR / "primitives.npy", mmap_mode="r")
    vertices = np.load(HISTORY_DIR / "vertices.npy", mmap_mode="r")
    output = np.lib.format.open_memmap(
        temporary_cache,
        mode="w+",
        dtype=np.float16,
        shape=(SEEDS, 4000, EVENTS, 69),
    )
    source = np.load(ANCHOR_DIR / "source.npy", mmap_mode="r")

    for seed in range(SEEDS):
        local_mask = np.asarray(mask[seed], dtype=bool)
        local_continuous = np.asarray(continuous[seed], dtype=np.float32)
        tau = local_continuous[..., 0] * 30_000.0
        angle = local_continuous[..., 1:5]
        angle_cycle = np.concatenate((np.sin(angle), np.cos(angle)), axis=-1)
        energy = np.stack(
            (
                np.log10(np.maximum(local_continuous[..., 5], 1e-30)) / 30.0,
                np.log10(np.maximum(local_continuous[..., 6], 1e-12)) / 12.0,
                np.sqrt(np.maximum(local_continuous[..., 6], 0.0)),
            ),
            axis=-1,
        )
        local_coefficient = np.asarray(coefficient[seed], dtype=np.complex64)
        coefficient_energy = np.abs(local_coefficient) ** 2
        coefficient_total = coefficient_energy.sum(axis=(-2, -1), keepdims=True)
        coefficient_port = coefficient_energy / np.maximum(coefficient_total, TINY)
        coefficient_phase = unit_complex(local_coefficient)
        local_port_energy = np.array(port_energy[seed], dtype=np.float32, copy=True)
        local_port_energy /= np.maximum(
            local_port_energy.sum(axis=(-2, -1), keepdims=True), TINY
        )
        local_depth = np.asarray(depth[seed], dtype=np.uint8)
        local_interactions = np.asarray(interactions[seed], dtype=np.uint8)
        interaction_hist = np.stack(
            [
                np.sum(
                    (local_interactions == interaction) & (
                        np.arange(local_interactions.shape[-1]) < local_depth[..., None]
                    ),
                    axis=-1,
                )
                for interaction in range(8)
            ],
            axis=-1,
        ).astype(np.float32) / 16.0
        hashes = sequence_hash_features(
            local_interactions,
            np.asarray(objects[seed]),
            np.asarray(primitives[seed]),
            local_depth,
        )
        vertex = vertex_features(
            np.asarray(vertices[seed], dtype=np.float32), local_depth, local_mask
        )
        feature = np.concatenate(
            (
                tau[..., None],
                angle_cycle,
                energy,
                coefficient_port.reshape(4000, EVENTS, PORTS),
                coefficient_phase.real.reshape(4000, EVENTS, PORTS),
                coefficient_phase.imag.reshape(4000, EVENTS, PORTS),
                local_port_energy.reshape(4000, EVENTS, PORTS),
                (local_depth[..., None].astype(np.float32) / 16.0),
                interaction_hist,
                hashes,
                vertex / 200.0,
                (source == 0)[..., None].astype(np.float32),
                (source == 1)[..., None].astype(np.float32),
            ),
            axis=-1,
        ).astype(np.float32)
        if feature.shape[-1] != output.shape[-1]:
            raise RuntimeError(f"unexpected E192 member feature width {feature.shape[-1]}")
        feature *= local_mask[..., None]
        output[seed] = feature.astype(np.float16)
        output.flush()
        print(f"E192 member features seed={seed + 1}/{SEEDS}", flush=True)
    del output
    os.replace(temporary_cache, FEATURE_CACHE)
    complete = {
        "status": "complete",
        "shape": [SEEDS, 4000, EVENTS, 69],
        "dtype": "float16",
        "bytes": FEATURE_CACHE.stat().st_size,
    }
    FEATURE_COMPLETE.write_text(
        json.dumps(complete, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "reused": False,
        "shape": [SEEDS, 4000, EVENTS, 69],
        "dtype": "float16",
        "seconds": float(time.time() - started),
    }


def build_labels(force: bool, config: Config) -> tuple[Labels, dict[str, object]]:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    if LABEL_CACHE.is_file() and not force:
        payload = np.load(LABEL_CACHE, allow_pickle=False)
        labels = Labels(
            mass_absolute=payload["mass_absolute"].astype(np.float32),
            mass_normalized=payload["mass_normalized"].astype(np.float32),
            delta=payload["delta"].astype(np.float32),
            port=payload["port"].astype(np.float32),
            phase=payload["phase"].astype(np.complex64),
            active=payload["active"].astype(bool),
            point_power=payload["point_power"].astype(np.float64),
            zero=payload["zero"].astype(bool),
        )
        return labels, json.loads(str(payload["report_json"]))

    started = time.time()
    anchor_mask = np.load(ANCHOR_DIR / "mask.npy", mmap_mode="r")
    history_mask = np.load(HISTORY_DIR / "mask.npy", mmap_mode="r")
    candidate_mask = np.asarray(anchor_mask, dtype=bool) & np.any(
        np.asarray(history_mask, dtype=bool), axis=0
    )
    representative = np.load(
        ANCHOR_DIR / "representative_slot.npy", mmap_mode="r"
    )
    raw_coordinate = np.load(BASE_COORDINATE, mmap_mode="r")
    candidate_coordinate = np.zeros((4000, EVENTS, 3), dtype=np.float32)
    rows, events = np.nonzero(candidate_mask)
    slots = np.asarray(representative[rows, events], dtype=np.int64)
    candidate_coordinate[rows, events] = raw_coordinate[rows, slots]

    component = np.load(COMPONENTS, allow_pickle=False)
    truth_coordinate = component["coordinates"].astype(np.float32)
    truth_coefficient = component["coefficients"].astype(np.complex64)
    point_power = component["point_power"].astype(np.float64)
    mass_absolute = np.zeros((4000, EVENTS), dtype=np.float32)
    delta = np.zeros((4000, EVENTS, 3), dtype=np.float32)
    port = np.full((4000, EVENTS, PORTS), 1.0 / PORTS, dtype=np.float32)
    phase = np.ones((4000, EVENTS, PORTS), dtype=np.complex64)
    matched_count = np.zeros(4000, dtype=np.int16)
    matched_fraction = np.zeros(4000, dtype=np.float64)
    available_fraction = np.zeros(4000, dtype=np.float64)

    for row in range(4000):
        candidate_ids = np.flatnonzero(candidate_mask[row])
        truth_energy = component_energy(truth_coefficient[row])
        truth_fraction = truth_energy / max(float(point_power[row]), TINY)
        truth_ids = np.flatnonzero(truth_fraction >= config.minimum_fraction)
        available_fraction[row] = float(np.sum(truth_fraction[truth_ids]))
        if len(candidate_ids) == 0 or len(truth_ids) == 0:
            continue
        cost = e79.support_cost(
            candidate_coordinate[row, candidate_ids], truth_coordinate[row, truth_ids]
        )
        utility = truth_fraction[truth_ids][None] * np.exp(
            -0.5 * cost / config.match_bandwidth
        )
        utility[cost > config.match_cost_limit] = 0.0
        candidate_local, truth_local = linear_sum_assignment(-utility)
        accepted = utility[candidate_local, truth_local] > 0.0
        candidate_local = candidate_local[accepted]
        truth_local = truth_local[accepted]
        assigned_candidate = candidate_ids[candidate_local]
        assigned_truth = truth_ids[truth_local]
        if len(assigned_candidate) == 0:
            continue
        coefficient = truth_coefficient[row, assigned_truth]
        mass_absolute[row, assigned_candidate] = truth_fraction[assigned_truth]
        for axis, period in enumerate(PERIODS):
            delta[row, assigned_candidate, axis] = e79.periodic_delta(
                truth_coordinate[row, assigned_truth, axis],
                candidate_coordinate[row, assigned_candidate, axis],
                float(period),
            )
        coefficient_energy = np.abs(coefficient) ** 2
        port[row, assigned_candidate] = coefficient_energy / np.maximum(
            coefficient_energy.sum(axis=1, keepdims=True), TINY
        )
        phase[row, assigned_candidate] = unit_complex(coefficient)
        matched_count[row] = len(assigned_candidate)
        matched_fraction[row] = float(np.sum(truth_fraction[assigned_truth]))

    mass_normalized = normalized_rows(mass_absolute, candidate_mask)
    active = mass_absolute > 0.0
    zero = point_power <= TINY
    report = {
        "candidate_events": int(np.sum(candidate_mask)),
        "history_backed_ratio": float(np.sum(candidate_mask) / np.sum(anchor_mask)),
        "nonzero_count": int(np.sum(~zero)),
        "matched_count_nonzero_mean": float(np.mean(matched_count[~zero])),
        "matched_fraction_nonzero_mean": float(np.mean(matched_fraction[~zero])),
        "matched_fraction_of_available": float(
            np.sum(matched_fraction) / max(float(np.sum(available_fraction)), TINY)
        ),
        "seconds": float(time.time() - started),
    }
    np.savez_compressed(
        LABEL_CACHE,
        mass_absolute=mass_absolute,
        mass_normalized=mass_normalized,
        delta=delta,
        port=port,
        phase=phase,
        active=active,
        point_power=point_power,
        zero=zero,
        report_json=np.asarray(json.dumps(report, ensure_ascii=False)),
    )
    return Labels(
        mass_absolute, mass_normalized, delta, port, phase, active, point_power, zero
    ), report


def prepare(force: bool, config: Config) -> dict[str, object]:
    started = time.time()
    missing = [PATHS.relative(path) for path in required_inputs() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing E192 relation inputs: {missing}")
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    feature = build_member_features(force)
    _, labels = build_labels(force, config)
    return {
        "feature": feature,
        "labels": labels,
        "seconds": float(time.time() - started),
    }


class RelationStore:
    def __init__(self, device: torch.device, labels: Labels) -> None:
        self.device = device

        def tensor(path: Path, dtype: torch.dtype) -> torch.Tensor:
            return torch.tensor(np.asarray(np.load(path, mmap_mode="r")), dtype=dtype, device=device)

        self.member_feature = tensor(FEATURE_CACHE, torch.float16)
        self.member_mask = tensor(HISTORY_DIR / "mask.npy", torch.bool)
        self.anchor_mask = tensor(ANCHOR_DIR / "mask.npy", torch.bool)
        self.event_mask = self.anchor_mask & self.member_mask.any(dim=0)
        self.representative = tensor(ANCHOR_DIR / "representative_slot.npy", torch.int64)
        self.raw_coordinate = tensor(BASE_COORDINATE, torch.float32)
        point = torch.arange(4000, device=device)[:, None]
        valid_slot = self.representative.clamp_min(0)
        self.coordinate = self.raw_coordinate[point, valid_slot]
        self.coordinate = torch.where(
            self.event_mask[..., None], self.coordinate, torch.zeros_like(self.coordinate)
        )
        source = tensor(ANCHOR_DIR / "source.npy", torch.int64)
        coordinate_phase = 2.0 * torch.pi * self.coordinate / torch.tensor(
            PERIODS, device=device
        )
        self.event_feature = torch.cat(
            (
                torch.sin(coordinate_phase),
                torch.cos(coordinate_phase),
                self.coordinate / torch.tensor(PERIODS, device=device),
                (source == 0)[..., None].to(torch.float32),
                (source == 1)[..., None].to(torch.float32),
            ),
            dim=2,
        )
        setup, positions, _, site, _ = load_positions()
        self.site = torch.as_tensor(site, dtype=torch.int64, device=device)
        relative = positions - setup.bs_positions[site]
        radial = np.linalg.norm(relative[:, :2], axis=1)
        bearing = np.arctan2(relative[:, 1], relative[:, 0])
        context = np.stack(
            (
                relative[:, 0] / 150.0,
                relative[:, 1] / 150.0,
                relative[:, 2] / 50.0,
                radial / 200.0,
                np.sin(bearing),
                np.cos(bearing),
                site == 0,
                site == 1,
            ),
            axis=1,
        ).astype(np.float32)
        self.context = torch.as_tensor(context, device=device)
        self.fold = tensor(FULL_FOLD, torch.int64)
        self.target_mass = torch.as_tensor(labels.mass_normalized, device=device)
        self.target_delta = torch.as_tensor(labels.delta, device=device)
        self.target_port = torch.as_tensor(labels.port, device=device)
        self.target_phase = torch.as_tensor(labels.phase, device=device)
        self.target_active = torch.as_tensor(labels.active, device=device)
        self.point_power = torch.as_tensor(labels.point_power, dtype=torch.float64, device=device)
        self.zero = torch.as_tensor(labels.zero, device=device)
        self.presence = self.member_mask.sum(dim=0).to(torch.float32)
        continuous = np.load(HISTORY_DIR / "continuous.npy", mmap_mode="r")
        fraction = np.asarray(continuous[..., 6], dtype=np.float32)
        representative_seed = np.argmax(fraction, axis=0).astype(np.int64)
        self.representative_seed = torch.as_tensor(
            representative_seed, dtype=torch.int64, device=device
        )
        point_np = np.arange(4000, dtype=np.int64)[:, None]
        event_np = np.arange(EVENTS, dtype=np.int64)[None, :]
        selected_fraction = fraction[representative_seed, point_np, event_np]
        selected_energy = np.asarray(continuous[..., 5], dtype=np.float32)[
            representative_seed, point_np, event_np
        ]
        event_mask_np = np.asarray(
            np.load(ANCHOR_DIR / "mask.npy", mmap_mode="r"), dtype=bool
        ) & np.any(
            np.asarray(np.load(HISTORY_DIR / "mask.npy", mmap_mode="r"), dtype=bool),
            axis=0,
        )
        presence_np = np.asarray(
            np.load(HISTORY_DIR / "mask.npy", mmap_mode="r"), dtype=np.float32
        ).sum(axis=0)
        anchored = np.maximum(selected_fraction, 1e-12) * np.sqrt(
            np.maximum(presence_np, 1.0) / SEEDS
        )
        base_mass = normalized_rows(anchored, event_mask_np)
        reference = np.load(
            HISTORY_DIR / "reference_coefficient.npy", mmap_mode="r"
        )
        selected_reference = np.asarray(
            reference[representative_seed, point_np, event_np], dtype=np.complex64
        )
        reference_energy = np.abs(selected_reference) ** 2
        base_port = np.divide(
            reference_energy,
            np.maximum(reference_energy.sum(axis=(-2, -1), keepdims=True), TINY),
            out=np.full_like(reference_energy, 1.0 / PORTS, dtype=np.float32),
        ).reshape(4000, EVENTS, PORTS)
        base_phase = unit_complex(selected_reference).reshape(4000, EVENTS, PORTS)
        base_power = np.sum(
            selected_energy * event_mask_np, axis=1, dtype=np.float64
        )
        base_zero = ~event_mask_np.any(axis=1) | (base_power <= TINY)
        self.base_mass = torch.as_tensor(base_mass, device=device)
        self.base_port = torch.as_tensor(base_port, device=device)
        self.base_phase = torch.as_tensor(base_phase, device=device)
        self.base_power = torch.as_tensor(
            base_power, dtype=torch.float64, device=device
        )
        self.base_zero = torch.as_tensor(base_zero, device=device)

    def normalization(self, rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        row = torch.as_tensor(rows, device=self.device)
        values = self.member_feature[:, row].float()
        mask = self.member_mask[:, row]
        selected = values[mask]
        mean = selected.mean(dim=0)
        std = selected.std(dim=0)
        std = torch.where(std < 1e-4, torch.ones_like(std), std)
        return mean, std

    def shuffle_index(self, seed: int) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        site = self.site.cpu().numpy()
        fold = self.fold.cpu().numpy()
        event_count = self.event_mask.sum(dim=1).cpu().numpy()
        bins = np.minimum(event_count // 16, 7)
        output = np.arange(4000, dtype=np.int64)
        for site_value in range(2):
            for fold_value in range(5):
                for bin_value in range(8):
                    ids = np.flatnonzero(
                        (site == site_value) & (fold == fold_value) & (bins == bin_value)
                    )
                    if len(ids) > 1:
                        output[ids] = rng.permutation(ids)
        return torch.as_tensor(output, dtype=torch.int64, device=self.device)


class RelationProbe(nn.Module):
    def __init__(
        self, feature_dim: int, event_dim: int, context_dim: int, config: Config
    ) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden
        self.member_input = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        self.seed_embedding = nn.Embedding(SEEDS, hidden)
        seed_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.heads,
            dim_feedforward=config.feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.seed_encoder = nn.TransformerEncoder(seed_layer, config.seed_layers)
        self.presence_input = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.event_input = nn.Sequential(
            nn.Linear(event_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.context_input = nn.Sequential(
            nn.Linear(context_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        event_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.heads,
            dim_feedforward=config.feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.event_encoder = nn.TransformerEncoder(event_layer, config.event_layers)
        self.output_norm = nn.LayerNorm(hidden)
        self.mass_head = nn.Linear(hidden, 1)
        self.coordinate_head = nn.Linear(hidden, 3)
        self.port_head = nn.Linear(hidden, PORTS)
        self.phase_head = nn.Linear(hidden, PORTS)
        self.point_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2)
        )
        for head in (
            self.mass_head,
            self.coordinate_head,
            self.port_head,
            self.phase_head,
            self.point_head[-1],
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        member: torch.Tensor,
        member_mask: torch.Tensor,
        event_mask: torch.Tensor,
        presence: torch.Tensor,
        event_feature: torch.Tensor,
        context: torch.Tensor,
        base_mass: torch.Tensor,
        base_port: torch.Tensor,
        base_phase: torch.Tensor,
        base_log_power: torch.Tensor,
        base_zero_logit: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, seeds, events, width = member.shape
        hidden = self.member_input(
            member.permute(0, 2, 1, 3).reshape(batch * events, seeds, width)
        )
        seed_ids = torch.arange(seeds, device=member.device)
        hidden = hidden + self.seed_embedding(seed_ids)[None]
        flat_mask = member_mask.permute(0, 2, 1).reshape(batch * events, seeds)
        safe_mask = flat_mask.clone()
        safe_mask[:, 0] |= ~safe_mask.any(dim=1)
        hidden = self.seed_encoder(hidden, src_key_padding_mask=~safe_mask)
        weight = flat_mask.to(hidden.dtype)
        pooled = (hidden * weight[..., None]).sum(dim=1) / weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        pooled = pooled.reshape(batch, events, self.config.hidden)
        context_hidden = self.context_input(context)
        pooled = (
            pooled
            + self.presence_input((presence / SEEDS)[..., None])
            + self.event_input(event_feature)
            + context_hidden[:, None]
        )
        safe_event = event_mask.clone()
        safe_event[:, 0] |= ~safe_event.any(dim=1)
        pooled = self.event_encoder(pooled, src_key_padding_mask=~safe_event)
        pooled = self.output_norm(pooled)
        mass_logits = (
            torch.log(base_mass.clamp_min(1e-12)) + self.mass_head(pooled)[..., 0]
        ).masked_fill(~event_mask, -1e4)
        coordinate = torch.tanh(self.coordinate_head(pooled)) * torch.tensor(
            COORDINATE_LIMITS, dtype=pooled.dtype, device=pooled.device
        )
        port = torch.softmax(
            torch.log(base_port.clamp_min(1e-8)) + self.port_head(pooled), dim=-1
        )
        phase_delta = torch.pi * torch.tanh(self.phase_head(pooled))
        phase = base_phase * torch.exp(1j * phase_delta)
        event_weight = event_mask.to(pooled.dtype)
        point_hidden = (pooled * event_weight[..., None]).sum(dim=1) / event_weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        point = self.point_head(point_hidden + context_hidden)
        mass = torch.softmax(mass_logits, dim=1) * event_mask.to(pooled.dtype)
        mass = mass / mass.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return {
            "mass_logits": mass_logits,
            "mass": mass,
            "coordinate": coordinate,
            "port": port,
            "phase": phase,
            "log_power": base_log_power + point[:, 0],
            "zero_logit": base_zero_logit + point[:, 1],
        }


def batch_from_store(
    store: RelationStore,
    rows: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    mode: str,
    shuffle_index: torch.Tensor,
) -> dict[str, torch.Tensor]:
    source_rows = shuffle_index[rows] if mode == "position_shuffle" else rows
    member = store.member_feature[:, source_rows].permute(1, 0, 2, 3).float()
    member_mask = store.member_mask[:, source_rows].permute(1, 0, 2)
    if mode == "summary":
        representative_seed = store.representative_seed[source_rows]
        gathered = representative_seed[:, None, :, None].expand(
            -1, 1, -1, member.shape[-1]
        )
        member = torch.gather(
            member,
            1,
            gathered,
        ).expand(-1, SEEDS, -1, -1)
        summary_mask = torch.zeros_like(member_mask)
        summary_mask[:, 0] = store.event_mask[source_rows]
        member_mask = summary_mask
    event_mask = store.event_mask[rows]
    member = ((member - mean) / std).clamp(-8.0, 8.0)
    member = member * member_mask[..., None]
    return {
        "member": member,
        "member_mask": member_mask,
        "event_mask": event_mask,
        "presence": store.presence[source_rows],
        "event_feature": store.event_feature[rows],
        "context": store.context[rows],
        "base_coordinate": store.coordinate[rows],
        "base_mass": store.base_mass[source_rows],
        "base_port": store.base_port[source_rows],
        "base_phase": store.base_phase[source_rows],
        "base_log_power": torch.log(
            store.base_power[source_rows].clamp_min(1e-30)
        ).float(),
        "base_zero": store.base_zero[source_rows],
        "base_zero_logit": torch.where(
            store.base_zero[source_rows],
            torch.full_like(store.base_power[source_rows], 6.0, dtype=torch.float64),
            torch.full_like(store.base_power[source_rows], -6.0, dtype=torch.float64),
        ).float(),
        "target_mass": store.target_mass[rows],
        "target_delta": store.target_delta[rows],
        "target_port": store.target_port[rows],
        "target_phase": store.target_phase[rows],
        "target_active": store.target_active[rows],
        "point_power": store.point_power[rows],
        "zero": store.zero[rows],
    }


def forward_probe(
    model: RelationProbe, batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return model(
        batch["member"],
        batch["member_mask"],
        batch["event_mask"],
        batch["presence"],
        batch["event_feature"],
        batch["context"],
        batch["base_mass"],
        batch["base_port"],
        batch["base_phase"],
        batch["base_log_power"],
        batch["base_zero_logit"],
    )


def synthesize_output(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> torch.Tensor:
    coordinate = torch.remainder(
        batch["base_coordinate"] + output["coordinate"],
        torch.tensor(PERIODS, dtype=torch.float32, device=output["mass"].device),
    )
    power = torch.exp(output["log_power"].clamp(-40.0, 5.0))
    zero_probability = torch.sigmoid(output["zero_logit"])
    coefficient_power = (
        power[:, None, None]
        * (1.0 - zero_probability[:, None, None])
        * output["mass"][:, :, None]
        * output["port"]
    )
    amplitude = torch.sqrt(coefficient_power.clamp_min(1e-30))
    amplitude = amplitude * batch["event_mask"][..., None]
    coefficient = torch.complex(amplitude, torch.zeros_like(amplitude)) * output[
        "phase"
    ]
    atom = component_atom_torch(coordinate)
    channel = torch.einsum("bkc,bkhvs->bchvs", coefficient, atom)
    return (
        channel.reshape(-1, 2, 4, 16, 8, 192)
        .permute(0, 1, 3, 4, 2, 5)
        .reshape(-1, 256, 4, 192)
    )


def component_atom_torch(coordinate: torch.Tensor) -> torch.Tensor:
    horizontal = torch.arange(16, dtype=torch.float32, device=coordinate.device)
    vertical = torch.arange(8, dtype=torch.float32, device=coordinate.device)
    subcarrier = torch.arange(192, dtype=torch.float32, device=coordinate.device)
    phase = (
        2.0
        * torch.pi
        * coordinate[..., 0, None, None, None]
        * horizontal[None, None, :, None, None]
        / 16.0
        + 2.0
        * torch.pi
        * coordinate[..., 1, None, None, None]
        * vertical[None, None, None, :, None]
        / 8.0
        - 2.0
        * torch.pi
        * coordinate[..., 2, None, None, None]
        * subcarrier[None, None, None, None, :]
        / 192.0
    )
    return torch.exp(1j * phase) / math.sqrt(16 * 8 * 192)


def state_scores(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    target_mass = batch["target_mass"]
    nonzero = ~batch["zero"]
    weight = target_mass / target_mass.sum(dim=1, keepdim=True).clamp_min(TINY)
    mass = (
        torch.sqrt(output["mass"].clamp_min(1e-12))
        * torch.sqrt(target_mass.clamp_min(0.0))
    ).sum(dim=1)
    coordinate_kernel = torch.exp(
        -0.5
        * (
            output["coordinate"]
            - batch["target_delta"]
        ).square().div(
            torch.tensor((1.5, 1.0, 2.0), device=target_mass.device) ** 2
        ).sum(dim=2)
    )
    coordinate = (weight * coordinate_kernel).sum(dim=1)
    port_affinity = (
        torch.sqrt(output["port"].clamp_min(1e-12))
        * torch.sqrt(batch["target_port"].clamp_min(0.0))
    ).sum(dim=2)
    port = (weight * port_affinity).sum(dim=1)
    phase_cosine = torch.real(output["phase"] * torch.conj(batch["target_phase"]))
    phase_event = (phase_cosine * batch["target_port"]).sum(dim=2)
    phase = (weight * phase_event).sum(dim=1)
    phase_mapped = 0.5 * (phase + 1.0)
    true_log_power = torch.log(batch["point_power"].clamp_min(1e-30)).float()
    power_error = (output["log_power"] - true_log_power).abs()
    power = torch.exp(-power_error / 2.0)
    zero_prediction = torch.sigmoid(output["zero_logit"])
    zero = torch.where(batch["zero"], zero_prediction, 1.0 - zero_prediction)
    composite = (
        0.25 * mass
        + 0.20 * coordinate
        + 0.15 * port
        + 0.25 * phase_mapped
        + 0.10 * power
        + 0.05 * zero
    )
    composite = torch.where(nonzero, composite, zero)
    return {
        "composite": composite,
        "mass": mass,
        "coordinate": coordinate,
        "port": port,
        "phase": phase_mapped,
        "power": power,
        "zero": zero,
        "nonzero": nonzero,
    }


def objective(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scores = state_scores(output, batch)
    nonzero = scores["nonzero"]
    target_mass = batch["target_mass"]
    active = batch["target_active"]
    weight = target_mass[active]
    weight = weight / weight.mean().clamp_min(TINY) if active.any() else weight
    listwise = -(
        target_mass * torch.log_softmax(output["mass_logits"], dim=1)
    ).sum(dim=1)
    listwise_loss = listwise[nonzero].mean() if nonzero.any() else listwise.mean() * 0.0
    existence = F.binary_cross_entropy_with_logits(
        output["mass_logits"][batch["event_mask"]],
        active[batch["event_mask"]].to(output["mass_logits"].dtype),
    )
    if active.any():
        coordinate = F.smooth_l1_loss(
            output["coordinate"][active], batch["target_delta"][active], reduction="none"
        ).mean(dim=1)
        coordinate_loss = (coordinate * weight).mean()
        port_loss = ((1.0 - (
            torch.sqrt(output["port"][active].clamp_min(1e-12))
            * torch.sqrt(batch["target_port"][active].clamp_min(0.0))
        ).sum(dim=1)) * weight).mean()
        phase_score = (
            torch.real(output["phase"][active] * torch.conj(batch["target_phase"][active]))
            * batch["target_port"][active]
        ).sum(dim=1)
        phase_loss = ((1.0 - phase_score) * weight).mean()
    else:
        coordinate_loss = output["mass"].sum() * 0.0
        port_loss = output["mass"].sum() * 0.0
        phase_loss = output["mass"].sum() * 0.0
    power_valid = ~batch["zero"]
    power_loss = F.smooth_l1_loss(
        output["log_power"][power_valid],
        torch.log(batch["point_power"][power_valid]).float(),
    ) if power_valid.any() else output["log_power"].sum() * 0.0
    zero_loss = F.binary_cross_entropy_with_logits(
        output["zero_logit"], batch["zero"].to(output["zero_logit"].dtype)
    )
    loss = (
        0.55 * listwise_loss
        + 0.08 * existence
        + 0.15 * coordinate_loss
        + 0.10 * port_loss
        + 0.25 * phase_loss
        + 0.08 * power_loss
        + 0.05 * zero_loss
    )
    return loss, scores


def official_objective(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    truth_pas: torch.Tensor | None = None,
    truth_pdp: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    shaped_prediction = prediction.reshape(-1, 2, 16, 8, 4, 192)
    prediction_pas = torch.fft.fft2(
        shaped_prediction, dim=(2, 3)
    ).abs().square().reshape_as(prediction)
    if truth_pas is None:
        shaped_truth = truth.reshape_as(shaped_prediction)
        truth_pas = torch.fft.fft2(
            shaped_truth, dim=(2, 3)
        ).abs().square().reshape_as(truth)
    pas_numerator = (prediction_pas * truth_pas).sum(dim=1, dtype=torch.float64)
    pas_denominator = torch.sqrt(
        (
            prediction_pas.square().sum(dim=1, dtype=torch.float64)
            * truth_pas.square().sum(dim=1, dtype=torch.float64)
        ).clamp_min(1e-24)
    )
    pas = (pas_numerator / pas_denominator).mean(dim=(1, 2))

    prediction_pdp = torch.fft.ifft(prediction, dim=-1).abs().square()
    if truth_pdp is None:
        truth_pdp = torch.fft.ifft(truth, dim=-1).abs().square()
    pdp_numerator = (prediction_pdp * truth_pdp).sum(dim=3, dtype=torch.float64)
    pdp_denominator = torch.sqrt(
        (
            prediction_pdp.square().sum(dim=3, dtype=torch.float64)
            * truth_pdp.square().sum(dim=3, dtype=torch.float64)
        ).clamp_min(1e-24)
    )
    pdp = (pdp_numerator / pdp_denominator).mean(dim=(1, 2))

    squared_error = (prediction - truth).abs().square().sum(
        dim=(1, 2, 3), dtype=torch.float64
    )
    truth_energy = truth.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
    nmse = squared_error.sum() / truth_energy.sum().clamp_min(TINY)
    nmse_score = 1.0 / (1.0 + nmse)
    score = 0.4 * pas.mean() + 0.4 * pdp.mean() + 0.2 * nmse_score
    return (1.0 - score).to(torch.float32), {
        "pas": pas.mean().to(torch.float32),
        "pdp": pdp.mean().to(torch.float32),
        "nmse": nmse.to(torch.float32),
        "nmse_score": nmse_score.to(torch.float32),
        "score": score.to(torch.float32),
    }


@torch.inference_mode()
def evaluate(
    model: RelationProbe,
    store: RelationStore,
    rows: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    mode: str,
    shuffle_index: torch.Tensor,
    batch_size: int,
) -> dict[str, object]:
    model.eval()
    values = {name: [] for name in ("composite", "mass", "coordinate", "port", "phase", "power", "zero")}
    for start in range(0, len(rows), batch_size):
        row = torch.as_tensor(rows[start : start + batch_size], device=store.device)
        batch = batch_from_store(store, row, mean, std, mode, shuffle_index)
        output = forward_probe(model, batch)
        scores = state_scores(output, batch)
        for name in values:
            values[name].append(scores[name].cpu().numpy())
    arrays = {name: np.concatenate(parts) for name, parts in values.items()}
    zero = store.zero[torch.as_tensor(rows, device=store.device)].cpu().numpy()
    nonzero = ~zero
    return {
        "count": int(len(rows)),
        **{
            name: float(np.mean(value if name in ("composite", "zero") else value[nonzero]))
            for name, value in arrays.items()
        },
        "per_point": arrays,
    }


def capability_rows(store: RelationStore, config: Config) -> np.ndarray:
    fold = store.fold.cpu().numpy()
    zero = store.zero.cpu().numpy()
    eligible = (fold != 0) & (fold != 1)
    nonzero_ids = np.flatnonzero(eligible & ~zero)
    zero_ids = np.flatnonzero(eligible & zero)
    if len(nonzero_ids) < config.capability_nonzero_points:
        raise RuntimeError("not enough nonzero rows for E192 capability gate")
    zero_count = config.capability_points - config.capability_nonzero_points
    if len(zero_ids) < zero_count:
        raise RuntimeError("not enough zero rows for E192 capability gate")
    rng = np.random.default_rng(config.random_state)
    rows = np.concatenate(
        (
            rng.choice(
                nonzero_ids, size=config.capability_nonzero_points, replace=False
            ),
            rng.choice(zero_ids, size=zero_count, replace=False),
        )
    )
    return rng.permutation(rows).astype(np.int64)


@torch.inference_mode()
def loss_on_rows(
    model: RelationProbe,
    store: RelationStore,
    rows: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    shuffle_index: torch.Tensor,
) -> float:
    model.eval()
    row = torch.as_tensor(rows, device=store.device)
    batch = batch_from_store(
        store, row, mean, std, "history", shuffle_index
    )
    output = forward_probe(model, batch)
    loss, _ = objective(output, batch)
    return float(loss.item())


def run_capability_gate(
    store: RelationStore, config: Config, shuffle_index: torch.Tensor
) -> dict[str, object]:
    started = time.time()
    rows = capability_rows(store, config)
    mean, std = store.normalization(rows)
    torch.manual_seed(config.random_state)
    torch.cuda.manual_seed_all(config.random_state)
    model = RelationProbe(
        store.member_feature.shape[-1],
        store.event_feature.shape[-1],
        store.context.shape[-1],
        config,
    ).to(store.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=0.0
    )
    initial = evaluate(
        model,
        store,
        rows,
        mean,
        std,
        "history",
        shuffle_index,
        config.capability_points,
    )
    initial_loss = loss_on_rows(model, store, rows, mean, std, shuffle_index)
    row = torch.as_tensor(rows, device=store.device)
    trace: list[dict[str, float | int]] = []
    for step in range(1, config.capability_steps + 1):
        model.train()
        batch = batch_from_store(
            store, row, mean, std, "history", shuffle_index
        )
        output = forward_probe(model, batch)
        loss, scores = objective(output, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if step == 1 or step % 20 == 0 or step == config.capability_steps:
            trace.append(
                {
                    "step": step,
                    "loss": float(loss.detach().item()),
                    "composite": float(scores["composite"].mean().detach().item()),
                }
            )
            print(
                f"E192 capability step={step}/{config.capability_steps} "
                f"loss={trace[-1]['loss']:.6f} "
                f"composite={trace[-1]['composite']:.6f}",
                flush=True,
            )
    final = evaluate(
        model,
        store,
        rows,
        mean,
        std,
        "history",
        shuffle_index,
        config.capability_points,
    )
    final_loss = loss_on_rows(model, store, rows, mean, std, shuffle_index)
    improvement = float(final["composite"] - initial["composite"])
    loss_reduction = float(
        (initial_loss - final_loss) / max(abs(initial_loss), 1e-12)
    )
    passed = bool(
        final["composite"] >= config.capability_minimum_composite
        and improvement >= config.capability_minimum_improvement
        and loss_reduction >= config.capability_minimum_loss_reduction
    )
    return {
        "experiment": VERSION,
        "stage": "capability",
        "status": "capability_passed" if passed else "capability_failed",
        "configuration": asdict(config),
        "rows": rows.tolist(),
        "zero_count": int(store.zero[row].sum().item()),
        "initial": {
            key: value for key, value in initial.items() if key != "per_point"
        },
        "final": {key: value for key, value in final.items() if key != "per_point"},
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "composite_improvement": improvement,
        "loss_reduction": loss_reduction,
        "gate": {
            "minimum_composite": config.capability_minimum_composite,
            "minimum_improvement": config.capability_minimum_improvement,
            "minimum_loss_reduction": config.capability_minimum_loss_reduction,
            "passed": passed,
        },
        "trace": trace,
        "performance": {
            "device": torch.cuda.get_device_name(store.device),
            "peak_torch_memory_gib": float(
                torch.cuda.max_memory_allocated() / 2**30
            ),
            "peak_rss_gib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            ),
            "seconds": float(time.time() - started),
        },
    }


def train_one(
    store: RelationStore,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    outer_rows: np.ndarray,
    mode: str,
    config: Config,
    shuffle_index: torch.Tensor,
    seed: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    mean, std = store.normalization(fit_rows)
    model = RelationProbe(
        store.member_feature.shape[-1],
        store.event_feature.shape[-1],
        store.context.shape[-1],
        config,
    ).to(store.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.08
    )
    rng = np.random.default_rng(seed)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_score = -math.inf
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(fit_rows)
        losses = []
        composites = []
        for start in range(0, len(order), config.batch_size):
            row = torch.as_tensor(order[start : start + config.batch_size], device=store.device)
            batch = batch_from_store(store, row, mean, std, mode, shuffle_index)
            output = forward_probe(model, batch)
            loss, scores = objective(output, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
            composites.append(float(scores["composite"].mean().detach().item()))
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_composite": float(np.mean(composites)),
        }
        if epoch >= config.minimum_epochs and epoch % config.validation_interval == 0:
            validation = evaluate(
                model, store, validation_rows, mean, std, mode, shuffle_index, config.batch_size
            )
            record["validation_composite"] = float(validation["composite"])
            if record["validation_composite"] > best_score:
                best_score = record["validation_composite"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
        history.append(record)
        if epoch % config.validation_interval == 0:
            print(
                f"E192 mode={mode} epoch={epoch}/{config.epochs} "
                f"train={record['train_composite']:.6f} "
                f"validation={record.get('validation_composite', float('nan')):.6f}",
                flush=True,
            )
    model.load_state_dict(best_state)
    outer = evaluate(
        model, store, outer_rows, mean, std, mode, shuffle_index, config.batch_size
    )
    predictions: dict[str, list[np.ndarray]] = {
        name: [] for name in ("mass", "coordinate", "port", "phase", "log_power", "zero_logit")
    }
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(outer_rows), config.batch_size):
            row = torch.as_tensor(outer_rows[start : start + config.batch_size], device=store.device)
            batch = batch_from_store(store, row, mean, std, mode, shuffle_index)
            output = forward_probe(model, batch)
            for name in predictions:
                value = output[name]
                predictions[name].append(value.cpu().numpy())
    prediction = {name: np.concatenate(parts) for name, parts in predictions.items()}
    return {
        "best_epoch": best_epoch,
        "validation_best": best_score,
        "outer": {key: value for key, value in outer.items() if key != "per_point"},
        "history": history,
    }, prediction


@torch.inference_mode()
def predict_channel(
    model: RelationProbe,
    store: RelationStore,
    rows: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    mode: str,
    shuffle_index: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output = np.empty((len(rows), 256, 4, 192), dtype=np.complex64)
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        row = torch.as_tensor(rows[start:end], device=store.device)
        batch = batch_from_store(store, row, mean, std, mode, shuffle_index)
        state = forward_probe(model, batch)
        output[start:end] = synthesize_output(state, batch).cpu().numpy()
    return output


def evaluate_channel(
    prediction: np.ndarray,
    rows: np.ndarray,
    truth_all: np.ndarray | torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float | int], PointMetricStatistics]:
    parts: list[PointMetricStatistics] = []
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        pred = torch.as_tensor(prediction[start:end], device=device)
        if isinstance(truth_all, torch.Tensor):
            row = torch.as_tensor(rows[start:end], device=device)
            truth = truth_all[row]
        else:
            truth = torch.as_tensor(
                np.asarray(truth_all[rows[start:end]], dtype=np.complex64),
                device=device,
            )
        parts.append(point_metric_statistics_torch(pred, truth))
    values = PointMetricStatistics(
        pas=np.concatenate([part.pas for part in parts]),
        pdp=np.concatenate([part.pdp for part in parts]),
        squared_error=np.concatenate([part.squared_error for part in parts]),
        truth_energy=np.concatenate([part.truth_energy for part in parts]),
    )
    return aggregate_statistics(values), values


def pretrain_state_model(
    store: RelationStore,
    fit_rows: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    config: Config,
    shuffle_index: torch.Tensor,
    seed: int,
    checkpoint_path: Path,
) -> tuple[RelationProbe, list[dict[str, float | int]]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = RelationProbe(
        store.member_feature.shape[-1],
        store.event_feature.shape[-1],
        store.context.shape[-1],
        config,
    ).to(store.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.08
    )
    rng = np.random.default_rng(seed)
    start_epoch = 1
    history: list[dict[str, float | int]] = []
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=store.device)
        if checkpoint.get("configuration") != asdict(config):
            raise RuntimeError("E192 direct checkpoint configuration mismatch")
        stage = checkpoint.get("stage")
        if stage == "state":
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            rng.bit_generator.state = checkpoint["rng_state"]
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            torch.cuda.set_rng_state(
                checkpoint["cuda_rng_state"].cpu(), device=store.device
            )
            history = checkpoint.get("state_history", [])
            start_epoch = int(checkpoint["epoch"]) + 1
            print(
                f"E192 direct resume state epoch={start_epoch - 1}/{config.epochs}",
                flush=True,
            )
        elif stage == "official":
            model.load_state_dict(checkpoint["state_pretrain_model"])
            history = checkpoint.get("state_history", [])
            return model, history
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        order = rng.permutation(fit_rows)
        losses: list[float] = []
        for start in range(0, len(order), config.batch_size):
            row = torch.as_tensor(
                order[start : start + config.batch_size], device=store.device
            )
            batch = batch_from_store(
                store, row, mean, std, "history", shuffle_index
            )
            output = forward_probe(model, batch)
            loss, _ = objective(output, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        scheduler.step()
        record = {"epoch": epoch, "loss": float(np.mean(losses))}
        history.append(record)
        atomic_torch_save(
            checkpoint_path,
            {
                "stage": "state",
                "epoch": epoch,
                "configuration": asdict(config),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng_state": rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(store.device),
                "state_history": history,
            },
        )
        if epoch % config.validation_interval == 0 or epoch == config.epochs:
            print(
                f"E192 direct state epoch={epoch}/{config.epochs} "
                f"loss={record['loss']:.6f}",
                flush=True,
            )
    return model, history


def train_direct_fold(
    store: RelationStore,
    truth_all: torch.Tensor,
    truth_pas_all: torch.Tensor,
    truth_pdp_all: torch.Tensor,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    outer_rows: np.ndarray,
    config: Config,
    shuffle_index: torch.Tensor,
    seed: int,
    checkpoint_path: Path,
) -> tuple[dict[str, object], np.ndarray]:
    fold_started = time.time()
    mean, std = store.normalization(fit_rows)
    model, state_history = pretrain_state_model(
        store,
        fit_rows,
        mean,
        std,
        config,
        shuffle_index,
        seed,
        checkpoint_path,
    )
    state_pretrain_state = copy.deepcopy(model.state_dict())
    baseline_prediction = predict_channel(
        model,
        store,
        validation_rows,
        mean,
        std,
        "history",
        shuffle_index,
        config.official_batch_size,
    )
    baseline, _ = evaluate_channel(
        baseline_prediction,
        validation_rows,
        truth_all,
        store.device,
        config.official_batch_size,
    )
    best_score = float(baseline["score"])
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.official_learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.official_epochs, 1),
        eta_min=config.official_learning_rate * 0.1,
    )
    rng = np.random.default_rng(seed + 17)
    history: list[dict[str, object]] = [
        {"epoch": 0, "validation": baseline, "fallback": "state_pretrain"}
    ]
    start_epoch = 1
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=store.device)
        if checkpoint.get("stage") == "official":
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            rng.bit_generator.state = checkpoint["rng_state"]
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            torch.cuda.set_rng_state(
                checkpoint["cuda_rng_state"].cpu(), device=store.device
            )
            state_pretrain_state = checkpoint["state_pretrain_model"]
            best_state = checkpoint["best_state"]
            best_score = float(checkpoint["best_score"])
            best_epoch = int(checkpoint["best_epoch"])
            history = checkpoint["official_history"]
            start_epoch = int(checkpoint["epoch"]) + 1
            print(
                f"E192 direct resume official epoch={start_epoch - 1}/"
                f"{config.official_epochs}",
                flush=True,
            )
    for epoch in range(start_epoch, config.official_epochs + 1):
        model.train()
        order = rng.permutation(fit_rows)
        train_scores: list[float] = []
        for start in range(0, len(order), config.official_batch_size):
            local = order[start : start + config.official_batch_size]
            row = torch.as_tensor(local, device=store.device)
            batch = batch_from_store(
                store, row, mean, std, "history", shuffle_index
            )
            state = forward_probe(model, batch)
            prediction = synthesize_output(state, batch)
            truth = truth_all[row]
            loss, values = official_objective(
                prediction, truth, truth_pas_all[row], truth_pdp_all[row]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_scores.append(float(values["score"].detach().item()))
        scheduler.step()
        record: dict[str, object] = {
            "epoch": epoch,
            "train_score": float(np.mean(train_scores)),
        }
        if epoch % config.official_validation_interval == 0:
            validation_prediction = predict_channel(
                model,
                store,
                validation_rows,
                mean,
                std,
                "history",
                shuffle_index,
                config.official_batch_size,
            )
            validation, _ = evaluate_channel(
                validation_prediction,
                validation_rows,
                truth_all,
                store.device,
                config.official_batch_size,
            )
            record["validation"] = validation
            if float(validation["score"]) > best_score:
                best_score = float(validation["score"])
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
        history.append(record)
        atomic_torch_save(
            checkpoint_path,
            {
                "stage": "official",
                "epoch": epoch,
                "configuration": asdict(config),
                "state_pretrain_model": state_pretrain_state,
                "state_history": state_history,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng_state": rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(store.device),
                "best_state": best_state,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "official_history": history,
            },
        )
        print(
            f"E192 direct official epoch={epoch}/{config.official_epochs} "
            f"train={record['train_score']:.6f} "
            f"validation={record.get('validation', {}).get('score', float('nan')):.6f}",
            flush=True,
        )
    model.load_state_dict(best_state)
    outer_prediction = predict_channel(
        model,
        store,
        outer_rows,
        mean,
        std,
        "history",
        shuffle_index,
        config.official_batch_size,
    )
    outer, _ = evaluate_channel(
        outer_prediction,
        outer_rows,
        truth_all,
        store.device,
        config.official_batch_size,
    )
    return {
        "best_epoch": best_epoch,
        "validation_best": best_score,
        "outer": outer,
        "state_history": state_history,
        "history": history,
        "performance": {
            "device": torch.cuda.get_device_name(store.device),
            "peak_torch_memory_gib": float(
                torch.cuda.max_memory_allocated() / 2**30
            ),
            "peak_rss_gib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            ),
            "seconds": float(time.time() - fold_started),
        },
    }, outer_prediction


def load_direct_truth(
    device: torch.device, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    started = time.time()
    source = np.load(TRAIN_CHANNEL, mmap_mode="r")
    truth = torch.empty(source.shape, dtype=torch.complex64, device=device)
    copy_seconds = 0.0
    for start in range(0, len(source), batch_size):
        end = min(start + batch_size, len(source))
        copy_started = time.time()
        truth[start:end].copy_(
            torch.from_numpy(np.array(source[start:end], dtype=np.complex64, copy=True))
            .to(device)
        )
        copy_seconds += time.time() - copy_started
    truth_pas = torch.empty(source.shape, dtype=torch.float32, device=device)
    truth_pdp = torch.empty(source.shape, dtype=torch.float32, device=device)
    transform_started = time.time()
    with torch.inference_mode():
        for start in range(0, len(source), batch_size):
            end = min(start + batch_size, len(source))
            shaped = truth[start:end].reshape(-1, 2, 16, 8, 4, 192)
            truth_pas[start:end] = (
                torch.fft.fft2(shaped, dim=(2, 3))
                .abs()
                .square()
                .reshape_as(truth[start:end])
            )
            truth_pdp[start:end] = torch.fft.ifft(
                truth[start:end], dim=-1
            ).abs().square()
    torch.cuda.synchronize(device)
    return truth, truth_pas, truth_pdp, {
        "copy_seconds": float(copy_seconds),
        "transform_seconds": float(time.time() - transform_started),
        "total_seconds": float(time.time() - started),
    }


def run_direct_history(
    store: RelationStore,
    config: Config,
    shuffle_index: torch.Tensor,
    outer_values: tuple[int, ...],
) -> dict[str, object]:
    started = time.time()
    DIRECT_FOLD_DIR.mkdir(parents=True, exist_ok=True)
    truth_all, truth_pas_all, truth_pdp_all, truth_loading = load_direct_truth(
        store.device, max(config.batch_size, config.official_batch_size, 64)
    )
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    all_rows = np.arange(4000, dtype=np.int64)
    fold = store.fold.cpu().numpy()
    reports: list[dict[str, object]] = []
    for outer in outer_values:
        paths = direct_fold_paths(outer)
        if all(paths[name].is_file() for name in ("prediction", "rows", "report")):
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            if report.get("configuration") != asdict(config):
                raise RuntimeError(f"E192 completed outer{outer} config mismatch")
            reports.append(report)
            print(
                f"E192 direct reused outer={outer} "
                f"score={report['outer_metrics']['score']:.6f}",
                flush=True,
            )
            continue
        validation_fold = (outer + 1) % 5
        fit_rows = all_rows[(fold != outer) & (fold != validation_fold)]
        validation_rows = query_ids[fold_id == validation_fold]
        outer_rows = query_ids[fold_id == outer]
        fold_started = time.time()
        try:
            report, prediction = train_direct_fold(
                store,
                truth_all,
                truth_pas_all,
                truth_pdp_all,
                fit_rows,
                validation_rows,
                outer_rows,
                config,
                shuffle_index,
                config.random_state + outer * 1000,
                paths["checkpoint"],
            )
            outer_metrics = report.pop("outer")
            report.update(
                {
                    "experiment": VERSION,
                    "stage": "direct_outer",
                    "status": "complete",
                    "configuration": asdict(config),
                    "outer_fold": outer,
                    "outer_metrics": outer_metrics,
                    "validation_fold": validation_fold,
                    "fit_count": int(len(fit_rows)),
                    "validation_count": int(len(validation_rows)),
                    "outer_count": int(len(outer_rows)),
                    "truth_loading": truth_loading,
                    "started_unix": float(fold_started),
                    "finished_unix": float(time.time()),
                }
            )
            if prediction.shape != (len(outer_rows), 256, 4, 192):
                raise RuntimeError(f"invalid E192 outer{outer} prediction shape")
            if prediction.dtype != np.complex64 or not np.isfinite(prediction).all():
                raise RuntimeError(f"invalid E192 outer{outer} prediction values")
            atomic_save_npy(paths["prediction"], prediction)
            atomic_save_npy(paths["rows"], outer_rows.astype(np.int32))
            atomic_write_json(paths["report"], report)
            paths["error"].unlink(missing_ok=True)
            reports.append(report)
            print(
                f"E192 direct completed outer={outer} "
                f"score={outer_metrics['score']:.6f}",
                flush=True,
            )
        except BaseException as error:
            failure = {
                "experiment": VERSION,
                "stage": "direct_outer",
                "status": "failed",
                "outer_fold": outer,
                "configuration": asdict(config),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "argv": sys.argv,
                "pid": os.getpid(),
                "device": torch.cuda.get_device_name(store.device),
                "seconds": float(time.time() - fold_started),
            }
            atomic_write_json(paths["error"], failure)
            raise
    return {
        "experiment": VERSION,
        "stage": "direct_outer",
        "status": "complete",
        "outer_folds": list(outer_values),
        "reports": reports,
        "performance": {
            "device": torch.cuda.get_device_name(store.device),
            "peak_torch_memory_gib": float(
                torch.cuda.max_memory_allocated() / 2**30
            ),
            "peak_rss_gib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            ),
            "seconds": float(time.time() - started),
        },
    }


def finalize_direct_history(
    config: Config, device: torch.device
) -> dict[str, object]:
    started = time.time()
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    query_lookup = np.full(4000, -1, dtype=np.int64)
    query_lookup[query_ids] = np.arange(len(query_ids), dtype=np.int64)
    channel = np.zeros((len(query_ids), 256, 4, 192), dtype=np.complex64)
    reports: list[dict[str, object]] = []
    for outer in range(5):
        paths = direct_fold_paths(outer)
        missing = [name for name in ("prediction", "rows", "report") if not paths[name].is_file()]
        if missing:
            raise RuntimeError(f"E192 outer{outer} incomplete: {missing}")
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        if report.get("configuration") != asdict(config):
            raise RuntimeError(f"E192 outer{outer} report config mismatch")
        rows = np.load(paths["rows"]).astype(np.int64)
        expected = query_ids[fold_id == outer]
        if not np.array_equal(rows, expected):
            raise RuntimeError(f"E192 outer{outer} row identity mismatch")
        prediction = np.load(paths["prediction"], mmap_mode="r")
        if prediction.shape != (len(rows), 256, 4, 192):
            raise RuntimeError(f"E192 outer{outer} prediction shape mismatch")
        channel[query_lookup[rows]] = prediction
        reports.append(report)
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    overall, values = evaluate_channel(
        channel, query_ids, truth_all, device, config.official_batch_size
    )
    fold_metrics = [
        {"fold": outer, **aggregate_statistics(values, fold_id == outer)}
        for outer in range(5)
    ]
    atomic_save_npy(DIRECT_CHANNEL, channel)
    starts = [float(report["started_unix"]) for report in reports]
    finishes = [float(report["finished_unix"]) for report in reports]
    result = {
        "experiment": VERSION,
        "stage": "direct",
        "status": "direct_passed"
        if float(overall["score"]) > 0.686375
        else "direct_failed",
        "configuration": asdict(config),
        "metrics": {"overall": overall, "folds": fold_metrics},
        "training": reports,
        "gate": {
            "v15_score": 0.686375,
            "target_score": 0.700000,
            "above_v15": bool(float(overall["score"]) > 0.686375),
            "target_reached": bool(float(overall["score"]) >= 0.700000),
        },
        "output": {
            "channel": PATHS.relative(DIRECT_CHANNEL),
            "shape": list(channel.shape),
            "dtype": str(channel.dtype),
            "finite": bool(np.isfinite(channel).all()),
        },
        "performance": {
            "finalize_device": torch.cuda.get_device_name(device),
            "parallel_wall_seconds": float(max(finishes) - min(starts)),
            "sum_fold_seconds": float(
                sum(float(report["performance"]["seconds"]) for report in reports)
            ),
            "finalize_seconds": float(time.time() - started),
            "fold_devices": [
                report["performance"]["device"] for report in reports
            ],
            "fold_peak_torch_memory_gib": [
                report["performance"]["peak_torch_memory_gib"] for report in reports
            ],
        },
    }
    atomic_write_json(DIRECT_REPORT, result)
    return result


def representative_state(
    rows: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    mask = np.load(HISTORY_DIR / "mask.npy", mmap_mode="r")
    continuous = np.load(HISTORY_DIR / "continuous.npy", mmap_mode="r")
    reference = np.load(
        HISTORY_DIR / "reference_coefficient.npy", mmap_mode="r"
    )
    anchor_mask = np.load(ANCHOR_DIR / "mask.npy", mmap_mode="r")
    representative = np.load(
        ANCHOR_DIR / "representative_slot.npy", mmap_mode="r"
    )
    raw_coordinate = np.load(BASE_COORDINATE, mmap_mode="r")
    local_mask = np.any(np.asarray(mask[:, rows], dtype=bool), axis=0) & np.asarray(
        anchor_mask[rows], dtype=bool
    )
    fraction = np.asarray(continuous[..., 6], dtype=np.float32)[:, rows]
    representative_seed = np.argmax(fraction, axis=0).astype(np.int64)
    point = np.arange(len(rows), dtype=np.int64)[:, None]
    event = np.arange(EVENTS, dtype=np.int64)[None, :]
    selected_fraction = fraction[representative_seed, point, event]
    selected_energy = np.asarray(continuous[..., 5], dtype=np.float32)[:, rows][
        representative_seed, point, event
    ]
    selected_coefficient = np.asarray(reference, dtype=np.complex64)[:, rows][
        representative_seed, point, event
    ]
    presence = np.asarray(mask, dtype=np.float32)[:, rows].sum(axis=0)
    anchored = np.maximum(selected_fraction, 1e-12) * np.sqrt(
        np.maximum(presence, 1.0) / SEEDS
    )
    mass = normalized_rows(anchored, local_mask)
    coefficient_energy = np.abs(selected_coefficient) ** 2
    port = np.divide(
        coefficient_energy,
        np.maximum(coefficient_energy.sum(axis=(-2, -1), keepdims=True), TINY),
        out=np.full_like(coefficient_energy, 1.0 / PORTS, dtype=np.float32),
    ).reshape(len(rows), EVENTS, PORTS).astype(np.float32)
    phase = unit_complex(selected_coefficient).reshape(
        len(rows), EVENTS, PORTS
    )
    coordinate = np.zeros((len(rows), EVENTS, 3), dtype=np.float32)
    active_rows, active_events = np.nonzero(local_mask)
    slots = np.asarray(
        representative[rows[active_rows], active_events], dtype=np.int64
    )
    coordinate[active_rows, active_events] = raw_coordinate[
        rows[active_rows], slots
    ]
    power = np.sum(selected_energy * local_mask, axis=1, dtype=np.float64)
    zero = ~local_mask.any(axis=1) | (power <= TINY)
    return mass, coordinate, port, phase, power, zero


@torch.inference_mode()
def evaluate_subspace(
    labels: Labels,
    predictable: dict[str, bool],
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    mass, coordinate, port, phase, power, zero = representative_state(query_ids)
    if predictable["mass"]:
        mass = np.array(labels.mass_normalized[query_ids], copy=True)
    if predictable["coordinate"]:
        coordinate = np.mod(
            coordinate + labels.delta[query_ids], PERIODS
        ).astype(np.float32)
    if predictable["port"]:
        port = np.array(labels.port[query_ids], copy=True)
    if predictable["phase"]:
        phase = np.array(labels.phase[query_ids], copy=True)
    if predictable["power"]:
        power = np.array(labels.point_power[query_ids], copy=True)
    if predictable["zero"]:
        zero = np.array(labels.zero[query_ids], copy=True)
    coefficient = (
        np.sqrt(
            np.maximum(
                power[:, None, None] * mass[:, :, None] * port,
                0.0,
            )
        )
        * phase
    ).astype(np.complex64)
    coefficient[zero] = 0.0
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    parts: list[PointMetricStatistics] = []
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        prediction = synthesize_batch(
            coordinate[start:end], coefficient[start:end], device
        )
        truth = torch.as_tensor(
            np.asarray(truth_all[query_ids[start:end]], dtype=np.complex64),
            device=device,
        )
        parts.append(point_metric_statistics_torch(prediction, truth))
    values = PointMetricStatistics(
        pas=np.concatenate([part.pas for part in parts]),
        pdp=np.concatenate([part.pdp for part in parts]),
        squared_error=np.concatenate([part.squared_error for part in parts]),
        truth_energy=np.concatenate([part.truth_energy for part in parts]),
    )
    return {
        "overall": aggregate_statistics(values),
        "folds": [
            {"fold": fold, **aggregate_statistics(values, fold_id == fold)}
            for fold in range(5)
        ],
        "predictable_states": predictable,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    config = Config(
        batch_size=args.batch_size,
        epochs=args.epochs,
        official_epochs=args.official_epochs,
        official_batch_size=args.official_batch_size,
    )
    if args.stage == "direct-finalize":
        device = torch.device(args.device)
        if device.type != "cuda":
            raise RuntimeError("E192 direct finalization requires CUDA")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
        direct = finalize_direct_history(config, device)
        print(json.dumps(direct, ensure_ascii=False, indent=2), flush=True)
        return
    prepare_report = prepare(args.force_prepare, config)
    if args.stage == "prepare":
        print(json.dumps(prepare_report, ensure_ascii=False, indent=2), flush=True)
        return

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("E192 relation probe requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    torch.set_float32_matmul_precision("high")
    labels, label_report = build_labels(False, config)
    store = RelationStore(device, labels)
    shuffle_index = store.shuffle_index(config.random_state)
    if args.stage == "capability":
        capability = run_capability_gate(store, config, shuffle_index)
        CAPABILITY_REPORT.write_text(
            json.dumps(capability, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(capability, ensure_ascii=False, indent=2), flush=True)
        return
    if not CAPABILITY_REPORT.is_file():
        raise RuntimeError("run E192 --stage capability before the formal relation probe")
    capability = json.loads(CAPABILITY_REPORT.read_text(encoding="utf-8"))
    if capability.get("status") != "capability_passed":
        raise RuntimeError("E192 capability gate did not pass")
    if args.stage == "direct":
        outer_values = tuple(
            sorted(set(args.direct_outer if args.direct_outer is not None else range(5)))
        )
        direct = run_direct_history(store, config, shuffle_index, outer_values)
        print(json.dumps(direct, ensure_ascii=False, indent=2), flush=True)
        return
    modes = ("summary", "history", "position_shuffle")
    outer_values = range(1) if args.stage == "screen" else range(5)
    reports: dict[str, list[dict[str, object]]] = {mode: [] for mode in modes}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    all_rows = np.arange(4000, dtype=np.int64)

    for outer in outer_values:
        validation_fold = (outer + 1) % 5
        fold = np.asarray(store.fold.cpu().numpy(), dtype=np.int64)
        fit_rows = all_rows[
            (fold != outer) & (fold != validation_fold)
        ]
        validation_rows = all_rows[fold == validation_fold]
        outer_rows = all_rows[fold == outer]
        for mode_index, mode in enumerate(modes):
            report, prediction = train_one(
                store,
                fit_rows,
                validation_rows,
                outer_rows,
                mode,
                config,
                shuffle_index,
                config.random_state + outer * 100,
            )
            report.update(
                {
                    "outer": outer,
                    "validation_fold": validation_fold,
                    "fit_count": int(len(fit_rows)),
                    "validation_count": int(len(validation_rows)),
                    "outer_count": int(len(outer_rows)),
                }
            )
            reports[mode].append(report)
            predictions[f"{mode}_outer{outer}"] = prediction
            predictions[f"{mode}_outer{outer}"]["rows"] = outer_rows.astype(
                np.int32
            )
            print(
                f"E192 completed mode={mode} outer={outer} "
                f"composite={report['outer']['composite']:.6f}",
                flush=True,
            )

    if args.stage == "screen":
        passed = bool(
            reports["history"][0]["outer"]["composite"]
            > reports["summary"][0]["outer"]["composite"]
            and (
                reports["history"][0]["outer"]["composite"]
                - reports["position_shuffle"][0]["outer"]["composite"]
            )
            / max(
                1.0 - reports["position_shuffle"][0]["outer"]["composite"], 1e-12
            )
            >= config.minimum_shuffle_error_reduction
        )
        status = "screen_passed" if passed else "screen_failed"
        gate: dict[str, object] = {"passed": passed}
        subspace = None
    else:
        history = np.asarray([row["outer"]["composite"] for row in reports["history"]])
        summary = np.asarray([row["outer"]["composite"] for row in reports["summary"]])
        shuffled = np.asarray(
            [row["outer"]["composite"] for row in reports["position_shuffle"]]
        )
        positive_folds = int(np.sum(history > summary))
        shuffle_reduction = float(
            (history.mean() - shuffled.mean()) / max(1.0 - shuffled.mean(), 1e-12)
        )
        relation_passed = bool(
            positive_folds >= config.minimum_positive_folds
            and shuffle_reduction >= config.minimum_shuffle_error_reduction
        )
        predictable: dict[str, bool] = {}
        state_relation: dict[str, dict[str, float | bool]] = {}
        for name in ("mass", "coordinate", "port", "phase", "power", "zero"):
            history_state = float(
                np.mean([row["outer"][name] for row in reports["history"]])
            )
            summary_state = float(
                np.mean([row["outer"][name] for row in reports["summary"]])
            )
            shuffle_state = float(
                np.mean(
                    [row["outer"][name] for row in reports["position_shuffle"]]
                )
            )
            reduction = float(
                (history_state - shuffle_state) / max(1.0 - shuffle_state, 1e-12)
            )
            predictable[name] = bool(
                history_state > summary_state
                and history_state > shuffle_state
                and reduction >= config.minimum_shuffle_error_reduction
            )
            state_relation[name] = {
                "history": history_state,
                "summary": summary_state,
                "position_shuffle": shuffle_state,
                "shuffle_error_reduction": reduction,
                "predictable": predictable[name],
            }
        subspace = evaluate_subspace(
            labels, predictable, device, min(config.batch_size, 16)
        )
        subspace_passed = bool(
            float(subspace["overall"]["score"]) >= config.minimum_subspace_score
        )
        passed = bool(relation_passed and subspace_passed)
        status = "relation_passed" if passed else "relation_failed"
        gate = {
            "positive_folds_vs_summary": positive_folds,
            "minimum_positive_folds": config.minimum_positive_folds,
            "shuffle_error_reduction": shuffle_reduction,
            "minimum_shuffle_error_reduction": config.minimum_shuffle_error_reduction,
            "state_relation": state_relation,
            "minimum_subspace_score": config.minimum_subspace_score,
            "subspace_passed": subspace_passed,
            "relation_passed_before_subspace": relation_passed,
            "passed": passed,
        }
    np.savez_compressed(PREDICTIONS, **{
        f"{group}_{name}": value
        for group, prediction in predictions.items()
        for name, value in prediction.items()
    })
    report = {
        "experiment": VERSION,
        "stage": args.stage,
        "status": status,
        "configuration": asdict(config),
        "prepare": prepare_report,
        "capability": {
            "status": capability["status"],
            "final_composite": capability["final"]["composite"],
            "composite_improvement": capability["composite_improvement"],
            "loss_reduction": capability["loss_reduction"],
        },
        "labels": label_report,
        "folds": reports,
        "gate": gate,
        "subspace": subspace,
        "performance": {
            "device": torch.cuda.get_device_name(device),
            "peak_torch_memory_gib": float(torch.cuda.max_memory_allocated() / 2**30),
            "peak_rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20),
            "seconds": float(time.time() - started),
        },
        "outputs": {"predictions": PATHS.relative(PREDICTIONS)},
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        failure_path = ANALYSIS_DIR / f"direct_process_error_{os.getpid()}.json"
        failure = {
            "experiment": VERSION,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "argv": sys.argv,
            "pid": os.getpid(),
            "unix_time": time.time(),
        }
        try:
            atomic_write_json(failure_path, failure)
        finally:
            traceback.print_exc()
        raise
