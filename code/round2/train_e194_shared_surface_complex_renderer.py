#!/usr/bin/env python3
"""Train a shared-surface complex path renderer on strict Round 2 folds."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import analyze_e129_sionna_path_state_variables as e129
from common.official_metrics import (
    PointMetricStatistics,
    aggregate_statistics,
    point_metric_statistics_torch,
)
from data import load_positions
from project_paths import PATHS
from run_path_component_transport import _component_atom
import train_e150_v7_pathwise_carrier_operator as e150
import train_e192_multiseed_relation_probe as e192


VERSION = "Round2-E194-shared-surface-complex-renderer"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e194_shared_surface_complex_renderer"
ANALYSIS_DIR = (
    PATHS.analysis / "results" / "round2_e194_shared_surface_complex_renderer"
)
STATIC_DIR = ARTIFACT_DIR / "static"
FOLD_DIR = ARTIFACT_DIR / "folds"
LOG_DIR = ANALYSIS_DIR / "logs"
REPORT = ANALYSIS_DIR / "fixed500_report.json"
OUTPUT_CHANNEL = ARTIFACT_DIR / "fixed500_channel.npy"
PREPARE_REPORT = ANALYSIS_DIR / "prepare.json"
PROFILE_REPORT = ANALYSIS_DIR / "profile.json"

PATH_CACHE = e192.PATH_CACHE
ANCHOR_DIR = e192.ANCHOR_DIR
HISTORY_DIR = e192.HISTORY_DIR
LABEL_CACHE = e192.LABEL_CACHE
PATH_FEATURES = (
    PATHS.artifacts
    / "round2"
    / "explicit_path_pdp_predictor"
    / "path_features_train_n4000.npy"
)
SEQUENCE_GEOMETRY = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "sequence_geometry_f16.npy"
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
INDEPENDENT_BASE = (
    PATHS.artifacts / "round2" / "e193_propagation_memory" / "oof_v11_channel.npy"
)
INDEPENDENT_BASE_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_e193_propagation_memory"
    / "independent_v11_base.json"
)
V15_REPORT = (
    PATHS.analysis
    / "results"
    / "round2_official_sionna_path_calibrator"
    / "fixed500_report.json"
)

REPRESENTATIVE_STATE = STATIC_DIR / "representative_state.npz"
SURFACE_FAMILY = STATIC_DIR / "surface_family_i16.npy"
BASE_COEFFICIENT = STATIC_DIR / "canonical_base_coefficient.npy"
STATIC_COMPLETE = STATIC_DIR / "complete.json"

POINTS = 4000
EVENTS = 128
STEPS = 16
PORTS = 8
PERIODS = (16.0, 8.0, 192.0)
COORDINATE_LIMITS = (1.5, 1.0, 3.0)
TINY = 1e-30
V15_SCORE = 0.6863750036074728
TARGET_SCORE = 0.700000


@dataclass(frozen=True)
class Config:
    hidden: int = 128
    surface_dim: int = 40
    interaction_dim: int = 16
    response_dim: int = 20
    heads: int = 4
    set_layers: int = 2
    feedforward: int = 384
    dropout: float = 0.08
    pretrain_epochs: int = 36
    official_epochs: int = 12
    validation_interval: int = 4
    official_validation_interval: int = 2
    state_batch_size: int = 64
    official_batch_size: int = 4
    metric_batch_size: int = 8
    learning_rate: float = 4e-4
    official_learning_rate: float = 8e-5
    weight_decay: float = 2e-4
    base_ridge: float = 0.02
    random_state: int = 20260813194


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "profile", "fold", "aggregate"), default="fold"
    )
    parser.add_argument("--outer", type=int, choices=range(5), default=0)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--force", action="store_true")
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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, value)
    os.replace(temporary, path)


def atomic_save_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def fold_paths(outer: int) -> dict[str, Path]:
    return {
        "checkpoint": FOLD_DIR / f"outer{outer}_model.pt",
        "channel": FOLD_DIR / f"outer{outer}_channel.npy",
        "report": ANALYSIS_DIR / f"outer{outer}_report.json",
    }


def required_inputs() -> tuple[Path, ...]:
    return (
        PATH_CACHE / "mask.npy",
        PATH_CACHE / "source.npy",
        PATH_CACHE / "interaction_count.npy",
        PATH_CACHE / "interactions.npy",
        PATH_CACHE / "primitives.npy",
        PATH_FEATURES,
        SEQUENCE_GEOMETRY,
        ANCHOR_DIR / "mask.npy",
        ANCHOR_DIR / "source.npy",
        ANCHOR_DIR / "representative_slot.npy",
        HISTORY_DIR / "mask.npy",
        HISTORY_DIR / "continuous.npy",
        HISTORY_DIR / "reference_coefficient.npy",
        LABEL_CACHE,
        FULL_FOLD,
        OOF_META,
        TRAIN_CHANNEL,
        INDEPENDENT_BASE,
        INDEPENDENT_BASE_REPORT,
        V15_REPORT,
    )


def validate_independent_base() -> dict[str, object]:
    report = json.loads(INDEPENDENT_BASE_REPORT.read_text(encoding="utf-8"))
    lineage = report.get("data_lineage", {})
    if lineage.get("historical_complete_channel_read") is not False:
        raise RuntimeError("E194 base lineage is not independent of historical channels")
    value = np.load(INDEPENDENT_BASE, mmap_mode="r")
    if value.shape != (500, 256, 4, 192) or value.dtype != np.complex64:
        raise ValueError(f"invalid independent base: {value.shape}/{value.dtype}")
    return report


def build_surface_family(force: bool) -> tuple[np.memmap, dict[str, object]]:
    if SURFACE_FAMILY.is_file() and not force:
        cached = np.load(SURFACE_FAMILY, mmap_mode="r")
        if cached.shape == (POINTS, EVENTS, STEPS) and cached.dtype == np.int16:
            return cached, {
                "reused": True,
                "shape": list(cached.shape),
                "dtype": str(cached.dtype),
                "family_count": int(np.max(cached)),
            }

    started = time.time()
    plane_id, plane_report = e129.canonical_face_planes(3)
    primitives = np.load(PATH_CACHE / "primitives.npy", mmap_mode="r")
    depth = np.load(PATH_CACHE / "interaction_count.npy", mmap_mode="r")
    maximum_primitive = int(np.max(primitives))
    mapping = np.empty(max(maximum_primitive + 1, len(plane_id)), dtype=np.int32)
    mapping[: len(plane_id)] = plane_id
    if maximum_primitive >= len(plane_id):
        extra = maximum_primitive + 1 - len(plane_id)
        mapping[len(plane_id) : maximum_primitive + 1] = (
            int(np.max(plane_id)) + 1 + np.arange(extra, dtype=np.int32)
        )
    family_count = int(np.max(mapping)) + 1
    if family_count + 1 > np.iinfo(np.int16).max:
        raise ValueError(f"surface family count exceeds int16: {family_count}")

    partial = SURFACE_FAMILY.with_name(f".{SURFACE_FAMILY.name}.{os.getpid()}.tmp")
    output = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.int16, shape=(POINTS, EVENTS, STEPS)
    )
    step = np.arange(STEPS, dtype=np.int64)[None, None, :]
    for start in range(0, POINTS, 256):
        end = min(start + 256, POINTS)
        local_depth = np.asarray(depth[start:end], dtype=np.int64)
        local = mapping[np.asarray(primitives[start:end], dtype=np.int64)] + 1
        local[step >= local_depth[..., None]] = 0
        output[start:end] = local.astype(np.int16)
    output.flush()
    del output
    os.replace(partial, SURFACE_FAMILY)
    cached = np.load(SURFACE_FAMILY, mmap_mode="r")
    return cached, {
        "reused": False,
        "shape": list(cached.shape),
        "dtype": str(cached.dtype),
        "family_count": family_count,
        "plane_report": plane_report,
        "maximum_primitive": maximum_primitive,
        "seconds": float(time.time() - started),
    }


def build_representative_state(force: bool) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    expected = {
        "mask": ((POINTS, EVENTS), np.bool_),
        "mass": ((POINTS, EVENTS), np.float32),
        "coordinate": ((POINTS, EVENTS, 3), np.float32),
        "port": ((POINTS, EVENTS, PORTS), np.float32),
        "phase": ((POINTS, EVENTS, PORTS), np.complex64),
        "power": ((POINTS,), np.float64),
        "zero": ((POINTS,), np.bool_),
    }
    if REPRESENTATIVE_STATE.is_file() and not force:
        payload = np.load(REPRESENTATIVE_STATE, allow_pickle=False)
        valid = all(
            name in payload
            and payload[name].shape == shape
            and payload[name].dtype == dtype
            for name, (shape, dtype) in expected.items()
        )
        if valid:
            return {name: payload[name] for name in expected}, {
                "reused": True,
                "path": PATHS.relative(REPRESENTATIVE_STATE),
            }

    started = time.time()
    rows = np.arange(POINTS, dtype=np.int64)
    mass, coordinate, port, phase, power, zero = e192.representative_state(rows)
    anchor_mask = np.asarray(
        np.load(ANCHOR_DIR / "mask.npy", mmap_mode="r"), dtype=bool
    )
    history_mask = np.load(HISTORY_DIR / "mask.npy", mmap_mode="r")
    mask = anchor_mask & np.any(np.asarray(history_mask, dtype=bool), axis=0)
    values = {
        "mask": mask.astype(bool),
        "mass": mass.astype(np.float32),
        "coordinate": coordinate.astype(np.float32),
        "port": port.astype(np.float32),
        "phase": phase.astype(np.complex64),
        "power": power.astype(np.float64),
        "zero": zero.astype(bool),
    }
    atomic_save_npz(REPRESENTATIVE_STATE, **values)
    return values, {
        "reused": False,
        "path": PATHS.relative(REPRESENTATIVE_STATE),
        "active_events": int(np.sum(mask)),
        "zero_points": int(np.sum(zero)),
        "seconds": float(time.time() - started),
    }


def render_coefficients(
    coordinate: torch.Tensor,
    coefficient: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    coefficient = torch.where(mask[..., None], coefficient, torch.zeros_like(coefficient))
    atom = _component_atom(coordinate, torch.complex64, coordinate.device)
    shaped = coefficient.reshape(len(coordinate), coordinate.shape[1], 2, 4)
    channel = torch.einsum("bkpu,bkhvs->bpuhvs", shaped, atom)
    return channel.permute(0, 1, 3, 4, 2, 5).reshape(-1, 256, 4, 192)


def dirichlet(delta: torch.Tensor, size: int) -> torch.Tensor:
    delta = delta.to(torch.float32)
    denominator = float(size) * torch.sin(torch.pi * delta / float(size))
    ratio = torch.sin(torch.pi * delta) / denominator.clamp(min=-1e30, max=1e30)
    near = denominator.abs() < 1e-5
    ratio = torch.where(near, torch.ones_like(ratio), ratio)
    phase = torch.exp(1j * torch.pi * delta * float(size - 1) / float(size))
    return ratio.to(torch.complex64) * phase


def candidate_gram(coordinate: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    delta = coordinate[:, None] - coordinate[:, :, None]
    horizontal = dirichlet(delta[..., 0], 16)
    vertical = dirichlet(delta[..., 1], 8)
    delay = dirichlet(-delta[..., 2], 192)
    gram = horizontal * vertical * delay
    valid = mask[:, :, None] & mask[:, None, :]
    return torch.where(valid, gram, torch.zeros_like(gram))


@torch.inference_mode()
def build_base_coefficient(
    state: dict[str, np.ndarray], device: torch.device, force: bool, ridge: float
) -> tuple[np.memmap, dict[str, object]]:
    if BASE_COEFFICIENT.is_file() and not force:
        cached = np.load(BASE_COEFFICIENT, mmap_mode="r")
        if cached.shape == (500, EVENTS, PORTS) and cached.dtype == np.complex64:
            return cached, {
                "reused": True,
                "shape": list(cached.shape),
                "dtype": str(cached.dtype),
            }

    started = time.time()
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    base = np.load(INDEPENDENT_BASE, mmap_mode="r")
    coordinate = state["coordinate"][query_ids]
    mask = state["mask"][query_ids]
    partial = BASE_COEFFICIENT.with_name(
        f".{BASE_COEFFICIENT.name}.{os.getpid()}.tmp"
    )
    partial.unlink(missing_ok=True)
    output = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.complex64,
        shape=(len(query_ids), EVENTS, PORTS),
    )
    residual_ratios: list[float] = []
    condition_samples: list[float] = []
    for start in range(0, len(query_ids), 4):
        end = min(start + 4, len(query_ids))
        local_coordinate = torch.as_tensor(coordinate[start:end], device=device)
        local_mask = torch.as_tensor(mask[start:end], device=device)
        local_base = torch.as_tensor(
            np.array(base[start:end], dtype=np.complex64, copy=True), device=device
        )
        atom = _component_atom(local_coordinate, torch.complex64, device)
        shaped = local_base.reshape(end - start, 2, 16, 8, 4, 192)
        correlation = torch.einsum(
            "bphvus,bkhvs->bkpu", shaped, torch.conj(atom)
        ).reshape(end - start, EVENTS, PORTS)
        correlation = torch.where(
            local_mask[..., None], correlation, torch.zeros_like(correlation)
        )
        gram = candidate_gram(local_coordinate, local_mask)
        identity = torch.eye(EVENTS, dtype=torch.complex64, device=device)[None]
        inactive = (~local_mask).to(torch.float32)
        system = gram + float(ridge) * identity + inactive[:, :, None] * identity
        coefficient = torch.linalg.solve(system, correlation)
        coefficient = torch.where(
            local_mask[..., None], coefficient, torch.zeros_like(coefficient)
        )
        output[start:end] = coefficient.cpu().numpy().astype(np.complex64)
        reconstructed = render_coefficients(local_coordinate, coefficient, local_mask)
        residual = local_base - reconstructed
        residual_ratios.extend(
            (
                residual.abs().square().sum(dim=(1, 2, 3))
                / local_base.abs().square().sum(dim=(1, 2, 3)).clamp_min(TINY)
            ).cpu().numpy().tolist()
        )
        if start == 0:
            condition_samples.extend(
                torch.linalg.cond(system).real.cpu().numpy().tolist()
            )
    output.flush()
    del output
    os.replace(partial, BASE_COEFFICIENT)
    cached = np.load(BASE_COEFFICIENT, mmap_mode="r")
    return cached, {
        "reused": False,
        "shape": list(cached.shape),
        "dtype": str(cached.dtype),
        "ridge": float(ridge),
        "residual_energy_ratio_mean": float(np.mean(residual_ratios)),
        "residual_energy_ratio_median": float(np.median(residual_ratios)),
        "condition_first_batch": condition_samples,
        "seconds": float(time.time() - started),
    }


def prepare_static(config: Config, device: torch.device, force: bool) -> dict[str, object]:
    started = time.time()
    missing = [PATHS.relative(path) for path in required_inputs() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing E194 inputs: {missing}")
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    lineage = validate_independent_base()
    family, family_report = build_surface_family(force)
    state, state_report = build_representative_state(force)
    coefficient, coefficient_report = build_base_coefficient(
        state, device, force, config.base_ridge
    )

    representative = np.load(ANCHOR_DIR / "representative_slot.npy", mmap_mode="r")
    event_mask = state["mask"]
    rows, events = np.nonzero(event_mask)
    slots = np.asarray(representative[rows, events], dtype=np.int64)
    original_mask = np.load(PATH_CACHE / "mask.npy", mmap_mode="r")
    if not np.all(np.asarray(original_mask[rows, slots], dtype=bool)):
        raise RuntimeError("representative event points to an inactive raw path")
    if len(np.unique(rows.astype(np.int64) * EVENTS + slots)) != len(slots):
        raise RuntimeError("representative raw path slots are duplicated within events")
    if np.max(family) <= 0:
        raise RuntimeError("surface family cache is empty")
    if not np.isfinite(coefficient.real).all() or not np.isfinite(coefficient.imag).all():
        raise FloatingPointError("non-finite canonical base coefficients")

    marker = {
        "experiment": VERSION,
        "status": "complete",
        "lineage": {
            "independent_base_report": PATHS.relative(INDEPENDENT_BASE_REPORT),
            "historical_complete_channel_read": False,
            "base_report_status": lineage.get("status"),
        },
        "surface_family": family_report,
        "representative_state": state_report,
        "base_coefficient": coefficient_report,
        "identity": {
            "active_events": int(len(slots)),
            "representative_raw_slot_unique": True,
            "surface_family_nonempty": True,
            "base_coefficient_finite": True,
        },
        "seconds": float(time.time() - started),
    }
    atomic_write_json(STATIC_COMPLETE, marker)
    atomic_write_json(PREPARE_REPORT, marker)
    return marker


class TensorStore:
    def __init__(self, device: torch.device) -> None:
        if not STATIC_COMPLETE.is_file():
            raise FileNotFoundError("E194 static cache is incomplete; run --stage prepare")
        self.device = device
        metadata = np.load(OOF_META, allow_pickle=False)
        self.query_ids_np = metadata["query_ids"].astype(np.int64)
        self.query_fold_np = metadata["fold_id"].astype(np.int8)
        self.query_lookup_np = np.full(POINTS, -1, dtype=np.int64)
        self.query_lookup_np[self.query_ids_np] = np.arange(len(self.query_ids_np))
        self.full_fold_np = np.load(FULL_FOLD).astype(np.int8)
        setup, positions, _, site, _ = load_positions()
        self.site_np = site.astype(np.int8)
        position = positions[:, :2].astype(np.float32)
        center = position.mean(axis=0)
        scale = np.maximum(position.std(axis=0), 1.0)
        bs = np.asarray(setup.bs_positions, dtype=np.float32)[site, :2]
        relative = position - bs
        relative_scale = np.maximum(relative.std(axis=0), 1.0)
        distance = np.linalg.norm(relative, axis=1, keepdims=True)
        distance_scale = max(float(np.median(distance)), 1.0)
        bearing = np.arctan2(relative[:, 1], relative[:, 0])[:, None]
        point_feature = np.concatenate(
            (
                (position - center) / scale,
                relative / relative_scale,
                distance / distance_scale,
                np.sin(bearing),
                np.cos(bearing),
                np.eye(2, dtype=np.float32)[site],
            ),
            axis=1,
        ).astype(np.float32)

        representative = np.load(
            ANCHOR_DIR / "representative_slot.npy", mmap_mode="r"
        )
        state = np.load(REPRESENTATIVE_STATE, allow_pickle=False)
        labels = np.load(LABEL_CACHE, allow_pickle=False)
        self.representative = torch.as_tensor(
            np.asarray(representative, dtype=np.int64), device=device
        )
        self.event_mask = torch.as_tensor(state["mask"].astype(bool), device=device)
        self.base_mass = torch.as_tensor(state["mass"].astype(np.float32), device=device)
        self.base_coordinate = torch.as_tensor(
            state["coordinate"].astype(np.float32), device=device
        )
        self.base_port = torch.as_tensor(state["port"].astype(np.float32), device=device)
        self.base_phase = torch.as_tensor(
            state["phase"].astype(np.complex64), device=device
        )
        self.base_power = torch.as_tensor(
            state["power"].astype(np.float32), device=device
        )
        self.base_zero = torch.as_tensor(state["zero"].astype(bool), device=device)
        self.anchor_source = torch.as_tensor(
            np.asarray(np.load(ANCHOR_DIR / "source.npy", mmap_mode="r"), dtype=np.int64),
            device=device,
        )
        self.path_features = torch.as_tensor(
            np.asarray(np.load(PATH_FEATURES, mmap_mode="r"), dtype=np.float16),
            device=device,
        )
        self.sequence_geometry = torch.as_tensor(
            np.asarray(np.load(SEQUENCE_GEOMETRY, mmap_mode="r"), dtype=np.float16),
            device=device,
        )
        self.raw_depth = torch.as_tensor(
            np.asarray(
                np.load(PATH_CACHE / "interaction_count.npy", mmap_mode="r"),
                dtype=np.int64,
            ),
            device=device,
        )
        self.raw_interactions = torch.as_tensor(
            np.asarray(np.load(PATH_CACHE / "interactions.npy", mmap_mode="r"), dtype=np.int64),
            device=device,
        )
        self.raw_family = torch.as_tensor(
            np.asarray(np.load(SURFACE_FAMILY, mmap_mode="r"), dtype=np.int64),
            device=device,
        )
        self.point_feature = torch.as_tensor(point_feature, device=device)
        self.site = torch.as_tensor(site.astype(np.int64), device=device)
        self.target_mass = torch.as_tensor(
            labels["mass_absolute"].astype(np.float32), device=device
        )
        self.target_delta = torch.as_tensor(
            labels["delta"].astype(np.float32), device=device
        )
        self.target_port = torch.as_tensor(labels["port"].astype(np.float32), device=device)
        self.target_phase = torch.as_tensor(
            labels["phase"].astype(np.complex64), device=device
        )
        self.target_active = torch.as_tensor(labels["active"].astype(bool), device=device)
        self.target_power = torch.as_tensor(
            labels["point_power"].astype(np.float32), device=device
        )
        self.target_zero = torch.as_tensor(labels["zero"].astype(bool), device=device)
        self.canonical_base = torch.as_tensor(
            np.array(np.load(INDEPENDENT_BASE, mmap_mode="r"), copy=True), device=device
        )
        truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
        self.canonical_truth = torch.as_tensor(
            np.asarray(truth_all[self.query_ids_np], dtype=np.complex64), device=device
        )
        self.canonical_base_coefficient = torch.as_tensor(
            np.asarray(np.load(BASE_COEFFICIENT, mmap_mode="r"), dtype=np.complex64),
            device=device,
        )
        shaped_truth = self.canonical_truth.reshape(-1, 2, 16, 8, 4, 192)
        self.truth_pas = torch.fft.fft2(shaped_truth, dim=(2, 3)).abs().square().reshape_as(
            self.canonical_truth
        )
        self.truth_pdp = torch.fft.ifft(self.canonical_truth, dim=-1).abs().square()
        self.family_count = int(self.raw_family.max().item())
        self.path_dim = int(self.path_features.shape[-1])
        self.sequence_dim = int(self.sequence_geometry.shape[-1])
        self.point_dim = int(self.point_feature.shape[-1])

    def normalization(self, rows: np.ndarray) -> dict[str, torch.Tensor]:
        selected = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        total = torch.zeros(self.path_dim, dtype=torch.float64, device=self.device)
        square = torch.zeros_like(total)
        count = 0
        for start in range(0, len(rows), 128):
            local = selected[start : start + 128]
            rep = self.representative[local].clamp_min(0)
            point = local[:, None]
            value = self.path_features[point, rep].float()
            valid = self.event_mask[local]
            active = value[valid].double()
            total += active.sum(dim=0)
            square += active.square().sum(dim=0)
            count += int(active.shape[0])
        mean = total / max(count, 1)
        variance = (square / max(count, 1) - mean.square()).clamp_min(1e-8)
        std = torch.sqrt(variance)
        std = torch.where(std < 1e-4, torch.ones_like(std), std)
        point_value = self.point_feature[selected]
        point_mean = point_value.mean(dim=0)
        point_std = point_value.std(dim=0)
        point_std = torch.where(
            point_std < 1e-4, torch.ones_like(point_std), point_std
        )
        return {
            "path_mean": mean.float(),
            "path_std": std.float(),
            "point_mean": point_mean,
            "point_std": point_std,
        }

    def batch(
        self, rows: torch.Tensor, normalization: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        rep = self.representative[rows].clamp_min(0)
        point = rows[:, None]
        mask = self.event_mask[rows]
        raw_path = self.path_features[point, rep].float()
        path = (raw_path - normalization["path_mean"]) / normalization["path_std"]
        path = torch.where(mask[..., None], path.clamp(-8.0, 8.0), torch.zeros_like(path))
        sequence = self.sequence_geometry[point, rep].float()
        sequence = torch.where(
            mask[..., None, None], sequence, torch.zeros_like(sequence)
        )
        depth = self.raw_depth[point, rep]
        interactions = self.raw_interactions[point, rep]
        family = self.raw_family[point, rep]
        point_feature = (
            self.point_feature[rows] - normalization["point_mean"]
        ) / normalization["point_std"]
        return {
            "rows": rows,
            "mask": mask,
            "path_feature": path,
            "sequence_geometry": sequence,
            "depth": depth,
            "interactions": interactions,
            "family": family,
            "source": self.anchor_source[rows],
            "site": self.site[rows],
            "point_feature": point_feature.clamp(-8.0, 8.0),
            "base_mass": self.base_mass[rows],
            "base_coordinate": self.base_coordinate[rows],
            "base_port": self.base_port[rows],
            "base_phase": self.base_phase[rows],
            "base_power": self.base_power[rows],
            "base_zero": self.base_zero[rows],
            "target_mass": self.target_mass[rows],
            "target_delta": self.target_delta[rows],
            "target_port": self.target_port[rows],
            "target_phase": self.target_phase[rows],
            "target_active": self.target_active[rows],
            "target_power": self.target_power[rows],
            "target_zero": self.target_zero[rows],
        }

    def canonical_batch(
        self, global_rows: torch.Tensor, normalization: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        local_np = self.query_lookup_np[global_rows.detach().cpu().numpy()]
        if np.any(local_np < 0):
            raise ValueError("official E194 batch contains non-canonical rows")
        local = torch.as_tensor(local_np, dtype=torch.long, device=self.device)
        batch = self.batch(global_rows, normalization)
        batch.update(
            {
                "canonical_local": local,
                "base_channel": self.canonical_base[local],
                "truth": self.canonical_truth[local],
                "truth_pas": self.truth_pas[local],
                "truth_pdp": self.truth_pdp[local],
                "base_coefficient": self.canonical_base_coefficient[local],
            }
        )
        return batch


class SharedSurfaceRenderer(nn.Module):
    def __init__(self, store: TensorStore, config: Config) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden
        self.family_embedding = nn.Embedding(
            store.family_count + 1, config.surface_dim, padding_idx=0
        )
        self.interaction_embedding = nn.Embedding(8, config.interaction_dim)
        self.step_embedding = nn.Embedding(STEPS, config.interaction_dim)
        self.source_embedding = nn.Embedding(3, config.interaction_dim)
        self.site_embedding = nn.Embedding(2, config.interaction_dim)
        step_input = store.sequence_dim + config.surface_dim + 2 * config.interaction_dim
        self.step_input = nn.Sequential(
            nn.Linear(step_input, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.ordered_sequence = nn.GRU(hidden, hidden, batch_first=True)
        self.surface_response = nn.Linear(hidden, config.response_dim)
        path_extra = 1 + 6 + PORTS + 2 * PORTS
        self.path_input = nn.Sequential(
            nn.Linear(store.path_dim + path_extra, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        self.point_input = nn.Sequential(
            nn.Linear(store.point_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden + config.response_dim + 2 * config.interaction_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.heads,
            dim_feedforward=config.feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=config.set_layers)
        self.output_norm = nn.LayerNorm(hidden)
        self.existence_head = nn.Linear(hidden, 1)
        self.mass_head = nn.Linear(hidden, 1)
        self.coordinate_head = nn.Linear(hidden, 3)
        self.port_head = nn.Linear(hidden, PORTS)
        self.phase_head = nn.Linear(hidden, PORTS)
        self.resolved_head = nn.Linear(hidden, 1)
        self.power_head = nn.Linear(hidden, 1)
        self.zero_head = nn.Linear(hidden, 1)
        self.gate_head = nn.Linear(hidden, 1)
        for layer_to_zero in (
            self.existence_head,
            self.mass_head,
            self.coordinate_head,
            self.port_head,
            self.phase_head,
            self.resolved_head,
            self.power_head,
            self.zero_head,
        ):
            nn.init.zeros_(layer_to_zero.weight)
            nn.init.zeros_(layer_to_zero.bias)
        nn.init.constant_(self.resolved_head.bias, 2.2)
        nn.init.constant_(self.zero_head.bias, -2.6)
        nn.init.normal_(self.surface_response.weight, std=1e-3)
        nn.init.zeros_(self.surface_response.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -2.2)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mask = batch["mask"]
        batch_size, paths = mask.shape
        step_ids = torch.arange(STEPS, device=mask.device)
        step_mask = step_ids[None, None] < batch["depth"][..., None]
        family = self.family_embedding(batch["family"].clamp(0, self.family_embedding.num_embeddings - 1))
        interaction = self.interaction_embedding(batch["interactions"].clamp(0, 7))
        step = self.step_embedding(step_ids)[None, None].expand(batch_size, paths, -1, -1)
        step_hidden = self.step_input(
            torch.cat((batch["sequence_geometry"], family, interaction, step), dim=-1)
        )
        flat = step_hidden.reshape(batch_size * paths, STEPS, self.config.hidden)
        ordered, _ = self.ordered_sequence(flat)
        depth_index = batch["depth"].clamp(1, STEPS).reshape(-1) - 1
        flat_index = torch.arange(len(flat), device=mask.device)
        ordered_state = ordered[flat_index, depth_index].reshape(
            batch_size, paths, self.config.hidden
        )
        ordered_state = ordered_state * (batch["depth"] > 0)[..., None]
        step_response = self.surface_response(step_hidden)
        response = (step_response * step_mask[..., None]).sum(dim=2)
        response = response / torch.sqrt(
            batch["depth"].clamp_min(1).to(response.dtype)[..., None]
        )

        coordinate_phase = 2.0 * torch.pi * batch["base_coordinate"] / torch.tensor(
            PERIODS, dtype=torch.float32, device=mask.device
        )
        base_extra = torch.cat(
            (
                torch.log(batch["base_mass"].clamp_min(1e-8))[..., None],
                torch.sin(coordinate_phase),
                torch.cos(coordinate_phase),
                torch.log(batch["base_port"].clamp_min(1e-8)),
                batch["base_phase"].real,
                batch["base_phase"].imag,
            ),
            dim=2,
        )
        path_state = self.path_input(torch.cat((batch["path_feature"], base_extra), dim=2))
        point_state = self.point_input(batch["point_feature"])
        source = self.source_embedding((batch["source"] + 1).clamp(0, 2))
        site = self.site_embedding(batch["site"])
        site = site[:, None].expand(-1, paths, -1)
        hidden = self.fusion(
            torch.cat((path_state, ordered_state, response, source, site), dim=2)
        ) + point_state[:, None]
        safe_mask = mask.clone()
        empty = ~safe_mask.any(dim=1)
        safe_mask[empty, 0] = True
        hidden = self.set_encoder(hidden, src_key_padding_mask=~safe_mask)
        hidden = self.output_norm(hidden)
        hidden = torch.where(mask[..., None], hidden, torch.zeros_like(hidden))

        existence_logit = self.existence_head(hidden)[..., 0] + 0.5 * response[..., 0]
        existence = torch.sigmoid(existence_logit) * mask.to(hidden.dtype)
        mass_logit = (
            torch.log(batch["base_mass"].clamp_min(1e-8))
            + 2.0 * torch.tanh(self.mass_head(hidden)[..., 0] + 0.5 * response[..., 0])
            + torch.log(existence.clamp_min(1e-8))
        )
        mass_logit = torch.where(mask, mass_logit, torch.full_like(mass_logit, -1e4))
        mass_normalized = torch.softmax(mass_logit, dim=1)
        mass_normalized = torch.where(mask, mass_normalized, torch.zeros_like(mass_normalized))
        valid = mask.to(hidden.dtype)
        pooled = (hidden * valid[..., None]).sum(dim=1) / valid.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        resolved_fraction = torch.sigmoid(self.resolved_head(pooled)[:, 0])
        zero_logit = self.zero_head(pooled)[:, 0]
        zero_probability = torch.sigmoid(zero_logit)
        mass_absolute = (
            mass_normalized
            * resolved_fraction[:, None]
            * (1.0 - zero_probability[:, None])
        )

        coordinate_raw = self.coordinate_head(hidden) + 0.5 * response[..., 1:4]
        coordinate_delta = torch.tanh(coordinate_raw) * torch.tensor(
            COORDINATE_LIMITS, dtype=torch.float32, device=mask.device
        )
        coordinate = torch.remainder(
            batch["base_coordinate"] + coordinate_delta,
            torch.tensor(PERIODS, dtype=torch.float32, device=mask.device),
        )
        port_raw = self.port_head(hidden) + 0.5 * response[..., 4:12]
        port = torch.softmax(
            torch.log(batch["base_port"].clamp_min(1e-8))
            + 2.0 * torch.tanh(port_raw),
            dim=2,
        )
        phase_raw = self.phase_head(hidden) + 0.5 * response[..., 12:20]
        phase_delta = torch.pi * torch.tanh(phase_raw)
        phase = batch["base_phase"] * torch.exp(1j * phase_delta)
        base_log_power = torch.log(batch["base_power"].clamp_min(1e-30))
        log_power = base_log_power + 2.0 * torch.tanh(self.power_head(pooled)[:, 0])
        coefficient_power = (
            torch.exp(log_power)[:, None, None]
            * mass_absolute[..., None]
            * port
        )
        coefficient = torch.sqrt(coefficient_power.clamp_min(1e-30)) * phase
        has_event = mask.any(dim=1).to(hidden.dtype)
        point_gate = torch.sigmoid(self.gate_head(pooled)[:, 0]) * has_event
        # Replacement strength is independent of predicted existence: a false
        # path must still be subtracted from the base field.
        action = point_gate[:, None] * mask.to(hidden.dtype)
        return {
            "existence_logit": existence_logit,
            "existence": existence,
            "mass_normalized": mass_normalized,
            "mass_absolute": mass_absolute,
            "resolved_fraction": resolved_fraction,
            "coordinate": coordinate,
            "coordinate_delta": coordinate_delta,
            "port": port,
            "phase": phase,
            "log_power": log_power,
            "zero_logit": zero_logit,
            "zero_probability": zero_probability,
            "coefficient": coefficient,
            "action": action,
            "point_gate": point_gate,
        }


def state_objective(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = batch["mask"]
    active = batch["target_active"] & mask
    target_mass = batch["target_mass"] * mask.to(batch["target_mass"].dtype)
    target_total = target_mass.sum(dim=1)
    target_normalized = target_mass / target_total[:, None].clamp_min(TINY)
    nonzero = ~batch["target_zero"]
    mass_affinity = (
        torch.sqrt(output["mass_normalized"].clamp_min(1e-12))
        * torch.sqrt(target_normalized.clamp_min(0.0))
    ).sum(dim=1)
    mass_loss = (1.0 - mass_affinity[nonzero]).mean() if nonzero.any() else mass_affinity.mean() * 0.0
    positive = active.sum().clamp_min(1)
    negative = (mask & ~active).sum().clamp_min(1)
    pos_weight = (negative / positive).clamp(1.0, 8.0).to(torch.float32)
    existence_loss = F.binary_cross_entropy_with_logits(
        output["existence_logit"][mask],
        active[mask].to(torch.float32),
        pos_weight=pos_weight,
    )
    resolved_loss = F.smooth_l1_loss(output["resolved_fraction"], target_total.clamp(0.0, 1.0))
    weight = target_normalized[active]
    weight = weight / weight.mean().clamp_min(TINY) if active.any() else weight
    if active.any():
        coordinate = F.smooth_l1_loss(
            output["coordinate_delta"][active], batch["target_delta"][active], reduction="none"
        ).mean(dim=1)
        coordinate_loss = (coordinate * weight).mean()
        port_affinity = (
            torch.sqrt(output["port"][active].clamp_min(1e-12))
            * torch.sqrt(batch["target_port"][active].clamp_min(0.0))
        ).sum(dim=1)
        port_loss = ((1.0 - port_affinity) * weight).mean()
        phase_score = (
            torch.real(output["phase"][active] * torch.conj(batch["target_phase"][active]))
            * batch["target_port"][active]
        ).sum(dim=1)
        phase_loss = ((1.0 - phase_score) * weight).mean()
    else:
        coordinate_loss = output["mass_absolute"].sum() * 0.0
        port_loss = output["mass_absolute"].sum() * 0.0
        phase_loss = output["mass_absolute"].sum() * 0.0
    power_loss = (
        F.smooth_l1_loss(
            output["log_power"][nonzero],
            torch.log(batch["target_power"][nonzero].clamp_min(1e-30)),
        )
        if nonzero.any()
        else output["log_power"].sum() * 0.0
    )
    zero_loss = F.binary_cross_entropy_with_logits(
        output["zero_logit"], batch["target_zero"].to(torch.float32)
    )
    loss = (
        0.55 * mass_loss
        + 0.10 * existence_loss
        + 0.08 * resolved_loss
        + 0.15 * coordinate_loss
        + 0.12 * port_loss
        + 0.28 * phase_loss
        + 0.08 * power_loss
        + 0.05 * zero_loss
    )
    return loss, {
        "mass": 1.0 - mass_loss,
        "existence": existence_loss,
        "resolved": resolved_loss,
        "coordinate": coordinate_loss,
        "port": 1.0 - port_loss,
        "phase": 1.0 - phase_loss,
        "power": power_loss,
        "zero": zero_loss,
    }


def render_complete(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> torch.Tensor:
    scale = output["action"] * batch["mask"].to(output["action"].dtype)
    modified = render_coefficients(
        output["coordinate"], output["coefficient"] * scale[..., None], batch["mask"]
    )
    base_paths = render_coefficients(
        batch["base_coordinate"],
        batch["base_coefficient"] * scale[..., None],
        batch["mask"],
    )
    prediction = batch["base_channel"] + modified - base_paths
    current_power = prediction.abs().square().sum(dim=(1, 2, 3))
    base_power = batch["base_channel"].abs().square().sum(dim=(1, 2, 3))
    desired_power = (
        (1.0 - output["point_gate"]) * base_power
        + output["point_gate"] * torch.exp(output["log_power"])
    )
    scale_power = torch.sqrt(desired_power / current_power.clamp_min(TINY))
    return prediction * scale_power[:, None, None, None]


def official_objective(
    prediction: torch.Tensor, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return e192.official_objective(
        prediction, batch["truth"], batch["truth_pas"], batch["truth_pdp"]
    )


def stable_step(
    loss: torch.Tensor, model: nn.Module, optimizer: torch.optim.Optimizer
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm_squared = torch.zeros((), dtype=torch.float64, device=loss.device)
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("non-finite E194 gradient")
        norm_squared += parameter.grad.double().square().sum()
    norm = torch.sqrt(norm_squared)
    scale = min(1.0, 2.0 / max(float(norm.detach().cpu()), 1e-30))
    if scale < 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
    optimizer.step()
    return float(norm.detach().cpu())


def run_state_epoch(
    model: SharedSurfaceRenderer,
    store: TensorStore,
    rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    config: Config,
    rng: np.random.Generator,
) -> dict[str, float]:
    model.train()
    order = rng.permutation(rows)
    totals: dict[str, list[float]] = {
        name: []
        for name in (
            "loss", "mass", "existence", "resolved", "coordinate", "port", "phase", "power", "zero", "gradient"
        )
    }
    for start in range(0, len(order), config.state_batch_size):
        local = torch.as_tensor(
            order[start : start + config.state_batch_size],
            dtype=torch.long,
            device=store.device,
        )
        batch = store.batch(local, normalization)
        output = model(batch)
        loss, values = state_objective(output, batch)
        gradient = stable_step(loss, model, optimizer)
        totals["loss"].append(float(loss.detach().cpu()))
        totals["gradient"].append(gradient)
        for name, value in values.items():
            totals[name].append(float(value.detach().cpu()))
    return {name: float(np.mean(values)) for name, values in totals.items()}


def run_official_epoch(
    model: SharedSurfaceRenderer,
    store: TensorStore,
    state_rows: np.ndarray,
    canonical_rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    config: Config,
    rng: np.random.Generator,
) -> dict[str, float]:
    state_values = run_state_epoch(
        model, store, state_rows, normalization, optimizer, config, rng
    )
    model.train()
    order = rng.permutation(canonical_rows)
    scores: list[float] = []
    gates: list[float] = []
    gradients: list[float] = []
    for start in range(0, len(order), config.official_batch_size):
        local = torch.as_tensor(
            order[start : start + config.official_batch_size],
            dtype=torch.long,
            device=store.device,
        )
        batch = store.canonical_batch(local, normalization)
        output = model(batch)
        prediction = render_complete(output, batch)
        score_loss, official = official_objective(prediction, batch)
        state_loss, _ = state_objective(output, batch)
        loss = score_loss + 0.25 * state_loss
        gradients.append(stable_step(loss, model, optimizer))
        scores.append(float(official["score"].detach().cpu()))
        gates.append(float(output["point_gate"].mean().detach().cpu()))
    return {
        **{f"state_{name}": value for name, value in state_values.items()},
        "official_score": float(np.mean(scores)),
        "point_gate": float(np.mean(gates)),
        "official_gradient": float(np.mean(gradients)),
    }


@torch.inference_mode()
def evaluate_rows(
    model: SharedSurfaceRenderer | None,
    store: TensorStore,
    rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    config: Config,
    collect_channel: bool = False,
) -> tuple[dict[str, float | int], np.ndarray | None, dict[str, float]]:
    if model is not None:
        model.eval()
    statistics = PointMetricStatistics(
        pas=np.empty(len(rows), dtype=np.float64),
        pdp=np.empty(len(rows), dtype=np.float64),
        squared_error=np.empty(len(rows), dtype=np.float64),
        truth_energy=np.empty(len(rows), dtype=np.float64),
    )
    channel = (
        np.empty((len(rows), 256, 4, 192), dtype=np.complex64)
        if collect_channel
        else None
    )
    gate_values: list[np.ndarray] = []
    action_values: list[np.ndarray] = []
    for start in range(0, len(rows), config.metric_batch_size):
        end = min(start + config.metric_batch_size, len(rows))
        local = torch.as_tensor(rows[start:end], dtype=torch.long, device=store.device)
        batch = store.canonical_batch(local, normalization)
        if model is None:
            prediction = batch["base_channel"]
        else:
            output = model(batch)
            prediction = render_complete(output, batch)
            gate_values.append(output["point_gate"].cpu().numpy())
            action_values.append(output["action"][batch["mask"]].cpu().numpy())
        values = point_metric_statistics_torch(prediction, batch["truth"])
        statistics.pas[start:end] = values.pas
        statistics.pdp[start:end] = values.pdp
        statistics.squared_error[start:end] = values.squared_error
        statistics.truth_energy[start:end] = values.truth_energy
        if channel is not None:
            channel[start:end] = prediction.cpu().numpy().astype(np.complex64)
    diagnostics = {
        "point_gate_mean": float(np.mean(np.concatenate(gate_values)))
        if gate_values
        else 0.0,
        "path_action_mean": float(np.mean(np.concatenate(action_values)))
        if action_values
        else 0.0,
    }
    return aggregate_statistics(statistics), channel, diagnostics


def train_screen(
    store: TensorStore,
    outer: int,
    config: Config,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, object], dict[str, torch.Tensor]]:
    inner = (outer + 1) % 5
    fit_rows = np.flatnonzero(
        (store.full_fold_np != outer) & (store.full_fold_np != inner)
    )
    validation_rows = store.query_ids_np[store.query_fold_np == inner]
    canonical_fit = store.query_ids_np[
        (store.query_fold_np != outer) & (store.query_fold_np != inner)
    ]
    normalization = store.normalization(fit_rows)
    torch.manual_seed(config.random_state + outer * 1009)
    torch.cuda.manual_seed_all(config.random_state + outer * 1009)
    model = SharedSurfaceRenderer(store, config).to(store.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.pretrain_epochs, eta_min=config.learning_rate * 0.08
    )
    rng = np.random.default_rng(config.random_state + outer * 2027)
    baseline, _, _ = evaluate_rows(
        None, store, validation_rows, normalization, config
    )
    best_score = float(baseline["score"])
    best_state: dict[str, torch.Tensor] | None = None
    best_stage = "base"
    best_epoch = 0
    history: list[dict[str, object]] = [
        {"stage": "base", "epoch": 0, "validation": baseline}
    ]

    for epoch in range(1, config.pretrain_epochs + 1):
        train = run_state_epoch(
            model, store, fit_rows, normalization, optimizer, config, rng
        )
        scheduler.step()
        record: dict[str, object] = {"stage": "state", "epoch": epoch, "train": train}
        if epoch % config.validation_interval == 0:
            validation, _, diagnostics = evaluate_rows(
                model, store, validation_rows, normalization, config
            )
            record["validation"] = validation
            record["diagnostics"] = diagnostics
            if float(validation["score"]) > best_score:
                best_score = float(validation["score"])
                best_state = copy.deepcopy(model.state_dict())
                best_stage = "state"
                best_epoch = epoch
        history.append(record)
        if epoch % config.validation_interval == 0:
            print(
                f"E194 outer={outer} state={epoch}/{config.pretrain_epochs} "
                f"loss={train['loss']:.6f} validation="
                f"{float(record['validation']['score']):.6f}",
                flush=True,
            )

    official_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.official_learning_rate,
        weight_decay=config.weight_decay,
    )
    official_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        official_optimizer,
        T_max=config.official_epochs,
        eta_min=config.official_learning_rate * 0.08,
    )
    for epoch in range(1, config.official_epochs + 1):
        train = run_official_epoch(
            model,
            store,
            fit_rows,
            canonical_fit,
            normalization,
            official_optimizer,
            config,
            rng,
        )
        official_scheduler.step()
        record = {"stage": "official", "epoch": epoch, "train": train}
        if epoch % config.official_validation_interval == 0:
            validation, _, diagnostics = evaluate_rows(
                model, store, validation_rows, normalization, config
            )
            record["validation"] = validation
            record["diagnostics"] = diagnostics
            if float(validation["score"]) > best_score:
                best_score = float(validation["score"])
                best_state = copy.deepcopy(model.state_dict())
                best_stage = "official"
                best_epoch = epoch
        history.append(record)
        if epoch % config.official_validation_interval == 0:
            print(
                f"E194 outer={outer} official={epoch}/{config.official_epochs} "
                f"train={train['official_score']:.6f} validation="
                f"{float(record['validation']['score']):.6f} gate="
                f"{float(record['diagnostics']['point_gate_mean']):.4f}",
                flush=True,
            )
    return best_state, {
        "fit_count": int(len(fit_rows)),
        "canonical_fit_count": int(len(canonical_fit)),
        "validation_count": int(len(validation_rows)),
        "baseline": baseline,
        "best_score": best_score,
        "best_stage": best_stage,
        "best_epoch": best_epoch,
        "history": history,
    }, normalization


def refit_and_predict(
    store: TensorStore,
    outer: int,
    best_stage: str,
    best_epoch: int,
    config: Config,
) -> tuple[SharedSurfaceRenderer | None, dict[str, torch.Tensor], dict[str, object]]:
    refit_rows = np.flatnonzero(store.full_fold_np != outer)
    canonical_refit = store.query_ids_np[store.query_fold_np != outer]
    normalization = store.normalization(refit_rows)
    if best_stage == "base":
        return None, normalization, {
            "stage": "base",
            "state_epochs": 0,
            "official_epochs": 0,
            "history": [],
        }
    torch.manual_seed(config.random_state + outer * 1009)
    torch.cuda.manual_seed_all(config.random_state + outer * 1009)
    model = SharedSurfaceRenderer(store, config).to(store.device)
    rng = np.random.default_rng(config.random_state + outer * 4099)
    history: list[dict[str, object]] = []
    state_epochs = best_epoch if best_stage == "state" else config.pretrain_epochs
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(state_epochs, 1), eta_min=config.learning_rate * 0.08
    )
    for epoch in range(1, state_epochs + 1):
        train = run_state_epoch(
            model, store, refit_rows, normalization, optimizer, config, rng
        )
        scheduler.step()
        history.append({"stage": "state", "epoch": epoch, "train": train})
    official_epochs = best_epoch if best_stage == "official" else 0
    if official_epochs:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.official_learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=official_epochs,
            eta_min=config.official_learning_rate * 0.08,
        )
        for epoch in range(1, official_epochs + 1):
            train = run_official_epoch(
                model,
                store,
                refit_rows,
                canonical_refit,
                normalization,
                optimizer,
                config,
                rng,
            )
            scheduler.step()
            history.append({"stage": "official", "epoch": epoch, "train": train})
    return model, normalization, {
        "stage": best_stage,
        "state_epochs": state_epochs,
        "official_epochs": official_epochs,
        "history": history,
    }


def run_fold(outer: int, config: Config, device: torch.device, force: bool) -> dict[str, object]:
    paths = fold_paths(outer)
    if not force and all(path.is_file() for path in paths.values()):
        return json.loads(paths["report"].read_text(encoding="utf-8"))
    started = time.time()
    started_unix = time.time()
    if device.type != "cuda":
        raise RuntimeError("E194 fold training requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    load_started = time.time()
    store = TensorStore(device)
    load_seconds = time.time() - load_started
    screen_started = time.time()
    best_state, screen, _ = train_screen(store, outer, config)
    screen_seconds = time.time() - screen_started
    refit_started = time.time()
    model, normalization, refit = refit_and_predict(
        store, outer, str(screen["best_stage"]), int(screen["best_epoch"]), config
    )
    refit_seconds = time.time() - refit_started
    held_rows = store.query_ids_np[store.query_fold_np == outer]
    selected, channel, diagnostics = evaluate_rows(
        model, store, held_rows, normalization, config, collect_channel=True
    )
    baseline, _, _ = evaluate_rows(
        None, store, held_rows, normalization, config, collect_channel=False
    )
    if channel is None:
        raise RuntimeError("E194 held channel was not collected")
    atomic_save_npy(paths["channel"], channel)
    checkpoint = {
        "experiment": VERSION,
        "outer": outer,
        "config": asdict(config),
        "state_dict": model.state_dict() if model is not None else None,
        "normalization": {key: value.detach().cpu() for key, value in normalization.items()},
        "best_stage": screen["best_stage"],
        "best_epoch": screen["best_epoch"],
        "surface_family_count": store.family_count,
        "historical_complete_channel_read": False,
    }
    atomic_torch_save(paths["checkpoint"], checkpoint)
    finished_unix = time.time()
    report = {
        "experiment": VERSION,
        "status": "fold_complete",
        "outer": outer,
        "configuration": asdict(config),
        "screen": screen,
        "refit": refit,
        "held": {
            "count": int(len(held_rows)),
            "baseline": baseline,
            "selected": selected,
            "delta_score": float(selected["score"] - baseline["score"]),
            "diagnostics": diagnostics,
        },
        "output": {
            "channel": PATHS.relative(paths["channel"]),
            "checkpoint": PATHS.relative(paths["checkpoint"]),
            "shape": list(channel.shape),
            "dtype": str(channel.dtype),
            "finite": bool(np.isfinite(channel).all()),
        },
        "performance": {
            "device": torch.cuda.get_device_name(device),
            "peak_torch_memory_gib": float(torch.cuda.max_memory_allocated(device) / 2**30),
            "peak_rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20),
            "load_seconds": float(load_seconds),
            "screen_seconds": float(screen_seconds),
            "refit_seconds": float(refit_seconds),
            "seconds": float(time.time() - started),
        },
        "started_unix": started_unix,
        "finished_unix": finished_unix,
    }
    atomic_write_json(paths["report"], report)
    return report


def profile(config: Config, device: torch.device) -> dict[str, object]:
    started = time.time()
    if device.type != "cuda":
        raise RuntimeError("E194 profile requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    load_started = time.time()
    store = TensorStore(device)
    load_seconds = time.time() - load_started
    fit_rows = np.flatnonzero((store.full_fold_np != 0) & (store.full_fold_np != 1))
    normalization = store.normalization(fit_rows)
    model = SharedSurfaceRenderer(store, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(config.random_state)
    state_rows = fit_rows[: config.state_batch_size * 2]
    canonical_rows = store.query_ids_np[
        (store.query_fold_np != 0) & (store.query_fold_np != 1)
    ][: config.official_batch_size * 2]
    torch.cuda.synchronize(device)
    state_started = time.time()
    state = run_state_epoch(
        model, store, state_rows, normalization, optimizer, config, rng
    )
    torch.cuda.synchronize(device)
    state_seconds = time.time() - state_started
    official_started = time.time()
    official = run_official_epoch(
        model,
        store,
        state_rows,
        canonical_rows,
        normalization,
        optimizer,
        config,
        rng,
    )
    torch.cuda.synchronize(device)
    official_seconds = time.time() - official_started
    validation_started = time.time()
    validation, _, diagnostics = evaluate_rows(
        model, store, canonical_rows, normalization, config
    )
    validation_seconds = time.time() - validation_started
    result = {
        "experiment": VERSION,
        "status": "profile_complete",
        "configuration": asdict(config),
        "shape_checks": {
            "path_dim": store.path_dim,
            "sequence_dim": store.sequence_dim,
            "family_count": store.family_count,
            "state_rows": int(len(state_rows)),
            "official_rows": int(len(canonical_rows)),
        },
        "state": state,
        "official": official,
        "validation": validation,
        "diagnostics": diagnostics,
        "performance": {
            "device": torch.cuda.get_device_name(device),
            "load_seconds": float(load_seconds),
            "state_two_batches_seconds": float(state_seconds),
            "official_two_batches_seconds": float(official_seconds),
            "validation_seconds": float(validation_seconds),
            "peak_torch_memory_gib": float(torch.cuda.max_memory_allocated(device) / 2**30),
            "peak_rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20),
            "expected_dominant_bottleneck": "state sequence/set forward-backward",
            "wall_seconds": float(time.time() - started),
        },
    }
    atomic_write_json(PROFILE_REPORT, result)
    return result


def aggregate(config: Config, device: torch.device) -> dict[str, object]:
    started = time.time()
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    lookup = np.full(POINTS, -1, dtype=np.int64)
    lookup[query_ids] = np.arange(len(query_ids), dtype=np.int64)
    channel = np.empty((len(query_ids), 256, 4, 192), dtype=np.complex64)
    fold_reports: list[dict[str, object]] = []
    for outer in range(5):
        paths = fold_paths(outer)
        if not paths["report"].is_file() or not paths["channel"].is_file():
            raise FileNotFoundError(f"missing E194 outer{outer} output")
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        expected = query_ids[fold_id == outer]
        local = np.load(paths["channel"], mmap_mode="r")
        if local.shape != (len(expected), 256, 4, 192) or local.dtype != np.complex64:
            raise ValueError(f"invalid E194 outer{outer} channel: {local.shape}/{local.dtype}")
        channel[lookup[expected]] = np.asarray(local, dtype=np.complex64)
        fold_reports.append(report)
    if not np.isfinite(channel.real).all() or not np.isfinite(channel.imag).all():
        raise FloatingPointError("non-finite E194 fixed500 channel")
    atomic_save_npy(OUTPUT_CHANNEL, channel)
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    truth = torch.as_tensor(
        np.asarray(truth_all[query_ids], dtype=np.complex64), device=device
    )
    prediction = torch.as_tensor(channel, device=device)
    values = point_metric_statistics_torch(prediction, truth)
    overall = aggregate_statistics(values)
    folds = [
        {"outer": outer, **aggregate_statistics(values, fold_id == outer)}
        for outer in range(5)
    ]
    v15 = json.loads(V15_REPORT.read_text(encoding="utf-8"))
    v15_folds = {int(row["outer"]): row["selected"] for row in v15["folds"]}
    fold_delta = [
        float(row["score"] - v15_folds[int(row["outer"])]["score"])
        for row in folds
    ]
    positive = int(sum(value > 0.0 for value in fold_delta))
    worst = float(min(fold_delta))
    above_v15 = bool(float(overall["score"]) > V15_SCORE)
    target_reached = bool(float(overall["score"]) >= TARGET_SCORE)
    status = "passed" if target_reached else ("improved_below_target" if above_v15 else "failed")
    phase_seconds = {
        f"outer{report['outer']}": float(report["performance"]["seconds"])
        for report in fold_reports
    }
    result = {
        "experiment": VERSION,
        "status": status,
        "configuration": asdict(config),
        "metrics": {"overall": overall, "folds": folds},
        "baseline": {
            "v15_score": V15_SCORE,
            "v15_online": 0.686591,
            "delta_score": float(overall["score"] - V15_SCORE),
            "target_score": TARGET_SCORE,
            "delta_target": float(overall["score"] - TARGET_SCORE),
            "fold_delta": fold_delta,
        },
        "gate": {
            "above_v15": above_v15,
            "target_reached": target_reached,
            "positive_folds": positive,
            "minimum_positive_folds": 4,
            "worst_fold_delta": worst,
            "minimum_worst_fold_delta": -0.001,
            "formal_pass": bool(target_reached and positive >= 4 and worst >= -0.001),
        },
        "lineage": {
            "historical_v7_v15_channel_read": False,
            "independent_base": PATHS.relative(INDEPENDENT_BASE),
            "official_labels": PATHS.relative(LABEL_CACHE),
            "raw_sionna_path_cache": PATHS.relative(PATH_CACHE),
            "surface_family": PATHS.relative(SURFACE_FAMILY),
        },
        "output": {
            "channel": PATHS.relative(OUTPUT_CHANNEL),
            "shape": list(channel.shape),
            "dtype": str(channel.dtype),
            "finite": True,
        },
        "performance": {
            "aggregate_device": torch.cuda.get_device_name(device),
            "fold_seconds": phase_seconds,
            "fold_devices": [report["performance"]["device"] for report in fold_reports],
            "fold_peak_torch_memory_gib": [
                report["performance"]["peak_torch_memory_gib"] for report in fold_reports
            ],
            "aggregate_seconds": float(time.time() - started),
            "measured_bottleneck": "state sequence/set forward-backward",
        },
    }
    atomic_write_json(REPORT, result)
    return result


def main() -> None:
    args = parse_args()
    config = Config()
    device = torch.device(args.device)
    try:
        if args.stage == "prepare":
            result = prepare_static(config, device, args.force)
        elif args.stage == "profile":
            result = profile(config, device)
        elif args.stage == "fold":
            result = run_fold(args.outer, config, device, args.force)
        else:
            result = aggregate(config, device)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
