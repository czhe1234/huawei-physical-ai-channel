#!/usr/bin/env python3
"""Train the E211 full physical path-state operator and score fixed500."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import time
import traceback
from typing import Any

import numpy as np
import torch
from torch import nn

import analyze_e114_path_state_variable_ceiling as e114
import analyze_e124_one_to_one_path_identity_ceiling as e124
import build_v7_complex_carrier_bank as carrier_bank
from carrier_calibration import estimate_carrier
from project_paths import PATHS
from round2.data import load_positions
from round2.physical_prior.source_port_full_state_tokens import (
    RAW_TOKEN_CACHE,
    TOKEN_REPORT,
)
from run_rbf_spectral_field import statistics
import train_e115_complete_path_state_operator as e115
import train_e150_v7_pathwise_carrier_operator as e150
import train_e205_ordered_map_path_calibrator as e205


VERSION = "Round2-E211-full-physical-path-state-residual"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e211_full_path_state_residual"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e211_full_path_state_residual"
E210_DIR = PATHS.artifacts / "round2" / "e210_source_port_full_channel"
E210_RAW = E210_DIR / "raw_tokens_n4000_k512.npz"
FULL_CHANNEL = ARTIFACT_DIR / "fixed500_channel.npy"
FULL_REPORT = ANALYSIS_DIR / "fixed500_report.json"

CANDIDATES = 128
CARRIERS = 17
PHYSICAL_STATES = 3
PROPOSALS = CARRIERS + PHYSICAL_STATES
PORTS = 8
STEPS = 16
TINY = 1e-30


@dataclass(frozen=True)
class Config(e150.Config):
    hidden: int = 192
    carrier_hidden: int = 96
    sequence_layers: int = 1
    pretrain_epochs: int = 18
    finetune_epochs: int = 12
    validation_interval: int = 2
    batch_size: int = 16
    metric_batch_size: int = 8
    carrier_neighbors: int = 32
    carrier_batch_size: int = 16
    action_limit: float = 1.0
    coordinate_limits: tuple[float, float, float] = (0.50, 0.35, 1.50)
    path_gain_limit: float = 3.0
    port_gain_limit: float = 1.0
    phase_limit: float = math.pi
    power_limit: float = 1.0
    required_score: float = 0.686375
    required_positive_folds: int = 4
    required_worst_v7_delta: float = -0.001
    seed: int = 20_260_815_211


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fold", "aggregate"), required=True)
    parser.add_argument("--outer", type=int, choices=range(5))
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def ids_digest(ids: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(ids).view(np.uint8)).hexdigest()


def fold_dir(outer: int) -> Path:
    return ARTIFACT_DIR / f"outer{outer}"


def cache_dir(outer: int, role: str) -> Path:
    return fold_dir(outer) / f"{role}_cache"


def shared_dir(outer: int) -> Path:
    return fold_dir(outer) / "shared"


def report_path(outer: int) -> Path:
    return ANALYSIS_DIR / f"outer{outer}.json"


def channel_path(outer: int) -> Path:
    return fold_dir(outer) / "held_channel.npy"


def held_ids_path(outer: int) -> Path:
    return fold_dir(outer) / "held_global_ids.npy"


def checkpoint_path(outer: int, name: str) -> Path:
    return fold_dir(outer) / f"{name}.pt"


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    value = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"invalid E211 checkpoint {path}")
    return value


def e210_input_dir(outer: int) -> Path:
    return E210_DIR / f"outer{outer}" / "inputs"


def carrier_reference_index() -> int:
    names = [item.name for item in carrier_bank.carrier_configs()]
    name = "radial_idw_k24"
    if name not in names:
        raise RuntimeError(f"missing carrier {name}")
    return names.index(name)


def split_ids(outer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fold, query_ids, canonical = e205.canonical_full_fold()
    validation = (outer + 1) % 5
    screen = np.flatnonzero((fold != outer) & (fold != validation)).astype(np.int64)
    refit = np.flatnonzero(fold != outer).astype(np.int64)
    validation_rows = np.flatnonzero(canonical == validation).astype(np.int64)
    held_rows = np.flatnonzero(canonical == outer).astype(np.int64)
    return screen, refit, validation_rows, held_rows


def _selected_physical_state(outer: int, force: bool) -> dict[str, Any]:
    directory = shared_dir(outer)
    state_path = directory / "state.json"
    required = (
        directory / "mask.npy",
        directory / "coordinate.npy",
        directory / "candidate_feature.npy",
        directory / "point_feature.npy",
        directory / "sequence_feature.npy",
        directory / "interaction.npy",
        directory / "sequence_mask.npy",
        directory / "physical_mask.npy",
        directory / "physical_coordinate.npy",
        directory / "physical_relative_energy.npy",
        directory / "physical_absolute_energy.npy",
        directory / "physical_coefficient.npy",
        directory / "target_positive.npy",
        directory / "target_coordinate.npy",
        directory / "target_coefficient.npy",
        directory / "target_point_power.npy",
    )
    if state_path.is_file() and all(path.is_file() for path in required) and not force:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "complete":
            return {"reused": True, **state}

    started = time.time()
    input_dir = e210_input_dir(outer)
    for name in (
        "selected_index_i32.npy",
        "coordinate_f32.npy",
        "candidate_feature_f16.npy",
        "point_feature_f32.npy",
        "sequence_feature_f16.npy",
        "interaction_type_i8.npy",
        "sequence_mask.npy",
        "selected_adapter.npz",
    ):
        if not (input_dir / name).is_file():
            raise FileNotFoundError(input_dir / name)
    if not RAW_TOKEN_CACHE.is_file() or not E210_RAW.is_file():
        raise FileNotFoundError("E211/E210 raw token cache is incomplete")

    selected = np.load(input_dir / "selected_index_i32.npy", mmap_mode="r")
    coordinate = np.asarray(
        np.load(input_dir / "coordinate_f32.npy", mmap_mode="r"), dtype=np.float32
    )
    with np.load(input_dir / "selected_adapter.npz", allow_pickle=False) as adapter:
        mask = adapter["mask"].astype(bool)
    if selected.shape != (4000, CANDIDATES) or coordinate.shape != (4000, CANDIDATES, 3):
        raise ValueError("unexpected E210 selected candidate shape")
    rows = np.arange(len(selected), dtype=np.int64)[:, None]
    safe = np.maximum(np.asarray(selected, dtype=np.int64), 0)

    with np.load(RAW_TOKEN_CACHE, allow_pickle=False) as raw, np.load(
        E210_RAW, allow_pickle=False
    ) as old:
        for name in ("mask", "source", "interaction_count"):
            if not np.array_equal(raw[name], old[name]):
                raise RuntimeError(f"E211 candidate identity changed for {name}")
        for name in ("tau", "theta_t", "phi_t", "theta_r", "phi_r"):
            delta = float(np.max(np.abs(raw[name].astype(np.float64) - old[name])))
            if delta > 1e-6:
                raise RuntimeError(f"E211 candidate identity drift {name}={delta}")
        physical_mask = raw["state_mask"][rows, safe]
        physical_coordinate = raw["state_coordinate"][rows, safe]
        physical_relative_energy = raw["state_relative_energy"][rows, safe]
        physical_absolute_energy = raw["state_absolute_energy"][rows, safe]
        physical_coefficient = raw["state_coefficient"][rows, safe]

    physical_mask &= mask[..., None]
    physical_coordinate = np.where(
        physical_mask[..., None], physical_coordinate, 0.0
    ).astype(np.float32)
    physical_relative_energy = np.where(
        physical_mask, physical_relative_energy, 0.0
    ).astype(np.float32)
    physical_absolute_energy = np.where(
        physical_mask, physical_absolute_energy, 0.0
    ).astype(np.float32)
    physical_coefficient = np.where(
        physical_mask[..., None], physical_coefficient, 0.0
    ).astype(np.complex64)

    component = np.load(e114.COMPONENTS, allow_pickle=False)
    targets, assignment = e124.build_one_to_one_targets(
        coordinate,
        mask,
        component["coordinates"].astype(np.float32),
        component["coefficients"].astype(np.complex64),
        component["point_power"].astype(np.float64),
    )
    target_mass = targets["mass_absolute"].astype(np.float32)
    target_port = targets["port"].astype(np.float32)
    target_phase = targets["phase"].astype(np.complex64)
    target_energy = (
        target_mass * component["point_power"].astype(np.float64)[:, None]
    ).astype(np.float32)
    target_coefficient = (
        np.sqrt(np.maximum(target_energy[..., None] * target_port, 0.0))
        * target_phase
    ).astype(np.complex64)

    temporary = directory.with_name(directory.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    arrays = {
        "mask": mask,
        "coordinate": coordinate,
        "candidate_feature": np.asarray(
            np.load(input_dir / "candidate_feature_f16.npy", mmap_mode="r"),
            dtype=np.float16,
        ),
        "point_feature": np.asarray(
            np.load(input_dir / "point_feature_f32.npy", mmap_mode="r"),
            dtype=np.float32,
        ),
        "sequence_feature": np.asarray(
            np.load(input_dir / "sequence_feature_f16.npy", mmap_mode="r"),
            dtype=np.float16,
        ),
        "interaction": np.asarray(
            np.load(input_dir / "interaction_type_i8.npy", mmap_mode="r"),
            dtype=np.int8,
        ),
        "sequence_mask": np.asarray(
            np.load(input_dir / "sequence_mask.npy", mmap_mode="r"), dtype=bool
        ),
        "physical_mask": physical_mask,
        "physical_coordinate": physical_coordinate,
        "physical_relative_energy": physical_relative_energy,
        "physical_absolute_energy": physical_absolute_energy,
        "physical_coefficient": physical_coefficient,
        "target_positive": target_mass > 0.0,
        "target_coordinate": targets["coordinate"].astype(np.float32),
        "target_coefficient": target_coefficient,
        "target_point_power": component["point_power"].astype(np.float64),
    }
    for name, value in arrays.items():
        with (temporary / f"{name}.npy").open("wb") as handle:
            np.save(handle, value)
    state = {
        "status": "complete",
        "experiment": VERSION,
        "outer": outer,
        "candidate_identity_equal_to_e210": True,
        "physical_state_members": int(np.sum(physical_mask)),
        "physical_state_member_mean": float(np.mean(physical_mask.sum(axis=2))),
        "matched_fraction_of_available_true_mass": assignment[
            "matched_fraction_of_available_true_mass"
        ],
        "seconds": float(time.time() - started),
    }
    atomic_json(temporary / "state.json", state)
    if directory.exists():
        shutil.rmtree(directory)
    os.replace(temporary, directory)
    return {"reused": False, **state}


def _carrier_cache(
    outer: int,
    role: str,
    donor_ids: np.ndarray,
    config: Config,
    device: torch.device,
    force: bool,
) -> dict[str, Any]:
    directory = cache_dir(outer, role)
    state_path = directory / "state.json"
    required = (directory / "carrier_coefficient.npy", directory / "carrier_log_power.npy")
    digest = ids_digest(donor_ids)
    if state_path.is_file() and all(path.is_file() for path in required) and not force:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "complete" and state.get("donor_sha256") == digest:
            return {"reused": True, **state}

    started = time.time()
    setup, positions, _, site, _ = load_positions()
    coordinate = np.load(shared_dir(outer) / "coordinate.npy", mmap_mode="r")
    distances, neighbors = e150.strict_neighbors(
        positions,
        site,
        donor_ids,
        np.arange(len(positions), dtype=np.int64),
        config.carrier_neighbors,
    )
    train_h = np.load(e205.TRAIN_CHANNEL, mmap_mode="r")
    component = np.load(e114.COMPONENTS, allow_pickle=False)
    calibration = estimate_carrier(
        train_h,
        positions,
        site,
        setup.bs_positions,
        donor_ids,
        component["point_power"].astype(np.float64),
        max_pairs_per_site=500,
        seed=config.seed + outer * 101 + (0 if role == "screen" else 50_000),
    )
    templates = carrier_bank.carrier_configs()
    configs_by_site = {
        int(site_value): [
            replace(
                template,
                beta=calibration[int(site_value)].beta,
                gamma=calibration[int(site_value)].gamma,
            )
            for template in templates
        ]
        for site_value in np.unique(site)
    }

    temporary = directory.with_name(directory.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    coefficient_output = np.lib.format.open_memmap(
        temporary / "carrier_coefficient.npy",
        mode="w+",
        dtype=np.complex64,
        shape=(len(positions), CANDIDATES, CARRIERS, PORTS),
    )
    power_output = np.lib.format.open_memmap(
        temporary / "carrier_log_power.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(positions), CARRIERS),
    )
    transfer_started = time.time()
    train_device = torch.as_tensor(np.asarray(train_h), device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    transfer_seconds = float(time.time() - transfer_started)
    first_batch_seconds = 0.0
    for start in range(0, len(positions), config.carrier_batch_size):
        batch_started = time.time()
        end = min(start + config.carrier_batch_size, len(positions))
        predicted = e150.batched_carrier_predictions(
            train_device,
            neighbors[start:end],
            positions[neighbors[start:end]],
            positions[start:end],
            distances[start:end],
            setup.bs_positions,
            site[start:end],
            configs_by_site,
        )
        point_power = predicted.abs().square().sum(dim=(2, 3, 4)).clamp_min(TINY)
        field = e150.complex_angle_delay(predicted.reshape(-1, *predicted.shape[2:]))
        local_coordinate = torch.as_tensor(
            np.asarray(coordinate[start:end], dtype=np.float32), device=device
        )
        repeated = (
            local_coordinate[:, None]
            .expand(-1, CARRIERS, -1, -1)
            .reshape((end - start) * CARRIERS, CANDIDATES, 3)
        )
        sampled = e150.periodic_trilinear_sample(field, repeated)
        sampled = sampled.reshape(end - start, CARRIERS, CANDIDATES, PORTS).permute(
            0, 2, 1, 3
        )
        coefficient_output[start:end] = sampled.cpu().numpy().astype(np.complex64)
        power_output[start:end] = torch.log(point_power).cpu().numpy().astype(np.float32)
        if start == 0:
            first_batch_seconds = float(time.time() - batch_started)
        print(f"E211 outer={outer} role={role} carrier {end}/4000", flush=True)
        del predicted, field, sampled, repeated
    coefficient_output.flush()
    power_output.flush()
    del train_device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    state = {
        "status": "complete",
        "experiment": VERSION,
        "outer": outer,
        "role": role,
        "donor_count": int(len(donor_ids)),
        "donor_sha256": digest,
        "zero_target_self_neighbor": True,
        "train_h_host_to_device_seconds": transfer_seconds,
        "first_batch_seconds": first_batch_seconds,
        "seconds": float(time.time() - started),
    }
    atomic_json(temporary / "state.json", state)
    if directory.exists():
        shutil.rmtree(directory)
    os.replace(temporary, directory)
    return {"reused": False, **state}


def _fixed_cache(outer: int, device: torch.device, force: bool) -> dict[str, Any]:
    directory = shared_dir(outer)
    path = directory / "fixed_v7_coefficient.npy"
    report = directory / "fixed_state.json"
    if path.is_file() and report.is_file() and not force:
        state = json.loads(report.read_text(encoding="utf-8"))
        if state.get("status") == "complete":
            return {"reused": True, **state}
    started = time.time()
    _, query_ids, _ = e205.canonical_full_fold()
    coordinate = np.load(directory / "coordinate.npy", mmap_mode="r")
    v7 = torch.as_tensor(
        np.array(np.load(e205.BASE_V7, mmap_mode="r"), dtype=np.complex64, copy=True),
        device=device,
    )
    output = np.empty((len(query_ids), CANDIDATES, PORTS), dtype=np.complex64)
    for start in range(0, len(query_ids), 32):
        end = min(start + 32, len(query_ids))
        field = e150.complex_angle_delay(v7[start:end])
        local_coordinate = torch.as_tensor(
            np.asarray(coordinate[query_ids[start:end]], dtype=np.float32), device=device
        )
        sampled = e150.periodic_trilinear_sample(field, local_coordinate)
        output[start:end] = sampled.cpu().numpy().astype(np.complex64)
    atomic_npy(path, output)
    state = {
        "status": "complete",
        "experiment": VERSION,
        "outer": outer,
        "shape": list(output.shape),
        "seconds": float(time.time() - started),
    }
    atomic_json(report, state)
    return {"reused": False, **state}


class StateStore:
    def __init__(self, outer: int, role: str, device: torch.device) -> None:
        self.outer = outer
        self.role = role
        self.device = device
        shared = shared_dir(outer)
        cache = cache_dir(outer, role)
        self.mask = torch.as_tensor(np.load(shared / "mask.npy"), device=device)
        self.coordinate = torch.as_tensor(
            np.asarray(np.load(shared / "coordinate.npy", mmap_mode="r"), np.float32),
            device=device,
        )
        self.candidate_feature = torch.as_tensor(
            np.asarray(
                np.load(shared / "candidate_feature.npy", mmap_mode="r"), np.float32
            ),
            device=device,
        )
        self.point_feature = torch.as_tensor(
            np.asarray(np.load(shared / "point_feature.npy", mmap_mode="r"), np.float32),
            device=device,
        )
        self.sequence_feature = torch.as_tensor(
            np.asarray(
                np.load(shared / "sequence_feature.npy", mmap_mode="r"), np.float32
            ),
            device=device,
        )
        self.interaction = torch.as_tensor(
            np.asarray(np.load(shared / "interaction.npy", mmap_mode="r"), np.int64),
            device=device,
        )
        self.sequence_mask = torch.as_tensor(
            np.load(shared / "sequence_mask.npy"), device=device
        )
        self.physical_mask = torch.as_tensor(
            np.load(shared / "physical_mask.npy"), device=device
        )
        self.physical_coordinate = torch.as_tensor(
            np.asarray(
                np.load(shared / "physical_coordinate.npy", mmap_mode="r"), np.float32
            ),
            device=device,
        )
        self.physical_relative_energy = torch.as_tensor(
            np.asarray(
                np.load(shared / "physical_relative_energy.npy", mmap_mode="r"),
                np.float32,
            ),
            device=device,
        )
        self.physical_absolute_energy = torch.as_tensor(
            np.asarray(
                np.load(shared / "physical_absolute_energy.npy", mmap_mode="r"),
                np.float32,
            ),
            device=device,
        )
        self.physical_coefficient = torch.as_tensor(
            np.asarray(
                np.load(shared / "physical_coefficient.npy", mmap_mode="r"),
                np.complex64,
            ),
            device=device,
        )
        self.target_positive = torch.as_tensor(
            np.load(shared / "target_positive.npy"), device=device
        )
        self.target_coordinate = torch.as_tensor(
            np.asarray(
                np.load(shared / "target_coordinate.npy", mmap_mode="r"), np.float32
            ),
            device=device,
        )
        self.target_coefficient = torch.as_tensor(
            np.asarray(
                np.load(shared / "target_coefficient.npy", mmap_mode="r"),
                np.complex64,
            ),
            device=device,
        )
        self.target_point_power = torch.as_tensor(
            np.asarray(
                np.load(shared / "target_point_power.npy", mmap_mode="r"), np.float64
            ),
            device=device,
        )
        self.carrier_coefficient = torch.as_tensor(
            np.asarray(
                np.load(cache / "carrier_coefficient.npy", mmap_mode="r"),
                np.complex64,
            ),
            device=device,
        )
        self.carrier_log_power = torch.as_tensor(
            np.asarray(
                np.load(cache / "carrier_log_power.npy", mmap_mode="r"), np.float32
            ),
            device=device,
        )
        self.reference_index = carrier_reference_index()
        self.fixed_v7_coefficient = torch.as_tensor(
            np.asarray(
                np.load(shared / "fixed_v7_coefficient.npy", mmap_mode="r"),
                np.complex64,
            ),
            device=device,
        )
        _, self.query_ids, self.canonical_fold = e205.canonical_full_fold()
        self.v7 = torch.as_tensor(
            np.array(np.load(e205.BASE_V7, mmap_mode="r"), dtype=np.complex64, copy=True),
            device=device,
        )
        truth_all = np.load(e205.TRAIN_CHANNEL, mmap_mode="r")
        self.truth = torch.as_tensor(
            np.asarray(truth_all[self.query_ids], dtype=np.complex64), device=device
        )
        base_values = e150.score_prediction_rows(
            self.v7, self.truth, Config().metric_batch_size
        )
        self.base_pas = torch.as_tensor(base_values[0], device=device)
        self.base_pdp = torch.as_tensor(base_values[1], device=device)
        self.base_error = torch.as_tensor(base_values[2], device=device)
        self.base_energy = torch.as_tensor(base_values[3], device=device)

    def normalization(self, global_ids: np.ndarray) -> dict[str, torch.Tensor]:
        rows = torch.as_tensor(global_ids, device=self.device)
        valid = self.mask[rows]
        candidate = self.candidate_feature[rows][valid]
        point = self.point_feature[rows]
        return {
            "candidate_mean": candidate.mean(dim=0),
            "candidate_std": candidate.std(dim=0).clamp_min(1e-4),
            "point_mean": point.mean(dim=0),
            "point_std": point.std(dim=0).clamp_min(1e-4),
        }

    def _common_batch(
        self, global_ids: torch.Tensor, normalization: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        candidate = (
            self.candidate_feature[global_ids] - normalization["candidate_mean"]
        ) / normalization["candidate_std"]
        point = (self.point_feature[global_ids] - normalization["point_mean"]) / normalization[
            "point_std"
        ]
        reference = self.carrier_coefficient[global_ids, :, self.reference_index]
        return {
            "mask": self.mask[global_ids],
            "coordinate": self.coordinate[global_ids],
            "candidate_feature": candidate.clamp(-8.0, 8.0),
            "point_feature": point.clamp(-8.0, 8.0),
            "sequence_feature": self.sequence_feature[global_ids],
            "interaction": self.interaction[global_ids],
            "sequence_mask": self.sequence_mask[global_ids],
            "physical_mask": self.physical_mask[global_ids],
            "physical_coordinate": self.physical_coordinate[global_ids],
            "physical_relative_energy": self.physical_relative_energy[global_ids],
            "physical_absolute_energy": self.physical_absolute_energy[global_ids],
            "physical_coefficient": self.physical_coefficient[global_ids],
            "carrier_coefficient": self.carrier_coefficient[global_ids],
            "carrier_log_power": self.carrier_log_power[global_ids],
            "reference_coefficient": reference,
            "v7_coefficient": reference,
            "v7_log_power": self.carrier_log_power[global_ids, self.reference_index],
            "target_positive": self.target_positive[global_ids],
            "target_coordinate": self.target_coordinate[global_ids],
            "target_coefficient": self.target_coefficient[global_ids],
            "target_point_power": self.target_point_power[global_ids],
        }

    def state_batch(
        self, global_ids: torch.Tensor, normalization: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return self._common_batch(global_ids, normalization)

    def fixed_batch(
        self, fixed_rows: torch.Tensor, normalization: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        global_ids = torch.as_tensor(self.query_ids, device=self.device)[fixed_rows]
        batch = self._common_batch(global_ids, normalization)
        batch.update(
            {
                "v7_coefficient": self.fixed_v7_coefficient[fixed_rows],
                "v7": self.v7[fixed_rows],
                "truth": self.truth[fixed_rows],
                "base_pas": self.base_pas[fixed_rows],
                "base_pdp": self.base_pdp[fixed_rows],
                "base_error": self.base_error[fixed_rows],
                "base_energy": self.base_energy[fixed_rows],
            }
        )
        return batch


class FullPathStateOperator(nn.Module):
    def __init__(self, candidate_dim: int, point_dim: int, config: Config) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden
        proposal_hidden = config.carrier_hidden
        self.candidate_input = nn.Sequential(
            nn.Linear(candidate_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.point_input = nn.Sequential(
            nn.Linear(point_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.physical_input = nn.Sequential(
            nn.Linear(PHYSICAL_STATES * 12, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.sequence_input = nn.Linear(18, hidden)
        self.interaction_embedding = nn.Embedding(8, hidden)
        self.step_embedding = nn.Embedding(STEPS, hidden)
        sequence_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.heads,
            dim_feedforward=3 * hidden,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(
            sequence_layer, num_layers=config.sequence_layers
        )
        proposal_dim = 1 + PORTS + 2 * PORTS + 1
        self.proposal_input = nn.Sequential(
            nn.Linear(proposal_dim, proposal_hidden),
            nn.GELU(),
            nn.LayerNorm(proposal_hidden),
            nn.Linear(proposal_hidden, proposal_hidden),
        )
        self.proposal_embedding = nn.Parameter(
            torch.randn(PROPOSALS, proposal_hidden) * 0.02
        )
        self.proposal_query = nn.Linear(hidden, proposal_hidden)
        self.proposal_key = nn.Linear(proposal_hidden, proposal_hidden)
        self.proposal_value = nn.Linear(proposal_hidden, hidden)
        set_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.heads,
            dim_feedforward=3 * hidden,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.path_encoder = nn.TransformerEncoder(set_layer, num_layers=config.layers)
        self.output_norm = nn.LayerNorm(hidden)
        self.existence_head = nn.Linear(hidden, 1)
        self.mass_head = nn.Linear(hidden, 1)
        self.coordinate_head = nn.Linear(hidden, 3)
        self.mix_logit_head = nn.Linear(hidden, PROPOSALS)
        self.mix_phase_head = nn.Linear(hidden, PROPOSALS)
        self.port_gain_head = nn.Linear(hidden, PORTS)
        self.port_phase_head = nn.Linear(hidden, PORTS)
        self.coverage_head = nn.Linear(hidden, 1)
        self.state_power_head = nn.Linear(hidden, 1)
        self.action_head = nn.Linear(hidden, 1)
        self.power_action_head = nn.Linear(hidden, 1)
        for layer in (self.action_head, self.power_action_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _physical_feature(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        coordinate = batch["physical_coordinate"]
        mask = batch["physical_mask"].to(coordinate.dtype)
        delay = (coordinate[..., 0] * 30_000.0 * 192.0).clamp(0.0, 192.0)
        angle = coordinate[..., 1:]
        feature = torch.cat(
            (
                mask[..., None],
                (delay / 32.0)[..., None],
                torch.log(batch["physical_relative_energy"].clamp_min(1e-12))[
                    ..., None
                ].clamp(-30.0, 0.0),
                torch.log(batch["physical_absolute_energy"].clamp_min(1e-30))[
                    ..., None
                ].clamp(-80.0, 20.0),
                torch.sin(angle),
                torch.cos(angle),
            ),
            dim=-1,
        )
        feature = feature * mask[..., None]
        return feature.reshape(*feature.shape[:2], -1)

    def _sequence_state(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        continuous = batch["sequence_feature"]
        interaction = batch["interaction"].clamp(0, 7)
        token_mask = batch["sequence_mask"].clone()
        empty = ~token_mask.any(dim=2)
        token_mask[..., 0] |= empty
        steps = torch.arange(STEPS, device=continuous.device)
        hidden = (
            self.sequence_input(continuous)
            + self.interaction_embedding(interaction)
            + self.step_embedding(steps)[None, None]
        )
        batch_size, paths, _, width = hidden.shape
        flat = hidden.reshape(batch_size * paths, STEPS, width)
        flat_mask = token_mask.reshape(batch_size * paths, STEPS)
        flat = self.sequence_encoder(flat, src_key_padding_mask=~flat_mask)
        weight = flat_mask.to(flat.dtype)
        pooled = (flat * weight[..., None]).sum(dim=1) / weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        return pooled.reshape(batch_size, paths, width)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mask = batch["mask"]
        point = self.point_input(batch["point_feature"])
        candidate = (
            self.candidate_input(batch["candidate_feature"])
            + point[:, None]
            + self.physical_input(self._physical_feature(batch))
            + self._sequence_state(batch)
        )

        carrier = batch["carrier_coefficient"]
        physical = batch["physical_coefficient"]
        coefficient = torch.cat((carrier, physical), dim=2)
        carrier_energy = carrier.abs().square().sum(dim=3).clamp_min(TINY)
        carrier_fraction = torch.exp(
            torch.log(carrier_energy)
            - batch["carrier_log_power"][:, None]
        ).clamp_min(1e-20)
        proposal_fraction = torch.cat(
            (carrier_fraction, batch["physical_relative_energy"].clamp_min(1e-20)),
            dim=2,
        )
        proposal_mask = torch.cat(
            (
                carrier_energy > 1e-24,
                batch["physical_mask"],
            ),
            dim=2,
        )
        energy = coefficient.abs().square().sum(dim=3).clamp_min(TINY)
        amplitude = coefficient.abs() / torch.sqrt(energy[..., None])
        reference = batch["reference_coefficient"]
        reference_unit = reference / reference.abs().clamp_min(1e-8)
        relative = coefficient / coefficient.abs().clamp_min(1e-8)
        relative = relative * torch.conj(reference_unit[:, :, None])
        source_flag = torch.cat(
            (
                torch.zeros(CARRIERS, device=coefficient.device),
                torch.ones(PHYSICAL_STATES, device=coefficient.device),
            )
        )
        proposal_feature = torch.cat(
            (
                torch.log(proposal_fraction).clamp(-30.0, 5.0)[..., None],
                amplitude,
                relative.real,
                relative.imag,
                source_flag[None, None, :, None].expand(
                    len(coefficient), CANDIDATES, -1, -1
                ),
            ),
            dim=3,
        )
        proposal = self.proposal_input(proposal_feature)
        proposal = proposal + self.proposal_embedding[None, None]
        query = self.proposal_query(candidate)
        attention = torch.einsum(
            "bkh,bkph->bkp", query, self.proposal_key(proposal)
        ) / math.sqrt(self.config.carrier_hidden)
        attention = attention.masked_fill(~proposal_mask, -1e4)
        attention = torch.softmax(attention, dim=2)
        pooled = torch.einsum(
            "bkp,bkph->bkh", attention, self.proposal_value(proposal)
        )
        hidden = self.path_encoder(candidate + pooled, src_key_padding_mask=~mask)
        hidden = self.output_norm(hidden)

        existence_logit = self.existence_head(hidden)[..., 0]
        existence = torch.sigmoid(existence_logit)
        base_log_mass = torch.logsumexp(
            torch.where(
                proposal_mask,
                torch.log(proposal_fraction.clamp_min(1e-20)),
                torch.full_like(proposal_fraction, -torch.inf),
            ),
            dim=2,
        ) - torch.log(proposal_mask.sum(dim=2).clamp_min(1))
        mass_logits = (
            base_log_mass
            + self.config.path_gain_limit * torch.tanh(self.mass_head(hidden)[..., 0])
            + torch.log(existence.clamp_min(1e-6))
        )
        mass_logits = torch.where(mask, mass_logits, torch.full_like(mass_logits, -torch.inf))
        mass = torch.softmax(mass_logits, dim=1)
        valid = mask.to(hidden.dtype)
        pooled_hidden = (hidden * valid[..., None]).sum(dim=1) / valid.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        coverage = torch.sigmoid(self.coverage_head(pooled_hidden)[:, 0])
        state_log_power = batch["v7_log_power"] + 3.0 * torch.tanh(
            self.state_power_head(pooled_hidden)[:, 0]
        )

        mix_logit = self.mix_logit_head(hidden).masked_fill(~proposal_mask, -1e4)
        mix_phase = math.pi * torch.tanh(self.mix_phase_head(hidden))
        mix_weight = torch.softmax(mix_logit, dim=2) * torch.exp(1j * mix_phase)
        normalized = coefficient / torch.sqrt(energy[..., None])
        mixed = torch.einsum("bkp,bkpu->bku", mix_weight, normalized)
        port_gain = self.config.port_gain_limit * torch.tanh(self.port_gain_head(hidden))
        port_phase = self.config.phase_limit * torch.tanh(self.port_phase_head(hidden))
        mixed = mixed * torch.exp(port_gain + 1j * port_phase)
        mixed = mixed / torch.sqrt(
            mixed.abs().square().sum(dim=2, keepdim=True).clamp_min(TINY)
        )
        path_energy = torch.exp(state_log_power)[:, None] * coverage[:, None] * mass
        state_coefficient = mixed * torch.sqrt(path_energy.clamp_min(TINY))[..., None]
        coordinate_scale = hidden.new_tensor(self.config.coordinate_limits)
        coordinate_delta = torch.tanh(self.coordinate_head(hidden)) * coordinate_scale
        state_coordinate = batch["coordinate"] + coordinate_delta
        action = self.config.action_limit * torch.tanh(self.action_head(hidden)[..., 0])
        action = action * mask.to(action.dtype)
        power_action = self.config.power_limit * torch.tanh(
            self.power_action_head(pooled_hidden)[:, 0]
        )
        return {
            "existence_logit": existence_logit,
            "existence": existence,
            "state_coefficient": state_coefficient,
            "state_coordinate": state_coordinate,
            "coordinate_delta": coordinate_delta,
            "state_log_power": state_log_power,
            "action": action,
            "power_action": power_action,
            "coverage": coverage,
        }

    def render(
        self,
        batch: dict[str, torch.Tensor],
        output: dict[str, torch.Tensor],
        alpha: float = 1.0,
    ) -> torch.Tensor:
        scale = float(alpha) * output["action"]
        modified = e150.render_coefficients(
            output["state_coordinate"],
            output["state_coefficient"] * scale[..., None],
            batch["mask"],
        )
        base = e150.render_coefficients(
            batch["coordinate"],
            batch["v7_coefficient"] * scale[..., None],
            batch["mask"],
        )
        prediction = batch["v7"] + modified - base
        prediction = prediction * torch.exp(
            0.5 * output["power_action"][:, None, None, None]
        )
        has_physical = batch["physical_mask"].any(dim=(1, 2))
        active = batch["v7"].abs().square().sum(dim=(1, 2, 3)) > 0.0
        keep = active & has_physical
        return torch.where(keep[:, None, None, None], prediction, batch["v7"])


def make_model(store: StateStore, config: Config) -> FullPathStateOperator:
    return FullPathStateOperator(
        store.candidate_feature.shape[-1], store.point_feature.shape[-1], config
    ).to(store.device)


def train_state(
    model: FullPathStateOperator,
    store: StateStore,
    train_ids: np.ndarray,
    normalization: dict[str, torch.Tensor],
    config: Config,
    epochs: int,
    seed: int,
) -> list[dict[str, float]]:
    for parameter in model.action_head.parameters():
        parameter.requires_grad = False
    for parameter in model.power_action_head.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=store.device.type == "cuda",
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_ids)
        losses: list[float] = []
        for start in range(0, len(order), config.batch_size):
            rows = torch.as_tensor(order[start : start + config.batch_size], device=store.device)
            batch = store.state_batch(rows, normalization)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss, diagnostics = e150.state_loss(output, batch, config)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"E211 non-finite state loss epoch={epoch}")
            loss.backward()
            e150.stable_clip_gradients(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(diagnostics["state_loss"]))
        row = {"epoch": float(epoch), "state_loss": float(np.mean(losses))}
        history.append(row)
        if epoch == 1 or epoch % 2 == 0 or epoch == epochs:
            print(
                f"E211 outer={store.outer} role={store.role} state "
                f"{epoch}/{epochs} loss={row['state_loss']:.6f}",
                flush=True,
            )
    for parameter in model.action_head.parameters():
        parameter.requires_grad = True
    for parameter in model.power_action_head.parameters():
        parameter.requires_grad = True
    return history


@torch.inference_mode()
def evaluate_fixed(
    model: FullPathStateOperator | None,
    store: StateStore,
    fixed_rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    config: Config,
    collect_channel: bool,
) -> tuple[dict[str, float | int], np.ndarray | None, dict[str, float]]:
    arrays = tuple(np.empty(len(fixed_rows), dtype=np.float64) for _ in range(4))
    channels: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    if model is not None:
        model.eval()
    for start in range(0, len(fixed_rows), config.metric_batch_size):
        end = min(start + config.metric_batch_size, len(fixed_rows))
        local_np = fixed_rows[start:end]
        local = torch.as_tensor(local_np, device=store.device)
        batch = store.fixed_batch(local, normalization)
        if model is None:
            prediction = batch["v7"]
        else:
            output = model(batch)
            prediction = model.render(batch, output, 1.0)
            actions.append(output["action"][batch["mask"]].cpu().numpy())
        values = e115.torch_official_metrics(prediction, batch["truth"])
        for destination, value in zip(arrays, values, strict=True):
            destination[start:end] = value.cpu().numpy()
        if collect_channel:
            channels.append(prediction.cpu().numpy().astype(np.complex64))
    diagnostics = {
        "action_abs_mean": float(np.mean(np.abs(np.concatenate(actions))))
        if actions
        else 0.0
    }
    channel = np.concatenate(channels) if channels else None
    return statistics(*arrays), channel, diagnostics


def finetune_screen(
    model: FullPathStateOperator,
    store: StateStore,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    config: Config,
    seed: int,
) -> tuple[int, dict[str, torch.Tensor], list[dict[str, Any]]]:
    baseline, _, _ = evaluate_fixed(
        None, store, validation_rows, normalization, config, False
    )
    best_score = float(baseline["score"])
    best_epoch = 0
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.finetune_learning_rate,
        weight_decay=config.weight_decay,
        fused=store.device.type == "cuda",
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = [
        {"epoch": 0, "validation": baseline, "selected": True}
    ]
    for epoch in range(1, config.finetune_epochs + 1):
        model.train()
        order = rng.permutation(train_rows)
        losses: list[float] = []
        batch_scores: list[float] = []
        for start in range(0, len(order), config.batch_size):
            local = torch.as_tensor(order[start : start + config.batch_size], device=store.device)
            batch = store.fixed_batch(local, normalization)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            state_loss, _ = e150.state_loss(output, batch, config)
            official_loss, diagnostics = e150.official_loss(model, output, batch, config)
            loss = official_loss + 0.20 * state_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"E211 non-finite official loss epoch={epoch}")
            loss.backward()
            e150.stable_clip_gradients(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            batch_scores.append(float(diagnostics["score"]))
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_batch_score": float(np.mean(batch_scores)),
        }
        if epoch == 1 or epoch % config.validation_interval == 0 or epoch == config.finetune_epochs:
            validation, _, diagnostics = evaluate_fixed(
                model, store, validation_rows, normalization, config, False
            )
            row["validation"] = validation
            row["diagnostics"] = diagnostics
            if float(validation["score"]) > best_score + 1e-10:
                best_score = float(validation["score"])
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        history.append(row)
        if epoch == 1 or epoch % 2 == 0 or epoch == config.finetune_epochs:
            message = (
                f"E211 outer={store.outer} screen {epoch}/{config.finetune_epochs} "
                f"train={row['train_batch_score']:.6f}"
            )
            if "validation" in row:
                message += f" validation={row['validation']['score']:.6f} best={best_score:.6f}"
            print(message, flush=True)
    return best_epoch, best_state, history


def finetune_refit(
    model: FullPathStateOperator,
    store: StateStore,
    train_rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    config: Config,
    epochs: int,
    seed: int,
) -> list[dict[str, float]]:
    if epochs == 0:
        return []
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.finetune_learning_rate,
        weight_decay=config.weight_decay,
        fused=store.device.type == "cuda",
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_rows)
        losses: list[float] = []
        scores: list[float] = []
        for start in range(0, len(order), config.batch_size):
            local = torch.as_tensor(order[start : start + config.batch_size], device=store.device)
            batch = store.fixed_batch(local, normalization)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            state_loss, _ = e150.state_loss(output, batch, config)
            official_loss, diagnostics = e150.official_loss(model, output, batch, config)
            loss = official_loss + 0.20 * state_loss
            loss.backward()
            e150.stable_clip_gradients(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            scores.append(float(diagnostics["score"]))
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(losses)),
                "train_batch_score": float(np.mean(scores)),
            }
        )
        print(
            f"E211 outer={store.outer} refit {epoch}/{epochs} "
            f"score={history[-1]['train_batch_score']:.6f}",
            flush=True,
        )
    return history


def run_fold(outer: int, config: Config, device: torch.device, force: bool) -> dict[str, Any]:
    if report_path(outer).is_file() and channel_path(outer).is_file() and not force:
        report = json.loads(report_path(outer).read_text(encoding="utf-8"))
        if report.get("status") == "complete":
            return {"reused": True, **report}
    started = time.time()
    torch.manual_seed(config.seed + outer)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    shared_record = _selected_physical_state(outer, force)
    fixed_record = _fixed_cache(outer, device, force)
    screen_ids, refit_ids, validation_rows, held_rows = split_ids(outer)
    screen_cache = _carrier_cache(
        outer, "screen", screen_ids, config, device, force
    )
    screen_store = StateStore(outer, "screen", device)
    screen_normalization = screen_store.normalization(screen_ids)
    model = make_model(screen_store, config)
    screen_state_path = checkpoint_path(outer, "screen_state")
    if screen_state_path.is_file() and not force:
        screen_state = load_checkpoint(screen_state_path, device)
        if screen_state.get("outer") != outer or screen_state.get("config") != asdict(config):
            raise RuntimeError(f"stale E211 screen checkpoint for outer={outer}")
        model.load_state_dict(screen_state["state_dict"])
        state_history = screen_state["history"]
    else:
        state_history = train_state(
            model,
            screen_store,
            screen_ids,
            screen_normalization,
            config,
            config.pretrain_epochs,
            config.seed + outer * 1000,
        )
        atomic_torch(
            screen_state_path,
            {
                "outer": outer,
                "config": asdict(config),
                "state_dict": model.state_dict(),
                "normalization": {
                    key: value.detach().cpu()
                    for key, value in screen_normalization.items()
                },
                "history": state_history,
            },
        )
    canonical = screen_store.canonical_fold
    validation_fold = (outer + 1) % 5
    screen_fixed_rows = np.flatnonzero(
        (canonical != outer) & (canonical != validation_fold)
    ).astype(np.int64)
    screen_selected_path = checkpoint_path(outer, "screen_selected")
    if screen_selected_path.is_file() and not force:
        screen_selected = load_checkpoint(screen_selected_path, device)
        if (
            screen_selected.get("outer") != outer
            or screen_selected.get("config") != asdict(config)
        ):
            raise RuntimeError(f"stale E211 screen selection for outer={outer}")
        best_epoch = int(screen_selected["best_epoch"])
        best_state = screen_selected["state_dict"]
        screen_history = screen_selected["history"]
    else:
        best_epoch, best_state, screen_history = finetune_screen(
            model,
            screen_store,
            screen_fixed_rows,
            validation_rows,
            screen_normalization,
            config,
            config.seed + outer * 2000 + 17,
        )
        atomic_torch(
            screen_selected_path,
            {
                "outer": outer,
                "config": asdict(config),
                "best_epoch": best_epoch,
                "state_dict": best_state,
                "history": screen_history,
            },
        )
    del model, screen_store
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if best_epoch == 0:
        refit_cache = {"skipped": True}
        refit_state_history: list[dict[str, float]] = []
        refit_history: list[dict[str, float]] = []
        baseline_store = StateStore(outer, "screen", device)
        normalization = baseline_store.normalization(screen_ids)
        selected, channel, diagnostics = evaluate_fixed(
            None, baseline_store, held_rows, normalization, config, True
        )
        baseline = selected
        del baseline_store
    else:
        refit_cache = _carrier_cache(
            outer, "refit", refit_ids, config, device, force
        )
        refit_store = StateStore(outer, "refit", device)
        normalization = refit_store.normalization(refit_ids)
        model = make_model(refit_store, config)
        refit_state_path = checkpoint_path(outer, "refit_state")
        if refit_state_path.is_file() and not force:
            refit_state = load_checkpoint(refit_state_path, device)
            if refit_state.get("outer") != outer or refit_state.get("config") != asdict(config):
                raise RuntimeError(f"stale E211 refit checkpoint for outer={outer}")
            model.load_state_dict(refit_state["state_dict"])
            refit_state_history = refit_state["history"]
        else:
            refit_state_history = train_state(
                model,
                refit_store,
                refit_ids,
                normalization,
                config,
                config.pretrain_epochs,
                config.seed + outer * 3000,
            )
            atomic_torch(
                refit_state_path,
                {
                    "outer": outer,
                    "config": asdict(config),
                    "state_dict": model.state_dict(),
                    "normalization": {
                        key: value.detach().cpu() for key, value in normalization.items()
                    },
                    "history": refit_state_history,
                },
            )
        refit_fixed_rows = np.flatnonzero(canonical != outer).astype(np.int64)
        refit_final_path = checkpoint_path(outer, "refit_final")
        if refit_final_path.is_file() and not force:
            refit_final = load_checkpoint(refit_final_path, device)
            if (
                refit_final.get("outer") != outer
                or refit_final.get("config") != asdict(config)
                or int(refit_final.get("best_epoch", -1)) != best_epoch
            ):
                raise RuntimeError(f"stale E211 final checkpoint for outer={outer}")
            model.load_state_dict(refit_final["state_dict"])
            refit_history = refit_final["history"]
        else:
            refit_history = finetune_refit(
                model,
                refit_store,
                refit_fixed_rows,
                normalization,
                config,
                best_epoch,
                config.seed + outer * 4000 + 29,
            )
            atomic_torch(
                refit_final_path,
                {
                    "outer": outer,
                    "best_epoch": best_epoch,
                    "config": asdict(config),
                    "state_dict": model.state_dict(),
                    "normalization": {
                        key: value.detach().cpu() for key, value in normalization.items()
                    },
                    "history": refit_history,
                },
            )
        selected, channel, diagnostics = evaluate_fixed(
            model, refit_store, held_rows, normalization, config, True
        )
        baseline, _, _ = evaluate_fixed(
            None, refit_store, held_rows, normalization, config, False
        )
        del model, refit_store
    if channel is None:
        raise RuntimeError("E211 held channel was not collected")
    atomic_npy(channel_path(outer), channel.astype(np.complex64))
    _, query_ids, _ = e205.canonical_full_fold()
    atomic_npy(held_ids_path(outer), query_ids[held_rows].astype(np.int32))
    delta = {
        key: float(selected[key]) - float(baseline[key])
        for key in ("pas", "pdp", "nmse_score", "score")
    }
    report = {
        "experiment": VERSION,
        "status": "complete",
        "outer": outer,
        "validation_fold": validation_fold,
        "configuration": asdict(config),
        "split": {
            "screen_state_count": int(len(screen_ids)),
            "refit_state_count": int(len(refit_ids)),
            "screen_fixed_count": int(len(screen_fixed_rows)),
            "validation_fixed_count": int(len(validation_rows)),
            "held_fixed_count": int(len(held_rows)),
            "outer_label_overlap": 0,
        },
        "cache": {
            "shared": shared_record,
            "fixed": fixed_record,
            "screen": screen_cache,
            "refit": refit_cache,
        },
        "selection": {"best_finetune_epoch": best_epoch},
        "held": {
            "selected": selected,
            "baseline": baseline,
            "delta": delta,
            "diagnostics": diagnostics,
        },
        "history": {
            "screen_state": state_history,
            "screen_finetune": screen_history,
            "refit_state": refit_state_history,
            "refit_finetune": refit_history,
        },
        "performance": {
            "total_seconds": float(time.time() - started),
            "gpu": e205.gpu_record(device),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "measured_bottleneck": "strict carrier cache and full complex state training",
        },
    }
    atomic_json(report_path(outer), report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def aggregate(config: Config, device: torch.device) -> dict[str, Any]:
    started = time.time()
    _, query_ids, canonical = e205.canonical_full_fold()
    output = np.empty(
        (len(query_ids), e115.MP * e115.MH * e115.MV, e115.N_UE, e115.S_SC),
        dtype=np.complex64,
    )
    assigned = np.zeros(len(query_ids), dtype=bool)
    fold_reports: list[dict[str, Any]] = []
    global_to_fixed = {int(value): index for index, value in enumerate(query_ids)}
    for outer in range(5):
        report = json.loads(report_path(outer).read_text(encoding="utf-8"))
        if report.get("status") != "complete":
            raise RuntimeError(f"E211 outer{outer} is incomplete")
        ids = np.load(held_ids_path(outer)).astype(np.int64)
        rows = np.asarray([global_to_fixed[int(value)] for value in ids], dtype=np.int64)
        channel = np.load(channel_path(outer), mmap_mode="r")
        if np.any(assigned[rows]) or not np.all(canonical[rows] == outer):
            raise RuntimeError(f"E211 outer{outer} assignment mismatch")
        output[rows] = channel
        assigned[rows] = True
        fold_reports.append(
            {
                "outer": outer,
                "selected": report["held"]["selected"],
                "baseline": report["held"]["baseline"],
                "delta": report["held"]["delta"],
                "best_epoch": report["selection"]["best_finetune_epoch"],
                "performance": report["performance"],
            }
        )
    if not np.all(assigned):
        raise RuntimeError(f"E211 missed {int(np.sum(~assigned))} fixed500 rows")
    atomic_npy(FULL_CHANNEL, output)
    truth_all = np.load(e205.TRAIN_CHANNEL, mmap_mode="r")
    truth = torch.as_tensor(
        np.asarray(truth_all[query_ids], dtype=np.complex64), device=device
    )
    prediction = torch.as_tensor(output, device=device)
    selected_values = e150.score_prediction_rows(
        prediction, truth, config.metric_batch_size
    )
    selected = statistics(*selected_values)
    v7 = torch.as_tensor(
        np.array(np.load(e205.BASE_V7, mmap_mode="r"), dtype=np.complex64, copy=True),
        device=device,
    )
    baseline_values = e150.score_prediction_rows(v7, truth, config.metric_batch_size)
    baseline = statistics(*baseline_values)
    delta = {
        key: float(selected[key]) - float(baseline[key])
        for key in ("pas", "pdp", "nmse_score", "score")
    }
    positive_folds = int(sum(row["delta"]["score"] > 0.0 for row in fold_reports))
    worst_v7_delta = float(min(row["delta"]["score"] for row in fold_reports))
    component_protection = bool(
        delta["pas"] >= -0.001
        and delta["pdp"] >= -0.001
        and delta["nmse_score"] >= -0.001
    )
    success = bool(
        float(selected["score"]) >= config.required_score
        and positive_folds >= config.required_positive_folds
        and worst_v7_delta >= config.required_worst_v7_delta
        and component_protection
    )
    report = {
        "experiment": VERSION,
        "status": "fixed500_passed" if success else "fixed500_failed",
        "configuration": asdict(config),
        "selected": selected,
        "baseline_v7": baseline,
        "delta_v7": delta,
        "comparisons": {
            "e210": 0.650915445,
            "v12": 0.677521,
            "v15_local": 0.686375,
            "delta_e210": float(selected["score"]) - 0.650915445,
            "delta_v12": float(selected["score"]) - 0.677521,
            "delta_v15_local": float(selected["score"]) - 0.686375,
            "delta_target_070": float(selected["score"]) - 0.700000,
        },
        "gates": {
            "required_score": config.required_score,
            "positive_folds": positive_folds,
            "required_positive_folds": config.required_positive_folds,
            "worst_v7_delta": worst_v7_delta,
            "component_protection": component_protection,
            "success": success,
        },
        "folds": fold_reports,
        "data_lineage": {
            "official_train_h": True,
            "official_train_positions": True,
            "official_map_setup": True,
            "vendored_physical_source": True,
            "sionna_package_imported": False,
            "historical_sionna_cache_used": False,
            "historical_v8_v15_prediction_used": False,
            "physical_state": [
                "three_seed_absolute_energy",
                "three_seed_2x4_complex_reference_coefficient",
                "three_seed_angle_delay_state",
                "ordered_interaction_geometry",
            ],
            "predicted_heads": [
                "path_existence",
                "path_coordinate",
                "path_absolute_complex_coefficient",
                "point_power",
                "bounded_path_replacement",
            ],
        },
        "output": {
            "channel": PATHS.relative(FULL_CHANNEL),
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "finite": bool(np.isfinite(output).all()),
        },
        "performance": {
            "aggregate_seconds": float(time.time() - started),
            "sum_fold_seconds": float(
                sum(row["performance"]["total_seconds"] for row in fold_reports)
            ),
            "gpu": e205.gpu_record(device),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "measured_bottleneck": "five strict carrier caches and full complex state training",
        },
        "failure_conclusion_if_failed": (
            "Complete E209 complex path state through the current V7-in-loop path "
            "replacement interface does not recover V15-level official state."
        ),
    }
    atomic_json(FULL_REPORT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    args = parse_args()
    config = Config()
    device = torch.device(args.device)
    try:
        if args.stage == "fold":
            if args.outer is None:
                raise ValueError("--outer is required for fold stage")
            run_fold(args.outer, config, device, args.force)
        else:
            aggregate(config, device)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
