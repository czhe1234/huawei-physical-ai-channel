#!/usr/bin/env python3
"""Measure the joint PAS/PDP ceiling of explicit multiseed Sionna paths."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import time

import h5py
import numpy as np
import torch

from analyze_v4_exact_joint_projection import (
    pas_distribution,
    pdp_distribution,
    truth_views,
)
from prepare_sionna_calibration_dataset import ARTIFACT_DIR as E40_DIR
from project_paths import PATHS
from round2.data import load_positions
from run_rbf_spectral_field import OOF_META, statistics
from run_v8_full_sionna_migration import OOF_V7, reconstruct_v7_nested
from run_v9_joint_sionna_migration import Candidate, apply_candidate, point_metrics
from train_v9_joint_sionna_endpoints import output_path as joint_output_path


VERSION = "Round2-E60-joint-explicit-path-token-oracle"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "joint_path_token_oracle"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_joint_path_token_oracle"
REPORT = ANALYSIS_DIR / "joint_path_token_oracle.json"
SIONNA_ROOT = PATHS.artifacts / "round2" / "sionna_multiseed"
BASE_NAME = "primary_d16_base_v4lambda_1m"
SCATTER_NAME = "primary_d10_scatter015_v4lambda_1m"
BASE_PATH_ROOT = SIONNA_ROOT / BASE_NAME / "full" / "paths"
SCATTER_PATH_ROOT = SIONNA_ROOT / SCATTER_NAME / "full" / "paths"
OLD_PAS = E40_DIR / "oof_pas_mean_f16.npy"
OLD_PDP = E40_DIR / "oof_pdp_mean_f16.npy"
JOINT_PAS = joint_output_path("pas", "oof")
JOINT_PDP = joint_output_path("pdp", "oof")
REGIME_CACHE = (
    PATHS.artifacts
    / "round2"
    / "regime_lifecycle_support"
    / "regime_assignments.npz"
)

N_SUBCARRIERS = 192
SUBCARRIER_SPACING = 30_000.0
FREQUENCIES = (
    np.arange(-N_SUBCARRIERS // 2, N_SUBCARRIERS // 2, dtype=np.float32)
    * SUBCARRIER_SPACING
)
TINY = 1e-30
SEED_PATTERN = re.compile(r"seed(\d+)\.h5$")


@dataclass
class PathRecord:
    tau: float
    theta_t: float
    phi_t: float
    theta_r: float
    phi_r: float
    coefficient: np.ndarray
    interactions: np.ndarray
    objects: np.ndarray
    primitives: np.ndarray
    vertices: np.ndarray
    interaction_count: int
    energy: float
    energy_fraction: float
    source: int
    seed: int
    presence: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--max-points", type=int, default=100)
    parser.add_argument("--seed-count", type=int, default=24)
    parser.add_argument("--max-paths", type=int, default=128)
    parser.add_argument("--base-reserve", type=int, default=48)
    parser.add_argument("--base-per-seed", type=int, default=96)
    parser.add_argument("--scatter-per-seed", type=int, default=64)
    parser.add_argument("--scatter-pool", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--include-scatter", action="store_true")
    parser.add_argument("--reuse-path-cache", action="store_true")
    parser.add_argument("--selection-seed", type=int, default=106)
    return parser.parse_args()


def seed_value(path: Path) -> int:
    match = SEED_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"cannot parse seed from {path}")
    return int(match.group(1))


def path_files(root: Path, count: int, split: str = "train") -> list[Path]:
    if split not in ("train", "test"):
        raise ValueError(f"unsupported path split: {split}")
    paths = sorted(root.glob(f"{split}_seed*.h5"), key=seed_value)
    if len(paths) < count:
        raise FileNotFoundError(f"expected {count} path files in {root}, found {len(paths)}")
    return paths[:count]


def normalized_cosine_torch(
    prediction: torch.Tensor, truth: torch.Tensor, domain: str
) -> torch.Tensor:
    if domain == "pas":
        axes = (1, 2, 3)
        average = (1, 2)
    elif domain == "pdp":
        axes = (3,)
        average = (1, 2)
    else:
        raise ValueError(domain)
    numerator = (prediction * truth).sum(dim=axes, dtype=torch.float64)
    denominator = torch.sqrt(
        prediction.square().sum(dim=axes, dtype=torch.float64)
        * truth.square().sum(dim=axes, dtype=torch.float64)
    )
    cosine = numerator / torch.where(denominator == 0.0, 1e-12, denominator)
    return cosine.mean(dim=average, dtype=torch.float64)


def path_coordinates(tau: float, theta_t: float, phi_t: float, config: dict, site: int) -> tuple[float, float, float]:
    yaw = math.radians(float(config["site_yaw_deg"][site]))
    tilt = math.radians(float(config["site_tilt_deg"][site]))
    horizontal_spacing = float(config["tx_horizontal_spacing"])
    vertical_spacing = float(config["tx_vertical_spacing"])
    horizontal_cycles = (
        horizontal_spacing * math.sin(theta_t) * math.sin(phi_t - yaw)
    )
    vertical_cycles = vertical_spacing * (
        math.sin(tilt) * math.sin(theta_t) * math.cos(phi_t - yaw)
        - math.cos(tilt) * math.cos(theta_t)
    )
    delay = (tau * SUBCARRIER_SPACING * N_SUBCARRIERS) % N_SUBCARRIERS
    horizontal = (16.0 * horizontal_cycles) % 16.0
    vertical = (8.0 * vertical_cycles) % 8.0
    return delay, horizontal, vertical


def expand_reference_coefficient(
    reference: np.ndarray,
    theta_t: np.ndarray,
    phi_t: np.ndarray,
    config: dict,
    site: int,
) -> np.ndarray:
    yaw = math.radians(float(config["site_yaw_deg"][site]))
    tilt = math.radians(float(config["site_tilt_deg"][site]))
    horizontal_spacing = float(config["tx_horizontal_spacing"])
    vertical_spacing = float(config["tx_vertical_spacing"])
    horizontal_phase = 2.0 * np.pi * horizontal_spacing * np.sin(theta_t) * np.sin(
        phi_t - yaw
    )
    vertical_phase = 2.0 * np.pi * vertical_spacing * (
        np.sin(tilt) * np.sin(theta_t) * np.cos(phi_t - yaw)
        - np.cos(tilt) * np.cos(theta_t)
    )
    horizontal = np.arange(16, dtype=np.float32)
    vertical = np.arange(8, dtype=np.float32)
    steering = np.exp(
        1j
        * (
            horizontal_phase[:, None, None] * horizontal[None, :, None]
            + vertical_phase[:, None, None] * vertical[None, None, :]
        )
    ).astype(np.complex64)
    return (
        reference[:, :, None, None, :] * steering[:, None, :, :, None]
    ).reshape(len(reference), 256, 4).astype(np.complex64)


def score_align_coefficient(coefficient: np.ndarray) -> np.ndarray:
    """Apply the fixed +2 vertical PAS-bin alignment established by E40."""

    vertical = np.arange(8, dtype=np.float32)
    ramp = np.exp(2j * np.pi * 2.0 * vertical / 8.0).astype(np.complex64)
    shaped = np.asarray(coefficient, dtype=np.complex64).reshape(-1, 2, 16, 8, 4)
    return (shaped * ramp[None, None, None, :, None]).reshape(-1, 256, 4)


def base_identity(record: PathRecord, config: dict, site: int) -> tuple[object, ...]:
    delay, horizontal, vertical = path_coordinates(
        record.tau, record.theta_t, record.phi_t, config, site
    )
    depth = record.interaction_count
    sequence = tuple(
        (
            int(record.interactions[index]),
            int(record.objects[index]),
            int(record.primitives[index]),
        )
        for index in range(depth)
    )
    return (
        int(round(delay * 50.0)),
        int(round(horizontal * 50.0)),
        int(round(vertical * 50.0)),
        sequence,
    )


def scatter_identity(record: PathRecord, config: dict, site: int) -> tuple[int, ...]:
    delay, horizontal, vertical = path_coordinates(
        record.tau, record.theta_t, record.phi_t, config, site
    )
    port_power = np.sum(np.abs(record.coefficient.reshape(2, 16, 8, 4)) ** 2, axis=(1, 2))
    port_class = int(np.argmax(port_power.reshape(-1)))
    return (
        int(round(delay * 4.0)),
        int(round(horizontal * 4.0)),
        int(round(vertical * 4.0)),
        min(record.interaction_count, 15),
        port_class,
    )


def selected_rows(
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    regime: np.ndarray,
    truth_power: np.ndarray,
    maximum: int,
    seed: int,
) -> np.ndarray:
    nonzero_rows = np.flatnonzero(truth_power > 0.0)
    if maximum <= 0 or maximum >= len(nonzero_rows):
        return nonzero_rows.astype(np.int64)
    rng = np.random.default_rng(seed)
    groups: list[np.ndarray] = []
    quota = max(1, maximum // 20)
    for fold in range(5):
        for code in range(4):
            rows = np.flatnonzero(
                (fold_id == fold) & (regime == code) & (truth_power > 0.0)
            )
            if len(rows):
                groups.append(
                    np.sort(rng.choice(rows, size=min(quota, len(rows)), replace=False))
                )
    chosen = np.unique(np.concatenate(groups)) if groups else np.empty(0, dtype=np.int64)
    if len(chosen) < maximum:
        remaining = np.setdiff1d(nonzero_rows, chosen, assume_unique=False)
        add = rng.choice(remaining, size=min(maximum - len(chosen), len(remaining)), replace=False)
        chosen = np.unique(np.concatenate([chosen, add]))
    if len(chosen) > maximum:
        chosen = np.sort(rng.choice(chosen, size=maximum, replace=False))
    return np.sort(chosen).astype(np.int64)


def read_records(
    path: Path,
    point_to_row: dict[int, int],
    per_seed: int,
    source: int,
) -> tuple[dict[int, list[PathRecord]], dict]:
    seed = seed_value(path)
    output: dict[int, list[PathRecord]] = {}
    with h5py.File(path, "r") as handle:
        config = json.loads(str(handle.attrs["configuration"]))
        coefficient_mode = str(handle.attrs["coefficient_mode"])
        for site_name in handle:
            site = int(site_name.replace("site", ""))
            site_group = handle[site_name]
            for batch in site_group.values():
                ids = np.asarray(batch["point_ids"], dtype=np.int64)
                offsets = np.asarray(batch["offsets"], dtype=np.int64)
                for local_row, point_id_value in enumerate(ids):
                    point_id = int(point_id_value)
                    sample_row = point_to_row.get(point_id)
                    if sample_row is None:
                        continue
                    start, end = int(offsets[local_row]), int(offsets[local_row + 1])
                    if end <= start:
                        output[sample_row] = []
                        continue
                    fraction = np.asarray(batch["path_energy_fraction"][start:end], dtype=np.float64)
                    count = min(per_seed, len(fraction))
                    selected = np.argpartition(fraction, len(fraction) - count)[-count:]
                    indices = np.sort(start + selected.astype(np.int64))
                    tau = np.asarray(batch["tau"][indices], dtype=np.float32)
                    theta_t = np.asarray(batch["theta_t"][indices], dtype=np.float32)
                    phi_t = np.asarray(batch["phi_t"][indices], dtype=np.float32)
                    theta_r = np.asarray(batch["theta_r"][indices], dtype=np.float32)
                    phi_r = np.asarray(batch["phi_r"][indices], dtype=np.float32)
                    if coefficient_mode == "full":
                        coefficient = np.asarray(
                            batch["full_coefficient"][indices], dtype=np.complex64
                        )
                    else:
                        coefficient = expand_reference_coefficient(
                            np.asarray(batch["reference_coefficient"][indices], dtype=np.complex64),
                            theta_t,
                            phi_t,
                            config,
                            site,
                        )
                    coefficient = score_align_coefficient(coefficient)
                    energy = np.asarray(batch["path_energy"][indices], dtype=np.float32)
                    selected_fraction = np.asarray(
                        batch["path_energy_fraction"][indices], dtype=np.float32
                    )
                    interactions = np.asarray(batch["interactions"][indices], dtype=np.uint8)
                    objects = np.asarray(batch["objects"][indices], dtype=np.uint32)
                    primitives = np.asarray(batch["primitives"][indices], dtype=np.uint32)
                    vertices = np.asarray(batch["vertices"][indices], dtype=np.float32)
                    interaction_count = np.asarray(
                        batch["interaction_count"][indices], dtype=np.uint8
                    )
                    records = []
                    for index in range(count):
                        records.append(
                            PathRecord(
                                tau=float(tau[index]),
                                theta_t=float(theta_t[index]),
                                phi_t=float(phi_t[index]),
                                theta_r=float(theta_r[index]),
                                phi_r=float(phi_r[index]),
                                coefficient=coefficient[index],
                                interactions=interactions[index],
                                objects=objects[index],
                                primitives=primitives[index],
                                vertices=vertices[index],
                                interaction_count=int(interaction_count[index]),
                                energy=float(energy[index]),
                                energy_fraction=float(selected_fraction[index]),
                                source=source,
                                seed=seed,
                            )
                        )
                    output[sample_row] = records
    return output, config


def prune_records(records: list[PathRecord], maximum: int) -> list[PathRecord]:
    if len(records) <= maximum:
        return records
    order = np.argsort(
        np.asarray([record.energy_fraction for record in records], dtype=np.float64)
    )[-maximum:]
    return [records[int(index)] for index in order]


def cache_paths(
    args: argparse.Namespace,
    rows: np.ndarray,
    query_ids: np.ndarray,
    site: np.ndarray,
    split: str = "train",
) -> dict[str, np.ndarray]:
    include_base = bool(getattr(args, "include_base", True))
    use_presence = bool(getattr(args, "use_presence", True))
    prefix = "" if split == "train" else f"{split}_"
    tag = (
        f"{prefix}n{len(rows):03d}_s{args.seed_count:02d}_k{args.max_paths:03d}_"
        f"b{args.base_per_seed:03d}_d{args.scatter_per_seed:03d}_"
        f"scatter{int(args.include_scatter)}_sel{args.selection_seed}"
    )
    if not include_base:
        tag += "_base0"
    if not use_presence:
        tag += "_presence0"
    cache_dir = ARTIFACT_DIR / tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_files = {
        "coefficient": cache_dir / "coefficient.npy",
        "tau": cache_dir / "tau.npy",
        "mask": cache_dir / "mask.npy",
        "source": cache_dir / "source.npy",
        "seed": cache_dir / "seed.npy",
        "angles": cache_dir / "angles.npy",
        "energy": cache_dir / "energy.npy",
        "energy_fraction": cache_dir / "energy_fraction.npy",
        "presence": cache_dir / "presence.npy",
        "interaction_count": cache_dir / "interaction_count.npy",
        "interactions": cache_dir / "interactions.npy",
        "objects": cache_dir / "objects.npy",
        "primitives": cache_dir / "primitives.npy",
        "vertices": cache_dir / "vertices.npy",
        "rows": cache_dir / "rows.npy",
    }
    if args.reuse_path_cache and all(path.is_file() for path in cache_files.values()):
        return {name: np.load(path, mmap_mode="r") for name, path in cache_files.items()}

    point_to_row = {int(query_ids[row]): index for index, row in enumerate(rows)}
    base_bank: list[dict[tuple[object, ...], PathRecord]] = [dict() for _ in rows]
    scatter_bank: list[list[PathRecord]] = [[] for _ in rows]
    if include_base:
        for file_index, path in enumerate(
            path_files(BASE_PATH_ROOT, args.seed_count, split)
        ):
            records, config = read_records(
                path, point_to_row, args.base_per_seed, source=0
            )
            for sample_row, local_records in records.items():
                local_site = int(site[query_ids[rows[sample_row]]])
                bank = base_bank[sample_row]
                for record in local_records:
                    key = base_identity(record, config, local_site)
                    previous = bank.get(key)
                    if previous is None:
                        bank[key] = record
                        continue
                    if use_presence:
                        previous.presence += 1
                    if record.energy_fraction > previous.energy_fraction:
                        record.presence = previous.presence
                        bank[key] = record
            print(
                f"path oracle base files {file_index + 1}/{args.seed_count}",
                flush=True,
            )

    if args.include_scatter:
        for file_index, path in enumerate(
            path_files(SCATTER_PATH_ROOT, args.seed_count, split)
        ):
            records, config = read_records(
                path, point_to_row, args.scatter_per_seed, source=1
            )
            for sample_row, local_records in records.items():
                local_site = int(site[query_ids[rows[sample_row]]])
                local_best: dict[tuple[int, ...], PathRecord] = {}
                for record in local_records:
                    key = scatter_identity(record, config, local_site)
                    previous = local_best.get(key)
                    if previous is None or record.energy_fraction > previous.energy_fraction:
                        local_best[key] = record
                scatter_bank[sample_row].extend(local_best.values())
                scatter_bank[sample_row] = prune_records(
                    scatter_bank[sample_row], args.scatter_pool
                )
            print(
                f"path oracle scatter files {file_index + 1}/{args.seed_count}",
                flush=True,
            )

    shape = (len(rows), args.max_paths)
    arrays: dict[str, np.ndarray] = {
        "coefficient": np.zeros(shape + (256, 4), dtype=np.complex64),
        "tau": np.zeros(shape, dtype=np.float32),
        "mask": np.zeros(shape, dtype=bool),
        "source": np.full(shape, -1, dtype=np.int8),
        "seed": np.full(shape, -1, dtype=np.int32),
        "angles": np.zeros(shape + (4,), dtype=np.float32),
        "energy": np.zeros(shape, dtype=np.float32),
        "energy_fraction": np.zeros(shape, dtype=np.float32),
        "presence": np.zeros(shape, dtype=np.uint8),
        "interaction_count": np.zeros(shape, dtype=np.uint8),
        "interactions": np.zeros(shape + (16,), dtype=np.uint8),
        "objects": np.zeros(shape + (16,), dtype=np.uint32),
        "primitives": np.zeros(shape + (16,), dtype=np.uint32),
        "vertices": np.zeros(shape + (16, 3), dtype=np.float32),
        "rows": rows.astype(np.int64),
    }
    for sample_row in range(len(rows)):
        base_records = list(base_bank[sample_row].values())
        base_records.sort(
            key=lambda record: record.energy_fraction
            * (1.0 + 0.05 * (record.presence - 1) if use_presence else 1.0),
            reverse=True,
        )
        base_take = min(args.base_reserve, args.max_paths, len(base_records))
        selected = base_records[:base_take]
        remaining = args.max_paths - len(selected)
        if remaining > 0 and args.include_scatter:
            scatter_records = sorted(
                scatter_bank[sample_row],
                key=lambda record: record.energy_fraction,
                reverse=True,
            )
            selected.extend(scatter_records[:remaining])
        if len(selected) < args.max_paths:
            selected.extend(base_records[base_take : base_take + args.max_paths - len(selected)])
        for slot, record in enumerate(selected[: args.max_paths]):
            arrays["coefficient"][sample_row, slot] = record.coefficient
            arrays["tau"][sample_row, slot] = record.tau
            arrays["mask"][sample_row, slot] = True
            arrays["source"][sample_row, slot] = record.source
            arrays["seed"][sample_row, slot] = record.seed
            arrays["angles"][sample_row, slot] = (
                record.theta_t,
                record.phi_t,
                record.theta_r,
                record.phi_r,
            )
            arrays["energy"][sample_row, slot] = record.energy
            arrays["energy_fraction"][sample_row, slot] = record.energy_fraction
            arrays["presence"][sample_row, slot] = min(record.presence, 255)
            arrays["interaction_count"][sample_row, slot] = record.interaction_count
            depth = min(record.interaction_count, 16)
            arrays["interactions"][sample_row, slot, :depth] = record.interactions[:depth]
            arrays["objects"][sample_row, slot, :depth] = record.objects[:depth]
            arrays["primitives"][sample_row, slot, :depth] = record.primitives[:depth]
            arrays["vertices"][sample_row, slot, :depth] = record.vertices[:depth]
    for name, path in cache_files.items():
        np.save(path, arrays[name])
    return {name: np.load(path, mmap_mode="r") for name, path in cache_files.items()}


def v9_channel(
    rows: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    v7 = reconstruct_v7_nested(device, max(batch_size, 1), True)
    old_pas = np.load(OLD_PAS, mmap_mode="r")
    old_pdp = np.load(OLD_PDP, mmap_mode="r")
    joint_pas = np.load(JOINT_PAS, mmap_mode="r")
    joint_pdp = np.load(JOINT_PDP, mmap_mode="r")
    output = np.empty((len(rows), 256, 4, 192), dtype=np.complex64)
    spec = Candidate(
        name="v9",
        pas_beta=0.75,
        pdp_beta=0.50,
        alpha_pas=0.45,
        alpha_pdp=0.35,
        cycles=4,
    )
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        local = rows[start:end]
        with torch.inference_mode():
            prediction = apply_candidate(
                torch.as_tensor(np.asarray(v7[local]), device=device),
                torch.as_tensor(np.asarray(old_pas[local], dtype=np.float32), device=device),
                torch.as_tensor(np.asarray(joint_pas[local], dtype=np.float32), device=device),
                torch.as_tensor(np.asarray(old_pdp[local], dtype=np.float32), device=device),
                torch.as_tensor(np.asarray(joint_pdp[local], dtype=np.float32), device=device),
                spec,
            )
        output[start:end] = prediction.cpu().numpy().astype(np.complex64)
    return output


def optimize_path_amplitudes(
    coefficients: np.ndarray,
    tau: np.ndarray,
    mask: np.ndarray,
    truth: np.ndarray,
    device: torch.device,
    steps: int,
    learning_rate: float,
    pas_weight: float = 0.5,
    pdp_weight: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if pas_weight < 0.0 or pdp_weight < 0.0 or pas_weight + pdp_weight <= 0.0:
        raise ValueError("PAS/PDP objective weights must be nonnegative with positive sum")
    weight_sum = pas_weight + pdp_weight
    pas_weight /= weight_sum
    pdp_weight /= weight_sum
    coefficient = torch.as_tensor(np.array(coefficients, copy=True), device=device)
    tau_tensor = torch.as_tensor(np.array(tau, copy=True), device=device, dtype=torch.float32)
    mask_tensor = torch.as_tensor(np.array(mask, copy=True), device=device, dtype=torch.float32)
    truth_tensor = torch.as_tensor(np.array(truth, copy=True), device=device)
    truth_pas, truth_pdp = truth_views(truth_tensor)
    frequency = torch.as_tensor(FREQUENCIES, device=device, dtype=torch.float32)
    phase = torch.exp(-2j * torch.pi * tau_tensor[:, :, None] * frequency[None, None])
    raw = torch.zeros(mask.shape, device=device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    best_score = torch.full((len(truth),), -1.0, device=device, dtype=torch.float64)
    best_raw = raw.detach().clone()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        amplitude = torch.exp(torch.clamp(raw, min=-8.0, max=8.0)) * mask_tensor
        prediction = torch.einsum(
            "bk,bkmu,bks->bmus", amplitude.to(coefficient.dtype), coefficient, phase
        )
        prediction_pas = pas_distribution(prediction)
        prediction_pdp = pdp_distribution(prediction)
        pas = normalized_cosine_torch(prediction_pas, truth_pas, "pas")
        pdp = normalized_cosine_torch(prediction_pdp, truth_pdp, "pdp")
        point_score = pas_weight * pas + pdp_weight * pdp
        loss = -point_score.mean() + 1e-6 * (raw.square() * mask_tensor).sum() / mask_tensor.sum().clamp_min(1.0)
        current_raw = raw.detach().clone()
        loss.backward()
        optimizer.step()
        improved = point_score.detach() > best_score
        best_score = torch.where(improved, point_score.detach(), best_score)
        best_raw[improved] = current_raw[improved]
        if (step + 1) % 25 == 0 or step + 1 == steps:
            print(
                f"path amplitude step {step + 1}/{steps} "
                f"objective={float(best_score.mean()):.6f} "
                f"weights={pas_weight:.2f}/{pdp_weight:.2f}",
                flush=True,
            )
    amplitude = torch.exp(torch.clamp(best_raw, min=-8.0, max=8.0)) * mask_tensor
    prediction = torch.einsum(
        "bk,bkmu,bks->bmus", amplitude.to(coefficient.dtype), coefficient, phase
    )
    truth_power = truth_tensor.abs().square().sum(dim=(1, 2, 3))
    prediction_power = prediction.abs().square().sum(dim=(1, 2, 3))
    prediction = prediction * torch.sqrt(
        truth_power / prediction_power.clamp_min(TINY)
    )[:, None, None, None]
    return (
        prediction.detach().cpu().numpy().astype(np.complex64),
        pas_distribution(prediction).detach().cpu().numpy().astype(np.float32),
        pdp_distribution(prediction).detach().cpu().numpy().astype(np.float32),
        amplitude.detach().cpu().numpy().astype(np.float32),
    )


def evaluate_projection_grid(
    base: np.ndarray,
    truth: np.ndarray,
    path_pas: np.ndarray,
    path_pdp: np.ndarray,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    alpha_pairs = (
        (0.35, 0.35),
        (0.50, 0.35),
        (0.50, 0.50),
        (0.75, 0.50),
        (0.75, 0.75),
        (1.00, 0.75),
        (1.00, 1.00),
    )
    base_tensor = torch.as_tensor(base, device=device)
    truth_tensor = torch.as_tensor(truth, device=device)
    path_pas_tensor = torch.as_tensor(path_pas, device=device)
    path_pdp_tensor = torch.as_tensor(path_pdp, device=device)
    truth_pas, truth_pdp = truth_views(truth_tensor)
    result = {
        key: np.empty((len(alpha_pairs) + 2, len(base)), dtype=np.float64)
        for key in ("pas", "pdp", "error", "energy")
    }
    names = ["v9_identity"]
    values = point_metrics(base_tensor, truth_tensor, truth_pas, truth_pdp)
    for key, value in zip(result, values):
        result[key][0] = value
    for index, (alpha_pas, alpha_pdp) in enumerate(alpha_pairs, start=1):
        spec = Candidate(
            name=f"path_ap{alpha_pas:.2f}_ad{alpha_pdp:.2f}",
            alpha_pas=alpha_pas,
            alpha_pdp=alpha_pdp,
            cycles=4,
        )
        with torch.inference_mode():
            prediction = apply_candidate(
                base_tensor,
                path_pas_tensor,
                path_pas_tensor,
                path_pdp_tensor,
                path_pdp_tensor,
                spec,
            )
            values = point_metrics(prediction, truth_tensor, truth_pas, truth_pdp)
        for key, value in zip(result, values):
            result[key][index] = value
        names.append(spec.name)
    truth_spec = Candidate(
        name="truth_endpoint_oracle", alpha_pas=1.0, alpha_pdp=1.0, cycles=4
    )
    with torch.inference_mode():
        prediction = apply_candidate(
            base_tensor,
            truth_pas,
            truth_pas,
            truth_pdp,
            truth_pdp,
            truth_spec,
        )
        values = point_metrics(prediction, truth_tensor, truth_pas, truth_pdp)
    for key, value in zip(result, values):
        result[key][-1] = value
    names.append(truth_spec.name)
    rows = []
    for index, name in enumerate(names):
        rows.append(
            {
                "candidate": name,
                **statistics(
                    result["pas"][index],
                    result["pdp"][index],
                    result["error"][index],
                    result["energy"][index],
                ),
            }
        )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    result["candidate_names"] = np.asarray(names, dtype="U40")
    return rows, result


def grouped_results(
    metrics: dict[str, np.ndarray],
    fold_id: np.ndarray,
    regime: np.ndarray,
) -> dict[str, object]:
    names = metrics["candidate_names"].astype(str)
    output: dict[str, object] = {}
    for index, name in enumerate(names):
        row: dict[str, object] = {
            "overall": statistics(
                metrics["pas"][index],
                metrics["pdp"][index],
                metrics["error"][index],
                metrics["energy"][index],
            ),
            "folds": [],
            "regimes": {},
        }
        for fold in range(5):
            row["folds"].append(
                statistics(
                    metrics["pas"][index],
                    metrics["pdp"][index],
                    metrics["error"][index],
                    metrics["energy"][index],
                    fold_id == fold,
                )
            )
        for code, label in enumerate(("clear", "light", "medium", "heavy")):
            row["regimes"][label] = statistics(
                metrics["pas"][index],
                metrics["pdp"][index],
                metrics["error"][index],
                metrics["energy"][index],
                regime == code,
            )
        output[name] = row
    return output


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    required = (OOF_META, REGIME_CACHE, OLD_PAS, OLD_PDP, JOINT_PAS, JOINT_PDP)
    missing = [PATHS.relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    regime = np.load(REGIME_CACHE, allow_pickle=False)["oof_regime"].astype(np.int8)
    train_h = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    truth_power = np.sum(
        np.abs(train_h[query_ids]) ** 2, axis=(1, 2, 3), dtype=np.float64
    )
    rows = selected_rows(
        query_ids,
        fold_id,
        regime,
        truth_power,
        args.max_points,
        args.selection_seed,
    )
    _, _, _, train_site, _ = load_positions()
    paths = cache_paths(args, rows, query_ids, train_site)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    truth = np.asarray(train_h[query_ids[rows]], dtype=np.complex64)
    base = v9_channel(rows, device, args.batch_size)

    path_channel_blocks = []
    path_pas_blocks = []
    path_pdp_blocks = []
    amplitude_blocks = []
    for start in range(0, len(rows), args.batch_size):
        end = min(start + args.batch_size, len(rows))
        values = optimize_path_amplitudes(
            np.asarray(paths["coefficient"][start:end]),
            np.asarray(paths["tau"][start:end]),
            np.asarray(paths["mask"][start:end]),
            truth[start:end],
            device,
            args.steps,
            args.learning_rate,
        )
        path_channel_blocks.append(values[0])
        path_pas_blocks.append(values[1])
        path_pdp_blocks.append(values[2])
        amplitude_blocks.append(values[3])
        print(f"path oracle optimized {end}/{len(rows)}", flush=True)
    path_channel = np.concatenate(path_channel_blocks)
    path_pas = np.concatenate(path_pas_blocks)
    path_pdp = np.concatenate(path_pdp_blocks)
    amplitudes = np.concatenate(amplitude_blocks)

    projection_rows, projection_metrics = evaluate_projection_grid(
        base, truth, path_pas, path_pdp, device
    )
    truth_tensor = torch.as_tensor(truth, device=device)
    truth_pas, truth_pdp = truth_views(truth_tensor)
    path_values = point_metrics(
        torch.as_tensor(path_channel, device=device), truth_tensor, truth_pas, truth_pdp
    )
    path_direct = statistics(*path_values)
    local_fold = fold_id[rows]
    local_regime = regime[rows]
    grouped = grouped_results(projection_metrics, local_fold, local_regime)
    baseline = grouped["v9_identity"]["overall"]
    fixed_path_rows = [
        row for row in projection_rows if row["candidate"].startswith("path_")
    ]
    selected = fixed_path_rows[0]
    selected_folds = grouped[str(selected["candidate"])]["folds"]
    baseline_folds = grouped["v9_identity"]["folds"]
    fold_deltas = [
        float(current["score"] - anchor["score"])
        for current, anchor in zip(selected_folds, baseline_folds)
    ]
    passed = bool(
        float(selected["score"]) >= 0.70
        and sum(delta > 0.0 for delta in fold_deltas) >= 4
    )
    tag = (
        f"n{len(rows):03d}_s{args.seed_count:02d}_k{args.max_paths:03d}_"
        f"b{args.base_per_seed:03d}_d{args.scatter_per_seed:03d}_"
        f"scatter{int(args.include_scatter)}_sel{args.selection_seed}"
    )
    metrics_path = ARTIFACT_DIR / f"oracle_metrics_{tag}.npz"
    np.savez_compressed(
        metrics_path,
        rows=rows,
        query_ids=query_ids[rows],
        fold_id=local_fold,
        regime=local_regime,
        path_count=np.sum(np.asarray(paths["mask"]), axis=1),
        amplitudes=amplitudes.astype(np.float16),
        **projection_metrics,
    )
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "PAS and PDP can be generated by one shared set of explicit propagation paths; "
            "only nonnegative per-path amplitudes are reallocated while path delay, full "
            "array/port coefficient, interaction sequence, and coherent phase stay fixed."
        ),
        "configuration": vars(args),
        "point_count": int(len(rows)),
        "query_rows": rows.tolist(),
        "query_ids": query_ids[rows].tolist(),
        "path_count": {
            "mean": float(np.mean(np.sum(np.asarray(paths["mask"]), axis=1))),
            "median": float(np.median(np.sum(np.asarray(paths["mask"]), axis=1))),
            "min": int(np.min(np.sum(np.asarray(paths["mask"]), axis=1))),
            "max": int(np.max(np.sum(np.asarray(paths["mask"]), axis=1))),
        },
        "v9_baseline": baseline,
        "path_only_channel": path_direct,
        "projection_grid": projection_rows,
        "grouped": grouped,
        "selected_fixed_path_projection": selected,
        "selected_fold_deltas_vs_v9": fold_deltas,
        "gate": {
            "required_score": 0.70,
            "required_positive_folds": 4,
            "passed": passed,
            "decision": (
                "Train path existence/energy model and reserve V10 if the full 500-point "
                "run also passes; otherwise reject this path library."
                if passed
                else "Do not train a path model from this candidate library."
            ),
        },
        "artifacts": {
            "metrics": PATHS.relative(metrics_path),
            "path_cache": PATHS.relative(ARTIFACT_DIR / tag),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
