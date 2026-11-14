#!/usr/bin/env python3
"""Prepare Round 1 official-H supervision for the Round 2 E197 calibrator."""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

import run_path_component_transport as component_transport
import train_e190_official_sionna_path_calibrator as e190
import train_explicit_path_pdp_predictor as path_features
import train_surface_interaction_energy_model as e85
from data import load_positions
from project_paths import PATHS


VERSION = "Round2-E197-cross-round-official-prepare"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e197_cross_round_official"
R1_DIR = ARTIFACT_DIR / "round1"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e197_cross_round_official"
REPORT = ANALYSIS_DIR / "round1_prepare.json"

V475_DIR = PATHS.version_cache / "v475"
V475_SHARDS = V475_DIR / "shards"
R1_CHANNEL = PATHS.round1 / "Round1_Train_Channel.npy"
R1_POSITION = PATHS.round1 / "Round1_Train_Pos.npy"
R1_GEOMETRY = PATHS.shared_features / "height_geometry_cache.npz"

BASE = R1_DIR / "base_features_f32.npy"
FRACTION = R1_DIR / "energy_fraction_f32.npy"
PRESENCE = R1_DIR / "presence_u8.npy"
COORDINATE = R1_DIR / "coordinate_f32.npy"
PORT = R1_DIR / "port_prior_f16.npy"
SEQUENCE_GEOMETRY = R1_DIR / "sequence_geometry_f16.npy"
SEQUENCE_HASH = R1_DIR / "sequence_hash_i32.npy"
DEPTH = R1_DIR / "interaction_count_u8.npy"
INTERACTIONS = R1_DIR / "interactions_u8.npy"
OBJECTS = R1_DIR / "objects_i32.npy"
PRIMITIVES = R1_DIR / "primitives_i32.npy"
MASK = R1_DIR / "mask_bool.npy"
TARGET_PAS = R1_DIR / "official_pas_f32.npy"
TARGET_PDP = R1_DIR / "official_pdp_f32.npy"
POINT_POWER = R1_DIR / "official_point_power_f64.npy"
LABELS = R1_DIR / "hungarian_labels.npz"
COMPONENTS = R1_DIR / "components" / "train_components_k24_os2_ref2.npz"

POINTS = 2000
PATHS_PER_POINT = 128
STEPS = 16
PORTS = 8
R1_BS = np.asarray((50.0, 0.0, 25.0), dtype=np.float32)
YAW_RAD = math.radians(-29.70)
TILT_RAD = math.radians(-7.65)
HORIZONTAL_SPACING = 0.5
VERTICAL_SPACING = 4.0
DELAY_OFFSET_BINS = 0.018516518971941634
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def expected_files() -> tuple[Path, ...]:
    return (
        BASE,
        FRACTION,
        PRESENCE,
        COORDINATE,
        PORT,
        SEQUENCE_GEOMETRY,
        SEQUENCE_HASH,
        DEPTH,
        INTERACTIONS,
        OBJECTS,
        PRIMITIVES,
        MASK,
        TARGET_PAS,
        TARGET_PDP,
        POINT_POWER,
        COMPONENTS,
        LABELS,
    )


def array_is_finite(path: Path, batch_size: int = 128) -> bool:
    value = np.load(path, mmap_mode="r")
    for start in range(0, len(value), batch_size):
        if not np.isfinite(np.asarray(value[start : start + batch_size])).all():
            return False
    return True


def regime_codes() -> np.ndarray:
    blocked = np.load(R1_GEOMETRY, allow_pickle=False)["train_blocked"]
    output = np.full(len(blocked), 2, dtype=np.int8)
    output[blocked <= 0] = 0
    output[(blocked >= 1) & (blocked <= 25)] = 1
    output[(blocked > 25) & (blocked <= 100)] = 2
    output[blocked > 100] = 3
    if output.shape != (POINTS,):
        raise ValueError(f"unexpected Round1 regime shape: {output.shape}")
    return output


def load_v475_train() -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {
        "valid": np.zeros((POINTS, PATHS_PER_POINT), dtype=bool),
        "tau": np.zeros((POINTS, PATHS_PER_POINT), dtype=np.float32),
        "angles": np.zeros((POINTS, PATHS_PER_POINT, 4), dtype=np.float32),
        "log_power": np.full(
            (POINTS, PATHS_PER_POINT), -np.inf, dtype=np.float32
        ),
        "fraction": np.zeros((POINTS, PATHS_PER_POINT), dtype=np.float32),
        "depth": np.zeros((POINTS, PATHS_PER_POINT), dtype=np.uint8),
        "interactions": np.zeros(
            (POINTS, PATHS_PER_POINT, STEPS), dtype=np.uint8
        ),
        "objects": np.full(
            (POINTS, PATHS_PER_POINT, STEPS), -1, dtype=np.int32
        ),
        "vertices": np.zeros(
            (POINTS, PATHS_PER_POINT, STEPS, 3), dtype=np.float32
        ),
        "reference": np.zeros(
            (POINTS, PATHS_PER_POINT), dtype=np.complex64
        ),
        "rx": np.zeros((POINTS, PATHS_PER_POINT, 4), dtype=np.float32),
        "tx_pol": np.zeros((POINTS, PATHS_PER_POINT, 2), dtype=np.float32),
    }
    expected_id = 0
    for shard_id in range(20):
        shard = V475_SHARDS / f"shard_{shard_id:05d}"
        metadata = np.load(shard / "metadata.npz", allow_pickle=False)
        path = np.load(shard / "path_targets.npz", allow_pickle=False)
        ids = metadata["position_id"].astype(np.int64)
        expected = np.arange(expected_id, expected_id + len(ids), dtype=np.int64)
        if not np.array_equal(ids, expected):
            raise RuntimeError(f"v475 shard {shard_id} position ordering mismatch")
        rows = slice(expected_id, expected_id + len(ids))
        fields["valid"][rows] = path["valid"]
        fields["tau"][rows] = path["tau"].astype(np.float32)
        fields["angles"][rows] = np.stack(
            (
                path["theta_t"],
                path["phi_t"],
                path["theta_r"],
                path["phi_r"],
            ),
            axis=-1,
        ).astype(np.float32)
        fields["log_power"][rows] = path["log_power"].astype(np.float32)
        fields["fraction"][rows] = path["normalized_power"].astype(np.float32)
        fields["depth"][rows] = path["interaction_count"].astype(np.uint8)
        fields["interactions"][rows] = path["interactions"].astype(np.uint8)
        fields["objects"][rows] = path["objects"].astype(np.int32)
        fields["vertices"][rows] = path["vertices"].astype(np.float32)
        fields["reference"][rows] = path["complex_reference"].astype(np.complex64)
        fields["rx"][rows] = path["rx_fingerprint"].astype(np.float32)
        fields["tx_pol"][rows] = path["tx_pol_fingerprint"].astype(np.float32)
        expected_id += len(ids)
    if expected_id != POINTS:
        raise RuntimeError(f"v475 loaded {expected_id} Round1 train rows")
    for name, value in fields.items():
        if value.dtype.kind in "fc" and not np.isfinite(value[fields["valid"]]).all():
            raise ValueError(f"non-finite valid v475 field: {name}")
    return fields


def corrected_coordinates(tau: np.ndarray, angles: np.ndarray) -> np.ndarray:
    theta = angles[..., 0]
    phi = angles[..., 1]
    horizontal = (
        e85.MH
        * HORIZONTAL_SPACING
        * np.sin(theta)
        * np.sin(phi - YAW_RAD)
    ) % e85.MH
    vertical = (
        e85.MV
        * VERTICAL_SPACING
        * (
            math.sin(TILT_RAD) * np.sin(theta) * np.cos(phi - YAW_RAD)
            - math.cos(TILT_RAD) * np.cos(theta)
        )
    ) % e85.MV
    delay = (tau * 30_000.0 * e85.S_SC + DELAY_OFFSET_BINS) % e85.S_SC
    return np.stack((horizontal, vertical, delay), axis=-1).astype(np.float32)


def normalized_port(tx_pol: np.ndarray, rx: np.ndarray) -> np.ndarray:
    value = tx_pol[..., :, None] * rx[..., None, :]
    value = np.maximum(value, 0.0)
    value /= np.maximum(value.sum(axis=(2, 3), keepdims=True), TINY)
    return value.astype(np.float32)


def build_base_features(
    data: dict[str, np.ndarray],
    coordinate: np.ndarray,
    port: np.ndarray,
    aligned_positions: np.ndarray,
    aligned_vertices: np.ndarray,
    regimes: np.ndarray,
) -> np.ndarray:
    mask = data["valid"]
    tau = data["tau"]
    angles = data["angles"]
    log_power = data["log_power"]
    fraction = data["fraction"]
    count = data["depth"].astype(np.int64)
    interactions = data["interactions"]
    n_points, n_paths = mask.shape
    delay = np.mod(tau * 30_000.0 * e85.S_SC + DELAY_OFFSET_BINS, e85.S_SC)
    delay_phase = 2.0 * np.pi * delay / e85.S_SC
    source = np.where(mask, 0, -1).astype(np.int8)
    presence = np.where(mask, 1.0, 0.0).astype(np.float32)
    log10_power = np.clip(log_power / math.log(10.0), -30.0, 10.0)
    blocks = [
        (delay / e85.S_SC)[..., None].astype(np.float32),
        np.sin(delay_phase)[..., None].astype(np.float32),
        np.cos(delay_phase)[..., None].astype(np.float32),
        np.sin(2.0 * delay_phase)[..., None].astype(np.float32),
        np.cos(2.0 * delay_phase)[..., None].astype(np.float32),
        np.concatenate((np.sin(angles), np.cos(angles)), axis=-1).astype(np.float32),
        log10_power[..., None].astype(np.float32),
        np.clip(np.log10(np.maximum(fraction, 1e-12)), -12.0, 0.0)[..., None],
        (presence / 24.0)[..., None],
        (np.minimum(count, STEPS) / float(STEPS))[..., None].astype(np.float32),
        path_features.one_hot(source, (0, 1)),
    ]
    rank = np.argsort(np.argsort(-fraction, axis=1), axis=1).astype(np.float32)
    blocks.append((rank / max(n_paths - 1, 1))[..., None])
    step = np.arange(STEPS, dtype=np.int64)[None, None]
    valid_step = step < count[..., None]
    denominator = np.maximum(count, 1)[..., None].astype(np.float32)
    interaction_hist = np.stack(
        [np.sum((interactions == value) & valid_step, axis=2) for value in (1, 2, 4)],
        axis=-1,
    ).astype(np.float32)
    blocks.append(interaction_hist / denominator)
    first_interaction = interactions[..., 0]
    last_index = np.clip(count - 1, 0, STEPS - 1)[..., None]
    last_interaction = np.take_along_axis(interactions, last_index, axis=2)[..., 0]
    blocks.extend(
        (
            path_features.one_hot(first_interaction, (0, 1, 2, 4)),
            path_features.one_hot(last_interaction, (0, 1, 2, 4)),
            np.zeros((n_points, n_paths, 6), dtype=np.float32),
            np.zeros((n_points, n_paths, 6), dtype=np.float32),
        )
    )

    setup, train_positions, _, train_site, _ = load_positions()
    center = train_positions[:, :2].mean(axis=0)
    scale = train_positions[:, :2].std(axis=0) + 1e-6
    train_relative = train_positions[:, :2] - setup.bs_positions[train_site, :2]
    relative_scale = np.std(train_relative, axis=0) + 1e-6
    distance_scale = np.median(np.linalg.norm(train_relative, axis=1)) + 1e-6
    relative = aligned_positions[:, :2] - setup.bs_positions[0, :2]
    distance = np.linalg.norm(relative, axis=1, keepdims=True)
    bearing = np.arctan2(relative[:, 1], relative[:, 0])[:, None]
    sites = np.zeros(n_points, dtype=np.int8)
    point = np.concatenate(
        (
            (aligned_positions[:, :2] - center) / scale,
            relative / relative_scale,
            distance / distance_scale,
            np.sin(bearing),
            np.cos(bearing),
            path_features.one_hot(sites, (0, 1)),
            path_features.one_hot(regimes, (0, 1, 2, 3)),
        ),
        axis=1,
    ).astype(np.float32)
    blocks.append(np.broadcast_to(point[:, None], (n_points, n_paths, point.shape[1])))

    first_vertex = aligned_vertices[..., 0, :2]
    gather_index = np.repeat(last_index[..., None], 2, axis=3)
    last_vertex = np.take_along_axis(
        aligned_vertices[..., :2], gather_index, axis=2
    )[..., 0, :]
    position_xy = aligned_positions[:, None, :2]
    vertex_scale = np.std(train_positions[:, :2], axis=0) + 1e-6
    blocks.extend(
        (
            ((first_vertex - position_xy) / vertex_scale).astype(np.float32),
            ((last_vertex - position_xy) / vertex_scale).astype(np.float32),
            ((last_vertex - first_vertex) / vertex_scale).astype(np.float32),
        )
    )
    segment = np.diff(aligned_vertices[..., :2], axis=2)
    segment_valid = np.arange(STEPS - 1)[None, None] < np.maximum(
        count - 1, 0
    )[..., None]
    route_length = np.sum(np.linalg.norm(segment, axis=-1) * segment_valid, axis=2)
    blocks.append(np.log1p(route_length)[..., None].astype(np.float32))

    scalar = np.concatenate(blocks, axis=-1).astype(np.float32)
    if scalar.shape[-1] != 63:
        raise ValueError(f"unexpected Round1 scalar feature width: {scalar.shape}")
    total_log_power = log10_power[..., None].astype(np.float32)
    port_flat = port.reshape(n_points, n_paths, PORTS)
    unavailable_phase = np.zeros((n_points, n_paths, 16), dtype=np.float32)
    coefficient = np.concatenate(
        (
            total_log_power,
            port_flat,
            unavailable_phase,
            unavailable_phase,
            unavailable_phase,
        ),
        axis=-1,
    ).astype(np.float32)
    if coefficient.shape[-1] != 57:
        raise ValueError(f"unexpected Round1 coefficient width: {coefficient.shape}")
    output = np.concatenate((scalar, coefficient), axis=-1).astype(np.float32)
    output[~mask] = 0.0
    if output.shape != (POINTS, PATHS_PER_POINT, 120):
        raise ValueError(f"unexpected Round1 base feature shape: {output.shape}")
    if not np.isfinite(output).all():
        raise ValueError("Round1 base features contain non-finite values")
    return output


def prepare_tokens(force: bool) -> dict[str, Any]:
    token_files = (
        BASE,
        FRACTION,
        PRESENCE,
        COORDINATE,
        PORT,
        SEQUENCE_GEOMETRY,
        SEQUENCE_HASH,
        DEPTH,
        INTERACTIONS,
        OBJECTS,
        PRIMITIVES,
        MASK,
    )
    if (
        not force
        and all(path.is_file() for path in token_files)
        and array_is_finite(BASE)
        and array_is_finite(SEQUENCE_GEOMETRY)
    ):
        return {
            "status": "reused",
            "base_shape": list(np.load(BASE, mmap_mode="r").shape),
            "sequence_shape": list(np.load(SEQUENCE_GEOMETRY, mmap_mode="r").shape),
        }
    started = time.time()
    data = load_v475_train()
    setup, train_positions, _, _, _ = load_positions()
    positions = np.load(R1_POSITION).astype(np.float32)
    shift = setup.bs_positions[0].astype(np.float32) - R1_BS
    aligned_positions = positions + shift
    step_valid = (
        np.arange(STEPS, dtype=np.int64)[None, None]
        < data["depth"].astype(np.int64)[..., None]
    )
    aligned_vertices = np.where(
        step_valid[..., None],
        np.nan_to_num(
            data["vertices"] + shift[None, None, None],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
    ).astype(np.float32)
    coordinate = corrected_coordinates(data["tau"], data["angles"])
    coordinate[~data["valid"]] = 0.0
    port = normalized_port(data["tx_pol"], data["rx"])
    regimes = regime_codes()
    base = build_base_features(
        data, coordinate, port, aligned_positions, aligned_vertices, regimes
    )
    center = np.concatenate(
        (train_positions[:, :2].mean(axis=0), np.asarray([1.5]))
    ).astype(np.float32)
    scale = np.concatenate(
        (np.maximum(train_positions[:, :2].std(axis=0), 1.0), np.asarray([20.0]))
    ).astype(np.float32)
    sites = np.zeros(POINTS, dtype=np.int8)
    sequence = e85.sequence_geometry_block(
        slice(0, POINTS),
        aligned_vertices,
        data["depth"],
        aligned_positions,
        sites,
        setup.bs_positions,
        center,
        scale,
    ).astype(np.float16)
    if sequence.shape[-1] != np.load(e85.SEQUENCE_GEOMETRY, mmap_mode="r").shape[-1]:
        raise ValueError(f"Round1/Round2 sequence width mismatch: {sequence.shape}")
    presence = np.where(data["valid"], 1, 0).astype(np.uint8)
    primitives = np.zeros_like(data["objects"], dtype=np.int32)
    sequence_hash = np.zeros((POINTS, PATHS_PER_POINT, 2), dtype=np.int32)
    for path, value in (
        (BASE, base),
        (FRACTION, data["fraction"].astype(np.float32)),
        (PRESENCE, presence),
        (COORDINATE, coordinate),
        (PORT, port.astype(np.float16)),
        (SEQUENCE_GEOMETRY, sequence),
        (SEQUENCE_HASH, sequence_hash),
        (DEPTH, data["depth"]),
        (INTERACTIONS, data["interactions"]),
        (OBJECTS, data["objects"]),
        (PRIMITIVES, primitives),
        (MASK, data["valid"]),
    ):
        atomic_save_npy(path, value)
    return {
        "status": "created",
        "valid_paths": int(data["valid"].sum()),
        "mean_paths": float(data["valid"].sum(axis=1).mean()),
        "base_shape": list(base.shape),
        "sequence_shape": list(sequence.shape),
        "coordinate_range": {
            "minimum": coordinate[data["valid"]].min(axis=0).tolist(),
            "maximum": coordinate[data["valid"]].max(axis=0).tolist(),
        },
        "bs_alignment_shift_m": shift.tolist(),
        "seconds": float(time.time() - started),
    }


def prepare_targets(device: torch.device, batch_size: int, force: bool) -> dict[str, Any]:
    if not force and all(
        path.is_file() for path in (TARGET_PAS, TARGET_PDP, POINT_POWER)
    ):
        return {
            "status": "reused",
            "pas_shape": list(np.load(TARGET_PAS, mmap_mode="r").shape),
            "pdp_shape": list(np.load(TARGET_PDP, mmap_mode="r").shape),
        }
    started = time.time()
    channel = np.load(R1_CHANNEL, mmap_mode="r")
    pas = np.empty((POINTS, e85.MP, e85.MH, e85.MV, e85.N_UE), dtype=np.float32)
    pdp = np.empty((POINTS, e85.MP, e85.N_UE, e85.S_SC), dtype=np.float32)
    power = np.empty(POINTS, dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, POINTS, batch_size):
            end = min(start + batch_size, POINTS)
            value = torch.as_tensor(
                np.array(channel[start:end], copy=True), device=device
            )
            shaped = value.reshape(
                -1, e85.MP, e85.MH, e85.MV, e85.N_UE, e85.S_SC
            )
            pas_power = torch.fft.fft2(shaped, dim=(2, 3)).abs().square().float()
            pas_direction = e85.normalized_direction(pas_power, (1, 2, 3)).mean(dim=5)
            delay_power = torch.fft.ifft(shaped, dim=5).abs().square().float()
            delay_direction = e85.normalized_direction(delay_power, (5,)).mean(
                dim=(2, 3)
            )
            pas[start:end] = pas_direction.cpu().numpy().astype(np.float32)
            pdp[start:end] = delay_direction.cpu().numpy().astype(np.float32)
            power[start:end] = (
                value.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64).cpu().numpy()
            )
            print(f"E197 Round1 official targets {end}/{POINTS}", flush=True)
    atomic_save_npy(TARGET_PAS, pas)
    atomic_save_npy(TARGET_PDP, pdp)
    atomic_save_npy(POINT_POWER, power)
    return {
        "status": "created",
        "pas_shape": list(pas.shape),
        "pdp_shape": list(pdp.shape),
        "nonzero_points": int(np.sum(power > 0.0)),
        "seconds": float(time.time() - started),
    }


def prepare_components(
    device: torch.device, batch_size: int, force: bool
) -> dict[str, Any]:
    if COMPONENTS.is_file() and not force:
        cached = np.load(COMPONENTS, allow_pickle=False)
        return {
            "status": "reused",
            "coordinate_shape": list(cached["coordinates"].shape),
            "coefficient_shape": list(cached["coefficients"].shape),
        }
    started = time.time()
    final_dir = COMPONENTS.parent
    build_dir = R1_DIR / f".components_build_{os.getpid()}"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    old_channel = component_transport.TRAIN_CHANNEL
    old_artifact = component_transport.ARTIFACT_DIR
    try:
        component_transport.TRAIN_CHANNEL = R1_CHANNEL
        component_transport.ARTIFACT_DIR = build_dir
        config = component_transport.Config(
            components=24, oversample=2, refinement_sweeps=2
        )
        raw = component_transport.extract_components(config, device, batch_size)
        refined = component_transport.refine_components(raw, config, device, batch_size)
        source_raw = component_transport.component_cache_path(config)
        source_refined = component_transport.refined_component_cache_path(config)
        final_dir.mkdir(parents=True, exist_ok=True)
        os.replace(source_raw, final_dir / source_raw.name)
        os.replace(source_refined, COMPONENTS)
    finally:
        component_transport.TRAIN_CHANNEL = old_channel
        component_transport.ARTIFACT_DIR = old_artifact
        shutil.rmtree(build_dir, ignore_errors=True)
    return {
        "status": "created",
        "coordinate_shape": list(refined.coordinates.shape),
        "coefficient_shape": list(refined.coefficients.shape),
        "median_explained_fraction": float(np.median(1.0 - refined.residual_fraction)),
        "seconds": float(time.time() - started),
    }


def prepare_labels(force: bool) -> dict[str, Any]:
    if LABELS.is_file() and not force:
        cached = np.load(LABELS, allow_pickle=False)
        report = json.loads(str(cached["report_json"].item()))
        return {"status": "reused", **report}
    started = time.time()
    component = np.load(COMPONENTS, allow_pickle=False)
    coordinate = np.load(COORDINATE, mmap_mode="r")
    mask = np.load(MASK, mmap_mode="r")
    fraction = np.asarray(np.load(FRACTION, mmap_mode="r"), dtype=np.float32)
    fallback_mass = e190.normalize_rows(fraction)
    fallback_port = np.asarray(np.load(PORT, mmap_mode="r"), dtype=np.float32).reshape(
        POINTS, PATHS_PER_POINT, PORTS
    )
    labels, report = e190.build_labels(
        np.asarray(coordinate, dtype=np.float32),
        np.asarray(mask, dtype=bool),
        component["coordinates"].astype(np.float32),
        component["coefficients"].astype(np.complex64),
        component["point_power"].astype(np.float64),
        fallback_mass,
        fallback_port,
        e190.Config(),
    )
    atomic_save_npz(
        LABELS,
        mass=labels.mass,
        delta=labels.delta,
        port=labels.port,
        matched_count=labels.matched_count,
        matched_fraction=labels.matched_fraction,
        valid=labels.valid,
        report_json=np.asarray(json.dumps(report, ensure_ascii=False)),
    )
    return {"status": "created", **report, "seconds_total": time.time() - started}


def validate_outputs() -> dict[str, Any]:
    missing = [PATHS.relative(path) for path in expected_files() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing E197 Round1 caches: {missing}")
    shapes = {
        "base": list(np.load(BASE, mmap_mode="r").shape),
        "fraction": list(np.load(FRACTION, mmap_mode="r").shape),
        "coordinate": list(np.load(COORDINATE, mmap_mode="r").shape),
        "port": list(np.load(PORT, mmap_mode="r").shape),
        "sequence_geometry": list(np.load(SEQUENCE_GEOMETRY, mmap_mode="r").shape),
        "target_pas": list(np.load(TARGET_PAS, mmap_mode="r").shape),
        "target_pdp": list(np.load(TARGET_PDP, mmap_mode="r").shape),
    }
    if shapes["base"] != [POINTS, PATHS_PER_POINT, 120]:
        raise ValueError(f"invalid E197 Round1 base shape: {shapes['base']}")
    finite_paths = (BASE, FRACTION, COORDINATE, PORT, SEQUENCE_GEOMETRY, TARGET_PAS, TARGET_PDP)
    for path in finite_paths:
        value = np.load(path, mmap_mode="r")
        for start in range(0, len(value), 128):
            if not np.isfinite(np.asarray(value[start : start + 128])).all():
                raise ValueError(f"non-finite E197 cache: {PATHS.relative(path)}")
    labels = np.load(LABELS, allow_pickle=False)
    label_valid = labels["valid"].astype(bool)
    mass_error = float(
        np.max(np.abs(labels["mass"].sum(axis=1)[label_valid] - 1.0))
    )
    mask = np.load(MASK, mmap_mode="r")
    port_sum = labels["port"].sum(axis=2)
    port_error = float(np.max(np.abs(port_sum[mask] - 1.0)))
    if mass_error > 2e-5 or port_error > 1e-3:
        raise ValueError(f"E197 label normalization failed: {mass_error}/{port_error}")
    return {
        "shapes": shapes,
        "label_mass_max_error": mass_error,
        "label_port_max_error": port_error,
        "all_finite": True,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    R1_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    missing_inputs = [
        PATHS.relative(path)
        for path in (V475_SHARDS, R1_CHANNEL, R1_POSITION, R1_GEOMETRY)
        if not path.exists()
    ]
    if missing_inputs:
        raise FileNotFoundError(f"missing E197 Round1 inputs: {missing_inputs}")
    report: dict[str, Any] = {
        "experiment": VERSION,
        "status": "running",
        "device": str(device),
        "stages": {},
    }
    atomic_json(REPORT, report)
    try:
        for name, function in (
            ("tokens", lambda: prepare_tokens(args.force)),
            ("official_targets", lambda: prepare_targets(device, args.batch_size, args.force)),
            ("components", lambda: prepare_components(device, args.batch_size, args.force)),
            ("labels", lambda: prepare_labels(args.force)),
        ):
            report["stages"][name] = function()
            atomic_json(REPORT, report)
        report["validation"] = validate_outputs()
        report["status"] = "complete"
        report["performance"] = {
            "device": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu",
            "peak_torch_memory_gib": float(
                torch.cuda.max_memory_allocated(device) / 2**30
            )
            if device.type == "cuda"
            else 0.0,
            "peak_rss_gib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            ),
        }
        report["seconds"] = float(time.time() - started)
        timing = {
            name: float(value.get("seconds", value.get("seconds_total", 0.0)))
            for name, value in report["stages"].items()
        }
        report["performance"]["measured_bottleneck"] = max(timing, key=timing.get)
        atomic_json(REPORT, report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        report["seconds"] = float(time.time() - started)
        atomic_json(REPORT, report)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
