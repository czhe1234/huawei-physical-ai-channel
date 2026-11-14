"""Build the exact deployment features consumed by the path-state calibrator."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .config import PipelineConfig, RuntimePaths
from .data import load_official_data
from .geometry import build_geometry_regime
from .io_utils import atomic_json, atomic_npy, atomic_npz
from .path_tokens import cache_dir


SEQUENCE_HASH_BUCKETS = 32749
TINY = 1e-30


def _one_hot(values: np.ndarray, classes: tuple[int, ...]) -> np.ndarray:
    return np.stack([values == value for value in classes], axis=-1).astype(np.float32)


def _angular_hash(values: np.ndarray, moduli: tuple[int, ...]) -> np.ndarray:
    blocks: list[np.ndarray] = []
    values = values.astype(np.float64)
    for modulus in moduli:
        phase = 2.0 * np.pi * np.mod(values, modulus) / float(modulus)
        blocks.extend((np.sin(phase)[..., None], np.cos(phase)[..., None]))
    return np.concatenate(blocks, axis=-1).astype(np.float32)


def _unit_complex_features(values: np.ndarray) -> np.ndarray:
    magnitude = np.abs(values)
    unit = np.divide(
        values, magnitude, out=np.ones_like(values), where=magnitude > 1e-20
    )
    return np.concatenate((unit.real, unit.imag), axis=-1).astype(np.float32)


def sequence_hashes(arrays: dict[str, np.ndarray]) -> np.ndarray:
    source = np.asarray(arrays["source"], dtype=np.int64)
    depth = np.asarray(arrays["interaction_count"], dtype=np.int64)
    interactions = np.asarray(arrays["interactions"], dtype=np.uint64)
    objects = np.asarray(arrays["objects"], dtype=np.uint64)
    primitives = np.asarray(arrays["primitives"], dtype=np.uint64)
    result = np.full(source.shape, np.uint64(1469598103934665603), dtype=np.uint64)
    result ^= np.asarray(source + 2, dtype=np.uint64)
    result *= np.uint64(1099511628211)
    result ^= np.asarray(depth, dtype=np.uint64)
    result *= np.uint64(1099511628211)
    for step in range(interactions.shape[2]):
        active = step < depth
        value = (
            interactions[..., step]
            ^ (objects[..., step] * np.uint64(0x9E3779B185EBCA87))
            ^ (primitives[..., step] * np.uint64(0xC2B2AE3D27D4EB4F))
            ^ np.uint64(step + 1)
        )
        updated = (result ^ value) * np.uint64(1099511628211)
        result = np.where(active, updated, result)
    return result


def initial_coordinates(
    tau: np.ndarray,
    angles: np.ndarray,
    sites: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    yaw_values = np.radians(np.asarray((-44.0, 73.5), dtype=np.float32))
    tilt_values = np.radians(np.asarray((-11.5, -10.5), dtype=np.float32))
    yaw = yaw_values[sites][:, None]
    tilt = tilt_values[sites][:, None]
    theta = np.asarray(angles[..., 0], dtype=np.float32)
    phi = np.asarray(angles[..., 1], dtype=np.float32)
    horizontal = (
        config.tx_horizontal * 0.5 * np.sin(theta) * np.sin(phi - yaw)
    ) % config.tx_horizontal
    vertical = (
        config.tx_vertical
        * 4.0
        * (
            np.sin(tilt) * np.sin(theta) * np.cos(phi - yaw)
            - np.cos(tilt) * np.cos(theta)
        )
        + 2.0
    ) % config.tx_vertical
    delay = (
        np.asarray(tau, dtype=np.float32)
        * config.subcarrier_spacing_hz
        * config.subcarriers
    ) % config.subcarriers
    return np.stack((horizontal, vertical, delay), axis=2).astype(np.float32)


def sequence_geometry(
    vertices: np.ndarray,
    depth: np.ndarray,
    positions: np.ndarray,
    sites: np.ndarray,
    bs_positions: np.ndarray,
    train_positions: np.ndarray,
    steps: int,
) -> np.ndarray:
    local_vertices = np.array(vertices, dtype=np.float32, copy=True)
    local_depth = np.asarray(depth, dtype=np.int64)
    point = positions[:, None, None, :].astype(np.float32)
    bs = bs_positions[sites][:, None, None, :].astype(np.float32)
    step = np.arange(steps, dtype=np.int64)[None, None, :]
    actual = step < local_depth[..., None]
    token_mask = step < np.maximum(local_depth, 1)[..., None]
    center = np.concatenate(
        (train_positions[:, :2].mean(axis=0), np.asarray([1.5], dtype=np.float32))
    ).astype(np.float32)
    scale = np.concatenate(
        (
            np.maximum(train_positions[:, :2].std(axis=0), 1.0),
            np.asarray([20.0], dtype=np.float32),
        )
    ).astype(np.float32)
    midpoint = 0.5 * (point + bs)
    local_vertices = np.where(
        ((local_depth == 0)[..., None, None] & (step == 0)[..., None]),
        midpoint,
        local_vertices,
    )
    previous = np.empty_like(local_vertices)
    previous[..., 0, :] = bs[..., 0, :]
    previous[..., 1:, :] = local_vertices[..., :-1, :]
    following = np.empty_like(local_vertices)
    following[..., :-1, :] = local_vertices[..., 1:, :]
    following[..., -1, :] = point[..., 0, :]
    last = step == np.maximum(local_depth - 1, 0)[..., None]
    following = np.where(last[..., None], point, following)
    incoming = local_vertices - previous
    outgoing = following - local_vertices
    incoming_length = np.linalg.norm(incoming, axis=-1, keepdims=True)
    outgoing_length = np.linalg.norm(outgoing, axis=-1, keepdims=True)
    incoming_unit = incoming / np.maximum(incoming_length, 1e-6)
    outgoing_unit = outgoing / np.maximum(outgoing_length, 1e-6)
    turn = np.sum(incoming_unit * outgoing_unit, axis=-1, keepdims=True)
    normalized_vertex = (local_vertices - center) / scale
    relative_ue = (local_vertices - point) / scale
    relative_bs = (local_vertices - bs) / scale
    step_fraction = np.broadcast_to(step / float(steps - 1), actual.shape)[..., None]
    depth_fraction = np.broadcast_to(
        local_depth[..., None, None] / float(steps),
        (*local_depth.shape, steps, 1),
    )
    output = np.concatenate(
        (
            normalized_vertex,
            relative_ue,
            relative_bs,
            incoming_unit,
            np.log1p(incoming_length) / 5.0,
            outgoing_unit,
            np.log1p(outgoing_length) / 5.0,
            turn,
            step_fraction.astype(np.float32),
            depth_fraction.astype(np.float32),
            actual[..., None].astype(np.float32),
        ),
        axis=-1,
    ).astype(np.float32)
    output *= token_mask[..., None]
    return output


def _load_tokens(paths: RuntimePaths, split: str) -> dict[str, np.ndarray]:
    root = cache_dir(paths, split)
    names = (
        "mask",
        "tau",
        "angles",
        "coefficient",
        "energy",
        "energy_fraction",
        "presence",
        "source",
        "interaction_count",
        "interactions",
        "objects",
        "primitives",
        "vertices",
    )
    missing = [name for name in names if not (root / f"{name}.npy").is_file()]
    if missing:
        raise FileNotFoundError(f"missing {split} path token fields: {missing}")
    return {name: np.load(root / f"{name}.npy", mmap_mode="r") for name in names}


def _path_features(
    tokens: dict[str, np.ndarray],
    positions: np.ndarray,
    sites: np.ndarray,
    train_positions: np.ndarray,
    train_sites: np.ndarray,
    bs_positions: np.ndarray,
    regime: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    mask = np.asarray(tokens["mask"])
    tau = np.asarray(tokens["tau"], dtype=np.float32)
    angles = np.asarray(tokens["angles"], dtype=np.float32)
    energy = np.asarray(tokens["energy"], dtype=np.float32)
    fraction = np.asarray(tokens["energy_fraction"], dtype=np.float32)
    presence = np.asarray(tokens["presence"], dtype=np.float32)
    source = np.asarray(tokens["source"], dtype=np.int8)
    count = np.asarray(tokens["interaction_count"], dtype=np.int64)
    interactions = np.asarray(tokens["interactions"], dtype=np.uint8)
    primitives = np.asarray(tokens["primitives"], dtype=np.uint32)
    vertices = np.asarray(tokens["vertices"], dtype=np.float32)
    coefficient = tokens["coefficient"]
    n_points, n_paths = mask.shape
    delay = np.mod(tau * config.subcarrier_spacing_hz * config.subcarriers, 192.0)
    delay_phase = 2.0 * np.pi * delay / 192.0
    blocks = [
        (delay / 192.0)[..., None].astype(np.float32),
        np.sin(delay_phase)[..., None].astype(np.float32),
        np.cos(delay_phase)[..., None].astype(np.float32),
        np.sin(2.0 * delay_phase)[..., None].astype(np.float32),
        np.cos(2.0 * delay_phase)[..., None].astype(np.float32),
        np.concatenate((np.sin(angles), np.cos(angles)), axis=-1).astype(np.float32),
        np.clip(np.log10(np.maximum(energy, 1e-30)), -30.0, 10.0)[..., None],
        np.clip(np.log10(np.maximum(fraction, 1e-12)), -12.0, 0.0)[..., None],
        np.minimum(presence, 24.0)[..., None] / 24.0,
        np.minimum(count, 16)[..., None].astype(np.float32) / 16.0,
        _one_hot(source, (0, 1)),
    ]
    rank = np.argsort(np.argsort(-fraction, axis=1), axis=1).astype(np.float32)
    blocks.append((rank / max(n_paths - 1, 1))[..., None])
    step = np.arange(16, dtype=np.int64)[None, None, :]
    valid_step = step < count[..., None]
    denominator = np.maximum(count, 1)[..., None].astype(np.float32)
    histogram = np.stack(
        [np.sum((interactions == value) & valid_step, axis=2) for value in (1, 2, 4)],
        axis=-1,
    ).astype(np.float32)
    blocks.append(histogram / denominator)
    first_interaction = interactions[..., 0]
    last_index = np.clip(count - 1, 0, 15)[..., None]
    last_interaction = np.take_along_axis(interactions, last_index, axis=2)[..., 0]
    blocks.extend(
        (
            _one_hot(first_interaction, (0, 1, 2, 4)),
            _one_hot(last_interaction, (0, 1, 2, 4)),
        )
    )
    first_primitive = primitives[..., 0]
    last_primitive = np.take_along_axis(primitives, last_index, axis=2)[..., 0]
    blocks.extend(
        (
            _angular_hash(first_primitive, (97, 389, 1543)),
            _angular_hash(last_primitive, (97, 389, 1543)),
        )
    )
    bs = bs_positions[sites, :2]
    relative = positions[:, :2] - bs
    distance = np.linalg.norm(relative, axis=1, keepdims=True)
    bearing = np.arctan2(relative[:, 1], relative[:, 0])[:, None]
    center = train_positions[:, :2].mean(axis=0)
    scale = train_positions[:, :2].std(axis=0) + 1e-6
    train_relative = train_positions[:, :2] - bs_positions[train_sites, :2]
    relative_scale = np.std(train_relative, axis=0) + 1e-6
    distance_scale = np.median(np.linalg.norm(train_relative, axis=1)) + 1e-6
    point = np.concatenate(
        (
            (positions[:, :2] - center) / scale,
            relative / relative_scale,
            distance / distance_scale,
            np.sin(bearing),
            np.cos(bearing),
            _one_hot(sites, (0, 1)),
            _one_hot(regime, (0, 1, 2, 3)),
        ),
        axis=1,
    ).astype(np.float32)
    blocks.append(np.broadcast_to(point[:, None], (n_points, n_paths, point.shape[1])))
    first_vertex = vertices[..., 0, :2]
    gather = np.repeat(last_index[..., None], 2, axis=3)
    last_vertex = np.take_along_axis(vertices[..., :2], gather, axis=2)[..., 0, :]
    position_xy = positions[:, None, :2]
    vertex_scale = np.std(train_positions[:, :2], axis=0) + 1e-6
    blocks.extend(
        (
            ((first_vertex - position_xy) / vertex_scale).astype(np.float32),
            ((last_vertex - position_xy) / vertex_scale).astype(np.float32),
            ((last_vertex - first_vertex) / vertex_scale).astype(np.float32),
        )
    )
    segment = np.diff(vertices[..., :2], axis=2)
    segment_valid = np.arange(15)[None, None] < np.maximum(count - 1, 0)[..., None]
    route_length = np.sum(np.linalg.norm(segment, axis=-1) * segment_valid, axis=2)
    blocks.append(np.log1p(route_length)[..., None].astype(np.float32))
    scalar_dim = sum(block.shape[-1] for block in blocks)
    coefficient_dim = 57
    output = np.empty(
        (n_points, n_paths, scalar_dim + coefficient_dim), dtype=np.float32
    )
    output[..., :scalar_dim] = np.concatenate(blocks, axis=-1)
    for start in range(0, n_points, 16):
        end = min(start + 16, n_points)
        shaped = np.asarray(coefficient[start:end]).reshape(
            end - start, n_paths, 2, 16, 8, 4
        )
        power = np.abs(shaped) ** 2
        port_power = power.sum(axis=(3, 4))
        port_fraction = port_power / np.maximum(
            port_power.sum(axis=(2, 3), keepdims=True), TINY
        )
        total_log_power = np.log10(np.maximum(power.sum(axis=(2, 3, 4, 5)), TINY))[
            ..., None
        ]
        reference = shaped[..., 0, 0, :].reshape(end - start, n_paths, 8)
        horizontal = np.sum(
            np.conj(shaped[..., :-1, :, :]) * shaped[..., 1:, :, :], axis=(3, 4)
        ).reshape(end - start, n_paths, 8)
        vertical = np.sum(
            np.conj(shaped[..., :, :-1, :]) * shaped[..., :, 1:, :], axis=(3, 4)
        ).reshape(end - start, n_paths, 8)
        output[start:end, :, scalar_dim:] = np.concatenate(
            (
                total_log_power.astype(np.float32),
                port_fraction.reshape(end - start, n_paths, 8).astype(np.float32),
                _unit_complex_features(reference),
                _unit_complex_features(horizontal),
                _unit_complex_features(vertical),
            ),
            axis=-1,
        )
    output[~mask] = 0.0
    if output.shape[-1] != 120:
        raise RuntimeError(f"path feature contract changed: {output.shape}")
    return output


def build_model_inputs(
    paths: RuntimePaths,
    config: PipelineConfig,
    split: str,
    force: bool = False,
) -> dict[str, Any]:
    output_dir = paths.work_dir / "model_inputs" / split
    feature_path = output_dir / "features.npy"
    auxiliary_path = output_dir / "auxiliary.npz"
    if feature_path.is_file() and auxiliary_path.is_file() and not force:
        features = np.load(feature_path, mmap_mode="r")
        return {
            "stage": "model_inputs",
            "status": "reused",
            "split": split,
            "feature_shape": list(features.shape),
        }
    started = time.time()
    data = load_official_data(paths.data_dir)
    geometry = build_geometry_regime(paths, force=False)
    positions = data.train_positions if split == "train" else data.test_positions
    sites = data.train_sites if split == "train" else data.test_sites
    regime = geometry["train_regime"] if split == "train" else geometry["test_regime"]
    tokens = _load_tokens(paths, split)
    features = _path_features(
        tokens,
        positions,
        sites,
        data.train_positions,
        data.train_sites,
        data.setup.bs_positions,
        regime,
        config,
    )
    coordinate = initial_coordinates(tokens["tau"], tokens["angles"], sites, config)
    shaped = np.asarray(tokens["coefficient"]).reshape(
        len(positions), config.paths_per_point, 2, 16, 8, 4
    )
    profile = np.abs(shaped) ** 2
    port = profile.sum(axis=(3, 4), dtype=np.float64)
    port /= np.maximum(port.sum(axis=(2, 3), keepdims=True), TINY)
    sequence = sequence_geometry(
        tokens["vertices"],
        tokens["interaction_count"],
        positions,
        sites,
        data.setup.bs_positions,
        data.train_positions,
        config.interaction_steps,
    )
    hashes = sequence_hashes(tokens)
    buckets = np.stack(
        (
            hashes % np.uint64(SEQUENCE_HASH_BUCKETS),
            (hashes >> np.uint64(32)) % np.uint64(SEQUENCE_HASH_BUCKETS),
        ),
        axis=-1,
    ).astype(np.int64)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_npy(feature_path, features)
    atomic_npz(
        auxiliary_path,
        coordinate=coordinate.astype(np.float32),
        port=port.astype(np.float32),
        geometry=sequence.astype(np.float32),
        hash=buckets,
        depth=np.asarray(tokens["interaction_count"], dtype=np.int64),
        interactions=np.asarray(tokens["interactions"], dtype=np.int64),
        objects=np.asarray(tokens["objects"], dtype=np.int64),
        primitives=np.asarray(tokens["primitives"], dtype=np.int64),
        fraction=np.asarray(tokens["energy_fraction"], dtype=np.float32),
        presence=np.asarray(tokens["presence"], dtype=np.float32),
        angles=np.asarray(tokens["angles"], dtype=np.float32),
    )
    report = {
        "stage": "model_inputs",
        "status": "complete",
        "split": split,
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "sequence_shape": list(sequence.shape),
        "finite": bool(
            np.isfinite(features).all()
            and np.isfinite(coordinate).all()
            and np.isfinite(port).all()
            and np.isfinite(sequence).all()
        ),
        "wall_seconds": float(time.time() - started),
        "measured_bottleneck": "full path coefficient feature extraction",
    }
    atomic_json(paths.report_dir / f"model_inputs_{split}.json", report)
    return report


def load_model_inputs(
    paths: RuntimePaths, split: str
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    root = paths.work_dir / "model_inputs" / split
    features = np.load(root / "features.npy", mmap_mode="r")
    with np.load(root / "auxiliary.npz", allow_pickle=False) as loaded:
        auxiliary = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    return features, auxiliary
