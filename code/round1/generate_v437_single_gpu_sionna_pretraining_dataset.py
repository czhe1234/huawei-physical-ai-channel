#!/usr/bin/env python3
"""Generate a large single-GPU Sionna pretraining dataset in the Etoile scene."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from project_paths import PATHS


VERSION_NUMBER = 437
VERSION = "v437_single_gpu_sionna_pretraining_dataset"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
SHARD_ROOT = OUT_DIR / "shards"
POSITIONS_PATH = OUT_DIR / "positions.npz"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
SCENE_XML = (
    PATHS.artifacts
    / "external"
    / "restored_sionna_official_etoile"
    / "etoile.xml"
)
BS_POSITION = np.asarray([50.0, 0.0, 25.0], dtype=np.float32)
NUM_SUBCARRIERS = 192
SUBCARRIER_SPACING = 30_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=1_000)
    parser.add_argument("--max-paths", type=int, default=128)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--samples-per-src", type=int, default=1_000_000)
    parser.add_argument("--max-num-paths-per-src", type=int, default=1_000_000)
    parser.add_argument("--frequency", type=float, default=6.693e9)
    parser.add_argument("--seed", type=int, default=43_701)
    parser.add_argument("--min-test-distance", type=float, default=1.0)
    parser.add_argument("--min-train-distance", type=float, default=0.05)
    parser.add_argument("--save-full-h", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-shards", action="store_true")
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--limit-shards", type=int)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def require_single_gpu(variant: str) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    if len(devices) != 1:
        raise RuntimeError(
            "v437 requires exactly one CUDA_VISIBLE_DEVICES entry; "
            f"received {visible!r}"
        )
    if not variant.startswith("cuda_"):
        raise ValueError(f"v437 requires a CUDA Mitsuba variant, got {variant!r}")
    return devices[0]


def pairwise_min_distance(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    output = np.full(len(points), np.inf, dtype=np.float32)
    for start in range(0, len(points), 4096):
        end = min(start + 4096, len(points))
        delta = points[start:end, None, :2] - reference[None, :, :2]
        distance2 = np.sum(delta * delta, axis=2, dtype=np.float64)
        output[start:end] = np.sqrt(np.min(distance2, axis=1)).astype(np.float32)
    return output


def train_neighbor_bank(train_xy: np.ndarray, count: int = 16) -> np.ndarray:
    delta = train_xy[:, None, :] - train_xy[None, :, :]
    distance2 = np.sum(delta * delta, axis=2, dtype=np.float64)
    np.fill_diagonal(distance2, np.inf)
    return np.argpartition(distance2, kth=count - 1, axis=1)[:, :count].astype(np.int32)


def generate_positions(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if POSITIONS_PATH.is_file() and not args.overwrite:
        cached = np.load(POSITIONS_PATH)
        if len(cached["positions"]) != args.count:
            raise ValueError(
                f"Existing {POSITIONS_PATH} has {len(cached['positions'])} positions, "
                f"requested {args.count}; use --overwrite to replace it"
            )
        return {key: cached[key] for key in cached.files}

    train = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    official_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    train_xy = train[:, :2]
    neighbors = train_neighbor_bank(train_xy)
    lower = np.min(train_xy, axis=0) - 8.0
    upper = np.max(train_xy, axis=0) + 8.0
    rng = np.random.default_rng(args.seed)
    accepted: list[np.ndarray] = []
    accepted_mode: list[np.ndarray] = []
    seen: set[tuple[int, int]] = set()

    while sum(len(item) for item in accepted) < args.count:
        remaining = args.count - sum(len(item) for item in accepted)
        draw = max(8192, int(remaining * 1.35))
        anchor_id = rng.integers(0, len(train), size=draw)
        mode = rng.choice(3, size=draw, p=np.asarray([0.50, 0.30, 0.20]))
        angle = rng.uniform(-math.pi, math.pi, size=draw)
        radius = np.empty(draw, dtype=np.float32)
        local = mode == 0
        medium = mode == 1
        radius[local] = np.sqrt(
            rng.uniform(0.15**2, 4.0**2, size=int(np.sum(local)))
        )
        radius[medium] = np.sqrt(
            rng.uniform(4.0**2, 18.0**2, size=int(np.sum(medium)))
        )
        radius[mode == 2] = 0.0
        candidate = train_xy[anchor_id].copy()
        candidate[:, 0] += radius * np.cos(angle)
        candidate[:, 1] += radius * np.sin(angle)

        bridge = mode == 2
        bridge_count = int(np.sum(bridge))
        if bridge_count:
            source = anchor_id[bridge]
            bank_column = rng.integers(0, neighbors.shape[1], size=bridge_count)
            destination = neighbors[source, bank_column]
            fraction = rng.uniform(0.05, 0.95, size=bridge_count).astype(np.float32)
            segment = train_xy[destination] - train_xy[source]
            norm = np.maximum(np.linalg.norm(segment, axis=1), 1e-3)
            normal = np.column_stack([-segment[:, 1], segment[:, 0]]) / norm[:, None]
            jitter = rng.normal(0.0, 1.2, size=bridge_count).astype(np.float32)
            candidate[bridge] = (
                train_xy[source]
                + fraction[:, None] * segment
                + jitter[:, None] * normal
            )

        inside = np.all((candidate >= lower) & (candidate <= upper), axis=1)
        candidate = candidate[inside]
        candidate_mode = mode[inside]
        if not len(candidate):
            continue
        min_test = pairwise_min_distance(candidate, official_test)
        min_train = pairwise_min_distance(candidate, train)
        keep = (
            (min_test >= args.min_test_distance)
            & (min_train >= args.min_train_distance)
        )
        candidate = candidate[keep]
        candidate_mode = candidate_mode[keep]
        unique_rows = []
        unique_modes = []
        for point, point_mode in zip(candidate, candidate_mode):
            key = (int(round(float(point[0]) * 100.0)), int(round(float(point[1]) * 100.0)))
            if key in seen:
                continue
            seen.add(key)
            unique_rows.append(point)
            unique_modes.append(point_mode)
            if sum(len(item) for item in accepted) + len(unique_rows) >= args.count:
                break
        if unique_rows:
            accepted.append(np.asarray(unique_rows, dtype=np.float32))
            accepted_mode.append(np.asarray(unique_modes, dtype=np.int8))

    xy = np.concatenate(accepted, axis=0)[: args.count]
    sample_mode = np.concatenate(accepted_mode, axis=0)[: args.count]
    positions = np.column_stack(
        [xy, np.full(args.count, 1.5, dtype=np.float32)]
    ).astype(np.float32)
    min_test_distance = pairwise_min_distance(positions, official_test)
    min_train_distance = pairwise_min_distance(positions, train)
    tile_x = np.floor((xy[:, 0] - lower[0]) / 20.0).astype(np.int32)
    tile_y = np.floor((xy[:, 1] - lower[1]) / 20.0).astype(np.int32)
    tile_hash = np.mod(tile_x * 73_856_093 + tile_y * 19_349_663, 10)
    split = np.zeros(args.count, dtype=np.int8)
    split[tile_hash == 0] = 1
    split[tile_hash == 1] = 2
    order = rng.permutation(args.count)
    payload = {
        "positions": positions[order],
        "sample_mode": sample_mode[order],
        "pretrain_split": split[order],
        "min_official_test_distance": min_test_distance[order],
        "min_official_train_distance": min_train_distance[order],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(POSITIONS_PATH, **payload)
    return payload


def tx_target(mi):
    radius = 140.0
    azimuth = math.radians(-30.0)
    tilt = math.radians(-9.53)
    return mi.Point3f(
        float(BS_POSITION[0] + radius * math.cos(azimuth)),
        float(BS_POSITION[1] + radius * math.sin(azimuth)),
        float(BS_POSITION[2] + radius * math.tan(tilt)),
    )


def convert_cfr(raw: np.ndarray) -> np.ndarray:
    expected = (4, 1, 256, 1, NUM_SUBCARRIERS)
    if raw.ndim != 6 or tuple(raw.shape[1:]) != expected:
        raise ValueError(f"Unexpected CFR shape {raw.shape}")
    return np.transpose(raw[:, :, 0, :, 0, :], (0, 2, 1, 3)).astype(np.complex64)


def normalized(value: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    denominator = np.sum(value, axis=axes, keepdims=True, dtype=np.float64)
    return (value / np.maximum(denominator, 1e-30)).astype(np.float32)


def spectral_targets(channel: np.ndarray) -> dict[str, np.ndarray]:
    count = len(channel)
    h6 = channel.reshape(count, 2, 16, 8, 4, NUM_SUBCARRIERS)
    point_power = np.sum(np.abs(channel) ** 2, axis=(1, 2, 3), dtype=np.float64)

    delay = np.fft.ifft(channel, axis=-1)
    delay_power = (np.abs(delay) ** 2).astype(np.float32)
    pdp = normalized(np.sum(delay_power, axis=(1, 2), dtype=np.float64), (1,))
    rx_pdp = normalized(np.sum(delay_power, axis=1, dtype=np.float64), (1, 2))
    delay6 = delay_power.reshape(count, 2, 16, 8, 4, NUM_SUBCARRIERS)
    tx_pol_pdp = normalized(np.sum(delay6, axis=(2, 3, 4), dtype=np.float64), (1, 2))
    h_delay = normalized(np.sum(delay6, axis=(1, 3, 4), dtype=np.float64), (1, 2))
    v_delay = normalized(np.sum(delay6, axis=(1, 2, 4), dtype=np.float64), (1, 2))
    pol_ue_delay = normalized(np.sum(delay6, axis=(2, 3), dtype=np.float64), (1, 2, 3))
    del delay, delay_power, delay6

    leaked_input = h6.transpose(0, 1, 5, 2, 4, 3)
    leaked = np.fft.fft2(leaked_input, axes=(2, 3))
    leaked_power = (np.abs(leaked) ** 2).astype(np.float32)
    joint_sh = normalized(np.sum(leaked_power, axis=(1, 4, 5), dtype=np.float64), (1, 2))
    pol_sh = normalized(np.sum(leaked_power, axis=(4, 5), dtype=np.float64), (1, 2, 3))
    del leaked_input, leaked, leaked_power

    array_angle = np.fft.fft2(h6, axes=(2, 3))
    array_power = (np.abs(array_angle) ** 2).astype(np.float32)
    conventional_hv = normalized(
        np.sum(array_power, axis=(1, 4, 5), dtype=np.float64),
        (1, 2),
    )
    del array_angle, array_power
    return {
        "log_point_power": np.log(np.maximum(point_power, 1e-30)).astype(np.float32),
        "pdp": pdp.astype(np.float16),
        "rx_pdp": rx_pdp.astype(np.float16),
        "tx_pol_pdp": tx_pol_pdp.astype(np.float16),
        "h_delay": h_delay.astype(np.float16),
        "v_delay": v_delay.astype(np.float16),
        "pol_ue_delay": pol_ue_delay.astype(np.float16),
        "joint_sh": joint_sh.astype(np.float16),
        "pol_sh": pol_sh.astype(np.float16),
        "conventional_hv": conventional_hv.astype(np.float16),
    }


def empty_path_targets(
    count: int,
    max_paths: int,
    max_depth: int,
) -> dict[str, np.ndarray]:
    shape = (count, max_paths)
    return {
        "valid": np.zeros(shape, dtype=bool),
        "tau": np.full(shape, np.nan, dtype=np.float32),
        "theta_t": np.full(shape, np.nan, dtype=np.float16),
        "phi_t": np.full(shape, np.nan, dtype=np.float16),
        "theta_r": np.full(shape, np.nan, dtype=np.float16),
        "phi_r": np.full(shape, np.nan, dtype=np.float16),
        "log_power": np.full(shape, -np.inf, dtype=np.float32),
        "normalized_power": np.zeros(shape, dtype=np.float16),
        "interaction_count": np.zeros(shape, dtype=np.int8),
        "interaction_mask": np.zeros(shape, dtype=np.int16),
        "interactions": np.zeros((count, max_paths, max_depth), dtype=np.int8),
        "objects": np.full((count, max_paths, max_depth), -1, dtype=np.int32),
        "vertices": np.full(
            (count, max_paths, max_depth, 3),
            np.nan,
            dtype=np.float32,
        ),
        "complex_reference": np.zeros(shape, dtype=np.complex64),
        "rx_fingerprint": np.zeros((count, max_paths, 4), dtype=np.float16),
        "tx_pol_fingerprint": np.zeros((count, max_paths, 2), dtype=np.float16),
        "tx_h_fingerprint": np.zeros((count, max_paths, 16), dtype=np.float16),
        "tx_v_fingerprint": np.zeros((count, max_paths, 8), dtype=np.float16),
    }


def extract_path_targets(paths, max_paths: int, max_depth: int) -> dict[str, np.ndarray]:
    valid = paths.valid.numpy()[:, 0].astype(bool)
    tau = paths.tau.numpy()[:, 0].astype(np.float32)
    theta_t = paths.theta_t.numpy()[:, 0].astype(np.float32)
    phi_t = paths.phi_t.numpy()[:, 0].astype(np.float32)
    theta_r = paths.theta_r.numpy()[:, 0].astype(np.float32)
    phi_r = paths.phi_r.numpy()[:, 0].astype(np.float32)
    interactions = paths.interactions.numpy()[:, :, 0].transpose(1, 2, 0).astype(np.int8)
    objects = paths.objects.numpy()[:, :, 0].transpose(1, 2, 0).astype(np.int64)
    objects = np.where(objects >= np.iinfo(np.uint32).max, -1, objects).astype(np.int32)
    vertices = paths.vertices.numpy()[:, :, 0].transpose(1, 2, 0, 3).astype(np.float32)
    cir, _ = paths.cir(normalize_delays=False, out_type="numpy")
    raw = cir[:, :, 0, :, :, 0].astype(np.complex64)
    element_power = (np.abs(raw) ** 2).astype(np.float32)
    path_power = np.sum(element_power, axis=(1, 2), dtype=np.float64)
    output = empty_path_targets(len(valid), max_paths, max_depth)

    for row in range(len(valid)):
        ids = np.flatnonzero(
            valid[row]
            & np.isfinite(tau[row])
            & (tau[row] >= 0.0)
            & np.isfinite(path_power[row])
            & (path_power[row] > 0.0)
        )
        if not len(ids):
            continue
        order = ids[np.argsort(path_power[row, ids])[::-1][:max_paths]]
        count = len(order)
        output["valid"][row, :count] = True
        output["tau"][row, :count] = tau[row, order]
        output["theta_t"][row, :count] = theta_t[row, order]
        output["phi_t"][row, :count] = phi_t[row, order]
        output["theta_r"][row, :count] = theta_r[row, order]
        output["phi_r"][row, :count] = phi_r[row, order]
        selected_power = path_power[row, order]
        output["log_power"][row, :count] = np.log(
            np.maximum(selected_power, 1e-30)
        ).astype(np.float32)
        output["normalized_power"][row, :count] = (
            selected_power / np.maximum(np.sum(selected_power), 1e-30)
        ).astype(np.float16)
        selected_interactions = interactions[row, order, :max_depth]
        output["interactions"][row, :count] = selected_interactions
        output["objects"][row, :count] = objects[row, order, :max_depth]
        output["vertices"][row, :count] = vertices[row, order, :max_depth]
        output["interaction_count"][row, :count] = np.sum(
            selected_interactions != 0,
            axis=1,
        ).astype(np.int8)
        output["interaction_mask"][row, :count] = np.bitwise_or.reduce(
            selected_interactions.astype(np.int16),
            axis=1,
        )
        output["complex_reference"][row, :count] = raw[row, 0, 0, order]

        selected_element = element_power[row][:, :, order].reshape(
            4, 2, 16, 8, count
        )
        denominator = np.maximum(selected_power, 1e-30)
        output["rx_fingerprint"][row, :count] = (
            np.sum(selected_element, axis=(1, 2, 3), dtype=np.float64).T
            / denominator[:, None]
        ).astype(np.float16)
        output["tx_pol_fingerprint"][row, :count] = (
            np.sum(selected_element, axis=(0, 2, 3), dtype=np.float64).T
            / denominator[:, None]
        ).astype(np.float16)
        output["tx_h_fingerprint"][row, :count] = (
            np.sum(selected_element, axis=(0, 1, 3), dtype=np.float64).T
            / denominator[:, None]
        ).astype(np.float16)
        output["tx_v_fingerprint"][row, :count] = (
            np.sum(selected_element, axis=(0, 1, 2), dtype=np.float64).T
            / denominator[:, None]
        ).astype(np.float16)
    return output


def allocate_target_buffers(count: int) -> dict[str, np.ndarray]:
    return {
        "log_point_power": np.empty(count, dtype=np.float32),
        "pdp": np.empty((count, 192), dtype=np.float16),
        "rx_pdp": np.empty((count, 4, 192), dtype=np.float16),
        "tx_pol_pdp": np.empty((count, 2, 192), dtype=np.float16),
        "h_delay": np.empty((count, 16, 192), dtype=np.float16),
        "v_delay": np.empty((count, 8, 192), dtype=np.float16),
        "pol_ue_delay": np.empty((count, 2, 4, 192), dtype=np.float16),
        "joint_sh": np.empty((count, 192, 16), dtype=np.float16),
        "pol_sh": np.empty((count, 2, 192, 16), dtype=np.float16),
        "conventional_hv": np.empty((count, 16, 8), dtype=np.float16),
    }


def copy_block(destination: dict[str, np.ndarray], block: dict[str, np.ndarray], start: int) -> None:
    end = start + len(next(iter(block.values())))
    for key, value in block.items():
        destination[key][start:end] = value


def create_scene(args: argparse.Namespace, mi):
    from sionna.rt import PathSolver, PlanarArray, Transmitter, load_scene

    scene = load_scene(str(SCENE_XML), merge_shapes=True)
    scene.frequency = args.frequency
    scene.tx_array = PlanarArray(
        num_rows=8,
        num_cols=16,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="VH",
    )
    scene.rx_array = PlanarArray(
        num_rows=2,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="cross",
    )
    tx = Transmitter(
        name="tx_v437",
        position=mi.Point3f(*map(float, BS_POSITION)),
        look_at=tx_target(mi),
        power_dbm=44.0,
    )
    scene.add(tx)
    return scene, tx, PathSolver()


def remove_receivers(scene, names: list[str]) -> None:
    for name in names:
        if scene.get(name) is not None:
            scene.remove(name)


def trace_shard(
    *,
    shard_id: int,
    positions: np.ndarray,
    position_ids: np.ndarray,
    metadata: dict[str, np.ndarray],
    args: argparse.Namespace,
    scene,
    tx,
    solver,
    Receiver,
    frequencies,
    mi,
    dr,
) -> dict[str, object]:
    shard_dir = SHARD_ROOT / f"shard_{shard_id:05d}"
    done_path = shard_dir / "done.json"
    if done_path.is_file() and not (args.overwrite or args.overwrite_shards):
        return json.loads(done_path.read_text(encoding="utf-8"))
    shard_dir.mkdir(parents=True, exist_ok=True)
    for partial in shard_dir.glob("*.partial.npy"):
        partial.unlink()

    count = len(positions)
    target_buffers = allocate_target_buffers(count)
    path_buffers = empty_path_targets(count, args.max_paths, args.max_depth)
    h_partial = shard_dir / "H.partial.npy"
    h_final = shard_dir / "H.npy"
    channel_output = None
    if args.save_full_h:
        channel_output = np.lib.format.open_memmap(
            h_partial,
            mode="w+",
            dtype=np.complex64,
            shape=(count, 256, 4, NUM_SUBCARRIERS),
        )

    receiver_names: list[str] = []
    batch_reports = []
    started = time.time()
    for start in range(0, count, args.batch_size):
        end = min(start + args.batch_size, count)
        remove_receivers(scene, receiver_names)
        receiver_names = []
        for local, position in enumerate(positions[start:end], start=start):
            name = f"rx_v437_{shard_id:05d}_{local:05d}"
            scene.add(
                Receiver(
                    name=name,
                    position=mi.Point3f(*map(float, position)),
                    look_at=tx,
                )
            )
            receiver_names.append(name)
        solve_start = time.time()
        paths = solver(
            scene,
            max_depth=args.max_depth,
            max_num_paths_per_src=args.max_num_paths_per_src,
            samples_per_src=args.samples_per_src,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=True,
            diffraction=False,
            edge_diffraction=False,
            diffraction_lit_region=True,
            seed=args.seed + shard_id * 1_000_003 + start,
        )
        dr.sync_thread()
        solve_seconds = time.time() - solve_start

        cfr_start = time.time()
        raw_cfr = paths.cfr(
            mi.TensorXf(frequencies),
            normalize_delays=False,
            normalize=False,
            out_type="numpy",
        )
        channel = convert_cfr(raw_cfr)
        dr.sync_thread()
        cfr_seconds = time.time() - cfr_start
        if not np.isfinite(channel).all():
            raise FloatingPointError(f"Non-finite CFR in shard {shard_id} batch {start}:{end}")
        target_start = time.time()
        target_block = spectral_targets(channel)
        path_block = extract_path_targets(paths, args.max_paths, args.max_depth)
        target_seconds = time.time() - target_start
        if channel_output is not None:
            channel_output[start:end] = channel
            channel_output.flush()
        copy_block(target_buffers, target_block, start)
        copy_block(path_buffers, path_block, start)
        path_count = np.sum(path_block["valid"], axis=1)
        batch_reports.append(
            {
                "start": int(start),
                "end": int(end),
                "solve_seconds": float(solve_seconds),
                "cfr_seconds": float(cfr_seconds),
                "target_seconds": float(target_seconds),
                "mean_retained_paths": float(np.mean(path_count)),
                "zero_path_points": int(np.sum(path_count == 0)),
                "zero_channel_points": int(
                    np.sum(target_block["log_point_power"] <= math.log(1e-30))
                ),
            }
        )
        print(
            f"v437 shard={shard_id} points={end}/{count} "
            f"solve={solve_seconds:.3f}s cfr={cfr_seconds:.3f}s "
            f"targets={target_seconds:.3f}s paths={float(np.mean(path_count)):.1f}",
            flush=True,
        )
        del paths, raw_cfr, channel, target_block, path_block
        dr.flush_malloc_cache()

    remove_receivers(scene, receiver_names)
    if channel_output is not None:
        channel_output.flush()
        del channel_output
        h_partial.replace(h_final)
    np.savez(
        shard_dir / "metadata.npz",
        position_id=position_ids,
        positions=positions,
        sample_mode=metadata["sample_mode"][position_ids],
        pretrain_split=metadata["pretrain_split"][position_ids],
        min_official_test_distance=metadata["min_official_test_distance"][position_ids],
        min_official_train_distance=metadata["min_official_train_distance"][position_ids],
    )
    np.savez(shard_dir / "spectral_targets.npz", **target_buffers)
    np.savez(shard_dir / "path_targets.npz", **path_buffers)
    path_count = np.sum(path_buffers["valid"], axis=1)
    report = {
        "shard_id": int(shard_id),
        "batch_size": int(args.batch_size),
        "samples_per_src": int(args.samples_per_src),
        "max_depth": int(args.max_depth),
        "position_start": int(position_ids[0]),
        "position_end": int(position_ids[-1]) + 1,
        "count": int(count),
        "seconds": float(time.time() - started),
        "mean_retained_paths": float(np.mean(path_count)),
        "zero_path_points": int(np.sum(path_count == 0)),
        "zero_channel_points": int(
            np.sum(target_buffers["log_point_power"] <= math.log(1e-30))
        ),
        "outputs": {
            "H": PATHS.relative(h_final) if args.save_full_h else None,
            "metadata": PATHS.relative(shard_dir / "metadata.npz"),
            "spectral_targets": PATHS.relative(shard_dir / "spectral_targets.npz"),
            "path_targets": PATHS.relative(shard_dir / "path_targets.npz"),
        },
        "batches": batch_reports,
    }
    atomic_json(done_path, report)
    return report


def main() -> None:
    args = parse_args()
    physical_gpu = require_single_gpu(args.mitsuba_variant)
    if args.count <= 0 or args.batch_size <= 0 or args.shard_size <= 0:
        raise ValueError("count, batch-size, and shard-size must be positive")
    if not SCENE_XML.is_file():
        raise FileNotFoundError(SCENE_XML)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = generate_positions(args)
    positions = metadata["positions"].astype(np.float32)

    import drjit as dr
    import mitsuba as mi
    import sionna

    mi.set_variant(args.mitsuba_variant)
    if "cuda" not in str(dr.backend_v(mi.Float([1.0]))).lower():
        raise RuntimeError("Mitsuba did not select the CUDA backend")
    from sionna.phy.channel import subcarrier_frequencies
    from sionna.rt import Receiver

    frequencies = (
        subcarrier_frequencies(NUM_SUBCARRIERS, SUBCARRIER_SPACING)
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    scene, tx, solver = create_scene(args, mi)
    total_shard_count = math.ceil(len(positions) / args.shard_size)
    if args.start_shard < 0 or args.start_shard >= total_shard_count:
        raise ValueError(
            f"start-shard must be in [0, {total_shard_count}), got {args.start_shard}"
        )
    stop_shard = total_shard_count
    if args.limit_shards is not None:
        stop_shard = min(stop_shard, args.start_shard + args.limit_shards)
    started = time.time()
    for shard_id in range(args.start_shard, stop_shard):
        start = shard_id * args.shard_size
        end = min(start + args.shard_size, len(positions))
        ids = np.arange(start, end, dtype=np.int64)
        trace_shard(
            shard_id=shard_id,
            positions=positions[start:end],
            position_ids=ids,
            metadata=metadata,
            args=args,
            scene=scene,
            tx=tx,
            solver=solver,
            Receiver=Receiver,
            frequencies=frequencies,
            mi=mi,
            dr=dr,
        )
        completed = [
            path
            for path in sorted(SHARD_ROOT.glob("shard_*/done.json"))
            if int(path.parent.name.split("_")[-1]) < total_shard_count
        ]
        partial_report = {
            "method": VERSION,
            "status": "generation_in_progress",
            "completed_shards": len(completed),
            "requested_shards": total_shard_count,
            "configuration": vars(args),
            "gpu": {
                "physical_cuda_visible_device": physical_gpu,
                "logical_device_count": 1,
                "mitsuba_variant": mi.variant(),
                "drjit_backend": str(dr.backend_v(mi.Float([1.0]))),
                "sionna_version": getattr(sionna, "__version__", "unknown"),
            },
            "shards": [
                json.loads(path.read_text(encoding="utf-8")) for path in completed
            ],
        }
        atomic_json(REPORT, partial_report)

    completed_paths = [
        path
        for path in sorted(SHARD_ROOT.glob("shard_*/done.json"))
        if int(path.parent.name.split("_")[-1]) < total_shard_count
    ]
    shard_reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in completed_paths
    ]
    total_points = int(sum(item["count"] for item in shard_reports))
    total_zeros = int(sum(item["zero_channel_points"] for item in shard_reports))
    complete = len(shard_reports) == total_shard_count
    report = {
        "method": VERSION,
        "status": (
            "single_gpu_pretraining_dataset_generated"
            if complete
            else "single_gpu_pretraining_dataset_partial"
        ),
        "primary_hypothesis": (
            "Dense same-scene random-location Sionna supervision can pretrain geometry-to-path "
            "birth/death, delay-angle, and energy modules that are later calibrated only with "
            "official training H."
        ),
        "scope": "pretraining data only; no official-test-coordinate channel labels",
        "configuration": vars(args),
        "gpu": {
            "physical_cuda_visible_device": physical_gpu,
            "logical_device_count": 1,
            "mitsuba_variant": mi.variant(),
            "drjit_backend": str(dr.backend_v(mi.Float([1.0]))),
            "sionna_version": getattr(sionna, "__version__", "unknown"),
        },
        "position_audit": {
            "source": "continuous samples derived only from official train coordinates",
            "minimum_test_distance_m": float(
                np.min(metadata["min_official_test_distance"])
            ),
            "minimum_train_distance_m": float(
                np.min(metadata["min_official_train_distance"])
            ),
            "split_counts": np.bincount(
                metadata["pretrain_split"], minlength=3
            ).astype(int).tolist(),
        },
        "summary": {
            "generated_points": total_points,
            "requested_points": int(args.count),
            "completed_shards": len(shard_reports),
            "requested_shards": total_shard_count,
            "zero_channel_points": total_zeros,
            "seconds": float(time.time() - started),
        },
        "outputs": {
            "positions": PATHS.relative(POSITIONS_PATH),
            "shard_root": PATHS.relative(SHARD_ROOT),
            "report": PATHS.relative(REPORT),
        },
        "shards": shard_reports,
    }
    atomic_json(REPORT, report)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
