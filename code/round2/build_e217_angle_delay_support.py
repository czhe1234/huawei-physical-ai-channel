#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback

import drjit as dr
import mitsuba as mi
import numpy as np

from data import load_positions
from project_paths import PATHS
from round2.physical_prior.physical_rt import (
    PathSolver,
    PlanarArray,
    Receiver,
    Transmitter,
    load_scene,
)
from round2.physical_prior.source_port_common import (
    CONFIGS,
    configure_materials,
    prepare_scene_files,
    solver_seed,
    tx_target,
)


VERSION = "Round2-E217-angle-delay-support-only"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e217_angle_delay_support_only"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e217_angle_delay_support_only"
SHARD_DIR = ARTIFACT_DIR / "support_shards"
COORDINATE_PATH = ARTIFACT_DIR / "candidate_coordinate_f32.npy"
MASK_PATH = ARTIFACT_DIR / "candidate_mask.npy"
POINT_IDS_PATH = ARTIFACT_DIR / "candidate_point_ids_i32.npy"
SUPPORT_REPORT = ANALYSIS_DIR / "support_report.json"
CAPACITY = 128
PERIODS = np.asarray((16.0, 8.0, 192.0), dtype=np.float32)
QUANTIZATION = np.asarray((0.25, 0.25, 1.0), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("generate", "merge", "all"), required=True)
    parser.add_argument("--site", type=int, choices=(0, 1))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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


def select_cuda_device(device: int) -> None:
    library = ctypes.CDLL(str(Path(dr.__file__).with_name("libdrjit-core.so")))
    setter = getattr(library, "_Z19jit_cuda_set_devicei")
    setter.argtypes = [ctypes.c_int]
    setter.restype = None
    setter(int(device))


def coordinates(tau: np.ndarray, theta: np.ndarray, phi: np.ndarray, site: int) -> np.ndarray:
    config = CONFIGS["base"]
    yaw = math.radians(float(config.site_yaw_deg[site]))
    tilt = math.radians(float(config.site_tilt_deg[site]))
    horizontal = (
        16.0 * config.tx_horizontal_spacing * np.sin(theta) * np.sin(phi - yaw)
    ) % 16.0
    vertical = (
        8.0
        * config.tx_vertical_spacing
        * (
            math.sin(tilt) * np.sin(theta) * np.cos(phi - yaw)
            - math.cos(tilt) * np.cos(theta)
        )
        + 2.0
    ) % 8.0
    delay = (tau * 30_000.0 * 192.0) % 192.0
    return np.stack((horizontal, vertical, delay), axis=1).astype(np.float32)


def periodic_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    delta = np.abs(left - right)
    delta = np.minimum(delta, PERIODS - delta) / PERIODS
    return np.square(delta, dtype=np.float32).sum(axis=1)


def select_support(value: np.ndarray) -> np.ndarray:
    if len(value) == 0:
        return value.astype(np.float32)
    value = np.remainder(np.asarray(value, dtype=np.float32), PERIODS)
    valid = np.all(np.isfinite(value), axis=1)
    value = value[valid]
    if len(value) == 0:
        return value
    cells = np.floor(value / QUANTIZATION).astype(np.int32)
    order = np.lexsort((value[:, 1], value[:, 0], value[:, 2]))
    ordered_cells = cells[order]
    keep = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        keep[1:] = np.any(ordered_cells[1:] != ordered_cells[:-1], axis=1)
    unique = value[order[keep]]
    if len(unique) <= CAPACITY:
        return unique
    selected = np.empty(CAPACITY, dtype=np.int64)
    selected[0] = 0
    minimum = periodic_distance(unique, unique[0])
    minimum[0] = -1.0
    for slot in range(1, CAPACITY):
        index = int(np.argmax(minimum))
        selected[slot] = index
        minimum = np.minimum(minimum, periodic_distance(unique, unique[index]))
        minimum[selected[: slot + 1]] = -1.0
    return unique[selected]


class CoordinateSession:
    def __init__(self, site: int, bs_position: np.ndarray, device: int) -> None:
        self.site = int(site)
        self.device = int(device)
        self.config = CONFIGS["base"]
        self.seed = 42
        select_cuda_device(device)
        scene_path = prepare_scene_files(
            scene_dir=ARTIFACT_DIR / "scene" / f"gpu_{device}"
        )
        self.scene = load_scene(
            str(scene_path), merge_shapes=True, remove_duplicate_vertices=True
        )
        self.scene.frequency = self.config.frequency_hz
        self.scene.tx_array = PlanarArray(
            num_rows=8,
            num_cols=16,
            vertical_spacing=self.config.tx_vertical_spacing,
            horizontal_spacing=self.config.tx_horizontal_spacing,
            pattern="tr38901",
            polarization="VH",
        )
        self.scene.rx_array = PlanarArray(
            num_rows=2,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="dipole",
            polarization="cross",
        )
        configure_materials(self.scene, self.config.scattering_coefficient)
        self.bs_position = np.asarray(bs_position, dtype=np.float32)
        self.transmitter = Transmitter(
            name=f"e217_tx_site{site}",
            position=mi.Point3f(*map(float, self.bs_position)),
            look_at=tx_target(
                self.bs_position,
                self.config.site_yaw_deg[site],
                self.config.site_tilt_deg[site],
            ),
            power_dbm=44.0,
        )
        self.scene.add(self.transmitter)
        self.solver = PathSolver()
        self.receivers: list[str] = []

    def solve(self, point_ids: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        for name in self.receivers:
            self.scene.remove(name)
        self.receivers = []
        for point_id, position in zip(point_ids, positions):
            name = f"e217_rx_{int(point_id):05d}"
            self.scene.add(
                Receiver(
                    name=name,
                    position=mi.Point3f(*map(float, position)),
                    look_at=self.transmitter,
                )
            )
            self.receivers.append(name)
        paths = self.solver(
            self.scene,
            max_depth=self.config.max_depth,
            max_num_paths_per_src=self.config.max_num_paths_per_src,
            samples_per_src=self.config.samples_per_src,
            synthetic_array=self.config.synthetic_array,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=True,
            diffraction=False,
            edge_diffraction=False,
            diffraction_lit_region=True,
            seed=solver_seed(self.seed, self.site, "train", int(point_ids[0])),
        )
        valid = np.asarray(paths.valid.numpy(), dtype=bool)
        tau = np.asarray(paths.tau.numpy(), dtype=np.float32)
        theta = np.asarray(paths.theta_t.numpy(), dtype=np.float32)
        phi = np.asarray(paths.phi_t.numpy(), dtype=np.float32)
        output = np.zeros((len(point_ids), CAPACITY, 3), dtype=np.float32)
        mask = np.zeros((len(point_ids), CAPACITY), dtype=bool)
        raw_count = 0
        for local in range(len(point_ids)):
            indices = np.flatnonzero(valid[local, 0])
            finite = (
                np.isfinite(tau[local, 0, indices])
                & np.isfinite(theta[local, 0, indices])
                & np.isfinite(phi[local, 0, indices])
                & (tau[local, 0, indices] >= 0.0)
            )
            indices = indices[finite]
            raw_count += len(indices)
            selected = select_support(
                coordinates(
                    tau[local, 0, indices],
                    theta[local, 0, indices],
                    phi[local, 0, indices],
                    self.site,
                )
            )
            output[local, : len(selected)] = selected
            mask[local, : len(selected)] = True
        del paths, valid, tau, theta, phi
        dr.sync_thread()
        return output, mask, raw_count

    def close(self) -> None:
        for name in self.receivers:
            self.scene.remove(name)
        self.receivers = []
        dr.sync_thread()


def shard_path(site: int, batch: int) -> Path:
    return SHARD_DIR / f"site_{site}" / f"batch_{batch:04d}.npz"


def valid_shard(path: Path, point_ids: np.ndarray) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            return (
                np.array_equal(payload["point_ids"], point_ids)
                and payload["coordinate"].shape == (len(point_ids), CAPACITY, 3)
                and payload["mask"].shape == (len(point_ids), CAPACITY)
            )
    except (OSError, ValueError, KeyError):
        return False


def generate(site: int, device: int, batch_size: int, force: bool) -> dict[str, object]:
    setup, train_positions, _, train_site, _ = load_positions()
    point_ids = np.flatnonzero(train_site == site).astype(np.int64)
    positions = train_positions[point_ids].astype(np.float32)
    started = time.time()
    completed = 0
    reused = 0
    raw_count = 0
    selected_count = 0
    session = CoordinateSession(site, setup.bs_positions[site], device)
    try:
        for batch, start in enumerate(range(0, len(point_ids), batch_size)):
            end = min(start + batch_size, len(point_ids))
            local_ids = point_ids[start:end]
            output = shard_path(site, batch)
            if not force and valid_shard(output, local_ids):
                reused += 1
                with np.load(output, allow_pickle=False) as payload:
                    selected_count += int(np.sum(payload["mask"]))
                    raw_count += int(payload["raw_count"])
                continue
            batch_started = time.time()
            coordinate, mask, local_raw = session.solve(local_ids, positions[start:end])
            atomic_npz(
                output,
                point_ids=local_ids.astype(np.int32),
                coordinate=coordinate,
                mask=mask,
                raw_count=np.asarray(local_raw, dtype=np.int64),
                seconds=np.asarray(time.time() - batch_started, dtype=np.float64),
            )
            completed += 1
            raw_count += local_raw
            selected_count += int(mask.sum())
            print(
                f"E217 support site={site} batch={batch + 1}/"
                f"{math.ceil(len(point_ids) / batch_size)} raw={local_raw} "
                f"selected={int(mask.sum())}",
                flush=True,
            )
    finally:
        session.close()
    report = {
        "experiment": VERSION,
        "status": "complete",
        "site": site,
        "device": device,
        "seed": 42,
        "configuration": "base",
        "point_count": int(len(point_ids)),
        "batch_size": batch_size,
        "completed_batches": completed,
        "reused_batches": reused,
        "raw_candidate_count": raw_count,
        "selected_candidate_count": selected_count,
        "selection_uses_energy": False,
        "saved_fields": ["point_ids", "coordinate", "mask"],
        "wall_seconds": float(time.time() - started),
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "exit_code": 0,
    }
    atomic_json(ANALYSIS_DIR / f"support_site{site}.json", report)
    return report


def merge() -> dict[str, object]:
    _, train_positions, _, train_site, _ = load_positions()
    coordinate = np.zeros((len(train_positions), CAPACITY, 3), dtype=np.float32)
    mask = np.zeros((len(train_positions), CAPACITY), dtype=bool)
    seen = np.zeros(len(train_positions), dtype=np.uint8)
    for site in (0, 1):
        point_ids = np.flatnonzero(train_site == site).astype(np.int64)
        paths = sorted((SHARD_DIR / f"site_{site}").glob("batch_*.npz"))
        if not paths:
            raise FileNotFoundError(f"no support shards for site {site}")
        covered: list[np.ndarray] = []
        for path in paths:
            with np.load(path, allow_pickle=False) as payload:
                rows = payload["point_ids"].astype(np.int64)
                if np.any(seen[rows]):
                    raise RuntimeError(f"duplicate point coverage in {path}")
                coordinate[rows] = payload["coordinate"].astype(np.float32)
                mask[rows] = payload["mask"].astype(bool)
                seen[rows] = 1
                covered.append(rows)
        if not np.array_equal(np.sort(np.concatenate(covered)), point_ids):
            raise RuntimeError(f"incomplete point coverage for site {site}")
    if not np.all(seen == 1) or not np.all(np.isfinite(coordinate)):
        raise RuntimeError("invalid merged support")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(COORDINATE_PATH, coordinate)
    np.save(MASK_PATH, mask)
    np.save(POINT_IDS_PATH, np.arange(len(train_positions), dtype=np.int32))
    counts = mask.sum(axis=1)
    report = {
        "experiment": VERSION,
        "status": "complete",
        "point_count": int(len(train_positions)),
        "candidate_count": int(mask.sum()),
        "per_point_mean": float(counts.mean()),
        "per_point_min": int(counts.min()),
        "per_point_max": int(counts.max()),
        "empty_points": int(np.sum(counts == 0)),
        "capacity_points": int(np.sum(counts == CAPACITY)),
        "selection_uses_energy": False,
        "model_visible_fields": ["coordinate", "mask"],
        "coordinate": PATHS.relative(COORDINATE_PATH),
        "mask": PATHS.relative(MASK_PATH),
        "exit_code": 0,
    }
    atomic_json(SUPPORT_REPORT, report)
    return report


def main() -> None:
    args = parse_args()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        result: dict[str, object] | list[dict[str, object]]
        if args.stage == "generate":
            if args.site is None:
                raise ValueError("--site is required for generate")
            result = generate(args.site, args.device, args.batch_size, args.force)
        elif args.stage == "merge":
            result = merge()
        else:
            if args.site is None:
                raise ValueError("--site is required for all")
            result = [
                generate(args.site, args.device, args.batch_size, args.force),
                merge(),
            ]
        print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
    except BaseException as error:
        atomic_json(
            ANALYSIS_DIR / f"failure_support_{args.stage}_site{args.site}.json",
            {
                "experiment": VERSION,
                "status": "failed",
                "stage": args.stage,
                "site": args.site,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "exit_code": 1,
            },
        )
        raise


if __name__ == "__main__":
    main()
