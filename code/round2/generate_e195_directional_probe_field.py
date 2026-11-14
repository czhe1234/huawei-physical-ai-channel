#!/usr/bin/env python3
"""Generate resumable directional Sionna probe paths for Round2-E195."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np

from generate_multiseed_sionna_dataset import (
    CONFIGS,
    SEEDS,
    prepare_runtime,
    select_drjit_cuda_device,
    tx_target,
)
from generate_v5_sionna_spectral_library import (
    SCENE_XML,
    configure_materials,
    prepare_scene,
)
from project_paths import PATHS
from round2.data import load_positions


VERSION = "Round2-E195-directional-probe-field"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e195_directional_probe_field"
REPORT_DIR = PATHS.analysis / "results" / "round2_e195_directional_probe_field"
STATE = ARTIFACT_DIR / "state.json"
POSITIONS = ARTIFACT_DIR / "positions.npy"
SITES = ARTIFACT_DIR / "site.npy"
ANCHOR_IDS = ARTIFACT_DIR / "anchor_id.npy"
OFFSET_IDS = ARTIFACT_DIR / "offset_id.npy"
SPLITS = ARTIFACT_DIR / "split.npy"
OFFSETS_PATH = ARTIFACT_DIR / "offsets_xy.npy"

MP = 2
MH = 16
MV = 8
N_UE = 4
PATH_SLOTS = 64
RADII = (0.75, 2.0, 5.0)
AZIMUTHS_DEG = tuple(range(0, 360, 45))
PROBES_PER_ANCHOR = len(RADII) * len(AZIMUTHS_DEG)
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "seed", "report"), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS[:3])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--site", choices=("all", "0", "1"), default="all")
    parser.add_argument("--rewrite-selected", action="store_true")
    parser.add_argument("--batch-shards", type=int, default=1)
    parser.add_argument("--batch-shard", type=int, default=0)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def digest_arrays(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def open_memmap(path: Path, dtype: Any, shape: tuple[int, ...]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        value = np.load(path, mmap_mode="r+")
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError(
                f"cache mismatch for {PATHS.relative(path)}: "
                f"{value.shape}/{value.dtype} != {shape}/{np.dtype(dtype)}"
            )
        return value
    value = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    value[...] = 0
    value.flush()
    return value


def directional_offsets() -> np.ndarray:
    output = []
    for radius in RADII:
        for angle_deg in AZIMUTHS_DEG:
            angle = math.radians(angle_deg)
            output.append((radius * math.cos(angle), radius * math.sin(angle)))
    return np.asarray(output, dtype=np.float32)


def prepare_positions(force: bool) -> dict[str, Any]:
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    paths = (POSITIONS, SITES, ANCHOR_IDS, OFFSET_IDS, SPLITS, OFFSETS_PATH, STATE)
    if force:
        for path in paths:
            path.unlink(missing_ok=True)

    if all(path.is_file() for path in paths):
        state = json.loads(STATE.read_text(encoding="utf-8"))
        state["prepare_status"] = "reused"
        return state

    _, train_positions, test_positions, train_site, test_site = load_positions()
    anchors = np.concatenate((train_positions, test_positions), axis=0).astype(np.float32)
    anchor_site = np.concatenate((train_site, test_site), axis=0).astype(np.int8)
    split = np.concatenate(
        (
            np.zeros(len(train_positions), dtype=np.int8),
            np.ones(len(test_positions), dtype=np.int8),
        )
    )
    offsets = directional_offsets()
    probe_positions = np.repeat(anchors, len(offsets), axis=0)
    probe_positions[:, :2] += np.tile(offsets, (len(anchors), 1))
    probe_site = np.repeat(anchor_site, len(offsets)).astype(np.int8)
    anchor_ids = np.repeat(np.arange(len(anchors), dtype=np.int32), len(offsets))
    offset_ids = np.tile(np.arange(len(offsets), dtype=np.int8), len(anchors))
    probe_split = np.repeat(split, len(offsets)).astype(np.int8)

    np.save(POSITIONS, probe_positions.astype(np.float32))
    np.save(SITES, probe_site)
    np.save(ANCHOR_IDS, anchor_ids)
    np.save(OFFSET_IDS, offset_ids)
    np.save(SPLITS, probe_split)
    np.save(OFFSETS_PATH, offsets)
    if not SCENE_XML.is_file():
        raise FileNotFoundError("run --stage prepare before seed generation")

    configuration = CONFIGS[0]
    state = {
        "experiment": VERSION,
        "status": "positions_prepared",
        "prepare_status": "created",
        "configuration": asdict(configuration),
        "seed_order": list(SEEDS[:3]),
        "anchor_count": int(len(anchors)),
        "train_anchor_count": int(len(train_positions)),
        "test_anchor_count": int(len(test_positions)),
        "probes_per_anchor": int(len(offsets)),
        "probe_count": int(len(probe_positions)),
        "path_slots": PATH_SLOTS,
        "radii_m": list(RADII),
        "azimuths_deg": list(AZIMUTHS_DEG),
        "digest": digest_arrays(
            probe_positions, probe_site, anchor_ids, offset_ids, probe_split, offsets
        ),
        "seconds": float(time.time() - started),
    }
    atomic_json(STATE, state)
    return state


def seed_paths(seed: int) -> dict[str, Path]:
    prefix = ARTIFACT_DIR / f"seed_{seed}"
    return {
        "coordinate": prefix.with_name(prefix.name + "_coordinate_f16.npy"),
        "raw": prefix.with_name(prefix.name + "_raw_f16.npy"),
        "mass": prefix.with_name(prefix.name + "_mass_f16.npy"),
        "port": prefix.with_name(prefix.name + "_port_f16.npy"),
        "mask": prefix.with_name(prefix.name + "_mask.npy"),
        "depth": prefix.with_name(prefix.name + "_depth.npy"),
        "path_count": prefix.with_name(prefix.name + "_path_count.npy"),
        "done": prefix.with_name(prefix.name + "_done.npy"),
        "report": REPORT_DIR / f"seed_{seed}.json",
    }


def compact_path_batch(
    traced: Any,
    batch_rows: np.ndarray,
    site_value: int,
    configuration: Any,
) -> dict[str, np.ndarray]:
    batch_count = len(batch_rows)
    coordinate = np.zeros((batch_count, PATH_SLOTS, 3), dtype=np.float32)
    raw = np.zeros((batch_count, PATH_SLOTS, 3), dtype=np.float32)
    mass = np.zeros((batch_count, PATH_SLOTS), dtype=np.float32)
    port = np.zeros((batch_count, PATH_SLOTS, MP, N_UE), dtype=np.float32)
    mask = np.zeros((batch_count, PATH_SLOTS), dtype=bool)
    depth = np.zeros((batch_count, PATH_SLOTS), dtype=np.uint8)
    path_count = np.zeros(batch_count, dtype=np.int32)

    valid = np.asarray(traced.valid.numpy(), dtype=bool)
    tau = np.asarray(traced.tau.numpy(), dtype=np.float32)
    theta_t = np.asarray(traced.theta_t.numpy(), dtype=np.float32)
    phi_t = np.asarray(traced.phi_t.numpy(), dtype=np.float32)
    interactions = np.asarray(traced.interactions.numpy(), dtype=np.uint32)
    raw_a_value = traced.a
    if isinstance(raw_a_value, tuple):
        coefficients = (
            np.asarray(raw_a_value[0].numpy(), dtype=np.float32)
            + 1j * np.asarray(raw_a_value[1].numpy(), dtype=np.float32)
        ).astype(np.complex64)
    else:
        raw_a = np.asarray(raw_a_value.numpy())
        if np.iscomplexobj(raw_a):
            coefficients = raw_a.astype(np.complex64)
        elif raw_a.ndim == 6 and raw_a.shape[0] == 2:
            coefficients = (raw_a[0] + 1j * raw_a[1]).astype(np.complex64)
        else:
            raise ValueError(
                f"unexpected path coefficient representation {raw_a.shape}"
            )

    yaw = math.radians(configuration.site_yaw_deg[site_value])
    tilt = math.radians(configuration.site_tilt_deg[site_value])
    for local in range(batch_count):
        indices = np.flatnonzero(valid[local, 0])
        path_count[local] = len(indices)
        if len(indices) == 0:
            continue
        local_coefficients = np.take(
            coefficients[local, :, 0, :, :], indices, axis=-1
        ).transpose(2, 1, 0)
        shaped = local_coefficients.reshape(len(indices), MP, MH, MV, N_UE)
        local_port_energy = np.square(np.abs(shaped), dtype=np.float32).sum(
            axis=(2, 3), dtype=np.float64
        ).astype(np.float32)
        path_energy = local_port_energy.sum(axis=(1, 2), dtype=np.float64).astype(
            np.float32
        )
        fraction = path_energy / np.maximum(path_energy.sum(dtype=np.float64), TINY)
        take = min(PATH_SLOTS, len(fraction))
        selected = np.argpartition(fraction, len(fraction) - take)[-take:]
        selected = selected[np.argsort(fraction[selected])[::-1]]
        local_mass = np.asarray(fraction[selected], dtype=np.float32)
        local_mass /= np.maximum(local_mass.sum(dtype=np.float64), TINY)
        local_port = np.asarray(local_port_energy[selected], dtype=np.float32)
        local_port /= np.maximum(local_port.sum(axis=(1, 2), keepdims=True), TINY)
        selected_paths = indices[selected]
        theta = np.asarray(theta_t[local, 0, selected_paths], dtype=np.float32)
        phi = np.asarray(phi_t[local, 0, selected_paths], dtype=np.float32)
        horizontal = (
            MH
            * configuration.tx_horizontal_spacing
            * np.sin(theta)
            * np.sin(phi - yaw)
        ) % MH
        vertical = (
            MV
            * configuration.tx_vertical_spacing
            * (
                math.sin(tilt) * np.sin(theta) * np.cos(phi - yaw)
                - math.cos(tilt) * np.cos(theta)
            )
            + 2.0
        ) % MV
        delay = (
            np.asarray(tau[local, 0, selected_paths], dtype=np.float32)
            * 30_000.0
            * 192.0
        ) % 192.0
        local_coordinate = np.stack((horizontal, vertical, delay), axis=-1)
        local_raw = np.stack((theta, phi, delay), axis=-1)

        coordinate[local, :take] = local_coordinate
        raw[local, :take] = local_raw
        mass[local, :take] = local_mass
        port[local, :take] = local_port
        mask[local, :take] = True
        depth[local, :take] = np.asarray(
            np.count_nonzero(
                np.take(interactions[:, local, 0, :], selected_paths, axis=-1).T,
                axis=1,
            ),
            dtype=np.uint8,
        )
    return {
        "coordinate": coordinate,
        "raw": raw,
        "mass": mass,
        "port": port,
        "mask": mask,
        "depth": depth,
        "path_count": path_count,
    }


def generate_seed(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed is None:
        raise ValueError("--seed is required for --stage seed")
    if not STATE.is_file():
        raise FileNotFoundError("run --stage prepare before seed generation")
    state = json.loads(STATE.read_text(encoding="utf-8"))
    positions = np.load(POSITIONS, mmap_mode="r")
    site = np.load(SITES, mmap_mode="r")
    configuration = CONFIGS[0]
    serialized_configuration = json.loads(json.dumps(asdict(configuration)))
    if state["configuration"] != serialized_configuration:
        raise ValueError("E195 configuration mismatch")

    paths = seed_paths(args.seed)
    if args.force:
        if args.site != "all":
            raise ValueError("--force requires --site all to avoid concurrent cache deletion")
        for path in paths.values():
            path.unlink(missing_ok=True)
        for site_value in (0, 1):
            (REPORT_DIR / f"seed_{args.seed}_site{site_value}.json").unlink(
                missing_ok=True
            )
    count = len(positions)
    stores = {
        "coordinate": open_memmap(
            paths["coordinate"], np.float16, (count, PATH_SLOTS, 3)
        ),
        "raw": open_memmap(paths["raw"], np.float16, (count, PATH_SLOTS, 3)),
        "mass": open_memmap(paths["mass"], np.float16, (count, PATH_SLOTS)),
        "port": open_memmap(
            paths["port"], np.float16, (count, PATH_SLOTS, MP, N_UE)
        ),
        "mask": open_memmap(paths["mask"], np.bool_, (count, PATH_SLOTS)),
        "depth": open_memmap(paths["depth"], np.uint8, (count, PATH_SLOTS)),
        "path_count": open_memmap(paths["path_count"], np.int32, (count,)),
        "done": open_memmap(paths["done"], np.bool_, (count,)),
    }
    selected_sites = (0, 1) if args.site == "all" else (int(args.site),)
    target_rows = np.flatnonzero(np.isin(np.asarray(site), selected_sites))
    assigned_parts: list[np.ndarray] = []
    for site_value in selected_sites:
        local_rows = np.flatnonzero(np.asarray(site) == site_value)
        for batch_index, offset in enumerate(range(0, len(local_rows), args.batch_size)):
            if batch_index % args.batch_shards == args.batch_shard:
                assigned_parts.append(local_rows[offset : offset + args.batch_size])
    assigned_rows = (
        np.concatenate(assigned_parts)
        if assigned_parts
        else np.empty(0, dtype=np.int64)
    )
    if args.rewrite_selected:
        if args.site == "all":
            raise ValueError("--rewrite-selected requires an explicit --site")
        for name, value in stores.items():
            value[target_rows] = False if name == "done" else 0
            value.flush()
    report_path = (
        paths["report"]
        if args.site == "all"
        else REPORT_DIR
        / (
            f"seed_{args.seed}_site{args.site}.json"
            if args.batch_shards == 1
            else f"seed_{args.seed}_site{args.site}_shard{args.batch_shard}of{args.batch_shards}.json"
        )
    )
    if np.all(stores["done"][assigned_rows]):
        if report_path.is_file():
            return json.loads(report_path.read_text(encoding="utf-8"))
        status = "reused"
    else:
        status = "generated"

    prepare_scene()
    runtime = prepare_runtime(args)
    setup, _, _, _, _ = load_positions()
    started = time.time()
    material_values: dict[str, float] = {}
    processed_batches = 0
    stopped_early = False
    for site_value in selected_sites:
        scene = runtime["load_scene"](
            str(SCENE_XML), merge_shapes=True, remove_duplicate_vertices=True
        )
        scene.frequency = configuration.frequency_hz
        scene.tx_array = runtime["PlanarArray"](
            num_rows=8,
            num_cols=16,
            vertical_spacing=configuration.tx_vertical_spacing,
            horizontal_spacing=configuration.tx_horizontal_spacing,
            pattern="tr38901",
            polarization="VH",
        )
        scene.rx_array = runtime["PlanarArray"](
            num_rows=2,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="dipole",
            polarization="cross",
        )
        material_values = configure_materials(scene, configuration.scattering_coefficient)
        bs_position = setup.bs_positions[site_value]
        transmitter = runtime["Transmitter"](
            name=f"e195_tx_site{site_value}_seed{args.seed}",
            position=runtime["mi"].Point3f(*map(float, bs_position)),
            look_at=tx_target(
                bs_position,
                configuration.site_yaw_deg[site_value],
                configuration.site_tilt_deg[site_value],
                runtime["mi"],
            ),
            power_dbm=44.0,
        )
        scene.add(transmitter)
        solver = runtime["PathSolver"]()
        receiver_names: list[str] = []
        local_rows = np.flatnonzero(np.asarray(site) == site_value)
        for batch_index, offset in enumerate(range(0, len(local_rows), args.batch_size)):
            if batch_index % args.batch_shards != args.batch_shard:
                continue
            batch_rows = local_rows[offset : offset + args.batch_size]
            pending = batch_rows[~np.asarray(stores["done"][batch_rows], dtype=bool)]
            if len(pending) == 0:
                continue
            for name in receiver_names:
                scene.remove(name)
            receiver_names = []
            for row in pending:
                name = f"e195_rx_{int(row):06d}"
                scene.add(
                    runtime["Receiver"](
                        name=name,
                        position=runtime["mi"].Point3f(*map(float, positions[row])),
                        look_at=transmitter,
                    )
                )
                receiver_names.append(name)
            select_drjit_cuda_device(runtime["dr"], args.cuda_device)
            traced = solver(
                scene,
                max_depth=configuration.max_depth,
                max_num_paths_per_src=configuration.max_num_paths_per_src,
                samples_per_src=configuration.samples_per_src,
                synthetic_array=configuration.synthetic_array,
                los=True,
                specular_reflection=True,
                diffuse_reflection=configuration.diffuse_reflection,
                refraction=True,
                diffraction=configuration.diffraction,
                edge_diffraction=configuration.edge_diffraction,
                diffraction_lit_region=True,
                seed=int(args.seed + 100_000 * site_value + int(pending[0])),
            )
            compact = compact_path_batch(
                traced, pending, site_value, configuration
            )
            for name, value in compact.items():
                stores[name][pending] = value.astype(stores[name].dtype)
            stores["done"][pending] = True
            for value in stores.values():
                value.flush()
            del compact, traced
            runtime["dr"].sync_thread()
            completed = int(
                np.sum(np.asarray(stores["done"][assigned_rows], dtype=np.int64))
            )
            if completed == len(assigned_rows) or completed % 512 < len(pending):
                print(
                    f"E195 seed={args.seed} site={site_value} "
                    f"{completed}/{len(assigned_rows)} assigned "
                    f"({int(np.sum(np.asarray(stores['done'], dtype=np.int64)))}/{count} global) "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
            processed_batches += 1
            if args.max_batches > 0 and processed_batches >= args.max_batches:
                stopped_early = True
                break
        for name in receiver_names:
            scene.remove(name)
        del scene, solver
        runtime["dr"].sync_thread()
        if stopped_early:
            break

    if stopped_early:
        return {
            "experiment": VERSION,
            "status": "partial_smoke_complete",
            "seed": int(args.seed),
            "site": args.site,
            "batch_shard": int(args.batch_shard),
            "batch_shards": int(args.batch_shards),
            "completed_rows": int(
                np.sum(np.asarray(stores["done"][assigned_rows], dtype=np.int64))
            ),
            "total_rows": int(len(assigned_rows)),
            "processed_batches": int(processed_batches),
            "seconds": float(time.time() - started),
        }
    if not np.all(stores["done"][assigned_rows]):
        raise RuntimeError(
            f"E195 seed {args.seed} site {args.site} did not complete"
        )
    report = {
        "experiment": VERSION,
        "status": status,
        "seed": int(args.seed),
        "site": args.site,
        "batch_shard": int(args.batch_shard),
        "batch_shards": int(args.batch_shards),
        "configuration": asdict(configuration),
        "rows": int(len(assigned_rows)),
        "zero_path_fraction": float(
            np.mean(np.asarray(stores["path_count"][assigned_rows]) == 0)
        ),
        "mean_path_count": float(
            np.mean(np.asarray(stores["path_count"][assigned_rows]))
        ),
        "materials": material_values,
        "outputs": {
            name: PATHS.relative(path)
            for name, path in paths.items()
            if name != "report"
        },
        "performance": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_cuda_device": int(args.cuda_device),
            "peak_rss_gib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            ),
            "measured_bottleneck": "Sionna PathSolver GPU compute",
        },
        "seconds": float(time.time() - started),
    }
    atomic_json(report_path, report)
    return report


def aggregate_report() -> dict[str, Any]:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    runs = []
    for seed in SEEDS[:3]:
        path = seed_paths(seed)["report"]
        if not path.is_file():
            raise FileNotFoundError(path)
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    report = {
        **state,
        "status": "probe_generation_complete",
        "runs": runs,
        "all_done": bool(
            all(
                np.all(np.load(seed_paths(seed)["done"], mmap_mode="r"))
                for seed in SEEDS[:3]
            )
        ),
    }
    atomic_json(REPORT_DIR / "generation.json", report)
    return report


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.batch_shards <= 0:
        raise ValueError("--batch-shards must be positive")
    if not 0 <= args.batch_shard < args.batch_shards:
        raise ValueError("--batch-shard must be in [0, batch-shards)")
    if args.stage == "prepare":
        report = prepare_positions(args.force)
    elif args.stage == "seed":
        report = generate_seed(args)
    else:
        report = aggregate_report()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
