#!/usr/bin/env python3
"""Generate one resumable compact top-path cache for an E200 finalist."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback
from typing import Any

import numpy as np

from e200_config_search import SearchConfig, portable_payload
from project_paths import PATHS
from round2.data import load_positions
from screen_e200_sionna_config import prepare_scene_files


NAMESPACE = os.environ.get("HW_E200_NAMESPACE", "e200").strip().lower()
if NAMESPACE not in {"e200", "e201"}:
    raise ValueError(f"unsupported E200-compatible namespace: {NAMESPACE}")
VERSION = f"Round2-{NAMESPACE.upper()}-finalist-compact-paths"
ROOT = PATHS.artifacts / "round2" / f"{NAMESPACE}_finalist_paths"
REPORT_ROOT = PATHS.analysis / "results" / f"round2_{NAMESPACE}_finalist_paths"
FINALISTS = (
    PATHS.artifacts
    / "round2"
    / f"{NAMESPACE}_config_search"
    / "manifests"
    / "finalists.json"
)
POINTS = 4000
PATHS_PER_CONFIG = 32
STEPS = 16
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=int, choices=range(4), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--report-tag", default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(portable_payload(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def select_drjit_cuda_device(dr, device: int) -> None:
    library_path = Path(dr.__file__).with_name("libdrjit-core.so")
    library = ctypes.CDLL(str(library_path))
    setter = getattr(library, "_Z19jit_cuda_set_devicei")
    setter.argtypes = [ctypes.c_int]
    setter.restype = None
    setter(int(device))


def tx_target(position: np.ndarray, yaw_deg: float, tilt_deg: float, mi):
    radius = 140.0
    yaw = math.radians(yaw_deg)
    tilt = math.radians(tilt_deg)
    return mi.Point3f(
        float(position[0] + radius * math.cos(yaw)),
        float(position[1] + radius * math.sin(yaw)),
        float(position[2] + radius * math.tan(tilt)),
    )


def config_digest(config: SearchConfig) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def output_paths(config: SearchConfig, report_tag: str = "") -> dict[str, Path]:
    root = ROOT / f"source{safe_name(str(config.name))}"
    report_suffix = f"_{safe_name(report_tag)}" if report_tag else ""
    names = (
        "coefficient",
        "tau",
        "mask",
        "source",
        "seed",
        "angles",
        "energy",
        "energy_fraction",
        "presence",
        "interaction_count",
        "interactions",
        "objects",
        "primitives",
        "vertices",
        "done",
    )
    return {
        **{name: root / f"{name}.npy" for name in names},
        "root": root,
        "state": root / "state.json",
        "report": REPORT_ROOT / f"{safe_name(config.name)}{report_suffix}.json",
    }


def array_specs() -> dict[str, tuple[np.dtype, tuple[int, ...], Any]]:
    shape = (POINTS, PATHS_PER_CONFIG)
    return {
        "coefficient": (np.dtype(np.complex64), shape + (256, 4), 0.0),
        "tau": (np.dtype(np.float32), shape, 0.0),
        "mask": (np.dtype(bool), shape, False),
        "source": (np.dtype(np.int8), shape, -1),
        "seed": (np.dtype(np.int32), shape, -1),
        "angles": (np.dtype(np.float32), shape + (4,), 0.0),
        "energy": (np.dtype(np.float32), shape, 0.0),
        "energy_fraction": (np.dtype(np.float32), shape, 0.0),
        "presence": (np.dtype(np.uint8), shape, 0),
        "interaction_count": (np.dtype(np.uint8), shape, 0),
        "interactions": (np.dtype(np.uint8), shape + (STEPS,), 0),
        "objects": (np.dtype(np.uint32), shape + (STEPS,), 0),
        "primitives": (np.dtype(np.uint32), shape + (STEPS,), 0),
        "vertices": (np.dtype(np.float32), shape + (STEPS, 3), 0.0),
        "done": (np.dtype(bool), (POINTS,), False),
    }


def open_arrays(paths: dict[str, Path], force: bool) -> dict[str, np.memmap]:
    paths["root"].mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.memmap] = {}
    for name, (dtype, shape, fill) in array_specs().items():
        path = paths[name]
        if force:
            path.unlink(missing_ok=True)
        if path.is_file():
            value = np.load(path, mmap_mode="r+")
            if value.shape != shape or value.dtype != dtype:
                raise RuntimeError(
                    f"E200 compact cache mismatch {path}: {value.shape}/{value.dtype}"
                )
        else:
            value = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
            value[...] = fill
            value.flush()
        arrays[name] = value
    return arrays


def raw_coefficients(traced) -> np.ndarray:
    raw = traced.a
    if isinstance(raw, tuple):
        return (
            np.asarray(raw[0].numpy(), dtype=np.float32)
            + 1j * np.asarray(raw[1].numpy(), dtype=np.float32)
        ).astype(np.complex64)
    value = np.asarray(raw.numpy())
    if np.iscomplexobj(value):
        return value.astype(np.complex64)
    if value.ndim == 6 and value.shape[0] == 2:
        return (value[0] + 1j * value[1]).astype(np.complex64)
    raise ValueError(f"unexpected E200 coefficient representation {value.shape}")


def compact_batch(
    traced,
    point_ids: np.ndarray,
    source: int,
    seed: int,
    alignment: np.ndarray,
) -> dict[str, np.ndarray]:
    count = len(point_ids)
    shape = (count, PATHS_PER_CONFIG)
    output: dict[str, np.ndarray] = {
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
        "interactions": np.zeros(shape + (STEPS,), dtype=np.uint8),
        "objects": np.zeros(shape + (STEPS,), dtype=np.uint32),
        "primitives": np.zeros(shape + (STEPS,), dtype=np.uint32),
        "vertices": np.zeros(shape + (STEPS, 3), dtype=np.float32),
    }
    valid = np.asarray(traced.valid.numpy(), dtype=bool)
    tau = np.asarray(traced.tau.numpy(), dtype=np.float32)
    theta_t = np.asarray(traced.theta_t.numpy(), dtype=np.float32)
    phi_t = np.asarray(traced.phi_t.numpy(), dtype=np.float32)
    theta_r = np.asarray(traced.theta_r.numpy(), dtype=np.float32)
    phi_r = np.asarray(traced.phi_r.numpy(), dtype=np.float32)
    interactions = np.asarray(traced.interactions.numpy(), dtype=np.uint32)
    objects = np.asarray(traced.objects.numpy(), dtype=np.uint32)
    primitives = np.asarray(traced.primitives.numpy(), dtype=np.uint32)
    vertices = np.asarray(traced.vertices.numpy(), dtype=np.float32)
    coefficients = raw_coefficients(traced)
    vertical = np.arange(8, dtype=np.float32)

    for local in range(count):
        indices = np.flatnonzero(valid[local, 0])
        if not len(indices):
            continue
        local_coefficient = np.take(
            coefficients[local, :, 0, :, :], indices, axis=-1
        ).transpose(2, 1, 0)
        shaped = local_coefficient.reshape(len(indices), 2, 16, 8, 4)
        energy = np.square(np.abs(shaped), dtype=np.float32).sum(
            axis=(1, 2, 3, 4), dtype=np.float64
        )
        total = max(float(np.sum(energy, dtype=np.float64)), TINY)
        take = min(PATHS_PER_CONFIG, len(indices))
        selected_local = np.argpartition(energy, len(energy) - take)[-take:]
        selected_local = selected_local[
            np.argsort(energy[selected_local], kind="stable")[::-1]
        ]
        selected = indices[selected_local]
        local_coefficient = local_coefficient[selected_local]
        ramp = np.exp(
            2j * np.pi * float(alignment[local]) * vertical / 8.0
        ).astype(np.complex64)
        local_coefficient = (
            local_coefficient.reshape(take, 2, 16, 8, 4)
            * ramp[None, None, None, :, None]
        ).reshape(take, 256, 4)
        local_interactions = np.take(
            interactions[:, local, 0, :], selected, axis=-1
        ).T
        local_objects = np.take(objects[:, local, 0, :], selected, axis=-1).T
        local_primitives = np.take(
            primitives[:, local, 0, :], selected, axis=-1
        ).T
        local_vertices = np.take(
            vertices[:, local, 0, :, :], selected, axis=1
        ).transpose(1, 0, 2)
        depth = np.count_nonzero(local_interactions, axis=1).astype(np.uint8)
        step_count = min(local_interactions.shape[1], STEPS)
        slots = slice(0, take)
        output["coefficient"][local, slots] = local_coefficient
        output["tau"][local, slots] = tau[local, 0, selected]
        output["mask"][local, slots] = True
        output["source"][local, slots] = source
        output["seed"][local, slots] = seed
        output["angles"][local, slots] = np.stack(
            (
                theta_t[local, 0, selected],
                phi_t[local, 0, selected],
                theta_r[local, 0, selected],
                phi_r[local, 0, selected],
            ),
            axis=1,
        )
        output["energy"][local, slots] = energy[selected_local].astype(np.float32)
        output["energy_fraction"][local, slots] = (
            energy[selected_local] / total
        ).astype(np.float32)
        output["presence"][local, slots] = 1
        output["interaction_count"][local, slots] = np.minimum(depth, STEPS)
        output["interactions"][local, slots, :step_count] = local_interactions[
            :, :step_count
        ].astype(np.uint8)
        output["objects"][local, slots, :step_count] = local_objects[:, :step_count]
        output["primitives"][local, slots, :step_count] = local_primitives[
            :, :step_count
        ]
        output["vertices"][local, slots, :step_count] = local_vertices[:, :step_count]
    return output


def gpu_snapshot() -> str:
    import subprocess

    physical = os.environ.get("E200_PHYSICAL_GPU", "0")
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical}",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip()


def main() -> None:
    args = parse_args()
    started = time.time()
    finalists = [
        SearchConfig.from_dict(value)
        for value in json.loads(FINALISTS.read_text(encoding="utf-8"))
    ]
    config = finalists[args.source]
    digest = config_digest(config)
    paths = output_paths(config, args.report_tag)
    assigned_ids = (
        np.asarray(np.load(args.ids_file), dtype=np.int64)
        if args.ids_file is not None
        else np.arange(POINTS, dtype=np.int64)
    )
    assigned_ids = np.unique(assigned_ids)
    if np.any((assigned_ids < 0) | (assigned_ids >= POINTS)):
        raise ValueError("E200 assigned point IDs are out of range")
    assigned_digest = hashlib.sha256(assigned_ids.tobytes()).hexdigest()
    shard_mode = args.ids_file is not None
    if args.force:
        for path in paths.values():
            if isinstance(path, Path) and path.is_file():
                path.unlink()
    if paths["report"].is_file() and not args.force:
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        if (
            report.get("status") in ("complete", "shard_complete")
            and report.get("config_digest") == digest
            and report.get("assigned_digest") == assigned_digest
        ):
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
            return
    arrays = open_arrays(paths, args.force)
    if paths["state"].is_file():
        state = json.loads(paths["state"].read_text(encoding="utf-8"))
        if state.get("config_digest") != digest:
            raise RuntimeError("E200 finalist cache configuration changed")
    elif not shard_mode:
        state = {
            "experiment": VERSION,
            "status": "running",
            "source": args.source,
            "configuration": config.to_dict(),
            "config_digest": digest,
        }
        atomic_json(paths["state"], state)
    else:
        raise FileNotFoundError(
            "E200 shard mode requires an initialized shared finalist cache"
        )

    setup, train_positions, _, train_site, _ = load_positions()
    screen_name = safe_name(config.name)
    screen_report = json.loads(
        (
            PATHS.analysis
            / "results"
            / f"round2_{NAMESPACE}_config_search"
            / "configs"
            / f"{screen_name}.json"
        ).read_text(encoding="utf-8")
    )
    align_by_site = np.asarray(
        [
            screen_report["angle_alignment"][f"site{site}"]["vertical_shift"]
            for site in (0, 1)
        ],
        dtype=np.float32,
    )
    status = "failed"
    error = None
    site_seconds: dict[str, float] = {}
    scene_identity: dict[str, Any] = {}
    path_counts: list[np.ndarray] = []
    try:
        import drjit as dr
        import mitsuba as mi

        if mi.variant() is None:
            mi.set_variant(args.mitsuba_variant)
        select_drjit_cuda_device(dr, args.cuda_device)
        import sionna
        from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

        if sionna.__version__ != "2.0.1":
            raise RuntimeError(f"expected Sionna 2.0.1, got {sionna.__version__}")
        scene_path = prepare_scene_files(config)
        for site_value in (0, 1):
            full_site_ids = np.flatnonzero(train_site == site_value)
            assigned_mask = np.zeros(POINTS, dtype=bool)
            assigned_mask[assigned_ids] = True
            site_batches: list[np.ndarray] = []
            for offset in range(0, len(full_site_ids), args.batch_size):
                batch_ids = full_site_ids[offset : offset + args.batch_size]
                selected = assigned_mask[batch_ids]
                if not np.any(selected):
                    continue
                if shard_mode and not np.all(selected):
                    raise ValueError(
                        "E200 shard IDs must contain complete original site batches"
                    )
                site_batches.append(batch_ids)
            site_started = time.time()
            scene = load_scene(
                str(scene_path),
                merge_shapes=NAMESPACE != "e201",
                remove_duplicate_vertices=True,
            )
            if NAMESPACE == "e201" and len(scene.objects) != 565:
                raise RuntimeError(
                    f"E201 expected 565 unmerged scene objects, got {len(scene.objects)}"
                )
            scene_identity[f"site{site_value}"] = {
                "scene": PATHS.relative(scene_path),
                "merge_shapes": NAMESPACE != "e201",
                "object_count": int(len(scene.objects)),
                "materials": sorted(str(name) for name in scene.radio_materials),
            }
            scene.frequency = config.frequency_hz
            scene.tx_array = PlanarArray(
                num_rows=8,
                num_cols=16,
                vertical_spacing=config.tx_vertical_spacing,
                horizontal_spacing=config.tx_horizontal_spacing,
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
            if NAMESPACE == "e201":
                for material in scene.radio_materials.values():
                    material.scattering_coefficient = 0.0
            else:
                scene.radio_materials["building_material"].scattering_coefficient = (
                    config.building_scattering
                )
                scene.radio_materials["ground_material"].scattering_coefficient = (
                    config.ground_scattering
                )
            bs_position = setup.bs_positions[site_value]
            transmitter = Transmitter(
                name=f"tx_site{site_value}",
                position=mi.Point3f(*map(float, bs_position)),
                look_at=tx_target(
                    bs_position,
                    config.site_yaw_deg[site_value],
                    config.site_tilt_deg[site_value],
                    mi,
                ),
                power_dbm=44.0,
            )
            scene.add(transmitter)
            solver = PathSolver()
            receiver_names: list[str] = []
            for batch_ids in site_batches:
                pending = batch_ids[~np.asarray(arrays["done"][batch_ids], dtype=bool)]
                if not len(pending):
                    continue
                for name in receiver_names:
                    scene.remove(name)
                receiver_names = []
                for point_id in pending:
                    name = f"rx_{int(point_id):05d}"
                    receiver_args: dict[str, Any] = {
                        "name": name,
                        "position": mi.Point3f(*map(float, train_positions[point_id])),
                    }
                    if config.rx_orientation == "track_bs":
                        receiver_args["look_at"] = transmitter
                    else:
                        receiver_args["orientation"] = mi.Point3f(0.0, 0.0, 0.0)
                    scene.add(Receiver(**receiver_args))
                    receiver_names.append(name)
                select_drjit_cuda_device(dr, args.cuda_device)
                traced = solver(
                    scene,
                    max_depth=config.max_depth,
                    max_num_paths_per_src=config.max_num_paths_per_src,
                    samples_per_src=config.samples_per_src,
                    synthetic_array=config.synthetic_array,
                    los=config.los,
                    specular_reflection=config.specular_reflection,
                    diffuse_reflection=config.diffuse_reflection,
                    refraction=config.refraction,
                    diffraction=config.diffraction,
                    edge_diffraction=config.edge_diffraction,
                    diffraction_lit_region=True,
                    seed=config.seed + 100_000 * site_value + int(batch_ids[0]),
                )
                valid = np.asarray(traced.valid.numpy(), dtype=bool).reshape(
                    len(pending), -1
                )
                path_counts.append(valid.sum(axis=1).astype(np.int32))
                compact = compact_batch(
                    traced,
                    pending,
                    args.source,
                    config.seed,
                    np.full(len(pending), align_by_site[site_value], dtype=np.float32),
                )
                for name, value in compact.items():
                    arrays[name][pending] = value
                    arrays[name].flush()
                arrays["done"][pending] = True
                arrays["done"].flush()
                del traced, compact, valid
                dr.sync_thread()
                completed = int(np.sum(arrays["done"][assigned_ids]))
                if completed == len(assigned_ids) or completed % 320 < len(pending):
                    print(
                        f"E200 source={args.source} {config.name} "
                        f"{completed}/{len(assigned_ids)} elapsed={time.time() - started:.1f}s",
                        flush=True,
                    )
            for name in receiver_names:
                scene.remove(name)
            del scene, solver
            dr.sync_thread()
            site_seconds[f"site{site_value}"] = float(time.time() - site_started)
        if not bool(np.all(arrays["done"][assigned_ids])):
            raise RuntimeError(
                "E200 finalist assigned cache incomplete: "
                f"{int(np.sum(arrays['done'][assigned_ids]))}/{len(assigned_ids)}"
            )
        status = "shard_complete" if shard_mode else "complete"
    except BaseException:
        error = traceback.format_exc()
        raise
    finally:
        valid_per_point = np.asarray(arrays["mask"][assigned_ids], dtype=bool).sum(axis=1)
        report = {
            "experiment": VERSION,
            "status": status,
            "source": args.source,
            "configuration": config.to_dict(),
            "config_digest": digest,
            "assigned_digest": assigned_digest,
            "assigned_points": int(len(assigned_ids)),
            "alignment": {
                "horizontal_shift": [
                    int(
                        screen_report["angle_alignment"][f"site{site}"][
                            "horizontal_shift"
                        ]
                    )
                    for site in (0, 1)
                ],
                "vertical_shift": align_by_site.astype(int).tolist(),
                "selection_train_points": 3500,
                "fixed500_used": False,
            },
            "scene_identity": scene_identity,
            "output": {
                "root": PATHS.relative(paths["root"]),
                "points": int(len(assigned_ids)),
                "slots": PATHS_PER_CONFIG,
                "complete_points": int(np.sum(arrays["done"][assigned_ids])),
                "mean_valid_slots": float(np.mean(valid_per_point)),
                "zero_path_fraction": float(np.mean(valid_per_point == 0)),
            },
            "performance": {
                "requested_batch_size": args.batch_size,
                "gpu": gpu_snapshot(),
                "peak_rss_gib": float(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
                ),
                "site_seconds": site_seconds,
                "wall_seconds": float(time.time() - started),
                "measured_bottleneck": "Sionna PathSolver and compact coefficient extraction",
            },
            "traceback": error,
        }
        if not shard_mode:
            state["status"] = status
            state["complete_points"] = int(np.sum(arrays["done"]))
            state["traceback"] = error
            atomic_json(paths["state"], state)
        atomic_json(paths["report"], report)
        print(json.dumps(portable_payload(report), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
