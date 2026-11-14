#!/usr/bin/env python3
"""Build per-seed event histories from the raw Round 2 propagation HDF5 files."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import resource
import time

import h5py
import numpy as np
import torch

import analyze_path_candidate_information_audit as e84
import train_surface_interaction_energy_model as e85
from data import load_positions
from project_paths import PATHS


VERSION = "Round2-E192-multiseed-event-history"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e192_multiseed_path_generator"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e192_multiseed_path_generator"
ANCHOR_DIR = ARTIFACT_DIR / "event_anchors_train"
HISTORY_DIR = ARTIFACT_DIR / "event_history_train"
JOB_DIR = HISTORY_DIR / "jobs"
REPORT = ANALYSIS_DIR / "event_history_train_report.json"

SIONNA_ROOT = PATHS.artifacts / "round2" / "sionna_multiseed"
BASE_NAME = "primary_d16_base_v4lambda_1m"
SCATTER_NAME = "primary_d10_scatter015_v4lambda_1m"
BASE_ROOT = SIONNA_ROOT / BASE_NAME / "full" / "paths"
SCATTER_ROOT = SIONNA_ROOT / SCATTER_NAME / "full" / "paths"
BASE_STATE = SIONNA_ROOT / BASE_NAME / "full" / "state.json"

SEED_VALUES = (
    42,
    101,
    211,
    307,
    401,
    503,
    601,
    701,
    809,
    907,
    1009,
    1103,
    1201,
    1301,
    1409,
    1511,
    1601,
    1709,
    1801,
    1901,
    2003,
    2111,
    2203,
    2309,
)
N_SEEDS = len(SEED_VALUES)
MAX_EVENTS = 128
STEPS = 16
UINT32_SENTINEL = np.uint32(np.iinfo(np.uint32).max)


@dataclass(frozen=True)
class Config:
    workers: int = 8
    scatter_topk: int = 96
    max_events: int = MAX_EVENTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=Config().workers)
    parser.add_argument("--scatter-topk", type=int, default=Config().scatter_topk)
    parser.add_argument(
        "--stage", choices=("anchors", "history", "all"), default="all"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sequence_hash(
    source: np.ndarray,
    depth: np.ndarray,
    interactions: np.ndarray,
    objects: np.ndarray,
    primitives: np.ndarray,
) -> np.ndarray:
    result = np.full(len(depth), np.uint64(1469598103934665603), dtype=np.uint64)
    result ^= np.asarray(source + 2, dtype=np.uint64)
    result *= np.uint64(1099511628211)
    result ^= np.asarray(depth, dtype=np.uint64)
    result *= np.uint64(1099511628211)
    for step in range(interactions.shape[1]):
        active = step < depth
        value = (
            interactions[:, step].astype(np.uint64)
            ^ (objects[:, step].astype(np.uint64) * np.uint64(0x9E3779B185EBCA87))
            ^ (
                primitives[:, step].astype(np.uint64)
                * np.uint64(0xC2B2AE3D27D4EB4F)
            )
            ^ np.uint64(step + 1)
        )
        updated = (result ^ value) * np.uint64(1099511628211)
        result = np.where(active, updated, result)
    return result


def _raw_coordinates(
    tau: np.ndarray,
    theta_t: np.ndarray,
    phi_t: np.ndarray,
    configuration: dict[str, object],
    site: int,
) -> np.ndarray:
    yaw = math.radians(float(configuration["site_yaw_deg"][site]))
    tilt = math.radians(float(configuration["site_tilt_deg"][site]))
    horizontal_spacing = float(configuration["tx_horizontal_spacing"])
    vertical_spacing = float(configuration["tx_vertical_spacing"])
    horizontal = (
        16.0
        * horizontal_spacing
        * np.sin(theta_t)
        * np.sin(phi_t - yaw)
    ) % 16.0
    vertical = (
        8.0
        * vertical_spacing
        * (
            math.sin(tilt) * np.sin(theta_t) * np.cos(phi_t - yaw)
            - math.cos(tilt) * np.cos(theta_t)
        )
    ) % 8.0
    delay = (tau * 30_000.0 * 192.0) % 192.0
    return np.stack((delay, horizontal, vertical), axis=1).astype(np.float32)


def _rounded_coordinates(coordinate: np.ndarray, source: int) -> np.ndarray:
    scale = 50.0 if source == 0 else 4.0
    return np.rint(coordinate * scale).astype(np.int16)


def _anchor_key(
    source: int,
    rounded: np.ndarray,
    sequence_hash: int,
    depth: int,
    port_class: int,
) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(source),
        int(rounded[0]),
        int(rounded[1]),
        int(rounded[2]),
        int(sequence_hash) if source == 0 else 0,
        min(int(depth), 15) if source == 1 else 0,
        int(port_class) if source == 1 else -1,
    )


def _required_anchor_inputs() -> tuple[Path, ...]:
    return (
        e85.PATH_CACHE / "mask.npy",
        e85.PATH_CACHE / "source.npy",
        e85.PATH_CACHE / "tau.npy",
        e85.PATH_CACHE / "angles.npy",
        e85.PATH_CACHE / "energy_fraction.npy",
        e85.PATH_CACHE / "interaction_count.npy",
        e85.PATH_CACHE / "interactions.npy",
        e85.PATH_CACHE / "objects.npy",
        e85.PATH_CACHE / "primitives.npy",
        e85.PATH_CACHE / "coefficient.npy",
        BASE_STATE,
    )


def _exact_port_classes(mask: np.ndarray, source: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    started = time.time()
    coefficient = np.load(e85.PATH_CACHE / "coefficient.npy", mmap_mode="r")
    output = np.full(source.shape, -1, dtype=np.int8)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(source), 128):
        stop = min(start + 128, len(source))
        active = (np.asarray(source[start:stop]) == 1) & np.asarray(mask[start:stop])
        if not np.any(active):
            continue
        value = torch.as_tensor(
            np.array(coefficient[start:stop], dtype=np.complex64, copy=True),
            device=device,
        ).reshape(stop - start, MAX_EVENTS, 2, 16, 8, 4)
        power = value.abs().square().sum(dim=(3, 4)).reshape(
            stop - start, MAX_EVENTS, 8
        )
        local = power.argmax(dim=2).cpu().numpy().astype(np.int8)
        output[start:stop][active] = local[active]
    return output, {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "seconds": float(time.time() - started),
        "peak_gpu_gib": float(torch.cuda.max_memory_allocated() / 2**30)
        if device.type == "cuda"
        else 0.0,
    }


def build_anchors(force: bool) -> dict[str, object]:
    started = time.time()
    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "mask": ANCHOR_DIR / "mask.npy",
        "source": ANCHOR_DIR / "source.npy",
        "representative_slot": ANCHOR_DIR / "representative_slot.npy",
        "rounded_coordinate": ANCHOR_DIR / "rounded_coordinate.npy",
        "sequence_hash": ANCHOR_DIR / "sequence_hash.npy",
        "depth": ANCHOR_DIR / "depth.npy",
        "port_class": ANCHOR_DIR / "port_class.npy",
        "event_count": ANCHOR_DIR / "event_count.npy",
    }
    if not force and all(path.is_file() for path in outputs.values()):
        event_count = np.load(outputs["event_count"], mmap_mode="r")
        return {
            "reused": True,
            "seconds": float(time.time() - started),
            "event_count_mean": float(np.mean(event_count)),
            "event_count_median": float(np.median(event_count)),
            "event_count_max": int(np.max(event_count)),
        }

    missing = [PATHS.relative(path) for path in _required_anchor_inputs() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing E192 anchor inputs: {missing}")

    mask = np.load(e85.PATH_CACHE / "mask.npy", mmap_mode="r")
    source = np.load(e85.PATH_CACHE / "source.npy", mmap_mode="r")
    tau = np.load(e85.PATH_CACHE / "tau.npy", mmap_mode="r")
    angles = np.load(e85.PATH_CACHE / "angles.npy", mmap_mode="r")
    fraction = np.load(e85.PATH_CACHE / "energy_fraction.npy", mmap_mode="r")
    depth = np.load(e85.PATH_CACHE / "interaction_count.npy", mmap_mode="r")
    interactions = np.load(e85.PATH_CACHE / "interactions.npy", mmap_mode="r")
    objects = np.load(e85.PATH_CACHE / "objects.npy", mmap_mode="r")
    primitives = np.load(e85.PATH_CACHE / "primitives.npy", mmap_mode="r")
    exact_port_class, port_class_report = _exact_port_classes(mask, source)
    state = json.loads(BASE_STATE.read_text(encoding="utf-8"))
    _, _, _, sites, _ = load_positions()

    count = len(mask)
    event_mask = np.zeros((count, MAX_EVENTS), dtype=bool)
    event_source = np.full((count, MAX_EVENTS), -1, dtype=np.int8)
    representative = np.full((count, MAX_EVENTS), -1, dtype=np.int16)
    event_rounded = np.zeros((count, MAX_EVENTS, 3), dtype=np.int16)
    event_hash = np.zeros((count, MAX_EVENTS), dtype=np.uint64)
    event_depth = np.zeros((count, MAX_EVENTS), dtype=np.uint8)
    event_port_class = np.full((count, MAX_EVENTS), -1, dtype=np.int8)
    event_count = np.zeros(count, dtype=np.int16)

    sequence_hash = e84.sequence_hashes(
        {
            "source": source,
            "interaction_count": depth,
            "interactions": interactions,
            "objects": objects,
            "primitives": primitives,
        }
    )
    for row in range(count):
        valid = np.flatnonzero(mask[row])
        if len(valid) == 0:
            continue
        coordinate = _raw_coordinates(
            np.asarray(tau[row, valid], dtype=np.float32),
            np.asarray(angles[row, valid, 0], dtype=np.float32),
            np.asarray(angles[row, valid, 1], dtype=np.float32),
            state["configuration"],
            int(sites[row]),
        )
        best: dict[tuple[int, int, int, int, int, int, int], tuple[int, float]] = {}
        metadata: dict[
            tuple[int, int, int, int, int, int, int],
            tuple[np.ndarray, int, int, int],
        ] = {}
        for local, slot_value in enumerate(valid):
            slot = int(slot_value)
            local_source = int(source[row, slot])
            rounded = _rounded_coordinates(coordinate[local : local + 1], local_source)[0]
            local_port_class = int(exact_port_class[row, slot]) if local_source == 1 else -1
            key = _anchor_key(
                local_source,
                rounded,
                int(sequence_hash[row, slot]),
                int(depth[row, slot]),
                local_port_class,
            )
            score = float(fraction[row, slot])
            previous = best.get(key)
            if previous is None or score > previous[1]:
                best[key] = (slot, score)
                metadata[key] = (
                    rounded,
                    int(sequence_hash[row, slot]),
                    int(depth[row, slot]),
                    local_port_class,
                )
        ordered = sorted(best, key=lambda key: best[key][1], reverse=True)[:MAX_EVENTS]
        event_count[row] = len(ordered)
        for event, key in enumerate(ordered):
            slot = best[key][0]
            rounded, local_hash, local_depth, local_port_class = metadata[key]
            event_mask[row, event] = True
            event_source[row, event] = int(source[row, slot])
            representative[row, event] = slot
            event_rounded[row, event] = rounded
            event_hash[row, event] = np.uint64(local_hash if key[0] == 0 else 0)
            event_depth[row, event] = np.uint8(local_depth)
            event_port_class[row, event] = np.int8(local_port_class)

    np.save(outputs["mask"], event_mask)
    np.save(outputs["source"], event_source)
    np.save(outputs["representative_slot"], representative)
    np.save(outputs["rounded_coordinate"], event_rounded)
    np.save(outputs["sequence_hash"], event_hash)
    np.save(outputs["depth"], event_depth)
    np.save(outputs["port_class"], event_port_class)
    np.save(outputs["event_count"], event_count)
    return {
        "reused": False,
        "seconds": float(time.time() - started),
        "event_count_mean": float(np.mean(event_count)),
        "event_count_median": float(np.median(event_count)),
        "event_count_p90": float(np.quantile(event_count, 0.90)),
        "event_count_min": int(np.min(event_count)),
        "event_count_max": int(np.max(event_count)),
        "raw_candidate_count_mean": float(np.mean(np.sum(mask, axis=1))),
        "duplicates_removed": int(np.sum(mask) - np.sum(event_mask)),
        "points_with_duplicate": float(
            np.mean(np.sum(mask, axis=1) > event_count)
        ),
        "exact_port_class": port_class_report,
    }


HISTORY_SPECS: dict[str, tuple[np.dtype | str, tuple[int, ...], object]] = {
    "mask": (np.uint8, (), 0),
    "continuous": (np.float32, (7,), 0.0),
    "reference_coefficient": (np.complex64, (2, 4), 0.0),
    "port_energy": (np.float32, (2, 4), 0.0),
    "interaction_count": (np.uint8, (), 0),
    "interactions": (np.uint8, (STEPS,), 0),
    "objects": (np.uint32, (STEPS,), UINT32_SENTINEL),
    "primitives": (np.uint32, (STEPS,), UINT32_SENTINEL),
    "vertices": (np.float16, (STEPS, 3), 0.0),
}


def _history_paths() -> dict[str, Path]:
    return {name: HISTORY_DIR / f"{name}.npy" for name in HISTORY_SPECS}


def _job_path(source: str, seed: int) -> Path:
    return JOB_DIR / f"{source}_seed{seed}.npz"


def _create_history_arrays(count: int, force: bool) -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    for name, (dtype, tail, fill) in HISTORY_SPECS.items():
        path = _history_paths()[name]
        if path.exists() and not force:
            continue
        output = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=dtype,
            shape=(N_SEEDS, count, MAX_EVENTS, *tail),
        )
        if fill != 0 and fill != 0.0:
            output[...] = fill
        else:
            output[...] = 0
        output.flush()
        del output


def _load_anchor_maps() -> tuple[list[dict[tuple[int, int, int, int, int, int, int], int]], int]:
    mask = np.load(ANCHOR_DIR / "mask.npy", mmap_mode="r")
    source = np.load(ANCHOR_DIR / "source.npy", mmap_mode="r")
    rounded = np.load(ANCHOR_DIR / "rounded_coordinate.npy", mmap_mode="r")
    sequence_hash = np.load(ANCHOR_DIR / "sequence_hash.npy", mmap_mode="r")
    depth = np.load(ANCHOR_DIR / "depth.npy", mmap_mode="r")
    port_class = np.load(ANCHOR_DIR / "port_class.npy", mmap_mode="r")
    maps: list[dict[tuple[int, int, int, int, int, int, int], int]] = []
    for row in range(len(mask)):
        local: dict[tuple[int, int, int, int, int, int, int], int] = {}
        for event_value in np.flatnonzero(mask[row]):
            event = int(event_value)
            key = _anchor_key(
                int(source[row, event]),
                np.asarray(rounded[row, event]),
                int(sequence_hash[row, event]),
                int(depth[row, event]),
                int(port_class[row, event]),
            )
            if key in local:
                raise RuntimeError(f"duplicate E192 anchor key row={row} event={event}")
            local[key] = event
        maps.append(local)
    return maps, len(mask)


def _seed_file(root: Path, seed: int) -> Path:
    path = root / f"train_seed{seed}.h5"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _worker(job: tuple[int, int, str, int]) -> dict[str, object]:
    seed_index, seed, source_name, scatter_topk = job
    source = 0 if source_name == "base" else 1
    root = BASE_ROOT if source == 0 else SCATTER_ROOT
    path = _seed_file(root, seed)
    started = time.time()
    anchor_maps, count = _load_anchor_maps()
    local: dict[str, np.ndarray] = {}
    for name, (dtype, tail, fill) in HISTORY_SPECS.items():
        array = np.empty((count, MAX_EVENTS, *tail), dtype=dtype)
        array[...] = fill
        local[name] = array

    total_paths = 0
    selected_paths = 0
    matched_records = 0
    matched_events = 0
    selected_mass = 0.0
    total_mass = 0.0
    with h5py.File(path, "r") as handle:
        configuration = json.loads(str(handle.attrs["configuration"]))
        for site_name in handle:
            site = int(site_name.replace("site", ""))
            for batch in handle[site_name].values():
                point_ids = np.asarray(batch["point_ids"], dtype=np.int64)
                offsets = np.asarray(batch["offsets"], dtype=np.int64)
                fraction_all = np.asarray(batch["path_energy_fraction"], dtype=np.float64)
                total_paths += len(fraction_all)
                total_mass += float(np.sum(fraction_all))
                tau_all = np.asarray(batch["tau"], dtype=np.float32)
                theta_t_all = np.asarray(batch["theta_t"], dtype=np.float32)
                phi_t_all = np.asarray(batch["phi_t"], dtype=np.float32)
                theta_r_all = np.asarray(batch["theta_r"], dtype=np.float32)
                phi_r_all = np.asarray(batch["phi_r"], dtype=np.float32)
                energy_all = np.asarray(batch["path_energy"], dtype=np.float32)
                reference_all = np.asarray(
                    batch["reference_coefficient"], dtype=np.complex64
                )
                port_energy_all = np.asarray(batch["port_energy"], dtype=np.float32)
                depth_all = np.asarray(batch["interaction_count"], dtype=np.uint8)
                interactions_all = np.asarray(batch["interactions"], dtype=np.uint8)
                objects_all = np.asarray(batch["objects"], dtype=np.uint32)
                primitives_all = np.asarray(batch["primitives"], dtype=np.uint32)
                vertices_all = np.asarray(batch["vertices"], dtype=np.float32)

                for local_row, point_id_value in enumerate(point_ids):
                    point_id = int(point_id_value)
                    start = int(offsets[local_row])
                    stop = int(offsets[local_row + 1])
                    if stop <= start or not anchor_maps[point_id]:
                        continue
                    if source == 1 and stop - start > scatter_topk:
                        fraction = fraction_all[start:stop]
                        chosen = np.argpartition(
                            fraction, len(fraction) - scatter_topk
                        )[-scatter_topk:]
                        indices = start + np.sort(chosen.astype(np.int64))
                    else:
                        indices = np.arange(start, stop, dtype=np.int64)
                    selected_paths += len(indices)
                    selected_mass += float(np.sum(fraction_all[indices]))
                    coordinate = _raw_coordinates(
                        tau_all[indices],
                        theta_t_all[indices],
                        phi_t_all[indices],
                        configuration,
                        site,
                    )
                    rounded = _rounded_coordinates(coordinate, source)
                    if source == 0:
                        hashes = _sequence_hash(
                            np.zeros(len(indices), dtype=np.int8),
                            depth_all[indices],
                            interactions_all[indices],
                            objects_all[indices],
                            primitives_all[indices],
                        )
                        port_classes = np.full(len(indices), -1, dtype=np.int8)
                    else:
                        hashes = np.zeros(len(indices), dtype=np.uint64)
                        port_classes = np.argmax(
                            np.abs(reference_all[indices].reshape(len(indices), -1)) ** 2,
                            axis=1,
                        ).astype(np.int8)
                    best: dict[int, int] = {}
                    for record, raw_index_value in enumerate(indices):
                        raw_index = int(raw_index_value)
                        key = _anchor_key(
                            source,
                            rounded[record],
                            int(hashes[record]),
                            int(depth_all[raw_index]),
                            int(port_classes[record]),
                        )
                        event = anchor_maps[point_id].get(key)
                        if event is None:
                            continue
                        previous = best.get(event)
                        if previous is None or fraction_all[raw_index] > fraction_all[previous]:
                            best[event] = raw_index
                    matched_records += len(best)
                    for event, raw_index in best.items():
                        local["mask"][point_id, event] = 1
                        local["continuous"][point_id, event] = (
                            tau_all[raw_index],
                            theta_t_all[raw_index],
                            phi_t_all[raw_index],
                            theta_r_all[raw_index],
                            phi_r_all[raw_index],
                            energy_all[raw_index],
                            fraction_all[raw_index],
                        )
                        local["reference_coefficient"][point_id, event] = reference_all[
                            raw_index
                        ]
                        local["port_energy"][point_id, event] = port_energy_all[raw_index]
                        local["interaction_count"][point_id, event] = depth_all[raw_index]
                        width = min(STEPS, interactions_all.shape[1])
                        local["interactions"][point_id, event, :width] = interactions_all[
                            raw_index, :width
                        ]
                        local["objects"][point_id, event, :width] = objects_all[
                            raw_index, :width
                        ]
                        local["primitives"][point_id, event, :width] = primitives_all[
                            raw_index, :width
                        ]
                        local["vertices"][point_id, event, :width] = vertices_all[
                            raw_index, :width
                        ].astype(np.float16)
                    matched_events += len(best)

    flat_index = np.flatnonzero(local["mask"].reshape(-1)).astype(np.int32)
    payload: dict[str, np.ndarray] = {"flat_index": flat_index}
    for name, (_, tail, _) in HISTORY_SPECS.items():
        if name == "mask":
            continue
        payload[name] = local[name].reshape(count * MAX_EVENTS, *tail)[flat_index]
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    job_path = _job_path(source_name, seed)
    np.savez(job_path, **payload)
    result = {
        "seed_index": seed_index,
        "seed": seed,
        "source": source_name,
        "seconds": float(time.time() - started),
        "total_paths": int(total_paths),
        "selected_paths": int(selected_paths),
        "matched_records": int(matched_records),
        "matched_events": int(matched_events),
        "selected_mass_ratio": float(selected_mass / max(total_mass, 1e-30)),
        "rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20),
        "job_file": PATHS.relative(job_path),
        "job_bytes": int(job_path.stat().st_size),
    }
    return result


def _assemble_job(
    outputs: dict[str, np.memmap],
    result: dict[str, object],
    count: int,
) -> None:
    seed_index = int(result["seed_index"])
    job_path = _job_path(str(result["source"]), int(result["seed"]))
    with np.load(job_path, allow_pickle=False) as payload:
        flat_index = payload["flat_index"].astype(np.int64)
        outputs["mask"][seed_index].reshape(count * MAX_EVENTS)[flat_index] = 1
        for name, (_, tail, _) in HISTORY_SPECS.items():
            if name == "mask":
                continue
            destination = outputs[name][seed_index].reshape(
                count * MAX_EVENTS, *tail
            )
            destination[flat_index] = payload[name]


def build_history(config: Config, force: bool) -> dict[str, object]:
    started = time.time()
    complete_path = HISTORY_DIR / "complete.json"
    if complete_path.is_file() and not force:
        payload = json.loads(complete_path.read_text(encoding="utf-8"))
        payload["reused"] = True
        return payload

    anchor_mask = np.load(ANCHOR_DIR / "mask.npy", mmap_mode="r")
    _create_history_arrays(len(anchor_mask), force=True)
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        name: np.lib.format.open_memmap(path, mode="r+")
        for name, path in _history_paths().items()
    }
    jobs = [
        (seed_index, seed, source, config.scatter_topk)
        for source in ("base", "scatter")
        for seed_index, seed in enumerate(SEED_VALUES)
    ]
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = {executor.submit(_worker, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            _assemble_job(outputs, result, len(anchor_mask))
            results.append(result)
            print(
                "E192 history "
                f"{index}/{len(jobs)} source={result['source']} seed={result['seed']} "
                f"seconds={result['seconds']:.2f} matched={result['matched_events']}",
                flush=True,
            )

    for output in outputs.values():
        output.flush()
    del outputs

    history_mask = np.load(HISTORY_DIR / "mask.npy", mmap_mode="r")
    presence = np.sum(history_mask, axis=0, dtype=np.int16)
    event_mask = np.load(ANCHOR_DIR / "mask.npy", mmap_mode="r")
    source = np.load(ANCHOR_DIR / "source.npy", mmap_mode="r")
    active = event_mask & (presence > 0)
    report = {
        "experiment": VERSION,
        "status": "history_complete",
        "reused": False,
        "configuration": asdict(config),
        "seconds": float(time.time() - started),
        "worker_results": sorted(results, key=lambda row: (str(row["source"]), int(row["seed_index"]))),
        "event_history": {
            "anchor_events": int(np.sum(event_mask)),
            "matched_events": int(np.sum(active)),
            "matched_event_ratio": float(np.sum(active) / max(int(np.sum(event_mask)), 1)),
            "presence_mean": float(np.mean(presence[active])) if np.any(active) else 0.0,
            "presence_median": float(np.median(presence[active])) if np.any(active) else 0.0,
            "presence_p90": float(np.quantile(presence[active], 0.90)) if np.any(active) else 0.0,
            "base_presence_mean": float(np.mean(presence[active & (source == 0)]))
            if np.any(active & (source == 0))
            else 0.0,
            "scatter_presence_mean": float(np.mean(presence[active & (source == 1)]))
            if np.any(active & (source == 1))
            else 0.0,
        },
        "files": {name: PATHS.relative(path) for name, path in _history_paths().items()},
    }
    complete_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    config = Config(workers=max(1, args.workers), scatter_topk=args.scatter_topk)
    anchor_report = build_anchors(args.force)
    if args.stage == "anchors":
        payload = {"experiment": VERSION, "anchors": anchor_report}
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return
    history_report = build_history(config, args.force)
    report = {
        "experiment": VERSION,
        "status": history_report["status"],
        "anchors": anchor_report,
        "history": history_report,
        "wall_seconds": float(time.time() - started),
        "host": {
            "cpu_count": os.cpu_count(),
            "peak_rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20),
        },
        "bottleneck": (
            "HDF5 LZF decompression and ragged path selection on CPU; the output is a "
            "single reusable tensor cache so model training never rereads HDF5."
        ),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
