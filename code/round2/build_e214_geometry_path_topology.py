#!/usr/bin/env python3
"""Export coefficient-free path topology for the E214 four-way experiment."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback
import zipfile

import numpy as np

from data import load_positions
from project_paths import PATHS


VERSION = "Round2-E214-geometry-path-topology"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e214_geometry_path_topology"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e214_geometry_path_topology"
RAW_ROOT = (
    PATHS.root
    / "final_code"
    / "artifacts"
    / "round2"
    / "physical_ai_retrained"
    / "propagation"
    / "train"
)

SEEDS = (
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

ALLOWED_FIELDS = (
    "point_ids",
    "offsets",
    "tau",
    "path_length_m",
    "geometric_length_m",
    "angles",
    "interactions",
    "objects",
    "primitives",
    "surface_plane",
    "vertices",
    "interaction_count",
    "geometry_valid",
    "seed",
    "source",
)
FORBIDDEN_FIELDS = (
    "absolute_energy",
    "relative_energy",
    "complex_coefficient",
    "full_complex_coefficient",
    "energy_rank",
)

POINT_COUNT = 4000
PATH_CAPACITY = 128
STEPS = 16
MH = 16
MV = 8
S_SC = 192
SINGLE_SEED = 42
SOURCE_ID = {"base": 0, "scatter": 1}
PER_SEED_LIMIT = {"base": 128, "scatter": 192}
SITE_YAW_DEG = np.asarray((-44.0, 73.5), dtype=np.float32)
SITE_TILT_DEG = np.asarray((-11.5, -10.5), dtype=np.float32)
TX_HORIZONTAL_SPACING = 0.5
TX_VERTICAL_SPACING = 4.0


@dataclass
class AccessAudit:
    shards: int = 0
    requested_members: int = 0
    compressed_bytes: int = 0
    uncompressed_bytes: int = 0

    def add(self, path: Path, fields: Iterable[str]) -> None:
        requested = tuple(fields)
        with zipfile.ZipFile(path) as archive:
            members = {item.filename.removesuffix(".npy"): item for item in archive.infolist()}
        missing = [field for field in requested if field not in members]
        if missing:
            raise KeyError(f"missing allowed fields in {path}: {missing}")
        self.shards += 1
        self.requested_members += len(requested)
        self.compressed_bytes += sum(members[field].compress_size for field in requested)
        self.uncompressed_bytes += sum(members[field].file_size for field in requested)

    def merge(self, other: "AccessAudit") -> None:
        self.shards += other.shards
        self.requested_members += other.requested_members
        self.compressed_bytes += other.compressed_bytes
        self.uncompressed_bytes += other.uncompressed_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("export", "merge", "all", "audit"), required=True)
    parser.add_argument("--source", choices=("base", "scatter", "all"), default="all")
    parser.add_argument("--workers", type=int, default=4)
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


def batch_output(source: str, seed_mode: str, site: int, name: str) -> Path:
    return ARTIFACT_DIR / "sanitized_batches" / f"{source}_{seed_mode}" / f"site_{site}" / name


def merged_dir(source: str, seed_mode: str) -> Path:
    return ARTIFACT_DIR / "topology" / f"{source}_{seed_mode}"


def report_path(source: str, stage: str) -> Path:
    return ANALYSIS_DIR / f"{stage}_{source}.json"


def failure_path(source: str, stage: str) -> Path:
    return ANALYSIS_DIR / f"failure_{stage}_{source}.json"


def raw_path(source: str, seed: int, site: int, name: str) -> Path:
    return RAW_ROOT / source / f"seed_{seed:04d}" / f"site_{site}" / name


def load_allowed(path: Path, audit: AccessAudit) -> dict[str, np.ndarray]:
    audit.add(path, ALLOWED_FIELDS)
    with np.load(path, allow_pickle=False) as archive:
        # This explicit list is the data-lineage boundary. Never replace it with archive.files.
        arrays = {field: np.array(archive[field], copy=True) for field in ALLOWED_FIELDS}
    return arrays


def mix_hash(value: np.ndarray, item: np.ndarray | int) -> np.ndarray:
    local = np.asarray(item, dtype=np.uint64)
    return (value ^ (local + np.uint64(0x9E3779B97F4A7C15))) * np.uint64(
        0xBF58476D1CE4E5B9
    )


def path_coordinates(tau: np.ndarray, angles: np.ndarray, site: int) -> np.ndarray:
    yaw = math.radians(float(SITE_YAW_DEG[site]))
    tilt = math.radians(float(SITE_TILT_DEG[site]))
    theta_t = np.asarray(angles[:, 0], dtype=np.float32)
    phi_t = np.asarray(angles[:, 1], dtype=np.float32)
    horizontal = (
        MH * TX_HORIZONTAL_SPACING * np.sin(theta_t) * np.sin(phi_t - yaw)
    ) % MH
    vertical = (
        MV
        * TX_VERTICAL_SPACING
        * (
            math.sin(tilt) * np.sin(theta_t) * np.cos(phi_t - yaw)
            - math.cos(tilt) * np.cos(theta_t)
        )
        + 2.0
    ) % MV
    delay = (np.asarray(tau, dtype=np.float32) * 30_000.0 * S_SC) % S_SC
    return np.stack((horizontal, vertical, delay), axis=1).astype(np.float32)


def family_hashes(
    source: str,
    coordinate: np.ndarray,
    depth: np.ndarray,
    interactions: np.ndarray,
    objects: np.ndarray,
    primitives: np.ndarray,
    planes: np.ndarray,
) -> np.ndarray:
    count = len(coordinate)
    value = np.full(count, np.uint64(SOURCE_ID[source] + 17), dtype=np.uint64)
    value = mix_hash(value, depth)
    value = mix_hash(value, np.floor(coordinate[:, 0] * 2.0).astype(np.int64))
    value = mix_hash(value, np.floor(coordinate[:, 1] * 2.0).astype(np.int64))
    value = mix_hash(value, np.floor(coordinate[:, 2] / 2.0).astype(np.int64))
    for step in range(min(4, interactions.shape[1])):
        active = step < depth
        interaction = np.where(active, interactions[:, step], 0)
        object_id = np.where(active, objects[:, step], 0)
        value = mix_hash(value, interaction)
        value = mix_hash(value, object_id)
        if source == "base":
            primitive_id = np.where(active, primitives[:, step], 0)
            value = mix_hash(value, primitive_id)
        else:
            plane = np.where(active[:, None], planes[:, step], 0)
            for axis in range(4):
                quantized = np.floor_divide(plane[:, axis].astype(np.int64), 2500)
                value = mix_hash(value, quantized)
    return value


def valid_local_indices(arrays: dict[str, np.ndarray], start: int, end: int) -> np.ndarray:
    local = np.arange(start, end, dtype=np.int64)
    finite = (
        np.isfinite(arrays["tau"][local])
        & np.isfinite(arrays["path_length_m"][local])
        & np.isfinite(arrays["geometric_length_m"][local])
        & np.all(np.isfinite(arrays["angles"][local]), axis=1)
    )
    valid = arrays["geometry_valid"][local].astype(bool) & finite
    valid &= arrays["interaction_count"][local] <= arrays["interactions"].shape[1]
    valid &= arrays["tau"][local] >= 0.0
    valid &= arrays["path_length_m"][local] > 0.0
    return local[valid]


def local_candidates(
    source: str,
    arrays: dict[str, np.ndarray],
    point_index: int,
    site: int,
) -> list[dict[str, object]]:
    start = int(arrays["offsets"][point_index])
    end = int(arrays["offsets"][point_index + 1])
    indices = valid_local_indices(arrays, start, end)
    if len(indices) == 0:
        return []
    coordinate = path_coordinates(arrays["tau"][indices], arrays["angles"][indices], site)
    depth = arrays["interaction_count"][indices].astype(np.int64)
    hashes = family_hashes(
        source,
        coordinate,
        depth,
        arrays["interactions"][indices],
        arrays["objects"][indices],
        arrays["primitives"][indices],
        arrays["surface_plane"][indices],
    )
    length = arrays["geometric_length_m"][indices].astype(np.float64)
    score = depth.astype(np.float64) + np.log1p(np.maximum(length, 0.0)) / 8.0
    order = np.lexsort((hashes, length, depth, score))
    selected: list[int] = []
    seen: set[int] = set()
    for local_index in order:
        key = int(hashes[local_index])
        if key in seen:
            continue
        seen.add(key)
        selected.append(int(local_index))
        if len(selected) >= PER_SEED_LIMIT[source]:
            break

    records: list[dict[str, object]] = []
    for local_index in selected:
        index = int(indices[local_index])
        local_depth = min(int(depth[local_index]), STEPS)
        interactions = np.zeros(STEPS, dtype=np.uint8)
        objects = np.full(STEPS, np.iinfo(np.uint32).max, dtype=np.uint32)
        primitives = np.full(STEPS, np.iinfo(np.uint32).max, dtype=np.uint32)
        planes = np.zeros((STEPS, 4), dtype=np.int32)
        vertices = np.zeros((STEPS, 3), dtype=np.float32)
        available_steps = min(STEPS, arrays["interactions"].shape[1])
        interactions[:available_steps] = arrays["interactions"][index, :available_steps]
        objects[:available_steps] = arrays["objects"][index, :available_steps]
        primitives[:available_steps] = arrays["primitives"][index, :available_steps]
        planes[:available_steps] = arrays["surface_plane"][index, :available_steps]
        vertices[:available_steps] = arrays["vertices"][index, :available_steps]
        records.append(
            {
                "hash": int(hashes[local_index]),
                "score": float(score[local_index]),
                "tau": float(arrays["tau"][index]),
                "path_length_m": float(arrays["path_length_m"][index]),
                "geometric_length_m": float(arrays["geometric_length_m"][index]),
                "angles": arrays["angles"][index].astype(np.float32, copy=True),
                "coordinate": coordinate[local_index].copy(),
                "depth": local_depth,
                "interactions": interactions,
                "objects": objects,
                "primitives": primitives,
                "planes": planes,
                "vertices": vertices,
                "source": SOURCE_ID[source],
                "geometry_valid": True,
            }
        )
    return records


def candidate_sort_key(record: dict[str, object]) -> tuple[object, ...]:
    coordinate = np.asarray(record["coordinate"])
    return (
        -int(record["presence_count"]),
        int(record["depth"]),
        float(record["geometric_length_m"]),
        int(coordinate[2] // 2.0),
        int(record["hash"]),
    )


def select_final(records: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(records, key=candidate_sort_key)
    selected: list[dict[str, object]] = []
    used_ids: set[int] = set()
    occupied: set[tuple[int, int, int, int, int]] = set()
    for record in ordered:
        coordinate = np.asarray(record["coordinate"])
        interaction = int(np.asarray(record["interactions"])[0]) if int(record["depth"]) else 0
        cell = (
            int(coordinate[0] // 2.0),
            int(coordinate[1]),
            int(coordinate[2] // 4.0),
            interaction,
            int(record["source"]),
        )
        if cell in occupied:
            continue
        occupied.add(cell)
        selected.append(record)
        used_ids.add(id(record))
        if len(selected) == PATH_CAPACITY:
            return selected
    for record in ordered:
        if id(record) in used_ids:
            continue
        selected.append(record)
        if len(selected) == PATH_CAPACITY:
            break
    return selected


def sequence_geometry(
    record: dict[str, object],
    position: np.ndarray,
    bs_position: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    output = np.zeros((STEPS, 16), dtype=np.float32)
    depth = int(record["depth"])
    token_count = max(depth, 1)
    vertices = np.asarray(record["vertices"], dtype=np.float32).copy()
    if depth == 0:
        vertices[0] = 0.5 * (position + bs_position)
    planes = np.asarray(record["planes"], dtype=np.float32)
    for step in range(min(token_count, STEPS)):
        vertex = vertices[step]
        output[step, 0:3] = (vertex - center) / scale
        output[step, 3:6] = (vertex - position) / scale
        output[step, 6:9] = (vertex - bs_position) / scale
        if step < depth:
            output[step, 9:13] = planes[step] / 100_000.0
        output[step, 13] = step / float(STEPS - 1)
        output[step, 14] = depth / float(STEPS)
        output[step, 15] = float(step < depth)
    return output


def empty_payload(count: int) -> dict[str, np.ndarray]:
    return {
        "point_ids": np.zeros(count, dtype=np.int32),
        "mask": np.zeros((count, PATH_CAPACITY), dtype=bool),
        "continuous": np.zeros((count, PATH_CAPACITY, 19), dtype=np.float16),
        "coordinate": np.zeros((count, PATH_CAPACITY, 3), dtype=np.float32),
        "presence": np.zeros((count, PATH_CAPACITY), dtype=np.float16),
        "depth": np.zeros((count, PATH_CAPACITY), dtype=np.uint8),
        "interactions": np.zeros((count, PATH_CAPACITY, STEPS), dtype=np.uint8),
        "objects": np.zeros((count, PATH_CAPACITY, STEPS), dtype=np.uint32),
        "primitives": np.zeros((count, PATH_CAPACITY, STEPS), dtype=np.uint32),
        "sequence_geometry": np.zeros(
            (count, PATH_CAPACITY, STEPS, 16), dtype=np.float16
        ),
        "family_hash": np.zeros((count, PATH_CAPACITY), dtype=np.uint64),
        "source": np.zeros((count, PATH_CAPACITY), dtype=np.int8),
    }


def write_point(
    payload: dict[str, np.ndarray],
    row: int,
    records: list[dict[str, object]],
    seed_count: int,
    position: np.ndarray,
    bs_position: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> None:
    for slot, record in enumerate(select_final(records)):
        presence = int(record["presence_count"]) / float(seed_count)
        tau = float(record["tau"])
        length = float(record["path_length_m"])
        geometric = float(record["geometric_length_m"])
        angles = np.asarray(record["angles"], dtype=np.float32)
        coordinate = np.asarray(record["coordinate"], dtype=np.float32)
        depth = int(record["depth"])
        continuous = np.concatenate(
            (
                np.asarray(
                    (
                        tau * 1e6,
                        math.log1p(max(length, 0.0)) / 8.0,
                        math.log1p(max(geometric, 0.0)) / 8.0,
                        (length - geometric) / max(geometric, 1.0),
                    ),
                    dtype=np.float32,
                ),
                np.sin(angles),
                np.cos(angles),
                coordinate / np.asarray((MH, MV, S_SC), dtype=np.float32),
                np.asarray(
                    (
                        depth / float(STEPS),
                        presence,
                        float(record["source"]),
                        float(record["geometry_valid"]),
                    ),
                    dtype=np.float32,
                ),
            )
        )
        if continuous.shape != (19,):
            raise RuntimeError(f"unexpected continuous shape {continuous.shape}")
        payload["mask"][row, slot] = True
        payload["continuous"][row, slot] = continuous.astype(np.float16)
        payload["coordinate"][row, slot] = coordinate
        payload["presence"][row, slot] = np.float16(presence)
        payload["depth"][row, slot] = depth
        payload["interactions"][row, slot] = np.asarray(record["interactions"], dtype=np.uint8)
        payload["objects"][row, slot] = np.asarray(record["objects"], dtype=np.uint32)
        payload["primitives"][row, slot] = np.asarray(record["primitives"], dtype=np.uint32)
        payload["sequence_geometry"][row, slot] = sequence_geometry(
            record, position, bs_position, center, scale
        ).astype(np.float16)
        payload["family_hash"][row, slot] = np.uint64(record["hash"])
        payload["source"][row, slot] = np.int8(record["source"])


def export_batch(
    source: str,
    site: int,
    name: str,
    positions: np.ndarray,
    sites: np.ndarray,
    bs_positions: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    audit: AccessAudit,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    single_by_point: list[list[dict[str, object]]] | None = None
    multi_by_point: list[dict[int, dict[str, object]]] | None = None
    expected_ids: np.ndarray | None = None
    for seed_index, seed in enumerate(SEEDS):
        path = raw_path(source, seed, site, name)
        if not path.is_file():
            raise FileNotFoundError(path)
        arrays = load_allowed(path, audit)
        point_ids = arrays["point_ids"].astype(np.int64)
        if expected_ids is None:
            expected_ids = point_ids
            single_by_point = [[] for _ in point_ids]
            multi_by_point = [dict() for _ in point_ids]
        elif not np.array_equal(point_ids, expected_ids):
            raise RuntimeError(f"point identity drift: {path}")
        if not np.all(sites[point_ids] == site):
            raise RuntimeError(f"site identity mismatch: {path}")
        if arrays["offsets"].shape != (len(point_ids) + 1,):
            raise ValueError(f"invalid offsets: {path}")
        if int(arrays["offsets"][-1]) != len(arrays["tau"]):
            raise ValueError(f"offset/path count mismatch: {path}")
        if len(arrays["seed"]) and not np.all(arrays["seed"] == seed):
            raise RuntimeError(f"seed identity mismatch: {path}")
        if len(arrays["source"]) and not np.all(arrays["source"] == SOURCE_ID[source]):
            raise RuntimeError(f"source identity mismatch: {path}")
        assert single_by_point is not None and multi_by_point is not None
        for local_point in range(len(point_ids)):
            records = local_candidates(source, arrays, local_point, site)
            if seed == SINGLE_SEED:
                single_by_point[local_point] = records
            for record in records:
                key = int(record["hash"])
                existing = multi_by_point[local_point].get(key)
                if existing is None:
                    stored = record.copy()
                    stored["seed_mask"] = 1 << seed_index
                    stored["presence_count"] = 1
                    multi_by_point[local_point][key] = stored
                else:
                    seed_mask = int(existing["seed_mask"])
                    bit = 1 << seed_index
                    if not seed_mask & bit:
                        existing["seed_mask"] = seed_mask | bit
                        existing["presence_count"] = int(existing["presence_count"]) + 1
                    if float(record["score"]) < float(existing["score"]):
                        presence_count = existing["presence_count"]
                        combined_mask = existing["seed_mask"]
                        existing.clear()
                        existing.update(record)
                        existing["seed_mask"] = combined_mask
                        existing["presence_count"] = presence_count

    if expected_ids is None or single_by_point is None or multi_by_point is None:
        raise RuntimeError(f"empty batch: {source}/{site}/{name}")
    single = empty_payload(len(expected_ids))
    multi = empty_payload(len(expected_ids))
    single["point_ids"] = expected_ids.astype(np.int32)
    multi["point_ids"] = expected_ids.astype(np.int32)
    for local_point, point_id in enumerate(expected_ids):
        single_records = single_by_point[local_point]
        for record in single_records:
            record["presence_count"] = 1
        write_point(
            single,
            local_point,
            single_records,
            1,
            positions[point_id].astype(np.float32),
            bs_positions[site].astype(np.float32),
            center,
            scale,
        )
        write_point(
            multi,
            local_point,
            list(multi_by_point[local_point].values()),
            len(SEEDS),
            positions[point_id].astype(np.float32),
            bs_positions[site].astype(np.float32),
            center,
            scale,
        )
    return single, multi


def source_batches(source: str) -> list[tuple[int, str]]:
    output: list[tuple[int, str]] = []
    for site in (0, 1):
        directory = RAW_ROOT / source / f"seed_{SINGLE_SEED:04d}" / f"site_{site}"
        names = sorted(path.name for path in directory.glob("batch_*.npz"))
        output.extend((site, name) for name in names)
    if len(output) != 250:
        raise RuntimeError(f"expected 250 {source} batches, found {len(output)}")
    return output


def export_batch_task(task: tuple[str, int, str]) -> dict[str, object]:
    source, site, name = task
    setup, positions, _, sites, _ = load_positions()
    positions = positions.astype(np.float32)
    center = positions.mean(axis=0).astype(np.float32)
    scale = np.maximum(positions.std(axis=0), np.asarray((1.0, 1.0, 20.0))).astype(np.float32)
    audit = AccessAudit()
    started = time.time()
    single, multi = export_batch(
        source,
        site,
        name,
        positions,
        sites,
        setup.bs_positions.astype(np.float32),
        center,
        scale,
        audit,
    )
    atomic_npz(batch_output(source, "s1", site, name), **single)
    atomic_npz(batch_output(source, "s24", site, name), **multi)
    return {
        "source": source,
        "site": site,
        "name": name,
        "seconds": float(time.time() - started),
        "shards": audit.shards,
        "requested_members": audit.requested_members,
        "compressed_bytes": audit.compressed_bytes,
        "uncompressed_bytes": audit.uncompressed_bytes,
    }


def export_source(source: str, force: bool, workers: int) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers must be positive")
    started = time.time()
    audit = AccessAudit()
    pending: list[tuple[str, int, str]] = []
    reused = 0
    for site, name in source_batches(source):
        ready = batch_output(source, "s1", site, name).is_file() and batch_output(
            source, "s24", site, name
        ).is_file()
        if ready and not force:
            reused += 1
        else:
            pending.append((source, site, name))
    completed = 0
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(export_batch_task, task) for task in pending]
            for future in as_completed(futures):
                result = future.result()
                audit.merge(
                    AccessAudit(
                        shards=int(result["shards"]),
                        requested_members=int(result["requested_members"]),
                        compressed_bytes=int(result["compressed_bytes"]),
                        uncompressed_bytes=int(result["uncompressed_bytes"]),
                    )
                )
                completed += 1
                if completed % 10 == 0 or completed == len(pending):
                    total_done = reused + completed
                    mean_seconds = (time.time() - started) / max(completed, 1)
                    print(
                        f"E214 export source={source} batch={total_done}/250 "
                        f"created={completed} reused={reused} worker_mean_wall={mean_seconds:.3f}",
                        flush=True,
                    )
    report = {
        "experiment": VERSION,
        "stage": "export",
        "status": "complete",
        "source": source,
        "created_batches": completed,
        "reused_batches": reused,
        "seed_count": len(SEEDS),
        "single_seed": SINGLE_SEED,
        "workers": workers,
        "allowed_fields_requested": list(ALLOWED_FIELDS),
        "forbidden_fields_requested": [],
        "forbidden_fields": list(FORBIDDEN_FIELDS),
        "access_audit": {
            "shards": audit.shards,
            "requested_members": audit.requested_members,
            "compressed_bytes": audit.compressed_bytes,
            "uncompressed_bytes": audit.uncompressed_bytes,
        },
        "wall_seconds": float(time.time() - started),
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "exit_code": 0,
    }
    atomic_json(report_path(source, "export"), report)
    return report


MERGED_SHAPES = {
    "mask": (POINT_COUNT, PATH_CAPACITY),
    "continuous": (POINT_COUNT, PATH_CAPACITY, 19),
    "coordinate": (POINT_COUNT, PATH_CAPACITY, 3),
    "presence": (POINT_COUNT, PATH_CAPACITY),
    "depth": (POINT_COUNT, PATH_CAPACITY),
    "interactions": (POINT_COUNT, PATH_CAPACITY, STEPS),
    "objects": (POINT_COUNT, PATH_CAPACITY, STEPS),
    "primitives": (POINT_COUNT, PATH_CAPACITY, STEPS),
    "sequence_geometry": (POINT_COUNT, PATH_CAPACITY, STEPS, 16),
    "family_hash": (POINT_COUNT, PATH_CAPACITY),
    "source": (POINT_COUNT, PATH_CAPACITY),
}
MERGED_DTYPES = {
    "mask": np.bool_,
    "continuous": np.float16,
    "coordinate": np.float32,
    "presence": np.float16,
    "depth": np.uint8,
    "interactions": np.uint8,
    "objects": np.uint32,
    "primitives": np.uint32,
    "sequence_geometry": np.float16,
    "family_hash": np.uint64,
    "source": np.int8,
}


def merge_one(source: str, seed_mode: str, force: bool) -> dict[str, object]:
    started = time.time()
    output_dir = merged_dir(source, seed_mode)
    state_path = output_dir / "state.json"
    outputs = {name: output_dir / f"{name}.npy" for name in MERGED_SHAPES}
    if state_path.is_file() and all(path.is_file() for path in outputs.values()) and not force:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "complete":
            return {"status": "reused", **state}
    output_dir.mkdir(parents=True, exist_ok=True)
    partials = {
        name: output_dir / f".{name}.npy.partial-{os.getpid()}" for name in MERGED_SHAPES
    }
    arrays = {
        name: np.lib.format.open_memmap(
            partials[name], mode="w+", dtype=MERGED_DTYPES[name], shape=shape
        )
        for name, shape in MERGED_SHAPES.items()
    }
    seen = np.zeros(POINT_COUNT, dtype=np.uint8)
    for site, name in source_batches(source):
        path = batch_output(source, seed_mode, site, name)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            point_ids = payload["point_ids"].astype(np.int64)
            if np.any(seen[point_ids]):
                raise RuntimeError(f"duplicate point ids in {path}")
            for field in MERGED_SHAPES:
                arrays[field][point_ids] = payload[field]
            seen[point_ids] = 1
    if not np.all(seen == 1):
        raise RuntimeError(f"missing {int(np.sum(seen == 0))} point ids for {source}/{seed_mode}")
    for array in arrays.values():
        array.flush()
    del arrays
    for field in MERGED_SHAPES:
        os.replace(partials[field], outputs[field])

    mask = np.load(outputs["mask"], mmap_mode="r")
    presence = np.load(outputs["presence"], mmap_mode="r")
    continuous = np.load(outputs["continuous"], mmap_mode="r")
    coordinate = np.load(outputs["coordinate"], mmap_mode="r")
    valid_presence = np.asarray(presence)[np.asarray(mask)]
    state = {
        "experiment": VERSION,
        "stage": "merge",
        "status": "complete",
        "source": source,
        "seed_mode": seed_mode,
        "seed_count": 1 if seed_mode == "s1" else len(SEEDS),
        "single_seed": SINGLE_SEED if seed_mode == "s1" else None,
        "point_count": POINT_COUNT,
        "capacity": PATH_CAPACITY,
        "valid_paths": int(np.sum(mask)),
        "valid_paths_mean": float(np.mean(np.sum(mask, axis=1))),
        "valid_paths_min": int(np.min(np.sum(mask, axis=1))),
        "valid_paths_max": int(np.max(np.sum(mask, axis=1))),
        "presence_mean": float(np.mean(valid_presence)) if len(valid_presence) else 0.0,
        "presence_full_fraction": float(np.mean(valid_presence == 1.0)) if len(valid_presence) else 0.0,
        "finite_continuous": bool(np.all(np.isfinite(continuous))),
        "finite_coordinate": bool(np.all(np.isfinite(coordinate))),
        "arrays": {
            field: {
                "path": PATHS.relative(outputs[field]),
                "shape": list(MERGED_SHAPES[field]),
                "dtype": str(np.dtype(MERGED_DTYPES[field])),
            }
            for field in MERGED_SHAPES
        },
        "allowed_fields": list(ALLOWED_FIELDS),
        "forbidden_fields_accessed": [],
        "wall_seconds": float(time.time() - started),
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "exit_code": 0,
    }
    atomic_json(state_path, state)
    return state


def merge_source(source: str, force: bool) -> dict[str, object]:
    results = {mode: merge_one(source, mode, force) for mode in ("s1", "s24")}
    report = {
        "experiment": VERSION,
        "stage": "merge",
        "status": "complete",
        "source": source,
        "results": results,
        "exit_code": 0,
    }
    atomic_json(report_path(source, "merge"), report)
    return report


def audit_all() -> dict[str, object]:
    entries = []
    for source in ("base", "scatter"):
        export_report = json.loads(report_path(source, "export").read_text(encoding="utf-8"))
        if export_report.get("forbidden_fields_requested"):
            raise RuntimeError(f"forbidden field access reported for {source}")
        for mode in ("s1", "s24"):
            state_path = merged_dir(source, mode) / "state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("status") != "complete" or state.get("forbidden_fields_accessed"):
                raise RuntimeError(f"invalid sanitized topology state {source}/{mode}")
            entries.append(state)
    report = {
        "experiment": VERSION,
        "stage": "audit",
        "status": "complete",
        "sources": entries,
        "allowed_fields": list(ALLOWED_FIELDS),
        "forbidden_fields": list(FORBIDDEN_FIELDS),
        "forbidden_fields_accessed": [],
        "exit_code": 0,
    }
    atomic_json(ANALYSIS_DIR / "topology_audit.json", report)
    return report


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    sources = ("base", "scatter") if args.source == "all" else (args.source,)
    try:
        results: dict[str, object] = {}
        if args.stage in ("export", "all"):
            results["export"] = {
                source: export_source(source, args.force, args.workers) for source in sources
            }
        if args.stage in ("merge", "all"):
            results["merge"] = {source: merge_source(source, args.force) for source in sources}
        if args.stage == "audit" or (args.stage == "all" and args.source == "all"):
            results["audit"] = audit_all()
        print(json.dumps(results, ensure_ascii=True, indent=2, allow_nan=False), flush=True)
    except Exception as error:
        source_name = args.source
        payload = {
            "experiment": VERSION,
            "status": "failed",
            "stage": args.stage,
            "source": source_name,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "exit_code": 1,
        }
        atomic_json(failure_path(source_name, args.stage), payload)
        raise


if __name__ == "__main__":
    main()
