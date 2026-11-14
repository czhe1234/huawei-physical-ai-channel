"""Resumable GPU path sampling and route-bank extraction."""

from __future__ import annotations

from collections import defaultdict
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import warp as wp

from project_paths import PATHS

from .config import (
    BranchConfig,
    INTERACTION_REFLECTION,
    INTERACTION_SCATTER,
    PhysicalPriorConfig,
)
from .geometry import GeometryData
from .kernels import MAX_TRACE_DEPTH, trace_paths


ARTIFACT_DIR = PATHS.artifacts / "round2" / "physical_prior"
TRACE_DIR = ARTIFACT_DIR / "traces"
ROUTE_DIR = ARTIFACT_DIR / "route_banks"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def trace_path(site: int, branch: str, seed: int, shard: int) -> Path:
    return TRACE_DIR / f"site{site}" / branch / f"seed{seed}" / f"shard{shard:02d}.npz"


def trace_report_path(site: int, branch: str, seed: int, shard: int) -> Path:
    return trace_path(site, branch, seed, shard).with_suffix(".json")


def route_bank_path(site: int, seed: int) -> Path:
    return ROUTE_DIR / f"site{site}_base_seed{seed}.npz"


def scatter_bank_path(site: int, seed: int) -> Path:
    return ROUTE_DIR / f"site{site}_scatter_seed{seed}.npz"


def shard_count(branch: BranchConfig) -> int:
    return int(math.ceil(branch.rays_per_site_seed / branch.shard_size))


def _branch(config: PhysicalPriorConfig, name: str) -> BranchConfig:
    for branch in config.branches:
        if branch.name == name:
            return branch
    raise KeyError(name)


class WarpTracer:
    """A thin owner for one resident Warp mesh and its face-family array."""

    def __init__(self, geometry: GeometryData, device: str):
        self.device = device
        wp.init()
        with wp.ScopedDevice(device):
            self.mesh = wp.Mesh(
                points=wp.array(geometry.vertices, dtype=wp.vec3, device=device),
                indices=wp.array(
                    geometry.faces.reshape(-1), dtype=wp.int32, device=device
                ),
            )
            self.face_family = wp.array(
                geometry.face_family, dtype=wp.int32, device=device
            )
        self.ground_family = int(geometry.ground_family)

    def trace(
        self,
        *,
        source: np.ndarray,
        seed: int,
        ray_offset: int,
        ray_count: int,
        branch: BranchConfig,
        config: PhysicalPriorConfig,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        allocate_started = time.time()
        with wp.ScopedDevice(self.device):
            counts = wp.zeros(ray_count, dtype=wp.uint8, device=self.device)
            families = wp.full(
                ray_count * MAX_TRACE_DEPTH,
                value=-1,
                dtype=wp.int32,
                device=self.device,
            )
            interactions = wp.zeros(
                ray_count * MAX_TRACE_DEPTH, dtype=wp.uint8, device=self.device
            )
            vertices = wp.zeros(
                ray_count * MAX_TRACE_DEPTH, dtype=wp.vec3, device=self.device
            )
            segment_length = wp.zeros(
                ray_count * MAX_TRACE_DEPTH, dtype=wp.float32, device=self.device
            )
            incidence = wp.zeros(
                ray_count * MAX_TRACE_DEPTH, dtype=wp.float32, device=self.device
            )
            energy_after = wp.zeros(
                ray_count * MAX_TRACE_DEPTH, dtype=wp.float32, device=self.device
            )
            initial_direction = wp.zeros(
                ray_count, dtype=wp.vec3, device=self.device
            )
            terminal_energy = wp.zeros(
                ray_count, dtype=wp.float32, device=self.device
            )
            total_length = wp.zeros(ray_count, dtype=wp.float32, device=self.device)
            allocate_seconds = time.time() - allocate_started
            launch_started = time.time()
            wp.launch(
                trace_paths,
                dim=ray_count,
                inputs=(
                    self.mesh.id,
                    self.face_family,
                    wp.vec3(*map(float, source)),
                    int(seed),
                    int(ray_offset),
                    int(branch.max_depth),
                    float(branch.diffuse_probability),
                    self.ground_family,
                    float(config.ray_epsilon_m),
                    float(config.maximum_path_length_m),
                    float(config.minimum_ray_energy),
                ),
                outputs=(
                    counts,
                    families,
                    interactions,
                    vertices,
                    segment_length,
                    incidence,
                    energy_after,
                    initial_direction,
                    terminal_energy,
                    total_length,
                ),
                device=self.device,
            )
            wp.synchronize_device(self.device)
            launch_seconds = time.time() - launch_started
            transfer_started = time.time()
            arrays = {
                "count": counts.numpy(),
                "family": families.numpy().reshape(ray_count, MAX_TRACE_DEPTH),
                "interaction": interactions.numpy().reshape(
                    ray_count, MAX_TRACE_DEPTH
                ),
                "vertices": vertices.numpy().reshape(
                    ray_count, MAX_TRACE_DEPTH, 3
                ),
                "segment_length": segment_length.numpy().reshape(
                    ray_count, MAX_TRACE_DEPTH
                ),
                "incidence": incidence.numpy().reshape(
                    ray_count, MAX_TRACE_DEPTH
                ),
                "energy_after": energy_after.numpy().reshape(
                    ray_count, MAX_TRACE_DEPTH
                ),
                "initial_direction": initial_direction.numpy(),
                "terminal_energy": terminal_energy.numpy(),
                "total_length": total_length.numpy(),
            }
            transfer_seconds = time.time() - transfer_started
        timing = {
            "allocate_seconds": float(allocate_seconds),
            "gpu_trace_seconds": float(launch_seconds),
            "device_to_host_seconds": float(transfer_seconds),
        }
        return arrays, timing


def run_trace_shard(
    *,
    geometry: GeometryData,
    bs_positions: np.ndarray,
    config: PhysicalPriorConfig,
    site: int,
    branch_name: str,
    seed: int,
    shard: int,
    device: str,
    force: bool = False,
) -> dict[str, Any]:
    branch = _branch(config, branch_name)
    output_path = trace_path(site, branch.name, seed, shard)
    report_path = trace_report_path(site, branch.name, seed, shard)
    if output_path.is_file() and report_path.is_file() and not force:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") == "complete":
            return {"reused": True, **report}
    start = shard * branch.shard_size
    end = min(start + branch.shard_size, branch.rays_per_site_seed)
    if start >= end:
        raise ValueError(f"empty shard {shard}")
    started = time.time()
    tracer = WarpTracer(geometry, device)
    arrays, timing = tracer.trace(
        source=bs_positions[site],
        seed=seed + 100_000 * site + 1_000_000 * branch.source,
        ray_offset=start,
        ray_count=end - start,
        branch=branch,
        config=config,
    )
    save_started = time.time()
    atomic_npz(
        output_path,
        **arrays,
        site=np.asarray(site, dtype=np.int8),
        seed=np.asarray(seed, dtype=np.int32),
        branch=np.asarray(branch.name),
        ray_start=np.asarray(start, dtype=np.int64),
        ray_end=np.asarray(end, dtype=np.int64),
    )
    save_seconds = time.time() - save_started
    valid = arrays["count"] > 0
    interactions = arrays["interaction"]
    report = {
        "status": "complete",
        "site": int(site),
        "branch": branch.name,
        "seed": int(seed),
        "shard": int(shard),
        "device": device,
        "ray_start": int(start),
        "ray_end": int(end),
        "ray_count": int(end - start),
        "hit_fraction": float(np.mean(valid)),
        "interaction_count_mean": float(np.mean(arrays["count"])),
        "reflection_events": int(np.sum(interactions == INTERACTION_REFLECTION)),
        "scatter_events": int(np.sum(interactions == INTERACTION_SCATTER)),
        "output_bytes": int(sum(value.nbytes for value in arrays.values())),
        **timing,
        "save_seconds": float(save_seconds),
        "total_seconds": float(time.time() - started),
        "output": PATHS.relative(output_path),
    }
    atomic_json(report_path, report)
    return report


def _void_rows(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value)
    return contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))).reshape(-1)


def _local_route_records(
    cache: np.lib.npyio.NpzFile,
    config: PhysicalPriorConfig,
) -> Iterable[tuple[tuple[int, ...], int, float, float]]:
    family = cache["family"].astype(np.int32)
    interaction = cache["interaction"].astype(np.uint8)
    energy_after = cache["energy_after"].astype(np.float32)
    reflection = interaction == INTERACTION_REFLECTION
    reflection_count = reflection.sum(axis=1, dtype=np.int16)
    sequence = np.full_like(family, -1)
    sequence_energy = np.zeros_like(energy_after)
    position = np.cumsum(reflection, axis=1, dtype=np.int16) - 1
    rows, columns = np.nonzero(reflection)
    slots = position[rows, columns]
    sequence[rows, slots] = family[rows, columns]
    sequence_energy[rows, slots] = energy_after[rows, columns]
    for depth in range(1, MAX_TRACE_DEPTH + 1):
        valid_rows = np.flatnonzero(reflection_count >= depth)
        if len(valid_rows) == 0:
            continue
        keys = sequence[valid_rows, :depth]
        view = _void_rows(keys)
        unique, first, inverse, counts = np.unique(
            view, return_index=True, return_inverse=True, return_counts=True
        )
        del unique
        maximum_energy = np.zeros(len(first), dtype=np.float32)
        np.maximum.at(
            maximum_energy,
            inverse,
            sequence_energy[valid_rows, depth - 1],
        )
        score = np.sqrt(counts.astype(np.float64)) * maximum_energy.astype(np.float64)
        keep = min(len(score), config.route_bank_per_depth * 4)
        selected = np.argpartition(score, len(score) - keep)[-keep:] if keep < len(score) else np.arange(len(score))
        for index in selected:
            key = tuple(int(value) for value in keys[first[index]])
            yield key, int(counts[index]), float(maximum_energy[index]), float(score[index])


def build_base_route_bank(
    site: int,
    seed: int,
    config: PhysicalPriorConfig,
    force: bool = False,
) -> dict[str, Any]:
    output_path = route_bank_path(site, seed)
    report_path = output_path.with_suffix(".json")
    if output_path.is_file() and report_path.is_file() and not force:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") == "complete":
            return {"reused": True, **report}
    started = time.time()
    aggregate: dict[tuple[int, ...], list[float]] = {}
    for shard in range(shard_count(config.base)):
        path = trace_path(site, config.base.name, seed, shard)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as cache:
            for key, count, energy, score in _local_route_records(cache, config):
                previous = aggregate.get(key)
                if previous is None:
                    aggregate[key] = [float(count), energy, score]
                else:
                    previous[0] += float(count)
                    previous[1] = max(previous[1], energy)
                    previous[2] += score
    by_depth: dict[int, list[tuple[tuple[int, ...], list[float]]]] = defaultdict(list)
    for key, values in aggregate.items():
        by_depth[len(key)].append((key, values))
    selected: list[tuple[tuple[int, ...], list[float]]] = []
    for depth, records in by_depth.items():
        records.sort(key=lambda item: item[1][2], reverse=True)
        selected.extend(records[: config.route_bank_per_depth])
    selected.sort(key=lambda item: item[1][2], reverse=True)
    selected = selected[: config.route_bank_maximum]
    family = np.full((len(selected), MAX_TRACE_DEPTH), -1, dtype=np.int32)
    depth = np.zeros(len(selected), dtype=np.uint8)
    count = np.zeros(len(selected), dtype=np.int32)
    energy = np.zeros(len(selected), dtype=np.float32)
    score = np.zeros(len(selected), dtype=np.float32)
    for row, (key, values) in enumerate(selected):
        family[row, : len(key)] = key
        depth[row] = len(key)
        count[row] = int(values[0])
        energy[row] = values[1]
        score[row] = values[2]
    atomic_npz(
        output_path,
        family=family,
        depth=depth,
        count=count,
        energy=energy,
        score=score,
        site=np.asarray(site, dtype=np.int8),
        seed=np.asarray(seed, dtype=np.int32),
    )
    report = {
        "status": "complete",
        "site": int(site),
        "seed": int(seed),
        "route_count": int(len(selected)),
        "routes_by_depth": {
            str(value): int(np.sum(depth == value)) for value in np.unique(depth)
        },
        "maximum_observed_depth": int(np.max(depth)) if len(depth) else 0,
        "seconds": float(time.time() - started),
        "output": PATHS.relative(output_path),
    }
    atomic_json(report_path, report)
    return report


def _scatter_events(cache: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    interaction = cache["interaction"].astype(np.uint8)
    diffuse = interaction == INTERACTION_SCATTER
    valid = np.any(diffuse, axis=1)
    rows = np.flatnonzero(valid)
    if len(rows) == 0:
        return {
            "family": np.empty((0, MAX_TRACE_DEPTH), dtype=np.int32),
            "interaction": np.empty((0, MAX_TRACE_DEPTH), dtype=np.uint8),
            "vertices": np.empty((0, MAX_TRACE_DEPTH, 3), dtype=np.float32),
            "incidence": np.empty((0, MAX_TRACE_DEPTH), dtype=np.float32),
            "depth": np.empty(0, dtype=np.uint8),
            "initial_direction": np.empty((0, 3), dtype=np.float32),
            "path_length": np.empty(0, dtype=np.float32),
            "energy": np.empty(0, dtype=np.float32),
        }
    last = np.where(diffuse[rows], np.arange(MAX_TRACE_DEPTH)[None], -1).max(axis=1)
    depth = last + 1
    index = (rows, last)
    segment = cache["segment_length"].astype(np.float32)
    return {
        "family": cache["family"][rows].astype(np.int32),
        "interaction": interaction[rows],
        "vertices": cache["vertices"][rows].astype(np.float32),
        "incidence": cache["incidence"][rows].astype(np.float32),
        "depth": depth.astype(np.uint8),
        "initial_direction": cache["initial_direction"][rows].astype(np.float32),
        "path_length": np.take_along_axis(
            np.cumsum(segment[rows], axis=1, dtype=np.float32), last[:, None], axis=1
        )[:, 0],
        "energy": cache["energy_after"][index].astype(np.float32),
    }


def build_scatter_bank(
    site: int,
    seed: int,
    config: PhysicalPriorConfig,
    force: bool = False,
) -> dict[str, Any]:
    output_path = scatter_bank_path(site, seed)
    report_path = output_path.with_suffix(".json")
    if output_path.is_file() and report_path.is_file() and not force:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") == "complete":
            return {"reused": True, **report}
    started = time.time()
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    for shard in range(shard_count(config.scatter)):
        path = trace_path(site, config.scatter.name, seed, shard)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as cache:
            local = _scatter_events(cache)
        for name, value in local.items():
            chunks[name].append(value)
    arrays = {
        name: np.concatenate(values, axis=0) if values else np.empty(0)
        for name, values in chunks.items()
    }
    count_before = int(len(arrays["depth"]))
    if count_before:
        rows = np.arange(count_before, dtype=np.int64)
        last = arrays["depth"].astype(np.int64) - 1
        point = arrays["vertices"][rows, last]
        voxel = np.floor(point / config.scatter_voxel_m).astype(np.int32)
        order = np.argsort(-arrays["energy"].astype(np.float64), kind="stable")
        kept: list[int] = []
        voxel_count: dict[tuple[int, int, int], int] = defaultdict(int)
        for row in order:
            key = tuple(int(value) for value in voxel[row])
            if voxel_count[key] >= config.scatter_events_per_voxel:
                continue
            voxel_count[key] += 1
            kept.append(int(row))
            if len(kept) >= config.scatter_events_per_seed:
                break
        selected = np.asarray(kept, dtype=np.int64)
        arrays = {name: value[selected] for name, value in arrays.items()}
    atomic_npz(
        output_path,
        **arrays,
        site=np.asarray(site, dtype=np.int8),
        seed=np.asarray(seed, dtype=np.int32),
    )
    report = {
        "status": "complete",
        "site": int(site),
        "seed": int(seed),
        "diffuse_events_before_pruning": count_before,
        "diffuse_events_after_pruning": int(len(arrays["depth"])),
        "seconds": float(time.time() - started),
        "output": PATHS.relative(output_path),
    }
    atomic_json(report_path, report)
    return report
