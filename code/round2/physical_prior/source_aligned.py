"""Source-aligned candidate tracing, exact image solving, and PathTokens."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import torch
import warp as wp

from project_paths import PATHS

from .config import LIGHT_SPEED_M_S
from .geometry import GeometryData
from .source_aligned_config import SourceAlignedConfig, SourceBranchConfig
from .source_aligned_geometry import SourceAlignedGeometry
from .source_aligned_kernels import (
    MAX_TRACE_DEPTH,
    solve_diffuse_routes,
    solve_image_routes,
    trace_source_paths,
)
from .source_aligned_material import MaterialTable, build_material_table
from .tokens import TOKEN_DTYPES, empty_token_arrays


ARTIFACT_DIR = PATHS.artifacts / "round2" / "physical_prior_source_aligned"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_physical_prior_source_aligned"
GEOMETRY_CACHE = ARTIFACT_DIR / "geometry_v1.npz"
RAW_TOKEN_CACHE = ARTIFACT_DIR / "raw_tokens_n4000_k512_v4.npz"
TRACE_DIR = ARTIFACT_DIR / "trace"
ROUTE_DIR = ARTIFACT_DIR / "routes_v2"
TOKEN_DIR = ARTIFACT_DIR / "token_shards_v3"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def trace_path(site: int, branch: str, seed: int, shard: int) -> Path:
    return TRACE_DIR / f"site{site}" / branch / f"seed{seed}_shard{shard:02d}.npz"


def route_path(site: int, branch: str, seed: int) -> Path:
    return ROUTE_DIR / f"site{site}_{branch}_seed{seed}.npz"


def diffuse_path(site: int, seed: int) -> Path:
    return ROUTE_DIR / f"site{site}_diffuse_seed{seed}.npz"


def token_shard_path(site: int, shard: int) -> Path:
    return TOKEN_DIR / f"site{site}_shard{shard:02d}.npz"


def shard_count(branch: SourceBranchConfig) -> int:
    return int(math.ceil(branch.rays_per_site_seed / branch.shard_size))


def tx_look_vector(site: int, config: SourceAlignedConfig) -> np.ndarray:
    yaw = math.radians(config.site_yaw_deg[site])
    tilt = math.radians(config.site_tilt_deg[site])
    return np.asarray(
        (
            math.cos(tilt) * math.cos(yaw),
            math.cos(tilt) * math.sin(yaw),
            math.sin(tilt),
        ),
        dtype=np.float32,
    )


class SourceAlignedTracer:
    """Own a resident Warp mesh and exact material tables."""

    def __init__(
        self,
        geometry: GeometryData,
        aligned: SourceAlignedGeometry,
        config: SourceAlignedConfig,
        device: str,
    ) -> None:
        self.device = device
        self.config = config
        wp.init()
        with wp.ScopedDevice(device):
            self.mesh = wp.Mesh(
                points=wp.array(geometry.vertices, dtype=wp.vec3, device=device),
                indices=wp.array(
                    geometry.faces.reshape(-1), dtype=wp.int32, device=device
                ),
            )
            self.face_plane = wp.array(
                aligned.face_plane, dtype=wp.int32, device=device
            )
            self.tables: dict[str, dict[str, wp.array]] = {}
            for branch in config.branches:
                table = build_material_table(
                    frequency_hz=config.frequency_hz,
                    thickness_m=config.concrete_thickness_m,
                    scattering_coefficient=branch.scattering_coefficient,
                    samples=config.material_table_samples,
                )
                self.tables[branch.name] = self._upload_table(table)

    def _upload_table(self, table: MaterialTable) -> dict[str, wp.array]:
        return {
            name: wp.array(getattr(table, name), dtype=wp.float32, device=self.device)
            for name in (
                "specular_probability",
                "diffuse_probability",
                "transmission_probability",
                "reflection_power",
                "transmission_power",
            )
        }

    def trace(
        self,
        *,
        source: np.ndarray,
        branch: SourceBranchConfig,
        seed: int,
        ray_offset: int,
        ray_count: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        allocation_started = time.time()
        total_slots = ray_count * MAX_TRACE_DEPTH
        with wp.ScopedDevice(self.device):
            counts = wp.zeros(ray_count, dtype=wp.uint8, device=self.device)
            planes = wp.full(total_slots, -1, dtype=wp.int32, device=self.device)
            faces = wp.full(total_slots, -1, dtype=wp.int32, device=self.device)
            interactions = wp.zeros(total_slots, dtype=wp.uint8, device=self.device)
            vertices = wp.zeros(total_slots, dtype=wp.vec3, device=self.device)
            incidence = wp.zeros(total_slots, dtype=wp.float32, device=self.device)
            event_pdf = wp.zeros(total_slots, dtype=wp.float32, device=self.device)
            prefix_power = wp.zeros(total_slots, dtype=wp.float32, device=self.device)
            initial_direction = wp.zeros(ray_count, dtype=wp.vec3, device=self.device)
            allocation_seconds = time.time() - allocation_started
            table = self.tables[branch.name]
            launch_started = time.time()
            wp.launch(
                trace_source_paths,
                dim=ray_count,
                inputs=(
                    self.mesh.id,
                    self.face_plane,
                    wp.vec3(*map(float, source)),
                    int(seed),
                    int(ray_offset),
                    int(branch.rays_per_site_seed),
                    int(branch.max_depth),
                    float(branch.scattering_coefficient),
                    float(self.config.ray_epsilon_m),
                    table["specular_probability"],
                    table["diffuse_probability"],
                    table["transmission_probability"],
                    table["reflection_power"],
                    table["transmission_power"],
                ),
                outputs=(
                    counts,
                    planes,
                    faces,
                    interactions,
                    vertices,
                    incidence,
                    event_pdf,
                    prefix_power,
                    initial_direction,
                ),
                device=self.device,
            )
            wp.synchronize_device(self.device)
            launch_seconds = time.time() - launch_started
            transfer_started = time.time()
            arrays = {
                "count": counts.numpy(),
                "plane": planes.numpy().reshape(ray_count, MAX_TRACE_DEPTH),
                "face": faces.numpy().reshape(ray_count, MAX_TRACE_DEPTH),
                "interaction": interactions.numpy().reshape(
                    ray_count, MAX_TRACE_DEPTH
                ),
                "vertices": vertices.numpy().reshape(
                    ray_count, MAX_TRACE_DEPTH, 3
                ),
                "incidence": incidence.numpy().reshape(
                    ray_count, MAX_TRACE_DEPTH
                ),
                "event_pdf": event_pdf.numpy().reshape(
                    ray_count, MAX_TRACE_DEPTH
                ),
                "prefix_power": prefix_power.numpy().reshape(
                    ray_count, MAX_TRACE_DEPTH
                ),
                "initial_direction": initial_direction.numpy(),
            }
            transfer_seconds = time.time() - transfer_started
        return arrays, {
            "allocation_seconds": float(allocation_seconds),
            "gpu_trace_seconds": float(launch_seconds),
            "device_to_host_seconds": float(transfer_seconds),
        }


def run_trace_shard(
    *,
    geometry: GeometryData,
    aligned: SourceAlignedGeometry,
    bs_positions: np.ndarray,
    config: SourceAlignedConfig,
    site: int,
    branch: SourceBranchConfig,
    seed: int,
    shard: int,
    device: str,
    force: bool = False,
) -> dict[str, Any]:
    output = trace_path(site, branch.name, seed, shard)
    report_path = output.with_suffix(".json")
    if output.is_file() and report_path.is_file() and not force:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") == "complete":
            return {"reused": True, **report}
    start = shard * branch.shard_size
    end = min(start + branch.shard_size, branch.rays_per_site_seed)
    if start >= end:
        raise ValueError(f"empty trace shard {shard}")
    started = time.time()
    tracer = SourceAlignedTracer(geometry, aligned, config, device)
    arrays, timing = tracer.trace(
        source=bs_positions[site],
        branch=branch,
        seed=seed,
        ray_offset=start,
        ray_count=end - start,
    )
    atomic_npz(
        output,
        ray_offset=np.asarray(start, dtype=np.int64),
        ray_count=np.asarray(end - start, dtype=np.int64),
        **arrays,
    )
    report = {
        "status": "complete",
        "site": int(site),
        "branch": branch.name,
        "seed": int(seed),
        "shard": int(shard),
        "device": device,
        "rays": int(end - start),
        "hit_fraction": float(np.mean(arrays["count"] > 0)),
        "mean_depth": float(np.mean(arrays["count"])),
        "diffuse_events": int(np.sum(arrays["interaction"] == 2)),
        "seconds": float(time.time() - started),
        **timing,
        "output": PATHS.relative(output),
    }
    atomic_json(report_path, report)
    return report


def _void_rows(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value)
    width = contiguous.dtype.itemsize * contiguous.shape[1]
    return contiguous.view(np.dtype((np.void, width))).reshape(-1)


def _topology_records(
    cache: np.lib.npyio.NpzFile,
    config: SourceAlignedConfig,
) -> Iterable[tuple[tuple[int, ...], int, float, float]]:
    count = cache["count"].astype(np.int16)
    interaction = cache["interaction"].astype(np.int16)
    plane = cache["plane"].astype(np.int32)
    prefix_power = cache["prefix_power"].astype(np.float32)
    for depth in range(1, MAX_TRACE_DEPTH + 1):
        rows = np.flatnonzero(count >= depth)
        if len(rows) == 0:
            continue
        local_interaction = interaction[rows, :depth]
        pure = ~np.any(local_interaction == 2, axis=1)
        rows = rows[pure]
        if len(rows) == 0:
            continue
        local_interaction = interaction[rows, :depth]
        local_plane = plane[rows, :depth]
        signature = np.where(
            local_interaction == 1,
            local_plane + 16,
            local_interaction,
        ).astype(np.int32)
        view = _void_rows(signature)
        _, first, inverse, occurrences = np.unique(
            view,
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )
        maximum_power = np.zeros(len(first), dtype=np.float32)
        np.maximum.at(maximum_power, inverse, prefix_power[rows, depth - 1])
        score = np.sqrt(occurrences.astype(np.float64)) * np.sqrt(
            np.maximum(maximum_power.astype(np.float64), 1e-30)
        )
        keep = min(len(score), config.routes_per_depth * 4)
        selected = (
            np.argpartition(score, len(score) - keep)[-keep:]
            if keep < len(score)
            else np.arange(len(score))
        )
        for index in selected:
            yield (
                tuple(int(value) for value in signature[first[index]]),
                int(occurrences[index]),
                float(maximum_power[index]),
                float(score[index]),
            )


def build_route_bank(
    *,
    site: int,
    branch: SourceBranchConfig,
    seed: int,
    config: SourceAlignedConfig,
    force: bool = False,
) -> dict[str, Any]:
    output = route_path(site, branch.name, seed)
    report_path = output.with_suffix(".json")
    if output.is_file() and report_path.is_file() and not force:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") == "complete":
            return {"reused": True, **report}
    started = time.time()
    aggregate: dict[tuple[int, ...], list[float]] = {}
    for shard in range(shard_count(branch)):
        path = trace_path(site, branch.name, seed, shard)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as cache:
            for key, count, power, score in _topology_records(cache, config):
                previous = aggregate.get(key)
                if previous is None:
                    aggregate[key] = [float(count), power, score]
                else:
                    previous[0] += float(count)
                    previous[1] = max(previous[1], power)
                    previous[2] += score
    by_depth: dict[int, list[tuple[tuple[int, ...], list[float]]]] = defaultdict(list)
    for key, values in aggregate.items():
        by_depth[len(key)].append((key, values))
    selected: list[tuple[tuple[int, ...], list[float]]] = []
    for depth, records in by_depth.items():
        records.sort(key=lambda item: item[1][2], reverse=True)
        selected.extend(records[: config.routes_per_depth])
    selected.sort(key=lambda item: item[1][2], reverse=True)
    selected = selected[: config.maximum_routes]

    interaction = np.zeros((len(selected), MAX_TRACE_DEPTH), dtype=np.uint8)
    plane = np.full((len(selected), MAX_TRACE_DEPTH), -1, dtype=np.int32)
    depth_array = np.zeros(len(selected), dtype=np.uint8)
    occurrence = np.zeros(len(selected), dtype=np.int32)
    trace_power = np.zeros(len(selected), dtype=np.float32)
    score = np.zeros(len(selected), dtype=np.float32)
    for row, (key, values) in enumerate(selected):
        depth_array[row] = len(key)
        occurrence[row] = int(values[0])
        trace_power[row] = values[1]
        score[row] = values[2]
        for step, token in enumerate(key):
            if token >= 16:
                interaction[row, step] = 1
                plane[row, step] = token - 16
            else:
                interaction[row, step] = token
    atomic_npz(
        output,
        interaction=interaction,
        plane=plane,
        depth=depth_array,
        occurrence=occurrence,
        trace_power=trace_power,
        score=score,
    )
    report = {
        "status": "complete",
        "site": int(site),
        "branch": branch.name,
        "seed": int(seed),
        "route_count": int(len(selected)),
        "routes_by_depth": {
            str(depth): int(np.sum(depth_array == depth))
            for depth in np.unique(depth_array)
        },
        "seconds": float(time.time() - started),
        "output": PATHS.relative(output),
    }
    atomic_json(report_path, report)
    return report


def _prune_diffuse_records(
    arrays: dict[str, np.ndarray],
    *,
    maximum: int,
    config: SourceAlignedConfig,
) -> dict[str, np.ndarray]:
    count = len(arrays["prefix_depth"])
    if count <= maximum:
        return arrays
    score = arrays["score"].astype(np.float64)
    preselect = min(count, maximum * 4)
    candidate = (
        np.argpartition(score, count - preselect)[-preselect:]
        if preselect < count
        else np.arange(count)
    )
    candidate = candidate[np.argsort(-score[candidate], kind="stable")]
    point = arrays["vertices"][
        candidate, arrays["prefix_depth"][candidate].astype(np.int64) - 1
    ]
    voxel = np.floor(point / config.scatter_voxel_m).astype(np.int32)
    voxel_count: dict[tuple[int, int, int], int] = defaultdict(int)
    seen_suffix: set[tuple[int, ...]] = set()
    kept: list[int] = []
    for local, row in enumerate(candidate):
        voxel_key = tuple(int(value) for value in voxel[local])
        if voxel_count[voxel_key] >= config.scatter_records_per_voxel:
            continue
        prefix_depth = int(arrays["prefix_depth"][row])
        total_depth = int(arrays["total_depth"][row])
        if total_depth > prefix_depth:
            signature = tuple(
                int(arrays["plane"][row, step]) + 16
                if int(arrays["interaction"][row, step]) == 1
                else int(arrays["interaction"][row, step])
                for step in range(total_depth)
            )
            if signature in seen_suffix:
                continue
            seen_suffix.add(signature)
        voxel_count[voxel_key] += 1
        kept.append(int(row))
        if len(kept) >= maximum:
            break
    selected = np.asarray(kept, dtype=np.int64)
    return {name: value[selected] for name, value in arrays.items()}


def build_diffuse_bank(
    *,
    site: int,
    seed: int,
    branch: SourceBranchConfig,
    config: SourceAlignedConfig,
    force: bool = False,
) -> dict[str, Any]:
    if branch.scattering_coefficient <= 0.0:
        raise ValueError("diffuse bank requires a non-zero scattering coefficient")
    output = diffuse_path(site, seed)
    report_path = output.with_suffix(".json")
    if output.is_file() and report_path.is_file() and not force:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") == "complete":
            return {"reused": True, **report}
    started = time.time()
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    candidate_count = 0
    for shard in range(shard_count(branch)):
        path = trace_path(site, branch.name, seed, shard)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as cache:
            count = cache["count"].astype(np.int16)
            interaction = cache["interaction"].astype(np.uint8)
            diffuse = interaction == 2
            depth_index = np.arange(MAX_TRACE_DEPTH, dtype=np.int16)[None]
            last_diffuse = np.maximum.accumulate(
                np.where(diffuse, depth_index, -1), axis=1
            )
            row_chunks: list[np.ndarray] = []
            prefix_chunks: list[np.ndarray] = []
            total_chunks: list[np.ndarray] = []
            score_chunks: list[np.ndarray] = []
            prefix_power = cache["prefix_power"].astype(np.float32)
            for total_depth in range(1, branch.max_depth + 1):
                rows = np.flatnonzero(count >= total_depth)
                if len(rows) == 0:
                    continue
                local_last = last_diffuse[rows, total_depth - 1]
                keep = local_last >= 0
                rows = rows[keep]
                local_last = local_last[keep]
                if len(rows) == 0:
                    continue
                prefix_depth = local_last + 1
                score = prefix_power[rows, local_last] / np.sqrt(
                    1.0 + total_depth - prefix_depth
                )
                row_chunks.append(rows.astype(np.int64))
                prefix_chunks.append(prefix_depth.astype(np.uint8))
                total_chunks.append(
                    np.full(len(rows), total_depth, dtype=np.uint8)
                )
                score_chunks.append(score.astype(np.float32))
            if not row_chunks:
                continue
            rows = np.concatenate(row_chunks)
            prefix_depth = np.concatenate(prefix_chunks)
            total_depth = np.concatenate(total_chunks)
            score = np.concatenate(score_chunks)
            candidate_count += len(rows)
            preselect = min(len(rows), config.scatter_records_per_seed * 2)
            selected = (
                np.argpartition(score, len(score) - preselect)[-preselect:]
                if preselect < len(score)
                else np.arange(len(score))
            )
            local = {
                "prefix_depth": prefix_depth[selected],
                "total_depth": total_depth[selected],
                "score": score[selected],
                "interaction": interaction[rows[selected]].copy(),
                "plane": cache["plane"][rows[selected]].astype(np.int32),
                "face": cache["face"][rows[selected]].astype(np.int32),
                "vertices": cache["vertices"][rows[selected]].astype(np.float32),
                "incidence": cache["incidence"][rows[selected]].astype(np.float32),
            }
        local = _prune_diffuse_records(
            local,
            maximum=config.scatter_records_per_seed,
            config=config,
        )
        for name, value in local.items():
            chunks[name].append(value)
    if chunks:
        arrays = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
        arrays = _prune_diffuse_records(
            arrays,
            maximum=config.scatter_records_per_seed,
            config=config,
        )
    else:
        arrays = {
            "prefix_depth": np.empty(0, dtype=np.uint8),
            "total_depth": np.empty(0, dtype=np.uint8),
            "score": np.empty(0, dtype=np.float32),
            "interaction": np.empty((0, MAX_TRACE_DEPTH), dtype=np.uint8),
            "plane": np.empty((0, MAX_TRACE_DEPTH), dtype=np.int32),
            "face": np.empty((0, MAX_TRACE_DEPTH), dtype=np.int32),
            "vertices": np.empty((0, MAX_TRACE_DEPTH, 3), dtype=np.float32),
            "incidence": np.empty((0, MAX_TRACE_DEPTH), dtype=np.float32),
        }
    arrays["presence"] = np.ones(len(arrays["prefix_depth"]), dtype=np.uint8)
    arrays["seed"] = np.full(len(arrays["prefix_depth"]), seed, dtype=np.int32)
    atomic_npz(output, **arrays)
    report = {
        "status": "complete",
        "site": int(site),
        "seed": int(seed),
        "candidate_records_before_pruning": int(candidate_count),
        "records_after_pruning": int(len(arrays["prefix_depth"])),
        "direct_diffuse_records": int(
            np.sum(arrays["prefix_depth"] == arrays["total_depth"])
        ),
        "diffuse_suffix_records": int(
            np.sum(arrays["prefix_depth"] < arrays["total_depth"])
        ),
        "seconds": float(time.time() - started),
        "output": PATHS.relative(output),
    }
    atomic_json(report_path, report)
    return report


def merge_diffuse_banks(
    site: int, config: SourceAlignedConfig
) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    for seed in config.seeds:
        path = diffuse_path(site, seed)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as cache:
            for name in cache.files:
                chunks[name].append(cache[name])
    arrays = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    maximum = config.scatter_records_per_seed * len(config.seeds)
    return _prune_diffuse_records(arrays, maximum=maximum, config=config)


def diffuse_suffix_images(
    records: dict[str, np.ndarray], aligned: SourceAlignedGeometry
) -> np.ndarray:
    count = len(records["prefix_depth"])
    images = np.zeros((count, MAX_TRACE_DEPTH, 3), dtype=np.float32)
    if count == 0:
        return images
    rows = np.arange(count, dtype=np.int64)
    current = records["vertices"][
        rows, records["prefix_depth"].astype(np.int64) - 1
    ].astype(np.float64)
    for suffix_step in range(MAX_TRACE_DEPTH):
        full_step = records["prefix_depth"].astype(np.int64) + suffix_step
        active = full_step < records["total_depth"].astype(np.int64)
        if not np.any(active):
            break
        active_rows = np.flatnonzero(active)
        step = full_step[active]
        interaction = records["interaction"][active_rows, step]
        specular_local = interaction == 1
        if np.any(specular_local):
            specular_rows = active_rows[specular_local]
            specular_step = step[specular_local]
            plane = records["plane"][specular_rows, specular_step]
            normal = aligned.plane_normal[plane].astype(np.float64)
            offset = aligned.plane_offset[plane].astype(np.float64)
            distance = np.einsum(
                "ri,ri->r", current[specular_rows], normal
            ) - offset
            current[specular_rows] -= 2.0 * distance[:, None] * normal
        images[active_rows, suffix_step] = current[active_rows].astype(np.float32)
    return images


def merge_route_banks(
    site: int,
    branch: SourceBranchConfig,
    config: SourceAlignedConfig,
) -> dict[str, np.ndarray]:
    records: dict[tuple[int, ...], dict[str, Any]] = {}
    for seed in config.seeds:
        path = route_path(site, branch.name, seed)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as cache:
            depth_values = cache["depth"].astype(np.int64)
            interaction_values = cache["interaction"].astype(np.uint8)
            plane_values = cache["plane"].astype(np.int32)
            occurrence_values = cache["occurrence"].astype(np.int64)
            score_values = cache["score"].astype(np.float64)
            for row, depth in enumerate(depth_values):
                signature: list[int] = []
                for step in range(depth):
                    interaction = int(interaction_values[row, step])
                    if interaction == 1:
                        signature.append(int(plane_values[row, step]) + 16)
                    else:
                        signature.append(interaction)
                key = tuple(signature)
                previous = records.get(key)
                if previous is None:
                    records[key] = {
                        "presence": 1,
                        "occurrence": int(occurrence_values[row]),
                        "score": float(score_values[row]),
                    }
                else:
                    previous["presence"] += 1
                    previous["occurrence"] += int(occurrence_values[row])
                    previous["score"] += float(score_values[row])
    by_depth: dict[int, list[tuple[tuple[int, ...], dict[str, Any]]]] = defaultdict(list)
    for key, value in records.items():
        by_depth[len(key)].append((key, value))
    selected: list[tuple[tuple[int, ...], dict[str, Any]]] = []
    for depth, local in by_depth.items():
        local.sort(key=lambda item: item[1]["score"], reverse=True)
        selected.extend(local[: config.routes_per_depth])
    selected.sort(key=lambda item: item[1]["score"], reverse=True)
    selected = selected[: config.maximum_routes]

    # Sionna hashes candidate chains together with each target. Rare direct
    # transmission chains therefore survive even when their global trace score
    # is low. Keep one such chain per supported depth so the global route bank
    # does not discard target-specific through-wall candidates.
    selected_keys = {key for key, _ in selected}
    for depth_value in range(1, branch.max_depth + 1):
        key = (4,) * depth_value
        if key in selected_keys:
            continue
        selected.append(
            (
                key,
                records.get(
                    key,
                    {
                        "presence": len(config.seeds),
                        "occurrence": 0,
                        "score": 0.0,
                    },
                ),
            )
        )

    # Depth zero is the exact line-of-sight candidate.
    count = len(selected) + 1
    interaction = np.zeros((count, MAX_TRACE_DEPTH), dtype=np.uint8)
    plane = np.full((count, MAX_TRACE_DEPTH), -1, dtype=np.int32)
    depth = np.zeros(count, dtype=np.uint8)
    presence = np.full(count, len(config.seeds), dtype=np.uint8)
    score = np.zeros(count, dtype=np.float32)
    score[0] = 1.0
    for row, (key, value) in enumerate(selected, start=1):
        depth[row] = len(key)
        presence[row] = min(int(value["presence"]), 255)
        score[row] = float(value["score"])
        for step, token in enumerate(key):
            if token >= 16:
                interaction[row, step] = 1
                plane[row, step] = token - 16
            else:
                interaction[row, step] = token
    return {
        "interaction": interaction,
        "plane": plane,
        "depth": depth,
        "presence": presence,
        "score": score,
    }


def route_images(
    source: np.ndarray,
    routes: dict[str, np.ndarray],
    aligned: SourceAlignedGeometry,
) -> np.ndarray:
    count = len(routes["depth"])
    images = np.zeros((count, MAX_TRACE_DEPTH, 3), dtype=np.float32)
    current = np.broadcast_to(
        np.asarray(source, dtype=np.float64), (count, 3)
    ).copy()
    for step in range(MAX_TRACE_DEPTH):
        active = routes["depth"].astype(np.int64) > step
        specular = active & (routes["interaction"][:, step] == 1)
        if np.any(specular):
            plane = routes["plane"][specular, step]
            normal = aligned.plane_normal[plane].astype(np.float64)
            offset = aligned.plane_offset[plane].astype(np.float64)
            distance = np.einsum("ri,ri->r", current[specular], normal) - offset
            current[specular] -= 2.0 * distance[:, None] * normal
        images[active, step] = current[active].astype(np.float32)
    return images


class SourceAlignedRouteSolver:
    def __init__(
        self,
        geometry: GeometryData,
        aligned: SourceAlignedGeometry,
        config: SourceAlignedConfig,
        branch: SourceBranchConfig,
        device: str,
    ) -> None:
        self.device = device
        self.config = config
        self.branch = branch
        wp.init()
        table = build_material_table(
            frequency_hz=config.frequency_hz,
            thickness_m=config.concrete_thickness_m,
            scattering_coefficient=branch.scattering_coefficient,
            samples=config.material_table_samples,
        )
        with wp.ScopedDevice(device):
            self.mesh = wp.Mesh(
                points=wp.array(geometry.vertices, dtype=wp.vec3, device=device),
                indices=wp.array(
                    geometry.faces.reshape(-1), dtype=wp.int32, device=device
                ),
            )
            self.face_plane = wp.array(
                aligned.face_plane, dtype=wp.int32, device=device
            )
            self.face_family = wp.array(
                geometry.face_family, dtype=wp.int32, device=device
            )
            self.face_normal = wp.array(
                aligned.face_normal, dtype=wp.vec3, device=device
            )
            self.table = {
                name: wp.array(getattr(table, name), dtype=wp.float32, device=device)
                for name in (
                    "specular_probability",
                    "diffuse_probability",
                    "transmission_probability",
                    "reflection_power",
                    "transmission_power",
                )
            }

    def solve(
        self,
        *,
        source: np.ndarray,
        tx_look: np.ndarray,
        targets: np.ndarray,
        routes: dict[str, np.ndarray],
        images: np.ndarray,
        maximum: int,
    ) -> dict[str, np.ndarray]:
        targets = np.asarray(targets, dtype=np.float32)
        route_count = len(routes["depth"])
        total = len(targets) * route_count
        slots = total * MAX_TRACE_DEPTH
        with wp.ScopedDevice(self.device):
            target_gpu = wp.array(targets, dtype=wp.vec3, device=self.device)
            depth_gpu = wp.array(routes["depth"], dtype=wp.uint8, device=self.device)
            interaction_gpu = wp.array(
                routes["interaction"].reshape(-1), dtype=wp.uint8, device=self.device
            )
            plane_gpu = wp.array(
                routes["plane"].reshape(-1), dtype=wp.int32, device=self.device
            )
            image_gpu = wp.array(
                images.reshape(-1, 3), dtype=wp.vec3, device=self.device
            )
            valid = wp.zeros(total, dtype=wp.uint8, device=self.device)
            path_length = wp.zeros(total, dtype=wp.float32, device=self.device)
            departure = wp.zeros(total, dtype=wp.vec3, device=self.device)
            arrival = wp.zeros(total, dtype=wp.vec3, device=self.device)
            energy = wp.zeros(total, dtype=wp.float32, device=self.device)
            output_interaction = wp.zeros(slots, dtype=wp.uint8, device=self.device)
            output_family = wp.full(slots, -1, dtype=wp.int32, device=self.device)
            output_face = wp.full(slots, -1, dtype=wp.int32, device=self.device)
            output_vertices = wp.zeros(slots, dtype=wp.vec3, device=self.device)
            output_incidence = wp.zeros(slots, dtype=wp.float32, device=self.device)
            wp.launch(
                solve_image_routes,
                dim=total,
                inputs=(
                    self.mesh.id,
                    self.face_plane,
                    self.face_family,
                    self.face_normal,
                    wp.vec3(*map(float, source)),
                    wp.vec3(*map(float, tx_look)),
                    target_gpu,
                    route_count,
                    depth_gpu,
                    interaction_gpu,
                    plane_gpu,
                    image_gpu,
                    float(self.branch.scattering_coefficient),
                    float(self.config.ray_epsilon_m),
                    self.table["specular_probability"],
                    self.table["diffuse_probability"],
                    self.table["transmission_probability"],
                    self.table["reflection_power"],
                    self.table["transmission_power"],
                ),
                outputs=(
                    valid,
                    path_length,
                    departure,
                    arrival,
                    energy,
                    output_interaction,
                    output_family,
                    output_face,
                    output_vertices,
                    output_incidence,
                ),
                device=self.device,
            )
            wp.synchronize_device(self.device)
            valid_t = wp.to_torch(valid).reshape(len(targets), route_count).bool()
            energy_t = wp.to_torch(energy).reshape(len(targets), route_count)
            ranked = torch.where(valid_t, energy_t, torch.full_like(energy_t, -torch.inf))
            keep = min(maximum, route_count)
            selected_energy, selected_route = torch.topk(ranked, keep, dim=1)
            selected_mask = torch.isfinite(selected_energy) & (selected_energy > 0.0)
            row = torch.arange(len(targets), device=selected_route.device)[:, None]
            flat = row * route_count + selected_route
            flat_slots = flat[..., None] * MAX_TRACE_DEPTH + torch.arange(
                MAX_TRACE_DEPTH, device=selected_route.device
            )

            def gather_scalar(value: wp.array) -> np.ndarray:
                return wp.to_torch(value)[flat].cpu().numpy()

            result = {
                "mask": selected_mask.cpu().numpy(),
                "route": selected_route.cpu().numpy().astype(np.int64),
                "path_length": gather_scalar(path_length).astype(np.float32),
                "departure": wp.to_torch(departure)[flat].cpu().numpy().astype(np.float32),
                "arrival": wp.to_torch(arrival)[flat].cpu().numpy().astype(np.float32),
                "energy": selected_energy.clamp_min(0.0).cpu().numpy().astype(np.float32),
                "interaction": wp.to_torch(output_interaction)[flat_slots]
                .cpu()
                .numpy()
                .astype(np.uint8),
                "family": wp.to_torch(output_family)[flat_slots]
                .cpu()
                .numpy()
                .astype(np.int32),
                "face": wp.to_torch(output_face)[flat_slots]
                .cpu()
                .numpy()
                .astype(np.int32),
                "vertices": wp.to_torch(output_vertices)[flat_slots]
                .cpu()
                .numpy()
                .astype(np.float32),
                "incidence": wp.to_torch(output_incidence)[flat_slots]
                .cpu()
                .numpy()
                .astype(np.float32),
            }
        return result


class SourceAlignedDiffuseSolver:
    def __init__(
        self,
        geometry: GeometryData,
        aligned: SourceAlignedGeometry,
        config: SourceAlignedConfig,
        records: dict[str, np.ndarray],
        images: np.ndarray,
        device: str,
    ) -> None:
        self.device = device
        self.config = config
        self.records = records
        wp.init()
        branch = config.scatter
        table = build_material_table(
            frequency_hz=config.frequency_hz,
            thickness_m=config.concrete_thickness_m,
            scattering_coefficient=branch.scattering_coefficient,
            samples=config.material_table_samples,
        )
        with wp.ScopedDevice(device):
            self.mesh = wp.Mesh(
                points=wp.array(geometry.vertices, dtype=wp.vec3, device=device),
                indices=wp.array(
                    geometry.faces.reshape(-1), dtype=wp.int32, device=device
                ),
            )
            self.face_plane = wp.array(
                aligned.face_plane, dtype=wp.int32, device=device
            )
            self.face_family = wp.array(
                geometry.face_family, dtype=wp.int32, device=device
            )
            self.face_normal = wp.array(
                aligned.face_normal, dtype=wp.vec3, device=device
            )
            self.prefix_depth = wp.array(
                records["prefix_depth"], dtype=wp.uint8, device=device
            )
            self.total_depth = wp.array(
                records["total_depth"], dtype=wp.uint8, device=device
            )
            self.interaction = wp.array(
                records["interaction"].reshape(-1),
                dtype=wp.uint8,
                device=device,
            )
            self.plane = wp.array(
                records["plane"].reshape(-1), dtype=wp.int32, device=device
            )
            self.face = wp.array(
                records["face"].reshape(-1), dtype=wp.int32, device=device
            )
            self.vertices = wp.array(
                records["vertices"].reshape(-1, 3), dtype=wp.vec3, device=device
            )
            self.images = wp.array(
                images.reshape(-1, 3), dtype=wp.vec3, device=device
            )
            self.table = {
                name: wp.array(getattr(table, name), dtype=wp.float32, device=device)
                for name in (
                    "specular_probability",
                    "diffuse_probability",
                    "transmission_probability",
                    "reflection_power",
                    "transmission_power",
                )
            }
            row = np.arange(len(records["prefix_depth"]), dtype=np.int64)
            point = records["vertices"][
                row, records["prefix_depth"].astype(np.int64) - 1
            ].astype(np.float32)
            self.point_t = torch.as_tensor(point, device=device)
            self.score_t = torch.as_tensor(
                np.maximum(records["score"], 1e-30), device=device
            )

    def solve(
        self,
        *,
        source: np.ndarray,
        tx_look: np.ndarray,
        targets: np.ndarray,
        maximum: int,
    ) -> dict[str, np.ndarray]:
        targets = np.asarray(targets, dtype=np.float32)
        target_t = torch.as_tensor(targets, device=self.device)
        distance_squared = torch.sum(
            torch.square(target_t[:, None] - self.point_t[None]), dim=2
        ).clamp_min_(1.0)
        upper = self.score_t[None] / distance_squared
        solve_count = min(self.config.scatter_solve_pool, upper.shape[1])
        _, selected = torch.topk(upper, solve_count, dim=1)
        total = len(targets) * solve_count
        slots = total * MAX_TRACE_DEPTH
        with wp.ScopedDevice(self.device):
            target_gpu = wp.array(targets, dtype=wp.vec3, device=self.device)
            selected_gpu = wp.from_torch(
                selected.reshape(-1).to(torch.int32), dtype=wp.int32
            )
            valid = wp.zeros(total, dtype=wp.uint8, device=self.device)
            path_length = wp.zeros(total, dtype=wp.float32, device=self.device)
            departure = wp.zeros(total, dtype=wp.vec3, device=self.device)
            arrival = wp.zeros(total, dtype=wp.vec3, device=self.device)
            energy = wp.zeros(total, dtype=wp.float32, device=self.device)
            output_interaction = wp.zeros(slots, dtype=wp.uint8, device=self.device)
            output_family = wp.full(slots, -1, dtype=wp.int32, device=self.device)
            output_face = wp.full(slots, -1, dtype=wp.int32, device=self.device)
            output_vertices = wp.zeros(slots, dtype=wp.vec3, device=self.device)
            output_incidence = wp.zeros(slots, dtype=wp.float32, device=self.device)
            wp.launch(
                solve_diffuse_routes,
                dim=total,
                inputs=(
                    self.mesh.id,
                    self.face_plane,
                    self.face_family,
                    self.face_normal,
                    wp.vec3(*map(float, source)),
                    wp.vec3(*map(float, tx_look)),
                    target_gpu,
                    solve_count,
                    selected_gpu,
                    self.prefix_depth,
                    self.total_depth,
                    self.interaction,
                    self.plane,
                    self.face,
                    self.vertices,
                    self.images,
                    float(self.config.scatter.scattering_coefficient),
                    int(self.config.scatter.rays_per_site_seed),
                    float(self.config.ray_epsilon_m),
                    self.table["specular_probability"],
                    self.table["diffuse_probability"],
                    self.table["transmission_probability"],
                    self.table["reflection_power"],
                    self.table["transmission_power"],
                ),
                outputs=(
                    valid,
                    path_length,
                    departure,
                    arrival,
                    energy,
                    output_interaction,
                    output_family,
                    output_face,
                    output_vertices,
                    output_incidence,
                ),
                device=self.device,
            )
            wp.synchronize_device(self.device)
            valid_t = wp.to_torch(valid).reshape(len(targets), solve_count).bool()
            energy_t = wp.to_torch(energy).reshape(len(targets), solve_count)
            ranked = torch.where(valid_t, energy_t, torch.full_like(energy_t, -torch.inf))
            keep = min(maximum, solve_count)
            selected_energy, local_index = torch.topk(ranked, keep, dim=1)
            selected_mask = torch.isfinite(selected_energy) & (selected_energy > 0.0)
            row = torch.arange(len(targets), device=local_index.device)[:, None]
            flat = row * solve_count + local_index
            flat_slots = flat[..., None] * MAX_TRACE_DEPTH + torch.arange(
                MAX_TRACE_DEPTH, device=local_index.device
            )
            record = torch.gather(selected, 1, local_index)

            def gather_scalar(value: wp.array) -> np.ndarray:
                return wp.to_torch(value)[flat].cpu().numpy()

            return {
                "mask": selected_mask.cpu().numpy(),
                "record": record.cpu().numpy().astype(np.int64),
                "path_length": gather_scalar(path_length).astype(np.float32),
                "departure": wp.to_torch(departure)[flat].cpu().numpy().astype(np.float32),
                "arrival": wp.to_torch(arrival)[flat].cpu().numpy().astype(np.float32),
                "energy": selected_energy.clamp_min(0.0).cpu().numpy().astype(np.float32),
                "interaction": wp.to_torch(output_interaction)[flat_slots]
                .cpu()
                .numpy()
                .astype(np.uint8),
                "family": wp.to_torch(output_family)[flat_slots]
                .cpu()
                .numpy()
                .astype(np.int32),
                "face": wp.to_torch(output_face)[flat_slots]
                .cpu()
                .numpy()
                .astype(np.int32),
                "vertices": wp.to_torch(output_vertices)[flat_slots]
                .cpu()
                .numpy()
                .astype(np.float32),
                "incidence": wp.to_torch(output_incidence)[flat_slots]
                .cpu()
                .numpy()
                .astype(np.float32),
            }


def direction_angles(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    unit = direction / np.maximum(norm, 1e-12)
    theta = np.arccos(np.clip(unit[..., 2], -1.0, 1.0))
    phi = np.arctan2(unit[..., 1], unit[..., 0])
    return theta.astype(np.float32), phi.astype(np.float32)


def solved_to_tokens(
    solved: dict[str, np.ndarray],
    routes: dict[str, np.ndarray],
    *,
    source_id: int,
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    count, maximum = solved["mask"].shape
    output = empty_token_arrays(count, maximum, MAX_TRACE_DEPTH)
    route = solved["route"]
    theta_t, phi_t = direction_angles(solved["departure"])
    theta_r, phi_r = direction_angles(solved["arrival"])
    output["mask"] = solved["mask"]
    output["source"].fill(source_id)
    output["path_length"] = solved["path_length"]
    output["tau"] = solved["path_length"] / np.float32(LIGHT_SPEED_M_S)
    output["theta_t"] = theta_t
    output["phi_t"] = phi_t
    output["theta_r"] = theta_r
    output["phi_r"] = phi_r
    output["relative_energy"] = solved["energy"]
    output["presence"] = routes["presence"][route]
    output["interaction_count"] = routes["depth"][route]
    output["interactions"] = solved["interaction"]
    output["surface_family"] = solved["family"]
    output["vertices"] = solved["vertices"]
    output["geometry_confidence"][solved["mask"]] = 1.0
    output["incidence"] = solved["incidence"]
    output["seed"].fill(-1)
    if normalize:
        for row in range(count):
            valid = output["mask"][row]
            total = float(
                np.sum(output["relative_energy"][row, valid], dtype=np.float64)
            )
            if total > 0.0:
                output["relative_energy"][row, valid] /= np.float32(total)
    return output


def solved_diffuse_to_tokens(
    solved: dict[str, np.ndarray],
    records: dict[str, np.ndarray],
    *,
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    count, maximum = solved["mask"].shape
    output = empty_token_arrays(count, maximum, MAX_TRACE_DEPTH)
    record = solved["record"]
    theta_t, phi_t = direction_angles(solved["departure"])
    theta_r, phi_r = direction_angles(solved["arrival"])
    output["mask"] = solved["mask"]
    output["source"].fill(1)
    output["path_length"] = solved["path_length"]
    output["tau"] = solved["path_length"] / np.float32(LIGHT_SPEED_M_S)
    output["theta_t"] = theta_t
    output["phi_t"] = phi_t
    output["theta_r"] = theta_r
    output["phi_r"] = phi_r
    output["relative_energy"] = solved["energy"]
    output["presence"] = records["presence"][record]
    output["interaction_count"] = records["total_depth"][record]
    output["interactions"] = solved["interaction"]
    output["surface_family"] = solved["family"]
    output["vertices"] = solved["vertices"]
    output["geometry_confidence"][solved["mask"]] = 1.0
    output["incidence"] = solved["incidence"]
    output["seed"] = records["seed"][record]
    if normalize:
        for row in range(count):
            valid = output["mask"][row]
            total = float(
                np.sum(output["relative_energy"][row, valid], dtype=np.float64)
            )
            if total > 0.0:
                output["relative_energy"][row, valid] /= np.float32(total)
    return output


def concatenate_candidate_groups(
    groups: list[dict[str, np.ndarray]],
    maximum: int,
    *,
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    if not groups:
        raise ValueError("at least one candidate group is required")
    count = groups[0]["mask"].shape[0]
    output = empty_token_arrays(count, maximum, MAX_TRACE_DEPTH)
    energy = np.concatenate([group["relative_energy"] for group in groups], axis=1)
    mask = np.concatenate([group["mask"] for group in groups], axis=1)
    ranked = np.where(mask, energy, -np.inf)
    keep = min(maximum, ranked.shape[1])
    index = np.argpartition(ranked, ranked.shape[1] - keep, axis=1)[:, -keep:]
    selected_energy = np.take_along_axis(ranked, index, axis=1)
    order = np.argsort(-selected_energy, axis=1)
    index = np.take_along_axis(index, order, axis=1)
    widths = np.cumsum([0, *[group["mask"].shape[1] for group in groups]])
    for row in range(count):
        for slot, global_index in enumerate(index[row]):
            group_id = int(np.searchsorted(widths[1:], global_index, side="right"))
            local = int(global_index - widths[group_id])
            group = groups[group_id]
            for name in TOKEN_DTYPES:
                output[name][row, slot] = group[name][row, local]
    if normalize:
        for row in range(count):
            valid = output["mask"][row]
            total = float(
                np.sum(output["relative_energy"][row, valid], dtype=np.float64)
            )
            if total > 0.0:
                output["relative_energy"][row, valid] /= np.float32(total)
    return output


def build_token_shard(
    *,
    positions: np.ndarray,
    point_ids: np.ndarray,
    site: int,
    shard: int,
    shard_size: int,
    bs_positions: np.ndarray,
    geometry: GeometryData,
    aligned: SourceAlignedGeometry,
    config: SourceAlignedConfig,
    device: str,
    force: bool = False,
) -> dict[str, Any]:
    output_path = token_shard_path(site, shard)
    report_path = output_path.with_suffix(".json")
    if output_path.is_file() and report_path.is_file() and not force:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") == "complete":
            return {"reused": True, **report}
    start = shard * shard_size
    end = min(start + shard_size, len(point_ids))
    selected_ids = point_ids[start:end]
    if len(selected_ids) == 0:
        raise ValueError(f"empty token shard site={site} shard={shard}")
    started = time.time()
    source = bs_positions[site].astype(np.float32)
    look = tx_look_vector(site, config)

    base_routes = merge_route_banks(site, config.base, config)
    base_images = route_images(source, base_routes, aligned)
    scatter_routes = merge_route_banks(site, config.scatter, config)
    scatter_images = route_images(source, scatter_routes, aligned)
    diffuse_records = merge_diffuse_banks(site, config)
    diffuse_images = diffuse_suffix_images(diffuse_records, aligned)

    base_solver = SourceAlignedRouteSolver(
        geometry, aligned, config, config.base, device
    )
    scatter_route_solver = SourceAlignedRouteSolver(
        geometry, aligned, config, config.scatter, device
    )
    diffuse_solver = SourceAlignedDiffuseSolver(
        geometry, aligned, config, diffuse_records, diffuse_images, device
    )
    arrays = empty_token_arrays(
        len(selected_ids), config.raw_candidate_count, MAX_TRACE_DEPTH
    )
    batch_timings: list[float] = []
    for batch_start in range(0, len(selected_ids), config.target_batch_size):
        batch_end = min(batch_start + config.target_batch_size, len(selected_ids))
        batch_ids = selected_ids[batch_start:batch_end]
        targets = positions[batch_ids].astype(np.float32)
        batch_started = time.time()
        base = solved_to_tokens(
            base_solver.solve(
                source=source,
                tx_look=look,
                targets=targets,
                routes=base_routes,
                images=base_images,
                maximum=config.base_pool,
            ),
            base_routes,
            source_id=0,
            normalize=False,
        )
        scatter_specular = solved_to_tokens(
            scatter_route_solver.solve(
                source=source,
                tx_look=look,
                targets=targets,
                routes=scatter_routes,
                images=scatter_images,
                maximum=config.scatter_pool,
            ),
            scatter_routes,
            source_id=1,
            normalize=False,
        )
        scatter_diffuse = solved_diffuse_to_tokens(
            diffuse_solver.solve(
                source=source,
                tx_look=look,
                targets=targets,
                maximum=config.scatter_pool,
            ),
            diffuse_records,
            normalize=False,
        )
        scatter = concatenate_candidate_groups(
            [scatter_specular, scatter_diffuse],
            config.scatter_pool,
            normalize=False,
        )
        combined = concatenate_candidate_groups(
            [base, scatter], config.raw_candidate_count, normalize=True
        )
        for name in TOKEN_DTYPES:
            arrays[name][batch_start:batch_end] = combined[name]
        batch_timings.append(float(time.time() - batch_started))
    atomic_npz(output_path, point_ids=selected_ids.astype(np.int64), **arrays)
    valid_per_point = arrays["mask"].sum(axis=1)
    report = {
        "status": "complete",
        "site": int(site),
        "shard": int(shard),
        "device": device,
        "point_count": int(len(selected_ids)),
        "valid_candidate_mean": float(np.mean(valid_per_point)),
        "valid_candidate_min": int(np.min(valid_per_point)),
        "empty_points": int(np.sum(valid_per_point == 0)),
        "base_candidates": int(
            np.sum(arrays["mask"] & (arrays["source"] == 0))
        ),
        "scatter_candidates": int(
            np.sum(arrays["mask"] & (arrays["source"] == 1))
        ),
        "seconds": float(time.time() - started),
        "mean_batch_seconds": float(np.mean(batch_timings)),
        "output": PATHS.relative(output_path),
    }
    atomic_json(report_path, report)
    return report


def assemble_token_shards(
    *,
    site_point_ids: dict[int, np.ndarray],
    point_count: int,
    shard_size: int,
    config: SourceAlignedConfig,
) -> dict[str, Any]:
    started = time.time()
    arrays = empty_token_arrays(
        point_count, config.raw_candidate_count, MAX_TRACE_DEPTH
    )
    assigned = np.zeros(point_count, dtype=bool)
    for site, point_ids in site_point_ids.items():
        for shard in range(int(math.ceil(len(point_ids) / shard_size))):
            path = token_shard_path(site, shard)
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as cache:
                ids = cache["point_ids"].astype(np.int64)
                if np.any(assigned[ids]):
                    raise RuntimeError("source-aligned token shards overlap")
                for name in TOKEN_DTYPES:
                    arrays[name][ids] = cache[name]
                assigned[ids] = True
    if not np.all(assigned):
        raise RuntimeError(
            f"source-aligned token assembly missed {int(np.sum(~assigned))} points"
        )
    mask = arrays["mask"]
    if not np.isfinite(arrays["path_length"][mask]).all():
        raise FloatingPointError("non-finite source-aligned path length")
    if not np.isfinite(arrays["relative_energy"][mask]).all():
        raise FloatingPointError("non-finite source-aligned relative energy")
    fallback_ids = np.flatnonzero(mask.sum(axis=1) == 0)
    mask[fallback_ids, 0] = True
    arrays["source"][fallback_ids, 0] = -1
    atomic_npz(RAW_TOKEN_CACHE, **arrays)
    return {
        "status": "complete",
        "point_count": int(point_count),
        "valid_candidate_mean": float(np.mean(mask.sum(axis=1))),
        "valid_candidate_min": int(np.min(mask.sum(axis=1))),
        "physical_empty_points": int(len(fallback_ids)),
        "adapter_fallback_points": int(len(fallback_ids)),
        "base_candidates": int(np.sum(mask & (arrays["source"] == 0))),
        "scatter_candidates": int(np.sum(mask & (arrays["source"] == 1))),
        "seconds": float(time.time() - started),
        "output": PATHS.relative(RAW_TOKEN_CACHE),
    }
