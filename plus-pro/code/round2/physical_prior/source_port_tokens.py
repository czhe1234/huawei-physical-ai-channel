"""Generate frozen raw512 PathTokens from the E209 source port."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np

from project_paths import PATHS
from round2.data import load_positions
from round2.physical_prior.config import (
    DELAY_BINS,
    HORIZONTAL_BINS,
    VERTICAL_BINS,
)
from round2.physical_prior.geometry import GeometryData, load_geometry
from round2.physical_prior.source_port import PathSupportSession
from round2.physical_prior.source_port_common import CONFIGS, E185_SEEDS


VERSION = "Round2-E210-source-port-full-channel"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e210_source_port_full_channel"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e210_source_port_full_channel"
SUPPORT_DIR = ARTIFACT_DIR / "support"
TOKEN_DIR = ARTIFACT_DIR / "token_shards"
RAW_TOKEN_CACHE = ARTIFACT_DIR / "raw_tokens_n4000_k512.npz"
TOKEN_REPORT = ANALYSIS_DIR / "raw_token_report.json"
GEOMETRY_CACHE = PATHS.artifacts / "round2" / "physical_prior" / "geometry.npz"

BATCH_SIZE = 16
RAW_CANDIDATES = 512
MAX_DEPTH = 16
BASE_PER_SEED = 96
SCATTER_PER_SEED = 64
BASE_POOL = 256
SCATTER_POOL = 256
SUBCARRIER_SPACING_HZ = 30_000.0

SUPPORT_FIELDS = (
    "tau",
    "path_length_m",
    "angles",
    "interactions",
    "primitives",
    "surface_plane",
    "vertices",
    "incidence_angle",
    "interaction_count",
    "relative_energy",
    "geometry_valid",
    "seed",
    "source",
)


def _token_helpers():
    from round2.physical_prior.tokens import TOKEN_DTYPES, empty_token_arrays

    return TOKEN_DTYPES, empty_token_arrays


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


def support_path(config: str, seed: int, site: int, batch: int) -> Path:
    return (
        SUPPORT_DIR
        / config
        / f"site{site}"
        / f"seed{seed}"
        / f"batch{batch:03d}.npz"
    )


def token_shard_path(site: int, batch: int) -> Path:
    return TOKEN_DIR / f"site{site}_batch{batch:03d}.npz"


def report_path(path: Path) -> Path:
    return path.with_suffix(".json")


def complete(path: Path) -> bool:
    if not path.is_file() or not report_path(path).is_file():
        return False
    try:
        return json.loads(report_path(path).read_text(encoding="utf-8")).get(
            "status"
        ) == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def site_point_ids(site: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    setup, positions, _, train_site, _ = load_positions()
    ids = np.flatnonzero(train_site == site).astype(np.int64)
    return ids, np.asarray(positions, dtype=np.float32), np.asarray(
        setup.bs_positions, dtype=np.float32
    )


def batch_count(site: int) -> int:
    ids, _, _ = site_point_ids(site)
    return int(math.ceil(len(ids) / BATCH_SIZE))


def _prune_support(
    support: dict[str, np.ndarray], maximum: int
) -> dict[str, np.ndarray]:
    point_ids = np.asarray(support["point_ids"], dtype=np.int64)
    offsets = np.asarray(support["offsets"], dtype=np.int64)
    selected_parts: list[np.ndarray] = []
    new_offsets = [0]
    for row in range(len(point_ids)):
        start, end = int(offsets[row]), int(offsets[row + 1])
        indices = np.arange(start, end, dtype=np.int64)
        if len(indices):
            valid = np.asarray(support["geometry_valid"][indices], dtype=bool)
            valid &= np.isfinite(support["relative_energy"][indices])
            valid &= np.isfinite(support["tau"][indices])
            indices = indices[valid]
        count = min(maximum, len(indices))
        if count:
            energy = np.asarray(
                support["relative_energy"][indices], dtype=np.float64
            )
            local = np.argpartition(energy, len(energy) - count)[-count:]
            chosen = np.sort(indices[local])
        else:
            chosen = np.empty(0, dtype=np.int64)
        selected_parts.append(chosen)
        new_offsets.append(new_offsets[-1] + len(chosen))
    selected = (
        np.concatenate(selected_parts)
        if selected_parts
        else np.empty(0, dtype=np.int64)
    )
    output = {
        "point_ids": point_ids,
        "offsets": np.asarray(new_offsets, dtype=np.int64),
    }
    for field in SUPPORT_FIELDS:
        output[field] = np.asarray(support[field][selected])
    return output


def build_support_worker(
    *,
    config_name: str,
    seed: int,
    site: int,
    cuda_device: int,
    only_batch: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    ids, positions, bs_positions = site_point_ids(site)
    maximum = BASE_PER_SEED if config_name == "base" else SCATTER_PER_SEED
    batches = [only_batch] if only_batch is not None else list(range(batch_count(site)))
    pending = [
        batch
        for batch in batches
        if force or not complete(support_path(config_name, seed, site, batch))
    ]
    if not pending:
        return {
            "status": "complete",
            "reused": True,
            "config": config_name,
            "seed": int(seed),
            "site": int(site),
            "batch_count": len(batches),
        }

    started = time.time()
    session = PathSupportSession(
        site=site,
        bs_position=bs_positions[site],
        config=config_name,
        seed=seed,
        cuda_device=cuda_device,
    )
    records: list[dict[str, Any]] = []
    try:
        for batch in pending:
            start = batch * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(ids))
            point_ids = ids[start:end]
            batch_started = time.time()
            support, timing = session.solve(
                positions=positions[point_ids],
                point_ids=point_ids,
                split="train",
            )
            raw_count = int(support["offsets"][-1])
            support = _prune_support(support, maximum)
            output = support_path(config_name, seed, site, batch)
            atomic_npz(output, **support)
            record = {
                "status": "complete",
                "version": VERSION,
                "config": config_name,
                "seed": int(seed),
                "site": int(site),
                "cuda_device": int(cuda_device),
                "batch": int(batch),
                "point_ids": point_ids.tolist(),
                "raw_path_count": raw_count,
                "selected_path_count": int(support["offsets"][-1]),
                "seconds": float(time.time() - batch_started),
                "timings": timing["timings"],
                "output": PATHS.relative(output),
            }
            atomic_json(report_path(output), record)
            records.append(record)
            print(
                f"E210 support {config_name} seed={seed} site={site} "
                f"batch={batch + 1}/{batch_count(site)} raw={raw_count} "
                f"kept={record['selected_path_count']} "
                f"seconds={record['seconds']:.2f}",
                flush=True,
            )
    finally:
        session.close()
    return {
        "status": "complete",
        "config": config_name,
        "seed": int(seed),
        "site": int(site),
        "cuda_device": int(cuda_device),
        "batch_count": len(records),
        "seconds": float(time.time() - started),
        "peak_rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        / (1024.0 * 1024.0),
    }


def _load_support(config: str, seed: int, site: int, batch: int) -> dict[str, np.ndarray]:
    path = support_path(config, seed, site, batch)
    if not complete(path):
        raise FileNotFoundError(f"incomplete E210 support: {path}")
    with np.load(path, allow_pickle=False) as cache:
        return {name: np.asarray(cache[name]) for name in cache.files}


def _path_coordinate(
    tau: float, theta_t: float, phi_t: float, config_name: str, site: int
) -> tuple[float, float, float]:
    config = CONFIGS[config_name]
    yaw = math.radians(config.site_yaw_deg[site])
    tilt = math.radians(config.site_tilt_deg[site])
    horizontal_cycles = config.tx_horizontal_spacing * math.sin(theta_t) * math.sin(
        phi_t - yaw
    )
    vertical_cycles = config.tx_vertical_spacing * (
        math.sin(tilt) * math.sin(theta_t) * math.cos(phi_t - yaw)
        - math.cos(tilt) * math.cos(theta_t)
    )
    return (
        (tau * SUBCARRIER_SPACING_HZ * DELAY_BINS) % DELAY_BINS,
        (HORIZONTAL_BINS * horizontal_cycles) % HORIZONTAL_BINS,
        (VERTICAL_BINS * vertical_cycles) % VERTICAL_BINS,
    )


def _sequence_key(support: dict[str, np.ndarray], index: int) -> tuple[Any, ...]:
    depth = min(int(support["interaction_count"][index]), MAX_DEPTH)
    return tuple(
        (
            int(support["interactions"][index, step]),
            *(
                int(value)
                for value in support["surface_plane"][index, step]
            ),
        )
        for step in range(depth)
    )


def _base_identity(
    support: dict[str, np.ndarray], index: int, site: int
) -> tuple[Any, ...]:
    delay, horizontal, vertical = _path_coordinate(
        float(support["tau"][index]),
        float(support["angles"][index, 0]),
        float(support["angles"][index, 1]),
        "base",
        site,
    )
    return (
        int(round(delay * 50.0)),
        int(round(horizontal * 50.0)),
        int(round(vertical * 50.0)),
        _sequence_key(support, index),
    )


def _scatter_identity(
    support: dict[str, np.ndarray], index: int, site: int
) -> tuple[Any, ...]:
    delay, horizontal, vertical = _path_coordinate(
        float(support["tau"][index]),
        float(support["angles"][index, 0]),
        float(support["angles"][index, 1]),
        "scatter",
        site,
    )
    depth = min(int(support["interaction_count"][index]), MAX_DEPTH)
    sequence = _sequence_key(support, index)
    boundary = sequence if depth <= 2 else (sequence[0], sequence[-1])
    return (
        int(round(delay * 4.0)),
        int(round(horizontal * 4.0)),
        int(round(vertical * 4.0)),
        depth,
        boundary,
    )


def _point_indices(support: dict[str, np.ndarray], row: int) -> np.ndarray:
    start = int(support["offsets"][row])
    end = int(support["offsets"][row + 1])
    return np.arange(start, end, dtype=np.int64)


def _record(
    support: dict[str, np.ndarray], index: int, presence: int = 1
) -> dict[str, Any]:
    return {
        "support": support,
        "index": int(index),
        "presence": int(presence),
    }


def _energy(record: dict[str, Any]) -> float:
    return float(record["support"]["relative_energy"][record["index"]])


def _family_plane_keys(geometry: GeometryData) -> dict[tuple[int, ...], list[int]]:
    output: dict[tuple[int, ...], list[int]] = {}
    for family in range(geometry.family_count):
        normal = np.asarray(geometry.family_normal[family], dtype=np.float64).copy()
        nonzero = np.flatnonzero(np.abs(normal) > 1e-8)
        if len(nonzero) and normal[nonzero[0]] < 0.0:
            normal = -normal
        offset = float(np.dot(normal, geometry.family_centroid[family]))
        key = (
            *np.rint(normal * 100_000.0).astype(np.int64).tolist(),
            int(np.rint(offset * 100.0)),
        )
        output.setdefault(key, []).append(family)
    return output


def _surface_family(
    geometry: GeometryData,
    plane_lookup: dict[tuple[int, ...], list[int]],
    primitive: int,
    plane: np.ndarray,
    vertex: np.ndarray,
) -> int:
    key = tuple(int(value) for value in plane)
    if 0 <= primitive < len(geometry.face_family):
        family = int(geometry.face_family[primitive])
        candidates = plane_lookup.get(key)
        if candidates is None or family in candidates:
            return family
    candidates = plane_lookup.get(key, ())
    if not candidates:
        return int(geometry.ground_family)
    centers = geometry.family_centroid[np.asarray(candidates, dtype=np.int64)]
    distance = np.linalg.norm(centers - np.asarray(vertex), axis=1)
    return int(candidates[int(np.argmin(distance))])


def _write_record(
    arrays: dict[str, np.ndarray],
    row: int,
    slot: int,
    record: dict[str, Any],
    geometry: GeometryData,
    plane_lookup: dict[tuple[int, ...], list[int]],
) -> None:
    support = record["support"]
    index = record["index"]
    depth = min(int(support["interaction_count"][index]), MAX_DEPTH)
    arrays["mask"][row, slot] = True
    arrays["source"][row, slot] = support["source"][index]
    arrays["tau"][row, slot] = support["tau"][index]
    arrays["path_length"][row, slot] = support["path_length_m"][index]
    arrays["theta_t"][row, slot] = support["angles"][index, 0]
    arrays["phi_t"][row, slot] = support["angles"][index, 1]
    arrays["theta_r"][row, slot] = support["angles"][index, 2]
    arrays["phi_r"][row, slot] = support["angles"][index, 3]
    arrays["relative_energy"][row, slot] = max(_energy(record), 0.0)
    arrays["presence"][row, slot] = min(record["presence"], 255)
    arrays["interaction_count"][row, slot] = depth
    arrays["interactions"][row, slot, :depth] = support["interactions"][
        index, :depth
    ]
    arrays["vertices"][row, slot, :depth] = support["vertices"][index, :depth]
    arrays["incidence"][row, slot, :depth] = np.cos(
        support["incidence_angle"][index, :depth]
    ).astype(np.float32)
    for step in range(depth):
        arrays["surface_family"][row, slot, step] = _surface_family(
            geometry,
            plane_lookup,
            int(support["primitives"][index, step]),
            support["surface_plane"][index, step],
            support["vertices"][index, step],
        )
    arrays["geometry_confidence"][row, slot] = 1.0
    arrays["seed"][row, slot] = support["seed"][index]


def build_token_shard(site: int, batch: int, force: bool = False) -> dict[str, Any]:
    output = token_shard_path(site, batch)
    if complete(output) and not force:
        return json.loads(report_path(output).read_text(encoding="utf-8"))
    supports = {
        (config, seed): _load_support(config, seed, site, batch)
        for config in CONFIGS
        for seed in E185_SEEDS
    }
    reference_ids = supports[("base", E185_SEEDS[0])]["point_ids"].astype(
        np.int64
    )
    for support in supports.values():
        if not np.array_equal(reference_ids, support["point_ids"]):
            raise RuntimeError(f"E210 support point mismatch site={site} batch={batch}")

    geometry = load_geometry(GEOMETRY_CACHE)
    plane_lookup = _family_plane_keys(geometry)
    _, empty_token_arrays = _token_helpers()
    arrays = empty_token_arrays(len(reference_ids), RAW_CANDIDATES, MAX_DEPTH)
    started = time.time()
    base_counts: list[int] = []
    scatter_counts: list[int] = []
    for row in range(len(reference_ids)):
        base_bank: dict[tuple[Any, ...], dict[str, Any]] = {}
        for seed in E185_SEEDS:
            support = supports[("base", seed)]
            for index in _point_indices(support, row):
                candidate = _record(support, int(index))
                key = _base_identity(support, int(index), site)
                previous = base_bank.get(key)
                if previous is None:
                    base_bank[key] = candidate
                    continue
                presence = previous["presence"] + 1
                if _energy(candidate) > _energy(previous):
                    candidate["presence"] = presence
                    base_bank[key] = candidate
                else:
                    previous["presence"] = presence
        base = sorted(
            base_bank.values(),
            key=lambda value: _energy(value)
            * (1.0 + 0.05 * (value["presence"] - 1)),
            reverse=True,
        )[:BASE_POOL]

        scatter: list[dict[str, Any]] = []
        for seed in E185_SEEDS:
            support = supports[("scatter", seed)]
            local: dict[tuple[Any, ...], dict[str, Any]] = {}
            for index in _point_indices(support, row):
                candidate = _record(support, int(index))
                key = _scatter_identity(support, int(index), site)
                previous = local.get(key)
                if previous is None or _energy(candidate) > _energy(previous):
                    local[key] = candidate
            scatter.extend(local.values())
            scatter = sorted(scatter, key=_energy, reverse=True)[:SCATTER_POOL]
        scatter = sorted(scatter, key=_energy, reverse=True)[:SCATTER_POOL]
        base_counts.append(len(base))
        scatter_counts.append(len(scatter))
        selected = [*base, *scatter]
        for slot, record in enumerate(selected[:RAW_CANDIDATES]):
            _write_record(
                arrays,
                row,
                slot,
                record,
                geometry,
                plane_lookup,
            )
        valid = arrays["mask"][row]
        total = float(
            np.sum(arrays["relative_energy"][row, valid], dtype=np.float64)
        )
        if total > 0.0:
            arrays["relative_energy"][row, valid] /= np.float32(total)
        if not np.any(valid):
            arrays["mask"][row, 0] = True
            arrays["source"][row, 0] = -1

    atomic_npz(output, point_ids=reference_ids, **arrays)
    record = {
        "status": "complete",
        "version": VERSION,
        "site": int(site),
        "batch": int(batch),
        "point_count": int(len(reference_ids)),
        "base_candidate_mean": float(np.mean(base_counts)),
        "scatter_candidate_mean": float(np.mean(scatter_counts)),
        "empty_points": int(np.sum(arrays["source"][:, 0] < 0)),
        "seconds": float(time.time() - started),
        "output": PATHS.relative(output),
    }
    atomic_json(report_path(output), record)
    return record


def build_all_token_shards(workers: int, force: bool = False) -> dict[str, Any]:
    jobs = [
        (site, batch)
        for site in range(2)
        for batch in range(batch_count(site))
        if force or not complete(token_shard_path(site, batch))
    ]
    started = time.time()
    records: list[dict[str, Any]] = []
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(build_token_shard, site, batch, force): (site, batch)
                for site, batch in jobs
            }
            for future in as_completed(futures):
                site, batch = futures[future]
                record = future.result()
                records.append(record)
                print(
                    f"E210 tokens site={site} batch={batch + 1}/{batch_count(site)} "
                    f"seconds={record['seconds']:.2f}",
                    flush=True,
                )
    return {
        "status": "complete",
        "jobs": len(records),
        "workers": int(workers),
        "seconds": float(time.time() - started),
    }


def assemble_tokens(force: bool = False) -> dict[str, Any]:
    if RAW_TOKEN_CACHE.is_file() and TOKEN_REPORT.is_file() and not force:
        return json.loads(TOKEN_REPORT.read_text(encoding="utf-8"))
    _, positions, _, train_site, _ = load_positions()
    token_dtypes, empty_token_arrays = _token_helpers()
    arrays = empty_token_arrays(len(positions), RAW_CANDIDATES, MAX_DEPTH)
    assigned = np.zeros(len(positions), dtype=bool)
    started = time.time()
    for site in range(2):
        for batch in range(batch_count(site)):
            path = token_shard_path(site, batch)
            if not complete(path):
                raise FileNotFoundError(f"incomplete token shard: {path}")
            with np.load(path, allow_pickle=False) as cache:
                ids = cache["point_ids"].astype(np.int64)
                if np.any(assigned[ids]):
                    raise RuntimeError("E210 token shards overlap")
                if not np.all(train_site[ids] == site):
                    raise RuntimeError("E210 token shard site mismatch")
                for name in token_dtypes:
                    arrays[name][ids] = cache[name]
                assigned[ids] = True
    if not np.all(assigned):
        raise RuntimeError(f"E210 missed {int(np.sum(~assigned))} train points")
    mask = arrays["mask"]
    if not np.isfinite(arrays["relative_energy"][mask]).all():
        raise FloatingPointError("E210 relative energy is non-finite")
    if not np.isfinite(arrays["path_length"][mask]).all():
        raise FloatingPointError("E210 path length is non-finite")
    atomic_npz(RAW_TOKEN_CACHE, **arrays)
    valid_count = mask.sum(axis=1)
    record = {
        "status": "complete",
        "version": VERSION,
        "point_count": int(len(positions)),
        "shape": list(mask.shape),
        "valid_candidate_mean": float(np.mean(valid_count)),
        "valid_candidate_min": int(np.min(valid_count)),
        "adapter_fallback_points": int(
            np.sum(np.any(mask & (arrays["source"] < 0), axis=1))
        ),
        "base_candidates": int(np.sum(mask & (arrays["source"] == 0))),
        "scatter_candidates": int(np.sum(mask & (arrays["source"] == 1))),
        "seconds": float(time.time() - started),
        "peak_rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        / (1024.0 * 1024.0),
        "output": PATHS.relative(RAW_TOKEN_CACHE),
    }
    atomic_json(TOKEN_REPORT, record)
    return record


__all__ = [
    "ANALYSIS_DIR",
    "ARTIFACT_DIR",
    "BATCH_SIZE",
    "RAW_TOKEN_CACHE",
    "TOKEN_REPORT",
    "VERSION",
    "assemble_tokens",
    "batch_count",
    "build_all_token_shards",
    "build_support_worker",
    "build_token_shard",
    "complete",
    "support_path",
    "token_shard_path",
]
