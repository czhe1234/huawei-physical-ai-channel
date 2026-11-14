#!/usr/bin/env python3
"""Align seed-42 base-path relative energy with the frozen E214 candidates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import time
import traceback
import zipfile

import numpy as np

from project_paths import PATHS
import build_e214_geometry_path_topology as e214_topology


VERSION = "Round2-E215-relative-energy-diagnostic"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e215_relative_energy"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e215_relative_energy"
ENERGY_PATH = ARTIFACT_DIR / "base_s1_relative_energy.npy"
STATE_PATH = ANALYSIS_DIR / "relative_energy_alignment.json"
FAILURE_PATH = ANALYSIS_DIR / "failure_alignment.json"

SOURCE = "base"
SEED = e214_topology.SINGLE_SEED
REQUESTED_FIELDS = e214_topology.ALLOWED_FIELDS + ("relative_energy",)
UNUSED_PROPAGATION_FIELDS = (
    "absolute_energy",
    "complex_coefficient",
    "full_complex_coefficient",
    "energy_rank",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def load_requested(path: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    with zipfile.ZipFile(path) as archive:
        members = {item.filename.removesuffix(".npy"): item for item in archive.infolist()}
    missing = [name for name in REQUESTED_FIELDS if name not in members]
    if missing:
        raise KeyError(f"missing requested fields in {path}: {missing}")
    with np.load(path, allow_pickle=False) as archive:
        # This explicit list is the E215 lineage boundary.
        arrays = {name: np.array(archive[name], copy=True) for name in REQUESTED_FIELDS}
    return arrays, {
        "members": len(REQUESTED_FIELDS),
        "compressed_bytes": sum(members[name].compress_size for name in REQUESTED_FIELDS),
        "uncompressed_bytes": sum(members[name].file_size for name in REQUESTED_FIELDS),
    }


def representative_energy(
    arrays: dict[str, np.ndarray], point_index: int, site: int
) -> tuple[list[dict[str, object]], dict[int, float]]:
    start = int(arrays["offsets"][point_index])
    end = int(arrays["offsets"][point_index + 1])
    indices = e214_topology.valid_local_indices(arrays, start, end)
    if len(indices) == 0:
        return [], {}
    coordinate = e214_topology.path_coordinates(
        arrays["tau"][indices], arrays["angles"][indices], site
    )
    depth = arrays["interaction_count"][indices].astype(np.int64)
    hashes = e214_topology.family_hashes(
        SOURCE,
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
    energy_by_hash: dict[int, float] = {}
    for local_index in order:
        key = int(hashes[local_index])
        if key in energy_by_hash:
            continue
        raw_index = int(indices[local_index])
        energy_by_hash[key] = float(arrays["relative_energy"][raw_index])
        if len(energy_by_hash) >= e214_topology.PER_SEED_LIMIT[SOURCE]:
            break

    records = e214_topology.local_candidates(SOURCE, arrays, point_index, site)
    for record in records:
        record["presence_count"] = 1
    selected = e214_topology.select_final(records)
    if any(int(record["hash"]) not in energy_by_hash for record in selected):
        raise RuntimeError(f"missing representative energy at site={site} point={point_index}")
    return selected, energy_by_hash


def build(force: bool) -> dict[str, object]:
    if ENERGY_PATH.is_file() and STATE_PATH.is_file() and not force:
        previous = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if previous.get("status") == "complete":
            return {"status": "reused", **previous}

    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    topology_dir = e214_topology.merged_dir(SOURCE, "s1")
    frozen_mask = np.load(topology_dir / "mask.npy", mmap_mode="r")
    frozen_hash = np.load(topology_dir / "family_hash.npy", mmap_mode="r")
    if frozen_mask.shape != (e214_topology.POINT_COUNT, e214_topology.PATH_CAPACITY):
        raise RuntimeError(f"unexpected frozen topology shape: {frozen_mask.shape}")

    partial = ENERGY_PATH.with_name(f".{ENERGY_PATH.name}.partial-{os.getpid()}")
    output = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.float32,
        shape=(e214_topology.POINT_COUNT, e214_topology.PATH_CAPACITY),
    )
    output[:] = 0.0
    seen = np.zeros(e214_topology.POINT_COUNT, dtype=np.uint8)
    shard_count = 0
    requested_members = 0
    compressed_bytes = 0
    uncompressed_bytes = 0
    aligned_paths = 0
    selected_mass: list[float] = []

    for batch_index, (site, name) in enumerate(e214_topology.source_batches(SOURCE), start=1):
        raw_path = e214_topology.raw_path(SOURCE, SEED, site, name)
        arrays, audit = load_requested(raw_path)
        shard_count += 1
        requested_members += audit["members"]
        compressed_bytes += audit["compressed_bytes"]
        uncompressed_bytes += audit["uncompressed_bytes"]
        point_ids = arrays["point_ids"].astype(np.int64)
        if np.any(seen[point_ids]):
            raise RuntimeError(f"duplicate point ids in {raw_path}")
        for local_point, point_id in enumerate(point_ids):
            selected, energy_by_hash = representative_energy(arrays, local_point, site)
            expected_count = int(np.sum(frozen_mask[point_id]))
            if len(selected) != expected_count:
                raise RuntimeError(
                    f"candidate count mismatch point={point_id}: {len(selected)} != {expected_count}"
                )
            selected_hash = np.asarray(
                [int(record["hash"]) for record in selected], dtype=np.uint64
            )
            if not np.array_equal(frozen_hash[point_id, :expected_count], selected_hash):
                raise RuntimeError(f"candidate order mismatch point={point_id}")
            values = np.asarray(
                [energy_by_hash[int(value)] for value in selected_hash], dtype=np.float32
            )
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise FloatingPointError(f"invalid relative energy point={point_id}")
            output[point_id, :expected_count] = values
            selected_mass.append(float(np.sum(values)))
            aligned_paths += expected_count
        seen[point_ids] = 1
        if batch_index % 25 == 0 or batch_index == 250:
            output.flush()
            print(
                f"E215 relative-energy alignment {batch_index}/250 paths={aligned_paths}",
                flush=True,
            )

    if not np.all(seen == 1):
        raise RuntimeError(f"missing {int(np.sum(seen == 0))} training points")
    valid_values = np.asarray(output)[np.asarray(frozen_mask)]
    if len(valid_values) != aligned_paths:
        raise RuntimeError("aligned path total changed")
    output.flush()
    del output
    os.replace(partial, ENERGY_PATH)

    quantiles = np.quantile(valid_values, (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0))
    report = {
        "experiment": VERSION,
        "status": "complete",
        "source": SOURCE,
        "seed": SEED,
        "frozen_topology": "E214/base_s1",
        "point_count": int(np.sum(seen)),
        "aligned_paths": aligned_paths,
        "exact_family_hash_order": True,
        "relative_energy": {
            "path": PATHS.relative(ENERGY_PATH),
            "shape": list(np.load(ENERGY_PATH, mmap_mode="r").shape),
            "dtype": str(np.load(ENERGY_PATH, mmap_mode="r").dtype),
            "minimum": float(np.min(valid_values)),
            "maximum": float(np.max(valid_values)),
            "mean": float(np.mean(valid_values)),
            "quantiles": [float(value) for value in quantiles],
            "selected_mass_mean": float(np.mean(selected_mass)),
            "selected_mass_min": float(np.min(selected_mass)),
        },
        "data_lineage": {
            "requested_fields": list(REQUESTED_FIELDS),
            "added_field": "relative_energy",
            "unused_propagation_fields": list(UNUSED_PROPAGATION_FIELDS),
            "complex_fields_accessed": [],
            "requested_members": requested_members,
            "compressed_bytes": compressed_bytes,
            "uncompressed_bytes": uncompressed_bytes,
            "shards": shard_count,
        },
        "wall_seconds": float(time.time() - started),
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "exit_code": 0,
    }
    atomic_json(STATE_PATH, report)
    return report


def main() -> None:
    args = parse_args()
    try:
        print(json.dumps(build(args.force), ensure_ascii=True, indent=2), flush=True)
    except Exception as error:
        payload = {
            "experiment": VERSION,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "exit_code": 1,
        }
        atomic_json(FAILURE_PATH, payload)
        raise


if __name__ == "__main__":
    main()
