#!/usr/bin/env python3
"""Merge four E200 finalist caches and build the fixed 128-path model inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np

import analyze_path_candidate_information_audit as e84
import train_surface_interaction_energy_model as e85
from e200_config_search import SearchConfig, portable_payload
from generate_e200_finalist_paths import (
    FINALISTS,
    PATHS_PER_CONFIG,
    ROOT as SOURCE_ROOT,
    safe_name,
)
from project_paths import PATHS
from round2.data import load_positions
from train_explicit_path_pdp_predictor import build_features


NAMESPACE = os.environ.get("HW_E200_NAMESPACE", "e200").strip().lower()
if NAMESPACE not in {"e200", "e201"}:
    raise ValueError(f"unsupported E200-compatible namespace: {NAMESPACE}")
VERSION = f"Round2-{NAMESPACE.upper()}-unified-candidate-cache"
ARTIFACT_DIR = PATHS.artifacts / "round2" / f"{NAMESPACE}_candidate_cache"
CACHE_DIR = ARTIFACT_DIR / "train_k128"
FEATURES = ARTIFACT_DIR / "path_features_train_n4000.npy"
COORDINATE = ARTIFACT_DIR / "coordinate_f32.npy"
PORT_PRIOR = ARTIFACT_DIR / "port_prior_f16.npy"
SEQUENCE_GEOMETRY = ARTIFACT_DIR / "sequence_geometry_f16.npy"
SEQUENCE_HASH = ARTIFACT_DIR / "sequence_hash_i32.npy"
REPORT = (
    PATHS.analysis / "results" / f"round2_{NAMESPACE}_candidate_cache" / "report.json"
)
POINTS = 4000
CANDIDATES = 128
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(portable_payload(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def digest_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(PATHS.relative(path).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def cache_specs() -> dict[str, tuple[np.dtype, tuple[int, ...], Any]]:
    shape = (POINTS, CANDIDATES)
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
        "interactions": (np.dtype(np.uint8), shape + (16,), 0),
        "objects": (np.dtype(np.uint32), shape + (16,), 0),
        "primitives": (np.dtype(np.uint32), shape + (16,), 0),
        "vertices": (np.dtype(np.float32), shape + (16, 3), 0.0),
        "rows": (np.dtype(np.int64), (POINTS,), 0),
    }


def valid_existing(lineage_digest: str) -> bool:
    if not REPORT.is_file():
        return False
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or report.get("lineage_digest") != lineage_digest:
        return False
    required = [CACHE_DIR / f"{name}.npy" for name in cache_specs()]
    required.extend((FEATURES, COORDINATE, PORT_PRIOR, SEQUENCE_GEOMETRY, SEQUENCE_HASH))
    return all(path.is_file() for path in required)


def main() -> None:
    args = parse_args()
    started = time.time()
    finalists = [
        SearchConfig.from_dict(value)
        for value in json.loads(FINALISTS.read_text(encoding="utf-8"))
    ]
    source_roots = [SOURCE_ROOT / f"source{safe_name(config.name)}" for config in finalists]
    source_files = [root / "state.json" for root in source_roots]
    missing = [PATHS.relative(path) for path in source_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing E200 finalist caches: {missing}")
    for path in source_files:
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("status") != "complete" or int(state.get("complete_points", 0)) != POINTS:
            raise RuntimeError(f"incomplete E200 source cache: {PATHS.relative(path)}")
    lineage_digest = digest_files(source_files + [FINALISTS])
    if not args.force and valid_existing(lineage_digest):
        print(REPORT.read_text(encoding="utf-8"), flush=True)
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for path in (FEATURES, COORDINATE, PORT_PRIOR, SEQUENCE_GEOMETRY, SEQUENCE_HASH):
        if args.force:
            path.unlink(missing_ok=True)
    outputs: dict[str, np.memmap] = {}
    for name, (dtype, shape, fill) in cache_specs().items():
        path = CACHE_DIR / f"{name}.npy"
        if args.force:
            path.unlink(missing_ok=True)
        value = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        value[...] = fill
        outputs[name] = value
    outputs["rows"][:] = np.arange(POINTS, dtype=np.int64)
    copied_bytes = 0
    names = tuple(name for name in cache_specs() if name != "rows")
    for source, root in enumerate(source_roots):
        target_slots = slice(source * PATHS_PER_CONFIG, (source + 1) * PATHS_PER_CONFIG)
        for name in names:
            source_array = np.load(root / f"{name}.npy", mmap_mode="r")
            for start in range(0, POINTS, args.batch_size):
                end = min(start + args.batch_size, POINTS)
                outputs[name][start:end, target_slots] = source_array[start:end]
            outputs[name].flush()
            copied_bytes += int(source_array.nbytes)
        print(f"E200 merged finalist {source + 1}/4", flush=True)
    for value in outputs.values():
        value.flush()

    setup, train_positions, _, train_site, _ = load_positions()
    tau = np.load(CACHE_DIR / "tau.npy", mmap_mode="r")
    angles = np.load(CACHE_DIR / "angles.npy", mmap_mode="r")
    coordinate = np.zeros((POINTS, CANDIDATES, 3), dtype=np.float32)
    alignment_records = []
    for source, config in enumerate(finalists):
        screen_report = json.loads(
            (
                PATHS.analysis
                / "results"
                / f"round2_{NAMESPACE}_config_search"
                / "configs"
                / f"{safe_name(config.name)}.json"
            ).read_text(encoding="utf-8")
        )
        slots = slice(source * PATHS_PER_CONFIG, (source + 1) * PATHS_PER_CONFIG)
        local = e84.sionna_coordinates(
            np.asarray(tau[:, slots], dtype=np.float32),
            np.asarray(angles[:, slots], dtype=np.float32),
            train_site,
            config.to_dict(),
            vertical_alignment=0.0,
        )
        horizontal = np.asarray(
            [
                screen_report["angle_alignment"][f"site{site}"]["horizontal_shift"]
                for site in train_site
            ],
            dtype=np.float32,
        )
        vertical = np.asarray(
            [
                screen_report["angle_alignment"][f"site{site}"]["vertical_shift"]
                for site in train_site
            ],
            dtype=np.float32,
        )
        local[..., 0] = np.mod(local[..., 0] + horizontal[:, None], 16.0)
        local[..., 1] = np.mod(local[..., 1] + vertical[:, None], 8.0)
        coordinate[:, slots] = local
        alignment_records.append(
            {
                "source": source,
                "name": config.name,
                "horizontal": [
                    int(
                        screen_report["angle_alignment"][f"site{site}"][
                            "horizontal_shift"
                        ]
                    )
                    for site in (0, 1)
                ],
                "vertical": [
                    int(
                        screen_report["angle_alignment"][f"site{site}"][
                            "vertical_shift"
                        ]
                    )
                    for site in (0, 1)
                ],
            }
        )
    np.save(COORDINATE, coordinate)

    coefficient = np.load(CACHE_DIR / "coefficient.npy", mmap_mode="r")
    port = np.lib.format.open_memmap(
        PORT_PRIOR,
        mode="w+",
        dtype=np.float16,
        shape=(POINTS, CANDIDATES, e85.MP, e85.N_UE),
    )
    for start in range(0, POINTS, args.batch_size):
        end = min(start + args.batch_size, POINTS)
        shaped = np.asarray(coefficient[start:end]).reshape(
            end - start, CANDIDATES, e85.MP, e85.MH, e85.MV, e85.N_UE
        )
        power = np.square(np.abs(shaped), dtype=np.float32).sum(
            axis=(3, 4), dtype=np.float64
        )
        power /= np.maximum(power.sum(axis=(2, 3), keepdims=True), TINY)
        port[start:end] = power.astype(np.float16)
    port.flush()

    vertices = np.load(CACHE_DIR / "vertices.npy", mmap_mode="r")
    depth = np.load(CACHE_DIR / "interaction_count.npy", mmap_mode="r")
    center = np.concatenate(
        (train_positions[:, :2].mean(axis=0), np.asarray([1.5], dtype=np.float32))
    ).astype(np.float32)
    scale = np.concatenate(
        (
            np.maximum(train_positions[:, :2].std(axis=0), 1.0),
            np.asarray([20.0], dtype=np.float32),
        )
    ).astype(np.float32)
    probe = e85.sequence_geometry_block(
        slice(0, 1),
        vertices,
        depth,
        train_positions,
        train_site,
        setup.bs_positions,
        center,
        scale,
    )
    sequence_geometry = np.lib.format.open_memmap(
        SEQUENCE_GEOMETRY,
        mode="w+",
        dtype=np.float16,
        shape=(POINTS, CANDIDATES, e85.STEPS, probe.shape[-1]),
    )
    for start in range(0, POINTS, args.batch_size):
        end = min(start + args.batch_size, POINTS)
        sequence_geometry[start:end] = e85.sequence_geometry_block(
            slice(start, end),
            vertices,
            depth,
            train_positions,
            train_site,
            setup.bs_positions,
            center,
            scale,
        ).astype(np.float16)
    sequence_geometry.flush()

    sequence_source = {
        name: np.load(CACHE_DIR / f"{name}.npy", mmap_mode="r")
        for name in (
            "source",
            "interaction_count",
            "interactions",
            "objects",
            "primitives",
        )
    }
    hashes = e84.sequence_hashes(sequence_source)
    buckets = np.stack(
        (
            hashes % np.uint64(e85.Config().sequence_hash_buckets),
            (hashes >> np.uint64(32))
            % np.uint64(e85.Config().sequence_hash_buckets),
        ),
        axis=-1,
    ).astype(np.int32)
    np.save(SEQUENCE_HASH, buckets)

    regime = np.load(
        PATHS.artifacts
        / "round2"
        / "explicit_path_pdp_predictor"
        / "path_regime_train_n4000.npy"
    ).astype(np.int8)
    rows = np.arange(POINTS, dtype=np.int64)
    features = build_features(
        rows,
        rows,
        regime,
        FEATURES,
        path_cache=CACHE_DIR,
        split="train",
        source_classes=(0, 1, 2, 3),
    )
    mask = np.load(CACHE_DIR / "mask.npy", mmap_mode="r")
    report = {
        "experiment": VERSION,
        "status": "complete",
        "lineage_digest": lineage_digest,
        "sources": [config.to_dict() for config in finalists],
        "alignment": alignment_records,
        "output": {
            "cache": PATHS.relative(CACHE_DIR),
            "features": PATHS.relative(FEATURES),
            "coordinate": PATHS.relative(COORDINATE),
            "port_prior": PATHS.relative(PORT_PRIOR),
            "sequence_geometry": PATHS.relative(SEQUENCE_GEOMETRY),
            "sequence_hash": PATHS.relative(SEQUENCE_HASH),
            "feature_shape": list(features.shape),
            "mean_valid_paths": float(np.mean(np.sum(mask, axis=1))),
            "minimum_valid_paths": int(np.min(np.sum(mask, axis=1))),
            "maximum_valid_paths": int(np.max(np.sum(mask, axis=1))),
        },
        "data_boundary": {
            "configuration_selection_used_fixed500": False,
            "candidate_generation_uses_same_position_sionna": True,
            "official_test_h_used": False,
            "historical_v7_v15_prediction_h_read": False,
        },
        "performance": {
            "copied_gib": copied_bytes / 2**30,
            "peak_rss_gib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            ),
            "wall_seconds": float(time.time() - started),
            "measured_bottleneck": "candidate coefficient merge and feature extraction",
        },
    }
    atomic_json(REPORT, report)
    print(json.dumps(portable_payload(report), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
