#!/usr/bin/env python3
"""Inventory geometric line-of-sight points in the official Round 2 map."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import trimesh

from data import load_positions
from geometry_features import GEOMETRY_CACHE
from project_paths import PATHS


VERSION = "Round2-E139-geometric-los-inventory"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "geometric_los_inventory"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_geometric_los_inventory"
LABELS = ARTIFACT_DIR / "geometric_los_labels.npz"
REPORT = ANALYSIS_DIR / "geometric_los_inventory.json"
MAP_PATH = PATHS.round2 / "Round2_Map.ply"
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)

ORIGIN_GUARD_M = 0.50
TARGET_GUARD_M = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--radii-m",
        type=float,
        nargs="+",
        default=(0.10, 0.25, 0.50),
    )
    return parser.parse_args()


def cast_clear_fan(
    mesh: trimesh.Trimesh,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    radius_m: float,
    batch_size: int,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    origins_center = np.asarray(bs_positions, dtype=np.float64)[site]
    delta_xy = positions[:, :2] - origins_center[:, :2]
    norm_xy = np.maximum(np.linalg.norm(delta_xy, axis=1, keepdims=True), 1e-12)
    perpendicular = np.column_stack((-delta_xy[:, 1], delta_xy[:, 0])) / norm_xy
    offsets = radius_m * np.asarray((-1.0, -0.5, 0.0, 0.5, 1.0))

    targets = np.repeat(positions[:, None, :], len(offsets), axis=1)
    targets[:, :, :2] += perpendicular[:, None, :] * offsets[None, :, None]
    origins = np.repeat(origins_center[:, None, :], len(offsets), axis=1)
    flat_origins = origins.reshape(-1, 3)
    flat_targets = targets.reshape(-1, 3)
    vectors = flat_targets - flat_origins
    lengths = np.linalg.norm(vectors, axis=1)
    directions = vectors / np.maximum(lengths[:, None], 1e-12)

    blocked = np.zeros(len(flat_origins), dtype=bool)
    for start in range(0, len(flat_origins), batch_size):
        end = min(start + batch_size, len(flat_origins))
        locations, ray_ids, _ = mesh.ray.intersects_location(
            flat_origins[start:end], directions[start:end], multiple_hits=True
        )
        if len(ray_ids) == 0:
            continue
        global_ids = ray_ids.astype(np.int64) + start
        distances = np.linalg.norm(locations - flat_origins[global_ids], axis=1)
        valid = (distances > ORIGIN_GUARD_M) & (
            distances < lengths[global_ids] - TARGET_GUARD_M
        )
        blocked[np.unique(global_ids[valid])] = True
    return (~blocked).reshape(len(positions), len(offsets))


def summarize(clear: np.ndarray, site: np.ndarray) -> dict[str, object]:
    center = clear[:, 2]
    stable_clear = np.all(clear, axis=1)
    stable_blocked = np.all(~clear, axis=1)
    boundary = ~(stable_clear | stable_blocked)

    def subset(mask: np.ndarray) -> dict[str, int | float]:
        count = int(np.sum(mask))
        return {
            "count": count,
            "center_clear": int(np.sum(center & mask)),
            "center_clear_fraction": float(np.mean(center[mask])) if count else 0.0,
            "stable_clear": int(np.sum(stable_clear & mask)),
            "stable_blocked": int(np.sum(stable_blocked & mask)),
            "boundary_sensitive": int(np.sum(boundary & mask)),
        }

    return {
        "all": subset(np.ones(len(clear), dtype=bool)),
        "site0": subset(site == 0),
        "site1": subset(site == 1),
    }


def radius_key(radius_m: float) -> str:
    return f"r{int(round(radius_m * 100.0)):03d}cm"


def summarize_center(center: np.ndarray) -> dict[str, int | float]:
    return {
        "count": int(len(center)),
        "center_clear": int(np.sum(center)),
        "center_blocked": int(np.sum(~center)),
        "center_clear_fraction": float(np.mean(center)),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    if any(radius <= 0.0 for radius in args.radii_m):
        raise ValueError("all radii must be positive")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    setup, train_positions, test_positions, train_site, test_site = load_positions()
    positions = np.concatenate((train_positions, test_positions), axis=0)
    site = np.concatenate((train_site, test_site), axis=0).astype(np.int64)

    mesh = trimesh.load(MAP_PATH, process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"expected Trimesh, got {type(mesh).__name__}")

    cached = np.load(GEOMETRY_CACHE, allow_pickle=False)
    names = cached["names"].astype(str).tolist()
    center_index = names.index("center_hit_count")
    cached_center = np.concatenate(
        (
            cached["train"][:, center_index] == 0.0,
            cached["test"][:, center_index] == 0.0,
        )
    )

    fan_by_radius: dict[str, np.ndarray] = {}
    summaries: dict[str, object] = {}
    for radius in args.radii_m:
        key = radius_key(radius)
        clear = cast_clear_fan(
            mesh,
            positions,
            site,
            setup.bs_positions,
            radius,
            args.batch_size,
        )
        if not np.array_equal(clear[:, 2], cached_center):
            raise RuntimeError(f"fresh center rays disagree with geometry cache at {key}")
        fan_by_radius[key] = clear
        summaries[key] = {
            "radius_m": float(radius),
            "train": summarize(clear[: len(train_positions)], train_site),
            "test": summarize(clear[len(train_positions) :], test_site),
        }

    train_center = cached_center[: len(train_positions)]
    test_center = cached_center[len(train_positions) :]
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    canonical_clear = train_center[query_ids]
    np.savez_compressed(
        LABELS,
        train_center_clear=train_center,
        test_center_clear=test_center,
        train_site=train_site.astype(np.int8),
        test_site=test_site.astype(np.int8),
        **{
            f"train_clear_fan_{key}": clear[: len(train_positions)]
            for key, clear in fan_by_radius.items()
        },
        **{
            f"test_clear_fan_{key}": clear[len(train_positions) :]
            for key, clear in fan_by_radius.items()
        },
    )

    payload = {
        "experiment": VERSION,
        "definition": {
            "center_los": (
                "No official-map triangle intersection on the serving-BS-to-UE segment "
                f"after {ORIGIN_GUARD_M:.2f} m from the BS and before "
                f"{TARGET_GUARD_M:.2f} m from the UE."
            ),
            "sensitivity_fan": (
                "Five parallel-in-local-azimuth endpoint rays at offsets "
                "[-r, -r/2, 0, r/2, r]."
            ),
            "uses_channel_h": False,
            "uses_sionna": False,
        },
        "map": {
            "vertices": int(len(mesh.vertices)),
            "triangles": int(len(mesh.faces)),
        },
        "center_counts": {
            "train": summarize_center(train_center),
            "test": summarize_center(test_center),
            "all_clear": int(np.sum(cached_center)),
            "all_points": int(len(cached_center)),
        },
        "canonical_fixed500": {
            "center_clear": int(np.sum(canonical_clear)),
            "center_blocked": int(np.sum(~canonical_clear)),
            "clear_by_fold": [
                int(np.sum(canonical_clear[fold_id == fold])) for fold in range(5)
            ],
        },
        "sensitivity": summaries,
        "outputs": {
            "labels": PATHS.relative(LABELS),
            "report": PATHS.relative(REPORT),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
