#!/usr/bin/env python3
"""Build global, wall-focused, and track-structured positions for v442."""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
from scipy.spatial import cKDTree

from project_paths import PATHS


VERSION_NUMBER = 442
VERSION = "v442_global_track_positions"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
POSITIONS_PATH = OUT_DIR / "positions.npz"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
SCENE_CACHE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"
V437_POSITIONS = PATHS.cache_dir(437) / "positions.npz"

LOWER = np.asarray([45.0, -205.0], dtype=np.float32)
UPPER = np.asarray([255.0, 105.0], dtype=np.float32)
MIN_V437_DISTANCE = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=44_201)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def tile_split(xy: np.ndarray) -> np.ndarray:
    tile_x = np.floor((xy[:, 0] - LOWER[0]) / 20.0).astype(np.int64)
    tile_y = np.floor((xy[:, 1] - LOWER[1]) / 20.0).astype(np.int64)
    value = np.mod(tile_x * 73_856_093 + tile_y * 19_349_663, 10)
    split = np.zeros(len(xy), dtype=np.int8)
    split[value == 0] = 1
    split[value == 1] = 2
    return split


def draw_candidates(
    rng: np.random.Generator,
    count: int,
    train_xy: np.ndarray,
    patches: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mode = rng.choice(4, size=count, p=np.asarray([0.40, 0.25, 0.25, 0.10]))
    output = np.empty((count, 2), dtype=np.float32)

    uniform = mode == 0
    output[uniform] = rng.uniform(LOWER, UPPER, size=(int(np.sum(uniform)), 2))

    local = mode == 1
    local_count = int(np.sum(local))
    if local_count:
        anchor = train_xy[rng.integers(0, len(train_xy), size=local_count)]
        angle = rng.uniform(-math.pi, math.pi, size=local_count)
        radius = np.sqrt(rng.uniform(0.2**2, 30.0**2, size=local_count))
        output[local] = anchor + np.column_stack(
            [radius * np.cos(angle), radius * np.sin(angle)]
        )

    wall = mode == 2
    wall_count = int(np.sum(wall))
    if wall_count:
        selected = patches[rng.integers(0, len(patches), size=wall_count)]
        normal = selected[:, :2]
        tangent = np.column_stack([-normal[:, 1], normal[:, 0]])
        along = rng.uniform(selected[:, 3], selected[:, 4])
        point = normal * selected[:, 2:3] + tangent * along[:, None]
        offset = rng.uniform(2.0, 40.0, size=wall_count)
        sign = rng.choice(np.asarray([-1.0, 1.0]), size=wall_count)
        output[wall] = point + normal * (offset * sign)[:, None]

    track = mode == 3
    track_count = int(np.sum(track))
    if track_count:
        anchor = rng.uniform(LOWER, UPPER, size=(track_count, 2))
        angle = rng.uniform(-math.pi, math.pi, size=track_count)
        direction = np.column_stack([np.cos(angle), np.sin(angle)])
        along = rng.integers(-80, 81, size=track_count).astype(np.float32) * 0.75
        cross = rng.normal(0.0, 0.35, size=track_count).astype(np.float32)
        normal = np.column_stack([-direction[:, 1], direction[:, 0]])
        output[track] = anchor + direction * along[:, None] + normal * cross[:, None]
    return output.astype(np.float32), mode.astype(np.int8)


def main() -> None:
    args = parse_args()
    if POSITIONS_PATH.is_file() and not args.overwrite:
        cached = np.load(POSITIONS_PATH)
        if len(cached["positions"]) != args.count:
            raise ValueError(
                f"Existing manifest has {len(cached['positions'])} points, "
                f"requested {args.count}; use --overwrite"
            )
        print(f"reused {PATHS.relative(POSITIONS_PATH)}", flush=True)
        return
    if args.count < 10:
        raise ValueError("count must be at least 10")
    for path in (SCENE_CACHE, V437_POSITIONS):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    train = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    official_test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    v437 = np.load(V437_POSITIONS)["positions"].astype(np.float32)
    scene = np.load(SCENE_CACHE)
    patches = scene["wall_patches"].astype(np.float32)
    trees = {
        "train": cKDTree(train[:, :2]),
        "test": cKDTree(official_test[:, :2]),
        "v437": cKDTree(v437[:, :2]),
    }
    validation_count = args.count // 10
    holdout_count = args.count // 10
    quota = np.asarray(
        [args.count - validation_count - holdout_count, validation_count, holdout_count],
        dtype=np.int64,
    )
    accepted_xy: list[list[np.ndarray]] = [[], [], []]
    accepted_mode: list[list[np.ndarray]] = [[], [], []]
    accepted_distance: list[list[np.ndarray]] = [[], [], []]
    counts = np.zeros(3, dtype=np.int64)
    seen: set[tuple[int, int]] = set()
    rng = np.random.default_rng(args.seed)
    started = time.time()
    rounds = 0
    while np.any(counts < quota):
        rounds += 1
        candidate, mode = draw_candidates(rng, 100_000, train[:, :2], patches)
        inside = np.all((candidate >= LOWER) & (candidate <= UPPER), axis=1)
        candidate = candidate[inside]
        mode = mode[inside]
        split = tile_split(candidate)
        distance_test = trees["test"].query(candidate, k=1, workers=-1)[0]
        distance_train = trees["train"].query(candidate, k=1, workers=-1)[0]
        distance_v437 = trees["v437"].query(candidate, k=1, workers=-1)[0]
        keep = (distance_test >= 1.0) & (distance_train >= 0.05)
        keep &= (split == 0) | (distance_v437 >= MIN_V437_DISTANCE)
        candidate = candidate[keep]
        mode = mode[keep]
        split = split[keep]
        distance_test = distance_test[keep]
        distance_train = distance_train[keep]
        distance_v437 = distance_v437[keep]
        for group in range(3):
            remaining = int(quota[group] - counts[group])
            if remaining <= 0:
                continue
            rows = np.flatnonzero(split == group)
            if not len(rows):
                continue
            rng.shuffle(rows)
            selected_rows = []
            for row in rows:
                point = candidate[row]
                key = (
                    int(round(float(point[0]) * 10.0)),
                    int(round(float(point[1]) * 10.0)),
                )
                if key in seen:
                    continue
                seen.add(key)
                selected_rows.append(int(row))
                if len(selected_rows) >= remaining:
                    break
            if not selected_rows:
                continue
            selected = np.asarray(selected_rows, dtype=np.int64)
            accepted_xy[group].append(candidate[selected])
            accepted_mode[group].append(mode[selected])
            accepted_distance[group].append(
                np.column_stack(
                    [
                        distance_test[selected],
                        distance_train[selected],
                        distance_v437[selected],
                    ]
                ).astype(np.float32)
            )
            counts[group] += len(selected)
        print(
            f"v442 positions round={rounds} counts={counts.tolist()} "
            f"quota={quota.tolist()}",
            flush=True,
        )
    xy_parts = []
    mode_parts = []
    split_parts = []
    distance_parts = []
    for group in range(3):
        xy = np.concatenate(accepted_xy[group], axis=0)[: quota[group]]
        sample_mode = np.concatenate(accepted_mode[group], axis=0)[: quota[group]]
        distance = np.concatenate(accepted_distance[group], axis=0)[: quota[group]]
        xy_parts.append(xy)
        mode_parts.append(sample_mode)
        split_parts.append(np.full(len(xy), group, dtype=np.int8))
        distance_parts.append(distance)
    xy = np.concatenate(xy_parts)
    sample_mode = np.concatenate(mode_parts)
    split = np.concatenate(split_parts)
    distance = np.concatenate(distance_parts)
    positions = np.column_stack(
        [xy, np.full(args.count, 1.5, dtype=np.float32)]
    ).astype(np.float32)
    order = rng.permutation(args.count)
    np.savez(
        POSITIONS_PATH,
        positions=positions[order],
        sample_mode=sample_mode[order],
        pretrain_split=split[order],
        min_official_test_distance=distance[order, 0],
        min_official_train_distance=distance[order, 1],
        min_v437_distance=distance[order, 2],
    )
    report = {
        "method": VERSION,
        "status": "positions_ready",
        "count": int(args.count),
        "split_counts": np.bincount(split, minlength=3).astype(int).tolist(),
        "sample_mode_counts": np.bincount(sample_mode, minlength=4).astype(int).tolist(),
        "bounds": {"lower": LOWER.tolist(), "upper": UPPER.tolist()},
        "independence": {
            "minimum_v437_distance_for_validation_holdout": MIN_V437_DISTANCE,
            "observed_validation_min": float(np.min(distance[split == 1, 2])),
            "observed_holdout_min": float(np.min(distance[split == 2, 2])),
            "official_test_min": float(np.min(distance[:, 0])),
        },
        "rounds": int(rounds),
        "seconds": float(time.time() - started),
        "output": PATHS.relative(POSITIONS_PATH),
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
