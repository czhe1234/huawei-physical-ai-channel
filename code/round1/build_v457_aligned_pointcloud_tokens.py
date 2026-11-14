#!/usr/bin/env python3
"""Build deployment-aligned point-cloud path tokens at v442 positions."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import build_v251_pointcloud_sionna_paths as pointcloud
from project_paths import PATHS


VERSION_NUMBER = 457
VERSION = "v457_aligned_pointcloud_tokens"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"

POSITION_SOURCE = PATHS.cache_dir(443) / "positions.npy"
SPLIT_SOURCE = PATHS.cache_dir(443) / "split.npy"
BS = np.asarray([50.0, 0.0, 25.0], dtype=np.float32)
LIGHT_SPEED = 299_792_458.0
DELAY_SCALE = 192.0 * 30_000.0
FEATURE_COUNT = 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-count", type=int, default=16_000)
    parser.add_argument("--validation-count", type=int, default=2_000)
    parser.add_argument("--holdout-count", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-paths", type=int, default=128)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--samples-per-src", type=int, default=1_000_000)
    parser.add_argument("--max-num-paths-per-src", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=45_701)
    parser.add_argument("--frequency", type=float, default=6.0e9)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    parser.add_argument("--full", action="store_true")
    return parser.parse_args()


def selected_ids(split: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.full:
        return np.arange(len(split), dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    counts = (args.train_count, args.validation_count, args.holdout_count)
    parts = []
    for value, count in enumerate(counts):
        available = np.flatnonzero(split == value)
        if count > len(available):
            raise ValueError(f"Requested {count} rows from split {value}, only {len(available)}")
        parts.append(np.sort(rng.choice(available, count, replace=False)))
    return np.concatenate(parts).astype(np.int64)


def open_memmap(name: str, dtype, shape: tuple[int, ...]) -> np.memmap:
    path = OUT_DIR / f"{name}.npy"
    partial = path.with_name(path.stem + ".partial.npy")
    if partial.exists():
        partial.unlink()
    return np.lib.format.open_memmap(partial, mode="w+", dtype=dtype, shape=shape)


def finish_memmap(name: str, value: np.memmap) -> None:
    value.flush()
    del value
    path = OUT_DIR / f"{name}.npy"
    path.with_name(path.stem + ".partial.npy").replace(path)


def path_features(item: dict[str, np.ndarray], ue: np.ndarray) -> np.ndarray:
    count = len(item["ids"])
    if count == 0:
        return np.empty((0, FEATURE_COUNT), dtype=np.float32)
    tau = item["tau"].astype(np.float32)
    length = tau * LIGHT_SPEED
    direct_length = float(np.linalg.norm(ue - BS))
    depth = item["interaction_count"].astype(np.float32)
    mask = item["interaction_mask"].astype(np.int16)
    angles = [item[name].astype(np.float32) for name in ("theta_t", "phi_t", "theta_r", "phi_r")]
    angle_features = np.column_stack(
        [value for angle in angles for value in (np.sin(angle), np.cos(angle))]
    )
    interactions = item["interactions"]
    vertices = item["vertices"].astype(np.float32)
    active = interactions != 0
    active_count = np.sum(active, axis=1)
    first_index = np.argmax(active, axis=1)
    last_index = np.maximum(active_count - 1, 0)
    row = np.arange(count)
    first = vertices[row, first_index].copy()
    last = vertices[row, last_index].copy()
    missing = active_count == 0
    first[missing] = BS
    last[missing] = BS
    global_vertex = np.column_stack(
        [
            (first[:, 0] - 150.0) / 150.0,
            (first[:, 1] + 50.0) / 200.0,
            first[:, 2] / 50.0,
            (last[:, 0] - 150.0) / 150.0,
            (last[:, 1] + 50.0) / 200.0,
            last[:, 2] / 50.0,
        ]
    )
    relative_vertex = np.column_stack(
        [
            (first[:, 0] - ue[0]) / 300.0,
            (first[:, 1] - ue[1]) / 300.0,
            (last[:, 0] - ue[0]) / 300.0,
            (last[:, 1] - ue[1]) / 300.0,
        ]
    )
    output = np.column_stack(
        [
            tau * 1e6 / 8.0,
            tau * DELAY_SCALE / 48.0,
            length / 600.0,
            np.maximum(length - direct_length, 0.0) / 500.0,
            depth / 16.0,
            (depth == 0).astype(np.float32),
            *[((mask & bit) != 0).astype(np.float32) for bit in (1, 2, 4, 8)],
            angle_features,
            global_vertex,
            relative_vertex,
        ]
    ).astype(np.float32)
    if output.shape != (count, FEATURE_COUNT):
        raise ValueError(f"Unexpected path feature shape {output.shape}")
    return np.clip(output, -4.0, 4.0)


def main() -> None:
    args = parse_args()
    for path in (POSITION_SOURCE, SPLIT_SOURCE, pointcloud.SOURCE_OBJ):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pointcloud.SCENE_DIR = OUT_DIR / "scene"
    positions = np.load(POSITION_SOURCE, mmap_mode="r")
    split = np.load(SPLIT_SOURCE, mmap_mode="r")
    ids = selected_ids(split, args)
    np.save(OUT_DIR / "source_ids.npy", ids)
    count = len(ids)
    feature = open_memmap(
        "path_candidate_feature", np.float16, (count, args.max_paths, FEATURE_COUNT)
    )
    valid = open_memmap("path_candidate_valid", np.bool_, (count, args.max_paths))
    feature[:] = 0
    valid[:] = False

    import mitsuba as mi

    mi.set_variant(args.mitsuba_variant)
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    scene_xml = pointcloud.prepare_scene_files()
    scene = load_scene(str(scene_xml), merge_shapes=True)
    scene.frequency = args.frequency
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    tx = Transmitter(
        name="tx",
        position=mi.Point3f(*map(float, BS)),
        look_at=mi.Point3f(0.0, 0.0, 1.5),
    )
    scene.add(tx)
    solver = PathSolver()
    receiver_names: list[str] = []
    path_counts = np.zeros(count, dtype=np.int16)
    started = time.time()
    for start in range(0, count, args.batch_size):
        end = min(start + args.batch_size, count)
        pointcloud.remove_receivers(scene, receiver_names)
        receiver_names = []
        for local, source_id in enumerate(ids[start:end]):
            name = f"rx_{start + local:06d}"
            scene.add(
                Receiver(
                    name=name,
                    position=mi.Point3f(*map(float, positions[source_id])),
                    look_at=tx,
                )
            )
            receiver_names.append(name)
        paths = solver(
            scene,
            max_depth=args.max_depth,
            max_num_paths_per_src=args.max_num_paths_per_src,
            samples_per_src=args.samples_per_src,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            diffuse_reflection=True,
            refraction=True,
            diffraction=True,
            edge_diffraction=True,
            diffraction_lit_region=True,
            seed=args.seed + start,
        )
        selected = pointcloud.select_paths(
            paths,
            args.max_paths,
            args.max_depth,
            30_000.0,
            True,
        )
        for local, item in enumerate(selected):
            row = start + local
            current = path_features(item, np.asarray(positions[ids[row]], dtype=np.float32))
            path_count = min(len(current), args.max_paths)
            if path_count:
                feature[row, :path_count] = current[:path_count].astype(np.float16)
                valid[row, :path_count] = True
            path_counts[row] = path_count
        if end % 1024 < args.batch_size or end == count:
            elapsed = time.time() - started
            print(
                f"{VERSION} {end}/{count} mean_paths={path_counts[:end].mean():.2f} "
                f"seconds={elapsed:.1f}",
                flush=True,
            )
    pointcloud.remove_receivers(scene, receiver_names)
    finish_memmap("path_candidate_feature", feature)
    finish_memmap("path_candidate_valid", valid)
    np.save(OUT_DIR / "path_count.npy", path_counts)
    report = {
        "method": VERSION,
        "status": "deployment_aligned_geometry_tokens_ready",
        "primary_hypothesis": (
            "Using the same official-point-cloud candidate generator for synthetic "
            "pretraining and official deployment removes v445's path-set membership bias."
        ),
        "parameters": vars(args),
        "count": int(count),
        "split_counts": {
            str(value): int(np.sum(np.asarray(split[ids]) == value)) for value in range(3)
        },
        "path_count": {
            "mean": float(np.mean(path_counts)),
            "median": float(np.median(path_counts)),
            "p90": float(np.quantile(path_counts, 0.9)),
            "zero": int(np.sum(path_counts == 0)),
        },
        "seconds": float(time.time() - started),
        "outputs": {
            "source_ids": PATHS.relative(OUT_DIR / "source_ids.npy"),
            "feature": PATHS.relative(OUT_DIR / "path_candidate_feature.npy"),
            "valid": PATHS.relative(OUT_DIR / "path_candidate_valid.npy"),
        },
        "input_exclusions": [
            "Sionna path power",
            "Sionna path coefficient",
            "official test H",
            "online score",
        ],
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
