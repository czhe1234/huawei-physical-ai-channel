#!/usr/bin/env python3
"""Trace official-point-cloud geometry candidates at v442 synthetic positions."""

from __future__ import annotations

import argparse
import json

import numpy as np

import build_v251_pointcloud_sionna_paths as base
from project_paths import PATHS


VERSION_NUMBER = 467
VERSION = "v467_synthetic_pointcloud_candidate_prior"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
SCENE_DIR = OUT_DIR / "scene"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / "v467_synthetic_pointcloud_candidates.json"
OUTPUT = OUT_DIR / "v467_synthetic_pointcloud_candidates.npz"
POSITION_SOURCE = PATHS.cache_dir(443) / "positions.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-paths", type=int, default=128)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--samples-per-src", type=int, default=1_000_000)
    parser.add_argument("--max-num-paths-per-src", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=46_701)
    parser.add_argument("--frequency", type=float, default=6.0e9)
    parser.add_argument("--subcarrier-spacing", type=float, default=30e3)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not POSITION_SOURCE.is_file():
        raise FileNotFoundError(POSITION_SOURCE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    base.OUT_DIR = OUT_DIR
    base.SCENE_DIR = SCENE_DIR
    scene_xml = base.prepare_scene_files()

    import mitsuba as mi

    mi.set_variant(args.mitsuba_variant)
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    all_positions = np.load(POSITION_SOURCE, mmap_mode="r")
    count = min(max(args.count, 0), len(all_positions))
    positions = np.asarray(all_positions[:count], dtype=np.float32)

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
        position=mi.Point3f(*map(float, base.BS_POS)),
        look_at=mi.Point3f(0.0, 0.0, 1.5),
    )
    scene.add(tx)

    trace_args = argparse.Namespace(
        batch_size=args.batch_size,
        max_paths=args.max_paths,
        max_depth=args.max_depth,
        samples_per_src=args.samples_per_src,
        max_num_paths_per_src=args.max_num_paths_per_src,
        seed=args.seed,
        subcarrier_spacing=args.subcarrier_spacing,
        diffuse_reflection=False,
        refraction=True,
        diffraction=False,
        edge_diffraction=False,
        depth_balanced_ranking=True,
    )
    arrays, trace_report = base.trace_split(
        prefix="synthetic",
        positions=positions,
        args=trace_args,
        scene=scene,
        tx=tx,
        mi=mi,
        Receiver=Receiver,
        PathSolver=PathSolver,
    )
    payload: dict[str, np.ndarray] = {
        "indices": np.arange(count, dtype=np.int32),
        "pos": positions,
        "bs_pos": base.BS_POS,
    }
    payload.update(arrays)
    np.savez_compressed(OUTPUT, **payload)

    report = {
        "method": VERSION,
        "status": "geometry_cache_only",
        "primary_hypothesis": (
            "A fixed official-point-cloud candidate universe at synthetic positions can be "
            "labeled by v442 true-scene paths without the v445 power-selected membership leak."
        ),
        "configuration": vars(args),
        "mitsuba_variant": mi.variant(),
        "trace": trace_report,
        "outputs": {
            "candidates": PATHS.relative(OUTPUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
