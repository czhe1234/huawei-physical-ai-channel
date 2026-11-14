#!/usr/bin/env python3
"""Trace v291-equivalent high-depth paths on the official Poisson mesh."""

from __future__ import annotations

import argparse
from pathlib import Path

import build_v251_pointcloud_sionna_paths as base
from project_paths import PATHS


VERSION_NUMBER = 323
VERSION = "v323_highdepth_poisson_paths"
SOURCE_MESH = PATHS.cache_dir(253) / "scene" / "official_poisson_mesh.ply"
SOURCE_SCENE = PATHS.cache_dir(253) / "scene" / "poisson_scene.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--test-count", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-paths", type=int, default=128)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--samples-per-src", type=int, default=1_000_000)
    parser.add_argument("--max-num-paths-per-src", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=32301)
    parser.add_argument("--frequency", type=float, default=6.0e9)
    parser.add_argument("--subcarrier-spacing", type=float, default=30e3)
    parser.add_argument("--mitsuba-variant", default="llvm_ad_mono_polarized")
    parser.add_argument(
        "--diffuse-reflection",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--refraction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--diffraction",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--edge-diffraction",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--depth-balanced-ranking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def prepare_scene_files() -> Path:
    if not SOURCE_MESH.is_file():
        raise FileNotFoundError(SOURCE_MESH)
    if not SOURCE_SCENE.is_file():
        raise FileNotFoundError(SOURCE_SCENE)
    return SOURCE_SCENE


base.VERSION_NUMBER = VERSION_NUMBER
base.VERSION = VERSION
base.OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
base.SCENE_DIR = base.OUT_DIR / "scene"
base.REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
base.REPORT = base.REPORT_DIR / f"{VERSION}.json"
base.OUTPUT = base.OUT_DIR / f"{VERSION}.npz"
base.SOURCE_OBJ = SOURCE_MESH
base.parse_args = parse_args
base.prepare_scene_files = prepare_scene_files


if __name__ == "__main__":
    base.main()
