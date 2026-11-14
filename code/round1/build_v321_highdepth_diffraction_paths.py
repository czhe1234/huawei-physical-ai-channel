#!/usr/bin/env python3
"""Trace depth-16 diffraction paths on the v291 official-pointcloud mesh."""

from __future__ import annotations

import build_v251_pointcloud_sionna_paths as base
from project_paths import PATHS


base.VERSION_NUMBER = 321
base.VERSION = "v321_highdepth_diffraction_paths"
base.OUT_DIR = PATHS.cache_dir(base.VERSION_NUMBER)
base.SCENE_DIR = base.OUT_DIR / "scene"
base.REPORT_DIR = PATHS.version_report_dir(base.VERSION_NUMBER)
base.REPORT = base.REPORT_DIR / f"{base.VERSION}.json"
base.OUTPUT = base.OUT_DIR / f"{base.VERSION}.npz"


if __name__ == "__main__":
    base.main()
