#!/usr/bin/env python3
"""Run the validated v437 Sionna pipeline on the v442 position manifest."""

from __future__ import annotations

import generate_v437_single_gpu_sionna_pretraining_dataset as generator

from project_paths import PATHS


generator.VERSION_NUMBER = 442
generator.VERSION = "v442_global_track_sionna_dataset"
generator.OUT_DIR = PATHS.cache_dir(442)
generator.SHARD_ROOT = generator.OUT_DIR / "shards"
generator.POSITIONS_PATH = generator.OUT_DIR / "positions.npz"
generator.REPORT_DIR = PATHS.version_report_dir(442)
generator.REPORT = generator.REPORT_DIR / f"{generator.VERSION}.json"


if __name__ == "__main__":
    generator.main()
