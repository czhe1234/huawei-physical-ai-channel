#!/usr/bin/env python3
"""Launch disjoint v442 workers on one visible GPU."""

from __future__ import annotations

import run_v437_single_gpu_generation as launcher

from project_paths import PATHS


launcher.GENERATOR = PATHS.code / "generate_v442_global_track_sionna_dataset.py"
launcher.LOG_DIR = PATHS.cache_dir(442) / "worker_logs"


if __name__ == "__main__":
    launcher.main()
