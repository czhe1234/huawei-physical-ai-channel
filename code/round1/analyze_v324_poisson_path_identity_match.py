#!/usr/bin/env python3
"""Match official-H components to v323 Poisson-mesh high-depth paths."""

from __future__ import annotations

import analyze_v312_highdepth_path_identity_match as base
from project_paths import PATHS


VERSION_NUMBER = 324
VERSION = "v324_poisson_path_identity_match"

base.VERSION_NUMBER = VERSION_NUMBER
base.VERSION = VERSION
base.OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
base.REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
base.OUT_DIR.mkdir(parents=True, exist_ok=True)
base.REPORT_DIR.mkdir(parents=True, exist_ok=True)
base.MATCH_CACHE = base.OUT_DIR / "v324_poisson_component_matches.npz"
base.REPORT = base.REPORT_DIR / f"{VERSION}.json"
base.PATH_CACHE = PATHS.cache_dir(323) / "v323_highdepth_poisson_paths.npz"


if __name__ == "__main__":
    base.main()
