#!/usr/bin/env python3
"""Build v443 point-cloud component supervision from v442 paths."""

from __future__ import annotations

import build_v439_component_path_supervision as builder

from project_paths import PATHS


builder.VERSION_NUMBER = 443
builder.VERSION = "v443_component_path_supervision"
builder.OUT_DIR = PATHS.cache_dir(443)
builder.REPORT_DIR = PATHS.version_report_dir(443)
builder.REPORT = builder.REPORT_DIR / f"{builder.VERSION}.json"
builder.SOURCE = PATHS.cache_dir(442)
builder.BASE_CACHE = PATHS.cache_dir(443)
builder.SHARD_ROOT = builder.SOURCE / "shards"
builder.COUNT = 200_000


if __name__ == "__main__":
    builder.main()
