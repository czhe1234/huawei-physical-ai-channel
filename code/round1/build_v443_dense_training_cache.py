#!/usr/bin/env python3
"""Build v443 route and compact targets from the validated v442 dataset."""

from __future__ import annotations

import build_v438_large_model_training_cache as builder

from project_paths import PATHS


builder.VERSION_NUMBER = 443
builder.VERSION = "v443_dense_training_cache"
builder.OUT_DIR = PATHS.cache_dir(443)
builder.REPORT_DIR = PATHS.version_report_dir(443)
builder.REPORT = builder.REPORT_DIR / f"{builder.VERSION}.json"
builder.SOURCE = PATHS.cache_dir(442)
builder.POSITIONS_SOURCE = builder.SOURCE / "positions.npz"
builder.SHARD_ROOT = builder.SOURCE / "shards"
builder.COUNT = 200_000


if __name__ == "__main__":
    builder.main()
