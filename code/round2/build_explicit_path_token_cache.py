#!/usr/bin/env python3
"""Build reusable explicit-path token caches for all Round 2 train or test points."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from analyze_joint_path_token_oracle import ARTIFACT_DIR, cache_paths
from project_paths import PATHS
from round2.data import load_positions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--seed-count", type=int, default=24)
    parser.add_argument("--max-paths", type=int, default=128)
    parser.add_argument("--base-reserve", type=int, default=48)
    parser.add_argument("--base-per-seed", type=int, default=96)
    parser.add_argument("--scatter-per-seed", type=int, default=64)
    parser.add_argument("--scatter-pool", type=int, default=256)
    parser.add_argument("--selection-seed", type=int, default=106)
    parser.add_argument("--include-scatter", action="store_true")
    parser.add_argument("--reuse-path-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    _, train_positions, test_positions, train_site, test_site = load_positions()
    if args.split == "train":
        count = len(train_positions)
        site = train_site
    else:
        count = len(test_positions)
        site = test_site
    point_ids = np.arange(count, dtype=np.int64)
    rows = np.arange(count, dtype=np.int64)
    cache = cache_paths(args, rows, point_ids, site, split=args.split)
    tag_prefix = "" if args.split == "train" else "test_"
    tag = (
        f"{tag_prefix}n{count:03d}_s{args.seed_count:02d}_k{args.max_paths:03d}_"
        f"b{args.base_per_seed:03d}_d{args.scatter_per_seed:03d}_"
        f"scatter{int(args.include_scatter)}_sel{args.selection_seed}"
    )
    report = {
        "split": args.split,
        "configuration": vars(args),
        "point_count": count,
        "path_count": {
            "mean": float(np.mean(np.sum(np.asarray(cache["mask"]), axis=1))),
            "median": float(np.median(np.sum(np.asarray(cache["mask"]), axis=1))),
            "min": int(np.min(np.sum(np.asarray(cache["mask"]), axis=1))),
            "max": int(np.max(np.sum(np.asarray(cache["mask"]), axis=1))),
        },
        "cache": PATHS.relative(ARTIFACT_DIR / tag),
        "seconds": float(time.time() - started),
    }
    report_path = ARTIFACT_DIR / f"{tag}_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
