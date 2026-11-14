#!/usr/bin/env python3
"""Generate high-budget Sionna path tokens at the official train/test positions."""

from __future__ import annotations

import argparse
import json

import numpy as np

import generate_v437_single_gpu_sionna_pretraining_dataset as generator
from project_paths import PATHS


VERSION_NUMBER = 475
VERSION = "v475_official_coordinate_path_tokens"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--max-paths", type=int, default=128)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--samples-per-src", type=int, default=1_000_000)
    parser.add_argument("--max-num-paths-per-src", type=int, default=1_000_000)
    parser.add_argument("--frequency", type=float, default=6.693e9)
    parser.add_argument("--seed", type=int, default=475_001)
    parser.add_argument("--overwrite-shards", action="store_true")
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--limit-shards", type=int)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_mono_polarized")
    return parser.parse_args()


def write_position_manifest(path) -> None:
    train = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    positions = np.concatenate([train, test], axis=0)
    official_split = np.concatenate(
        [
            np.zeros(len(train), dtype=np.int8),
            np.ones(len(test), dtype=np.int8),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        positions=positions,
        sample_mode=official_split,
        pretrain_split=official_split,
        min_official_test_distance=np.zeros(len(positions), dtype=np.float32),
        min_official_train_distance=np.zeros(len(positions), dtype=np.float32),
        official_split=official_split,
    )


def main() -> None:
    args = parse_args()
    generator.VERSION_NUMBER = VERSION_NUMBER
    generator.VERSION = VERSION
    generator.OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
    generator.SHARD_ROOT = generator.OUT_DIR / "shards"
    generator.POSITIONS_PATH = generator.OUT_DIR / "positions.npz"
    generator.REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
    generator.REPORT = generator.REPORT_DIR / f"{VERSION}.json"
    write_position_manifest(generator.POSITIONS_PATH)

    generator_args = argparse.Namespace(
        count=2500,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        max_paths=args.max_paths,
        max_depth=args.max_depth,
        samples_per_src=args.samples_per_src,
        max_num_paths_per_src=args.max_num_paths_per_src,
        frequency=args.frequency,
        seed=args.seed,
        min_test_distance=0.0,
        min_train_distance=0.0,
        save_full_h=False,
        overwrite=False,
        overwrite_shards=args.overwrite_shards,
        start_shard=args.start_shard,
        limit_shards=args.limit_shards,
        mitsuba_variant=args.mitsuba_variant,
    )
    generator.parse_args = lambda: generator_args
    generator.main()
    report = json.loads(generator.REPORT.read_text(encoding="utf-8"))
    report["method"] = VERSION
    report["scope"] = (
        "diagnostic path-token cache at all 2000 official-train and 500 "
        "official-test coordinates"
    )
    report["position_audit"] = {
        "source": "official Round1 train/test coordinate arrays",
        "train_count": 2000,
        "test_count": 500,
        "ordering": "train rows followed by test rows",
    }
    generator.REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
