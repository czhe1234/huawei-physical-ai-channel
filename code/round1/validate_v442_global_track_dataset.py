#!/usr/bin/env python3
"""Validate every compact spectral/path shard of the v442 dataset."""

from __future__ import annotations

import json
import time

import numpy as np

import validate_v437_sionna_pretraining_dataset as v437_validator
from project_paths import PATHS


VERSION_NUMBER = 442
VERSION = "v442_global_track_dataset_validation"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
SHARD_ROOT = OUT_DIR / "shards"
POSITIONS_PATH = OUT_DIR / "positions.npz"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"

COUNT = 200_000
SHARD_COUNT = 200
SHARD_SIZE = 1_000


def main() -> None:
    started = time.time()
    manifest = np.load(POSITIONS_PATH)
    positions = manifest["positions"].astype(np.float32)
    split = manifest["pretrain_split"].astype(np.int8)
    if positions.shape != (COUNT, 3):
        raise ValueError(f"Unexpected positions shape {positions.shape}")
    if np.bincount(split, minlength=3).tolist() != [160_000, 20_000, 20_000]:
        raise ValueError("Unexpected v442 split counts")
    if float(np.min(manifest["min_official_test_distance"])) < 1.0:
        raise ValueError("A v442 point is less than 1m from an official test point")
    min_v437 = manifest["min_v437_distance"].astype(np.float32)
    if float(np.min(min_v437[split > 0])) < 3.0:
        raise ValueError("A v442 validation/holdout point is less than 3m from v437")

    total_paths = 0
    zero_paths = 0
    zero_channels = 0
    maximum_mass_error = 0.0
    maximum_path_mass_error = 0.0
    total_bytes = 0
    batch_sizes: dict[int, int] = {}
    summaries = []
    for shard_id in range(SHARD_COUNT):
        shard_dir = SHARD_ROOT / f"shard_{shard_id:05d}"
        files = {
            "metadata": shard_dir / "metadata.npz",
            "targets": shard_dir / "spectral_targets.npz",
            "paths": shard_dir / "path_targets.npz",
            "done": shard_dir / "done.json",
        }
        for path in files.values():
            if not path.is_file():
                raise FileNotFoundError(path)
            total_bytes += path.stat().st_size
        if (shard_dir / "H.npy").exists() or (shard_dir / "H.partial.npy").exists():
            raise ValueError(f"Shard {shard_id} unexpectedly contains full H")
        done = json.loads(files["done"].read_text(encoding="utf-8"))
        if int(done["count"]) != SHARD_SIZE:
            raise ValueError(f"Shard {shard_id} count is not {SHARD_SIZE}")
        if done["outputs"]["H"] is not None:
            raise ValueError(f"Shard {shard_id} reports a full H output")
        batch_size = int(done["batch_size"])
        batch_sizes[batch_size] = batch_sizes.get(batch_size, 0) + 1

        start = shard_id * SHARD_SIZE
        end = start + SHARD_SIZE
        expected_ids = np.arange(start, end, dtype=np.int64)
        metadata = np.load(files["metadata"])
        if not np.array_equal(metadata["position_id"], expected_ids):
            raise ValueError(f"Shard {shard_id} IDs are not contiguous")
        if not np.array_equal(metadata["positions"], positions[start:end]):
            raise ValueError(f"Shard {shard_id} positions differ from manifest")
        if not np.array_equal(metadata["pretrain_split"], split[start:end]):
            raise ValueError(f"Shard {shard_id} split differs from manifest")

        targets = np.load(files["targets"])
        if set(targets.files) != set(v437_validator.TARGET_TAIL_SHAPES):
            raise ValueError(f"Shard {shard_id} target keys differ")
        for name, tail in v437_validator.TARGET_TAIL_SHAPES.items():
            value = targets[name]
            if value.shape != (SHARD_SIZE, *tail):
                raise ValueError(f"Shard {shard_id} {name} shape {value.shape}")
            if not np.isfinite(value).all():
                raise FloatingPointError(f"Shard {shard_id} {name} is non-finite")
        nonzero = targets["log_point_power"] > np.log(1e-30) + 1e-4
        zero_channels += int(np.sum(~nonzero))
        for name in v437_validator.TARGET_TAIL_SHAPES:
            if name != "log_point_power":
                maximum_mass_error = max(
                    maximum_mass_error,
                    v437_validator.check_distribution(name, targets[name], nonzero),
                )

        path_targets = np.load(files["paths"])
        if set(path_targets.files) != set(v437_validator.PATH_TAIL_SHAPES):
            raise ValueError(f"Shard {shard_id} path keys differ")
        for name, tail in v437_validator.PATH_TAIL_SHAPES.items():
            if path_targets[name].shape != (SHARD_SIZE, *tail):
                raise ValueError(
                    f"Shard {shard_id} {name} shape {path_targets[name].shape}"
                )
        valid = path_targets["valid"].astype(bool)
        path_count = np.sum(valid, axis=1)
        total_paths += int(np.sum(path_count))
        zero_paths += int(np.sum(path_count == 0))
        for name in (
            "tau",
            "theta_t",
            "phi_t",
            "theta_r",
            "phi_r",
            "log_power",
            "complex_reference",
            "rx_fingerprint",
            "tx_pol_fingerprint",
            "tx_h_fingerprint",
            "tx_v_fingerprint",
        ):
            if not np.isfinite(path_targets[name][valid]).all():
                raise FloatingPointError(f"Shard {shard_id} valid {name} is non-finite")
        active_vertex = path_targets["interactions"] != 0
        if not np.isfinite(path_targets["vertices"][active_vertex]).all():
            raise FloatingPointError(f"Shard {shard_id} active vertices are non-finite")
        path_mass = np.sum(
            path_targets["normalized_power"].astype(np.float32),
            axis=1,
            dtype=np.float64,
        )
        current_error = float(
            np.max(np.abs(path_mass - (path_count > 0).astype(np.float64)))
        )
        maximum_path_mass_error = max(maximum_path_mass_error, current_error)
        if current_error > 0.01:
            raise ValueError(f"Shard {shard_id} path mass error {current_error}")
        summaries.append(
            {
                "shard_id": shard_id,
                "mean_paths": float(np.mean(path_count)),
                "zero_paths": int(np.sum(path_count == 0)),
                "zero_channels": int(np.sum(~nonzero)),
            }
        )
        print(f"validated v442 shard {shard_id + 1}/{SHARD_COUNT}", flush=True)

    if zero_paths != zero_channels:
        raise ValueError(
            f"zero path/channel mismatch: {zero_paths} vs {zero_channels}"
        )
    report = {
        "method": VERSION,
        "status": "pass",
        "summary": {
            "point_count": COUNT,
            "shard_count": SHARD_COUNT,
            "full_h_saved": False,
            "total_bytes": int(total_bytes),
            "total_paths": int(total_paths),
            "mean_paths_per_point": float(total_paths / COUNT),
            "zero_path_points": int(zero_paths),
            "zero_channel_points": int(zero_channels),
            "split_counts": np.bincount(split, minlength=3).astype(int).tolist(),
            "validation_min_v437_distance": float(np.min(min_v437[split == 1])),
            "holdout_min_v437_distance": float(np.min(min_v437[split == 2])),
            "batch_size_shard_counts": {
                str(key): value for key, value in sorted(batch_sizes.items())
            },
            "maximum_target_mass_error": maximum_mass_error,
            "maximum_path_mass_error": maximum_path_mass_error,
            "seconds": float(time.time() - started),
        },
        "shards": summaries,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
