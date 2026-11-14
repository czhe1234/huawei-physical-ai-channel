#!/usr/bin/env python3
"""Validate every shard of the v437 Sionna pretraining dataset."""

from __future__ import annotations

import json
import time

import numpy as np

import generate_v437_single_gpu_sionna_pretraining_dataset as v437
from project_paths import PATHS


OUT_DIR = PATHS.cache_dir(437)
SHARD_ROOT = OUT_DIR / "shards"
POSITIONS_PATH = OUT_DIR / "positions.npz"
REPORT_DIR = PATHS.version_report_dir(437)
REPORT = REPORT_DIR / "v437_dataset_validation.json"


TARGET_TAIL_SHAPES = {
    "log_point_power": (),
    "pdp": (192,),
    "rx_pdp": (4, 192),
    "tx_pol_pdp": (2, 192),
    "h_delay": (16, 192),
    "v_delay": (8, 192),
    "pol_ue_delay": (2, 4, 192),
    "joint_sh": (192, 16),
    "pol_sh": (2, 192, 16),
    "conventional_hv": (16, 8),
}
PATH_TAIL_SHAPES = {
    "valid": (128,),
    "tau": (128,),
    "theta_t": (128,),
    "phi_t": (128,),
    "theta_r": (128,),
    "phi_r": (128,),
    "log_power": (128,),
    "normalized_power": (128,),
    "interaction_count": (128,),
    "interaction_mask": (128,),
    "interactions": (128, 16),
    "objects": (128, 16),
    "vertices": (128, 16, 3),
    "complex_reference": (128,),
    "rx_fingerprint": (128, 4),
    "tx_pol_fingerprint": (128, 2),
    "tx_h_fingerprint": (128, 16),
    "tx_v_fingerprint": (128, 8),
}


def check_distribution(name: str, value: np.ndarray, nonzero: np.ndarray) -> float:
    axes = tuple(range(1, value.ndim))
    mass = np.sum(value.astype(np.float32), axis=axes, dtype=np.float64)
    expected = nonzero.astype(np.float64)
    error = float(np.max(np.abs(mass - expected)))
    if error > 0.01:
        raise ValueError(f"{name} normalization error {error}")
    return error


def main() -> None:
    started = time.time()
    positions_file = np.load(POSITIONS_PATH)
    positions = positions_file["positions"].astype(np.float32)
    if positions.shape != (100_000, 3):
        raise ValueError(f"Unexpected positions shape {positions.shape}")
    if float(np.min(positions_file["min_official_test_distance"])) < 1.0:
        raise ValueError("A generated position is less than 1m from an official test point")

    batch_sizes: dict[int, int] = {}
    total_paths = 0
    zero_path_points = 0
    zero_channel_points = 0
    maximum_power_error = 0.0
    maximum_target_mass_error = 0.0
    maximum_recomputed_pdp_error = 0.0
    maximum_recomputed_joint_sh_error = 0.0
    total_bytes = 0
    shard_summaries = []

    for shard_id in range(100):
        shard_dir = SHARD_ROOT / f"shard_{shard_id:05d}"
        paths = {
            "H": shard_dir / "H.npy",
            "metadata": shard_dir / "metadata.npz",
            "targets": shard_dir / "spectral_targets.npz",
            "paths": shard_dir / "path_targets.npz",
            "done": shard_dir / "done.json",
        }
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
            total_bytes += path.stat().st_size
        done = json.loads(paths["done"].read_text(encoding="utf-8"))
        count = int(done["count"])
        if count != 1_000:
            raise ValueError(f"Shard {shard_id} count is {count}")
        batch_size = int(done["batch_size"])
        batch_sizes[batch_size] = batch_sizes.get(batch_size, 0) + 1

        metadata = np.load(paths["metadata"])
        expected_ids = np.arange(shard_id * 1_000, (shard_id + 1) * 1_000)
        if not np.array_equal(metadata["position_id"], expected_ids):
            raise ValueError(f"Shard {shard_id} position IDs are not contiguous")
        if not np.array_equal(metadata["positions"], positions[expected_ids]):
            raise ValueError(f"Shard {shard_id} positions differ from the global manifest")

        targets = np.load(paths["targets"])
        if set(targets.files) != set(TARGET_TAIL_SHAPES):
            raise ValueError(f"Shard {shard_id} target keys differ")
        for name, tail in TARGET_TAIL_SHAPES.items():
            value = targets[name]
            if value.shape != (count, *tail):
                raise ValueError(f"Shard {shard_id} {name} shape {value.shape}")
            if not np.isfinite(value).all():
                raise FloatingPointError(f"Shard {shard_id} {name} is non-finite")
        nonzero = targets["log_point_power"] > np.log(1e-30) + 1e-4
        zero_channel_points += int(np.sum(~nonzero))
        for name in TARGET_TAIL_SHAPES:
            if name != "log_point_power":
                maximum_target_mass_error = max(
                    maximum_target_mass_error,
                    check_distribution(name, targets[name], nonzero),
                )

        path_targets = np.load(paths["paths"])
        if set(path_targets.files) != set(PATH_TAIL_SHAPES):
            raise ValueError(f"Shard {shard_id} path keys differ")
        for name, tail in PATH_TAIL_SHAPES.items():
            if path_targets[name].shape != (count, *tail):
                raise ValueError(
                    f"Shard {shard_id} {name} shape {path_targets[name].shape}"
                )
        valid = path_targets["valid"].astype(bool)
        path_count = np.sum(valid, axis=1)
        total_paths += int(np.sum(path_count))
        zero_path_points += int(np.sum(path_count == 0))
        for name in (
            "tau",
            "theta_t",
            "phi_t",
            "theta_r",
            "phi_r",
            "log_power",
            "complex_reference",
        ):
            if not np.isfinite(path_targets[name][valid]).all():
                raise FloatingPointError(f"Shard {shard_id} valid {name} is non-finite")
        for name in (
            "rx_fingerprint",
            "tx_pol_fingerprint",
            "tx_h_fingerprint",
            "tx_v_fingerprint",
        ):
            if not np.isfinite(path_targets[name][valid]).all():
                raise FloatingPointError(f"Shard {shard_id} valid {name} is non-finite")
        path_mass = np.sum(
            path_targets["normalized_power"].astype(np.float32),
            axis=1,
            dtype=np.float64,
        )
        path_mass_error = float(
            np.max(np.abs(path_mass - (path_count > 0).astype(np.float64)))
        )
        if path_mass_error > 0.01:
            raise ValueError(f"Shard {shard_id} path mass error {path_mass_error}")
        active_vertex = path_targets["interactions"] != 0
        if not np.isfinite(path_targets["vertices"][active_vertex]).all():
            raise FloatingPointError(f"Shard {shard_id} active vertices are non-finite")

        channel = np.load(paths["H"], mmap_mode="r")
        if channel.shape != (count, 256, 4, 192) or channel.dtype != np.complex64:
            raise ValueError(
                f"Shard {shard_id} H has shape/dtype {channel.shape}/{channel.dtype}"
            )
        for start in range(0, count, 32):
            block = np.asarray(channel[start : start + 32])
            if not np.isfinite(block).all():
                raise FloatingPointError(
                    f"Shard {shard_id} H is non-finite at {start}:{start + len(block)}"
                )
            power = np.sum(np.abs(block) ** 2, axis=(1, 2, 3), dtype=np.float64)
            log_power = np.log(np.maximum(power, 1e-30))
            maximum_power_error = max(
                maximum_power_error,
                float(
                    np.max(
                        np.abs(
                            log_power
                            - targets["log_point_power"][start : start + len(block)]
                        )
                    )
                ),
            )

        sample_id = int(np.flatnonzero(nonzero)[0]) if np.any(nonzero) else 0
        recomputed = v437.spectral_targets(np.asarray(channel[sample_id : sample_id + 1]))
        maximum_recomputed_pdp_error = max(
            maximum_recomputed_pdp_error,
            float(
                np.max(
                    np.abs(
                        recomputed["pdp"].astype(np.float32)
                        - targets["pdp"][sample_id : sample_id + 1].astype(np.float32)
                    )
                )
            ),
        )
        maximum_recomputed_joint_sh_error = max(
            maximum_recomputed_joint_sh_error,
            float(
                np.max(
                    np.abs(
                        recomputed["joint_sh"].astype(np.float32)
                        - targets["joint_sh"][sample_id : sample_id + 1].astype(np.float32)
                    )
                )
            ),
        )
        shard_summaries.append(
            {
                "shard_id": shard_id,
                "batch_size": batch_size,
                "mean_paths": float(np.mean(path_count)),
                "zero_paths": int(np.sum(path_count == 0)),
                "zero_channels": int(np.sum(~nonzero)),
            }
        )
        print(f"validated shard {shard_id + 1}/100", flush=True)

    if maximum_power_error > 1e-4:
        raise ValueError(f"Stored H/target log-power mismatch {maximum_power_error}")
    float16_recompute_tolerance = 5e-4
    if maximum_recomputed_pdp_error > float16_recompute_tolerance:
        raise ValueError(f"Stored/recomputed PDP mismatch {maximum_recomputed_pdp_error}")
    if maximum_recomputed_joint_sh_error > float16_recompute_tolerance:
        raise ValueError(
            f"Stored/recomputed joint S-H mismatch {maximum_recomputed_joint_sh_error}"
        )

    report = {
        "method": "v437_sionna_pretraining_dataset_validation",
        "status": "pass",
        "summary": {
            "point_count": 100_000,
            "shard_count": 100,
            "total_bytes": int(total_bytes),
            "total_paths": int(total_paths),
            "mean_paths_per_point": float(total_paths / 100_000),
            "zero_path_points": int(zero_path_points),
            "zero_channel_points": int(zero_channel_points),
            "batch_size_shard_counts": {
                str(key): value for key, value in sorted(batch_sizes.items())
            },
            "maximum_log_power_error": maximum_power_error,
            "maximum_target_mass_error": maximum_target_mass_error,
            "maximum_recomputed_pdp_error": maximum_recomputed_pdp_error,
            "maximum_recomputed_joint_sh_error": maximum_recomputed_joint_sh_error,
            "seconds": float(time.time() - started),
        },
        "shards": shard_summaries,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
