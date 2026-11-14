#!/usr/bin/env python3
"""Validate the Round 2 high-budget multi-seed Sionna dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import h5py
import numpy as np

from project_paths import PATHS
from round2.generate_multiseed_sionna_dataset import (
    ARTIFACT_DIR,
    CHECKPOINTS,
    CONFIGS,
    DOMAINS,
    REPORT_DIR,
    SEEDS,
    VERSION,
    checkpoint_path,
    count_shape,
    domain_shape,
    expand_count,
)


REPORT = REPORT_DIR / "sionna_multiseed_validation.json"
SPECTRAL_AXES = {
    "pas": (1, 2, 3),
    "pdp": (-1,),
    "ad": (1, 2, 3),
}
FLOAT_PATH_DATASETS = (
    "tau",
    "theta_t",
    "phi_t",
    "theta_r",
    "phi_r",
    "vertices",
    "path_energy",
    "path_energy_fraction",
    "port_energy",
    "reference_coefficient",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        choices=("all", "primary", *(config.name for config in CONFIGS)),
        default="primary",
    )
    parser.add_argument("--selection-tag", default="full")
    parser.add_argument("--chunk-rows", type=int, default=4)
    parser.add_argument("--skip-path-values", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def choose_configs(name: str):
    if name == "all":
        return list(CONFIGS)
    if name == "primary":
        return [config for config in CONFIGS if config.name.startswith("primary_")]
    return [config for config in CONFIGS if config.name == name]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_checkpoint_domain(
    root: Path,
    checkpoint: int,
    split: str,
    name: str,
    rows: int,
    chunk_rows: int,
) -> dict[str, Any]:
    expected_domain_shape = domain_shape(name, rows)
    expected_count_shape = count_shape(name, rows)
    mean = np.load(
        checkpoint_path(root, checkpoint, split, name, "mean"), mmap_mode="r"
    )
    variance = np.load(
        checkpoint_path(root, checkpoint, split, name, "variance"), mmap_mode="r"
    )
    count = np.load(
        checkpoint_path(root, checkpoint, split, name, "valid_count"), mmap_mode="r"
    )
    require(mean.shape == expected_domain_shape, f"bad mean shape for {name}")
    require(variance.shape == expected_domain_shape, f"bad variance shape for {name}")
    require(count.shape == expected_count_shape, f"bad count shape for {name}")
    require(mean.dtype == np.float32, f"bad mean dtype for {name}")
    require(variance.dtype == np.float32, f"bad variance dtype for {name}")
    require(count.dtype == np.uint16, f"bad count dtype for {name}")

    finite = True
    variance_min = float("inf")
    variance_max = 0.0
    normalization_error = 0.0
    zero_count_value_max = 0.0
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        local_mean = np.asarray(mean[start:end])
        local_variance = np.asarray(variance[start:end])
        local_count = np.asarray(count[start:end])
        finite = finite and bool(
            np.isfinite(local_mean).all() and np.isfinite(local_variance).all()
        )
        variance_min = min(variance_min, float(local_variance.min(initial=0.0)))
        variance_max = max(variance_max, float(local_variance.max(initial=0.0)))
        expanded = expand_count(name, local_count)
        zero_count_value_max = max(
            zero_count_value_max,
            float(
                np.maximum(
                    np.where(expanded == 0, np.abs(local_mean), 0.0),
                    np.where(expanded == 0, np.abs(local_variance), 0.0),
                ).max(initial=0.0)
            ),
        )
        if name in SPECTRAL_AXES:
            sums = local_mean.sum(axis=SPECTRAL_AXES[name], dtype=np.float64)
            valid = local_count > 0
            if np.any(valid):
                normalization_error = max(
                    normalization_error,
                    float(np.max(np.abs(sums[valid] - 1.0))),
                )

    require(finite, f"non-finite checkpoint values for {split}/{name}/n{checkpoint}")
    require(variance_min >= -1e-12, f"negative variance for {split}/{name}")
    require(int(count.max(initial=0)) <= checkpoint, f"count exceeds checkpoint")
    require(zero_count_value_max <= 1e-12, f"nonzero value with zero count for {name}")
    if name in SPECTRAL_AXES:
        require(normalization_error <= 2e-6, f"normalization failed for {name}")
    if name == "path_count":
        require(int(count.min(initial=checkpoint)) == checkpoint, "path count missing")
    return {
        "shape": list(mean.shape),
        "dtype": str(mean.dtype),
        "count_min": int(count.min(initial=checkpoint)),
        "count_max": int(count.max(initial=0)),
        "variance_min": variance_min,
        "variance_max": variance_max,
        "normalization_max_abs": normalization_error,
        "zero_count_value_max_abs": zero_count_value_max,
    }


def convergence(
    root: Path,
    split: str,
    name: str,
    rows: int,
    first: int,
    second: int,
    chunk_rows: int,
) -> dict[str, float | int]:
    left = np.load(
        checkpoint_path(root, first, split, name, "mean"), mmap_mode="r"
    )
    right = np.load(
        checkpoint_path(root, second, split, name, "mean"), mmap_mode="r"
    )
    cosine_parts: list[np.ndarray] = []
    relative_parts: list[np.ndarray] = []
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        x = np.asarray(left[start:end], dtype=np.float64).reshape(end - start, -1)
        y = np.asarray(right[start:end], dtype=np.float64).reshape(end - start, -1)
        dot = np.einsum("ij,ij->i", x, y)
        x_norm = np.linalg.norm(x, axis=1)
        y_norm = np.linalg.norm(y, axis=1)
        denominator = x_norm * y_norm
        valid = denominator > 0
        if np.any(valid):
            cosine_parts.append(dot[valid] / denominator[valid])
        valid_right = y_norm > 0
        if np.any(valid_right):
            relative_parts.append(
                np.linalg.norm(y[valid_right] - x[valid_right], axis=1)
                / y_norm[valid_right]
            )
    cosine = np.concatenate(cosine_parts)
    relative = np.concatenate(relative_parts)
    return {
        "point_count": int(len(relative)),
        "cosine_mean": float(np.mean(cosine)),
        "cosine_min": float(np.min(cosine)),
        "relative_l2_mean": float(np.mean(relative)),
        "relative_l2_p90": float(np.quantile(relative, 0.90)),
    }


def validate_path_group(
    group: h5py.Group,
    coefficient_mode: str,
    read_values: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    row_indices = np.asarray(group["row_indices"], dtype=np.int64)
    point_ids = np.asarray(group["point_ids"], dtype=np.int64)
    offsets = np.asarray(group["offsets"], dtype=np.int64)
    require(len(row_indices) == len(point_ids), "row/point length mismatch")
    require(len(offsets) == len(row_indices) + 1, "offset length mismatch")
    require(offsets[0] == 0 and np.all(np.diff(offsets) >= 0), "bad offsets")
    path_count = int(offsets[-1])
    required = (
        "tau",
        "theta_t",
        "phi_t",
        "theta_r",
        "phi_r",
        "interactions",
        "objects",
        "primitives",
        "vertices",
        "path_energy",
        "path_energy_fraction",
        "port_energy",
        "reference_coefficient",
        "interaction_count",
    )
    for name in required:
        require(name in group, f"missing path dataset {name}")
        require(group[name].shape[0] == path_count, f"bad path length for {name}")
    if coefficient_mode == "full":
        require("full_coefficient" in group, "missing full path coefficient")
        require(
            group["full_coefficient"].shape == (path_count, 256, 4),
            "bad full coefficient shape",
        )
    else:
        require("full_coefficient" not in group, "unexpected full coefficient")
    require(group["port_energy"].shape == (path_count, 2, 4), "bad port energy")
    require(
        group["reference_coefficient"].shape == (path_count, 2, 4),
        "bad reference coefficient",
    )
    require(group["vertices"].shape[-1] == 3, "bad vertex shape")

    fraction_error = 0.0
    if read_values:
        names = list(FLOAT_PATH_DATASETS)
        if coefficient_mode == "full":
            names.append("full_coefficient")
        for name in names:
            require(np.isfinite(np.asarray(group[name])).all(), f"non-finite {name}")
        path_energy = np.asarray(group["path_energy"], dtype=np.float64)
        fractions = np.asarray(group["path_energy_fraction"], dtype=np.float64)
        require(np.all(path_energy >= 0), "negative path energy")
        require(np.all(fractions >= 0), "negative path energy fraction")
        for row in range(len(row_indices)):
            start, end = int(offsets[row]), int(offsets[row + 1])
            if start == end or np.sum(path_energy[start:end]) <= 1e-30:
                continue
            fraction_error = max(
                fraction_error,
                abs(float(np.sum(fractions[start:end])) - 1.0),
            )
    return row_indices, point_ids, np.diff(offsets), fraction_error


def validate_paths(
    root: Path,
    config,
    split: str,
    ids: np.ndarray,
    read_values: bool,
) -> dict[str, Any]:
    rows = len(ids)
    per_seed_counts: list[np.ndarray] = []
    maximum_fraction_error = 0.0
    group_count = 0
    total_paths = 0
    for seed in SEEDS:
        path = root / "paths" / f"{split}_seed{seed}.h5"
        require(path.is_file(), f"missing path file {path}")
        row_parts: list[np.ndarray] = []
        point_parts: list[np.ndarray] = []
        count_parts: list[np.ndarray] = []
        with h5py.File(path, "r") as handle:
            require(bool(handle.attrs.get("complete", False)), f"incomplete {path}")
            require(handle.attrs.get("split") == split, f"bad split in {path}")
            require(int(handle.attrs.get("seed")) == seed, f"bad seed in {path}")
            require(
                handle.attrs.get("coefficient_mode") == config.path_coefficient_mode,
                f"bad coefficient mode in {path}",
            )
            expected_config = json.loads(json.dumps(asdict(config)))
            actual_config = json.loads(handle.attrs["configuration"])
            require(actual_config == expected_config, f"bad configuration in {path}")
            for site_group in handle.values():
                require(isinstance(site_group, h5py.Group), "unexpected HDF5 object")
                for group in site_group.values():
                    local_rows, local_points, local_counts, fraction_error = (
                        validate_path_group(
                            group,
                            config.path_coefficient_mode,
                            read_values,
                        )
                    )
                    row_parts.append(local_rows)
                    point_parts.append(local_points)
                    count_parts.append(local_counts)
                    maximum_fraction_error = max(
                        maximum_fraction_error, fraction_error
                    )
                    group_count += 1
                    total_paths += int(np.sum(local_counts))
        all_rows = np.concatenate(row_parts)
        all_points = np.concatenate(point_parts)
        all_counts = np.concatenate(count_parts)
        order = np.argsort(all_rows)
        require(np.array_equal(all_rows[order], np.arange(rows)), f"bad rows in {path}")
        require(np.array_equal(all_points[order], ids), f"bad point ids in {path}")
        per_seed_counts.append(all_counts[order].astype(np.float64))

    path_counts = np.stack(per_seed_counts)
    expected_mean = np.load(
        checkpoint_path(root, 24, split, "path_count", "mean"), mmap_mode="r"
    )
    expected_variance = np.load(
        checkpoint_path(root, 24, split, "path_count", "variance"), mmap_mode="r"
    )
    mean_error = float(np.max(np.abs(path_counts.mean(axis=0) - expected_mean)))
    variance_error = float(
        np.max(np.abs(path_counts.var(axis=0, ddof=1) - expected_variance))
    )
    mean_tolerance = 1e-6 * max(1.0, float(np.max(np.abs(expected_mean))))
    require(mean_error <= mean_tolerance, f"path count mean mismatch for {split}")
    require(variance_error <= 0.1, f"path count variance mismatch for {split}")
    require(maximum_fraction_error <= 2e-5, "path energy fractions do not sum to one")
    return {
        "file_count": len(SEEDS),
        "group_count": group_count,
        "total_paths": total_paths,
        "mean_paths_per_point_seed": float(path_counts.mean()),
        "zero_path_fraction": float(np.mean(path_counts == 0)),
        "path_count_mean_max_abs_error": mean_error,
        "path_count_mean_tolerance": mean_tolerance,
        "path_count_variance_max_abs_error": variance_error,
        "path_energy_fraction_max_abs_error": maximum_fraction_error,
        "path_values_scanned": read_values,
    }


def validate_config(config, tag: str, chunk_rows: int, read_values: bool):
    root = ARTIFACT_DIR / config.name / tag
    require(root.is_dir(), f"missing configuration root {root}")
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    expected_config = json.loads(json.dumps(asdict(config)))
    require(state["version"] == VERSION, f"bad state version for {config.name}")
    require(state["configuration"] == expected_config, f"bad state config")
    require(state["selection_tag"] == tag, f"bad selection tag")
    require(state["seed_order"] == list(SEEDS), f"bad seed order")

    split_ids: dict[str, np.ndarray] = {}
    for split, rows in (("train", 4000), ("test", 500)):
        ids = np.load(root / f"{split}_ids.npy")
        require(ids.shape == (rows,), f"bad {split} ids shape")
        require(ids.dtype == np.int64, f"bad {split} ids dtype")
        require(np.array_equal(ids, np.arange(rows)), f"non-full {split} selection")
        split_ids[split] = ids
        split_state = state["splits"][split]
        require(split_state["count"] == len(SEEDS), f"incomplete {split} state")
        require(split_state["completed_seeds"] == list(SEEDS), f"bad seed state")
        require(split_state["active_seed"] is None, f"active seed remains")

    checkpoints: dict[str, Any] = {}
    for checkpoint in CHECKPOINTS:
        print(f"validating {config.name} checkpoint n{checkpoint:02d}", flush=True)
        checkpoint_summary: dict[str, Any] = {}
        for split, rows in (("train", 4000), ("test", 500)):
            checkpoint_summary[split] = {
                name: validate_checkpoint_domain(
                    root, checkpoint, split, name, rows, chunk_rows
                )
                for name in DOMAINS
            }
        checkpoints[f"n{checkpoint:02d}"] = checkpoint_summary

    convergence_summary: dict[str, Any] = {}
    for first, second in ((6, 12), (12, 24)):
        key = f"n{first:02d}_to_n{second:02d}"
        convergence_summary[key] = {}
        for split, rows in (("train", 4000), ("test", 500)):
            convergence_summary[key][split] = {
                name: convergence(
                    root, split, name, rows, first, second, chunk_rows
                )
                for name in SPECTRAL_AXES
            }

    paths = {}
    for split in ("train", "test"):
        print(f"validating {config.name} {split} path tables", flush=True)
        paths[split] = validate_paths(
            root, config, split, split_ids[split], read_values
        )
    storage_bytes = sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
    return {
        "root": PATHS.relative(root),
        "storage_bytes": storage_bytes,
        "state_complete": True,
        "checkpoints": checkpoints,
        "convergence": convergence_summary,
        "paths": paths,
    }


def main() -> None:
    args = parse_args()
    require(args.chunk_rows > 0, "--chunk-rows must be positive")
    started = time.time()
    selected = choose_configs(args.config)
    summaries = {
        config.name: validate_config(
            config,
            args.selection_tag,
            args.chunk_rows,
            not args.skip_path_values,
        )
        for config in selected
    }
    report = {
        "experiment": VERSION,
        "status": "validated",
        "all_checks_passed": True,
        "arguments": vars(args),
        "validation_seconds": float(time.time() - started),
        "configurations": summaries,
    }
    atomic_json(REPORT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
