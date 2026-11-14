#!/usr/bin/env python3
"""Build v560 power-free tokens and audit geometry-only candidate coverage."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import build_v445_geometry_ordered_path_tokens as legacy_builder
from project_paths import PATHS


VERSION_NUMBER = 560
FEATURES = legacy_builder.FEATURES
BS = legacy_builder.BS
LIGHT_SPEED = legacy_builder.LIGHT_SPEED
DELAY_SCALE = legacy_builder.DELAY_SCALE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("pilot", "official"), default="pilot")
    parser.add_argument("--max-paths", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def open_output(path: Path, dtype, shape: tuple[int, ...]) -> np.memmap:
    partial = path.with_name(path.stem + ".partial.npy")
    if partial.exists():
        partial.unlink()
    return np.lib.format.open_memmap(partial, mode="w+", dtype=dtype, shape=shape)


def finish(path: Path, value: np.memmap) -> None:
    value.flush()
    del value
    path.with_name(path.stem + ".partial.npy").replace(path)


def token_features(
    data: np.lib.npyio.NpzFile,
    local: int,
    order: np.ndarray,
    ue: np.ndarray,
    first: np.ndarray,
    last: np.ndarray,
) -> np.ndarray:
    count = len(order)
    tau = data["tau"][local, order].astype(np.float32)
    delay = tau * DELAY_SCALE
    length = tau * LIGHT_SPEED
    direct = float(np.linalg.norm(ue - BS))
    depth = data["interaction_count"][local, order].astype(np.float32)
    mask = data["interaction_mask"][local, order].astype(np.int16)
    angles = [
        data[name][local, order].astype(np.float32)
        for name in ("theta_t", "phi_t", "theta_r", "phi_r")
    ]
    angle_feature = np.column_stack(
        [value for angle in angles for value in (np.sin(angle), np.cos(angle))]
    )
    first_value = first[local, order]
    last_value = last[local, order]
    global_vertex = np.column_stack(
        [
            (first_value[:, 0] - 150.0) / 150.0,
            (first_value[:, 1] + 50.0) / 200.0,
            first_value[:, 2] / 50.0,
            (last_value[:, 0] - 150.0) / 150.0,
            (last_value[:, 1] + 50.0) / 200.0,
            last_value[:, 2] / 50.0,
        ]
    )
    relative_vertex = np.column_stack(
        [
            (first_value[:, 0] - ue[0]) / 300.0,
            (first_value[:, 1] - ue[1]) / 300.0,
            (last_value[:, 0] - ue[0]) / 300.0,
            (last_value[:, 1] - ue[1]) / 300.0,
        ]
    )
    current = np.column_stack(
        [
            tau * 1e6 / 8.0,
            delay / 48.0,
            length / 600.0,
            np.maximum(length - direct, 0.0) / 500.0,
            depth / 16.0,
            (depth == 0).astype(np.float32),
            *[((mask & bit) != 0).astype(np.float32) for bit in (1, 2, 4, 8)],
            angle_feature,
            global_vertex,
            relative_vertex,
        ]
    ).astype(np.float32)
    if current.shape != (count, FEATURES):
        raise ValueError(f"Unexpected token feature shape {current.shape}")
    return np.clip(current, -4.0, 4.0).astype(np.float16)


def summary(value: np.ndarray) -> dict[str, float | int]:
    value = np.asarray(value, dtype=np.float64)
    return {
        "count": int(len(value)),
        "mean": float(np.mean(value)),
        "p10": float(np.quantile(value, 0.10)),
        "median": float(np.median(value)),
        "p90": float(np.quantile(value, 0.90)),
        "p99": float(np.quantile(value, 0.99)),
        "min": float(np.min(value)),
        "max": float(np.max(value)),
    }


def legacy_sources(dataset: str) -> tuple[Path, Path, Path] | None:
    if dataset == "pilot":
        root = PATHS.cache_dir(445)
        paths = tuple(root / f"path_candidate_{name}.npy" for name in ("feature", "valid", "mass"))
    else:
        root = PATHS.cache_dir(477)
        paths = (
            root / "v477_path_candidate_feature.npy",
            root / "v477_path_candidate_valid.npy",
            Path(),
        )
    if paths[0].is_file() and paths[1].is_file():
        return paths
    return None


def compare_legacy(
    dataset: str,
    feature: np.ndarray,
    valid: np.ndarray,
    mass: np.ndarray,
) -> dict[str, object] | None:
    sources = legacy_sources(dataset)
    if sources is None:
        return None
    old_feature = np.load(sources[0], mmap_mode="r")[: len(feature), :128]
    old_valid = np.load(sources[1], mmap_mode="r")[: len(feature), :128]
    new_feature = np.asarray(feature[:, :128])
    new_valid = np.asarray(valid[:, :128])
    valid_equal = np.all(old_valid == new_valid, axis=1)
    feature_equal = np.all(old_feature == new_feature, axis=(1, 2))
    changed = ~(valid_equal & feature_equal)
    output: dict[str, object] = {
        "legacy_feature": PATHS.relative(sources[0]),
        "legacy_valid": PATHS.relative(sources[1]),
        "row_count": int(len(changed)),
        "changed_row_count": int(np.sum(changed)),
        "changed_row_fraction": float(np.mean(changed)),
        "valid_mask_changed_fraction": float(np.mean(~valid_equal)),
        "feature_changed_fraction": float(np.mean(~feature_equal)),
    }
    if sources[2].is_file():
        old_mass = np.load(sources[2], mmap_mode="r")[: len(feature), :128]
        mass_l1 = np.sum(
            np.abs(np.asarray(old_mass, dtype=np.float32) - mass[:, :128].astype(np.float32)),
            axis=1,
        )
        output["mass_l1"] = summary(mass_l1)
    return output


def main() -> None:
    args = parse_args()
    root = PATHS.cache_dir(VERSION_NUMBER) / args.dataset
    source = root / "shards"
    manifest = root / "positions.npz"
    report_dir = PATHS.version_report_dir(VERSION_NUMBER)
    report_path = report_dir / f"v560_{args.dataset}_geometry_only_path_tokens.json"
    for path in (source, manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    positions_data = np.load(manifest)
    positions = positions_data["positions"].astype(np.float32)
    count = len(positions)
    output_paths = {
        "feature": root / "path_candidate_feature.npy",
        "valid": root / "path_candidate_valid.npy",
        "mass": root / "path_candidate_mass.npy",
    }
    if not args.overwrite and all(path.is_file() for path in output_paths.values()):
        print(f"v560 {args.dataset} tokens already exist", flush=True)
        return

    feature = open_output(output_paths["feature"], np.float16, (count, args.max_paths, FEATURES))
    valid_out = open_output(output_paths["valid"], np.bool_, (count, args.max_paths))
    mass_out = open_output(output_paths["mass"], np.float16, (count, args.max_paths))
    feature[:] = 0
    valid_out[:] = False
    mass_out[:] = 0
    retained = np.ones(count, dtype=np.float32)
    eligible = np.zeros(count, dtype=np.int32)
    selected = np.zeros(count, dtype=np.int32)
    overlap = np.ones(count, dtype=np.float32)
    seen = np.zeros(count, dtype=bool)
    started = time.time()

    shard_paths = sorted(source.glob("shard_*/path_targets.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No completed path shards under {source}")
    for shard_index, shard in enumerate(shard_paths):
        data = np.load(shard)
        if data["valid"].shape[1] < args.max_paths:
            raise ValueError(
                f"{shard} stores {data['valid'].shape[1]} paths, requested {args.max_paths}"
            )
        metadata = np.load(shard.parent / "metadata.npz")
        ids = metadata["position_id"].astype(np.int64)
        interactions = data["interactions"].astype(np.int8)
        first, last = legacy_builder.first_last_vertices(
            data["vertices"].astype(np.float32), interactions
        )
        for local, global_id in enumerate(ids):
            if seen[global_id]:
                raise ValueError(f"Duplicate v560 row {global_id}")
            seen[global_id] = True
            order = np.flatnonzero(data["valid"][local])[: args.max_paths]
            current_count = len(order)
            if current_count:
                feature[global_id, :current_count] = token_features(
                    data, local, order, positions[global_id], first, last
                )
                valid_out[global_id, :current_count] = True
                current_mass = data["normalized_power"][local, order].astype(np.float32)
                current_mass /= max(float(np.sum(current_mass)), 1e-30)
                mass_out[global_id, :current_count] = current_mass.astype(np.float16)
            retained[global_id] = data["retained_power_fraction"][local]
            eligible[global_id] = data["eligible_path_count"][local]
            selected[global_id] = data["selected_path_count"][local]
            overlap[global_id] = data["power_topk_overlap"][local]
        if (shard_index + 1) % 10 == 0 or shard_index + 1 == len(shard_paths):
            print(
                f"v560 {args.dataset} tokens shards={shard_index + 1}/{len(shard_paths)}",
                flush=True,
            )
    if not np.all(seen):
        missing = np.flatnonzero(~seen)
        raise ValueError(f"Missing {len(missing)} rows, first={missing[:10].tolist()}")

    for name, value in (("feature", feature), ("valid", valid_out), ("mass", mass_out)):
        finish(output_paths[name], value)
    feature_value = np.load(output_paths["feature"], mmap_mode="r")
    valid_value = np.load(output_paths["valid"], mmap_mode="r")
    mass_value = np.load(output_paths["mass"], mmap_mode="r")
    mass_float = np.asarray(mass_value, dtype=np.float32)
    k128_mass = np.sum(mass_float[:, :128], axis=1)
    coverage_128 = np.clip(retained * k128_mass, 0.0, 1.0)
    overlap_128 = np.ones(count, dtype=np.float32)
    for row in range(count):
        path_count = int(selected[row])
        if path_count <= 128:
            continue
        top = np.argsort(mass_float[row, :path_count], kind="stable")[::-1][:128]
        overlap_128[row] = float(np.mean(top < 128))
    nonempty = eligible > 0
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "method": f"v560_{args.dataset}_geometry_only_path_tokens",
        "status": "geometry_only_tokens_ready",
        "source": PATHS.relative(source),
        "count": count,
        "max_paths": args.max_paths,
        "input_exclusions": [
            "path power",
            "complex coefficient",
            "power rank",
            "port fingerprints",
        ],
        "candidate_statistics": {
            "eligible_paths": summary(eligible),
            "selected_paths": summary(selected),
            "rows_above_128": int(np.sum(eligible > 128)),
            "rows_above_128_fraction": float(np.mean(eligible > 128)),
            "geometry_vs_power_topk_overlap_k128": summary(overlap_128),
            "retained_energy_k128_all_rows": summary(coverage_128),
            "retained_energy_k128_nonempty_rows": summary(coverage_128[nonempty]),
            f"retained_energy_k{args.max_paths}": summary(retained),
        },
        "legacy_comparison_k128": compare_legacy(
            args.dataset, feature_value, valid_value, mass_value
        ),
        "rerun_boundary": (
            "This pilot is a mixed-hardware Monte Carlo rerun: shard 0 was regenerated on "
            "H100 while shards 1-19 were retained from A800 generation. The legacy comparison "
            "therefore must not be interpreted as a pure top-K cleanup effect. The controlled "
            "transfer probe uses original v445 rows with path_count<128 instead."
        ),
        "validation": {
            name: {
                "path": PATHS.relative(path),
                "shape": list(np.load(path, mmap_mode="r").shape),
                "dtype": str(np.load(path, mmap_mode="r").dtype),
                "finite": bool(np.isfinite(np.load(path, mmap_mode="r")).all()),
            }
            for name, path in output_paths.items()
        },
        "seconds": float(time.time() - started),
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
