#!/usr/bin/env python3
"""Prepare absolute-coordinate Round 1 complex-field inputs for E198."""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

import prepare_e197_cross_round_official as e197_prepare
import train_surface_interaction_energy_model as e85
from data import load_positions
from project_paths import PATHS


VERSION = "Round2-E198-cross-round-complex-field-prepare"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e198_cross_round_complex_field"
FIELD_DIR = ARTIFACT_DIR / "field"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e198_cross_round_complex_field"
REPORT = ANALYSIS_DIR / "prepare_report.json"

DONOR_IDS = FIELD_DIR / "round1_donor_ids_i32.npy"
DONOR_DISTANCE = FIELD_DIR / "round1_donor_distance_f32.npy"
DONOR_FEATURE = FIELD_DIR / "round1_complex_field_f16.npy"
COVERAGE = FIELD_DIR / "round1_coverage_f32.npy"
CANDIDATE_PHASE = FIELD_DIR / "round2_candidate_phase_f16.npy"

R1_POSITION = PATHS.round1 / "Round1_Train_Pos.npy"
R2_POSITION = PATHS.round2 / "Round2_Train_Pos.npy"
R1_COMPONENTS = e197_prepare.COMPONENTS
R2_COEFFICIENT = e85.PATH_CACHE / "coefficient.npy"
R2_MASK = e85.PATH_CACHE / "mask.npy"

R1_BS = np.asarray((50.0, 0.0, 25.0), dtype=np.float32)
PERIODS = np.asarray((e85.MH, e85.MV, e85.S_SC), dtype=np.float32)
POSITION_SCALE = np.asarray((50.0, 50.0, 20.0), dtype=np.float32)
DONORS = 8
COMPONENTS = 24
PORTS = e85.MP * e85.N_UE
FIELD_FEATURES = 63
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--phase-batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def expected_files() -> tuple[Path, ...]:
    return (DONOR_IDS, DONOR_DISTANCE, DONOR_FEATURE, COVERAGE, CANDIDATE_PHASE)


def validate_array(
    path: Path, shape: tuple[int, ...], dtype: np.dtype[Any]
) -> np.memmap:
    value = np.load(path, mmap_mode="r")
    if value.shape != shape or value.dtype != dtype:
        raise ValueError(f"invalid E198 cache {path}: {value.shape}/{value.dtype}")
    for start in range(0, len(value), 128):
        block = np.asarray(value[start : start + 128])
        if block.dtype.kind in "fc" and not np.isfinite(block).all():
            raise ValueError(f"non-finite E198 cache: {path}")
    return value


def _broadcast_point(value: np.ndarray, components: int = COMPONENTS) -> np.ndarray:
    return np.broadcast_to(value[:, :, None], (*value.shape[:2], components, value.shape[2]))


def build_field(batch_size: int, force: bool) -> dict[str, Any]:
    expected_shape = (4000, DONORS, COMPONENTS, FIELD_FEATURES)
    if (
        not force
        and DONOR_FEATURE.is_file()
        and DONOR_IDS.is_file()
        and DONOR_DISTANCE.is_file()
        and COVERAGE.is_file()
    ):
        feature = validate_array(DONOR_FEATURE, expected_shape, np.dtype(np.float16))
        ids = validate_array(DONOR_IDS, (4000, DONORS), np.dtype(np.int32))
        distance = validate_array(
            DONOR_DISTANCE, (4000, DONORS), np.dtype(np.float32)
        )
        coverage = validate_array(COVERAGE, (4000,), np.dtype(np.float32))
        return {
            "status": "reused",
            "feature_shape": list(feature.shape),
            "feature_dtype": str(feature.dtype),
            "nearest_distance_quantiles": np.quantile(
                np.asarray(distance[:, 0], dtype=np.float64), (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
            ).tolist(),
            "coverage_quantiles": np.quantile(
                np.asarray(coverage, dtype=np.float64), (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
            ).tolist(),
            "donor_id_digest": int(np.asarray(ids, dtype=np.int64).sum()),
        }

    started = time.time()
    setup, r2_position, _, r2_site, _ = load_positions()
    r1_position = np.load(R1_POSITION).astype(np.float32)
    component = np.load(R1_COMPONENTS, allow_pickle=False)
    coordinate = component["coordinates"].astype(np.float32)
    coefficient = component["coefficients"].astype(np.complex64)
    point_power = component["point_power"].astype(np.float64)
    residual_fraction = component["residual_fraction"].astype(np.float32)
    if coordinate.shape != (2000, COMPONENTS, 3):
        raise ValueError(f"unexpected Round1 coordinates: {coordinate.shape}")
    if coefficient.shape != (2000, COMPONENTS, PORTS):
        raise ValueError(f"unexpected Round1 coefficients: {coefficient.shape}")

    distance, donor_ids = cKDTree(r1_position[:, :2]).query(
        r2_position[:, :2], k=DONORS
    )
    distance = np.asarray(distance, dtype=np.float32)
    donor_ids = np.asarray(donor_ids, dtype=np.int32)
    coverage = np.exp(-distance[:, 0] / 30.0).astype(np.float32)
    coverage *= np.where(r2_site == 0, 1.0, 0.1).astype(np.float32)

    FIELD_DIR.mkdir(parents=True, exist_ok=True)
    ids_tmp = DONOR_IDS.with_name(f".{DONOR_IDS.name}.{os.getpid()}.tmp.npy")
    distance_tmp = DONOR_DISTANCE.with_name(
        f".{DONOR_DISTANCE.name}.{os.getpid()}.tmp.npy"
    )
    coverage_tmp = COVERAGE.with_name(f".{COVERAGE.name}.{os.getpid()}.tmp.npy")
    np.save(ids_tmp, donor_ids)
    np.save(distance_tmp, distance)
    np.save(coverage_tmp, coverage)

    feature_tmp = DONOR_FEATURE.with_name(
        f".{DONOR_FEATURE.name}.{os.getpid()}.tmp.npy"
    )
    feature_tmp.unlink(missing_ok=True)
    output = np.lib.format.open_memmap(
        feature_tmp,
        mode="w+",
        dtype=np.float16,
        shape=expected_shape,
    )
    donor_rank = np.arange(DONORS, dtype=np.float32)[None, :, None, None]
    component_rank = np.arange(COMPONENTS, dtype=np.float32)[None, None, :, None]
    for start in range(0, len(r2_position), batch_size):
        end = min(start + batch_size, len(r2_position))
        local_ids = donor_ids[start:end]
        local_coordinate = coordinate[local_ids]
        local_coefficient = coefficient[local_ids]
        local_abs = np.abs(local_coefficient).astype(np.float32)
        local_energy = np.sum(
            np.square(local_abs, dtype=np.float32), axis=3, dtype=np.float64
        ).astype(np.float32)
        local_power = point_power[local_ids].astype(np.float32)
        fraction = np.divide(
            local_energy,
            np.maximum(local_power[..., None], TINY),
            out=np.zeros_like(local_energy),
        )
        port = np.divide(
            local_abs,
            np.sqrt(np.maximum(local_energy[..., None], TINY)),
            out=np.zeros_like(local_abs),
        )
        unit = np.divide(
            local_coefficient,
            np.maximum(local_abs, 1e-12),
            out=np.ones_like(local_coefficient),
        )
        valid = (
            (local_power[..., None] > 0.0)
            & (local_energy > np.maximum(local_power[..., None], TINY) * 1e-8)
        )

        target = r2_position[start:end].astype(np.float32)
        donor = r1_position[local_ids]
        target_r2_bs = setup.bs_positions[r2_site[start:end]].astype(np.float32)
        offset = donor - target[:, None]
        local_distance = np.linalg.norm(offset, axis=2, keepdims=True)
        donor_relative_r1 = (donor - R1_BS) / 200.0
        target_relative_r1 = np.broadcast_to(
            ((target - R1_BS) / 200.0)[:, None],
            (end - start, DONORS, 3),
        )
        target_relative_r2 = np.broadcast_to(
            ((target - target_r2_bs) / 200.0)[:, None],
            (end - start, DONORS, 3),
        )
        donor_bearing = np.arctan2(offset[..., 1], offset[..., 0])
        target_r1_bearing = np.arctan2(
            target[:, 1] - R1_BS[1], target[:, 0] - R1_BS[0]
        )
        target_r2_bearing = np.arctan2(
            target[:, 1] - target_r2_bs[:, 1],
            target[:, 0] - target_r2_bs[:, 0],
        )
        coordinate_phase = 2.0 * math.pi * local_coordinate / PERIODS
        site_one_hot = np.stack(
            ((r2_site[start:end] == 0), (r2_site[start:end] == 1)), axis=1
        ).astype(np.float32)
        coverage_block = coverage[start:end, None, None, None]

        blocks = (
            local_coordinate / PERIODS,
            np.sin(coordinate_phase).astype(np.float32),
            np.cos(coordinate_phase).astype(np.float32),
            np.clip(np.log10(np.maximum(fraction, 1e-12)) / 12.0, -1.0, 0.0)[
                ..., None
            ].astype(np.float32),
            np.sqrt(np.maximum(fraction, 0.0))[..., None].astype(np.float32),
            port.astype(np.float32),
            unit.real.astype(np.float32),
            unit.imag.astype(np.float32),
            _broadcast_point((offset / POSITION_SCALE).astype(np.float32)),
            _broadcast_point((local_distance / 100.0).astype(np.float32)),
            _broadcast_point((np.log1p(local_distance) / 5.0).astype(np.float32)),
            _broadcast_point(donor_relative_r1.astype(np.float32)),
            _broadcast_point(target_relative_r1.astype(np.float32)),
            _broadcast_point(target_relative_r2.astype(np.float32)),
            np.broadcast_to(
                np.sin(donor_bearing)[:, :, None, None],
                (end - start, DONORS, COMPONENTS, 1),
            ).astype(np.float32),
            np.broadcast_to(
                np.cos(donor_bearing)[:, :, None, None],
                (end - start, DONORS, COMPONENTS, 1),
            ).astype(np.float32),
            np.broadcast_to(
                np.sin(target_r1_bearing)[:, None, None, None],
                (end - start, DONORS, COMPONENTS, 1),
            ).astype(np.float32),
            np.broadcast_to(
                np.cos(target_r1_bearing)[:, None, None, None],
                (end - start, DONORS, COMPONENTS, 1),
            ).astype(np.float32),
            np.broadcast_to(
                np.sin(target_r2_bearing)[:, None, None, None],
                (end - start, DONORS, COMPONENTS, 1),
            ).astype(np.float32),
            np.broadcast_to(
                np.cos(target_r2_bearing)[:, None, None, None],
                (end - start, DONORS, COMPONENTS, 1),
            ).astype(np.float32),
            np.broadcast_to(
                donor_rank / max(DONORS - 1, 1),
                (end - start, DONORS, COMPONENTS, 1),
            ),
            np.broadcast_to(
                component_rank / max(COMPONENTS - 1, 1),
                (end - start, DONORS, COMPONENTS, 1),
            ),
            np.broadcast_to(
                np.clip((np.log10(np.maximum(local_power, 1e-30)) + 12.0) / 8.0, -2.0, 2.0)[
                    :, :, None, None
                ],
                (end - start, DONORS, COMPONENTS, 1),
            ).astype(np.float32),
            np.broadcast_to(
                residual_fraction[local_ids][..., None, None],
                (end - start, DONORS, COMPONENTS, 1),
            ).astype(np.float32),
            valid[..., None].astype(np.float32),
            np.broadcast_to(
                site_one_hot[:, None, None],
                (end - start, DONORS, COMPONENTS, 2),
            ),
            np.broadcast_to(
                coverage_block,
                (end - start, DONORS, COMPONENTS, 1),
            ).astype(np.float32),
        )
        local_feature = np.concatenate(blocks, axis=3).astype(np.float32)
        if local_feature.shape != (end - start, DONORS, COMPONENTS, FIELD_FEATURES):
            raise RuntimeError(f"E198 field feature mismatch: {local_feature.shape}")
        if not np.isfinite(local_feature).all():
            raise FloatingPointError(f"non-finite E198 field block {start}:{end}")
        output[start:end] = local_feature.astype(np.float16)
        if end % 512 < batch_size or end == len(r2_position):
            output.flush()
            print(f"E198 Round1 field {end}/{len(r2_position)}", flush=True)
    output.flush()
    del output
    os.replace(ids_tmp, DONOR_IDS)
    os.replace(distance_tmp, DONOR_DISTANCE)
    os.replace(coverage_tmp, COVERAGE)
    os.replace(feature_tmp, DONOR_FEATURE)

    validate_array(DONOR_FEATURE, expected_shape, np.dtype(np.float16))
    return {
        "status": "created",
        "feature_shape": list(expected_shape),
        "feature_dtype": "float16",
        "nearest_distance_quantiles": np.quantile(
            distance[:, 0], (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
        ).tolist(),
        "coverage_quantiles": np.quantile(
            coverage, (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
        ).tolist(),
        "site_nearest_distance": {
            str(site): np.quantile(
                distance[r2_site == site, 0], (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
            ).tolist()
            for site in (0, 1)
        },
        "seconds": float(time.time() - started),
    }


def build_candidate_phase(batch_size: int, force: bool) -> dict[str, Any]:
    expected_shape = (4000, e85.PATHS_PER_POINT, 2 * PORTS)
    if not force and CANDIDATE_PHASE.is_file():
        cached = validate_array(
            CANDIDATE_PHASE, expected_shape, np.dtype(np.float16)
        )
        return {
            "status": "reused",
            "shape": list(cached.shape),
            "dtype": str(cached.dtype),
        }

    started = time.time()
    coefficient = np.load(R2_COEFFICIENT, mmap_mode="r")
    mask = np.load(R2_MASK, mmap_mode="r")
    if coefficient.shape != (4000, e85.PATHS_PER_POINT, 256, 4):
        raise ValueError(f"unexpected Round2 path coefficient: {coefficient.shape}")
    temporary = CANDIDATE_PHASE.with_name(
        f".{CANDIDATE_PHASE.name}.{os.getpid()}.tmp.npy"
    )
    temporary.unlink(missing_ok=True)
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float16, shape=expected_shape
    )
    for start in range(0, len(coefficient), batch_size):
        end = min(start + batch_size, len(coefficient))
        shaped = np.asarray(coefficient[start:end], dtype=np.complex64).reshape(
            end - start, e85.PATHS_PER_POINT, e85.MP, e85.MH, e85.MV, e85.N_UE
        )
        reference = shaped[:, :, :, 0, 0, :].reshape(
            end - start, e85.PATHS_PER_POINT, PORTS
        )
        magnitude = np.abs(reference)
        unit = np.divide(
            reference,
            np.maximum(magnitude, 1e-12),
            out=np.ones_like(reference),
        ).astype(np.complex64)
        local_mask = np.asarray(mask[start:end], dtype=bool)
        unit = np.where(local_mask[..., None], unit, np.ones_like(unit))
        value = np.concatenate((unit.real, unit.imag), axis=2).astype(np.float32)
        if not np.isfinite(value).all():
            raise FloatingPointError(f"non-finite candidate phase {start}:{end}")
        output[start:end] = value.astype(np.float16)
        if end % 256 < batch_size or end == len(coefficient):
            output.flush()
            print(f"E198 candidate phase {end}/{len(coefficient)}", flush=True)
    output.flush()
    del output
    os.replace(temporary, CANDIDATE_PHASE)
    cached = validate_array(CANDIDATE_PHASE, expected_shape, np.dtype(np.float16))
    return {
        "status": "created",
        "shape": list(cached.shape),
        "dtype": str(cached.dtype),
        "seconds": float(time.time() - started),
    }


def main() -> None:
    args = parse_args()
    FIELD_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    missing = [
        PATHS.relative(path)
        for path in (R1_POSITION, R2_POSITION, R1_COMPONENTS, R2_COEFFICIENT, R2_MASK)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing E198 prepare inputs: {missing}")
    started = time.time()
    try:
        field = build_field(args.batch_size, args.force)
        phase = build_candidate_phase(args.phase_batch_size, args.force)
        report = {
            "experiment": VERSION,
            "status": "complete",
            "field": field,
            "candidate_phase": phase,
            "data_lineage": {
                "round1_official_train_h_components": PATHS.relative(R1_COMPONENTS),
                "round1_absolute_coordinates": True,
                "round1_scene_translation_applied": False,
                "round2_same_position_external_path_coefficients": PATHS.relative(
                    R2_COEFFICIENT
                ),
                "historical_prediction_h_read": False,
                "hidden_test_truth_read": False,
            },
            "performance": {
                "peak_rss_gib": float(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
                ),
                "measured_bottleneck": "candidate_phase_sequential_read"
                if phase.get("status") == "created"
                else "cache_validation",
            },
            "seconds": float(time.time() - started),
        }
        atomic_json(REPORT, report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    except Exception:
        failure = {
            "experiment": VERSION,
            "status": "failed",
            "traceback": traceback.format_exc(),
        }
        atomic_json(REPORT, failure)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
