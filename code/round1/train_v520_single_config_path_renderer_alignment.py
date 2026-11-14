#!/usr/bin/env python3
"""Repeat v519 with deployable v475 single-configuration path candidates."""

from __future__ import annotations

import json

import numpy as np

import analyze_v507_complex_path_frequency_oracle as v507
import analyze_v510_joint_delay_h_complex_oracle as v510
import train_v519_end_to_end_path_spectral_renderer as v519
from project_paths import PATHS


VERSION_NUMBER = 520
VERSION = "v520_single_config_path_renderer_alignment"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
REPORT = REPORT_DIR / f"{VERSION}.json"
POINT_METRICS = CACHE_DIR / "v520_point_metrics.npz"
COEFFICIENT_CACHE = CACHE_DIR / "v520_single_config_coefficients_points160.npz"
V516_CACHE = PATHS.cache_dir(516) / "v516_path_coefficients_points160.npz"
V515_CFR = PATHS.cache_dir(515) / "v515_base_cfr_points160.npy"
V475_SHARDS = PATHS.cache_dir(475) / "shards"
TRAIN_H = PATHS.round1 / "Round1_Train_Channel.npy"
PATH_LIMIT = v519.v516.PATH_LIMIT


def load_v475_geometry() -> dict[str, np.ndarray]:
    output = {
        "valid": np.zeros((2000, 128), dtype=bool),
        "tau": np.zeros((2000, 128), dtype=np.float32),
        "phi_t": np.zeros((2000, 128), dtype=np.float32),
        "interaction_count": np.zeros((2000, 128), dtype=np.int16),
    }
    seen = np.zeros(2000, dtype=bool)
    for path in sorted(V475_SHARDS.glob("shard_*/path_targets.npz")):
        metadata = np.load(path.parent / "metadata.npz")
        point_ids = metadata["position_id"].astype(np.int64)
        keep = point_ids < 2000
        if not np.any(keep):
            continue
        with np.load(path) as data:
            for name in output:
                output[name][point_ids[keep]] = data[name][keep]
        seen[point_ids[keep]] = True
    if not np.all(seen):
        raise RuntimeError(f"missing v475 train rows: {np.flatnonzero(~seen)[:10]}")
    return output


def build_coefficients() -> None:
    if COEFFICIENT_CACHE.is_file():
        with np.load(COEFFICIENT_CACHE) as loaded:
            if all(
                loaded[name].shape[1] == PATH_LIMIT
                for name in (
                    "tau",
                    "h_center",
                    "depth",
                    "valid",
                    "source_coefficients",
                    "official_coefficients",
                )
            ):
                return
    for path in (V516_CACHE, V515_CFR, TRAIN_H):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(V516_CACHE) as reference:
        point_ids = reference["indices"].astype(np.int64)
    geometry = load_v475_geometry()
    source = np.load(V515_CFR, mmap_mode="r")
    truth = np.load(TRAIN_H, mmap_mode="r")
    count = len(point_ids)
    tau_out = np.zeros((count, PATH_LIMIT), dtype=np.float64)
    h_out = np.zeros_like(tau_out)
    depth_out = np.zeros((count, PATH_LIMIT), dtype=np.float32)
    valid_out = np.zeros((count, PATH_LIMIT), dtype=bool)
    source_coefficients = np.zeros(
        (count, PATH_LIMIT, 64), dtype=np.complex64
    )
    official_coefficients = np.zeros_like(source_coefficients)
    source_fit_coherence = np.zeros(count, dtype=np.float32)
    official_fit_coherence = np.zeros(count, dtype=np.float32)

    for row, point_id in enumerate(point_ids):
        valid = geometry["valid"][point_id]
        tau, h_center, depth = v510.path_candidates(
            geometry["tau"][point_id, valid],
            geometry["phi_t"][point_id, valid],
            geometry["interaction_count"][point_id, valid],
        )
        if not len(tau):
            continue
        matrix = v510.joint_basis(tau, h_center)
        source_target = v510.target_matrix(np.asarray(source[row], dtype=np.complex64))
        official_target = v510.target_matrix(
            np.asarray(truth[int(point_id)], dtype=np.complex64)
        )
        if len(tau) > PATH_LIMIT:
            coefficients = v507.ridge_coefficients(matrix, source_target)
            energy = np.sum(np.abs(coefficients) ** 2, axis=1, dtype=np.float64)
            score = np.log(np.maximum(energy, v507.TINY)) - 1e-6 * depth
            selected = np.argsort(score)[-PATH_LIMIT:]
        else:
            selected = np.arange(len(tau), dtype=np.int64)
        matrix = matrix[:, selected]
        tau = tau[selected]
        h_center = h_center[selected]
        depth = depth[selected]
        source_local = v507.ridge_coefficients(matrix, source_target)
        official_local = v507.ridge_coefficients(matrix, official_target)
        local_count = len(selected)
        tau_out[row, :local_count] = tau
        h_out[row, :local_count] = h_center
        depth_out[row, :local_count] = depth
        valid_out[row, :local_count] = True
        source_coefficients[row, :local_count] = source_local.astype(np.complex64)
        official_coefficients[row, :local_count] = official_local.astype(np.complex64)
        source_fit = v510.channel_from_matrix(matrix @ source_local)
        official_fit = v510.channel_from_matrix(matrix @ official_local)
        source_fit_coherence[row] = v507.complex_metrics(
            source_fit, np.asarray(source[row])
        )["complex_coherence"]
        official_fit_coherence[row] = v507.complex_metrics(
            official_fit, np.asarray(truth[int(point_id)])
        )["complex_coherence"]
        if (row + 1) % 10 == 0 or row + 1 == count:
            print(f"v520 coefficient fit {row + 1}/{count}", flush=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        COEFFICIENT_CACHE,
        indices=point_ids,
        tau=tau_out,
        h_center=h_out,
        depth=depth_out,
        valid=valid_out,
        source_coefficients=source_coefficients,
        official_coefficients=official_coefficients,
        source_fit_coherence=source_fit_coherence,
        official_fit_coherence=official_fit_coherence,
    )


def configure_v519() -> None:
    v519.VERSION_NUMBER = VERSION_NUMBER
    v519.VERSION = VERSION
    v519.REPORT_DIR = REPORT_DIR
    v519.CACHE_DIR = CACHE_DIR
    v519.CHECKPOINT_DIR = CHECKPOINT_DIR
    v519.FOLD_DIR = CACHE_DIR / "fold_predictions"
    v519.REPORT = REPORT
    v519.POINT_METRICS = POINT_METRICS
    v519.COEFFICIENT_CACHE = COEFFICIENT_CACHE
    # The official targets and point ordering are identical to v519.
    v519.TARGET_CACHE = PATHS.cache_dir(519) / "v519_official_target_spectra_points160.npz"


def main() -> None:
    build_coefficients()
    configure_v519()
    v519.main()
    if REPORT.is_file():
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        report["method"] = VERSION
        report["primary_hypothesis"] = (
            "The v519 end-to-end path spectral gain persists when the 16-configuration "
            "v506 union is replaced by the single-configuration v475 path candidates "
            "already available at all 2500 official coordinates."
        )
        report["candidate_source"] = {
            "version": 475,
            "scope": "all 2000 official-train and 500 official-test coordinates",
            "selection": "top 32 source-CFR-energy delay-H atoms",
        }
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
