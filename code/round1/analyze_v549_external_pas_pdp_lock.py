#!/usr/bin/env python3
"""Test whether the external six-seed PAS complements the v540 PDP carrier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np

import analyze_v205_sionna_config_calibration as calibration
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v424_sionna_pas_full_pdp_lock_diagnostic as v424
from project_paths import PATHS


VERSION_NUMBER = 549
VERSION = "v549_external_pas_pdp_lock_diagnostic"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
POINT_METRICS = OUT_DIR / f"{VERSION}_point_metrics.npz"

BASE = PATHS.cache_dir(540) / "v540_Round1_Test_Channel.npy"
REFERENCE = PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"
MEMBER = "Round1_Test_Channel.npy"
ALPHAS = (0.50, 0.75, 1.00)
PAS_CYCLES = 4
METRICS = ("pas", "pdp", "leaked_sh", "ad_support", "complex_coherence")
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-archive", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def empty_point_metrics(rows: int) -> dict[str, np.ndarray]:
    return {name: np.empty(rows, dtype=np.float32) for name in METRICS}


def update_scores(
    output: dict[str, np.ndarray],
    start: int,
    end: int,
    prediction: np.ndarray,
    target_views: dict[str, np.ndarray],
) -> None:
    scores = calibration.spectral_scores(prediction, target_views)
    for name in METRICS:
        output[name][start:end] = np.asarray(scores[name], dtype=np.float32)


def summarize_scores(
    point_metrics: dict[str, np.ndarray], error: float, energy: float
) -> dict[str, float]:
    means = {name: float(np.mean(values)) for name, values in point_metrics.items()}
    nmse = float(error / max(energy, TINY))
    nmse_score = float(1.0 / (1.0 + nmse))
    return {
        **means,
        "nmse": nmse,
        "nmse_score": nmse_score,
        "score_conventional": float(
            0.4 * means["pas"] + 0.4 * means["pdp"] + 0.2 * nmse_score
        ),
        "score_traceback_leaked": float(
            0.4 * means["leaked_sh"] + 0.4 * means["pdp"] + 0.2 * nmse_score
        ),
    }


def main() -> None:
    args = parse_args()
    archive_path = args.external_archive.resolve()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    for path in (archive_path, BASE, REFERENCE):
        if not path.is_file():
            raise FileNotFoundError(path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = np.load(BASE, mmap_mode="r")
    reference = np.load(REFERENCE, mmap_mode="r")
    if base.shape != reference.shape or base.dtype != np.complex64:
        raise ValueError("v540/reference shape or dtype mismatch")

    variants = ("v540", "external", *(f"alpha_{alpha:.2f}" for alpha in ALPHAS))
    point_metrics = {name: empty_point_metrics(len(base)) for name in variants}
    errors = {name: 0.0 for name in variants}
    prediction_energy = {name: 0.0 for name in variants}
    reference_energy = 0.0
    structural = {
        f"alpha_{alpha:.2f}": {
            "pdp_lock_max": 0.0,
            "power_ratio_sum": 0.0,
            "nmse_vs_v540_error": 0.0,
            "nmse_vs_external_error": 0.0,
            "pas_to_external": [],
            "leaked_sh_to_external": [],
        }
        for alpha in ALPHAS
    }
    base_energy = 0.0
    external_energy = 0.0
    external_finite = True

    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("external ZIP CRC failed")
        with archive.open(MEMBER) as handle:
            version = np.lib.format.read_magic(handle)
            shape, fortran_order, dtype = np.lib.format._read_array_header(
                handle, version
            )
            if (
                tuple(shape) != tuple(base.shape)
                or fortran_order
                or dtype != np.dtype(np.complex64)
            ):
                raise ValueError(
                    f"invalid external NPY: {shape=} {fortran_order=} {dtype=}"
                )
            row_values = int(np.prod(shape[1:]))
            for start in range(0, len(base), args.batch_size):
                end = min(start + args.batch_size, len(base))
                count = (end - start) * row_values
                raw = handle.read(count * dtype.itemsize)
                if len(raw) != count * dtype.itemsize:
                    raise EOFError(f"external NPY ended at row {start}")
                external = np.frombuffer(raw, dtype=dtype, count=count).reshape(
                    end - start, *shape[1:]
                )
                external_finite = external_finite and bool(np.isfinite(external).all())
                base_h = np.asarray(base[start:end], dtype=np.complex64)
                truth_h = np.asarray(reference[start:end], dtype=np.complex64)
                truth_views = calibration.spectral_views(truth_h)
                external_views = calibration.spectral_views(external)
                base_support, _ = v423.leaked_support(base_h)
                external_support, _ = v423.leaked_support(external)
                base_pas = v423.leaked_marginals(base_support)
                external_pas = v423.leaked_marginals(external_support)

                block_variants: dict[str, np.ndarray] = {
                    "v540": base_h,
                    "external": external,
                }
                for alpha in ALPHAS:
                    key = f"alpha_{alpha:.2f}"
                    desired = v423.desired_pas(
                        base_pas,
                        external_pas,
                        np.full(end - start, alpha, dtype=np.float32),
                    )
                    candidate = base_h.copy()
                    for _ in range(PAS_CYCLES):
                        candidate = v423.project_pas(candidate, desired)
                        candidate = v424.restore_full_linkwise_pdp(candidate, base_h)
                    block_variants[key] = candidate
                    pdp_error = v424.full_linkwise_pdp_error(candidate, base_h)
                    structural[key]["pdp_lock_max"] = max(
                        structural[key]["pdp_lock_max"], float(np.max(pdp_error))
                    )
                    candidate_power = np.sum(
                        np.abs(candidate) ** 2, axis=(1, 2, 3), dtype=np.float64
                    )
                    local_base_power = np.sum(
                        np.abs(base_h) ** 2, axis=(1, 2, 3), dtype=np.float64
                    )
                    structural[key]["power_ratio_sum"] += float(
                        np.sum(candidate_power / np.maximum(local_base_power, TINY))
                    )
                    structural[key]["nmse_vs_v540_error"] += float(
                        np.sum(np.abs(candidate - base_h) ** 2, dtype=np.float64)
                    )
                    structural[key]["nmse_vs_external_error"] += float(
                        np.sum(np.abs(candidate - external) ** 2, dtype=np.float64)
                    )
                    teacher_scores = calibration.spectral_scores(
                        candidate, external_views
                    )
                    structural[key]["pas_to_external"].append(
                        np.asarray(teacher_scores["pas"], dtype=np.float32)
                    )
                    structural[key]["leaked_sh_to_external"].append(
                        np.asarray(teacher_scores["leaked_sh"], dtype=np.float32)
                    )

                reference_energy += float(
                    np.sum(np.abs(truth_h) ** 2, dtype=np.float64)
                )
                base_energy += float(np.sum(np.abs(base_h) ** 2, dtype=np.float64))
                external_energy += float(
                    np.sum(np.abs(external) ** 2, dtype=np.float64)
                )
                for name, prediction in block_variants.items():
                    update_scores(
                        point_metrics[name], start, end, prediction, truth_views
                    )
                    errors[name] += float(
                        np.sum(np.abs(prediction - truth_h) ** 2, dtype=np.float64)
                    )
                    prediction_energy[name] += float(
                        np.sum(np.abs(prediction) ** 2, dtype=np.float64)
                    )
                print(f"v549 external PAS diagnostic {end}/{len(base)}", flush=True)

    summaries = {
        name: {
            **summarize_scores(point_metrics[name], errors[name], reference_energy),
            "power_ratio_to_reference": float(
                prediction_energy[name] / max(reference_energy, TINY)
            ),
        }
        for name in variants
    }
    for alpha in ALPHAS:
        key = f"alpha_{alpha:.2f}"
        structural[key] = {
            "pdp_lock_max": float(structural[key]["pdp_lock_max"]),
            "point_power_ratio_mean": float(
                structural[key]["power_ratio_sum"] / len(base)
            ),
            "nmse_vs_v540": float(
                structural[key]["nmse_vs_v540_error"] / max(base_energy, TINY)
            ),
            "nmse_vs_external": float(
                structural[key]["nmse_vs_external_error"] / max(external_energy, TINY)
            ),
            "pas_to_external": float(
                np.mean(np.concatenate(structural[key]["pas_to_external"]))
            ),
            "leaked_sh_to_external": float(
                np.mean(np.concatenate(structural[key]["leaked_sh_to_external"]))
            ),
        }

    np.savez_compressed(
        POINT_METRICS,
        **{
            f"{variant}_{metric}": values
            for variant, metrics in point_metrics.items()
            for metric, values in metrics.items()
        },
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "The external six-seed result contains PAS information complementary to "
            "v540, and that information can be transplanted while exactly retaining "
            "v540 linkwise CIR magnitudes, PDP, and point power."
        ),
        "scope": (
            "The external output is used only as a diagnostic PAS teacher. This script "
            "does not generate a submission and does not cache the external full H."
        ),
        "inputs": {
            "external_archive": PATHS.relative(archive_path),
            "base": PATHS.relative(BASE),
            "reference": PATHS.relative(REFERENCE),
        },
        "parameters": {
            "alphas": list(ALPHAS),
            "pas_cycles": PAS_CYCLES,
            "batch_size": args.batch_size,
            "pas_marginals": ["sh", "psh", "shue", "shv"],
            "pdp_lock": "restore v540 full linkwise CIR magnitude after each cycle",
        },
        "external_integrity": {
            "shape": list(base.shape),
            "dtype": "complex64",
            "finite": external_finite,
            "zip_crc": True,
        },
        "v207_diagnostic": summaries,
        "structural": structural,
        "outputs": {
            "point_metrics": PATHS.relative(POINT_METRICS),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
