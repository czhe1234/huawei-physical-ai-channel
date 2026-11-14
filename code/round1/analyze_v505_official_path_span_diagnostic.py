#!/usr/bin/env python3
"""Separate Sionna path-set geometry error from path-weight error."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

import analyze_v501_full_official_sionna_inverse_calibration as v501
from project_paths import PATHS


VERSION_NUMBER = 505
VERSION = "v505_official_path_span_diagnostic"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
POINT_METRICS = CACHE_DIR / "v505_path_span_point_metrics.npz"
V503_REPORT = PATHS.version_report_dir(503) / "v503_refine.json"
PATH_ROOT = PATHS.cache_dir(475) / "shards"
TRAIN_H = PATHS.round1 / "Round1_Train_Channel.npy"

N_DELAY = 192
N_H = 16
SUBCARRIER_SPACING = 30_000.0
TX_ORIENTATION_RAD = math.radians(-30.0)
REGIME_NAMES = ("clear", "light", "medium", "heavy")
TINY = 1e-30


def normalize_sum(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    return (value / max(float(np.sum(value)), TINY)).astype(np.float32)


def row_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64).reshape(-1)
    right64 = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left64) * np.linalg.norm(right64))
    if denominator <= TINY:
        return 0.0
    return float(np.clip(np.dot(left64, right64) / denominator, 0.0, 1.0))


def delay_kernel(tau: np.ndarray, fft_sign: int) -> np.ndarray:
    """Exact finite-band Dirichlet power atom for IFFT (-) or FFT (+)."""
    grid = np.arange(N_DELAY, dtype=np.float64) / N_DELAY
    delta = SUBCARRIER_SPACING * tau[:, None].astype(np.float64)
    delta = delta + grid[None, :] if fft_sign > 0 else delta - grid[None, :]
    numerator = np.sinc(N_DELAY * delta)
    denominator = np.sinc(delta)
    value = (numerator / np.maximum(np.abs(denominator), TINY)) ** 2
    value /= np.maximum(value.sum(axis=1, keepdims=True), TINY)
    return value.astype(np.float32)


def horizontal_kernel(phi_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mod(
        8.0 * np.sin(phi_t.astype(np.float64) - TX_ORIENTATION_RAD), N_H
    )
    element = np.arange(N_H, dtype=np.float64)
    steering = np.exp(2j * np.pi * center[:, None] * element[None, :] / N_H)
    value = np.abs(np.fft.fft(steering, axis=1)) ** 2
    value /= np.maximum(value.sum(axis=1, keepdims=True), TINY)
    return value.astype(np.float32), center.astype(np.float32)


def fit_nonnegative_atoms(
    target: np.ndarray, atoms: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    atoms64 = np.asarray(atoms, dtype=np.float64).reshape(len(atoms), -1)
    target64 = np.asarray(target, dtype=np.float64).reshape(-1)
    keep = np.all(np.isfinite(atoms64), axis=1) & (np.sum(atoms64, axis=1) > TINY)
    atoms64 = atoms64[keep]
    if len(atoms64) == 0 or np.linalg.norm(target64) <= TINY:
        return np.zeros_like(target, dtype=np.float32), np.zeros(len(atoms), dtype=np.float32)
    atoms64 /= np.maximum(atoms64.sum(axis=1, keepdims=True), TINY)
    weight, _ = nnls(atoms64.T, target64, maxiter=max(300, len(atoms64) * 20))
    if np.sum(weight) <= TINY:
        weight[int(np.argmax(atoms64 @ target64))] = 1.0
    reconstruction = normalize_sum(weight @ atoms64).reshape(target.shape)
    full_weight = np.zeros(len(atoms), dtype=np.float32)
    full_weight[keep] = (weight / max(float(np.sum(weight)), TINY)).astype(np.float32)
    return reconstruction.astype(np.float32), full_weight


def official_targets(channel: np.ndarray) -> dict[str, np.ndarray]:
    channel = np.asarray(channel, dtype=np.complex64)
    h6 = channel.reshape(2, 16, 8, 4, N_DELAY)
    delay = np.fft.ifft(channel, axis=-1)
    delay_power = np.abs(delay) ** 2
    delay6 = delay_power.reshape(2, 16, 8, 4, N_DELAY)
    pdp = normalize_sum(np.sum(delay_power, axis=(0, 1), dtype=np.float64))
    pol_ue_delay = normalize_sum(np.sum(delay6, axis=(1, 2), dtype=np.float64))

    leaked_input = h6.transpose(0, 4, 1, 3, 2)
    leaked = np.fft.fft2(leaked_input, axes=(1, 2))
    joint_sh = normalize_sum(np.sum(np.abs(leaked) ** 2, axis=(0, 3, 4)))
    return {"pdp": pdp, "pol_ue_delay": pol_ue_delay, "joint_sh": joint_sh}


def load_selected_rows(indices: np.ndarray) -> dict[str, np.ndarray]:
    path_keys = (
        "valid",
        "tau",
        "phi_t",
        "normalized_power",
        "interaction_count",
        "tx_pol_fingerprint",
        "rx_fingerprint",
    )
    spectral_keys = ("pdp", "pol_ue_delay", "joint_sh")
    rows: dict[str, list[np.ndarray]] = {key: [] for key in path_keys + spectral_keys}
    opened: dict[int, tuple[np.lib.npyio.NpzFile, np.lib.npyio.NpzFile]] = {}
    try:
        for point_id in indices:
            shard_id = int(point_id) // 100
            row = int(point_id) % 100
            if shard_id not in opened:
                shard = PATH_ROOT / f"shard_{shard_id:05d}"
                opened[shard_id] = (
                    np.load(shard / "path_targets.npz"),
                    np.load(shard / "spectral_targets.npz"),
                )
            path_data, spectral_data = opened[shard_id]
            for key in path_keys:
                rows[key].append(np.asarray(path_data[key][row]))
            for key in spectral_keys:
                rows[key].append(np.asarray(spectral_data[key][row], dtype=np.float32))
    finally:
        for path_data, spectral_data in opened.values():
            path_data.close()
            spectral_data.close()
    return {key: np.stack(value) for key, value in rows.items()}


def circular_distance(grid: np.ndarray, center: np.ndarray, size: int) -> np.ndarray:
    delta = np.abs(grid[None, :] - center[:, None])
    return np.minimum(delta, size - delta)


def support_coverage(
    target_pdp: np.ndarray,
    target_sh: np.ndarray,
    pdp_centers: np.ndarray,
    sh_centers: np.ndarray,
    h_centers: np.ndarray,
) -> tuple[float, float]:
    delay_grid = np.arange(N_DELAY, dtype=np.float32)
    h_grid = np.arange(N_H, dtype=np.float32)
    pdp_mask = np.any(circular_distance(delay_grid, pdp_centers, N_DELAY) <= 1.5, axis=0)
    pdp_coverage = float(np.sum(target_pdp[pdp_mask]))

    delay_near = circular_distance(delay_grid, sh_centers, N_DELAY) <= 1.5
    h_near = circular_distance(h_grid, h_centers, N_H) <= 1.5
    sh_mask = np.any(delay_near[:, :, None] & h_near[:, None, :], axis=0)
    sh_coverage = float(np.sum(target_sh[sh_mask]))
    return pdp_coverage, sh_coverage


def summarize(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    selected = np.asarray(values, dtype=np.float64)[mask]
    if len(selected) == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    return {
        "count": int(len(selected)),
        "mean": float(np.mean(selected)),
        "median": float(np.median(selected)),
        "p10": float(np.quantile(selected, 0.10)),
        "p90": float(np.quantile(selected, 0.90)),
    }


def grouped_summary(
    metrics: dict[str, np.ndarray], regimes: np.ndarray, nonzero: np.ndarray
) -> dict[str, object]:
    groups: dict[str, object] = {
        "overall_nonzero": {
            key: summarize(value, nonzero) for key, value in metrics.items()
        }
    }
    for code, name in enumerate(REGIME_NAMES):
        mask = nonzero & (regimes == code)
        groups[name] = {key: summarize(value, mask) for key, value in metrics.items()}
    return groups


def main() -> None:
    for path in (V503_REPORT, TRAIN_H):
        if not path.is_file():
            raise FileNotFoundError(path)
    path_files = sorted(PATH_ROOT.glob("shard_*/path_targets.npz"))
    if len(path_files) != 25:
        raise ValueError(f"Expected 25 v475 path shards, found {len(path_files)}")

    v503_report = json.loads(V503_REPORT.read_text(encoding="utf-8"))
    indices = np.asarray(v503_report["indices"], dtype=np.int64)
    if len(indices) != 160 or np.max(indices) >= 2000:
        raise ValueError("v505 expects the 160 official-train rows used by v503 refine")
    paths = load_selected_rows(indices)
    truth_h = np.load(TRAIN_H, mmap_mode="r")
    regimes = v501.regime_codes()[indices]

    metric_names = (
        "raw_pdp_cosine",
        "rendered_pdp_to_raw_cosine",
        "oracle_pdp_cosine",
        "raw_pol_ue_cosine",
        "rendered_pol_ue_to_raw_cosine",
        "oracle_pol_ue_cosine",
        "raw_joint_sh_cosine",
        "rendered_joint_sh_to_raw_cosine",
        "oracle_joint_sh_cosine",
        "pdp_support_coverage",
        "joint_sh_support_coverage",
        "oracle_pdp_recoverable_fraction",
        "oracle_joint_sh_recoverable_fraction",
        "oracle_pol_ue_recoverable_fraction",
        "oracle_mean_interaction_depth",
        "valid_path_count",
    )
    metrics = {name: np.zeros(len(indices), dtype=np.float32) for name in metric_names}
    nonzero = np.zeros(len(indices), dtype=bool)

    for output_row, point_id in enumerate(indices):
        target = official_targets(truth_h[point_id])
        nonzero[output_row] = np.linalg.norm(target["pdp"]) > TINY
        valid = paths["valid"][output_row].astype(bool)
        tau = paths["tau"][output_row, valid].astype(np.float32)
        phi_t = paths["phi_t"][output_row, valid].astype(np.float32)
        raw_weight = paths["normalized_power"][output_row, valid].astype(np.float32)
        interaction = paths["interaction_count"][output_row, valid].astype(np.float32)
        metrics["valid_path_count"][output_row] = len(tau)
        if not nonzero[output_row] or len(tau) == 0:
            continue

        pdp_kernel = delay_kernel(tau, fft_sign=-1)
        sh_delay_kernel = delay_kernel(tau, fft_sign=1)
        h_kernel, h_center = horizontal_kernel(phi_t)
        sh_atoms = sh_delay_kernel[:, :, None] * h_kernel[:, None, :]
        pol_ue_pattern = (
            paths["tx_pol_fingerprint"][output_row, valid].astype(np.float32)[:, :, None]
            * paths["rx_fingerprint"][output_row, valid].astype(np.float32)[:, None, :]
        )
        pol_ue_pattern /= np.maximum(
            pol_ue_pattern.sum(axis=(1, 2), keepdims=True), TINY
        )
        pol_ue_atoms = pol_ue_pattern[:, :, :, None] * pdp_kernel[:, None, None, :]

        rendered_pdp = normalize_sum(np.sum(pdp_kernel * raw_weight[:, None], axis=0))
        rendered_sh = normalize_sum(np.sum(sh_atoms * raw_weight[:, None, None], axis=0))
        rendered_pol_ue = normalize_sum(
            np.sum(pol_ue_atoms * raw_weight[:, None, None, None], axis=0)
        )
        oracle_pdp, weight_pdp = fit_nonnegative_atoms(target["pdp"], pdp_kernel)
        oracle_sh, weight_sh = fit_nonnegative_atoms(target["joint_sh"], sh_atoms)
        oracle_pol_ue, weight_pol_ue = fit_nonnegative_atoms(
            target["pol_ue_delay"], pol_ue_atoms
        )

        metrics["raw_pdp_cosine"][output_row] = row_cosine(
            target["pdp"], paths["pdp"][output_row]
        )
        metrics["rendered_pdp_to_raw_cosine"][output_row] = row_cosine(
            rendered_pdp, paths["pdp"][output_row]
        )
        metrics["oracle_pdp_cosine"][output_row] = row_cosine(target["pdp"], oracle_pdp)
        metrics["raw_pol_ue_cosine"][output_row] = row_cosine(
            target["pol_ue_delay"], paths["pol_ue_delay"][output_row]
        )
        metrics["rendered_pol_ue_to_raw_cosine"][output_row] = row_cosine(
            rendered_pol_ue, paths["pol_ue_delay"][output_row]
        )
        metrics["oracle_pol_ue_cosine"][output_row] = row_cosine(
            target["pol_ue_delay"], oracle_pol_ue
        )
        metrics["raw_joint_sh_cosine"][output_row] = row_cosine(
            target["joint_sh"], paths["joint_sh"][output_row]
        )
        metrics["rendered_joint_sh_to_raw_cosine"][output_row] = row_cosine(
            rendered_sh, paths["joint_sh"][output_row]
        )
        metrics["oracle_joint_sh_cosine"][output_row] = row_cosine(
            target["joint_sh"], oracle_sh
        )

        pdp_centers = np.argmax(pdp_kernel, axis=1).astype(np.float32)
        sh_centers = np.argmax(sh_delay_kernel, axis=1).astype(np.float32)
        pdp_coverage, sh_coverage = support_coverage(
            target["pdp"], target["joint_sh"], pdp_centers, sh_centers, h_center
        )
        metrics["pdp_support_coverage"][output_row] = pdp_coverage
        metrics["joint_sh_support_coverage"][output_row] = sh_coverage

        raw_pdp = metrics["raw_pdp_cosine"][output_row]
        raw_sh = metrics["raw_joint_sh_cosine"][output_row]
        raw_pol_ue = metrics["raw_pol_ue_cosine"][output_row]
        metrics["oracle_pdp_recoverable_fraction"][output_row] = (
            max(metrics["oracle_pdp_cosine"][output_row] - raw_pdp, 0.0)
        ) / max(1.0 - raw_pdp, 1e-6)
        metrics["oracle_joint_sh_recoverable_fraction"][output_row] = (
            max(metrics["oracle_joint_sh_cosine"][output_row] - raw_sh, 0.0)
        ) / max(1.0 - raw_sh, 1e-6)
        metrics["oracle_pol_ue_recoverable_fraction"][output_row] = (
            max(metrics["oracle_pol_ue_cosine"][output_row] - raw_pol_ue, 0.0)
        ) / max(1.0 - raw_pol_ue, 1e-6)

        aggregate_weight = (weight_pdp + weight_sh + weight_pol_ue) / 3.0
        metrics["oracle_mean_interaction_depth"][output_row] = float(
            np.sum(aggregate_weight * interaction)
        )
        if (output_row + 1) % 20 == 0 or output_row + 1 == len(indices):
            print(f"v505 path span {output_row + 1}/{len(indices)}", flush=True)

    grouped = grouped_summary(metrics, regimes, nonzero)
    overall = grouped["overall_nonzero"]
    heavy = grouped["heavy"]
    conclusion = {
        "overall_delay_path_span_is_near_complete": bool(
            overall["oracle_pdp_cosine"]["mean"] >= 0.95
        ),
        "heavy_delay_path_span_is_near_complete": bool(
            heavy["oracle_pdp_cosine"]["mean"] >= 0.95
        ),
        "heavy_delay_angle_path_span_is_near_complete": bool(
            heavy["oracle_joint_sh_cosine"]["mean"] >= 0.90
        ),
        "heavy_port_path_span_is_near_complete": bool(
            heavy["oracle_pol_ue_cosine"]["mean"] >= 0.90
        ),
        "heavy_points_below_0p8": {
            "delay": int(
                np.sum(
                    nonzero
                    & (regimes == 3)
                    & (metrics["oracle_pdp_cosine"] < 0.8)
                )
            ),
            "delay_angle": int(
                np.sum(
                    nonzero
                    & (regimes == 3)
                    & (metrics["oracle_joint_sh_cosine"] < 0.8)
                )
            ),
            "polarization_ue_delay": int(
                np.sum(
                    nonzero
                    & (regimes == 3)
                    & (metrics["oracle_pol_ue_cosine"] < 0.8)
                )
            ),
        },
        "decision_rule": (
            "If the nonnegative per-path oracle remains low, the missing official "
            "configuration changes path birth/death or geometry. If it is high while raw "
            "Sionna is low, focus on path coefficients, polarization, and post-processing."
        ),
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        POINT_METRICS,
        indices=indices,
        regimes=regimes,
        nonzero=nonzero,
        **metrics,
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "The remaining mismatch can be localized by freeing only per-path nonnegative "
            "energy while keeping Sionna delay, departure angle, and port fingerprints fixed."
        ),
        "inputs": {
            "official_train_h": PATHS.relative(TRAIN_H),
            "point_indices": "same 160 balanced nonzero official rows as v503 refine",
            "sionna_path_tokens": PATHS.relative(PATH_ROOT),
            "sionna_configuration": (
                "v475: Sionna 2.0.1, 6.693GHz, depth16, 1m, LOS+specular+refraction"
            ),
        },
        "point_count": int(len(indices)),
        "nonzero_count": int(np.sum(nonzero)),
        "metrics": grouped,
        "conclusion": conclusion,
        "outputs": {
            "point_metrics": PATHS.relative(POINT_METRICS),
            "report": PATHS.relative(REPORT),
            "submission_zip": None,
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "point_count": len(indices),
                "raw_pdp": overall["raw_pdp_cosine"]["mean"],
                "oracle_pdp": overall["oracle_pdp_cosine"]["mean"],
                "raw_joint_sh": overall["raw_joint_sh_cosine"]["mean"],
                "oracle_joint_sh": overall["oracle_joint_sh_cosine"]["mean"],
                "raw_pol_ue": overall["raw_pol_ue_cosine"]["mean"],
                "oracle_pol_ue": overall["oracle_pol_ue_cosine"]["mean"],
                "report": PATHS.relative(REPORT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
