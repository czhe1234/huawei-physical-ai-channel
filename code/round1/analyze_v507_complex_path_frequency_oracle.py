#!/usr/bin/env python3
"""Cross-frequency complex oracle on the v506 multi-mechanism delay union."""

from __future__ import annotations

import json
import math

import numpy as np

import analyze_v501_full_official_sionna_inverse_calibration as v501
import analyze_v503_fixed_material_sionna_version_inverse as v503
import analyze_v505_official_path_span_diagnostic as v505
import analyze_v506_path_mechanism_union_oracle as v506
from project_paths import PATHS


VERSION_NUMBER = 507
VERSION = "v507_complex_path_frequency_oracle"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
POINT_METRICS = CACHE_DIR / "v507_complex_frequency_point_metrics.npz"
TRAIN_H = PATHS.round1 / "Round1_Train_Channel.npy"
V503_REPORT = PATHS.version_report_dir(503) / "v503_refine.json"
V506_REPORT = PATHS.version_report_dir(506) / "v506_path_mechanism_union_oracle.json"

N_SUBCARRIERS = 192
SUBCARRIER_SPACING = 30_000.0
MAX_DELAYS = 64
RIDGE_RELATIVE = 1e-4
REGIME_NAMES = ("clear", "light", "medium", "heavy")
TINY = 1e-30


def load_union(indices: np.ndarray) -> dict[str, np.ndarray]:
    path_sets = []
    for config in v506.configs():
        path = v506.path_set_path(config, len(indices))
        if not path.is_file():
            raise FileNotFoundError(path)
        data = np.load(path)
        path_sets.append({key: np.asarray(data[key]) for key in data.files})
    return v506.union_path_sets(path_sets, indices)


def unique_delays(
    tau: np.ndarray, interaction_count: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(tau) == 0:
        return tau.astype(np.float64), interaction_count.astype(np.float32)
    center = np.mod(
        N_SUBCARRIERS * SUBCARRIER_SPACING * tau.astype(np.float64),
        N_SUBCARRIERS,
    )
    key = np.rint(center * 8.0).astype(np.int32)
    order = np.lexsort((tau, interaction_count))
    selected = []
    seen: set[int] = set()
    for index in order:
        local_key = int(key[index])
        if local_key in seen:
            continue
        seen.add(local_key)
        selected.append(int(index))
    keep = np.asarray(selected, dtype=np.int64)
    return tau[keep].astype(np.float64), interaction_count[keep].astype(np.float32)


def basis(tau: np.ndarray) -> np.ndarray:
    frequency = (
        np.arange(N_SUBCARRIERS, dtype=np.float64) - N_SUBCARRIERS // 2
    ) * SUBCARRIER_SPACING
    return np.exp(-2j * np.pi * frequency[:, None] * tau[None, :]).astype(np.complex128)


def ridge_coefficients(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    gram = matrix.conj().T @ matrix
    scale = float(np.trace(gram).real / max(len(gram), 1))
    regularized = gram + np.eye(len(gram), dtype=np.complex128) * (
        RIDGE_RELATIVE * max(scale, TINY)
    )
    return np.linalg.solve(regularized, matrix.conj().T @ target)


def select_delays(
    matrix: np.ndarray, target: np.ndarray, interaction_count: np.ndarray
) -> np.ndarray:
    if matrix.shape[1] <= MAX_DELAYS:
        return np.arange(matrix.shape[1], dtype=np.int64)
    coefficients = ridge_coefficients(matrix, target)
    energy = np.sum(np.abs(coefficients) ** 2, axis=1, dtype=np.float64)
    # Tiny tie-break favors simpler paths without overriding measured energy.
    score = np.log(np.maximum(energy, TINY)) - 1e-6 * interaction_count
    return np.argsort(score)[-MAX_DELAYS:].astype(np.int64)


def fit_predict(
    matrix: np.ndarray,
    target: np.ndarray,
    train_rows: np.ndarray,
    predict_rows: np.ndarray,
    interaction_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = select_delays(
        matrix[train_rows], target[train_rows], interaction_count
    )
    coefficients = ridge_coefficients(
        matrix[train_rows][:, selected], target[train_rows]
    )
    prediction = matrix[predict_rows][:, selected] @ coefficients
    return prediction, selected


def complex_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.complex64)
    truth = np.asarray(truth, dtype=np.complex64)
    metric = v503.channel_metrics(prediction[None], truth[None])
    pred_flat = prediction.reshape(-1).astype(np.complex128)
    truth_flat = truth.reshape(-1).astype(np.complex128)
    error = float(np.sum(np.abs(pred_flat - truth_flat) ** 2))
    truth_power = float(np.sum(np.abs(truth_flat) ** 2))
    nmse = error / max(truth_power, TINY)
    inv_nmse = 1.0 / (1.0 + nmse)
    pdp = float(metric["pdp"][0])
    leaked = float(metric["leaked_full"][0])
    return {
        "pdp": pdp,
        "leaked_full": leaked,
        "complex_coherence": float(metric["complex_coherence"][0]),
        "nmse": nmse,
        "inv_nmse": inv_nmse,
        "traceback_like": 0.4 * pdp + 0.4 * leaked + 0.2 * inv_nmse,
    }


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


def grouped(metrics: dict[str, np.ndarray], regimes: np.ndarray) -> dict[str, object]:
    groups = {
        "overall": {
            key: summarize(value, np.ones(len(regimes), dtype=bool))
            for key, value in metrics.items()
        }
    }
    for code, name in enumerate(REGIME_NAMES):
        mask = regimes == code
        groups[name] = {key: summarize(value, mask) for key, value in metrics.items()}
    return groups


def main() -> None:
    for path in (TRAIN_H, V503_REPORT, V506_REPORT):
        if not path.is_file():
            raise FileNotFoundError(path)
    indices = np.asarray(
        json.loads(V503_REPORT.read_text(encoding="utf-8"))["indices"],
        dtype=np.int64,
    )
    union = load_union(indices)
    truth = np.load(TRAIN_H, mmap_mode="r")
    regimes = v501.regime_codes()[indices]

    metric_names = (
        "crossfit_pdp",
        "crossfit_leaked_full",
        "crossfit_complex_coherence",
        "crossfit_nmse",
        "crossfit_inv_nmse",
        "crossfit_traceback_like",
        "fullfit_pdp",
        "fullfit_leaked_full",
        "fullfit_complex_coherence",
        "fullfit_nmse",
        "fullfit_inv_nmse",
        "fullfit_traceback_like",
        "candidate_delay_count",
        "selected_delay_count_even",
        "selected_delay_count_odd",
        "selected_delay_count_full",
        "selected_mean_depth_full",
    )
    metrics = {name: np.zeros(len(indices), dtype=np.float32) for name in metric_names}
    even = np.arange(0, N_SUBCARRIERS, 2, dtype=np.int64)
    odd = np.arange(1, N_SUBCARRIERS, 2, dtype=np.int64)

    for row, point_id in enumerate(indices):
        valid = union["valid"][row].astype(bool)
        tau, depth = unique_delays(
            union["tau"][row, valid], union["interaction_count"][row, valid]
        )
        channel = np.asarray(truth[point_id], dtype=np.complex64)
        target = channel.reshape(-1, N_SUBCARRIERS).T.astype(np.complex128)
        metrics["candidate_delay_count"][row] = len(tau)
        if len(tau) == 0:
            continue
        matrix = basis(tau)

        odd_prediction, selected_even = fit_predict(
            matrix, target, even, odd, depth
        )
        even_prediction, selected_odd = fit_predict(
            matrix, target, odd, even, depth
        )
        cross_prediction = np.empty_like(target)
        cross_prediction[odd] = odd_prediction
        cross_prediction[even] = even_prediction
        cross_channel = cross_prediction.T.reshape(channel.shape)
        cross = complex_metrics(cross_channel, channel)
        for key, value in cross.items():
            metrics[f"crossfit_{key}"][row] = value

        selected_full = select_delays(matrix, target, depth)
        full_coefficients = ridge_coefficients(matrix[:, selected_full], target)
        full_prediction = matrix[:, selected_full] @ full_coefficients
        full_channel = full_prediction.T.reshape(channel.shape)
        full = complex_metrics(full_channel, channel)
        for key, value in full.items():
            metrics[f"fullfit_{key}"][row] = value

        metrics["selected_delay_count_even"][row] = len(selected_even)
        metrics["selected_delay_count_odd"][row] = len(selected_odd)
        metrics["selected_delay_count_full"][row] = len(selected_full)
        full_energy = np.sum(np.abs(full_coefficients) ** 2, axis=1, dtype=np.float64)
        full_energy /= max(float(np.sum(full_energy)), TINY)
        metrics["selected_mean_depth_full"][row] = float(
            np.sum(full_energy * depth[selected_full])
        )
        if (row + 1) % 20 == 0 or row + 1 == len(indices):
            print(f"v507 complex frequency {row + 1}/{len(indices)}", flush=True)

    groups = grouped(metrics, regimes)
    control_data = np.load(
        PATHS.cache_dir(501)
        / "point_metrics"
        / "physical_d16_1m_spec_refract_6p693.npz"
    )
    control = {
        "pdp": float(np.mean(control_data["pdp"][indices])),
        "leaked_full": float(np.mean(control_data["leaked_full"][indices])),
        "complex_coherence": float(np.mean(control_data["complex_coherence"][indices])),
    }
    heavy = groups["heavy"]
    conclusion = {
        "cross_frequency_delay_dictionary_is_strong": bool(
            heavy["crossfit_pdp"]["mean"] >= 0.90
            and heavy["crossfit_leaked_full"]["mean"] >= 0.80
        ),
        "interpretation": (
            "High even/odd cross-frequency scores mean the union delays generate the official "
            "frequency law without fitting each evaluated subcarrier. The remaining mismatch "
            "then lies in deployable path complex coefficients and spatial/port structure."
        ),
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        POINT_METRICS, indices=indices, regimes=regimes, **metrics
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "The v506 mechanism union contains the official path delays, and the remaining "
            "error is primarily the complex path coefficient field rather than path geometry."
        ),
        "method_details": {
            "candidate_source": "v506 16-configuration path union",
            "delay_dedup_resolution_bins": 0.125,
            "max_selected_delays": MAX_DELAYS,
            "ridge_relative": RIDGE_RELATIVE,
            "crossfit": "fit even subcarriers to predict odd and vice versa",
            "coefficient_freedom": "independent complex coefficient per TX-antenna/UE-port link and path",
        },
        "point_count": int(len(indices)),
        "current_sionna_control": control,
        "metrics": groups,
        "conclusion": conclusion,
        "outputs": {
            "report": PATHS.relative(REPORT),
            "point_metrics": PATHS.relative(POINT_METRICS),
            "submission_zip": None,
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "overall_crossfit": {
                    "pdp": groups["overall"]["crossfit_pdp"]["mean"],
                    "leaked_full": groups["overall"]["crossfit_leaked_full"]["mean"],
                    "coherence": groups["overall"]["crossfit_complex_coherence"]["mean"],
                    "traceback_like": groups["overall"]["crossfit_traceback_like"]["mean"],
                },
                "heavy_crossfit": {
                    "pdp": heavy["crossfit_pdp"]["mean"],
                    "leaked_full": heavy["crossfit_leaked_full"]["mean"],
                    "coherence": heavy["crossfit_complex_coherence"]["mean"],
                    "traceback_like": heavy["crossfit_traceback_like"]["mean"],
                },
                "report": PATHS.relative(REPORT),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
