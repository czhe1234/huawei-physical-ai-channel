#!/usr/bin/env python3
"""Transport path phase at base-supported continuous delays instead of geometry delays."""

from __future__ import annotations

import json

import numpy as np

import analyze_v297_continuous_delay_component_transport as v297
import analyze_v304_inverse_scatterer_identifiability as v304
import analyze_v305_pointcloud_path_identity_match as v305
import analyze_v306_matched_patch_energy_transformer as v306
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v325_path_phase_transport_identifiability as v325
import analyze_v326_complex_path_subspace_transport as v326
import analyze_v328_patch_existence_support_field as v328
import analyze_v329_highprecision_complex_path_transport as v329
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 331
VERSION = "v331_base_supported_delay_phase_transport"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"

DELAY_OFFSETS = np.arange(-1.0, 1.0001, 0.125, dtype=np.float64)


def refine_delays(base: np.ndarray, geometry_delay: np.ndarray) -> np.ndarray:
    frequency = np.arange(v297.N_DELAY, dtype=np.float64)
    h = base.reshape(-1, v297.N_DELAY)
    refined = np.empty_like(geometry_delay, dtype=np.float64)
    used: list[float] = []
    for row, center in enumerate(geometry_delay):
        trial = center + DELAY_OFFSETS
        steering = np.exp(
            -2j * np.pi * frequency[:, None] * trial[None, :] / v297.N_DELAY
        ).astype(np.complex64)
        gain = steering.conj().T @ h.T / v297.N_DELAY
        energy = np.sum(np.abs(gain) ** 2, axis=1, dtype=np.float64)
        order = np.argsort(energy)[::-1]
        selected = float(trial[int(order[0])])
        for candidate in order:
            value = float(trial[int(candidate)])
            if all(abs(value - previous) >= 0.25 for previous in used):
                selected = value
                break
        refined[row] = selected
        used.append(selected)
    return refined


def project_one(
    base: np.ndarray,
    query_xy: np.ndarray,
    candidate: dict[str, np.ndarray],
    query_row: int,
    bank: v329.SelectivePatchBank,
    beta: float,
) -> tuple[np.ndarray, dict[str, float]]:
    selected = bank.candidates(query_xy, candidate, query_row)
    if not selected:
        return base.copy(), {
            "path_count": 0.0,
            "distance_mean": 0.0,
            "delay_shift_abs_mean": 0.0,
        }
    candidate_ids = np.asarray([item[1] for item in selected], dtype=np.int64)
    bank_rows = np.asarray([item[2] for item in selected], dtype=np.int64)
    distance = np.asarray([item[3] for item in selected], dtype=np.float32)
    path_length = candidate["path_length"][query_row, candidate_ids].astype(np.float64)
    calibration = json.loads(v304.REPORT.read_text(encoding="utf-8"))["los_calibration"]
    geometry_delay = (
        float(calibration["intercept_bins"])
        + float(calibration["slope_bins_per_meter"]) * path_length
    )
    delay = refine_delays(base, geometry_delay)
    frequency = np.arange(v297.N_DELAY, dtype=np.float64)
    steering = np.exp(
        -2j * np.pi * frequency[:, None] * delay[None, :] / v297.N_DELAY
    ).astype(np.complex64)
    h = base.reshape(-1, v297.N_DELAY)
    gram = steering.conj().T @ steering
    regularized = gram + np.eye(len(selected), dtype=np.complex64) * (
        v326.RIDGE * np.trace(gram).real / max(len(selected), 1)
    )
    base_gain = np.linalg.solve(regularized, steering.conj().T @ h.T)

    transported = bank.gain[bank_rows].reshape(len(selected), -1).copy()
    delta_length = path_length - bank.rows["length"][bank_rows]
    transported *= np.exp(-1j * beta * delta_length)[:, None].astype(np.complex64)
    reflection = candidate["reflection_point"][query_row, candidate_ids]
    query_h = v305.expected_h_bin(reflection).astype(np.float32)
    delta_h = np.remainder(
        query_h - bank.rows["h_center"][bank_rows] + 8.0, 16.0
    ) - 8.0
    h_coordinate = np.arange(16, dtype=np.float32) - 7.5
    h_phase = np.exp(
        1j * 2.0 * np.pi * delta_h[:, None] * h_coordinate[None, :] / 16.0
    ).astype(np.complex64)
    transported = transported.reshape(len(selected), 2, 16, 8, 4)
    transported *= h_phase[:, None, :, None, None]
    transported = transported.reshape(len(selected), -1)
    base_norm = np.linalg.norm(base_gain, axis=1)
    transported_norm = np.linalg.norm(transported, axis=1)
    transported *= (
        base_norm / np.maximum(transported_norm, 1e-20)
    )[:, None].astype(np.float32)
    correction = (steering @ (transported - base_gain)).T.reshape(base.shape)
    output = (base + correction).astype(np.complex64)
    base_power = float(np.sum(np.abs(base) ** 2, dtype=np.float64))
    output_power = float(np.sum(np.abs(output) ** 2, dtype=np.float64))
    if base_power > 0.0 and output_power > 0.0:
        output *= np.float32(np.sqrt(base_power / output_power))
    return output, {
        "path_count": float(len(selected)),
        "distance_mean": float(np.mean(distance)),
        "delay_shift_abs_mean": float(np.mean(np.abs(delay - geometry_delay))),
    }


def corrected_predictions(
    base: np.ndarray,
    query_pos: np.ndarray,
    candidate: dict[str, np.ndarray],
    bank: v329.SelectivePatchBank,
    beta: float,
) -> tuple[np.ndarray, dict[str, object]]:
    output = np.empty_like(base)
    path_count = np.zeros(len(base), dtype=np.float32)
    distance = np.zeros(len(base), dtype=np.float32)
    delay_shift = np.zeros(len(base), dtype=np.float32)
    for row in range(len(base)):
        output[row], diagnostics = project_one(
            base[row], query_pos[row, :2], candidate, row, bank, beta
        )
        path_count[row] = diagnostics["path_count"]
        distance[row] = diagnostics["distance_mean"]
        delay_shift[row] = diagnostics["delay_shift_abs_mean"]
    active = path_count > 0
    return output, {
        "path_count": summarize(path_count),
        "active_fraction": float(np.mean(active)),
        "distance_mean_active": summarize(distance[active]),
        "delay_shift_abs_active": summarize(delay_shift[active]),
    }


def main() -> None:
    v329_report = json.loads(v329.REPORT.read_text(encoding="utf-8"))
    gain_data = v325.recover_center_gains()
    rows = v325.matched_rows(gain_data)
    gain = v326.recover_matched_gain_tensors(rows)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    channel = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    component = np.load(v297.COMPONENT_CACHE)
    valid_ids = np.flatnonzero(component["valid"].any(axis=1)).astype(np.int64)
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])
    data = np.load(v306.TRAIN_CANDIDATES)
    candidate = {key: data[key] for key in data.files}
    candidate_valid, label = v328.build_labels(candidate)
    fold_rows = []

    for fold, query_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, query_ids)
        threshold = float(v329_report["fold_rows"][fold]["threshold"])
        beta = float(v329_report["fold_rows"][fold]["outer_beta"])
        selector, _truth = v329.selector_matrix(
            candidate,
            candidate_valid,
            label,
            fit_ids,
            query_ids,
            train_pos[:, :2],
        )
        base = v326.base_predictions(train_pos[query_ids], fit_ids, train_pos, channel)
        bank = v329.SelectivePatchBank(
            rows,
            gain,
            fit_ids,
            train_pos[:, :2],
            selector=selector,
            threshold=threshold,
        )
        corrected, diagnostics = corrected_predictions(
            base,
            train_pos[query_ids],
            v326.slice_candidates(candidate, query_ids),
            bank,
            beta,
        )
        truth = np.asarray(channel[query_ids])
        base_metrics = v326.score(base, truth)
        metrics = v326.score(corrected, truth)
        fold_rows.append(
            {
                "fold": fold,
                "threshold": threshold,
                "beta": beta,
                "diagnostics": diagnostics,
                "base": base_metrics,
                "prediction": metrics,
                "score_delta": metrics["score"] - base_metrics["score"],
                "pas_delta": metrics["pas"] - base_metrics["pas"],
                "pdp_delta": metrics["pdp"] - base_metrics["pdp"],
                "inv_nmse_delta": metrics["inv_nmse"] - base_metrics["inv_nmse"],
            }
        )
        print(
            f"v331 fold={fold} active={diagnostics['active_fraction']:.3f} "
            f"delay_shift={diagnostics['delay_shift_abs_active']['mean']:.3f} "
            f"score={fold_rows[-1]['score_delta']:+.6f}",
            flush=True,
        )

    summary = {
        key: float(np.mean([row[key] for row in fold_rows]))
        for key in ("score_delta", "pas_delta", "pdp_delta", "inv_nmse_delta")
    }
    summary["fold_wins"] = int(sum(row["score_delta"] > 0.0 for row in fold_rows))
    passed = (
        summary["score_delta"] >= v326.MIN_SCORE_DELTA
        and summary["fold_wins"] >= 4
        and summary["pas_delta"] >= -v326.MAX_SPECTRAL_DEGRADATION
        and summary["pdp_delta"] >= -v326.MAX_SPECTRAL_DEGRADATION
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Geometry should identify the path family while the current full-H base supplies "
            "the precise continuous delay used for phase-only tensor replacement."
        ),
        "parameters": {
            "delay_search_offsets_bins": [
                float(DELAY_OFFSETS[0]),
                float(DELAY_OFFSETS[-1]),
                0.125,
            ],
            "selector_thresholds_and_beta": "frozen from v329 outer folds",
            "alpha": 1.0,
        },
        "fold_rows": fold_rows,
        "summary": summary,
        "decision_rule": {
            "score_delta": f">={v326.MIN_SCORE_DELTA}",
            "fold_wins": ">=4/5",
            "pas_delta": f">=-{v326.MAX_SPECTRAL_DEGRADATION}",
            "pdp_delta": f">=-{v326.MAX_SPECTRAL_DEGRADATION}",
            "passed": bool(passed),
        },
        "outputs": {"report": PATHS.relative(REPORT), "zip": None},
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "decision": report["decision_rule"]}, indent=2))


if __name__ == "__main__":
    main()
