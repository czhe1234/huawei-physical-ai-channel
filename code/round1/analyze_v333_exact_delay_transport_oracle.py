#!/usr/bin/env python3
"""Measure the full-H ceiling with oracle path existence and exact path delay."""

from __future__ import annotations

import json

import numpy as np

import analyze_v297_continuous_delay_component_transport as v297
import analyze_v305_pointcloud_path_identity_match as v305
import analyze_v306_matched_patch_energy_transformer as v306
import analyze_v309_unmatched_effective_scatterers as v309
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v325_path_phase_transport_identifiability as v325
import analyze_v326_complex_path_subspace_transport as v326
import analyze_v328_patch_existence_support_field as v328
import analyze_v329_highprecision_complex_path_transport as v329
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 333
VERSION = "v333_exact_delay_transport_oracle"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"


def exact_delay_labels(
    candidate_shape: tuple[int, int], component_delay: np.ndarray
) -> np.ndarray:
    output = np.full(candidate_shape, np.nan, dtype=np.float32)
    match = np.load(v305.MATCH_CACHE)
    good = (
        (match["length_error_bins"] <= v309.GOOD_LENGTH_ERROR)
        & (match["h_error_bins"] <= v309.GOOD_H_ERROR)
    )
    for point_id, component_id, candidate_id, mass in zip(
        match["point_id"][good],
        match["component_id"][good],
        match["candidate_id"][good],
        match["mass"][good],
    ):
        point_id = int(point_id)
        candidate_id = int(candidate_id)
        current = output[point_id, candidate_id]
        if not np.isfinite(current):
            output[point_id, candidate_id] = component_delay[point_id, int(component_id)]
    return output


def project_one(
    base: np.ndarray,
    query_xy: np.ndarray,
    candidate: dict[str, np.ndarray],
    query_row: int,
    bank: v329.SelectivePatchBank,
    beta: float,
    exact_delay: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    selected = bank.candidates(query_xy, candidate, query_row)
    selected = [item for item in selected if np.isfinite(exact_delay[query_row, item[1]])]
    if not selected:
        return base.copy(), {"path_count": 0.0, "distance_mean": 0.0}
    candidate_ids = np.asarray([item[1] for item in selected], dtype=np.int64)
    bank_rows = np.asarray([item[2] for item in selected], dtype=np.int64)
    distance = np.asarray([item[3] for item in selected], dtype=np.float32)
    delay = exact_delay[query_row, candidate_ids].astype(np.float64)
    path_length = candidate["path_length"][query_row, candidate_ids].astype(np.float64)
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
    }


def corrected_predictions(
    base: np.ndarray,
    query_pos: np.ndarray,
    candidate: dict[str, np.ndarray],
    bank: v329.SelectivePatchBank,
    beta: float,
    exact_delay: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    output = np.empty_like(base)
    path_count = np.zeros(len(base), dtype=np.float32)
    distance = np.zeros(len(base), dtype=np.float32)
    for row in range(len(base)):
        output[row], diagnostics = project_one(
            base[row], query_pos[row, :2], candidate, row, bank, beta, exact_delay
        )
        path_count[row] = diagnostics["path_count"]
        distance[row] = diagnostics["distance_mean"]
    active = path_count > 0
    return output, {
        "path_count": summarize(path_count),
        "active_fraction": float(np.mean(active)),
        "distance_mean_active": summarize(distance[active]),
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
    exact_delay = exact_delay_labels(candidate_valid.shape, component["delay"])
    fold_rows = []

    for fold, query_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, query_ids)
        threshold = float(v329_report["fold_rows"][fold]["threshold"])
        beta = float(v329_report["fold_rows"][fold]["outer_beta"])
        selector, truth_label = v329.selector_matrix(
            candidate,
            candidate_valid,
            label,
            fit_ids,
            query_ids,
            train_pos[:, :2],
        )
        true_positive = (selector >= threshold) & (truth_label > 0.0)
        binary_selector = true_positive.astype(np.float32)
        base = v326.base_predictions(train_pos[query_ids], fit_ids, train_pos, channel)
        bank = v329.SelectivePatchBank(
            rows,
            gain,
            fit_ids,
            train_pos[:, :2],
            selector=binary_selector,
            threshold=0.5,
        )
        corrected, diagnostics = corrected_predictions(
            base,
            train_pos[query_ids],
            v326.slice_candidates(candidate, query_ids),
            bank,
            beta,
            exact_delay[query_ids],
        )
        truth = np.asarray(channel[query_ids])
        base_metrics = v326.score(base, truth)
        metrics = v326.score(corrected, truth)
        fold_rows.append(
            {
                "fold": fold,
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
            f"v333 fold={fold} active={diagnostics['active_fraction']:.3f} "
            f"score={fold_rows[-1]['score_delta']:+.6f}",
            flush=True,
        )

    summary = {
        key: float(np.mean([row[key] for row in fold_rows]))
        for key in ("score_delta", "pas_delta", "pdp_delta", "inv_nmse_delta")
    }
    summary["fold_wins"] = int(sum(row["score_delta"] > 0.0 for row in fold_rows))
    passed = summary["score_delta"] >= 0.01 and summary["fold_wins"] >= 4
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "diagnostic_truth_policy": (
            "Held-out path existence and exact continuous delay are used only for an oracle "
            "ceiling after v329 thresholds are frozen; no test prediction is produced."
        ),
        "fold_rows": fold_rows,
        "summary": summary,
        "decision_rule": {
            "score_delta": ">=0.01",
            "fold_wins": ">=4/5",
            "passed": bool(passed),
        },
        "outputs": {"report": PATHS.relative(REPORT), "zip": None},
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "decision": report["decision_rule"]}, indent=2))


if __name__ == "__main__":
    main()
