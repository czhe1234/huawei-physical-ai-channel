#!/usr/bin/env python3
"""Screen local spectral-flow transport on the Round 2 spatial holes."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from analyze_v3_spectral_interpolators import (
    KernelConfig,
    cosine,
    predict_spectrum,
    solve_weights,
)
from data import load_positions
from project_paths import PATHS


ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v3_spectral_transport"
REPORT = ANALYSIS_DIR / "local_spectral_flow_screen.json"

PRE_DIR = PATHS.artifacts / "round2" / "pre_sionna"
DESCRIPTORS = PRE_DIR / "round2_train_spectral_descriptors.npz"
OOF_WEIGHTS = (
    PATHS.artifacts
    / "round2"
    / "rbf_spectral_field"
    / "round2_rbf_oof_weights.npz"
)


@dataclass(frozen=True)
class FlowConfig:
    name: str
    fit_neighbors: int
    distance_power: float
    ridge: float


CONFIGS = tuple(
    FlowConfig(f"k{k}_p{power:g}_r{ridge:g}", k, power, ridge)
    for k in (8, 16, 32, 64)
    for power in (0.0, 1.0, 2.0)
    for ridge in (0.1, 1.0, 10.0)
)
ALPHAS = (0.25, 0.50, 0.75, 1.0)
RBF = KernelConfig("flow_rbf", "negative_linear", 64, 10.0)


def normalized(value: np.ndarray) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = array.sum(axis=1, keepdims=True, dtype=np.float64)
    return np.divide(
        array,
        np.maximum(total, 1e-20),
        out=np.zeros_like(array),
        where=total > 0.0,
    )


def circular(index: np.ndarray, bins: int) -> np.ndarray:
    angle = 2.0 * np.pi * np.asarray(index, dtype=np.float64) / bins
    return np.column_stack([np.cos(angle), np.sin(angle)]).astype(np.float64)


def circular_index(value: np.ndarray, bins: int) -> np.ndarray:
    angle = np.mod(np.arctan2(value[..., 1], value[..., 0]), 2.0 * np.pi)
    return np.rint(angle * bins / (2.0 * np.pi)).astype(np.int64) % bins


def signed_delta(destination: np.ndarray, source: np.ndarray, bins: int) -> np.ndarray:
    return np.asarray(
        (np.asarray(destination) - np.asarray(source) + bins // 2) % bins
        - bins // 2,
        dtype=np.int64,
    )


def fit_local(
    donor_xy: np.ndarray,
    query_xy: np.ndarray,
    target: np.ndarray,
    distance_power: float,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    offset = np.asarray(donor_xy, dtype=np.float64) - np.asarray(query_xy)[None, :]
    scale = max(float(np.median(np.linalg.norm(offset, axis=1))), 1.0)
    design = np.column_stack(
        [np.ones(len(offset)), offset[:, 0] / scale, offset[:, 1] / scale]
    )
    distance = np.linalg.norm(offset, axis=1)
    weight = 1.0 / np.maximum(distance, 0.5) ** distance_power
    gram = design.T @ (weight[:, None] * design)
    gram += np.diag([1e-8, ridge, ridge])
    right = design.T @ (weight[:, None] * np.asarray(target, dtype=np.float64))
    try:
        coefficients = np.linalg.solve(gram, right)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(gram, right, rcond=1e-10)[0]
    return coefficients, design


def shift_rows_zero(values: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    output = np.zeros_like(values)
    for row, shift in enumerate(np.asarray(shifts, dtype=np.int64)):
        if shift > 0:
            output[row, shift:] = values[row, :-shift]
        elif shift < 0:
            output[row, :shift] = values[row, -shift:]
        else:
            output[row] = values[row]
    return output


def shift_pas_rows(
    values: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
) -> np.ndarray:
    shaped = values.reshape(-1, 2, 16, 8)
    output = np.empty_like(shaped)
    for row, (h_shift, v_shift) in enumerate(zip(horizontal, vertical)):
        output[row] = np.roll(
            shaped[row], (int(h_shift), int(v_shift)), axis=(1, 2)
        )
    return output.reshape(len(values), -1)


def interpolate_shifted(
    donor: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    amplitude = np.maximum(
        np.tensordot(weights.astype(np.float32), np.sqrt(np.maximum(donor, 0.0)), axes=(0, 0)),
        0.0,
    )
    return normalized((amplitude * amplitude)[None, :])[0]


def blend(base: np.ndarray, moved: np.ndarray, alpha: float) -> np.ndarray:
    amplitude = (1.0 - alpha) * np.sqrt(np.maximum(base, 0.0)) + alpha * np.sqrt(
        np.maximum(moved, 0.0)
    )
    return normalized(amplitude * amplitude)


def summarize(values: np.ndarray, fold: np.ndarray, site: np.ndarray) -> dict[str, object]:
    return {
        "mean": float(np.mean(values)),
        "fold": [float(np.mean(values[fold == item])) for item in np.unique(fold)],
        "site": {
            str(int(item)): float(np.mean(values[site == item]))
            for item in np.unique(site)
        },
    }


def main() -> None:
    started = time.time()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    descriptors = np.load(DESCRIPTORS)
    global_pas = descriptors["global_pas"].astype(np.float32)
    global_pdp = descriptors["global_pdp"].astype(np.float32)
    weights_cache = np.load(OOF_WEIGHTS)
    query_ids = weights_cache["query_ids"].astype(np.int64)
    fold_id = weights_cache["fold_id"].astype(np.int8)
    donor_ids = weights_cache["neighbor_ids"].astype(np.int64)
    rbf_weights = weights_cache["weights"].astype(np.float32)
    _, positions, _, train_site, _ = load_positions()
    query_site = train_site[query_ids]

    pas_shaped = global_pas.reshape(-1, 2, 16, 8)
    delay_peak = np.argmax(global_pdp, axis=1).astype(np.int64)
    horizontal_peak = np.argmax(pas_shaped.sum(axis=(1, 3)), axis=1).astype(np.int64)
    vertical_peak = np.argmax(pas_shaped.sum(axis=(1, 2)), axis=1).astype(np.int64)
    peak_target = np.column_stack(
        [
            delay_peak,
            circular(horizontal_peak, 16),
            circular(vertical_peak, 8),
        ]
    )
    base_pas = predict_spectrum(global_pas, donor_ids, rbf_weights)
    base_pdp = predict_spectrum(global_pdp, donor_ids, rbf_weights)
    truth_pas = global_pas[query_ids]
    truth_pdp = global_pdp[query_ids]
    base_pas_score = cosine(base_pas, truth_pas)
    base_pdp_score = cosine(base_pdp, truth_pdp)
    baseline = 0.5 * (base_pas_score + base_pdp_score)

    rows: list[dict[str, object]] = []
    for config in CONFIGS:
        transported_pas = np.empty_like(base_pas)
        transported_pdp = np.empty_like(base_pdp)
        post_pas = np.empty_like(base_pas)
        post_pdp = np.empty_like(base_pdp)
        for row, query_id in enumerate(query_ids):
            fit_ids = donor_ids[row, : config.fit_neighbors]
            coefficients, design = fit_local(
                positions[fit_ids, :2],
                positions[query_id, :2],
                peak_target[fit_ids],
                config.distance_power,
                config.ridge,
            )
            fitted = design @ coefficients
            query_value = coefficients[0]

            predicted_delay = int(np.rint(query_value[0]))
            donor_delay = np.rint(fitted[:, 0]).astype(np.int64)
            delay_shift = np.clip(predicted_delay - donor_delay, -3, 3)
            moved_pdp = shift_rows_zero(global_pdp[fit_ids], delay_shift)

            predicted_h = int(circular_index(query_value[1:3][None, :], 16)[0])
            donor_h = circular_index(fitted[:, 1:3], 16)
            h_shift = np.clip(signed_delta(predicted_h, donor_h, 16), -2, 2)
            predicted_v = int(circular_index(query_value[3:5][None, :], 8)[0])
            donor_v = circular_index(fitted[:, 3:5], 8)
            v_shift = np.clip(signed_delta(predicted_v, donor_v, 8), -1, 1)
            moved_pas = shift_pas_rows(global_pas[fit_ids], h_shift, v_shift)

            local_weights = solve_weights(
                positions[fit_ids, :2], positions[query_id, :2], RBF
            )
            transported_pdp[row] = interpolate_shifted(moved_pdp, local_weights)
            transported_pas[row] = interpolate_shifted(moved_pas, local_weights)

            base_delay = int(np.argmax(base_pdp[row]))
            post_pdp[row] = shift_rows_zero(
                base_pdp[row : row + 1],
                np.asarray([np.clip(predicted_delay - base_delay, -2, 2)]),
            )[0]
            base_shape = base_pas[row].reshape(2, 16, 8)
            base_h = int(np.argmax(base_shape.sum(axis=(0, 2))))
            base_v = int(np.argmax(base_shape.sum(axis=(0, 1))))
            post_pas[row] = shift_pas_rows(
                base_pas[row : row + 1],
                np.asarray([np.clip(signed_delta(predicted_h, base_h, 16), -2, 2)]),
                np.asarray([np.clip(signed_delta(predicted_v, base_v, 8), -1, 1)]),
            )[0]

        variants = {"gradient_transport": (transported_pas, transported_pdp), "post_shift": (post_pas, post_pdp)}
        action_rows: list[dict[str, object]] = []
        for variant, (candidate_pas, candidate_pdp) in variants.items():
            for alpha in ALPHAS:
                pas_score = cosine(blend(base_pas, candidate_pas, alpha), truth_pas)
                pdp_score = cosine(blend(base_pdp, candidate_pdp, alpha), truth_pdp)
                combined = 0.5 * (pas_score + pdp_score)
                action_rows.append(
                    {
                        "variant": variant,
                        "alpha": alpha,
                        "pas": summarize(pas_score, fold_id, query_site),
                        "pdp": summarize(pdp_score, fold_id, query_site),
                        "combined": summarize(combined, fold_id, query_site),
                        "delta_fold": [
                            float(np.mean((combined - baseline)[fold_id == item]))
                            for item in np.unique(fold_id)
                        ],
                    }
                )
        best = max(action_rows, key=lambda value: float(value["combined"]["mean"]))
        result = {
            "config": asdict(config),
            "best": best,
            "delta_vs_v2": float(best["combined"]["mean"] - np.mean(baseline)),
        }
        rows.append(result)
        print(
            f"{config.name}: {best['variant']} a={best['alpha']:.2f} "
            f"PAS={best['pas']['mean']:.6f} PDP={best['pdp']['mean']:.6f} "
            f"delta={result['delta_vs_v2']:+.6f}",
            flush=True,
        )

    ranking = sorted(rows, key=lambda value: float(value["best"]["combined"]["mean"]), reverse=True)
    payload = {
        "experiment": "Round2-E4-V3-local-spectral-flow-screen",
        "status": "analysis_only",
        "sionna_used": False,
        "official_inputs_only": True,
        "validation": "fixed 500-point five-fold spatial holes",
        "baseline": {
            "pas": summarize(base_pas_score, fold_id, query_site),
            "pdp": summarize(base_pdp_score, fold_id, query_site),
            "combined": summarize(baseline, fold_id, query_site),
        },
        "results": rows,
        "ranking": ranking,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(ranking[:8], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
