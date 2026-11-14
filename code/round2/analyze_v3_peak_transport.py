#!/usr/bin/env python3
"""Test deployable peak transport using only official Round 2 observables."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from data import load_positions
from project_paths import PATHS


ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v3_spectral_transport"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v3_spectral_transport"
REPORT = ANALYSIS_DIR / "peak_transport_screen.json"
PREDICTIONS = ARTIFACT_DIR / "peak_transport_oof_predictions.npz"

DESCRIPTORS = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_train_spectral_descriptors.npz"
)
GEOMETRY = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_geometry_visibility_features.npz"
)
OOF_WEIGHTS = (
    PATHS.artifacts
    / "round2"
    / "rbf_spectral_field"
    / "round2_rbf_oof_weights.npz"
)
FOLD_CACHE = (
    PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    family: str
    estimators: int
    min_samples_leaf: int
    max_features: float


CONFIGS = (
    ModelConfig("extra_leaf2", "extra", 480, 2, 1.0),
    ModelConfig("extra_leaf5", "extra", 480, 5, 1.0),
    ModelConfig("extra_leaf10", "extra", 480, 10, 1.0),
    ModelConfig("extra_leaf5_mf07", "extra", 480, 5, 0.7),
    ModelConfig("forest_leaf2", "forest", 360, 2, 0.8),
    ModelConfig("forest_leaf5", "forest", 360, 5, 0.8),
)


def normalized(value: np.ndarray) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = array.sum(axis=1, keepdims=True, dtype=np.float64)
    return np.divide(
        array,
        np.maximum(total, 1e-20),
        out=np.zeros_like(array),
        where=total > 0.0,
    )


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(left * left, axis=1, dtype=np.float64)
        * np.sum(right * right, axis=1, dtype=np.float64)
    )
    return np.divide(
        numerator,
        np.maximum(denominator, 1e-30),
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )


def positional_features(geometry: np.ndarray) -> np.ndarray:
    value = np.asarray(geometry, dtype=np.float32)
    xy = value[:, :2]
    features = [value]
    for scale in (4.0, 8.0, 16.0, 32.0, 64.0, 128.0):
        features.extend([np.sin(xy / scale), np.cos(xy / scale)])
    return np.concatenate(features, axis=1).astype(np.float32)


def circular_targets(index: np.ndarray, bins: int) -> np.ndarray:
    angle = 2.0 * np.pi * np.asarray(index, dtype=np.float64) / bins
    return np.column_stack([np.cos(angle), np.sin(angle)]).astype(np.float32)


def circular_prediction(values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    angle = np.arctan2(values[:, 1], values[:, 0])
    index = np.rint(np.mod(angle, 2.0 * np.pi) * bins / (2.0 * np.pi)).astype(np.int64)
    confidence = np.linalg.norm(values, axis=1)
    return index % bins, confidence.astype(np.float32)


def make_model(config: ModelConfig, seed: int):
    common = dict(
        n_estimators=config.estimators,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        random_state=seed,
        n_jobs=-1,
    )
    if config.family == "extra":
        return ExtraTreesRegressor(**common)
    if config.family == "forest":
        return RandomForestRegressor(**common)
    raise ValueError(config.family)


def predict_v2_global(
    cache: np.ndarray,
    donor_ids: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    donor = np.sqrt(np.maximum(cache[donor_ids], 0.0)).astype(np.float32)
    amplitude = np.maximum(
        np.einsum("qk,qkd->qd", weights, donor, optimize=True), 0.0
    )
    return normalized(amplitude * amplitude)


def shift_zero_padded(values: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    output = np.zeros_like(values)
    for row, shift in enumerate(np.asarray(shifts, dtype=np.int64)):
        if shift > 0:
            output[row, shift:] = values[row, :-shift]
        elif shift < 0:
            output[row, :shift] = values[row, -shift:]
        else:
            output[row] = values[row]
    return output


def shift_pas(values: np.ndarray, h_shift: np.ndarray, v_shift: np.ndarray) -> np.ndarray:
    shaped = values.reshape(-1, 2, 16, 8)
    output = np.empty_like(shaped)
    for row, (horizontal, vertical) in enumerate(zip(h_shift, v_shift)):
        output[row] = np.roll(
            shaped[row], (int(horizontal), int(vertical)), axis=(1, 2)
        )
    return output.reshape(len(values), -1)


def signed_circular_delta(destination: np.ndarray, source: np.ndarray, bins: int) -> np.ndarray:
    return ((destination - source + bins // 2) % bins - bins // 2).astype(np.int64)


def blend_spectrum(base: np.ndarray, moved: np.ndarray, alpha: float) -> np.ndarray:
    amplitude = (1.0 - alpha) * np.sqrt(np.maximum(base, 0.0)) + alpha * np.sqrt(
        np.maximum(moved, 0.0)
    )
    return normalized(amplitude * amplitude)


def summary(values: np.ndarray, fold: np.ndarray, site: np.ndarray) -> dict[str, object]:
    return {
        "mean": float(np.mean(values)),
        "fold": [float(np.mean(values[fold == item])) for item in np.unique(fold)],
        "site": {
            str(int(item)): float(np.mean(values[site == item]))
            for item in np.unique(site)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20_260_805)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    descriptors = np.load(DESCRIPTORS)
    global_pas = descriptors["global_pas"].astype(np.float32)
    global_pdp = descriptors["global_pdp"].astype(np.float32)
    valid = descriptors["valid"].astype(bool)
    geometry_cache = np.load(GEOMETRY)
    feature = positional_features(geometry_cache["train"].astype(np.float32))
    weight_cache = np.load(OOF_WEIGHTS)
    query_ids = weight_cache["query_ids"].astype(np.int64)
    fold_id = weight_cache["fold_id"].astype(np.int8)
    donor_ids = weight_cache["neighbor_ids"].astype(np.int64)
    weights = weight_cache["weights"].astype(np.float32)
    _, positions, _, site, _ = load_positions()
    query_site = site[query_ids]
    folds = np.load(FOLD_CACHE)

    base_pas = predict_v2_global(global_pas, donor_ids, weights)
    base_pdp = predict_v2_global(global_pdp, donor_ids, weights)
    truth_pas = global_pas[query_ids]
    truth_pdp = global_pdp[query_ids]
    base_pas_score = cosine(base_pas, truth_pas)
    base_pdp_score = cosine(base_pdp, truth_pdp)

    pas_shaped = global_pas.reshape(-1, 2, 16, 8)
    delay_peak = np.argmax(global_pdp, axis=1).astype(np.int64)
    horizontal_peak = np.argmax(pas_shaped.sum(axis=(1, 3)), axis=1).astype(np.int64)
    vertical_peak = np.argmax(pas_shaped.sum(axis=(1, 2)), axis=1).astype(np.int64)
    targets = np.column_stack(
        [
            delay_peak.astype(np.float32),
            circular_targets(horizontal_peak, 16),
            circular_targets(vertical_peak, 8),
        ]
    ).astype(np.float32)

    base_delay_peak = np.argmax(base_pdp, axis=1).astype(np.int64)
    base_pas_shaped = base_pas.reshape(-1, 2, 16, 8)
    base_h_peak = np.argmax(base_pas_shaped.sum(axis=(1, 3)), axis=1).astype(np.int64)
    base_v_peak = np.argmax(base_pas_shaped.sum(axis=(1, 2)), axis=1).astype(np.int64)

    results: list[dict[str, object]] = []
    saved: dict[str, np.ndarray] = {
        "query_ids": query_ids.astype(np.int32),
        "fold_id": fold_id,
        "base_pas": base_pas.astype(np.float16),
        "base_pdp": base_pdp.astype(np.float16),
    }
    alphas = (0.25, 0.50, 0.75, 1.00)
    max_delay_values = (1, 2, 4)
    max_h_values = (1, 2, 4)
    max_v_values = (1, 2)

    for config_index, config in enumerate(CONFIGS):
        prediction = np.zeros((len(query_ids), targets.shape[1]), dtype=np.float32)
        for held_fold in np.unique(fold_id):
            rows = np.flatnonzero(fold_id == held_fold)
            fit_ids = folds[f"fold{int(held_fold)}_fit"].astype(np.int64)
            for site_value in np.unique(query_site[rows]):
                selected_rows = rows[query_site[rows] == site_value]
                fit = fit_ids[(site[fit_ids] == site_value) & valid[fit_ids]]
                model = make_model(
                    config,
                    args.seed + config_index * 1009 + int(held_fold) * 97 + int(site_value),
                )
                model.fit(feature[fit], targets[fit])
                prediction[selected_rows] = model.predict(feature[query_ids[selected_rows]])

        predicted_delay = np.rint(prediction[:, 0]).astype(np.int64)
        predicted_delay = np.clip(predicted_delay, 0, global_pdp.shape[1] - 1)
        predicted_h, confidence_h = circular_prediction(prediction[:, 1:3], 16)
        predicted_v, confidence_v = circular_prediction(prediction[:, 3:5], 8)
        raw_delay_shift = predicted_delay - base_delay_peak
        raw_h_shift = signed_circular_delta(predicted_h, base_h_peak, 16)
        raw_v_shift = signed_circular_delta(predicted_v, base_v_peak, 8)

        config_rows: list[dict[str, object]] = []
        for max_delay in max_delay_values:
            delay_shift = np.clip(raw_delay_shift, -max_delay, max_delay)
            moved_pdp = shift_zero_padded(base_pdp, delay_shift)
            for max_h in max_h_values:
                h_shift = np.clip(raw_h_shift, -max_h, max_h)
                for max_v in max_v_values:
                    v_shift = np.clip(raw_v_shift, -max_v, max_v)
                    moved_pas = shift_pas(base_pas, h_shift, v_shift)
                    for alpha in alphas:
                        pas_score = cosine(
                            blend_spectrum(base_pas, moved_pas, alpha), truth_pas
                        )
                        pdp_score = cosine(
                            blend_spectrum(base_pdp, moved_pdp, alpha), truth_pdp
                        )
                        combined = 0.5 * (pas_score + pdp_score)
                        config_rows.append(
                            {
                                "alpha": alpha,
                                "max_delay_shift": max_delay,
                                "max_h_shift": max_h,
                                "max_v_shift": max_v,
                                "pas": summary(pas_score, fold_id, query_site),
                                "pdp": summary(pdp_score, fold_id, query_site),
                                "combined": summary(combined, fold_id, query_site),
                            }
                        )
        best = max(config_rows, key=lambda row: float(row["combined"]["mean"]))
        row = {
            "config": asdict(config),
            "delay_peak_accuracy": float(
                np.mean(predicted_delay == delay_peak[query_ids])
            ),
            "horizontal_peak_accuracy": float(
                np.mean(predicted_h == horizontal_peak[query_ids])
            ),
            "vertical_peak_accuracy": float(
                np.mean(predicted_v == vertical_peak[query_ids])
            ),
            "horizontal_confidence_mean": float(np.mean(confidence_h)),
            "vertical_confidence_mean": float(np.mean(confidence_v)),
            "best": best,
            "delta_vs_v2": {
                "pas": float(best["pas"]["mean"] - np.mean(base_pas_score)),
                "pdp": float(best["pdp"]["mean"] - np.mean(base_pdp_score)),
                "combined": float(
                    best["combined"]["mean"]
                    - 0.5 * (np.mean(base_pas_score) + np.mean(base_pdp_score))
                ),
            },
        }
        results.append(row)
        saved[f"{config.name}_prediction"] = prediction
        print(
            f"{config.name}: peak_acc={row['delay_peak_accuracy']:.3f}/"
            f"{row['horizontal_peak_accuracy']:.3f}/{row['vertical_peak_accuracy']:.3f} "
            f"best PAS={best['pas']['mean']:.6f} PDP={best['pdp']['mean']:.6f} "
            f"delta={row['delta_vs_v2']['combined']:+.6f}",
            flush=True,
        )

    ranking = sorted(
        results, key=lambda row: float(row["best"]["combined"]["mean"]), reverse=True
    )
    payload = {
        "experiment": "Round2-E4-V3-peak-transport-screen",
        "status": "analysis_only",
        "sionna_used": False,
        "official_inputs_only": True,
        "validation": "fixed 500-point five-fold spatial holes",
        "baseline": {
            "pas": summary(base_pas_score, fold_id, query_site),
            "pdp": summary(base_pdp_score, fold_id, query_site),
            "combined": summary(
                0.5 * (base_pas_score + base_pdp_score), fold_id, query_site
            ),
        },
        "results": results,
        "ranking": [
            {
                "name": row["config"]["name"],
                "best": row["best"],
                "delta_vs_v2": row["delta_vs_v2"],
            }
            for row in ranking
        ],
        "seconds": float(time.time() - started),
    }
    np.savez_compressed(PREDICTIONS, **saved)
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["ranking"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
