#!/usr/bin/env python3
"""Screen low-rank linkwise PAS/PDP targets for Round 2 V3."""

from __future__ import annotations

import json
import time

import numpy as np

from data import load_positions
from project_paths import PATHS


ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v3_spectral_transport"
REPORT = ANALYSIS_DIR / "linkwise_shrinkage_screen.json"

PRE_DIR = PATHS.artifacts / "round2" / "pre_sionna"
TRAIN_PAS = PRE_DIR / "round2_train_pas_link_support_f16.npy"
TRAIN_PDP = PRE_DIR / "round2_train_pdp_link_support_f16.npy"
OOF_WEIGHTS = (
    PATHS.artifacts
    / "round2"
    / "rbf_spectral_field"
    / "round2_rbf_oof_weights.npz"
)

ALPHAS = (0.0, 0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.0)


def normalize(value: np.ndarray, axis: int) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = array.sum(axis=axis, keepdims=True, dtype=np.float64)
    return np.divide(
        array,
        np.maximum(total, 1e-20),
        out=np.zeros_like(array),
        where=total > 0.0,
    )


def spectral_target(
    cache: np.ndarray,
    donor_ids: np.ndarray,
    weights: np.ndarray,
    axis: int,
) -> np.ndarray:
    donor = np.sqrt(np.maximum(np.asarray(cache[donor_ids], dtype=np.float32), 0.0))
    amplitude = np.maximum(
        np.tensordot(weights.astype(np.float32), donor, axes=(0, 0)), 0.0
    )
    return normalize(amplitude * amplitude, axis)


def blend(base: np.ndarray, target: np.ndarray, alpha: float, axis: int) -> np.ndarray:
    amplitude = (1.0 - alpha) * np.sqrt(np.maximum(base, 0.0)) + alpha * np.sqrt(
        np.maximum(target, 0.0)
    )
    return normalize(amplitude * amplitude, axis)


def pas_variants(value: np.ndarray) -> dict[str, np.ndarray]:
    # PAS has shape [M, N, S] and is normalized over M for every N,S link.
    return {
        "link": value,
        "ue_shared_frequency": np.broadcast_to(
            value.mean(axis=2, keepdims=True, dtype=np.float64).astype(np.float32),
            value.shape,
        ),
        "frequency_shared_ue": np.broadcast_to(
            value.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32),
            value.shape,
        ),
        "global": np.broadcast_to(
            value.mean(axis=(1, 2), keepdims=True, dtype=np.float64).astype(np.float32),
            value.shape,
        ),
    }


def pdp_variants(value: np.ndarray) -> dict[str, np.ndarray]:
    # PDP has shape [M, N, S] and is normalized over S for every M,N link.
    shaped = value.reshape(2, 16, 8, 4, 192)
    polarization = shaped.mean(axis=(1, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    return {
        "link": value,
        "bs_shared": np.broadcast_to(
            value.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32),
            value.shape,
        ),
        "ue_shared": np.broadcast_to(
            value.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32),
            value.shape,
        ),
        "global": np.broadcast_to(
            value.mean(axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32),
            value.shape,
        ),
        "within_polarization_shared": np.broadcast_to(
            polarization, shaped.shape
        ).reshape(value.shape),
    }


def cosine_vector(prediction: np.ndarray, truth: np.ndarray, axis: int) -> float:
    numerator = np.sum(prediction * truth, axis=axis, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(prediction * prediction, axis=axis, dtype=np.float64)
        * np.sum(truth * truth, axis=axis, dtype=np.float64)
    )
    values = np.divide(
        numerator,
        np.maximum(denominator, 1e-30),
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    return float(np.mean(values, dtype=np.float64))


def grouped_summary(
    values: np.ndarray,
    fold_id: np.ndarray,
    site: np.ndarray,
) -> dict[str, object]:
    return {
        "mean": float(np.mean(values)),
        "fold": [float(np.mean(values[fold_id == fold])) for fold in np.unique(fold_id)],
        "site": {
            str(int(value)): float(np.mean(values[site == value]))
            for value in np.unique(site)
        },
    }


def main() -> None:
    started = time.time()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    weights_cache = np.load(OOF_WEIGHTS)
    query_ids = weights_cache["query_ids"].astype(np.int64)
    fold_id = weights_cache["fold_id"].astype(np.int8)
    donor_ids = weights_cache["neighbor_ids"].astype(np.int64)
    weights = weights_cache["weights"].astype(np.float32)
    _, _, _, train_site, _ = load_positions()
    query_site = train_site[query_ids]
    train_pas = np.load(TRAIN_PAS, mmap_mode="r")
    train_pdp = np.load(TRAIN_PDP, mmap_mode="r")

    pas_names = tuple(pas_variants(np.zeros((256, 4, 192), dtype=np.float32)))
    pdp_names = tuple(pdp_variants(np.zeros((256, 4, 192), dtype=np.float32)))
    pas_values = {
        (name, alpha): np.empty(len(query_ids), dtype=np.float64)
        for name in pas_names
        for alpha in ALPHAS
    }
    pdp_values = {
        (name, alpha): np.empty(len(query_ids), dtype=np.float64)
        for name in pdp_names
        for alpha in ALPHAS
    }

    for row, query_id in enumerate(query_ids):
        pas = spectral_target(train_pas, donor_ids[row], weights[row], axis=0)
        pdp = spectral_target(train_pdp, donor_ids[row], weights[row], axis=2)
        truth_pas = np.asarray(train_pas[query_id], dtype=np.float32)
        truth_pdp = np.asarray(train_pdp[query_id], dtype=np.float32)
        for name, target in pas_variants(pas).items():
            target = normalize(target, axis=0)
            for alpha in ALPHAS:
                pas_values[(name, alpha)][row] = cosine_vector(
                    blend(pas, target, alpha, axis=0), truth_pas, axis=0
                )
        for name, target in pdp_variants(pdp).items():
            target = normalize(target, axis=2)
            for alpha in ALPHAS:
                pdp_values[(name, alpha)][row] = cosine_vector(
                    blend(pdp, target, alpha, axis=2), truth_pdp, axis=2
                )
        if (row + 1) % 25 == 0:
            print(f"linkwise screen {row + 1}/{len(query_ids)}", flush=True)

    pas_rows = [
        {
            "variant": name,
            "alpha": alpha,
            "score": grouped_summary(values, fold_id, query_site),
        }
        for (name, alpha), values in pas_values.items()
    ]
    pdp_rows = [
        {
            "variant": name,
            "alpha": alpha,
            "score": grouped_summary(values, fold_id, query_site),
        }
        for (name, alpha), values in pdp_values.items()
    ]
    best_pas = max(pas_rows, key=lambda row: float(row["score"]["mean"]))
    best_pdp = max(pdp_rows, key=lambda row: float(row["score"]["mean"]))
    payload = {
        "experiment": "Round2-E4-V3-linkwise-shrinkage-screen",
        "status": "analysis_only",
        "sionna_used": False,
        "official_inputs_only": True,
        "validation": "fixed 500-point five-fold spatial holes",
        "alphas": ALPHAS,
        "pas_results": pas_rows,
        "pdp_results": pdp_rows,
        "best_pas": best_pas,
        "best_pdp": best_pdp,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"best_pas": best_pas, "best_pdp": best_pdp}, indent=2), flush=True)


if __name__ == "__main__":
    main()
