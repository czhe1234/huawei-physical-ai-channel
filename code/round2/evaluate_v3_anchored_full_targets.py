#!/usr/bin/env python3
"""Build and score full linkwise targets from the anchored neural RBF model."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from data import load_positions
from project_paths import PATHS
from run_v3_anchored_neural_rbf import (
    AnchoredWeightCorrector,
    Config,
    GEOMETRY,
    OOF_META,
    V2_WEIGHTS,
    validation_context,
)


ARTIFACT_DIR = PATHS.artifacts / "round2" / "v3_anchored_neural_rbf"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v3_anchored_neural_rbf"
REPORT = ANALYSIS_DIR / "full_target_scores.json"
CORRECTED_WEIGHTS = ARTIFACT_DIR / "oof_epoch6_corrected_weights.npz"

PRE_DIR = PATHS.artifacts / "round2" / "pre_sionna"
TRAIN_AD = PRE_DIR / "round2_train_ad_support_f16.npy"
TRAIN_PAS = PRE_DIR / "round2_train_pas_link_support_f16.npy"
TRAIN_PDP = PRE_DIR / "round2_train_pdp_link_support_f16.npy"
DESCRIPTORS = PRE_DIR / "round2_train_spectral_descriptors.npz"

TARGET_AD_BASE = ARTIFACT_DIR / "oof_ad_base_f16.npy"
TARGET_AD_AVERAGE = ARTIFACT_DIR / "oof_ad_average_f16.npy"
TARGET_AD_PAS = ARTIFACT_DIR / "oof_ad_pas_f16.npy"
TARGET_AD_PDP = ARTIFACT_DIR / "oof_ad_pdp_f16.npy"
TARGET_PAS = ARTIFACT_DIR / "oof_pas_neural_f16.npy"
TARGET_PDP = ARTIFACT_DIR / "oof_pdp_neural_f16.npy"

PAS_SHRINK = (0.0, 0.10, 0.25, 0.40)
PDP_SHRINK = (0.0, 0.40, 0.55, 0.70, 0.85, 1.0)


def normalize(value: np.ndarray, axes: int | tuple[int, ...]) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = array.sum(axis=axes, keepdims=True, dtype=np.float64)
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
    axes: int | tuple[int, ...],
) -> np.ndarray:
    donor = np.sqrt(np.maximum(np.asarray(cache[donor_ids], dtype=np.float32), 0.0))
    amplitude = np.maximum(
        np.tensordot(weights.astype(np.float32), donor, axes=(0, 0)), 0.0
    )
    return normalize(amplitude * amplitude, axes)


def blend(base: np.ndarray, target: np.ndarray, alpha: float, axis: int) -> np.ndarray:
    amplitude = (1.0 - alpha) * np.sqrt(np.maximum(base, 0.0)) + alpha * np.sqrt(
        np.maximum(target, 0.0)
    )
    return normalize(amplitude * amplitude, axis)


def cosine_mean(prediction: np.ndarray, truth: np.ndarray, axis: int | tuple[int, ...]) -> float:
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


def load_corrected_weights(device: torch.device) -> dict[str, np.ndarray]:
    if CORRECTED_WEIGHTS.is_file():
        cached = np.load(CORRECTED_WEIGHTS)
        if int(cached["epoch"]) == 6:
            return {key: cached[key] for key in cached.files}

    descriptors = np.load(DESCRIPTORS)
    spectra_numpy = np.concatenate(
        [
            np.sqrt(np.maximum(descriptors["global_pas"].astype(np.float32), 0.0)),
            np.sqrt(np.maximum(descriptors["global_pdp"].astype(np.float32), 0.0)),
        ],
        axis=1,
    ).astype(np.float32)
    geometry_numpy = np.load(GEOMETRY)["train"].astype(np.float32)
    _, positions_numpy, _, _, _ = load_positions()
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    v2 = np.load(V2_WEIGHTS)
    donor_ids = v2["neighbor_ids"].astype(np.int64)
    pas_weights = np.empty_like(v2["weights"], dtype=np.float32)
    pdp_weights = np.empty_like(v2["weights"], dtype=np.float32)

    positions = torch.as_tensor(positions_numpy[:, :2], dtype=torch.float32, device=device)
    spectra = torch.as_tensor(spectra_numpy, dtype=torch.float32, device=device)
    for fold in np.unique(fold_id):
        rows = np.flatnonzero(fold_id == fold)
        checkpoint = torch.load(
            ARTIFACT_DIR / f"fold{int(fold)}_final.pt",
            map_location=device,
            weights_only=False,
        )
        if int(checkpoint["epoch"]) != 6:
            raise RuntimeError(f"fold {fold} final checkpoint is not epoch 6")
        config = Config(**checkpoint["config"])
        selected_geometry = geometry_numpy[:, 3:]
        geometry_numpy_fold = (
            (selected_geometry - checkpoint["feature_mean"][None, :])
            / checkpoint["feature_std"][None, :]
        ).astype(np.float32)
        geometry = torch.as_tensor(geometry_numpy_fold, dtype=torch.float32, device=device)
        model = AnchoredWeightCorrector(geometry.shape[1], config).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        table = validation_context(int(fold), query_ids, fold_id)
        for start in range(0, len(rows), 128):
            end = min(start + 128, len(rows))
            local_donor = torch.as_tensor(
                table.donor_ids[start:end], dtype=torch.long, device=device
            )
            query = torch.as_tensor(
                table.query_ids[start:end], dtype=torch.long, device=device
            )
            distance = torch.as_tensor(
                table.distances[start:end], dtype=torch.float32, device=device
            )
            base_weight = torch.as_tensor(
                table.weights[start:end], dtype=torch.float32, device=device
            )
            with torch.inference_mode():
                _, _, diagnostics = model(
                    geometry[query],
                    positions[query],
                    geometry[local_donor],
                    positions[local_donor],
                    spectra[local_donor],
                    distance,
                    base_weight,
                )
            pas_weights[rows[start:end]] = diagnostics["pas_weights"].cpu().numpy()
            pdp_weights[rows[start:end]] = diagnostics["pdp_weights"].cpu().numpy()
    np.savez_compressed(
        CORRECTED_WEIGHTS,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        donor_ids=donor_ids.astype(np.int32),
        base_weights=v2["weights"].astype(np.float32),
        pas_weights=pas_weights,
        pdp_weights=pdp_weights,
        epoch=np.int32(6),
    )
    return {key: value for key, value in np.load(CORRECTED_WEIGHTS).items()}


def build_and_score(device: torch.device) -> dict[str, object]:
    weights = load_corrected_weights(device)
    query_ids = weights["query_ids"].astype(np.int64)
    donor_ids = weights["donor_ids"].astype(np.int64)
    base_weights = weights["base_weights"].astype(np.float32)
    pas_weights = weights["pas_weights"].astype(np.float32)
    pdp_weights = weights["pdp_weights"].astype(np.float32)
    average_weights = 0.5 * (pas_weights + pdp_weights)

    train_ad = np.load(TRAIN_AD, mmap_mode="r")
    train_pas = np.load(TRAIN_PAS, mmap_mode="r")
    train_pdp = np.load(TRAIN_PDP, mmap_mode="r")
    ad_outputs = {
        "base": np.lib.format.open_memmap(
            TARGET_AD_BASE, mode="w+", dtype=np.float16, shape=(len(query_ids), 16, 8, 192)
        ),
        "average": np.lib.format.open_memmap(
            TARGET_AD_AVERAGE, mode="w+", dtype=np.float16, shape=(len(query_ids), 16, 8, 192)
        ),
        "pas": np.lib.format.open_memmap(
            TARGET_AD_PAS, mode="w+", dtype=np.float16, shape=(len(query_ids), 16, 8, 192)
        ),
        "pdp": np.lib.format.open_memmap(
            TARGET_AD_PDP, mode="w+", dtype=np.float16, shape=(len(query_ids), 16, 8, 192)
        ),
    }
    pas_output = np.lib.format.open_memmap(
        TARGET_PAS, mode="w+", dtype=np.float16, shape=(len(query_ids), 256, 4, 192)
    )
    pdp_output = np.lib.format.open_memmap(
        TARGET_PDP, mode="w+", dtype=np.float16, shape=(len(query_ids), 256, 4, 192)
    )
    ad_score = {name: np.empty(len(query_ids), dtype=np.float64) for name in ad_outputs}
    pas_score = {alpha: np.empty(len(query_ids), dtype=np.float64) for alpha in PAS_SHRINK}
    pdp_score = {alpha: np.empty(len(query_ids), dtype=np.float64) for alpha in PDP_SHRINK}

    ad_weight_values = {
        "base": base_weights,
        "average": average_weights,
        "pas": pas_weights,
        "pdp": pdp_weights,
    }
    for row, query_id in enumerate(query_ids):
        truth_ad = np.asarray(train_ad[query_id], dtype=np.float32)
        truth_pas = np.asarray(train_pas[query_id], dtype=np.float32)
        truth_pdp = np.asarray(train_pdp[query_id], dtype=np.float32)
        for name, values in ad_weight_values.items():
            target = spectral_target(train_ad, donor_ids[row], values[row], (0, 1, 2))
            ad_outputs[name][row] = target.astype(np.float16)
            ad_score[name][row] = cosine_mean(target, truth_ad, (0, 1, 2))

        pas = spectral_target(train_pas, donor_ids[row], pas_weights[row], 0)
        pdp = spectral_target(train_pdp, donor_ids[row], pdp_weights[row], 2)
        pas_output[row] = pas.astype(np.float16)
        pdp_output[row] = pdp.astype(np.float16)
        global_pas = np.broadcast_to(
            pas.mean(axis=(1, 2), keepdims=True, dtype=np.float64).astype(np.float32),
            pas.shape,
        )
        global_pdp = np.broadcast_to(
            pdp.mean(axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32),
            pdp.shape,
        )
        for alpha in PAS_SHRINK:
            pas_score[alpha][row] = cosine_mean(
                blend(pas, global_pas, alpha, 0), truth_pas, 0
            )
        for alpha in PDP_SHRINK:
            pdp_score[alpha][row] = cosine_mean(
                blend(pdp, global_pdp, alpha, 2), truth_pdp, 2
            )
        if (row + 1) % 25 == 0:
            print(f"full targets {row + 1}/{len(query_ids)}", flush=True)
    for output in [*ad_outputs.values(), pas_output, pdp_output]:
        output.flush()

    fold_id = weights["fold_id"].astype(np.int8)
    _, _, _, train_site, _ = load_positions()
    query_site = train_site[query_ids]

    def summarize(values: np.ndarray) -> dict[str, object]:
        return {
            "mean": float(np.mean(values)),
            "fold": [
                float(np.mean(values[fold_id == fold])) for fold in np.unique(fold_id)
            ],
            "site": {
                str(int(site)): float(np.mean(values[query_site == site]))
                for site in np.unique(query_site)
            },
        }

    return {
        "ad": {name: summarize(value) for name, value in ad_score.items()},
        "pas": {str(alpha): summarize(value) for alpha, value in pas_score.items()},
        "pdp": {str(alpha): summarize(value) for alpha, value in pdp_score.items()},
        "target_paths": {
            "ad_base": PATHS.relative(TARGET_AD_BASE),
            "ad_average": PATHS.relative(TARGET_AD_AVERAGE),
            "ad_pas": PATHS.relative(TARGET_AD_PAS),
            "ad_pdp": PATHS.relative(TARGET_AD_PDP),
            "pas": PATHS.relative(TARGET_PAS),
            "pdp": PATHS.relative(TARGET_PDP),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    results = build_and_score(torch.device(args.device))
    payload = {
        "experiment": "Round2-E4-V3-anchored-full-targets",
        "status": "analysis_only",
        "sionna_used": False,
        "official_inputs_only": True,
        "epoch": 6,
        "results": results,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
