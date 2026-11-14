#!/usr/bin/env python3
"""Measure noncoherent path-track PAS/PDP targets on Round 2 spatial holes."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from data import load_positions
from project_paths import PATHS


ARTIFACT_DIR = PATHS.artifacts / "round2" / "v4_path_track_spectral_targets"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v4_path_track_spectral_targets"
REPORT = ANALYSIS_DIR / "path_track_spectral_targets.json"

PATH_ARTIFACT_DIR = PATHS.artifacts / "round2" / "v4_path_component_transport"
TRUE_COMPONENTS = PATH_ARTIFACT_DIR / "train_components_k24_os2_ref2.npz"
PREDICTED_COMPONENTS = (
    PATH_ARTIFACT_DIR
    / "oof_path_parameters_k16_os2_ref2_n40_dir3600_r1.npz"
)
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
TRAIN_AD = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_train_ad_support_f16.npy"
)
TRAIN_PAS = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_train_pas_link_support_f16.npy"
)
TRAIN_PDP = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_train_pdp_link_support_f16.npy"
)

DEPLOYABLE_AD = ARTIFACT_DIR / "oof_track_ad_f16.npy"
DEPLOYABLE_PAS = ARTIFACT_DIR / "oof_track_pas_f16.npy"
DEPLOYABLE_PDP = ARTIFACT_DIR / "oof_track_pdp_f16.npy"

MP = 2
MH = 16
MV = 8
N_UE = 4
S_SC = 192
PORTS = MP * N_UE
TINY = 1e-30


@dataclass(frozen=True)
class RenderCase:
    name: str
    coordinates: np.ndarray
    coefficients: np.ndarray


def normalized(value: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    value = value.clamp_min(0.0)
    total = value.sum(dim=dims, keepdim=True)
    return torch.where(
        total > 0.0,
        value / total.clamp_min(TINY),
        torch.zeros_like(value),
    )


def periodic_delta(destination: np.ndarray, source: np.ndarray, period: float) -> np.ndarray:
    return (destination - source + period / 2.0) % period - period / 2.0


def match_by_support(
    predicted_coordinates: np.ndarray,
    true_coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one-to-one path matches based only on delay-angle support."""

    h_delta = periodic_delta(
        predicted_coordinates[:, None, 0], true_coordinates[None, :, 0], MH
    )
    v_delta = periodic_delta(
        predicted_coordinates[:, None, 1], true_coordinates[None, :, 1], MV
    )
    d_delta = periodic_delta(
        predicted_coordinates[:, None, 2], true_coordinates[None, :, 2], S_SC
    )
    cost = (h_delta / 1.5) ** 2 + v_delta**2 + (d_delta / 2.0) ** 2
    predicted_rows, true_columns = linear_sum_assignment(cost)
    return predicted_rows.astype(np.int64), true_columns.astype(np.int64)


def matched_cases(
    predicted_coordinates: np.ndarray,
    predicted_coefficients: np.ndarray,
    true_coordinates: np.ndarray,
    true_coefficients: np.ndarray,
) -> list[RenderCase]:
    count, paths, _ = predicted_coordinates.shape
    true_top_coordinates = np.asarray(true_coordinates[:, :paths], dtype=np.float32)
    true_top_coefficients = np.asarray(true_coefficients[:, :paths], dtype=np.complex64)
    predicted_support_true_energy = np.zeros_like(predicted_coefficients)
    true_support_predicted_energy = np.zeros_like(predicted_coefficients)

    for row in range(count):
        predicted_rows, true_columns = match_by_support(
            predicted_coordinates[row], true_top_coordinates[row]
        )
        predicted_support_true_energy[row, predicted_rows] = true_top_coefficients[
            row, true_columns
        ]
        true_support_predicted_energy[row, true_columns] = predicted_coefficients[
            row, predicted_rows
        ]

    return [
        RenderCase(
            "true_support_true_energy",
            true_top_coordinates,
            true_top_coefficients,
        ),
        RenderCase(
            "predicted_support_true_energy",
            predicted_coordinates,
            predicted_support_true_energy,
        ),
        RenderCase(
            "true_support_predicted_energy",
            true_top_coordinates,
            true_support_predicted_energy,
        ),
        RenderCase(
            "predicted_support_predicted_energy",
            predicted_coordinates,
            predicted_coefficients,
        ),
    ]


def spectral_kernels(
    coordinates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = coordinates.device
    horizontal_index = torch.arange(MH, dtype=torch.float32, device=device)
    vertical_index = torch.arange(MV, dtype=torch.float32, device=device)
    subcarrier_index = torch.arange(S_SC, dtype=torch.float32, device=device)

    horizontal_wave = torch.exp(
        2j
        * math.pi
        * coordinates[..., 0, None]
        * horizontal_index[None, None, :]
        / MH
    )
    vertical_wave = torch.exp(
        2j
        * math.pi
        * coordinates[..., 1, None]
        * vertical_index[None, None, :]
        / MV
    )
    delay_wave = torch.exp(
        -2j
        * math.pi
        * coordinates[..., 2, None]
        * subcarrier_index[None, None, :]
        / S_SC
    )
    horizontal = torch.fft.fft(horizontal_wave, dim=-1).abs().square()
    vertical = torch.fft.fft(vertical_wave, dim=-1).abs().square()
    delay = torch.fft.ifft(delay_wave, dim=-1).abs().square()
    horizontal /= horizontal.sum(dim=-1, keepdim=True).clamp_min(TINY)
    vertical /= vertical.sum(dim=-1, keepdim=True).clamp_min(TINY)
    delay /= delay.sum(dim=-1, keepdim=True).clamp_min(TINY)
    return horizontal.float(), vertical.float(), delay.float()


@torch.inference_mode()
def render_noncoherent(
    coordinates: np.ndarray,
    coefficients: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinate_tensor = torch.as_tensor(coordinates, dtype=torch.float32, device=device)
    coefficient_tensor = torch.as_tensor(coefficients, dtype=torch.complex64, device=device)
    energy = coefficient_tensor.abs().square().reshape(
        len(coordinate_tensor), -1, MP, N_UE
    )
    horizontal, vertical, delay = spectral_kernels(coordinate_tensor)

    pas_small = torch.einsum(
        "bkpu,bkh,bkv->bphvu", energy, horizontal, vertical
    )
    pas_small = normalized(pas_small, (1, 2, 3))
    pas = (
        pas_small[..., None]
        .expand(-1, -1, -1, -1, -1, S_SC)
        .reshape(-1, MP * MH * MV, N_UE, S_SC)
    )

    pdp_small = torch.einsum("bkpu,bkd->bpud", energy, delay)
    pdp_small = normalized(pdp_small, (3,))
    pdp = (
        pdp_small[:, :, None, None, :, :]
        .expand(-1, -1, MH, MV, -1, -1)
        .reshape(-1, MP * MH * MV, N_UE, S_SC)
    )

    collapsed_energy = energy.sum(dim=(2, 3))
    ad = torch.einsum(
        "bk,bkh,bkv,bkd->bhvd",
        collapsed_energy,
        horizontal,
        vertical,
        delay,
    )
    ad = normalized(ad, (1, 2, 3))
    return ad.cpu().numpy(), pas.cpu().numpy(), pdp.cpu().numpy()


def cosine(
    prediction: np.ndarray,
    truth: np.ndarray,
    axis: int | tuple[int, ...],
) -> np.ndarray:
    prediction64 = np.asarray(prediction, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    numerator = np.sum(prediction64 * truth64, axis=axis, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(prediction64 * prediction64, axis=axis, dtype=np.float64)
        * np.sum(truth64 * truth64, axis=axis, dtype=np.float64)
    )
    return np.divide(
        numerator,
        np.where(denominator == 0.0, 1e-12, denominator),
    )


def summarize(
    values: np.ndarray,
    fold_id: np.ndarray,
    site: np.ndarray,
) -> dict[str, object]:
    return {
        "mean": float(np.mean(values)),
        "fold": [
            float(np.mean(values[fold_id == fold])) for fold in np.unique(fold_id)
        ],
        "site": {
            str(int(site_value)): float(np.mean(values[site == site_value]))
            for site_value in np.unique(site)
        },
    }


def evaluate_cases(
    cases: list[RenderCase],
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    query_site: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    truth_ad = np.load(TRAIN_AD, mmap_mode="r")
    truth_pas = np.load(TRAIN_PAS, mmap_mode="r")
    truth_pdp = np.load(TRAIN_PDP, mmap_mode="r")
    metrics = {
        case.name: {
            "ad": np.zeros(len(query_ids), dtype=np.float64),
            "pas": np.zeros(len(query_ids), dtype=np.float64),
            "pdp": np.zeros(len(query_ids), dtype=np.float64),
        }
        for case in cases
    }

    deployable = next(
        case for case in cases if case.name == "predicted_support_predicted_energy"
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ad_output = np.lib.format.open_memmap(
        DEPLOYABLE_AD,
        mode="w+",
        dtype=np.float16,
        shape=(len(query_ids), MH, MV, S_SC),
    )
    pas_output = np.lib.format.open_memmap(
        DEPLOYABLE_PAS,
        mode="w+",
        dtype=np.float16,
        shape=(len(query_ids), MP * MH * MV, N_UE, S_SC),
    )
    pdp_output = np.lib.format.open_memmap(
        DEPLOYABLE_PDP,
        mode="w+",
        dtype=np.float16,
        shape=(len(query_ids), MP * MH * MV, N_UE, S_SC),
    )

    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        selected_ids = query_ids[start:end]
        block_truth_ad = np.asarray(truth_ad[selected_ids], dtype=np.float32)
        block_truth_pas = np.asarray(truth_pas[selected_ids], dtype=np.float32)
        block_truth_pdp = np.asarray(truth_pdp[selected_ids], dtype=np.float32)
        for case in cases:
            ad, pas, pdp = render_noncoherent(
                case.coordinates[start:end], case.coefficients[start:end], device
            )
            metrics[case.name]["ad"][start:end] = cosine(
                ad, block_truth_ad, (1, 2, 3)
            )
            metrics[case.name]["pas"][start:end] = cosine(
                pas, block_truth_pas, 1
            ).mean(axis=(1, 2), dtype=np.float64)
            metrics[case.name]["pdp"][start:end] = cosine(
                pdp, block_truth_pdp, 3
            ).mean(axis=(1, 2), dtype=np.float64)
            if case is deployable:
                ad_output[start:end] = ad.astype(np.float16)
                pas_output[start:end] = pas.astype(np.float16)
                pdp_output[start:end] = pdp.astype(np.float16)
        print(f"path-track target evaluation {end}/{len(query_ids)}", flush=True)

    ad_output.flush()
    pas_output.flush()
    pdp_output.flush()
    return {
        name: {
            metric: summarize(values, fold_id, query_site)
            for metric, values in rows.items()
        }
        for name, rows in metrics.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    predicted = np.load(PREDICTED_COMPONENTS)
    if not np.array_equal(predicted["query_ids"].astype(np.int64), query_ids):
        raise RuntimeError("predicted component rows do not match V3 OOF rows")
    true = np.load(TRUE_COMPONENTS)
    _, _, _, train_site, _ = load_positions()
    query_site = train_site[query_ids]

    cases = matched_cases(
        predicted["coordinates"].astype(np.float32),
        predicted["coefficients"].astype(np.complex64),
        true["coordinates"][query_ids].astype(np.float32),
        true["coefficients"][query_ids].astype(np.complex64),
    )
    results = evaluate_cases(
        cases,
        query_ids,
        fold_id,
        query_site,
        torch.device(args.device),
        args.batch_size,
    )
    payload = {
        "experiment": "Round2-E9-path-track-spectral-targets",
        "status": "analysis_only",
        "primary_hypothesis": (
            "Continuous delay-angle paths can improve PAS/PDP without predicting "
            "spatially undersampled absolute carrier phase."
        ),
        "sionna_used": False,
        "official_inputs_only": True,
        "validation": "fixed 500-point five-fold spatial holes",
        "paths": int(cases[0].coordinates.shape[1]),
        "results": results,
        "deployable_targets": {
            "ad": PATHS.relative(DEPLOYABLE_AD),
            "pas": PATHS.relative(DEPLOYABLE_PAS),
            "pdp": PATHS.relative(DEPLOYABLE_PDP),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
