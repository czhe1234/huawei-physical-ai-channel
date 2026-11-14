#!/usr/bin/env python3
"""v255: official-holdout oracle audit for global spectral transport.

This is an analysis-only upper-bound test. For each validation query and each
of its labelled neighbours, the validation truth is used only to find the best
single circular shift of the neighbour S-H/PDP spectrum. If even this oracle
alignment cannot beat raw Euclidean interpolation by a useful margin, a
learned pairwise shift model cannot be a breakthrough direction.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree

import analyze_v254_route_topology_spectral_field as v254
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 255
VERSION = "v255_global_spectral_transport_oracle"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"

FOLD_SEEDS = v254.FOLD_SEEDS
VAL_COUNT = 320
NEIGHBORS = 16
POWER = 2.0
MAX_D_SHIFT = 8
MAX_S_SHIFT = 8
MAX_H_SHIFT = 4


def normalize(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    total = out.reshape(len(out), -1).sum(axis=1)
    valid = total > 1e-20
    out[valid] /= total[valid].reshape(
        (int(np.sum(valid)),) + (1,) * (out.ndim - 1)
    )
    out[~valid] = 0.0
    return out.astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    y = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def metrics(
    pred_sh: np.ndarray,
    pred_d: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
) -> dict[str, float]:
    sh = float(np.mean(row_cos(pred_sh, truth_sh)))
    d = float(np.mean(row_cos(pred_d, truth_d)))
    return {"sh": sh, "d": d, "weighted": 0.52 * sh + 0.48 * d}


def best_1d_shifts(query: np.ndarray, bank: np.ndarray, max_shift: int) -> np.ndarray:
    query_fft = np.fft.fft(query, axis=-1)
    bank_fft = np.fft.fft(bank, axis=-1)
    correlation = np.fft.ifft(
        query_fft[:, None, :] * np.conj(bank_fft), axis=-1
    ).real
    shifts = np.arange(-max_shift, max_shift + 1, dtype=np.int16)
    restricted = correlation[..., shifts % query.shape[-1]]
    return shifts[np.argmax(restricted, axis=-1)]


def best_2d_shifts(
    query: np.ndarray,
    bank: np.ndarray,
    max_s_shift: int,
    max_h_shift: int,
) -> tuple[np.ndarray, np.ndarray]:
    query_fft = np.fft.fftn(query, axes=(-2, -1))
    bank_fft = np.fft.fftn(bank, axes=(-2, -1))
    correlation = np.fft.ifftn(
        query_fft[:, None, :, :] * np.conj(bank_fft), axes=(-2, -1)
    ).real
    s_shifts = np.arange(-max_s_shift, max_s_shift + 1, dtype=np.int16)
    h_shifts = np.arange(-max_h_shift, max_h_shift + 1, dtype=np.int16)
    candidates = correlation[
        ...,
        (s_shifts[:, None] % query.shape[-2]),
        (h_shifts[None, :] % query.shape[-1]),
    ]
    flat = np.argmax(candidates.reshape(*candidates.shape[:-2], -1), axis=-1)
    h_count = len(h_shifts)
    return s_shifts[flat // h_count], h_shifts[flat % h_count]


def apply_1d_shifts(bank: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    output = np.empty_like(bank, dtype=np.float32)
    for shift in np.unique(shifts):
        mask = shifts == shift
        output[mask] = np.roll(bank[mask], int(shift), axis=-1)
    return output


def apply_2d_shifts(
    bank: np.ndarray, s_shifts: np.ndarray, h_shifts: np.ndarray
) -> np.ndarray:
    output = np.empty_like(bank, dtype=np.float32)
    pairs = np.stack([s_shifts.reshape(-1), h_shifts.reshape(-1)], axis=1)
    for s_shift, h_shift in np.unique(pairs, axis=0):
        mask = (s_shifts == s_shift) & (h_shifts == h_shift)
        output[mask] = np.roll(
            bank[mask], (int(s_shift), int(h_shift)), axis=(-2, -1)
        )
    return output


def weighted_average(bank: np.ndarray, weight: np.ndarray) -> np.ndarray:
    shape = weight.shape + (1,) * (bank.ndim - 2)
    return normalize(
        np.sum(bank * weight.reshape(shape), axis=1, dtype=np.float64).astype(np.float32)
    )


def main() -> None:
    if not TARGET_CACHE.is_file():
        raise FileNotFoundError(TARGET_CACHE)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)[:, :2]
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)[:, :2]
    data = np.load(TARGET_CACHE)
    sh = data["sh"].astype(np.float32)
    d = data["d"].astype(np.float32)
    valid_ids = np.flatnonzero(data["valid"].astype(bool)).astype(np.int64)

    folds: list[dict[str, object]] = []
    all_d_shift: list[np.ndarray] = []
    all_s_shift: list[np.ndarray] = []
    all_h_shift: list[np.ndarray] = []
    all_distance: list[np.ndarray] = []
    for fold_index, seed in enumerate(FOLD_SEEDS):
        matched = v254.choose_seeded_matched_indices(train_pos, test_pos, VAL_COUNT, seed)
        val_ids = np.intersect1d(matched, valid_ids)
        fit_ids = np.setdiff1d(valid_ids, val_ids)
        distance, local = cKDTree(train_pos[fit_ids]).query(
            train_pos[val_ids], k=NEIGHBORS
        )
        neighbour_ids = fit_ids[local]
        weight = 1.0 / np.maximum(distance, 0.05) ** POWER
        weight /= weight.sum(axis=1, keepdims=True)
        neighbour_sh = sh[neighbour_ids]
        neighbour_d = d[neighbour_ids]
        truth_sh = sh[val_ids]
        truth_d = d[val_ids]

        raw_sh = weighted_average(neighbour_sh, weight)
        raw_d = weighted_average(neighbour_d, weight)
        d_shift = best_1d_shifts(truth_d, neighbour_d, MAX_D_SHIFT)
        s_shift, h_shift = best_2d_shifts(
            truth_sh, neighbour_sh, MAX_S_SHIFT, MAX_H_SHIFT
        )
        aligned_d = weighted_average(apply_1d_shifts(neighbour_d, d_shift), weight)
        aligned_sh = weighted_average(
            apply_2d_shifts(neighbour_sh, s_shift, h_shift), weight
        )
        hybrid_d = normalize(0.5 * raw_d + 0.5 * aligned_d)
        hybrid_sh = normalize(0.5 * raw_sh + 0.5 * aligned_sh)

        folds.append(
            {
                "fold": fold_index,
                "seed": seed,
                "fit_count": int(len(fit_ids)),
                "validation_count": int(len(val_ids)),
                "raw": metrics(raw_sh, raw_d, truth_sh, truth_d),
                "oracle_aligned": metrics(aligned_sh, aligned_d, truth_sh, truth_d),
                "half_blend": metrics(hybrid_sh, hybrid_d, truth_sh, truth_d),
                "mean_neighbour_oracle_cos": {
                    "sh": float(np.mean(row_cos(
                        apply_2d_shifts(neighbour_sh, s_shift, h_shift).reshape(
                            -1, *truth_sh.shape[1:]
                        ),
                        np.repeat(truth_sh, NEIGHBORS, axis=0),
                    ))),
                    "d": float(np.mean(row_cos(
                        apply_1d_shifts(neighbour_d, d_shift).reshape(
                            -1, truth_d.shape[-1]
                        ),
                        np.repeat(truth_d, NEIGHBORS, axis=0),
                    ))),
                },
            }
        )
        all_d_shift.append(d_shift.reshape(-1))
        all_s_shift.append(s_shift.reshape(-1))
        all_h_shift.append(h_shift.reshape(-1))
        all_distance.append(distance.reshape(-1))
        print(f"v255 oracle fold {fold_index + 1}/{len(FOLD_SEEDS)}", flush=True)

    def mean_metric(key: str) -> dict[str, float]:
        return {
            metric: float(np.mean([fold[key][metric] for fold in folds]))
            for metric in ("sh", "d", "weighted")
        }

    raw_mean = mean_metric("raw")
    oracle_mean = mean_metric("oracle_aligned")
    blend_mean = mean_metric("half_blend")
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Neighbour interpolation is limited mainly by coherent global S/H/delay shifts; "
            "if true, oracle shift alignment should provide a large official holdout gain."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H leaked S-H and PDP targets",
                "official train/test coordinates",
                "validation truth only for an analysis-only oracle upper bound",
            ],
            "does_not_use": [
                "synthetic or Sionna channel labels",
                "teacher outputs",
                "local replay truth",
                "hidden test labels",
                "raw RT H",
            ],
            "deployment": "none; the oracle cannot be used to generate a candidate",
        },
        "parameters": {
            "fold_seeds": list(FOLD_SEEDS),
            "neighbours": NEIGHBORS,
            "power": POWER,
            "max_d_shift": MAX_D_SHIFT,
            "max_s_shift": MAX_S_SHIFT,
            "max_h_shift": MAX_H_SHIFT,
        },
        "folds": folds,
        "mean": {
            "raw": raw_mean,
            "oracle_aligned": oracle_mean,
            "half_blend": blend_mean,
            "oracle_delta": {
                key: oracle_mean[key] - raw_mean[key]
                for key in ("sh", "d", "weighted")
            },
        },
        "shift_diagnostics": {
            "distance": summarize(np.concatenate(all_distance)),
            "d_shift": summarize(np.concatenate(all_d_shift)),
            "s_shift": summarize(np.concatenate(all_s_shift)),
            "h_shift": summarize(np.concatenate(all_h_shift)),
            "d_at_boundary_fraction": float(
                np.mean(np.abs(np.concatenate(all_d_shift)) == MAX_D_SHIFT)
            ),
            "s_at_boundary_fraction": float(
                np.mean(np.abs(np.concatenate(all_s_shift)) == MAX_S_SHIFT)
            ),
            "h_at_boundary_fraction": float(
                np.mean(np.abs(np.concatenate(all_h_shift)) == MAX_H_SHIFT)
            ),
        },
        "local_gate": None,
        "outputs": {
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
