#!/usr/bin/env python3
"""Audit a spatially varying official-train calibration of Sionna joint S-H."""

from __future__ import annotations

import json

import numpy as np

import analyze_v401_dualdomain_pas_preserving_confocal as v401
import analyze_v432_sionna_link_energy_calibration as v432
import analyze_v434_sionna_joint_sh_calibration as v434
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 462
VERSION = "v462_spatial_sionna_residual_calibration"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OOF_CACHE = OUT_DIR / f"{VERSION}_oof.npz"

TRAIN_OFFICIAL = PATHS.round1 / "Round1_Train_Channel.npy"
TRAIN_HIGH = PATHS.cache_dir(208) / "v208_sionna_train_clean_H.npy"
TRAIN_LOWS = tuple(
    PATHS.cache_dir(205) / name / "train_clean_H.npy"
    for name in (
        "benchmark_sector_m30_track_d5_f2p6",
        "benchmark_sector_m30_track_d5_f6p0",
        "benchmark_sector_m30_track_d5_f6p7",
    )
)
OLD_OOF = PATHS.cache_dir(434) / "v434_sionna_joint_sh_calibration_oof.npz"

REGIMES = ("clear", "light", "medium", "heavy")
STRENGTHS = np.asarray([0.0, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90], dtype=np.float32)
ALPHAS = np.linspace(0.05, 0.55, 11, dtype=np.float32)
OVERLAY_ALPHAS = np.asarray(
    [0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80], dtype=np.float32
)
K_NEIGHBORS = 48
SOURCE_DISTANCE_SCALE = 18.0
LOG_FLOOR = 1e-10


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    return value / np.maximum(value.sum(axis=1, keepdims=True), TINY)


def source_marginals(source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid = source.reshape(len(source), 192, 16)
    delay = normalize_rows(grid.sum(axis=2, dtype=np.float64).astype(np.float32))
    angle = normalize_rows(grid.sum(axis=1, dtype=np.float64).astype(np.float32))
    return delay, angle


def cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.sqrt(np.sum(left * left, axis=1, dtype=np.float64))
    right_norm = np.sqrt(np.sum(right * right, axis=1, dtype=np.float64))
    numerator = left.astype(np.float64) @ right.astype(np.float64).T
    return (
        numerator / np.maximum(left_norm[:, None] * right_norm[None, :], 1e-30)
    ).astype(np.float32)


def conditioned_neighbor_bank(
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
    positions: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
    source: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    xy = positions[:, :2].astype(np.float32)
    spatial = np.linalg.norm(
        xy[query_ids, None] - xy[None, fit_ids], axis=2
    ).astype(np.float32)
    block_delta = np.abs(
        blocked[query_ids, None] - blocked[None, fit_ids]
    ).astype(np.float32)
    same = regimes[query_ids, None] == regimes[None, fit_ids]
    delay, angle = source_marginals(source)
    delay_cos = cosine_matrix(delay[query_ids], delay[fit_ids])
    angle_cos = cosine_matrix(angle[query_ids], angle[fit_ids])
    spectral_distance = 0.70 * (1.0 - delay_cos) + 0.30 * (1.0 - angle_cos)
    effective = (
        spatial
        + 0.09 * block_delta
        + 5.0 * (~same)
        + SOURCE_DISTANCE_SCALE * spectral_distance
    ).astype(np.float32)
    local_k = min(K_NEIGHBORS, len(fit_ids))
    nearest = np.argpartition(effective, local_k - 1, axis=1)[:, :local_k]
    rows = np.arange(len(query_ids))[:, None]
    nearest = nearest[rows, np.argsort(effective[rows, nearest], axis=1)]
    selected = effective[rows, nearest]
    selected_spatial = spatial[rows, nearest]
    selected_spectral = spectral_distance[rows, nearest]
    selected_same = same[rows, nearest].astype(np.float32)
    weight = np.exp(-selected / 20.0).astype(np.float32)
    weight *= 0.45 + 0.55 * selected_same
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), TINY)
    return fit_ids[nearest], weight, {
        "mean_spatial_distance": float(np.mean(selected_spatial)),
        "mean_spectral_distance": float(np.mean(selected_spectral)),
        "same_regime_fraction": float(np.mean(selected_same)),
    }


def predict_local_residual(
    residual: np.ndarray,
    neighbor_ids: np.ndarray,
    weights: np.ndarray,
    query_batch: int = 32,
) -> np.ndarray:
    output = np.empty((len(neighbor_ids), residual.shape[1]), dtype=np.float32)
    for start in range(0, len(neighbor_ids), query_batch):
        end = min(start + query_batch, len(neighbor_ids))
        values = residual[neighbor_ids[start:end]]
        output[start:end] = np.sum(
            values * weights[start:end, :, None], axis=1, dtype=np.float64
        ).astype(np.float32)
    return output


def main() -> None:
    required = (TRAIN_OFFICIAL, TRAIN_HIGH, *TRAIN_LOWS, OLD_OOF)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked = geometry["train_blocked"].astype(np.float32)
    regimes = regime_of(blocked).astype("U8")
    truth, _ = v434.load_sh(TRAIN_OFFICIAL)
    high, high_total = v434.load_sh(TRAIN_HIGH)
    lows = [v434.load_sh(path) for path in TRAIN_LOWS]
    raw_source, point_quality, _ = v434.robust_source(
        high,
        high_total,
        [value[0] for value in lows],
        [value[1] for value in lows],
    )

    old_cache = np.load(OLD_OOF)
    baseline = old_cache["baseline"].astype(np.float32)
    old_selected = old_cache["selected"].astype(np.float32)
    old_score = v432.row_cosine(truth, old_selected)
    baseline_score = v432.row_cosine(truth, baseline)
    all_ids = np.arange(len(positions), dtype=np.int64)
    folds = v401.confirmatory_folds(all_ids, positions[:, :2])
    fold_id = np.full(len(positions), -1, dtype=np.int16)
    corrected_by_strength = {
        float(strength): np.zeros_like(truth) for strength in STRENGTHS
    }
    neighbor_rows = []

    for fold, query_ids in enumerate(folds):
        fit_ids = np.setdiff1d(all_ids, query_ids)
        fold_id[query_ids] = fold
        ratios = v432.fit_ratio(truth, raw_source, fit_ids, regimes)
        global_all = v432.apply_ratio(raw_source, all_ids, regimes, ratios)
        fit_residual = np.zeros_like(truth)
        fit_residual[fit_ids] = np.clip(
            np.log(np.maximum(truth[fit_ids], LOG_FLOOR))
            - np.log(np.maximum(global_all[fit_ids], LOG_FLOOR)),
            -2.5,
            2.5,
        ).astype(np.float32)
        neighbor_ids, weights, diagnostics = conditioned_neighbor_bank(
            fit_ids,
            query_ids,
            positions,
            blocked,
            regimes,
            global_all,
        )
        local_residual = predict_local_residual(
            fit_residual, neighbor_ids, weights
        )
        neighbor_rows.append({"fold": fold, **diagnostics})
        for strength in STRENGTHS:
            corrected = global_all[query_ids] * np.exp(
                np.clip(strength * local_residual, -1.5, 1.5)
            )
            corrected_by_strength[float(strength)][query_ids] = normalize_rows(corrected)
        print(f"v462 fold {fold + 1}/5 residual complete", flush=True)

    score_grid = np.empty((len(STRENGTHS), len(ALPHAS), len(truth)), dtype=np.float32)
    for strength_index, strength in enumerate(STRENGTHS):
        target = corrected_by_strength[float(strength)]
        for alpha_index, alpha in enumerate(ALPHAS):
            effective = np.clip(alpha * point_quality, 0.0, 1.0)
            candidate = normalize_rows(
                baseline * (1.0 - effective[:, None])
                + target * effective[:, None]
            )
            score_grid[strength_index, alpha_index] = v432.row_cosine(truth, candidate)
    means = score_grid.mean(axis=2)
    selected_flat = int(np.argmax(means))
    strength_index, alpha_index = np.unravel_index(selected_flat, means.shape)
    selected_strength = float(STRENGTHS[strength_index])
    selected_alpha = float(ALPHAS[alpha_index])
    selected_score = score_grid[strength_index, alpha_index]
    selected_target = corrected_by_strength[selected_strength]
    global_target = corrected_by_strength[0.0]
    residual_direction = np.clip(
        np.log(np.maximum(selected_target, LOG_FLOOR))
        - np.log(np.maximum(global_target, LOG_FLOOR)),
        -1.5,
        1.5,
    ).astype(np.float32)
    overlay_scores = np.empty((len(OVERLAY_ALPHAS), len(truth)), dtype=np.float32)
    for overlay_index, overlay_alpha in enumerate(OVERLAY_ALPHAS):
        overlay = normalize_rows(
            old_selected
            * np.exp(
                overlay_alpha * point_quality[:, None] * residual_direction
            )
        )
        overlay_scores[overlay_index] = v432.row_cosine(truth, overlay)
    overlay_index = int(np.argmax(overlay_scores.mean(axis=1)))
    overlay_alpha = float(OVERLAY_ALPHAS[overlay_index])
    overlay_selected = overlay_scores[overlay_index]
    fold_rows = []
    for fold in range(5):
        mask = fold_id == fold
        fold_rows.append(
            {
                "fold": fold,
                "count": int(np.sum(mask)),
                "baseline": float(np.mean(baseline_score[mask])),
                "old_selected": float(np.mean(old_score[mask])),
                "selected": float(np.mean(selected_score[mask])),
                "gain_vs_old": float(np.mean(selected_score[mask] - old_score[mask])),
            }
        )
    gain_vs_old = float(np.mean(selected_score - old_score))
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "The Sionna-to-official spectral mismatch is a spatially varying field. "
            "Interpolating train-only log-support residuals among spatially and spectrally "
            "similar points should improve the global/regime calibration used by v458."
        ),
        "parameters": {
            "neighbors": K_NEIGHBORS,
            "source_distance_scale": SOURCE_DISTANCE_SCALE,
            "strengths": STRENGTHS.tolist(),
            "alphas": ALPHAS.tolist(),
            "folds": "same five spatial folds as v434",
        },
        "selection": {
            "strength": selected_strength,
            "alpha": selected_alpha,
            "baseline_mean": float(np.mean(baseline_score)),
            "old_v434_mean": float(np.mean(old_score)),
            "selected_mean": float(np.mean(selected_score)),
            "gain_vs_old": gain_vs_old,
            "fold_wins_vs_old": int(sum(row["gain_vs_old"] > 0.0 for row in fold_rows)),
            "folds": fold_rows,
        },
        "overlay_selection": {
            "operator": "old_v434_selected * exp(alpha * quality * local_residual_direction)",
            "alphas": OVERLAY_ALPHAS.tolist(),
            "selected_alpha": overlay_alpha,
            "old_v434_mean": float(np.mean(old_score)),
            "selected_mean": float(np.mean(overlay_selected)),
            "gain_vs_old": float(np.mean(overlay_selected - old_score)),
            "fold_wins_vs_old": int(
                sum(
                    float(np.mean(overlay_selected[fold_id == fold] - old_score[fold_id == fold]))
                    > 0.0
                    for fold in range(5)
                )
            ),
            "folds": [
                {
                    "fold": fold,
                    "gain_vs_old": float(
                        np.mean(
                            overlay_selected[fold_id == fold]
                            - old_score[fold_id == fold]
                        )
                    ),
                }
                for fold in range(5)
            ],
            "mean_by_alpha": overlay_scores.mean(axis=1).tolist(),
        },
        "neighbor_diagnostics": neighbor_rows,
        "deployment_gate": {
            "pass": bool(
                gain_vs_old >= 0.005
                and sum(row["gain_vs_old"] > 0.0 for row in fold_rows) >= 4
            ),
            "minimum_gain_vs_old": 0.005,
            "minimum_fold_wins": 4,
        },
        "score_grid": means.tolist(),
        "outputs": {
            "oof": PATHS.relative(OOF_CACHE),
            "report": PATHS.relative(REPORT),
        },
    }
    np.savez_compressed(
        OOF_CACHE,
        fold_id=fold_id,
        point_quality=point_quality,
        truth=truth,
        baseline=baseline,
        old_selected=old_selected,
        selected_target=selected_target,
        global_target=global_target,
        residual_direction=residual_direction,
        overlay_selected_score=overlay_selected,
        selected_score=selected_score,
    )
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
