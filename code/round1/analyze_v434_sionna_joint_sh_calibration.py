#!/usr/bin/env python3
"""Audit official-train calibration of Sionna joint delay-horizontal support."""

from __future__ import annotations

import json

import numpy as np

import analyze_v401_dualdomain_pas_preserving_confocal as v401
import analyze_v432_sionna_link_energy_calibration as v432
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 434
VERSION = "v434_sionna_joint_sh_calibration"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OOF_CACHE = CACHE_DIR / "v434_sionna_joint_sh_calibration_oof.npz"

OFFICIAL = PATHS.round1 / "Round1_Train_Channel.npy"
HIGH = PATHS.cache_dir(208) / "v208_sionna_train_clean_H.npy"
LOWS = tuple(
    PATHS.cache_dir(205) / name / "train_clean_H.npy"
    for name in (
        "benchmark_sector_m30_track_d5_f2p6",
        "benchmark_sector_m30_track_d5_f6p0",
        "benchmark_sector_m30_track_d5_f6p7",
    )
)
REGIMES = ("clear", "light", "medium", "heavy")
ALPHA_GRID = np.linspace(0.0, 1.5, 31, dtype=np.float32)
BATCH = 2
EPS = 1e-10


def load_sh(path) -> tuple[np.ndarray, np.ndarray]:
    channel = np.load(path, mmap_mode="r")
    output = np.empty((len(channel), 192 * 16), dtype=np.float32)
    total = np.empty(len(channel), dtype=np.float32)
    for start in range(0, len(channel), BATCH):
        end = min(start + BATCH, len(channel))
        support, point_total = v423.leaked_support(
            np.asarray(channel[start:end], dtype=np.complex64)
        )
        sh = v423.leaked_marginals(support)["sh"]
        output[start:end] = sh.reshape(end - start, -1)
        total[start:end] = point_total
    return v432.normalize_rows(output), total


def robust_source(
    high: np.ndarray,
    high_total: np.ndarray,
    lows: list[np.ndarray],
    low_totals: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = np.stack(lows, axis=0).astype(np.float32)
    point_valid = np.stack([value > 1e-20 for value in low_totals], axis=0)
    valid = point_valid[:, :, None] & (stack > EPS)
    coverage = valid.mean(axis=0).astype(np.float32)
    low_log = np.log(np.maximum(stack, EPS))
    spread = np.std(low_log, axis=0).astype(np.float32)
    cell_quality = coverage * np.exp(-0.42 * spread)
    low_mean = np.sum(stack * valid, axis=0) / np.maximum(valid.sum(axis=0), 1.0)
    low_mean = v432.normalize_rows(low_mean)
    high_multi = v432.row_cosine(high, low_mean)
    point_quality = np.clip(
        0.35 * point_valid.mean(axis=0)
        + 0.25 * np.quantile(cell_quality, 0.75, axis=1)
        + 0.40 * high_multi,
        0.0,
        1.0,
    ).astype(np.float32)
    point_quality[high_total <= 1e-20] = 0.0
    target = v432.normalize_rows(
        high * (1.0 - 0.18 * cell_quality)
        + low_mean * (0.18 * cell_quality)
    )
    return target, point_quality, cell_quality


def marginal_cosine(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    truth_grid = truth.reshape(len(truth), 192, 16)
    pred_grid = prediction.reshape(len(prediction), 192, 16)
    delay_truth = v432.normalize_rows(truth_grid.sum(axis=2))
    delay_pred = v432.normalize_rows(pred_grid.sum(axis=2))
    angle_truth = v432.normalize_rows(truth_grid.sum(axis=1))
    angle_pred = v432.normalize_rows(pred_grid.sum(axis=1))
    return (
        v432.row_cosine(delay_truth, delay_pred),
        v432.row_cosine(angle_truth, angle_pred),
    )


def main() -> None:
    for path in (OFFICIAL, HIGH, *LOWS):
        if not path.is_file():
            raise FileNotFoundError(path)
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked = geometry["train_blocked"].astype(np.float32)
    regimes = regime_of(blocked).astype("U8")

    truth, truth_total = load_sh(OFFICIAL)
    high, high_total = load_sh(HIGH)
    low_loaded = [load_sh(path) for path in LOWS]
    low_support = [value[0] for value in low_loaded]
    low_total = [value[1] for value in low_loaded]
    raw_source, point_quality, cell_quality = robust_source(
        high, high_total, low_support, low_total
    )

    all_ids = np.arange(len(positions), dtype=np.int64)
    folds = v401.confirmatory_folds(all_ids, positions[:, :2])
    fold_id = np.full(len(positions), -1, dtype=np.int16)
    baseline = np.zeros_like(truth)
    calibrated = np.zeros_like(truth)
    for fold, query_ids in enumerate(folds):
        fit_ids = np.setdiff1d(all_ids, query_ids)
        fold_id[query_ids] = fold
        baseline[query_ids] = v432.normalize_rows(
            v432.neighbor_prediction(
                fit_ids,
                query_ids,
                positions,
                blocked,
                regimes,
                truth,
                k=48,
            )
        )
        ratio = v432.fit_ratio(truth, raw_source, fit_ids, regimes)
        calibrated[query_ids] = v432.apply_ratio(
            raw_source, query_ids, regimes, ratio
        )

    baseline_cos = v432.row_cosine(truth, baseline)
    raw_cos = v432.row_cosine(truth, raw_source)
    calibrated_cos = v432.row_cosine(truth, calibrated)
    score_grid: dict[float, np.ndarray] = {}
    candidates: dict[float, np.ndarray] = {}
    for alpha in ALPHA_GRID:
        effective = np.clip(alpha * point_quality, 0.0, 1.0)
        candidate = v432.normalize_rows(
            baseline * (1.0 - effective[:, None])
            + calibrated * effective[:, None]
        )
        candidates[float(alpha)] = candidate
        score_grid[float(alpha)] = v432.row_cosine(truth, candidate)
    selected_alpha = max(score_grid, key=lambda value: float(np.mean(score_grid[value])))
    selected = candidates[selected_alpha]
    selected_cos = score_grid[selected_alpha]
    baseline_delay, baseline_angle = marginal_cosine(truth, baseline)
    selected_delay, selected_angle = marginal_cosine(truth, selected)

    fold_rows = []
    for fold in range(5):
        mask = fold_id == fold
        fold_rows.append(
            {
                "fold": fold,
                "count": int(np.sum(mask)),
                "baseline_joint_cosine": float(np.mean(baseline_cos[mask])),
                "selected_joint_cosine": float(np.mean(selected_cos[mask])),
                "joint_gain": float(np.mean(selected_cos[mask] - baseline_cos[mask])),
                "delay_gain": float(np.mean(selected_delay[mask] - baseline_delay[mask])),
                "angle_gain": float(np.mean(selected_angle[mask] - baseline_angle[mask])),
            }
        )

    mean_gain = float(np.mean(selected_cos - baseline_cos))
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "v424 used raw test-coordinate Sionna S-H support. Official-train calibration may "
            "remove systematic Sionna angular-delay bias and add reusable joint path information "
            "beyond official-H spatial interpolation."
        ),
        "evidence_boundary": {
            "selection": "five-fold official-train holdout",
            "test_sionna_used": False,
            "online_feedback_used": False,
        },
        "selected_alpha": float(selected_alpha),
        "joint_sh": {
            "baseline": v432.summarize(baseline_cos),
            "raw_sionna": v432.summarize(raw_cos),
            "calibrated_sionna": v432.summarize(calibrated_cos),
            "selected": v432.summarize(selected_cos),
            "mean_gain": mean_gain,
            "fold_wins": int(sum(float(row["joint_gain"]) > 0.0 for row in fold_rows)),
            "delay_gain": float(np.mean(selected_delay - baseline_delay)),
            "angle_gain": float(np.mean(selected_angle - baseline_angle)),
            "by_regime": {
                regime: {
                    "count": int(np.sum(regimes == regime)),
                    "baseline": float(np.mean(baseline_cos[regimes == regime])),
                    "selected": float(np.mean(selected_cos[regimes == regime])),
                    "gain": float(np.mean(selected_cos[regimes == regime] - baseline_cos[regimes == regime])),
                }
                for regime in REGIMES
            },
        },
        "quality": {
            "point_quality": v432.summarize(point_quality),
            "cell_quality": v432.summarize(cell_quality),
        },
        "folds": fold_rows,
        "deployment_gate": {
            "pass": bool(
                mean_gain >= 0.005
                and sum(float(row["joint_gain"]) > 0.0 for row in fold_rows) >= 4
                and float(np.mean(selected_delay - baseline_delay)) >= 0.0
                and float(np.mean(selected_angle - baseline_angle)) >= 0.0
            )
        },
        "outputs": {
            "oof": PATHS.relative(OOF_CACHE),
            "report": PATHS.relative(REPORT),
        },
    }
    np.savez_compressed(
        OOF_CACHE,
        fold_id=fold_id,
        regimes=regimes,
        point_quality=point_quality,
        cell_quality=cell_quality,
        truth=truth,
        baseline=baseline,
        calibrated=calibrated,
        selected=selected,
        baseline_cosine=baseline_cos,
        selected_cosine=selected_cos,
    )
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
