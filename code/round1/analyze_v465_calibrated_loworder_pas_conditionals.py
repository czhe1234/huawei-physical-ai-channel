#!/usr/bin/env python3
"""Audit train-calibrated low-order leaked-PAS conditionals beyond joint S-H."""

from __future__ import annotations

import json

import numpy as np

import analyze_v401_dualdomain_pas_preserving_confocal as v401
import analyze_v432_sionna_link_energy_calibration as v432
import analyze_v447_sionna_linkwise_pdp_shape_calibration as v447
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 465
VERSION = "v465_calibrated_loworder_pas_conditionals"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"

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
QUALITY_CACHE = PATHS.cache_dir(434) / "v434_sionna_joint_sh_calibration_oof.npz"
GEOMETRY = PATHS.shared_features / "height_geometry_cache.npz"

MARGINS = {
    "psh": {"shape": (2, 192, 16), "conditional_axis": 1, "minimum_gain": 0.002},
    "shue": {"shape": (192, 16, 4), "conditional_axis": 3, "minimum_gain": 0.003},
    "shv": {"shape": (192, 16, 8), "conditional_axis": 3, "minimum_gain": 0.004},
}
ALPHAS = np.linspace(0.0, 0.80, 17, dtype=np.float32)
BUILD_BATCH = 2
DIM_BATCH = 512
QUERY_BATCH = 8
LOG_FLOOR = 1e-10


def require_file(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    return value / np.maximum(value.sum(axis=1, keepdims=True), TINY)


def cache_path(name: str, kind: str):
    return OUT_DIR / f"{name}_{kind}.npy"


def build_marginal_caches() -> None:
    required = [
        cache_path(name, kind)
        for name in MARGINS
        for kind in ("truth", "source")
    ]
    if all(path.is_file() for path in required):
        return
    official = np.load(OFFICIAL, mmap_mode="r")
    high = np.load(HIGH, mmap_mode="r")
    lows = [np.load(path, mmap_mode="r") for path in LOWS]
    outputs = {}
    for name, config in MARGINS.items():
        dimension = int(np.prod(config["shape"]))
        outputs[(name, "truth")] = np.lib.format.open_memmap(
            cache_path(name, "truth"),
            mode="w+",
            dtype=np.float32,
            shape=(len(official), dimension),
        )
        outputs[(name, "source")] = np.lib.format.open_memmap(
            cache_path(name, "source"),
            mode="w+",
            dtype=np.float32,
            shape=(len(official), dimension),
        )
    for start in range(0, len(official), BUILD_BATCH):
        end = min(start + BUILD_BATCH, len(official))
        truth_support, _ = v423.leaked_support(
            np.asarray(official[start:end], dtype=np.complex64)
        )
        high_support, high_total = v423.leaked_support(
            np.asarray(high[start:end], dtype=np.complex64)
        )
        low_loaded = [
            v423.leaked_support(np.asarray(value[start:end], dtype=np.complex64))
            for value in lows
        ]
        truth_margins = v423.leaked_marginals(truth_support)
        high_margins = v423.leaked_marginals(high_support)
        low_margins = [v423.leaked_marginals(value[0]) for value in low_loaded]
        low_valid = np.stack([value[1] > 1e-20 for value in low_loaded], axis=0)
        high_valid = high_total > 1e-20
        for name in MARGINS:
            truth = truth_margins[name].reshape(end - start, -1)
            high_value = high_margins[name].reshape(end - start, -1)
            low_stack = np.stack(
                [value[name].reshape(end - start, -1) for value in low_margins],
                axis=0,
            ).astype(np.float32)
            valid = low_valid[:, :, None].astype(np.float32)
            low_mean = np.sum(low_stack * valid, axis=0) / np.maximum(
                np.sum(valid, axis=0), 1.0
            )
            low_mean = normalize_rows(low_mean)
            source = normalize_rows(0.82 * high_value + 0.18 * low_mean)
            source[~high_valid] = low_mean[~high_valid]
            outputs[(name, "truth")][start:end] = truth
            outputs[(name, "source")][start:end] = source
        if end % 100 == 0 or end == len(official):
            for output in outputs.values():
                output.flush()
            print(f"v465 marginal cache {end}/{len(official)}", flush=True)


def fit_ratio_chunk(
    truth: np.ndarray,
    source: np.ndarray,
    fit_ids: np.ndarray,
    regimes: np.ndarray,
    start: int,
    end: int,
) -> dict[str, np.ndarray]:
    delta = np.clip(
        np.log(np.maximum(np.asarray(truth[fit_ids, start:end]), LOG_FLOOR))
        - np.log(np.maximum(np.asarray(source[fit_ids, start:end]), LOG_FLOOR)),
        -3.0,
        3.0,
    )
    global_ratio = np.median(delta, axis=0).astype(np.float32)
    output = {"global": global_ratio}
    for regime in v447.REGIMES:
        ids = fit_ids[regimes[fit_ids] == regime]
        if len(ids) < 40:
            output[regime] = global_ratio
            continue
        local = np.median(
            np.clip(
                np.log(np.maximum(np.asarray(truth[ids, start:end]), LOG_FLOOR))
                - np.log(np.maximum(np.asarray(source[ids, start:end]), LOG_FLOOR)),
                -3.0,
                3.0,
            ),
            axis=0,
        ).astype(np.float32)
        shrink = np.float32(len(ids) / (len(ids) + 120.0))
        output[regime] = shrink * local + (1.0 - shrink) * global_ratio
    return output


def build_oof_predictions(
    name: str,
    folds: list[np.ndarray],
    positions: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
) -> tuple[np.memmap, np.memmap, np.ndarray]:
    truth = np.load(cache_path(name, "truth"), mmap_mode="r")
    source = np.load(cache_path(name, "source"), mmap_mode="r")
    baseline_path = cache_path(name, "baseline_oof")
    calibrated_path = cache_path(name, "calibrated_oof")
    baseline = np.lib.format.open_memmap(
        baseline_path, mode="w+", dtype=np.float32, shape=truth.shape
    )
    calibrated = np.lib.format.open_memmap(
        calibrated_path, mode="w+", dtype=np.float32, shape=truth.shape
    )
    all_ids = np.arange(len(truth), dtype=np.int64)
    fold_id = np.full(len(truth), -1, dtype=np.int16)
    for fold, query_ids in enumerate(folds):
        fit_ids = np.setdiff1d(all_ids, query_ids)
        fold_id[query_ids] = fold
        neighbor_ids, weights = v447.neighbor_bank(
            fit_ids, query_ids, positions, blocked, regimes
        )
        for start in range(0, truth.shape[1], DIM_BATCH):
            end = min(start + DIM_BATCH, truth.shape[1])
            for query_start in range(0, len(query_ids), QUERY_BATCH):
                query_end = min(query_start + QUERY_BATCH, len(query_ids))
                ids = neighbor_ids[query_start:query_end]
                values = np.asarray(truth[ids, start:end], dtype=np.float32)
                baseline[
                    query_ids[query_start:query_end], start:end
                ] = np.sum(
                    values * weights[query_start:query_end, :, None],
                    axis=1,
                    dtype=np.float64,
                ).astype(np.float32)
            ratio = fit_ratio_chunk(
                truth, source, fit_ids, regimes, start, end
            )
            for regime in v447.REGIMES:
                local = regimes[query_ids] == regime
                if not np.any(local):
                    continue
                ids = query_ids[local]
                calibrated[ids, start:end] = (
                    np.asarray(source[ids, start:end], dtype=np.float32)
                    * np.exp(ratio[regime])[None]
                )
        calibrated[query_ids] = normalize_rows(
            np.asarray(calibrated[query_ids], dtype=np.float32)
        )
        baseline.flush()
        calibrated.flush()
        print(f"v465 {name} fold {fold + 1}/5 complete", flush=True)
    return baseline, calibrated, fold_id


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1, dtype=np.float64)
    ll = np.sum(left * left, axis=1, dtype=np.float64)
    rr = np.sum(right * right, axis=1, dtype=np.float64)
    return (numerator / np.maximum(np.sqrt(ll * rr), 1e-30)).astype(np.float32)


def conditional_cosine(
    name: str,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    config = MARGINS[name]
    shape = (len(truth), *config["shape"])
    truth_grid = np.asarray(truth, dtype=np.float32).reshape(shape)
    prediction_grid = np.asarray(prediction, dtype=np.float32).reshape(shape)
    axis = int(config["conditional_axis"])
    truth_mass = np.sum(truth_grid, axis=axis, dtype=np.float64).astype(np.float32)
    prediction_mass = np.sum(
        prediction_grid, axis=axis, dtype=np.float64
    ).astype(np.float32)
    truth_conditional = truth_grid / np.maximum(
        np.expand_dims(truth_mass, axis=axis), TINY
    )
    prediction_conditional = prediction_grid / np.maximum(
        np.expand_dims(prediction_mass, axis=axis), TINY
    )
    numerator = np.sum(
        truth_conditional * prediction_conditional, axis=axis, dtype=np.float64
    )
    ll = np.sum(truth_conditional**2, axis=axis, dtype=np.float64)
    rr = np.sum(prediction_conditional**2, axis=axis, dtype=np.float64)
    cosine = numerator / np.maximum(np.sqrt(ll * rr), 1e-30)
    weight = truth_mass / np.maximum(
        truth_mass.reshape(len(truth), -1).sum(axis=1)[:, None, None], TINY
    )
    return np.sum(cosine * weight, axis=(1, 2), dtype=np.float64).astype(np.float32)


def joint_cosine(
    name: str,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    config = MARGINS[name]
    shape = (len(truth), *config["shape"])
    axis = int(config["conditional_axis"])
    truth_joint = np.sum(
        np.asarray(truth).reshape(shape), axis=axis, dtype=np.float64
    ).reshape(len(truth), -1).astype(np.float32)
    prediction_joint = np.sum(
        np.asarray(prediction).reshape(shape), axis=axis, dtype=np.float64
    ).reshape(len(truth), -1).astype(np.float32)
    return row_cosine(normalize_rows(truth_joint), normalize_rows(prediction_joint))


def evaluate_margin(
    name: str,
    baseline: np.ndarray,
    calibrated: np.ndarray,
    fold_id: np.ndarray,
    quality: np.ndarray,
) -> dict[str, object]:
    truth = np.load(cache_path(name, "truth"), mmap_mode="r")
    scores = np.zeros((len(ALPHAS), len(truth)), dtype=np.float32)
    for start in range(0, len(truth), 32):
        end = min(start + 32, len(truth))
        truth_block = np.asarray(truth[start:end], dtype=np.float32)
        baseline_block = np.asarray(baseline[start:end], dtype=np.float32)
        calibrated_block = np.asarray(calibrated[start:end], dtype=np.float32)
        for index, alpha in enumerate(ALPHAS):
            effective = np.clip(alpha * quality[start:end], 0.0, 1.0)
            candidate = (
                baseline_block * (1.0 - effective[:, None])
                + calibrated_block * effective[:, None]
            )
            scores[index, start:end] = row_cosine(truth_block, candidate)
    mean_by_alpha = scores.mean(axis=1)
    selected_index = int(np.argmax(mean_by_alpha))
    selected_alpha = float(ALPHAS[selected_index])
    baseline_score = scores[0]
    selected_score = scores[selected_index]
    conditional_baseline = np.empty(len(truth), dtype=np.float32)
    conditional_selected = np.empty(len(truth), dtype=np.float32)
    joint_baseline = np.empty(len(truth), dtype=np.float32)
    joint_selected = np.empty(len(truth), dtype=np.float32)
    for start in range(0, len(truth), 16):
        end = min(start + 16, len(truth))
        truth_block = np.asarray(truth[start:end], dtype=np.float32)
        baseline_block = np.asarray(baseline[start:end], dtype=np.float32)
        effective = np.clip(selected_alpha * quality[start:end], 0.0, 1.0)
        selected_block = (
            baseline_block * (1.0 - effective[:, None])
            + np.asarray(calibrated[start:end], dtype=np.float32)
            * effective[:, None]
        )
        conditional_baseline[start:end] = conditional_cosine(
            name, truth_block, baseline_block
        )
        conditional_selected[start:end] = conditional_cosine(
            name, truth_block, selected_block
        )
        joint_baseline[start:end] = joint_cosine(name, truth_block, baseline_block)
        joint_selected[start:end] = joint_cosine(name, truth_block, selected_block)
    folds = []
    for fold in range(5):
        mask = fold_id == fold
        folds.append(
            {
                "fold": fold,
                "marginal_gain": float(
                    np.mean(selected_score[mask] - baseline_score[mask])
                ),
                "conditional_gain": float(
                    np.mean(conditional_selected[mask] - conditional_baseline[mask])
                ),
                "joint_gain": float(
                    np.mean(joint_selected[mask] - joint_baseline[mask])
                ),
            }
        )
    return {
        "selected_alpha": selected_alpha,
        "baseline_marginal": float(np.mean(baseline_score)),
        "selected_marginal": float(np.mean(selected_score)),
        "marginal_gain": float(np.mean(selected_score - baseline_score)),
        "baseline_conditional": float(np.mean(conditional_baseline)),
        "selected_conditional": float(np.mean(conditional_selected)),
        "conditional_gain": float(
            np.mean(conditional_selected - conditional_baseline)
        ),
        "joint_gain": float(np.mean(joint_selected - joint_baseline)),
        "conditional_fold_wins": int(
            sum(row["conditional_gain"] > 0.0 for row in folds)
        ),
        "folds": folds,
        "mean_by_alpha": mean_by_alpha.tolist(),
    }


def main() -> None:
    for path in (OFFICIAL, HIGH, *LOWS, QUALITY_CACHE, GEOMETRY):
        require_file(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    build_marginal_caches()
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    geometry = np.load(GEOMETRY)
    blocked = geometry["train_blocked"].astype(np.float32)
    regimes = regime_of(blocked).astype("U8")
    all_ids = np.arange(len(positions), dtype=np.int64)
    folds = v401.confirmatory_folds(all_ids, positions[:, :2])
    quality = np.load(QUALITY_CACHE)["point_quality"].astype(np.float32)
    results = {}
    for name in MARGINS:
        baseline, calibrated, fold_id = build_oof_predictions(
            name, folds, positions, blocked, regimes
        )
        results[name] = evaluate_margin(
            name, baseline, calibrated, fold_id, quality
        )
    conditional_mean_gain = float(
        np.mean([results[name]["conditional_gain"] for name in MARGINS])
    )
    marginal_mean_gain = float(
        np.mean([results[name]["marginal_gain"] for name in MARGINS])
    )
    passed = bool(
        conditional_mean_gain >= 0.004
        and marginal_mean_gain >= 0.005
        and all(
            results[name]["conditional_gain"] >= MARGINS[name]["minimum_gain"]
            and results[name]["conditional_fold_wins"] >= 4
            for name in MARGINS
        )
        and all(results[name]["joint_gain"] >= -0.001 for name in MARGINS)
    )
    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Official-train global/regime calibration of polarization, UE-port and "
            "vertical conditionals adds stable PAS information beyond joint S-H."
        ),
        "evidence_boundary": {
            "selection": "five official-train spatial folds",
            "test_sionna_used": False,
            "online_feedback_used_for_parameters": False,
        },
        "parameters": {
            "margins": MARGINS,
            "alphas": ALPHAS.tolist(),
            "source": "0.82 high-budget plus 0.18 valid three-frequency mean",
            "calibration": "global plus regime-shrunk median log truth/source per cell",
        },
        "margins": results,
        "summary": {
            "conditional_mean_gain": conditional_mean_gain,
            "marginal_mean_gain": marginal_mean_gain,
        },
        "decision_rule": {
            "conditional_mean_gain": ">=0.004",
            "marginal_mean_gain": ">=0.005",
            "per_margin": "minimum conditional gain and >=4/5 fold wins",
            "joint_guard": ">=-0.001 for every margin",
            "passed": passed,
        },
        "outputs": {
            "cache_dir": PATHS.relative(OUT_DIR),
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
