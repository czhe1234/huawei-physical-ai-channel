#!/usr/bin/env python3
"""Audit official-train calibration of scorer-aligned conventional PAS."""

from __future__ import annotations

import json

import numpy as np

import analyze_v401_dualdomain_pas_preserving_confocal as v401
import analyze_v447_sionna_linkwise_pdp_shape_calibration as v447
import generate_v427_sionna_conventional_pas_dual_lock_diagnostic as v427
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 448
VERSION = "v448_sionna_conventional_pas_calibration"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TRUTH_CACHE = CACHE_DIR / "official_conventional_pas.npy"
SOURCE_CACHE = CACHE_DIR / "robust_sionna_conventional_pas.npy"
QUALITY_CACHE = CACHE_DIR / "sionna_pas_quality.npy"
OOF_CACHE = CACHE_DIR / "v448_conventional_pas_calibration_oof.npz"

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
BUILD_BATCH = 2
QUERY_BATCH = 8
CELL_BATCH = 32
K_NEIGHBORS = 48
LOG_FLOOR = 1e-10


def require_file(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def flatten_pas(shape: np.ndarray) -> np.ndarray:
    return shape.transpose(0, 4, 5, 1, 2, 3).reshape(len(shape), 4 * 192, 256)


def build_spectral_cache() -> None:
    if TRUTH_CACHE.is_file() and SOURCE_CACHE.is_file() and QUALITY_CACHE.is_file():
        return
    official = np.load(OFFICIAL, mmap_mode="r")
    high = np.load(HIGH, mmap_mode="r")
    lows = [np.load(path, mmap_mode="r") for path in LOWS]
    count = len(official)
    truth_out = np.lib.format.open_memmap(
        TRUTH_CACHE, mode="w+", dtype=np.float32, shape=(count, 4 * 192, 256)
    )
    source_out = np.lib.format.open_memmap(
        SOURCE_CACHE, mode="w+", dtype=np.float32, shape=(count, 4 * 192, 256)
    )
    quality_out = np.lib.format.open_memmap(
        QUALITY_CACHE, mode="w+", dtype=np.float32, shape=(count, 4 * 192)
    )
    for start in range(0, count, BUILD_BATCH):
        end = min(start + BUILD_BATCH, count)
        truth_h = np.asarray(official[start:end], dtype=np.complex64)
        high_h = np.asarray(high[start:end], dtype=np.complex64)
        low_h = [np.asarray(value[start:end], dtype=np.complex64) for value in lows]
        truth_shape, _ = v427.conventional_pas(truth_h)
        source_shape, diagnostics = v427.robust_pas_target(
            truth_h, high_h, low_h
        )
        truth_out[start:end] = flatten_pas(truth_shape)
        source_out[start:end] = flatten_pas(source_shape)
        quality_out[start:end] = diagnostics["cell_quality"].reshape(
            end - start, -1
        )
        truth_out.flush()
        source_out.flush()
        quality_out.flush()
        if end % 100 == 0 or end == count:
            print(f"v448 cache {end}/{count}", flush=True)


def fit_ratio_chunk(
    truth: np.ndarray,
    source: np.ndarray,
    fit_ids: np.ndarray,
    regimes: np.ndarray,
    start: int,
    end: int,
) -> dict[str, np.ndarray]:
    truth_chunk = np.asarray(truth[fit_ids, start:end], dtype=np.float32)
    source_chunk = np.asarray(source[fit_ids, start:end], dtype=np.float32)
    delta = np.clip(
        np.log(np.maximum(truth_chunk, LOG_FLOOR))
        - np.log(np.maximum(source_chunk, LOG_FLOOR)),
        -3.0,
        3.0,
    )
    global_ratio = np.median(delta, axis=0).astype(np.float32)
    ratios = {"global": global_ratio}
    fit_regimes = regimes[fit_ids]
    for regime in REGIMES:
        mask = fit_regimes == regime
        if int(np.sum(mask)) < 40:
            ratios[regime] = global_ratio
            continue
        local = np.median(delta[mask], axis=0).astype(np.float32)
        count = int(np.sum(mask))
        shrink = np.float32(count / (count + 120.0))
        ratios[regime] = shrink * local + (1.0 - shrink) * global_ratio
    return ratios


def apply_ratio_chunk(
    source: np.ndarray,
    query_ids: np.ndarray,
    regimes: np.ndarray,
    ratios: dict[str, np.ndarray],
    start: int,
    end: int,
) -> np.ndarray:
    output = np.empty((len(query_ids), end - start, 256), dtype=np.float32)
    for regime in REGIMES:
        mask = regimes[query_ids] == regime
        if not np.any(mask):
            continue
        values = np.asarray(source[query_ids[mask], start:end], dtype=np.float32)
        output[mask] = v447.normalize_links(
            values * np.exp(ratios[regime])[None]
        )
    return output


def baseline_chunk(
    truth: np.ndarray,
    neighbor_ids: np.ndarray,
    weights: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    output = np.empty((len(neighbor_ids), end - start, 256), dtype=np.float32)
    for query_start in range(0, len(neighbor_ids), QUERY_BATCH):
        query_end = min(query_start + QUERY_BATCH, len(neighbor_ids))
        ids = neighbor_ids[query_start:query_end]
        values = np.asarray(truth[ids, start:end], dtype=np.float32)
        output[query_start:query_end] = np.sum(
            values * weights[query_start:query_end, :, None, None], axis=1
        )
    return v447.normalize_links(output)


def main() -> None:
    for path in (OFFICIAL, HIGH, *LOWS):
        require_file(path)
    build_spectral_cache()
    truth = np.load(TRUTH_CACHE, mmap_mode="r")
    source = np.load(SOURCE_CACHE, mmap_mode="r")
    quality = np.load(QUALITY_CACHE, mmap_mode="r")
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked = geometry["train_blocked"].astype(np.float32)
    regimes = regime_of(blocked).astype("U8")
    all_ids = np.arange(len(positions), dtype=np.int64)
    folds = v401.confirmatory_folds(all_ids, positions[:, :2])
    fold_id = np.full(len(positions), -1, dtype=np.int16)
    scores = np.zeros((len(ALPHA_GRID), len(positions)), dtype=np.float64)
    baseline_score = np.zeros(len(positions), dtype=np.float64)
    raw_score = np.zeros(len(positions), dtype=np.float64)
    calibrated_score = np.zeros(len(positions), dtype=np.float64)
    cells = truth.shape[1]

    for fold, query_ids in enumerate(folds):
        fit_ids = np.setdiff1d(all_ids, query_ids)
        fold_id[query_ids] = fold
        neighbor_ids, weights = v447.neighbor_bank(
            fit_ids, query_ids, positions, blocked, regimes
        )
        for cell_start in range(0, cells, CELL_BATCH):
            cell_end = min(cell_start + CELL_BATCH, cells)
            ratios = fit_ratio_chunk(
                truth, source, fit_ids, regimes, cell_start, cell_end
            )
            baseline = baseline_chunk(
                truth, neighbor_ids, weights, cell_start, cell_end
            )
            calibrated = apply_ratio_chunk(
                source, query_ids, regimes, ratios, cell_start, cell_end
            )
            truth_chunk = np.asarray(
                truth[query_ids, cell_start:cell_end], dtype=np.float32
            )
            source_chunk = np.asarray(
                source[query_ids, cell_start:cell_end], dtype=np.float32
            )
            quality_chunk = np.asarray(
                quality[query_ids, cell_start:cell_end], dtype=np.float32
            )
            baseline_score[query_ids] += v447.link_cosine(
                truth_chunk, baseline
            ).sum(axis=1)
            raw_score[query_ids] += v447.link_cosine(
                truth_chunk, source_chunk
            ).sum(axis=1)
            calibrated_score[query_ids] += v447.link_cosine(
                truth_chunk, calibrated
            ).sum(axis=1)
            for alpha_index, alpha in enumerate(ALPHA_GRID):
                effective = np.clip(alpha * quality_chunk, 0.0, 1.0)
                candidate = v447.normalize_links(
                    baseline * (1.0 - effective[..., None])
                    + calibrated * effective[..., None]
                )
                scores[alpha_index, query_ids] += v447.link_cosine(
                    truth_chunk, candidate
                ).sum(axis=1)
        print(f"v448 fold {fold + 1}/5 complete", flush=True)

    baseline_score /= cells
    raw_score /= cells
    calibrated_score /= cells
    scores /= cells
    mean_by_alpha = scores.mean(axis=1)
    selected_index = int(np.argmax(mean_by_alpha))
    selected_alpha = float(ALPHA_GRID[selected_index])
    selected_score = scores[selected_index]
    fold_rows = []
    for fold in range(5):
        mask = fold_id == fold
        fold_rows.append(
            {
                "fold": fold,
                "count": int(np.sum(mask)),
                "baseline": float(np.mean(baseline_score[mask])),
                "selected": float(np.mean(selected_score[mask])),
                "gain": float(np.mean(selected_score[mask] - baseline_score[mask])),
            }
        )
    mean_gain = float(np.mean(selected_score - baseline_score))
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "v427 failed because it projected raw conventional Sionna PAS. "
            "Official-train calibration may remove the systematic array-angle bias "
            "for every UE and subcarrier under the corrected official PAS mouth."
        ),
        "evidence_boundary": {
            "selection": "five-fold official-train holdout",
            "test_sionna_used": False,
            "online_feedback_used_for_parameters": False,
        },
        "parameters": {
            "alpha_grid": ALPHA_GRID.tolist(),
            "selected_alpha": selected_alpha,
            "neighbors": K_NEIGHBORS,
            "cell_batch": CELL_BATCH,
            "pas_object": "256 array-angle powers independently normalized for each UE and subcarrier",
            "ratio": "global plus regime-shrunk median log truth/source per PAS angle cell",
        },
        "conventional_pas": {
            "baseline": v447.summarize(baseline_score),
            "raw_sionna": v447.summarize(raw_score),
            "calibrated_sionna": v447.summarize(calibrated_score),
            "selected": v447.summarize(selected_score),
            "mean_gain": mean_gain,
            "fold_wins": int(sum(row["gain"] > 0.0 for row in fold_rows)),
            "by_regime": {
                regime: {
                    "count": int(np.sum(regimes == regime)),
                    "baseline": float(np.mean(baseline_score[regimes == regime])),
                    "selected": float(np.mean(selected_score[regimes == regime])),
                    "gain": float(
                        np.mean(
                            selected_score[regimes == regime]
                            - baseline_score[regimes == regime]
                        )
                    ),
                }
                for regime in REGIMES
            },
        },
        "folds": fold_rows,
        "deployment_gate": {
            "pass": bool(
                selected_alpha > 0.0
                and mean_gain >= 0.005
                and sum(row["gain"] > 0.0 for row in fold_rows) >= 4
            ),
            "minimum_mean_gain": 0.005,
            "minimum_fold_wins": 4,
        },
        "outputs": {
            "truth_cache": PATHS.relative(TRUTH_CACHE),
            "source_cache": PATHS.relative(SOURCE_CACHE),
            "quality_cache": PATHS.relative(QUALITY_CACHE),
            "oof": PATHS.relative(OOF_CACHE),
            "report": PATHS.relative(REPORT),
        },
    }
    np.savez_compressed(
        OOF_CACHE,
        fold_id=fold_id,
        regimes=regimes,
        alpha_grid=ALPHA_GRID,
        mean_by_alpha=mean_by_alpha,
        baseline_score=baseline_score,
        raw_score=raw_score,
        calibrated_score=calibrated_score,
        selected_score=selected_score,
    )
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
