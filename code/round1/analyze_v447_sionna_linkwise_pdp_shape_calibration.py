#!/usr/bin/env python3
"""Audit official-train calibration of every Sionna linkwise PDP shape."""

from __future__ import annotations

import json

import numpy as np

import analyze_v401_dualdomain_pas_preserving_confocal as v401
import analyze_v432_sionna_link_energy_calibration as v432
import generate_v426_sionna_linkwise_pdp_completion_diagnostic as v426
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 447
VERSION = "v447_sionna_linkwise_pdp_shape_calibration"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TRUTH_CACHE = CACHE_DIR / "official_linkwise_pdp.npy"
SOURCE_CACHE = CACHE_DIR / "robust_sionna_linkwise_pdp.npy"
QUALITY_CACHE = CACHE_DIR / "sionna_link_quality.npy"
OOF_CACHE = CACHE_DIR / "v447_linkwise_pdp_calibration_oof.npz"

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
QUERY_BATCH = 4
LINK_BATCH = 16
K_NEIGHBORS = 48
LOG_FLOOR = 1e-10


def require_file(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def normalize_links(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    return value / np.maximum(value.sum(axis=-1, keepdims=True), TINY)


def link_cosine(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    numerator = np.sum(truth * prediction, axis=-1, dtype=np.float64)
    left = np.sum(truth * truth, axis=-1, dtype=np.float64)
    right = np.sum(prediction * prediction, axis=-1, dtype=np.float64)
    return (numerator / np.maximum(np.sqrt(left * right), 1e-30)).astype(np.float32)


def build_spectral_cache() -> None:
    if TRUTH_CACHE.is_file() and SOURCE_CACHE.is_file() and QUALITY_CACHE.is_file():
        return
    official = np.load(OFFICIAL, mmap_mode="r")
    high = np.load(HIGH, mmap_mode="r")
    lows = [np.load(path, mmap_mode="r") for path in LOWS]
    count = len(official)
    links = official.shape[1] * official.shape[2]
    delays = official.shape[3]
    truth_out = np.lib.format.open_memmap(
        TRUTH_CACHE, mode="w+", dtype=np.float32, shape=(count, links, delays)
    )
    source_out = np.lib.format.open_memmap(
        SOURCE_CACHE, mode="w+", dtype=np.float32, shape=(count, links, delays)
    )
    quality_out = np.lib.format.open_memmap(
        QUALITY_CACHE, mode="w+", dtype=np.float32, shape=(count, links)
    )
    for start in range(0, count, BUILD_BATCH):
        end = min(start + BUILD_BATCH, count)
        truth_h = np.asarray(official[start:end], dtype=np.complex64)
        high_h = np.asarray(high[start:end], dtype=np.complex64)
        low_h = [np.asarray(value[start:end], dtype=np.complex64) for value in lows]
        truth_shape, _ = v426.linkwise_pdp(truth_h)
        source_shape, diagnostics = v426.robust_link_target(
            truth_h, high_h, low_h
        )
        truth_out[start:end] = truth_shape.reshape(end - start, links, delays)
        source_out[start:end] = source_shape.reshape(end - start, links, delays)
        quality_out[start:end] = diagnostics["link_quality"].reshape(
            end - start, links
        )
        truth_out.flush()
        source_out.flush()
        quality_out.flush()
        if end % 100 == 0 or end == count:
            print(f"v447 cache {end}/{count}", flush=True)


def neighbor_bank(
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
    positions: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xy = positions[:, :2].astype(np.float32)
    distance = np.linalg.norm(
        xy[query_ids, None, :] - xy[None, fit_ids, :], axis=2
    ).astype(np.float32)
    block_delta = np.abs(
        blocked[query_ids, None] - blocked[None, fit_ids]
    ).astype(np.float32)
    same = regimes[query_ids, None] == regimes[None, fit_ids]
    effective = distance + 0.09 * block_delta + 5.0 * (~same)
    local_k = min(K_NEIGHBORS, len(fit_ids))
    nearest = np.argpartition(effective, local_k - 1, axis=1)[:, :local_k]
    rows = np.arange(len(query_ids))[:, None]
    nearest = nearest[rows, np.argsort(effective[rows, nearest], axis=1)]
    selected = effective[rows, nearest]
    selected_same = same[rows, nearest].astype(np.float32)
    weight = np.exp(-selected / 20.0).astype(np.float32)
    weight *= 0.45 + 0.55 * selected_same
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), TINY)
    return fit_ids[nearest], weight


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
    output = np.empty((len(query_ids), end - start, source.shape[-1]), dtype=np.float32)
    for regime in REGIMES:
        mask = regimes[query_ids] == regime
        if not np.any(mask):
            continue
        values = np.asarray(source[query_ids[mask], start:end], dtype=np.float32)
        output[mask] = normalize_links(values * np.exp(ratios[regime])[None])
    return output


def baseline_chunk(
    truth: np.ndarray,
    neighbor_ids: np.ndarray,
    weights: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    output = np.empty((len(neighbor_ids), end - start, truth.shape[-1]), dtype=np.float32)
    for query_start in range(0, len(neighbor_ids), QUERY_BATCH):
        query_end = min(query_start + QUERY_BATCH, len(neighbor_ids))
        ids = neighbor_ids[query_start:query_end]
        values = np.asarray(truth[ids, start:end], dtype=np.float32)
        output[query_start:query_end] = np.sum(
            values * weights[query_start:query_end, :, None, None], axis=1
        )
    return normalize_links(output)


def summarize(value: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, dtype=np.float64)
    return {
        "mean": float(np.mean(value)),
        "p10": float(np.quantile(value, 0.10)),
        "median": float(np.median(value)),
        "p90": float(np.quantile(value, 0.90)),
    }


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
    links = truth.shape[1]

    for fold, query_ids in enumerate(folds):
        fit_ids = np.setdiff1d(all_ids, query_ids)
        fold_id[query_ids] = fold
        neighbor_ids, weights = neighbor_bank(
            fit_ids, query_ids, positions, blocked, regimes
        )
        for link_start in range(0, links, LINK_BATCH):
            link_end = min(link_start + LINK_BATCH, links)
            ratios = fit_ratio_chunk(
                truth, source, fit_ids, regimes, link_start, link_end
            )
            baseline = baseline_chunk(
                truth, neighbor_ids, weights, link_start, link_end
            )
            calibrated = apply_ratio_chunk(
                source, query_ids, regimes, ratios, link_start, link_end
            )
            truth_chunk = np.asarray(
                truth[query_ids, link_start:link_end], dtype=np.float32
            )
            source_chunk = np.asarray(
                source[query_ids, link_start:link_end], dtype=np.float32
            )
            quality_chunk = np.asarray(
                quality[query_ids, link_start:link_end], dtype=np.float32
            )
            baseline_score[query_ids] += link_cosine(truth_chunk, baseline).sum(axis=1)
            raw_score[query_ids] += link_cosine(truth_chunk, source_chunk).sum(axis=1)
            calibrated_score[query_ids] += link_cosine(truth_chunk, calibrated).sum(axis=1)
            for alpha_index, alpha in enumerate(ALPHA_GRID):
                effective = np.clip(alpha * quality_chunk, 0.0, 1.0)
                candidate = normalize_links(
                    baseline * (1.0 - effective[..., None])
                    + calibrated * effective[..., None]
                )
                scores[alpha_index, query_ids] += link_cosine(
                    truth_chunk, candidate
                ).sum(axis=1)
        print(f"v447 fold {fold + 1}/5 complete", flush=True)

    baseline_score /= links
    raw_score /= links
    calibrated_score /= links
    scores /= links
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
            "The corrected leaderboard validates scorer-aligned full linkwise PDP. "
            "Official-train log-ratio calibration may remove systematic Sionna bias "
            "inside every 256x4x192 normalized PDP while retaining spatial interpolation."
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
            "link_batch": LINK_BATCH,
            "ratio": "global plus regime-shrunk median log truth/source per link-delay cell",
        },
        "linkwise_pdp": {
            "baseline": summarize(baseline_score),
            "raw_sionna": summarize(raw_score),
            "calibrated_sionna": summarize(calibrated_score),
            "selected": summarize(selected_score),
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
