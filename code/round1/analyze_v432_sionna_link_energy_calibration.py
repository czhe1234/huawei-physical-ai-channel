#!/usr/bin/env python3
"""Audit train-calibrated Sionna point and link energy on official H."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor

import analyze_v401_dualdomain_pas_preserving_confocal as v401
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 432
VERSION = "v432_sionna_link_energy_calibration"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OOF_CACHE = CACHE_DIR / "v432_sionna_link_energy_calibration_oof.npz"

OFFICIAL = PATHS.round1 / "Round1_Train_Channel.npy"
HIGH = PATHS.cache_dir(208) / "v208_sionna_train_clean_H.npy"
LOW_PATHS = (
    PATHS.cache_dir(205) / "benchmark_sector_m30_track_d5_f2p6" / "train_clean_H.npy",
    PATHS.cache_dir(205) / "benchmark_sector_m30_track_d5_f6p0" / "train_clean_H.npy",
    PATHS.cache_dir(205) / "benchmark_sector_m30_track_d5_f6p7" / "train_clean_H.npy",
)
ALPHA_GRID = np.linspace(0.0, 1.5, 31, dtype=np.float32)
REGIMES = ("clear", "light", "medium", "heavy")
LINK_EPS = 1e-10


def load_energy(path) -> tuple[np.ndarray, np.ndarray]:
    channel = np.load(path, mmap_mode="r")
    point_power = np.empty(len(channel), dtype=np.float64)
    link_share = np.empty((len(channel), 256 * 4), dtype=np.float32)
    for start in range(0, len(channel), 8):
        end = min(start + 8, len(channel))
        block = np.asarray(channel[start:end], dtype=np.complex64)
        link = np.sum(np.abs(block) ** 2, axis=-1, dtype=np.float64).reshape(end - start, -1)
        total = np.maximum(link.sum(axis=1), TINY)
        point_power[start:end] = total
        link_share[start:end] = (link / total[:, None]).astype(np.float32)
    return np.log(np.maximum(point_power, TINY)).astype(np.float32), link_share


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    return value / np.maximum(value.sum(axis=1, keepdims=True), TINY)


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1, dtype=np.float64)
    ll = np.sum(left * left, axis=1, dtype=np.float64)
    rr = np.sum(right * right, axis=1, dtype=np.float64)
    return (numerator / np.maximum(np.sqrt(ll * rr), 1e-30)).astype(np.float32)


def neighbor_prediction(
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
    positions: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
    target: np.ndarray,
    k: int = 48,
) -> np.ndarray:
    xy = positions[:, :2].astype(np.float32)
    distance = np.linalg.norm(
        xy[query_ids, None, :] - xy[None, fit_ids, :], axis=2
    ).astype(np.float32)
    block_delta = np.abs(blocked[query_ids, None] - blocked[None, fit_ids]).astype(np.float32)
    same = regimes[query_ids, None] == regimes[None, fit_ids]
    effective = distance + 0.09 * block_delta + 5.0 * (~same)
    local_k = min(k, len(fit_ids))
    nearest = np.argpartition(effective, local_k - 1, axis=1)[:, :local_k]
    rows = np.arange(len(query_ids))[:, None]
    nearest = nearest[rows, np.argsort(effective[rows, nearest], axis=1)]
    selected = effective[rows, nearest]
    selected_same = same[rows, nearest].astype(np.float32)
    weight = np.exp(-selected / 20.0).astype(np.float32) * (0.45 + 0.55 * selected_same)
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), TINY)
    values = target[fit_ids[nearest]]
    return np.sum(values * weight[..., None], axis=1).astype(np.float32)


def fit_ratio(
    truth: np.ndarray,
    source: np.ndarray,
    fit_ids: np.ndarray,
    regimes: np.ndarray,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    global_ratio = np.median(
        np.clip(
            np.log(np.maximum(truth[fit_ids], LINK_EPS))
            - np.log(np.maximum(source[fit_ids], LINK_EPS)),
            -3.0,
            3.0,
        ),
        axis=0,
    ).astype(np.float32)
    output["global"] = global_ratio
    for regime in REGIMES:
        ids = fit_ids[regimes[fit_ids] == regime]
        if len(ids) < 40:
            output[regime] = global_ratio
            continue
        local = np.median(
            np.clip(
                np.log(np.maximum(truth[ids], LINK_EPS))
                - np.log(np.maximum(source[ids], LINK_EPS)),
                -3.0,
                3.0,
            ),
            axis=0,
        ).astype(np.float32)
        shrink = np.float32(len(ids) / (len(ids) + 120.0))
        output[regime] = shrink * local + (1.0 - shrink) * global_ratio
    return output


def apply_ratio(
    source: np.ndarray,
    query_ids: np.ndarray,
    regimes: np.ndarray,
    ratios: dict[str, np.ndarray],
) -> np.ndarray:
    output = np.empty((len(query_ids), source.shape[1]), dtype=np.float32)
    for regime in REGIMES:
        local = regimes[query_ids] == regime
        if not np.any(local):
            continue
        output[local] = source[query_ids[local]] * np.exp(ratios[regime])[None, :]
    return normalize_rows(output)


def multi_target(
    high: np.ndarray,
    lows: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = np.stack(lows, axis=0).astype(np.float32)
    valid = stack > LINK_EPS
    coverage = valid.mean(axis=0).astype(np.float32)
    low_log = np.log(np.maximum(stack, LINK_EPS))
    spread = np.std(low_log, axis=0).astype(np.float32)
    cell_quality = coverage * np.exp(-0.55 * spread)
    low_mean = np.sum(stack * valid, axis=0) / np.maximum(valid.sum(axis=0), 1.0)
    low_mean = normalize_rows(low_mean)
    high_multi = row_cosine(high, low_mean)
    point_quality = np.clip(
        0.45 * coverage.mean(axis=1)
        + 0.30 * np.quantile(cell_quality, 0.75, axis=1)
        + 0.25 * high_multi,
        0.0,
        1.0,
    ).astype(np.float32)
    target = normalize_rows(
        high * (1.0 - 0.22 * cell_quality)
        + low_mean * (0.22 * cell_quality)
    )
    return target, point_quality, cell_quality


def power_features(
    positions: np.ndarray,
    blocked: np.ndarray,
    logs: list[np.ndarray],
) -> np.ndarray:
    rel = positions - np.asarray([50.0, 0.0, 25.0], dtype=np.float32)[None, :]
    distance = np.linalg.norm(rel, axis=1)
    azimuth = np.arctan2(rel[:, 1], rel[:, 0])
    stack = np.stack(logs, axis=1)
    return np.column_stack(
        [
            positions[:, :2] / 250.0,
            rel[:, :2] / 250.0,
            distance / 350.0,
            np.log1p(distance),
            np.sin(azimuth),
            np.cos(azimuth),
            blocked / 250.0,
            np.log1p(blocked),
            stack,
            stack.std(axis=1),
            stack.max(axis=1) - stack.min(axis=1),
        ]
    ).astype(np.float32)


def summarize(value: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, dtype=np.float64)
    return {
        "mean": float(np.mean(value)),
        "p10": float(np.quantile(value, 0.10)),
        "median": float(np.median(value)),
        "p90": float(np.quantile(value, 0.90)),
    }


def main() -> None:
    for path in (OFFICIAL, HIGH, *LOW_PATHS):
        if not path.is_file():
            raise FileNotFoundError(path)

    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    blocked = geometry["train_blocked"].astype(np.float32)
    regimes = regime_of(blocked).astype("U8")
    official_log, official_share = load_energy(OFFICIAL)
    high_log, high_share = load_energy(HIGH)
    low_loaded = [load_energy(path) for path in LOW_PATHS]
    low_logs = [item[0] for item in low_loaded]
    low_shares = [item[1] for item in low_loaded]
    raw_multi, point_quality, cell_quality = multi_target(high_share, low_shares)
    features = power_features(positions, blocked, [high_log, *low_logs])

    valid_ids = np.arange(len(positions), dtype=np.int64)
    folds = v401.confirmatory_folds(valid_ids, positions[:, :2])
    fold_id = np.full(len(positions), -1, dtype=np.int16)
    baseline_share = np.zeros_like(official_share)
    sionna_share = np.zeros_like(official_share)
    baseline_log = np.zeros(len(positions), dtype=np.float32)
    model_log = np.zeros(len(positions), dtype=np.float32)

    all_ids = np.arange(len(positions), dtype=np.int64)
    fold_rows: list[dict[str, object]] = []
    for fold, query_ids in enumerate(folds):
        fit_ids = np.setdiff1d(all_ids, query_ids)
        fold_id[query_ids] = fold
        baseline_share[query_ids] = normalize_rows(
            neighbor_prediction(
                fit_ids, query_ids, positions, blocked, regimes, official_share
            )
        )
        baseline_log[query_ids] = neighbor_prediction(
            fit_ids,
            query_ids,
            positions,
            blocked,
            regimes,
            official_log[:, None],
        )[:, 0]
        ratio = fit_ratio(official_share, raw_multi, fit_ids, regimes)
        sionna_share[query_ids] = apply_ratio(raw_multi, query_ids, regimes, ratio)
        model = ExtraTreesRegressor(
            n_estimators=360,
            min_samples_leaf=6,
            max_features=0.85,
            n_jobs=-1,
            random_state=4320 + fold,
        )
        model.fit(features[fit_ids], official_log[fit_ids])
        model_log[query_ids] = model.predict(features[query_ids]).astype(np.float32)
        fold_rows.append({"fold": fold, "fit": int(len(fit_ids)), "query": int(len(query_ids))})

    base_cos = row_cosine(official_share, baseline_share)
    sionna_cos = row_cosine(official_share, sionna_share)
    raw_cos = row_cosine(official_share, raw_multi)
    blended_scores: dict[float, np.ndarray] = {}
    for alpha in ALPHA_GRID:
        effective = np.clip(alpha * point_quality, 0.0, 1.0)
        candidate = normalize_rows(
            baseline_share * (1.0 - effective[:, None])
            + sionna_share * effective[:, None]
        )
        blended_scores[float(alpha)] = row_cosine(official_share, candidate)
    selected_alpha = max(
        blended_scores,
        key=lambda alpha: float(np.mean(blended_scores[alpha])),
    )
    selected_cos = blended_scores[selected_alpha]

    power_alpha_grid = np.linspace(0.0, 1.0, 21, dtype=np.float32)
    power_mae = {}
    for alpha in power_alpha_grid:
        prediction = baseline_log + alpha * (model_log - baseline_log)
        power_mae[float(alpha)] = float(np.mean(np.abs(prediction - official_log)))
    selected_power_alpha = min(power_mae, key=power_mae.get)
    selected_log = baseline_log + selected_power_alpha * (model_log - baseline_log)

    fold_summary = []
    for row in fold_rows:
        mask = fold_id == row["fold"]
        row.update(
            {
                "baseline_link_cosine": float(np.mean(base_cos[mask])),
                "raw_sionna_link_cosine": float(np.mean(raw_cos[mask])),
                "calibrated_sionna_link_cosine": float(np.mean(sionna_cos[mask])),
                "selected_link_cosine": float(np.mean(selected_cos[mask])),
                "link_gain": float(np.mean(selected_cos[mask] - base_cos[mask])),
                "baseline_power_mae": float(np.mean(np.abs(baseline_log[mask] - official_log[mask]))),
                "selected_power_mae": float(np.mean(np.abs(selected_log[mask] - official_log[mask]))),
            }
        )
        fold_summary.append(row)

    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "v426 completes each link's normalized PDP but preserves the old 256x4 link energies. "
            "Official-train calibration should reveal whether high-budget and multi-frequency Sionna "
            "contain reusable link-energy or point-power information beyond spatial interpolation."
        ),
        "evidence_boundary": {
            "selection": "five-fold official-train holdout",
            "test_sionna_used": False,
            "online_feedback_used": False,
        },
        "selected": {
            "link_alpha_scale": float(selected_alpha),
            "point_power_alpha": float(selected_power_alpha),
        },
        "link_energy": {
            "baseline_cosine": summarize(base_cos),
            "raw_sionna_cosine": summarize(raw_cos),
            "calibrated_sionna_cosine": summarize(sionna_cos),
            "selected_cosine": summarize(selected_cos),
            "mean_gain": float(np.mean(selected_cos - base_cos)),
            "fold_wins": int(sum(float(row["link_gain"]) > 0.0 for row in fold_summary)),
            "by_regime": {
                regime: {
                    "count": int(np.sum(regimes == regime)),
                    "baseline": float(np.mean(base_cos[regimes == regime])),
                    "selected": float(np.mean(selected_cos[regimes == regime])),
                    "gain": float(np.mean(selected_cos[regimes == regime] - base_cos[regimes == regime])),
                }
                for regime in REGIMES
            },
        },
        "point_power": {
            "baseline_mae_log": float(np.mean(np.abs(baseline_log - official_log))),
            "model_mae_log": float(np.mean(np.abs(model_log - official_log))),
            "selected_mae_log": float(np.mean(np.abs(selected_log - official_log))),
            "baseline_spearman": float(spearmanr(official_log, baseline_log).statistic),
            "model_spearman": float(spearmanr(official_log, model_log).statistic),
            "selected_spearman": float(spearmanr(official_log, selected_log).statistic),
            "fold_wins": int(
                sum(
                    float(row["selected_power_mae"]) < float(row["baseline_power_mae"])
                    for row in fold_summary
                )
            ),
        },
        "quality": {
            "point_quality": summarize(point_quality),
            "cell_quality": summarize(cell_quality),
        },
        "folds": fold_summary,
        "deployment_gate": {
            "link_pass": bool(
                np.mean(selected_cos - base_cos) >= 0.002
                and sum(float(row["link_gain"]) > 0.0 for row in fold_summary) >= 4
            ),
            "power_pass": bool(
                np.mean(np.abs(selected_log - official_log))
                <= np.mean(np.abs(baseline_log - official_log)) - 0.005
                and sum(
                    float(row["selected_power_mae"]) < float(row["baseline_power_mae"])
                    for row in fold_summary
                )
                >= 4
            ),
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
        baseline_link_share=baseline_share,
        calibrated_sionna_link_share=sionna_share,
        baseline_link_cosine=base_cos,
        selected_link_cosine=selected_cos,
        official_log_power=official_log,
        baseline_log_power=baseline_log,
        model_log_power=model_log,
        selected_log_power=selected_log,
    )
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
