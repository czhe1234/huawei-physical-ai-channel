#!/usr/bin/env python3
"""Learn which calibrated Sionna linkwise-PDP actions transfer on official train H."""

from __future__ import annotations

import json
import zipfile

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import analyze_v208_strict_historical_replay as v208
import analyze_v401_dualdomain_pas_preserving_confocal as v401
import analyze_v447_sionna_linkwise_pdp_shape_calibration as v447
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v425_sionna_full_slice_pas_completion_diagnostic as v425
import generate_v426_sionna_linkwise_pdp_completion_diagnostic as v426
import generate_v435_train_calibrated_sionna_joint_sh_diagnostic as v435
import generate_v458_sionna_multimouth_extreme_diagnostic as v458
from generate_v159_frontier_consensus_support_soup import digest
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 459
VERSION = "v459_link_action_response_selector"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = OUT_DIR / "v459_link_features.npy"
BEST_ALPHA = OUT_DIR / "v459_best_alpha.npy"
BEST_GAIN = OUT_DIR / "v459_best_gain.npy"
SCORE_GRID_CACHE = OUT_DIR / "v459_score_grid.npy"
FOLD_CACHE = OUT_DIR / "v459_fold_id.npy"
OOF_PREDICTION = OUT_DIR / "v459_oof_prediction.npz"
OUTPUT = OUT_DIR / "v459_Round1_Test_Channel.npy"
SUPPORT_OUTPUT = OUT_DIR / "v459_selected_support.npy"
ACTION_OUT = OUT_DIR / "v459_link_selector_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

BASE = PATHS.cache_dir(426) / "v426_Round1_Test_Channel.npy"
V458 = PATHS.cache_dir(458) / "v458_Round1_Test_Channel.npy"
JOINT_ACTIONS = PATHS.cache_dir(435) / "v435_train_calibrated_joint_sh_actions.npz"
POWER_ACTIONS = PATHS.cache_dir(433) / "v433_sionna_calibrated_power_release_actions.npz"
GEOMETRY = PATHS.shared_features / "height_geometry_cache.npz"
TRAIN_POS = PATHS.round1 / "Round1_Train_Pos.npy"
TEST_POS = PATHS.round1 / "Round1_Test_Pos.npy"

ALPHA_GRID = np.asarray([0.0, 0.10, 0.25, 0.40, 0.60, 0.80, 1.0], dtype=np.float32)
LINK_BATCH = 16
POINT_BATCH = 2
FEATURE_COUNT = 28
TRAIN_SAMPLE_PER_FOLD = 240_000
FULL_TRAIN_SAMPLE = 600_000
RANDOM_SEED = 45_901
POWER_SCALE = 10.0
JOINT_CYCLES = 5


def require_file(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def normalize(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    return value / np.maximum(value.sum(axis=-1, keepdims=True), TINY)


def link_grid_features(link_ids: np.ndarray) -> np.ndarray:
    antenna = link_ids // 4
    ue = link_ids % 4
    pol = antenna // (16 * 8)
    rem = antenna % (16 * 8)
    horizontal = rem // 8
    vertical = rem % 8
    return np.column_stack(
        [
            pol.astype(np.float32),
            ue.astype(np.float32) / 3.0,
            horizontal.astype(np.float32) / 15.0,
            vertical.astype(np.float32) / 7.0,
            np.sin(2.0 * np.pi * horizontal / 16.0),
            np.cos(2.0 * np.pi * horizontal / 16.0),
            np.sin(2.0 * np.pi * vertical / 8.0),
            np.cos(2.0 * np.pi * vertical / 8.0),
        ]
    ).astype(np.float32)


def spectrum_features(
    raw: np.ndarray,
    calibrated: np.ndarray,
    quality: np.ndarray,
    positions: np.ndarray,
    blocked: np.ndarray,
    regimes: np.ndarray,
    link_ids: np.ndarray,
) -> np.ndarray:
    raw = normalize(raw)
    calibrated = normalize(calibrated)
    bins = np.arange(raw.shape[-1], dtype=np.float32)
    raw_peak = np.argmax(raw, axis=-1).astype(np.float32) / 191.0
    calibrated_peak = np.argmax(calibrated, axis=-1).astype(np.float32) / 191.0
    raw_mean = np.sum(raw * bins, axis=-1) / 191.0
    calibrated_mean = np.sum(calibrated * bins, axis=-1) / 191.0
    raw_var = np.sum(raw * (bins[None, None] / 191.0 - raw_mean[..., None]) ** 2, axis=-1)
    calibrated_var = np.sum(
        calibrated * (bins[None, None] / 191.0 - calibrated_mean[..., None]) ** 2,
        axis=-1,
    )
    raw_entropy = -np.sum(raw * np.log(np.maximum(raw, 1e-12)), axis=-1) / np.log(192.0)
    calibrated_entropy = -np.sum(
        calibrated * np.log(np.maximum(calibrated, 1e-12)), axis=-1
    ) / np.log(192.0)
    relation = v447.link_cosine(raw, calibrated)
    l1 = np.sum(np.abs(raw - calibrated), axis=-1)
    point_count, link_count = raw.shape[:2]
    distance = np.linalg.norm(positions[:, :2] - np.asarray([50.0, 0.0]), axis=1) / 400.0
    point = np.column_stack(
        [
            distance,
            np.clip(blocked / 200.0, 0.0, 2.0),
            (regimes == "clear").astype(np.float32),
            (regimes == "light").astype(np.float32),
            (regimes == "medium").astype(np.float32),
            (regimes == "heavy").astype(np.float32),
        ]
    )
    point = np.broadcast_to(point[:, None, :], (point_count, link_count, 6))
    link = np.broadcast_to(
        link_grid_features(link_ids)[None], (point_count, link_count, 8)
    )
    features = np.concatenate(
        [
            quality[..., None],
            relation[..., None],
            l1[..., None],
            raw_peak[..., None],
            calibrated_peak[..., None],
            np.abs(raw_peak - calibrated_peak)[..., None],
            raw_mean[..., None],
            calibrated_mean[..., None],
            np.sqrt(np.maximum(raw_var, 0.0))[..., None],
            np.sqrt(np.maximum(calibrated_var, 0.0))[..., None],
            raw_entropy[..., None],
            calibrated_entropy[..., None],
            np.max(raw, axis=-1)[..., None],
            np.max(calibrated, axis=-1)[..., None],
            point,
            link,
        ],
        axis=-1,
    ).astype(np.float32)
    if features.shape[-1] != FEATURE_COUNT:
        raise ValueError(f"Unexpected feature count {features.shape}")
    return features


def build_training_cache() -> None:
    if all(path.is_file() for path in (FEATURES, BEST_ALPHA, BEST_GAIN, SCORE_GRID_CACHE, FOLD_CACHE)):
        return
    truth = np.load(v447.TRUTH_CACHE, mmap_mode="r")
    source = np.load(v447.SOURCE_CACHE, mmap_mode="r")
    quality = np.load(v447.QUALITY_CACHE, mmap_mode="r")
    positions = np.load(TRAIN_POS).astype(np.float32)
    geometry = np.load(GEOMETRY)
    blocked = geometry["train_blocked"].astype(np.float32)
    regimes = regime_of(blocked).astype("U8")
    all_ids = np.arange(len(positions), dtype=np.int64)
    folds = v401.confirmatory_folds(all_ids, positions[:, :2])
    fold_id = np.full(len(positions), -1, dtype=np.int16)
    for fold, ids in enumerate(folds):
        fold_id[ids] = fold
    np.save(FOLD_CACHE, fold_id)
    links = truth.shape[1]
    feature_out = np.lib.format.open_memmap(
        FEATURES, mode="w+", dtype=np.float32, shape=(len(truth), links, FEATURE_COUNT)
    )
    alpha_out = np.lib.format.open_memmap(
        BEST_ALPHA, mode="w+", dtype=np.float16, shape=(len(truth), links)
    )
    gain_out = np.lib.format.open_memmap(
        BEST_GAIN, mode="w+", dtype=np.float32, shape=(len(truth), links)
    )
    score_out = np.lib.format.open_memmap(
        SCORE_GRID_CACHE,
        mode="w+",
        dtype=np.float32,
        shape=(len(truth), links, len(ALPHA_GRID)),
    )
    for fold, query_ids in enumerate(folds):
        fit_ids = np.setdiff1d(all_ids, query_ids)
        for start in range(0, links, LINK_BATCH):
            end = min(start + LINK_BATCH, links)
            ratios = v447.fit_ratio_chunk(
                truth, source, fit_ids, regimes, start, end
            )
            raw = np.asarray(source[query_ids, start:end], dtype=np.float32)
            calibrated = v447.apply_ratio_chunk(
                source, query_ids, regimes, ratios, start, end
            )
            truth_chunk = np.asarray(truth[query_ids, start:end], dtype=np.float32)
            score_grid = np.empty(
                (len(query_ids), end - start, len(ALPHA_GRID)), dtype=np.float32
            )
            for index, alpha in enumerate(ALPHA_GRID):
                candidate = normalize((1.0 - alpha) * raw + alpha * calibrated)
                score_grid[..., index] = v447.link_cosine(truth_chunk, candidate)
            best_index = np.argmax(score_grid, axis=-1)
            best_score = np.take_along_axis(
                score_grid, best_index[..., None], axis=-1
            )[..., 0]
            feature_out[query_ids, start:end] = spectrum_features(
                raw,
                calibrated,
                np.asarray(quality[query_ids, start:end], dtype=np.float32),
                positions[query_ids],
                blocked[query_ids],
                regimes[query_ids],
                np.arange(start, end, dtype=np.int64),
            )
            alpha_out[query_ids, start:end] = ALPHA_GRID[best_index].astype(np.float16)
            gain_out[query_ids, start:end] = best_score - score_grid[..., 0]
            score_out[query_ids, start:end] = score_grid
        feature_out.flush()
        alpha_out.flush()
        gain_out.flush()
        score_out.flush()
        print(f"v459 training cache fold {fold + 1}/5", flush=True)


def fit_model(features: np.ndarray, target: np.ndarray, seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.07,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        l2_regularization=2.0,
        random_state=seed,
    ).fit(features, target)


def interpolate_scores(score_grid: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    upper = np.searchsorted(ALPHA_GRID, alpha, side="right")
    upper = np.clip(upper, 1, len(ALPHA_GRID) - 1)
    lower = upper - 1
    left = ALPHA_GRID[lower]
    right = ALPHA_GRID[upper]
    weight = (alpha - left) / np.maximum(right - left, 1e-8)
    low_score = np.take_along_axis(score_grid, lower[..., None], axis=-1)[..., 0]
    high_score = np.take_along_axis(score_grid, upper[..., None], axis=-1)[..., 0]
    return low_score * (1.0 - weight) + high_score * weight


def train_oof_selector() -> tuple[dict[str, object], HistGradientBoostingRegressor, HistGradientBoostingRegressor]:
    features = np.load(FEATURES, mmap_mode="r")
    best_alpha = np.load(BEST_ALPHA, mmap_mode="r")
    best_gain = np.load(BEST_GAIN, mmap_mode="r")
    score_grid = np.load(SCORE_GRID_CACHE, mmap_mode="r")
    fold_id = np.load(FOLD_CACHE)
    rng = np.random.default_rng(RANDOM_SEED)
    pred_alpha = np.zeros(best_alpha.shape, dtype=np.float32)
    pred_gain = np.zeros(best_gain.shape, dtype=np.float32)
    fold_rows = []
    for fold in range(5):
        train_points = np.flatnonzero(fold_id != fold)
        query_points = np.flatnonzero(fold_id == fold)
        train_rows = train_points[:, None] * best_alpha.shape[1] + np.arange(best_alpha.shape[1])[None]
        train_rows = train_rows.reshape(-1)
        sample_count = min(TRAIN_SAMPLE_PER_FOLD, len(train_rows))
        sample = rng.choice(train_rows, sample_count, replace=False)
        flat_features = features.reshape(-1, FEATURE_COUNT)
        flat_alpha = best_alpha.reshape(-1)
        flat_gain = best_gain.reshape(-1)
        alpha_model = fit_model(
            np.asarray(flat_features[sample], dtype=np.float32),
            np.asarray(flat_alpha[sample], dtype=np.float32),
            RANDOM_SEED + fold,
        )
        gain_model = fit_model(
            np.asarray(flat_features[sample], dtype=np.float32),
            np.asarray(flat_gain[sample], dtype=np.float32),
            RANDOM_SEED + 100 + fold,
        )
        for start in range(0, len(query_points), 32):
            ids = query_points[start : start + 32]
            value = np.asarray(features[ids], dtype=np.float32).reshape(-1, FEATURE_COUNT)
            pred_alpha[ids] = alpha_model.predict(value).reshape(len(ids), -1)
            pred_gain[ids] = gain_model.predict(value).reshape(len(ids), -1)
        fold_rows.append({"fold": fold, "query_points": int(len(query_points)), "train_rows": int(sample_count)})
        print(f"v459 selector fold {fold + 1}/5", flush=True)
    grid_rows = []
    for quantile in (0.0, 0.25, 0.50, 0.70, 0.85, 0.93):
        threshold = float(np.quantile(pred_gain, quantile))
        gate = pred_gain >= threshold
        for scale in (0.5, 1.0, 1.5, 2.0):
            alpha = np.where(gate, np.clip(scale * pred_alpha, 0.0, 1.0), 0.0)
            selected_score = interpolate_scores(score_grid, alpha)
            grid_rows.append(
                {
                    "gain_quantile": float(quantile),
                    "gain_threshold": threshold,
                    "alpha_scale": float(scale),
                    "active_fraction": float(np.mean(gate)),
                    "mean_score": float(np.mean(selected_score)),
                    "gain_vs_raw": float(np.mean(selected_score - score_grid[..., 0])),
                    "positive_folds": int(
                        sum(
                            np.mean(selected_score[fold_id == fold] - score_grid[fold_id == fold, :, 0]) > 0
                            for fold in range(5)
                        )
                    ),
                }
            )
    winner = max(grid_rows, key=lambda row: row["gain_vs_raw"])
    fixed_v447_alpha = 0.15 * np.load(v447.QUALITY_CACHE, mmap_mode="r")
    fixed_v447_score = interpolate_scores(score_grid, fixed_v447_alpha)
    np.savez_compressed(
        OOF_PREDICTION,
        pred_alpha=pred_alpha.astype(np.float16),
        pred_gain=pred_gain.astype(np.float32),
        fold_id=fold_id,
    )
    all_rows = np.arange(best_alpha.size, dtype=np.int64)
    sample = rng.choice(all_rows, min(FULL_TRAIN_SAMPLE, len(all_rows)), replace=False)
    flat_features = features.reshape(-1, FEATURE_COUNT)
    alpha_model = fit_model(
        np.asarray(flat_features[sample], dtype=np.float32),
        np.asarray(best_alpha.reshape(-1)[sample], dtype=np.float32),
        RANDOM_SEED + 500,
    )
    gain_model = fit_model(
        np.asarray(flat_features[sample], dtype=np.float32),
        np.asarray(best_gain.reshape(-1)[sample], dtype=np.float32),
        RANDOM_SEED + 600,
    )
    report = {
        "folds": fold_rows,
        "grid": grid_rows,
        "winner": winner,
        "raw_mean": float(np.mean(score_grid[..., 0])),
        "oracle_mean": float(np.mean(np.max(score_grid, axis=-1))),
        "fixed_v447_mean": float(np.mean(fixed_v447_score)),
        "fixed_v447_gain": float(np.mean(fixed_v447_score - score_grid[..., 0])),
    }
    return report, alpha_model, gain_model


def build_test_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.load(v458.RAW_LINK_TARGET, mmap_mode="r")
    calibrated = np.load(v458.CALIBRATED_LINK_TARGET, mmap_mode="r")
    quality = np.load(v458.LINK_QUALITY, mmap_mode="r")
    positions = np.load(TEST_POS).astype(np.float32)
    geometry = np.load(GEOMETRY)
    blocked = geometry["test_blocked"].astype(np.float32)
    regimes = regime_of(blocked).astype("U8")
    features = np.empty((len(raw), raw.shape[1], FEATURE_COUNT), dtype=np.float32)
    for start in range(0, raw.shape[1], LINK_BATCH):
        end = min(start + LINK_BATCH, raw.shape[1])
        features[:, start:end] = spectrum_features(
            np.asarray(raw[:, start:end], dtype=np.float32),
            np.asarray(calibrated[:, start:end], dtype=np.float32),
            np.asarray(quality[:, start:end], dtype=np.float32),
            positions,
            blocked,
            regimes,
            np.arange(start, end, dtype=np.int64),
        )
    return features, np.asarray(calibrated, dtype=np.float32), regimes, np.asarray(quality, dtype=np.float32)


def generate_candidate(
    alpha_model: HistGradientBoostingRegressor,
    gain_model: HistGradientBoostingRegressor,
    selector: dict[str, float],
) -> dict[str, object]:
    test_features, calibrated_link, regimes, quality = build_test_features()
    flat = test_features.reshape(-1, FEATURE_COUNT)
    predicted_alpha = alpha_model.predict(flat).reshape(test_features.shape[:2]).astype(np.float32)
    predicted_gain = gain_model.predict(flat).reshape(test_features.shape[:2]).astype(np.float32)
    # Use the OOF quantile on test predictions, not the raw numerical threshold,
    # because train/test gain calibration can shift while ranking remains useful.
    threshold = float(np.quantile(predicted_gain, selector["gain_quantile"]))
    active = predicted_gain >= threshold
    effective_alpha = np.where(
        active,
        np.clip(selector["alpha_scale"] * predicted_alpha, 0.0, 1.0),
        0.0,
    ).astype(np.float32)
    source = np.load(BASE, mmap_mode="r")
    joint = np.load(JOINT_ACTIONS)
    point_quality = joint["point_quality"].astype(np.float32)
    calibrated_joint = joint["calibrated_target"].astype(np.float32)
    joint_profile = next(profile for profile in v458.JOINT_PROFILES if profile.name == "tail085")
    joint_alpha = v458.joint_alpha(joint_profile, regimes, point_quality)
    support = np.lib.format.open_memmap(
        SUPPORT_OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape
    )
    for start in range(0, len(source), POINT_BATCH):
        end = min(start + POINT_BATCH, len(source))
        base_h = np.asarray(source[start:end], dtype=np.complex64)
        base_support, _ = v423.leaked_support(base_h)
        desired_support, _, _ = v435.build_desired_support(
            base_support, calibrated_joint[start:end], joint_alpha[start:end]
        )
        base_link, _ = v426.linkwise_pdp(base_h)
        target_link = calibrated_link[start:end].reshape(
            end - start, source.shape[1], source.shape[2], source.shape[3]
        )
        desired_link = v426.desired_linkwise_pdp(
            base_link,
            target_link,
            effective_alpha[start:end].reshape(end - start, source.shape[1], source.shape[2]),
        )
        candidate = base_h.copy()
        for _ in range(JOINT_CYCLES):
            candidate = v425.project_full_support(candidate, desired_support)
            candidate = v426.restore_desired_pdp(candidate, base_h, desired_link)
        support[start:end] = candidate
        if end % 50 == 0 or end == len(source):
            support.flush()
    log_delta = np.load(POWER_ACTIONS)["final_log_delta"].astype(np.float32)
    amplitude = np.exp(0.5 * np.float32(POWER_SCALE) * log_delta).astype(np.float32)
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape
    )
    for start in range(0, len(source), 10):
        end = min(start + 10, len(source))
        value = np.asarray(support[start:end], dtype=np.complex64).copy()
        value *= amplitude[start:end, None, None, None]
        output[start:end] = value
        output.flush()
    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes,
        predicted_alpha=predicted_alpha.astype(np.float16),
        predicted_gain=predicted_gain.astype(np.float32),
        effective_alpha=effective_alpha.astype(np.float16),
        active=active,
        quality=quality.astype(np.float16),
    )
    return {
        "active_fraction": float(np.mean(active)),
        "effective_alpha_mean": float(np.mean(effective_alpha)),
        "effective_alpha_active_mean": float(np.mean(effective_alpha[active])),
        "gain_threshold": threshold,
    }


def write_zip(report: dict[str, object]) -> None:
    dependencies = (
        "generate_v459_link_action_response_selector.py",
        "generate_v458_sionna_multimouth_extreme_diagnostic.py",
        "analyze_v447_sionna_linkwise_pdp_shape_calibration.py",
        "generate_v435_train_calibrated_sionna_joint_sh_diagnostic.py",
        "generate_v426_sionna_linkwise_pdp_completion_diagnostic.py",
        "generate_v425_sionna_full_slice_pas_completion_diagnostic.py",
        "generate_v423_joint_sionna_pas_pdp_diagnostic.py",
        "analyze_v208_strict_historical_replay.py",
        "powerfield_common.py",
        "project_paths.py",
    )
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for name in dependencies:
            archive.write(PATHS.code / name, name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    value = np.load(OUTPUT, mmap_mode="r")
    report["integrity"] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(np.isfinite(value).all()),
        "zip_crc": True,
        "zip_members": members,
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }


def main() -> None:
    for path in (
        v447.TRUTH_CACHE,
        v447.SOURCE_CACHE,
        v447.QUALITY_CACHE,
        v458.CALIBRATED_LINK_TARGET,
        v458.RAW_LINK_TARGET,
        BASE,
        V458,
        JOINT_ACTIONS,
        POWER_ACTIONS,
    ):
        require_file(path)
    build_training_cache()
    oof, alpha_model, gain_model = train_oof_selector()
    selector = oof["winner"]
    action_summary = generate_candidate(alpha_model, gain_model, selector)
    candidate = v208.score_prediction(OUTPUT)
    baseline = v208.score_prediction(V458)
    report = {
        "method": VERSION,
        "status": "candidate_pending_diagnostic_noncompliant",
        "primary_hypothesis": (
            "v458 over-transfer comes from globally correcting links whose calibrated "
            "Sionna PDP does not generalize. A train-H-supervised link action-response "
            "model can keep strong corrections only where they historically transfer."
        ),
        "compliance_boundary": {
            "diagnostic_only": True,
            "explicitly_noncompliant_for_final_submission": True,
            "reason": "candidate still uses Sionna references at the 500 official test coordinates",
        },
        "parameters": {
            "feature_count": FEATURE_COUNT,
            "alpha_grid": ALPHA_GRID.tolist(),
            "train_sample_per_fold": TRAIN_SAMPLE_PER_FOLD,
            "full_train_sample": FULL_TRAIN_SAMPLE,
            "power_scale": POWER_SCALE,
            "joint_profile": "tail085",
            "joint_cycles": JOINT_CYCLES,
        },
        "official_train_oof": oof,
        "test_action": action_summary,
        "baseline_v458": baseline,
        "candidate": candidate,
        "replay_delta_vs_v458": float(
            candidate["score_traceback_leaked"] - baseline["score_traceback_leaked"]
        ),
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "actions": PATHS.relative(ACTION_OUT),
            "oof": PATHS.relative(OOF_PREDICTION),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }
    write_zip(report)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "oof_winner": selector,
        "test_action": action_summary,
        "candidate": candidate,
        "replay_delta_vs_v458": report["replay_delta_vs_v458"],
        "zip": report["outputs"]["zip"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
