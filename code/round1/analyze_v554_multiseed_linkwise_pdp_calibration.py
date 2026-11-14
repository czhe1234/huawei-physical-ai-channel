#!/usr/bin/env python3
"""Cross-fit the incremental multi-seed linkwise-PDP information for v554."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import analyze_v401_dualdomain_pas_preserving_confocal as v401
import analyze_v447_sionna_linkwise_pdp_shape_calibration as v447
import generate_v458_sionna_multimouth_extreme_diagnostic as v458
import generate_v554_multiseed_linkwise_pdp_statistics as source
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 554
VERSION = "v554_multiseed_linkwise_pdp_calibration"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER) / "linkwise_pdp_calibration"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OLD_CALIBRATED_OOF = OUT_DIR / "old_calibrated_linkwise_pdp_oof.npy"
OLD_CALIBRATED_DONE = OUT_DIR / "old_calibrated_linkwise_pdp_oof.done"
OOF_CACHE = OUT_DIR / "v554_multiseed_linkwise_pdp_calibration_oof.npz"
TEST_ACTION = OUT_DIR / "v554_test_linkwise_pdp_log_action.npy"
TEST_TARGET = OUT_DIR / "v554_test_linkwise_pdp_target.npy"
TEST_QUALITY = OUT_DIR / "v554_test_linkwise_pdp_quality.npz"

OFFICIAL_TRUTH = v447.TRUTH_CACHE
OLD_SOURCE = v447.SOURCE_CACHE
OLD_TEST_TARGET = v458.CALIBRATED_LINK_TARGET
REGIMES = v447.REGIMES
CHECKPOINTS = source.CHECKPOINTS
QUALITY_SCALES = (0.0, 4.0, 8.0, 16.0)
ALPHAS = (0.125, 0.25, 0.375, 0.50, 0.75, 1.0)
REGIME_POLICIES = {
    "all": REGIMES,
    "nlos_only": ("medium", "heavy"),
}
LINK_BATCH = 16
LOG_FLOOR = 1e-10
ACTION_CLIP = 3.0
SHUFFLE_SEED = 20260728


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def checkpoint_path(checkpoint: int, split: str, kind: str) -> Path:
    return source.OUT_DIR / "checkpoints" / f"n{checkpoint:02d}_{split}_{kind}.npy"


def as_links(value: np.ndarray) -> np.ndarray:
    return value.reshape(len(value), -1, value.shape[-1])


def summarize(value: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, dtype=np.float64)
    return {
        "mean": float(np.mean(value)),
        "p10": float(np.quantile(value, 0.10)),
        "median": float(np.median(value)),
        "p90": float(np.quantile(value, 0.90)),
        "max": float(np.max(value)),
    }


def source_quality(
    mean: np.ndarray,
    variance: np.ndarray,
    valid_count: np.ndarray,
    checkpoint: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del mean
    instability = np.sqrt(
        np.maximum(np.sum(variance, axis=-1, dtype=np.float64), 0.0)
    ).astype(np.float32)
    coverage = np.clip(
        valid_count.astype(np.float32) / np.float32(checkpoint), 0.0, 1.0
    )
    valid = valid_count > 0
    return instability, coverage, valid


def geometric_action(
    baseline: np.ndarray,
    target: np.ndarray,
    effective: np.ndarray,
) -> np.ndarray:
    log_delta = np.clip(
        np.log(np.maximum(target, LOG_FLOOR)) - np.log(np.maximum(baseline, LOG_FLOOR)),
        -ACTION_CLIP,
        ACTION_CLIP,
    )
    candidate = baseline * np.exp(effective[..., None] * log_delta).astype(np.float32)
    return v447.normalize_links(candidate)


def build_old_calibrated_oof(
    truth: np.ndarray,
    old_source: np.ndarray,
    folds: list[np.ndarray],
    regimes: np.ndarray,
) -> np.memmap:
    if OLD_CALIBRATED_OOF.is_file() and OLD_CALIBRATED_DONE.is_file():
        cached = np.load(OLD_CALIBRATED_OOF, mmap_mode="r")
        if cached.shape == truth.shape:
            return cached
    OLD_CALIBRATED_OOF.unlink(missing_ok=True)
    OLD_CALIBRATED_DONE.unlink(missing_ok=True)
    output = np.lib.format.open_memmap(
        OLD_CALIBRATED_OOF, mode="w+", dtype=np.float32, shape=truth.shape
    )
    all_ids = np.arange(len(truth), dtype=np.int64)
    for fold, query_ids in enumerate(folds):
        fit_ids = np.setdiff1d(all_ids, query_ids)
        for start in range(0, truth.shape[1], LINK_BATCH):
            end = min(start + LINK_BATCH, truth.shape[1])
            ratios = v447.fit_ratio_chunk(
                truth, old_source, fit_ids, regimes, start, end
            )
            output[query_ids, start:end] = v447.apply_ratio_chunk(
                old_source, query_ids, regimes, ratios, start, end
            )
        output.flush()
        print(f"v554 old-source calibration fold {fold + 1}/5", flush=True)
    OLD_CALIBRATED_DONE.write_text("complete\n", encoding="ascii")
    return np.load(OLD_CALIBRATED_OOF, mmap_mode="r")


def evaluate_checkpoint(
    *,
    checkpoint: int,
    truth: np.ndarray,
    old_oof: np.ndarray,
    folds: list[np.ndarray],
    fold_id: np.ndarray,
    regimes: np.ndarray,
) -> dict[str, object]:
    mean = as_links(
        np.load(checkpoint_path(checkpoint, "train", "mean"), mmap_mode="r")
    )
    variance = as_links(
        np.load(checkpoint_path(checkpoint, "train", "variance"), mmap_mode="r")
    )
    valid_count = np.load(
        checkpoint_path(checkpoint, "train", "valid_count"), mmap_mode="r"
    ).reshape(len(truth), -1)
    configs = [
        (quality_scale, alpha) for quality_scale in QUALITY_SCALES for alpha in ALPHAS
    ]
    score = np.zeros((len(configs), len(truth)), dtype=np.float64)
    raw_score = np.zeros(len(truth), dtype=np.float64)
    calibrated_score = np.zeros(len(truth), dtype=np.float64)
    baseline_score = np.zeros(len(truth), dtype=np.float64)
    instability_point = np.zeros(len(truth), dtype=np.float64)
    coverage_point = np.zeros(len(truth), dtype=np.float64)
    links = truth.shape[1]
    all_ids = np.arange(len(truth), dtype=np.int64)

    for fold, query_ids in enumerate(folds):
        fit_ids = np.setdiff1d(all_ids, query_ids)
        for start in range(0, links, LINK_BATCH):
            end = min(start + LINK_BATCH, links)
            ratios = v447.fit_ratio_chunk(truth, mean, fit_ids, regimes, start, end)
            calibrated = v447.apply_ratio_chunk(
                mean, query_ids, regimes, ratios, start, end
            )
            truth_chunk = np.asarray(truth[query_ids, start:end], dtype=np.float32)
            old_chunk = np.asarray(old_oof[query_ids, start:end], dtype=np.float32)
            raw_chunk = np.asarray(mean[query_ids, start:end], dtype=np.float32)
            variance_chunk = np.asarray(
                variance[query_ids, start:end], dtype=np.float32
            )
            count_chunk = np.asarray(valid_count[query_ids, start:end], dtype=np.uint16)
            instability, coverage, valid = source_quality(
                raw_chunk, variance_chunk, count_chunk, checkpoint
            )
            calibrated = np.where(valid[..., None], calibrated, old_chunk)
            baseline_score[query_ids] += v447.link_cosine(truth_chunk, old_chunk).sum(
                axis=1
            )
            raw_score[query_ids] += v447.link_cosine(truth_chunk, raw_chunk).sum(axis=1)
            calibrated_score[query_ids] += v447.link_cosine(
                truth_chunk, calibrated
            ).sum(axis=1)
            instability_point[query_ids] += instability.sum(axis=1)
            coverage_point[query_ids] += coverage.sum(axis=1)
            for config_index, (quality_scale, alpha) in enumerate(configs):
                effective = np.clip(
                    np.float32(alpha)
                    * coverage
                    * np.exp(-np.float32(quality_scale) * instability),
                    0.0,
                    1.0,
                ).astype(np.float32)
                candidate = geometric_action(old_chunk, calibrated, effective)
                score[config_index, query_ids] += v447.link_cosine(
                    truth_chunk, candidate
                ).sum(axis=1)
        print(f"v554 checkpoint={checkpoint} fold {fold + 1}/5", flush=True)

    for value in (score,):
        value /= links
    for value in (
        baseline_score,
        raw_score,
        calibrated_score,
        instability_point,
        coverage_point,
    ):
        value /= links
    grid = []
    best = None
    for config_index, (quality_scale, alpha) in enumerate(configs):
        for policy, active_regimes in REGIME_POLICIES.items():
            active = np.isin(regimes, active_regimes)
            policy_score = np.where(active, score[config_index], baseline_score)
            mean_score = float(np.mean(policy_score))
            grid.append(
                {
                    "quality_scale": float(quality_scale),
                    "alpha": float(alpha),
                    "regime_policy": policy,
                    "score": mean_score,
                    "gain": float(mean_score - np.mean(baseline_score)),
                }
            )
            if best is None or mean_score > best[0]:
                best = (mean_score, config_index, policy, policy_score)
    assert best is not None
    _, best_index, regime_policy, selected = best
    quality_scale, alpha = configs[best_index]
    fold_rows = []
    for fold in range(len(folds)):
        mask = fold_id == fold
        fold_rows.append(
            {
                "fold": fold,
                "count": int(np.sum(mask)),
                "baseline": float(np.mean(baseline_score[mask])),
                "selected": float(np.mean(selected[mask])),
                "gain": float(np.mean(selected[mask] - baseline_score[mask])),
            }
        )
    by_regime = {}
    for regime in REGIMES:
        mask = regimes == regime
        by_regime[regime] = {
            "count": int(np.sum(mask)),
            "baseline": float(np.mean(baseline_score[mask])),
            "selected": float(np.mean(selected[mask])),
            "gain": float(np.mean(selected[mask] - baseline_score[mask])),
        }
    return {
        "checkpoint": checkpoint,
        "baseline": summarize(baseline_score),
        "raw_multiseed": summarize(raw_score),
        "calibrated_multiseed": summarize(calibrated_score),
        "instability": summarize(instability_point),
        "coverage": summarize(coverage_point),
        "grid": grid,
        "selected": {
            "quality_scale": float(quality_scale),
            "alpha": float(alpha),
            "regime_policy": regime_policy,
            "score": float(np.mean(selected)),
            "gain": float(np.mean(selected - baseline_score)),
            "fold_wins": int(sum(row["gain"] > 0.0 for row in fold_rows)),
            "folds": fold_rows,
            "by_regime": by_regime,
        },
        "arrays": {
            "baseline_score": baseline_score,
            "selected_score": selected,
            "grid_scores": score,
            "configs": np.asarray(configs, dtype=np.float32),
        },
    }


def shuffle_permutations(
    folds: list[np.ndarray], regimes: np.ndarray
) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(SHUFFLE_SEED)
    output = {}
    for fold, query_ids in enumerate(folds):
        permutation = query_ids.copy()
        for regime in REGIMES:
            local = np.flatnonzero(regimes[query_ids] == regime)
            permutation[local] = query_ids[rng.permutation(local)]
        output[fold] = permutation
    return output


def shuffled_score(
    *,
    checkpoint: int,
    quality_scale: float,
    alpha: float,
    regime_policy: str,
    truth: np.ndarray,
    old_oof: np.ndarray,
    folds: list[np.ndarray],
    regimes: np.ndarray,
) -> np.ndarray:
    mean = as_links(
        np.load(checkpoint_path(checkpoint, "train", "mean"), mmap_mode="r")
    )
    variance = as_links(
        np.load(checkpoint_path(checkpoint, "train", "variance"), mmap_mode="r")
    )
    valid_count = np.load(
        checkpoint_path(checkpoint, "train", "valid_count"), mmap_mode="r"
    ).reshape(len(truth), -1)
    permutations = shuffle_permutations(folds, regimes)
    result = np.zeros(len(truth), dtype=np.float64)
    all_ids = np.arange(len(truth), dtype=np.int64)
    for fold, query_ids in enumerate(folds):
        fit_ids = np.setdiff1d(all_ids, query_ids)
        shuffled_ids = permutations[fold]
        for start in range(0, truth.shape[1], LINK_BATCH):
            end = min(start + LINK_BATCH, truth.shape[1])
            ratios = v447.fit_ratio_chunk(truth, mean, fit_ids, regimes, start, end)
            calibrated = v447.apply_ratio_chunk(
                mean, shuffled_ids, regimes, ratios, start, end
            )
            truth_chunk = np.asarray(truth[query_ids, start:end], dtype=np.float32)
            old_chunk = np.asarray(old_oof[query_ids, start:end], dtype=np.float32)
            mean_chunk = np.asarray(mean[shuffled_ids, start:end], dtype=np.float32)
            variance_chunk = np.asarray(
                variance[shuffled_ids, start:end], dtype=np.float32
            )
            count_chunk = np.asarray(
                valid_count[shuffled_ids, start:end], dtype=np.uint16
            )
            instability, coverage, valid = source_quality(
                mean_chunk, variance_chunk, count_chunk, checkpoint
            )
            calibrated = np.where(valid[..., None], calibrated, old_chunk)
            effective = np.clip(
                np.float32(alpha)
                * coverage
                * np.exp(-np.float32(quality_scale) * instability),
                0.0,
                1.0,
            ).astype(np.float32)
            active = np.isin(regimes[query_ids], REGIME_POLICIES[regime_policy])
            effective[~active] = 0.0
            candidate = geometric_action(old_chunk, calibrated, effective)
            result[query_ids] += v447.link_cosine(truth_chunk, candidate).sum(axis=1)
    result /= truth.shape[1]
    return result


def build_test_action(
    *,
    checkpoint: int,
    quality_scale: float,
    alpha: float,
    regime_policy: str,
    truth: np.ndarray,
    regimes: np.ndarray,
    test_regimes: np.ndarray,
) -> dict[str, object]:
    train_mean = as_links(
        np.load(checkpoint_path(checkpoint, "train", "mean"), mmap_mode="r")
    )
    test_mean = as_links(
        np.load(checkpoint_path(checkpoint, "test", "mean"), mmap_mode="r")
    )
    test_variance = as_links(
        np.load(checkpoint_path(checkpoint, "test", "variance"), mmap_mode="r")
    )
    test_count = np.load(
        checkpoint_path(checkpoint, "test", "valid_count"), mmap_mode="r"
    ).reshape(len(test_mean), -1)
    old_target = np.load(OLD_TEST_TARGET, mmap_mode="r")
    if old_target.shape != test_mean.shape:
        raise ValueError(
            f"old test target mismatch {old_target.shape} != {test_mean.shape}"
        )
    target_out = np.lib.format.open_memmap(
        TEST_TARGET, mode="w+", dtype=np.float32, shape=test_mean.shape
    )
    action_out = np.lib.format.open_memmap(
        TEST_ACTION, mode="w+", dtype=np.float32, shape=test_mean.shape
    )
    quality_out = np.empty(test_count.shape, dtype=np.float32)
    instability_out = np.empty(test_count.shape, dtype=np.float32)
    coverage_out = np.empty(test_count.shape, dtype=np.float32)
    fit_ids = np.arange(len(truth), dtype=np.int64)
    query_ids = np.arange(len(test_mean), dtype=np.int64)
    for start in range(0, test_mean.shape[1], LINK_BATCH):
        end = min(start + LINK_BATCH, test_mean.shape[1])
        ratios = v447.fit_ratio_chunk(truth, train_mean, fit_ids, regimes, start, end)
        calibrated = v447.apply_ratio_chunk(
            test_mean, query_ids, test_regimes, ratios, start, end
        )
        old_chunk = np.asarray(old_target[:, start:end], dtype=np.float32)
        mean_chunk = np.asarray(test_mean[:, start:end], dtype=np.float32)
        variance_chunk = np.asarray(test_variance[:, start:end], dtype=np.float32)
        count_chunk = np.asarray(test_count[:, start:end], dtype=np.uint16)
        instability, coverage, valid = source_quality(
            mean_chunk, variance_chunk, count_chunk, checkpoint
        )
        calibrated = np.where(valid[..., None], calibrated, old_chunk)
        effective = np.clip(
            np.float32(alpha)
            * coverage
            * np.exp(-np.float32(quality_scale) * instability),
            0.0,
            1.0,
        ).astype(np.float32)
        active = np.isin(test_regimes, REGIME_POLICIES[regime_policy])
        effective[~active] = 0.0
        target = geometric_action(old_chunk, calibrated, effective)
        action = np.clip(
            np.log(np.maximum(target, LOG_FLOOR))
            - np.log(np.maximum(old_chunk, LOG_FLOOR)),
            -ACTION_CLIP,
            ACTION_CLIP,
        ).astype(np.float32)
        target_out[:, start:end] = target
        action_out[:, start:end] = action
        quality_out[:, start:end] = effective
        instability_out[:, start:end] = instability
        coverage_out[:, start:end] = coverage
        if end % 128 == 0 or end == test_mean.shape[1]:
            target_out.flush()
            action_out.flush()
            print(f"v554 test target {end}/{test_mean.shape[1]} links", flush=True)
    np.savez_compressed(
        TEST_QUALITY,
        effective_weight=quality_out,
        instability=instability_out,
        coverage=coverage_out,
    )
    return {
        "effective_weight": summarize(quality_out),
        "instability": summarize(instability_out),
        "coverage": summarize(coverage_out),
        "absolute_log_action": summarize(np.abs(action_out)),
    }


def main() -> None:
    required = [
        OFFICIAL_TRUTH,
        OLD_SOURCE,
        OLD_TEST_TARGET,
        PATHS.round1 / "Round1_Train_Pos.npy",
        PATHS.shared_features / "height_geometry_cache.npz",
    ]
    for checkpoint in CHECKPOINTS:
        for split in ("train", "test"):
            for kind in ("mean", "variance", "valid_count"):
                required.append(checkpoint_path(checkpoint, split, kind))
    for path in required:
        require(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    truth = np.load(OFFICIAL_TRUTH, mmap_mode="r")
    old_source = np.load(OLD_SOURCE, mmap_mode="r")
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    regimes = regime_of(geometry["train_blocked"].astype(np.float32)).astype("U8")
    test_regimes = regime_of(geometry["test_blocked"].astype(np.float32)).astype("U8")
    all_ids = np.arange(len(truth), dtype=np.int64)
    folds = v401.confirmatory_folds(all_ids, positions[:, :2])
    fold_id = np.full(len(truth), -1, dtype=np.int16)
    for fold, ids in enumerate(folds):
        fold_id[ids] = fold
    old_oof = build_old_calibrated_oof(truth, old_source, folds, regimes)
    checkpoint_results = []
    for checkpoint in CHECKPOINTS:
        result = evaluate_checkpoint(
            checkpoint=checkpoint,
            truth=truth,
            old_oof=old_oof,
            folds=folds,
            fold_id=fold_id,
            regimes=regimes,
        )
        checkpoint_results.append(result)
    selected_result = max(checkpoint_results, key=lambda row: row["selected"]["score"])
    selected = selected_result["selected"]
    checkpoint = int(selected_result["checkpoint"])
    shuffled = shuffled_score(
        checkpoint=checkpoint,
        quality_scale=float(selected["quality_scale"]),
        alpha=float(selected["alpha"]),
        regime_policy=str(selected["regime_policy"]),
        truth=truth,
        old_oof=old_oof,
        folds=folds,
        regimes=regimes,
    )
    baseline_score = selected_result["arrays"]["baseline_score"]
    selected_score = selected_result["arrays"]["selected_score"]
    test_diagnostics = build_test_action(
        checkpoint=checkpoint,
        quality_scale=float(selected["quality_scale"]),
        alpha=float(selected["alpha"]),
        regime_policy=str(selected["regime_policy"]),
        truth=truth,
        regimes=regimes,
        test_regimes=test_regimes,
    )
    np.savez_compressed(
        OOF_CACHE,
        fold_id=fold_id,
        regimes=regimes,
        checkpoint=np.int16(checkpoint),
        quality_scale=np.float32(selected["quality_scale"]),
        alpha=np.float32(selected["alpha"]),
        baseline_score=baseline_score,
        selected_score=selected_score,
        shuffled_score=shuffled,
    )
    report_results = []
    for result in checkpoint_results:
        report_results.append(
            {key: value for key, value in result.items() if key != "arrays"}
        )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "A calibrated 24-realization linkwise-PDP expectation contains incremental "
            "delay-path information beyond the calibrated single-realization/multifrequency "
            "physical source already inherited by v549."
        ),
        "evidence_boundary": {
            "selection": "five mutually exclusive official-train spatial folds",
            "online_score_used": False,
            "comparison": "new multi-seed calibrated source versus old calibrated physical source",
        },
        "parameters": {
            "checkpoints": list(CHECKPOINTS),
            "quality_scales": list(QUALITY_SCALES),
            "alphas": list(ALPHAS),
            "regime_policies": {
                name: list(values) for name, values in REGIME_POLICIES.items()
            },
            "action_clip": ACTION_CLIP,
            "link_batch": LINK_BATCH,
            "calibration": "global plus regime-shrunk median log truth/source per link-delay cell",
            "variance_gate": "coverage * exp(-quality_scale * sqrt(sum delay variance))",
        },
        "checkpoints": report_results,
        "selected": {
            "checkpoint": checkpoint,
            **selected,
            "shuffled_score": float(np.mean(shuffled)),
            "shuffled_gain_vs_baseline": float(np.mean(shuffled - baseline_score)),
            "real_minus_shuffled": float(np.mean(selected_score - shuffled)),
        },
        "test_action": test_diagnostics,
        "deployment_gate": {
            "pass": bool(
                selected["gain"] > 0.0
                and selected["fold_wins"] >= 4
                and np.mean(selected_score - shuffled) > 0.0
            ),
            "requirements": "positive mean gain, at least 4/5 fold wins, positive same-location margin",
        },
        "outputs": {
            "old_calibrated_oof": PATHS.relative(OLD_CALIBRATED_OOF),
            "oof": PATHS.relative(OOF_CACHE),
            "test_action": PATHS.relative(TEST_ACTION),
            "test_target": PATHS.relative(TEST_TARGET),
            "test_quality": PATHS.relative(TEST_QUALITY),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
