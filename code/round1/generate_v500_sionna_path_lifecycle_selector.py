#!/usr/bin/env python3
"""Generate an action-matched Sionna path-lifecycle selector candidate."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile

import joblib
import numpy as np
from sklearn.decomposition import PCA

import analyze_v208_strict_historical_replay as v208
import analyze_v432_sionna_link_energy_calibration as v432
import analyze_v434_sionna_joint_sh_calibration as v434
import analyze_v494_sionna_configuration_stacking as v494
import analyze_v498_sionna_path_lifecycle_profiles as v498
import analyze_v499_sionna_path_lifecycle_selector as v499
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v425_sionna_full_slice_pas_completion_diagnostic as v425
import generate_v426_sionna_linkwise_pdp_completion_diagnostic as v426
import generate_v435_train_calibrated_sionna_joint_sh_diagnostic as v435
import generate_v458_sionna_multimouth_extreme_diagnostic as v458
import generate_v496_sionna_configuration_selector_joint as v496
import train_v469_neural_sionna_calibrator as v469
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 500
VERSION = "v500_sionna_path_lifecycle_selector"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUTPUT = CACHE_DIR / "v500_Round1_Test_Channel.npy"
TARGET = CACHE_DIR / "v500_path_lifecycle_joint_target.npy"
ACTIONS = CACHE_DIR / "v500_path_lifecycle_selector_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
CHECKPOINT = CHECKPOINT_DIR / "path_lifecycle_selector.joblib"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"
V498_CACHE = PATHS.cache_dir(498) / "v498_sionna_path_lifecycle_oof.npz"
V499_CACHE = PATHS.cache_dir(499) / "v499_sionna_path_lifecycle_selector_oof.npz"
V491_OUTPUT = PATHS.cache_dir(491) / "v491_Round1_Test_Channel.npy"
REGIMES = ("clear", "light", "medium", "heavy")
LOG_RATIO_CAP = 1.5
BATCH = 2


def digest(path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 1e-10)
    return value / np.maximum(value.sum(axis=1, keepdims=True), TINY)


def fitted_position_features(
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    train_blocked: np.ndarray,
    test_blocked: np.ndarray,
    train_regimes: np.ndarray,
    test_regimes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train_positions[:, :2].mean(axis=0, keepdims=True)
    std = np.maximum(train_positions[:, :2].std(axis=0, keepdims=True), 1.0)
    block_mean = float(np.mean(train_blocked))
    block_std = max(float(np.std(train_blocked)), 1.0)

    def transform(
        positions: np.ndarray, blocked: np.ndarray, regimes: np.ndarray
    ) -> np.ndarray:
        normalized = (positions[:, :2] - mean) / std
        blocks = [normalized, positions[:, 2:3] / 20.0]
        for frequency in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
            angle = np.float32(math.pi * frequency) * normalized
            blocks.extend([np.sin(angle), np.cos(angle)])
        blocks.append(((blocked - block_mean) / block_std)[:, None])
        blocks.append(
            np.stack(
                [(regimes == name).astype(np.float32) for name in REGIMES], axis=1
            )
        )
        return np.concatenate(blocks, axis=1).astype(np.float32)

    return (
        transform(train_positions, train_blocked, train_regimes),
        transform(test_positions, test_blocked, test_regimes),
    )


def selector_features(
    train_robust: np.ndarray,
    test_robust: np.ndarray,
    profiles: np.ndarray,
    valid: np.ndarray,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    train_blocked: np.ndarray,
    test_blocked: np.ndarray,
    train_regimes: np.ndarray,
    test_regimes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    train_profiles = np.asarray(profiles[:2000], dtype=np.float32)
    test_profiles = np.asarray(profiles[2000:], dtype=np.float32)
    train_valid = np.asarray(valid[:2000], dtype=bool)
    test_valid = np.asarray(valid[2000:], dtype=bool)
    train_delay = normalize_rows(train_robust.reshape(2000, 192, 16).sum(axis=2))
    test_delay = normalize_rows(test_robust.reshape(500, 192, 16).sum(axis=2))
    train_profile_features = v499.profile_features(
        train_profiles, train_valid, train_delay
    )
    test_profile_features = v499.profile_features(
        test_profiles, test_valid, test_delay
    )
    path_summary = np.asarray(v499.build_path_summary(), dtype=np.float32)
    train_position, test_position = fitted_position_features(
        train_positions,
        test_positions,
        train_blocked,
        test_blocked,
        train_regimes,
        test_regimes,
    )
    train_raw = train_profiles.reshape(2000, -1)
    test_raw = test_profiles.reshape(500, -1)
    pca = PCA(
        n_components=v499.PCA_COMPONENTS,
        whiten=True,
        svd_solver="randomized",
        random_state=v499.SEED,
    )
    pca.fit(train_raw)
    train_features = np.concatenate(
        [
            path_summary[:2000],
            train_profile_features,
            train_position,
            pca.transform(train_raw),
        ],
        axis=1,
    ).astype(np.float32)
    test_features = np.concatenate(
        [
            path_summary[2000:],
            test_profile_features,
            test_position,
            pca.transform(test_raw),
        ],
        axis=1,
    ).astype(np.float32)
    return train_features, test_features, pca


def oof_calibrated_sources(
    truth: np.ndarray,
    sources: list[np.ndarray],
    fold_id: np.ndarray,
    regimes: np.ndarray,
) -> np.ndarray:
    output = np.zeros((len(sources), len(truth), truth.shape[1]), dtype=np.float32)
    for fold in range(5):
        fit_ids = np.flatnonzero(fold_id != fold)
        query_ids = np.flatnonzero(fold_id == fold)
        output[:, query_ids] = v494.calibrated_sources(
            truth, sources, fit_ids, query_ids, regimes
        )
    return output


def validated_alphas(
    truth: np.ndarray,
    sources: list[np.ndarray],
    quality: np.ndarray,
    regimes: np.ndarray,
    fold_id: np.ndarray,
) -> list[dict[str, float]]:
    calibrated = oof_calibrated_sources(truth, sources, fold_id, regimes)
    baseline = np.load(v434.OOF_CACHE)["baseline"].astype(np.float32)
    ids = np.arange(len(truth), dtype=np.int64)
    return [
        v494.select_alpha(
            truth, baseline, calibrated[index], quality, ids, regimes
        )
        for index in range(len(sources))
    ]


def calibrated_test_sources(
    truth: np.ndarray,
    train_sources: list[np.ndarray],
    test_sources: list[np.ndarray],
    train_regimes: np.ndarray,
    test_regimes: np.ndarray,
) -> np.ndarray:
    fit_ids = np.arange(len(truth), dtype=np.int64)
    query_ids = np.arange(len(test_sources[0]), dtype=np.int64)
    rows = []
    for train_source, test_source in zip(train_sources, test_sources, strict=True):
        ratio = v432.fit_ratio(truth, train_source, fit_ids, train_regimes)
        rows.append(v432.apply_ratio(test_source, query_ids, test_regimes, ratio))
    return np.stack(rows, axis=0).astype(np.float32)


def build_target(
    calibrated: np.ndarray,
    selected: np.ndarray,
    source_alpha: list[dict[str, float]],
    test_regimes: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    selected_target = calibrated[selected, np.arange(len(selected))]
    old_target = calibrated[0]
    neural = normalize_rows(np.load(v469.JOINT_PRED, mmap_mode="r"))
    delta = np.clip(
        np.log(np.maximum(selected_target, 1e-10))
        - np.log(np.maximum(old_target, 1e-10)),
        -LOG_RATIO_CAP,
        LOG_RATIO_CAP,
    ).astype(np.float32)
    joint = np.load(v458.JOINT_ACTIONS)
    deployment_regimes = joint["regimes"].astype("U8")
    if not np.array_equal(deployment_regimes, test_regimes):
        raise RuntimeError("test regime mismatch")
    profile = next(item for item in v458.JOINT_PROFILES if item.name == "tail085")
    deployment_alpha = v458.joint_alpha(
        profile,
        deployment_regimes,
        joint["point_quality"].astype(np.float32),
    )
    validated = np.asarray(
        [source_alpha[int(source)][str(regime)] for source, regime in zip(selected, test_regimes, strict=True)],
        dtype=np.float32,
    )
    scale = np.minimum(1.0, validated / np.maximum(deployment_alpha, 1e-6))
    scale[selected == 0] = 0.0
    effective_delta = delta * scale[:, None]
    corrected = normalize_rows(neural * np.exp(effective_delta).astype(np.float32))
    np.save(TARGET, corrected.astype(np.float32))
    diagnostics = {
        "active_points": int(np.sum(selected != 0)),
        "transfer_scale": {
            "mean_active": float(np.mean(scale[selected != 0])),
            "p10_active": float(np.quantile(scale[selected != 0], 0.10)),
            "median_active": float(np.median(scale[selected != 0])),
            "p90_active": float(np.quantile(scale[selected != 0], 0.90)),
        },
        "effective_log_delta_abs": {
            "mean": float(np.mean(np.abs(effective_delta))),
            "p90": float(np.quantile(np.abs(effective_delta), 0.90)),
            "p99": float(np.quantile(np.abs(effective_delta), 0.99)),
        },
    }
    return corrected, diagnostics


def render(target: np.ndarray) -> dict[str, object]:
    source = np.load(v458.BASE, mmap_mode="r")
    teacher_link = np.load(v458.CALIBRATED_LINK_TARGET, mmap_mode="r")
    link_quality = np.load(v458.LINK_QUALITY, mmap_mode="r")
    joint = np.load(v458.JOINT_ACTIONS)
    regimes = joint["regimes"].astype("U8")
    point_quality = joint["point_quality"].astype(np.float32)
    profile = next(item for item in v458.JOINT_PROFILES if item.name == "tail085")
    alpha_joint = v458.joint_alpha(profile, regimes, point_quality)
    alpha_link = v458.link_alpha(
        0.40, regimes, np.asarray(link_quality, dtype=np.float32)
    )
    power_delta = np.load(v458.POWER_ACTIONS)["final_log_delta"].astype(np.float32)
    amplitude = np.exp(0.5 * 10.0 * power_delta).astype(np.float32)
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape
    )
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        base_h = np.asarray(source[start:end], dtype=np.complex64)
        base_support, _ = v423.leaked_support(base_h)
        desired_support, _, _ = v435.build_desired_support(
            base_support, target[start:end], alpha_joint[start:end]
        )
        base_link, _ = v426.linkwise_pdp(base_h)
        desired_link = v426.desired_linkwise_pdp(
            base_link,
            np.asarray(teacher_link[start:end], dtype=np.float32).reshape(
                end - start, 256, 4, 192
            ),
            alpha_link[start:end].reshape(end - start, 256, 4),
        )
        candidate = base_h.copy()
        for _ in range(v458.JOINT_CYCLES):
            candidate = v425.project_full_support(candidate, desired_support)
            candidate = v426.restore_desired_pdp(candidate, base_h, desired_link)
        output[start:end] = (
            candidate * amplitude[start:end, None, None, None]
        ).astype(np.complex64)
        if end % 50 == 0 or end == len(source):
            output.flush()
            print(f"v500 render {end}/{len(source)}", flush=True)
    output.flush()
    replay = v208.score_prediction(OUTPUT)
    v491_replay = v208.score_prediction(V491_OUTPUT)
    return {
        "replay": replay,
        "v491_replay": v491_replay,
        "replay_delta_vs_v491": float(
            replay["score_traceback_leaked"]
            - v491_replay["score_traceback_leaked"]
        ),
    }


def compare_to_v491() -> dict[str, float]:
    left = np.load(OUTPUT, mmap_mode="r")
    right = np.load(V491_OUTPUT, mmap_mode="r")
    error = 0.0
    energy = 0.0
    cosine = 0.0
    count = 0
    for start in range(0, len(left), 5):
        end = min(start + 5, len(left))
        a = np.asarray(left[start:end], dtype=np.complex64)
        b = np.asarray(right[start:end], dtype=np.complex64)
        error += float(np.sum(np.abs(a - b) ** 2, dtype=np.float64))
        energy += float(np.sum(np.abs(b) ** 2, dtype=np.float64))
        inner = np.sum(np.conj(b) * a, axis=(1, 2, 3), dtype=np.complex128)
        norm = np.sqrt(
            np.sum(np.abs(a) ** 2, axis=(1, 2, 3), dtype=np.float64)
            * np.sum(np.abs(b) ** 2, axis=(1, 2, 3), dtype=np.float64)
        )
        cosine += float(np.sum(np.abs(inner) / np.maximum(norm, 1e-30)))
        count += end - start
    return {
        "nmse": error / max(energy, 1e-30),
        "complex_cosine": cosine / max(count, 1),
    }


def write_package(report: dict[str, object]) -> None:
    dependencies = (
        "generate_v500_sionna_path_lifecycle_selector.py",
        "analyze_v499_sionna_path_lifecycle_selector.py",
        "analyze_v498_sionna_path_lifecycle_profiles.py",
        "analyze_v494_sionna_configuration_stacking.py",
        "analyze_v495_sionna_configuration_selector.py",
        "train_v469_neural_sionna_calibrator.py",
        "generate_v458_sionna_multimouth_extreme_diagnostic.py",
        "generate_v435_train_calibrated_sionna_joint_sh_diagnostic.py",
        "generate_v426_sionna_linkwise_pdp_completion_diagnostic.py",
        "generate_v425_sionna_full_slice_pas_completion_diagnostic.py",
        "generate_v423_joint_sionna_pas_pdp_diagnostic.py",
        "analyze_v434_sionna_joint_sh_calibration.py",
        "analyze_v432_sionna_link_energy_calibration.py",
        "analyze_v208_strict_historical_replay.py",
        "project_paths.py",
    )
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        archive.write(PATHS.code / dependencies[0], dependencies[0])
        archive.writestr("model/", b"")
        archive.write(CHECKPOINT, "model/path_lifecycle_selector.joblib")
        for name in dependencies[1:]:
            archive.write(PATHS.code / name, name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    value = np.load(OUTPUT, mmap_mode="r")
    finite = all(
        np.isfinite(np.asarray(value[start : start + 5])).all()
        for start in range(0, len(value), 5)
    )
    report["integrity"] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(finite),
        "zip_crc": True,
        "zip_members": members,
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }


def main() -> None:
    required = (
        V498_CACHE,
        V499_CACHE,
        PATHS.cache_dir(498) / "v498_path_delay_profiles.npy",
        PATHS.cache_dir(498) / "v498_path_delay_profile_valid.npy",
        v469.JOINT_PRED,
        V491_OUTPUT,
        v458.BASE,
        v458.CALIBRATED_LINK_TARGET,
        v458.LINK_QUALITY,
        v458.POWER_ACTIONS,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    truth, base_sources, quality = v494.load_sources()
    train_robust = base_sources[0]
    test_robust = v496.load_test_sources()[0][0]
    profiles = np.load(
        PATHS.cache_dir(498) / "v498_path_delay_profiles.npy", mmap_mode="r"
    )
    valid = np.load(
        PATHS.cache_dir(498) / "v498_path_delay_profile_valid.npy", mmap_mode="r"
    )
    train_sources = v498.path_conditioned_sources(
        train_robust, profiles[:2000], valid[:2000]
    )
    test_sources = v498.path_conditioned_sources(
        test_robust, profiles[2000:], valid[2000:]
    )
    train_positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_positions = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    train_blocked = geometry["train_blocked"].astype(np.float32)
    test_blocked = geometry["test_blocked"].astype(np.float32)
    train_regimes = regime_of(train_blocked).astype("U8")
    test_regimes = regime_of(test_blocked).astype("U8")
    train_features, test_features, pca = selector_features(
        train_robust,
        test_robust,
        profiles,
        valid,
        train_positions,
        test_positions,
        train_blocked,
        test_blocked,
        train_regimes,
        test_regimes,
    )
    v498_cache = np.load(V498_CACHE)
    source_score = v498_cache["source_score"].astype(np.float32)
    gains = (source_score - source_score[0:1]).T.astype(np.float32)
    model = v499.make_model(v499.SEED)
    model.fit(train_features, gains)
    predicted = model.predict(test_features).astype(np.float32)
    predicted[:, 0] = 0.0
    selected = np.argmax(predicted, axis=1).astype(np.int16)
    margin = predicted[np.arange(len(test_features)), selected]
    selected[margin < 0.001] = 0
    selected_valid = np.ones(len(selected), dtype=bool)
    nonzero = selected > 0
    selected_valid[nonzero] = valid[
        2000 + np.flatnonzero(nonzero), selected[nonzero] - 1
    ]
    selected[~selected_valid] = 0
    joblib.dump(
        {
            "model": model,
            "pca": pca,
            "profile_names": v498.PROFILE_NAMES,
            "feature_dim": int(train_features.shape[1]),
            "fallback_margin": 0.001,
        },
        CHECKPOINT,
        compress=0,
    )
    fold_id = np.load(v434.OOF_CACHE)["fold_id"].astype(np.int16)
    source_alpha = validated_alphas(
        truth, train_sources, quality, train_regimes, fold_id
    )
    calibrated = calibrated_test_sources(
        truth, train_sources, test_sources, train_regimes, test_regimes
    )
    target, action_report = build_target(
        calibrated, selected, source_alpha, test_regimes
    )
    rendering = render(target)
    comparison = compare_to_v491()
    np.savez_compressed(
        ACTIONS,
        profile_names=np.asarray(v498.PROFILE_NAMES),
        selected_source=selected,
        predicted_margin=margin,
        predicted_gain=predicted,
        selected_valid=selected_valid,
        source_alpha=np.asarray(
            [[row[regime] for regime in REGIMES] for row in source_alpha],
            dtype=np.float32,
        ),
    )
    report: dict[str, object] = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "The strict v499 selector can identify the reliable LOS/reflection/refraction/depth "
            "lifecycle profile per test point, and OOF-action-matched residual transfer can add "
            "that path-level information to v491 without repeating v496 overdrive."
        ),
        "compliance": {
            "status": "diagnostic_noncompliant",
            "reason": "uses Sionna paths traced at the 500 official test coordinates",
        },
        "evidence": {
            "v499_mean_gain_vs_old": 0.00909789651632309,
            "v499_fold_wins": 5,
            "v499_gain_over_shuffled": 0.006980666890740395,
            "v498_profile_oracle_gain": 0.023445842787623405,
        },
        "selector": {
            "feature_dim": int(train_features.shape[1]),
            "selected_counts": {
                v498.PROFILE_NAMES[index]: int(np.sum(selected == index))
                for index in range(len(v498.PROFILE_NAMES))
            },
            "invalid_fallback_count": int(np.sum(~selected_valid)),
            "predicted_margin": {
                "mean": float(np.mean(margin)),
                "p10": float(np.quantile(margin, 0.10)),
                "median": float(np.median(margin)),
                "p90": float(np.quantile(margin, 0.90)),
            },
        },
        "action_matching": action_report,
        "renderer": {
            "base": PATHS.relative(v458.BASE),
            "joint_anchor": PATHS.relative(v469.JOINT_PRED),
            "joint_profile": "tail085",
            "link_scale": 0.40,
            "power_scale": 10.0,
            "cycles": v458.JOINT_CYCLES,
            "preserved": ["v458 linkwise PDP", "v458 point power", "base complex phase"],
        },
        **rendering,
        "comparison_to_v491": comparison,
        "online_score": None,
        "outputs": {
            "channel": PATHS.relative(OUTPUT),
            "target": PATHS.relative(TARGET),
            "actions": PATHS.relative(ACTIONS),
            "checkpoint": PATHS.relative(CHECKPOINT),
            "archive": PATHS.relative(ZIP_OUT),
            "report": PATHS.relative(REPORT),
        },
    }
    write_package(report)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "archive": PATHS.relative(ZIP_OUT),
                "selected_counts": report["selector"]["selected_counts"],
                "replay": rendering["replay"]["score_traceback_leaked"],
                "replay_delta_vs_v491": rendering["replay_delta_vs_v491"],
                "nmse_to_v491": comparison["nmse"],
                "sha256": report["integrity"]["sha256"],
                "md5": report["integrity"]["md5"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
