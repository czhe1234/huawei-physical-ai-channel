#!/usr/bin/env python3
"""Generate v496 from a learned per-point Sionna configuration selector."""

from __future__ import annotations

import hashlib
import json
import zipfile

import joblib
import numpy as np
from sklearn.decomposition import PCA

import analyze_v208_strict_historical_replay as v208
import analyze_v432_sionna_link_energy_calibration as v432
import analyze_v434_sionna_joint_sh_calibration as v434
import analyze_v494_sionna_configuration_stacking as v494
import analyze_v495_sionna_configuration_selector as v495
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v425_sionna_full_slice_pas_completion_diagnostic as v425
import generate_v426_sionna_linkwise_pdp_completion_diagnostic as v426
import generate_v435_train_calibrated_sionna_joint_sh_diagnostic as v435
import generate_v458_sionna_multimouth_extreme_diagnostic as v458
import train_v469_neural_sionna_calibrator as v469
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 496
VERSION = "v496_sionna_configuration_selector_joint"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUTPUT = CACHE_DIR / "v496_Round1_Test_Channel.npy"
TARGET = CACHE_DIR / "v496_selected_joint_target.npy"
ACTIONS = CACHE_DIR / "v496_sionna_configuration_selector_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
CHECKPOINT = CHECKPOINT_DIR / "sionna_configuration_selector.joblib"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"
V494_CACHE = PATHS.cache_dir(494) / "v494_sionna_configuration_stacking_oof.npz"
V491_OUTPUT = PATHS.cache_dir(491) / "v491_Round1_Test_Channel.npy"
FLOOR = 1e-10
BATCH = 2
LOG_RATIO_CAP = 1.5


def require(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def digest(path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), FLOOR)
    return value / np.maximum(value.sum(axis=1, keepdims=True), TINY)


def load_test_sources() -> tuple[list[np.ndarray], np.ndarray]:
    high, high_total = v434.load_sh(v435.TEST_HIGH)
    low_loaded = [v434.load_sh(path) for path in v435.TEST_LOWS]
    lows = [item[0] for item in low_loaded]
    low_totals = [item[1] for item in low_loaded]
    robust, quality, _ = v434.robust_source(
        high, high_total, lows, low_totals
    )
    all_interaction = normalize_rows(
        np.load(PATHS.cache_dir(461) / "test_joint_sh.npy", mmap_mode="r").reshape(
            len(robust), -1
        )
    )
    raw = [robust, high, *lows, all_interaction]
    cleaned: list[np.ndarray] = []
    for source in raw:
        source = normalize_rows(source)
        valid = source.sum(axis=1) > 0.5
        source = source.copy()
        source[~valid] = robust[~valid]
        cleaned.append(normalize_rows(source))
    return cleaned, quality.astype(np.float32)


def test_neighbor_baseline(
    truth: np.ndarray,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    train_blocked: np.ndarray,
    test_blocked: np.ndarray,
    train_regimes: np.ndarray,
    test_regimes: np.ndarray,
) -> np.ndarray:
    positions = np.concatenate([train_positions, test_positions], axis=0)
    blocked = np.concatenate([train_blocked, test_blocked])
    regimes = np.concatenate([train_regimes, test_regimes])
    target = np.zeros((len(positions), truth.shape[1]), dtype=np.float32)
    target[: len(truth)] = truth
    return normalize_rows(
        v432.neighbor_prediction(
            np.arange(len(truth), dtype=np.int64),
            np.arange(len(truth), len(positions), dtype=np.int64),
            positions,
            blocked,
            regimes,
            target,
            k=48,
        )
    )


def train_selector(
    train_sources: list[np.ndarray],
    test_sources: list[np.ndarray],
    train_baseline: np.ndarray,
    test_baseline: np.ndarray,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    train_blocked: np.ndarray,
    test_blocked: np.ndarray,
    train_regimes: np.ndarray,
    test_regimes: np.ndarray,
    gains: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    train_summary = v495.spectral_features(train_sources, train_baseline)
    test_summary = v495.spectral_features(test_sources, test_baseline)
    all_positions = np.concatenate([train_positions, test_positions], axis=0)
    all_blocked = np.concatenate([train_blocked, test_blocked])
    all_regimes = np.concatenate([train_regimes, test_regimes])
    all_position_features = v495.position_features(
        all_positions, all_blocked, all_regimes
    )
    train_raw_pca = v495.pca_input(train_sources)
    test_raw_pca = v495.pca_input(test_sources)
    pca = PCA(
        n_components=v495.PCA_COMPONENTS,
        whiten=True,
        svd_solver="randomized",
        random_state=v495.SEED,
    )
    pca.fit(train_raw_pca)
    train_features = np.concatenate(
        [
            train_summary,
            all_position_features[: len(train_positions)],
            pca.transform(train_raw_pca).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    test_features = np.concatenate(
        [
            test_summary,
            all_position_features[len(train_positions) :],
            pca.transform(test_raw_pca).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    model = v495.make_model(v495.SEED)
    model.fit(train_features, gains)
    predicted = model.predict(test_features).astype(np.float32)
    predicted[:, 0] = 0.0
    selected = np.argmax(predicted, axis=1).astype(np.int16)
    margin = predicted[np.arange(len(test_features)), selected]
    selected[margin < 0.001] = 0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "pca": pca,
            "source_names": v494.SOURCE_NAMES,
            "feature_dim": int(train_features.shape[1]),
            "fallback_margin": 0.001,
        },
        CHECKPOINT,
        compress=0,
    )
    diagnostics = {
        "feature_dim": int(train_features.shape[1]),
        "selected_counts": {
            v494.SOURCE_NAMES[index]: int(np.sum(selected == index))
            for index in range(len(v494.SOURCE_NAMES))
        },
        "predicted_margin": {
            "mean": float(np.mean(margin)),
            "p10": float(np.quantile(margin, 0.10)),
            "median": float(np.median(margin)),
            "p90": float(np.quantile(margin, 0.90)),
        },
    }
    return selected, margin, predicted, diagnostics


def calibrated_test_targets(
    truth: np.ndarray,
    train_sources: list[np.ndarray],
    test_sources: list[np.ndarray],
    train_regimes: np.ndarray,
    test_regimes: np.ndarray,
) -> np.ndarray:
    fit_ids = np.arange(len(truth), dtype=np.int64)
    query_ids = np.arange(len(test_sources[0]), dtype=np.int64)
    output = []
    for train_source, test_source in zip(train_sources, test_sources, strict=True):
        ratio = v432.fit_ratio(truth, train_source, fit_ids, train_regimes)
        output.append(
            v432.apply_ratio(test_source, query_ids, test_regimes, ratio)
        )
    return np.stack(output, axis=0).astype(np.float32)


def build_target(
    calibrated: np.ndarray,
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    old_target = calibrated[0]
    selected_target = calibrated[selected, np.arange(len(selected))]
    neural = normalize_rows(np.load(v469.JOINT_PRED, mmap_mode="r"))
    log_delta = np.clip(
        np.log(np.maximum(selected_target, FLOOR))
        - np.log(np.maximum(old_target, FLOOR)),
        -LOG_RATIO_CAP,
        LOG_RATIO_CAP,
    ).astype(np.float32)
    active = selected != 0
    log_delta[~active] = 0.0
    corrected = normalize_rows(neural * np.exp(log_delta).astype(np.float32))
    np.save(TARGET, corrected.astype(np.float32))
    return corrected, selected_target, log_delta


def render(corrected_target: np.ndarray) -> dict[str, object]:
    source = np.load(v458.BASE, mmap_mode="r")
    teacher_link = np.load(v458.CALIBRATED_LINK_TARGET, mmap_mode="r")
    link_quality = np.load(v458.LINK_QUALITY, mmap_mode="r")
    joint_actions = np.load(v458.JOINT_ACTIONS)
    regimes = joint_actions["regimes"].astype("U8")
    point_quality = joint_actions["point_quality"].astype(np.float32)
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
            base_support,
            corrected_target[start:end],
            alpha_joint[start:end],
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
            print(f"v496 render {end}/{len(source)}", flush=True)
    output.flush()
    replay = v208.score_prediction(OUTPUT)
    baseline_replay = v208.score_prediction(V491_OUTPUT)
    return {
        "replay": replay,
        "v491_replay": baseline_replay,
        "replay_delta_vs_v491": float(
            replay["score_traceback_leaked"]
            - baseline_replay["score_traceback_leaked"]
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
        "generate_v496_sionna_configuration_selector_joint.py",
        "analyze_v495_sionna_configuration_selector.py",
        "analyze_v494_sionna_configuration_stacking.py",
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
        archive.write(CHECKPOINT, "model/sionna_configuration_selector.joblib")
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
        V494_CACHE,
        V491_OUTPUT,
        v469.JOINT_PRED,
        v458.BASE,
        v458.CALIBRATED_LINK_TARGET,
        v458.LINK_QUALITY,
        v458.POWER_ACTIONS,
        PATHS.cache_dir(461) / "test_joint_sh.npy",
    )
    for path in required:
        require(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    truth, train_sources, _ = v494.load_sources()
    test_sources, test_quality = load_test_sources()
    train_positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_positions = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    train_blocked = geometry["train_blocked"].astype(np.float32)
    test_blocked = geometry["test_blocked"].astype(np.float32)
    train_regimes = regime_of(train_blocked).astype("U8")
    test_regimes = regime_of(test_blocked).astype("U8")
    v494_cache = np.load(V494_CACHE)
    train_baseline = v494_cache["baseline"].astype(np.float32)
    test_baseline = test_neighbor_baseline(
        truth,
        train_positions,
        test_positions,
        train_blocked,
        test_blocked,
        train_regimes,
        test_regimes,
    )
    source_score = v494_cache["source_score"].astype(np.float32)
    gains = (source_score - source_score[0:1]).T.astype(np.float32)
    selected, margin, predicted, selector_report = train_selector(
        train_sources,
        test_sources,
        train_baseline,
        test_baseline,
        train_positions,
        test_positions,
        train_blocked,
        test_blocked,
        train_regimes,
        test_regimes,
        gains,
    )
    calibrated = calibrated_test_targets(
        truth,
        train_sources,
        test_sources,
        train_regimes,
        test_regimes,
    )
    corrected, selected_target, log_delta = build_target(calibrated, selected)
    rendering = render(corrected)
    comparison = compare_to_v491()
    np.savez_compressed(
        ACTIONS,
        source_names=np.asarray(v494.SOURCE_NAMES),
        selected_source=selected,
        predicted_margin=margin,
        predicted_gain=predicted,
        test_quality=test_quality,
        selected_target=selected_target,
        log_delta=log_delta,
    )
    report: dict[str, object] = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "The strict v495 selector can transfer only the calibrated Sionna configuration "
            "residual that is appropriate for each test point onto the online-positive v491 "
            "neural joint target, while retaining v458 linkwise PDP, power and phase."
        ),
        "compliance": {
            "status": "diagnostic_noncompliant",
            "reason": "uses Sionna references traced at the 500 official test coordinates",
        },
        "evidence": {
            "v495_mean_gain_vs_old": 0.005241366568952799,
            "v495_fold_wins": 5,
            "v495_gain_over_shuffled": 0.00629607867449522,
            "v494_source_oracle_gain": 0.02397097833454609,
        },
        "selector": selector_report,
        "joint_transfer": {
            "anchor": PATHS.relative(v469.JOINT_PRED),
            "old_target": "full-train calibrated robust_v434",
            "selected_target": "full-train calibrated per-point Sionna source",
            "log_ratio_cap": LOG_RATIO_CAP,
            "active_points": int(np.sum(selected != 0)),
            "log_delta_abs_mean": float(np.mean(np.abs(log_delta))),
            "log_delta_abs_p90": float(np.quantile(np.abs(log_delta), 0.90)),
        },
        "renderer": {
            "base": PATHS.relative(v458.BASE),
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
                "active_points": int(np.sum(selected != 0)),
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
