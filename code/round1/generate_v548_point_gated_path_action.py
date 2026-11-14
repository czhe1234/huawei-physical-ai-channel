#!/usr/bin/env python3
"""Deploy the v547 point-gated path action on the current v540 carrier."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np

import analyze_v545_path_block_family_oracle as v545
import analyze_v546_local_global_path_event_model as v546
import analyze_v547_pointwise_path_action_gate as v547
import generate_v533_expanded_physical_sbi_action_match as v533
import generate_v543_disagreement_conditioned_cell_fusion as v543
from project_paths import PATHS


VERSION_NUMBER = 548
VERSION = "v548_point_gated_path_action"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUTPUT = CACHE_DIR / "v548_Round1_Test_Channel.npy"
TARGET = CACHE_DIR / "v548_point_gated_joint_target.npy"
ACTIONS = CACHE_DIR / "v548_point_gated_path_actions.npz"
TEST_BANK = CACHE_DIR / "v548_test_candidate_bank.npy"
TEST_META = CACHE_DIR / "v548_test_path_metadata.npz"
TEST_EVENTS = CACHE_DIR / "v548_test_path_event_records.npz"
TEST_FEATURES = CACHE_DIR / "v548_test_point_gate_features.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

V540_OUTPUT = PATHS.cache_dir(540) / "v540_Round1_Test_Channel.npy"
V540_TARGET = PATHS.cache_dir(540) / "v540_seed42_medium_joint_target.npy"
V543_OUTPUT = PATHS.cache_dir(543) / "v543_Round1_Test_Channel.npy"
V543_ACTIONS = PATHS.cache_dir(543) / "v543_cellwise_fusion_actions.npz"
PATH_ROOT = PATHS.cache_dir(475) / "shards"
TEST_POS = PATHS.round1 / "Round1_Test_Pos.npy"

HEIGHT = v545.HEIGHT
WIDTH = v545.WIDTH
SOURCE_COUNT = v545.SOURCE_COUNT
MAX_PATHS = v545.MAX_PATHS
OWNER_NONE = v545.OWNER_NONE
REGIMES = v545.REGIMES
FLOOR = 1e-10


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def validate_inputs() -> None:
    required = (
        V540_OUTPUT,
        V540_TARGET,
        V543_OUTPUT,
        V543_ACTIONS,
        v545.OOF_SOURCE,
        v545.PATH_META,
        v546.EVENT_CACHE,
        v546.OOF_CACHE,
        v547.FEATURE_CACHE,
        v547.OOF_CACHE,
        v547.REPORT,
        TEST_POS,
    )
    missing = [PATHS.relative(path) for path in required if not path.is_file()]
    shards = sorted(PATH_ROOT.glob("shard_*/path_targets.npz"))
    if len(shards) != 25:
        missing.append(f"expected 25 v475 path shards, found {len(shards)}")
    if missing:
        raise FileNotFoundError(f"missing v548 inputs: {missing}")


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    return value / np.maximum(value.sum(axis=1, keepdims=True), 1e-30)


def build_test_bank() -> tuple[np.ndarray, list[str]]:
    if TEST_BANK.is_file():
        bank = np.load(TEST_BANK, mmap_mode="r")
        if bank.shape != (500, SOURCE_COUNT, HEIGHT * WIDTH):
            raise ValueError(f"invalid cached test bank shape: {bank.shape}")
        metadata = np.load(V543_ACTIONS)
        return bank, ["official_48nn_baseline", *metadata["source_names"].astype(str)]

    truth, train_sources, test_sources, _, _, source_names = v543.load_sources()
    train_positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_positions = np.load(TEST_POS).astype(np.float32)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    train_blocked = geometry["train_blocked"].astype(np.float32)
    test_blocked = geometry["test_blocked"].astype(np.float32)
    train_regimes = v543.regime_of(train_blocked).astype("U8")
    test_regimes = v543.regime_of(test_blocked).astype("U8")
    fit_ids = np.arange(len(truth), dtype=np.int64)
    baseline = v543.test_neighbor_baseline(
        truth,
        fit_ids,
        train_positions,
        test_positions,
        train_blocked,
        test_blocked,
        train_regimes,
        test_regimes,
    )
    candidates = v543.calibrated_test_sources(
        truth,
        train_sources,
        test_sources,
        fit_ids,
        train_regimes,
        test_regimes,
    )
    bank = np.concatenate([baseline[None], candidates], axis=0).transpose(1, 0, 2)
    bank = normalize_rows(bank.reshape(-1, HEIGHT * WIDTH)).reshape(bank.shape)
    np.save(TEST_BANK, bank.astype(np.float32))
    return np.load(TEST_BANK, mmap_mode="r"), [
        "official_48nn_baseline",
        *source_names,
    ]


def build_test_path_metadata() -> dict[str, np.ndarray]:
    if TEST_META.is_file():
        with np.load(TEST_META) as cached:
            return {key: np.asarray(cached[key]) for key in cached.files}

    train_meta_file = np.load(v545.PATH_META)
    train_meta = {
        key: np.asarray(train_meta_file[key]) for key in train_meta_file.files
    }
    family_maps: dict[str, dict[tuple[int, ...], int]] = {
        mode: {} for mode in v545.IDENTITY_MODES
    }
    owner = np.full((500, HEIGHT * WIDTH), OWNER_NONE, dtype=np.uint8)
    path_valid = np.zeros((500, MAX_PATHS), dtype=bool)
    path_count = np.zeros(500, dtype=np.int16)
    family_arrays = {
        mode: np.full((500, MAX_PATHS), -1, dtype=np.int32)
        for mode in v545.IDENTITY_MODES
    }
    global_offset = 0
    for shard_id in range(25):
        shard = PATH_ROOT / f"shard_{shard_id:05d}" / "path_targets.npz"
        with np.load(shard) as paths:
            rows = len(paths["valid"])
            for row in range(rows):
                global_point = global_offset + row
                valid_ids = np.flatnonzero(paths["valid"][row])
                is_test = global_point >= 2000
                if is_test:
                    test_point = global_point - 2000
                    path_valid[test_point, valid_ids] = True
                    path_count[test_point] = len(valid_ids)
                    if len(valid_ids):
                        tau = paths["tau"][row, valid_ids].astype(np.float32)
                        delay = v545.v498.delay_fingerprint(tau)
                        horizontal = paths["tx_h_fingerprint"][row, valid_ids].astype(
                            np.float32
                        )
                        horizontal /= np.maximum(
                            horizontal.sum(axis=1, keepdims=True), v545.TINY
                        )
                        mass = paths["normalized_power"][row, valid_ids].astype(
                            np.float32
                        )
                        mass /= max(float(mass.sum()), v545.TINY)
                        atoms = (
                            mass[:, None, None]
                            * delay[:, :, None]
                            * horizontal[:, None, :]
                        )
                        owner_local = np.argmax(atoms, axis=0)
                        owner[test_point] = (
                            valid_ids[owner_local].reshape(-1).astype(np.uint8)
                        )
                for original_path in valid_ids:
                    count = int(paths["interaction_count"][row, original_path])
                    mask = int(paths["interaction_mask"][row, original_path])
                    for mode in v545.IDENTITY_MODES:
                        key = v545.family_key(
                            paths["interactions"][row, original_path],
                            paths["objects"][row, original_path],
                            count,
                            mask,
                            mode,
                        )
                        mapping = family_maps[mode]
                        if key not in mapping:
                            mapping[key] = len(mapping)
                        family_id = mapping[key]
                        if is_test:
                            family_arrays[mode][
                                global_point - 2000, original_path
                            ] = family_id
                        elif (
                            train_meta[f"family_{mode}"][global_point, original_path]
                            != family_id
                        ):
                            raise ValueError(
                                f"train family replay mismatch at point={global_point} "
                                f"path={original_path} mode={mode}"
                            )
            global_offset += rows
        print(f"v548 path metadata shard {shard_id + 1}/25", flush=True)
    if global_offset != 2500:
        raise ValueError(f"expected 2500 path rows, found {global_offset}")
    result = {
        "owner_path": owner,
        "path_valid": path_valid,
        "path_count": path_count,
        **{f"family_{mode}": value for mode, value in family_arrays.items()},
    }
    np.savez_compressed(TEST_META, **result)
    return result


def build_test_events(
    bank: np.ndarray,
    anchor: np.ndarray,
    base: np.ndarray,
    metadata: dict[str, np.ndarray],
    positions: np.ndarray,
) -> dict[str, np.ndarray]:
    if TEST_EVENTS.is_file():
        with np.load(TEST_EVENTS) as cached:
            return {key: np.asarray(cached[key]) for key in cached.files}

    point_list: list[int] = []
    path_list: list[int] = []
    cells_list: list[int] = []
    family_lists = {mode: [] for mode in v545.IDENTITY_MODES}
    candidate_features: list[np.ndarray] = []
    block_features: list[np.ndarray] = []
    owner = metadata["owner_path"]
    record_by_point: list[list[int]] = [[] for _ in range(500)]

    for point in range(500):
        paths = np.unique(owner[point][owner[point] != OWNER_NONE]).astype(np.int64)
        for path in paths:
            mask = owner[point] == path
            candidate = np.asarray(bank[point], dtype=np.float64)[:, mask]
            source_features, summary = v546.block_candidate_features(
                candidate,
                anchor[point, mask],
                base[point, mask],
            )
            point_list.append(point)
            path_list.append(int(path))
            cells_list.append(int(np.sum(mask)))
            for mode in v545.IDENTITY_MODES:
                family_lists[mode].append(int(metadata[f"family_{mode}"][point, path]))
            candidate_features.append(source_features)
            block_features.append(summary)
            record_by_point[point].append(len(point_list) - 1)

    records = {
        "point": np.asarray(point_list, dtype=np.int32),
        "path": np.asarray(path_list, dtype=np.int16),
        "cells": np.asarray(cells_list, dtype=np.int32),
        "candidate_features": np.asarray(candidate_features, dtype=np.float32),
        "block_features": np.asarray(block_features, dtype=np.float32),
        **{
            f"family_{mode}": np.asarray(value, dtype=np.int32)
            for mode, value in family_lists.items()
        },
    }
    path_features = np.zeros(
        (len(records["point"]), v546.PATH_FEATURE_DIM), dtype=np.float32
    )
    global_offset = 2000
    for shard_id in range(20, 25):
        shard = PATH_ROOT / f"shard_{shard_id:05d}" / "path_targets.npz"
        with np.load(shard) as data:
            rows = len(data["valid"])
            for row in range(rows):
                point = global_offset + row - 2000
                for record in record_by_point[point]:
                    path_features[record] = v546.path_features(
                        data,
                        row,
                        int(records["path"][record]),
                        positions[point],
                    )
            global_offset += rows
        print(f"v548 test path features shard {shard_id - 19}/5", flush=True)
    if global_offset != 2500:
        raise ValueError(f"expected final path offset 2500, found {global_offset}")
    records["path_features"] = path_features
    np.savez_compressed(TEST_EVENTS, **records)
    return records


def combine_local_records(
    train: dict[str, np.ndarray], test: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    output = {
        "point": np.concatenate([train["point"], test["point"].astype(np.int32) + 2000])
    }
    for mode in v545.IDENTITY_MODES:
        output[f"family_{mode}"] = np.concatenate(
            [train[f"family_{mode}"], test[f"family_{mode}"]]
        )
    return output


def predict_test_events(
    train_records: dict[str, np.ndarray],
    test_records: dict[str, np.ndarray],
    train_positions: np.ndarray,
    test_positions: np.ndarray,
) -> np.ndarray:
    train_common, fold_id = v547.load_common_event_features(train_records)
    _, static_test, _ = v543.load_static_features()
    test_common = np.concatenate(
        [
            static_test[test_records["point"]],
            test_records["path_features"],
            test_records["block_features"],
            test_records["candidate_features"],
        ],
        axis=1,
    ).astype(np.float32)
    test_common = np.nan_to_num(test_common, nan=0.0, posinf=1e6, neginf=-1e6)
    combined = combine_local_records(train_records, test_records)
    positions = np.concatenate([train_positions, test_positions], axis=0)
    train_count = len(train_records["point"])
    total_count = train_count + len(test_records["point"])
    utility = np.zeros((total_count, SOURCE_COUNT), dtype=np.float32)
    log_gain = np.zeros_like(utility)
    utility[:train_count] = train_records["utility_model"]
    log_gain[:train_count] = train_records["log_gain"]
    target = np.concatenate(
        [train_records["utility_model"], train_records["log_gain"]], axis=1
    ).astype(np.float32)
    record_fold = fold_id[train_records["point"]]
    predictions: list[np.ndarray] = []

    for outer in range(5):
        fit_train = record_fold != outer
        fit_combined = np.zeros(total_count, dtype=bool)
        fit_combined[:train_count] = fit_train
        local = v546.local_target_features(
            combined,
            positions,
            fit_combined,
            utility,
            log_gain,
        )
        train_features = np.concatenate(
            [train_common, local[:train_count]], axis=1
        ).astype(np.float32)
        test_features = np.concatenate(
            [test_common, local[train_count:]], axis=1
        ).astype(np.float32)
        model = v546.make_model(v546.SEED + 100 + outer, v546.MODEL_TREES)
        model.fit(
            train_features[fit_train],
            target[fit_train],
            sample_weight=train_records["event_weight"][fit_train],
        )
        predictions.append(model.predict(test_features).astype(np.float32))
        print(f"v548 first-layer test ensemble fold {outer + 1}/5", flush=True)
    return np.mean(np.stack(predictions, axis=0), axis=0).astype(np.float32)


def build_test_gate_features(
    test_records: dict[str, np.ndarray],
    predicted: np.ndarray,
    choices: np.ndarray,
    gains: np.ndarray,
    base: np.ndarray,
    action: np.ndarray,
    anchor: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    _, static_test, static_names = v543.load_static_features()
    event, event_names = v547.aggregate_event_features(
        test_records, predicted, choices, gains
    )
    spectral, spectral_names = v547.spectral_action_features(base, action, anchor)
    features = np.concatenate(
        [static_test, event[: len(static_test)], spectral], axis=1
    ).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    names = list(static_names) + event_names + spectral_names
    train_feature_cache = np.load(v547.FEATURE_CACHE)
    train_names = train_feature_cache["feature_names"].astype(str).tolist()
    if names != train_names:
        mismatch = next(
            (
                index
                for index, (left, right) in enumerate(zip(names, train_names))
                if left != right
            ),
            min(len(names), len(train_names)),
        )
        raise ValueError(f"train/test gate feature mismatch at column {mismatch}")
    np.savez_compressed(
        TEST_FEATURES,
        features=features,
        feature_names=np.asarray(names, dtype="U96"),
    )
    return features, names


def predict_test_gate(features: np.ndarray) -> dict[str, np.ndarray]:
    train_cache = np.load(v547.FEATURE_CACHE)
    train_features = train_cache["features"].astype(np.float32)
    oof = np.load(v547.OOF_CACHE)
    true_gain = oof["true_gain"].astype(np.float32)
    valid = oof["valid"].astype(bool)
    fold_id = oof["fold_id"].astype(np.int16)
    gain_predictions: list[np.ndarray] = []
    probability_predictions: list[np.ndarray] = []
    for outer in range(5):
        fit = valid & (fold_id != outer)
        target = true_gain[fit]
        label = target > 0.0
        regressor = v547.make_regressor(v547.SEED + outer * 17)
        classifier = v547.make_classifier(v547.SEED + 10000 + outer * 17)
        regressor.fit(train_features[fit], target)
        classifier.fit(train_features[fit], label)
        gain_predictions.append(regressor.predict(features).astype(np.float32))
        probability_predictions.append(
            classifier.predict_proba(features)[:, 1].astype(np.float32)
        )
        print(f"v548 point-gate test ensemble fold {outer + 1}/5", flush=True)
    predicted_gain = np.mean(np.stack(gain_predictions, axis=0), axis=0)
    probability = np.mean(np.stack(probability_predictions, axis=0), axis=0)
    active = (predicted_gain > 0.0) & (probability > 0.5)
    return {
        "predicted_gain": predicted_gain.astype(np.float32),
        "positive_probability": probability.astype(np.float32),
        "active": active,
        "fold_predicted_gain": np.stack(gain_predictions, axis=0),
        "fold_probability": np.stack(probability_predictions, axis=0),
    }


def build_target(
    base_physical: np.ndarray,
    path_action: np.ndarray,
    gate: dict[str, np.ndarray],
    test_regimes: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    v543_actions = np.load(V543_ACTIONS)
    deployment_alpha = v543_actions["deployment_alpha"].astype(np.float32)
    gate_active = gate["active"].astype(bool)
    effective_active = gate_active & (deployment_alpha > 1e-6)
    desired_action = np.clip(
        np.log(np.maximum(path_action, FLOOR))
        - np.log(np.maximum(base_physical, FLOOR)),
        -1.5,
        1.5,
    ).astype(np.float32)
    desired_action[~effective_active] = 0.0
    target_log_delta = np.zeros_like(desired_action)
    target_log_delta[effective_active] = (
        desired_action[effective_active] / deployment_alpha[effective_active, None]
    )
    target_log_delta = np.clip(target_log_delta, -1.5, 1.5)
    v540_target = normalize_rows(np.load(V540_TARGET, mmap_mode="r"))
    corrected = normalize_rows(
        v540_target * np.exp(target_log_delta).astype(np.float32)
    )
    corrected[~effective_active] = v540_target[~effective_active]
    diagnostics = {
        "gate_active_count": int(np.sum(gate_active)),
        "gate_active_fraction": float(np.mean(gate_active)),
        "effective_active_count": int(np.sum(effective_active)),
        "effective_active_fraction": float(np.mean(effective_active)),
        "active_by_regime": {
            regime: int(np.sum(effective_active & (test_regimes == regime)))
            for regime in REGIMES
        },
        "active_fraction_by_regime": {
            regime: float(np.mean(effective_active[test_regimes == regime]))
            for regime in REGIMES
        },
        "predicted_gain": {
            "mean": float(np.mean(gate["predicted_gain"])),
            "p10": float(np.quantile(gate["predicted_gain"], 0.10)),
            "median": float(np.median(gate["predicted_gain"])),
            "p90": float(np.quantile(gate["predicted_gain"], 0.90)),
        },
        "positive_probability": {
            "mean": float(np.mean(gate["positive_probability"])),
            "p10": float(np.quantile(gate["positive_probability"], 0.10)),
            "median": float(np.median(gate["positive_probability"])),
            "p90": float(np.quantile(gate["positive_probability"], 0.90)),
        },
        "desired_action_abs": {
            "mean_all": float(np.mean(np.abs(desired_action))),
            "mean_active": float(
                np.mean(np.abs(desired_action[effective_active]))
                if np.any(effective_active)
                else 0.0
            ),
            "p90_active": float(
                np.quantile(np.abs(desired_action[effective_active]), 0.90)
                if np.any(effective_active)
                else 0.0
            ),
        },
        "target_log_delta_abs": {
            "mean": float(np.mean(np.abs(target_log_delta))),
            "p90": float(np.quantile(np.abs(target_log_delta), 0.90)),
            "p99": float(np.quantile(np.abs(target_log_delta), 0.99)),
        },
        "inactive_target_max_abs": float(
            np.max(
                np.abs(corrected[~effective_active] - v540_target[~effective_active])
            )
        ),
    }
    return corrected.astype(np.float32), diagnostics


def compare_channels(left_path: Path, right_path: Path) -> dict[str, object]:
    left = np.load(left_path, mmap_mode="r")
    right = np.load(right_path, mmap_mode="r")
    actions = np.load(ACTIONS)
    active = actions["effective_active"].astype(bool)
    error = 0.0
    energy = 0.0
    cosine = 0.0
    count = 0
    point_nmse = np.zeros(len(left), dtype=np.float64)
    maximum_inactive_abs = 0.0
    for start in range(0, len(left), 5):
        stop = min(start + 5, len(left))
        a = np.asarray(left[start:stop], dtype=np.complex64)
        b = np.asarray(right[start:stop], dtype=np.complex64)
        local_error = np.sum(np.abs(a - b) ** 2, axis=(1, 2, 3), dtype=np.float64)
        local_energy = np.sum(np.abs(b) ** 2, axis=(1, 2, 3), dtype=np.float64)
        point_nmse[start:stop] = local_error / np.maximum(local_energy, 1e-30)
        error += float(np.sum(local_error))
        energy += float(np.sum(local_energy))
        inner = np.sum(np.conj(b) * a, axis=(1, 2, 3), dtype=np.complex128)
        norm = np.sqrt(
            np.sum(np.abs(a) ** 2, axis=(1, 2, 3), dtype=np.float64) * local_energy
        )
        cosine += float(np.sum(np.abs(inner) / np.maximum(norm, 1e-30)))
        count += stop - start
        local_inactive = ~active[start:stop]
        if np.any(local_inactive):
            maximum_inactive_abs = max(
                maximum_inactive_abs,
                float(np.max(np.abs(a[local_inactive] - b[local_inactive]))),
            )
    return {
        "nmse": error / max(energy, 1e-30),
        "complex_cosine": cosine / max(count, 1),
        "active_nmse_mean": (
            float(np.mean(point_nmse[active])) if np.any(active) else 0.0
        ),
        "inactive_nmse_mean": float(np.mean(point_nmse[~active])),
        "inactive_maximum_abs": maximum_inactive_abs,
    }


def write_package(report: dict[str, object]) -> None:
    neural_checkpoint = (
        PATHS.checkpoints / v543.v469.VERSION / "neural_sionna_calibrator.pt"
    )
    require(neural_checkpoint)
    dependencies = (
        "generate_v548_point_gated_path_action.py",
        "analyze_v547_pointwise_path_action_gate.py",
        "analyze_v546_local_global_path_event_model.py",
        "analyze_v545_path_block_family_oracle.py",
        "generate_v543_disagreement_conditioned_cell_fusion.py",
        "generate_v533_expanded_physical_sbi_action_match.py",
        "analyze_v541_corrected_full_library_sbi.py",
        "analyze_v531_expanded_sbi_configuration_library.py",
        "analyze_v494_sionna_configuration_stacking.py",
        "train_v469_neural_sionna_calibrator.py",
        "generate_v458_sionna_multimouth_extreme_diagnostic.py",
        "generate_v435_train_calibrated_sionna_joint_sh_diagnostic.py",
        "generate_v426_sionna_linkwise_pdp_completion_diagnostic.py",
        "generate_v425_sionna_full_slice_pas_completion_diagnostic.py",
        "generate_v423_joint_sionna_pas_pdp_diagnostic.py",
        "analyze_v208_strict_historical_replay.py",
        "powerfield_common.py",
        "project_paths.py",
    )
    for name in dependencies:
        require(PATHS.code / name)
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        archive.writestr("model/", b"")
        archive.write(neural_checkpoint, "model/neural_sionna_calibrator.pt")
        archive.write(REPORT, "model/v548_path_action_gate_report.json")
        for name in dependencies:
            archive.write(PATHS.code / name, name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    output = np.load(OUTPUT, mmap_mode="r")
    finite = all(
        np.isfinite(np.asarray(output[start : start + 5])).all()
        for start in range(0, len(output), 5)
    )
    report["integrity"] = {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": bool(finite),
        "zip_crc": True,
        "zip_members": members,
        "zip_bytes": int(ZIP_OUT.stat().st_size),
        "sha256": v533.digest(ZIP_OUT, "sha256"),
        "md5": v533.digest(ZIP_OUT, "md5"),
    }


def main() -> None:
    validate_inputs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    test_positions = np.load(TEST_POS).astype(np.float32)
    train_positions = np.load(v545.TRAIN_POS).astype(np.float32)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    test_regimes = v543.regime_of(geometry["test_blocked"]).astype("U8")
    v543_actions = np.load(V543_ACTIONS)
    if not np.array_equal(test_regimes, v543_actions["test_regimes"].astype("U8")):
        raise ValueError("test regime mismatch")
    base_physical = normalize_rows(v543_actions["physical_target"].astype(np.float32))
    anchor = normalize_rows(v543_actions["v541_final"].astype(np.float32))

    bank, source_names = build_test_bank()
    metadata = build_test_path_metadata()
    test_records = build_test_events(
        bank, anchor, base_physical, metadata, test_positions
    )
    train_records = v546.build_event_cache()
    predicted = predict_test_events(
        train_records,
        test_records,
        train_positions,
        test_positions,
    )
    choices, gains = v547.derive_event_actions(predicted)
    path_action = v546.apply_event_actions(
        base_physical,
        bank,
        metadata["owner_path"],
        test_records,
        np.arange(len(test_records["point"]), dtype=np.int64),
        choices,
        gains,
    )
    test_features, feature_names = build_test_gate_features(
        test_records,
        predicted,
        choices,
        gains,
        base_physical,
        path_action,
        anchor,
    )
    gate = predict_test_gate(test_features)
    corrected_target, target_diagnostics = build_target(
        base_physical, path_action, gate, test_regimes
    )
    np.save(TARGET, corrected_target)
    deployment_alpha = v543_actions["deployment_alpha"].astype(np.float32)
    effective_active = gate["active"] & (deployment_alpha > 1e-6)
    np.savez_compressed(
        ACTIONS,
        source_names=np.asarray(source_names),
        test_regimes=test_regimes,
        event_point=test_records["point"],
        event_path=test_records["path"],
        event_choices=choices,
        event_gains=gains,
        event_predictions=predicted,
        base_physical=base_physical,
        path_action=path_action,
        predicted_gain=gate["predicted_gain"],
        positive_probability=gate["positive_probability"],
        gate_active=gate["active"],
        effective_active=effective_active,
        fold_predicted_gain=gate["fold_predicted_gain"],
        fold_probability=gate["fold_probability"],
        corrected_target=corrected_target,
    )

    v533.OUTPUT = OUTPUT
    rendering = v533.render(corrected_target)
    comparisons = {
        "v540": compare_channels(OUTPUT, V540_OUTPUT),
        "v543": compare_channels(OUTPUT, V543_OUTPUT),
    }
    v547_report = json.loads(v547.REPORT.read_text(encoding="utf-8"))
    report: dict[str, object] = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "The v547 point-level net-gain gate can transfer the v546 path-event action to "
            "hidden test points while retaining the online-best v540 carrier wherever the "
            "complete action is not predicted to help."
        ),
        "new_information": [
            "complete v475 test-coordinate path route tokens",
            "19-source path-block candidate composition",
            "five-model v546 path-event utility ensemble",
            "five-model v547 point-level net-gain ensemble",
        ],
        "frozen_oof_evidence": v547_report["decision"],
        "deployment": {
            "path_events": int(len(test_records["point"])),
            "point_feature_count": int(len(feature_names)),
            "candidate_sources": SOURCE_COUNT,
            "first_layer_models": 5,
            "point_gate_models": 5,
            "gate_rule": "mean predicted gain > 0 and mean positive probability > 0.5",
            "carrier": "v540 joint target, v458 linkwise PDP/power/phase renderer",
            "action_mapping": "v546 path action relative to v543 physical target, mapped through deployment alpha",
            **target_diagnostics,
        },
        "rendering": rendering,
        "comparisons": comparisons,
        "outputs": {
            "channel": PATHS.relative(OUTPUT),
            "target": PATHS.relative(TARGET),
            "actions": PATHS.relative(ACTIONS),
            "test_bank": PATHS.relative(TEST_BANK),
            "test_path_metadata": PATHS.relative(TEST_META),
            "test_events": PATHS.relative(TEST_EVENTS),
            "test_features": PATHS.relative(TEST_FEATURES),
            "report": PATHS.relative(REPORT),
            "archive": PATHS.relative(ZIP_OUT),
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_package(report)
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "archive": report["outputs"]["archive"],
                "active": target_diagnostics["effective_active_count"],
                "comparison_v540": comparisons["v540"],
                "replay": rendering["replay"],
                "sha256": report["integrity"]["sha256"],
                "md5": report["integrity"]["md5"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
