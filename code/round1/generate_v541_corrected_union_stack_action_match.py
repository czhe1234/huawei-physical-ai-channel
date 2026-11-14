#!/usr/bin/env python3
"""Deploy the v541 old/corrected 18-source deterministic stack."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np

import analyze_v434_sionna_joint_sh_calibration as v434
import analyze_v494_sionna_configuration_stacking as v494
import analyze_v531_expanded_sbi_configuration_library as v531
import analyze_v541_corrected_full_library_sbi as v541
import generate_v533_expanded_physical_sbi_action_match as v533
import generate_v458_sionna_multimouth_extreme_diagnostic as v458
import train_v469_neural_sionna_calibrator as v469
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 541
VERSION = "v541_corrected_union_stack_action_match"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
SOURCE_DIR = CACHE_DIR / "physical_sources"
V537_SOURCE_DIR = PATHS.cache_dir(537) / "physical_sources"
OUTPUT = CACHE_DIR / "v541_Round1_Test_Channel.npy"
TARGET = CACHE_DIR / "v541_corrected_union_stack_joint_target.npy"
ACTIONS = CACHE_DIR / "v541_corrected_union_stack_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"
V531_REPORT = PATHS.version_report_dir(531) / "v531_expanded_sbi_configuration_library.json"
V531_OOF = PATHS.cache_dir(531) / "v531_expanded_sbi_configuration_library_oof.npz"
V541_REPORT = PATHS.version_report_dir(541) / "v541_old_corrected_union_sbi.json"
V541_OOF = (
    PATHS.cache_dir(541)
    / "v541_old_corrected_union_sbi"
    / "v541_old_corrected_union_sbi_oof.npz"
)


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def source_paths(name: str, selection: str = "all") -> dict[str, Path]:
    suffix = "" if selection == "all" else f"_{selection}"
    prefix = f"{name}_test{suffix}"
    return {
        "joint": SOURCE_DIR / f"{prefix}_joint_sh.npy",
        "log_power": SOURCE_DIR / f"{prefix}_log_power.npy",
        "done": SOURCE_DIR / f"{prefix}_done.npy",
        "indices": SOURCE_DIR / f"{prefix}_indices.npy",
    }


def load_complete_test(name: str) -> tuple[np.ndarray, np.ndarray]:
    paths = source_paths(name)
    for path in paths.values():
        require(path)
    done = np.load(paths["done"])
    indices = np.load(paths["indices"]).astype(np.int64)
    if not np.all(done) or not np.array_equal(indices, np.arange(500)):
        raise ValueError(f"incomplete corrected test source: {name}")
    value = np.load(paths["joint"], mmap_mode="r").reshape(500, -1)
    total = np.exp(np.load(paths["log_power"]).astype(np.float64))
    return v533.normalize_rows(value), total.astype(np.float32)


def seed_mean(paths: tuple[Path, ...], fallback: np.ndarray) -> np.ndarray:
    values = []
    for path in paths:
        done_path = path.with_name(path.name.replace("_joint_sh.npy", "_done.npy"))
        index_path = path.with_name(path.name.replace("_joint_sh.npy", "_indices.npy"))
        for required in (path, done_path, index_path):
            require(required)
        if not np.all(np.load(done_path)):
            raise ValueError(f"incomplete seed source: {path}")
        indices = np.load(index_path).astype(np.int64)
        if not np.array_equal(indices, np.arange(500)):
            raise ValueError(f"unexpected seed indices: {path}")
        value = np.load(path, mmap_mode="r").reshape(500, -1)
        values.append(v531.clean_source(value, fallback))
    return v533.normalize_rows(np.mean(values, axis=0))


def load_corrected_test_sources(
    test_regimes: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    high, high_total = load_complete_test("corrected_depth16_f6p0")
    low_loaded = [
        load_complete_test(name)
        for name in (
            "corrected_depth5_f2p6",
            "corrected_depth5_f6p0",
            "corrected_depth5_f6p7",
        )
    ]
    lows = [item[0] for item in low_loaded]
    low_totals = [item[1] for item in low_loaded]
    robust, quality, _ = v434.robust_source(high, high_total, lows, low_totals)
    all_interaction, _ = load_complete_test("corrected_depth5_all_f6p0")
    shallow = seed_mean(
        tuple(
            V537_SOURCE_DIR
            / f"corrected_array_d10_sc015_seed{seed}_test_joint_sh.npy"
            for seed in (42, 206, 420)
        ),
        robust,
    )
    deep = seed_mean(
        tuple(
            SOURCE_DIR
            / f"corrected_depth20_scatter015_seed{seed}_test_joint_sh.npy"
            for seed in (42, 206)
        ),
        robust,
    )

    edge_paths = source_paths("corrected_depth16_nonsynthetic_edge", selection="heavy")
    for path in edge_paths.values():
        require(path)
    edge_ids = np.load(edge_paths["indices"]).astype(np.int64)
    expected_edge_ids = np.flatnonzero(test_regimes == "heavy").astype(np.int64)
    if not np.array_equal(edge_ids, expected_edge_ids) or not np.all(
        np.load(edge_paths["done"])
    ):
        raise ValueError("incomplete corrected heavy-edge test source")
    edge_rows = np.load(edge_paths["joint"], mmap_mode="r").reshape(len(edge_ids), -1)
    edge = robust.copy()
    edge[edge_ids] = v531.clean_source(edge_rows, edge[edge_ids])

    sources = [robust, high, *lows, all_interaction, shallow, deep, edge]
    sources = [v531.clean_source(value, robust) for value in sources]
    return sources, quality.astype(np.float32), list(v541.CORRECTED_NAMES)


def effective_final(
    baseline: np.ndarray,
    target: np.ndarray,
    alpha_by_regime: dict[str, float],
    quality: np.ndarray,
    regimes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    effective = np.clip(
        np.asarray([alpha_by_regime[value] for value in regimes], dtype=np.float32)
        * quality,
        0.0,
        1.0,
    )
    final = v533.normalize_rows(
        baseline * (1.0 - effective[:, None]) + target * effective[:, None]
    )
    return final, effective


def oof_evidence(regimes: np.ndarray) -> dict[str, object]:
    with np.load(V531_OOF) as old_cache, np.load(V541_OOF) as new_cache:
        old_score = np.asarray(
            old_cache[f"{v531.HYBRID_METHOD}_score"], dtype=np.float64
        )
        new_score = np.asarray(new_cache["expanded_stack_score"], dtype=np.float64)
        fold_id = np.asarray(old_cache["fold_id"], dtype=np.int16)
    gain = new_score - old_score
    fold_gains = [float(np.mean(gain[fold_id == fold])) for fold in range(5)]
    return {
        "comparison": "v541 expanded deterministic stack versus v531 hybrid",
        "mean_gain": float(np.mean(gain)),
        "fold_gains": fold_gains,
        "fold_wins": int(sum(value > 0.0 for value in fold_gains)),
        "regime_gains": {
            regime: float(np.mean(gain[regimes == regime]))
            for regime in v494.REGIMES
        },
        "deployment_policy": (
            "Local OOF is diagnostic only. The user explicitly removed the fixed +0.001 "
            "packaging gate because online and local rankings are not identical."
        ),
    }


def build_target() -> tuple[np.ndarray, dict[str, object]]:
    old_report = json.loads(V531_REPORT.read_text(encoding="utf-8"))
    union_report = json.loads(V541_REPORT.read_text(encoding="utf-8"))
    if union_report["decision_gate"]["best_method"] != "expanded_stack":
        raise ValueError("v541 best union method is no longer expanded_stack")

    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    train_regimes = regime_of(geometry["train_blocked"]).astype("U8")
    test_regimes = regime_of(geometry["test_blocked"]).astype("U8")
    train_positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_positions = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)

    truth, corrected_train, corrected_quality_train, corrected_names = (
        v541.load_corrected_sources()
    )
    old_train = v531.load_expanded_sources()
    old_names = list(old_train[3])
    train_sources = list(old_train[1]) + list(corrected_train)
    train_names = [f"old_{name}" for name in old_names] + corrected_names
    report_names = [str(value) for value in union_report["sources"]]
    if train_names != report_names:
        raise ValueError(f"v541 train source mismatch: {train_names} != {report_names}")
    if not np.allclose(truth, old_train[0]):
        raise ValueError("old and corrected train truth arrays differ")

    old_test, old_quality_test = v533.load_test_sources(old_names)
    corrected_test, corrected_quality_test, corrected_test_names = (
        load_corrected_test_sources(test_regimes)
    )
    if corrected_test_names != corrected_names:
        raise ValueError("corrected train/test source order differs")
    test_sources = list(old_test) + list(corrected_test)
    union_quality = np.maximum(old_quality_test, corrected_quality_test)

    calibrated = v533.v496.calibrated_test_targets(
        truth,
        train_sources,
        test_sources,
        train_regimes,
        test_regimes,
    )
    baseline = v533.v496.test_neighbor_baseline(
        truth,
        train_positions,
        test_positions,
        geometry["train_blocked"].astype(np.float32),
        geometry["test_blocked"].astype(np.float32),
        train_regimes,
        test_regimes,
    )

    six_weights = v533.old_mean_weights(old_report)
    new_weights = v533.mean_weights(union_report, "expanded_stack")
    six_target = v533.v529.weighted_rows(
        calibrated[: len(v494.SOURCE_NAMES)], six_weights, test_regimes
    )
    new_target = v533.v529.weighted_rows(calibrated, new_weights, test_regimes)
    six_final, six_effective = effective_final(
        baseline,
        six_target,
        v533.mean_alpha(old_report, "old_stack"),
        old_quality_test,
        test_regimes,
    )
    new_final, new_effective = effective_final(
        baseline,
        new_target,
        v533.mean_alpha(union_report, "expanded_stack"),
        union_quality,
        test_regimes,
    )
    desired_action = np.clip(
        np.log(np.maximum(new_final, v533.FLOOR))
        - np.log(np.maximum(six_final, v533.FLOOR)),
        -1.5,
        1.5,
    ).astype(np.float32)

    joint_actions = np.load(v458.JOINT_ACTIONS)
    if not np.array_equal(joint_actions["regimes"].astype("U8"), test_regimes):
        raise ValueError("v458 and v541 regime labels differ")
    point_quality = joint_actions["point_quality"].astype(np.float32)
    profile = next(item for item in v458.JOINT_PROFILES if item.name == "tail085")
    deployment_alpha = v458.joint_alpha(profile, test_regimes, point_quality)
    target_log_delta = np.zeros_like(desired_action)
    active = deployment_alpha > 1e-6
    target_log_delta[active] = desired_action[active] / deployment_alpha[active, None]
    target_log_delta = np.clip(target_log_delta, -1.5, 1.5)
    neural = v533.normalize_rows(np.load(v469.JOINT_PRED, mmap_mode="r"))
    corrected = v533.normalize_rows(neural * np.exp(target_log_delta).astype(np.float32))

    np.save(TARGET, corrected.astype(np.float32))
    np.savez_compressed(
        ACTIONS,
        source_names=np.asarray(report_names),
        test_regimes=test_regimes,
        old_quality=old_quality_test,
        corrected_quality=corrected_quality_test,
        union_quality=union_quality,
        baseline=baseline,
        six_target=six_target,
        new_target=new_target,
        six_final=six_final,
        new_final=new_final,
        six_effective=six_effective,
        new_effective=new_effective,
        desired_action=desired_action,
        deployment_alpha=deployment_alpha,
        target_log_delta=target_log_delta,
        new_weights=np.asarray([new_weights[value] for value in v494.REGIMES]),
    )
    evidence = oof_evidence(train_regimes)
    evidence.update(
        {
            "selected_method": "expanded_stack",
            "source_count": len(report_names),
            "quality_rule": "max(old_v531_quality, corrected_array_quality)",
            "desired_action_abs": {
                "mean": float(np.mean(np.abs(desired_action))),
                "p90": float(np.quantile(np.abs(desired_action), 0.90)),
                "p99": float(np.quantile(np.abs(desired_action), 0.99)),
            },
            "target_log_delta_abs": {
                "mean": float(np.mean(np.abs(target_log_delta))),
                "p90": float(np.quantile(np.abs(target_log_delta), 0.90)),
                "p99": float(np.quantile(np.abs(target_log_delta), 0.99)),
            },
        }
    )
    return corrected, evidence


def compare_to(path: Path) -> dict[str, float]:
    require(path)
    left = np.load(OUTPUT, mmap_mode="r")
    right = np.load(path, mmap_mode="r")
    error = 0.0
    energy = 0.0
    cosine = 0.0
    count = 0
    for start in range(0, len(left), 5):
        stop = min(start + 5, len(left))
        a = np.asarray(left[start:stop], dtype=np.complex64)
        b = np.asarray(right[start:stop], dtype=np.complex64)
        error += float(np.sum(np.abs(a - b) ** 2, dtype=np.float64))
        energy += float(np.sum(np.abs(b) ** 2, dtype=np.float64))
        inner = np.sum(np.conj(b) * a, axis=(1, 2, 3), dtype=np.complex128)
        norm = np.sqrt(
            np.sum(np.abs(a) ** 2, axis=(1, 2, 3), dtype=np.float64)
            * np.sum(np.abs(b) ** 2, axis=(1, 2, 3), dtype=np.float64)
        )
        cosine += float(np.sum(np.abs(inner) / np.maximum(norm, 1e-30)))
        count += stop - start
    return {
        "nmse": error / max(energy, 1e-30),
        "complex_cosine": cosine / max(count, 1),
    }


def write_package(report: dict[str, object]) -> None:
    checkpoint = PATHS.checkpoints / v469.VERSION / "neural_sionna_calibrator.pt"
    require(checkpoint)
    dependencies = (
        "generate_v541_corrected_union_stack_action_match.py",
        "generate_v541_corrected_full_library_spectra.py",
        "analyze_v541_corrected_full_library_sbi.py",
        "analyze_v531_expanded_sbi_configuration_library.py",
        "analyze_v529_sbi_configuration_posterior.py",
        "analyze_v494_sionna_configuration_stacking.py",
        "generate_v533_expanded_physical_sbi_action_match.py",
        "generate_v496_sionna_configuration_selector_joint.py",
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
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        archive.writestr("model/", b"")
        archive.write(checkpoint, "model/neural_sionna_calibrator.pt")
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
        "sha256": v533.digest(ZIP_OUT, "sha256"),
        "md5": v533.digest(ZIP_OUT, "md5"),
    }


def main() -> None:
    required = (
        V531_REPORT,
        V531_OOF,
        V541_REPORT,
        V541_OOF,
        v533.V491_OUTPUT,
        v458.BASE,
        v458.CALIBRATED_LINK_TARGET,
        v458.LINK_QUALITY,
        v458.POWER_ACTIONS,
        v458.JOINT_ACTIONS,
        v469.JOINT_PRED,
    )
    for path in required:
        require(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    v533.OUTPUT = OUTPUT
    target, evidence = build_target()
    rendering = v533.render(target)
    comparison = {
        "v491": compare_to(v533.V491_OUTPUT),
        "v533": compare_to(PATHS.cache_dir(533) / "v533_Round1_Test_Channel.npy"),
        "v540": compare_to(PATHS.cache_dir(540) / "v540_Round1_Test_Channel.npy"),
    }
    report: dict[str, object] = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "The corrected 4-wavelength physical library contains source-specific modes "
            "that are lost by full replacement. Deploying the exact v541 18-source "
            "deterministic stack provides an online probe without treating local OOF as "
            "a hard submission gate."
        ),
        "evidence": evidence,
        "renderer": {
            "base": PATHS.relative(v458.BASE),
            "joint_anchor": PATHS.relative(v469.JOINT_PRED),
            "joint_profile": "tail085",
            "link_scale": 0.40,
            "power_scale": 10.0,
            "cycles": v458.JOINT_CYCLES,
            "preserved": [
                "v458 calibrated linkwise PDP",
                "v458 point power",
                "base complex phase",
            ],
        },
        "rendering": rendering,
        "comparison": comparison,
        "online_score": None,
        "outputs": {
            "channel": PATHS.relative(OUTPUT),
            "target": PATHS.relative(TARGET),
            "actions": PATHS.relative(ACTIONS),
            "archive": PATHS.relative(ZIP_OUT),
            "report": PATHS.relative(REPORT),
        },
    }
    write_package(report)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "archive": PATHS.relative(ZIP_OUT),
                "replay": rendering["replay"]["score_traceback_leaked"],
                "oof_mean_gain": evidence["mean_gain"],
                "oof_fold_wins": evidence["fold_wins"],
                "comparison": comparison,
                "sha256": report["integrity"]["sha256"],
                "md5": report["integrity"]["md5"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
