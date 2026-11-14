#!/usr/bin/env python3
"""Deploy the validated corrected-array SBI source only for medium test points."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np

import analyze_v494_sionna_configuration_stacking as v494
import analyze_v531_expanded_sbi_configuration_library as v531
import analyze_v538_corrected_array_medium_only as v538
import generate_v533_expanded_physical_sbi_action_match as v533
import generate_v458_sionna_multimouth_extreme_diagnostic as v458
import train_v469_neural_sionna_calibrator as v469
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 538
VERSION = "v538_corrected_array_medium_sbi_action_match"
ANALYSIS = v538
SOURCE_COMBINATION = "mean"
ENTRY_SCRIPT = "generate_v538_corrected_array_medium_sbi_action_match.py"
ANALYSIS_SCRIPT = "analyze_v538_corrected_array_medium_only.py"
PRIMARY_HYPOTHESIS = (
    "The v534/v535-corrected three-seed array source improves medium propagation "
    "support, while exact retention of v533 outside medium prevents the observed "
    "light-regime regression."
)
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUTPUT = CACHE_DIR / "v538_Round1_Test_Channel.npy"
TARGET = CACHE_DIR / "v538_corrected_array_medium_joint_target.npy"
ACTIONS = CACHE_DIR / "v538_corrected_array_medium_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"
V531_REPORT = PATHS.version_report_dir(531) / "v531_expanded_sbi_configuration_library.json"
ANALYSIS_REPORT = PATHS.version_report_dir(538) / "v538_corrected_array_medium_only.json"
SOURCE_DIR = PATHS.cache_dir(537) / "physical_sources"


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def test_source_paths() -> tuple[Path, ...]:
    return tuple(
        SOURCE_DIR / f"corrected_array_d10_sc015_seed{seed}_test_joint_sh.npy"
        for seed in ANALYSIS.SEEDS
    )


def combined_test_source(paths: tuple[Path, ...], fallback: np.ndarray) -> np.ndarray:
    values = []
    for path in paths:
        require(path)
        value = np.load(path, mmap_mode="r").reshape(len(fallback), -1)
        values.append(v531.clean_source(value, fallback))
    if SOURCE_COMBINATION == "mean":
        combined = np.mean(values, axis=0)
    elif SOURCE_COMBINATION == "max":
        combined = np.max(np.stack(values), axis=0)
    else:
        raise ValueError(f"unsupported source combination: {SOURCE_COMBINATION}")
    return v533.normalize_rows(combined)


def load_test_sources(source_names: list[str]) -> tuple[list[np.ndarray], np.ndarray]:
    old_names = source_names[:-1]
    if source_names[-1] != ANALYSIS.SOURCE_NAME:
        raise ValueError(f"unexpected corrected source order: {source_names}")
    sources, quality = v533.load_test_sources(old_names)
    corrected = combined_test_source(test_source_paths(), sources[0])
    sources.append(corrected)
    return sources, quality


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


def build_target() -> tuple[np.ndarray, dict[str, object]]:
    old_report = json.loads(V531_REPORT.read_text(encoding="utf-8"))
    new_report = json.loads(ANALYSIS_REPORT.read_text(encoding="utf-8"))
    gate = new_report["medium_only_incremental_gate_vs_v533_oof"]
    if not bool(gate["pass"]):
        raise RuntimeError(f"v538 medium-only gate did not pass: {gate}")
    old_method = str(old_report["decision_gate"]["best_method"])
    new_method = str(gate["new_method"])
    old_source_names = [str(value) for value in old_report["sources"]]
    source_names = [str(value) for value in new_report["sources"]]
    if source_names[:-1] != old_source_names or source_names[-1] != ANALYSIS.SOURCE_NAME:
        raise ValueError("v538 source library is not a one-source extension of v531")

    truth, train_sources, _, train_names = ANALYSIS.load_corrected_sources(False)
    if train_names != source_names:
        raise ValueError(f"train source mismatch: {train_names} != {source_names}")
    test_sources, test_quality = load_test_sources(source_names)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    train_regimes = regime_of(geometry["train_blocked"]).astype("U8")
    test_regimes = regime_of(geometry["test_blocked"]).astype("U8")
    train_positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_positions = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
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
    old_weights = v533.mean_weights(old_report, old_method)
    new_weights = v533.mean_weights(new_report, new_method)
    six_target = v533.v529.weighted_rows(
        calibrated[: len(v494.SOURCE_NAMES)], six_weights, test_regimes
    )
    old_target = v533.v529.weighted_rows(
        calibrated[: len(old_source_names)], old_weights, test_regimes
    )
    new_target = v533.v529.weighted_rows(calibrated, new_weights, test_regimes)
    six_final, six_effective = effective_final(
        baseline,
        six_target,
        v533.mean_alpha(old_report, "old_stack"),
        test_quality,
        test_regimes,
    )
    old_final, old_effective = effective_final(
        baseline,
        old_target,
        v533.mean_alpha(old_report, old_method),
        test_quality,
        test_regimes,
    )
    new_final, new_effective = effective_final(
        baseline,
        new_target,
        v533.mean_alpha(new_report, new_method),
        test_quality,
        test_regimes,
    )
    medium = test_regimes == "medium"
    deployed_final = old_final.copy()
    deployed_final[medium] = new_final[medium]
    desired_action = np.clip(
        np.log(np.maximum(deployed_final, v533.FLOOR))
        - np.log(np.maximum(six_final, v533.FLOOR)),
        -1.5,
        1.5,
    ).astype(np.float32)

    joint_actions = np.load(v458.JOINT_ACTIONS)
    if not np.array_equal(joint_actions["regimes"].astype("U8"), test_regimes):
        raise ValueError("v458 and v538 regime labels differ")
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
        source_names=np.asarray(source_names),
        test_regimes=test_regimes,
        test_quality=test_quality,
        medium_mask=medium,
        six_final=six_final,
        old_v533_final=old_final,
        corrected_final=new_final,
        deployed_final=deployed_final,
        six_effective=six_effective,
        old_effective=old_effective,
        new_effective=new_effective,
        desired_action=desired_action,
        deployment_alpha=deployment_alpha,
        target_log_delta=target_log_delta,
    )
    incremental = np.log(np.maximum(deployed_final, v533.FLOOR)) - np.log(
        np.maximum(old_final, v533.FLOOR)
    )
    diagnostics = {
        "selected_method": new_method,
        "deployment_rule": "corrected-array source only for medium; retain v533 elsewhere",
        "source_names": source_names,
        "medium_test_count": int(np.count_nonzero(medium)),
        "strict_oof_gate": gate,
        "incremental_log_action_abs": {
            "all_mean": float(np.mean(np.abs(incremental))),
            "medium_mean": float(np.mean(np.abs(incremental[medium]))),
            "medium_p90": float(np.quantile(np.abs(incremental[medium]), 0.90)),
            "nonmedium_max": float(np.max(np.abs(incremental[~medium]))),
        },
    }
    return corrected, diagnostics


def write_package(report: dict[str, object]) -> None:
    checkpoint = PATHS.checkpoints / v469.VERSION / "neural_sionna_calibrator.pt"
    require(checkpoint)
    dependencies = (
        ENTRY_SCRIPT,
        ANALYSIS_SCRIPT,
        "generate_v538_corrected_array_medium_sbi_action_match.py",
        "analyze_v537_corrected_array_sbi_extension.py",
        "analyze_v538_corrected_array_medium_only.py",
        "generate_v537_corrected_array_spectra.py",
        "generate_v533_expanded_physical_sbi_action_match.py",
        "analyze_v531_expanded_sbi_configuration_library.py",
        "analyze_v529_sbi_configuration_posterior.py",
        "analyze_v494_sionna_configuration_stacking.py",
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
    for path in (
        V531_REPORT,
        ANALYSIS_REPORT,
        *ANALYSIS.SOURCE_PATHS,
        *ANALYSIS.SOURCE_DONE,
        *test_source_paths(),
        v533.V491_OUTPUT,
        v458.BASE,
        v458.CALIBRATED_LINK_TARGET,
        v458.LINK_QUALITY,
        v458.POWER_ACTIONS,
        v458.JOINT_ACTIONS,
        v469.JOINT_PRED,
    ):
        require(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    v533.OUTPUT = OUTPUT
    target, evidence = build_target()
    rendering = v533.render(target)
    comparison = v533.compare_to_v491()
    report: dict[str, object] = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": PRIMARY_HYPOTHESIS,
        "evidence": evidence,
        "rendering": rendering,
        "comparison_to_v491": comparison,
        "output": PATHS.relative(OUTPUT),
        "archive": PATHS.relative(ZIP_OUT),
    }
    write_package(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
