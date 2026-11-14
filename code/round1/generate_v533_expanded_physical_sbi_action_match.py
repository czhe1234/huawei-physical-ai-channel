#!/usr/bin/env python3
"""Deploy the validated expanded physical posterior as a matched v491 action."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

import analyze_v208_strict_historical_replay as v208
import analyze_v494_sionna_configuration_stacking as v494
import analyze_v529_sbi_configuration_posterior as v529
import analyze_v531_expanded_sbi_configuration_library as v531
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v425_sionna_full_slice_pas_completion_diagnostic as v425
import generate_v426_sionna_linkwise_pdp_completion_diagnostic as v426
import generate_v435_train_calibrated_sionna_joint_sh_diagnostic as v435
import generate_v458_sionna_multimouth_extreme_diagnostic as v458
import generate_v496_sionna_configuration_selector_joint as v496
import train_v469_neural_sionna_calibrator as v469
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 533
VERSION = "v533_expanded_physical_sbi_action_match"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUTPUT = CACHE_DIR / "v533_Round1_Test_Channel.npy"
TARGET = CACHE_DIR / "v533_expanded_joint_target.npy"
ACTIONS = CACHE_DIR / "v533_expanded_physical_sbi_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"
V531_REPORT = PATHS.version_report_dir(531) / "v531_expanded_sbi_configuration_library.json"
V491_OUTPUT = PATHS.cache_dir(491) / "v491_Round1_Test_Channel.npy"
V530_DIR = PATHS.cache_dir(530)
V514_DIR = PATHS.cache_dir(514) / "reference_configs"
FLOOR = 1e-10
BATCH = 2


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), FLOOR)
    return value / np.maximum(value.sum(axis=1, keepdims=True), TINY)


def load_test_seed_average(paths: tuple[Path, ...], fallback: np.ndarray) -> np.ndarray:
    values = []
    for path in paths:
        require(path)
        value = np.load(path, mmap_mode="r").reshape(len(fallback), -1)
        values.append(v531.clean_source(value, fallback))
    return normalize_rows(np.mean(values, axis=0))


def load_test_sources(source_names: list[str]) -> tuple[list[np.ndarray], np.ndarray]:
    old_sources, quality = v496.load_test_sources()
    sources = list(old_sources)
    names = list(v494.SOURCE_NAMES)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    test_regimes = regime_of(geometry["test_blocked"]).astype("U8")
    for name in source_names[len(names) :]:
        if name == "depth10_scatter015_synthetic":
            paths = (V530_DIR / "depth10_scatter015_synthetic_test_joint_sh.npy",)
            sources.append(load_test_seed_average(paths, sources[0]))
        elif name == "depth10_scatter015_synthetic_seed_marginal_3":
            paths = (
                V530_DIR / "depth10_scatter015_synthetic_test_joint_sh.npy",
                V530_DIR / "depth10_scatter015_synthetic_seed206_test_joint_sh.npy",
                V530_DIR / "depth10_scatter015_synthetic_seed420_test_joint_sh.npy",
            )
            sources.append(load_test_seed_average(paths, sources[0]))
        elif name == "depth20_scatter015_synthetic_seed206":
            paths = (
                V530_DIR / "depth20_scatter015_synthetic_seed206_test_joint_sh.npy",
            )
            sources.append(load_test_seed_average(paths, sources[0]))
        elif name == "depth20_scatter015_synthetic_seed_marginal_2":
            paths = (
                V530_DIR / "depth20_scatter015_synthetic_seed206_test_joint_sh.npy",
                V530_DIR / "depth20_scatter015_synthetic_seed42_test_joint_sh.npy",
            )
            sources.append(load_test_seed_average(paths, sources[0]))
        elif name == "depth16_nonsynthetic_edge_heavy":
            edge_path = V514_DIR / "nonsynthetic_edge_points279_sh.npy"
            require(edge_path)
            edge_ids = np.flatnonzero(test_regimes == "heavy").astype(np.int64)
            edge_rows = np.load(edge_path, mmap_mode="r").reshape(len(edge_ids), -1)
            edge_source = sources[0].copy()
            edge_source[edge_ids] = v531.clean_source(
                edge_rows, edge_source[edge_ids]
            )
            sources.append(normalize_rows(edge_source))
        else:
            raise ValueError(f"unsupported expanded source: {name}")
        names.append(name)
    if names != source_names:
        raise ValueError(f"source order mismatch: report={source_names}, test={names}")
    return sources, quality.astype(np.float32)


def mean_weights(
    report: dict[str, object], method: str
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for regime in v494.REGIMES:
        local_method = method
        if method == v531.HYBRID_METHOD:
            local_method = "expanded_stack" if regime == "heavy" else "expanded_sbi_map"
        if local_method == "expanded_stack":
            rows = [fold["expanded_weights"][regime] for fold in report["folds"]]
        else:
            key = "weight" if local_method == "expanded_sbi" else "map_weight"
            rows = [
                fold["posterior"][regime][key] for fold in report["posteriors"]
            ]
        value = np.mean(np.asarray(rows, dtype=np.float64), axis=0)
        value = np.maximum(value, 0.0)
        output[regime] = (value / np.maximum(value.sum(), 1e-30)).astype(np.float32)
    return output


def old_mean_weights(report: dict[str, object]) -> dict[str, np.ndarray]:
    return {
        regime: np.mean(
            np.asarray(
                [fold["old_weights"][regime] for fold in report["folds"]],
                dtype=np.float32,
            ),
            axis=0,
        ).astype(np.float32)
        for regime in v494.REGIMES
    }


def mean_alpha(report: dict[str, object], method: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for regime in v494.REGIMES:
        local_method = method
        if method == v531.HYBRID_METHOD:
            local_method = "expanded_stack" if regime == "heavy" else "expanded_sbi_map"
        output[regime] = float(
            np.mean(
                [fold["alphas"][local_method][regime] for fold in report["folds"]]
            )
        )
    return output


def build_target() -> tuple[np.ndarray, dict[str, object]]:
    report = json.loads(V531_REPORT.read_text(encoding="utf-8"))
    gate = report["decision_gate"]
    if not bool(gate["pass"]):
        raise RuntimeError(f"v531 deployment gate did not pass: {gate}")
    method = str(gate["best_method"])
    source_names = [str(value) for value in report["sources"]]
    truth, train_sources, _, train_names = v531.load_expanded_sources(
        include_nonmetal=False
    )
    if train_names != source_names:
        raise ValueError(f"v531 train source mismatch: {train_names} != {source_names}")
    test_sources, test_quality = load_test_sources(source_names)

    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    train_regimes = regime_of(geometry["train_blocked"]).astype("U8")
    test_regimes = regime_of(geometry["test_blocked"]).astype("U8")
    train_positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_positions = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    calibrated = v496.calibrated_test_targets(
        truth,
        train_sources,
        test_sources,
        train_regimes,
        test_regimes,
    )
    new_weights = mean_weights(report, method)
    old_weights = old_mean_weights(report)
    old_target = v529.weighted_rows(
        calibrated[: len(v494.SOURCE_NAMES)], old_weights, test_regimes
    )
    new_target = v529.weighted_rows(calibrated, new_weights, test_regimes)
    baseline = v496.test_neighbor_baseline(
        truth,
        train_positions,
        test_positions,
        geometry["train_blocked"].astype(np.float32),
        geometry["test_blocked"].astype(np.float32),
        train_regimes,
        test_regimes,
    )

    old_alpha_by_regime = mean_alpha(report, "old_stack")
    new_alpha_by_regime = mean_alpha(report, method)
    old_effective = np.clip(
        np.asarray([old_alpha_by_regime[value] for value in test_regimes], dtype=np.float32)
        * test_quality,
        0.0,
        1.0,
    )
    new_effective = np.clip(
        np.asarray([new_alpha_by_regime[value] for value in test_regimes], dtype=np.float32)
        * test_quality,
        0.0,
        1.0,
    )
    old_final = normalize_rows(
        baseline * (1.0 - old_effective[:, None])
        + old_target * old_effective[:, None]
    )
    new_final = normalize_rows(
        baseline * (1.0 - new_effective[:, None])
        + new_target * new_effective[:, None]
    )
    desired_action = np.clip(
        np.log(np.maximum(new_final, FLOOR))
        - np.log(np.maximum(old_final, FLOOR)),
        -1.5,
        1.5,
    ).astype(np.float32)

    joint_actions = np.load(v458.JOINT_ACTIONS)
    deployment_regimes = joint_actions["regimes"].astype("U8")
    if not np.array_equal(deployment_regimes, test_regimes):
        raise ValueError("v458 and v531 regime labels differ")
    point_quality = joint_actions["point_quality"].astype(np.float32)
    profile = next(item for item in v458.JOINT_PROFILES if item.name == "tail085")
    deployment_alpha = v458.joint_alpha(profile, test_regimes, point_quality)
    target_log_delta = np.zeros_like(desired_action)
    active = deployment_alpha > 1e-6
    target_log_delta[active] = desired_action[active] / deployment_alpha[active, None]
    target_log_delta = np.clip(target_log_delta, -1.5, 1.5)
    neural = normalize_rows(np.load(v469.JOINT_PRED, mmap_mode="r"))
    corrected = normalize_rows(neural * np.exp(target_log_delta).astype(np.float32))
    np.save(TARGET, corrected.astype(np.float32))
    np.savez_compressed(
        ACTIONS,
        source_names=np.asarray(source_names),
        test_regimes=test_regimes,
        test_quality=test_quality,
        old_target=old_target,
        new_target=new_target,
        baseline=baseline,
        old_effective=old_effective,
        new_effective=new_effective,
        desired_action=desired_action,
        deployment_alpha=deployment_alpha,
        target_log_delta=target_log_delta,
        old_weights=np.asarray([old_weights[value] for value in v494.REGIMES]),
        new_weights=np.asarray([new_weights[value] for value in v494.REGIMES]),
    )
    diagnostics = {
        "selected_method": method,
        "source_names": source_names,
        "validated_gain_vs_old_stack": float(gate["best_mean_gain_vs_old_stack"]),
        "validated_fold_wins": int(gate["fold_wins"]),
        "old_alpha": old_alpha_by_regime,
        "new_alpha": new_alpha_by_regime,
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
        stop = min(start + BATCH, len(source))
        base_h = np.asarray(source[start:stop], dtype=np.complex64)
        base_support, _ = v423.leaked_support(base_h)
        desired_support, _, _ = v435.build_desired_support(
            base_support, target[start:stop], alpha_joint[start:stop]
        )
        base_link, _ = v426.linkwise_pdp(base_h)
        desired_link = v426.desired_linkwise_pdp(
            base_link,
            np.asarray(teacher_link[start:stop], dtype=np.float32).reshape(
                stop - start, 256, 4, 192
            ),
            alpha_link[start:stop].reshape(stop - start, 256, 4),
        )
        candidate = base_h.copy()
        for _ in range(v458.JOINT_CYCLES):
            candidate = v425.project_full_support(candidate, desired_support)
            candidate = v426.restore_desired_pdp(candidate, base_h, desired_link)
        output[start:stop] = (
            candidate * amplitude[start:stop, None, None, None]
        ).astype(np.complex64)
        if stop % 50 == 0 or stop == len(source):
            output.flush()
            print(f"v533 render {stop}/{len(source)}", flush=True)
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
    for path in (
        V531_REPORT,
        V491_OUTPUT,
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
    target, action_report = build_target()
    rendering = render(target)
    comparison = compare_to_v491()
    report: dict[str, object] = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "A posterior over physically distinct shallow-scattering, deep-scattering, "
            "and non-synthetic edge configurations adds path support missing from the old "
            "six-source library. Matching its strict OOF action onto v491 should transfer "
            "the new information without repeating selector overdrive."
        ),
        "evidence": action_report,
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
        **rendering,
        "comparison_to_v491": comparison,
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
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "archive": PATHS.relative(ZIP_OUT),
                "replay": rendering["replay"]["score_traceback_leaked"],
                "replay_delta_vs_v491": rendering["replay_delta_vs_v491"],
                "nmse_to_v491": comparison["nmse"],
                "evidence": action_report,
                "sha256": report["integrity"]["sha256"],
                "md5": report["integrity"]["md5"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
