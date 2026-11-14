#!/usr/bin/env python3
"""Generate an action-matched deployment of the v495 Sionna selector."""

from __future__ import annotations

import hashlib
import json
import zipfile

import numpy as np

import analyze_v208_strict_historical_replay as v208
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v425_sionna_full_slice_pas_completion_diagnostic as v425
import generate_v426_sionna_linkwise_pdp_completion_diagnostic as v426
import generate_v435_train_calibrated_sionna_joint_sh_diagnostic as v435
import generate_v458_sionna_multimouth_extreme_diagnostic as v458
import generate_v496_sionna_configuration_selector_joint as v496
import train_v469_neural_sionna_calibrator as v469
from powerfield_common import TINY
from project_paths import PATHS


VERSION_NUMBER = 497
VERSION = "v497_sionna_selector_action_matched"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUTPUT = CACHE_DIR / "v497_Round1_Test_Channel.npy"
TARGET = CACHE_DIR / "v497_action_matched_joint_target.npy"
ACTIONS = CACHE_DIR / "v497_sionna_selector_action_matched_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"
V495_REPORT = PATHS.version_report_dir(495) / "v495_sionna_configuration_selector.json"
V496_ACTIONS = PATHS.cache_dir(496) / "v496_sionna_configuration_selector_actions.npz"
V491_OUTPUT = PATHS.cache_dir(491) / "v491_Round1_Test_Channel.npy"
BATCH = 2
REGIMES = ("clear", "light", "medium", "heavy")


def digest(path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 1e-10)
    return value / np.maximum(value.sum(axis=1, keepdims=True), TINY)


def mean_source_alpha() -> dict[str, dict[str, float]]:
    report = json.loads(V495_REPORT.read_text(encoding="utf-8"))
    output: dict[str, dict[str, float]] = {}
    for source in report["source_names"]:
        output[source] = {}
        for regime in REGIMES:
            output[source][regime] = float(
                np.mean(
                    [
                        fold["source_alphas"][source][regime]
                        for fold in json.loads(
                            (
                                PATHS.version_report_dir(494)
                                / "v494_sionna_configuration_stacking.json"
                            ).read_text(encoding="utf-8")
                        )["folds"]
                    ]
                )
            )
    return output


def build_target() -> tuple[np.ndarray, dict[str, object]]:
    data = np.load(V496_ACTIONS)
    selected = data["selected_source"].astype(np.int16)
    source_names = data["source_names"].astype("U32")
    original_delta = data["log_delta"].astype(np.float32)
    joint = np.load(v458.JOINT_ACTIONS)
    regimes = joint["regimes"].astype("U8")
    point_quality = joint["point_quality"].astype(np.float32)
    profile = next(item for item in v458.JOINT_PROFILES if item.name == "tail085")
    deployment_alpha = v458.joint_alpha(profile, regimes, point_quality)
    alpha_lookup = mean_source_alpha()
    validated_alpha = np.asarray(
        [
            alpha_lookup[str(source_names[source])][str(regime)]
            for source, regime in zip(selected, regimes, strict=True)
        ],
        dtype=np.float32,
    )
    transfer_scale = np.minimum(
        1.0, validated_alpha / np.maximum(deployment_alpha, 1e-6)
    ).astype(np.float32)
    transfer_scale[selected == 0] = 0.0
    effective_delta = original_delta * transfer_scale[:, None]
    neural = normalize_rows(np.load(v469.JOINT_PRED, mmap_mode="r"))
    corrected = normalize_rows(neural * np.exp(effective_delta).astype(np.float32))
    np.save(TARGET, corrected.astype(np.float32))
    np.savez_compressed(
        ACTIONS,
        source_names=source_names,
        selected_source=selected,
        deployment_alpha=deployment_alpha,
        validated_alpha=validated_alpha,
        transfer_scale=transfer_scale,
        original_log_delta=original_delta,
        effective_log_delta=effective_delta,
    )
    diagnostics = {
        "active_points": int(np.sum(selected != 0)),
        "transfer_scale": {
            "mean_active": float(np.mean(transfer_scale[selected != 0])),
            "p10_active": float(np.quantile(transfer_scale[selected != 0], 0.10)),
            "median_active": float(np.median(transfer_scale[selected != 0])),
            "p90_active": float(np.quantile(transfer_scale[selected != 0], 0.90)),
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
            print(f"v497 render {end}/{len(source)}", flush=True)
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
        "generate_v497_sionna_selector_action_matched.py",
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
        archive.write(
            v496.CHECKPOINT, "model/sionna_configuration_selector.joblib"
        )
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
    for path in (
        V495_REPORT,
        V496_ACTIONS,
        v496.CHECKPOINT,
        v469.JOINT_PRED,
        V491_OUTPUT,
        v458.BASE,
        v458.CALIBRATED_LINK_TARGET,
        v458.LINK_QUALITY,
        v458.POWER_ACTIONS,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    target, action_report = build_target()
    rendering = render(target)
    comparison = compare_to_v491()
    report: dict[str, object] = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v496 over-applied a selector validated at smaller regime-specific action levels. "
            "Matching the transferred target residual to the strict v495 source alpha should "
            "retain the selector signal without turning it into a new joint overdrive."
        ),
        "compliance": {
            "status": "diagnostic_noncompliant",
            "reason": "uses Sionna references traced at the 500 official test coordinates",
        },
        "evidence": {
            "v495_mean_gain_vs_old": 0.005241366568952799,
            "v495_fold_wins": 5,
            "v495_gain_over_shuffled": 0.00629607867449522,
            "v496_replay_delta_vs_v491": -0.003460635558622971,
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
            "checkpoint": PATHS.relative(v496.CHECKPOINT),
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
                "transfer_scale": action_report["transfer_scale"],
                "sha256": report["integrity"]["sha256"],
                "md5": report["integrity"]["md5"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
