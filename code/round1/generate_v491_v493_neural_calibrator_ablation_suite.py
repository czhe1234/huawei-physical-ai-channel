#!/usr/bin/env python3
"""Generate strict joint/link/power ablations of v469 around the v458 teacher."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import analyze_v208_strict_historical_replay as v208
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v425_sionna_full_slice_pas_completion_diagnostic as v425
import generate_v426_sionna_linkwise_pdp_completion_diagnostic as v426
import generate_v435_train_calibrated_sionna_joint_sh_diagnostic as v435
import generate_v458_sionna_multimouth_extreme_diagnostic as v458
import train_v469_neural_sionna_calibrator as v469
from project_paths import PATHS


VERSION = "v491_v493_neural_calibrator_ablation_suite"
REPORT_DIR = PATHS.version_report_dir(491)
SUITE_REPORT = REPORT_DIR / f"{VERSION}.json"
BATCH = 2
POWER_SCALE = 10.0


@dataclass(frozen=True)
class Probe:
    version: int
    name: str
    changed_component: str

    @property
    def cache(self) -> Path:
        return PATHS.cache_dir(self.version)

    @property
    def output(self) -> Path:
        return self.cache / f"v{self.version}_Round1_Test_Channel.npy"

    @property
    def report_dir(self) -> Path:
        return PATHS.version_report_dir(self.version)

    @property
    def report(self) -> Path:
        return self.report_dir / f"{self.name}.json"

    @property
    def archive(self) -> Path:
        return PATHS.round1_versions / f"{self.name}_candidate_pending.zip"


PROBES = (
    Probe(491, "v491_v469_neural_joint_only", "joint_s_h"),
    Probe(492, "v492_v469_neural_link_only", "linkwise_pdp"),
    Probe(493, "v493_v469_neural_power_only", "point_power"),
)


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def compare_channels(left_path: Path, right_path: Path) -> dict[str, float]:
    left = np.load(left_path, mmap_mode="r")
    right = np.load(right_path, mmap_mode="r")
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


def render_suite() -> dict[str, float | int]:
    source = np.load(v458.BASE, mmap_mode="r")
    teacher_support = np.load(v458.BEST_SUPPORT, mmap_mode="r")
    teacher_joint_actions = np.load(v458.JOINT_ACTIONS)
    teacher_joint = teacher_joint_actions["calibrated_target"].astype(np.float32)
    neural_joint = np.load(v469.JOINT_PRED, mmap_mode="r")
    teacher_link = np.load(v458.CALIBRATED_LINK_TARGET, mmap_mode="r")
    neural_link = np.load(v469.LINK_PRED, mmap_mode="r")
    link_quality = np.load(v458.LINK_QUALITY, mmap_mode="r")
    teacher_power = np.load(v458.POWER_ACTIONS)["final_log_delta"].astype(np.float32)
    neural_power = np.load(v469.POWER_PRED).astype(np.float32)
    regimes = teacher_joint_actions["regimes"].astype("U8")
    point_quality = teacher_joint_actions["point_quality"].astype(np.float32)
    profile = next(item for item in v458.JOINT_PROFILES if item.name == "tail085")
    alpha_joint = v458.joint_alpha(profile, regimes, point_quality)
    alpha_link = v458.link_alpha(
        0.40, regimes, np.asarray(link_quality, dtype=np.float32)
    )
    teacher_amplitude = np.exp(0.5 * POWER_SCALE * teacher_power).astype(np.float32)
    neural_amplitude = np.exp(0.5 * POWER_SCALE * neural_power).astype(np.float32)

    for probe in PROBES:
        probe.cache.mkdir(parents=True, exist_ok=True)
        probe.report_dir.mkdir(parents=True, exist_ok=True)
    joint_output = np.lib.format.open_memmap(
        PROBES[0].output, mode="w+", dtype=np.complex64, shape=source.shape
    )
    link_output = np.lib.format.open_memmap(
        PROBES[1].output, mode="w+", dtype=np.complex64, shape=source.shape
    )
    power_output = np.lib.format.open_memmap(
        PROBES[2].output, mode="w+", dtype=np.complex64, shape=source.shape
    )

    rebuild_error = 0.0
    rebuild_energy = 0.0
    rebuild_max_abs = 0.0
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        base_h = np.asarray(source[start:end], dtype=np.complex64)
        base_support, _ = v423.leaked_support(base_h)
        teacher_desired_support, _, _ = v435.build_desired_support(
            base_support,
            teacher_joint[start:end],
            alpha_joint[start:end],
        )
        neural_desired_support, _, _ = v435.build_desired_support(
            base_support,
            np.asarray(neural_joint[start:end], dtype=np.float32),
            alpha_joint[start:end],
        )
        base_link, _ = v426.linkwise_pdp(base_h)
        teacher_desired_link = v426.desired_linkwise_pdp(
            base_link,
            np.asarray(teacher_link[start:end], dtype=np.float32).reshape(
                end - start, 256, 4, 192
            ),
            alpha_link[start:end].reshape(end - start, 256, 4),
        )
        neural_desired_link = v426.desired_linkwise_pdp(
            base_link,
            np.asarray(neural_link[start:end], dtype=np.float32).reshape(
                end - start, 256, 4, 192
            ),
            alpha_link[start:end].reshape(end - start, 256, 4),
        )

        rebuilt = base_h.copy()
        joint_candidate = base_h.copy()
        link_candidate = base_h.copy()
        for _ in range(v458.JOINT_CYCLES):
            rebuilt = v425.project_full_support(rebuilt, teacher_desired_support)
            rebuilt = v426.restore_desired_pdp(
                rebuilt, base_h, teacher_desired_link
            )
            joint_candidate = v425.project_full_support(
                joint_candidate, neural_desired_support
            )
            joint_candidate = v426.restore_desired_pdp(
                joint_candidate, base_h, teacher_desired_link
            )
            link_candidate = v425.project_full_support(
                link_candidate, teacher_desired_support
            )
            link_candidate = v426.restore_desired_pdp(
                link_candidate, base_h, neural_desired_link
            )

        expected_support = np.asarray(teacher_support[start:end], dtype=np.complex64)
        rebuild_error += float(
            np.sum(np.abs(rebuilt - expected_support) ** 2, dtype=np.float64)
        )
        rebuild_energy += float(
            np.sum(np.abs(expected_support) ** 2, dtype=np.float64)
        )
        rebuild_max_abs = max(
            rebuild_max_abs, float(np.max(np.abs(rebuilt - expected_support)))
        )
        joint_output[start:end] = (
            joint_candidate * teacher_amplitude[start:end, None, None, None]
        ).astype(np.complex64)
        link_output[start:end] = (
            link_candidate * teacher_amplitude[start:end, None, None, None]
        ).astype(np.complex64)
        power_output[start:end] = (
            expected_support * neural_amplitude[start:end, None, None, None]
        ).astype(np.complex64)
        if end % 50 == 0 or end == len(source):
            joint_output.flush()
            link_output.flush()
            power_output.flush()
            print(f"v491-v493 render {end}/{len(source)}", flush=True)
    joint_output.flush()
    link_output.flush()
    power_output.flush()
    return {
        "teacher_support_rebuild_nmse": rebuild_error
        / max(rebuild_energy, 1e-30),
        "teacher_support_rebuild_max_abs": rebuild_max_abs,
        "point_count": len(source),
    }


def write_package(probe: Probe) -> dict[str, object]:
    checkpoint = PATHS.checkpoints / v469.VERSION / "neural_sionna_calibrator.pt"
    dependencies = (
        "train_v469_neural_sionna_calibrator.py",
        "generate_v458_sionna_multimouth_extreme_diagnostic.py",
        "generate_v435_train_calibrated_sionna_joint_sh_diagnostic.py",
        "generate_v426_sionna_linkwise_pdp_completion_diagnostic.py",
        "generate_v425_sionna_full_slice_pas_completion_diagnostic.py",
        "generate_v423_joint_sionna_pas_pdp_diagnostic.py",
        "analyze_v447_sionna_linkwise_pdp_shape_calibration.py",
        "analyze_v434_sionna_joint_sh_calibration.py",
        "project_paths.py",
    )
    with zipfile.ZipFile(
        probe.archive, "w", compression=zipfile.ZIP_STORED
    ) as archive:
        archive.write(probe.output, "Round1_Test_Channel.npy")
        archive.write(__file__, Path(__file__).name)
        archive.writestr("model/", b"")
        archive.write(checkpoint, "model/neural_sionna_calibrator.pt")
        for dependency in dependencies:
            archive.write(PATHS.code / dependency, dependency)
    with zipfile.ZipFile(probe.archive) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure for {probe.name}: {bad}")
    value = np.load(probe.output, mmap_mode="r")
    finite = all(
        np.isfinite(np.asarray(value[start : start + 5])).all()
        for start in range(0, len(value), 5)
    )
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(finite),
        "zip_crc": True,
        "zip_members": members,
        "sha256": digest(probe.archive, "sha256"),
        "md5": digest(probe.archive, "md5"),
    }


def main() -> None:
    reconstruction = render_suite()
    baseline_replay = v208.score_prediction(v458.OUTPUT)
    suite_rows = []
    for probe in PROBES:
        replay = v208.score_prediction(probe.output)
        comparison = compare_channels(probe.output, v458.OUTPUT)
        integrity = write_package(probe)
        report = {
            "method": probe.name,
            "status": "candidate_pending",
            "primary_hypothesis": (
                f"The near-zero v469-v458 online delta may hide cancellation; isolate the "
                f"neural {probe.changed_component} action while preserving both other v458 actions."
            ),
            "base": {
                "version": 458,
                "online_score": 0.739203,
                "replay": baseline_replay,
            },
            "changed_component": probe.changed_component,
            "preserved_components": [
                value
                for value in ("joint_s_h", "linkwise_pdp", "point_power")
                if value != probe.changed_component
            ],
            "renderer": {
                "base_channel": PATHS.relative(v458.BASE),
                "joint_profile": "tail085",
                "link_scale": 0.40,
                "power_scale": POWER_SCALE,
                "cycles": v458.JOINT_CYCLES,
            },
            "teacher_reconstruction": reconstruction,
            "replay": replay,
            "replay_delta_vs_v458": float(
                replay["score_traceback_leaked"]
                - baseline_replay["score_traceback_leaked"]
            ),
            "comparison_to_v458": comparison,
            "integrity": integrity,
            "online_score": None,
            "outputs": {
                "channel": PATHS.relative(probe.output),
                "archive": PATHS.relative(probe.archive),
                "report": PATHS.relative(probe.report),
            },
        }
        probe.report.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        suite_rows.append(
            {
                "version": probe.version,
                "name": probe.name,
                "changed_component": probe.changed_component,
                "replay_score": replay["score_traceback_leaked"],
                "replay_delta_vs_v458": report["replay_delta_vs_v458"],
                "nmse_to_v458": comparison["nmse"],
                "archive": PATHS.relative(probe.archive),
                "sha256": integrity["sha256"],
                "md5": integrity["md5"],
            }
        )
        print(json.dumps(suite_rows[-1], ensure_ascii=False), flush=True)
    SUITE_REPORT.write_text(
        json.dumps(
            {
                "method": VERSION,
                "status": "three_candidates_pending",
                "teacher_reconstruction": reconstruction,
                "baseline_v458_replay": baseline_replay,
                "probes": suite_rows,
                "report": PATHS.relative(SUITE_REPORT),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
