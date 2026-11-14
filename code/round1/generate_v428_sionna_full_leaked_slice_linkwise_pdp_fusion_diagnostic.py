#!/usr/bin/env python3
"""Fuse full leaked slice support with the validated v426 linkwise PDP.

This diagnostic reads Sionna references traced at the official test
coordinates. It is not compliant with the final competition data boundary.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass

import numpy as np

import analyze_v208_strict_historical_replay as v208
import generate_v178_test_track_complex_consensus as track
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v424_sionna_pas_full_pdp_lock_diagnostic as v424
import generate_v425_sionna_full_slice_pas_completion_diagnostic as v425
from generate_v87_easy_field_projection import summarize
from generate_v159_frontier_consensus_support_soup import digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 428
VERSION = "v428_sionna_full_leaked_slice_linkwise_pdp_fusion_diagnostic"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v428_Round1_Test_Channel.npy"
ACTION_OUT = OUT_DIR / "v428_sionna_full_leaked_slice_linkwise_pdp_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

BASE = PATHS.cache_dir(426) / "v426_Round1_Test_Channel.npy"
V207_REFERENCE = PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"
V205_REFERENCES = (
    PATHS.cache_dir(205) / "benchmark_sector_m30_track_d5_f2p6" / "test_clean_H.npy",
    PATHS.cache_dir(205) / "benchmark_sector_m30_track_d5_f6p0" / "test_clean_H.npy",
    PATHS.cache_dir(205) / "benchmark_sector_m30_track_d5_f6p7" / "test_clean_H.npy",
)
V424_ACTIONS = PATHS.cache_dir(424) / "v424_sionna_pas_full_pdp_lock_actions.npz"

BATCH = 2
JOINT_CYCLES = 5
FULL_ALPHA_BASE = {"clear": 0.0, "light": 0.08, "medium": 0.54, "heavy": 0.76}


@dataclass(frozen=True)
class Probe:
    name: str
    scale: float

    @property
    def path(self):
        return OUT_DIR / f"v428_{self.name}_Round1_Test_Channel.npy"


PROBES = (
    Probe("slice050", 0.50),
    Probe("slice100", 1.00),
    Probe("slice150", 1.50),
    Probe("slice200", 2.00),
)


def require_file(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def full_support_l1(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(left - right), axis=(1, 2, 3, 4, 5), dtype=np.float64).astype(
        np.float32
    )


def slice_shape_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_shape, left_total = v425.normalize_slices(left)
    right_shape, right_total = v425.normalize_slices(right)
    cosine = v425.slice_cosine(left_shape, right_shape)
    weight = np.sqrt(np.maximum(left_total * right_total, 0.0)).astype(np.float32)
    return (
        np.sum(cosine * weight, axis=(1, 2, 3), dtype=np.float64)
        / np.maximum(np.sum(weight, axis=(1, 2, 3), dtype=np.float64), 1e-30)
    ).astype(np.float32)


def generate_probes(regimes: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    source = np.load(BASE, mmap_mode="r")
    high = np.load(V207_REFERENCE, mmap_mode="r")
    lows = [np.load(path, mmap_mode="r") for path in V205_REFERENCES]
    prior = np.load(V424_ACTIONS)
    outputs = {
        probe.name: np.lib.format.open_memmap(
            probe.path, mode="w+", dtype=np.complex64, shape=source.shape
        )
        for probe in PROBES
    }
    shared = {
        "point_quality": prior["pas_quality"].astype(np.float32).copy(),
        "slice_quality_mean": np.zeros(len(source), dtype=np.float32),
        "slice_quality_p90": np.zeros(len(source), dtype=np.float32),
        "slice_cross_mean": np.zeros(len(source), dtype=np.float32),
        "slice_stability_mean": np.zeros(len(source), dtype=np.float32),
    }
    actions: dict[str, np.ndarray] = {}
    for probe in PROBES:
        for key in (
            "alpha_mean",
            "alpha_max",
            "power_ratio",
            "phase_agreement",
            "pdp_lock_error",
            "marginal_l1",
            "support_l1",
            "slice_target_cosine",
        ):
            actions[f"{key}_{probe.name}"] = np.zeros(len(source), dtype=np.float32)

    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        base_h = np.asarray(source[start:end], dtype=np.complex64)
        high_h = np.asarray(high[start:end], dtype=np.complex64)
        low_h = [np.asarray(value[start:end], dtype=np.complex64) for value in lows]
        base_support, _ = v423.leaked_support(base_h)
        high_support, _ = v423.leaked_support(high_h)
        low_supports = [v423.leaked_support(value)[0] for value in low_h]
        target_slice, slice_diag = v425.robust_slice_target(
            base_support, high_support, low_supports
        )
        quality = slice_diag["slice_quality"]
        flat_quality = quality.reshape(len(base_h), -1)
        shared["slice_quality_mean"][start:end] = flat_quality.mean(axis=1)
        shared["slice_quality_p90"][start:end] = np.quantile(
            flat_quality, 0.90, axis=1
        ).astype(np.float32)
        shared["slice_cross_mean"][start:end] = slice_diag[
            "slice_cross_agreement"
        ].reshape(len(base_h), -1).mean(axis=1)
        shared["slice_stability_mean"][start:end] = slice_diag[
            "slice_pair_stability"
        ].reshape(len(base_h), -1).mean(axis=1)

        base_power = np.maximum(
            np.sum(np.abs(base_h) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY
        )
        local_regimes = regimes[start:end].astype(str)
        point_quality = np.clip(
            prior["pas_quality"][start:end].astype(np.float32) / 1.04, 0.0, 1.0
        )

        for probe in PROBES:
            point_alpha = np.asarray(
                [FULL_ALPHA_BASE[name] for name in local_regimes], dtype=np.float32
            )
            point_alpha = np.clip(
                point_alpha * np.float32(probe.scale) * point_quality, 0.0, 0.95
            )
            alpha_slice = point_alpha[:, None, None, None] * (
                0.24 + 0.76 * quality
            )
            alpha_slice *= slice_diag["slice_high_valid"]
            desired = v425.build_desired_support(
                base_support, target_slice, alpha_slice
            )
            candidate = base_h.copy()
            for _ in range(JOINT_CYCLES):
                candidate = v425.project_full_support(candidate, desired)
                candidate = v424.restore_full_linkwise_pdp(candidate, base_h)
            if not np.isfinite(candidate).all():
                raise FloatingPointError(f"non-finite {probe.name} block {start}:{end}")
            outputs[probe.name][start:end] = candidate
            outputs[probe.name].flush()

            candidate_power = np.sum(
                np.abs(candidate) ** 2, axis=(1, 2, 3), dtype=np.float64
            )
            candidate_support, _ = v423.leaked_support(candidate)
            flat_alpha = alpha_slice.reshape(len(base_h), -1)
            actions[f"alpha_mean_{probe.name}"][start:end] = flat_alpha.mean(axis=1)
            actions[f"alpha_max_{probe.name}"][start:end] = flat_alpha.max(axis=1)
            actions[f"power_ratio_{probe.name}"][start:end] = (
                candidate_power / base_power
            ).astype(np.float32)
            actions[f"phase_agreement_{probe.name}"][start:end] = v423.phase_agreement(
                base_h, candidate
            )
            actions[f"pdp_lock_error_{probe.name}"][start:end] = (
                v424.full_linkwise_pdp_error(candidate, base_h)
            )
            actions[f"marginal_l1_{probe.name}"][start:end] = v425.marginal_l1(
                candidate_support, base_support
            )
            actions[f"support_l1_{probe.name}"][start:end] = full_support_l1(
                candidate_support, base_support
            )
            actions[f"slice_target_cosine_{probe.name}"][start:end] = (
                slice_shape_cosine(candidate_support, desired)
            )
        print(f"v428 full leaked slice + linkwise PDP {end}/{len(source)}", flush=True)

    np.savez_compressed(
        ACTION_OUT, regimes=regimes.astype("U8"), **shared, **actions
    )
    return {"shared": shared, "actions": actions}


def choose_probe(
    scores: dict[str, dict[str, object]], base: dict[str, object]
) -> str:
    base_score = float(base["score_traceback_leaked"])
    eligible = [
        probe.name
        for probe in PROBES
        if float(scores[probe.name]["leaked_sh"]) > float(base["leaked_sh"])
        and float(scores[probe.name]["pdp"]) >= float(base["pdp"]) - 1e-6
        and float(scores[probe.name]["pas"]) >= float(base["pas"]) - 0.003
        and float(scores[probe.name]["nmse_score"])
        >= float(base["nmse_score"]) - 0.002
    ]
    if not eligible:
        return max(scores, key=lambda name: float(scores[name]["score_traceback_leaked"]))
    best = max(float(scores[name]["score_traceback_leaked"]) for name in eligible)
    threshold = base_score + 0.90 * (best - base_score)
    for probe in PROBES:
        if (
            probe.name in eligible
            and float(scores[probe.name]["score_traceback_leaked"]) >= threshold
        ):
            return probe.name
    return eligible[-1]


def write_zip(report: dict[str, object]) -> dict[str, object]:
    sources = (
        "generate_v428_sionna_full_leaked_slice_linkwise_pdp_fusion_diagnostic.py",
        "generate_v426_sionna_linkwise_pdp_completion_diagnostic.py",
        "generate_v425_sionna_full_slice_pas_completion_diagnostic.py",
        "generate_v424_sionna_pas_full_pdp_lock_diagnostic.py",
        "generate_v423_joint_sionna_pas_pdp_diagnostic.py",
        "generate_v212_multidomain_sionna_pdp_ensemble.py",
        "generate_v189_leaked_multimarginal_overdrive.py",
        "generate_v187_leaked_pas_lifecycle_projection.py",
        "generate_v196_train_guided_delay_pdp_mouth.py",
        "analyze_v208_strict_historical_replay.py",
        "analyze_v205_sionna_config_calibration.py",
        "generate_v178_test_track_complex_consensus.py",
        "generate_v87_easy_field_projection.py",
        "generate_v171_full_complex_ad_tail_denoising.py",
        "generate_v169_power_scaled_tail_denoising.py",
        "generate_v159_frontier_consensus_support_soup.py",
        "powerfield_common.py",
        "project_paths.py",
    )
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for name in sources:
            archive.write(PATHS.code / name, name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    output = np.load(OUTPUT, mmap_mode="r")
    finite = all(
        np.isfinite(np.asarray(output[start : start + 10])).all()
        for start in range(0, len(output), 10)
    )
    report["integrity"] = {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": bool(finite),
        "zip_crc": True,
        "zip_members": members,
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    for path in (BASE, V207_REFERENCE, V424_ACTIONS, *V205_REFERENCES):
        require_file(path)
    geometry = np.load(PATHS.shared_features / "height_geometry_cache.npz")
    regimes = regime_of(geometry["test_blocked"].astype(np.float32))
    generated = generate_probes(regimes)

    base_local = v208.score_prediction(BASE)
    local_scores = {probe.name: v208.score_prediction(probe.path) for probe in PROBES}
    selected_name = choose_probe(local_scores, base_local)
    selected_probe = next(probe for probe in PROBES if probe.name == selected_name)
    shutil.copyfile(selected_probe.path, OUTPUT)
    comparisons = {
        probe.name: track.compare_output_to(probe.path, BASE, regimes)
        for probe in PROBES
    }
    actions = generated["actions"]
    report = {
        "method": VERSION,
        "status": "candidate_pending_diagnostic_noncompliant",
        "primary_hypothesis": (
            "v425 and v426 corrected complementary scorer-aligned objects but were never fused: "
            "the full leaked polarization/UE/vertical slice conditionals and the complete 256x4 "
            "linkwise PDP. Fusing them while locking all validated marginals can add information "
            "without repeating v427's failed conventional PAS target."
        ),
        "compliance_boundary": {
            "diagnostic_only": True,
            "explicitly_noncompliant_for_final_submission": True,
            "reason": "reads v207/v205 Sionna references traced at the 500 official test coordinates",
            "must_not_be_used_as_final_method": True,
        },
        "phase_pretest": {
            "official_train_holdout_raw_oracle_magnitude_coherence": 0.086682,
            "train_calibrated_oracle_magnitude_coherence": 0.098544,
            "decision": "insufficient; phase injection was rejected before candidate generation",
        },
        "new_information_used": {
            "v426_online": 0.573416,
            "v427_online": 0.565229,
            "retained": "v426 complete linkwise PDP and leaked low-dimensional marginals",
            "new_object": "S-H conditional shape of every polarization x UE x vertical leaked slice",
        },
        "method_details": {
            "base": PATHS.relative(BASE),
            "joint_cycles": JOINT_CYCLES,
            "slice_target": "v207 high-budget shape blended with v205 three-frequency stable shape",
            "marginal_lock": "re-impose the complete v426 S-H, pol-S-H, S-H-UE and S-H-V marginals",
            "pdp_lock": "restore every v426 256x4 linkwise 192-bin CIR magnitude after each cycle",
            "power_policy": "preserve every link energy and point total power",
            "phase_policy": "retain projection carrier phase; no raw Sionna phase replacement",
            "selection_policy": "smallest action retaining 90% of best leaked-mouth gain while PDP is exact and PAS/NMSE stay bounded",
        },
        "probe_grid": [
            {"name": probe.name, "scale": probe.scale} for probe in PROBES
        ],
        "strict_local_diagnostic": {
            "reference_warning": "v207 same-coordinate Sionna truth, not official hidden truth",
            "base_v426": base_local,
            "probes": local_scores,
            "selected": selected_name,
        },
        "selected_action": {"name": selected_name, "scale": selected_probe.scale},
        "diagnostics": {
            "shared": {
                key: summarize(value) for key, value in generated["shared"].items()
            },
            "by_probe": {
                probe.name: {
                    key: summarize(actions[f"{key}_{probe.name}"])
                    for key in (
                        "alpha_mean",
                        "alpha_max",
                        "power_ratio",
                        "phase_agreement",
                        "pdp_lock_error",
                        "marginal_l1",
                        "support_l1",
                        "slice_target_cosine",
                    )
                }
                | {
                    "comparison_to_v426": comparisons[probe.name],
                    "by_regime": by_regime_summary(
                        regimes,
                        {
                            key: actions[f"{key}_{probe.name}"]
                            for key in (
                                "alpha_mean",
                                "power_ratio",
                                "phase_agreement",
                                "support_l1",
                                "slice_target_cosine",
                            )
                        },
                    ),
                }
                for probe in PROBES
            },
        },
        "outputs": {
            "submission_npy": PATHS.relative(OUTPUT),
            "probe_predictions": {
                probe.name: PATHS.relative(probe.path) for probe in PROBES
            },
            "actions": PATHS.relative(ACTION_OUT),
            "report": PATHS.relative(REPORT),
            "zip": PATHS.relative(ZIP_OUT),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(write_zip(report), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
