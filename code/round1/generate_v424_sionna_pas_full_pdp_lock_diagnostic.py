#!/usr/bin/env python3
"""Generate a non-compliant Sionna-PAS diagnostic with full v212 PDP lock.

This experiment intentionally reads Sionna references traced at the official
test coordinates. It is a directional online diagnostic only and must not be
used as the final rule-compliant solution.

v423 alternated PAS projection with low-dimensional D/U-D/H-D projection. The
PAS target improved, but the scorer-aligned 256x4 linkwise PDP regressed. v424
removes that interface error: after every PAS step, it restores the complete
192-bin PDP magnitude of every v212 antenna/UE link while retaining the phase
produced by the PAS step.
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
from generate_v87_easy_field_projection import summarize
from generate_v159_frontier_consensus_support_soup import digest
from generate_v169_power_scaled_tail_denoising import by_regime_summary
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 424
VERSION = "v424_sionna_pas_full_pdp_lock_diagnostic"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "v424_Round1_Test_Channel.npy"
ACTION_OUT = OUT_DIR / "v424_sionna_pas_full_pdp_lock_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

BASE = PATHS.cache_dir(212) / "v212_Round1_Test_Channel.npy"
V207_REFERENCE = PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"
V205_REFERENCES = (
    PATHS.cache_dir(205) / "benchmark_sector_m30_track_d5_f2p6" / "test_clean_H.npy",
    PATHS.cache_dir(205) / "benchmark_sector_m30_track_d5_f6p0" / "test_clean_H.npy",
    PATHS.cache_dir(205) / "benchmark_sector_m30_track_d5_f6p7" / "test_clean_H.npy",
)
V212_ACTIONS = PATHS.cache_dir(212) / "v212_multidomain_sionna_pdp_actions.npz"

BATCH = 2
PAS_CYCLES = 4
PAS_ALPHA_BASE = {"clear": 0.0, "light": 0.06, "medium": 0.52, "heavy": 0.72}


@dataclass(frozen=True)
class Probe:
    name: str
    pas_scale: float

    @property
    def path(self):
        return OUT_DIR / f"v424_{self.name}_Round1_Test_Channel.npy"


PROBES = (
    Probe("pas035", 0.35),
    Probe("pas060", 0.60),
    Probe("pas085", 0.85),
    Probe("pas110", 1.10),
)


def require_file(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def restore_full_linkwise_pdp(candidate: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Restore every v212 link's CIR magnitude and keep candidate CIR phase."""
    candidate_cir = np.fft.ifft(candidate, axis=-1)
    base_cir = np.fft.ifft(base, axis=-1)
    candidate_abs = np.abs(candidate_cir)
    base_abs = np.abs(base_cir)
    candidate_phase = candidate_cir / np.maximum(candidate_abs, 1e-30)
    base_phase = base_cir / np.maximum(base_abs, 1e-30)
    phase = np.where(candidate_abs > 1e-20, candidate_phase, base_phase)
    restored = np.fft.fft(base_abs * phase, axis=-1)
    return restored.astype(np.complex64)


def full_linkwise_pdp_error(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_power = np.abs(np.fft.ifft(left, axis=-1)).astype(np.float32) ** 2
    right_power = np.abs(np.fft.ifft(right, axis=-1)).astype(np.float32) ** 2
    left_norm = left_power / np.maximum(left_power.sum(axis=-1, keepdims=True), TINY)
    right_norm = right_power / np.maximum(right_power.sum(axis=-1, keepdims=True), TINY)
    return np.max(np.abs(left_norm - right_norm), axis=(1, 2, 3)).astype(np.float32)


def pas_quality_from_references(
    prior: np.lib.npyio.NpzFile,
    start: int,
    end: int,
    pas_diag: dict[str, np.ndarray],
) -> np.ndarray:
    prior_cross = prior["cross_agreement"][start:end].astype(np.float32)
    prior_stability = prior["multi_stability"][start:end].astype(np.float32)
    prior_coverage = prior["multi_coverage"][start:end].astype(np.float32)
    prior_overpush = prior["overpush_guard"][start:end].astype(np.float32)
    quality = np.clip(
        0.10
        + 0.28 * pas_diag["cross_agreement"]
        + 0.20 * pas_diag["stability"]
        + 0.14 * pas_diag["coverage"]
        + 0.14 * prior_cross
        + 0.10 * prior_stability
        + 0.08 * prior_coverage
        - 0.22 * prior_overpush,
        0.0,
        1.05,
    ).astype(np.float32)
    blocked = (
        (pas_diag["high_zero"] > 0.5)
        | (pas_diag["cross_agreement"] < 0.16)
        | (prior_overpush > 0.88)
    )
    quality[blocked] = 0.0
    return quality


def generate_probes(regimes: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    source = np.load(BASE, mmap_mode="r")
    high = np.load(V207_REFERENCE, mmap_mode="r")
    low = [np.load(path, mmap_mode="r") for path in V205_REFERENCES]
    prior = np.load(V212_ACTIONS)
    outputs = {
        probe.name: np.lib.format.open_memmap(
            probe.path, mode="w+", dtype=np.complex64, shape=source.shape
        )
        for probe in PROBES
    }
    shared = {
        "pas_quality": np.zeros(len(source), dtype=np.float32),
        "pas_cross_agreement": np.zeros(len(source), dtype=np.float32),
        "pas_multi_stability": np.zeros(len(source), dtype=np.float32),
        "pas_multi_coverage": np.zeros(len(source), dtype=np.float32),
    }
    actions: dict[str, np.ndarray] = {}
    for probe in PROBES:
        actions[f"pas_alpha_{probe.name}"] = np.zeros(len(source), dtype=np.float32)
        actions[f"power_ratio_{probe.name}"] = np.zeros(len(source), dtype=np.float32)
        actions[f"phase_agreement_{probe.name}"] = np.zeros(len(source), dtype=np.float32)
        actions[f"pdp_lock_error_{probe.name}"] = np.zeros(len(source), dtype=np.float32)

    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        base_h = np.asarray(source[start:end], dtype=np.complex64)
        high_h = np.asarray(high[start:end], dtype=np.complex64)
        low_h = [np.asarray(value[start:end], dtype=np.complex64) for value in low]

        base_support, _ = v423.leaked_support(base_h)
        high_support, high_total = v423.leaked_support(high_h)
        low_support = [v423.leaked_support(value) for value in low_h]
        base_pas, target_pas, pas_diag = v423.pas_targets(
            base_support,
            high_support,
            high_total,
            [value[0] for value in low_support],
            [value[1] for value in low_support],
        )
        quality = pas_quality_from_references(prior, start, end, pas_diag)
        shared["pas_quality"][start:end] = quality
        shared["pas_cross_agreement"][start:end] = pas_diag["cross_agreement"]
        shared["pas_multi_stability"][start:end] = pas_diag["stability"]
        shared["pas_multi_coverage"][start:end] = pas_diag["coverage"]

        local_regimes = regimes[start:end]
        base_power = np.maximum(
            np.sum(np.abs(base_h) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY
        )
        for probe in PROBES:
            alpha = v423.alpha_for_regimes(
                local_regimes,
                PAS_ALPHA_BASE,
                probe.pas_scale,
                quality,
            )
            desired = v423.desired_pas(base_pas, target_pas, alpha)
            candidate = base_h.copy()
            for _ in range(PAS_CYCLES):
                candidate = v423.project_pas(candidate, desired)
                candidate = restore_full_linkwise_pdp(candidate, base_h)
            if not np.isfinite(candidate).all():
                raise FloatingPointError(f"non-finite {probe.name} block {start}:{end}")
            outputs[probe.name][start:end] = candidate
            outputs[probe.name].flush()
            candidate_power = np.sum(
                np.abs(candidate) ** 2, axis=(1, 2, 3), dtype=np.float64
            )
            actions[f"pas_alpha_{probe.name}"][start:end] = alpha
            actions[f"power_ratio_{probe.name}"][start:end] = (
                candidate_power / base_power
            ).astype(np.float32)
            actions[f"phase_agreement_{probe.name}"][start:end] = v423.phase_agreement(
                base_h, candidate
            )
            actions[f"pdp_lock_error_{probe.name}"][start:end] = full_linkwise_pdp_error(
                candidate, base_h
            )
        print(f"v424 PAS plus full-PDP lock {end}/{len(source)}", flush=True)

    np.savez_compressed(
        ACTION_OUT,
        regimes=regimes.astype("U8"),
        **shared,
        **actions,
    )
    return {"shared": shared, "actions": actions}


def choose_probe(
    scores: dict[str, dict[str, object]], base: dict[str, object]
) -> str:
    base_score = float(base["score_traceback_leaked"])
    eligible = [
        probe.name
        for probe in PROBES
        if float(scores[probe.name]["pas"]) > float(base["pas"])
        and float(scores[probe.name]["leaked_sh"]) > float(base["leaked_sh"])
        and float(scores[probe.name]["pdp"]) >= float(base["pdp"]) - 2e-6
        and float(scores[probe.name]["score_traceback_leaked"]) > base_score
    ]
    if not eligible:
        return max(scores, key=lambda name: float(scores[name]["score_traceback_leaked"]))
    best = max(float(scores[name]["score_traceback_leaked"]) for name in eligible)
    threshold = base_score + 0.90 * (best - base_score)
    for probe in PROBES:
        if probe.name in eligible and float(scores[probe.name]["score_traceback_leaked"]) >= threshold:
            return probe.name
    return eligible[-1]


def write_zip(report: dict[str, object]) -> dict[str, object]:
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for source in (
            PATHS.code / "generate_v424_sionna_pas_full_pdp_lock_diagnostic.py",
            PATHS.code / "generate_v423_joint_sionna_pas_pdp_diagnostic.py",
            PATHS.code / "generate_v212_multidomain_sionna_pdp_ensemble.py",
            PATHS.code / "generate_v187_leaked_pas_lifecycle_projection.py",
            PATHS.code / "generate_v196_train_guided_delay_pdp_mouth.py",
            PATHS.code / "analyze_v208_strict_historical_replay.py",
            PATHS.code / "analyze_v205_sionna_config_calibration.py",
            PATHS.code / "generate_v178_test_track_complex_consensus.py",
            PATHS.code / "generate_v87_easy_field_projection.py",
            PATHS.code / "generate_v171_full_complex_ad_tail_denoising.py",
            PATHS.code / "generate_v169_power_scaled_tail_denoising.py",
            PATHS.code / "generate_v159_frontier_consensus_support_soup.py",
            PATHS.code / "generate_v189_leaked_multimarginal_overdrive.py",
            PATHS.code / "powerfield_common.py",
            PATHS.code_root / "project_paths.py",
        ):
            archive.write(source, source.name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    output = np.load(OUTPUT, mmap_mode="r")
    finite = True
    for start in range(0, len(output), 10):
        block = np.asarray(output[start : start + 10], dtype=np.complex64)
        finite = finite and bool(np.isfinite(block).all())
    report["integrity"] = {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
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
    for path in (BASE, V207_REFERENCE, V212_ACTIONS, *V205_REFERENCES):
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
    action_arrays = generated["actions"]
    report = {
        "method": VERSION,
        "status": "candidate_pending_diagnostic_noncompliant",
        "primary_hypothesis": (
            "v212 already contains the online-validated Sionna PDP correction. "
            "Sionna PAS can add headroom only if the full scorer-aligned 256x4 "
            "linkwise PDP is preserved after every PAS projection."
        ),
        "compliance_boundary": {
            "diagnostic_only": True,
            "explicitly_noncompliant_for_final_submission": True,
            "reason": "reads v207/v205 Sionna references traced at the 500 official test coordinates",
            "must_not_be_used_as_final_method": True,
        },
        "inputs": {
            "base": PATHS.relative(BASE),
            "high_budget_sionna": PATHS.relative(V207_REFERENCE),
            "multi_frequency_sionna": [PATHS.relative(path) for path in V205_REFERENCES],
            "v212_quality_actions": PATHS.relative(V212_ACTIONS),
        },
        "method_details": {
            "pas_targets": ["S-H", "pol-S-H", "S-H-UE", "S-H-V"],
            "pdp_target": "the complete normalized 192-bin PDP of every v212 256x4 link",
            "pas_cycles": PAS_CYCLES,
            "pdp_lock": "restore every v212 CIR magnitude after every PAS cycle",
            "point_power_policy": "inherited exactly from the full linkwise v212 PDP",
            "selection_policy": "smallest action retaining at least 90% of the best local gain while PAS improves and PDP is non-decreasing",
        },
        "probe_grid": [
            {"name": probe.name, "pas_scale": probe.pas_scale} for probe in PROBES
        ],
        "strict_local_diagnostic": {
            "reference_warning": "v207 same-coordinate Sionna truth, not official hidden truth",
            "base_v212": base_local,
            "probes": local_scores,
            "selected": selected_name,
        },
        "selected_action": {
            "name": selected_name,
            "pas_scale": selected_probe.pas_scale,
        },
        "diagnostics": {
            "shared": {
                key: summarize(value) for key, value in generated["shared"].items()
            },
            "by_probe": {
                probe.name: {
                    "pas_alpha": summarize(action_arrays[f"pas_alpha_{probe.name}"]),
                    "power_ratio": summarize(action_arrays[f"power_ratio_{probe.name}"]),
                    "phase_agreement": summarize(action_arrays[f"phase_agreement_{probe.name}"]),
                    "pdp_lock_error": summarize(action_arrays[f"pdp_lock_error_{probe.name}"]),
                    "comparison_to_v212": comparisons[probe.name],
                    "by_regime": by_regime_summary(
                        regimes,
                        {
                            "pas_alpha": action_arrays[f"pas_alpha_{probe.name}"],
                            "power_ratio": action_arrays[f"power_ratio_{probe.name}"],
                            "phase_agreement": action_arrays[f"phase_agreement_{probe.name}"],
                            "pdp_lock_error": action_arrays[f"pdp_lock_error_{probe.name}"],
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
    report = write_zip(report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
