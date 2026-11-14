#!/usr/bin/env python3
"""Alternate exact external-PAS and v549-PDP magnitude constraints."""

from __future__ import annotations

import hashlib
import json
import zipfile

import numpy as np

import analyze_v205_sionna_config_calibration as calibration
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v424_sionna_pas_full_pdp_lock_diagnostic as v424
import generate_v558_external_pas_v549_pdp_fusion as v558
from project_paths import PATHS


VERSION_NUMBER = 559
VERSION = "v559_alternating_external_pas_v549_pdp_phase_retrieval"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUTPUT = CACHE_DIR / "v559_Round1_Test_Channel.npy"
ACTIONS = CACHE_DIR / "v559_alternating_projection_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

BASE = v558.BASE
REFERENCE = v558.REFERENCE
EXTERNAL_ARCHIVE = v558.EXTERNAL_ARCHIVE
NEURAL_CHECKPOINT = v558.NEURAL_CHECKPOINT
BATCH_SIZE = 2
FLOOR = 1e-20
TINY = 1e-30
CHECKPOINTS = (1, 2, 4, 8, 16)
BASE_CHECKPOINTS = (1, 4, 8)
METRICS = v558.METRICS


def require(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def digest(path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def external_pas_shape(external_h: np.ndarray) -> np.ndarray:
    power = v423.leak.leaked_cell_power(v423.leak.leaked_forward(external_h))
    total = power.sum(axis=(2, 3), keepdims=True, dtype=np.float64)
    return (power / np.maximum(total, TINY)).astype(np.float32)


def project_exact_pas(
    candidate: np.ndarray, external_shape: np.ndarray, external_h: np.ndarray
) -> np.ndarray:
    spec = v423.leak.leaked_forward(candidate)
    power = v423.leak.leaked_cell_power(spec)
    slice_energy = power.sum(axis=(2, 3), keepdims=True, dtype=np.float64)
    target_abs = np.sqrt(np.maximum(external_shape, FLOOR) * slice_energy).astype(
        np.float32
    )
    current_abs = np.abs(spec).astype(np.float32)
    external_spec = v423.leak.leaked_forward(external_h)
    external_abs = np.abs(external_spec).astype(np.float32)
    phase = spec / np.maximum(current_abs, FLOOR)
    external_phase = external_spec / np.maximum(external_abs, FLOOR)
    phase = np.where(current_abs > FLOOR, phase, external_phase)
    return v423.leak.leaked_inverse(target_abs * phase).astype(np.complex64)


def initial_candidate(
    start: str, base_h: np.ndarray, external_h: np.ndarray
) -> np.ndarray:
    if start == "base":
        return base_h.copy()
    if start == "external_phase":
        return v424.restore_full_linkwise_pdp(external_h, base_h)
    raise ValueError(start)


def iterate(
    start: str,
    cycles: int,
    base_h: np.ndarray,
    external_h: np.ndarray,
    external_shape: np.ndarray,
) -> np.ndarray:
    candidate = initial_candidate(start, base_h, external_h)
    for _ in range(cycles):
        candidate = project_exact_pas(candidate, external_shape, external_h)
        candidate = v424.restore_full_linkwise_pdp(candidate, base_h)
    return candidate


def variant_specs() -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (f"external_phase_c{cycles:02d}", "external_phase", cycles)
        for cycles in CHECKPOINTS
    ) + tuple((f"base_c{cycles:02d}", "base", cycles) for cycles in BASE_CHECKPOINTS)


def empty_metrics(rows: int) -> dict[str, np.ndarray]:
    return {name: np.empty(rows, dtype=np.float32) for name in METRICS}


def analyze() -> tuple[str, dict[str, object], dict[str, np.ndarray]]:
    base = np.load(BASE, mmap_mode="r")
    reference = np.load(REFERENCE, mmap_mode="r")
    specs = variant_specs()
    names = ("v549", "external", "v558", *(item[0] for item in specs))
    metrics = {name: empty_metrics(len(base)) for name in names}
    errors = {name: 0.0 for name in names}
    structure = {
        name: {
            "pdp_lock_max": 0.0,
            "pas_target_l1": np.empty(len(base), dtype=np.float32),
            "phase_agreement": np.empty(len(base), dtype=np.float32),
            "nmse_vs_v549_error": 0.0,
        }
        for name, _, _ in specs
    }
    truth_energy = 0.0
    base_energy = 0.0

    for start, end, external_h in v558.external_batches(BATCH_SIZE):
        base_h = np.asarray(base[start:end], dtype=np.complex64)
        truth_h = np.asarray(reference[start:end], dtype=np.complex64)
        target_shape = external_pas_shape(external_h)
        candidates = {
            "v549": base_h,
            "external": external_h,
            "v558": initial_candidate("external_phase", base_h, external_h),
        }

        for start_name in ("external_phase", "base"):
            checkpoints = (
                CHECKPOINTS if start_name == "external_phase" else BASE_CHECKPOINTS
            )
            candidate = initial_candidate(start_name, base_h, external_h)
            for cycle in range(1, max(checkpoints) + 1):
                candidate = project_exact_pas(candidate, target_shape, external_h)
                candidate = v424.restore_full_linkwise_pdp(candidate, base_h)
                if cycle not in checkpoints:
                    continue
                name = (
                    f"external_phase_c{cycle:02d}"
                    if start_name == "external_phase"
                    else f"base_c{cycle:02d}"
                )
                candidates[name] = candidate.copy()
                actual_shape = external_pas_shape(candidate)
                structure[name]["pas_target_l1"][start:end] = np.sum(
                    np.abs(actual_shape - target_shape),
                    axis=(1, 2, 3, 4, 5),
                    dtype=np.float64,
                ).astype(np.float32)
                structure[name]["pdp_lock_max"] = max(
                    structure[name]["pdp_lock_max"],
                    float(np.max(v424.full_linkwise_pdp_error(candidate, base_h))),
                )
                structure[name]["phase_agreement"][start:end] = v423.phase_agreement(
                    base_h, candidate
                )
                structure[name]["nmse_vs_v549_error"] += float(
                    np.sum(np.abs(candidate - base_h) ** 2, dtype=np.float64)
                )

        truth_views = calibration.spectral_views(truth_h)
        truth_energy += float(np.sum(np.abs(truth_h) ** 2, dtype=np.float64))
        base_energy += float(np.sum(np.abs(base_h) ** 2, dtype=np.float64))
        for name, candidate in candidates.items():
            score = calibration.spectral_scores(candidate, truth_views)
            for metric in METRICS:
                metrics[name][metric][start:end] = score[metric]
            errors[name] += float(
                np.sum(np.abs(candidate - truth_h) ** 2, dtype=np.float64)
            )
        if end % 20 == 0 or end == len(base):
            print(f"v559 alternating audit {end}/{len(base)}", flush=True)

    scores = {
        name: v558.finalize_score(metrics[name], errors[name], truth_energy)
        for name in names
    }
    eligible = [
        name
        for name, _, _ in specs
        if scores[name]["leaked_sh"] > scores["v558"]["leaked_sh"]
        and scores[name]["pdp"] >= scores["v549"]["pdp"] - 2e-6
        and structure[name]["pdp_lock_max"] <= 2e-6
    ]
    if not eligible:
        raise RuntimeError("no alternating projection improved v558")
    selected = max(eligible, key=lambda name: scores[name]["score_traceback_leaked"])
    action_arrays: dict[str, np.ndarray] = {}
    structural_report = {}
    for name, _, _ in specs:
        target_l1 = structure[name]["pas_target_l1"]
        phase = structure[name]["phase_agreement"]
        action_arrays[f"{name}_pas_target_l1"] = target_l1
        action_arrays[f"{name}_phase_agreement"] = phase
        structural_report[name] = {
            "pdp_lock_max": float(structure[name]["pdp_lock_max"]),
            "pas_target_l1_mean": float(np.mean(target_l1)),
            "phase_agreement_mean": float(np.mean(phase)),
            "nmse_vs_v549": float(
                structure[name]["nmse_vs_v549_error"] / max(base_energy, TINY)
            ),
        }
    return selected, {"scores": scores, "structural": structural_report}, action_arrays


def generate_selected(selected: str) -> None:
    spec = {name: (start, cycles) for name, start, cycles in variant_specs()}
    start_name, cycles = spec[selected]
    base = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    for start, end, external_h in v558.external_batches(BATCH_SIZE):
        base_h = np.asarray(base[start:end], dtype=np.complex64)
        target_shape = external_pas_shape(external_h)
        output[start:end] = iterate(
            start_name, cycles, base_h, external_h, target_shape
        )
        if end % 20 == 0 or end == len(base):
            output.flush()
            print(f"v559 selected render {end}/{len(base)}", flush=True)
    output.flush()


def write_package(report: dict[str, object]) -> dict[str, object]:
    dependencies = (
        "generate_v559_alternating_pas_pdp_phase_retrieval.py",
        "generate_v558_external_pas_v549_pdp_fusion.py",
        "generate_v424_sionna_pas_full_pdp_lock_diagnostic.py",
        "generate_v423_joint_sionna_pas_pdp_diagnostic.py",
        "generate_v187_leaked_pas_lifecycle_projection.py",
        "generate_v189_leaked_multimarginal_overdrive.py",
        "generate_v171_full_complex_ad_tail_denoising.py",
        "powerfield_common.py",
        "project_paths.py",
    )
    for name in dependencies:
        require(PATHS.code / name)
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        archive.writestr("model/", b"")
        archive.write(NEURAL_CHECKPOINT, "model/neural_sionna_calibrator.pt")
        archive.write(REPORT, "model/v559_phase_retrieval_report.json")
        for name in dependencies:
            archive.write(PATHS.code / name, name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    channel = np.load(OUTPUT, mmap_mode="r")
    finite = all(
        np.isfinite(np.asarray(channel[start : start + 5])).all()
        for start in range(0, len(channel), 5)
    )
    integrity = {
        "shape": list(channel.shape),
        "dtype": str(channel.dtype),
        "finite": bool(finite),
        "zip_crc": True,
        "zip_members": members,
        "zip_bytes": int(ZIP_OUT.stat().st_size),
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }
    report["integrity"] = integrity
    return integrity


def main() -> None:
    for path in (BASE, REFERENCE, EXTERNAL_ARCHIVE, NEURAL_CHECKPOINT):
        require(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    selected, audit, action_arrays = analyze()
    generate_selected(selected)
    np.savez_compressed(ACTIONS, selected=np.asarray(selected), **action_arrays)
    report: dict[str, object] = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v558 proves the external phase and v549 CIR magnitude are complementary. "
            "Alternating exact scorer-shaped PAS-slice magnitude projection with exact "
            "v549 linkwise-CIR magnitude projection can satisfy both constraints more "
            "closely than a one-pass phase transplant."
        ),
        "new_information": [
            "external per-(polarization, UE, vertical) S-H PAS magnitude shape",
            "v549 per-link delay-domain magnitude",
            "phase-retrieval consistency between the two measured magnitude domains",
        ],
        "parameters": {
            "batch_size": BATCH_SIZE,
            "external_phase_checkpoints": list(CHECKPOINTS),
            "base_checkpoints": list(BASE_CHECKPOINTS),
            "pas_projection": (
                "exact per-(P,UE,V) S-H magnitude with current slice energy"
            ),
            "pdp_projection": "exact v549 CIR magnitude after every PAS projection",
        },
        "selected": selected,
        "diagnostic_warning": (
            "The v207 traceback score is a same-coordinate Sionna diagnostic, not the "
            "hidden online score. Online feedback remains the final evidence."
        ),
        "audit": audit,
        "outputs": {
            "channel": PATHS.relative(OUTPUT),
            "actions": PATHS.relative(ACTIONS),
            "report": PATHS.relative(REPORT),
            "archive": PATHS.relative(ZIP_OUT),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    integrity = write_package(report)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected": selected,
                "archive": report["outputs"]["archive"],
                "v549": audit["scores"]["v549"],
                "v558": audit["scores"]["v558"],
                "candidate": audit["scores"][selected],
                "selected_structure": audit["structural"][selected],
                "sha256": integrity["sha256"],
                "md5": integrity["md5"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
