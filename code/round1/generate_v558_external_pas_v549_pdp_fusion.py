#!/usr/bin/env python3
"""Fuse the external six-seed PAS with the v549 PDP/carrier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

import analyze_v205_sionna_config_calibration as calibration
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v424_sionna_pas_full_pdp_lock_diagnostic as v424
import generate_v425_sionna_full_slice_pas_completion_diagnostic as v425
from project_paths import PATHS


VERSION_NUMBER = 558
VERSION = "v558_external_high_order_pas_v549_pdp_fusion"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUTPUT = CACHE_DIR / "v558_Round1_Test_Channel.npy"
ACTIONS = CACHE_DIR / "v558_external_pas_fusion_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

BASE = PATHS.cache_dir(549) / "v549_Round1_Test_Channel.npy"
REFERENCE = PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"
EXTERNAL_ARCHIVE = (
    PATHS.root
    / "v395_six_seed_high_order_pas_projection_action075_complement050_candidate_pending.zip"
)
EXTERNAL_MEMBER = "Round1_Test_Channel.npy"
NEURAL_CHECKPOINT = (
    PATHS.checkpoints / "v469_neural_sionna_calibrator" / "neural_sionna_calibrator.pt"
)

BATCH_SIZE = 2
PAS_CYCLES = 4
FLOOR = 1e-12
TINY = 1e-30
METRICS = ("pas", "pdp", "leaked_sh", "ad_support", "complex_coherence")
VARIANTS = (
    ("full_log_050", "full_log", 0.50),
    ("full_log_075", "full_log", 0.75),
    ("full_log_100", "full_log", 1.00),
    ("full_log_125", "full_log", 1.25),
    ("full_log_150", "full_log", 1.50),
    ("full_linear_100", "full_linear", 1.00),
    ("marginal_100", "marginal", 1.00),
    ("complex_050", "complex", 0.50),
    ("complex_075", "complex", 0.75),
    ("complex_100", "complex", 1.00),
)


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def normalize_support(value: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(value, dtype=np.float32), FLOOR)
    output /= np.maximum(
        output.sum(axis=(1, 2, 3, 4, 5), keepdims=True, dtype=np.float64), TINY
    )
    return output.astype(np.float32)


def external_batches(batch_size: int):
    with zipfile.ZipFile(EXTERNAL_ARCHIVE) as archive:
        with archive.open(EXTERNAL_MEMBER) as handle:
            version = np.lib.format.read_magic(handle)
            shape, fortran_order, dtype = np.lib.format._read_array_header(
                handle, version
            )
            if tuple(shape) != (500, 256, 4, 192):
                raise ValueError(f"unexpected external shape: {shape}")
            if fortran_order or dtype != np.dtype(np.complex64):
                raise ValueError(
                    f"unexpected external NPY header: {fortran_order=} {dtype=}"
                )
            row_values = int(np.prod(shape[1:]))
            for start in range(0, shape[0], batch_size):
                end = min(start + batch_size, shape[0])
                count = (end - start) * row_values
                raw = handle.read(count * dtype.itemsize)
                if len(raw) != count * dtype.itemsize:
                    raise EOFError(f"external NPY ended at row {start}")
                yield start, end, np.frombuffer(raw, dtype=dtype, count=count).reshape(
                    end - start, *shape[1:]
                )
            if handle.read(1):
                raise ValueError("external NPY has trailing payload")


def full_target(
    base_support: np.ndarray,
    external_support: np.ndarray,
    alpha: float,
    logarithmic: bool,
) -> np.ndarray:
    if not logarithmic:
        return normalize_support(
            (1.0 - np.float32(alpha)) * base_support
            + np.float32(alpha) * external_support
        )
    log_ratio = np.clip(
        np.log(np.maximum(external_support, FLOOR))
        - np.log(np.maximum(base_support, FLOOR)),
        -6.0,
        6.0,
    )
    return normalize_support(
        base_support * np.exp(np.float32(alpha) * log_ratio).astype(np.float32)
    )


def render_variant(
    base_h: np.ndarray,
    external_h: np.ndarray,
    base_support: np.ndarray,
    external_support: np.ndarray,
    kind: str,
    alpha: float,
) -> np.ndarray:
    if kind == "complex":
        mixed = (
            (1.0 - np.float32(alpha)) * base_h + np.float32(alpha) * external_h
        ).astype(np.complex64)
        return v424.restore_full_linkwise_pdp(mixed, base_h)

    candidate = base_h.copy()
    if kind in {"full_log", "full_linear"}:
        desired = full_target(
            base_support,
            external_support,
            alpha,
            logarithmic=kind == "full_log",
        )
        for _ in range(PAS_CYCLES):
            candidate = v425.project_full_support(candidate, desired)
            candidate = v424.restore_full_linkwise_pdp(candidate, base_h)
        return candidate

    if kind == "marginal":
        base_marginal = v423.leaked_marginals(base_support)
        external_marginal = v423.leaked_marginals(external_support)
        desired = v423.desired_pas(
            base_marginal,
            external_marginal,
            np.full(len(base_h), alpha, dtype=np.float32),
        )
        for _ in range(PAS_CYCLES):
            candidate = v423.project_pas(candidate, desired)
            candidate = v424.restore_full_linkwise_pdp(candidate, base_h)
        return candidate
    raise ValueError(kind)


def empty_metrics(rows: int) -> dict[str, np.ndarray]:
    return {name: np.empty(rows, dtype=np.float32) for name in METRICS}


def finalize_score(
    metrics: dict[str, np.ndarray], error: float, truth_energy: float
) -> dict[str, float]:
    means = {name: float(np.mean(value)) for name, value in metrics.items()}
    nmse = float(error / max(truth_energy, TINY))
    nmse_score = float(1.0 / (1.0 + nmse))
    return {
        **means,
        "global_nmse": nmse,
        "nmse_score": nmse_score,
        "score_conventional": float(
            0.4 * means["pas"] + 0.4 * means["pdp"] + 0.2 * nmse_score
        ),
        "score_traceback_leaked": float(
            0.4 * means["leaked_sh"] + 0.4 * means["pdp"] + 0.2 * nmse_score
        ),
    }


def analyze_variants() -> tuple[str, dict[str, object], dict[str, np.ndarray]]:
    base = np.load(BASE, mmap_mode="r")
    reference = np.load(REFERENCE, mmap_mode="r")
    names = ("v549", "external", *(item[0] for item in VARIANTS))
    metrics = {name: empty_metrics(len(base)) for name in names}
    errors = {name: 0.0 for name in names}
    structural = {
        name: {
            "pdp_lock_max": 0.0,
            "power_ratio": np.empty(len(base), dtype=np.float32),
            "phase_agreement": np.empty(len(base), dtype=np.float32),
            "nmse_vs_v549_error": 0.0,
        }
        for name, _, _ in VARIANTS
    }
    truth_energy = 0.0
    base_energy = 0.0

    for start, end, external_h in external_batches(BATCH_SIZE):
        base_h = np.asarray(base[start:end], dtype=np.complex64)
        truth_h = np.asarray(reference[start:end], dtype=np.complex64)
        if not np.isfinite(external_h).all():
            raise FloatingPointError(f"non-finite external block {start}:{end}")
        base_support, _ = v423.leaked_support(base_h)
        external_support, _ = v423.leaked_support(external_h)
        candidates = {"v549": base_h, "external": external_h}
        for name, kind, alpha in VARIANTS:
            candidate = render_variant(
                base_h,
                external_h,
                base_support,
                external_support,
                kind,
                alpha,
            )
            if not np.isfinite(candidate).all():
                raise FloatingPointError(f"non-finite {name} block {start}:{end}")
            candidates[name] = candidate
            structural[name]["pdp_lock_max"] = max(
                structural[name]["pdp_lock_max"],
                float(np.max(v424.full_linkwise_pdp_error(candidate, base_h))),
            )
            base_power = np.sum(np.abs(base_h) ** 2, axis=(1, 2, 3), dtype=np.float64)
            candidate_power = np.sum(
                np.abs(candidate) ** 2, axis=(1, 2, 3), dtype=np.float64
            )
            structural[name]["power_ratio"][start:end] = (
                candidate_power / np.maximum(base_power, TINY)
            ).astype(np.float32)
            structural[name]["phase_agreement"][start:end] = v423.phase_agreement(
                base_h, candidate
            )
            structural[name]["nmse_vs_v549_error"] += float(
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
            print(f"v558 fusion audit {end}/{len(base)}", flush=True)

    scores = {
        name: finalize_score(metrics[name], errors[name], truth_energy)
        for name in names
    }
    eligible = [
        name
        for name, _, _ in VARIANTS
        if scores[name]["leaked_sh"] > scores["v549"]["leaked_sh"]
        and scores[name]["pdp"] >= scores["v549"]["pdp"] - 2e-6
        and structural[name]["pdp_lock_max"] <= 2e-6
    ]
    if not eligible:
        raise RuntimeError("no v558 fusion variant passed the structural gate")
    selected = max(eligible, key=lambda name: scores[name]["score_traceback_leaked"])
    structural_report = {}
    action_arrays: dict[str, np.ndarray] = {}
    for name, _, _ in VARIANTS:
        power_ratio = structural[name]["power_ratio"]
        phase_agreement = structural[name]["phase_agreement"]
        structural_report[name] = {
            "pdp_lock_max": float(structural[name]["pdp_lock_max"]),
            "power_ratio_mean": float(np.mean(power_ratio)),
            "power_ratio_max_abs_error": float(np.max(np.abs(power_ratio - 1.0))),
            "phase_agreement_mean": float(np.mean(phase_agreement)),
            "nmse_vs_v549": float(
                structural[name]["nmse_vs_v549_error"] / max(base_energy, TINY)
            ),
        }
        action_arrays[f"{name}_power_ratio"] = power_ratio
        action_arrays[f"{name}_phase_agreement"] = phase_agreement
    return selected, {"scores": scores, "structural": structural_report}, action_arrays


def generate_selected(selected: str) -> None:
    specification = {name: (kind, alpha) for name, kind, alpha in VARIANTS}
    kind, alpha = specification[selected]
    base = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    for start, end, external_h in external_batches(BATCH_SIZE):
        base_h = np.asarray(base[start:end], dtype=np.complex64)
        base_support, _ = v423.leaked_support(base_h)
        external_support, _ = v423.leaked_support(external_h)
        output[start:end] = render_variant(
            base_h,
            external_h,
            base_support,
            external_support,
            kind,
            alpha,
        )
        if end % 20 == 0 or end == len(base):
            output.flush()
            print(f"v558 selected render {end}/{len(base)}", flush=True)
    output.flush()


def write_package(report: dict[str, object]) -> dict[str, object]:
    dependencies = (
        "generate_v558_external_pas_v549_pdp_fusion.py",
        "generate_v425_sionna_full_slice_pas_completion_diagnostic.py",
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
        archive.write(REPORT, "model/v558_fusion_report.json")
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
    with zipfile.ZipFile(EXTERNAL_ARCHIVE) as archive:
        bad = archive.testzip()
    if bad is not None:
        raise ValueError(f"external ZIP CRC failure: {bad}")

    selected, audit, action_arrays = analyze_variants()
    generate_selected(selected)
    np.savez_compressed(ACTIONS, selected=np.asarray(selected), **action_arrays)
    report: dict[str, object] = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "The external six-seed output has a stronger scorer-shaped high-order PAS, "
            "while v549 has the stronger online-validated linkwise PDP and carrier. "
            "Alternating high-order PAS projection with an exact v549 CIR-magnitude "
            "restore should combine the complementary information instead of averaging H."
        ),
        "new_information": [
            "external six-realization full conditional (P,S,H,UE,V) PAS state",
            "v549 online-validated 256x4x192 linkwise PDP and point power",
            "explicit comparison against marginal-only and complex-phase fusion controls",
        ],
        "inputs": {
            "carrier": PATHS.relative(BASE),
            "external_pas_teacher": PATHS.relative(EXTERNAL_ARCHIVE),
            "reference_diagnostic": PATHS.relative(REFERENCE),
        },
        "parameters": {
            "batch_size": BATCH_SIZE,
            "pas_cycles": PAS_CYCLES,
            "variants": [
                {"name": name, "kind": kind, "alpha": alpha}
                for name, kind, alpha in VARIANTS
            ],
            "selection": (
                "highest v207 traceback score among variants with improved leaked PAS "
                "and exact v549 linkwise-PDP lock"
            ),
        },
        "selected": selected,
        "diagnostic_warning": (
            "v207 is a same-coordinate Sionna reference, not hidden online truth. "
            "The external endpoint's approximately 0.743 online result and v549's "
            "0.740891 result are the independent online evidence for complementarity."
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
                "external": audit["scores"]["external"],
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
