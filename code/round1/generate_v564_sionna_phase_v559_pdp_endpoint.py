#!/usr/bin/env python3
"""Use the Sionna complex phase endpoint with the v559 linkwise PDP skeleton."""

from __future__ import annotations

import hashlib
import json
import zipfile

import numpy as np

import analyze_v208_strict_historical_replay as replay
import generate_v424_sionna_pas_full_pdp_lock_diagnostic as v424
from project_paths import PATHS


VERSION_NUMBER = 564
VERSION = "v564_sionna_phase_v559_pdp_endpoint"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUTPUT = CACHE_DIR / "v564_Round1_Test_Channel.npy"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

BASE = PATHS.cache_dir(559) / "v559_Round1_Test_Channel.npy"
PHASE_SOURCE = PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"
BATCH = 2


def require(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def digest(path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def point_power(channel: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(channel) ** 2, axis=(1, 2, 3), dtype=np.float64)


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
    }


def generate() -> dict[str, object]:
    base = np.load(BASE, mmap_mode="r")
    phase_source = np.load(PHASE_SOURCE, mmap_mode="r")
    if base.shape != phase_source.shape or base.dtype != np.dtype("complex64"):
        raise ValueError(f"shape/dtype mismatch: {base.shape=} {base.dtype=}")
    if phase_source.dtype != np.dtype("complex64"):
        raise ValueError(f"unexpected phase source dtype: {phase_source.dtype}")

    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    power_ratio = np.empty(len(base), dtype=np.float32)
    phase_agreement = np.empty(len(base), dtype=np.float32)
    pdp_error = np.empty(len(base), dtype=np.float32)
    finite = True
    max_abs = 0.0
    for start in range(0, len(base), BATCH):
        end = min(start + BATCH, len(base))
        base_h = np.asarray(base[start:end], dtype=np.complex64)
        source_h = np.asarray(phase_source[start:end], dtype=np.complex64)
        candidate = v424.restore_full_linkwise_pdp(source_h, base_h)
        finite = finite and bool(np.isfinite(candidate).all())
        max_abs = max(max_abs, float(np.max(np.abs(candidate))))
        output[start:end] = candidate
        power_ratio[start:end] = (
            point_power(candidate) / np.maximum(point_power(base_h), 1e-30)
        ).astype(np.float32)
        phase_agreement[start:end] = v424.v423.phase_agreement(base_h, candidate)
        pdp_error[start:end] = v424.full_linkwise_pdp_error(candidate, base_h)
        if end == len(base) or end % 50 == 0:
            output.flush()
            print(f"v564 Sionna-phase PDP endpoint {end}/{len(base)}", flush=True)
    output.flush()
    return {
        "power_ratio": summarize(power_ratio),
        "phase_agreement_to_v559": summarize(phase_agreement),
        "full_linkwise_pdp_error": summarize(pdp_error),
        "finite": bool(finite),
        "max_abs": max_abs,
    }


def write_package(report: dict[str, object]) -> None:
    dependencies = (
        "generate_v564_sionna_phase_v559_pdp_endpoint.py",
        "generate_v424_sionna_pas_full_pdp_lock_diagnostic.py",
        "generate_v423_joint_sionna_pas_pdp_diagnostic.py",
        "generate_v187_leaked_pas_lifecycle_projection.py",
        "generate_v171_full_complex_ad_tail_denoising.py",
        "analyze_v208_strict_historical_replay.py",
        "project_paths.py",
    )
    for name in dependencies:
        require(PATHS.code / name)
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        archive.writestr("model/", b"")
        archive.write(REPORT, "model/v564_sionna_phase_report.json")
        for name in dependencies:
            archive.write(PATHS.code / name, name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failure: {bad}")
    channel = np.load(OUTPUT, mmap_mode="r")
    finite = all(
        np.isfinite(np.asarray(channel[start : start + BATCH])).all()
        for start in range(0, len(channel), BATCH)
    )
    report["integrity"] = {
        "shape": list(channel.shape),
        "dtype": str(channel.dtype),
        "finite": bool(finite),
        "zip_crc": True,
        "zip_members": members,
        "zip_bytes": int(ZIP_OUT.stat().st_size),
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }


def main() -> None:
    for path in (BASE, PHASE_SOURCE):
        require(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    generation = generate()
    base_replay = replay.score_prediction(BASE)
    candidate_replay = replay.score_prediction(OUTPUT)
    report: dict[str, object] = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v559 preserves the best known high-order PAS/PDP skeleton but has weak "
            "complex coherence. The same-coordinate Sionna reference is a manufactured "
            "physical endpoint with coherent array/phase organization; v564 borrows "
            "only that phase organization and locks every v559 linkwise-CIR magnitude "
            "back exactly."
        ),
        "new_information": [
            "same-coordinate Sionna complex phase and array organization",
            "v559 online-best linkwise PDP magnitude skeleton",
            "v207 probe where Sionna phase plus v559 PDP reaches a much higher replay score",
        ],
        "parameters": {
            "base": str(BASE.relative_to(PATHS.root)),
            "phase_source": str(PHASE_SOURCE.relative_to(PATHS.root)),
            "operation": "restore_full_linkwise_pdp(phase_source, base)",
            "pdp_lock": "exact v559 per-(TX antenna, UE port, delay) CIR magnitude",
        },
        "generation": generation,
        "v207_replay_diagnostic": {
            "warning": (
                "same-coordinate Sionna reference was used as the phase endpoint and "
                "as this diagnostic reference; online score is the final evidence"
            ),
            "base": base_replay,
            "candidate": candidate_replay,
            "delta": {
                key: float(candidate_replay[key] - base_replay[key])
                for key in (
                    "pas",
                    "pdp",
                    "leaked_sh",
                    "ad_support",
                    "complex_coherence",
                    "nmse_score",
                    "score_traceback_leaked",
                    "score_conventional",
                )
            },
        },
        "outputs": {
            "channel": str(OUTPUT.relative_to(PATHS.root)),
            "report": str(REPORT.relative_to(PATHS.root)),
            "archive": str(ZIP_OUT.relative_to(PATHS.root)),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_package(report)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
