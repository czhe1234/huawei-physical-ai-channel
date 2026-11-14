#!/usr/bin/env python3
"""Package the existing v207 raw Sionna test channel as an online diagnostic."""

from __future__ import annotations

import hashlib
import json
import zipfile

import numpy as np

from project_paths import PATHS


VERSION_NUMBER = 364
VERSION = "v364_direct_sionna_full_h_diagnostic"
SOURCE = PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}_package.json"
EXPECTED_SHAPE = (500, 256, 4, 192)


def digests(path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            sha256.update(block)
            md5.update(block)
    return sha256.hexdigest(), md5.hexdigest()


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    channel = np.load(SOURCE, mmap_mode="r")
    if channel.shape != EXPECTED_SHAPE or channel.dtype != np.complex64:
        raise ValueError(f"invalid source layout: {channel.shape} {channel.dtype}")
    finite = True
    point_power = np.empty(len(channel), dtype=np.float64)
    for start in range(0, len(channel), 16):
        block = np.asarray(channel[start : start + 16], dtype=np.complex64)
        finite = finite and bool(
            np.isfinite(block.real).all() and np.isfinite(block.imag).all()
        )
        point_power[start : start + len(block)] = np.sum(
            np.abs(block) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
    if not finite:
        raise ValueError("raw Sionna channel contains non-finite values")

    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(SOURCE, "Round1_Test_Channel.npy")
        for source in (
            PATHS.code / "package_v364_direct_sionna_full_h_diagnostic.py",
            PATHS.code / "generate_v207_sionna_test_reference.py",
            PATHS.code_root / "project_paths.py",
        ):
            archive.write(source, source.name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
    if bad_member is not None:
        raise ValueError(f"ZIP CRC failure: {bad_member}")
    sha256, md5 = digests(ZIP_OUT)
    report = {
        "method": VERSION,
        "status": "diagnostic_pending_not_compliant",
        "purpose": (
            "Measure how closely the v207 reconstructed high-budget Sionna configuration "
            "matches the hidden official test channel when submitted without any AI model, "
            "hybrid carrier, PDP projection, phase correction, or power correction."
        ),
        "compliance_warning": (
            "This is a pure Sionna/PathSolver diagnostic and must not be treated as the final "
            "AI/Physical-AI competition solution."
        ),
        "source": {
            "channel": PATHS.relative(SOURCE),
            "generator": "code/generate_v207_sionna_test_reference.py",
            "configuration": {
                "frequency_hz": 6.0e9,
                "tx_azimuth_deg": -30.0,
                "tx_tilt_deg": -9.53,
                "max_depth": 16,
                "samples_per_src": 1_000_000,
                "max_num_paths_per_src": 1_000_000,
                "los": True,
                "specular_reflection": True,
                "refraction": True,
                "diffuse_reflection": False,
                "diffraction": False,
                "seed_policy": "42 + batch_start",
            },
        },
        "transform": "none; raw v207 complex64 full H is archived byte-for-byte",
        "integrity": {
            "shape": list(channel.shape),
            "dtype": str(channel.dtype),
            "finite": finite,
            "zero_power_count": int(np.sum(point_power == 0.0)),
            "point_power": {
                "mean": float(np.mean(point_power)),
                "median": float(np.median(point_power)),
                "p10": float(np.quantile(point_power, 0.10)),
                "p90": float(np.quantile(point_power, 0.90)),
            },
            "zip_crc": True,
            "zip_members": members,
            "sha256": sha256,
            "md5": md5,
        },
        "outputs": {
            "zip": PATHS.relative(ZIP_OUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
