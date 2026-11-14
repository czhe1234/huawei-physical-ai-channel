#!/usr/bin/env python3
"""Generate an all-zero channel as an explicit online-score floor diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from project_paths import PATHS


VERSION_NUMBER = 521
VERSION = "v521_zero_channel_floor_diagnostic"
OUTPUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUTPUT = OUTPUT_DIR / "v521_Round1_Test_Channel.npy"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
ARCHIVE = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"
SHAPE = (500, 256, 4, 192)
DTYPE = np.dtype(np.complex64)
CHUNK = 8 * 1024 * 1024


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            value.update(block)
    return value.hexdigest()


def generate_output() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    channel = np.lib.format.open_memmap(
        OUTPUT,
        mode="w+",
        dtype=DTYPE,
        shape=SHAPE,
    )
    for start in range(0, SHAPE[0], 10):
        channel[start : start + 10] = 0.0 + 0.0j
    channel.flush()
    del channel


def package() -> None:
    PATHS.round1_versions.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        archive.write(Path(__file__), Path(__file__).name)
        archive.write(PATHS.code_root / "project_paths.py", "project_paths.py")


def validate() -> dict[str, object]:
    channel = np.load(OUTPUT, mmap_mode="r")
    if channel.shape != SHAPE:
        raise ValueError(f"unexpected shape: {channel.shape}")
    if channel.dtype != DTYPE:
        raise ValueError(f"unexpected dtype: {channel.dtype}")
    if not np.all(np.isfinite(channel)):
        raise ValueError("non-finite output")
    if np.count_nonzero(channel) != 0:
        raise ValueError("output is not exactly zero")
    with zipfile.ZipFile(ARCHIVE) as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
        info = archive.getinfo("Round1_Test_Channel.npy")
    if bad_member is not None:
        raise ValueError(f"bad ZIP member: {bad_member}")
    if not members or members[0] != "Round1_Test_Channel.npy":
        raise ValueError(f"unexpected ZIP layout: {members}")
    return {
        "shape": list(channel.shape),
        "dtype": str(channel.dtype),
        "finite": True,
        "nonzero_count": 0,
        "point_power_sum": 0.0,
        "zip_crc": True,
        "zip_compression": "stored",
        "zip_members": members,
        "npy_file_size": int(OUTPUT.stat().st_size),
        "zip_file_size": int(ARCHIVE.stat().st_size),
        "npy_member_size": int(info.file_size),
    }


def main() -> None:
    generate_output()
    package()
    integrity = validate()
    integrity["sha256"] = digest(ARCHIVE, "sha256")
    integrity["md5"] = digest(ARCHIVE, "md5")
    report = {
        "method": VERSION,
        "status": "candidate_pending_diagnostic_noncompliant",
        "primary_hypothesis": (
            "An exact-zero prediction provides a deterministic lower-score control for the "
            "corrected online metric and submission pipeline."
        ),
        "compliance_boundary": {
            "diagnostic_only": True,
            "explicitly_noncompliant_for_final_submission": True,
            "reason": "constant all-zero output is not an AI prediction method",
        },
        "expected_behavior": {
            "pas": "approximately zero against nonzero hidden truth",
            "pdp": "approximately zero against nonzero hidden truth",
            "nmse_score": "approximately 0.5 before the 0.2 weight, so total near 0.1",
        },
        "outputs": {
            "channel": PATHS.relative(OUTPUT),
            "archive": PATHS.relative(ARCHIVE),
            "report": PATHS.relative(REPORT),
        },
        "integrity": integrity,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
