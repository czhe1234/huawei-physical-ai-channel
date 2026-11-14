#!/usr/bin/env python3
"""Apply a guarded v552 point-power correction on top of v559."""

from __future__ import annotations

import hashlib
import json
import zipfile

import numpy as np

import analyze_v208_strict_historical_replay as replay
from project_paths import PATHS


VERSION_NUMBER = 563
VERSION = "v563_guarded_v552_power_overdrive"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUTPUT = CACHE_DIR / "v563_Round1_Test_Channel.npy"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

BASE = PATHS.cache_dir(559) / "v559_Round1_Test_Channel.npy"
POWER_SOURCE = PATHS.cache_dir(552) / "v552_Round1_Test_Channel.npy"
ALPHA = np.float32(2.0)
LOG_RATIO_CLIP = np.float32(1.5)
BATCH = 4
TINY = 1e-30


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
    source = np.load(POWER_SOURCE, mmap_mode="r")
    if base.shape != source.shape or base.dtype != np.dtype("complex64"):
        raise ValueError(f"shape/dtype mismatch: {base.shape=} {base.dtype=}")
    if source.dtype != np.dtype("complex64"):
        raise ValueError(f"unexpected source dtype: {source.dtype}")

    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    power_ratio = np.empty(len(base), dtype=np.float32)
    unclipped_power_ratio = np.empty(len(base), dtype=np.float32)
    scale_values = np.empty(len(base), dtype=np.float32)
    clipped = np.zeros(len(base), dtype=bool)
    finite = True
    max_abs = 0.0
    for start in range(0, len(base), BATCH):
        end = min(start + BATCH, len(base))
        base_h = np.asarray(base[start:end], dtype=np.complex64)
        source_h = np.asarray(source[start:end], dtype=np.complex64)
        base_power = np.maximum(point_power(base_h), TINY)
        source_power = np.maximum(point_power(source_h), TINY)
        raw_log_ratio = np.log(source_power) - np.log(base_power)
        clipped_log_ratio = np.clip(raw_log_ratio, -LOG_RATIO_CLIP, LOG_RATIO_CLIP)
        target_ratio = np.exp(np.float64(ALPHA) * clipped_log_ratio)
        scale = np.sqrt(target_ratio).astype(np.float32)
        candidate = (base_h * scale[:, None, None, None]).astype(np.complex64)
        finite = finite and bool(np.isfinite(candidate).all())
        max_abs = max(max_abs, float(np.max(np.abs(candidate))))
        output[start:end] = candidate
        power_ratio[start:end] = target_ratio.astype(np.float32)
        unclipped_power_ratio[start:end] = (source_power / base_power).astype(np.float32)
        scale_values[start:end] = scale
        clipped[start:end] = raw_log_ratio != clipped_log_ratio
        if end == len(base) or end % 50 == 0:
            output.flush()
            print(f"v563 guarded point-power scale {end}/{len(base)}", flush=True)
    output.flush()
    return {
        "alpha": float(ALPHA),
        "log_ratio_clip": float(LOG_RATIO_CLIP),
        "power_ratio": summarize(power_ratio),
        "unclipped_source_over_base_power_ratio": summarize(unclipped_power_ratio),
        "scale": summarize(scale_values),
        "clipped_fraction": float(np.mean(clipped)),
        "finite": bool(finite),
        "max_abs": max_abs,
    }


def write_package(report: dict[str, object]) -> None:
    dependencies = (
        "generate_v563_guarded_v552_power_overdrive.py",
        "analyze_v208_strict_historical_replay.py",
        "project_paths.py",
    )
    for name in dependencies:
        require(PATHS.code / name)
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        archive.writestr("model/", b"")
        archive.write(REPORT, "model/v563_guarded_power_report.json")
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
    for path in (BASE, POWER_SOURCE):
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
            "v559 already has the strongest PAS/PDP skeleton, but its point-power field "
            "may still cap the NMSE term. v552 carries an independent learned power "
            "field, so v563 applies that field more strongly than v562 while clipping "
            "extreme log-ratios to protect numerical spectrum floors."
        ),
        "new_information": [
            "v552 carrier-conditioned full-H model point-power field",
            "v559 online-best high-order PAS/PDP phase-retrieved skeleton",
            "v207 guarded sweep showing alpha=2.0 with ±1.5 log-ratio clip beats v562",
        ],
        "parameters": {
            "base": str(BASE.relative_to(PATHS.root)),
            "power_source": str(POWER_SOURCE.relative_to(PATHS.root)),
            "alpha": float(ALPHA),
            "log_ratio_clip": float(LOG_RATIO_CLIP),
            "mapping": (
                "power_ratio = exp(alpha * clip(log(v552_power/v559_power), "
                "-log_ratio_clip, log_ratio_clip))"
            ),
        },
        "generation": generation,
        "v207_replay_diagnostic": {
            "warning": "same-coordinate Sionna reference; online score is final evidence",
            "base": base_replay,
            "candidate": candidate_replay,
            "delta": {
                key: float(candidate_replay[key] - base_replay[key])
                for key in (
                    "pas",
                    "pdp",
                    "leaked_sh",
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
