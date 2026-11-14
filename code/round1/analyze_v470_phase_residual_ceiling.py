#!/usr/bin/env python3
"""Measure the official-train ceiling of phase-only residual correction."""

from __future__ import annotations

import json

import numpy as np

import analyze_v205_sionna_config_calibration as metric
import analyze_v454_v453_official_fullh_local_score as v454
from project_paths import PATHS


VERSION_NUMBER = 470
VERSION = "v470_phase_residual_ceiling"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"

BASE_AD = PATHS.shared_features / "train_v98like_loo_base_ad.npy"
TRUTH = PATHS.round1 / "Round1_Train_Channel.npy"
SIONNA_HIGH = PATHS.cache_dir(208) / "v208_sionna_train_clean_H.npy"
BATCH = 4
TINY = 1e-30


def phase_unit(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    magnitude = np.abs(value)
    fallback_magnitude = np.abs(fallback)
    primary = value / np.maximum(magnitude, TINY)
    backup = fallback / np.maximum(fallback_magnitude, TINY)
    return np.where(magnitude > 1e-20, primary, backup).astype(np.complex64)


def replace_cir_phase(base: np.ndarray, phase_source: np.ndarray) -> np.ndarray:
    base_cir = np.fft.ifft(base, axis=-1)
    phase_cir = np.fft.ifft(phase_source, axis=-1)
    output = np.abs(base_cir).astype(np.float32) * phase_unit(phase_cir, base_cir)
    return np.fft.fft(output, axis=-1).astype(np.complex64)


def replace_cir_magnitude(base: np.ndarray, magnitude_source: np.ndarray) -> np.ndarray:
    base_cir = np.fft.ifft(base, axis=-1)
    source_cir = np.fft.ifft(magnitude_source, axis=-1)
    output = np.abs(source_cir).astype(np.float32) * phase_unit(base_cir, source_cir)
    return np.fft.fft(output, axis=-1).astype(np.complex64)


def empty(count: int) -> dict[str, np.ndarray]:
    return {
        "pas": np.zeros(count, dtype=np.float32),
        "pdp": np.zeros(count, dtype=np.float32),
        "error": np.zeros(count, dtype=np.float64),
        "energy": np.zeros(count, dtype=np.float64),
    }


def add_scores(
    store: dict[str, np.ndarray], ids: np.ndarray, prediction: np.ndarray, truth: np.ndarray
) -> None:
    views = metric.spectral_views(truth)
    scores = metric.spectral_scores(prediction, views)
    store["pas"][ids] = scores["pas"]
    store["pdp"][ids] = scores["pdp"]
    store["error"][ids] = np.sum(
        np.abs(prediction - truth) ** 2, axis=(1, 2, 3), dtype=np.float64
    )
    store["energy"][ids] = views["energy"]


def aggregate(store: dict[str, np.ndarray]) -> dict[str, float]:
    pas = float(np.mean(store["pas"]))
    pdp = float(np.mean(store["pdp"]))
    nmse = float(np.sum(store["error"]) / max(float(np.sum(store["energy"])), TINY))
    nmse_score = 1.0 / (1.0 + nmse)
    return {
        "pas": pas,
        "pdp": pdp,
        "global_nmse": nmse,
        "nmse_score": nmse_score,
        "official_score": 0.4 * pas + 0.4 * pdp + 0.2 * nmse_score,
    }


def delta(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: float(left[key] - right[key]) for key in left}


def main() -> None:
    for path in (BASE_AD, TRUTH, SIONNA_HIGH):
        if not path.is_file():
            raise FileNotFoundError(path)
    base_ad = np.load(BASE_AD, mmap_mode="r")
    truth_h = np.load(TRUTH, mmap_mode="r")
    sionna_h = np.load(SIONNA_HIGH, mmap_mode="r")
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    variants = {
        "base": empty(len(base_ad)),
        "truth_phase": empty(len(base_ad)),
        "sionna_phase": empty(len(base_ad)),
        "truth_magnitude": empty(len(base_ad)),
    }
    for start in range(0, len(base_ad), BATCH):
        end = min(start + BATCH, len(base_ad))
        ids = np.arange(start, end, dtype=np.int64)
        base = v454.restore_raw_h(
            np.asarray(base_ad[start:end], dtype=np.complex64), positions[start:end]
        )
        truth = np.asarray(truth_h[start:end], dtype=np.complex64)
        sionna = np.asarray(sionna_h[start:end], dtype=np.complex64)
        candidates = {
            "base": base,
            "truth_phase": replace_cir_phase(base, truth),
            "sionna_phase": replace_cir_phase(base, sionna),
            "truth_magnitude": replace_cir_magnitude(base, truth),
        }
        for name, prediction in candidates.items():
            add_scores(variants[name], ids, prediction, truth)
        if end % 100 == 0 or end == len(base_ad):
            print(f"v470 phase ceiling {end}/{len(base_ad)}", flush=True)
    scores = {name: aggregate(value) for name, value in variants.items()}
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "If phase is the main remaining ceiling after v458/v469 magnitude support correction, "
            "replacing only CIR phase should produce a large official-train score gain while PDP stays fixed."
        ),
        "scores": scores,
        "deltas_vs_base": {
            name: delta(value, scores["base"])
            for name, value in scores.items()
            if name != "base"
        },
        "decision": {
            "phase_has_0p03_ceiling": bool(
                scores["truth_phase"]["official_score"] - scores["base"]["official_score"] >= 0.03
            ),
            "raw_sionna_phase_positive": bool(
                scores["sionna_phase"]["official_score"] > scores["base"]["official_score"]
            ),
        },
        "outputs": {"report": PATHS.relative(REPORT), "prediction": None, "zip": None},
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
