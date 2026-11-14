#!/usr/bin/env python3
"""Attribute frozen local PDP headroom to marginal hierarchy levels."""

from __future__ import annotations

import json

import numpy as np

import analyze_v208_strict_historical_replay as v208
from project_paths import PATHS


VERSION_NUMBER = 341
VERSION = "v341_local_pdp_hierarchy_oracle"
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / f"{VERSION}.json"
SOURCE = PATHS.cache_dir(282) / "v282_Round1_Test_Channel.npy"
TRUTH = v208.TEST_TRUTH

BATCH = 5
MP = 2
MH = 16
MV = 8
UE = 4
D = 192
FLOOR = 1e-12


MARGINALS: dict[str, tuple[int, ...]] = {
    "D": (1, 2, 3, 4),
    "UD": (1, 2, 3),
    "HD": (1, 3, 4),
    "PD": (2, 3, 4),
    "VD": (1, 2, 4),
    "PUD": (2, 3),
    "HUD": (1, 3),
    "VUD": (1, 2),
    "PHD": (3, 4),
    "HVD": (1, 4),
    "PVD": (2, 4),
    "PHUD": (3,),
    "HVUD": (1,),
    "PVUD": (2,),
    "PHVD": (4,),
}


VARIANTS = {
    "base": (),
    "D": ("D",),
    "D_UD_HD": ("D", "UD", "HD"),
    "D_UD_HD_PD_VD": ("D", "UD", "HD", "PD", "VD"),
    "low_plus_HUD": ("D", "UD", "HD", "PD", "VD", "HUD"),
    "low_plus_PHUD": ("D", "UD", "HD", "PD", "VD", "PHUD"),
    "low_plus_HVUD": ("D", "UD", "HD", "PD", "VD", "HVUD"),
    "all_leave_one_axis": (
        "D", "UD", "HD", "PD", "VD", "PHUD", "HVUD", "PVUD", "PHVD"
    ),
    "full_link": ("FULL",),
}


def normalize_marginal(value: np.ndarray) -> np.ndarray:
    total = value.sum(axis=-1, keepdims=True, dtype=np.float64)
    return (value / np.maximum(total, FLOOR)).astype(np.float32)


def apply_ratio(work: np.ndarray, target: np.ndarray, axes: tuple[int, ...]) -> None:
    current = normalize_marginal(work.sum(axis=axes, dtype=np.float64).astype(np.float32))
    ratio = np.maximum(target, FLOOR) / np.maximum(current, FLOOR)
    shape = list(work.shape)
    for axis in axes:
        shape[axis] = 1
    work *= ratio.reshape(shape)
    work /= np.maximum(work.sum(axis=-1, keepdims=True), FLOOR)


def link_cos(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    numerator = np.sum(pred * truth, axis=-1, dtype=np.float64)
    norm = np.sqrt(
        np.sum(pred * pred, axis=-1, dtype=np.float64)
        * np.sum(truth * truth, axis=-1, dtype=np.float64)
    )
    return (numerator / np.maximum(norm, 1e-30)).mean(axis=(1, 2, 3, 4)).astype(np.float32)


def main() -> None:
    source = np.load(SOURCE, mmap_mode="r")
    truth_h = np.load(TRUTH, mmap_mode="r")
    scores = {name: [] for name in VARIANTS}
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        pred_h = np.asarray(source[start:end], dtype=np.complex64)
        true_h = np.asarray(truth_h[start:end], dtype=np.complex64)
        pred = np.abs(np.fft.ifft(pred_h, axis=-1)).astype(np.float32) ** 2
        true = np.abs(np.fft.ifft(true_h, axis=-1)).astype(np.float32) ** 2
        pred = pred.reshape(end - start, MP, MH, MV, UE, D)
        true = true.reshape(end - start, MP, MH, MV, UE, D)
        pred /= np.maximum(pred.sum(axis=-1, keepdims=True), FLOOR)
        true /= np.maximum(true.sum(axis=-1, keepdims=True), FLOOR)
        target = {
            name: normalize_marginal(true.sum(axis=axes, dtype=np.float64).astype(np.float32))
            for name, axes in MARGINALS.items()
        }
        for name, levels in VARIANTS.items():
            if levels == ("FULL",):
                work = true.copy()
            else:
                work = pred.copy()
                for _ in range(8):
                    for level in levels:
                        apply_ratio(work, target[level], MARGINALS[level])
            scores[name].append(link_cos(work, true))
    merged = {name: np.concatenate(parts) for name, parts in scores.items()}
    base = float(np.mean(merged["base"]))
    result = {
        name: {
            "mean": float(np.mean(value)),
            "delta_vs_base": float(np.mean(value) - base),
            "p10": float(np.quantile(value, 0.10)),
            "median": float(np.median(value)),
        }
        for name, value in merged.items()
    }
    report = {
        "method": VERSION,
        "status": "analysis_only_oracle_no_candidate",
        "purpose": (
            "Determine which exact linkwise PDP hierarchy explains the gap between strict v282 "
            "and the frozen v207/v212-like local target. Oracle marginals are diagnostic only."
        ),
        "source": PATHS.relative(SOURCE),
        "truth_usage": "frozen synthetic local attribution only; never a final model input or label",
        "variants": {name: list(levels) for name, levels in VARIANTS.items()},
        "results": result,
        "report": PATHS.relative(REPORT),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
