#!/usr/bin/env python3
"""Generate v566: relative-noise weak-tail guard on top of v559.

The v565 train-only analysis showed that the low spectral floor is much better
explained by a relative-power noise model than by a fixed variance.  v566 tests
that information at the safest interface: shrink only very weak cells, protect
strong support, and preserve v559's proven power/PDP skeleton.
"""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from dataclasses import dataclass

import numpy as np

import analyze_v205_sionna_config_calibration as calibration
import analyze_v208_strict_historical_replay as replay
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v424_sionna_pas_full_pdp_lock_diagnostic as v424
import generate_v558_external_pas_v549_pdp_fusion as v558
from project_paths import PATHS


VERSION_NUMBER = 566
VERSION = "v566_relative_noise_tail_guard"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUTPUT = CACHE_DIR / "v566_Round1_Test_Channel.npy"
ACTIONS = CACHE_DIR / "v566_relative_noise_tail_guard_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

BASE = PATHS.cache_dir(559) / "v559_Round1_Test_Channel.npy"
REFERENCE = PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"
NOISE_REPORT = PATHS.version_report_dir(565) / "v565_train_noise_distribution_analysis.json"

BATCH = 2
TINY = 1e-30
RELATIVE_NOISE_RATIO = 0.0087415852887559
METRICS = ("pas", "pdp", "leaked_sh", "ad_support", "complex_coherence")


@dataclass(frozen=True)
class Variant:
    name: str
    mode: str
    alpha: float
    protect: float
    min_keep: float
    power_lock: str


VARIANTS = (
    Variant("cir_mild_a020_p2", "cir", 0.20, 2.0, 0.92, "point_power"),
    Variant("cir_mid_a035_p3", "cir", 0.35, 3.0, 0.86, "point_power"),
    Variant("cir_strong_a055_p4", "cir", 0.55, 4.0, 0.78, "point_power"),
    Variant("leaked_mild_a020_p2", "leaked", 0.20, 2.0, 0.92, "linkwise_pdp"),
    Variant("leaked_mid_a035_p3", "leaked", 0.35, 3.0, 0.86, "linkwise_pdp"),
    Variant("dual_mild_a018_p2", "dual", 0.18, 2.4, 0.90, "point_power"),
)


def require(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def digest(path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def summarize(values: np.ndarray) -> dict[str, float]:
    value = np.asarray(values, dtype=np.float64)
    value = value[np.isfinite(value)]
    return {
        "mean": float(np.mean(value)),
        "p05": float(np.quantile(value, 0.05)),
        "p10": float(np.quantile(value, 0.10)),
        "median": float(np.median(value)),
        "p90": float(np.quantile(value, 0.90)),
        "p95": float(np.quantile(value, 0.95)),
        "max": float(np.max(value)),
    }


def point_power(channel: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(channel) ** 2, axis=(1, 2, 3), dtype=np.float64)


def restore_point_power(candidate: np.ndarray, base: np.ndarray) -> np.ndarray:
    current = point_power(candidate)
    target = point_power(base)
    scale = np.sqrt(target / np.maximum(current, TINY)).astype(np.float32)
    return (candidate * scale[:, None, None, None]).astype(np.complex64)


def soft_gain(power: np.ndarray, floor: np.ndarray, alpha: float, protect: float, min_keep: float) -> np.ndarray:
    floor = np.maximum(floor, TINY)
    protected = np.clip(
        (power - floor) / np.maximum((protect - 1.0) * floor, TINY),
        0.0,
        1.0,
    )
    noise_like = 1.0 - protected
    retained = np.maximum(power - alpha * floor * noise_like, min_keep * power)
    return np.sqrt(retained / np.maximum(power, TINY)).astype(np.float32)


def cir_tail_guard(channel: np.ndarray, variant: Variant) -> np.ndarray:
    cir = np.fft.ifft(channel, axis=-1)
    power = np.abs(cir).astype(np.float32) ** 2
    link_mean = np.mean(power, axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)
    kth = int(round((channel.shape[-1] - 1) * 0.25))
    q25 = np.partition(power, kth, axis=-1)[..., [kth]].astype(np.float32)
    q25_floor = q25 / np.float32(-math.log(0.75))
    relative_floor = np.float32(RELATIVE_NOISE_RATIO) * link_mean
    # q25 can be inflated by dense real paths.  The relative-power cap is the
    # train-derived part of the estimator and keeps the action from eating
    # structured medium paths.
    floor = np.minimum(q25_floor, relative_floor)
    gain = soft_gain(power, floor, variant.alpha, variant.protect, variant.min_keep)
    guarded = np.fft.fft(cir * gain, axis=-1).astype(np.complex64)
    return guarded


def leaked_tail_guard(channel: np.ndarray, variant: Variant) -> np.ndarray:
    spec = v423.leak.leaked_forward(channel)
    power = np.abs(spec).astype(np.float32) ** 2
    # The scorer-like leaked PAS target is normalized per (P, UE, V) S-H slice.
    # Use a per-slice floor, not a global floor.
    slice_mean = np.mean(power, axis=(2, 3), keepdims=True, dtype=np.float64).astype(np.float32)
    flat = power.reshape(power.shape[0], power.shape[1], -1, power.shape[4], power.shape[5])
    kth = int(round((flat.shape[2] - 1) * 0.25))
    q25 = np.partition(flat, kth, axis=2)[:, :, [kth], :, :]
    q25 = q25.reshape(power.shape[0], power.shape[1], 1, 1, power.shape[4], power.shape[5])
    q25_floor = q25 / np.float32(-math.log(0.75))
    relative_floor = np.float32(RELATIVE_NOISE_RATIO) * slice_mean
    floor = np.minimum(q25_floor, relative_floor)
    gain = soft_gain(power, floor, variant.alpha, variant.protect, variant.min_keep)
    return v423.leak.leaked_inverse(spec * gain).astype(np.complex64)


def apply_variant(base_h: np.ndarray, variant: Variant) -> np.ndarray:
    if variant.mode == "cir":
        candidate = cir_tail_guard(base_h, variant)
    elif variant.mode == "leaked":
        candidate = leaked_tail_guard(base_h, variant)
    elif variant.mode == "dual":
        candidate = leaked_tail_guard(cir_tail_guard(base_h, variant), variant)
    else:
        raise ValueError(variant.mode)

    if variant.power_lock == "point_power":
        return restore_point_power(candidate, base_h)
    if variant.power_lock == "linkwise_pdp":
        return v424.restore_full_linkwise_pdp(candidate, base_h)
    raise ValueError(variant.power_lock)


def empty_metric_arrays(n: int) -> dict[str, np.ndarray]:
    return {name: np.empty(n, dtype=np.float32) for name in METRICS}


def finalize_score(metrics: dict[str, np.ndarray], error: float, truth_energy: float) -> dict[str, float]:
    means = {name: float(np.mean(value)) for name, value in metrics.items()}
    nmse = float(error / max(truth_energy, TINY))
    nmse_score = float(1.0 / (1.0 + nmse))
    return {
        **means,
        "global_nmse": nmse,
        "nmse_score": nmse_score,
        "score_conventional": float(0.4 * means["pas"] + 0.4 * means["pdp"] + 0.2 * nmse_score),
        "score_traceback_leaked": float(0.4 * means["leaked_sh"] + 0.4 * means["pdp"] + 0.2 * nmse_score),
    }


def analyze_variants() -> tuple[str, dict[str, object], dict[str, np.ndarray]]:
    base = np.load(BASE, mmap_mode="r")
    reference = np.load(REFERENCE, mmap_mode="r")
    names = ("v559", *(variant.name for variant in VARIANTS))
    metrics = {name: empty_metric_arrays(len(base)) for name in names}
    errors = {name: 0.0 for name in names}
    structure = {
        variant.name: {
            "point_power_ratio": np.empty(len(base), dtype=np.float32),
            "phase_agreement": np.empty(len(base), dtype=np.float32),
            "pdp_error": np.empty(len(base), dtype=np.float32),
            "nmse_vs_v559_error": 0.0,
        }
        for variant in VARIANTS
    }
    truth_energy = 0.0
    base_energy = 0.0

    for start in range(0, len(base), BATCH):
        end = min(start + BATCH, len(base))
        base_h = np.asarray(base[start:end], dtype=np.complex64)
        truth_h = np.asarray(reference[start:end], dtype=np.complex64)
        truth_views = calibration.spectral_views(truth_h)
        truth_energy += float(np.sum(np.abs(truth_h) ** 2, dtype=np.float64))
        base_energy += float(np.sum(np.abs(base_h) ** 2, dtype=np.float64))

        candidates = {"v559": base_h}
        for variant in VARIANTS:
            candidate = apply_variant(base_h, variant)
            if not np.isfinite(candidate).all():
                raise FloatingPointError(f"non-finite {variant.name} block {start}:{end}")
            candidates[variant.name] = candidate
            structure[variant.name]["point_power_ratio"][start:end] = (
                point_power(candidate) / np.maximum(point_power(base_h), TINY)
            ).astype(np.float32)
            structure[variant.name]["phase_agreement"][start:end] = v423.phase_agreement(base_h, candidate)
            structure[variant.name]["pdp_error"][start:end] = v424.full_linkwise_pdp_error(candidate, base_h)
            structure[variant.name]["nmse_vs_v559_error"] += float(
                np.sum(np.abs(candidate - base_h) ** 2, dtype=np.float64)
            )

        for name, candidate in candidates.items():
            score = calibration.spectral_scores(candidate, truth_views)
            for metric in METRICS:
                metrics[name][metric][start:end] = score[metric]
            errors[name] += float(np.sum(np.abs(candidate - truth_h) ** 2, dtype=np.float64))

        if end % 50 == 0 or end == len(base):
            print(f"v566 relative-noise probe {end}/{len(base)}", flush=True)

    scores = {name: finalize_score(metrics[name], errors[name], truth_energy) for name in names}
    structural_report = {}
    for variant in VARIANTS:
        item = structure[variant.name]
        structural_report[variant.name] = {
            "parameters": variant.__dict__,
            "point_power_ratio": summarize(item["point_power_ratio"]),
            "phase_agreement_to_v559": summarize(item["phase_agreement"]),
            "full_linkwise_pdp_error": summarize(item["pdp_error"]),
            "nmse_vs_v559": float(item["nmse_vs_v559_error"] / max(base_energy, TINY)),
        }

    # Prefer replay-positive variants, but reject anything that materially hurts
    # the already-validated v559 PDP in the same-coordinate diagnostic.
    base_score = scores["v559"]
    eligible = [
        variant.name
        for variant in VARIANTS
        if scores[variant.name]["pdp"] >= base_score["pdp"] - 0.0025
        and scores[variant.name]["leaked_sh"] >= base_score["leaked_sh"] - 0.0025
        and structural_report[variant.name]["nmse_vs_v559"] <= 0.012
    ]
    if not eligible:
        eligible = [VARIANTS[0].name]
    selected = max(eligible, key=lambda name: scores[name]["score_traceback_leaked"])
    action_arrays = {
        "selected": np.asarray(selected),
        "variant_names": np.asarray([variant.name for variant in VARIANTS]),
    }
    for variant in VARIANTS:
        action_arrays[f"{variant.name}_point_power_ratio"] = structure[variant.name]["point_power_ratio"]
        action_arrays[f"{variant.name}_phase_agreement"] = structure[variant.name]["phase_agreement"]
        action_arrays[f"{variant.name}_pdp_error"] = structure[variant.name]["pdp_error"]
    return selected, {"scores": scores, "structural": structural_report, "eligible": eligible}, action_arrays


def render_selected(selected: str) -> None:
    variant_by_name = {variant.name: variant for variant in VARIANTS}
    variant = variant_by_name[selected]
    base = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape)
    for start in range(0, len(base), BATCH):
        end = min(start + BATCH, len(base))
        base_h = np.asarray(base[start:end], dtype=np.complex64)
        output[start:end] = apply_variant(base_h, variant)
        if end % 50 == 0 or end == len(base):
            output.flush()
            print(f"v566 selected render {end}/{len(base)}", flush=True)
    output.flush()


def write_package(report: dict[str, object]) -> dict[str, object]:
    dependencies = (
        "generate_v566_relative_noise_tail_guard.py",
        "analyze_v565_train_noise_distribution.py",
        "generate_v559_alternating_pas_pdp_phase_retrieval.py",
        "generate_v558_external_pas_v549_pdp_fusion.py",
        "generate_v424_sionna_pas_full_pdp_lock_diagnostic.py",
        "generate_v423_joint_sionna_pas_pdp_diagnostic.py",
        "generate_v187_leaked_pas_lifecycle_projection.py",
        "analyze_v208_strict_historical_replay.py",
        "analyze_v205_sionna_config_calibration.py",
        "project_paths.py",
    )
    for name in dependencies:
        require(PATHS.code / name)
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        archive.writestr("model/", b"")
        archive.write(REPORT, "model/v566_relative_noise_tail_guard_report.json")
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
    for path in (BASE, REFERENCE, NOISE_REPORT):
        require(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    selected, audit, action_arrays = analyze_variants()
    render_selected(selected)
    np.savez_compressed(ACTIONS, **action_arrays)
    base_replay = replay.score_prediction(BASE)
    candidate_replay = replay.score_prediction(OUTPUT)
    report: dict[str, object] = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v565 shows that official train noise is approximately relative-power "
            "rather than fixed-variance.  v566 applies that new information only as "
            "a weak tail reliability guard on the online-best v559 carrier, protecting "
            "strong support and preserving either point power or v559 linkwise PDP."
        ),
        "new_information": [
            "v565 relative noise ratio median 0.0087415853",
            "train finding that KNN residuals are structured and must not be globally subtracted",
            "v559 online-best PAS/PDP phase-retrieval carrier",
        ],
        "parameters": {
            "base": PATHS.relative(BASE),
            "relative_noise_ratio": RELATIVE_NOISE_RATIO,
            "variants": [variant.__dict__ for variant in VARIANTS],
            "selected": selected,
            "selection_rule": (
                "maximize v207 traceback-like score among variants that keep v559 "
                "PDP/leaked-SH within 0.0025 and stay within NMSE-to-v559 <= 0.012"
            ),
        },
        "diagnostic_warning": (
            "v207 replay is a same-coordinate Sionna diagnostic and is not the hidden "
            "online score.  It is used only to reject destructive denoising actions."
        ),
        "audit": audit,
        "v207_replay_diagnostic": {
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
            "channel": PATHS.relative(OUTPUT),
            "actions": PATHS.relative(ACTIONS),
            "report": PATHS.relative(REPORT),
            "archive": PATHS.relative(ZIP_OUT),
        },
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    integrity = write_package(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected": selected,
                "archive": report["outputs"]["archive"],
                "base_traceback": base_replay["score_traceback_leaked"],
                "candidate_traceback": candidate_replay["score_traceback_leaked"],
                "delta": report["v207_replay_diagnostic"]["delta"],
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
