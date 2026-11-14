#!/usr/bin/env python3
"""Generate v567: low-rank path-matrix denoising on the v559 carrier."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from dataclasses import asdict, dataclass

import numpy as np

import analyze_v205_sionna_config_calibration as calibration
import generate_v423_joint_sionna_pas_pdp_diagnostic as v423
import generate_v424_sionna_pas_full_pdp_lock_diagnostic as v424
from project_paths import PATHS


VERSION_NUMBER = 567
VERSION = "v567_lowrank_path_noise_deconvolution"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUTPUT = CACHE_DIR / "v567_Round1_Test_Channel.npy"
ACTIONS = CACHE_DIR / "v567_lowrank_path_noise_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

BASE = PATHS.cache_dir(559) / "v559_Round1_Test_Channel.npy"
REFERENCE = PATHS.cache_dir(207) / "v207_sionna_test_reference_H.npy"
TRAIN = PATHS.round1 / "Round1_Train_Channel.npy"
NOISE_REPORT = (
    PATHS.version_report_dir(565) / "v565_train_noise_distribution_analysis.json"
)

RELATIVE_NOISE_RATIO = np.float32(0.0087415852887559)
DELAY_BINS = 192
SPATIAL_ROWS = 128
POL_UE_COLUMNS = 8
RANDOM_SEEDS = (56711, 56729, 56747)
SYNTHETIC_POINTS_PER_SEED = 60
BATCH = 1
TINY = 1e-30
METRICS = ("pas", "pdp", "leaked_sh", "ad_support", "complex_coherence")


@dataclass(frozen=True)
class Variant:
    name: str
    floor_mode: str
    edge_fraction: float


VARIANTS = (
    Variant("relative_edge050", "relative_link", 0.50),
    Variant("relative_edge064", "relative_link", 0.64),
    Variant("relative_edge075", "relative_link", 0.75),
    Variant("relative_edge090", "relative_link", 0.90),
    Variant("empirical_edge075", "empirical_capped", 0.75),
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


def to_path_matrices(cir: np.ndarray) -> np.ndarray:
    batch = len(cir)
    return (
        cir.reshape(batch, 2, 16, 8, 4, DELAY_BINS)
        .transpose(0, 5, 2, 3, 1, 4)
        .reshape(batch, DELAY_BINS, SPATIAL_ROWS, POL_UE_COLUMNS)
    )


def from_path_matrices(matrices: np.ndarray) -> np.ndarray:
    batch = len(matrices)
    return (
        matrices.reshape(batch, DELAY_BINS, 16, 8, 2, 4)
        .transpose(0, 4, 2, 3, 5, 1)
        .reshape(batch, 256, 4, DELAY_BINS)
    )


def link_noise_floor(cir: np.ndarray, mode: str) -> np.ndarray:
    power = np.abs(cir).astype(np.float32) ** 2
    link_mean = np.mean(power, axis=-1, keepdims=True, dtype=np.float64).astype(
        np.float32
    )
    relative = RELATIVE_NOISE_RATIO * link_mean
    if mode == "relative_link":
        floor = relative
    elif mode == "empirical_capped":
        kth = int(round((DELAY_BINS - 1) * 0.25))
        q25 = np.partition(power, kth, axis=-1)[..., [kth]]
        empirical = q25 / np.float32(-math.log(0.75))
        floor = np.minimum(empirical, relative)
    else:
        raise ValueError(mode)
    return np.broadcast_to(floor, cir.shape)


def deconvolve(channel: np.ndarray, variant: Variant) -> np.ndarray:
    cir = np.fft.ifft(channel, axis=-1)
    floor = to_path_matrices(link_noise_floor(cir, variant.floor_mode))
    matrices = to_path_matrices(cir)
    scale = np.sqrt(np.maximum(floor, TINY)).astype(np.float32)
    whitened = matrices / scale
    left, singular, right = np.linalg.svd(whitened, full_matrices=False)
    edge_squared = float(
        (math.sqrt(SPATIAL_ROWS) + math.sqrt(POL_UE_COLUMNS)) ** 2
    )
    retained = np.sqrt(
        np.maximum(
            singular * singular - np.float32(variant.edge_fraction * edge_squared),
            0.0,
        )
    ).astype(np.float32)
    reconstructed = (left * retained[..., None, :]) @ right
    denoised_cir = from_path_matrices(reconstructed * scale)
    return np.fft.fft(denoised_cir, axis=-1).astype(np.complex64)


def empty_metric_sums() -> dict[str, float]:
    return {name: 0.0 for name in METRICS}


def finalize(
    metric_sums: dict[str, float], error: float, truth_energy: float, count: int
) -> dict[str, float]:
    means = {name: value / max(count, 1) for name, value in metric_sums.items()}
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
            0.4 * means["leaked_sh"]
            + 0.4 * means["pdp"]
            + 0.2 * nmse_score
        ),
    }


def score_candidates(
    truth: np.ndarray, candidates: dict[str, np.ndarray]
) -> dict[str, dict[str, object]]:
    truth_views = calibration.spectral_views(truth)
    truth_energy = float(np.sum(np.abs(truth) ** 2, dtype=np.float64))
    output = {}
    for name, candidate in candidates.items():
        scores = calibration.spectral_scores(candidate, truth_views)
        metric_sums = {
            metric: float(np.sum(scores[metric], dtype=np.float64))
            for metric in METRICS
        }
        error = float(np.sum(np.abs(candidate - truth) ** 2, dtype=np.float64))
        output[name] = {
            "metric_sums": metric_sums,
            "error": error,
            "truth_energy": truth_energy,
            "count": len(truth),
        }
    return output


def synthetic_noise_validation() -> tuple[str, dict[str, object]]:
    train = np.load(TRAIN, mmap_mode="r")
    nonzero = np.flatnonzero(
        np.sum(np.abs(train), axis=(1, 2, 3), dtype=np.float64) > 0.0
    )
    rows = []
    per_variant_deltas = {variant.name: [] for variant in VARIANTS}

    for seed in RANDOM_SEEDS:
        rng = np.random.default_rng(seed)
        ids = np.sort(
            rng.choice(nonzero, size=SYNTHETIC_POINTS_PER_SEED, replace=False)
        )
        totals = {
            name: {
                "metric_sums": empty_metric_sums(),
                "error": 0.0,
                "truth_energy": 0.0,
                "count": 0,
            }
            for name in ("noisy", *(variant.name for variant in VARIANTS))
        }
        for offset in range(0, len(ids), BATCH):
            local_ids = ids[offset : offset + BATCH]
            truth = np.asarray(train[local_ids], dtype=np.complex64)
            link_power = np.mean(
                np.abs(truth) ** 2,
                axis=-1,
                keepdims=True,
                dtype=np.float64,
            )
            sigma = np.sqrt(RELATIVE_NOISE_RATIO * link_power / 2.0).astype(
                np.float32
            )
            noise = (
                rng.standard_normal(truth.shape, dtype=np.float32)
                + 1j * rng.standard_normal(truth.shape, dtype=np.float32)
            ) * sigma
            noisy = (truth + noise).astype(np.complex64)
            candidates = {"noisy": noisy}
            for variant in VARIANTS:
                candidates[variant.name] = deconvolve(noisy, variant)
            local = score_candidates(truth, candidates)
            for name, values in local.items():
                target = totals[name]
                for metric in METRICS:
                    target["metric_sums"][metric] += values["metric_sums"][metric]
                target["error"] += values["error"]
                target["truth_energy"] += values["truth_energy"]
                target["count"] += values["count"]

        finalized = {
            name: finalize(
                values["metric_sums"],
                values["error"],
                values["truth_energy"],
                values["count"],
            )
            for name, values in totals.items()
        }
        base = finalized["noisy"]
        seed_row = {"seed": seed, "noisy": base, "variants": {}}
        for variant in VARIANTS:
            current = finalized[variant.name]
            delta = {
                "score_conventional": float(
                    current["score_conventional"] - base["score_conventional"]
                ),
                "score_traceback_leaked": float(
                    current["score_traceback_leaked"]
                    - base["score_traceback_leaked"]
                ),
                "global_nmse": float(current["global_nmse"] - base["global_nmse"]),
            }
            seed_row["variants"][variant.name] = {"score": current, "delta": delta}
            per_variant_deltas[variant.name].append(delta)
        rows.append(seed_row)
        print(f"v567 synthetic validation seed={seed} complete", flush=True)

    summary = {}
    eligible = []
    for variant in VARIANTS:
        deltas = per_variant_deltas[variant.name]
        conv = np.asarray([row["score_conventional"] for row in deltas])
        traceback = np.asarray([row["score_traceback_leaked"] for row in deltas])
        nmse = np.asarray([row["global_nmse"] for row in deltas])
        summary[variant.name] = {
            "conventional_delta": summarize(conv),
            "traceback_delta": summarize(traceback),
            "nmse_delta": summarize(nmse),
            "all_seed_conventional_positive": bool(np.all(conv > 0.0)),
            "all_seed_traceback_positive": bool(np.all(traceback > 0.0)),
            "selection_value": float(np.mean(0.5 * (conv + traceback))),
        }
        if np.all(conv > 0.0) and np.all(traceback > 0.0) and np.all(nmse < 0.0):
            eligible.append(variant.name)
    if not eligible:
        raise RuntimeError("no low-rank variant passed all synthetic-noise seeds")
    selected = max(eligible, key=lambda name: summary[name]["selection_value"])
    return selected, {"per_seed": rows, "summary": summary, "eligible": eligible}


def audit_v207() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    base = np.load(BASE, mmap_mode="r")
    reference = np.load(REFERENCE, mmap_mode="r")
    names = ("v559", *(variant.name for variant in VARIANTS))
    totals = {
        name: {
            "metric_sums": empty_metric_sums(),
            "error": 0.0,
            "truth_energy": 0.0,
            "count": 0,
        }
        for name in names
    }
    structure = {
        variant.name: {
            "point_power_ratio": np.empty(len(base), dtype=np.float32),
            "phase_agreement": np.empty(len(base), dtype=np.float32),
            "pdp_error": np.empty(len(base), dtype=np.float32),
            "nmse_vs_v559": np.empty(len(base), dtype=np.float32),
        }
        for variant in VARIANTS
    }

    for start in range(0, len(base), BATCH):
        end = min(start + BATCH, len(base))
        base_h = np.asarray(base[start:end], dtype=np.complex64)
        truth_h = np.asarray(reference[start:end], dtype=np.complex64)
        candidates = {"v559": base_h}
        for variant in VARIANTS:
            candidate = deconvolve(base_h, variant)
            if not np.isfinite(candidate).all():
                raise FloatingPointError(f"non-finite {variant.name} at {start}:{end}")
            candidates[variant.name] = candidate
            base_power = point_power(base_h)
            structure[variant.name]["point_power_ratio"][start:end] = (
                point_power(candidate) / np.maximum(base_power, TINY)
            ).astype(np.float32)
            structure[variant.name]["phase_agreement"][start:end] = (
                v423.phase_agreement(base_h, candidate)
            )
            structure[variant.name]["pdp_error"][start:end] = (
                v424.full_linkwise_pdp_error(candidate, base_h)
            )
            structure[variant.name]["nmse_vs_v559"][start:end] = (
                np.sum(
                    np.abs(candidate - base_h) ** 2,
                    axis=(1, 2, 3),
                    dtype=np.float64,
                )
                / np.maximum(base_power, TINY)
            ).astype(np.float32)
        local = score_candidates(truth_h, candidates)
        for name, values in local.items():
            target = totals[name]
            for metric in METRICS:
                target["metric_sums"][metric] += values["metric_sums"][metric]
            target["error"] += values["error"]
            target["truth_energy"] += values["truth_energy"]
            target["count"] += values["count"]
        if end % 50 == 0 or end == len(base):
            print(f"v567 v207 audit {end}/{len(base)}", flush=True)

    scores = {
        name: finalize(
            values["metric_sums"],
            values["error"],
            values["truth_energy"],
            values["count"],
        )
        for name, values in totals.items()
    }
    structural_report = {
        name: {key: summarize(value) for key, value in values.items()}
        for name, values in structure.items()
    }
    arrays = {
        f"{name}_{key}": value
        for name, values in structure.items()
        for key, value in values.items()
    }
    return {"scores": scores, "structural": structural_report}, arrays


def render_selected(selected: str) -> None:
    variant = {item.name: item for item in VARIANTS}[selected]
    base = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    for start in range(0, len(base), BATCH):
        end = min(start + BATCH, len(base))
        output[start:end] = deconvolve(
            np.asarray(base[start:end], dtype=np.complex64), variant
        )
        if end % 50 == 0 or end == len(base):
            output.flush()
            print(f"v567 selected render {end}/{len(base)}", flush=True)
    output.flush()


def write_package(report: dict[str, object]) -> dict[str, object]:
    dependencies = (
        "generate_v567_lowrank_path_noise_deconvolution.py",
        "analyze_v565_train_noise_distribution.py",
        "generate_v423_joint_sionna_pas_pdp_diagnostic.py",
        "generate_v424_sionna_pas_full_pdp_lock_diagnostic.py",
        "generate_v187_leaked_pas_lifecycle_projection.py",
        "analyze_v205_sionna_config_calibration.py",
        "project_paths.py",
    )
    for name in dependencies:
        require(PATHS.code / name)
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        archive.writestr("model/", b"")
        archive.write(REPORT, "model/v567_lowrank_noise_report.json")
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
    for path in (BASE, REFERENCE, TRAIN, NOISE_REPORT):
        require(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    selected, synthetic = synthetic_noise_validation()
    audit, action_arrays = audit_v207()
    render_selected(selected)
    np.savez_compressed(
        ACTIONS,
        selected=np.asarray(selected),
        variant_names=np.asarray([variant.name for variant in VARIANTS]),
        **action_arrays,
    )

    report: dict[str, object] = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "A physical path at one delay is approximately low-rank over the 128 "
            "array positions and 8 polarization/UE channels, whereas relative complex "
            "Gaussian noise is full-rank. Random-matrix singular-value deconvolution "
            "should separate them more effectively than v566 independent tail shrinkage."
        ),
        "new_information": [
            "v565 relative per-link noise scale inferred from official train H",
            "within-point train analysis showing median log floor-vs-link-power slope near 0.89",
            "Sionna path outer-product structure at fixed delay",
            "multi-seed synthetic-noise recovery on official train H",
        ],
        "parameters": {
            "base": PATHS.relative(BASE),
            "relative_noise_ratio": float(RELATIVE_NOISE_RATIO),
            "path_matrix": [SPATIAL_ROWS, POL_UE_COLUMNS],
            "array_reshape": "[2 polarization,16 H,8 V,4 UE] -> [128 spatial,8 P-UE]",
            "random_matrix_edge": "(sqrt(128)+sqrt(8))^2",
            "variants": [asdict(variant) for variant in VARIANTS],
            "synthetic_seeds": list(RANDOM_SEEDS),
            "synthetic_points_per_seed": SYNTHETIC_POINTS_PER_SEED,
            "selected": selected,
            "power_restore": False,
            "selection_rule": (
                "all three injected-noise seeds must improve conventional score, "
                "traceback-like score, and NMSE; maximize mean of the two score gains"
            ),
        },
        "synthetic_noise_validation": synthetic,
        "v207_diagnostic": {
            "warning": (
                "The same-coordinate v207 Sionna channel is only a destructive-action "
                "audit and did not select the variant. Online feedback is final."
            ),
            **audit,
        },
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
                "synthetic_selected": synthetic["summary"][selected],
                "v559": audit["scores"]["v559"],
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
