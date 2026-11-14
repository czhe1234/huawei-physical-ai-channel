#!/usr/bin/env python3
"""Deploy the v576 dual PAS/PDP response model as an online probe on v567."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

import generate_v187_leaked_pas_lifecycle_projection as leak
import generate_v458_sionna_multimouth_extreme_diagnostic as v458
import generate_v543_disagreement_conditioned_cell_fusion as v543
import generate_v575_physical_response_basis as v575_source
import generate_v577_test_physical_responses as response_source
import train_v575_physical_response_basis as v575
import train_v576_dual_metric_physical_response as v576
from project_paths import PATHS


VERSION_NUMBER = 577
VERSION = "v577_dual_metric_physical_response_probe"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUTPUT = CACHE_DIR / "v577_Round1_Test_Channel.npy"
ACTIONS = CACHE_DIR / "v577_dual_metric_physical_response_actions.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"
TARGET_DIR = CACHE_DIR / "predicted_targets"

BASE = PATHS.cache_dir(567) / "v567_Round1_Test_Channel.npy"
CHECKPOINT_DIR = PATHS.checkpoints / v576.VERSION
PAS_ANCHORS = {
    "sh": PATHS.cache_dir(543) / "v543_cellwise_joint_target.npy",
    "psh": (
        PATHS.cache_dir(549)
        / "pas_calibration"
        / "test_targets"
        / "n24_test_psh_calibrated.npy"
    ),
    "shue": (
        PATHS.cache_dir(549)
        / "pas_calibration"
        / "test_targets"
        / "n24_test_shue_calibrated.npy"
    ),
    "shv": (
        PATHS.cache_dir(549)
        / "pas_calibration"
        / "test_targets"
        / "n24_test_shv_calibrated.npy"
    ),
}
PDP_ANCHOR = v458.CALIBRATED_LINK_TARGET
TARGET_PATHS = {
    name: TARGET_DIR / f"v577_{name}_target.npy"
    for name in (*v575.PAS_NAMES, "pdp")
}

ROWS = 500
MODEL_WIDTH = 192
COEFFICIENT_CAP = 1.5
PREDICTION_BATCH = 64
TARGET_BATCH = 4
RENDER_BATCH = 2
LOG_RATIO_CLIP = 1.5
PAS_STRENGTH = 1.0
PDP_STRENGTH = 1.0
IPF_ITERS = 4
FLOOR = 1e-10
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-device", type=int, default=0)
    return parser.parse_args()


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def normalize_global(value: np.ndarray) -> np.ndarray:
    axes = tuple(range(1, value.ndim))
    total = value.sum(axis=axes, keepdims=True, dtype=np.float64)
    return (value / np.maximum(total, TINY)).astype(np.float32)


def normalize_last(value: np.ndarray) -> np.ndarray:
    total = value.sum(axis=-1, keepdims=True, dtype=np.float64)
    output = value / np.maximum(total, TINY)
    output[total[..., 0] <= TINY] = 0.0
    return output.astype(np.float32)


def load_summary_matrix() -> np.ndarray:
    pair_columns: list[np.ndarray] = []
    for item in v575_source.PAIRS:
        seed_paths = [
            response_source.response_paths(item, seed)
            for seed in v575_source.PAIR_SEEDS
        ]
        for paths in seed_paths:
            for path in paths.values():
                require(path)
        local_columns: list[np.ndarray] = []
        for name in (*v575.PAS_NAMES, "pdp"):
            first = np.load(seed_paths[0][name], mmap_mode="r")
            second = np.load(seed_paths[1][name], mmap_mode="r")
            instability = np.empty(ROWS, dtype=np.float32)
            for start in range(0, ROWS, TARGET_BATCH):
                stop = min(start + TARGET_BATCH, ROWS)
                difference = (
                    np.asarray(first[start:stop], dtype=np.float32)
                    - np.asarray(second[start:stop], dtype=np.float32)
                )
                instability[start:stop] = np.sqrt(
                    np.mean(
                        difference.astype(np.float64) ** 2,
                        axis=tuple(range(1, difference.ndim)),
                    )
                ).astype(np.float32)
            local_columns.append(instability[:, None])

        seed_summaries = [np.load(paths["summary"]) for paths in seed_paths]
        for key in (
            "minus_path_count",
            "plus_path_count",
            "minus_log_power",
            "plus_log_power",
            "sh_response_rms",
            "psh_response_rms",
            "shue_response_rms",
            "shv_response_rms",
            "pdp_response_rms",
        ):
            first = np.asarray(seed_summaries[0][key], dtype=np.float32)
            second = np.asarray(seed_summaries[1][key], dtype=np.float32)
            mean = 0.5 * (first + second)
            spread = np.abs(first - second)
            if key == "plus_path_count":
                minus = 0.5 * (
                    np.asarray(
                        seed_summaries[0]["minus_path_count"], dtype=np.float32
                    )
                    + np.asarray(
                        seed_summaries[1]["minus_path_count"], dtype=np.float32
                    )
                )
                local_columns.extend(
                    [
                        np.log1p(np.maximum(mean, 0.0))[:, None],
                        ((mean - minus) / np.maximum(1.0, mean + minus))[:, None],
                    ]
                )
            elif key == "plus_log_power":
                minus = 0.5 * (
                    np.asarray(
                        seed_summaries[0]["minus_log_power"], dtype=np.float32
                    )
                    + np.asarray(
                        seed_summaries[1]["minus_log_power"], dtype=np.float32
                    )
                )
                local_columns.append((mean - minus)[:, None])
            elif key.startswith("minus_"):
                continue
            else:
                local_columns.extend([mean[:, None], spread[:, None]])
        pair_columns.append(np.concatenate(local_columns, axis=1))
    return np.concatenate(pair_columns, axis=1).astype(np.float32)


def predict_coefficients(
    summary: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, static_test, _ = v543.load_static_features()
    features = np.concatenate([static_test, summary], axis=1).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    pas_models = []
    pdp_models = []
    for fold in range(5):
        checkpoint_path = CHECKPOINT_DIR / f"main_fold_{fold}.pt"
        require(checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
        if len(mean) != features.shape[1]:
            raise ValueError(
                f"v576 feature mismatch: checkpoint={len(mean)} test={features.shape[1]}"
            )
        standardized = (features - mean) / std
        feature_tensor = torch.from_numpy(standardized).to(device)
        model = v576.DualCoefficientModel(
            features.shape[1], MODEL_WIDTH, COEFFICIENT_CAP
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        pas = np.empty((ROWS, v575.PAIR_COUNT), dtype=np.float32)
        pdp = np.empty((ROWS, v575.PAIR_COUNT), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, ROWS, PREDICTION_BATCH):
                stop = min(start + PREDICTION_BATCH, ROWS)
                local_pas, local_pdp = model(feature_tensor[start:stop])
                pas[start:stop] = local_pas.cpu().numpy()
                pdp[start:stop] = local_pdp.cpu().numpy()
        pas_models.append(pas)
        pdp_models.append(pdp)
    pas_stack = np.stack(pas_models)
    pdp_stack = np.stack(pdp_models)
    return (
        pas_stack.mean(axis=0).astype(np.float32),
        pdp_stack.mean(axis=0).astype(np.float32),
        pas_stack.std(axis=0).astype(np.float32),
        pdp_stack.std(axis=0).astype(np.float32),
    )


def render_response_target(
    name: str, anchor_path: Path, coefficients: np.ndarray
) -> np.ndarray:
    require(anchor_path)
    anchor = np.load(anchor_path, mmap_mode="r")
    if len(anchor) != ROWS:
        raise ValueError(f"unexpected {name} anchor rows: {anchor.shape}")
    flat_dim = int(np.prod(anchor.shape[1:]))
    output_shape = (ROWS, 1024, 192) if name == "pdp" else (ROWS, flat_dim)
    output = np.lib.format.open_memmap(
        TARGET_PATHS[name], mode="w+", dtype=np.float32, shape=output_shape
    )
    response_arrays = []
    for item in v575_source.PAIRS:
        first = np.load(
            response_source.response_paths(item, v575_source.PAIR_SEEDS[0])[name],
            mmap_mode="r",
        )
        second = np.load(
            response_source.response_paths(item, v575_source.PAIR_SEEDS[1])[name],
            mmap_mode="r",
        )
        response_arrays.append((first, second))

    for start in range(0, ROWS, TARGET_BATCH):
        stop = min(start + TARGET_BATCH, ROWS)
        local_anchor = np.asarray(anchor[start:stop], dtype=np.float32).reshape(
            stop - start, *output_shape[1:]
        )
        amplitude = np.sqrt(np.maximum(local_anchor, FLOOR)).astype(np.float32)
        for pair_index, (first, second) in enumerate(response_arrays):
            response = np.float32(0.5) * (
                np.asarray(first[start:stop], dtype=np.float32)
                + np.asarray(second[start:stop], dtype=np.float32)
            )
            coefficient = coefficients[start:stop, pair_index]
            coefficient = coefficient.reshape(
                (stop - start,) + (1,) * (response.ndim - 1)
            )
            amplitude += coefficient * response
        power = np.maximum(amplitude, 0.0) ** 2
        output[start:stop] = (
            normalize_last(power) if name == "pdp" else normalize_global(power)
        )
        if stop == ROWS or stop % 100 == 0:
            output.flush()
            print(f"v577 target {name} {stop}/{ROWS}", flush=True)
    output.flush()
    return np.load(TARGET_PATHS[name], mmap_mode="r")


def marginals(full: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "sh": full.sum(axis=(1, 4, 5), dtype=np.float64).astype(np.float32),
        "psh": full.sum(axis=(4, 5), dtype=np.float64).astype(np.float32),
        "shue": full.sum(axis=(1, 5), dtype=np.float64).astype(np.float32),
        "shv": full.sum(axis=(1, 4), dtype=np.float64).astype(np.float32),
    }


def transfer_residual(
    current: np.ndarray,
    predicted: np.ndarray,
    anchor: np.ndarray,
    strength: float,
    last_axis: bool = False,
) -> np.ndarray:
    log_ratio = np.log(np.maximum(predicted, FLOOR)) - np.log(
        np.maximum(anchor, FLOOR)
    )
    log_ratio = np.clip(log_ratio, -LOG_RATIO_CLIP, LOG_RATIO_CLIP)
    desired = current * np.exp(np.float32(strength) * log_ratio)
    return normalize_last(desired) if last_axis else normalize_global(desired)


def apply_ratio(work: np.ndarray, ratio: np.ndarray, name: str, exponent: float) -> None:
    ratio = np.power(np.maximum(ratio, FLOOR), exponent, dtype=np.float32)
    if name == "sh":
        work *= ratio[:, None, :, :, None, None]
    elif name == "psh":
        work *= ratio[:, :, :, :, None, None]
    elif name == "shue":
        work *= ratio[:, None, :, :, :, None]
    elif name == "shv":
        work *= ratio[:, None, :, :, None, :]
    else:
        raise ValueError(name)


def build_full_pas_target(
    current: np.ndarray, desired: dict[str, np.ndarray]
) -> np.ndarray:
    work = np.asarray(current, dtype=np.float32).copy()
    exponents = {"sh": 0.70, "psh": 0.42, "shue": 0.46, "shv": 0.42}
    for _ in range(IPF_ITERS):
        current_marginals = marginals(work)
        for name in v575.PAS_NAMES:
            ratio = desired[name] / np.maximum(current_marginals[name], FLOOR)
            apply_ratio(work, ratio, name, exponents[name])
        work = normalize_global(work)
    return work


def phase_unit(value: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    magnitude = np.abs(value).astype(np.float32)
    output = value / np.maximum(magnitude, 1e-20)
    if fallback is not None:
        fallback_magnitude = np.abs(fallback).astype(np.float32)
        fallback_phase = fallback / np.maximum(fallback_magnitude, 1e-20)
        output = np.where(magnitude > 1e-20, output, fallback_phase)
    return output.astype(np.complex64)


def render_output(
    pas_targets: dict[str, np.ndarray], pdp_target: np.ndarray
) -> dict[str, np.ndarray]:
    base = np.load(BASE, mmap_mode="r")
    pas_anchors = {
        name: np.load(path, mmap_mode="r") for name, path in PAS_ANCHORS.items()
    }
    pdp_anchor = np.load(PDP_ANCHOR, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    diagnostics: dict[str, list[np.ndarray]] = {
        "pas_full_l1": [],
        "pdp_link_l1": [],
        "nmse_to_v567": [],
        "point_power_ratio": [],
        "final_pdp_target_max_abs": [],
    }

    for start in range(0, ROWS, RENDER_BATCH):
        stop = min(start + RENDER_BATCH, ROWS)
        base_h = np.asarray(base[start:stop], dtype=np.complex64)
        spec = leak.leaked_forward(base_h)
        power = leak.leaked_cell_power(spec).astype(np.float32)
        total = power.sum(axis=(1, 2, 3, 4, 5), keepdims=True, dtype=np.float64)
        support = (power / np.maximum(total, TINY)).astype(np.float32)
        current_marginals = marginals(support)
        desired_marginals = {}
        for name in v575.PAS_NAMES:
            predicted = np.asarray(pas_targets[name][start:stop], dtype=np.float32)
            anchor = np.asarray(pas_anchors[name][start:stop], dtype=np.float32)
            predicted = predicted.reshape(current_marginals[name].shape)
            anchor = anchor.reshape(current_marginals[name].shape)
            desired_marginals[name] = transfer_residual(
                current_marginals[name], predicted, anchor, PAS_STRENGTH
            )
        full_target = build_full_pas_target(support, desired_marginals)
        target_abs = np.sqrt(
            np.maximum(full_target * total.astype(np.float32), 0.0)
        ).astype(np.float32)
        candidate = leak.leaked_inverse(target_abs * phase_unit(spec)).astype(
            np.complex64
        )

        base_cir = np.fft.ifft(base_h, axis=-1).astype(np.complex64)
        candidate_cir = np.fft.ifft(candidate, axis=-1).astype(np.complex64)
        base_cir_power = np.abs(base_cir).astype(np.float32) ** 2
        link_energy = base_cir_power.sum(
            axis=-1, keepdims=True, dtype=np.float64
        ).astype(np.float32)
        current_pdp = normalize_last(base_cir_power).reshape(stop - start, 1024, 192)
        predicted_pdp = np.asarray(pdp_target[start:stop], dtype=np.float32)
        anchor_pdp = np.asarray(pdp_anchor[start:stop], dtype=np.float32)
        desired_pdp = transfer_residual(
            current_pdp, predicted_pdp, anchor_pdp, PDP_STRENGTH, last_axis=True
        ).reshape(stop - start, 256, 4, 192)
        target_cir_abs = np.sqrt(
            np.maximum(desired_pdp * link_energy, 0.0)
        ).astype(np.float32)
        final_cir = target_cir_abs * phase_unit(candidate_cir, base_cir)
        final_h = np.fft.fft(final_cir, axis=-1).astype(np.complex64)
        if not (np.isfinite(final_h.real).all() and np.isfinite(final_h.imag).all()):
            raise FloatingPointError(f"non-finite v577 output block {start}:{stop}")
        output[start:stop] = final_h

        final_cir_power = np.abs(np.fft.ifft(final_h, axis=-1)).astype(np.float32) ** 2
        final_pdp = normalize_last(final_cir_power).reshape(stop - start, 1024, 192)
        base_energy = np.sum(np.abs(base_h) ** 2, axis=(1, 2, 3), dtype=np.float64)
        final_energy = np.sum(np.abs(final_h) ** 2, axis=(1, 2, 3), dtype=np.float64)
        error = np.sum(np.abs(final_h - base_h) ** 2, axis=(1, 2, 3), dtype=np.float64)
        diagnostics["pas_full_l1"].append(
            np.sum(np.abs(full_target - support), axis=(1, 2, 3, 4, 5)).astype(
                np.float32
            )
        )
        diagnostics["pdp_link_l1"].append(
            np.mean(
                np.sum(np.abs(desired_pdp.reshape(stop - start, 1024, 192) - current_pdp), axis=-1),
                axis=1,
            ).astype(np.float32)
        )
        diagnostics["nmse_to_v567"].append((error / np.maximum(base_energy, TINY)).astype(np.float32))
        diagnostics["point_power_ratio"].append(
            (final_energy / np.maximum(base_energy, TINY)).astype(np.float32)
        )
        diagnostics["final_pdp_target_max_abs"].append(
            np.max(np.abs(final_pdp - desired_pdp.reshape(stop - start, 1024, 192)), axis=(1, 2)).astype(np.float32)
        )
        if stop == ROWS or stop % 50 == 0:
            output.flush()
            print(f"v577 render {stop}/{ROWS}", flush=True)
    output.flush()
    return {
        key: np.concatenate(values).astype(np.float32)
        for key, values in diagnostics.items()
    }


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
    }


def write_package(report: dict[str, object]) -> dict[str, object]:
    dependencies = (
        "generate_v577_dual_metric_physical_response_probe.py",
        "generate_v577_test_physical_responses.py",
        "train_v576_dual_metric_physical_response.py",
        "train_v575_physical_response_basis.py",
        "generate_v575_physical_response_basis.py",
        "project_paths.py",
    )
    for name in dependencies:
        require(PATHS.code / name)
    checkpoints = [CHECKPOINT_DIR / f"main_fold_{fold}.pt" for fold in range(5)]
    for checkpoint in checkpoints:
        require(checkpoint)
    manifest = {
        "method": VERSION,
        "checkpoints": [f"model/{path.name}" for path in checkpoints],
        "response_pairs": list(v575.PAIR_NAMES),
        "base": "v567 online 0.750787",
    }
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        archive.writestr("model/", b"")
        archive.writestr(
            "model/manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
        archive.write(REPORT, "model/v577_probe_report.json")
        for checkpoint in checkpoints:
            archive.write(checkpoint, f"model/{checkpoint.name}")
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
    args = parse_args()
    required = [BASE, PDP_ANCHOR, *PAS_ANCHORS.values()]
    for item in v575_source.PAIRS:
        for seed in v575_source.PAIR_SEEDS:
            required.extend(response_source.response_paths(item, seed).values())
    for path in required:
        require(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("v577 coefficient inference requires CUDA")
    device = torch.device(f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)

    summary = load_summary_matrix()
    pas_coefficients, pdp_coefficients, pas_std, pdp_std = predict_coefficients(
        summary, device
    )
    pas_targets = {
        name: render_response_target(name, PAS_ANCHORS[name], pas_coefficients)
        for name in v575.PAS_NAMES
    }
    pdp_target = render_response_target("pdp", PDP_ANCHOR, pdp_coefficients)
    diagnostics = render_output(pas_targets, pdp_target)
    np.savez_compressed(
        ACTIONS,
        pair_names=np.asarray(v575.PAIR_NAMES),
        pas_coefficients=pas_coefficients,
        pdp_coefficients=pdp_coefficients,
        pas_coefficient_std=pas_std,
        pdp_coefficient_std=pdp_std,
        **diagnostics,
    )

    report: dict[str, object] = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "v576 proved that paired Sionna response directions contain location-specific "
            "PAS/PDP information. Transferring only the predicted response residual from "
            "the v575 anchors onto v567 should retain the online-validated carrier while "
            "testing whether the physical response source migrates to hidden test points."
        ),
        "single_change": (
            "apply the five-fold v576 dual-head response residual to v567 with one PAS "
            "multi-marginal projection followed by one exact linkwise-PDP restoration"
        ),
        "parameters": {
            "base": PATHS.relative(BASE),
            "models": 5,
            "response_pairs": list(v575.PAIR_NAMES),
            "pas_strength": PAS_STRENGTH,
            "pdp_strength": PDP_STRENGTH,
            "log_ratio_clip": LOG_RATIO_CLIP,
            "ipf_iterations": IPF_ITERS,
            "pas_ipf_exponents": {
                "sh": 0.70,
                "psh": 0.42,
                "shue": 0.46,
                "shv": 0.42,
            },
            "projection_order": "PAS full-cell projection then exact linkwise PDP",
            "power_constraint": "preserve every v567 link CIR energy",
        },
        "diagnostics": {
            key: summarize(value) for key, value in diagnostics.items()
        },
        "coefficient_statistics": {
            "pas_abs_mean": float(np.mean(np.abs(pas_coefficients))),
            "pdp_abs_mean": float(np.mean(np.abs(pdp_coefficients))),
            "pas_fold_std_mean": float(np.mean(pas_std)),
            "pdp_fold_std_mean": float(np.mean(pdp_std)),
        },
        "warning": (
            "v576 did not pass its predeclared +0.02 local threshold. This package is "
            "generated only because the user explicitly requested an online probe; the "
            "online score is the final evidence."
        ),
        "outputs": {
            "channel": PATHS.relative(OUTPUT),
            "actions": PATHS.relative(ACTIONS),
            "targets": {
                name: PATHS.relative(path) for name, path in TARGET_PATHS.items()
            },
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
                "archive": report["outputs"]["archive"],
                "diagnostics": report["diagnostics"],
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
