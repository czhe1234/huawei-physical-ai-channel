#!/usr/bin/env python3
"""Fit the official v445 path-residual adapter and project the v209 carrier."""

from __future__ import annotations

import json
import time
import zipfile

import numpy as np
import torch

import analyze_v208_strict_historical_replay as v208
import analyze_v451_official_path_representation_probe as v451
import analyze_v452_v445_representation_over_strong_anchor as v452
import generate_v187_leaked_pas_lifecycle_projection as leak
from generate_v159_frontier_consensus_support_soup import digest
from generate_v171_full_complex_ad_tail_denoising import top_mask
from project_paths import PATHS


VERSION_NUMBER = 453
VERSION = "v453_v445_path_residual_adapter"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OUTPUT = OUT_DIR / "v453_Round1_Test_Channel.npy"
ADAPTER = OUT_DIR / "v453_official_path_residual_adapter.npz"
PATH_ENCODER = OUT_DIR / "v453_path_encoder.pt"
ZIP_OUT = PATHS.round1_versions / f"{VERSION}_candidate_pending.zip"

BASE = PATHS.cache_dir(209) / "v209_Round1_Test_Channel.npy"
TARGET_SOURCE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
GEOMETRY_SOURCE = PATHS.shared_features / "height_geometry_cache.npz"
EMBEDDING_SOURCE = PATHS.cache_dir(451) / "v445_path_embedding.npy"
V445_CHECKPOINT = (
    PATHS.checkpoints / "v445_candidate_path_token_model" / "component_route_best.pt"
)

RIDGE = 10.0
RESIDUAL_STRENGTH = 0.15
BATCH = 8
FLOOR = v451.FLOOR
TINY = 1e-30


def summarize(value: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(value, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def save_pruned_path_encoder() -> dict[str, object]:
    checkpoint = torch.load(V445_CHECKPOINT, map_location="cpu", weights_only=True)
    prefixes = (
        "candidate_encoder.",
        "candidate_position.",
        "candidate_special",
        "candidate_transformer.",
        "candidate_mass_head.",
    )
    state = {
        name: value
        for name, value in checkpoint["model"].items()
        if name.startswith(prefixes)
    }
    if not state:
        raise ValueError("No candidate path encoder tensors found in v445 checkpoint")
    torch.save(
        {
            "state_dict": state,
            "source": PATHS.relative(V445_CHECKPOINT),
            "feature_count": 28,
            "embedding": "special + mass_weighted + valid_mean, 3x256",
        },
        PATH_ENCODER,
    )
    return {
        "tensor_count": int(len(state)),
        "parameter_count": int(sum(value.numel() for value in state.values())),
        "bytes": int(PATH_ENCODER.stat().st_size),
    }


def fit_adapter() -> tuple[np.ndarray, dict[str, object]]:
    target = np.load(TARGET_SOURCE)
    truth_sh = target["sh"].astype(np.float32)
    valid_ids = np.flatnonzero(target["valid"].astype(bool)).astype(np.int64)
    positions = np.vstack(
        [
            np.load(PATHS.round1 / "Round1_Train_Pos.npy"),
            np.load(PATHS.round1 / "Round1_Test_Pos.npy"),
        ]
    ).astype(np.float32)
    geometry = np.load(GEOMETRY_SOURCE)
    blocked = geometry["train_blocked"].astype(np.float32)
    embedding = np.load(EMBEDDING_SOURCE).astype(np.float32)

    train_anchor = v452.knn_anchor(
        valid_ids,
        valid_ids,
        positions[:2000],
        blocked,
        truth_sh,
        exclude_self=True,
    )
    residual = v452.residual_target(truth_sh[valid_ids], train_anchor)
    local_ids = np.arange(len(valid_ids), dtype=np.int64)
    latent, pca = v451.prepare_latent_target(residual, local_ids, VERSION_NUMBER)

    feature = embedding[valid_ids]
    feature_mean = np.mean(feature, axis=0, dtype=np.float64).astype(np.float32)
    feature_std = np.std(feature, axis=0, dtype=np.float64).astype(np.float32)
    feature_std = np.maximum(feature_std, 1e-4)
    standardized = ((feature - feature_mean) / feature_std).astype(np.float32)
    coefficient, intercept = v451.ridge_fit(standardized, latent, RIDGE)

    test_standardized = (
        (embedding[2000:] - feature_mean) / feature_std
    ).astype(np.float32)
    test_latent = test_standardized @ coefficient + intercept
    test_delta = pca.inverse_transform(test_latent).astype(np.float32)
    test_delta = np.clip(test_delta, -6.0, 6.0).reshape(500, 192, 16)

    np.savez_compressed(
        ADAPTER,
        feature_mean=feature_mean,
        feature_std=feature_std,
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        ridge_coefficient=coefficient,
        ridge_intercept=intercept,
        ridge=np.asarray(RIDGE, dtype=np.float32),
        residual_strength=np.asarray(RESIDUAL_STRENGTH, dtype=np.float32),
        valid_ids=valid_ids.astype(np.int32),
    )
    report = {
        "valid_train_count": int(len(valid_ids)),
        "feature_dim": int(feature.shape[1]),
        "latent_dim": int(pca.n_components_),
        "ridge": RIDGE,
        "residual_strength": RESIDUAL_STRENGTH,
        "train_anchor_joint_cosine": float(
            np.mean(v451.v450.row_cosine(train_anchor, truth_sh[valid_ids]))
        ),
        "test_delta_l2": summarize(
            np.linalg.norm(test_delta.reshape(500, -1), axis=1)
        ),
        "test_scaled_log_ratio_abs": summarize(
            np.mean(np.abs(RESIDUAL_STRENGTH * test_delta), axis=(1, 2))
        ),
        "adapter_bytes": int(ADAPTER.stat().st_size),
    }
    return test_delta, report


def project_carrier(test_delta: np.ndarray) -> dict[str, object]:
    source = np.load(BASE, mmap_mode="r")
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=source.shape
    )
    support_l1 = []
    sh_l1 = []
    gain_min = []
    gain_p10 = []
    gain_p90 = []
    gain_max = []
    power_factor = []
    for start in range(0, len(source), BATCH):
        end = min(start + BATCH, len(source))
        h0 = np.asarray(source[start:end], dtype=np.complex64)
        spec0 = leak.leaked_forward(h0)
        power0 = leak.leaked_cell_power(spec0)
        total0 = np.maximum(
            np.sum(power0, axis=(1, 2, 3, 4, 5), dtype=np.float64), TINY
        ).astype(np.float32)
        support0 = power0 / total0[:, None, None, None, None, None]
        current_sh = np.sum(support0, axis=(1, 4, 5), dtype=np.float64).astype(
            np.float32
        )
        ratio = np.exp(
            np.clip(
                RESIDUAL_STRENGTH * test_delta[start:end], -1.8, 1.8
            )
        ).astype(np.float32)
        desired_sh = current_sh * ratio
        desired_sh /= np.maximum(
            np.sum(desired_sh, axis=(1, 2), keepdims=True, dtype=np.float64),
            TINY,
        )
        cell_ratio = desired_sh / np.maximum(current_sh, FLOOR)
        gain = np.sqrt(cell_ratio)[:, None, :, :, None, None].astype(np.float32)
        gain = np.broadcast_to(gain, support0.shape).copy()
        gain = np.clip(gain, 0.32, 3.20)
        flat = support0.reshape(end - start, -1)
        strong128 = top_mask(flat, 128).reshape(support0.shape)
        strong1024 = top_mask(flat, 1024).reshape(support0.shape)
        strong8192 = top_mask(flat, 8192).reshape(support0.shape)
        gain[strong8192] = np.clip(gain[strong8192], 0.55, 2.00)
        gain[strong1024] = np.clip(gain[strong1024], 0.70, 1.55)
        gain[strong128] = np.clip(gain[strong128], 0.82, 1.25)

        spec1 = spec0 * gain
        h1 = leak.leaked_inverse(spec1)
        h_power0 = np.maximum(
            np.sum(np.abs(h0) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY
        )
        h_power1 = np.maximum(
            np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64), TINY
        )
        h1 *= np.sqrt(h_power0 / h_power1)[:, None, None, None].astype(np.float32)
        if not np.isfinite(h1).all():
            raise FloatingPointError(f"Non-finite v453 projection {start}:{end}")
        output[start:end] = h1.astype(np.complex64)
        output.flush()

        power1 = leak.leaked_cell_power(leak.leaked_forward(h1))
        support1 = power1 / np.maximum(
            np.sum(power1, axis=(1, 2, 3, 4, 5), keepdims=True, dtype=np.float64),
            TINY,
        )
        sh1 = np.sum(support1, axis=(1, 4, 5), dtype=np.float64)
        support_l1.append(
            np.sum(np.abs(support1 - support0), axis=(1, 2, 3, 4, 5)).astype(
                np.float32
            )
        )
        sh_l1.append(
            np.sum(np.abs(sh1 - current_sh), axis=(1, 2)).astype(np.float32)
        )
        flat_gain = gain.reshape(end - start, -1)
        gain_min.append(np.min(flat_gain, axis=1).astype(np.float32))
        gain_p10.append(np.quantile(flat_gain, 0.10, axis=1).astype(np.float32))
        gain_p90.append(np.quantile(flat_gain, 0.90, axis=1).astype(np.float32))
        gain_max.append(np.max(flat_gain, axis=1).astype(np.float32))
        power_factor.append(
            (
                np.sum(np.abs(h1) ** 2, axis=(1, 2, 3), dtype=np.float64)
                / h_power0
            ).astype(np.float32)
        )
        print(f"v453 projection {end}/{len(source)}", flush=True)
    output.flush()
    del output
    return {
        "support_l1": summarize(np.concatenate(support_l1)),
        "joint_sh_l1": summarize(np.concatenate(sh_l1)),
        "gain_min": summarize(np.concatenate(gain_min)),
        "gain_p10": summarize(np.concatenate(gain_p10)),
        "gain_p90": summarize(np.concatenate(gain_p90)),
        "gain_max": summarize(np.concatenate(gain_max)),
        "power_factor": summarize(np.concatenate(power_factor)),
    }


def validate_output() -> dict[str, object]:
    output = np.load(OUTPUT, mmap_mode="r")
    if output.shape != (500, 256, 4, 192) or output.dtype != np.complex64:
        raise ValueError(f"Unexpected output {output.shape}/{output.dtype}")
    finite = all(
        np.isfinite(np.asarray(output[start : start + 8])).all()
        for start in range(0, len(output), 8)
    )
    if not finite:
        raise FloatingPointError("v453 output is non-finite")
    return {"shape": list(output.shape), "dtype": str(output.dtype), "finite": True}


def package() -> dict[str, object]:
    source_files = (
        "generate_v453_v445_path_residual_adapter.py",
        "analyze_v452_v445_representation_over_strong_anchor.py",
        "analyze_v451_official_path_representation_probe.py",
        "analyze_v450_official_geometry_token_transfer.py",
        "generate_v218_official_multispectrum_physics_ai.py",
        "generate_v215_pointcloud_physics_neural_field.py",
        "generate_v187_leaked_pas_lifecycle_projection.py",
        "generate_v171_full_complex_ad_tail_denoising.py",
        "generate_v159_frontier_consensus_support_soup.py",
        "analyze_v208_strict_historical_replay.py",
        "project_paths.py",
    )
    model_files = (ADAPTER, PATH_ENCODER)
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(OUTPUT, "Round1_Test_Channel.npy")
        for name in source_files:
            path = PATHS.code / name
            if not path.is_file():
                raise FileNotFoundError(path)
            archive.write(path, name)
        for path in model_files:
            archive.write(path, path.name)
    with zipfile.ZipFile(ZIP_OUT) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise ValueError(f"ZIP CRC failed for {bad}")
    nested = [name for name in members if "/" in name.strip("/")]
    if nested:
        raise ValueError(f"ZIP must be flat: {nested}")
    return {
        "zip_crc": True,
        "members": members,
        "bytes": int(ZIP_OUT.stat().st_size),
        "sha256": digest(ZIP_OUT, "sha256"),
        "md5": digest(ZIP_OUT, "md5"),
    }


def main() -> None:
    required = (
        BASE,
        TARGET_SOURCE,
        GEOMETRY_SOURCE,
        EMBEDDING_SOURCE,
        V445_CHECKPOINT,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    test_delta, adapter_report = fit_adapter()
    encoder_report = save_pruned_path_encoder()
    projection_report = project_carrier(test_delta)
    output_check = validate_output()
    local_base = v208.score_prediction(BASE)
    local_model = v208.score_prediction(OUTPUT)
    local_delta = {
        key: float(local_model[key] - local_base[key])
        for key in (
            "score_traceback_leaked",
            "leaked_sh",
            "pdp",
            "nmse_score",
        )
    }
    package_report = package()
    report = {
        "method": VERSION,
        "status": "candidate_pending",
        "primary_hypothesis": (
            "The v445 path representation contains a geometry-conditioned joint support "
            "residual that complements the pre-Sionna v209 carrier after official-H calibration."
        ),
        "official_oof_evidence": {
            "source": "v452",
            "strong_anchor_weighted_gain": 0.0022984266281127486,
            "positive_folds": 5,
            "random_encoder_gain": 0.0005206131935119318,
            "shuffled_encoder_gain": -0.00004029035568242634,
        },
        "compliance_boundary": {
            "uses": [
                "official train H targets and official point-cloud geometry",
                "official train/test coordinates",
                "AI path encoder pretrained on self-generated Sionna data",
                "official-train PCA/ridge residual adapter",
                "pre-Sionna v209 complex carrier",
            ],
            "does_not_use": [
                "official test H",
                "Sionna CFR or path coefficients at official test coordinates",
                "v212/v426/v436/v449 prediction teachers",
                "hidden online labels for training or selection",
            ],
            "eligibility_note": (
                "Synthetic pretraining is used only through an AI encoder; final inputs are "
                "official geometry. Competition-rule confirmation for auxiliary synthetic "
                "pretraining remains required before treating this as the final audited method."
            ),
        },
        "adapter": adapter_report,
        "pruned_path_encoder": encoder_report,
        "projection": projection_report,
        "local_diagnostic": {
            "base_v209": local_base,
            "v453": local_model,
            "delta": local_delta,
            "selection_policy": (
                "not used for model fitting; reported only after official OOF configuration froze"
            ),
        },
        "integrity": {**output_check, **package_report},
        "seconds": float(time.time() - started),
        "outputs": {
            "prediction": PATHS.relative(OUTPUT),
            "adapter": PATHS.relative(ADAPTER),
            "path_encoder": PATHS.relative(PATH_ENCODER),
            "zip": PATHS.relative(ZIP_OUT),
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
