#!/usr/bin/env python3
"""Audit transfer of v445 path-token learning to official point-cloud geometry."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

import build_v439_component_path_supervision as component_builder
import train_v439_component_route_long_model as v439
import train_v445_candidate_path_token_model as v445
from project_paths import PATHS


VERSION_NUMBER = 450
VERSION = "v450_official_geometry_token_transfer"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"

PATH_SOURCE = PATHS.cache_dir(291) / "v291_highdepth_pointcloud_paths.npz"
ROUTE_SOURCE = PATHS.shared_features / "v415_route_raster_features.npz"
TARGET_SOURCE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
COMPONENT_GEOMETRY_SOURCE = PATHS.cache_dir(439) / "component_geometry.npy"
V443_CHECKPOINT = (
    PATHS.checkpoints / "v443_dense_component_route_model" / "component_route_best.pt"
)
V445_CHECKPOINT = (
    PATHS.checkpoints / "v445_candidate_path_token_model" / "component_route_best.pt"
)
SYNTHETIC_TOKEN_SOURCE = PATHS.cache_dir(445)

BS = np.asarray([50.0, 0.0, 25.0], dtype=np.float32)
FEATURE_COUNT = 28
MAX_PATHS = 128
DELAY_SCALE = 192.0 * 30_000.0
LIGHT_SPEED = 299_792_458.0
FEATURE_NAMES = (
    "tau_us_over_8",
    "delay_bin_over_48",
    "length_over_600",
    "excess_length_over_500",
    "depth_over_16",
    "is_los",
    "has_specular",
    "has_diffuse",
    "has_refraction",
    "has_diffraction",
    "sin_theta_t",
    "cos_theta_t",
    "sin_phi_t",
    "cos_phi_t",
    "sin_theta_r",
    "cos_theta_r",
    "sin_phi_r",
    "cos_phi_r",
    "first_x_global",
    "first_y_global",
    "first_z_global",
    "last_x_global",
    "last_y_global",
    "last_z_global",
    "first_x_ue_relative",
    "first_y_ue_relative",
    "last_x_ue_relative",
    "last_y_ue_relative",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--synthetic-sample", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=45_001)
    parser.add_argument("--build-only", action="store_true")
    return parser.parse_args()


def cache_path(name: str) -> object:
    return OUT_DIR / f"{name}.npy"


def position_features(positions: np.ndarray) -> np.ndarray:
    xy = np.asarray(positions[:, :2], dtype=np.float32)
    delta = xy - BS[None, :2]
    distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-4)
    angle = np.arctan2(delta[:, 1], delta[:, 0])
    base = np.column_stack(
        [
            (xy[:, 0] - 150.0) / 120.0,
            (xy[:, 1] + 50.0) / 160.0,
            delta[:, 0] / 250.0,
            delta[:, 1] / 350.0,
            distance / 350.0,
            np.log1p(distance) / 6.0,
            np.sin(angle),
            np.cos(angle),
        ]
    )
    harmonics: list[np.ndarray] = []
    for scale in (5.0, 10.0, 20.0, 40.0, 80.0, 160.0):
        phase_x = 2.0 * np.pi * xy[:, 0] / scale
        phase_y = 2.0 * np.pi * xy[:, 1] / scale
        harmonics.extend(
            [np.sin(phase_x), np.cos(phase_x), np.sin(phase_y), np.cos(phase_y)]
        )
    return np.column_stack([base, *harmonics]).astype(np.float32)


def first_last_vertices(
    vertices: np.ndarray, interactions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    active = interactions != 0
    count = np.sum(active, axis=2)
    first_index = np.argmax(active, axis=2)
    last_index = np.maximum(count - 1, 0)
    row = np.arange(len(vertices))[:, None]
    path = np.arange(vertices.shape[1])[None, :]
    first = vertices[row, path, first_index].copy()
    last = vertices[row, path, last_index].copy()
    missing = count == 0
    first[missing] = BS
    last[missing] = BS
    return first, last


def geometry_order(data: np.lib.npyio.NpzFile, prefix: str, row: int) -> np.ndarray:
    valid = np.flatnonzero(data[f"{prefix}_valid"][row])
    if not len(valid):
        return valid
    local = np.lexsort(
        (
            data[f"{prefix}_phi_r"][row, valid].astype(np.float32),
            data[f"{prefix}_phi_t"][row, valid].astype(np.float32),
            data[f"{prefix}_interaction_count"][row, valid].astype(np.int16),
            data[f"{prefix}_tau"][row, valid].astype(np.float64),
        )
    )
    return valid[local]


def build_token_split(
    data: np.lib.npyio.NpzFile, prefix: str, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    count = len(positions)
    feature = np.zeros((count, MAX_PATHS, FEATURE_COUNT), dtype=np.float16)
    valid_out = np.zeros((count, MAX_PATHS), dtype=bool)
    interactions = data[f"{prefix}_interactions"].astype(np.int8)
    vertices = data[f"{prefix}_vertices"].astype(np.float32)
    first, last = first_last_vertices(vertices, interactions)
    for row in range(count):
        order = geometry_order(data, prefix, row)
        path_count = min(len(order), MAX_PATHS)
        if path_count == 0:
            continue
        order = order[:path_count]
        tau = data[f"{prefix}_tau"][row, order].astype(np.float32)
        delay = tau * DELAY_SCALE
        length = tau * LIGHT_SPEED
        ue = np.asarray(positions[row], dtype=np.float32)
        direct = float(np.linalg.norm(ue - BS))
        depth = data[f"{prefix}_interaction_count"][row, order].astype(np.float32)
        mask = data[f"{prefix}_interaction_mask"][row, order].astype(np.int16)
        angles = [
            data[f"{prefix}_{name}"][row, order].astype(np.float32)
            for name in ("theta_t", "phi_t", "theta_r", "phi_r")
        ]
        angle_feature = np.column_stack(
            [value for angle in angles for value in (np.sin(angle), np.cos(angle))]
        )
        first_value = first[row, order]
        last_value = last[row, order]
        global_vertex = np.column_stack(
            [
                (first_value[:, 0] - 150.0) / 150.0,
                (first_value[:, 1] + 50.0) / 200.0,
                first_value[:, 2] / 50.0,
                (last_value[:, 0] - 150.0) / 150.0,
                (last_value[:, 1] + 50.0) / 200.0,
                last_value[:, 2] / 50.0,
            ]
        )
        relative_vertex = np.column_stack(
            [
                (first_value[:, 0] - ue[0]) / 300.0,
                (first_value[:, 1] - ue[1]) / 300.0,
                (last_value[:, 0] - ue[0]) / 300.0,
                (last_value[:, 1] - ue[1]) / 300.0,
            ]
        )
        current = np.column_stack(
            [
                tau * 1e6 / 8.0,
                delay / 48.0,
                length / 600.0,
                np.maximum(length - direct, 0.0) / 500.0,
                depth / 16.0,
                (depth == 0).astype(np.float32),
                *[((mask & bit) != 0).astype(np.float32) for bit in (1, 2, 4, 8)],
                angle_feature,
                global_vertex,
                relative_vertex,
            ]
        ).astype(np.float32)
        if current.shape != (path_count, FEATURE_COUNT):
            raise ValueError(f"Unexpected {prefix} token shape {current.shape}")
        if not np.isfinite(current).all():
            raise FloatingPointError(f"Non-finite {prefix} token at row {row}")
        feature[row, :path_count] = np.clip(current, -4.0, 4.0).astype(np.float16)
        valid_out[row, :path_count] = True
    return feature, valid_out


def load_or_build_inputs() -> dict[str, np.ndarray]:
    names = (
        "positions",
        "route_raster",
        "component_features",
        "path_candidate_feature",
        "path_candidate_valid",
        "split",
    )
    if all(cache_path(name).is_file() for name in names):
        return {
            name: np.load(cache_path(name), mmap_mode="r")
            for name in names
        }

    path_data = np.load(PATH_SOURCE)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    positions = np.vstack([train_pos, test_pos]).astype(np.float32)
    route_data = np.load(ROUTE_SOURCE)
    route = np.concatenate([route_data["train"], route_data["test"]], axis=0)
    geometry = np.load(COMPONENT_GEOMETRY_SOURCE).astype(np.float32)
    component_builder.COUNT = len(positions)
    component = component_builder.build_component_features(positions, geometry)
    train_feature, train_valid = build_token_split(path_data, "train", train_pos)
    test_feature, test_valid = build_token_split(path_data, "test", test_pos)
    feature = np.concatenate([train_feature, test_feature], axis=0)
    valid = np.concatenate([train_valid, test_valid], axis=0)
    split = np.concatenate(
        [np.zeros(len(train_pos), dtype=np.int8), np.ones(len(test_pos), dtype=np.int8)]
    )
    values = {
        "positions": positions,
        "route_raster": route,
        "component_features": component,
        "path_candidate_feature": feature,
        "path_candidate_valid": valid,
        "split": split,
    }
    for name, value in values.items():
        np.save(cache_path(name), value)
    return {
        name: np.load(cache_path(name), mmap_mode="r")
        for name in names
    }


def summary(value: np.ndarray) -> dict[str, float | int]:
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


def domain_report(
    official_feature: np.ndarray,
    official_valid: np.ndarray,
    synthetic_sample: int,
    seed: int,
) -> dict[str, object]:
    synthetic_feature = np.load(
        SYNTHETIC_TOKEN_SOURCE / "path_candidate_feature.npy", mmap_mode="r"
    )
    synthetic_valid = np.load(
        SYNTHETIC_TOKEN_SOURCE / "path_candidate_valid.npy", mmap_mode="r"
    )
    rng = np.random.default_rng(seed)
    sample_ids = np.sort(
        rng.choice(
            len(synthetic_feature),
            min(synthetic_sample, len(synthetic_feature)),
            replace=False,
        )
    )
    official_flat = np.asarray(official_feature[official_valid], dtype=np.float32)
    sample_feature = np.asarray(synthetic_feature[sample_ids], dtype=np.float32)
    sample_valid = np.asarray(synthetic_valid[sample_ids], dtype=bool)
    synthetic_flat = sample_feature[sample_valid]
    official_mean = np.mean(official_flat, axis=0)
    synthetic_mean = np.mean(synthetic_flat, axis=0)
    synthetic_std = np.maximum(np.std(synthetic_flat, axis=0), 1e-4)
    standardized_shift = np.abs(official_mean - synthetic_mean) / synthetic_std
    rows = []
    for index, name in enumerate(FEATURE_NAMES):
        rows.append(
            {
                "feature": name,
                "official_mean": float(official_mean[index]),
                "synthetic_mean": float(synthetic_mean[index]),
                "synthetic_std": float(synthetic_std[index]),
                "standardized_mean_shift": float(standardized_shift[index]),
            }
        )
    rows.sort(key=lambda row: row["standardized_mean_shift"], reverse=True)
    official_count = np.sum(official_valid, axis=1)
    synthetic_count = np.sum(sample_valid, axis=1)
    return {
        "official_valid_token_count": int(len(official_flat)),
        "synthetic_sample_valid_token_count": int(len(synthetic_flat)),
        "official_path_count": summary(official_count),
        "synthetic_path_count": summary(synthetic_count),
        "zero_path_count": {
            "official": int(np.sum(official_count == 0)),
            "synthetic_sample": int(np.sum(synthetic_count == 0)),
        },
        "standardized_mean_shift": summary(standardized_shift),
        "largest_feature_shifts": rows[:12],
    }


def row_cosine(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = prediction.reshape(len(prediction), -1).astype(np.float64)
    truth = target.reshape(len(target), -1).astype(np.float64)
    numerator = np.sum(pred * truth, axis=1)
    denominator = np.linalg.norm(pred, axis=1) * np.linalg.norm(truth, axis=1)
    return (numerator / np.maximum(denominator, 1e-30)).astype(np.float32)


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    inputs: dict[str, np.ndarray],
    ids: np.ndarray,
    device: torch.device,
    batch_size: int,
    *,
    geometry_ids: np.ndarray | None = None,
    path_ids: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    joint_parts: list[np.ndarray] = []
    power_parts: list[np.ndarray] = []
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        batch_ids = ids[start:end]
        geometry_source = batch_ids if geometry_ids is None else geometry_ids[start:end]
        path_source = geometry_source if path_ids is None else path_ids[start:end]
        route = torch.from_numpy(
            np.asarray(inputs["route_raster"][geometry_source], dtype=np.float32)
        ).to(device)
        position = torch.from_numpy(
            position_features(np.asarray(inputs["positions"][batch_ids]))
        ).to(device)
        component = torch.from_numpy(
            np.asarray(inputs["component_features"][geometry_source], dtype=np.float32)
        ).to(device)
        optional = {}
        if getattr(model, "uses_path_candidates", False):
            optional = {
                "path_candidate_feature": torch.from_numpy(
                    np.asarray(
                        inputs["path_candidate_feature"][path_source], dtype=np.float32
                    )
                ).to(device),
                "path_candidate_valid": torch.from_numpy(
                    np.asarray(inputs["path_candidate_valid"][path_source], dtype=bool)
                ).to(device),
            }
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(route, position, component, **optional)
        joint_parts.append(
            F.softmax(output["joint_logits"].flatten(1), dim=1)
            .reshape(-1, 192, 16)
            .float()
            .cpu()
            .numpy()
        )
        power_parts.append(output["power"].float().cpu().numpy())
    joint = np.concatenate(joint_parts)
    reverse = np.concatenate([[0], np.arange(191, 0, -1)])
    return {
        "joint_sh": joint,
        "pdp": np.sum(joint[:, reverse], axis=2).astype(np.float32),
        "power_standardized": np.concatenate(power_parts).astype(np.float32),
    }


def metrics(
    prediction: dict[str, np.ndarray],
    target: np.lib.npyio.NpzFile,
    ids: np.ndarray,
    power_mean: float,
    power_std: float,
) -> dict[str, float | int]:
    valid = target["valid"][ids].astype(bool)
    joint_cosine = row_cosine(
        prediction["joint_sh"][valid], target["sh"][ids[valid]]
    )
    pdp_cosine = row_cosine(prediction["pdp"][valid], target["d"][ids[valid]])
    predicted_power = prediction["power_standardized"] * power_std + power_mean
    power_error = np.abs(predicted_power[valid] - target["log_power"][ids[valid]])
    return {
        "score_60_joint_40_pdp": float(
            0.60 * np.mean(joint_cosine) + 0.40 * np.mean(pdp_cosine)
        ),
        "joint_sh_cosine": float(np.mean(joint_cosine)),
        "pdp_cosine": float(np.mean(pdp_cosine)),
        "power_log_mae": float(np.mean(power_error)),
        "count": int(len(ids)),
        "valid_count": int(np.sum(valid)),
    }


def evaluate_models(
    inputs: dict[str, np.ndarray], args: argparse.Namespace
) -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"v450 requires exactly one visible CUDA GPU, got {torch.cuda.device_count()}"
        )
    device = torch.device("cuda")
    target = np.load(TARGET_SOURCE)
    ids = np.flatnonzero(np.asarray(inputs["split"]) == 0).astype(np.int64)
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(ids)

    checkpoint443 = torch.load(V443_CHECKPOINT, map_location="cpu", weights_only=True)
    base = v439.ComponentRouteTransformer(coordinate_dropout=0.0)
    base.load_state_dict(checkpoint443["model"], strict=True)
    base = base.to(device)
    base_prediction = predict(base, inputs, ids, device, args.batch_size)
    base_metrics = metrics(
        base_prediction,
        target,
        ids,
        float(checkpoint443["power_mean"]),
        float(checkpoint443["power_std"]),
    )
    base_shuffled = predict(
        base,
        inputs,
        ids,
        device,
        args.batch_size,
        geometry_ids=permutation,
    )
    base_shuffled_metrics = metrics(
        base_shuffled,
        target,
        ids,
        float(checkpoint443["power_mean"]),
        float(checkpoint443["power_std"]),
    )
    del base, base_prediction, base_shuffled
    torch.cuda.empty_cache()

    checkpoint445 = torch.load(V445_CHECKPOINT, map_location="cpu", weights_only=True)
    candidate = v445.CandidatePathTokenModel(coordinate_dropout=0.0)
    candidate.load_state_dict(checkpoint445["model"], strict=True)
    candidate = candidate.to(device)
    candidate_prediction = predict(candidate, inputs, ids, device, args.batch_size)
    candidate_metrics = metrics(
        candidate_prediction,
        target,
        ids,
        float(checkpoint445["power_mean"]),
        float(checkpoint445["power_std"]),
    )
    path_shuffled = predict(
        candidate,
        inputs,
        ids,
        device,
        args.batch_size,
        path_ids=permutation,
    )
    path_shuffled_metrics = metrics(
        path_shuffled,
        target,
        ids,
        float(checkpoint445["power_mean"]),
        float(checkpoint445["power_std"]),
    )
    all_shuffled = predict(
        candidate,
        inputs,
        ids,
        device,
        args.batch_size,
        geometry_ids=permutation,
    )
    all_shuffled_metrics = metrics(
        all_shuffled,
        target,
        ids,
        float(checkpoint445["power_mean"]),
        float(checkpoint445["power_std"]),
    )
    result = {
        "v443_official_geometry": base_metrics,
        "v443_shuffled_all_geometry": base_shuffled_metrics,
        "v445_official_geometry_tokens": candidate_metrics,
        "v445_shuffled_path_tokens_only": path_shuffled_metrics,
        "v445_shuffled_all_geometry": all_shuffled_metrics,
        "deltas": {
            "v445_minus_v443": {
                key: float(candidate_metrics[key] - base_metrics[key])
                for key in (
                    "score_60_joint_40_pdp",
                    "joint_sh_cosine",
                    "pdp_cosine",
                )
            },
            "v445_minus_path_shuffled": {
                key: float(candidate_metrics[key] - path_shuffled_metrics[key])
                for key in (
                    "score_60_joint_40_pdp",
                    "joint_sh_cosine",
                    "pdp_cosine",
                )
            },
        },
    }
    del candidate, candidate_prediction, path_shuffled, all_shuffled
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    required = (
        PATH_SOURCE,
        ROUTE_SOURCE,
        TARGET_SOURCE,
        COMPONENT_GEOMETRY_SOURCE,
        V443_CHECKPOINT,
        V445_CHECKPOINT,
        SYNTHETIC_TOKEN_SOURCE / "path_candidate_feature.npy",
        SYNTHETIC_TOKEN_SOURCE / "path_candidate_valid.npy",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    inputs = load_or_build_inputs()
    domain = domain_report(
        np.asarray(inputs["path_candidate_feature"]),
        np.asarray(inputs["path_candidate_valid"]),
        args.synthetic_sample,
        args.seed,
    )
    evaluation = None if args.build_only else evaluate_models(inputs, args)
    transfer_passed = bool(
        evaluation is not None
        and evaluation["deltas"]["v445_minus_v443"]["score_60_joint_40_pdp"] > 0.0
        and evaluation["deltas"]["v445_minus_path_shuffled"][
            "score_60_joint_40_pdp"
        ]
        > 0.0
    )
    report = {
        "method": VERSION,
        "status": "zero_shot_transfer_evaluated" if evaluation else "inputs_ready",
        "primary_hypothesis": (
            "The power-free path existence, delay, angle, interaction, and vertex rule "
            "learned by v445 transfers to candidates reconstructed only from the official "
            "point cloud and official coordinates."
        ),
        "compliance_boundary": {
            "uses": [
                "official Round1_Map.ply derived route/component geometry",
                "official train/test coordinates",
                "official-point-cloud geometry-only PathSolver tokens",
                "official train H-derived denoised joint S-H and PDP for evaluation",
                "v443/v445 pretrained model weights for architecture transfer research",
            ],
            "does_not_use": [
                "Sionna CFR or path coefficients at official test coordinates",
                "v436/v449 predictions or teacher labels",
                "hidden test H or online score selection",
                "path power in model inputs or candidate ordering",
            ],
        },
        "parameters": {
            "batch_size": int(args.batch_size),
            "synthetic_domain_sample": int(args.synthetic_sample),
            "seed": int(args.seed),
            "single_visible_gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() and torch.cuda.device_count() == 1
                else None
            ),
        },
        "official_input_validation": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "finite": bool(
                    np.isfinite(value).all()
                    if value.dtype != np.bool_
                    else True
                ),
            }
            for name, value in inputs.items()
        },
        "token_domain": domain,
        "zero_shot": evaluation,
        "decision": {
            "criterion": (
                "real official geometry must beat both frozen v443 and path-token shuffle"
            ),
            "passed": transfer_passed,
            "next_step": (
                "run strict five-fold official-H adapter comparison"
                if transfer_passed
                else "stop full training and diagnose token-domain mismatch"
            ),
        },
        "seconds": float(time.time() - started),
        "outputs": {
            "cache": PATHS.relative(OUT_DIR),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
