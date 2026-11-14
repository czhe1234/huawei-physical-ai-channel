#!/usr/bin/env python3
"""Train a strict point selector between sequence and paired-control endpoints."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor

import analyze_e86_surface_endpoint_v11_blend as e86
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from build_geodesic_corridor_features import CACHE as GEODESIC_CACHE
from data import load_positions
from project_paths import PATHS
from run_rbf_spectral_field import statistics
from run_v6_regime_component_lifecycle import REGIME_CACHE
from run_v9_joint_sionna_migration import point_metrics
import train_surface_interaction_energy_model as e85


VERSION = "Round2-E97-surface-sequence-reversion-selector"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "surface_sequence_reversion_selector"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_surface_sequence_reversion_selector"
SELECTIONS = ARTIFACT_DIR / "strict_oof_reversion_selections.npz"
REPORT = ANALYSIS_DIR / "surface_sequence_reversion_selector.json"
VISIBILITY = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_geometry_visibility_features.npz"
)
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--trees", type=int, default=640)
    parser.add_argument("--min-leaf", type=int, default=8)
    return parser.parse_args()


def cosine_feature(
    left: torch.Tensor, right: torch.Tensor, dims: tuple[int, ...]
) -> torch.Tensor:
    numerator = (left * right).sum(dim=dims, dtype=torch.float64)
    denominator = torch.sqrt(
        left.square().sum(dim=dims, dtype=torch.float64)
        * right.square().sum(dim=dims, dtype=torch.float64)
    )
    return (numerator / torch.where(denominator == 0.0, 1e-12, denominator)).float()


def reduce_group(value: torch.Tensor, dim: int, prefix: str) -> tuple[np.ndarray, list[str]]:
    blocks = (
        value.mean(dim=dim),
        value.std(dim=dim),
        value.amin(dim=dim),
        value.amax(dim=dim),
    )
    output = torch.cat(blocks, dim=1).cpu().numpy().astype(np.float32)
    names = []
    width = blocks[0].shape[1]
    for statistic in ("mean", "std", "min", "max"):
        names.extend(f"{prefix}_{statistic}_{index}" for index in range(width))
    return output, names


def spectral_point_features(
    base_pas: torch.Tensor,
    base_pdp: torch.Tensor,
    sequence_pas: torch.Tensor,
    sequence_pdp: torch.Tensor,
    control_pas: torch.Tensor,
    control_pdp: torch.Tensor,
) -> tuple[np.ndarray, list[str]]:
    features = []
    names = []
    pas_pairs = (
        (sequence_pas, control_pas, "pas_sequence_control"),
        (base_pas, sequence_pas, "pas_base_sequence"),
        (base_pas, control_pas, "pas_base_control"),
    )
    for left, right, name in pas_pairs:
        cosine = cosine_feature(left, right, (1, 2, 3))
        block, block_names = reduce_group(cosine, 2, name)
        features.append(block)
        names.extend(block_names)
        distance = (
            torch.sqrt(left.clamp_min(0.0)) - torch.sqrt(right.clamp_min(0.0))
        ).square().sum(dim=(1, 2, 3))
        block, block_names = reduce_group(distance, 2, f"{name}_root_l2")
        features.append(block)
        names.extend(block_names)
    pdp_pairs = (
        (sequence_pdp, control_pdp, "pdp_sequence_control"),
        (base_pdp, sequence_pdp, "pdp_base_sequence"),
        (base_pdp, control_pdp, "pdp_base_control"),
    )
    for left, right, name in pdp_pairs:
        cosine = cosine_feature(left, right, (3,)).reshape(len(left), 256, 4)
        block, block_names = reduce_group(cosine, 1, name)
        features.append(block)
        names.extend(block_names)
        distance = (
            torch.sqrt(left.clamp_min(0.0)) - torch.sqrt(right.clamp_min(0.0))
        ).square().sum(dim=3).reshape(len(left), 256, 4)
        block, block_names = reduce_group(distance, 1, f"{name}_root_l2")
        features.append(block)
        names.extend(block_names)
    return np.concatenate(features, axis=1), names


def weighted_moments(
    value: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    normalized = weight / np.maximum(weight.sum(axis=1, keepdims=True), TINY)
    mean = (value * normalized[..., None]).sum(axis=1)
    variance = (
        np.square(value - mean[:, None]) * normalized[..., None]
    ).sum(axis=1)
    return mean.astype(np.float32), np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)


def path_point_features(query_ids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    geometry = np.load(
        e85.ARTIFACT_DIR / "strict_oof_geometry_sequence_path_endpoint.npz",
        allow_pickle=False,
    )
    control = np.load(
        e85.ARTIFACT_DIR / "strict_oof_no_sequence_path_endpoint.npz",
        allow_pickle=False,
    )
    if not np.array_equal(geometry["query_ids"].astype(np.int64), query_ids):
        raise ValueError("geometry endpoint query ids changed")
    if not np.array_equal(control["query_ids"].astype(np.int64), query_ids):
        raise ValueError("control endpoint query ids changed")
    mass_g = geometry["mass"].astype(np.float32)
    mass_c = control["mass"].astype(np.float32)
    coordinate_g = geometry["coordinate"].astype(np.float32)
    coordinate_c = control["coordinate"].astype(np.float32)
    port_g = geometry["port"].astype(np.float32).reshape(len(query_ids), 128, 8)
    port_c = control["port"].astype(np.float32).reshape(len(query_ids), 128, 8)
    blocks = []
    names = []
    for mass, label in ((mass_g, "sequence"), (mass_c, "control")):
        summary = np.stack(
            (
                mass.max(axis=1),
                np.sort(mass, axis=1)[:, -8:].sum(axis=1),
                -(mass * np.log(np.maximum(mass, 1e-12))).sum(axis=1),
                1.0 / np.maximum(np.square(mass).sum(axis=1), TINY),
            ),
            axis=1,
        ).astype(np.float32)
        blocks.append(summary)
        names.extend(
            (
                f"{label}_mass_max",
                f"{label}_mass_top8",
                f"{label}_mass_entropy",
                f"{label}_mass_effective_count",
            )
        )
    mass_pair = np.stack(
        (
            np.abs(mass_g - mass_c).sum(axis=1),
            (np.sqrt(mass_g) * np.sqrt(mass_c)).sum(axis=1),
        ),
        axis=1,
    ).astype(np.float32)
    blocks.append(mass_pair)
    names.extend(("mass_pair_l1", "mass_pair_affinity"))
    for coordinate, mass, label in (
        (coordinate_g, mass_g, "sequence"),
        (coordinate_c, mass_c, "control"),
    ):
        mean, std = weighted_moments(coordinate, mass)
        blocks.extend((mean, std))
        names.extend(f"{label}_coordinate_mean_{axis}" for axis in "hvd")
        names.extend(f"{label}_coordinate_std_{axis}" for axis in "hvd")
    coordinate_delta = np.concatenate(
        weighted_moments(np.abs(coordinate_g - coordinate_c), 0.5 * (mass_g + mass_c)),
        axis=1,
    )
    blocks.append(coordinate_delta)
    names.extend(f"coordinate_pair_abs_mean_{axis}" for axis in "hvd")
    names.extend(f"coordinate_pair_abs_std_{axis}" for axis in "hvd")
    for port, mass, label in (
        (port_g, mass_g, "sequence"),
        (port_c, mass_c, "control"),
    ):
        mean, std = weighted_moments(port, mass)
        blocks.extend((mean, std))
        names.extend(f"{label}_port_mean_{index}" for index in range(8))
        names.extend(f"{label}_port_std_{index}" for index in range(8))
    mask = np.asarray(np.load(e85.PATH_CACHE / "mask.npy", mmap_mode="r")[query_ids], dtype=bool)
    path = np.asarray(np.load(e85.PATH_FEATURES, mmap_mode="r")[query_ids], dtype=np.float32)
    valid = mask[..., None].astype(np.float32)
    count = valid.sum(axis=1).clip(min=1.0)
    path_mean = (path * valid).sum(axis=1) / count
    path_variance = (np.square(path - path_mean[:, None]) * valid).sum(axis=1) / count
    blocks.extend((path_mean, np.sqrt(np.maximum(path_variance, 0.0))))
    names.extend(f"path_feature_mean_{index}" for index in range(path.shape[2]))
    names.extend(f"path_feature_std_{index}" for index in range(path.shape[2]))
    sequence_geometry = np.asarray(
        np.load(e85.SEQUENCE_GEOMETRY, mmap_mode="r")[query_ids], dtype=np.float32
    )
    sequence_mean = sequence_geometry.mean(axis=(1, 2))
    sequence_std = sequence_geometry.std(axis=(1, 2))
    blocks.extend((sequence_mean, sequence_std))
    names.extend(f"sequence_geometry_mean_{index}" for index in range(sequence_mean.shape[1]))
    names.extend(f"sequence_geometry_std_{index}" for index in range(sequence_std.shape[1]))
    depth = np.asarray(
        np.load(e85.PATH_CACHE / "interaction_count.npy", mmap_mode="r")[query_ids],
        dtype=np.int64,
    )
    depth_histogram = np.stack(
        [((depth == level) & mask).sum(axis=1) for level in range(8)], axis=1
    ).astype(np.float32)
    depth_histogram /= np.maximum(mask.sum(axis=1, keepdims=True), 1)
    blocks.append(depth_histogram)
    names.extend(f"interaction_depth_fraction_{level}" for level in range(8))
    return np.concatenate(blocks, axis=1).astype(np.float32), names


def geometry_point_features(
    query_ids: np.ndarray, fold_id: np.ndarray
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    _, train_positions, _, train_site, _ = load_positions()
    positions = train_positions[query_ids, :2].astype(np.float32)
    center = train_positions[:, :2].mean(axis=0)
    scale = train_positions[:, :2].std(axis=0) + 1e-6
    normalized_position = ((positions - center) / scale).astype(np.float32)
    site = np.eye(2, dtype=np.float32)[train_site[query_ids]]
    position_only = np.concatenate((normalized_position, site), axis=1)
    position_names = ["position_x", "position_y", "site_0", "site_1"]
    geodesic = np.load(GEODESIC_CACHE, allow_pickle=False)["train_features"][query_ids].astype(np.float32)
    visibility = np.load(VISIBILITY, allow_pickle=False)["train"][query_ids].astype(np.float32)
    regime_cache = np.load(REGIME_CACHE, allow_pickle=False)
    if not np.array_equal(regime_cache["oof_regime"].shape, fold_id.shape):
        raise ValueError("regime cache shape changed")
    regime = np.eye(4, dtype=np.float32)[regime_cache["oof_regime"].astype(np.int8)]
    risk = regime_cache["oof_risk"].astype(np.float32)[:, None]
    geometry = np.concatenate((position_only, geodesic, visibility, regime, risk), axis=1)
    names = position_names.copy()
    names.extend(f"geodesic_{index}" for index in range(geodesic.shape[1]))
    names.extend(f"visibility_{index}" for index in range(visibility.shape[1]))
    names.extend(f"regime_{index}" for index in range(4))
    names.append("regime_risk")
    return geometry, names, position_only, position_names


def model(args: argparse.Namespace, seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=args.trees,
        min_samples_leaf=args.min_leaf,
        max_features=0.70,
        bootstrap=False,
        n_jobs=-1,
        random_state=seed,
    )


def choose_threshold(
    prediction: np.ndarray,
    rows: np.ndarray,
    sequence_values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    control_values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[float, float, int]:
    local = prediction[rows]
    candidates = np.unique(
        np.concatenate(
            (
                np.asarray((-np.inf, 0.0, np.inf), dtype=np.float64),
                np.quantile(local, np.linspace(0.10, 0.90, 9)),
            )
        )
    )
    best = None
    for threshold in candidates:
        switch = local > threshold
        stitched = tuple(np.array(value[rows], copy=True) for value in sequence_values)
        for output, source in zip(stitched, control_values):
            output[switch] = source[rows][switch]
        score = float(statistics(*stitched)["score"])
        record = (score, -int(switch.sum()), float(threshold), int(switch.sum()))
        if best is None or record[:2] > best[:2]:
            best = record
    if best is None:
        raise RuntimeError("threshold selection failed")
    return best[2], best[0], best[3]


def strict_selector(
    name: str,
    features: np.ndarray,
    target: np.ndarray,
    fold_id: np.ndarray,
    sequence_values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    control_values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
    shuffled: bool = False,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[np.ndarray]]:
    selected = np.zeros(len(fold_id), dtype=bool)
    prediction = np.zeros(len(fold_id), dtype=np.float32)
    fold_records = []
    importances = []
    for outer in range(5):
        validation_fold = (outer + 1) % 5
        train = np.flatnonzero((fold_id != outer) & (fold_id != validation_fold))
        validation = np.flatnonzero(fold_id == validation_fold)
        held = np.flatnonzero(fold_id == outer)
        local_target = target[train].copy()
        seed = 2026080797 + 1000 * outer + (100 if shuffled else 0)
        if shuffled:
            rng = np.random.default_rng(seed)
            rng.shuffle(local_target)
        estimator = model(args, seed)
        estimator.fit(features[train], local_target)
        local_prediction = estimator.predict(features).astype(np.float32)
        prediction[held] = local_prediction[held]
        threshold, validation_score, validation_switches = choose_threshold(
            local_prediction,
            validation,
            sequence_values,
            control_values,
        )
        selected[held] = local_prediction[held] > threshold
        rank = spearmanr(local_prediction[held], target[held]).statistic
        fold_records.append(
            {
                "fold": outer,
                "validation_fold": validation_fold,
                "train_count": int(len(train)),
                "validation_count": int(len(validation)),
                "held_count": int(len(held)),
                "threshold": threshold,
                "validation_score": validation_score,
                "validation_switches": validation_switches,
                "held_switches": int(selected[held].sum()),
                "held_target_spearman": float(rank) if np.isfinite(rank) else None,
            }
        )
        importances.append(estimator.feature_importances_.astype(np.float64))
    stitched = tuple(np.array(value, copy=True) for value in sequence_values)
    for output, source in zip(stitched, control_values):
        output[selected] = source[selected]
    baseline = statistics(*sequence_values)
    summary = statistics(*stitched)
    for record in fold_records:
        mask = fold_id == record["fold"]
        record["held_delta"] = float(
            statistics(*stitched, mask)["score"]
            - statistics(*sequence_values, mask)["score"]
        )
    result = {
        "name": name,
        "metrics": summary,
        "delta": float(summary["score"] - baseline["score"]),
        "switch_count": int(selected.sum()),
        "positive_folds": int(sum(row["held_delta"] > 0.0 for row in fold_records)),
        "worst_fold_delta": float(min(row["held_delta"] for row in fold_records)),
        "prediction_target_spearman": float(spearmanr(prediction, target).statistic),
        "folds": fold_records,
    }
    return result, selected, prediction, importances


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    started = time.time()
    metadata = np.load(e86.OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    base_np = np.load(e86.V11_CHANNEL, mmap_mode="r")
    truth_np = np.asarray(
        np.load(e86.TRAIN_CHANNEL, mmap_mode="r")[query_ids], dtype=np.complex64
    )
    base = torch.as_tensor(np.array(base_np, copy=True), device=device)
    truth = torch.as_tensor(truth_np, device=device)
    base_pas = pas_distribution(base)
    base_pdp = pdp_distribution(base)
    truth_pas = pas_distribution(truth)
    truth_pdp = pdp_distribution(truth)
    sequence_pas = torch.as_tensor(
        np.asarray(
            np.load(e86.RENDERED["geometry_sequence"][0], mmap_mode="r"),
            dtype=np.float32,
        ),
        device=device,
    )
    sequence_pdp = torch.as_tensor(
        np.asarray(
            np.load(e86.RENDERED["geometry_sequence"][1], mmap_mode="r"),
            dtype=np.float32,
        ),
        device=device,
    )
    control_pas = torch.as_tensor(
        np.asarray(np.load(e86.RENDERED["no_sequence"][0], mmap_mode="r"), dtype=np.float32),
        device=device,
    )
    control_pdp = torch.as_tensor(
        np.asarray(np.load(e86.RENDERED["no_sequence"][1], mmap_mode="r"), dtype=np.float32),
        device=device,
    )
    spec = e86.Candidate("endpoint", "sqrt", 0.20, 0.20, "pas_pdp", 4)
    with torch.inference_mode():
        sequence_channel = e86.make_candidate(
            base, base_pas, base_pdp, sequence_pas, sequence_pdp, spec
        )
        control_channel = e86.make_candidate(
            base, base_pas, base_pdp, control_pas, control_pdp, spec
        )
        sequence_values = point_metrics(
            sequence_channel, truth, truth_pas, truth_pdp
        )
        control_values = point_metrics(control_channel, truth, truth_pas, truth_pdp)
        spectral, spectral_names = spectral_point_features(
            base_pas,
            base_pdp,
            sequence_pas,
            sequence_pdp,
            control_pas,
            control_pdp,
        )
    point_score_sequence = (
        0.4 * sequence_values[0]
        + 0.4 * sequence_values[1]
        + 0.2
        / (1.0 + sequence_values[2] / np.maximum(sequence_values[3], TINY))
    )
    point_score_control = (
        0.4 * control_values[0]
        + 0.4 * control_values[1]
        + 0.2 / (1.0 + control_values[2] / np.maximum(control_values[3], TINY))
    )
    target = (point_score_control - point_score_sequence).astype(np.float32)
    geometry, geometry_names, position, position_names = geometry_point_features(
        query_ids, fold_id
    )
    path, path_names = path_point_features(query_ids)
    full = np.concatenate((geometry, spectral, path), axis=1).astype(np.float32)
    full_names = geometry_names + spectral_names + path_names
    full = np.nan_to_num(full, nan=0.0, posinf=1e6, neginf=-1e6)
    geometry = np.nan_to_num(geometry, nan=0.0, posinf=1e6, neginf=-1e6)
    position = np.nan_to_num(position, nan=0.0, posinf=1e6, neginf=-1e6)
    primary, primary_mask, primary_prediction, importances = strict_selector(
        "full_physics",
        full,
        target,
        fold_id,
        sequence_values,
        control_values,
        args,
    )
    geometry_control, geometry_mask, geometry_prediction, _ = strict_selector(
        "geometry_only",
        geometry,
        target,
        fold_id,
        sequence_values,
        control_values,
        args,
    )
    position_control, position_mask, position_prediction, _ = strict_selector(
        "position_only",
        position,
        target,
        fold_id,
        sequence_values,
        control_values,
        args,
    )
    shuffled_control, shuffled_mask, shuffled_prediction, _ = strict_selector(
        "shuffled_target",
        full,
        target,
        fold_id,
        sequence_values,
        control_values,
        args,
        shuffled=True,
    )
    mean_importance = np.mean(np.stack(importances), axis=0)
    order = np.argsort(mean_importance)[::-1][:30]
    top_features = [
        {"name": full_names[index], "importance": float(mean_importance[index])}
        for index in order
    ]
    np.savez_compressed(
        SELECTIONS,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        target=target,
        primary_prediction=primary_prediction,
        primary_switch=primary_mask,
        geometry_prediction=geometry_prediction,
        geometry_switch=geometry_mask,
        position_prediction=position_prediction,
        position_switch=position_mask,
        shuffled_prediction=shuffled_prediction,
        shuffled_switch=shuffled_mask,
    )
    oracle = np.maximum(point_score_sequence, point_score_control)
    report = {
        "experiment": VERSION,
        "status": "strict_complete",
        "primary_hypothesis": (
            "Deployable paired endpoint, path-sequence, and map features predict where "
            "the ordered surface-sequence correction should revert to its no-sequence control."
        ),
        "configuration": {
            **vars(args),
            "validation_fold": "(outer + 1) % 5",
            "feature_count": int(full.shape[1]),
            "geometry_feature_count": int(geometry.shape[1]),
            "position_feature_count": int(position.shape[1]),
        },
        "V12": statistics(*sequence_values),
        "control_endpoint": statistics(*control_values),
        "target": {
            "mean": float(target.mean()),
            "std": float(target.std()),
            "positive_count": int((target > 0.0).sum()),
            "point_proxy_oracle_mean": float(oracle.mean()),
        },
        "selectors": {
            "primary": primary,
            "geometry_only": geometry_control,
            "position_only": position_control,
            "shuffled_target": shuffled_control,
        },
        "top_primary_features": top_features,
        "decision": {
            "passed": bool(primary["delta"] >= 0.001 and primary["positive_folds"] == 5),
            "minimum_delta": 0.001,
            "required_positive_folds": 5,
            "candidate_allowed": False,
        },
        "output": PATHS.relative(SELECTIONS),
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
