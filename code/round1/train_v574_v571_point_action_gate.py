#!/usr/bin/env python3
"""Learn a strict nested-CV point gate for the frozen v571 spectrum action."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

import generate_v543_disagreement_conditioned_cell_fusion as v543
import train_v570_global_path_family_graph_renderer as v570
from project_paths import PATHS


VERSION_NUMBER = 574
VERSION = "v574_v571_point_action_gate"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
FEATURE_CACHE = CACHE_DIR / "v574_v571_point_action_gate_features.npz"
OOF_CACHE = CACHE_DIR / "v574_v571_point_action_gate_oof.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"

V543_OOF = PATHS.cache_dir(543) / "v543_cellwise_fusion_oof.npz"
V571_OOF = PATHS.cache_dir(571) / "v571_soft_component_transport_oof.npz"
V572_SURFACE = PATHS.cache_dir(572) / "v572_aligned_surface_sequences_2500.npz"
V573_VISIBILITY = (
    PATHS.cache_dir(573) / "v573_official_pointcloud_route_visibility_2000.npz"
)
TRAIN_POS = PATHS.round1 / "Round1_Train_Pos.npy"

ALPHAS = np.asarray([0.15, 0.30, 0.50, 0.75, 1.00], dtype=np.float32)
ACTIVE_FRACTIONS = np.asarray(
    [0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00],
    dtype=np.float32,
)
REGIME_NAMES = tuple(v543.REGIMES)
FLOOR = np.float32(1e-10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-estimators", type=int, default=128)
    parser.add_argument("--min-leaves", type=int, nargs="+", default=[6, 18, 48])
    parser.add_argument("--max-features", type=float, default=0.45)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=57401)
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def require_inputs() -> None:
    required = (V543_OOF, V571_OOF, V572_SURFACE, V573_VISIBILITY, TRAIN_POS)
    missing = [PATHS.relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing v574 inputs: {missing}")


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    return value / np.maximum(value.sum(axis=1, keepdims=True), 1e-30)


def distribution_features(
    value: np.ndarray,
    prefix: str,
) -> tuple[np.ndarray, list[str]]:
    value = normalize_rows(value)
    count = value.shape[1]
    coordinate = np.linspace(0.0, 1.0, count, dtype=np.float32)
    center = value @ coordinate
    spread = np.sqrt(
        np.maximum(
            np.sum(value * (coordinate[None] - center[:, None]) ** 2, axis=1),
            0.0,
        )
    )
    entropy = -np.sum(value * np.log(np.maximum(value, FLOOR)), axis=1)
    entropy /= np.float32(max(math.log(count), 1.0))
    ordered = np.sort(value, axis=1)
    top_sizes = tuple(size for size in (1, 2, 4, 8, 16, 32) if size <= count)
    columns = [
        entropy,
        center,
        spread,
        np.argmax(value, axis=1).astype(np.float32) / np.float32(max(count - 1, 1)),
        np.max(value, axis=1),
    ]
    names = [
        f"{prefix}_entropy",
        f"{prefix}_center",
        f"{prefix}_spread",
        f"{prefix}_argmax",
        f"{prefix}_maximum",
    ]
    for size in top_sizes:
        columns.append(np.sum(ordered[:, -size:], axis=1))
        names.append(f"{prefix}_top{size}_mass")
    return np.stack(columns, axis=1).astype(np.float32), names


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    numerator = np.sum(left64 * right64, axis=1)
    denominator = np.sqrt(
        np.sum(left64 * left64, axis=1) * np.sum(right64 * right64, axis=1)
    )
    return (numerator / np.maximum(denominator, 1e-30)).astype(np.float32)


def action_features(
    base: np.ndarray,
    action: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    base = normalize_rows(base)
    action = normalize_rows(action)
    base_grid = base.reshape(-1, v543.HEIGHT, v543.WIDTH)
    action_grid = action.reshape(-1, v543.HEIGHT, v543.WIDTH)
    base_delay = normalize_rows(base_grid.sum(axis=2))
    action_delay = normalize_rows(action_grid.sum(axis=2))
    base_angle = normalize_rows(base_grid.sum(axis=1))
    action_angle = normalize_rows(action_grid.sum(axis=1))

    blocks: list[np.ndarray] = []
    names: list[str] = []
    for value, prefix in (
        (base, "base_joint"),
        (action, "action_joint"),
        (base_delay, "base_delay"),
        (action_delay, "action_delay"),
        (base_angle, "base_angle"),
        (action_angle, "action_angle"),
    ):
        local, local_names = distribution_features(value, prefix)
        blocks.append(local)
        names.extend(local_names)

    delta = action - base
    log_ratio = np.clip(
        np.log(np.maximum(action, FLOOR)) - np.log(np.maximum(base, FLOOR)),
        -6.0,
        6.0,
    )
    abs_delta = np.abs(delta)
    abs_log = np.abs(log_ratio)
    quantiles = (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
    scalar_columns = [
        np.sum(abs_delta, axis=1),
        np.sqrt(np.sum(delta * delta, axis=1)),
        cosine_rows(base, action),
        np.sum(np.minimum(base, action), axis=1),
        np.sum(np.maximum(delta, 0.0), axis=1),
        np.sum(np.maximum(-delta, 0.0), axis=1),
        np.mean(abs_log, axis=1),
        np.max(abs_log, axis=1),
    ]
    scalar_names = [
        "action_joint_l1",
        "action_joint_l2",
        "action_joint_cosine",
        "action_joint_overlap",
        "action_positive_mass",
        "action_negative_mass",
        "action_abs_log_mean",
        "action_abs_log_max",
    ]
    for quantile in quantiles:
        scalar_columns.append(np.quantile(abs_delta, quantile, axis=1))
        scalar_names.append(f"action_abs_delta_q{int(100 * quantile):02d}")
        scalar_columns.append(np.quantile(abs_log, quantile, axis=1))
        scalar_names.append(f"action_abs_log_q{int(100 * quantile):02d}")

    for marginal_name, left, right in (
        ("delay", base_delay, action_delay),
        ("angle", base_angle, action_angle),
    ):
        difference = right - left
        scalar_columns.extend(
            [
                np.sum(np.abs(difference), axis=1),
                np.sqrt(np.sum(difference * difference, axis=1)),
                cosine_rows(left, right),
                np.sum(np.minimum(left, right), axis=1),
                np.argmax(right, axis=1).astype(np.float32)
                - np.argmax(left, axis=1).astype(np.float32),
            ]
        )
        scalar_names.extend(
            [
                f"action_{marginal_name}_l1",
                f"action_{marginal_name}_l2",
                f"action_{marginal_name}_cosine",
                f"action_{marginal_name}_overlap",
                f"action_{marginal_name}_argmax_shift",
            ]
        )
    blocks.append(np.stack(scalar_columns, axis=1).astype(np.float32))
    names.extend(scalar_names)

    delay_group = 8
    pooled_base_delay = base_delay.reshape(len(base), -1, delay_group).sum(axis=2)
    pooled_action_delay = action_delay.reshape(len(base), -1, delay_group).sum(axis=2)
    pooled_delay_log = np.clip(
        np.log(np.maximum(pooled_action_delay, FLOOR))
        - np.log(np.maximum(pooled_base_delay, FLOOR)),
        -6.0,
        6.0,
    )
    for value, prefix in (
        (pooled_base_delay, "base_delay_pool8"),
        (pooled_action_delay, "action_delay_pool8"),
        (pooled_action_delay - pooled_base_delay, "delta_delay_pool8"),
        (pooled_delay_log, "logratio_delay_pool8"),
        (base_angle, "base_angle_bin"),
        (action_angle, "action_angle_bin"),
        (action_angle - base_angle, "delta_angle_bin"),
    ):
        blocks.append(value.astype(np.float32))
        names.extend([f"{prefix}_{index}" for index in range(value.shape[1])])

    delay_block = 12
    pooled_delta = delta.reshape(
        len(base), v543.HEIGHT // delay_block, delay_block, v543.WIDTH
    ).sum(axis=2)
    pooled_delta = np.sign(pooled_delta) * np.sqrt(np.abs(pooled_delta))
    blocks.append(pooled_delta.reshape(len(base), -1).astype(np.float32))
    names.extend(
        [
            f"joint_delta_pool12_d{delay}_h{angle}"
            for delay in range(pooled_delta.shape[1])
            for angle in range(pooled_delta.shape[2])
        ]
    )
    output = np.concatenate(blocks, axis=1).astype(np.float32)
    return np.nan_to_num(output, nan=0.0, posinf=8.0, neginf=-8.0), names


def masked_moments(
    value: np.ndarray,
    valid: np.ndarray,
    axes: tuple[int, ...],
    prefix: str,
    include_minimum: bool = True,
) -> tuple[np.ndarray, list[str]]:
    value = np.asarray(value, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    mask = valid[..., None]
    count = np.maximum(np.sum(valid, axis=axes), 1).astype(np.float32)
    total = np.sum(value * mask, axis=axes, dtype=np.float64).astype(np.float32)
    mean = total / count[:, None]
    second = (
        np.sum(value * value * mask, axis=axes, dtype=np.float64).astype(np.float32)
        / count[:, None]
    )
    std = np.sqrt(np.maximum(second - mean * mean, 0.0))
    maximum = np.max(np.where(mask, value, -1e6), axis=axes)
    maximum[~np.any(valid, axis=axes)] = 0.0
    blocks = [mean, std, maximum]
    labels = ["mean", "std", "max"]
    if include_minimum:
        minimum = np.min(np.where(mask, value, 1e6), axis=axes)
        minimum[~np.any(valid, axis=axes)] = 0.0
        blocks.append(minimum)
        labels.append("min")
    output = np.concatenate(blocks, axis=1).astype(np.float32)
    names = [
        f"{prefix}_{label}_{index}"
        for label, block in zip(labels, blocks)
        for index in range(block.shape[1])
    ]
    return output, names


def physical_features() -> tuple[np.ndarray, list[str]]:
    with np.load(V572_SURFACE) as surface, np.load(V573_VISIBILITY) as visibility:
        aligned = surface["aligned_index"][:2000].astype(np.int16)
        path_valid = aligned >= 0
        step_valid = surface["step_valid"][:2000].astype(bool)
        interaction = surface["interaction"][:2000].astype(np.int16)
        object_token = surface["object_token"][:2000].astype(np.int16)
        patch_id = surface["patch_id"][:2000].astype(np.int16)
        step_numeric = surface["step_numeric"][:2000].astype(np.float32)
        depth = surface["depth"][:2000].astype(np.int16)
        route_numeric = visibility["route_numeric"].astype(np.float32)
        step_visibility = visibility["step_visibility"].astype(np.float32)
        point_context = visibility["point_context"].astype(np.float32)

    blocks: list[np.ndarray] = [point_context]
    names = [f"point_context_{index}" for index in range(point_context.shape[1])]

    summary_columns = [
        np.mean(path_valid, axis=1),
        np.sum(path_valid, axis=1).astype(np.float32) / 32.0,
        np.sum(step_valid, axis=(1, 2)).astype(np.float32) / (32.0 * 16.0),
        np.sum(depth * path_valid, axis=1)
        / np.maximum(np.sum(path_valid, axis=1), 1),
        np.max(np.where(path_valid, depth, 0), axis=1).astype(np.float32) / 16.0,
    ]
    summary_names = [
        "surface_valid_path_fraction",
        "surface_valid_path_count_scaled",
        "surface_valid_step_fraction",
        "surface_mean_depth",
        "surface_max_depth_scaled",
    ]
    for threshold in (1, 2, 4, 8, 12, 16):
        summary_columns.append(np.mean(path_valid & (depth >= threshold), axis=1))
        summary_names.append(f"surface_depth_ge_{threshold}_fraction")
    blocks.append(np.stack(summary_columns, axis=1).astype(np.float32))
    names.extend(summary_names)

    depth_hist = np.stack(
        [np.mean(path_valid & (depth == value), axis=1) for value in range(17)], axis=1
    ).astype(np.float32)
    blocks.append(depth_hist)
    names.extend([f"surface_depth_hist_{value}" for value in range(17)])

    interaction_hist = np.stack(
        [np.mean(step_valid & (interaction == value), axis=(1, 2)) for value in range(16)],
        axis=1,
    ).astype(np.float32)
    object_hist = np.stack(
        [np.mean(step_valid & (object_token == value), axis=(1, 2)) for value in range(8)],
        axis=1,
    ).astype(np.float32)
    blocks.extend([interaction_hist, object_hist])
    names.extend([f"surface_interaction_hist_{value}" for value in range(16)])
    names.extend([f"surface_object_hist_{value}" for value in range(8)])

    patch_summary = np.zeros((len(path_valid), 4), dtype=np.float32)
    for point in range(len(path_valid)):
        ids = patch_id[point][step_valid[point] & (patch_id[point] > 0)]
        if len(ids) == 0:
            continue
        _, counts = np.unique(ids, return_counts=True)
        probability = counts.astype(np.float64) / np.sum(counts)
        patch_summary[point] = (
            len(counts) / 965.0,
            np.max(probability),
            -np.sum(probability * np.log(np.maximum(probability, 1e-30)))
            / max(math.log(len(counts)), 1.0),
            np.sum(counts >= 2) / max(len(counts), 1),
        )
    blocks.append(patch_summary)
    names.extend(
        [
            "surface_unique_patch_fraction",
            "surface_dominant_patch_mass",
            "surface_patch_entropy",
            "surface_repeated_patch_fraction",
        ]
    )

    step_block, step_names = masked_moments(
        step_numeric,
        step_valid,
        axes=(1, 2),
        prefix="surface_step",
    )
    route_block, route_names = masked_moments(
        route_numeric,
        path_valid,
        axes=(1,),
        prefix="route",
    )
    visibility_block, visibility_names = masked_moments(
        step_visibility,
        step_valid,
        axes=(1, 2),
        prefix="visibility",
        include_minimum=False,
    )
    blocks.extend([step_block, route_block, visibility_block])
    names.extend(step_names + route_names + visibility_names)

    for limit in (4, 8, 16, 32):
        local_valid = path_valid[:, :limit]
        local_value = route_numeric[:, :limit]
        count = np.maximum(np.sum(local_valid, axis=1), 1).astype(np.float32)
        local_mean = np.sum(
            local_value * local_valid[..., None], axis=1, dtype=np.float64
        ).astype(np.float32) / count[:, None]
        blocks.append(local_mean)
        names.extend([f"route_top{limit}_mean_{index}" for index in range(37)])

    output = np.concatenate(blocks, axis=1).astype(np.float32)
    output = np.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
    return output, names


def build_features(force: bool) -> tuple[np.ndarray, list[str]]:
    if FEATURE_CACHE.is_file() and not force:
        with np.load(FEATURE_CACHE) as loaded:
            return loaded["features"].astype(np.float32), loaded["names"].astype(str).tolist()

    with np.load(V543_OOF) as source, np.load(V571_OOF) as action_source:
        base = source["calibrated"].astype(np.float32)
        action = action_source["prediction"].astype(np.float32)
    static_train, _, static_names = v543.load_static_features()
    action_block, action_names = action_features(base, action)
    physical_block, physical_names = physical_features()
    features = np.concatenate(
        [static_train.astype(np.float32), action_block, physical_block], axis=1
    ).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=1e4, neginf=-1e4)
    names = [f"static_{name}" for name in static_names] + action_names + physical_names
    if features.shape[1] != len(names):
        raise RuntimeError(f"feature/name mismatch: {features.shape[1]} vs {len(names)}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FEATURE_CACHE,
        features=features.astype(np.float32),
        names=np.asarray(names),
    )
    return features, names


def action_gain_targets(
    truth: np.ndarray,
    base: np.ndarray,
    action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base_score = v543.spectrum_scores(truth, base)["weighted"]
    gains = np.empty((len(base), len(ALPHAS)), dtype=np.float32)
    for index, alpha in enumerate(ALPHAS):
        prediction = v543.log_blend(base, action, float(alpha))
        gains[:, index] = v543.spectrum_scores(truth, prediction)["weighted"] - base_score
    return gains, base_score.astype(np.float32)


def shuffle_targets_within_regime(
    target: np.ndarray,
    ids: np.ndarray,
    regimes: np.ndarray,
    seed: int,
) -> np.ndarray:
    output = target[ids].copy()
    rng = np.random.default_rng(seed)
    local_regimes = regimes[ids]
    for regime in REGIME_NAMES:
        local = np.flatnonzero(local_regimes == regime)
        if len(local) > 1:
            output[local] = output[rng.permutation(local)]
    return output


@dataclass
class GateEnsemble:
    models: list[ExtraTreesRegressor]
    importances: np.ndarray


def fit_ensemble(
    features: np.ndarray,
    targets: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> GateEnsemble:
    models: list[ExtraTreesRegressor] = []
    importances: list[np.ndarray] = []
    for offset, leaf in enumerate(args.min_leaves):
        model = ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            min_samples_leaf=int(leaf),
            max_features=args.max_features,
            n_jobs=args.n_jobs,
            random_state=seed + 101 * offset,
        )
        model.fit(features, targets)
        models.append(model)
        importances.append(model.feature_importances_.astype(np.float64))
    return GateEnsemble(models=models, importances=np.mean(importances, axis=0))


def ensemble_predictions(
    model: GateEnsemble,
    features: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [np.asarray(estimator.predict(features), dtype=np.float32) for estimator in model.models],
        axis=0,
    )


def top_fraction_mask(score: np.ndarray, fraction: float) -> np.ndarray:
    count = len(score)
    selected = max(1, min(count, int(math.ceil(float(fraction) * count))))
    order = np.argsort(score, kind="stable")
    output = np.zeros(count, dtype=bool)
    output[order[-selected:]] = True
    return output


def score_views(predictions: np.ndarray) -> dict[str, np.ndarray]:
    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)
    median = np.median(predictions, axis=0)
    return {
        "mean": mean,
        "lcb025": mean - 0.25 * std,
        "lcb050": mean - 0.50 * std,
        "median": median,
    }


def select_config(
    predictions: np.ndarray,
    gain_targets: np.ndarray,
    folds: np.ndarray,
) -> dict[str, object]:
    views = score_views(predictions)
    unique_folds = np.unique(folds)
    candidates: list[dict[str, object]] = []
    for view_name, view in views.items():
        for alpha_index, alpha in enumerate(ALPHAS):
            for fraction in ACTIVE_FRACTIONS:
                fold_gains: list[float] = []
                fold_precision: list[float] = []
                for fold in unique_folds:
                    local = np.flatnonzero(folds == fold)
                    active = top_fraction_mask(view[local, alpha_index], float(fraction))
                    local_gain = gain_targets[local, alpha_index]
                    fold_gains.append(float(np.mean(local_gain * active)))
                    fold_precision.append(float(np.mean(local_gain[active] > 0.0)))
                mean_gain = float(np.mean(fold_gains))
                std_gain = float(np.std(fold_gains))
                minimum_gain = float(np.min(fold_gains))
                positive_folds = int(np.sum(np.asarray(fold_gains) > 0.0))
                objective = mean_gain - 0.20 * std_gain - 0.35 * max(0.0, -minimum_gain)
                if positive_folds < max(3, len(unique_folds) - 1):
                    objective -= 0.01
                candidates.append(
                    {
                        "view": view_name,
                        "alpha_index": alpha_index,
                        "alpha": float(alpha),
                        "active_fraction": float(fraction),
                        "fold_gains": fold_gains,
                        "fold_precision": fold_precision,
                        "mean_gain": mean_gain,
                        "std_gain": std_gain,
                        "minimum_gain": minimum_gain,
                        "positive_folds": positive_folds,
                        "objective": objective,
                    }
                )
    selected = max(candidates, key=lambda row: float(row["objective"]))
    if float(selected["objective"]) <= 0.0:
        return {
            "view": "fallback",
            "alpha_index": 0,
            "alpha": 0.0,
            "active_fraction": 0.0,
            "fold_gains": [0.0 for _ in unique_folds],
            "fold_precision": [0.0 for _ in unique_folds],
            "mean_gain": 0.0,
            "std_gain": 0.0,
            "minimum_gain": 0.0,
            "positive_folds": len(unique_folds),
            "objective": 0.0,
        }
    return selected


def apply_selected_config(
    predictions: np.ndarray,
    config: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    if float(config["active_fraction"]) <= 0.0:
        return np.zeros(predictions.shape[1], dtype=bool), np.zeros(
            predictions.shape[1], dtype=np.float32
        )
    view = score_views(predictions)[str(config["view"])]
    alpha_index = int(config["alpha_index"])
    score = view[:, alpha_index]
    active = top_fraction_mask(score, float(config["active_fraction"]))
    return active, score.astype(np.float32)


def metrics_summary(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        key: float(np.mean(value))
        for key, value in v543.spectrum_scores(truth, prediction).items()
    }


def gate_prediction(
    base: np.ndarray,
    action: np.ndarray,
    active: np.ndarray,
    alpha: float,
) -> np.ndarray:
    output = base.copy()
    if alpha > 0.0 and np.any(active):
        output[active] = v543.log_blend(base[active], action[active], alpha)
    return output


def run_nested_gate(
    features: np.ndarray,
    gain_targets: np.ndarray,
    truth: np.ndarray,
    base: np.ndarray,
    action: np.ndarray,
    folds: np.ndarray,
    regimes: np.ndarray,
    valid: np.ndarray,
    names: list[str],
    args: argparse.Namespace,
    control: bool,
) -> dict[str, object]:
    label = "shuffled_control" if control else "main"
    output = base.copy()
    active_all = np.zeros(len(base), dtype=bool)
    score_all = np.zeros(len(base), dtype=np.float32)
    alpha_all = np.zeros(len(base), dtype=np.float32)
    fraction_all = np.zeros(len(base), dtype=np.float32)
    fold_rows: list[dict[str, object]] = []
    importance_sum = np.zeros(features.shape[1], dtype=np.float64)

    for outer_fold in range(5):
        started = time.monotonic()
        fit_ids = np.flatnonzero(valid & (folds != outer_fold))
        query_ids = np.flatnonzero(valid & (folds == outer_fold))
        inner_predictions = np.zeros(
            (len(args.min_leaves), len(fit_ids), len(ALPHAS)), dtype=np.float32
        )
        inner_fold_codes = folds[fit_ids]
        id_to_local = np.full(len(base), -1, dtype=np.int32)
        id_to_local[fit_ids] = np.arange(len(fit_ids), dtype=np.int32)

        for inner_fold in np.unique(inner_fold_codes):
            inner_train = np.flatnonzero(valid & (folds != outer_fold) & (folds != inner_fold))
            inner_query = np.flatnonzero(valid & (folds == inner_fold))
            train_targets = gain_targets[inner_train]
            if control:
                train_targets = shuffle_targets_within_regime(
                    gain_targets,
                    inner_train,
                    regimes,
                    args.seed + 100_000 + outer_fold * 1000 + int(inner_fold),
                )
            ensemble = fit_ensemble(
                features[inner_train],
                train_targets,
                args,
                args.seed + outer_fold * 10_000 + int(inner_fold) * 100 + (50_000 if control else 0),
            )
            inner_predictions[:, id_to_local[inner_query]] = ensemble_predictions(
                ensemble, features[inner_query]
            )

        config = select_config(
            inner_predictions,
            gain_targets[fit_ids],
            inner_fold_codes,
        )
        final_targets = gain_targets[fit_ids]
        if control:
            final_targets = shuffle_targets_within_regime(
                gain_targets,
                fit_ids,
                regimes,
                args.seed + 200_000 + outer_fold,
            )
        final_ensemble = fit_ensemble(
            features[fit_ids],
            final_targets,
            args,
            args.seed + outer_fold * 10_000 + 9000 + (50_000 if control else 0),
        )
        outer_predictions = ensemble_predictions(final_ensemble, features[query_ids])
        active, gate_score = apply_selected_config(outer_predictions, config)
        alpha = float(config["alpha"])
        prediction = gate_prediction(base[query_ids], action[query_ids], active, alpha)
        output[query_ids] = prediction
        active_all[query_ids] = active
        score_all[query_ids] = gate_score
        alpha_all[query_ids] = alpha
        fraction_all[query_ids] = float(config["active_fraction"])
        importance_sum += final_ensemble.importances

        base_metrics = metrics_summary(truth[query_ids], base[query_ids])
        action_metrics = metrics_summary(truth[query_ids], action[query_ids])
        gate_metrics = metrics_summary(truth[query_ids], prediction)
        selected_gain = gain_targets[query_ids, int(config["alpha_index"])]
        fold_rows.append(
            {
                "fold": outer_fold,
                "query_points": len(query_ids),
                "config": config,
                "active_points": int(np.sum(active)),
                "active_fraction_realized": float(np.mean(active)),
                "active_positive_fraction": (
                    float(np.mean(selected_gain[active] > 0.0)) if np.any(active) else 0.0
                ),
                "base": base_metrics,
                "full_v571": action_metrics,
                "gated": gate_metrics,
                "gain_vs_v543": {
                    key: float(gate_metrics[key] - base_metrics[key]) for key in base_metrics
                },
                "elapsed_seconds": float(time.monotonic() - started),
            }
        )
        print(
            f"v574 {label} outer={outer_fold} alpha={alpha:.2f} "
            f"fraction={float(config['active_fraction']):.2f} active={int(np.sum(active))}/"
            f"{len(query_ids)} gain={gate_metrics['weighted'] - base_metrics['weighted']:+.6f} "
            f"elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )

    scored = valid
    base_metrics = metrics_summary(truth[scored], base[scored])
    action_metrics = metrics_summary(truth[scored], action[scored])
    gate_metrics = metrics_summary(truth[scored], output[scored])
    importance = importance_sum / 5.0
    top = np.argsort(importance)[::-1][:50]
    regime_rows: dict[str, object] = {}
    for regime in REGIME_NAMES:
        local = scored & (regimes == regime)
        local_base = metrics_summary(truth[local], base[local])
        local_gate = metrics_summary(truth[local], output[local])
        regime_rows[regime] = {
            "points": int(np.sum(local)),
            "active_fraction": float(np.mean(active_all[local])),
            "base": local_base,
            "gated": local_gate,
            "gain_vs_v543": {
                key: float(local_gate[key] - local_base[key]) for key in local_base
            },
        }
    return {
        "label": label,
        "prediction": output,
        "active": active_all,
        "gate_score": score_all,
        "alpha": alpha_all,
        "active_fraction": fraction_all,
        "folds": fold_rows,
        "summary": {
            "base": base_metrics,
            "full_v571": action_metrics,
            "gated": gate_metrics,
            "gain_vs_v543": {
                key: float(gate_metrics[key] - base_metrics[key]) for key in base_metrics
            },
            "active_points": int(np.sum(active_all[scored])),
            "active_fraction": float(np.mean(active_all[scored])),
            "positive_outer_regions": int(
                np.sum(
                    [
                        float(row["gain_vs_v543"]["weighted"]) > 0.0
                        for row in fold_rows
                    ]
                )
            ),
            "minimum_outer_region_gain": float(
                np.min([row["gain_vs_v543"]["weighted"] for row in fold_rows])
            ),
        },
        "by_regime": regime_rows,
        "feature_importance": [
            {"name": names[index], "importance": float(importance[index])}
            for index in top
        ],
    }


def oracle_summary(
    truth: np.ndarray,
    base: np.ndarray,
    action: np.ndarray,
    gain_targets: np.ndarray,
    valid: np.ndarray,
    folds: np.ndarray,
) -> dict[str, object]:
    full_gain = gain_targets[:, -1]
    full_oracle = base.copy()
    positive_full = valid & (full_gain > 0.0)
    full_oracle[positive_full] = action[positive_full]

    best_index = np.argmax(np.maximum(gain_targets, 0.0), axis=1)
    best_gain = np.max(np.maximum(gain_targets, 0.0), axis=1)
    alpha_oracle = base.copy()
    for alpha_index, alpha in enumerate(ALPHAS):
        local = valid & (best_index == alpha_index) & (best_gain > 0.0)
        if np.any(local):
            alpha_oracle[local] = v543.log_blend(base[local], action[local], float(alpha))

    base_metrics = metrics_summary(truth[valid], base[valid])
    full_metrics = metrics_summary(truth[valid], full_oracle[valid])
    alpha_metrics = metrics_summary(truth[valid], alpha_oracle[valid])
    fold_rows = []
    for fold in range(5):
        local = valid & (folds == fold)
        local_base = metrics_summary(truth[local], base[local])
        local_full = metrics_summary(truth[local], full_oracle[local])
        local_alpha = metrics_summary(truth[local], alpha_oracle[local])
        fold_rows.append(
            {
                "fold": fold,
                "full_action_oracle_gain": float(
                    local_full["weighted"] - local_base["weighted"]
                ),
                "multi_alpha_oracle_gain": float(
                    local_alpha["weighted"] - local_base["weighted"]
                ),
            }
        )
    return {
        "full_action_positive_points": int(np.sum(positive_full)),
        "full_action_positive_fraction": float(np.mean(positive_full[valid])),
        "full_action_oracle": full_metrics,
        "full_action_oracle_gain": float(full_metrics["weighted"] - base_metrics["weighted"]),
        "multi_alpha_positive_points": int(np.sum(valid & (best_gain > 0.0))),
        "multi_alpha_oracle": alpha_metrics,
        "multi_alpha_oracle_gain": float(alpha_metrics["weighted"] - base_metrics["weighted"]),
        "folds": fold_rows,
    }


def main() -> None:
    args = parse_args()
    require_inputs()
    started = time.monotonic()
    features, names = build_features(args.force_features)
    with np.load(V543_OOF) as source, np.load(V571_OOF) as action_source:
        truth = source["truth"].astype(np.float32)
        base = source["calibrated"].astype(np.float32)
        regimes = source["regimes"].astype("U8")
        action = action_source["prediction"].astype(np.float32)
        valid = action_source["valid"].astype(bool)
    positions = np.load(TRAIN_POS).astype(np.float32)
    folds = v570.contiguous_fold_codes(positions)
    if not np.array_equal(folds, np.load(V571_OOF)["fold_id"]):
        raise RuntimeError("v574 fold codes do not match frozen v571 OOF")
    gain_targets, base_score = action_gain_targets(truth, base, action)
    oracle = oracle_summary(truth, base, action, gain_targets, valid, folds)

    print(
        f"v574 features={features.shape} full_v571_gain="
        f"{float(np.mean(gain_targets[valid, -1])):+.6f} "
        f"oracle={oracle['full_action_oracle_gain']:+.6f}",
        flush=True,
    )
    main_result = run_nested_gate(
        features,
        gain_targets,
        truth,
        base,
        action,
        folds,
        regimes,
        valid,
        names,
        args,
        control=False,
    )
    control_result = run_nested_gate(
        features,
        gain_targets,
        truth,
        base,
        action,
        folds,
        regimes,
        valid,
        names,
        args,
        control=True,
    )

    main_gain = float(main_result["summary"]["gain_vs_v543"]["weighted"])
    control_gain = float(control_result["summary"]["gain_vs_v543"]["weighted"])
    margin = main_gain - control_gain
    positive_regions = int(main_result["summary"]["positive_outer_regions"])
    success = main_gain >= 0.005 and positive_regions >= 4 and margin >= 0.003

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OOF_CACHE,
        valid=valid,
        fold_id=folds.astype(np.int16),
        regimes=regimes,
        base_score=base_score,
        gain_targets=gain_targets,
        prediction=np.asarray(main_result["prediction"], dtype=np.float32),
        active=np.asarray(main_result["active"], dtype=bool),
        gate_score=np.asarray(main_result["gate_score"], dtype=np.float32),
        alpha=np.asarray(main_result["alpha"], dtype=np.float32),
        active_fraction=np.asarray(main_result["active_fraction"], dtype=np.float32),
        control_prediction=np.asarray(control_result["prediction"], dtype=np.float32),
        control_active=np.asarray(control_result["active"], dtype=bool),
        control_gate_score=np.asarray(control_result["gate_score"], dtype=np.float32),
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "The frozen v571 path action contains useful pointwise corrections, but its "
            "average loss is caused by missing execute-or-fallback decisions. Geometry, "
            "action shape, ordered surfaces, and pointcloud visibility can predict that decision."
        ),
        "new_information": [
            "pointwise v571 action geometry and multi-alpha response supervision",
            "ordered v572 official-pointcloud surface sequence summaries",
            "v573 route obstruction, clearance, and receiver pointcloud context",
        ],
        "protocol": {
            "outer_folds": 5,
            "outer_split": "frozen contiguous v570/v571 regions",
            "inner_protocol": "leave-one-of-four-fit-regions-out predictions",
            "model": "multi-leaf ExtraTrees multi-output gain regressor",
            "n_estimators_per_leaf": args.n_estimators,
            "min_samples_leaf": args.min_leaves,
            "max_features": args.max_features,
            "alphas": ALPHAS.tolist(),
            "active_fractions": ACTIVE_FRACTIONS.tolist(),
            "control": "shuffle the full multi-alpha gain curve within propagation regime",
        },
        "scope": {
            "points": len(features),
            "scored_nonzero_points": int(np.sum(valid)),
            "feature_dim": features.shape[1],
            "submission_generated": False,
        },
        "oracle": oracle,
        "main": {
            key: value
            for key, value in main_result.items()
            if key not in {"prediction", "active", "gate_score", "alpha", "active_fraction"}
        },
        "shuffled_control": {
            key: value
            for key, value in control_result.items()
            if key not in {"prediction", "active", "gate_score", "alpha", "active_fraction"}
        },
        "decision": {
            "main_weighted_gain": main_gain,
            "positive_outer_regions": positive_regions,
            "shuffled_control_weighted_gain": control_gain,
            "margin_vs_shuffled_control": margin,
            "success_gate": (
                "main gain >= 0.005, at least 4/5 positive outer regions, "
                "and margin vs shuffled control >= 0.003"
            ),
            "passed": success,
            "deploy_test_candidate": success,
            "submission_generated": False,
        },
        "outputs": {
            "feature_cache": PATHS.relative(FEATURE_CACHE),
            "oof_cache": PATHS.relative(OOF_CACHE),
            "report": PATHS.relative(REPORT),
            "submission_zip": None,
        },
        "elapsed_seconds": float(time.monotonic() - started),
    }
    REPORT.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(report["decision"]), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
