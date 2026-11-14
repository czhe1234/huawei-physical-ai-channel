#!/usr/bin/env python3
"""Learn when each non-Sionna V6 component action improves the V4 backbone."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass

import joblib
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor
import torch

from data import load_positions
from project_paths import PATHS
from run_rbf_spectral_field import OOF_META, statistics
from run_v6_regime_component_lifecycle import (
    METRICS as V6_METRICS,
    OOF_CURRENT_AD,
    PROFILE_ALPHA,
    REGIME_CACHE,
    TARGET_PATHS,
    TEST_CURRENT_AD,
    V4_TEST,
    _candidate_from_name,
    apply_candidate,
    candidate_list,
)


VERSION = "Round2-E42-learned-component-action-controller"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "learned_component_action_controller"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_learned_component_action_controller"
VERSION_DIR = PATHS.analysis / "results" / "version_reports" / "V7"

GEOMETRY_CACHE = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_geometry_visibility_features.npz"
)
GEODESIC_CACHE = (
    PATHS.artifacts
    / "round2"
    / "regime_lifecycle_support"
    / "geodesic_corridor_features.npz"
)
TRAIN_AD = (
    PATHS.artifacts / "round2" / "pre_sionna" / "round2_train_ad_support_f16.npy"
)
OOF_WEIGHTS = (
    PATHS.artifacts
    / "round2"
    / "v3_anchored_neural_rbf"
    / "oof_epoch6_corrected_weights.npz"
)
TEST_WEIGHTS = (
    PATHS.artifacts
    / "round2"
    / "v3_anchored_neural_rbf"
    / "test_corrected_weights.npz"
)

FEATURE_CACHE = ARTIFACT_DIR / "v7_action_features.npz"
MODEL_PATH = ARTIFACT_DIR / "v7_action_utility_ensemble.joblib"
SELECTION_PATH = ARTIFACT_DIR / "v7_selection.json"
OUTPUT = ARTIFACT_DIR / "Round2_V7_Learned_Component_Action_Test_Channel.npy"
REPORT = ANALYSIS_DIR / "learned_component_action_controller.json"
VERSION_REPORT = VERSION_DIR / "V7_learned_component_action_controller.json"

REGIME_NAMES = ("clear", "light", "medium", "heavy")
FAMILY_NAMES = tuple(TARGET_PATHS)
PROFILE_NAMES = tuple(PROFILE_ALPHA)
MODE_NAMES = ("ad", "metric", "joint")
TINY = 1e-30


@dataclass(frozen=True)
class ModelConfig:
    n_estimators: int = 240
    min_samples_leaf: int = 10
    max_features: float = 0.62
    seeds: tuple[int, ...] = (20_260_807, 20_260_908)


def _normalize_support(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = value.sum(axis=(1, 2, 3), keepdims=True, dtype=np.float64)
    return np.divide(
        value,
        np.maximum(total, TINY),
        out=np.zeros_like(value),
        where=total > 0.0,
    )


def _safe_root_normalize(value: np.ndarray) -> np.ndarray:
    value = np.sqrt(np.maximum(np.asarray(value, dtype=np.float32), 0.0))
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    return np.divide(
        value,
        np.maximum(norm, 1e-20),
        out=np.zeros_like(value),
        where=norm > 0.0,
    )


def support_summary(value: np.ndarray) -> np.ndarray:
    """Compress a sparse 16x8x192 support without assigning path identities."""

    support = _normalize_support(value)
    count = len(support)
    horizontal = support.sum(axis=(2, 3), dtype=np.float64).astype(np.float32)
    vertical = support.sum(axis=(1, 3), dtype=np.float64).astype(np.float32)
    delay = support.sum(axis=(1, 2), dtype=np.float64).astype(np.float32)
    delay_pool = delay.reshape(count, 24, 8).sum(axis=2)
    horizontal_delay = support.sum(axis=2, dtype=np.float64).astype(np.float32)
    horizontal_delay = horizontal_delay.reshape(count, 4, 4, 12, 16).sum(
        axis=(2, 4)
    )

    flat = support.reshape(count, -1)
    top_count = min(64, flat.shape[1])
    top_index = np.argpartition(flat, -top_count, axis=1)[:, -top_count:]
    top_mass = np.take_along_axis(flat, top_index, axis=1)
    order = np.argsort(top_mass, axis=1)[:, ::-1]
    top_index = np.take_along_axis(top_index, order, axis=1)
    top_mass = np.take_along_axis(top_mass, order, axis=1)

    peak_count = 8
    peak_index = top_index[:, :peak_count]
    peak_h = peak_index // (8 * 192)
    remainder = peak_index % (8 * 192)
    peak_v = remainder // 192
    peak_d = remainder % 192
    peaks = np.stack(
        [
            peak_h.astype(np.float32) / 15.0,
            peak_v.astype(np.float32) / 7.0,
            peak_d.astype(np.float32) / 191.0,
            np.sqrt(np.maximum(top_mass[:, :peak_count], 0.0)),
        ],
        axis=2,
    ).reshape(count, -1)

    cumulative = np.stack(
        [top_mass[:, :rank].sum(axis=1) for rank in (1, 2, 4, 8, 16, 32, 64)],
        axis=1,
    )
    entropy = -np.sum(flat * np.log(np.maximum(flat, 1e-30)), axis=1)
    entropy /= math.log(flat.shape[1])
    support_mass = flat.sum(axis=1)
    effective = np.divide(
        1.0,
        np.maximum(np.sum(flat * flat, axis=1), 1e-30),
        out=np.zeros(count, dtype=np.float32),
        where=support_mass > 0.0,
    )
    effective /= flat.shape[1]

    delay_coordinate = np.linspace(0.0, 1.0, 192, dtype=np.float32)
    delay_mean = delay @ delay_coordinate
    delay_std = np.sqrt(
        np.maximum(
            np.sum(
                delay
                * np.square(delay_coordinate[None, :] - delay_mean[:, None]),
                axis=1,
            ),
            0.0,
        )
    )
    horizontal_angle = 2.0 * np.pi * np.arange(16, dtype=np.float32) / 16.0
    horizontal_sin = horizontal @ np.sin(horizontal_angle)
    horizontal_cos = horizontal @ np.cos(horizontal_angle)
    horizontal_concentration = np.sqrt(horizontal_sin**2 + horizontal_cos**2)
    vertical_coordinate = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    vertical_mean = vertical @ vertical_coordinate
    vertical_std = np.sqrt(
        np.maximum(
            np.sum(
                vertical
                * np.square(vertical_coordinate[None, :] - vertical_mean[:, None]),
                axis=1,
            ),
            0.0,
        )
    )
    scalars = np.column_stack(
        [
            cumulative,
            entropy,
            effective,
            delay_mean,
            delay_std,
            horizontal_sin,
            horizontal_cos,
            horizontal_concentration,
            vertical_mean,
            vertical_std,
            flat.sum(axis=1) == 0.0,
        ]
    ).astype(np.float32)
    return np.concatenate(
        [
            _safe_root_normalize(horizontal),
            _safe_root_normalize(vertical),
            _safe_root_normalize(delay_pool),
            _safe_root_normalize(horizontal_delay.reshape(count, -1)),
            peaks.astype(np.float32),
            scalars,
        ],
        axis=1,
    ).astype(np.float32)


def support_relation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = _normalize_support(left)
    right = _normalize_support(right)
    left_flat = left.reshape(len(left), -1)
    right_flat = right.reshape(len(right), -1)
    root_left = np.sqrt(np.maximum(left_flat, 0.0))
    root_right = np.sqrt(np.maximum(right_flat, 0.0))
    cosine = np.sum(root_left * root_right, axis=1)
    l1 = np.sum(np.abs(left_flat - right_flat), axis=1)
    midpoint = 0.5 * (left_flat + right_flat)
    js = 0.5 * np.sum(
        left_flat * np.log(np.maximum(left_flat, 1e-30) / np.maximum(midpoint, 1e-30)),
        axis=1,
    )
    js += 0.5 * np.sum(
        right_flat
        * np.log(np.maximum(right_flat, 1e-30) / np.maximum(midpoint, 1e-30)),
        axis=1,
    )

    left_h = left.sum(axis=(2, 3), dtype=np.float64).astype(np.float32)
    right_h = right.sum(axis=(2, 3), dtype=np.float64).astype(np.float32)
    left_v = left.sum(axis=(1, 3), dtype=np.float64).astype(np.float32)
    right_v = right.sum(axis=(1, 3), dtype=np.float64).astype(np.float32)
    left_d = left.sum(axis=(1, 2), dtype=np.float64).astype(np.float32)
    right_d = right.sum(axis=(1, 2), dtype=np.float64).astype(np.float32)

    marginal_cosines = []
    for first, second in ((left_h, right_h), (left_v, right_v), (left_d, right_d)):
        first_root = _safe_root_normalize(first)
        second_root = _safe_root_normalize(second)
        marginal_cosines.append(np.sum(first_root * second_root, axis=1))

    delay_coordinate = np.linspace(0.0, 1.0, 192, dtype=np.float32)
    vertical_coordinate = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    left_delay = left_d @ delay_coordinate
    right_delay = right_d @ delay_coordinate
    left_vertical = left_v @ vertical_coordinate
    right_vertical = right_v @ vertical_coordinate
    left_entropy = -np.sum(left_flat * np.log(np.maximum(left_flat, 1e-30)), axis=1)
    right_entropy = -np.sum(
        right_flat * np.log(np.maximum(right_flat, 1e-30)), axis=1
    )
    return np.column_stack(
        [
            cosine,
            l1,
            js,
            *marginal_cosines,
            right_delay - left_delay,
            right_vertical - left_vertical,
            (right_entropy - left_entropy) / math.log(left_flat.shape[1]),
            np.max(right_flat, axis=1) - np.max(left_flat, axis=1),
        ]
    ).astype(np.float32)


def summarize_support_file(path, batch_size: int = 32) -> np.ndarray:
    value = np.load(path, mmap_mode="r")
    output = None
    for start in range(0, len(value), batch_size):
        end = min(start + batch_size, len(value))
        block = support_summary(np.asarray(value[start:end], dtype=np.float32))
        if output is None:
            output = np.empty((len(value), block.shape[1]), dtype=np.float32)
        output[start:end] = block
    if output is None:
        raise RuntimeError(f"empty support file: {path}")
    return output


def weight_features(
    weight_path,
    query_positions: np.ndarray,
    train_positions: np.ndarray,
    train_summary: np.ndarray,
    current_summary: np.ndarray,
) -> np.ndarray:
    cached = np.load(weight_path, allow_pickle=False)
    donor_ids = cached["donor_ids" if "donor_ids" in cached.files else "neighbor_ids"].astype(
        np.int64
    )
    base = cached[
        "base_weights" if "base_weights" in cached.files else "weights"
    ].astype(np.float32)
    pas = cached["pas_weights"].astype(np.float32) if "pas_weights" in cached.files else base
    pdp = cached["pdp_weights"].astype(np.float32) if "pdp_weights" in cached.files else base
    distances = np.linalg.norm(
        train_positions[donor_ids, :2] - query_positions[:, None, :2], axis=2
    ).astype(np.float32)

    parts = []
    for weights in (base, pas, pdp):
        absolute = np.abs(weights)
        total = np.maximum(absolute.sum(axis=1, keepdims=True), 1e-20)
        probability = absolute / total
        entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-30)), axis=1)
        entropy /= math.log(weights.shape[1])
        parts.append(
            np.column_stack(
                [
                    weights.sum(axis=1),
                    absolute.sum(axis=1),
                    np.maximum(-weights, 0.0).sum(axis=1),
                    np.max(absolute, axis=1),
                    1.0 / np.maximum(np.sum(probability * probability, axis=1), 1e-20),
                    entropy,
                ]
            )
        )
    distance_features = np.column_stack(
        [
            distances[:, 0],
            np.mean(distances[:, :4], axis=1),
            np.mean(distances[:, :16], axis=1),
            np.mean(distances, axis=1),
            distances[:, -1],
        ]
    )
    probability = np.abs(0.5 * (pas + pdp))
    probability /= np.maximum(probability.sum(axis=1, keepdims=True), 1e-20)
    donor_mean = np.einsum("nk,nkf->nf", probability, train_summary[donor_ids])
    donor_delta = donor_mean - current_summary
    donor_delta_features = np.column_stack(
        [
            np.mean(np.abs(donor_delta), axis=1),
            np.sqrt(np.mean(np.square(donor_delta), axis=1)),
            np.max(np.abs(donor_delta), axis=1),
            np.sum(donor_mean * current_summary, axis=1)
            / np.maximum(
                np.linalg.norm(donor_mean, axis=1)
                * np.linalg.norm(current_summary, axis=1),
                1e-20,
            ),
            np.mean(np.std(train_summary[donor_ids[:, :16]], axis=1), axis=1),
        ]
    )
    return np.concatenate([*parts, distance_features, donor_delta_features], axis=1).astype(
        np.float32
    )


def query_graph_features(
    positions: np.ndarray,
    site: np.ndarray,
    summary: np.ndarray,
    group: np.ndarray | None,
) -> np.ndarray:
    output = np.zeros((len(positions), 11), dtype=np.float32)
    shape = summary[:, :96]
    shape /= np.maximum(np.linalg.norm(shape, axis=1, keepdims=True), 1e-20)
    for site_value in np.unique(site):
        rows_site = np.flatnonzero(site == site_value)
        groups = np.unique(group[rows_site]) if group is not None else np.asarray([0])
        for group_value in groups:
            if group is None:
                rows = rows_site
            else:
                rows = rows_site[group[rows_site] == group_value]
            if len(rows) <= 1:
                continue
            count = min(9, len(rows))
            distances, local = cKDTree(positions[rows, :2]).query(
                positions[rows, :2], k=count, workers=-1
            )
            distances = np.atleast_2d(distances)[:, 1:]
            neighbors = rows[np.atleast_2d(local)[:, 1:]]
            cosine = np.sum(shape[rows, None, :] * shape[neighbors], axis=2)
            take4 = min(4, cosine.shape[1])
            take8 = min(8, cosine.shape[1])
            output[rows] = np.column_stack(
                [
                    distances[:, 0],
                    np.mean(distances[:, :take4], axis=1),
                    np.mean(distances[:, :take8], axis=1),
                    np.min(cosine, axis=1),
                    np.max(cosine, axis=1),
                    np.mean(cosine[:, :take4], axis=1),
                    np.mean(cosine[:, :take8], axis=1),
                    np.std(cosine[:, :take8], axis=1),
                    np.mean(np.abs(summary[rows] - summary[neighbors[:, 0]]), axis=1),
                    np.sqrt(
                        np.mean(np.square(summary[rows] - summary[neighbors[:, 0]]), axis=1)
                    ),
                    np.max(np.abs(summary[rows] - summary[neighbors[:, 0]]), axis=1),
                ]
            )
    return output


def build_point_features() -> dict[str, np.ndarray]:
    if FEATURE_CACHE.is_file():
        cached = np.load(FEATURE_CACHE, allow_pickle=False)
        required = {
            "oof_point",
            "test_point",
            "oof_summary",
            "test_summary",
            "oof_target_summary",
            "test_target_summary",
            "oof_relations",
            "test_relations",
        }
        if required.issubset(cached.files):
            return {key: cached[key] for key in cached.files}

    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    _, train_positions, test_positions, train_site, test_site = load_positions()
    geometry = np.load(GEOMETRY_CACHE, allow_pickle=False)
    geodesic = np.load(GEODESIC_CACHE, allow_pickle=False)
    regimes = np.load(REGIME_CACHE, allow_pickle=False)

    train_summary = summarize_support_file(TRAIN_AD)
    oof_summary = summarize_support_file(OOF_CURRENT_AD)
    test_summary = summarize_support_file(TEST_CURRENT_AD)

    oof_target_summary = []
    test_target_summary = []
    oof_relations = []
    test_relations = []
    oof_current = np.load(OOF_CURRENT_AD, mmap_mode="r")
    test_current = np.load(TEST_CURRENT_AD, mmap_mode="r")
    for family in FAMILY_NAMES:
        oof_path, test_path = TARGET_PATHS[family]
        oof_target_summary.append(summarize_support_file(oof_path))
        test_target_summary.append(summarize_support_file(test_path))
        oof_target = np.load(oof_path, mmap_mode="r")
        test_target = np.load(test_path, mmap_mode="r")
        relation_oof = []
        relation_test = []
        for start in range(0, len(oof_current), 32):
            end = min(start + 32, len(oof_current))
            relation_oof.append(
                support_relation(
                    np.asarray(oof_current[start:end], dtype=np.float32),
                    np.asarray(oof_target[start:end], dtype=np.float32),
                )
            )
        for start in range(0, len(test_current), 32):
            end = min(start + 32, len(test_current))
            relation_test.append(
                support_relation(
                    np.asarray(test_current[start:end], dtype=np.float32),
                    np.asarray(test_target[start:end], dtype=np.float32),
                )
            )
        oof_relations.append(np.concatenate(relation_oof, axis=0))
        test_relations.append(np.concatenate(relation_test, axis=0))

    oof_target_summary_array = np.stack(oof_target_summary, axis=1)
    test_target_summary_array = np.stack(test_target_summary, axis=1)
    oof_relations_array = np.stack(oof_relations, axis=1)
    test_relations_array = np.stack(test_relations, axis=1)

    oof_weight = weight_features(
        OOF_WEIGHTS,
        train_positions[query_ids],
        train_positions,
        train_summary,
        oof_summary,
    )
    test_weight = weight_features(
        TEST_WEIGHTS,
        test_positions,
        train_positions,
        train_summary,
        test_summary,
    )
    oof_graph = query_graph_features(
        train_positions[query_ids], train_site[query_ids], oof_summary, fold_id
    )
    test_graph = query_graph_features(test_positions, test_site, test_summary, None)

    oof_regime = regimes["oof_regime"].astype(np.int8)
    test_regime = regimes["test_regime"].astype(np.int8)
    oof_site = train_site[query_ids].astype(np.int8)
    test_site = test_site.astype(np.int8)
    oof_state = np.concatenate(
        [
            np.eye(2, dtype=np.float32)[oof_site],
            np.eye(4, dtype=np.float32)[oof_regime],
            regimes["oof_risk"].astype(np.float32)[:, None],
        ],
        axis=1,
    )
    test_state = np.concatenate(
        [
            np.eye(2, dtype=np.float32)[test_site],
            np.eye(4, dtype=np.float32)[test_regime],
            regimes["test_risk"].astype(np.float32)[:, None],
        ],
        axis=1,
    )

    oof_point = np.concatenate(
        [
            geometry["train"][query_ids].astype(np.float32),
            geodesic["train_features"][query_ids].astype(np.float32),
            oof_state,
            oof_summary,
            oof_relations_array.reshape(len(query_ids), -1),
            oof_weight,
            oof_graph,
        ],
        axis=1,
    ).astype(np.float32)
    test_point = np.concatenate(
        [
            geometry["test"].astype(np.float32),
            geodesic["test_features"].astype(np.float32),
            test_state,
            test_summary,
            test_relations_array.reshape(len(test_positions), -1),
            test_weight,
            test_graph,
        ],
        axis=1,
    ).astype(np.float32)
    payload = {
        "oof_point": oof_point,
        "test_point": test_point,
        "oof_summary": oof_summary,
        "test_summary": test_summary,
        "oof_target_summary": oof_target_summary_array,
        "test_target_summary": test_target_summary_array,
        "oof_relations": oof_relations_array,
        "test_relations": test_relations_array,
        "query_ids": query_ids,
        "fold_id": fold_id,
        "oof_site": oof_site,
        "test_site": test_site,
        "oof_regime": oof_regime,
        "test_regime": test_regime,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEATURE_CACHE, **payload)
    return payload


def action_features(
    point: np.ndarray,
    current_summary: np.ndarray,
    target_summary: np.ndarray,
    relations: np.ndarray,
    regime: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    candidates = candidate_list()
    blocks = []
    names = []
    for candidate in candidates:
        family = FAMILY_NAMES.index(candidate.target)
        profile = PROFILE_NAMES.index(candidate.profile)
        mode = MODE_NAMES.index(candidate.mode)
        alpha = PROFILE_ALPHA[candidate.profile][regime].astype(np.float32)
        delta = (target_summary[:, family] - current_summary) * alpha[:, None]
        static = np.concatenate(
            [
                np.eye(len(FAMILY_NAMES), dtype=np.float32)[
                    np.full(len(point), family, dtype=np.int64)
                ],
                np.eye(len(PROFILE_NAMES), dtype=np.float32)[
                    np.full(len(point), profile, dtype=np.int64)
                ],
                np.eye(len(MODE_NAMES), dtype=np.float32)[
                    np.full(len(point), mode, dtype=np.int64)
                ],
                alpha[:, None],
                np.square(alpha)[:, None],
                relations[:, family],
                relations[:, family] * alpha[:, None],
            ],
            axis=1,
        )
        blocks.append(np.concatenate([point, delta, static], axis=1).astype(np.float32))
        names.append(candidate.name)
    return np.stack(blocks, axis=1), names


def target_components(metrics: dict[str, np.ndarray]) -> np.ndarray:
    count = metrics["pas"].shape[1]
    baseline_nmse = float(metrics["error"][0].sum() / metrics["energy"][0].sum())
    error_scale = 0.2 * count / (
        (1.0 + baseline_nmse) ** 2 * float(metrics["energy"][0].sum())
    )
    return np.stack(
        [
            0.4 * (metrics["pas"][1:] - metrics["pas"][0]),
            0.4 * (metrics["pdp"][1:] - metrics["pdp"][0]),
            -error_scale * (metrics["error"][1:] - metrics["error"][0]),
        ],
        axis=2,
    ).transpose(1, 0, 2).astype(np.float32)


def selected_statistics(metrics: dict[str, np.ndarray], selected: np.ndarray) -> dict[str, float | int]:
    point = np.arange(len(selected))
    return statistics(
        metrics["pas"][selected, point],
        metrics["pdp"][selected, point],
        metrics["error"][selected, point],
        metrics["energy"][selected, point],
    )


def _fit_ensemble(
    features: np.ndarray,
    targets: np.ndarray,
    config: ModelConfig,
    seed_offset: int,
) -> list[ExtraTreesRegressor]:
    models = []
    for seed in config.seeds:
        model = ExtraTreesRegressor(
            n_estimators=config.n_estimators,
            min_samples_leaf=config.min_samples_leaf,
            max_features=config.max_features,
            n_jobs=-1,
            random_state=seed + seed_offset,
        )
        model.fit(features, targets)
        models.append(model)
    return models


def _predict_ensemble(
    models: list[ExtraTreesRegressor], features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.stack([model.predict(features) for model in models], axis=0)
    utility = values.sum(axis=2)
    return utility.mean(axis=0), utility.std(axis=0)


def choose_actions(
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
    uncertainty: float,
    fraction: float,
) -> np.ndarray:
    adjusted = mean - uncertainty * std
    best = np.argmax(adjusted, axis=1)
    best_value = adjusted[np.arange(len(best)), best]
    selected = best + 1
    selected[best_value <= threshold] = 0
    if fraction < 1.0:
        active = np.flatnonzero(selected > 0)
        keep = int(round(fraction * len(selected)))
        if len(active) > keep:
            order = active[np.argsort(best_value[active])[::-1]]
            selected[order[keep:]] = 0
    return selected.astype(np.int32)


def _candidate_controls() -> list[tuple[float, float, float]]:
    thresholds = (-0.001, 0.0, 0.00025, 0.0005, 0.001, 0.002, 0.003)
    uncertainty = (0.0, 0.5, 1.0)
    fractions = (0.20, 0.35, 0.50, 0.70, 1.0)
    return [
        (threshold, penalty, fraction)
        for threshold in thresholds
        for penalty in uncertainty
        for fraction in fractions
    ]


def fit_nested_selector(
    action_x: np.ndarray,
    target_y: np.ndarray,
    metrics: dict[str, np.ndarray],
    fold_id: np.ndarray,
    config: ModelConfig,
) -> dict[str, object]:
    point_count, action_count, feature_count = action_x.shape
    selected = np.zeros(point_count, dtype=np.int32)
    records = []
    controls = _candidate_controls()
    for outer in np.unique(fold_id):
        outer_train = np.flatnonzero(fold_id != outer)
        held = np.flatnonzero(fold_id == outer)
        inner_mean = np.zeros((len(outer_train), action_count), dtype=np.float32)
        inner_std = np.zeros_like(inner_mean)
        for inner in np.unique(fold_id[outer_train]):
            fit_points = outer_train[fold_id[outer_train] != inner]
            valid_points = outer_train[fold_id[outer_train] == inner]
            models = _fit_ensemble(
                action_x[fit_points].reshape(-1, feature_count),
                target_y[fit_points].reshape(-1, target_y.shape[2]),
                config,
                seed_offset=10_000 * int(outer) + 1_000 * int(inner),
            )
            mean, std = _predict_ensemble(
                models, action_x[valid_points].reshape(-1, feature_count)
            )
            destination = np.searchsorted(outer_train, valid_points)
            inner_mean[destination] = mean.reshape(len(valid_points), action_count)
            inner_std[destination] = std.reshape(len(valid_points), action_count)
        best_control = None
        best_score = -np.inf
        for control in controls:
            local_selected = choose_actions(inner_mean, inner_std, *control)
            value = selected_statistics(
                {
                    key: metrics[key][:, outer_train]
                    if key in ("pas", "pdp", "error", "energy")
                    else metrics[key]
                    for key in metrics
                },
                local_selected,
            )
            if float(value["score"]) > best_score:
                best_score = float(value["score"])
                best_control = control
        if best_control is None:
            raise RuntimeError("no selector control was evaluated")
        models = _fit_ensemble(
            action_x[outer_train].reshape(-1, feature_count),
            target_y[outer_train].reshape(-1, target_y.shape[2]),
            config,
            seed_offset=100_000 + 10_000 * int(outer),
        )
        held_mean, held_std = _predict_ensemble(
            models, action_x[held].reshape(-1, feature_count)
        )
        held_selected = choose_actions(
            held_mean.reshape(len(held), action_count),
            held_std.reshape(len(held), action_count),
            *best_control,
        )
        selected[held] = held_selected
        held_metrics = {
            key: metrics[key][:, held]
            if key in ("pas", "pdp", "error", "energy")
            else metrics[key]
            for key in metrics
        }
        value = selected_statistics(held_metrics, held_selected)
        baseline = selected_statistics(
            held_metrics, np.zeros(len(held), dtype=np.int32)
        )
        records.append(
            {
                "fold": int(outer),
                "control": {
                    "threshold": float(best_control[0]),
                    "uncertainty": float(best_control[1]),
                    "fraction": float(best_control[2]),
                },
                "inner_score": best_score,
                "held": value,
                "held_delta": float(value["score"] - baseline["score"]),
                "active": int(np.sum(held_selected > 0)),
            }
        )
        print(
            f"V7 selector outer={outer} delta={records[-1]['held_delta']:+.6f} "
            f"active={records[-1]['active']} control={best_control}",
            flush=True,
        )
    return {
        "selected": selected,
        "overall": selected_statistics(metrics, selected),
        "folds": records,
        "positive_folds": int(sum(row["held_delta"] > 0.0 for row in records)),
    }


def fit_final_selector(
    action_x: np.ndarray,
    target_y: np.ndarray,
    metrics: dict[str, np.ndarray],
    fold_id: np.ndarray,
    config: ModelConfig,
) -> tuple[list[ExtraTreesRegressor], tuple[float, float, float], dict[str, object]]:
    point_count, action_count, feature_count = action_x.shape
    oof_mean = np.zeros((point_count, action_count), dtype=np.float32)
    oof_std = np.zeros_like(oof_mean)
    for held_fold in np.unique(fold_id):
        fit_points = np.flatnonzero(fold_id != held_fold)
        held = np.flatnonzero(fold_id == held_fold)
        models = _fit_ensemble(
            action_x[fit_points].reshape(-1, feature_count),
            target_y[fit_points].reshape(-1, target_y.shape[2]),
            config,
            seed_offset=300_000 + 10_000 * int(held_fold),
        )
        mean, std = _predict_ensemble(models, action_x[held].reshape(-1, feature_count))
        oof_mean[held] = mean.reshape(len(held), action_count)
        oof_std[held] = std.reshape(len(held), action_count)
    best_control = None
    best_value = None
    for control in _candidate_controls():
        selected = choose_actions(oof_mean, oof_std, *control)
        value = selected_statistics(metrics, selected)
        if best_value is None or float(value["score"]) > float(best_value["score"]):
            best_control = control
            best_value = value
    if best_control is None or best_value is None:
        raise RuntimeError("final selector control was not selected")
    models = _fit_ensemble(
        action_x.reshape(-1, feature_count),
        target_y.reshape(-1, target_y.shape[2]),
        config,
        seed_offset=900_000,
    )
    return models, best_control, {"oof": best_value}


def generate_output(
    selected: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    base = np.load(V4_TEST, mmap_mode="r")
    current = np.load(TEST_CURRENT_AD, mmap_mode="r")
    regimes = np.load(REGIME_CACHE, allow_pickle=False)["test_regime"].astype(np.int8)
    _, _, _, _, site = load_positions()
    targets = {
        family: np.load(TARGET_PATHS[family][1], mmap_mode="r") for family in FAMILY_NAMES
    }
    names = ["identity"] + [candidate.name for candidate in candidate_list()]
    output = np.lib.format.open_memmap(
        OUTPUT, mode="w+", dtype=np.complex64, shape=base.shape
    )
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        block = np.array(base[start:end], dtype=np.complex64, copy=True)
        local_selected = selected[start:end]
        for candidate_index in np.unique(local_selected):
            if int(candidate_index) == 0:
                continue
            rows = np.flatnonzero(local_selected == candidate_index)
            candidate = _candidate_from_name(names[int(candidate_index)])
            with torch.inference_mode():
                prediction = apply_candidate(
                    torch.as_tensor(block[rows], device=device),
                    torch.as_tensor(
                        np.asarray(current[start:end][rows], dtype=np.float32),
                        device=device,
                    ),
                    torch.as_tensor(
                        np.asarray(
                            targets[candidate.target][start:end][rows], dtype=np.float32
                        ),
                        device=device,
                    ),
                    regimes[start:end][rows],
                    site[start:end][rows],
                    candidate,
                )
            block[rows] = prediction.cpu().numpy().astype(np.complex64)
        output[start:end] = block
        output.flush()
        print(f"V7 test output {end}/{len(base)}", flush=True)
    finite = True
    for start in range(0, len(output), 8):
        block = np.asarray(output[start : start + 8])
        finite = finite and bool(
            np.isfinite(block.real).all() and np.isfinite(block.imag).all()
        )
    return {
        "path": PATHS.relative(OUTPUT),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "bytes": int(OUTPUT.stat().st_size),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("all", "features", "evaluate", "test"), default="all"
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    VERSION_DIR.mkdir(parents=True, exist_ok=True)
    features = build_point_features()
    if args.phase == "features":
        print(
            json.dumps(
                {
                    "feature_cache": PATHS.relative(FEATURE_CACHE),
                    "oof_shape": list(features["oof_point"].shape),
                    "test_shape": list(features["test_point"].shape),
                },
                indent=2,
            ),
            flush=True,
        )
        return

    metrics_file = np.load(V6_METRICS, allow_pickle=False)
    metrics = {key: metrics_file[key] for key in metrics_file.files}
    oof_action_x, action_names = action_features(
        features["oof_point"],
        features["oof_summary"],
        features["oof_target_summary"],
        features["oof_relations"],
        features["oof_regime"],
    )
    expected_names = metrics["candidate_names"].astype(str).tolist()[1:]
    if action_names != expected_names:
        raise RuntimeError("V7 action order does not match V6 metric order")
    target_y = target_components(metrics)
    config = ModelConfig()
    nested = fit_nested_selector(
        oof_action_x, target_y, metrics, features["fold_id"], config
    )
    baseline = selected_statistics(metrics, np.zeros(len(features["fold_id"]), dtype=np.int32))
    v6_selection = json.loads(
        (
            PATHS.artifacts
            / "round2"
            / "regime_lifecycle_support"
            / "v6_selection.json"
        ).read_text(encoding="utf-8")
    )
    v6_nested = v6_selection["winner"]["nested"]
    report = {
        "experiment": VERSION,
        "status": "analysis_only",
        "official_inputs_only": True,
        "sionna_used": False,
        "primary_hypothesis": (
            "V6 target supports contain useful path events, but their benefit is point- and "
            "action-dependent; a shared utility model can identify observable survival, birth "
            "and shift states without assigning brittle path identities."
        ),
        "validation": (
            "strict five outer spatial holes; every outer control is selected by an inner "
            "leave-one-hole-out prediction over the remaining four folds"
        ),
        "feature_shapes": {
            "point": list(features["oof_point"].shape),
            "action": list(oof_action_x.shape),
        },
        "model": {
            "type": "three-seed ExtraTrees multi-output utility ensemble",
            "config": config.__dict__,
        },
        "baseline": baseline,
        "v6_nested": v6_nested,
        "nested": {
            key: value for key, value in nested.items() if key != "selected"
        },
        "nested_gain_vs_v4": float(nested["overall"]["score"] - baseline["score"]),
        "nested_gain_vs_v6": float(nested["overall"]["score"] - v6_nested["score"]),
        "seconds": float(time.time() - started),
    }
    if args.phase == "evaluate":
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        VERSION_REPORT.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return

    models, control, final_diagnostics = fit_final_selector(
        oof_action_x, target_y, metrics, features["fold_id"], config
    )
    test_action_x, test_names = action_features(
        features["test_point"],
        features["test_summary"],
        features["test_target_summary"],
        features["test_relations"],
        features["test_regime"],
    )
    if test_names != action_names:
        raise RuntimeError("OOF/test action order mismatch")
    mean, std = _predict_ensemble(models, test_action_x.reshape(-1, test_action_x.shape[2]))
    selected_test = choose_actions(
        mean.reshape(len(test_action_x), len(action_names)),
        std.reshape(len(test_action_x), len(action_names)),
        *control,
    )
    joblib.dump(
        {
            "models": models,
            "control": control,
            "action_names": action_names,
            "feature_count": int(oof_action_x.shape[2]),
            "config": config.__dict__,
        },
        MODEL_PATH,
        compress=3,
    )
    selection_payload = {
        "control": {
            "threshold": float(control[0]),
            "uncertainty": float(control[1]),
            "fraction": float(control[2]),
        },
        "action_names": ["identity", *action_names],
        "test_selected": selected_test.tolist(),
        "test_active": int(np.sum(selected_test > 0)),
        "test_action_counts": {
            (["identity", *action_names])[index]: int(np.sum(selected_test == index))
            for index in np.unique(selected_test)
        },
        "final_oof": final_diagnostics,
    }
    SELECTION_PATH.write_text(
        json.dumps(selection_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output = generate_output(selected_test, args.batch_size, torch.device(args.device))
    report.update(
        {
            "status": "candidate_ready",
            "final_control": selection_payload["control"],
            "final_oof": final_diagnostics,
            "test_selection": {
                "active": selection_payload["test_active"],
                "action_counts": selection_payload["test_action_counts"],
            },
            "output": output,
            "outputs": {
                "features": PATHS.relative(FEATURE_CACHE),
                "model": PATHS.relative(MODEL_PATH),
                "selection": PATHS.relative(SELECTION_PATH),
                "channel": PATHS.relative(OUTPUT),
                "report": PATHS.relative(REPORT),
            },
            "seconds": float(time.time() - started),
        }
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    REPORT.write_text(text, encoding="utf-8")
    VERSION_REPORT.write_text(text, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
