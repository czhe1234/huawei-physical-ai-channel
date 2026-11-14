#!/usr/bin/env python3
"""Learn local-global path events instead of sharing one action per path family."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor

import analyze_v545_path_block_family_oracle as v545
import generate_v543_disagreement_conditioned_cell_fusion as v543
from project_paths import PATHS


VERSION_NUMBER = 546
VERSION = "v546_local_global_path_event_model"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
EVENT_CACHE = CACHE_DIR / "v546_path_event_records.npz"
OOF_CACHE = CACHE_DIR / "v546_path_event_model_oof.npz"
OUTPUT_DIR = PATHS.analysis / "visualizations" / "局部全局路径事件模型"

FIGURES = {
    "ladder": OUTPUT_DIR / "01_路径事件模型与现有方法对照.png",
    "folds": OUTPUT_DIR / "02_五折与传播区域增量.png",
    "actions": OUTPUT_DIR / "03_路径事件动作覆盖与候选来源.png",
    "heavy": OUTPUT_DIR / "04_重遮挡点路径事件收益空间分布.png",
}

PATH_ROOT = PATHS.cache_dir(475) / "shards"
SOURCE_COUNT = v545.SOURCE_COUNT
HEIGHT = v545.HEIGHT
WIDTH = v545.WIDTH
OWNER_NONE = v545.OWNER_NONE
PATH_FEATURE_DIM = 80
CANDIDATE_FEATURES_PER_SOURCE = 10
LOCAL_RADIUS = 20.0
LOCAL_NEIGHBORS = 8
MODEL_TREES = 900
CONTROL_TREES = 400
MIN_SAMPLES_LEAF = 8
MAX_FEATURES = 0.70
SEED = 54601
LOG_GAIN_SCALE = math.log(v545.GAIN_MAX)
REGIMES = v545.REGIMES
REGIME_CN = v545.REGIME_CN


def configure_chinese_font() -> None:
    selected = "DejaVu Sans"
    for family in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"):
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=family),
                fallback_to_default=False,
            )
            font_manager.fontManager.addfont(path)
            selected = font_manager.FontProperties(fname=path).get_name()
            break
        except ValueError:
            continue
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 260,
        }
    )


def validate_inputs() -> None:
    required = (
        v545.OOF_SOURCE,
        v545.BANK_CACHE,
        v545.BANK_META,
        v545.PATH_META,
        v545.TRAIN_POS,
        v545.TRAIN_POWER,
        v545.SCENE,
    )
    missing = [PATHS.relative(path) for path in required if not path.is_file()]
    shards = sorted(PATH_ROOT.glob("shard_*/path_targets.npz"))
    if len(shards) != 25:
        missing.append(f"expected 25 v475 shards, found {len(shards)}")
    if missing:
        raise FileNotFoundError(f"missing v546 inputs: {missing}")


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    return value / np.maximum(value.sum(axis=1, keepdims=True), 1e-30)


def entropy(value: np.ndarray) -> float:
    value = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    value /= max(float(value.sum()), 1e-30)
    return float(-np.sum(value * np.log(np.maximum(value, 1e-30))))


def circular_stats(value: np.ndarray) -> tuple[float, float, float]:
    value = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    value /= max(float(value.sum()), 1e-30)
    phase = np.exp(2j * np.pi * np.arange(len(value)) / max(len(value), 1))
    moment = np.sum(value * phase)
    return float(moment.real), float(moment.imag), float(abs(moment))


def direction(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros(3, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def path_features(
    data: np.lib.npyio.NpzFile,
    row: int,
    path: int,
    position: np.ndarray,
) -> np.ndarray:
    count = int(data["interaction_count"][row, path])
    interactions = data["interactions"][row, path, :count].astype(np.int16)
    vertices = data["vertices"][row, path, :count].astype(np.float64)
    tau = float(data["tau"][row, path])
    delay_bin = (-HEIGHT * 30_000.0 * tau) % HEIGHT
    delay_phase = 2.0 * math.pi * delay_bin / HEIGHT
    values: list[float] = [
        tau * 1e7,
        delay_bin / HEIGHT,
        math.sin(delay_phase),
        math.cos(delay_phase),
        float(data["log_power"][row, path]),
        float(data["normalized_power"][row, path]),
        count / 16.0,
        float(data["interaction_mask"][row, path]) / 255.0,
    ]
    for key in ("theta_t", "phi_t", "theta_r", "phi_r"):
        angle = float(data[key][row, path])
        values.extend([math.sin(angle), math.cos(angle)])
    for kind in range(9):
        values.append(float(np.sum(interactions == kind)) / 16.0)
    mask = int(data["interaction_mask"][row, path])
    values.extend([float(bool(mask & (1 << bit))) for bit in range(8)])

    horizontal = data["tx_h_fingerprint"][row, path].astype(np.float64)
    h_real, h_imag, h_abs = circular_stats(horizontal)
    values.extend(
        [
            entropy(horizontal) / math.log(16.0),
            float(np.max(horizontal)),
            h_real,
            h_imag,
            h_abs,
        ]
    )
    vertical = data["tx_v_fingerprint"][row, path].astype(np.float64)
    vertical /= max(float(vertical.sum()), 1e-30)
    vertical_index = np.arange(8, dtype=np.float64) / 7.0
    values.extend(
        [
            entropy(vertical) / math.log(8.0),
            float(np.max(vertical)),
            float(np.sum(vertical * vertical_index)),
            float(
                np.sqrt(
                    np.sum(
                        vertical
                        * (vertical_index - np.sum(vertical * vertical_index)) ** 2
                    )
                )
            ),
        ]
    )
    rx = data["rx_fingerprint"][row, path].astype(np.float64)
    rx /= max(float(rx.sum()), 1e-30)
    values.extend(rx.tolist())
    values.extend([entropy(rx) / math.log(4.0), float(np.max(rx))])
    polar = data["tx_pol_fingerprint"][row, path].astype(np.float64)
    polar /= max(float(polar.sum()), 1e-30)
    values.extend(polar.tolist())

    start = np.asarray([50.0, 0.0, 25.0], dtype=np.float64)
    end = np.asarray(position, dtype=np.float64)
    nodes = np.vstack([start, vertices, end])
    segments = np.diff(nodes, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    direct = float(np.linalg.norm(end - start))
    total = float(np.sum(lengths))
    values.extend(
        [
            direct / 500.0,
            total / 1000.0,
            (total - direct) / max(direct, 1.0),
            len(lengths) / 17.0,
            float(np.mean(lengths)) / 200.0,
            float(np.std(lengths)) / 200.0,
            float(np.min(lengths)) / 200.0,
            float(np.max(lengths)) / 500.0,
        ]
    )
    values.extend(direction(segments[0]).tolist())
    values.extend(direction(segments[-1]).tolist())
    if len(vertices):
        values.extend(
            [
                float(np.mean(vertices[:, 2])) / 100.0,
                float(np.std(vertices[:, 2])) / 100.0,
                float(np.min(vertices[:, 2])) / 100.0,
                float(np.max(vertices[:, 2])) / 100.0,
                float(np.mean(vertices[:, 0] - end[0])) / 500.0,
                float(np.mean(vertices[:, 1] - end[1])) / 500.0,
            ]
        )
    else:
        values.extend([0.0] * 6)
    output = np.zeros(PATH_FEATURE_DIM, dtype=np.float32)
    if len(values) > PATH_FEATURE_DIM:
        raise ValueError(f"path feature overflow: {len(values)} > {PATH_FEATURE_DIM}")
    output[: len(values)] = np.nan_to_num(
        np.asarray(values, dtype=np.float32),
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    )
    return output


def block_candidate_features(
    candidate: np.ndarray,
    anchor: np.ndarray,
    model: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = np.asarray(candidate, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    model = np.asarray(model, dtype=np.float64)
    anchor_norm = math.sqrt(float(np.sum(anchor * anchor)))
    model_norm = math.sqrt(float(np.sum(model * model)))
    anchor_mass = float(np.sum(anchor))
    model_mass = float(np.sum(model))
    features = np.zeros((SOURCE_COUNT, CANDIDATE_FEATURES_PER_SOURCE), dtype=np.float32)
    floor = 1e-10
    for source in range(SOURCE_COUNT):
        value = candidate[source]
        value_norm = math.sqrt(float(np.sum(value * value)))
        value_mass = float(np.sum(value))
        log_value = np.log(np.maximum(value, floor))
        log_anchor = np.log(np.maximum(anchor, floor))
        log_model = np.log(np.maximum(model, floor))
        weight = np.maximum(value + anchor + model, floor)
        weight /= max(float(weight.sum()), 1e-30)
        anchor_delta = log_value - log_anchor
        model_delta = log_value - log_model
        features[source] = np.asarray(
            [
                value_mass,
                value_norm,
                float(np.sum(value * anchor)) / max(value_norm * anchor_norm, 1e-30),
                float(np.sum(np.abs(value - anchor)))
                / max(value_mass + anchor_mass, 1e-30),
                float(np.sum(weight * anchor_delta)),
                float(
                    np.sqrt(
                        np.sum(
                            weight * (anchor_delta - np.sum(weight * anchor_delta)) ** 2
                        )
                    )
                ),
                float(np.sum(value * model)) / max(value_norm * model_norm, 1e-30),
                float(np.sum(np.abs(value - model)))
                / max(value_mass + model_mass, 1e-30),
                float(np.sum(weight * model_delta)),
                float(
                    np.sqrt(
                        np.sum(
                            weight * (model_delta - np.sum(weight * model_delta)) ** 2
                        )
                    )
                ),
            ],
            dtype=np.float32,
        )
    summary = np.asarray(
        [
            len(anchor) / (HEIGHT * WIDTH),
            anchor_mass,
            anchor_norm,
            model_mass,
            model_norm,
            float(np.mean(features[:, 0])),
            float(np.std(features[:, 0])),
            float(np.max(features[:, 0])),
            float(np.mean(features[:, 3])),
            float(np.max(features[:, 3])),
            float(np.mean(features[:, 7])),
            float(np.max(features[:, 7])),
        ],
        dtype=np.float32,
    )
    return features.reshape(-1), summary


def utility_targets(
    candidate: np.ndarray,
    target: np.ndarray,
    base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = np.asarray(candidate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    cross = np.sum(candidate * target[None, :], axis=1)
    candidate_sq = np.sum(candidate * candidate, axis=1)
    gain = np.clip(
        cross / np.maximum(candidate_sq, 1e-30),
        v545.GAIN_MIN,
        v545.GAIN_MAX,
    )
    candidate_error = np.sum((candidate * gain[:, None] - target[None, :]) ** 2, axis=1)
    base_error = float(np.sum((base - target) ** 2))
    utility = (base_error - candidate_error) / max(base_error, 1e-12)
    utility = np.clip(utility, -1.0, 1.0)
    log_gain = np.log(gain) / LOG_GAIN_SCALE
    return utility.astype(np.float32), log_gain.astype(np.float32)


def build_event_cache() -> dict[str, np.ndarray]:
    if EVENT_CACHE.is_file():
        cached = np.load(EVENT_CACHE)
        return {key: np.asarray(cached[key]) for key in cached.files}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    oof = np.load(v545.OOF_SOURCE)
    truth = oof["truth"].astype(np.float32)
    anchor = oof["anchor"].astype(np.float32)
    model = oof["calibrated"].astype(np.float32)
    bank = np.load(v545.BANK_CACHE, mmap_mode="r")
    path_meta_file = np.load(v545.PATH_META)
    path_meta = {key: np.asarray(path_meta_file[key]) for key in path_meta_file.files}
    owner = path_meta["owner_path"]
    power = np.load(v545.TRAIN_POWER).astype(np.float64)
    valid_points = power != 0.0
    positions = np.load(v545.TRAIN_POS).astype(np.float32)

    point_list: list[int] = []
    path_list: list[int] = []
    cells_list: list[int] = []
    family_lists = {mode: [] for mode in v545.IDENTITY_MODES}
    candidate_features: list[np.ndarray] = []
    block_features: list[np.ndarray] = []
    utility_anchor: list[np.ndarray] = []
    utility_model: list[np.ndarray] = []
    log_gain: list[np.ndarray] = []
    event_weight: list[float] = []

    for point in np.flatnonzero(valid_points):
        point_owner = owner[point]
        paths = np.unique(point_owner[point_owner != OWNER_NONE]).astype(np.int64)
        point_rows: list[int] = []
        point_mass: list[float] = []
        for path in paths:
            mask = point_owner == path
            candidate = np.asarray(bank[point], dtype=np.float64)[:, mask]
            target = truth[point, mask].astype(np.float64)
            local_anchor = anchor[point, mask].astype(np.float64)
            local_model = model[point, mask].astype(np.float64)
            source_features, summary = block_candidate_features(
                candidate, local_anchor, local_model
            )
            point_list.append(int(point))
            path_list.append(int(path))
            cells_list.append(int(np.sum(mask)))
            for mode in v545.IDENTITY_MODES:
                family_lists[mode].append(int(path_meta[f"family_{mode}"][point, path]))
            candidate_features.append(source_features)
            block_features.append(summary)
            local_utility, local_gain = utility_targets(candidate, target, local_anchor)
            model_utility, _ = utility_targets(candidate, target, local_model)
            utility_anchor.append(local_utility)
            utility_model.append(model_utility)
            log_gain.append(local_gain)
            point_rows.append(len(point_list) - 1)
            point_mass.append(max(float(np.sum(local_anchor)), 1e-5))
        if point_rows:
            weights = np.sqrt(np.asarray(point_mass, dtype=np.float64))
            weights /= max(float(weights.sum()), 1e-30)
            event_weight.extend(weights.tolist())

    records = {
        "point": np.asarray(point_list, dtype=np.int32),
        "path": np.asarray(path_list, dtype=np.int16),
        "cells": np.asarray(cells_list, dtype=np.int32),
        "candidate_features": np.asarray(candidate_features, dtype=np.float32),
        "block_features": np.asarray(block_features, dtype=np.float32),
        "utility_anchor": np.asarray(utility_anchor, dtype=np.float32),
        "utility_model": np.asarray(utility_model, dtype=np.float32),
        "log_gain": np.asarray(log_gain, dtype=np.float32),
        "event_weight": np.asarray(event_weight, dtype=np.float32),
        **{
            f"family_{mode}": np.asarray(value, dtype=np.int32)
            for mode, value in family_lists.items()
        },
    }

    path_feature_array = np.zeros(
        (len(records["point"]), PATH_FEATURE_DIM), dtype=np.float32
    )
    record_by_point: list[list[int]] = [[] for _ in range(2000)]
    for record, point in enumerate(records["point"]):
        record_by_point[int(point)].append(record)
    offset = 0
    for shard_id in range(20):
        shard = PATH_ROOT / f"shard_{shard_id:05d}" / "path_targets.npz"
        with np.load(shard) as data:
            rows = len(data["valid"])
            for row in range(rows):
                point = offset + row
                for record in record_by_point[point]:
                    path_feature_array[record] = path_features(
                        data,
                        row,
                        int(records["path"][record]),
                        positions[point],
                    )
        offset += rows
        print(f"v546 path features {offset}/2000", flush=True)
    if offset != 2000:
        raise ValueError(f"expected 2000 train path rows, found {offset}")
    records["path_features"] = path_feature_array
    np.savez_compressed(EVENT_CACHE, **records)
    return records


def local_target_features(
    records: dict[str, np.ndarray],
    positions: np.ndarray,
    fit_mask: np.ndarray,
    utility: np.ndarray,
    log_gain: np.ndarray,
) -> np.ndarray:
    point = records["point"].astype(np.int64)
    output_blocks: list[np.ndarray] = []
    for mode in v545.IDENTITY_MODES:
        family = records[f"family_{mode}"].astype(np.int64)
        output = np.zeros((len(point), 3 + 2 * SOURCE_COUNT), dtype=np.float32)
        order = np.argsort(family, kind="stable")
        sorted_family = family[order]
        boundaries = np.flatnonzero(np.diff(sorted_family)) + 1
        for group in np.split(order, boundaries):
            fit = group[fit_mask[group]]
            if not len(fit):
                continue
            tree = cKDTree(positions[point[fit], :2])
            k = min(LOCAL_NEIGHBORS + 4, len(fit))
            distance, local_index = tree.query(
                positions[point[group], :2],
                k=k,
                distance_upper_bound=LOCAL_RADIUS,
            )
            if k == 1:
                distance = distance[:, None]
                local_index = local_index[:, None]
            for local_row, record in enumerate(group):
                valid_neighbor = np.isfinite(distance[local_row]) & (
                    local_index[local_row] < len(fit)
                )
                neighbors = fit[local_index[local_row, valid_neighbor]]
                distances = distance[local_row, valid_neighbor]
                different_point = point[neighbors] != point[record]
                neighbors = neighbors[different_point][:LOCAL_NEIGHBORS]
                distances = distances[different_point][:LOCAL_NEIGHBORS]
                if not len(neighbors):
                    continue
                weight = 1.0 / np.maximum(distances + 2.0, 1e-6)
                weight /= max(float(weight.sum()), 1e-30)
                output[record, 0] = len(neighbors) / LOCAL_NEIGHBORS
                output[record, 1] = float(np.min(distances)) / LOCAL_RADIUS
                output[record, 2] = float(np.mean(distances)) / LOCAL_RADIUS
                output[record, 3 : 3 + SOURCE_COUNT] = np.sum(
                    utility[neighbors] * weight[:, None], axis=0
                )
                output[record, 3 + SOURCE_COUNT :] = np.sum(
                    log_gain[neighbors] * weight[:, None], axis=0
                )
        output_blocks.append(output)
    return np.concatenate(output_blocks, axis=1)


def make_model(seed: int, trees: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        bootstrap=False,
        n_jobs=-1,
        random_state=seed,
    )


def shuffled_targets(
    target: np.ndarray,
    record_regimes: np.ndarray,
    fit_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.asarray(target, dtype=np.float32).copy()
    for regime in REGIMES:
        ids = np.flatnonzero(fit_mask & (record_regimes == regime))
        output[ids] = target[ids][rng.permutation(len(ids))]
    return output


def apply_event_actions(
    base: np.ndarray,
    bank: np.ndarray,
    owner: np.ndarray,
    records: dict[str, np.ndarray],
    query_records: np.ndarray,
    choices: np.ndarray,
    gains: np.ndarray,
) -> np.ndarray:
    output = np.asarray(base, dtype=np.float32).copy()
    for record in query_records:
        source = int(choices[record])
        if source < 0:
            continue
        point = int(records["point"][record])
        path = int(records["path"][record])
        mask = owner[point] == path
        output[point, mask] = np.asarray(
            bank[point, source, mask], dtype=np.float32
        ) * np.float32(gains[record])
    return normalize_rows(output)


def evaluate_variant(
    name: str,
    base: np.ndarray,
    utility: np.ndarray,
    records: dict[str, np.ndarray],
    common_features: np.ndarray,
    positions: np.ndarray,
    fold_id: np.ndarray,
    regimes: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    bank: np.ndarray,
    owner: np.ndarray,
    *,
    run_control: bool,
) -> dict[str, object]:
    record_fold = fold_id[records["point"]]
    record_regimes = regimes[records["point"]]
    target = np.concatenate([utility, records["log_gain"]], axis=1).astype(np.float32)
    choices = np.full(len(record_fold), -1, dtype=np.int16)
    gains = np.ones(len(record_fold), dtype=np.float32)
    predicted_margin = np.zeros(len(record_fold), dtype=np.float32)
    source_only = np.asarray(base, dtype=np.float32).copy()
    prediction = np.asarray(base, dtype=np.float32).copy()
    control_prediction = np.asarray(base, dtype=np.float32).copy()
    control_choices = np.full(len(record_fold), -1, dtype=np.int16)
    control_gains = np.ones(len(record_fold), dtype=np.float32)
    fold_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(SEED + (0 if name == "anchor" else 1000))

    for outer in range(5):
        fit_mask = record_fold != outer
        query_mask = record_fold == outer
        query_records = np.flatnonzero(query_mask)
        local = local_target_features(
            records,
            positions,
            fit_mask,
            utility,
            records["log_gain"],
        )
        features = np.concatenate([common_features, local], axis=1).astype(np.float32)
        model = make_model(SEED + outer + (0 if name == "anchor" else 100), MODEL_TREES)
        model.fit(
            features[fit_mask],
            target[fit_mask],
            sample_weight=records["event_weight"][fit_mask],
        )
        predicted = model.predict(features[query_mask]).astype(np.float32)
        utility_prediction = predicted[:, :SOURCE_COUNT]
        local_choice = np.argmax(utility_prediction, axis=1).astype(np.int16)
        local_margin = utility_prediction[np.arange(len(local_choice)), local_choice]
        active = local_margin > 0.0
        local_gain = np.exp(
            np.clip(
                predicted[np.arange(len(local_choice)), SOURCE_COUNT + local_choice],
                -1.0,
                1.0,
            )
            * LOG_GAIN_SCALE
        ).astype(np.float32)
        local_gain = np.clip(local_gain, v545.GAIN_MIN, v545.GAIN_MAX)
        local_choice[~active] = -1
        local_gain[~active] = 1.0
        choices[query_records] = local_choice
        gains[query_records] = local_gain
        predicted_margin[query_records] = local_margin

        source_only = apply_event_actions(
            source_only,
            bank,
            owner,
            records,
            query_records,
            choices,
            np.ones_like(gains),
        )
        prediction = apply_event_actions(
            prediction,
            bank,
            owner,
            records,
            query_records,
            choices,
            gains,
        )

        if run_control:
            shuffled_utility = shuffled_targets(utility, record_regimes, fit_mask, rng)
            shuffled_gain = shuffled_targets(
                records["log_gain"], record_regimes, fit_mask, rng
            )
            shuffled_target = np.concatenate(
                [shuffled_utility, shuffled_gain], axis=1
            ).astype(np.float32)
            control_local = local_target_features(
                records,
                positions,
                fit_mask,
                shuffled_utility,
                shuffled_gain,
            )
            control_features = np.concatenate(
                [common_features, control_local], axis=1
            ).astype(np.float32)
            control = make_model(SEED + 500 + outer, CONTROL_TREES)
            control.fit(
                control_features[fit_mask],
                shuffled_target[fit_mask],
                sample_weight=records["event_weight"][fit_mask],
            )
            control_pred = control.predict(control_features[query_mask]).astype(
                np.float32
            )
            control_utility = control_pred[:, :SOURCE_COUNT]
            local_control_choice = np.argmax(control_utility, axis=1).astype(np.int16)
            control_margin = control_utility[
                np.arange(len(local_control_choice)), local_control_choice
            ]
            control_active = control_margin > 0.0
            local_control_gain = np.exp(
                np.clip(
                    control_pred[
                        np.arange(len(local_control_choice)),
                        SOURCE_COUNT + local_control_choice,
                    ],
                    -1.0,
                    1.0,
                )
                * LOG_GAIN_SCALE
            ).astype(np.float32)
            local_control_choice[~control_active] = -1
            local_control_gain[~control_active] = 1.0
            control_choices[query_records] = local_control_choice
            control_gains[query_records] = local_control_gain
            control_prediction = apply_event_actions(
                control_prediction,
                bank,
                owner,
                records,
                query_records,
                control_choices,
                control_gains,
            )

        point_query = fold_id == outer
        base_summary, base_metrics = v545.score_prediction(truth, base, valid, regimes)
        pred_summary, pred_metrics = v545.score_prediction(
            truth, prediction, valid, regimes
        )
        fold_rows.append(
            {
                "fold": outer,
                "points": int(np.sum(point_query)),
                "records": int(np.sum(query_mask)),
                "active_record_fraction": float(np.mean(active)),
                "mean_predicted_utility": float(np.mean(local_margin)),
                "gain": float(
                    np.mean(
                        pred_metrics["weighted"][point_query & valid]
                        - base_metrics["weighted"][point_query & valid]
                    )
                ),
            }
        )
        print(
            f"v546 {name} fold {outer + 1}/5 "
            f"gain={fold_rows[-1]['gain']:+.6f} "
            f"active={fold_rows[-1]['active_record_fraction']:.3f}",
            flush=True,
        )

    source_only = normalize_rows(source_only)
    prediction = normalize_rows(prediction)
    source_summary, source_metrics = v545.score_prediction(
        truth, source_only, valid, regimes
    )
    summary, metrics = v545.score_prediction(truth, prediction, valid, regimes)
    base_summary, base_metrics = v545.score_prediction(truth, base, valid, regimes)
    result: dict[str, object] = {
        "name": name,
        "base": base,
        "source_only": source_only,
        "prediction": prediction,
        "choices": choices,
        "gains": gains,
        "predicted_margin": predicted_margin,
        "base_summary": base_summary,
        "source_summary": source_summary,
        "summary": summary,
        "base_metrics": base_metrics,
        "source_metrics": source_metrics,
        "metrics": metrics,
        "folds": fold_rows,
    }
    if run_control:
        control_prediction = normalize_rows(control_prediction)
        control_summary, control_metrics = v545.score_prediction(
            truth, control_prediction, valid, regimes
        )
        result.update(
            {
                "control_prediction": control_prediction,
                "control_summary": control_summary,
                "control_metrics": control_metrics,
                "control_choices": control_choices,
            }
        )
    return result


def compact_summary(result: dict[str, object], valid: np.ndarray) -> dict[str, object]:
    base_score = float(result["base_summary"]["overall"]["weighted"])
    source_score = float(result["source_summary"]["overall"]["weighted"])
    score = float(result["summary"]["overall"]["weighted"])
    choices = np.asarray(result["choices"])
    output: dict[str, object] = {
        "base": result["base_summary"],
        "source_only": result["source_summary"],
        "source_gain": result["summary"],
        "source_only_gain": source_score - base_score,
        "source_gain_gain": score - base_score,
        "active_record_fraction": float(np.mean(choices >= 0)),
        "active_source_counts": {
            str(source): int(np.sum(choices == source))
            for source in range(SOURCE_COUNT)
        },
        "gain_distribution": v545.summarize(
            np.asarray(result["metrics"]["weighted"])[valid]
            - np.asarray(result["base_metrics"]["weighted"])[valid]
        ),
        "folds": result["folds"],
    }
    if "control_summary" in result:
        control_score = float(result["control_summary"]["overall"]["weighted"])
        output["shuffled_control"] = result["control_summary"]
        output["shuffled_control_gain"] = control_score - base_score
        output["gain_over_shuffled"] = score - control_score
    return output


def load_background() -> dict[str, object]:
    scene = np.load(v545.SCENE)
    wall = np.log1p(scene["grid_vertical_count"].astype(np.float32))
    positive = wall[wall > 0]
    x_min = float(scene["x_min"])
    y_min = float(scene["y_min"])
    resolution = float(scene["resolution"])
    return {
        "wall": wall,
        "extent": (
            x_min,
            x_min + wall.shape[1] * resolution,
            y_min,
            y_min + wall.shape[0] * resolution,
        ),
        "vmax": max(float(np.quantile(positive, 0.98)), 1.0),
    }


def decorate_map(axis: plt.Axes, background: dict[str, object]) -> None:
    axis.imshow(
        background["wall"],
        origin="lower",
        extent=background["extent"],
        cmap="Greys",
        vmin=0.0,
        vmax=background["vmax"],
        alpha=0.20,
        interpolation="nearest",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("横坐标（米）")
    axis.set_ylabel("纵坐标（米）")


def plot_results(
    anchor_result: dict[str, object],
    model_result: dict[str, object],
    methods: dict[str, float],
    positions: np.ndarray,
    regimes: np.ndarray,
    valid: np.ndarray,
) -> None:
    labels = [
        "物理锚点",
        "v543严格OOF",
        "v545路径族共享",
        "v546锚点路径事件",
        "v546叠加v543",
        "点内路径块oracle",
        "逐格oracle",
    ]
    values = [
        methods["anchor"],
        methods["v543"],
        methods["v545_crossfit"],
        float(anchor_result["summary"]["overall"]["weighted"]),
        float(model_result["summary"]["overall"]["weighted"]),
        methods["path_oracle"],
        methods["cell_oracle"],
    ]
    colors = [
        "#64748b",
        "#dc2626",
        "#0891b2",
        "#2563eb",
        "#16a34a",
        "#7c3aed",
        "#f59e0b",
    ]
    figure, axis = plt.subplots(figsize=(13, 6.3), constrained_layout=True)
    bars = axis.bar(np.arange(len(values)), values, color=colors)
    axis.bar_label(bars, fmt="%.4f", padding=3)
    axis.set_xticks(np.arange(len(values)), labels, rotation=14)
    axis.set_ylim(max(0.72, min(values) - 0.04), 1.0)
    axis.set_ylabel("训练集严格OOF联合谱综合相似度")
    axis.grid(axis="y", alpha=0.24)
    axis.set_title(
        "局部与全局路径事件能否把物理候选上限转成可迁移收益",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(FIGURES["ladder"], bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(15, 5.7), constrained_layout=True)
    folds = np.arange(1, 6)
    axes[0].plot(
        folds,
        [row["gain"] for row in anchor_result["folds"]],
        marker="o",
        linewidth=2,
        label="从物理锚点出发",
        color="#2563eb",
    )
    axes[0].plot(
        folds,
        [row["gain"] for row in model_result["folds"]],
        marker="o",
        linewidth=2,
        label="叠加v543",
        color="#16a34a",
    )
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xticks(folds)
    axes[0].set_xlabel("空间折")
    axes[0].set_ylabel("相对各自底座的综合分增量")
    axes[0].set_title("五个互斥空间折", fontweight="bold")
    axes[0].grid(alpha=0.24)
    axes[0].legend()
    x = np.arange(len(REGIMES))
    width = 0.36
    anchor_gain = [
        float(anchor_result["summary"]["by_regime"][regime]["weighted"])
        - float(anchor_result["base_summary"]["by_regime"][regime]["weighted"])
        for regime in REGIMES
    ]
    model_gain = [
        float(model_result["summary"]["by_regime"][regime]["weighted"])
        - float(model_result["base_summary"]["by_regime"][regime]["weighted"])
        for regime in REGIMES
    ]
    axes[1].bar(x - width / 2, anchor_gain, width, label="从物理锚点出发")
    axes[1].bar(x + width / 2, model_gain, width, label="叠加v543")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(x, [REGIME_CN[value] for value in REGIMES])
    axes[1].set_ylabel("综合分增量")
    axes[1].set_title("不同传播区域", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.24)
    axes[1].legend()
    figure.suptitle("路径事件模型的迁移稳定性", fontsize=15, fontweight="bold")
    figure.savefig(FIGURES["folds"], bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(15, 5.7), constrained_layout=True)
    for axis, result, title, color in (
        (axes[0], anchor_result, "从物理锚点出发", "#2563eb"),
        (axes[1], model_result, "叠加v543", "#16a34a"),
    ):
        choices = np.asarray(result["choices"])
        counts = np.asarray(
            [np.sum(choices == source) for source in range(SOURCE_COUNT)]
        )
        axis.bar(np.arange(SOURCE_COUNT), counts, color=color)
        axis.set_xlabel("候选源编号")
        axis.set_ylabel("被选择的路径事件数")
        axis.set_title(f"{title}：激活{np.mean(choices >= 0):.1%}", fontweight="bold")
        axis.set_xticks(np.arange(SOURCE_COUNT))
        axis.grid(axis="y", alpha=0.22)
    figure.suptitle("模型实际选择了哪些物理候选", fontsize=15, fontweight="bold")
    figure.savefig(FIGURES["actions"], bbox_inches="tight")
    plt.close(figure)

    heavy = valid & (regimes == "heavy")
    background = load_background()
    anchor_gain_point = np.asarray(anchor_result["metrics"]["weighted"]) - np.asarray(
        anchor_result["base_metrics"]["weighted"]
    )
    model_gain_point = np.asarray(model_result["metrics"]["weighted"]) - np.asarray(
        model_result["base_metrics"]["weighted"]
    )
    limit = max(
        float(np.quantile(np.abs(anchor_gain_point[heavy]), 0.98)),
        float(np.quantile(np.abs(model_gain_point[heavy]), 0.98)),
        1e-4,
    )
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.0), constrained_layout=True)
    for axis, gain, title in (
        (axes[0], anchor_gain_point, "从物理锚点出发"),
        (axes[1], model_gain_point, "叠加v543"),
    ):
        decorate_map(axis, background)
        scatter = axis.scatter(
            positions[heavy, 0],
            positions[heavy, 1],
            c=gain[heavy],
            cmap="RdBu_r",
            norm=norm,
            s=20,
            edgecolors="none",
        )
        axis.set_title(title, fontweight="bold")
        figure.colorbar(scatter, ax=axis, label="逐点综合分增量")
    figure.suptitle(
        "重遮挡区域中路径事件动作的收益与伤害位置",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(FIGURES["heavy"], bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    configure_chinese_font()
    validate_inputs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    oof = np.load(v545.OOF_SOURCE)
    truth = oof["truth"].astype(np.float32)
    anchor = oof["anchor"].astype(np.float32)
    cell_model = oof["calibrated"].astype(np.float32)
    cell_oracle = oof["cell_oracle"].astype(np.float32)
    fold_id = oof["fold_id"].astype(np.int16)
    regimes = oof["regimes"].astype("U8")
    positions = np.load(v545.TRAIN_POS).astype(np.float32)
    power = np.load(v545.TRAIN_POWER).astype(np.float64)
    valid = power != 0.0
    bank = np.load(v545.BANK_CACHE, mmap_mode="r")
    path_meta_file = np.load(v545.PATH_META)
    owner = np.asarray(path_meta_file["owner_path"])
    records = build_event_cache()
    static_train, _, _ = v543.load_static_features()
    common_features = np.concatenate(
        [
            static_train[records["point"]],
            records["path_features"],
            records["block_features"],
            records["candidate_features"],
        ],
        axis=1,
    ).astype(np.float32)
    common_features = np.nan_to_num(common_features, nan=0.0, posinf=1e6, neginf=-1e6)
    print(
        f"v546 records={len(records['point'])} common_features={common_features.shape[1]}",
        flush=True,
    )

    anchor_result = evaluate_variant(
        "anchor",
        anchor,
        records["utility_anchor"],
        records,
        common_features,
        positions,
        fold_id,
        regimes,
        truth,
        valid,
        bank,
        owner,
        run_control=True,
    )
    model_result = evaluate_variant(
        "v543",
        cell_model,
        records["utility_model"],
        records,
        common_features,
        positions,
        fold_id,
        regimes,
        truth,
        valid,
        bank,
        owner,
        run_control=False,
    )

    anchor_summary, _ = v545.score_prediction(truth, anchor, valid, regimes)
    model_summary, _ = v545.score_prediction(truth, cell_model, valid, regimes)
    cell_summary, _ = v545.score_prediction(truth, cell_oracle, valid, regimes)
    point_oracle = np.load(v545.POINT_ORACLE_CACHE).astype(np.float32)
    point_summary, _ = v545.score_prediction(truth, point_oracle, valid, regimes)
    v545_report = json.loads(v545.REPORT.read_text(encoding="utf-8"))
    v545_crossfit = float(v545_report["decision"]["best_exact_crossfit_score"])
    methods = {
        "anchor": float(anchor_summary["overall"]["weighted"]),
        "v543": float(model_summary["overall"]["weighted"]),
        "v545_crossfit": v545_crossfit,
        "path_oracle": float(point_summary["overall"]["weighted"]),
        "cell_oracle": float(cell_summary["overall"]["weighted"]),
    }
    plot_results(
        anchor_result,
        model_result,
        methods,
        positions,
        regimes,
        valid,
    )

    anchor_compact = compact_summary(anchor_result, valid)
    model_compact = compact_summary(model_result, valid)
    anchor_score = float(anchor_result["summary"]["overall"]["weighted"])
    model_score = float(model_result["summary"]["overall"]["weighted"])
    v543_score = methods["v543"]
    control_score = float(anchor_result["control_summary"]["overall"]["weighted"])
    decision = {
        "anchor_event_score": anchor_score,
        "v543_baseline_score": v543_score,
        "v543_event_score": model_score,
        "anchor_event_gain": anchor_score - methods["anchor"],
        "v543_event_gain": model_score - v543_score,
        "anchor_event_gain_over_shuffled": anchor_score - control_score,
        "anchor_event_exceeds_v543": bool(anchor_score > v543_score),
        "event_model_improves_v543": bool(model_score > v543_score),
        "heavy_not_harmed_vs_v543": bool(
            float(model_result["summary"]["by_regime"]["heavy"]["weighted"])
            >= float(model_result["base_summary"]["by_regime"]["heavy"]["weighted"])
        ),
        "generate_test_candidate": bool(
            model_score > v543_score
            and float(model_result["summary"]["by_regime"]["heavy"]["weighted"])
            >= float(model_result["base_summary"]["by_regime"]["heavy"]["weighted"])
        ),
    }
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "A path event model that combines global route geometry, same-point physical "
            "candidate disagreement and 20 m fit-only lifecycle evidence can predict source "
            "utility and shared path energy better than one fixed action per path family."
        ),
        "scope": {
            "train_points": 2000,
            "scored_nonzero_points": int(np.sum(valid)),
            "path_event_records": int(len(records["point"])),
            "test_data_used": False,
            "outer_spatial_folds": 5,
        },
        "model": {
            "type": "ExtraTrees multi-output path-event utility and gain regressor",
            "trees": MODEL_TREES,
            "control_trees": CONTROL_TREES,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "max_features": MAX_FEATURES,
            "common_feature_count": int(common_features.shape[1]),
            "local_radius_m": LOCAL_RADIUS,
            "local_neighbors": LOCAL_NEIGHBORS,
            "path_identity_modes": list(v545.IDENTITY_MODES),
            "candidate_sources": SOURCE_COUNT,
            "gain_range": [v545.GAIN_MIN, v545.GAIN_MAX],
            "no_action_rule": "maximum predicted utility <= 0",
        },
        "reference_scores": methods,
        "anchor_variant": anchor_compact,
        "v543_variant": model_compact,
        "decision": decision,
        "outputs": {
            "event_cache": PATHS.relative(EVENT_CACHE),
            "oof": PATHS.relative(OOF_CACHE),
            "figures": {key: PATHS.relative(path) for key, path in FIGURES.items()},
            "report": PATHS.relative(REPORT),
        },
    }
    np.savez_compressed(
        OOF_CACHE,
        fold_id=fold_id,
        valid=valid,
        regimes=regimes,
        anchor_prediction=np.asarray(anchor_result["prediction"], dtype=np.float32),
        anchor_source_only=np.asarray(anchor_result["source_only"], dtype=np.float32),
        anchor_control=np.asarray(
            anchor_result["control_prediction"], dtype=np.float32
        ),
        v543_prediction=np.asarray(model_result["prediction"], dtype=np.float32),
        anchor_choices=np.asarray(anchor_result["choices"], dtype=np.int16),
        v543_choices=np.asarray(model_result["choices"], dtype=np.int16),
        anchor_gains=np.asarray(anchor_result["gains"], dtype=np.float32),
        v543_gains=np.asarray(model_result["gains"], dtype=np.float32),
        anchor_weighted=np.asarray(
            anchor_result["metrics"]["weighted"], dtype=np.float32
        ),
        v543_weighted=np.asarray(model_result["metrics"]["weighted"], dtype=np.float32),
    )
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
