#!/usr/bin/env python3
"""Learn strict-OOF reliability for deployable wall-phase actions."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

from analyze_v4_map_virtual_source_phase import (
    METERS_PER_DELAY_BIN,
    Config as MapConfig,
    build_wall_bank,
    image_sources,
)
from analyze_v5_deployable_wall_phase_transport import (
    PHASE_CACHE,
    gate_masks,
    mix_full_phase,
)
from analyze_v5_explicit_virtual_path_identity import (
    circular_abs,
    horizontal_bin,
    load_fold_parameters,
)
from analyze_v5_path_relative_port_phase import (
    fixed_coordinate_decomposition,
    ports_to_channel,
    preserve_power,
)
from analyze_v5_preprojection_wall_phase import (
    Config as ReplayConfig,
    load_v4_decisions,
    replay_v4,
)
from analyze_v5_static_link_phase_calibration import TRAIN_CHANNEL, V4_OOF_CHANNEL
from data import load_positions
from evaluate_v3_anchored_projection import target_batch
from official_metrics import point_metric_statistics
from project_paths import PATHS
from run_rbf_spectral_field import OOF_BASE, OOF_META, statistics
from run_v4_calibrated_exact_projection import (
    OOF_CALIBRATED_PAS,
    OOF_CALIBRATED_PDP,
    TARGET_AD,
    TARGET_PAS,
    TARGET_PDP,
)


VERSION = "Round2-E36-supervised-wall-phase-selector"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v5_supervised_wall_phase_selector"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v5_supervised_wall_phase_selector"
FEATURE_CACHE = ARTIFACT_DIR / "oof_wall_phase_path_features.npz"
SELECTOR_CACHE = ARTIFACT_DIR / "oof_wall_phase_selector_predictions.npz"
METRIC_CACHE = ARTIFACT_DIR / "oof_supervised_wall_phase_metrics.npz"
REPORT = ANALYSIS_DIR / "supervised_wall_phase_selector.json"
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    paths: int = 16
    phase_alpha: float = 0.10
    decomposition_ridge: float = 0.003
    trees: int = 384
    minimum_leaf: int = 8
    maximum_features: float = 0.75
    family_prior_observations: float = 10.0
    target_clip: float = 2.0
    deployment_score_threshold: float = 0.645
    deployment_positive_folds: int = 4
    seed: int = 20_260_805


VARIANT_NAMES = (
    "identity",
    "quality_top75_correct",
    "all_active_correct",
    "family_selector_correct",
    "model_selector_correct",
    "shuffled_selector_correct",
    "model_selector_rotated_control",
    "model_selector_wrong_beta_control",
    "oracle_selector_correct",
)


def unit_vector(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    return value / np.maximum(norm, 1e-20)


def phase_mix_numpy(base: np.ndarray, source: np.ndarray, alpha: float) -> np.ndarray:
    base_unit = base / np.maximum(np.abs(base), 1e-20)
    source_unit = source / np.maximum(np.abs(source), 1e-20)
    direction = (1.0 - alpha) * base_unit + alpha * source_unit
    direction /= np.maximum(np.abs(direction), 1e-20)
    return np.abs(base) * direction


def vector_cosine(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_unit = unit_vector(left)
    right_unit = unit_vector(right)
    inner = np.sum(np.conj(left_unit) * right_unit, axis=-1)
    return np.real(inner), np.abs(inner)


def selected_wall_geometry(
    point: np.ndarray,
    bs_position: np.ndarray,
    source: np.ndarray,
    normal: np.ndarray,
    offset: float,
    orientation: float,
) -> tuple[float, float]:
    delta = point - source
    length = float(np.linalg.norm(delta))
    denominator = float(np.dot(delta[:2], normal))
    numerator = float(offset - np.dot(source[:2], normal))
    parameter = numerator / denominator if abs(denominator) > 1e-12 else 0.5
    reflection = source + parameter * delta
    departure = reflection[:2] - bs_position[:2]
    azimuth = math.atan2(float(departure[1]), float(departure[0]))
    horizontal = float(horizontal_bin(np.asarray(azimuth), orientation))
    return length, horizontal


@torch.inference_mode()
def build_path_features(
    arrays: dict[str, np.ndarray],
    config: Config,
    device: torch.device,
    batch_size: int,
    rebuild: bool,
) -> dict[str, np.ndarray]:
    if FEATURE_CACHE.is_file() and not rebuild:
        return {key: value for key, value in np.load(FEATURE_CACHE).items()}
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    setup, train_positions, _, train_site, _ = load_positions()
    site = train_site[query_ids].astype(np.int8)
    base_all = np.load(OOF_BASE, mmap_mode="r")
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    base_coefficient = np.zeros((len(query_ids), config.paths, 8), dtype=np.complex64)
    truth_coefficient = np.zeros_like(base_coefficient)
    point_base_power = np.zeros(len(query_ids), dtype=np.float64)
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        base = torch.as_tensor(
            np.array(base_all[start:end], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True),
            device=device,
        )
        base_value, _, _ = fixed_coordinate_decomposition(
            base,
            arrays["coordinates"][start:end],
            config.decomposition_ridge,
            device,
        )
        truth_value, _, _ = fixed_coordinate_decomposition(
            truth,
            arrays["coordinates"][start:end],
            config.decomposition_ridge,
            device,
        )
        base_coefficient[start:end] = base_value.cpu().numpy()
        truth_coefficient[start:end] = truth_value.cpu().numpy()
        point_base_power[start:end] = (
            base.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64).cpu().numpy()
        )
        if end % 100 == 0 or end == len(query_ids):
            print(f"wall selector decomposition {end}/{len(query_ids)}", flush=True)

    active = arrays["active"].astype(bool)
    point_index, path_index = np.nonzero(active)
    source = arrays["source"][point_index, path_index].astype(np.complex64)
    rotated_source = arrays["rotated_source"][point_index, path_index].astype(np.complex64)
    wrong_source = arrays["wrong_beta_source"][point_index, path_index].astype(np.complex64)
    base = base_coefficient[point_index, path_index]
    truth = truth_coefficient[point_index, path_index]
    candidate = phase_mix_numpy(base, source, config.phase_alpha)
    truth_energy = np.sum(np.abs(truth) ** 2, axis=1, dtype=np.float64)
    base_error = np.sum(np.abs(base - truth) ** 2, axis=1, dtype=np.float64)
    candidate_error = np.sum(
        np.abs(candidate - truth) ** 2, axis=1, dtype=np.float64
    )
    target = (base_error - candidate_error) / np.maximum(truth_energy, 1e-20)
    target = np.clip(target, -config.target_clip, config.target_clip)
    path_energy = np.sum(np.abs(base_coefficient) ** 2, axis=2, dtype=np.float64)
    path_fraction = path_energy / np.maximum(
        np.sum(path_energy, axis=1, keepdims=True), 1e-20
    )

    wall_bank = build_wall_bank(MapConfig())
    image_source = {
        site_value: image_sources(wall_bank, setup.bs_positions[site_value])
        for site_value in range(2)
    }
    fold_parameters = load_fold_parameters()
    features: list[list[float]] = []
    feature_names = [
        "x",
        "y",
        "relative_x",
        "relative_y",
        "radius",
        "azimuth_sin",
        "azimuth_cos",
        "path_rank",
        "horizontal_sin",
        "horizontal_cos",
        "vertical_sin",
        "vertical_cos",
        "delay_sin",
        "delay_cos",
        "fit_coherence",
        "rotated_coherence",
        "coherence_margin",
        "log_observation_count",
        "nearest_distance_m",
        "delay_error_bins",
        "horizontal_error_bins",
        "wall_normal_x",
        "wall_normal_y",
        "wall_offset",
        "wall_tangent_width",
        "wall_height",
        "log_wall_area",
        "log_wall_faces",
        "source_relative_x",
        "source_relative_y",
        "source_distance",
        "log_path_energy",
        "path_energy_fraction",
        "log_point_base_power",
        "base_source_real",
        "base_source_abs",
        "base_rotated_real",
        "base_wrong_real",
        "source_rotated_real",
        "source_wrong_real",
    ] + [f"port_fraction_{index}" for index in range(8)]
    base_source_real, base_source_abs = vector_cosine(base, source)
    base_rotated_real, _ = vector_cosine(base, rotated_source)
    base_wrong_real, _ = vector_cosine(base, wrong_source)
    source_rotated_real, _ = vector_cosine(source, rotated_source)
    source_wrong_real, _ = vector_cosine(source, wrong_source)
    port_energy = np.abs(base) ** 2
    port_fraction = port_energy / np.maximum(
        np.sum(port_energy, axis=1, keepdims=True), 1e-20
    )
    for row, (point_row, path) in enumerate(zip(point_index, path_index)):
        query_id = int(query_ids[point_row])
        site_value = int(site[point_row])
        wall_id = int(arrays["path_id"][point_row, path])
        point = train_positions[query_id]
        bs_position = setup.bs_positions[site_value]
        relative = point[:2] - bs_position[:2]
        radius = float(np.linalg.norm(relative))
        azimuth = math.atan2(float(relative[1]), float(relative[0]))
        coordinate = arrays["coordinates"][point_row, path]
        source_position = image_source[site_value][wall_id]
        orientation = fold_parameters[(int(fold_id[point_row]), site_value)][
            "orientation"
        ]
        length, expected_h = selected_wall_geometry(
            point,
            bs_position,
            source_position,
            wall_bank.normal[wall_id],
            float(wall_bank.offset[wall_id]),
            float(orientation),
        )
        delay_error = float(
            circular_abs(
                np.asarray(coordinate[2]),
                np.asarray(length / METERS_PER_DELAY_BIN),
                192.0,
            )
        )
        horizontal_error = float(
            circular_abs(np.asarray(coordinate[0]), np.asarray(expected_h), 8.0)
        )
        source_relative = source_position[:2] - bs_position[:2]
        feature = [
            float(point[0]),
            float(point[1]),
            float(relative[0]),
            float(relative[1]),
            radius,
            math.sin(azimuth),
            math.cos(azimuth),
            float(path) / max(config.paths - 1, 1),
            math.sin(2.0 * math.pi * float(coordinate[0]) / 8.0),
            math.cos(2.0 * math.pi * float(coordinate[0]) / 8.0),
            math.sin(2.0 * math.pi * float(coordinate[1]) / 16.0),
            math.cos(2.0 * math.pi * float(coordinate[1]) / 16.0),
            math.sin(2.0 * math.pi * float(coordinate[2]) / 192.0),
            math.cos(2.0 * math.pi * float(coordinate[2]) / 192.0),
            float(arrays["fit_coherence"][point_row, path]),
            float(arrays["rotated_coherence"][point_row, path]),
            float(
                arrays["fit_coherence"][point_row, path]
                - arrays["rotated_coherence"][point_row, path]
            ),
            math.log1p(float(arrays["observation_count"][point_row, path])),
            float(arrays["nearest_distance"][point_row, path]),
            delay_error,
            horizontal_error,
            float(wall_bank.normal[wall_id, 0]),
            float(wall_bank.normal[wall_id, 1]),
            float(wall_bank.offset[wall_id]),
            float(wall_bank.tangent_max[wall_id] - wall_bank.tangent_min[wall_id]),
            float(wall_bank.z_max[wall_id] - wall_bank.z_min[wall_id]),
            math.log1p(float(wall_bank.area[wall_id])),
            math.log1p(float(wall_bank.face_count[wall_id])),
            float(source_relative[0]),
            float(source_relative[1]),
            float(np.linalg.norm(source_relative)),
            math.log(max(float(path_energy[point_row, path]), 1e-30)),
            float(path_fraction[point_row, path]),
            math.log(max(float(point_base_power[point_row]), 1e-30)),
            float(base_source_real[row]),
            float(base_source_abs[row]),
            float(base_rotated_real[row]),
            float(base_wrong_real[row]),
            float(source_rotated_real[row]),
            float(source_wrong_real[row]),
            *[float(value) for value in port_fraction[row]],
        ]
        features.append(feature)

    output = {
        "features": np.asarray(features, dtype=np.float32),
        "feature_names": np.asarray(feature_names),
        "target": target.astype(np.float32),
        "truth_energy": truth_energy.astype(np.float64),
        "base_error": base_error.astype(np.float64),
        "candidate_error": candidate_error.astype(np.float64),
        "point_index": point_index.astype(np.int16),
        "path_index": path_index.astype(np.int8),
        "fold_id": fold_id[point_index].astype(np.int8),
        "site": site[point_index].astype(np.int8),
        "path_id": arrays["path_id"][point_index, path_index].astype(np.int32),
        "query_ids": query_ids,
        "base_coefficient": base_coefficient,
        "truth_coefficient": truth_coefficient,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEATURE_CACHE, **output)
    return output


def family_encoding(
    family: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    train: np.ndarray,
    query: np.ndarray,
    prior_observations: float,
    leave_one_out: bool,
) -> tuple[np.ndarray, np.ndarray]:
    train_weight = weight[train]
    global_mean = float(
        np.sum(train_weight * target[train]) / max(float(np.sum(train_weight)), TINY)
    )
    prior_mass = prior_observations * max(float(np.median(train_weight)), 1e-20)
    total_weight: dict[int, float] = {}
    weighted_target: dict[int, float] = {}
    count: dict[int, int] = {}
    for index in np.flatnonzero(train):
        key = int(family[index])
        total_weight[key] = total_weight.get(key, 0.0) + float(weight[index])
        weighted_target[key] = weighted_target.get(key, 0.0) + float(
            weight[index] * target[index]
        )
        count[key] = count.get(key, 0) + 1
    encoded = np.empty(int(np.sum(query)), dtype=np.float32)
    support = np.empty_like(encoded)
    for output_row, index in enumerate(np.flatnonzero(query)):
        key = int(family[index])
        mass = total_weight.get(key, 0.0)
        value = weighted_target.get(key, 0.0)
        observations = count.get(key, 0)
        if leave_one_out and train[index]:
            mass -= float(weight[index])
            value -= float(weight[index] * target[index])
            observations -= 1
        encoded[output_row] = np.float32(
            (value + prior_mass * global_mean) / max(mass + prior_mass, 1e-20)
        )
        support[output_row] = np.float32(math.log1p(max(observations, 0)))
    return encoded, support


def permute_targets(
    target: np.ndarray,
    site: np.ndarray,
    weight: np.ndarray,
    train: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.array(target, copy=True)
    log_weight = np.log(np.maximum(weight, 1e-30))
    for site_value in np.unique(site[train]):
        local = np.flatnonzero(train & (site == site_value))
        edges = np.quantile(log_weight[local], [0.25, 0.50, 0.75])
        group = np.digitize(log_weight[local], edges)
        for group_value in range(4):
            rows = local[group == group_value]
            if len(rows) > 1:
                output[rows] = output[rng.permutation(rows)]
    return output


def fit_model(
    features: np.ndarray,
    family: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    train: np.ndarray,
    held: np.ndarray,
    config: Config,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_encoding, train_support = family_encoding(
        family,
        target,
        weight,
        train,
        train,
        config.family_prior_observations,
        True,
    )
    held_encoding, held_support = family_encoding(
        family,
        target,
        weight,
        train,
        held,
        config.family_prior_observations,
        False,
    )
    train_features = np.column_stack(
        [features[train], train_encoding, train_support]
    ).astype(np.float32)
    held_features = np.column_stack(
        [features[held], held_encoding, held_support]
    ).astype(np.float32)
    median_weight = max(float(np.median(weight[train])), 1e-20)
    sample_weight = np.clip(weight[train] / median_weight, 0.1, 20.0)
    model = ExtraTreesRegressor(
        n_estimators=config.trees,
        min_samples_leaf=config.minimum_leaf,
        max_features=config.maximum_features,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(train_features, target[train], sample_weight=sample_weight)
    return model.predict(held_features).astype(np.float32), held_encoding


def selector_summary(
    name: str,
    prediction: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    fold_id: np.ndarray,
) -> dict[str, object]:
    selected = prediction > 0.0
    positive = target > 0.0
    total_positive_gain = float(np.sum(weight * np.maximum(target, 0.0)))
    selected_gain = float(np.sum(weight[selected] * target[selected]))
    selected_positive_gain = float(
        np.sum(weight[selected] * np.maximum(target[selected], 0.0))
    )
    denominator = math.sqrt(
        max(float(np.sum(weight * prediction * prediction)), TINY)
        * max(float(np.sum(weight * target * target)), TINY)
    )
    correlation = float(np.sum(weight * prediction * target) / denominator)
    return {
        "name": name,
        "selected_path_fraction": float(np.mean(selected)),
        "selected_energy_fraction": float(
            np.sum(weight[selected]) / max(float(np.sum(weight)), TINY)
        ),
        "weighted_positive_precision": float(
            np.sum(weight[selected & positive]) / max(float(np.sum(weight[selected])), TINY)
        ),
        "net_normalized_gain": selected_gain / max(float(np.sum(weight)), TINY),
        "positive_gain_recall": selected_positive_gain / max(total_positive_gain, TINY),
        "weighted_cosine_prediction_target": correlation,
        "fold_net_normalized_gain": [
            float(
                np.sum(weight[(fold_id == fold) & selected] * target[(fold_id == fold) & selected])
                / max(float(np.sum(weight[fold_id == fold])), TINY)
            )
            for fold in range(5)
        ],
    }


def build_selectors(
    feature: dict[str, np.ndarray], config: Config, rebuild: bool
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if SELECTOR_CACHE.is_file() and not rebuild:
        cached = {key: value for key, value in np.load(SELECTOR_CACHE).items()}
        diagnostics = json.loads(str(cached.pop("diagnostics_json").item()))
        return cached, diagnostics
    features = feature["features"].astype(np.float32)
    target = feature["target"].astype(np.float32)
    weight = feature["truth_energy"].astype(np.float64)
    fold_id = feature["fold_id"].astype(np.int8)
    site = feature["site"].astype(np.int8)
    family = feature["path_id"].astype(np.int32)
    model_prediction = np.zeros(len(target), dtype=np.float32)
    shuffled_prediction = np.zeros_like(model_prediction)
    family_prediction = np.zeros_like(model_prediction)
    rng = np.random.default_rng(config.seed)
    for held_fold in range(5):
        train = fold_id != held_fold
        held = fold_id == held_fold
        prediction, prior = fit_model(
            features,
            family,
            target,
            weight,
            train,
            held,
            config,
            config.seed + held_fold * 1009,
        )
        shuffled_target = permute_targets(target, site, weight, train, rng)
        shuffled, _ = fit_model(
            features,
            family,
            shuffled_target,
            weight,
            train,
            held,
            config,
            config.seed + 50_000 + held_fold * 1009,
        )
        model_prediction[held] = prediction
        shuffled_prediction[held] = shuffled
        family_prediction[held] = prior
        print(
            f"wall selector held_fold={held_fold} paths={int(np.sum(held))}",
            flush=True,
        )
    selectors = {
        "model_prediction": model_prediction,
        "shuffled_prediction": shuffled_prediction,
        "family_prediction": family_prediction,
        "oracle_prediction": target,
    }
    diagnostics = {
        "target": {
            "path_count": int(len(target)),
            "positive_path_fraction": float(np.mean(target > 0.0)),
            "positive_energy_fraction": float(
                np.sum(weight[target > 0.0]) / max(float(np.sum(weight)), TINY)
            ),
            "oracle_net_normalized_gain": float(
                np.sum(weight * np.maximum(target, 0.0))
                / max(float(np.sum(weight)), TINY)
            ),
        },
        "selectors": [
            selector_summary(name, prediction, target, weight, fold_id)
            for name, prediction in selectors.items()
        ],
    }
    np.savez_compressed(
        SELECTOR_CACHE,
        **selectors,
        diagnostics_json=np.asarray(json.dumps(diagnostics)),
    )
    return selectors, diagnostics


def build_point_masks(
    arrays: dict[str, np.ndarray],
    feature: dict[str, np.ndarray],
    selectors: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    shape = arrays["active"].shape
    masks: dict[str, np.ndarray] = {
        "all_active": arrays["active"].astype(bool),
    }
    gate_names, gate_values = gate_masks(arrays)
    masks["quality_top75"] = gate_values[gate_names.index("quality_top75")]
    for name, key in (
        ("family_selector", "family_prediction"),
        ("model_selector", "model_prediction"),
        ("shuffled_selector", "shuffled_prediction"),
        ("oracle_selector", "oracle_prediction"),
    ):
        mask = np.zeros(shape, dtype=bool)
        mask[
            feature["point_index"].astype(np.int64),
            feature["path_index"].astype(np.int64),
        ] = selectors[key] > 0.0
        masks[name] = mask
    return masks


def candidate_rows(
    metric: dict[str, np.ndarray], fold_id: np.ndarray
) -> list[dict[str, object]]:
    rows = []
    for index, name in enumerate(VARIANT_NAMES):
        overall = statistics(
            metric["pas"][index],
            metric["pdp"][index],
            metric["error"][index],
            metric["energy"][index],
        )
        folds = [
            {
                "fold": fold,
                **statistics(
                    metric["pas"][index],
                    metric["pdp"][index],
                    metric["error"][index],
                    metric["energy"][index],
                    fold_id == fold,
                ),
            }
            for fold in range(5)
        ]
        rows.append({"name": name, **overall, "folds": folds})
    baseline = rows[0]
    for row in rows:
        row["delta_vs_identity"] = float(row["score"] - baseline["score"])
        row["fold_delta_vs_identity"] = [
            float(current["score"] - old["score"])
            for current, old in zip(row["folds"], baseline["folds"])
        ]
        row["positive_fold_count"] = int(
            np.sum(np.asarray(row["fold_delta_vs_identity"]) > 0.0)
        )
    return rows


def nested_identity_model(
    metric: dict[str, np.ndarray], fold_id: np.ndarray, site: np.ndarray
) -> dict[str, object]:
    selectable = (0, VARIANT_NAMES.index("model_selector_correct"))
    selected = np.zeros(len(fold_id), dtype=np.int64)
    decisions = []
    for held_fold in range(5):
        for site_value in range(2):
            train = (fold_id != held_fold) & (site == site_value)
            held = (fold_id == held_fold) & (site == site_value)
            score = [
                statistics(
                    metric["pas"][index],
                    metric["pdp"][index],
                    metric["error"][index],
                    metric["energy"][index],
                    train,
                )["score"]
                for index in selectable
            ]
            chosen = int(selectable[int(np.argmax(score))])
            selected[held] = chosen
            decisions.append(
                {
                    "held_fold": held_fold,
                    "site": site_value,
                    "candidate": VARIANT_NAMES[chosen],
                    "training_score": float(max(score)),
                }
            )
    point = np.arange(len(fold_id))
    result = statistics(
        metric["pas"][selected, point],
        metric["pdp"][selected, point],
        metric["error"][selected, point],
        metric["energy"][selected, point],
    )
    baseline = statistics(
        metric["pas"][0], metric["pdp"][0], metric["error"][0], metric["energy"][0]
    )
    fold_delta = []
    for fold in range(5):
        mask = fold_id == fold
        current = statistics(
            metric["pas"][selected, point],
            metric["pdp"][selected, point],
            metric["error"][selected, point],
            metric["energy"][selected, point],
            mask,
        )
        old = statistics(
            metric["pas"][0],
            metric["pdp"][0],
            metric["error"][0],
            metric["energy"][0],
            mask,
        )
        fold_delta.append(float(current["score"] - old["score"]))
    result["delta_vs_identity"] = float(result["score"] - baseline["score"])
    result["fold_delta_vs_identity"] = fold_delta
    result["positive_fold_count"] = int(np.sum(np.asarray(fold_delta) > 0.0))
    result["decisions"] = decisions
    return result


@torch.inference_mode()
def evaluate_full_replay(
    arrays: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    config: Config,
    device: torch.device,
    batch_size: int,
    rebuild: bool,
) -> dict[str, object]:
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    site = arrays["site"].astype(np.int8)
    if METRIC_CACHE.is_file() and not rebuild:
        cached = np.load(METRIC_CACHE)
        metric = {key: cached[key] for key in ("pas", "pdp", "error", "energy")}
    else:
        base_all = np.load(OOF_BASE, mmap_mode="r")
        truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
        v4_reference = np.load(V4_OOF_CHANNEL, mmap_mode="r")
        ad_cache = np.load(TARGET_AD, mmap_mode="r")
        pas_cache = np.load(TARGET_PAS, mmap_mode="r")
        pdp_cache = np.load(TARGET_PDP, mmap_mode="r")
        calibrated_pas_cache = np.load(OOF_CALIBRATED_PAS, mmap_mode="r")
        calibrated_pdp_cache = np.load(OOF_CALIBRATED_PDP, mmap_mode="r")
        decisions = load_v4_decisions()
        metric = {
            key: np.zeros((len(VARIANT_NAMES), len(query_ids)), dtype=np.float64)
            for key in ("pas", "pdp", "error", "energy")
        }
        replay_config = ReplayConfig()
        identity_max_difference = 0.0
        for start in range(0, len(query_ids), batch_size):
            end = min(start + batch_size, len(query_ids))
            ad, raw_pas, raw_pdp = target_batch(
                ad_cache, pas_cache, pdp_cache, start, end, device
            )
            calibrated_pas = torch.as_tensor(
                np.asarray(calibrated_pas_cache[start:end], dtype=np.float32),
                device=device,
            )
            calibrated_pdp = torch.as_tensor(
                np.asarray(calibrated_pdp_cache[start:end], dtype=np.float32),
                device=device,
            )
            base = torch.as_tensor(
                np.array(base_all[start:end], dtype=np.complex64, copy=True), device=device
            )
            truth = torch.as_tensor(
                np.array(truth_all[query_ids[start:end]], dtype=np.complex64, copy=True),
                device=device,
            )
            coefficient, residual, atoms = fixed_coordinate_decomposition(
                base,
                arrays["coordinates"][start:end],
                replay_config.decomposition_ridge,
                device,
            )
            variant_definition = {
                "identity": None,
                "quality_top75_correct": ("quality_top75", "source"),
                "all_active_correct": ("all_active", "source"),
                "family_selector_correct": ("family_selector", "source"),
                "model_selector_correct": ("model_selector", "source"),
                "shuffled_selector_correct": ("shuffled_selector", "source"),
                "model_selector_rotated_control": (
                    "model_selector",
                    "rotated_source",
                ),
                "model_selector_wrong_beta_control": (
                    "model_selector",
                    "wrong_beta_source",
                ),
                "oracle_selector_correct": ("oracle_selector", "source"),
            }
            preprojection: dict[str, torch.Tensor] = {"identity": base}
            for name, definition in variant_definition.items():
                if definition is None:
                    continue
                mask_name, source_name = definition
                mask = torch.as_tensor(masks[mask_name][start:end], device=device)
                source = torch.as_tensor(arrays[source_name][start:end], device=device)
                modified = mix_full_phase(
                    coefficient, source, mask, config.phase_alpha
                )
                ports = residual + torch.einsum(
                    "bkc,bkhvs->bchvs", modified, atoms
                )
                preprojection[name] = preserve_power(ports_to_channel(ports), base)
            truth_numpy = truth.cpu().numpy()
            for index, name in enumerate(VARIANT_NAMES):
                prediction = replay_v4(
                    preprojection[name],
                    base,
                    ad,
                    raw_pas,
                    raw_pdp,
                    calibrated_pas,
                    calibrated_pdp,
                    fold_id[start:end],
                    site[start:end],
                    decisions,
                )
                point = point_metric_statistics(prediction.cpu().numpy(), truth_numpy)
                metric["pas"][index, start:end] = point.pas
                metric["pdp"][index, start:end] = point.pdp
                metric["error"][index, start:end] = point.squared_error
                metric["energy"][index, start:end] = point.truth_energy
                if name == "identity":
                    reference = torch.as_tensor(
                        np.array(v4_reference[start:end], dtype=np.complex64, copy=True),
                        device=device,
                    )
                    identity_max_difference = max(
                        identity_max_difference,
                        float((prediction - reference).abs().max().item()),
                    )
            if end % 20 == 0 or end == len(query_ids):
                print(f"supervised wall replay {end}/{len(query_ids)}", flush=True)
        np.savez_compressed(
            METRIC_CACHE,
            **metric,
            variant_names=np.asarray(VARIANT_NAMES),
            fold_id=fold_id,
            site=site,
            identity_max_difference=np.asarray(identity_max_difference),
        )

    rows = candidate_rows(metric, fold_id)
    by_name = {str(row["name"]): row for row in rows}
    nested = nested_identity_model(metric, fold_id, site)
    model = by_name["model_selector_correct"]
    controls = (
        by_name["shuffled_selector_correct"],
        by_name["model_selector_rotated_control"],
        by_name["model_selector_wrong_beta_control"],
    )
    deployment_ready = bool(
        float(nested["score"]) >= config.deployment_score_threshold
        and int(nested["positive_fold_count"]) >= config.deployment_positive_folds
        and all(float(model["score"]) > float(row["score"]) for row in controls)
    )
    return {
        "candidates": sorted(rows, key=lambda row: float(row["score"]), reverse=True),
        "nested_identity_model": nested,
        "deployment_ready": deployment_ready,
        "deployment_rule": {
            "minimum_nested_score": config.deployment_score_threshold,
            "minimum_nested_positive_folds": config.deployment_positive_folds,
            "model_must_beat_shuffled_rotated_and_wrong_beta_controls": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    config = Config()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in np.load(PHASE_CACHE).items()}
    device = torch.device(args.device)
    feature = build_path_features(
        arrays, config, device, args.batch_size, args.rebuild
    )
    selectors, selector_diagnostics = build_selectors(
        feature, config, args.rebuild
    )
    masks = build_point_masks(arrays, feature, selectors)
    full_replay = evaluate_full_replay(
        arrays,
        masks,
        config,
        device,
        args.batch_size,
        args.rebuild,
    )
    report = {
        "experiment": VERSION,
        "status": "candidate_ready" if full_replay["deployment_ready"] else "analysis_only",
        "primary_hypothesis": (
            "Strict-OOF observable path features can identify which predicted wall "
            "identities have positive carrier-phase value, retaining the path oracle "
            "while rejecting mixed or incorrect wall assignments."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "config": asdict(config),
        "selector_diagnostics": selector_diagnostics,
        "full_replay": full_replay,
        "mask_fractions": {
            name: float(np.mean(mask)) for name, mask in masks.items()
        },
        "outputs": {
            "features": PATHS.relative(FEATURE_CACHE),
            "selectors": PATHS.relative(SELECTOR_CACHE),
            "metrics": PATHS.relative(METRIC_CACHE),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
