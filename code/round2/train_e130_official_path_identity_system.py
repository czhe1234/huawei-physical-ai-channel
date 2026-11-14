#!/usr/bin/env python3
"""Reconstruct path dynamics from official H and train a setwise identity selector."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import analyze_e114_path_state_variable_ceiling as e114
import analyze_e121_standalone_unique_path_carrier as e121
import analyze_e123_complete_path_variable_system as e123
import analyze_e124_one_to_one_path_identity_ceiling as e124
import analyze_e127_candidate_lifecycle_information as e127
import analyze_e128_path_aligned_map_evidence as e128
import train_e115_complete_path_state_operator as e115
import train_e118_hierarchical_unique_path_operator as e118
import train_e125_identity_first_path_operator as e125
from common.official_metrics import PointMetricStatistics, aggregate_statistics
from data import load_positions
from project_paths import PATHS


VERSION = "Round2-E130-official-path-identity-system"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "official_path_identity_system"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_official_path_identity_system"
REPORT = ANALYSIS_DIR / "official_path_identity_fivefold.json"
METRICS = ARTIFACT_DIR / "official_path_identity_fivefold_metrics.npz"
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"

IDENTITY_EPOCHS = 30
SELECTOR_EPOCHS = 18
VALIDATION_INTERVAL = 2
SELECTOR_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
TRAJECTORY_SPATIAL_SCALE_M = 12.0
MAXIMUM_DISTANCE_M = 90.0
TINY = 1e-30


BOUNDARY_NAMES = (
    "active_donor_fraction",
    "weighted_active_donor_fraction",
    "nearest_active_distance_m",
    "nearest_inactive_distance_m",
    "signed_boundary_margin_m",
    "active_fraction_within_3m",
    "active_fraction_within_6m",
    "active_fraction_within_12m",
    "active_fraction_within_24m",
    "active_distance_mean_m",
    "active_distance_std_m",
    "active_distance_max_m",
    "active_angular_coverage",
)
STABILITY_NAMES = (
    "member_count",
    "unique_active_donor_count",
    "members_per_active_donor_mean",
    "members_per_active_donor_std",
    "members_per_active_donor_max",
    "path_mass_mean",
    "path_mass_std",
    "path_mass_coefficient_variation",
    "horizontal_coordinate_std_bin",
    "vertical_coordinate_std_bin",
    "delay_coordinate_std_bin",
    "active_position_isotropy",
)
MASS_JACOBIAN_NAMES = (
    "log_mass_intercept",
    "log_mass_gradient_x",
    "log_mass_gradient_y",
    "log_mass_rmse",
    "log_mass_r2",
    "idw_path_mass",
)
COORDINATE_JACOBIAN_NAMES = tuple(
    f"{axis}_{quantity}"
    for axis in ("horizontal", "vertical", "delay")
    for quantity in ("intercept_delta", "gradient_x", "gradient_y", "rmse", "r2")
)
JOINT_JACOBIAN_NAMES = (
    "coordinate_jacobian_norm",
    "horizontal_vertical_jacobian_determinant",
)
COMPETITION_NAMES = (
    "component_rank_mean",
    "component_rank_std",
    "component_rank_min",
    "member_share_mean",
    "member_share_max",
    "member_log_power_mean",
    "member_residual_fraction_mean",
    "candidate_member_mass_sum",
)
TRAJECTORY_NAMES = (
    BOUNDARY_NAMES
    + STABILITY_NAMES
    + MASS_JACOBIAN_NAMES
    + COORDINATE_JACOBIAN_NAMES
    + JOINT_JACOBIAN_NAMES
    + COMPETITION_NAMES
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--identity-epochs", type=int, default=IDENTITY_EPOCHS)
    parser.add_argument("--selector-epochs", type=int, default=SELECTOR_EPOCHS)
    parser.add_argument(
        "--validation-interval", type=int, default=VALIDATION_INTERVAL
    )
    parser.add_argument("--seed", type=int, default=20_260_808_130)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--force-models", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def periodic_delta(destination: np.ndarray, source: float, period: float) -> np.ndarray:
    return np.remainder(destination - source + period / 2.0, period) - period / 2.0


def weighted_linear_fit(
    offset: np.ndarray,
    value: np.ndarray,
    weight: np.ndarray,
    ridge: float = 0.08,
) -> tuple[float, float, float, float, float]:
    if len(value) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    x = np.column_stack((np.ones(len(offset)), offset[:, 0], offset[:, 1]))
    weight = np.maximum(np.asarray(weight, dtype=np.float64), 1e-8)
    gram = x.T @ (weight[:, None] * x)
    gram += np.diag((1e-6, ridge, ridge))
    cross = x.T @ (weight * value)
    try:
        coefficient = np.linalg.solve(gram, cross)
    except np.linalg.LinAlgError:
        coefficient = np.linalg.lstsq(gram, cross, rcond=None)[0]
    prediction = x @ coefficient
    total_weight = max(float(weight.sum()), TINY)
    rmse = math.sqrt(
        max(float(np.sum(weight * np.square(prediction - value)) / total_weight), 0.0)
    )
    mean = float(np.sum(weight * value) / total_weight)
    total = float(np.sum(weight * np.square(value - mean)))
    residual = float(np.sum(weight * np.square(prediction - value)))
    r2 = 1.0 - residual / max(total, 1e-8)
    return (
        float(coefficient[0]),
        float(coefficient[1]),
        float(coefficient[2]),
        float(rmse),
        float(np.clip(r2, -2.0, 1.0)),
    )


def circular_weighted_mean(
    value: np.ndarray, weight: np.ndarray, period: float
) -> float:
    angle = 2.0 * np.pi * value.astype(np.float64) / period
    sine = float(np.sum(weight * np.sin(angle)))
    cosine = float(np.sum(weight * np.cos(angle)))
    return float(np.mod(np.arctan2(sine, cosine), 2.0 * np.pi) * period / (2.0 * np.pi))


def aggregate_donor_state(
    member: np.ndarray,
    donor_inverse: np.ndarray,
    raw_coordinate: np.ndarray,
    raw_fraction: np.ndarray,
    donor_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active = np.unique(donor_inverse[member])
    mass = np.zeros(len(active), dtype=np.float64)
    coordinate = np.zeros((len(active), 3), dtype=np.float64)
    recurrence = np.zeros(len(active), dtype=np.float64)
    for local, donor in enumerate(active):
        selected = member[donor_inverse[member] == donor]
        local_mass = np.maximum(raw_fraction[selected], 1e-12)
        mass[local] = float(local_mass.sum())
        recurrence[local] = float(len(selected))
        for axis, period in enumerate(e115.PERIODS):
            coordinate[local, axis] = circular_weighted_mean(
                raw_coordinate[selected, axis], local_mass, float(period)
            )
    if np.any(active >= donor_count):
        raise RuntimeError("trajectory donor index exceeds donor table")
    return active, mass, coordinate, recurrence


def trajectory_cache_path(outer: int) -> Path:
    return ARTIFACT_DIR / f"outer{outer}_official_trajectory_features.npy"


def map_feature_cache_path(outer: int) -> Path:
    return ARTIFACT_DIR / f"outer{outer}_candidate_map_features.npy"


def build_official_trajectory_features(
    outer: int,
    directory: Path,
    force: bool,
    row_limit: int | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    output_path = trajectory_cache_path(outer)
    expected_shape = (4000, e118.Config().candidate_count, len(TRAJECTORY_NAMES))
    if output_path.is_file() and not force and row_limit is None:
        cached = np.load(output_path, mmap_mode="r")
        if cached.shape == expected_shape and cached.dtype == np.float16:
            return cached, {"reused": True, "shape": list(cached.shape)}

    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    source_directory = PATHS.root / state["source_cache"]
    raw_feature = np.load(source_directory / "candidate_feature.npy", mmap_mode="r")
    token_cluster = np.load(directory / "token_cluster.npy", mmap_mode="r")
    candidate_coordinate = np.load(directory / "coordinate.npy", mmap_mode="r")
    candidate_mask = np.load(directory / "mask.npy", mmap_mode="r")
    count = len(candidate_mask) if row_limit is None else min(row_limit, len(candidate_mask))
    output = np.zeros((count, candidate_mask.shape[1], len(TRAJECTORY_NAMES)), np.float32)
    donor_counts: list[int] = []
    active_counts: list[int] = []

    for row in range(count):
        cluster = np.asarray(token_cluster[row], dtype=np.int64)
        valid_token = cluster >= 0
        if not np.any(valid_token):
            continue
        feature = np.asarray(raw_feature[row], dtype=np.float32)
        raw_offset = feature[:, 9:11] * 30.0
        rounded_offset = np.round(raw_offset[valid_token], 2)
        donor_offset, valid_inverse = np.unique(
            rounded_offset, axis=0, return_inverse=True
        )
        donor_inverse = np.full(len(cluster), -1, dtype=np.int64)
        donor_inverse[valid_token] = valid_inverse
        donor_distance = np.linalg.norm(donor_offset, axis=1)
        donor_weight = np.exp(-donor_distance / TRAJECTORY_SPATIAL_SCALE_M)
        donor_weight /= max(float(donor_weight.sum()), TINY)
        donor_counts.append(len(donor_offset))

        raw_coordinate = feature[:, 6:9] * e115.PERIODS[None]
        raw_fraction = np.exp(np.clip(feature[:, 17], -30.0, 5.0))
        raw_component_rank = feature[:, 21] * max(e115.Config().donor_components - 1, 1)
        raw_log_power = feature[:, 22]
        raw_residual = feature[:, 23]
        donor_total = np.zeros(len(donor_offset), dtype=np.float64)
        for donor in range(len(donor_offset)):
            donor_total[donor] = float(raw_fraction[donor_inverse == donor].sum())

        for candidate in np.flatnonzero(candidate_mask[row]):
            member = np.flatnonzero(cluster == candidate)
            if len(member) == 0:
                continue
            active, mass, coordinate, recurrence = aggregate_donor_state(
                member,
                donor_inverse,
                raw_coordinate,
                raw_fraction,
                len(donor_offset),
            )
            active_counts.append(len(active))
            active_flag = np.zeros(len(donor_offset), dtype=bool)
            active_flag[active] = True
            active_distance = donor_distance[active]
            inactive_distance = donor_distance[~active_flag]
            nearest_active = float(np.min(active_distance, initial=MAXIMUM_DISTANCE_M))
            nearest_inactive = float(
                np.min(inactive_distance, initial=MAXIMUM_DISTANCE_M)
            )
            threshold_values = []
            for radius in (3.0, 6.0, 12.0, 24.0):
                local = donor_distance <= radius
                threshold_values.append(
                    float(np.mean(active_flag[local])) if np.any(local) else 0.0
                )
            if len(active) >= 2:
                centered = donor_offset[active] - donor_offset[active].mean(axis=0)
                covariance = centered.T @ centered / max(len(active), 1)
                eigenvalue = np.linalg.eigvalsh(covariance)
                isotropy = float(eigenvalue[0] / max(eigenvalue[-1], 1e-8))
                angle = np.arctan2(donor_offset[active, 1], donor_offset[active, 0])
                quadrant = (
                    (np.cos(angle) >= 0).astype(np.int8)
                    + 2 * (np.sin(angle) >= 0).astype(np.int8)
                )
                angular_coverage = float(len(np.unique(quadrant)) / 4.0)
            else:
                isotropy = 0.0
                angular_coverage = 0.25 if len(active) else 0.0
            boundary = (
                float(len(active) / max(len(donor_offset), 1)),
                float(donor_weight[active].sum()),
                nearest_active,
                nearest_inactive,
                nearest_inactive - nearest_active,
                *threshold_values,
                float(np.mean(active_distance)) if len(active_distance) else 0.0,
                float(np.std(active_distance)) if len(active_distance) else 0.0,
                float(np.max(active_distance, initial=0.0)),
                angular_coverage,
            )

            mass_mean = float(np.mean(mass))
            coordinate_delta = np.stack(
                [
                    periodic_delta(
                        coordinate[:, axis],
                        float(candidate_coordinate[row, candidate, axis]),
                        float(e115.PERIODS[axis]),
                    )
                    for axis in range(3)
                ],
                axis=1,
            )
            coordinate_std = np.std(coordinate_delta, axis=0)
            stability = (
                float(len(member)),
                float(len(active)),
                float(np.mean(recurrence)),
                float(np.std(recurrence)),
                float(np.max(recurrence, initial=0.0)),
                mass_mean,
                float(np.std(mass)),
                float(np.std(mass) / max(mass_mean, 1e-12)),
                float(coordinate_std[0]),
                float(coordinate_std[1]),
                float(coordinate_std[2]),
                isotropy,
            )
            active_weight = np.exp(-active_distance / TRAJECTORY_SPATIAL_SCALE_M)
            active_weight /= max(float(active_weight.sum()), TINY)
            mass_fit = weighted_linear_fit(
                donor_offset[active], np.log(np.maximum(mass, 1e-12)), active_weight
            )
            idw_mass = float(np.sum(active_weight * mass))
            mass_jacobian = (*mass_fit, idw_mass)
            coordinate_fits = [
                weighted_linear_fit(
                    donor_offset[active], coordinate_delta[:, axis], active_weight
                )
                for axis in range(3)
            ]
            coordinate_jacobian = tuple(value for fit in coordinate_fits for value in fit)
            gradient = np.asarray([[fit[1], fit[2]] for fit in coordinate_fits])
            joint_jacobian = (
                float(np.linalg.norm(gradient)),
                float(np.linalg.det(gradient[:2])),
            )
            member_share = raw_fraction[member] / np.maximum(
                donor_total[donor_inverse[member]], 1e-12
            )
            competition = (
                float(np.mean(raw_component_rank[member])),
                float(np.std(raw_component_rank[member])),
                float(np.min(raw_component_rank[member])),
                float(np.mean(member_share)),
                float(np.max(member_share, initial=0.0)),
                float(np.mean(raw_log_power[member])),
                float(np.mean(raw_residual[member])),
                float(np.sum(raw_fraction[member])),
            )
            values = boundary + stability + mass_jacobian + coordinate_jacobian + joint_jacobian + competition
            if len(values) != len(TRAJECTORY_NAMES):
                raise RuntimeError((len(values), len(TRAJECTORY_NAMES)))
            output[row, candidate] = np.nan_to_num(
                np.asarray(values, dtype=np.float32),
                nan=0.0,
                posinf=1e4,
                neginf=-1e4,
            )
        if (row + 1) % 500 == 0:
            print(f"E130 trajectory outer={outer} rows={row + 1}/{count}", flush=True)

    diagnostics = {
        "reused": False,
        "shape": list(output.shape),
        "feature_count": len(TRAJECTORY_NAMES),
        "donor_count_mean": float(np.mean(donor_counts)),
        "donor_count_min": int(min(donor_counts)) if donor_counts else 0,
        "donor_count_max": int(max(donor_counts)) if donor_counts else 0,
        "active_donor_count_mean": float(np.mean(active_counts)),
        "active_donor_count_median": float(np.median(active_counts)),
        "finite": bool(np.isfinite(output).all()),
    }
    if row_limit is None:
        np.save(output_path, output.astype(np.float16))
        return np.load(output_path, mmap_mode="r"), diagnostics
    return output, diagnostics


def build_map_features(
    outer: int,
    fit_rows: np.ndarray,
    directory: Path,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    component: np.lib.npyio.NpzFile,
    force: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    output_path = map_feature_cache_path(outer)
    coordinate = np.asarray(np.load(directory / "coordinate.npy", mmap_mode="r"), np.float32)
    mask = np.asarray(np.load(directory / "mask.npy", mmap_mode="r"), bool)
    expected = (len(positions), mask.shape[1], 22 + 2 * len(e128.META_NAMES))
    if output_path.is_file() and not force:
        cached = np.load(output_path, mmap_mode="r")
        if cached.shape == expected and cached.dtype == np.float16:
            return cached, {"reused": True, "shape": list(cached.shape)}

    previous_cache = e128.MAP_CACHE
    e128.MAP_CACHE = ARTIFACT_DIR / f"outer{outer}_fit_only_map_path_bank.npz"
    try:
        map_coordinate, map_metadata, map_valid, diagnostics = e128.build_map_cache(
            fit_rows,
            positions,
            site,
            bs_positions,
            component,
            force,
        )
    finally:
        e128.MAP_CACHE = previous_cache
    features = e128.map_aligned_features(
        coordinate, mask, map_coordinate, map_metadata, map_valid
    )
    if features.shape != expected or not np.isfinite(features).all():
        raise RuntimeError((features.shape, expected, np.isfinite(features).all()))
    np.save(output_path, features.astype(np.float16))
    return np.load(output_path, mmap_mode="r"), {
        "reused": False,
        "shape": list(features.shape),
        "map": diagnostics,
    }


def identity_checkpoint(outer: int) -> Path:
    return ARTIFACT_DIR / f"outer{outer}_identity_core.pt"


def selector_checkpoint(outer: int) -> Path:
    return ARTIFACT_DIR / f"outer{outer}_official_selector.pt"


def instantiate_identity_model(
    store: e118.ClusterTensorStore,
    config: e118.Config,
    device: torch.device,
) -> e118.HierarchicalUniquePathOperator:
    return e118.HierarchicalUniquePathOperator(
        store.raw.candidate_feature.shape[-1],
        store.cluster_feature.shape[-1],
        store.raw.context.shape[-1],
        store.graph_feature.shape[-1],
        config,
        e115.VARIANTS[0],
    ).to(device)


def load_or_train_identity(
    outer: int,
    store: e118.ClusterTensorStore,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    hard_mass: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[
    e118.HierarchicalUniquePathOperator,
    dict[str, torch.Tensor],
    dict[str, object],
]:
    path = identity_checkpoint(outer)
    if path.is_file() and not args.force_models:
        checkpoint = torch.load(path, map_location=args.device, weights_only=False)
        config = e118.Config(**checkpoint["config"])
        model = instantiate_identity_model(store, config, store.device)
        model.load_state_dict(checkpoint["state_dict"])
        normalization = {
            key: value.to(store.device) for key, value in checkpoint["normalization"].items()
        }
        return model, normalization, {
            "reused": True,
            "best_epoch": int(checkpoint["best_epoch"]),
        }

    seed = args.seed + 1009 * outer
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    normalization = store.normalization(fit_rows)
    config = e118.Config(
        hidden=192,
        heads=8,
        token_layers=3,
        candidate_layers=3,
        graph_layers=2,
        dropout=0.08,
        score_limit=6.0,
        random_state=seed,
    )
    model = instantiate_identity_model(store, config, store.device)
    stage, state = e125.train_stage(
        "core",
        model,
        e115.VARIANTS[0],
        store,
        fit_rows,
        validation_rows,
        normalization,
        hard_mass,
        args.identity_epochs,
        args.batch_size,
        args.validation_interval,
        seed,
    )
    model.load_state_dict(state)
    torch.save(
        {
            "experiment": VERSION,
            "outer": outer,
            "config": asdict(config),
            "state_dict": state,
            "normalization": {
                key: value.detach().cpu() for key, value in normalization.items()
            },
            "best_epoch": int(stage["best_epoch"]),
            "official_h_only": True,
            "sionna_or_v12_used": False,
        },
        path,
    )
    return model, normalization, {"reused": False, **stage}


def static_features(
    identity_logits: np.ndarray,
    identity_mass: np.ndarray,
    directory: Path,
    mask: np.ndarray,
) -> np.ndarray:
    base_log_mass = np.asarray(
        np.load(directory / "base_log_mass.npy", mmap_mode="r"), np.float32
    )
    member_count = np.asarray(
        np.load(directory / "member_count.npy", mmap_mode="r"), np.float32
    )
    coordinate = np.asarray(
        np.load(directory / "coordinate.npy", mmap_mode="r"), np.float32
    )
    cluster = np.asarray(
        np.load(directory / "cluster_feature.npy", mmap_mode="r"), np.float32
    )
    entropy = -np.sum(
        identity_mass * np.log(np.maximum(identity_mass, 1e-8)), axis=1
    )
    count = mask.sum(axis=1).astype(np.float32)
    order = np.argsort(-identity_logits, axis=1)
    rank = np.empty_like(identity_logits, dtype=np.float32)
    rank[np.arange(len(rank))[:, None], order] = np.arange(mask.shape[1], dtype=np.float32)
    phase = 2.0 * np.pi * coordinate / e115.PERIODS[None, None]
    unary = np.stack(
        (
            np.clip(identity_logits, -20.0, 20.0),
            np.log(np.maximum(identity_mass, 1e-8)),
            np.clip(base_log_mass, -20.0, 20.0),
            np.log1p(member_count),
            rank / max(mask.shape[1] - 1, 1),
            np.broadcast_to(entropy[:, None], mask.shape),
            np.broadcast_to(count[:, None] / mask.shape[1], mask.shape),
        ),
        axis=2,
    )
    return np.concatenate((unary, np.sin(phase), np.cos(phase), cluster), axis=2).astype(
        np.float32
    )


def candidate_shuffle_dynamic(
    features: np.ndarray,
    mask: np.ndarray,
    static_dim: int,
    seed: int,
) -> np.ndarray:
    output = np.array(features, copy=True)
    rng = np.random.default_rng(seed)
    for row in range(len(output)):
        ids = np.flatnonzero(mask[row])
        if len(ids) <= 1:
            continue
        shift = int(rng.integers(1, len(ids)))
        output[row, ids, static_dim:] = features[row, np.roll(ids, shift), static_dim:]
    return output


class OfficialPathIdentitySelector(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 192, heads: int = 8) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=3 * hidden,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.norm = nn.LayerNorm(hidden)
        self.residual = nn.Linear(hidden, 1)
        self.coverage = nn.Linear(hidden, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.coverage.weight)
        nn.init.constant_(self.coverage.bias, 4.0)

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.input(features)
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        hidden = self.norm(hidden)
        residual = 6.0 * torch.tanh(self.residual(hidden)[..., 0])
        valid = mask.to(hidden.dtype)
        pooled = (hidden * valid[..., None]).sum(dim=1) / valid.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        coverage = self.coverage(pooled)[:, 0]
        return residual, coverage


def feature_normalization(
    features: np.ndarray, mask: np.ndarray, fit_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    valid = mask[fit_rows]
    values = features[fit_rows][valid].astype(np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-4] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def selector_output(
    batch: dict[str, torch.Tensor],
    base_logits: torch.Tensor,
    residual: torch.Tensor,
    coverage_logit: torch.Tensor,
    alpha: float,
) -> dict[str, torch.Tensor]:
    output = e125.base_output(batch)
    logits = (base_logits + alpha * residual).masked_fill(~batch["mask"], -1e4)
    output["mass_logits"] = logits
    output["mass"] = torch.softmax(logits, dim=1)
    output["score_residual"] = alpha * residual
    coverage_probability = torch.sigmoid(coverage_logit).clamp_min(1e-6)
    output["log_power"] = batch["base_log_power"] + 2.0 * alpha * torch.log(
        coverage_probability
    )
    return output


def full_score_loss(
    output: dict[str, torch.Tensor], truth: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = e115.render_complex_channel(output)
    pas, pdp, error, energy = e115.torch_official_metrics(prediction, truth)
    nmse = error.sum() / energy.sum().clamp_min(TINY)
    score = 0.4 * pas.mean() + 0.4 * pdp.mean() + 0.2 / (1.0 + nmse)
    return 1.0 - score, {
        "score": float(score.detach().cpu()),
        "pas": float(pas.mean().detach().cpu()),
        "pdp": float(pdp.mean().detach().cpu()),
        "nmse": float(nmse.detach().cpu()),
    }


@torch.inference_mode()
def selector_predictions(
    model: OfficialPathIdentitySelector,
    features: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    residual_parts: list[np.ndarray] = []
    coverage_parts: list[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        residual, coverage = model(features[start:stop], mask[start:stop])
        residual_parts.append(residual.cpu().numpy())
        coverage_parts.append(coverage.cpu().numpy())
    return np.concatenate(residual_parts), np.concatenate(coverage_parts)


@torch.inference_mode()
def evaluate_variant(
    residual: np.ndarray,
    coverage: np.ndarray,
    base_logits: np.ndarray,
    store: e118.ClusterTensorStore,
    rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    hard_mass: torch.Tensor,
    alpha: float,
    batch_size: int,
    collect_point_metrics: bool = False,
) -> tuple[dict[str, object], dict[str, np.ndarray] | None]:
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    pas_parts: list[np.ndarray] = []
    pdp_parts: list[np.ndarray] = []
    error_parts: list[np.ndarray] = []
    energy_parts: list[np.ndarray] = []
    affinity_parts: list[np.ndarray] = []
    top16_parts: list[np.ndarray] = []
    active_parts: list[np.ndarray] = []
    logit_parts: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        local = rows[start : start + batch_size]
        index = torch.as_tensor(local, dtype=torch.long, device=store.device)
        batch = e125.hard_batch(store, index, normalization, hard_mass)
        local_residual = torch.as_tensor(residual[local], device=store.device)
        local_coverage = torch.as_tensor(coverage[local], device=store.device)
        local_base = torch.as_tensor(base_logits[local], device=store.device)
        output = selector_output(
            batch, local_base, local_residual, local_coverage, alpha
        )
        truth = torch.as_tensor(
            np.asarray(truth_all[local], dtype=np.complex64), device=store.device
        )
        prediction = e115.render_complex_channel(output)
        pas, pdp, error, energy = e115.torch_official_metrics(prediction, truth)
        pas_parts.append(pas.cpu().numpy())
        pdp_parts.append(pdp.cpu().numpy())
        error_parts.append(error.cpu().numpy())
        energy_parts.append(energy.cpu().numpy())
        target_absolute = batch["target_mass"]
        target = target_absolute / target_absolute.sum(dim=1, keepdim=True).clamp_min(TINY)
        nonzero = target_absolute.sum(dim=1) > 0.0
        affinity = (
            torch.sqrt(output["mass"].clamp_min(1e-12))
            * torch.sqrt(target.clamp_min(0.0))
        ).sum(dim=1)
        affinity_parts.append(affinity[nonzero].cpu().numpy())
        top16_parts.append(
            e125.top_mass_recall(output["mass"], target, 16)[nonzero].cpu().numpy()
        )
        active_parts.append((target_absolute > 0.0).cpu().numpy())
        logit_parts.append(output["mass_logits"].cpu().numpy())
    values = PointMetricStatistics(
        pas=np.concatenate(pas_parts),
        pdp=np.concatenate(pdp_parts),
        squared_error=np.concatenate(error_parts),
        truth_energy=np.concatenate(energy_parts),
    )
    metrics = aggregate_statistics(values)
    metrics.update(
        {
            "count": int(len(rows)),
            "mass_bhattacharyya": float(np.mean(np.concatenate(affinity_parts))),
            "top16_target_mass_recall": float(np.mean(np.concatenate(top16_parts))),
        }
    )
    active = np.concatenate(active_parts)
    logits = np.concatenate(logit_parts)
    valid = np.asarray(store.mask[torch.as_tensor(rows, device=store.device)].cpu())
    metrics["existence_ap"] = e127.average_precision_score(
        active[valid].astype(np.int8), logits[valid]
    )
    point = None
    if collect_point_metrics:
        point = {
            "pas": values.pas,
            "pdp": values.pdp,
            "squared_error": values.squared_error,
            "truth_energy": values.truth_energy,
        }
    return metrics, point


def train_selector(
    outer: int,
    store: e118.ClusterTensorStore,
    features: np.ndarray,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    hard_mass: torch.Tensor,
    base_logits: np.ndarray,
    args: argparse.Namespace,
) -> tuple[
    OfficialPathIdentitySelector,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    path = selector_checkpoint(outer)
    mean, std = feature_normalization(features, np.asarray(store.mask.cpu()), fit_rows)
    normalized = np.clip((features.astype(np.float32) - mean) / std, -8.0, 8.0)
    feature_tensor = torch.as_tensor(normalized, device=store.device)
    mask = store.mask
    if path.is_file() and not args.force_models:
        checkpoint = torch.load(path, map_location=args.device, weights_only=False)
        model = OfficialPathIdentitySelector(features.shape[-1]).to(store.device)
        model.load_state_dict(checkpoint["state_dict"])
        residual, coverage = selector_predictions(
            model, feature_tensor, mask, args.batch_size
        )
        return model, residual, coverage, {
            "reused": True,
            "best_epoch": int(checkpoint["best_epoch"]),
            "selected_alpha": float(checkpoint["selected_alpha"]),
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
        }

    seed = args.seed + 2029 * outer
    torch.manual_seed(seed)
    model = OfficialPathIdentitySelector(features.shape[-1]).to(store.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=3e-4)
    train_h = np.load(TRAIN_CHANNEL, mmap_mode="r")
    base_logits_tensor = torch.as_tensor(base_logits, device=store.device)
    best_score = -np.inf
    best_epoch = 0
    best_alpha = 0.0
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    history: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for epoch in range(1, args.selector_epochs + 1):
        model.train()
        shuffled = rng.permutation(fit_rows)
        losses = []
        score_values = []
        for start in range(0, len(shuffled), args.batch_size):
            local = shuffled[start : start + args.batch_size]
            index = torch.as_tensor(local, dtype=torch.long, device=store.device)
            batch = e125.hard_batch(store, index, normalization, hard_mass)
            optimizer.zero_grad(set_to_none=True)
            residual, coverage = model(feature_tensor[index], batch["mask"])
            output = selector_output(
                batch, base_logits_tensor[index], residual, coverage, 1.0
            )
            identity_loss, _ = e125.identity_loss(output, batch)
            truth = torch.as_tensor(
                np.asarray(train_h[local], dtype=np.complex64), device=store.device
            )
            official_loss, official = full_score_loss(output, truth)
            nonzero_target = (batch["point_power"] > 0.0).to(coverage.dtype)
            zero_weight = torch.where(
                nonzero_target > 0.5,
                torch.ones_like(nonzero_target),
                torch.full_like(nonzero_target, 6.0),
            )
            coverage_loss = F.binary_cross_entropy_with_logits(
                coverage, nonzero_target, weight=zero_weight
            )
            loss = 0.55 * identity_loss + official_loss + 0.08 * coverage_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            score_values.append(official["score"])
        if epoch % args.validation_interval == 0 or epoch == args.selector_epochs:
            model.eval()
            residual_all, coverage_all = selector_predictions(
                model, feature_tensor, mask, args.batch_size
            )
            alpha_rows = []
            for alpha in SELECTOR_ALPHAS:
                metrics, _ = evaluate_variant(
                    residual_all,
                    coverage_all,
                    base_logits,
                    store,
                    validation_rows,
                    normalization,
                    hard_mass,
                    alpha,
                    args.batch_size,
                )
                alpha_rows.append({"alpha": alpha, "metrics": metrics})
            winner = max(alpha_rows, key=lambda row: row["metrics"]["score"])
            record = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_batch_score": float(np.mean(score_values)),
                "alpha_results": alpha_rows,
                "selected_alpha": float(winner["alpha"]),
                "selected_validation_score": float(winner["metrics"]["score"]),
            }
            history.append(record)
            print(
                f"E130 outer={outer} selector epoch={epoch}/{args.selector_epochs} "
                f"alpha={winner['alpha']:.2f} validation={winner['metrics']['score']:.6f}",
                flush=True,
            )
            if float(winner["metrics"]["score"]) > best_score + 1e-9:
                best_score = float(winner["metrics"]["score"])
                best_epoch = epoch
                best_alpha = float(winner["alpha"])
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
    model.load_state_dict(best_state)
    residual, coverage = selector_predictions(model, feature_tensor, mask, args.batch_size)
    torch.save(
        {
            "experiment": VERSION,
            "outer": outer,
            "feature_dim": int(features.shape[-1]),
            "state_dict": best_state,
            "feature_mean": mean,
            "feature_std": std,
            "best_epoch": best_epoch,
            "selected_alpha": best_alpha,
            "official_h_only": True,
            "sionna_or_v12_used": False,
        },
        path,
    )
    return model, residual, coverage, {
        "reused": False,
        "best_epoch": best_epoch,
        "selected_alpha": best_alpha,
        "best_validation_score": best_score,
        "history": history,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
    }


def smoke(args: argparse.Namespace) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    directory = e118.cache_dir(e123.cache_tag(0))
    features, diagnostics = build_official_trajectory_features(
        0, directory, force=True, row_limit=24
    )
    mask = np.asarray(np.load(directory / "mask.npy", mmap_mode="r")[:24], bool)
    model = OfficialPathIdentitySelector(features.shape[-1]).to(args.device)
    tensor = torch.tensor(features, device=args.device)
    mask_tensor = torch.tensor(mask, device=args.device)
    residual, coverage = model(tensor, mask_tensor)
    if residual.shape != mask.shape or coverage.shape != (len(mask),):
        raise RuntimeError((residual.shape, coverage.shape, mask.shape))
    if not torch.isfinite(residual).all() or not torch.isfinite(coverage).all():
        raise RuntimeError("non-finite smoke output")
    print(
        json.dumps(
            {
                "status": "smoke_passed",
                "trajectory": diagnostics,
                "residual_shape": list(residual.shape),
                "coverage_shape": list(coverage.shape),
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.smoke:
        smoke(args)
        return
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    setup, positions, _, site, _ = load_positions()
    fold_id = np.load(e115.FULL_FOLD).astype(np.int8)
    component = np.load(e114.COMPONENTS, allow_pickle=False)
    endpoint = np.load(e121.ENDPOINT, allow_pickle=False)
    query_ids = endpoint["query_ids"].astype(np.int64)
    query_fold = endpoint["full_fold"].astype(np.int8)
    if not np.array_equal(fold_id[query_ids], query_fold):
        raise RuntimeError("canonical query fold mismatch")

    variant_names = (
        "identity_baseline",
        "mass_only",
        "coverage_only",
        "official_physics",
        "candidate_shuffle",
    )
    point_values = {
        name: {
            "pas": np.zeros(len(query_ids), np.float64),
            "pdp": np.zeros(len(query_ids), np.float64),
            "squared_error": np.zeros(len(query_ids), np.float64),
            "truth_energy": np.zeros(len(query_ids), np.float64),
        }
        for name in variant_names
    }
    fold_records = []
    feature_records = []
    for outer in range(5):
        inner = (outer + 1) % 5
        fit_rows = np.flatnonzero((fold_id != outer) & (fold_id != inner))
        validation_rows = np.flatnonzero(fold_id == inner)
        canonical_local = np.flatnonzero(query_fold == outer)
        held_rows = query_ids[canonical_local]
        directory = e118.cache_dir(e123.cache_tag(outer))
        store = e118.ClusterTensorStore(directory, device)
        coordinate = np.asarray(
            np.load(directory / "coordinate.npy", mmap_mode="r"), np.float32
        )
        mask = np.asarray(np.load(directory / "mask.npy", mmap_mode="r"), bool)
        targets, assignment = e124.build_one_to_one_targets(
            coordinate,
            mask,
            component["coordinates"].astype(np.float32),
            component["coefficients"].astype(np.complex64),
            component["point_power"].astype(np.float64),
        )
        hard_mass = torch.as_tensor(
            targets["mass_absolute"], dtype=torch.float32, device=device
        )
        identity_model, normalization, identity_record = load_or_train_identity(
            outer,
            store,
            fit_rows,
            validation_rows,
            hard_mass,
            args,
        )
        identity_logits, identity_mass = e127.identity_predictions(
            identity_model, store, normalization, args.batch_size
        )
        del identity_model

        trajectory, trajectory_record = build_official_trajectory_features(
            outer, directory, args.force_features
        )
        map_features, map_record = build_map_features(
            outer,
            fit_rows,
            directory,
            positions,
            site,
            setup.bs_positions,
            component,
            args.force_features,
        )
        static = static_features(identity_logits, identity_mass, directory, mask)
        features = np.concatenate(
            (
                static,
                np.asarray(trajectory, np.float32),
                np.asarray(map_features, np.float32),
            ),
            axis=2,
        )
        if not np.isfinite(features).all():
            raise RuntimeError(f"non-finite feature outer={outer}")
        selector, residual, coverage, selector_record = train_selector(
            outer,
            store,
            features,
            fit_rows,
            validation_rows,
            normalization,
            hard_mass,
            identity_logits,
            args,
        )
        del selector
        alpha = float(selector_record["selected_alpha"])
        baseline_residual = np.zeros_like(residual)
        baseline_coverage = np.full_like(coverage, 30.0)
        shuffled_features = candidate_shuffle_dynamic(
            features,
            mask,
            static.shape[2],
            args.seed + 4099 * outer,
        )
        mean = np.asarray(selector_record["feature_mean"], dtype=np.float32)
        std = np.asarray(selector_record["feature_std"], dtype=np.float32)
        normalized_shuffle = np.clip((shuffled_features - mean) / std, -8.0, 8.0)
        checkpoint = torch.load(
            selector_checkpoint(outer), map_location=device, weights_only=False
        )
        shuffle_model = OfficialPathIdentitySelector(features.shape[-1]).to(device)
        shuffle_model.load_state_dict(checkpoint["state_dict"])
        shuffle_residual, shuffle_coverage = selector_predictions(
            shuffle_model,
            torch.as_tensor(normalized_shuffle, device=device),
            store.mask,
            args.batch_size,
        )
        del shuffle_model

        variants = {
            "identity_baseline": (baseline_residual, baseline_coverage, 0.0),
            "mass_only": (residual, baseline_coverage, alpha),
            "coverage_only": (baseline_residual, coverage, alpha),
            "official_physics": (residual, coverage, alpha),
            "candidate_shuffle": (shuffle_residual, shuffle_coverage, alpha),
        }
        fold_metrics = {}
        for name, (local_residual, local_coverage, local_alpha) in variants.items():
            metrics, points = evaluate_variant(
                local_residual,
                local_coverage,
                identity_logits,
                store,
                held_rows,
                normalization,
                hard_mass,
                local_alpha,
                args.batch_size,
                collect_point_metrics=True,
            )
            if points is None:
                raise RuntimeError("point metrics were not collected")
            fold_metrics[name] = metrics
            for key, value in points.items():
                point_values[name][key][canonical_local] = value
        fold_record = {
            "outer": outer,
            "inner": inner,
            "fit_count": int(len(fit_rows)),
            "validation_count": int(len(validation_rows)),
            "canonical_held_count": int(len(held_rows)),
            "selected_alpha": alpha,
            "identity": identity_record,
            "selector": selector_record,
            "assignment": assignment,
            "metrics": fold_metrics,
            "delta_vs_identity": float(
                fold_metrics["official_physics"]["score"]
                - fold_metrics["identity_baseline"]["score"]
            ),
            "margin_over_candidate_shuffle": float(
                fold_metrics["official_physics"]["score"]
                - fold_metrics["candidate_shuffle"]["score"]
            ),
        }
        fold_records.append(fold_record)
        feature_records.append(
            {
                "outer": outer,
                "static_dim": int(static.shape[2]),
                "trajectory": trajectory_record,
                "map": map_record,
                "total_feature_dim": int(features.shape[2]),
            }
        )
        print(
            f"E130 outer={outer} canonical baseline="
            f"{fold_metrics['identity_baseline']['score']:.6f} primary="
            f"{fold_metrics['official_physics']['score']:.6f} shuffled="
            f"{fold_metrics['candidate_shuffle']['score']:.6f}",
            flush=True,
        )
        del store, hard_mass, features, shuffled_features, normalized_shuffle
        torch.cuda.empty_cache()

    aggregate = {}
    for name in variant_names:
        values = PointMetricStatistics(**point_values[name])
        aggregate[name] = aggregate_statistics(values)
    baseline = aggregate["identity_baseline"]
    mass_only = aggregate["mass_only"]
    coverage_only = aggregate["coverage_only"]
    primary = aggregate["official_physics"]
    shuffled = aggregate["candidate_shuffle"]
    fold_delta = [float(row["delta_vs_identity"]) for row in fold_records]
    decomposition = {
        "status": "posthoc_frozen_checkpoint_diagnostic",
        "mass_only_gain": float(mass_only["score"] - baseline["score"]),
        "coverage_only_gain": float(coverage_only["score"] - baseline["score"]),
        "full_gain": float(primary["score"] - baseline["score"]),
        "mass_coverage_interaction": float(
            primary["score"] - mass_only["score"] - coverage_only["score"]
            + baseline["score"]
        ),
        "candidate_alignment_margin": float(primary["score"] - shuffled["score"]),
    }
    gate = {
        "minimum_fixed500_score": 0.62,
        "require_pas_gain": True,
        "require_pdp_gain": True,
        "require_nmse_nonincrease": True,
        "require_five_positive_folds": True,
        "require_positive_shuffle_margin": True,
        "fixed500_score": float(primary["score"]),
        "score_gain": float(primary["score"] - baseline["score"]),
        "pas_gain": float(primary["pas"] - baseline["pas"]),
        "pdp_gain": float(primary["pdp"] - baseline["pdp"]),
        "nmse_change": float(primary["nmse"] - baseline["nmse"]),
        "positive_folds": int(sum(value > 0.0 for value in fold_delta)),
        "minimum_fold_delta": float(min(fold_delta)),
        "shuffle_margin": float(primary["score"] - shuffled["score"]),
    }
    gate["passed"] = bool(
        gate["fixed500_score"] >= gate["minimum_fixed500_score"]
        and gate["pas_gain"] > 0.0
        and gate["pdp_gain"] > 0.0
        and gate["nmse_change"] <= 0.0
        and gate["positive_folds"] == 5
        and gate["shuffle_margin"] > 0.0
    )
    report = {
        "experiment": VERSION,
        "status": "fivefold_complete",
        "primary_hypothesis": (
            "Official-H member trajectories reconstruct path boundary and Jacobian "
            "state, and a setwise selector can turn that state into one-to-one path "
            "identity and mass gains under the complete official score."
        ),
        "strict_protocol": {
            "query_count": int(len(query_ids)),
            "canonical_fixed500": True,
            "outer_and_inner_excluded_from_candidate_donors": True,
            "inner_selects_identity_epoch_selector_epoch_and_alpha": True,
            "held_used_once_for_final_scoring": True,
            "official_h_only": True,
            "test_sionna_loaded": False,
            "sionna_labels_or_weights_loaded": False,
            "v12_loaded": False,
            "phase_head_trained": False,
            "candidate_generated": False,
        },
        "configuration": {
            "identity_epochs": args.identity_epochs,
            "selector_epochs": args.selector_epochs,
            "validation_interval": args.validation_interval,
            "batch_size": args.batch_size,
            "selector_alphas": SELECTOR_ALPHAS,
            "trajectory_spatial_scale_m": TRAJECTORY_SPATIAL_SCALE_M,
            "trajectory_feature_names": TRAJECTORY_NAMES,
        },
        "features": feature_records,
        "folds": fold_records,
        "aggregate": aggregate,
        "posthoc_decomposition": decomposition,
        "reference": {
            "e123_candidate_base": 0.515323,
            "e123_mass_oracle": 0.648175,
            "v12_fixed500": 0.677521,
            "v12_online": 0.685857,
        },
        "gate": gate,
        "decision": {
            "identity_mass_stage_solved": bool(gate["passed"]),
            "candidate_or_zip_allowed": False,
            "reason": (
                "E130 is an analysis experiment; a candidate remains forbidden unless "
                "the pre-registered fixed-500 gate passes."
            ),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        METRICS,
        query_ids=query_ids.astype(np.int32),
        full_fold=query_fold.astype(np.int8),
        variant_names=np.asarray(variant_names),
        **{
            f"{name}_{key}": value
            for name, payload in point_values.items()
            for key, value in payload.items()
        },
    )
    print(
        json.dumps(
            {"aggregate": aggregate, "gate": gate, "seconds": report["seconds"]},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
