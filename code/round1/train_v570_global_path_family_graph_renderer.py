#!/usr/bin/env python3
"""Learn cross-position path-family states and render the final joint S-H spectrum."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as functional

import analyze_v545_path_block_family_oracle as v545
import analyze_v546_local_global_path_event_model as v546
import generate_v543_disagreement_conditioned_cell_fusion as v543
from project_paths import PATHS


VERSION_NUMBER = 570
VERSION = "v570_global_path_family_graph_renderer"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
FOLD_DIR = CACHE_DIR / "fold_predictions"
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = CACHE_DIR / "v570_global_path_family_graph_oof.npz"
FIG_DIR = REPORT_DIR / "figures"

OOF_SOURCE = v545.OOF_SOURCE
BANK_CACHE = v545.BANK_CACHE
PATH_META = v545.PATH_META
EVENT_CACHE = v546.EVENT_CACHE
TRAIN_POS = v545.TRAIN_POS
TRAIN_POWER = v545.TRAIN_POWER

POINT_COUNT = 2000
CELL_COUNT = v545.HEIGHT * v545.WIDTH
SOURCE_COUNT = v545.SOURCE_COUNT
OPTION_COUNT = SOURCE_COUNT + 1
BASE_OPTION = SOURCE_COUNT
FAMILY_MODES = v545.IDENTITY_MODES
NEIGHBORS_PER_MODE = 8
FAMILY_RADIUS_M = {
    "exact": 80.0,
    "endpoint": 50.0,
    "first": 30.0,
}
NEIGHBOR_TARGET_DIM = 2 * SOURCE_COUNT
NEIGHBOR_TOKEN_DIM = NEIGHBOR_TARGET_DIM + 4 + 2 + len(FAMILY_MODES) + 12 + 16
MAX_LOG_GAIN = math.log(v545.GAIN_MAX)
SEED = 570_001
FLOOR = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-control", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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
            "figure.dpi": 140,
            "savefig.dpi": 260,
        }
    )


def validate_inputs() -> None:
    required = (
        OOF_SOURCE,
        BANK_CACHE,
        PATH_META,
        EVENT_CACHE,
        TRAIN_POS,
        TRAIN_POWER,
    )
    missing = [PATHS.relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing v570 inputs: {missing}")


def contiguous_fold_codes(positions: np.ndarray) -> np.ndarray:
    model = KMeans(n_clusters=5, n_init=20, random_state=570)
    labels = model.fit_predict(np.asarray(positions[:, :2], dtype=np.float64))
    centers = model.cluster_centers_
    order = np.lexsort((centers[:, 0], centers[:, 1]))
    remap = np.empty(5, dtype=np.int16)
    remap[order] = np.arange(5, dtype=np.int16)
    return remap[labels]


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    return value / np.maximum(value.sum(axis=1, keepdims=True), 1e-30)


def load_data() -> dict[str, np.ndarray]:
    oof = np.load(OOF_SOURCE)
    records_file = np.load(EVENT_CACHE)
    records = {key: np.asarray(records_file[key]) for key in records_file.files}
    positions = np.load(TRAIN_POS).astype(np.float32)
    power = np.load(TRAIN_POWER).astype(np.float64)
    valid_points = power != 0.0
    static_train, _, _ = v543.load_static_features()
    common = np.concatenate(
        [
            static_train[records["point"]],
            records["path_features"],
            records["block_features"],
            records["candidate_features"],
        ],
        axis=1,
    ).astype(np.float32)
    common = np.nan_to_num(common, nan=0.0, posinf=1e6, neginf=-1e6)

    path_meta = np.load(PATH_META)
    owner = np.asarray(path_meta["owner_path"])
    point_event_count = np.bincount(records["point"], minlength=POINT_COUNT)
    maximum_events = int(np.max(point_event_count))
    point_events = np.full((POINT_COUNT, maximum_events), -1, dtype=np.int32)
    event_slot = np.empty(len(records["point"]), dtype=np.int16)
    cursor = np.zeros(POINT_COUNT, dtype=np.int16)
    event_lookup: dict[tuple[int, int], int] = {}
    for event, (point, path) in enumerate(zip(records["point"], records["path"])):
        point_id = int(point)
        slot = int(cursor[point_id])
        point_events[point_id, slot] = event
        event_slot[event] = slot
        cursor[point_id] += 1
        event_lookup[(point_id, int(path))] = event

    cell_event = np.full((POINT_COUNT, CELL_COUNT), -1, dtype=np.int32)
    for point in np.flatnonzero(point_event_count):
        point_owner = owner[point]
        for path in np.unique(point_owner[point_owner != v545.OWNER_NONE]):
            event = event_lookup.get((int(point), int(path)))
            if event is not None:
                cell_event[point, point_owner == path] = event

    target = np.concatenate(
        [records["utility_model"], records["log_gain"]], axis=1
    ).astype(np.float32)
    folds = contiguous_fold_codes(positions)
    return {
        "truth": oof["truth"].astype(np.float32),
        "anchor": oof["anchor"].astype(np.float32),
        "base": oof["calibrated"].astype(np.float32),
        "cell_oracle": oof["cell_oracle"].astype(np.float32),
        "regimes": oof["regimes"].astype("U8"),
        "positions": positions,
        "valid_points": valid_points,
        "folds": folds,
        "common": common,
        "target": target,
        "point": records["point"].astype(np.int64),
        "path": records["path"].astype(np.int64),
        "cells": records["cells"].astype(np.float32),
        "event_weight": records["event_weight"].astype(np.float32),
        "block_features": records["block_features"].astype(np.float32),
        "path_features": records["path_features"].astype(np.float32),
        "utility_model": records["utility_model"].astype(np.float32),
        "log_gain": records["log_gain"].astype(np.float32),
        **{
            f"family_{mode}": records[f"family_{mode}"].astype(np.int64)
            for mode in FAMILY_MODES
        },
        "point_events": point_events,
        "event_slot": event_slot,
        "cell_event": cell_event,
    }


def shuffled_fit_targets(
    data: dict[str, np.ndarray],
    support_events: np.ndarray,
    seed: int,
) -> np.ndarray:
    output = data["target"].copy()
    rng = np.random.default_rng(seed)
    event_regimes = data["regimes"][data["point"]]
    for regime in v545.REGIMES:
        ids = support_events[event_regimes[support_events] == regime]
        if len(ids) > 1:
            output[ids] = data["target"][ids[rng.permutation(len(ids))]]
    return output


def build_neighbor_tokens(
    data: dict[str, np.ndarray],
    support_points: np.ndarray,
    *,
    shuffled: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    event_count = len(data["point"])
    mode_count = len(FAMILY_MODES)
    tokens = np.zeros(
        (event_count, mode_count * NEIGHBORS_PER_MODE, NEIGHBOR_TOKEN_DIM),
        dtype=np.float32,
    )
    valid = np.zeros((event_count, mode_count * NEIGHBORS_PER_MODE), dtype=bool)
    support_event_mask = support_points[data["point"]]
    support_events = np.flatnonzero(support_event_mask)
    target_values = (
        shuffled_fit_targets(data, support_events, seed)
        if shuffled
        else data["target"]
    )
    positions = data["positions"][:, :2].astype(np.float64)

    for mode_index, mode in enumerate(FAMILY_MODES):
        family = data[f"family_{mode}"]
        order = np.argsort(family, kind="stable")
        boundaries = np.flatnonzero(np.diff(family[order])) + 1
        radius = FAMILY_RADIUS_M[mode]
        for group in np.split(order, boundaries):
            support = group[support_event_mask[group]]
            if not len(support):
                continue
            tree = cKDTree(positions[data["point"][support]])
            query_positions = positions[data["point"][group]]
            k = min(NEIGHBORS_PER_MODE + 4, len(support))
            distance, local = tree.query(
                query_positions,
                k=k,
                distance_upper_bound=radius,
            )
            if k == 1:
                distance = distance[:, None]
                local = local[:, None]
            for query_row, event in enumerate(group):
                keep = np.isfinite(distance[query_row]) & (local[query_row] < len(support))
                neighbor = support[local[query_row, keep]]
                local_distance = distance[query_row, keep]
                different_point = data["point"][neighbor] != data["point"][event]
                neighbor = neighbor[different_point][:NEIGHBORS_PER_MODE]
                local_distance = local_distance[different_point][:NEIGHBORS_PER_MODE]
                for rank, (neighbor_event, distance_m) in enumerate(
                    zip(neighbor, local_distance)
                ):
                    slot = mode_index * NEIGHBORS_PER_MODE + rank
                    delta = (
                        positions[data["point"][neighbor_event]]
                        - positions[data["point"][event]]
                    )
                    values = np.concatenate(
                        [
                            target_values[neighbor_event],
                            np.asarray(
                                [
                                    delta[0] / 160.0,
                                    delta[1] / 160.0,
                                    distance_m / 160.0,
                                    math.log1p(distance_m) / math.log(161.0),
                                ],
                                dtype=np.float32,
                            ),
                            np.asarray(
                                [
                                    data["event_weight"][neighbor_event],
                                    data["cells"][neighbor_event] / CELL_COUNT,
                                ],
                                dtype=np.float32,
                            ),
                            np.eye(mode_count, dtype=np.float32)[mode_index],
                            data["block_features"][neighbor_event],
                            data["path_features"][neighbor_event, :16],
                        ]
                    ).astype(np.float32)
                    if len(values) != NEIGHBOR_TOKEN_DIM:
                        raise ValueError(
                            f"neighbor token size {len(values)} != {NEIGHBOR_TOKEN_DIM}"
                        )
                    tokens[event, slot] = values
                    valid[event, slot] = True

    counts = np.sum(valid, axis=1)
    diagnostics = {
        "event_fraction_with_neighbor": float(np.mean(counts > 0)),
        "median_neighbor_count": float(np.median(counts)),
        "p90_neighbor_count": float(np.quantile(counts, 0.90)),
        "maximum_neighbor_count": int(np.max(counts)),
    }
    return tokens, valid, diagnostics


class PathFamilyGraphRenderer(nn.Module):
    def __init__(self, common_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.common_encoder = nn.Sequential(
            nn.Linear(common_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(NEIGHBOR_TOKEN_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.family_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=8,
            dropout=0.08,
            batch_first=True,
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=4 * hidden_dim,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.point_transformer = nn.TransformerEncoder(
            layer,
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * SOURCE_COUNT + 1),
        )
        final = self.output[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            final.bias[SOURCE_COUNT] = -2.0

    def forward(
        self,
        common: torch.Tensor,
        neighbor_tokens: torch.Tensor,
        neighbor_valid: torch.Tensor,
        point_events: torch.Tensor,
        event_point: torch.Tensor,
        event_slot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.common_encoder(common)
        neighbor = self.neighbor_encoder(neighbor_tokens)
        safe_valid = neighbor_valid.clone()
        empty = ~torch.any(safe_valid, dim=1)
        if torch.any(empty):
            safe_valid[empty, 0] = True
            neighbor = neighbor.clone()
            neighbor[empty, 0] = 0.0
        attended, _ = self.family_attention(
            query[:, None, :],
            neighbor,
            neighbor,
            key_padding_mask=~safe_valid,
            need_weights=False,
        )
        support_fraction = neighbor_valid.float().mean(dim=1, keepdim=True)
        event_state = self.fusion(
            torch.cat([query, attended[:, 0], support_fraction], dim=1)
        )

        point_mask = point_events >= 0
        safe_events = torch.clamp(point_events, min=0)
        point_state = event_state[safe_events] * point_mask[..., None]
        safe_mask = point_mask.clone()
        empty_points = ~torch.any(safe_mask, dim=1)
        if torch.any(empty_points):
            safe_mask[empty_points, 0] = True
        point_state = self.point_transformer(
            point_state,
            src_key_padding_mask=~safe_mask,
        )
        event_state = point_state[event_point, event_slot]
        output = self.output(event_state)
        source_logits = output[:, :SOURCE_COUNT]
        action_raw = output[:, SOURCE_COUNT]
        log_gain = MAX_LOG_GAIN * torch.tanh(output[:, SOURCE_COUNT + 1 :])
        return source_logits, action_raw, log_gain


def torch_row_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    numerator = torch.sum(left * right, dim=1)
    denominator = torch.sqrt(
        torch.sum(left * left, dim=1) * torch.sum(right * right, dim=1) + 1e-20
    )
    return numerator / denominator


def torch_scores(
    truth: torch.Tensor,
    prediction: torch.Tensor,
) -> dict[str, torch.Tensor]:
    truth_grid = truth.reshape(-1, v545.HEIGHT, v545.WIDTH)
    pred_grid = prediction.reshape(-1, v545.HEIGHT, v545.WIDTH)
    joint = torch_row_cosine(truth_grid.flatten(1), pred_grid.flatten(1))
    truth_delay = truth_grid.sum(dim=2)
    pred_delay = pred_grid.sum(dim=2)
    truth_angle = truth_grid.sum(dim=1)
    pred_angle = pred_grid.sum(dim=1)
    delay = torch_row_cosine(truth_delay, pred_delay)
    angle = torch_row_cosine(truth_angle, pred_angle)
    weighted = 0.50 * joint + 0.25 * delay + 0.25 * angle
    return {"joint": joint, "delay": delay, "angle": angle, "weighted": weighted}


def render_points(
    point_ids: torch.Tensor,
    source_logits: torch.Tensor,
    action_raw: torch.Tensor,
    log_gain: torch.Tensor,
    bank: torch.Tensor,
    base: torch.Tensor,
    cell_event: torch.Tensor,
) -> torch.Tensor:
    local_event = cell_event[point_ids]
    has_event = local_event >= 0
    safe_event = torch.clamp(local_event, min=0)
    source_weight = torch.softmax(source_logits[safe_event], dim=-1)
    source_gain = torch.exp(log_gain[safe_event])
    source_value = bank[point_ids].transpose(1, 2)
    source_mix = torch.sum(
        source_weight * source_gain * source_value,
        dim=-1,
    )
    initial_softplus = functional.softplus(
        torch.as_tensor(-2.0, device=action_raw.device, dtype=action_raw.dtype)
    )
    action = functional.softplus(action_raw[safe_event]) - initial_softplus
    action = torch.clamp(action, min=-0.50, max=1.50)
    log_delta = torch.clamp(
        torch.log(torch.clamp(source_mix, min=1e-12))
        - torch.log(torch.clamp(base[point_ids], min=1e-12)),
        min=-3.0,
        max=3.0,
    )
    prediction = base[point_ids] * torch.exp(action * log_delta)
    prediction = torch.where(has_event, prediction, base[point_ids])
    prediction = torch.clamp(prediction, min=0.0)
    return prediction / torch.clamp(prediction.sum(dim=1, keepdim=True), min=1e-20)


def event_auxiliary_loss(
    source_logits: torch.Tensor,
    action_raw: torch.Tensor,
    log_gain: torch.Tensor,
    event_ids: torch.Tensor,
    utility: torch.Tensor,
    gain_target: torch.Tensor,
    event_weight: torch.Tensor,
) -> torch.Tensor:
    local_utility = utility[event_ids]
    best_utility, best_source = torch.max(local_utility, dim=1)
    label = torch.where(
        best_utility > 0.0,
        best_source,
        torch.full_like(best_source, BASE_OPTION),
    )
    weight = event_weight[event_ids]
    weight = weight / torch.clamp(torch.mean(weight), min=1e-6)
    active = label != BASE_OPTION
    if torch.any(active):
        classification = functional.cross_entropy(
            source_logits[event_ids[active]],
            label[active],
            reduction="none",
        )
        selected_gain = log_gain[event_ids[active], label[active]]
        target_gain = MAX_LOG_GAIN * gain_target[event_ids[active], label[active]]
        regression = functional.smooth_l1_loss(
            selected_gain,
            target_gain,
            reduction="none",
        )
        gain_loss = torch.sum(regression * weight[active]) / torch.sum(weight[active])
        class_loss = torch.sum(classification * weight[active]) / torch.sum(weight[active])
    else:
        gain_loss = source_logits.new_zeros(())
        class_loss = source_logits.new_zeros(())
    initial_softplus = functional.softplus(
        torch.as_tensor(-2.0, device=action_raw.device, dtype=action_raw.dtype)
    )
    predicted_action = functional.softplus(action_raw[event_ids]) - initial_softplus
    target_action = torch.clamp(best_utility, min=0.0, max=1.0)
    action_loss = functional.smooth_l1_loss(
        predicted_action,
        target_action,
        reduction="none",
    )
    action_loss = torch.sum(action_loss * weight) / torch.sum(weight)
    return class_loss + 0.25 * gain_loss + 0.50 * action_loss


def model_state(
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def prepare_tensors(
    data: dict[str, np.ndarray],
    neighbor_tokens: np.ndarray,
    neighbor_valid: np.ndarray,
    support_points: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    fit_events = support_points[data["point"]]
    mean = data["common"][fit_events].mean(axis=0, keepdims=True)
    std = np.maximum(data["common"][fit_events].std(axis=0, keepdims=True), 1e-4)
    common = ((data["common"] - mean) / std).astype(np.float32)
    return {
        "common": torch.from_numpy(common).to(device),
        "neighbor_tokens": torch.from_numpy(neighbor_tokens).to(device),
        "neighbor_valid": torch.from_numpy(neighbor_valid).to(device),
        "point_events": torch.from_numpy(data["point_events"].astype(np.int64)).to(device),
        "event_point": torch.from_numpy(data["point"]).to(device),
        "event_slot": torch.from_numpy(data["event_slot"].astype(np.int64)).to(device),
        "cell_event": torch.from_numpy(data["cell_event"].astype(np.int64)).to(device),
        "truth": torch.from_numpy(data["truth"]).to(device),
        "base": torch.from_numpy(data["base"]).to(device),
        "utility": torch.from_numpy(data["utility_model"]).to(device),
        "gain_target": torch.from_numpy(data["log_gain"]).to(device),
        "event_weight": torch.from_numpy(data["event_weight"]).to(device),
        "feature_mean": torch.from_numpy(mean.astype(np.float32)),
        "feature_std": torch.from_numpy(std.astype(np.float32)),
    }


@torch.no_grad()
def evaluate_model(
    model: PathFamilyGraphRenderer,
    tensors: dict[str, torch.Tensor],
    point_ids: np.ndarray,
    bank: torch.Tensor,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    model.eval()
    source_logits, action_raw, log_gain = model(
        tensors["common"],
        tensors["neighbor_tokens"],
        tensors["neighbor_valid"],
        tensors["point_events"],
        tensors["event_point"],
        tensors["event_slot"],
    )
    ids = torch.from_numpy(np.asarray(point_ids, dtype=np.int64)).to(bank.device)
    prediction = render_points(
        ids,
        source_logits,
        action_raw,
        log_gain,
        bank,
        tensors["base"],
        tensors["cell_event"],
    )
    metrics = torch_scores(tensors["truth"][ids], prediction)
    output = {
        key: value.float().cpu().numpy().astype(np.float32)
        for key, value in metrics.items()
    }
    return output, prediction.float().cpu().numpy().astype(np.float32)


def train_model(
    data: dict[str, np.ndarray],
    train_points: np.ndarray,
    validation_points: np.ndarray | None,
    neighbor_support: np.ndarray,
    bank: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    *,
    shuffled: bool,
    seed: int,
    fixed_epochs: int | None = None,
) -> tuple[PathFamilyGraphRenderer, dict[str, object], dict[str, torch.Tensor]]:
    neighbor_tokens, neighbor_valid, neighbor_diagnostics = build_neighbor_tokens(
        data,
        neighbor_support,
        shuffled=shuffled,
        seed=seed + 50_000,
    )
    tensors = prepare_tensors(
        data,
        neighbor_tokens,
        neighbor_valid,
        neighbor_support,
        device,
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PathFamilyGraphRenderer(data["common"].shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_point_ids = np.flatnonzero(train_points).astype(np.int64)
    train_ids = torch.from_numpy(train_point_ids).to(device)
    train_event_ids = torch.from_numpy(
        np.flatnonzero(train_points[data["point"]]).astype(np.int64)
    ).to(device)
    maximum_epochs = fixed_epochs if fixed_epochs is not None else args.epochs
    best_score = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    pretrain_history: list[dict[str, float | int]] = []
    stale = 0
    start_time = time.monotonic()

    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            source_logits, action_raw, log_gain = model(
                tensors["common"],
                tensors["neighbor_tokens"],
                tensors["neighbor_valid"],
                tensors["point_events"],
                tensors["event_point"],
                tensors["event_slot"],
            )
            auxiliary = event_auxiliary_loss(
                source_logits,
                action_raw,
                log_gain,
                train_event_ids,
                tensors["utility"],
                tensors["gain_target"],
                tensors["event_weight"],
            )
        auxiliary.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.pretrain_epochs:
            with torch.no_grad():
                local_utility = tensors["utility"][train_event_ids]
                best_utility, best_source = torch.max(local_utility, dim=1)
                active = best_utility > 0.0
                accuracy = float(
                    torch.mean(
                        (torch.argmax(source_logits[train_event_ids], dim=1)[active]
                        == best_source[active]).float()
                    ).cpu()
                )
            row = {
                "epoch": epoch,
                "auxiliary_loss": float(auxiliary.detach().cpu()),
                "active_source_accuracy": accuracy,
            }
            pretrain_history.append(row)
            print(
                f"v570 {'control' if shuffled else 'main'} pretrain={epoch:03d} "
                f"aux={row['auxiliary_loss']:.6f} accuracy={accuracy:.3f}",
                flush=True,
            )

    final_layer = model.output[-1]
    assert isinstance(final_layer, nn.Linear)
    with torch.no_grad():
        final_layer.weight[SOURCE_COUNT].zero_()
        final_layer.bias[SOURCE_COUNT] = -2.0

    if validation_points is not None:
        validation_ids = np.flatnonzero(validation_points).astype(np.int64)
        pretrained_validation, _ = evaluate_model(
            model,
            tensors,
            validation_ids,
            bank,
        )
        best_score = float(np.mean(pretrained_validation["weighted"]))
    else:
        pretrained_train, _ = evaluate_model(
            model,
            tensors,
            train_point_ids,
            bank,
        )
        best_score = float(np.mean(pretrained_train["weighted"]))
    best_epoch = 0
    best_state = model_state(model)
    print(
        f"v570 {'control' if shuffled else 'main'} pretrained_score={best_score:.6f}",
        flush=True,
    )

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            source_logits, action_raw, log_gain = model(
                tensors["common"],
                tensors["neighbor_tokens"],
                tensors["neighbor_valid"],
                tensors["point_events"],
                tensors["event_point"],
                tensors["event_slot"],
            )
            prediction = render_points(
                train_ids,
                source_logits,
                action_raw,
                log_gain,
                bank,
                tensors["base"],
                tensors["cell_event"],
            )
            score = torch_scores(tensors["truth"][train_ids], prediction)["weighted"]
            spectral_loss = 1.0 - torch.mean(score)
            auxiliary = event_auxiliary_loss(
                source_logits,
                action_raw,
                log_gain,
                train_event_ids,
                tensors["utility"],
                tensors["gain_target"],
                tensors["event_weight"],
            )
            loss = spectral_loss + 0.01 * auxiliary
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        should_evaluate = epoch == 1 or epoch % 5 == 0 or epoch == maximum_epochs
        if should_evaluate:
            train_score = float(torch.mean(score.detach()).cpu())
            if validation_points is not None:
                validation_ids = np.flatnonzero(validation_points).astype(np.int64)
                validation, _ = evaluate_model(model, tensors, validation_ids, bank)
                validation_score = float(np.mean(validation["weighted"]))
            else:
                validation_score = train_score
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu()),
                    "train_score": train_score,
                    "validation_score": validation_score,
                }
            )
            print(
                f"v570 {'control' if shuffled else 'main'} epoch={epoch:03d} "
                f"train={train_score:.6f} validation={validation_score:.6f}",
                flush=True,
            )
            if validation_score > best_score + 1e-5:
                best_score = validation_score
                best_epoch = epoch
                best_state = model_state(model)
                stale = 0
            else:
                stale += 5
                if fixed_epochs is None and stale >= args.patience:
                    break

    if fixed_epochs is None:
        model.load_state_dict(best_state)
    else:
        best_epoch = fixed_epochs
        if history:
            best_score = history[-1]["validation_score"]
    details: dict[str, object] = {
        "best_epoch": int(best_epoch),
        "best_validation_score": float(best_score),
        "elapsed_seconds": float(time.monotonic() - start_time),
        "neighbor_diagnostics": neighbor_diagnostics,
        "pretrain_history": pretrain_history,
        "history": history,
    }
    return model, details, tensors


def numpy_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, np.ndarray]:
    return v543.spectrum_scores(truth, prediction)


def run_fold(
    fold: int,
    data: dict[str, np.ndarray],
    bank: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, object]:
    fold_path = FOLD_DIR / f"fold_{fold}.npz"
    checkpoint_path = CHECKPOINT_DIR / f"fold_{fold}.pt"
    if fold_path.is_file() and not args.force:
        with np.load(fold_path) as loaded:
            return json.loads(str(loaded["details"].item()))

    valid = data["valid_points"]
    outer = data["folds"] == fold
    fit = valid & ~outer
    query = valid & outer
    tune_fold = (fold + 1) % 5
    inner_validation = fit & (data["folds"] == tune_fold)
    inner_train = fit & ~inner_validation
    print(
        f"v570 fold={fold} inner_train={int(np.sum(inner_train))} "
        f"inner_validation={int(np.sum(inner_validation))} outer={int(np.sum(query))}",
        flush=True,
    )

    inner_model, selection, _ = train_model(
        data,
        inner_train,
        inner_validation,
        inner_train,
        bank,
        device,
        args,
        shuffled=False,
        seed=args.seed + 100 * fold,
    )
    selected_epoch = max(int(selection["best_epoch"]), 0)
    del inner_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model, final_training, tensors = train_model(
        data,
        fit,
        None,
        fit,
        bank,
        device,
        args,
        shuffled=False,
        seed=args.seed + 100 * fold + 1,
        fixed_epochs=selected_epoch,
    )
    metrics, prediction = evaluate_model(model, tensors, np.flatnonzero(query), bank)
    query_ids = np.flatnonzero(query)
    base_metrics = numpy_metrics(data["truth"][query], data["base"][query])
    oracle_metrics = numpy_metrics(data["truth"][query], data["cell_oracle"][query])
    torch.save(
        {
            "state_dict": model_state(model),
            "feature_mean": tensors["feature_mean"],
            "feature_std": tensors["feature_std"],
            "fold": fold,
            "selected_epoch": selected_epoch,
            "common_dim": data["common"].shape[1],
            "hidden_dim": args.hidden_dim,
        },
        checkpoint_path,
    )

    control_metrics: dict[str, np.ndarray] | None = None
    control_prediction: np.ndarray | None = None
    control_training: dict[str, object] | None = None
    if not args.skip_control:
        control, control_training, control_tensors = train_model(
            data,
            fit,
            None,
            fit,
            bank,
            device,
            args,
            shuffled=True,
            seed=args.seed + 100 * fold + 2,
            fixed_epochs=selected_epoch,
        )
        control_metrics, control_prediction = evaluate_model(
            control,
            control_tensors,
            query_ids,
            bank,
        )
        del control, control_tensors

    details: dict[str, object] = {
        "fold": fold,
        "tune_fold": tune_fold,
        "fit_points": int(np.sum(fit)),
        "query_points": int(np.sum(query)),
        "selected_epoch": selected_epoch,
        "selection": selection,
        "final_training": final_training,
        "baseline": {key: float(np.mean(value)) for key, value in base_metrics.items()},
        "prediction": {key: float(np.mean(value)) for key, value in metrics.items()},
        "oracle": {key: float(np.mean(value)) for key, value in oracle_metrics.items()},
        "gain": {
            key: float(np.mean(metrics[key] - base_metrics[key])) for key in metrics
        },
        "control": (
            {key: float(np.mean(value)) for key, value in control_metrics.items()}
            if control_metrics is not None
            else None
        ),
        "margin_over_control": (
            float(np.mean(metrics["weighted"] - control_metrics["weighted"]))
            if control_metrics is not None
            else None
        ),
        "control_training": control_training,
    }
    save_values: dict[str, np.ndarray] = {
        "query_ids": query_ids.astype(np.int32),
        "prediction": prediction.astype(np.float32),
        **{f"metric_{key}": value for key, value in metrics.items()},
        "details": np.asarray(json.dumps(details, ensure_ascii=False)),
    }
    if control_prediction is not None and control_metrics is not None:
        save_values["control_prediction"] = control_prediction.astype(np.float32)
        for key, value in control_metrics.items():
            save_values[f"control_metric_{key}"] = value
    np.savez_compressed(fold_path, **save_values)
    print(json.dumps(details["gain"], indent=2), flush=True)
    del model, tensors
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return details


def summarize_by_regime(
    truth: np.ndarray,
    prediction: np.ndarray,
    base: np.ndarray,
    valid: np.ndarray,
    regimes: np.ndarray,
) -> dict[str, object]:
    pred = numpy_metrics(truth, prediction)
    base_metrics = numpy_metrics(truth, base)
    output: dict[str, object] = {}
    for regime in ("overall", *v545.REGIMES):
        mask = valid if regime == "overall" else valid & (regimes == regime)
        output[regime] = {
            "count": int(np.sum(mask)),
            **{key: float(np.mean(pred[key][mask])) for key in pred},
            **{
                f"gain_{key}": float(np.mean(pred[key][mask] - base_metrics[key][mask]))
                for key in pred
            },
        }
    return output


def plot_results(
    data: dict[str, np.ndarray],
    prediction: np.ndarray,
    control: np.ndarray,
    fold_details: list[dict[str, object]],
    summary: dict[str, object],
) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "scores": FIG_DIR / "00_连续区域路径族图模型评分.png",
        "folds": FIG_DIR / "01_五个连续留出区域增量.png",
        "waterfall": FIG_DIR / "02_未见区域路径出生消失热图.png",
        "space": FIG_DIR / "03_点云上的逐点收益分布.png",
    }
    valid = data["valid_points"]
    base_metrics = numpy_metrics(data["truth"], data["base"])
    pred_metrics = numpy_metrics(data["truth"], prediction)
    control_metrics = numpy_metrics(data["truth"], control)

    figure, axis = plt.subplots(figsize=(9.2, 5.5), constrained_layout=True)
    labels = ["v543基线", "路径族图模型", "路径关系打乱", "逐格候选上限"]
    values = [
        float(np.mean(base_metrics["weighted"][valid])),
        float(np.mean(pred_metrics["weighted"][valid])),
        float(np.mean(control_metrics["weighted"][valid])),
        float(np.mean(numpy_metrics(data["truth"], data["cell_oracle"])["weighted"][valid])),
    ]
    bars = axis.bar(labels, values, color=["#64748b", "#2563eb", "#ef4444", "#16a34a"])
    axis.bar_label(bars, fmt="%.6f", padding=4)
    axis.set_ylim(max(0.65, min(values) - 0.04), min(1.0, max(values) + 0.04))
    axis.set_ylabel("joint/delay/angle综合分")
    axis.set_title("连续空间留出：模型是否真正利用路径族状态", fontweight="bold")
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(outputs["scores"], bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10.0, 5.5), constrained_layout=True)
    folds = np.arange(5)
    gain = np.asarray([row["gain"]["weighted"] for row in fold_details])
    margin = np.asarray([row["margin_over_control"] for row in fold_details])
    width = 0.36
    axis.bar(folds - width / 2, gain, width, label="相对v543", color="#2563eb")
    axis.bar(folds + width / 2, margin, width, label="相对打乱对照", color="#f59e0b")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(folds, [f"区域{fold}" for fold in folds])
    axis.set_ylabel("综合分增量")
    axis.set_title("五个完全留出的连续空间区域", fontweight="bold")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(outputs["folds"], bbox_inches="tight")
    plt.close(figure)

    target_fold = int(np.argmax(gain))
    mask = valid & (data["folds"] == target_fold)
    ids = np.flatnonzero(mask)
    local_pos = data["positions"][ids, :2]
    centered = local_pos - local_pos.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    route = centered @ vh[0]
    order = np.argsort(route)
    ids = ids[order]
    panels = [
        (data["truth"][ids], "官方真实谱"),
        (data["base"][ids], "v543基线"),
        (prediction[ids], "路径族图模型"),
        (np.abs(prediction[ids] - data["truth"][ids]), "绝对误差"),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(18.5, 6.0), constrained_layout=True)
    for axis, (value, title) in zip(axes, panels):
        delay = value.reshape(-1, v545.HEIGHT, v545.WIDTH).sum(axis=2)
        image = axis.imshow(
            np.log10(np.maximum(delay.T, 1e-8)),
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=-5.0,
            vmax=0.0,
        )
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("沿连续区域主方向排序的未见点")
        axis.set_ylabel("时延bin")
    figure.colorbar(image, ax=axes, label="归一化能量log10")
    figure.suptitle(
        f"区域{target_fold}：主路径是否在正确位置出生、移动和消失",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(outputs["waterfall"], bbox_inches="tight")
    plt.close(figure)

    point_gain = pred_metrics["weighted"] - base_metrics["weighted"]
    limit = max(float(np.quantile(np.abs(point_gain[valid]), 0.98)), 1e-4)
    figure, axis = plt.subplots(figsize=(9.5, 7.2), constrained_layout=True)
    scatter = axis.scatter(
        data["positions"][valid, 0],
        data["positions"][valid, 1],
        c=point_gain[valid],
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        s=20,
        edgecolors="none",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("横坐标 x（米）")
    axis.set_ylabel("纵坐标 y（米）")
    axis.set_title("路径族图模型在官方训练点上的严格OOF收益", fontweight="bold")
    figure.colorbar(scatter, ax=axis, label="相对v543逐点综合分增量")
    figure.savefig(outputs["space"], bbox_inches="tight")
    plt.close(figure)
    return outputs


def finalize(
    data: dict[str, np.ndarray],
    fold_details: list[dict[str, object]],
) -> None:
    prediction = data["base"].copy()
    control = data["base"].copy()
    for fold in range(5):
        with np.load(FOLD_DIR / f"fold_{fold}.npz") as loaded:
            ids = loaded["query_ids"].astype(np.int64)
            prediction[ids] = loaded["prediction"]
            if "control_prediction" in loaded.files:
                control[ids] = loaded["control_prediction"]
    valid = data["valid_points"]
    summary = summarize_by_regime(
        data["truth"],
        prediction,
        data["base"],
        valid,
        data["regimes"],
    )
    control_summary = summarize_by_regime(
        data["truth"],
        control,
        data["base"],
        valid,
        data["regimes"],
    )
    overall_gain = float(summary["overall"]["gain_weighted"])
    control_margin = float(
        summary["overall"]["weighted"] - control_summary["overall"]["weighted"]
    )
    fold_gain = np.asarray([row["gain"]["weighted"] for row in fold_details])
    joint_gain = float(summary["overall"]["gain_joint"])
    delay_gain = float(summary["overall"]["gain_delay"])
    angle_gain = float(summary["overall"]["gain_angle"])
    success = bool(
        overall_gain >= 0.03
        and np.all(fold_gain > 0.0)
        and joint_gain > 0.0
        and delay_gain > 0.0
        and angle_gain > 0.0
        and control_margin >= 0.70 * overall_gain
    )
    figures = plot_results(data, prediction, control, fold_details, summary)
    np.savez_compressed(
        OOF_CACHE,
        fold_id=data["folds"],
        valid=valid,
        regimes=data["regimes"],
        prediction=prediction.astype(np.float32),
        control=control.astype(np.float32),
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "A cross-position latent state attached to each Sionna path family can predict "
            "path birth, disappearance and energy in a contiguous unseen region better than "
            "independent event selection or point-level fallback gates."
        ),
        "scope": {
            "train_points": POINT_COUNT,
            "scored_nonzero_points": int(np.sum(valid)),
            "path_events": int(len(data["point"])),
            "candidate_sources": SOURCE_COUNT,
            "contiguous_outer_regions": 5,
            "test_data_used": False,
            "submission_generated": False,
        },
        "model": {
            "type": "path-family conditional graph Transformer with differentiable spectrum renderer",
            "common_feature_count": int(data["common"].shape[1]),
            "neighbor_token_dim": NEIGHBOR_TOKEN_DIM,
            "neighbors_per_family_mode": NEIGHBORS_PER_MODE,
            "family_modes": list(FAMILY_MODES),
            "family_radius_m": FAMILY_RADIUS_M,
            "options": "19 physical candidate sources plus strict v543 fallback",
            "loss": "direct 0.50 joint + 0.25 delay + 0.25 angle cosine, with weak event auxiliary",
            "negative_control": "shuffle fit-only path-state targets within propagation regime",
        },
        "folds": fold_details,
        "summary": summary,
        "control_summary": control_summary,
        "decision": {
            "overall_gain_vs_v543": overall_gain,
            "joint_gain": joint_gain,
            "delay_gain": delay_gain,
            "angle_gain": angle_gain,
            "positive_contiguous_regions": int(np.sum(fold_gain > 0.0)),
            "minimum_region_gain": float(np.min(fold_gain)),
            "margin_over_shuffled_control": control_margin,
            "strict_success": success,
            "success_gate": (
                "overall gain >= 0.03; all five contiguous regions positive; joint, delay "
                "and angle all positive; shuffled-control margin >= 70% of gain"
            ),
            "generate_submission": False,
        },
        "outputs": {
            "oof": PATHS.relative(OOF_CACHE),
            "fold_predictions": PATHS.relative(FOLD_DIR),
            "checkpoints": PATHS.relative(CHECKPOINT_DIR),
            "figures": {key: PATHS.relative(path) for key, path in figures.items()},
            "report": PATHS.relative(REPORT),
            "submission_zip": None,
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    configure_chinese_font()
    validate_inputs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    bank_array = np.array(np.load(BANK_CACHE, mmap_mode="r"), copy=True)
    bank = torch.from_numpy(bank_array).to(device)
    selected_folds = [args.fold] if args.fold is not None else list(range(5))
    fold_details = [run_fold(fold, data, bank, device, args) for fold in selected_folds]
    if args.fold is not None:
        print(
            json.dumps(
                {
                    "fold": int(args.fold),
                    "gain": fold_details[0]["gain"],
                    "margin_over_control": fold_details[0]["margin_over_control"],
                    "cache": PATHS.relative(FOLD_DIR / f"fold_{args.fold}.npz"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return
    finalize(data, fold_details)


if __name__ == "__main__":
    main()
