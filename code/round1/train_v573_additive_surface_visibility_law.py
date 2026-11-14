#!/usr/bin/env python3
"""Learn additive log-domain surface and official-pointcloud visibility laws."""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn

import train_v572_surface_sequence_path_gain_renderer as v572
from project_paths import PATHS


VERSION_NUMBER = 573
VERSION = "v573_additive_surface_visibility_law"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
FOLD_DIR = CACHE_DIR / "fold_results"
VISIBILITY_CACHE = CACHE_DIR / "v573_official_pointcloud_route_visibility_2000.npz"
REPORT = REPORT_DIR / f"{VERSION}.json"

SEGMENT_COUNT = v572.MAX_STEPS + 1
SAMPLES_PER_SEGMENT = 12
SEGMENT_FEATURE_DIM = 8
SEED = 573_001
FLOOR = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--sequence-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--max-log-gain", type=float, default=4.0)
    parser.add_argument("--train-alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force-visibility-cache", action="store_true")
    parser.add_argument("--force-fold", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    return parser.parse_args()


def segment_visibility_features(
    start: np.ndarray,
    end: np.ndarray,
    scene: dict[str, np.ndarray | float],
) -> np.ndarray:
    count = len(start)
    output = np.zeros((count, SEGMENT_FEATURE_DIM), dtype=np.float32)
    if count == 0:
        return output
    z_max = scene["z_max"]
    grid_count = scene["count"]
    vertical_count = scene["vertical_count"]
    wall_x = scene["wall_x"]
    wall_y = scene["wall_y"]
    x_min = float(scene["x_min"])
    y_min = float(scene["y_min"])
    resolution = float(scene["resolution"])
    ny, nx = z_max.shape
    fractions = np.linspace(0.08, 0.92, SAMPLES_PER_SEGMENT, dtype=np.float64)
    for chunk_start in range(0, count, 16_384):
        chunk_end = min(chunk_start + 16_384, count)
        local_start = start[chunk_start:chunk_end, None].astype(np.float64)
        local_end = end[chunk_start:chunk_end, None].astype(np.float64)
        point = (
            local_start * (1.0 - fractions[None, :, None])
            + local_end * fractions[None, :, None]
        )
        ix = np.floor((point[..., 0] - x_min) / resolution).astype(np.int64)
        iy = np.floor((point[..., 1] - y_min) / resolution).astype(np.int64)
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        safe_x = np.clip(ix, 0, nx - 1)
        safe_y = np.clip(iy, 0, ny - 1)
        local_z = z_max[safe_y, safe_x]
        local_count = grid_count[safe_y, safe_x]
        local_vertical = vertical_count[safe_y, safe_x]
        local_wall_x = wall_x[safe_y, safe_x]
        local_wall_y = wall_y[safe_y, safe_x]
        local_z = np.where(inside & (local_count > 0), local_z, -50.0)
        deficit = local_z - point[..., 2]
        positive = np.maximum(deficit, 0.0)
        vertical_fraction = local_vertical / np.maximum(local_count, 1)
        direction = local_end[:, 0, :2] - local_start[:, 0, :2]
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
        alignment = np.abs(
            local_wall_x * direction[:, None, 0]
            + local_wall_y * direction[:, None, 1]
        )
        alignment = np.where(local_vertical > 0, alignment, 0.0)
        output[chunk_start:chunk_end] = np.stack(
            [
                np.mean(deficit > 0.25, axis=1),
                np.mean(deficit > 1.00, axis=1),
                np.mean(positive, axis=1) / 10.0,
                np.maximum(np.max(deficit, axis=1), 0.0) / 30.0,
                np.maximum(np.quantile(deficit, 0.90, axis=1), 0.0) / 30.0,
                np.mean(local_z > 2.0, axis=1),
                np.mean(vertical_fraction, axis=1),
                np.mean(alignment, axis=1),
            ],
            axis=1,
        ).astype(np.float32)
    return np.nan_to_num(output, nan=0.0, posinf=10.0, neginf=0.0)


def build_visibility_cache(
    data: dict[str, np.ndarray],
    force: bool,
) -> dict[str, object]:
    if VISIBILITY_CACHE.is_file() and not force:
        with np.load(VISIBILITY_CACHE) as loaded:
            return json.loads(str(loaded["summary_json"].item()))

    with np.load(v572.SURFACE_CACHE) as loaded:
        depth = loaded["depth"][: v572.POINT_COUNT].astype(np.int16)
    with np.load(v572.SCENE_CACHE) as loaded:
        point_context = loaded["train_features"].astype(np.float32)
        scene = {
            "z_max": np.nan_to_num(loaded["grid_z_max"], nan=-50.0).astype(np.float32),
            "count": loaded["grid_count"].astype(np.float32),
            "vertical_count": loaded["grid_vertical_count"].astype(np.float32),
            "wall_x": np.nan_to_num(loaded["grid_wall_nx"], nan=0.0).astype(np.float32),
            "wall_y": np.nan_to_num(loaded["grid_wall_ny"], nan=0.0).astype(np.float32),
            "x_min": float(loaded["x_min"]),
            "y_min": float(loaded["y_min"]),
            "resolution": float(loaded["resolution"]),
        }

    scale = np.asarray([500.0, 500.0, 50.0], dtype=np.float32)
    vertices = data["step_numeric"][..., :3] * scale
    segment_valid = np.zeros(
        (v572.POINT_COUNT, v572.PATH_LIMIT, SEGMENT_COUNT), dtype=bool
    )
    segment_features = np.zeros(
        (
            v572.POINT_COUNT,
            v572.PATH_LIMIT,
            SEGMENT_COUNT,
            SEGMENT_FEATURE_DIM,
        ),
        dtype=np.float16,
    )
    segment_length = np.zeros(segment_valid.shape, dtype=np.float32)
    started = time.monotonic()
    for segment in range(SEGMENT_COUNT):
        active = data["path_valid"] & (segment <= depth)
        if segment == 0:
            start = np.broadcast_to(
                v572.BS_POSITION.astype(np.float32),
                (v572.POINT_COUNT, v572.PATH_LIMIT, 3),
            )
        else:
            start = vertices[:, :, segment - 1]
        if segment < v572.MAX_STEPS:
            end = np.where(
                (segment < depth)[..., None],
                vertices[:, :, segment],
                data["positions"][:, None, :],
            )
        else:
            end = np.broadcast_to(
                data["positions"][:, None, :],
                (v572.POINT_COUNT, v572.PATH_LIMIT, 3),
            )
        flat_start = start[active]
        flat_end = end[active]
        local = segment_visibility_features(flat_start, flat_end, scene)
        segment_valid[:, :, segment] = active
        segment_features[:, :, segment][active] = local.astype(np.float16)
        segment_length[:, :, segment][active] = np.linalg.norm(
            flat_end - flat_start, axis=1
        ).astype(np.float32)
        print(
            f"v573 segment visibility {segment + 1}/{SEGMENT_COUNT} "
            f"active={int(np.sum(active))} elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )

    step_visibility = np.zeros(
        (
            v572.POINT_COUNT,
            v572.PATH_LIMIT,
            v572.MAX_STEPS,
            2 * SEGMENT_FEATURE_DIM,
        ),
        dtype=np.float16,
    )
    for step in range(v572.MAX_STEPS):
        step_visibility[:, :, step, :SEGMENT_FEATURE_DIM] = segment_features[
            :, :, step
        ]
        step_visibility[:, :, step, SEGMENT_FEATURE_DIM:] = segment_features[
            :, :, step + 1
        ]
    step_visibility *= data["step_valid"][..., None]

    valid_count = np.maximum(segment_valid.sum(axis=2), 1)
    segment_float = segment_features.astype(np.float32)
    mean = np.sum(segment_float * segment_valid[..., None], axis=2) / valid_count[..., None]
    maximum = np.max(
        np.where(segment_valid[..., None], segment_float, -1e4), axis=2
    )
    maximum[~data["path_valid"]] = 0.0
    first = segment_float[:, :, 0]
    last = np.take_along_axis(
        segment_float,
        depth[:, :, None, None],
        axis=2,
    )[:, :, 0]
    blocked_segments = np.sum(
        segment_valid & (segment_float[..., 0] > 0.25), axis=2
    ) / valid_count
    total_length = np.sum(segment_length, axis=2)
    direct = np.linalg.norm(
        data["positions"] - v572.BS_POSITION[None], axis=1
    )[:, None]
    route_numeric = np.concatenate(
        [
            mean,
            maximum,
            first,
            last,
            blocked_segments[..., None].astype(np.float32),
            (valid_count / SEGMENT_COUNT)[..., None].astype(np.float32),
            (total_length / 1000.0)[..., None],
            ((total_length - direct) / np.maximum(direct, 1.0))[..., None],
            (np.mean(segment_length, axis=2) / 250.0)[..., None],
        ],
        axis=2,
    ).astype(np.float32)
    route_numeric *= data["path_valid"][..., None]
    summary = {
        "samples_per_segment": SAMPLES_PER_SEGMENT,
        "segment_feature_dim": SEGMENT_FEATURE_DIM,
        "step_visibility_dim": int(step_visibility.shape[-1]),
        "route_numeric_dim": int(route_numeric.shape[-1]),
        "point_context_dim": int(point_context.shape[-1]),
        "valid_segments": int(np.sum(segment_valid)),
        "mean_obstructed_fraction": float(
            np.mean(segment_float[..., 0][segment_valid])
        ),
        "p90_obstructed_fraction": float(
            np.quantile(segment_float[..., 0][segment_valid], 0.90)
        ),
        "paths_with_blocked_segment": int(
            np.sum(data["path_valid"] & (blocked_segments > 0.0))
        ),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        VISIBILITY_CACHE,
        route_numeric=route_numeric.astype(np.float16),
        step_visibility=step_visibility,
        point_context=point_context.astype(np.float16),
        summary_json=np.asarray(json.dumps(summary, ensure_ascii=False)),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def load_data(
    force_visibility: bool,
) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, object]]:
    data, surface_summary, _oracle_summary = v572.load_data()
    visibility_summary = build_visibility_cache(data, force_visibility)
    with np.load(VISIBILITY_CACHE) as loaded:
        data["route_numeric"] = loaded["route_numeric"].astype(np.float32)
        data["step_visibility"] = loaded["step_visibility"].astype(np.float32)
        data["point_context"] = loaded["point_context"].astype(np.float32)
    data["surface_step_numeric"] = np.concatenate(
        [data["step_numeric"], data["step_visibility"]], axis=3
    ).astype(np.float32)
    return data, surface_summary, visibility_summary


def standardizers(
    data: dict[str, np.ndarray],
    fit_ids: np.ndarray,
) -> tuple[np.ndarray, ...]:
    path_mask = data["path_valid"][fit_ids]
    base = data["base_features"][fit_ids][path_mask]
    route = data["route_numeric"][fit_ids][path_mask]
    step_mask = data["step_valid"][fit_ids]
    step = data["surface_step_numeric"][fit_ids][step_mask]
    point = data["point_context"][fit_ids]
    return (
        base.mean(axis=0).astype(np.float32),
        np.maximum(base.std(axis=0), 1e-4).astype(np.float32),
        route.mean(axis=0).astype(np.float32),
        np.maximum(route.std(axis=0), 1e-4).astype(np.float32),
        step.mean(axis=0).astype(np.float32),
        np.maximum(step.std(axis=0), 1e-4).astype(np.float32),
        point.mean(axis=0).astype(np.float32),
        np.maximum(point.std(axis=0), 1e-4).astype(np.float32),
    )


def gather_path_array(
    value: np.ndarray,
    ids: np.ndarray,
    path_map: np.ndarray | None,
) -> np.ndarray:
    if path_map is None:
        return value[ids]
    flat = path_map[ids]
    return value[flat // v572.PATH_LIMIT, flat % v572.PATH_LIMIT]


def batch_tensors(
    ids: np.ndarray,
    data: dict[str, np.ndarray],
    normalization: tuple[np.ndarray, ...],
    device: torch.device,
    path_map: np.ndarray | None = None,
    point_map: np.ndarray | None = None,
) -> tuple[torch.Tensor, ...]:
    (
        base_mean,
        base_std,
        route_mean,
        route_std,
        step_mean,
        step_std,
        point_mean,
        point_std,
    ) = normalization
    base = (
        (data["base_features"][ids] - base_mean[None, None])
        / base_std[None, None]
    ).astype(np.float32)
    base *= data["path_valid"][ids, :, None]
    route = gather_path_array(data["route_numeric"], ids, path_map)
    route = ((route - route_mean[None, None]) / route_std[None, None]).astype(
        np.float32
    )
    route *= data["path_valid"][ids, :, None]
    step = gather_path_array(data["surface_step_numeric"], ids, path_map)
    step_valid = gather_path_array(data["step_valid"], ids, path_map)
    interaction = gather_path_array(data["interaction"], ids, path_map)
    object_token = gather_path_array(data["object_token"], ids, path_map)
    patch_id = gather_path_array(data["patch_id"], ids, path_map)
    step = ((step - step_mean[None, None, None]) / step_std[None, None, None]).astype(
        np.float32
    )
    step *= step_valid[..., None]
    context_ids = ids if point_map is None else point_map[ids]
    context = (
        (data["point_context"][context_ids] - point_mean[None]) / point_std[None]
    ).astype(np.float32)
    return (
        torch.from_numpy(base).to(device),
        torch.from_numpy(data["path_valid"][ids]).to(device),
        torch.from_numpy(route).to(device),
        torch.from_numpy(step).to(device),
        torch.from_numpy(interaction).to(device),
        torch.from_numpy(object_token).to(device),
        torch.from_numpy(patch_id).to(device),
        torch.from_numpy(step_valid).to(device),
        torch.from_numpy(context).to(device),
        torch.from_numpy(data["source_energy"][ids]).to(device),
        torch.from_numpy(data["delay_kernel"][ids].astype(np.float32)).to(device),
        torch.from_numpy(data["angle_kernel"][ids].astype(np.float32)).to(device),
        torch.from_numpy(data["base"][ids]).to(device),
        torch.from_numpy(data["truth"][ids]).to(device),
    )


class AdditiveSurfaceLaw(nn.Module):
    def __init__(
        self,
        data: dict[str, np.ndarray],
        surface_summary: dict[str, object],
        args: argparse.Namespace,
    ) -> None:
        super().__init__()
        hidden = args.hidden_dim
        sequence = args.sequence_dim
        self.max_log_gain = args.max_log_gain
        object_count = max(
            int(value) for value in surface_summary["object_lookup"].values()
        )
        patch_count = int(surface_summary["wall_patch_count"]) + 1
        self.base_encoder = nn.Sequential(
            nn.Linear(data["base_features"].shape[-1], hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.route_encoder = nn.Sequential(
            nn.Linear(data["route_numeric"].shape[-1], hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.point_encoder = nn.Sequential(
            nn.Linear(data["point_context"].shape[-1], hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.interaction_embedding = nn.Embedding(8, 12, padding_idx=0)
        self.object_embedding = nn.Embedding(object_count + 1, 10, padding_idx=0)
        self.patch_embedding = nn.Embedding(patch_count + 1, 20, padding_idx=0)
        nn.init.zeros_(self.patch_embedding.weight)
        self.patch_log_gain = nn.Embedding(patch_count + 1, 1, padding_idx=0)
        self.interaction_log_gain = nn.Embedding(8, 1, padding_idx=0)
        nn.init.zeros_(self.patch_log_gain.weight)
        nn.init.zeros_(self.interaction_log_gain.weight)
        step_input = data["surface_step_numeric"].shape[-1] + 45
        self.step_encoder = nn.Sequential(
            nn.Linear(step_input, sequence),
            nn.LayerNorm(sequence),
            nn.GELU(),
            nn.Linear(sequence, sequence),
            nn.GELU(),
        )
        self.continuous_step_gain = nn.Linear(sequence, 1)
        nn.init.zeros_(self.continuous_step_gain.weight)
        nn.init.zeros_(self.continuous_step_gain.bias)
        self.surface_projection = nn.Sequential(
            nn.Linear(2 * sequence, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(4 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        path_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=8,
            dim_feedforward=4 * hidden,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.path_encoder = nn.TransformerEncoder(path_layer, num_layers=3)
        self.contextual_gain = nn.Linear(hidden, 1)
        self.route_gain = nn.Linear(hidden, 1)
        nn.init.zeros_(self.contextual_gain.weight)
        nn.init.zeros_(self.contextual_gain.bias)
        nn.init.zeros_(self.route_gain.weight)
        nn.init.zeros_(self.route_gain.bias)

    def forward(
        self,
        base: torch.Tensor,
        path_valid: torch.Tensor,
        route: torch.Tensor,
        step: torch.Tensor,
        interaction: torch.Tensor,
        object_token: torch.Tensor,
        patch_id: torch.Tensor,
        step_valid: torch.Tensor,
        point: torch.Tensor,
    ) -> torch.Tensor:
        batch, paths, steps, _ = step.shape
        position = torch.arange(steps, device=step.device, dtype=torch.float32)
        phase = 2.0 * math.pi * position / max(steps, 1)
        position_feature = torch.stack(
            [position / max(steps - 1, 1), torch.sin(phase), torch.cos(phase)],
            dim=1,
        )[None, None].expand(batch, paths, -1, -1)
        step_token = self.step_encoder(
            torch.cat(
                [
                    step,
                    self.interaction_embedding(interaction),
                    self.object_embedding(object_token),
                    self.patch_embedding(patch_id),
                    position_feature,
                ],
                dim=-1,
            )
        )
        mask = step_valid.to(step_token.dtype)[..., None]
        count = torch.clamp(mask.sum(dim=2), min=1.0)
        mean = torch.sum(step_token * mask, dim=2) / count
        maximum = torch.max(
            torch.where(step_valid[..., None], step_token, torch.full_like(step_token, -1e4)),
            dim=2,
        ).values
        empty = ~torch.any(step_valid, dim=2)
        maximum[empty] = 0.0
        surface = self.surface_projection(torch.cat([mean, maximum], dim=-1))
        base_token = self.base_encoder(base)
        route_token = self.route_encoder(route)
        point_token = self.point_encoder(point)[:, None].expand(-1, paths, -1)
        token = self.combine(
            torch.cat([base_token, route_token, point_token, surface], dim=-1)
        )
        safe_valid = path_valid.clone()
        no_path = ~torch.any(safe_valid, dim=1)
        if torch.any(no_path):
            safe_valid[no_path, 0] = True
        token = self.path_encoder(token, src_key_padding_mask=~safe_valid)

        step_gain = self.continuous_step_gain(step_token)[..., 0]
        step_gain += self.patch_log_gain(patch_id)[..., 0]
        step_gain += self.interaction_log_gain(interaction)[..., 0]
        additive = torch.sum(step_gain * step_valid.to(step_gain.dtype), dim=2)
        additive /= torch.sqrt(torch.clamp(step_valid.sum(dim=2), min=1).to(additive.dtype))
        raw = additive + self.route_gain(route_token)[..., 0]
        raw += self.contextual_gain(token)[..., 0]
        log_gain = self.max_log_gain * torch.tanh(raw / self.max_log_gain)
        return log_gain * path_valid.to(log_gain.dtype)


def make_model(
    data: dict[str, np.ndarray],
    surface_summary: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> AdditiveSurfaceLaw:
    return AdditiveSurfaceLaw(data, surface_summary, args).to(device)


@torch.no_grad()
def evaluate(
    model: AdditiveSurfaceLaw,
    ids: np.ndarray,
    data: dict[str, np.ndarray],
    normalization: tuple[np.ndarray, ...],
    device: torch.device,
    args: argparse.Namespace,
    path_map: np.ndarray | None = None,
    point_map: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    model.eval()
    output = np.zeros((len(ids), v572.HEIGHT * v572.WIDTH), dtype=np.float32)
    gains = []
    for start in range(0, len(ids), args.batch_size):
        local_ids = ids[start : start + args.batch_size]
        tensors = batch_tensors(
            local_ids,
            data,
            normalization,
            device,
            path_map=path_map,
            point_map=point_map,
        )
        log_gain = model(*tensors[:9])
        physical = v572.render_spectrum(
            tensors[9], tensors[10], tensors[11], log_gain, tensors[1]
        )
        output[start : start + len(local_ids)] = physical.cpu().numpy()
        gains.extend(torch.abs(log_gain[tensors[1]]).cpu().numpy().tolist())
    return output, {
        "mean_abs_log_gain": float(np.mean(gains)) if gains else 0.0,
        "p90_abs_log_gain": float(np.quantile(gains, 0.90)) if gains else 0.0,
    }


def train_model(
    train_ids: np.ndarray,
    validation_ids: np.ndarray | None,
    data: dict[str, np.ndarray],
    normalization: tuple[np.ndarray, ...],
    surface_summary: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    fixed_epochs: int | None = None,
) -> tuple[AdditiveSurfaceLaw, dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(data, surface_summary, args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(seed)
    epochs = args.epochs if fixed_epochs is None else fixed_epochs
    best_score = float("-inf")
    best_epoch = 0
    best_alpha = 0.0
    best_state = v572.state_dict_cpu(model)
    history = []
    stale = 0
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_ids)
        running = 0.0
        for start in range(0, len(order), args.batch_size):
            ids = order[start : start + args.batch_size]
            tensors = batch_tensors(ids, data, normalization, device)
            log_gain = model(*tensors[:9])
            physical = v572.render_spectrum(
                tensors[9], tensors[10], tensors[11], log_gain, tensors[1]
            )
            prediction = v572.torch_log_blend(tensors[12], physical, args.train_alpha)
            score = v572.torch_scores(tensors[13], prediction)[-1]
            source_weight = tensors[9] / torch.clamp(
                tensors[9].sum(dim=1, keepdim=True), min=1e-20
            )
            regularization = torch.mean(
                torch.sum(source_weight * log_gain * log_gain, dim=1)
            )
            loss = 1.0 - torch.mean(score) + 2e-4 * regularization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(ids)

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            if validation_ids is not None:
                physical, diagnostics = evaluate(
                    model,
                    validation_ids,
                    data,
                    normalization,
                    device,
                    args,
                )
                alpha, validation_score, _ = v572.select_alpha(
                    data["truth"][validation_ids],
                    data["base"][validation_ids],
                    physical,
                )
            else:
                diagnostics = {"mean_abs_log_gain": 0.0, "p90_abs_log_gain": 0.0}
                alpha = 1.0
                validation_score = float("nan")
            row = {
                "epoch": epoch,
                "train_loss": running / len(train_ids),
                "validation_score": validation_score,
                "validation_alpha": alpha,
                **diagnostics,
            }
            history.append(row)
            print(
                f"v573 epoch={epoch:03d} loss={row['train_loss']:.6f} "
                f"validation={validation_score:.6f} alpha={alpha:.2f} "
                f"move={row['mean_abs_log_gain']:.3f}",
                flush=True,
            )
            if validation_ids is not None:
                if validation_score > best_score + 1e-5:
                    best_score = validation_score
                    best_epoch = epoch
                    best_alpha = alpha
                    best_state = v572.state_dict_cpu(model)
                    stale = 0
                else:
                    stale += 5
                    if stale >= args.patience:
                        break
    if validation_ids is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
    return model, {
        "best_epoch": int(best_epoch),
        "best_validation_score": (
            float(best_score) if validation_ids is not None else None
        ),
        "best_alpha": float(best_alpha),
        "elapsed_seconds": float(time.monotonic() - started),
        "history": history,
    }


def shuffled_point_map(
    ids: np.ndarray,
    data: dict[str, np.ndarray],
    seed: int,
) -> np.ndarray:
    mapping = np.arange(v572.POINT_COUNT, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for regime in v572.REGIME_NAMES:
        local = ids[data["regimes"][ids] == regime]
        if len(local) > 1:
            mapping[local] = rng.permutation(local)
    return mapping


def run_fold(
    fold: int,
    data: dict[str, np.ndarray],
    surface_summary: dict[str, object],
    visibility_summary: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    fold_path = FOLD_DIR / f"fold_{fold}.npz"
    if fold_path.is_file() and not args.force_fold:
        with np.load(fold_path) as loaded:
            return json.loads(str(loaded["details"].item()))
    valid = data["point_valid"]
    outer = data["folds"] == fold
    fit_mask = valid & ~outer
    query_mask = valid & outer
    tune_fold = (fold + 1) % 5
    inner_validation_mask = fit_mask & (data["folds"] == tune_fold)
    inner_train_mask = fit_mask & ~inner_validation_mask
    inner_train_ids = np.flatnonzero(inner_train_mask)
    inner_validation_ids = np.flatnonzero(inner_validation_mask)
    fit_ids = np.flatnonzero(fit_mask)
    query_ids = np.flatnonzero(query_mask)
    print(
        f"v573 fold={fold} inner_train={len(inner_train_ids)} "
        f"inner_validation={len(inner_validation_ids)} outer={len(query_ids)}",
        flush=True,
    )
    inner_norm = standardizers(data, inner_train_ids)
    selected_model, selection = train_model(
        inner_train_ids,
        inner_validation_ids,
        data,
        inner_norm,
        surface_summary,
        args,
        device,
        args.seed + 1000 * fold,
    )
    selected_epoch = int(selection["best_epoch"])
    selected_alpha = float(selection["best_alpha"])
    del selected_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    final_norm = standardizers(data, fit_ids)
    model, final_training = train_model(
        fit_ids,
        None,
        data,
        final_norm,
        surface_summary,
        args,
        device,
        args.seed + 1000 * fold + 1,
        fixed_epochs=selected_epoch,
    )
    physical, diagnostics = evaluate(
        model, query_ids, data, final_norm, device, args
    )
    prediction = v572.v543.log_blend(
        data["base"][query_ids], physical, selected_alpha
    )
    path_map = v572.path_shuffle_map(
        query_ids, data, args.seed + 20_000 + fold
    )
    shuffled_path_physical, shuffled_path_diagnostics = evaluate(
        model,
        query_ids,
        data,
        final_norm,
        device,
        args,
        path_map=path_map,
    )
    shuffled_path = v572.v543.log_blend(
        data["base"][query_ids], shuffled_path_physical, selected_alpha
    )
    point_map = shuffled_point_map(
        query_ids, data, args.seed + 30_000 + fold
    )
    shuffled_all_physical, shuffled_all_diagnostics = evaluate(
        model,
        query_ids,
        data,
        final_norm,
        device,
        args,
        path_map=path_map,
        point_map=point_map,
    )
    shuffled_all = v572.v543.log_blend(
        data["base"][query_ids], shuffled_all_physical, selected_alpha
    )
    truth = data["truth"][query_ids]
    baseline_metrics = v572.metric_summary(truth, data["base"][query_ids])
    prediction_metrics = v572.metric_summary(truth, prediction)
    shuffled_path_metrics = v572.metric_summary(truth, shuffled_path)
    shuffled_all_metrics = v572.metric_summary(truth, shuffled_all)
    v572_reference = None
    v572_fold = v572.FOLD_DIR / f"fold_{fold}.npz"
    if v572_fold.is_file():
        with np.load(v572_fold) as loaded:
            saved_ids = loaded["query_ids"].astype(np.int64)
            if np.array_equal(saved_ids, query_ids):
                v572_reference = v572.metric_summary(
                    truth, loaded["surface_prediction"].astype(np.float32)
                )
    details = {
        "fold": fold,
        "query_points": int(len(query_ids)),
        "model_config": {
            "max_log_gain": float(args.max_log_gain),
            "train_alpha": float(args.train_alpha),
            "hidden_dim": int(args.hidden_dim),
            "sequence_dim": int(args.sequence_dim),
        },
        "selected_epoch": selected_epoch,
        "selected_alpha": selected_alpha,
        "selection": selection,
        "final_training": final_training,
        "diagnostics": diagnostics,
        "baseline": baseline_metrics,
        "prediction": prediction_metrics,
        "gain_vs_v543": {
            key: float(prediction_metrics[key] - baseline_metrics[key])
            for key in prediction_metrics
        },
        "v572_surface_reference": v572_reference,
        "margin_vs_v572": (
            float(prediction_metrics["weighted"] - v572_reference["weighted"])
            if v572_reference is not None
            else None
        ),
        "shuffled_path_physics": {
            "metrics": shuffled_path_metrics,
            "diagnostics": shuffled_path_diagnostics,
            "margin": float(
                prediction_metrics["weighted"] - shuffled_path_metrics["weighted"]
            ),
        },
        "shuffled_path_and_point_context": {
            "metrics": shuffled_all_metrics,
            "diagnostics": shuffled_all_diagnostics,
            "margin": float(
                prediction_metrics["weighted"] - shuffled_all_metrics["weighted"]
            ),
        },
        "decision": {
            "hard_region_success": bool(
                prediction_metrics["weighted"] - baseline_metrics["weighted"] >= 0.005
                and prediction_metrics["weighted"] - shuffled_path_metrics["weighted"] >= 0.003
                and prediction_metrics["weighted"] - shuffled_all_metrics["weighted"] >= 0.003
            )
        },
    }
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": VERSION,
            "fold": fold,
            "state_dict": v572.state_dict_cpu(model),
            "normalization": [torch.from_numpy(value) for value in final_norm],
            "details": details,
        },
        CHECKPOINT_DIR / f"fold_{fold}.pt",
    )
    np.savez_compressed(
        fold_path,
        query_ids=query_ids.astype(np.int32),
        prediction=prediction.astype(np.float32),
        shuffled_path=shuffled_path.astype(np.float32),
        shuffled_all=shuffled_all.astype(np.float32),
        details=np.asarray(json.dumps(v572.json_safe(details), ensure_ascii=False)),
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "Path power is a product of free-space and ordered interaction gains, so the "
            "correction should be additive in log power along the surface sequence. Official "
            "pointcloud segment clearance identifies false or disappearing Sionna paths."
        ),
        "new_information": [
            "per-segment official-pointcloud clearance and building-crossing statistics",
            "ordered additive wall-patch and interaction log-gain law",
            "73-dimensional official pointcloud context at each receiver",
        ],
        "surface_alignment": surface_summary,
        "visibility": visibility_summary,
        "result": details,
        "decision": {
            "expand_five_fold": bool(details["decision"]["hard_region_success"]),
            "submission_generated": False,
        },
        "outputs": {
            "visibility_cache": PATHS.relative(VISIBILITY_CACHE),
            "fold_result": PATHS.relative(fold_path),
            "checkpoint": PATHS.relative(CHECKPOINT_DIR / f"fold_{fold}.pt"),
            "report": PATHS.relative(REPORT),
            "submission_zip": None,
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(v572.json_safe(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(details["gain_vs_v543"], indent=2), flush=True)
    print(json.dumps(details["decision"], indent=2), flush=True)
    return details


def main() -> None:
    args = parse_args()
    for path in (v572.SURFACE_CACHE, v572.SOURCE_DATA, v572.OOF_SOURCE, v572.SCENE_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    data, surface_summary, visibility_summary = load_data(
        args.force_visibility_cache
    )
    if args.build_only:
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    run_fold(
        args.fold,
        data,
        surface_summary,
        visibility_summary,
        args,
        device,
    )


if __name__ == "__main__":
    main()
