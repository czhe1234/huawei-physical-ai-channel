#!/usr/bin/env python3
"""v293: shared interaction-level physical law for depth-16 path geometry.

The model never receives Sionna path coefficients. It converts official-pointcloud
geometry-only paths into PDP atoms and learns one globally shared, monotone path-loss
law from official train H. Per-point freedom is restricted to geometry-derived path
sets and a bounded anchor/path gate.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v208_strict_historical_replay as v208
import analyze_v225_official_graph_residual_field as v225
import analyze_v254_route_topology_spectral_field as v254
import analyze_v292_highdepth_path_weight_transformer as v292
import generate_v215_pointcloud_physics_neural_field as v215
from generate_v87_easy_field_projection import S_SC, summarize
from powerfield_common import TINY, regime_of
from project_paths import PATHS


VERSION_NUMBER = 293
VERSION = "v293_interaction_physical_path_law"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
FEATURE_CACHE = OUT_DIR / "v293_interaction_features.npz"
TARGETS_OUT = OUT_DIR / "v293_physical_path_targets.npz"
OUTPUT = OUT_DIR / "v293_Round1_Test_Channel.npy"
PATH_CACHE = PATHS.cache_dir(291) / "v291_highdepth_pointcloud_paths.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"
GEOM_CACHE = PATHS.shared_features / "height_geometry_cache.npz"

LOCAL_GATE = 0.645604
OUTER_SEEDS = (293, 1293, 2293, 3293, 4293)
OUTER_COUNT = 320
INNER_COUNT = 240
EPOCHS = 80
FINAL_SEEDS = (29301, 29331, 29361)
MIN_OFFICIAL_DELTA = 5e-4
ALPHA_GRID = np.linspace(0.0, 1.5, 13, dtype=np.float32)
NEIGHBORS = 16
MAX_PATHS = 128
MAX_DEPTH = 16
SHIFTS = (-0.85, 0.0, 0.85)
TAIL_ATOMS = 6
BATCH_SIZE = 48


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    out /= np.maximum(out.sum(axis=1, keepdims=True), TINY)
    return out.astype(np.float32)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    y = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def fixed_atoms(max_delay: np.ndarray) -> np.ndarray:
    x = np.arange(S_SC, dtype=np.float32)[None, :]
    early0 = np.exp(-0.5 * (x / 0.55) ** 2)
    early1 = np.exp(-0.5 * ((x - 1.0) / 0.75) ** 2)
    early2 = np.exp(-0.5 * ((x - 2.0) / 1.05) ** 2)
    broad = np.exp(-0.5 * ((x - 10.0) / 4.5) ** 2)
    delta = np.maximum(x - max_delay[:, None], 0.0)
    causal = (x >= max_delay[:, None]).astype(np.float32)
    tail8 = np.exp(-delta / 8.0) * causal
    tail22 = np.exp(-delta / 22.0) * causal
    atoms = np.stack(
        [
            np.broadcast_to(early0, tail8.shape),
            np.broadcast_to(early1, tail8.shape),
            np.broadcast_to(early2, tail8.shape),
            np.broadcast_to(broad, tail8.shape),
            tail8,
            tail22,
        ],
        axis=1,
    )
    atoms /= np.maximum(atoms.sum(axis=2, keepdims=True), 1e-30)
    return atoms.astype(np.float32)


def _unit(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = np.linalg.norm(vector, axis=2)
    return vector / np.maximum(length[..., None], 1e-6), length


def interaction_features(
    cache: np.lib.npyio.NpzFile,
    prefix: str,
    positions: np.ndarray,
) -> dict[str, np.ndarray]:
    valid = cache[f"{prefix}_valid"].astype(bool)
    delay = np.nan_to_num(
        cache[f"{prefix}_delay_bin_physical"].astype(np.float32), nan=0.0
    )
    length = np.nan_to_num(
        cache[f"{prefix}_path_length"].astype(np.float32),
        nan=0.0,
        posinf=2500.0,
    )
    depth = cache[f"{prefix}_interaction_count"].astype(np.int16)
    interactions = cache[f"{prefix}_interactions"].astype(np.int16)
    vertices = np.nan_to_num(
        cache[f"{prefix}_vertices"].astype(np.float32), nan=0.0
    )
    theta_t = np.nan_to_num(cache[f"{prefix}_theta_t"].astype(np.float32), nan=0.0)
    phi_t = np.nan_to_num(cache[f"{prefix}_phi_t"].astype(np.float32), nan=0.0)
    theta_r = np.nan_to_num(cache[f"{prefix}_theta_r"].astype(np.float32), nan=0.0)
    phi_r = np.nan_to_num(cache[f"{prefix}_phi_r"].astype(np.float32), nan=0.0)

    count = len(valid)
    bs = np.broadcast_to(
        cache["bs_pos"].astype(np.float32)[None, None, :],
        (count, MAX_PATHS, 3),
    )
    ue = np.broadcast_to(
        positions.astype(np.float32)[:, None, :], (count, MAX_PATHS, 3)
    )
    reflection_count = np.zeros_like(length)
    refraction_count = np.zeros_like(length)
    ground_reflection = np.zeros_like(length)
    reflection_cos_sum = np.zeros_like(length)
    reflection_cos_min = np.ones_like(length)
    reflection_cos_max = np.zeros_like(length)
    refraction_bend_sum = np.zeros_like(length)
    refraction_bend_max = np.zeros_like(length)
    segment_sum = np.zeros_like(length)
    segment_sq_sum = np.zeros_like(length)
    segment_min = np.full_like(length, np.inf)
    segment_count = np.zeros_like(length)

    previous = bs
    for step in range(MAX_DEPTH):
        active = valid & (depth > step)
        vertex = vertices[:, :, step]
        has_next = depth > (step + 1)
        next_point = np.where(has_next[..., None], vertices[:, :, min(step + 1, MAX_DEPTH - 1)], ue)
        incoming, incoming_length = _unit(vertex - previous)
        outgoing, _ = _unit(next_point - vertex)
        turn_dot = np.clip(np.sum(incoming * outgoing, axis=2), -1.0, 1.0)
        reflection_cos = np.sqrt(np.maximum((1.0 - turn_dot) * 0.5, 0.0))
        bend = np.sqrt(np.maximum((1.0 - turn_dot) * 0.5, 0.0))
        interaction = interactions[:, :, step]
        is_reflection = active & (interaction == 1)
        is_refraction = active & (interaction == 4)
        is_ground = is_reflection & (vertex[:, :, 2] <= 0.20)

        reflection_count += is_reflection
        refraction_count += is_refraction
        ground_reflection += is_ground
        reflection_cos_sum += reflection_cos * is_reflection
        reflection_cos_min = np.where(
            is_reflection, np.minimum(reflection_cos_min, reflection_cos), reflection_cos_min
        )
        reflection_cos_max = np.where(
            is_reflection, np.maximum(reflection_cos_max, reflection_cos), reflection_cos_max
        )
        refraction_bend_sum += bend * is_refraction
        refraction_bend_max = np.where(
            is_refraction, np.maximum(refraction_bend_max, bend), refraction_bend_max
        )
        segment_sum += incoming_length * active
        segment_sq_sum += incoming_length**2 * active
        segment_min = np.where(active, np.minimum(segment_min, incoming_length), segment_min)
        segment_count += active
        previous = vertex

    last_vertex = np.zeros_like(vertices[:, :, 0])
    for step in range(MAX_DEPTH):
        last_vertex = np.where((depth == step + 1)[..., None], vertices[:, :, step], last_vertex)
    tail_start = np.where((depth > 0)[..., None], last_vertex, bs)
    _, tail_length = _unit(ue - tail_start)
    segment_sum += tail_length * valid
    segment_sq_sum += tail_length**2 * valid
    segment_min = np.where(valid, np.minimum(segment_min, tail_length), segment_min)
    segment_count += valid

    reflection_mean = reflection_cos_sum / np.maximum(reflection_count, 1.0)
    reflection_cos_min = np.where(reflection_count > 0, reflection_cos_min, 0.0)
    refraction_bend_mean = refraction_bend_sum / np.maximum(refraction_count, 1.0)
    segment_mean = segment_sum / np.maximum(segment_count, 1.0)
    segment_var = segment_sq_sum / np.maximum(segment_count, 1.0) - segment_mean**2
    segment_cv = np.sqrt(np.maximum(segment_var, 0.0)) / np.maximum(segment_mean, 1e-3)
    segment_min = np.where(valid, segment_min, 0.0)
    wall_reflection = np.maximum(reflection_count - ground_reflection, 0.0)

    minimum_length = np.min(np.where(valid, length, np.inf), axis=1, keepdims=True)
    minimum_length = np.where(np.isfinite(minimum_length), minimum_length, 1.0)
    relative_log_length = np.log(
        np.maximum(length, 1.0) / np.maximum(minimum_length, 1.0)
    )
    fractional_delay = delay - np.floor(delay)
    path_features = np.stack(
        [
            valid.astype(np.float32),
            np.log1p(length) / 8.0,
            relative_log_length / 5.0,
            delay / 48.0,
            fractional_delay,
            depth.astype(np.float32) / MAX_DEPTH,
            reflection_count / MAX_DEPTH,
            refraction_count / MAX_DEPTH,
            wall_reflection / MAX_DEPTH,
            ground_reflection / MAX_DEPTH,
            reflection_mean,
            reflection_cos_min,
            reflection_cos_max,
            refraction_bend_mean,
            refraction_bend_max,
            np.log1p(segment_min) / 7.0,
            np.clip(segment_cv, 0.0, 4.0) / 4.0,
            np.sin(theta_t),
            np.cos(theta_t),
            np.sin(phi_t),
            np.cos(phi_t),
            np.sin(theta_r),
            np.cos(theta_r),
            np.sin(phi_r),
            np.cos(phi_r),
        ],
        axis=2,
    ).astype(np.float32)
    path_features[~valid] = 0.0
    max_delay = np.max(np.where(valid, delay, 0.0), axis=1)
    depth_hist = np.stack(
        [np.mean(valid & (depth == value), axis=1) for value in range(MAX_DEPTH + 1)],
        axis=1,
    ).astype(np.float32)
    summary = np.concatenate(
        [
            valid.mean(axis=1, keepdims=True).astype(np.float32),
            (np.max(depth, axis=1, keepdims=True) / MAX_DEPTH).astype(np.float32),
            (max_delay[:, None] / 48.0).astype(np.float32),
            depth_hist,
        ],
        axis=1,
    ).astype(np.float32)
    return {
        "valid": valid,
        "delay": delay.astype(np.float32),
        "depth": depth.astype(np.float32),
        "reflection_count": reflection_count.astype(np.float32),
        "refraction_count": refraction_count.astype(np.float32),
        "ground_reflection": ground_reflection.astype(np.float32),
        "reflection_mean": reflection_mean.astype(np.float32),
        "relative_log_length": relative_log_length.astype(np.float32),
        "path_features": np.nan_to_num(path_features, nan=0.0, posinf=8.0, neginf=-8.0),
        "summary": np.nan_to_num(summary, nan=0.0, posinf=8.0, neginf=-8.0),
        "tail_atoms": fixed_atoms(max_delay),
    }


def build_feature_cache(path_cache: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    if FEATURE_CACHE.is_file():
        data = np.load(FEATURE_CACHE)
        return {key: data[key] for key in data.files}
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float32)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float32)
    train = interaction_features(path_cache, "train", train_pos)
    test = interaction_features(path_cache, "test", test_pos)
    payload = {
        f"train_{key}": value for key, value in train.items()
    } | {f"test_{key}": value for key, value in test.items()}
    np.savez_compressed(FEATURE_CACHE, **payload)
    return payload


@dataclass
class Prepared:
    path_features: np.ndarray
    summary: np.ndarray
    valid: np.ndarray
    delay: np.ndarray
    depth: np.ndarray
    reflection_count: np.ndarray
    refraction_count: np.ndarray
    ground_reflection: np.ndarray
    reflection_mean: np.ndarray
    relative_log_length: np.ndarray
    tail_atoms: np.ndarray

    def take(self, ids: np.ndarray) -> "Prepared":
        return Prepared(**{key: getattr(self, key)[ids] for key in self.__dataclass_fields__})


def prepared(features: dict[str, np.ndarray], prefix: str) -> Prepared:
    return Prepared(
        **{
            key: features[f"{prefix}_{key}"]
            for key in Prepared.__dataclass_fields__
        }
    )


def standardize(train: Prepared, others: list[Prepared]) -> tuple[Prepared, list[Prepared]]:
    values = train.path_features[train.valid]
    p_mean = values.mean(axis=0).astype(np.float32)
    p_std = np.maximum(values.std(axis=0), 1e-5).astype(np.float32)
    g_mean = train.summary.mean(axis=0).astype(np.float32)
    g_std = np.maximum(train.summary.std(axis=0), 1e-5).astype(np.float32)

    def convert(value: Prepared) -> Prepared:
        path = np.clip(
            (value.path_features - p_mean[None, None]) / p_std[None, None],
            -8.0,
            8.0,
        ).astype(np.float32)
        path[~value.valid] = 0.0
        summary = np.clip((value.summary - g_mean) / g_std, -8.0, 8.0).astype(np.float32)
        return Prepared(
            path,
            summary,
            value.valid,
            value.delay,
            value.depth,
            value.reflection_count,
            value.refraction_count,
            value.ground_reflection,
            value.reflection_mean,
            value.relative_log_length,
            value.tail_atoms,
        )

    return convert(train), [convert(item) for item in others]


class PhysicalPathLaw(nn.Module):
    def __init__(self, path_dim: int, summary_dim: int):
        super().__init__()
        self.raw_gamma = nn.Parameter(torch.tensor(0.55))
        self.raw_wall_loss = nn.Parameter(torch.tensor(-0.25))
        self.raw_ground_loss = nn.Parameter(torch.tensor(-0.65))
        self.raw_refraction_loss = nn.Parameter(torch.tensor(-0.15))
        self.path_residual = nn.Sequential(
            nn.Linear(path_dim, 48),
            nn.SiLU(),
            nn.Linear(48, 24),
            nn.SiLU(),
            nn.Linear(24, 1),
        )
        self.shift_residual = nn.Sequential(
            nn.Linear(path_dim, 32), nn.SiLU(), nn.Linear(32, len(SHIFTS))
        )
        self.tail_head = nn.Sequential(
            nn.Linear(summary_dim, 48), nn.SiLU(), nn.Linear(48, TAIL_ATOMS)
        )
        self.gate_head = nn.Sequential(
            nn.Linear(summary_dim, 32), nn.SiLU(), nn.Linear(32, 1)
        )
        self.shift_bias = nn.Parameter(torch.zeros(len(SHIFTS)))
        self.tail_bias = nn.Parameter(
            torch.tensor([0.6, 0.3, 0.0, -0.8, -1.6, -2.0])
        )
        nn.init.zeros_(self.path_residual[-1].weight)
        nn.init.zeros_(self.path_residual[-1].bias)
        nn.init.zeros_(self.shift_residual[-1].weight)
        nn.init.zeros_(self.shift_residual[-1].bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, -1.5)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma = 1.0 + F.softplus(self.raw_gamma)
        wall_loss = F.softplus(self.raw_wall_loss)
        ground_loss = F.softplus(self.raw_ground_loss)
        refraction_loss = F.softplus(self.raw_refraction_loss)
        wall_count = torch.clamp(
            batch["reflection_count"] - batch["ground_reflection"], min=0.0
        )
        physical_score = (
            -gamma * batch["relative_log_length"]
            - wall_loss * wall_count * batch["reflection_mean"]
            - ground_loss * batch["ground_reflection"] * batch["reflection_mean"]
            - refraction_loss * batch["refraction_count"]
        )
        residual = 0.85 * torch.tanh(self.path_residual(batch["path_features"]).squeeze(-1))
        path_score = physical_score + residual
        shift_score = self.shift_bias[None, None] + 0.55 * torch.tanh(
            self.shift_residual(batch["path_features"])
        )
        path_score = path_score[..., None] + shift_score
        path_score = path_score.masked_fill(~batch["valid"][..., None], -1e4)
        tail_score = self.tail_bias[None] + 0.75 * torch.tanh(
            self.tail_head(batch["summary"])
        )
        all_score = torch.cat([path_score.flatten(1), tail_score], dim=1)
        weight = torch.softmax(all_score, dim=1)
        path_weight = weight[:, : MAX_PATHS * len(SHIFTS)].reshape(
            -1, MAX_PATHS, len(SHIFTS)
        )
        tail_weight = weight[:, MAX_PATHS * len(SHIFTS) :]

        bins = torch.arange(S_SC, device=anchor.device, dtype=torch.float32)
        shifts = torch.tensor(SHIFTS, device=anchor.device, dtype=torch.float32)
        center = batch["delay"][..., None, None] + shifts[None, None, :, None]
        sigma = torch.where(batch["depth"] == 0, 0.62, 0.96)[..., None, None]
        atoms = torch.exp(-0.5 * ((bins[None, None, None] - center) / sigma) ** 2)
        atoms = atoms / torch.clamp(atoms.sum(dim=3, keepdim=True), min=1e-20)
        path_pdp = torch.sum(path_weight[..., None] * atoms, dim=(1, 2))
        path_pdp = path_pdp + torch.sum(
            tail_weight[..., None] * batch["tail_atoms"], dim=1
        )
        path_pdp = path_pdp / torch.clamp(path_pdp.sum(dim=1, keepdim=True), min=1e-20)
        gate = 0.45 * torch.sigmoid(self.gate_head(batch["summary"]))
        model = (1.0 - gate) * anchor + gate * path_pdp
        model = model / torch.clamp(model.sum(dim=1, keepdim=True), min=1e-20)
        return model, path_pdp, gate.squeeze(1)


def tensor_batch(data: Prepared, ids: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    keys = Prepared.__dataclass_fields__.keys()
    output: dict[str, torch.Tensor] = {}
    for key in keys:
        value = getattr(data, key)[ids]
        if value.dtype == bool:
            output[key] = torch.from_numpy(value).to(device=device, dtype=torch.bool)
        else:
            output[key] = torch.from_numpy(value).to(device=device, dtype=torch.float32)
    return output


def cosine_loss(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(prediction, truth, dim=1).mean()


def predict(
    model: PhysicalPathLaw,
    data: Prepared,
    anchor: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    model_parts: list[np.ndarray] = []
    path_parts: list[np.ndarray] = []
    gate_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(anchor), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(anchor))
            ids = np.arange(start, end)
            batch = tensor_batch(data, ids, device)
            anchor_t = torch.from_numpy(anchor[start:end]).to(device)
            model_d, path_d, gate = model(batch, anchor_t)
            model_parts.append(model_d.cpu().numpy().astype(np.float32))
            path_parts.append(path_d.cpu().numpy().astype(np.float32))
            gate_parts.append(gate.cpu().numpy().astype(np.float32))
    return (
        np.concatenate(model_parts),
        np.concatenate(path_parts),
        np.concatenate(gate_parts),
    )


def train_model(
    train: Prepared,
    train_anchor: np.ndarray,
    train_truth: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
    validation: tuple[Prepared, np.ndarray, np.ndarray] | None,
) -> tuple[PhysicalPathLaw, dict[str, object]]:
    set_seed(seed)
    model = PhysicalPathLaw(
        train.path_features.shape[2], train.summary.shape[1]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)
    best_epoch = epochs
    best_score = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    order = np.arange(len(train_anchor))
    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(order)
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start : start + BATCH_SIZE]
            batch = tensor_batch(train, ids, device)
            anchor_t = torch.from_numpy(train_anchor[ids]).to(device)
            truth_t = torch.from_numpy(train_truth[ids]).to(device)
            model_d, path_d, gate = model(batch, anchor_t)
            loss = (
                cosine_loss(model_d, truth_t)
                + 0.06 * cosine_loss(path_d, truth_t)
                + 0.002 * torch.mean(gate**2)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            item: dict[str, object] = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
            }
            if validation is not None:
                val_data, val_anchor, val_truth = validation
                val_model, val_path, val_gate = predict(model, val_data, val_anchor, device)
                value = float(np.mean(row_cos(val_model, val_truth)))
                item.update(
                    {
                        "val_cos": value,
                        "val_path_cos": float(np.mean(row_cos(val_path, val_truth))),
                        "val_gate_mean": float(np.mean(val_gate)),
                    }
                )
                if value > best_score:
                    best_score = value
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            history.append(item)
            print(f"v293 seed={seed} {json.dumps(item)}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_cos": None if not np.isfinite(best_score) else float(best_score),
        "history": history,
    }


def choose_alpha(base: np.ndarray, model: np.ndarray, truth: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    curve = []
    for alpha in ALPHA_GRID:
        value = normalize(base + float(alpha) * (model - base))
        curve.append(
            {"alpha": float(alpha), "cosine": float(np.mean(row_cos(value, truth)))}
        )
    return float(max(curve, key=lambda item: item["cosine"])["alpha"]), curve


def main() -> None:
    for path in (PATH_CACHE, TARGET_CACHE, BASE, GEOM_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    path_cache = np.load(PATH_CACHE)
    features = build_feature_cache(path_cache)
    train_data_all = prepared(features, "train")
    test_data_all = prepared(features, "test")
    targets = np.load(TARGET_CACHE)
    truth_d = normalize(targets["d"].astype(np.float32))
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"v293 device={device} valid={len(valid_ids)}", flush=True)

    outer_rows: list[dict[str, object]] = []
    oof_ids_parts: list[np.ndarray] = []
    oof_anchor_parts: list[np.ndarray] = []
    oof_model_parts: list[np.ndarray] = []
    selected_epochs: list[int] = []
    for fold, seed in enumerate(OUTER_SEEDS):
        outer_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], OUTER_COUNT, seed
            ),
            valid_ids,
        )
        outer_fit = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], INNER_COUNT, seed + 50_000
            ),
            outer_fit,
        )
        inner_fit = np.setdiff1d(outer_fit, inner_ids)
        inner_train_anchor = v292.knn_anchor(
            inner_fit, inner_fit, train_pos, train_pos, truth_d, True
        )
        inner_val_anchor = v292.knn_anchor(
            inner_ids, inner_fit, train_pos, train_pos, truth_d, False
        )
        inner_train_raw = train_data_all.take(inner_fit)
        inner_val_raw = train_data_all.take(inner_ids)
        inner_train, [inner_val] = standardize(inner_train_raw, [inner_val_raw])
        _, inner_training = train_model(
            inner_train,
            inner_train_anchor,
            truth_d[inner_fit],
            seed,
            EPOCHS,
            device,
            (inner_val, inner_val_anchor, truth_d[inner_ids]),
        )
        selected_epoch = int(inner_training["best_epoch"])
        selected_epochs.append(selected_epoch)

        outer_train_anchor = v292.knn_anchor(
            outer_fit, outer_fit, train_pos, train_pos, truth_d, True
        )
        outer_val_anchor = v292.knn_anchor(
            outer_ids, outer_fit, train_pos, train_pos, truth_d, False
        )
        outer_train_raw = train_data_all.take(outer_fit)
        outer_val_raw = train_data_all.take(outer_ids)
        outer_train, [outer_val] = standardize(outer_train_raw, [outer_val_raw])
        model, outer_training = train_model(
            outer_train,
            outer_train_anchor,
            truth_d[outer_fit],
            seed + 100_000,
            selected_epoch,
            device,
            None,
        )
        model_d, path_d, gate = predict(model, outer_val, outer_val_anchor, device)
        baseline = float(np.mean(row_cos(outer_val_anchor, truth_d[outer_ids])))
        model_score = float(np.mean(row_cos(model_d, truth_d[outer_ids])))
        outer_rows.append(
            {
                "fold": fold,
                "seed": seed,
                "fit_count": int(len(outer_fit)),
                "validation_count": int(len(outer_ids)),
                "selected_epoch": selected_epoch,
                "inner_training": inner_training,
                "outer_training": outer_training,
                "baseline": baseline,
                "model": model_score,
                "raw_delta": model_score - baseline,
                "path_cosine": float(np.mean(row_cos(path_d, truth_d[outer_ids]))),
                "gate": summarize(gate),
            }
        )
        oof_ids_parts.append(outer_ids)
        oof_anchor_parts.append(outer_val_anchor)
        oof_model_parts.append(model_d)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"v293 outer {fold + 1}/5 epoch={selected_epoch} "
            f"base={baseline:.6f} model={model_score:.6f} "
            f"delta={model_score - baseline:+.6f}",
            flush=True,
        )

    oof_ids = np.concatenate(oof_ids_parts)
    oof_anchor = np.concatenate(oof_anchor_parts)
    oof_model = np.concatenate(oof_model_parts)
    alpha, alpha_curve = choose_alpha(oof_anchor, oof_model, truth_d[oof_ids])
    fold_delta = []
    for row, ids, anchor, model in zip(
        outer_rows, oof_ids_parts, oof_anchor_parts, oof_model_parts
    ):
        blended = normalize(anchor + alpha * (model - anchor))
        score = float(np.mean(row_cos(blended, truth_d[ids])))
        delta = score - float(row["baseline"])
        row["blended"] = score
        row["blended_delta"] = delta
        fold_delta.append(delta)
    mean_delta = float(np.mean(fold_delta))
    wins = int(np.sum(np.asarray(fold_delta) > 0.0))
    selected = bool(mean_delta >= MIN_OFFICIAL_DELTA and wins >= 3 and alpha > 0.0)

    final_training: list[dict[str, object]] = []
    projection = None
    local = None
    test_gate = None
    if selected:
        full_anchor = v292.knn_anchor(
            valid_ids, valid_ids, train_pos, train_pos, truth_d, True
        )
        test_anchor = v292.knn_anchor(
            np.arange(len(test_pos)), valid_ids, test_pos, train_pos, truth_d, False
        )
        full_raw = train_data_all.take(valid_ids)
        full_train, [test_data] = standardize(full_raw, [test_data_all])
        test_predictions = []
        test_gates = []
        final_epoch = int(np.median(selected_epochs))
        for seed in FINAL_SEEDS:
            model, training = train_model(
                full_train,
                full_anchor,
                truth_d[valid_ids],
                seed,
                final_epoch,
                device,
                None,
            )
            model_d, _, gate = predict(model, test_data, test_anchor, device)
            test_predictions.append(model_d)
            test_gates.append(gate)
            final_training.append(training)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        test_model = normalize(np.mean(test_predictions, axis=0))
        test_gate = summarize(np.mean(test_gates, axis=0))
        current = v225.current_delay_marginals(BASE)
        target_d = normalize(current["d"] + alpha * (test_model - test_anchor))
        geom = np.load(GEOM_CACHE)
        regimes = regime_of(geom["test_blocked"].astype(np.float32))
        np.savez_compressed(
            TARGETS_OUT,
            anchor=test_anchor,
            model=test_model,
            current=current["d"],
            target=target_d,
            alpha=np.asarray(alpha, dtype=np.float32),
        )
        projection_raw = v225.apply_delay_targets(
            BASE,
            OUTPUT,
            {"d": target_d, "du": current["du"], "dh": current["dh"]},
            regimes,
            1.0,
        )
        projection = {key: summarize(value) for key, value in projection_raw.items()}
        local = v208.score_prediction(OUTPUT)

    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Per-interaction segment length, reflection turn/incidence, ground-vs-wall "
            "reflection and refraction counts let one shared monotone physical law "
            "generalize depth-16 path weights better than free per-point attention."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H PDP",
                "official train/test coordinates and point-cloud geometry",
                "v291 geometry-only paths, vertices and interaction types",
                "shared PyTorch physical law trained only on official train folds",
                "strict v239 carrier only after official selection",
            ],
            "does_not_use": [
                "Paths.cfr or path coefficients",
                "synthetic/Sionna channel labels",
                "v212/v217 teacher output",
                "hidden labels or local truth for training/selection",
            ],
            "local_truth_usage": "read once after official selection" if selected else "not read",
        },
        "parameters": {
            "neighbors": NEIGHBORS,
            "epochs": EPOCHS,
            "outer_seeds": list(OUTER_SEEDS),
            "outer_count": OUTER_COUNT,
            "inner_count": INNER_COUNT,
            "alpha_grid": ALPHA_GRID.tolist(),
            "minimum_official_delta": MIN_OFFICIAL_DELTA,
            "final_seeds": list(FINAL_SEEDS),
        },
        "outer_folds": outer_rows,
        "official_selection": {
            "residual_alpha": alpha,
            "alpha_curve": alpha_curve,
            "mean_pdp_delta": mean_delta,
            "fold_deltas": fold_delta,
            "wins": wins,
            "selected_epochs": selected_epochs,
            "selected": selected,
        },
        "final_training": final_training,
        "test_gate": test_gate,
        "projection": projection,
        "local_gate": local,
        "outputs": {
            "features": PATHS.relative(FEATURE_CACHE),
            "targets": PATHS.relative(TARGETS_OUT) if selected else None,
            "prediction": PATHS.relative(OUTPUT) if selected else None,
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
