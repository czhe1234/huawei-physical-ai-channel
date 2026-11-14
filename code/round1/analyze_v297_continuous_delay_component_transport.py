#!/usr/bin/env python3
"""Recover shared continuous-delay components and transport them in 2D.

The released channel at one UE contains 1024 antenna-link measurements of the
same propagation delays. This experiment extracts a compact continuous-delay
set from those shared measurements, then uses an official-only set network to
transport neighboring component sets to unseen spatial locations.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import nnls

import analyze_v208_strict_historical_replay as v208
import analyze_v225_official_graph_residual_field as v225
import analyze_v254_route_topology_spectral_field as v254
import analyze_v273_planar_barrier_kernel as v273
import analyze_v274_planar_barrier_attention as v274
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 297
VERSION = "v297_continuous_delay_component_transport"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
COMPONENT_CACHE = OUT_DIR / "v297_train_continuous_delay_components.npz"
PROFILE_CACHE = OUT_DIR / "v297_train_oversampled_delay_profiles.npz"
TARGETS_OUT = OUT_DIR / "v297_continuous_delay_targets.npz"
OUTPUT = OUT_DIR / "v297_Round1_Test_Channel.npy"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
BASE = PATHS.cache_dir(284) / "v284_Round1_Test_Channel.npy"

N_DELAY = 192
OVERSAMPLE = 4
N_OVERSAMPLED = N_DELAY * OVERSAMPLE
COMPONENT_COUNT = 12
CANDIDATE_COUNT = 32
MIN_PEAK_DISTANCE = 3
PROFILE_BATCH = 8
K_NEIGHBORS = v274.K_NEIGHBORS
OUTER_SEEDS = (297, 1297, 2297, 3297, 4297)
OUTER_COUNT = 320
INNER_COUNT = 240
EPOCHS = 60
EVAL_EVERY = 5
BATCH_SIZE = 32
OFFSETS = (-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
ALPHA_GRID = np.linspace(0.0, 1.5, 13, dtype=np.float32)
MIN_OFFICIAL_PDP_DELTA = 0.01
FINAL_SEEDS = (29701, 29731, 29761)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_rows(value: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = out.reshape(len(out), -1).sum(axis=1)
    valid = total > 1e-20
    out[valid] /= total[valid].reshape(
        (int(np.sum(valid)),) + (1,) * (out.ndim - 1)
    )
    out[~valid] = 0.0
    return out.astype(np.float32)


def row_cos(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = np.asarray(left, dtype=np.float64).reshape(len(left), -1)
    y = np.asarray(right, dtype=np.float64).reshape(len(right), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def circular_distance(left: np.ndarray, right: np.ndarray, period: float) -> np.ndarray:
    delta = np.abs(np.asarray(left) - np.asarray(right))
    return np.minimum(delta, period - delta)


def dirichlet_atoms(delays: np.ndarray) -> np.ndarray:
    tau = np.asarray(delays, dtype=np.float64)[None, :]
    bins = np.arange(N_DELAY, dtype=np.float64)[:, None]
    delta = (bins - tau + N_DELAY / 2.0) % N_DELAY - N_DELAY / 2.0
    numerator = np.sin(math.pi * delta)
    denominator = N_DELAY * np.sin(math.pi * delta / N_DELAY)
    atom = np.empty_like(delta)
    near_zero = np.abs(delta) < 1e-9
    atom[near_zero] = 1.0
    atom[~near_zero] = (numerator[~near_zero] / denominator[~near_zero]) ** 2
    atom /= np.maximum(atom.sum(axis=0, keepdims=True), 1e-30)
    return atom.astype(np.float32)


def select_peak_delays(profile: np.ndarray, count: int) -> np.ndarray:
    value = np.asarray(profile, dtype=np.float64)
    local = (value >= np.roll(value, 1)) & (value >= np.roll(value, -1))
    candidates = np.flatnonzero(local)
    if len(candidates) == 0:
        candidates = np.arange(len(value))
    candidates = candidates[np.argsort(value[candidates])[::-1]]
    selected: list[int] = []
    for index in candidates:
        distance = [
            min(abs(int(index) - old), len(value) - abs(int(index) - old))
            for old in selected
        ]
        if not distance or min(distance) >= MIN_PEAK_DISTANCE:
            selected.append(int(index))
        if len(selected) == count:
            break
    if len(selected) < count:
        for index in np.argsort(value)[::-1]:
            if int(index) not in selected:
                selected.append(int(index))
            if len(selected) == count:
                break

    refined = np.empty(len(selected), dtype=np.float32)
    log_value = np.log(np.maximum(value, 1e-30))
    for row, index in enumerate(selected):
        left = log_value[(index - 1) % len(value)]
        center = log_value[index]
        right = log_value[(index + 1) % len(value)]
        denominator = left - 2.0 * center + right
        offset = 0.0 if abs(denominator) < 1e-12 else 0.5 * (left - right) / denominator
        refined[row] = ((index + float(np.clip(offset, -0.5, 0.5))) / OVERSAMPLE) % N_DELAY
    return refined


def build_profiles(device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if PROFILE_CACHE.is_file():
        cached = np.load(PROFILE_CACHE)
        return cached["full"], cached["even"], cached["odd"]

    train_h = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    full = np.empty((len(train_h), N_OVERSAMPLED), dtype=np.float32)
    even = np.empty_like(full)
    odd = np.empty_like(full)
    for start in range(0, len(train_h), PROFILE_BATCH):
        end = min(start + PROFILE_BATCH, len(train_h))
        block = np.array(train_h[start:end], dtype=np.complex64, copy=True).reshape(
            end - start, -1, N_DELAY
        )
        tensor = torch.from_numpy(block).to(device)
        cir = torch.fft.ifft(tensor, n=N_OVERSAMPLED, dim=2)
        power = torch.abs(cir).square()
        profiles = (
            power.sum(dim=1),
            power[:, 0::2].sum(dim=1),
            power[:, 1::2].sum(dim=1),
        )
        for output, profile in zip((full, even, odd), profiles):
            value = profile.float()
            value /= value.sum(dim=1, keepdim=True).clamp_min(1e-30)
            output[start:end] = value.cpu().numpy()
        if end % 200 == 0 or end == len(train_h):
            print(f"v297 shared-delay profiles {end}/{len(train_h)}", flush=True)
    np.savez_compressed(PROFILE_CACHE, full=full, even=even, odd=odd)
    return full, even, odd


def build_components(device: torch.device, truth_d: np.ndarray) -> dict[str, np.ndarray]:
    if COMPONENT_CACHE.is_file():
        data = np.load(COMPONENT_CACHE)
        return {key: data[key] for key in data.files}

    full, even, odd = build_profiles(device)
    count = len(full)
    delay = np.zeros((count, COMPONENT_COUNT), dtype=np.float32)
    weight = np.zeros_like(delay)
    valid = np.zeros((count, COMPONENT_COUNT), dtype=bool)
    diffuse = np.zeros(count, dtype=np.float32)
    reconstruction_cosine = np.zeros(count, dtype=np.float32)
    half_profile_cosine = row_cos(even, odd)
    half_component_distance = np.zeros(count, dtype=np.float32)
    uniform = np.full((N_DELAY, 1), 1.0 / N_DELAY, dtype=np.float32)

    for index in range(count):
        if float(np.sum(truth_d[index])) <= 1e-20:
            continue
        candidates = select_peak_delays(full[index], CANDIDATE_COUNT)
        design = np.concatenate([dirichlet_atoms(candidates), uniform], axis=1)
        try:
            coefficient = nnls(design, truth_d[index], maxiter=10_000)[0]
        except RuntimeError:
            coefficient = np.maximum(
                np.linalg.lstsq(design, truth_d[index], rcond=None)[0], 0.0
            )
        order = np.argsort(coefficient[:-1])[::-1][:COMPONENT_COUNT]
        selected_delay = candidates[order]
        selected_weight = coefficient[order]
        retained = selected_weight > max(float(selected_weight[0]) * 1e-4, 1e-8)
        total = float(np.sum(selected_weight[retained]) + coefficient[-1])
        if total <= 1e-20:
            continue
        delay[index, : len(order)] = selected_delay
        weight[index, : len(order)] = selected_weight / total
        valid[index, : len(order)] = retained
        diffuse[index] = float(coefficient[-1] / total)
        reconstruction = (
            dirichlet_atoms(selected_delay) @ weight[index]
            + diffuse[index] / N_DELAY
        )
        reconstruction_cosine[index] = row_cos(
            reconstruction[None, :], truth_d[index : index + 1]
        )[0]

        even_delay = select_peak_delays(even[index], COMPONENT_COUNT)
        odd_delay = select_peak_delays(odd[index], COMPONENT_COUNT)
        distance_even = circular_distance(
            selected_delay[:, None], even_delay[None, :], N_DELAY
        ).min(axis=1)
        distance_odd = circular_distance(
            selected_delay[:, None], odd_delay[None, :], N_DELAY
        ).min(axis=1)
        component_weight = weight[index] * valid[index]
        half_component_distance[index] = float(
            np.sum(component_weight * 0.5 * (distance_even + distance_odd))
            / max(float(np.sum(component_weight)), 1e-20)
        )
        if (index + 1) % 200 == 0 or index + 1 == count:
            print(f"v297 continuous components {index + 1}/{count}", flush=True)

    payload = {
        "delay": delay,
        "weight": weight,
        "valid": valid,
        "diffuse": diffuse,
        "reconstruction_cosine": reconstruction_cosine,
        "half_profile_cosine": half_profile_cosine,
        "half_component_distance": half_component_distance,
    }
    np.savez_compressed(COMPONENT_CACHE, **payload)
    return payload


@dataclass
class ComponentData:
    delay: np.ndarray
    weight: np.ndarray
    valid: np.ndarray
    diffuse: np.ndarray
    reconstruction_cosine: np.ndarray


class ContinuousComponentTransport(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        hidden = 96
        self.query_encoder = nn.Sequential(
            nn.Linear(node_dim, 192), nn.SiLU(), nn.LayerNorm(192), nn.Linear(192, hidden)
        )
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(node_dim, 192), nn.SiLU(), nn.LayerNorm(192), nn.Linear(192, hidden)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 96), nn.SiLU(), nn.Linear(96, 48)
        )
        self.component_encoder = nn.Sequential(
            nn.Linear(13, 96), nn.SiLU(), nn.Linear(96, 48)
        )
        self.token_encoder = nn.Sequential(
            nn.Linear(hidden * 3 + 96, 192),
            nn.SiLU(),
            nn.LayerNorm(192),
            nn.Linear(192, hidden),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=256,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.token_score = nn.Linear(hidden, 1)
        self.shift_score = nn.Linear(hidden, len(OFFSETS))
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.SiLU(), nn.Linear(128, 1)
        )
        nn.init.constant_(self.gate[-1].bias, -1.6)
        self.register_buffer("offsets", torch.tensor(OFFSETS, dtype=torch.float32))
        self.register_buffer(
            "shift_prior",
            torch.tensor([-0.75 * abs(value) for value in OFFSETS], dtype=torch.float32),
        )

    def render_sparse(
        self,
        component_delay: torch.Tensor,
        token_energy: torch.Tensor,
        shift_probability: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        batch = len(component_delay)
        destination = component_delay.reshape(batch, -1, 1) + self.offsets
        lower_float = torch.floor(destination)
        fraction = destination - lower_float
        lower = torch.remainder(lower_float.long(), N_DELAY)
        upper = torch.remainder(lower + 1, N_DELAY)
        mass = token_energy[:, :, None] * shift_probability
        sparse = torch.zeros((batch, N_DELAY), device=anchor.device, dtype=anchor.dtype)
        sparse.scatter_add_(1, lower.reshape(batch, -1), (mass * (1.0 - fraction)).reshape(batch, -1))
        sparse.scatter_add_(1, upper.reshape(batch, -1), (mass * fraction).reshape(batch, -1))
        sparse = 0.10 * torch.roll(sparse, 1, dims=1) + 0.80 * sparse + 0.10 * torch.roll(sparse, -1, dims=1)
        sparse /= sparse.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return sparse

    def forward(
        self,
        query_node: torch.Tensor,
        neighbor_node: torch.Tensor,
        edge: torch.Tensor,
        base_weights: torch.Tensor,
        component_delay: torch.Tensor,
        component_weight: torch.Tensor,
        component_valid: torch.Tensor,
        neighbor_diffuse: torch.Tensor,
        neighbor_reconstruction: torch.Tensor,
        anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, neighbors, components = component_delay.shape
        q = self.query_encoder(query_node)
        n = self.neighbor_encoder(neighbor_node)
        e = self.edge_encoder(edge)
        angle = 2.0 * math.pi * component_delay / N_DELAY
        rank = torch.arange(components, device=component_delay.device).float()
        rank = rank[None, None, :].expand(batch, neighbors, -1)
        component_feature = torch.stack(
            [
                torch.sin(angle),
                torch.cos(angle),
                torch.sin(2.0 * angle),
                torch.cos(2.0 * angle),
                torch.sin(4.0 * angle),
                torch.cos(4.0 * angle),
                torch.log(component_weight.clamp_min(1e-12)) / 12.0,
                component_weight,
                rank / max(components - 1, 1),
                component_valid.float(),
                neighbor_diffuse[:, :, None].expand(-1, -1, components),
                neighbor_reconstruction[:, :, None].expand(-1, -1, components),
                torch.log(base_weights[:, :, None].expand(-1, -1, components).clamp_min(1e-12)) / 8.0,
            ],
            dim=3,
        )
        c = self.component_encoder(component_feature)
        q_token = q[:, None, None, :].expand(-1, neighbors, components, -1)
        n_token = n[:, :, None, :].expand(-1, -1, components, -1)
        e_token = e[:, :, None, :].expand(-1, -1, components, -1)
        token = self.token_encoder(
            torch.cat([q_token, n_token, q_token * n_token, e_token, c], dim=3)
        ).reshape(batch, neighbors * components, -1)
        mask = ~component_valid.reshape(batch, -1)
        token = self.transformer(token, src_key_padding_mask=mask)

        prior = torch.log(
            (
                base_weights[:, :, None] * component_weight.clamp_min(1e-12)
            ).reshape(batch, -1)
        )
        token_logits = prior + 2.0 * torch.tanh(self.token_score(token).squeeze(2))
        token_logits = token_logits.masked_fill(mask, -1e9)
        token_energy = torch.softmax(token_logits, dim=1)
        shift_probability = torch.softmax(
            self.shift_prior[None, None, :]
            + 1.75 * torch.tanh(self.shift_score(token)),
            dim=2,
        )

        sparse = self.render_sparse(
            component_delay, token_energy, shift_probability, anchor
        )
        pooled = torch.sum(token_energy[:, :, None] * token, dim=1)
        gate = 0.65 * torch.sigmoid(self.gate(torch.cat([q, pooled], dim=1)))
        prediction = (1.0 - gate) * anchor + gate * sparse
        prediction /= prediction.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return prediction, {
            "gate": gate.squeeze(1),
            "token_energy": token_energy,
            "shift_probability": shift_probability,
        }


MODEL_CLASS = ContinuousComponentTransport


def component_data(raw: dict[str, np.ndarray]) -> ComponentData:
    return ComponentData(
        delay=raw["delay"].astype(np.float32),
        weight=raw["weight"].astype(np.float32),
        valid=raw["valid"].astype(bool),
        diffuse=raw["diffuse"].astype(np.float32),
        reconstruction_cosine=raw["reconstruction_cosine"].astype(np.float32),
    )


def model_arrays(
    data: v274.ModelSet,
    node_standardized: np.ndarray,
    components: ComponentData,
    truth_d: np.ndarray,
) -> dict[str, np.ndarray]:
    neighbor = data.neighbor_ids
    anchor = normalize_rows(
        np.sum(
            data.base_weights[:, :, None] * truth_d[neighbor],
            axis=1,
            dtype=np.float64,
        )
    )
    return {
        "query_ids": data.query_ids,
        "query_node": data.query_node,
        "neighbor_node": node_standardized[neighbor].astype(np.float32),
        "edge": data.edge.astype(np.float32),
        "base_weights": data.base_weights.astype(np.float32),
        "component_delay": components.delay[neighbor],
        "component_weight": components.weight[neighbor],
        "component_valid": components.valid[neighbor],
        "neighbor_diffuse": components.diffuse[neighbor],
        "neighbor_reconstruction": components.reconstruction_cosine[neighbor],
        "anchor": anchor,
    }


def tensor_batch(
    arrays: dict[str, np.ndarray], rows: np.ndarray, device: torch.device
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key, value in arrays.items():
        if key == "query_ids":
            continue
        tensor = torch.from_numpy(np.asarray(value[rows]))
        output[key] = tensor.to(device)
    return output


@torch.no_grad()
def predict(
    model: ContinuousComponentTransport,
    arrays: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    output = np.empty((len(arrays["query_ids"]), N_DELAY), dtype=np.float32)
    gate = np.empty(len(output), dtype=np.float32)
    for start in range(0, len(output), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(output))
        batch = tensor_batch(arrays, np.arange(start, end), device)
        prediction, diagnostics = model(**batch)
        output[start:end] = prediction.cpu().numpy()
        gate[start:end] = diagnostics["gate"].cpu().numpy()
    return output, gate


def loss_function(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    cosine = F.cosine_similarity(prediction, truth, dim=1)
    root_cosine = F.cosine_similarity(
        torch.sqrt(prediction.clamp_min(1e-12)),
        torch.sqrt(truth.clamp_min(1e-12)),
        dim=1,
    )
    return (1.0 - cosine).mean() + 0.12 * (1.0 - root_cosine).mean()


def fit_model(
    train_arrays: dict[str, np.ndarray],
    truth_d: np.ndarray,
    node_dim: int,
    edge_dim: int,
    seed: int,
    epochs: int,
    device: torch.device,
    validation_arrays: dict[str, np.ndarray] | None = None,
) -> tuple[ContinuousComponentTransport, dict[str, object]]:
    set_seed(seed)
    model = MODEL_CLASS(node_dim, edge_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.2e-4, weight_decay=1e-4)
    target = truth_d[train_arrays["query_ids"]].astype(np.float32)
    baseline = None
    best_score = -np.inf
    best_epoch = epochs
    best_state = None
    history: list[dict[str, float | int]] = []
    if validation_arrays is not None:
        validation_truth = truth_d[validation_arrays["query_ids"]]
        baseline = float(np.mean(row_cos(validation_arrays["anchor"], validation_truth)))
        best_score = baseline
        best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(target))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start : start + BATCH_SIZE]
            batch = tensor_batch(train_arrays, rows, device)
            truth = torch.from_numpy(target[rows]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, _diagnostics = model(**batch)
            loss = loss_function(prediction, truth)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite v297 loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if validation_arrays is not None and (epoch % EVAL_EVERY == 0 or epoch == epochs):
            validation_prediction, validation_gate = predict(model, validation_arrays, device)
            score = float(np.mean(row_cos(validation_prediction, validation_truth)))
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(np.mean(losses)),
                    "validation_pdp": score,
                    "validation_gate_mean": float(np.mean(validation_gate)),
                }
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        if epoch % 20 == 0 or epoch == epochs:
            print(
                f"v297 seed={seed} epoch={epoch}/{epochs} loss={np.mean(losses):.6f}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "baseline_pdp": baseline,
        "best_validation_pdp": None if not np.isfinite(best_score) else float(best_score),
        "history": history,
    }


def standardized_sets(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    all_pos: np.ndarray,
    signatures: np.ndarray,
    blocker_count: np.ndarray,
    footprint: np.ndarray,
    z_max: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
    node_raw: np.ndarray,
    stats: v274.Standardizer | None = None,
) -> tuple[v274.ModelSet, v274.Standardizer, np.ndarray]:
    raw = v274.build_raw_set(
        query_ids,
        bank_ids,
        all_pos,
        signatures,
        blocker_count,
        footprint,
        z_max,
        x_min,
        y_min,
        resolution,
    )
    if stats is None:
        stats = v274.fit_standardizer(node_raw, bank_ids, raw)
    model_set = v274.apply_standardizer(raw, node_raw, stats)
    node_standardized = np.clip(
        (node_raw - stats.node_mean) / stats.node_std, -8.0, 8.0
    ).astype(np.float32)
    return model_set, stats, node_standardized


def choose_alpha(
    base: np.ndarray, model: np.ndarray, truth: np.ndarray
) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for alpha in ALPHA_GRID:
        blended = normalize_rows(base + float(alpha) * (model - base))
        rows.append(
            {
                "alpha": float(alpha),
                "pdp": float(np.mean(row_cos(blended, truth))),
            }
        )
    best = max(rows, key=lambda row: row["pdp"])
    return float(best["alpha"]), rows


def main() -> None:
    for path in (TARGET_CACHE, v274.SCENE_CACHE, v274.FOOTPRINT_CACHE, v274.SIGNATURE_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = np.load(TARGET_CACHE)
    truth_d = targets["d"].astype(np.float32)
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    component_raw = build_components(device, truth_d)
    components = component_data(component_raw)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    (
        all_pos,
        signatures,
        blocker_count,
        scene_features,
        footprint,
        z_max,
        x_min,
        y_min,
        resolution,
    ) = v274.load_geometry()
    node_raw = v274.build_node_features(
        all_pos, signatures, blocker_count, scene_features
    )

    outer_rows: list[dict[str, object]] = []
    outer_base: list[np.ndarray] = []
    outer_model: list[np.ndarray] = []
    outer_truth: list[np.ndarray] = []
    selected_epochs: list[int] = []
    for fold, seed in enumerate(OUTER_SEEDS):
        outer_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], OUTER_COUNT, seed
            ),
            valid_ids,
        )
        outer_fit = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = v273.matched_subset(
            outer_fit,
            train_pos[:, :2],
            test_pos[:, :2],
            INNER_COUNT,
            seed + 10_000,
        )
        inner_fit = np.setdiff1d(outer_fit, inner_ids)
        inner_train_set, inner_stats, inner_node = standardized_sets(
            inner_fit, inner_fit, all_pos, signatures, blocker_count, footprint,
            z_max, x_min, y_min, resolution, node_raw,
        )
        inner_val_set, _stats, _node = standardized_sets(
            inner_ids, inner_fit, all_pos, signatures, blocker_count, footprint,
            z_max, x_min, y_min, resolution, node_raw, inner_stats,
        )
        inner_train = model_arrays(inner_train_set, inner_node, components, truth_d)
        inner_val = model_arrays(inner_val_set, inner_node, components, truth_d)
        _model, inner_report = fit_model(
            inner_train,
            truth_d,
            inner_train["query_node"].shape[1],
            inner_train["edge"].shape[2],
            seed,
            EPOCHS,
            device,
            inner_val,
        )
        selected_epoch = int(inner_report["best_epoch"])
        selected_epochs.append(selected_epoch)

        outer_train_set, outer_stats, outer_node = standardized_sets(
            outer_fit, outer_fit, all_pos, signatures, blocker_count, footprint,
            z_max, x_min, y_min, resolution, node_raw,
        )
        outer_val_set, _stats, _node = standardized_sets(
            outer_ids, outer_fit, all_pos, signatures, blocker_count, footprint,
            z_max, x_min, y_min, resolution, node_raw, outer_stats,
        )
        outer_train_arrays = model_arrays(
            outer_train_set, outer_node, components, truth_d
        )
        outer_val_arrays = model_arrays(
            outer_val_set, outer_node, components, truth_d
        )
        if selected_epoch == 0:
            model_prediction = outer_val_arrays["anchor"].copy()
            gate = np.zeros(len(model_prediction), dtype=np.float32)
            outer_training = {"seed": seed, "best_epoch": 0}
        else:
            model, outer_training = fit_model(
                outer_train_arrays,
                truth_d,
                outer_train_arrays["query_node"].shape[1],
                outer_train_arrays["edge"].shape[2],
                seed + 50_000,
                selected_epoch,
                device,
            )
            model_prediction, gate = predict(model, outer_val_arrays, device)
        base = outer_val_arrays["anchor"]
        truth = truth_d[outer_ids]
        base_score = float(np.mean(row_cos(base, truth)))
        model_score = float(np.mean(row_cos(model_prediction, truth)))
        outer_base.append(base)
        outer_model.append(model_prediction)
        outer_truth.append(truth)
        outer_rows.append(
            {
                "fold": fold,
                "seed": seed,
                "outer_count": int(len(outer_ids)),
                "selected_epoch": selected_epoch,
                "base_pdp": base_score,
                "model_pdp": model_score,
                "raw_delta": model_score - base_score,
                "gate": summarize(gate),
                "inner": inner_report,
                "outer_training": outer_training,
            }
        )
        print(
            f"v297 outer {fold + 1}/5 epoch={selected_epoch} "
            f"base={base_score:.6f} model={model_score:.6f} "
            f"delta={model_score - base_score:+.6f}",
            flush=True,
        )

    base_all = np.concatenate(outer_base)
    model_all = np.concatenate(outer_model)
    truth_all = np.concatenate(outer_truth)
    alpha, alpha_rows = choose_alpha(base_all, model_all, truth_all)
    blended_all = normalize_rows(base_all + alpha * (model_all - base_all))
    base_score = float(np.mean(row_cos(base_all, truth_all)))
    blended_score = float(np.mean(row_cos(blended_all, truth_all)))
    fold_blended = []
    start = 0
    for row, base, model, truth in zip(outer_rows, outer_base, outer_model, outer_truth):
        end = start + len(base)
        blended = blended_all[start:end]
        score = float(np.mean(row_cos(blended, truth)))
        delta = score - float(row["base_pdp"])
        row["blended_pdp"] = score
        row["blended_delta"] = delta
        fold_blended.append(delta)
        start = end
    official_delta = blended_score - base_score
    official_selection = bool(
        official_delta >= MIN_OFFICIAL_PDP_DELTA
        and sum(value > 0.0 for value in fold_blended) >= 4
    )

    deployment: dict[str, object] | None = None
    if official_selection:
        if not BASE.is_file():
            raise FileNotFoundError(BASE)
        final_predictions = []
        final_epochs = max(1, int(np.median([value for value in selected_epochs if value > 0])))
        test_ids = np.arange(len(train_pos), len(all_pos), dtype=np.int64)
        final_train_set, final_stats, final_node = standardized_sets(
            valid_ids, valid_ids, all_pos, signatures, blocker_count, footprint,
            z_max, x_min, y_min, resolution, node_raw,
        )
        final_test_set, _stats, _node = standardized_sets(
            test_ids, valid_ids, all_pos, signatures, blocker_count, footprint,
            z_max, x_min, y_min, resolution, node_raw, final_stats,
        )
        final_train = model_arrays(final_train_set, final_node, components, truth_d)
        final_test = model_arrays(final_test_set, final_node, components, truth_d)
        for seed in FINAL_SEEDS:
            model, train_report = fit_model(
                final_train,
                truth_d,
                final_train["query_node"].shape[1],
                final_train["edge"].shape[2],
                seed,
                final_epochs,
                device,
            )
            checkpoint = CHECKPOINT_DIR / f"seed_{seed}.pt"
            torch.save(model.state_dict(), checkpoint)
            prediction, gate = predict(model, final_test, device)
            final_predictions.append(prediction)
            train_report["checkpoint"] = PATHS.relative(checkpoint)
        model_test = normalize_rows(np.mean(final_predictions, axis=0))
        target_d = normalize_rows(
            final_test["anchor"] + alpha * (model_test - final_test["anchor"])
        )
        np.savez_compressed(
            TARGETS_OUT,
            d=target_d,
            base_d=final_test["anchor"],
            model_d=model_test,
            alpha=np.float32(alpha),
        )
        current = v225.current_delay_marginals(BASE)
        regimes = regime_of(test_pos[:, 2])
        diagnostics = v225.apply_delay_targets(
            BASE,
            OUTPUT,
            {"d": target_d, "du": current["du"], "dh": current["dh"]},
            regimes,
            0.85,
        )
        deployment = {
            "final_epochs": final_epochs,
            "targets": PATHS.relative(TARGETS_OUT),
            "prediction": PATHS.relative(OUTPUT),
            "local_score": v208.score_prediction(OUTPUT),
            "diagnostics": {
                key: summarize(value) for key, value in diagnostics.items()
            },
        }

    valid_component = targets["valid"].astype(bool)
    component_report = {
        "raw_half_profile_cosine": summarize(component_raw["half_profile_cosine"][valid_component]),
        "continuous_half_distance_bins": summarize(component_raw["half_component_distance"][valid_component]),
        "path_reconstruction_cosine": summarize(component_raw["reconstruction_cosine"][valid_component]),
        "active_components": summarize(component_raw["valid"][valid_component].sum(axis=1)),
        "diffuse_mass": summarize(component_raw["diffuse"][valid_component]),
    }
    report = {
        "method": VERSION,
        "status": "official_selection_passed" if official_selection else "analysis_only_no_candidate",
        "primary_hypothesis": (
            "The 1024 antenna links at one UE share continuous propagation delays; "
            "recovering and transporting that latent set should improve unseen-point "
            "PDP beyond interpolation of noisy gridded power."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H",
                "official train/test planar coordinates",
                "official point-cloud barrier and scene features",
                "PyTorch set Transformer trained only on fit-fold official labels",
            ],
            "does_not_use": [
                "synthetic or Sionna channel labels",
                "v212 or v217 teacher outputs",
                "hidden online labels",
                "RT channel coefficients or local replay truth for selection",
            ],
        },
        "component_extraction": component_report,
        "renderer": MODEL_CLASS.__name__,
        "outer_rows": outer_rows,
        "oof": {
            "base_pdp": base_score,
            "model_pdp": float(np.mean(row_cos(model_all, truth_all))),
            "residual_alpha": alpha,
            "blended_pdp": blended_score,
            "blended_delta": official_delta,
            "fold_deltas": fold_blended,
            "fold_wins": int(sum(value > 0.0 for value in fold_blended)),
            "minimum_required_delta": MIN_OFFICIAL_PDP_DELTA,
            "alpha_grid": alpha_rows,
            "official_selection": official_selection,
        },
        "deployment": deployment,
        "outputs": {
            "components": PATHS.relative(COMPONENT_CACHE),
            "profiles": PATHS.relative(PROFILE_CACHE),
            "targets": PATHS.relative(TARGETS_OUT) if TARGETS_OUT.is_file() else None,
            "prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None,
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["component_extraction"], indent=2), flush=True)
    print(json.dumps(report["oof"], indent=2), flush=True)


if __name__ == "__main__":
    main()
