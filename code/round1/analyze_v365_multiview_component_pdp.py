#!/usr/bin/env python3
"""Strict official-only PDP transport with per-path multi-view reliability.

This experiment deliberately predicts only the normalized global delay profile.
It tests whether a donor component that repeats across independent antenna/link
views is easier to transport to an unseen planar coordinate than an equally
strong but view-inconsistent component.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

import analyze_v274_planar_barrier_attention as v274
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v238_raw_path_conditional_angle as v238
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 365
VERSION = "v365_multiview_component_pdp"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
RELIABILITY_CACHE = OUT_DIR / "v365_train_component_multiview_reliability.npz"
OOF_OUT = OUT_DIR / "v365_multiview_component_pdp_oof.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"

N_DELAY = 192
OVERSAMPLE = 4
N_OVERSAMPLED = N_DELAY * OVERSAMPLE
COMPONENT_COUNT = 12
NEIGHBORS = 32
OFFSETS = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
PROFILE_RADIUS = 4
PROFILE_BATCH = 4
TRAIN_BATCH = 16
MAX_EPOCHS = 24
EVAL_EVERY = 2
LEARNING_RATE = 3.0e-4

MIN_COSINE_LIFT = 0.010
MIN_TOP_RECALL_LIFT = 0.05
MIN_EMD_REDUCTION = 0.10
MIN_FOLD_WINS = 4
MIN_RELIABILITY_CONTROL_LIFT = 0.002
MAX_CLEAR_DEGRADATION = 0.002

VIEW_PAIRS = (
    "horizontal_half",
    "vertical_half",
    "ue_port_half",
    "polarization",
    "array_checkerboard",
    "link_checkerboard",
)


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


def circular_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    delta = np.abs(np.asarray(left) - np.asarray(right))
    return np.minimum(delta, N_DELAY - delta)


def link_pair_masks(device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    p, h, v, u = torch.meshgrid(
        torch.arange(2, device=device),
        torch.arange(16, device=device),
        torch.arange(8, device=device),
        torch.arange(4, device=device),
        indexing="ij",
    )
    groups = (
        h < 8,
        v < 4,
        u < 2,
        p == 0,
        torch.remainder(h + v, 2) == 0,
        torch.remainder(p + h + v + u, 2) == 0,
    )
    return [(group.reshape(-1), (~group).reshape(-1)) for group in groups]


def build_multiview_reliability(device: torch.device) -> dict[str, np.ndarray]:
    if RELIABILITY_CACHE.is_file():
        cached = np.load(RELIABILITY_CACHE)
        return {key: cached[key] for key in cached.files}
    if not v297.COMPONENT_CACHE.is_file():
        raise FileNotFoundError(v297.COMPONENT_CACHE)

    component = np.load(v297.COMPONENT_CACHE)
    delay = component["delay"].astype(np.float32)
    valid = component["valid"].astype(bool)
    channel = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    feature = np.zeros((len(channel), COMPONENT_COUNT, len(VIEW_PAIRS) * 3), dtype=np.float32)
    aggregate = np.zeros((len(channel), COMPONENT_COUNT), dtype=np.float32)
    masks = link_pair_masks(device)
    window_offset = torch.arange(
        -PROFILE_RADIUS, PROFILE_RADIUS + 1, device=device, dtype=torch.long
    )

    for start in range(0, len(channel), PROFILE_BATCH):
        end = min(start + PROFILE_BATCH, len(channel))
        block = np.array(channel[start:end], dtype=np.complex64, copy=True).reshape(
            end - start, -1, N_DELAY
        )
        cir = torch.fft.ifft(torch.from_numpy(block).to(device), n=N_OVERSAMPLED, dim=2)
        power = torch.abs(cir).square()
        centers = torch.round(
            torch.from_numpy(delay[start:end]).to(device) * OVERSAMPLE
        ).long()
        gather_index = torch.remainder(
            centers[:, :, None] + window_offset[None, None, :], N_OVERSAMPLED
        )
        pair_quality: list[torch.Tensor] = []
        for pair_index, (first_mask, second_mask) in enumerate(masks):
            profiles = []
            for mask in (first_mask, second_mask):
                profile = power[:, mask].sum(dim=1)
                profile /= profile.sum(dim=1, keepdim=True).clamp_min(1e-30)
                profiles.append(torch.gather(profile, 1, gather_index.reshape(end - start, -1)).reshape(
                    end - start, COMPONENT_COUNT, -1
                ))
            first, second = profiles
            first_mass = first.sum(dim=2)
            second_mass = second.sum(dim=2)
            agreement = 2.0 * torch.sqrt(first_mass * second_mass).clamp_min(0.0)
            agreement /= (first_mass + second_mass).clamp_min(1e-30)
            first_peak = torch.argmax(first, dim=2).float()
            second_peak = torch.argmax(second, dim=2).float()
            peak_agreement = torch.exp(-torch.abs(first_peak - second_peak) / OVERSAMPLE)
            window_cosine = F.cosine_similarity(first, second, dim=2, eps=1e-12).clamp(0.0, 1.0)
            feature[start:end, :, pair_index] = agreement.cpu().numpy()
            feature[start:end, :, len(VIEW_PAIRS) + pair_index] = peak_agreement.cpu().numpy()
            feature[start:end, :, 2 * len(VIEW_PAIRS) + pair_index] = window_cosine.cpu().numpy()
            if pair_index in (0, 1, 4, 5):
                pair_quality.append(
                    agreement * peak_agreement * (0.5 + 0.5 * window_cosine)
                )
        quality = torch.stack(pair_quality, dim=2).clamp_min(1e-8)
        aggregate[start:end] = torch.exp(torch.mean(torch.log(quality), dim=2)).cpu().numpy()
        feature[start:end] *= valid[start:end, :, None]
        aggregate[start:end] *= valid[start:end]
        del cir, power
        if end % 200 == 0 or end == len(channel):
            print(f"v365 multi-view profiles {end}/{len(channel)}", flush=True)

    np.savez_compressed(
        RELIABILITY_CACHE,
        feature=feature,
        aggregate=aggregate,
        pair_names=np.asarray(VIEW_PAIRS),
    )
    return {
        "feature": feature,
        "aggregate": aggregate,
        "pair_names": np.asarray(VIEW_PAIRS),
    }


@dataclass
class FoldData:
    query_ids: np.ndarray
    query_node: np.ndarray
    neighbor_ids: np.ndarray
    neighbor_node: np.ndarray
    edge: np.ndarray
    base_weights: np.ndarray
    component_delay: np.ndarray
    component_weight: np.ndarray
    component_valid: np.ndarray
    neighbor_diffuse: np.ndarray
    neighbor_reconstruction: np.ndarray
    reliability_feature: np.ndarray
    reliability_aggregate: np.ndarray
    anchor: np.ndarray


def trim_set(data: v274.ModelSet) -> v274.ModelSet:
    weight = data.base_weights[:, :NEIGHBORS]
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-20)
    return v274.ModelSet(
        query_ids=data.query_ids,
        query_node=data.query_node,
        neighbor_ids=data.neighbor_ids[:, :NEIGHBORS],
        edge=data.edge[:, :NEIGHBORS],
        base_weights=weight.astype(np.float32),
    )


def make_fold_data(
    data: v274.ModelSet,
    node_standardized: np.ndarray,
    component: dict[str, np.ndarray],
    truth_d: np.ndarray,
    reliability_feature: np.ndarray,
    reliability_aggregate: np.ndarray,
) -> FoldData:
    data = trim_set(data)
    neighbor = data.neighbor_ids
    anchor = normalize_rows(
        np.sum(
            data.base_weights[:, :, None] * truth_d[neighbor],
            axis=1,
            dtype=np.float64,
        )
    )
    return FoldData(
        query_ids=data.query_ids,
        query_node=data.query_node.astype(np.float32),
        neighbor_ids=neighbor,
        neighbor_node=node_standardized[neighbor].astype(np.float32),
        edge=data.edge.astype(np.float32),
        base_weights=data.base_weights.astype(np.float32),
        component_delay=component["delay"][neighbor].astype(np.float32),
        component_weight=component["weight"][neighbor].astype(np.float32),
        component_valid=component["valid"][neighbor].astype(bool),
        neighbor_diffuse=component["diffuse"][neighbor].astype(np.float32),
        neighbor_reconstruction=component["reconstruction_cosine"][neighbor].astype(np.float32),
        reliability_feature=reliability_feature[neighbor].astype(np.float32),
        reliability_aggregate=reliability_aggregate[neighbor].astype(np.float32),
        anchor=anchor,
    )


def shuffled_reliability(
    feature: np.ndarray,
    aggregate: np.ndarray,
    valid: np.ndarray,
    bank_ids: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    out_feature = np.zeros_like(feature)
    out_aggregate = np.zeros_like(aggregate)
    mask = valid[bank_ids]
    source_feature = feature[bank_ids][mask]
    source_aggregate = aggregate[bank_ids][mask]
    order = np.random.default_rng(seed).permutation(len(source_aggregate))
    bank_feature = out_feature[bank_ids]
    bank_aggregate = out_aggregate[bank_ids]
    bank_feature[mask] = source_feature[order]
    bank_aggregate[mask] = source_aggregate[order]
    out_feature[bank_ids] = bank_feature
    out_aggregate[bank_ids] = bank_aggregate
    return out_feature, out_aggregate


class MultiViewComponentTransport(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, reliability_dim: int):
        super().__init__()
        hidden = 128
        self.query_encoder = nn.Sequential(
            nn.Linear(node_dim, 192), nn.SiLU(), nn.LayerNorm(192), nn.Linear(192, hidden)
        )
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(node_dim, 192), nn.SiLU(), nn.LayerNorm(192), nn.Linear(192, hidden)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 128), nn.SiLU(), nn.LayerNorm(128), nn.Linear(128, 64)
        )
        self.component_encoder = nn.Sequential(
            nn.Linear(13 + reliability_dim + 1, 160),
            nn.SiLU(),
            nn.LayerNorm(160),
            nn.Linear(160, 64),
        )
        self.token_encoder = nn.Sequential(
            nn.Linear(hidden * 3 + 128, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, hidden),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=8,
            dim_feedforward=384,
            dropout=0.04,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.token_score = nn.Linear(hidden, 1)
        self.shift_score = nn.Linear(hidden, len(OFFSETS))
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.SiLU(), nn.Linear(128, 1)
        )
        nn.init.zeros_(self.token_score.weight)
        nn.init.zeros_(self.token_score.bias)
        nn.init.zeros_(self.shift_score.weight)
        nn.init.zeros_(self.shift_score.bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.2)
        self.register_buffer("offsets", torch.tensor(OFFSETS, dtype=torch.float32))
        self.register_buffer(
            "shift_prior",
            torch.tensor([-1.35 * abs(value) for value in OFFSETS], dtype=torch.float32),
        )

    def render(
        self,
        delay: torch.Tensor,
        token_energy: torch.Tensor,
        shift_probability: torch.Tensor,
    ) -> torch.Tensor:
        batch = len(delay)
        destination = delay.reshape(batch, -1, 1) + self.offsets
        bins = torch.arange(N_DELAY, device=delay.device, dtype=delay.dtype)
        delta = bins[None, None, None, :] - destination[:, :, :, None]
        delta = torch.remainder(delta + N_DELAY / 2.0, N_DELAY) - N_DELAY / 2.0
        atom = torch.square(torch.sinc(delta) / torch.sinc(delta / N_DELAY))
        atom /= atom.sum(dim=3, keepdim=True).clamp_min(1e-30)
        mass = token_energy[:, :, None] * shift_probability
        output = torch.einsum("bto,bton->bn", mass, atom)
        return output / output.sum(dim=1, keepdim=True).clamp_min(1e-30)

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
        reliability_feature: torch.Tensor,
        reliability_aggregate: torch.Tensor,
        anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, neighbors, components = component_delay.shape
        q = self.query_encoder(query_node)
        n = self.neighbor_encoder(neighbor_node)
        e = self.edge_encoder(edge)
        angle = 2.0 * math.pi * component_delay / N_DELAY
        rank = torch.arange(components, device=component_delay.device).float()
        rank = rank[None, None, :].expand(batch, neighbors, -1)
        scalar = torch.stack(
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
        component_feature = torch.cat(
            [scalar, reliability_feature, reliability_aggregate[:, :, :, None]], dim=3
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
            (base_weights[:, :, None] * component_weight.clamp_min(1e-12)).reshape(batch, -1)
        )
        logits = prior + 2.25 * torch.tanh(self.token_score(token).squeeze(2))
        logits = logits.masked_fill(mask, -1e9)
        token_energy = torch.softmax(logits, dim=1)
        shift_probability = torch.softmax(
            self.shift_prior[None, None, :] + 2.0 * torch.tanh(self.shift_score(token)), dim=2
        )
        proposal = self.render(component_delay, token_energy, shift_probability)
        pooled = torch.sum(token_energy[:, :, None] * token, dim=1)
        gate = 0.75 * torch.sigmoid(self.gate(torch.cat([q, pooled], dim=1)))
        prediction = (1.0 - gate) * anchor + gate * proposal
        prediction /= prediction.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return prediction, {
            "gate": gate[:, 0],
            "token_energy": token_energy,
            "shift_probability": shift_probability,
        }


def tensor_batch(data: FoldData, rows: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key in (
        "query_node",
        "neighbor_node",
        "edge",
        "base_weights",
        "component_delay",
        "component_weight",
        "component_valid",
        "neighbor_diffuse",
        "neighbor_reconstruction",
        "reliability_feature",
        "reliability_aggregate",
        "anchor",
    ):
        output[key] = torch.from_numpy(np.asarray(getattr(data, key)[rows])).to(device)
    return output


def model_loss(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    cosine = 1.0 - F.cosine_similarity(prediction, truth, dim=1).mean()
    root = 1.0 - F.cosine_similarity(
        torch.sqrt(prediction.clamp_min(1e-12)),
        torch.sqrt(truth.clamp_min(1e-12)),
        dim=1,
    ).mean()
    emd = torch.mean(torch.abs(torch.cumsum(prediction - truth, dim=1)))
    return cosine + 0.08 * root + 0.12 * emd


@torch.no_grad()
def predict(
    model: MultiViewComponentTransport,
    data: FoldData,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    output = np.empty((len(data.query_ids), N_DELAY), dtype=np.float32)
    gate = np.empty(len(data.query_ids), dtype=np.float32)
    token_energy = np.empty((len(data.query_ids), NEIGHBORS * COMPONENT_COUNT), dtype=np.float32)
    for start in range(0, len(output), TRAIN_BATCH):
        end = min(start + TRAIN_BATCH, len(output))
        batch = tensor_batch(data, np.arange(start, end), device)
        block, diagnostics = model(**batch)
        output[start:end] = block.cpu().numpy()
        gate[start:end] = diagnostics["gate"].cpu().numpy()
        token_energy[start:end] = diagnostics["token_energy"].cpu().numpy()
    return output, gate, token_energy


def fit_model(
    train: FoldData,
    validation: FoldData | None,
    truth_d: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
) -> tuple[MultiViewComponentTransport, dict[str, object]]:
    set_seed(seed)
    model = MultiViewComponentTransport(
        train.query_node.shape[1], train.edge.shape[2], train.reliability_feature.shape[3]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=2.0e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    rng = np.random.default_rng(seed)
    best_epoch = epochs
    best_score = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    if validation is not None:
        best_epoch = 0
        best_score = float(np.mean(row_cos(validation.anchor, truth_d[validation.query_ids])))
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        order = rng.permutation(len(train.query_ids))
        for start in range(0, len(order), TRAIN_BATCH):
            rows = order[start : start + TRAIN_BATCH]
            batch = tensor_batch(train, rows, device)
            target = torch.from_numpy(truth_d[train.query_ids[rows]]).to(device)
            prediction, _ = model(**batch)
            loss = model_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if validation is not None and (epoch % EVAL_EVERY == 0 or epoch == epochs):
            validation_prediction, validation_gate, _ = predict(model, validation, device)
            score = float(np.mean(row_cos(validation_prediction, truth_d[validation.query_ids])))
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(np.mean(losses)),
                    "validation_pdp": score,
                    "validation_gate": float(np.mean(validation_gate)),
                }
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
        if epoch % 8 == 0 or epoch == epochs:
            print(
                f"v365 seed={seed} epoch={epoch}/{epochs} loss={np.mean(losses):.6f}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_validation_pdp": (
            float(best_score) if validation is not None and np.isfinite(best_score) else None
        ),
        "history": history,
    }


def build_sets(
    train_ids: np.ndarray,
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    geometry: tuple,
    node_raw: np.ndarray,
) -> tuple[v274.ModelSet, v274.ModelSet, np.ndarray]:
    (
        all_pos,
        signatures,
        blocker_count,
        _scene_features,
        footprint,
        z_max,
        x_min,
        y_min,
        resolution,
    ) = geometry
    raw_train = v274.build_raw_set(
        train_ids, bank_ids, all_pos, signatures, blocker_count, footprint,
        z_max, x_min, y_min, resolution,
    )
    stats = v274.fit_standardizer(node_raw, bank_ids, raw_train)
    raw_query = v274.build_raw_set(
        query_ids, bank_ids, all_pos, signatures, blocker_count, footprint,
        z_max, x_min, y_min, resolution,
    )
    train_set = v274.apply_standardizer(raw_train, node_raw, stats)
    query_set = v274.apply_standardizer(raw_query, node_raw, stats)
    standardized_node = np.clip(
        (node_raw - stats.node_mean) / stats.node_std, -8.0, 8.0
    ).astype(np.float32)
    return train_set, query_set, standardized_node


def top_path_recall(
    prediction: np.ndarray,
    truth: np.ndarray,
    query_ids: np.ndarray,
    component: dict[str, np.ndarray],
) -> np.ndarray:
    output = np.zeros(len(query_ids), dtype=np.float32)
    bins = np.arange(N_DELAY, dtype=np.float32)
    for row, point_id in enumerate(query_ids):
        active = np.flatnonzero(component["valid"][point_id])
        if len(active) == 0:
            continue
        order = active[np.argsort(component["weight"][point_id, active])[::-1]]
        selected: list[int] = []
        accumulated = 0.0
        for component_id in order:
            selected.append(int(component_id))
            accumulated += float(component["weight"][point_id, component_id])
            if accumulated >= 0.80 or len(selected) >= 6:
                break
        support = np.zeros(N_DELAY, dtype=bool)
        for component_id in selected:
            distance = circular_distance(bins, component["delay"][point_id, component_id])
            support |= distance <= 1.5
        truth_mass = float(np.sum(truth[row, support]))
        predicted_mass = float(np.sum(prediction[row, support]))
        output[row] = min(predicted_mass, truth_mass) / max(truth_mass, 1e-20)
    return output


def delay_emd(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return (
        np.sum(np.abs(np.cumsum(prediction - truth, axis=1)), axis=1) / N_DELAY
    ).astype(np.float32)


def token_ap(
    token_energy: np.ndarray,
    data: FoldData,
    component: dict[str, np.ndarray],
) -> float:
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    candidate = data.component_delay.reshape(len(data.query_ids), -1)
    valid_candidate = data.component_valid.reshape(len(data.query_ids), -1)
    for row, point_id in enumerate(data.query_ids):
        active = np.flatnonzero(component["valid"][point_id])
        if len(active) == 0:
            continue
        truth_delay = component["delay"][point_id, active]
        distance = circular_distance(candidate[row, :, None], truth_delay[None, :]).min(axis=1)
        keep = valid_candidate[row]
        labels.append((distance[keep] <= 0.25).astype(np.int8))
        scores.append(token_energy[row, keep])
    if not labels:
        return float("nan")
    y_true = np.concatenate(labels)
    y_score = np.concatenate(scores)
    return float(average_precision_score(y_true, y_score))


def evaluate(
    prediction: np.ndarray,
    truth: np.ndarray,
    query_ids: np.ndarray,
    component: dict[str, np.ndarray],
) -> dict[str, np.ndarray | float]:
    cosine = row_cos(prediction, truth)
    recall = top_path_recall(prediction, truth, query_ids, component)
    emd = delay_emd(prediction, truth)
    return {
        "cosine_rows": cosine,
        "top_recall_rows": recall,
        "emd_rows": emd,
        "cosine": float(np.mean(cosine)),
        "top_path_mass_recall": float(np.mean(recall)),
        "delay_emd": float(np.mean(emd)),
    }


def metric_summary(metric: dict[str, np.ndarray | float]) -> dict[str, float]:
    return {
        "cosine": float(metric["cosine"]),
        "top_path_mass_recall": float(metric["top_path_mass_recall"]),
        "delay_emd": float(metric["delay_emd"]),
    }


def prior_token_energy(data: FoldData) -> np.ndarray:
    value = data.base_weights[:, :, None] * data.component_weight
    value *= data.component_valid
    value = value.reshape(len(data.query_ids), -1)
    value /= np.maximum(value.sum(axis=1, keepdims=True), 1e-20)
    return value.astype(np.float32)


def reliability_audit(
    reliability: dict[str, np.ndarray],
    component: dict[str, np.ndarray],
    valid_ids: np.ndarray,
) -> dict[str, object]:
    valid = component["valid"][valid_ids]
    feature = reliability["feature"][valid_ids][valid]
    aggregate = reliability["aggregate"][valid_ids][valid]
    weight = component["weight"][valid_ids][valid]
    pair_summary = {}
    for index, name in enumerate(VIEW_PAIRS):
        pair_summary[name] = {
            "mass_agreement": summarize(feature[:, index]),
            "peak_agreement": summarize(feature[:, len(VIEW_PAIRS) + index]),
            "window_cosine": summarize(feature[:, 2 * len(VIEW_PAIRS) + index]),
        }
    correlation = spearmanr(aggregate, weight).statistic
    return {
        "pair_summary": pair_summary,
        "aggregate": summarize(aggregate),
        "aggregate_vs_component_mass_spearman": float(correlation),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    for path in (
        TARGET_CACHE,
        v297.COMPONENT_CACHE,
        v274.SCENE_CACHE,
        v274.FOOTPRINT_CACHE,
        v274.SIGNATURE_CACHE,
        v238.GEOM_CACHE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = np.load(TARGET_CACHE)
    truth_d = normalize_rows(targets["d"].astype(np.float32))
    valid_ids = np.flatnonzero(targets["valid"].astype(bool)).astype(np.int64)
    component_data = np.load(v297.COMPONENT_CACHE)
    component = {key: component_data[key] for key in component_data.files}
    reliability = build_multiview_reliability(device)
    audit = reliability_audit(reliability, component, valid_ids)
    if args.audit_only:
        aggregate = audit["aggregate"]
        report = {
            "method": VERSION,
            "status": "analysis_only_no_candidate",
            "scope": "per-component official-H reliability audit only; training stopped before OOF model fitting",
            "primary_hypothesis": (
                "A real path repeats across independent official-H link views, so local "
                "mass/peak/window agreement might separate transportable paths from noise."
            ),
            "historical_control": (
                "v356 already found point/fingerprint half-view reliability nearly saturated; "
                "v365 tests finer local component mass and peak agreement before any model fit."
            ),
            "reliability_audit": audit,
            "decision": {
                "aggregate_p10": float(aggregate["p10"]),
                "aggregate_median": float(aggregate["median"]),
                "entered_model_training": False,
                "reason": (
                    "P10 and median remain near one and reproduce v356 saturation; "
                    "the feature does not create enough dynamic range for a causal transport test."
                ),
            },
            "compliance": {
                "uses": ["official train H and per-point continuous components"],
                "does_not_use": [
                    "outer-fold model fitting",
                    "Sionna/synthetic labels or v212 teacher",
                    "test prediction, full-H projection, or ZIP",
                ],
            },
            "outputs": {
                "reliability_cache": PATHS.relative(RELIABILITY_CACHE),
                "report": PATHS.relative(REPORT),
                "oof": None,
                "prediction": None,
                "zip": None,
            },
        }
        REPORT.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report["decision"], ensure_ascii=False, indent=2), flush=True)
        return
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    geometry = v274.load_geometry()
    (
        all_pos,
        signatures,
        blocker_count,
        scene_features,
        _footprint,
        _z_max,
        _x_min,
        _y_min,
        _resolution,
    ) = geometry
    node_raw = v274.build_node_features(all_pos, signatures, blocker_count, scene_features)
    folds = v315.disjoint_folds(valid_ids, train_pos[:, :2])

    fold_rows: list[dict[str, object]] = []
    oof: dict[str, list[np.ndarray]] = {
        "ids": [],
        "truth": [],
        "baseline": [],
        "shuffled": [],
        "full": [],
    }
    propagation_geometry = np.load(v238.GEOM_CACHE)
    all_regime = regime_of(propagation_geometry["train_blocked"].astype(np.float32))
    for fold, outer_ids in enumerate(folds):
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = folds[(fold + 1) % len(folds)]
        inner_ids = np.intersect1d(inner_ids, fit_ids)
        inner_fit = np.setdiff1d(fit_ids, inner_ids)
        if np.intersect1d(outer_ids, fit_ids).size:
            raise AssertionError("outer H entered v365 donor bank")

        inner_train_set, inner_val_set, inner_node = build_sets(
            inner_fit, inner_ids, inner_fit, geometry, node_raw
        )
        outer_train_set, outer_val_set, outer_node = build_sets(
            fit_ids, outer_ids, fit_ids, geometry, node_raw
        )
        inner_shuffle_feature, inner_shuffle_aggregate = shuffled_reliability(
            reliability["feature"], reliability["aggregate"], component["valid"],
            inner_fit, 365_000 + fold,
        )
        outer_shuffle_feature, outer_shuffle_aggregate = shuffled_reliability(
            reliability["feature"], reliability["aggregate"], component["valid"],
            fit_ids, 366_000 + fold,
        )

        modes = {
            "full": (
                reliability["feature"], reliability["aggregate"],
                reliability["feature"], reliability["aggregate"],
            ),
            "shuffled": (
                inner_shuffle_feature, inner_shuffle_aggregate,
                outer_shuffle_feature, outer_shuffle_aggregate,
            ),
        }
        mode_predictions: dict[str, np.ndarray] = {}
        mode_tokens: dict[str, np.ndarray] = {}
        mode_reports: dict[str, object] = {}
        outer_data_for_metric: FoldData | None = None
        for mode_index, (mode, values) in enumerate(modes.items()):
            inner_feature, inner_aggregate, outer_feature, outer_aggregate = values
            inner_train = make_fold_data(
                inner_train_set, inner_node, component, truth_d,
                inner_feature, inner_aggregate,
            )
            inner_val = make_fold_data(
                inner_val_set, inner_node, component, truth_d,
                inner_feature, inner_aggregate,
            )
            inner_model, inner_report = fit_model(
                inner_train, inner_val, truth_d,
                3650 + fold * 10 + mode_index, MAX_EPOCHS, device,
            )
            selected_epoch = int(inner_report["best_epoch"])
            del inner_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            outer_train = make_fold_data(
                outer_train_set, outer_node, component, truth_d,
                outer_feature, outer_aggregate,
            )
            outer_val = make_fold_data(
                outer_val_set, outer_node, component, truth_d,
                outer_feature, outer_aggregate,
            )
            outer_data_for_metric = outer_val
            if selected_epoch == 0:
                prediction = outer_val.anchor.copy()
                gate = np.zeros(len(outer_ids), dtype=np.float32)
                token_energy = prior_token_energy(outer_val)
                outer_report: dict[str, object] = {"best_epoch": 0}
            else:
                model, outer_report = fit_model(
                    outer_train, None, truth_d,
                    4650 + fold * 10 + mode_index, selected_epoch, device,
                )
                prediction, gate, token_energy = predict(model, outer_val, device)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            mode_predictions[mode] = prediction
            mode_tokens[mode] = token_energy
            mode_reports[mode] = {
                "selected_epoch": selected_epoch,
                "inner": inner_report,
                "outer": outer_report,
                "gate": summarize(gate),
            }

        if outer_data_for_metric is None:
            raise AssertionError("missing outer fold data")
        truth = truth_d[outer_ids]
        baseline_metric = evaluate(
            outer_data_for_metric.anchor, truth, outer_ids, component
        )
        shuffled_metric = evaluate(
            mode_predictions["shuffled"], truth, outer_ids, component
        )
        full_metric = evaluate(mode_predictions["full"], truth, outer_ids, component)
        regime_rows = {}
        for regime_name in ("clear", "light", "medium", "heavy"):
            mask = all_regime[outer_ids] == regime_name
            if not np.any(mask):
                continue
            regime_rows[regime_name] = {
                "count": int(np.sum(mask)),
                "baseline_cosine": float(np.mean(baseline_metric["cosine_rows"][mask])),
                "full_cosine": float(np.mean(full_metric["cosine_rows"][mask])),
                "cosine_lift": float(np.mean(
                    full_metric["cosine_rows"][mask] - baseline_metric["cosine_rows"][mask]
                )),
            }
        row = {
            "fold": fold,
            "fit_count": int(len(fit_ids)),
            "inner_fit_count": int(len(inner_fit)),
            "inner_validation_count": int(len(inner_ids)),
            "outer_count": int(len(outer_ids)),
            "baseline": metric_summary(baseline_metric),
            "shuffled_reliability": metric_summary(shuffled_metric),
            "full": metric_summary(full_metric),
            "full_delta": {
                "cosine": float(full_metric["cosine"] - baseline_metric["cosine"]),
                "top_path_mass_recall": float(
                    full_metric["top_path_mass_recall"] - baseline_metric["top_path_mass_recall"]
                ),
                "delay_emd": float(full_metric["delay_emd"] - baseline_metric["delay_emd"]),
            },
            "full_vs_shuffled_cosine": float(
                full_metric["cosine"] - shuffled_metric["cosine"]
            ),
            "path_existence_ap": {
                "baseline_prior": token_ap(
                    prior_token_energy(outer_data_for_metric), outer_data_for_metric, component
                ),
                "shuffled": token_ap(
                    mode_tokens["shuffled"], outer_data_for_metric, component
                ),
                "full": token_ap(mode_tokens["full"], outer_data_for_metric, component),
            },
            "regimes": regime_rows,
            "training": mode_reports,
            "leakage_assertions": {
                "outer_in_donor_bank": 0,
                "outer_in_inner_training": 0,
            },
        }
        fold_rows.append(row)
        oof["ids"].append(outer_ids)
        oof["truth"].append(truth)
        oof["baseline"].append(outer_data_for_metric.anchor)
        oof["shuffled"].append(mode_predictions["shuffled"])
        oof["full"].append(mode_predictions["full"])
        print(
            f"v365 fold={fold} base={baseline_metric['cosine']:.6f} "
            f"shuffle={shuffled_metric['cosine']:.6f} full={full_metric['cosine']:.6f} "
            f"delta={full_metric['cosine']-baseline_metric['cosine']:+.6f}",
            flush=True,
        )

    merged = {key: np.concatenate(value) for key, value in oof.items()}
    order = np.argsort(merged["ids"])
    for key in merged:
        merged[key] = merged[key][order]
    np.savez_compressed(OOF_OUT, **merged)
    base_metric = evaluate(merged["baseline"], merged["truth"], merged["ids"], component)
    shuffled_metric = evaluate(
        merged["shuffled"], merged["truth"], merged["ids"], component
    )
    full_metric = evaluate(merged["full"], merged["truth"], merged["ids"], component)
    cosine_lift = float(full_metric["cosine"] - base_metric["cosine"])
    recall_lift = float(
        full_metric["top_path_mass_recall"] - base_metric["top_path_mass_recall"]
    )
    emd_reduction = float(
        (base_metric["delay_emd"] - full_metric["delay_emd"])
        / max(float(base_metric["delay_emd"]), 1e-20)
    )
    fold_wins = int(sum(row["full_delta"]["cosine"] > 0.0 for row in fold_rows))
    full_vs_shuffled = float(full_metric["cosine"] - shuffled_metric["cosine"])
    regime_decision = {}
    for regime_name in ("clear", "light", "medium", "heavy"):
        mask = all_regime[merged["ids"]] == regime_name
        regime_decision[regime_name] = {
            "count": int(np.sum(mask)),
            "cosine_lift": float(np.mean(
                full_metric["cosine_rows"][mask] - base_metric["cosine_rows"][mask]
            )),
        }
    passed = bool(
        cosine_lift >= MIN_COSINE_LIFT
        and recall_lift >= MIN_TOP_RECALL_LIFT
        and emd_reduction >= MIN_EMD_REDUCTION
        and fold_wins >= MIN_FOLD_WINS
        and full_vs_shuffled >= MIN_RELIABILITY_CONTROL_LIFT
        and regime_decision["medium"]["cosine_lift"] > 0.0
        and regime_decision["heavy"]["cosine_lift"] > 0.0
        and regime_decision["clear"]["cosine_lift"] >= -MAX_CLEAR_DEGRADATION
    )
    report = {
        "method": VERSION,
        "status": "strict_pdp_module_passed" if passed else "analysis_only_no_candidate",
        "scope": "normalized global D only; no full-H, phase, total power, Sionna prior, or ZIP",
        "primary_hypothesis": (
            "A real continuous-delay path repeats across independent official-H link views. "
            "Per-component view agreement should disambiguate donor-query path transport "
            "without assigning a global virtual-source identity."
        ),
        "historical_control": (
            "v356 used point/fingerprint-level half-view cosine to rescale global source mass; "
            "v365 instead measures local mass/peak/window agreement for each continuous path "
            "and supplies it to a query-conditioned donor-token model."
        ),
        "compliance": {
            "uses": [
                "official train H and normalized PDP labels",
                "official train/test planar coordinates",
                "official point-cloud geometry features",
            ],
            "does_not_use": [
                "Sionna/synthetic channel labels or v212 teacher",
                "outer-fold H in features, donor bank, training, or model selection",
                "hidden/local/online truth",
                "full-H projection or submission packaging",
            ],
        },
        "parameters": {
            "view_pairs": list(VIEW_PAIRS),
            "neighbors": NEIGHBORS,
            "components_per_neighbor": COMPONENT_COUNT,
            "offsets": list(OFFSETS),
            "max_epochs": MAX_EPOCHS,
            "folds": "v315 five mutually exclusive isolation-stratified spatial folds",
            "control": "same architecture with active donor-component reliability shuffled inside each fit bank",
        },
        "reliability_audit": audit,
        "fold_rows": fold_rows,
        "summary": {
            "baseline": metric_summary(base_metric),
            "shuffled_reliability": metric_summary(shuffled_metric),
            "full": metric_summary(full_metric),
            "cosine_lift": cosine_lift,
            "top_path_mass_recall_lift": recall_lift,
            "delay_emd_reduction_fraction": emd_reduction,
            "fold_wins": fold_wins,
            "full_vs_shuffled_cosine": full_vs_shuffled,
            "regimes": regime_decision,
        },
        "decision_rule": {
            "cosine_lift": f">={MIN_COSINE_LIFT}",
            "top_path_mass_recall_lift": f">={MIN_TOP_RECALL_LIFT}",
            "delay_emd_reduction_fraction": f">={MIN_EMD_REDUCTION}",
            "fold_wins": f">={MIN_FOLD_WINS}/5",
            "full_vs_shuffled_cosine": f">={MIN_RELIABILITY_CONTROL_LIFT}",
            "medium_heavy": "both positive",
            "clear": f">=-{MAX_CLEAR_DEGRADATION}",
            "passed": passed,
        },
        "outputs": {
            "reliability_cache": PATHS.relative(RELIABILITY_CACHE),
            "oof": PATHS.relative(OOF_OUT),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(report["decision_rule"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
