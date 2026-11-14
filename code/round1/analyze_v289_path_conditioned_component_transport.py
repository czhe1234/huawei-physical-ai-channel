#!/usr/bin/env python3
"""Strict path-family-conditioned transport of official angle-delay components."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v254_route_topology_spectral_field as v254
import analyze_v257_neural_pairwise_spectral_transport as v257
import analyze_v264_patch_scatterer_transformer as v264
import analyze_v265_scatterer_delay_angle_oracle as v265
import analyze_v274_planar_barrier_attention as v274
import analyze_v279_joint_component_trajectory as v279
import analyze_v283_joint_geometry_path_renderer as v283
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 289
VERSION = "v289_path_conditioned_component_transport"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
TARGETS_OUT = OUT_DIR / "v289_path_transport_targets.npz"
OUTPUT = OUT_DIR / "v289_Round1_Test_Channel.npy"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
TRAIN_CANDIDATES = PATHS.cache_dir(263) / "v263_full_wall_reflection_candidates.npz"
TEST_CANDIDATES = PATHS.cache_dir(264) / "v264_test_wall_reflection_candidates.npz"
SCENE_CACHE = PATHS.shared_features / "official_pointcloud_scene_cache.npz"
V265_REPORT = PATHS.version_report_dir(265) / "v265_scatterer_delay_angle_oracle.json"

LOCAL_GATE = 0.645604
NEIGHBORS = 16
PEAKS = 8
PATH_OPTIONS = 4
OPTIONS = PATH_OPTIONS + 1
EPOCHS = 60
EVAL_EVERY = 5
BATCH_SIZE = 48
OUTER_SEEDS = (289, 1289, 2289, 3289, 4289)
OUTER_COUNT = 320
INNER_COUNT = 240
FINAL_SEEDS = (28901, 28931, 28961)
MIN_OFFICIAL_DELTA = 5e-4
S_SIZE = 192
H_SIZE = 16


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def signed_circular_delta(target: float, source: float, size: int) -> float:
    return float((target - source + size / 2.0) % size - size / 2.0)


@dataclass
class PathGeometry:
    path_id: np.ndarray
    valid: np.ndarray
    center_s: np.ndarray
    center_h: np.ndarray
    delay: np.ndarray
    score: np.ndarray

    def take(self, ids: np.ndarray) -> "PathGeometry":
        return PathGeometry(
            self.path_id[ids],
            self.valid[ids],
            self.center_s[ids],
            self.center_h[ids],
            self.delay[ids],
            self.score[ids],
        )


@dataclass
class TransportSet:
    query_ids: np.ndarray
    neighbor_ids: np.ndarray
    query_node: np.ndarray
    neighbor_node: np.ndarray
    anchor: np.ndarray
    source_index: np.ndarray
    source_mass: np.ndarray
    token_numeric: np.ndarray
    option_destination: np.ndarray
    option_patch_id: np.ndarray
    option_numeric: np.ndarray
    option_prior: np.ndarray
    option_valid: np.ndarray
    shared_path_count: np.ndarray

    def standardize_nodes(
        self, mean: np.ndarray, std: np.ndarray
    ) -> "TransportSet":
        return TransportSet(
            self.query_ids,
            self.neighbor_ids,
            np.clip((self.query_node - mean) / std, -8.0, 8.0).astype(np.float32),
            np.clip((self.neighbor_node - mean) / std, -8.0, 8.0).astype(np.float32),
            self.anchor,
            self.source_index,
            self.source_mass,
            self.token_numeric,
            self.option_destination,
            self.option_patch_id,
            self.option_numeric,
            self.option_prior,
            self.option_valid,
            self.shared_path_count,
        )


def build_path_geometry(
    positions: np.ndarray,
    candidates: dict[str, np.ndarray],
    node: np.ndarray,
    patches: np.ndarray,
    spec: dict[str, object],
) -> PathGeometry:
    feature = v264.build_features(positions, candidates, patches, node)
    azimuth, valid, path_id = v265.departure_azimuth(positions, candidates)
    center_h = v265.h_center_for(
        azimuth,
        float(spec["orientation"]),
        str(spec["h_mode"]),
        float(spec["h_sign"]),
        float(spec["h_shift"]),
    )
    center_s = np.mod(float(spec["s_sign"]) * feature.delay, float(S_SIZE))
    score = np.ones_like(feature.delay, dtype=np.float32)
    score[:, 1:] = candidates["score"][:, : v264.REFLECTIONS].astype(np.float32)
    return PathGeometry(
        path_id=path_id.astype(np.int64),
        valid=valid.astype(bool),
        center_s=center_s.astype(np.float32),
        center_h=center_h.astype(np.float32),
        delay=feature.delay.astype(np.float32),
        score=np.clip(score, 0.0, 1.0).astype(np.float32),
    )


def top_peaks(sh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(sh, dtype=np.float32).reshape(len(sh), -1)
    indices = np.argpartition(flat, -PEAKS, axis=1)[:, -PEAKS:]
    values = np.take_along_axis(flat, indices, axis=1)
    order = np.argsort(values, axis=1)[:, ::-1]
    return (
        np.take_along_axis(indices, order, axis=1).astype(np.int64),
        np.take_along_axis(values, order, axis=1).astype(np.float32),
    )


def neighbor_query(
    query_pos: np.ndarray,
    bank_pos: np.ndarray,
    bank_ids: np.ndarray,
    exclude_ids: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    extra = 1 if exclude_ids is not None else 0
    distance, local = cKDTree(bank_pos[bank_ids, :2]).query(
        query_pos[:, :2], k=NEIGHBORS + extra
    )
    if distance.ndim == 1:
        distance = distance[:, None]
        local = local[:, None]
    absolute_all = bank_ids[local]
    if exclude_ids is None:
        absolute = absolute_all
        kept_distance = distance
    else:
        absolute = np.empty((len(query_pos), NEIGHBORS), dtype=np.int64)
        kept_distance = np.empty((len(query_pos), NEIGHBORS), dtype=np.float64)
        for row, query_id in enumerate(exclude_ids):
            keep = absolute_all[row] != query_id
            absolute[row] = absolute_all[row, keep][:NEIGHBORS]
            kept_distance[row] = distance[row, keep][:NEIGHBORS]
    weight = 1.0 / np.maximum(kept_distance, 0.05) ** 2
    weight /= weight.sum(axis=1, keepdims=True)
    return absolute, kept_distance.astype(np.float32), weight.astype(np.float32)


def build_transport_set(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    query_pos_all: np.ndarray,
    bank_pos: np.ndarray,
    query_node_all: np.ndarray,
    bank_node: np.ndarray,
    query_geometry: PathGeometry,
    bank_geometry: PathGeometry,
    bank_sh: np.ndarray,
    exclude_self: bool,
) -> TransportSet:
    query_pos = query_pos_all[query_ids]
    exclude = query_ids if exclude_self else None
    neighbor_ids, distance, weight = neighbor_query(
        query_pos, bank_pos, bank_ids, exclude
    )
    peak_index_all, peak_mass_all = top_peaks(bank_sh)
    peak_index = peak_index_all[neighbor_ids]
    peak_mass = peak_mass_all[neighbor_ids]
    neighbor_sh = bank_sh[neighbor_ids]
    anchor = v279.normalize(
        np.sum(
            neighbor_sh * weight[:, :, None, None],
            axis=1,
            dtype=np.float64,
        ).astype(np.float32)
    )

    count = len(query_ids)
    tokens = NEIGHBORS * PEAKS
    source_index = peak_index.reshape(count, tokens).astype(np.int64)
    source_mass = (
        peak_mass * weight[:, :, None]
    ).reshape(count, tokens).astype(np.float32)
    token_numeric = np.zeros((count, tokens, 12), dtype=np.float32)
    option_destination = np.repeat(
        source_index[:, :, None], OPTIONS, axis=2
    ).astype(np.int64)
    option_patch_id = np.zeros((count, tokens, OPTIONS), dtype=np.int64)
    option_numeric = np.zeros((count, tokens, OPTIONS, 14), dtype=np.float32)
    option_prior = np.full((count, tokens, OPTIONS), -4.0, dtype=np.float32)
    option_valid = np.zeros((count, tokens, OPTIONS), dtype=bool)
    option_valid[:, :, 0] = True
    option_prior[:, :, 0] = 1.65
    option_numeric[:, :, 0, 0] = 1.0
    shared_path_count = np.zeros((count, NEIGHBORS), dtype=np.int16)

    for row, query_id in enumerate(query_ids):
        q_valid = np.flatnonzero(query_geometry.valid[query_id])
        q_lookup = {
            int(query_geometry.path_id[query_id, path]): int(path)
            for path in q_valid
        }
        qx, qy = query_pos_all[query_id, :2]
        for neighbor_rank, donor_id in enumerate(neighbor_ids[row]):
            donor_id = int(donor_id)
            donor_paths = [
                int(path)
                for path in np.flatnonzero(bank_geometry.valid[donor_id])
                if int(bank_geometry.path_id[donor_id, path]) in q_lookup
            ]
            shared_path_count[row, neighbor_rank] = len(donor_paths)
            dx = float(qx - bank_pos[donor_id, 0])
            dy = float(qy - bank_pos[donor_id, 1])
            edge_distance = float(distance[row, neighbor_rank])
            for peak_rank in range(PEAKS):
                token = neighbor_rank * PEAKS + peak_rank
                source = int(peak_index[row, neighbor_rank, peak_rank])
                source_s = source // H_SIZE
                source_h = source % H_SIZE
                mass = float(peak_mass[row, neighbor_rank, peak_rank])
                top_mass = max(float(peak_mass[row, neighbor_rank, 0]), 1e-12)
                angle_s = 2.0 * math.pi * source_s / S_SIZE
                angle_h = 2.0 * math.pi * source_h / H_SIZE
                token_numeric[row, token] = np.asarray(
                    [
                        math.sin(angle_s),
                        math.cos(angle_s),
                        math.sin(angle_h),
                        math.cos(angle_h),
                        math.log(max(mass, 1e-12)) / 12.0,
                        mass / top_mass,
                        peak_rank / max(PEAKS - 1, 1),
                        float(weight[row, neighbor_rank]),
                        dx / 20.0,
                        dy / 20.0,
                        edge_distance / 20.0,
                        math.log1p(len(donor_paths)) / 4.0,
                    ],
                    dtype=np.float32,
                )

                ranked: list[tuple[float, int, int, float, float, float, float]] = []
                for donor_path in donor_paths:
                    path_id = int(bank_geometry.path_id[donor_id, donor_path])
                    query_path = q_lookup[path_id]
                    source_ds = signed_circular_delta(
                        float(bank_geometry.center_s[donor_id, donor_path]),
                        float(source_s),
                        S_SIZE,
                    )
                    source_dh = signed_circular_delta(
                        float(bank_geometry.center_h[donor_id, donor_path]),
                        float(source_h),
                        H_SIZE,
                    )
                    proximity = (source_ds / 2.2) ** 2 + (source_dh / 1.8) ** 2
                    delta_s = signed_circular_delta(
                        float(query_geometry.center_s[query_id, query_path]),
                        float(bank_geometry.center_s[donor_id, donor_path]),
                        S_SIZE,
                    )
                    delta_h = signed_circular_delta(
                        float(query_geometry.center_h[query_id, query_path]),
                        float(bank_geometry.center_h[donor_id, donor_path]),
                        H_SIZE,
                    )
                    ranked.append(
                        (
                            proximity,
                            donor_path,
                            query_path,
                            source_ds,
                            source_dh,
                            delta_s,
                            delta_h,
                        )
                    )
                ranked.sort(key=lambda value: value[0])
                for option, item in enumerate(ranked[:PATH_OPTIONS], start=1):
                    (
                        proximity,
                        donor_path,
                        query_path,
                        source_ds,
                        source_dh,
                        delta_s,
                        delta_h,
                    ) = item
                    path_id = int(bank_geometry.path_id[donor_id, donor_path])
                    destination_s = (source_s + int(np.rint(delta_s))) % S_SIZE
                    destination_h = (source_h + int(np.rint(delta_h))) % H_SIZE
                    option_destination[row, token, option] = (
                        destination_s * H_SIZE + destination_h
                    )
                    option_patch_id[row, token, option] = path_id + 1
                    donor_score = float(bank_geometry.score[donor_id, donor_path])
                    query_score = float(query_geometry.score[query_id, query_path])
                    option_numeric[row, token, option] = np.asarray(
                        [
                            0.0,
                            source_ds / 4.0,
                            source_dh / 4.0,
                            delta_s / 4.0,
                            delta_h / 4.0,
                            float(bank_geometry.delay[donor_id, donor_path]) / 24.0,
                            float(query_geometry.delay[query_id, query_path]) / 24.0,
                            donor_score,
                            query_score,
                            math.sqrt(max(donor_score * query_score, 0.0)),
                            proximity / 8.0,
                            float(path_id == 0),
                            float(abs(delta_s) <= 2.0),
                            float(abs(delta_h) <= 2.0),
                        ],
                        dtype=np.float32,
                    )
                    option_prior[row, token, option] = float(
                        -0.42 * proximity
                        + 0.35 * math.sqrt(max(donor_score * query_score, 0.0))
                    )
                    option_valid[row, token, option] = proximity <= 9.0

    return TransportSet(
        query_ids=query_ids.astype(np.int64),
        neighbor_ids=neighbor_ids,
        query_node=query_node_all[query_ids].astype(np.float32),
        neighbor_node=bank_node[neighbor_ids].astype(np.float32),
        anchor=anchor,
        source_index=source_index,
        source_mass=source_mass,
        token_numeric=token_numeric,
        option_destination=option_destination,
        option_patch_id=option_patch_id,
        option_numeric=option_numeric,
        option_prior=option_prior,
        option_valid=option_valid,
        shared_path_count=shared_path_count,
    )


class PathTransportNet(nn.Module):
    def __init__(self, node_dim: int, patch_count: int) -> None:
        super().__init__()
        self.node = nn.Sequential(
            nn.Linear(node_dim, 192),
            nn.SiLU(),
            nn.LayerNorm(192),
            nn.Linear(192, 80),
        )
        self.token = nn.Sequential(
            nn.Linear(12, 80), nn.SiLU(), nn.Linear(80, 48)
        )
        self.option = nn.Sequential(
            nn.Linear(14, 80), nn.SiLU(), nn.Linear(80, 48)
        )
        self.patch = nn.Embedding(patch_count + 2, 24)
        self.trunk = nn.Sequential(
            nn.Linear(80 * 3 + 48 * 2 + 24, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 160),
            nn.SiLU(),
            nn.Linear(160, 1),
        )

    def forward(
        self,
        query_node: torch.Tensor,
        neighbor_node: torch.Tensor,
        anchor: torch.Tensor,
        source_index: torch.Tensor,
        source_mass: torch.Tensor,
        token_numeric: torch.Tensor,
        option_destination: torch.Tensor,
        option_patch_id: torch.Tensor,
        option_numeric: torch.Tensor,
        option_prior: torch.Tensor,
        option_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, options = option_destination.shape
        query = self.node(query_node)[:, None, None, :].expand(-1, tokens, options, -1)
        donor = self.node(neighbor_node)
        donor = donor[:, :, None, :].expand(-1, -1, PEAKS, -1)
        donor = donor.reshape(batch, tokens, -1)[:, :, None, :].expand(-1, -1, options, -1)
        token = self.token(token_numeric)[:, :, None, :].expand(-1, -1, options, -1)
        option = self.option(option_numeric)
        hidden = torch.cat(
            [query, donor, query * donor, token, option, self.patch(option_patch_id)],
            dim=3,
        )
        delta = 2.4 * torch.tanh(self.trunk(hidden).squeeze(3))
        logits = option_prior + delta
        logits = logits.masked_fill(~option_valid, -1e9)
        probability = torch.softmax(logits, dim=2)

        removed = torch.zeros_like(anchor)
        removed.scatter_add_(1, source_index, source_mass)
        added = torch.zeros_like(anchor)
        added.scatter_add_(
            1,
            option_destination.reshape(batch, -1),
            (source_mass[:, :, None] * probability).reshape(batch, -1),
        )
        prediction = torch.clamp(anchor - removed + added, min=0.0)
        prediction /= prediction.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return prediction, probability


def batch_forward(
    model: PathTransportNet,
    data: TransportSet,
    rows: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return model(
        torch.from_numpy(data.query_node[rows]).to(device),
        torch.from_numpy(data.neighbor_node[rows]).to(device),
        torch.from_numpy(data.anchor[rows].reshape(len(rows), -1)).to(device),
        torch.from_numpy(data.source_index[rows]).to(device),
        torch.from_numpy(data.source_mass[rows]).to(device),
        torch.from_numpy(data.token_numeric[rows]).to(device),
        torch.from_numpy(data.option_destination[rows]).to(device),
        torch.from_numpy(data.option_patch_id[rows]).to(device),
        torch.from_numpy(data.option_numeric[rows]).to(device),
        torch.from_numpy(data.option_prior[rows]).to(device),
        torch.from_numpy(data.option_valid[rows]).to(device),
    )


@torch.no_grad()
def predict(
    model: PathTransportNet,
    data: TransportSet,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    output = np.empty((len(data.query_ids), S_SIZE, H_SIZE), dtype=np.float32)
    move = np.empty(len(data.query_ids), dtype=np.float32)
    for start in range(0, len(data.query_ids), BATCH_SIZE * 2):
        end = min(start + BATCH_SIZE * 2, len(data.query_ids))
        rows = np.arange(start, end, dtype=np.int64)
        value, probability = batch_forward(model, data, rows, device)
        output[start:end] = value.reshape(-1, S_SIZE, H_SIZE).cpu().numpy()
        move[start:end] = (
            1.0 - probability[:, :, 0]
        ).mean(dim=1).cpu().numpy()
    return v279.normalize(output), move


def train_model(
    train_data: TransportSet,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
    patch_count: int,
    validation: TransportSet | None,
) -> tuple[PathTransportNet, dict[str, object]]:
    set_seed(seed)
    model = PathTransportNet(train_data.query_node.shape[1], patch_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    rng = np.random.default_rng(seed)
    best_metric = -np.inf
    best_epoch = epochs
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, object]] = []
    validation_base = None
    if validation is not None:
        validation_base = v279.metrics(
            validation.anchor,
            truth_sh[validation.query_ids],
            truth_d[validation.query_ids],
        )
        best_metric = validation_base["weighted"]
        best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_data.query_ids))
        losses: list[float] = []
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start : start + BATCH_SIZE]
            prediction, probability = batch_forward(model, train_data, rows, device)
            target_sh = torch.from_numpy(
                truth_sh[train_data.query_ids[rows]].reshape(len(rows), -1)
            ).to(device)
            target_d = torch.from_numpy(truth_d[train_data.query_ids[rows]]).to(device)
            pred_d = v279.derive_d_torch(prediction)
            sh_loss = 1.0 - F.cosine_similarity(prediction, target_sh, dim=1).mean()
            d_loss = 1.0 - F.cosine_similarity(pred_d, target_d, dim=1).mean()
            move = (1.0 - probability[:, :, 0]).mean()
            loss = 0.52 * sh_loss + 0.48 * d_loss + 8e-5 * move
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        if epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == epochs:
            item: dict[str, object] = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
            }
            if validation is not None:
                value, move = predict(model, validation, device)
                metric = v279.metrics(
                    value,
                    truth_sh[validation.query_ids],
                    truth_d[validation.query_ids],
                )
                item["validation"] = metric
                item["baseline"] = validation_base
                item["move"] = summarize(move)
                if metric["weighted"] > best_metric:
                    best_metric = metric["weighted"]
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            history.append(item)
        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"v289 seed={seed} epoch={epoch}/{epochs} loss={np.mean(losses):.6f}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_weighted": float(best_metric) if np.isfinite(best_metric) else None,
        "baseline": validation_base,
        "history": history,
    }


def node_stats(node: np.ndarray, fit_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = node[fit_ids].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = node[fit_ids].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-5] = 1.0
    return mean, std


def main() -> None:
    for path in (
        BASE,
        TARGET_CACHE,
        TRAIN_CANDIDATES,
        TEST_CANDIDATES,
        SCENE_CACHE,
        V265_REPORT,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    target_data = np.load(TARGET_CACHE)
    truth_sh = v279.normalize(target_data["sh"].astype(np.float32))
    truth_d = v279.normalize(target_data["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target_data["valid"].astype(bool)).astype(np.int64)
    train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_pos = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)
    train_node, test_node = v257.build_node_features(
        train_pos[:, :2], test_pos[:, :2]
    )
    patches = np.load(SCENE_CACHE)["wall_patches"].astype(np.float32)
    train_candidates = dict(np.load(TRAIN_CANDIDATES))
    test_candidates = dict(np.load(TEST_CANDIDATES))
    with V265_REPORT.open("r", encoding="utf-8") as handle:
        best_spec = json.load(handle)["selection"]["best"]
    train_geometry = build_path_geometry(
        train_pos, train_candidates, train_node, patches, best_spec
    )
    test_geometry = build_path_geometry(
        test_pos, test_candidates, test_node, patches, best_spec
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"v289 device={device} valid={len(valid_ids)} node_dim={train_node.shape[1]}",
        flush=True,
    )

    outer_rows: list[dict[str, object]] = []
    oof_ids_parts: list[np.ndarray] = []
    oof_base_parts: list[np.ndarray] = []
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

        mean, std = node_stats(train_node, inner_fit)
        inner_train = build_transport_set(
            inner_fit,
            inner_fit,
            train_pos,
            train_pos,
            train_node,
            train_node,
            train_geometry,
            train_geometry,
            truth_sh,
            True,
        ).standardize_nodes(mean, std)
        inner_validation = build_transport_set(
            inner_ids,
            inner_fit,
            train_pos,
            train_pos,
            train_node,
            train_node,
            train_geometry,
            train_geometry,
            truth_sh,
            False,
        ).standardize_nodes(mean, std)
        _, inner_training = train_model(
            inner_train,
            truth_sh,
            truth_d,
            seed,
            EPOCHS,
            device,
            len(patches),
            inner_validation,
        )
        selected_epoch = int(inner_training["best_epoch"])
        selected_epochs.append(selected_epoch)

        mean, std = node_stats(train_node, outer_fit)
        outer_train = build_transport_set(
            outer_fit,
            outer_fit,
            train_pos,
            train_pos,
            train_node,
            train_node,
            train_geometry,
            train_geometry,
            truth_sh,
            True,
        ).standardize_nodes(mean, std)
        outer_validation = build_transport_set(
            outer_ids,
            outer_fit,
            train_pos,
            train_pos,
            train_node,
            train_node,
            train_geometry,
            train_geometry,
            truth_sh,
            False,
        ).standardize_nodes(mean, std)
        if selected_epoch == 0:
            model_sh = outer_validation.anchor.copy()
            move = np.zeros(len(outer_ids), dtype=np.float32)
            outer_training = {"seed": seed + 100_000, "best_epoch": 0, "history": []}
        else:
            model, outer_training = train_model(
                outer_train,
                truth_sh,
                truth_d,
                seed + 100_000,
                selected_epoch,
                device,
                len(patches),
                None,
            )
            model_sh, move = predict(model, outer_validation, device)
        baseline = v279.metrics(
            outer_validation.anchor, truth_sh[outer_ids], truth_d[outer_ids]
        )
        model_metric = v279.metrics(
            model_sh, truth_sh[outer_ids], truth_d[outer_ids]
        )
        delta = {
            key: model_metric[key] - baseline[key]
            for key in ("sh", "d", "weighted")
        }
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
                "model": model_metric,
                "raw_delta": delta,
                "move": summarize(move),
                "shared_paths": summarize(
                    outer_validation.shared_path_count.reshape(-1)
                ),
            }
        )
        oof_ids_parts.append(outer_ids)
        oof_base_parts.append(outer_validation.anchor)
        oof_model_parts.append(model_sh)
        print(
            f"v289 outer {fold + 1}/{len(OUTER_SEEDS)} epoch={selected_epoch} "
            f"base={baseline['weighted']:.6f} model={model_metric['weighted']:.6f} "
            f"delta={delta['weighted']:+.6f}",
            flush=True,
        )

    oof_ids = np.concatenate(oof_ids_parts)
    oof_base = np.concatenate(oof_base_parts)
    oof_model = np.concatenate(oof_model_parts)
    alpha, alpha_grid = v279.choose_joint_alpha(
        oof_base, oof_model, truth_sh[oof_ids], truth_d[oof_ids]
    )
    fold_delta: list[dict[str, float]] = []
    for row, ids, base_sh, model_sh in zip(
        outer_rows, oof_ids_parts, oof_base_parts, oof_model_parts
    ):
        blended = v279.normalize(base_sh + alpha * (model_sh - base_sh))
        metric = v279.metrics(blended, truth_sh[ids], truth_d[ids])
        delta = {
            key: metric[key] - row["baseline"][key]
            for key in ("sh", "d", "weighted")
        }
        row["blended"] = metric
        row["blended_delta"] = delta
        fold_delta.append(delta)
    mean_delta = {
        key: float(np.mean([value[key] for value in fold_delta]))
        for key in ("sh", "d", "weighted")
    }
    weighted_wins = int(np.sum([value["weighted"] > 0.0 for value in fold_delta]))
    official_selected = bool(
        alpha > 0.0
        and mean_delta["weighted"] >= MIN_OFFICIAL_DELTA
        and weighted_wins >= 3
        and mean_delta["sh"] > 0.0
        and mean_delta["d"] > 0.0
    )

    final_training: list[dict[str, object]] = []
    projection_diagnostics = None
    local_score = None
    test_move = None
    if official_selected:
        positive_epochs = [value for value in selected_epochs if value > 0]
        final_epoch = max(int(np.median(positive_epochs)), 1)
        mean, std = node_stats(train_node, valid_ids)
        final_train = build_transport_set(
            valid_ids,
            valid_ids,
            train_pos,
            train_pos,
            train_node,
            train_node,
            train_geometry,
            train_geometry,
            truth_sh,
            True,
        ).standardize_nodes(mean, std)
        test_ids = np.arange(len(test_pos), dtype=np.int64)
        final_test = build_transport_set(
            test_ids,
            valid_ids,
            test_pos,
            train_pos,
            test_node,
            train_node,
            test_geometry,
            train_geometry,
            truth_sh,
            False,
        ).standardize_nodes(mean, std)
        test_parts: list[np.ndarray] = []
        move_parts: list[np.ndarray] = []
        for seed in FINAL_SEEDS:
            model, training = train_model(
                final_train,
                truth_sh,
                truth_d,
                seed,
                final_epoch,
                device,
                len(patches),
                None,
            )
            test_model, move = predict(model, final_test, device)
            test_parts.append(test_model)
            move_parts.append(move)
            checkpoint = CHECKPOINT_DIR / f"seed_{seed}.pt"
            torch.save(
                {
                    "version": VERSION,
                    "seed": seed,
                    "epoch": final_epoch,
                    "model_state": model.state_dict(),
                    "node_mean": mean,
                    "node_std": std,
                },
                checkpoint,
            )
            final_training.append(
                {**training, "checkpoint": PATHS.relative(checkpoint)}
            )
        test_model = v279.normalize(np.mean(test_parts, axis=0))
        test_move = np.mean(move_parts, axis=0).astype(np.float32)
        current_sh, _ = v274.extract_current_marginals(BASE)
        residual = test_model - final_test.anchor
        target_sh = v279.normalize(current_sh + alpha * residual)
        target_d = v279.derive_d_numpy(target_sh)
        np.savez_compressed(
            TARGETS_OUT,
            anchor_sh=final_test.anchor,
            model_sh=test_model,
            current_sh=current_sh,
            target_sh=target_sh,
            target_d=target_d,
            residual=residual,
            alpha=np.asarray(alpha, dtype=np.float32),
            move=test_move,
            shared_path_count=final_test.shared_path_count,
        )
        projection_diagnostics = v247.project(target_sh, target_d, OUTPUT)
        local_score = v208.score_prediction(OUTPUT)

    report = {
        "method": VERSION,
        "status": (
            "local_gate_passed_needs_packaging"
            if local_score is not None
            and float(local_score["score_traceback_leaked"]) >= LOCAL_GATE
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "Each angle-delay component follows its own planar wall-path family. The query-minus-donor "
            "change in path length and departure angle should transport donor peaks without the "
            "component mixing caused by shared dense spatial fields."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H S-H/PDP labels",
                "official train/test planar coordinates",
                "official point-cloud full wall reflection families",
                "strict v239 official-only carrier after official selection",
                "PyTorch path-option network trained only on official train folds",
            ],
            "does_not_use": [
                "synthetic/Sionna channel labels",
                "v212/v217 teacher output",
                "hidden labels or RT H/path coefficients",
                "local replay truth for training, epoch, alpha, or architecture selection",
            ],
            "local_truth_usage": (
                "one final gate score after nested official selection"
                if official_selected
                else "not read"
            ),
        },
        "parameters": {
            "neighbors": NEIGHBORS,
            "peaks_per_neighbor": PEAKS,
            "path_options_per_peak": PATH_OPTIONS,
            "epochs": EPOCHS,
            "outer_seeds": list(OUTER_SEEDS),
            "outer_count": OUTER_COUNT,
            "inner_count": INNER_COUNT,
            "final_seeds": list(FINAL_SEEDS),
            "minimum_official_delta": MIN_OFFICIAL_DELTA,
        },
        "path_center_spec": best_spec,
        "outer_folds": outer_rows,
        "official_selection": {
            "residual_alpha": alpha,
            "alpha_grid": alpha_grid,
            "mean_delta": mean_delta,
            "weighted_wins": weighted_wins,
            "selected_epochs": selected_epochs,
            "selected": official_selected,
        },
        "final_training": final_training,
        "test_move": summarize(test_move) if test_move is not None else None,
        "projection_diagnostics": (
            {key: summarize(value) for key, value in projection_diagnostics.items()}
            if projection_diagnostics is not None
            else None
        ),
        "local_gate": (
            {
                "required": LOCAL_GATE,
                "score": local_score,
                "passed": bool(
                    float(local_score["score_traceback_leaked"]) >= LOCAL_GATE
                ),
                "gap": float(local_score["score_traceback_leaked"]) - LOCAL_GATE,
            }
            if local_score is not None
            else None
        ),
        "outputs": {
            "targets": PATHS.relative(TARGETS_OUT) if TARGETS_OUT.is_file() else None,
            "prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None,
            "report": PATHS.relative(REPORT),
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
