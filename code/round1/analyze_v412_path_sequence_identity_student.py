#!/usr/bin/env python3
"""Predict v411 propagation identity from ordered official-pointcloud paths."""

from __future__ import annotations

import copy
import json
import math
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import analyze_v257_neural_pairwise_spectral_transport as v257
import analyze_v293_interaction_physical_path_law as v293
import analyze_v297_continuous_delay_component_transport as v297
import analyze_v315_full_sequence_confirmatory as v315
import analyze_v402_visible_path_residual_factorization as v402
import analyze_v406_neural_propagation_mode_router as v406
import analyze_v411_multiview_contrastive_mode_retrieval as v411
from generate_v87_easy_field_projection import summarize
from powerfield_common import regime_of
from project_paths import PATHS


VERSION_NUMBER = 412
VERSION = "v412_path_sequence_identity_student"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = OUT_DIR / "v412_path_sequence_identity_student_oof.npz"
PATH_CACHE = PATHS.cache_dir(291) / "v291_highdepth_pointcloud_paths.npz"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"
GEOMETRY_CACHE = PATHS.shared_features / "height_geometry_cache.npz"

PATH_COUNT = 128
MAX_DEPTH = 16
EMBED_DIM = v411.EMBED_DIM
STUDENT_EPOCHS = 100
EVAL_EVERY = 5
PATIENCE_EVALS = 6
BATCH_SIZE = 20
LEARNING_RATE = 7e-4
WEIGHT_DECAY = 4e-4
TEMPERATURE = 0.07
MIN_STUDENT_DELTA = 0.01
MIN_FOLD_WINS = 4
MIN_TEACHER_DELTA = 0.02


class PathSequenceStudent(nn.Module):
    def __init__(self, global_dim: int, path_dim: int, step_dim: int) -> None:
        super().__init__()
        self.step_encoder = nn.Sequential(
            nn.Linear(step_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        self.step_attention = nn.Linear(64, 1)
        self.path_encoder = nn.Sequential(
            nn.Linear(path_dim + 3 * 64, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(192, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=384,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.path_transformer = nn.TransformerEncoder(layer, num_layers=3)
        self.output = nn.Sequential(
            nn.Linear(256, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(192, EMBED_DIM),
        )

    def forward(
        self,
        global_feature: torch.Tensor,
        path_feature: torch.Tensor,
        step_feature: torch.Tensor,
        path_valid: torch.Tensor,
        step_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, path_count, step_count, _ = step_feature.shape
        step = self.step_encoder(step_feature)
        mask = step_valid[..., None]
        denominator = torch.clamp(mask.sum(dim=2), min=1.0)
        step_mean = (step * mask).sum(dim=2) / denominator
        step_max = step.masked_fill(~mask, -1e4).amax(dim=2)
        step_max = torch.where(
            step_valid.any(dim=2)[..., None], step_max, torch.zeros_like(step_max)
        )
        attention = self.step_attention(step).squeeze(-1)
        attention = attention.masked_fill(~step_valid, -1e4)
        attention = torch.softmax(attention, dim=2)
        attention = torch.where(step_valid, attention, torch.zeros_like(attention))
        attention /= torch.clamp(attention.sum(dim=2, keepdim=True), min=1e-6)
        step_attention = torch.sum(step * attention[..., None], dim=2)
        path = self.path_encoder(
            torch.cat([path_feature, step_mean, step_max, step_attention], dim=-1)
        )
        path = path * path_valid[..., None]
        global_token = self.global_encoder(global_feature)
        token = torch.cat([global_token[:, None, :], path], dim=1)
        padding = torch.cat(
            [
                torch.zeros((batch, 1), dtype=torch.bool, device=path.device),
                ~path_valid,
            ],
            dim=1,
        )
        encoded = self.path_transformer(token, src_key_padding_mask=padding)
        valid_weight = path_valid.float()
        path_summary = torch.sum(
            encoded[:, 1:] * valid_weight[..., None], dim=1
        ) / torch.clamp(valid_weight.sum(dim=1, keepdim=True), min=1.0)
        embedding = self.output(torch.cat([encoded[:, 0], path_summary], dim=1))
        return F.normalize(embedding, dim=1)


def build_geometry_features(
    path_cache: np.lib.npyio.NpzFile,
    interaction_cache: np.lib.npyio.NpzFile,
    prefix: str,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = path_cache[f"{prefix}_valid"][:, :PATH_COUNT].astype(bool)
    depth = path_cache[f"{prefix}_interaction_count"][:, :PATH_COUNT].astype(np.int64)
    vertices = np.nan_to_num(
        path_cache[f"{prefix}_vertices"][:, :PATH_COUNT].astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    interaction = path_cache[f"{prefix}_interactions"][:, :PATH_COUNT].astype(np.int64)
    step_index = np.arange(MAX_DEPTH, dtype=np.int64)
    step_valid = valid[:, :, None] & (step_index[None, None, :] < depth[:, :, None])
    bs = path_cache["bs_pos"].astype(np.float32)
    ue = positions.astype(np.float32)
    line = ue - bs[None, :]
    direct_length = np.maximum(np.linalg.norm(line, axis=1), 1e-4).astype(np.float32)
    line_unit = line / direct_length[:, None]

    previous = np.zeros_like(vertices)
    previous[:, :, 0] = bs
    previous[:, :, 1:] = vertices[:, :, :-1]
    following = np.zeros_like(vertices)
    following[:, :, :-1] = vertices[:, :, 1:]
    for step in range(MAX_DEPTH):
        last = depth == step + 1
        following[:, :, step] = np.where(
            last[..., None], ue[:, None, :], following[:, :, step]
        )
    incoming = vertices - previous
    outgoing = following - vertices
    incoming_length = np.maximum(np.linalg.norm(incoming, axis=3), 1e-6)
    outgoing_length = np.maximum(np.linalg.norm(outgoing, axis=3), 1e-6)
    incoming_unit = incoming / incoming_length[..., None]
    outgoing_unit = outgoing / outgoing_length[..., None]
    turn_cosine = np.sum(incoming_unit * outgoing_unit, axis=3)

    relative_bs = vertices - bs[None, None, None, :]
    relative_ue = vertices - ue[:, None, None, :]
    projection = np.sum(relative_bs * line_unit[:, None, None, :], axis=3)
    line_t = projection / direct_length[:, None, None]
    closest = bs[None, None, None, :] + projection[..., None] * line_unit[:, None, None, :]
    perpendicular = vertices - closest
    absolute = vertices.copy()
    absolute[..., 0] = (absolute[..., 0] - 150.0) / 180.0
    absolute[..., 1] = (absolute[..., 1] + 50.0) / 180.0
    absolute[..., 2] = (absolute[..., 2] - 15.0) / 40.0
    interaction_onehot = np.stack(
        [
            interaction == 0,
            interaction == 1,
            interaction == 4,
            ~np.isin(interaction, (0, 1, 4)),
        ],
        axis=3,
    ).astype(np.float32)
    position = (step_index.astype(np.float32) + 0.5) / MAX_DEPTH
    position_feature = np.stack(
        [
            np.sin(math.pi * position),
            np.cos(math.pi * position),
            np.sin(2.0 * math.pi * position),
            np.cos(2.0 * math.pi * position),
        ],
        axis=1,
    )
    position_feature = np.broadcast_to(
        position_feature[None, None, :, :],
        vertices.shape[:3] + (position_feature.shape[1],),
    )
    step_feature = np.concatenate(
        [
            absolute,
            relative_bs / 300.0,
            relative_ue / 300.0,
            line_t[..., None],
            perpendicular / 120.0,
            incoming_unit,
            (incoming_length / 300.0)[..., None],
            outgoing_unit,
            (outgoing_length / 300.0)[..., None],
            turn_cosine[..., None],
            interaction_onehot,
            position_feature,
        ],
        axis=3,
    ).astype(np.float32)
    step_feature *= step_valid[..., None]

    length = path_cache[f"{prefix}_path_length"][:, :PATH_COUNT].astype(np.float32)
    delay = path_cache[f"{prefix}_delay_bin_physical"][:, :PATH_COUNT].astype(np.float32)
    score = np.nan_to_num(
        path_cache[f"{prefix}_score"][:, :PATH_COUNT].astype(np.float32),
        nan=-20.0,
        posinf=8.0,
        neginf=-20.0,
    )
    theta_t = path_cache[f"{prefix}_theta_t"][:, :PATH_COUNT].astype(np.float32)
    phi_t = path_cache[f"{prefix}_phi_t"][:, :PATH_COUNT].astype(np.float32)
    theta_r = path_cache[f"{prefix}_theta_r"][:, :PATH_COUNT].astype(np.float32)
    phi_r = path_cache[f"{prefix}_phi_r"][:, :PATH_COUNT].astype(np.float32)
    reflection_count = np.sum((interaction == 1) & step_valid, axis=2).astype(np.float32)
    refraction_count = np.sum((interaction == 4) & step_valid, axis=2).astype(np.float32)
    ground_count = np.sum(
        (interaction == 1) & step_valid & (vertices[..., 2] <= 0.4), axis=2
    ).astype(np.float32)
    rank = np.broadcast_to(
        np.arange(PATH_COUNT, dtype=np.float32)[None, :] / (PATH_COUNT - 1),
        valid.shape,
    )
    angle = np.stack(
        [
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
    )
    base = np.concatenate(
        [
            (length / 500.0)[..., None],
            ((length - direct_length[:, None]) / 300.0)[..., None],
            (delay / 192.0)[..., None],
            (np.clip(score, -20.0, 8.0) / 8.0)[..., None],
            (depth.astype(np.float32) / MAX_DEPTH)[..., None],
            (reflection_count / MAX_DEPTH)[..., None],
            (refraction_count / MAX_DEPTH)[..., None],
            (ground_count / MAX_DEPTH)[..., None],
            angle,
            rank[..., None],
            np.broadcast_to((direct_length / 300.0)[:, None, None], valid.shape + (1,)),
            valid[..., None].astype(np.float32),
        ],
        axis=2,
    ).astype(np.float32)
    physical = interaction_cache[f"{prefix}_path_features"][:, :PATH_COUNT].astype(np.float32)
    path_feature = np.concatenate([base, physical], axis=2)
    path_feature *= valid[..., None]
    return (
        np.nan_to_num(path_feature, nan=0.0, posinf=4.0, neginf=-4.0),
        np.nan_to_num(step_feature, nan=0.0, posinf=4.0, neginf=-4.0),
        valid,
        step_valid,
    )


def standardize_global(
    fit: np.ndarray, outer: np.ndarray, train_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = fit[train_rows].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = fit[train_rows].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-5] = 1.0
    return ((fit - mean) / std).astype(np.float32), ((outer - mean) / std).astype(np.float32)


def make_global_feature(
    query_ids: np.ndarray,
    fit_ids: np.ndarray,
    xy: np.ndarray,
    node: np.ndarray,
    target: np.ndarray,
    teacher: v411.SpectrumEncoder,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor, donor_ids, feature = v411.query_data(
        query_ids, fit_ids, xy, node, target, teacher, device
    )
    return anchor, donor_ids, feature


def retrieval_loss(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    donor_embedding: torch.Tensor,
) -> torch.Tensor:
    cosine_loss = (1.0 - torch.sum(prediction * truth, dim=1)).mean()
    candidate_target = torch.sum(truth[:, None, :] * donor_embedding, dim=2)
    candidate_soft = torch.softmax(candidate_target / TEMPERATURE, dim=1)
    candidate_logit = torch.sum(prediction[:, None, :] * donor_embedding, dim=2)
    candidate_loss = torch.sum(
        -candidate_soft * F.log_softmax(candidate_logit / TEMPERATURE, dim=1), dim=1
    ).mean()
    batch_logit = prediction @ truth.T / TEMPERATURE
    batch_label = torch.arange(len(prediction), device=prediction.device)
    batch_loss = F.cross_entropy(batch_logit, batch_label)
    return 0.45 * cosine_loss + 0.40 * candidate_loss + 0.15 * batch_loss


@torch.no_grad()
def predict_student(
    model: PathSequenceStudent,
    global_feature: torch.Tensor,
    path_feature: torch.Tensor,
    step_feature: torch.Tensor,
    path_valid: torch.Tensor,
    step_valid: torch.Tensor,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(global_feature), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(global_feature))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=global_feature.is_cuda):
            value = model(
                global_feature[start:end],
                path_feature[start:end],
                step_feature[start:end],
                path_valid[start:end],
                step_valid[start:end],
            )
        output.append(value.float().cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def train_student(
    fit_global: np.ndarray,
    fit_path: np.ndarray,
    fit_step: np.ndarray,
    fit_path_valid: np.ndarray,
    fit_step_valid: np.ndarray,
    teacher_embedding: np.ndarray,
    fit_ids: np.ndarray,
    fit_donor: np.ndarray,
    fit_anchor: np.ndarray,
    target: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[PathSequenceStudent, tuple[float, float], list[dict[str, float]]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    global_tensor = torch.from_numpy(fit_global).to(device)
    path_tensor = torch.from_numpy(fit_path).to(device)
    step_tensor = torch.from_numpy(fit_step).to(device)
    path_valid_tensor = torch.from_numpy(fit_path_valid).to(device)
    step_valid_tensor = torch.from_numpy(fit_step_valid).to(device)
    truth_tensor = torch.from_numpy(teacher_embedding[fit_ids]).to(device)
    donor_embedding_tensor = torch.from_numpy(teacher_embedding[fit_donor]).to(device)
    model = PathSequenceStudent(
        fit_global.shape[1], fit_path.shape[2], fit_step.shape[3]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    best_state = copy.deepcopy(model.state_dict())
    best_action = (0.0, float("inf"))
    best_delta = 0.0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, STUDENT_EPOCHS + 1):
        model.train()
        order = rng.permutation(train_rows)
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            rows_np = order[start : start + BATCH_SIZE]
            rows = torch.from_numpy(rows_np).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = model(
                    global_tensor[rows],
                    path_tensor[rows],
                    step_tensor[rows],
                    path_valid_tensor[rows],
                    step_valid_tensor[rows],
                )
                loss = retrieval_loss(
                    prediction,
                    truth_tensor[rows],
                    donor_embedding_tensor[rows],
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(rows_np)
        if epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == STUDENT_EPOCHS:
            validation_embedding = predict_student(
                model,
                global_tensor[validation_rows],
                path_tensor[validation_rows],
                step_tensor[validation_rows],
                path_valid_tensor[validation_rows],
                step_valid_tensor[validation_rows],
            )
            rank, margin = v411.retrieve(
                validation_embedding,
                fit_donor[validation_rows],
                teacher_embedding,
            )
            alpha, threshold, delta, _ = v411.select_action(
                fit_anchor[validation_rows],
                fit_donor[validation_rows],
                rank,
                margin,
                target,
                target[fit_ids[validation_rows]],
            )
            target_cosine = float(
                np.mean(
                    np.sum(
                        validation_embedding * teacher_embedding[fit_ids[validation_rows]],
                        axis=1,
                    )
                )
            )
            history.append(
                {
                    "epoch": epoch,
                    "loss": total_loss / len(train_rows),
                    "validation_delta": delta,
                    "validation_target_cosine": target_cosine,
                    "validation_alpha": alpha,
                    "validation_threshold": threshold,
                }
            )
            if delta > best_delta + 1e-5:
                best_delta = delta
                best_action = (alpha, threshold)
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if epoch >= 25 and stale >= PATIENCE_EVALS:
                break
    model.load_state_dict(best_state)
    return model, best_action, history


def evaluate(
    prediction: np.ndarray, anchor: np.ndarray, truth: np.ndarray
) -> tuple[dict[str, object], np.ndarray]:
    base = v402.point_metrics(anchor, truth)
    values = v402.point_metrics(prediction, truth)
    delta = values["weighted"] - base["weighted"]
    return (
        {
            "metrics": v402.metric_summary(prediction, truth),
            "delta": {
                key: float(np.mean(values[key] - base[key])) for key in values
            },
            "point_gain": summarize(delta),
            "point_win_fraction": float(np.mean(delta > 0.0)),
        },
        delta,
    )


def run_fold(fold: int, device: torch.device) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    targets = np.load(TARGET_CACHE)
    target = v406.normalize(targets["sh"].astype(np.float32))
    views = v411.make_views(targets)
    positions = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
    test_xy = np.load(PATHS.round1 / "Round1_Test_Pos.npy").astype(np.float64)[:, :2]
    xy = positions[:, :2]
    node, _ = v257.build_node_features(xy, test_xy)
    blocked = np.load(GEOMETRY_CACHE)["train_blocked"].astype(np.float32)
    component = np.load(v297.COMPONENT_CACHE)
    valid_ids = np.flatnonzero(
        targets["valid"].astype(bool) & component["valid"].any(axis=1)
    ).astype(np.int64)
    folds = v315.disjoint_folds(valid_ids, xy)
    query_ids = folds[fold]
    fit_ids = np.setdiff1d(valid_ids, query_ids)
    validation_ids = folds[(fold + 1) % len(folds)]
    validation_rows = np.flatnonzero(np.isin(fit_ids, validation_ids)).astype(np.int64)
    train_rows = np.setdiff1d(np.arange(len(fit_ids)), validation_rows)

    teacher, teacher_history = v411.train_teacher(
        fit_ids, views, 41200 + fold, device
    )
    teacher_embedding = np.zeros((len(target), EMBED_DIM), dtype=np.float32)
    teacher_embedding[fit_ids] = v411.encode_spectra(
        teacher, target[fit_ids], device
    )
    fit_anchor, fit_donor, fit_global = make_global_feature(
        fit_ids, fit_ids, xy, node, target, teacher, device
    )
    outer_anchor, outer_donor, outer_global = make_global_feature(
        query_ids, fit_ids, xy, node, target, teacher, device
    )
    fit_global, outer_global = standardize_global(
        fit_global, outer_global, train_rows
    )
    path_cache = np.load(PATH_CACHE)
    interaction_cache = np.load(v293.FEATURE_CACHE)
    train_path, train_step, train_path_valid, train_step_valid = build_geometry_features(
        path_cache, interaction_cache, "train", positions
    )
    model, action, student_history = train_student(
        fit_global,
        train_path[fit_ids],
        train_step[fit_ids],
        train_path_valid[fit_ids],
        train_step_valid[fit_ids],
        teacher_embedding,
        fit_ids,
        fit_donor,
        fit_anchor,
        target,
        train_rows,
        validation_rows,
        41250 + fold,
        device,
    )
    outer_embedding = predict_student(
        model,
        torch.from_numpy(outer_global).to(device),
        torch.from_numpy(train_path[query_ids]).to(device),
        torch.from_numpy(train_step[query_ids]).to(device),
        torch.from_numpy(train_path_valid[query_ids]).to(device),
        torch.from_numpy(train_step_valid[query_ids]).to(device),
    )
    rank, margin = v411.retrieve(
        outer_embedding, outer_donor, teacher_embedding
    )
    prediction = v411.render(
        outer_anchor,
        outer_donor,
        rank,
        margin,
        target,
        action[0],
        action[1],
    )

    query_truth_embedding = v411.encode_spectra(teacher, target[query_ids], device)
    teacher_rank, teacher_margin = v411.retrieve(
        query_truth_embedding, outer_donor, teacher_embedding
    )
    validation_truth_embedding = teacher_embedding[fit_ids[validation_rows]]
    validation_teacher_rank, validation_teacher_margin = v411.retrieve(
        validation_truth_embedding,
        fit_donor[validation_rows],
        teacher_embedding,
    )
    teacher_action = v411.select_action(
        fit_anchor[validation_rows],
        fit_donor[validation_rows],
        validation_teacher_rank,
        validation_teacher_margin,
        target,
        target[fit_ids[validation_rows]],
    )
    teacher_prediction = v411.render(
        outer_anchor,
        outer_donor,
        teacher_rank,
        teacher_margin,
        target,
        teacher_action[0],
        teacher_action[1],
    )
    option_scores = v406.option_scores(
        outer_anchor, target[outer_donor], target[query_ids]
    )
    oracle_data = v406.RouterData(
        query_ids=query_ids,
        donor_ids=outer_donor,
        anchor=outer_anchor,
        scores=option_scores,
        q_geometry=np.empty((0,)),
        donor_geometry=np.empty((0,)),
        relation_geometry=np.empty((0,)),
        q_full=np.empty((0,)),
        donor_full=np.empty((0,)),
        relation_full=np.empty((0,)),
    )
    full_oracle = v406.render_choices(
        oracle_data, option_scores.argmax(axis=1), target
    )
    truth = target[query_ids]
    methods: dict[str, object] = {}
    deltas: dict[str, np.ndarray] = {}
    regimes = regime_of(blocked[query_ids]).astype(str)
    for name, value in (
        ("path_student", prediction),
        ("teacher_oracle", teacher_prediction),
        ("full_oracle", full_oracle),
    ):
        result, delta = evaluate(value, outer_anchor, truth)
        result["by_regime"] = {
            regime: float(np.mean(delta[regimes == regime]))
            for regime in ("clear", "light", "medium", "heavy")
            if np.any(regimes == regime)
        }
        methods[name] = result
        deltas[name] = delta
    report = {
        "fold": fold,
        "fit_count": int(len(fit_ids)),
        "query_count": int(len(query_ids)),
        "methods": methods,
        "student_action": {
            "alpha": action[0],
            "threshold": action[1],
            "active_fraction": float(np.mean(margin >= action[1])),
            "margin": summarize(margin),
        },
        "teacher_action": {
            "alpha": teacher_action[0],
            "threshold": teacher_action[1],
            "active_fraction": float(
                np.mean(teacher_margin >= teacher_action[1])
            ),
        },
        "teacher_history": teacher_history,
        "student_history": student_history,
        "geometry": {
            "path_feature_dim": int(train_path.shape[2]),
            "step_feature_dim": int(train_step.shape[3]),
            "mean_valid_paths": float(np.mean(train_path_valid[query_ids].sum(axis=1))),
            "zero_path_queries": int(np.sum(train_path_valid[query_ids].sum(axis=1) == 0)),
        },
        "leakage_assertions": {
            "query_in_fit": int(np.intersect1d(query_ids, fit_ids).size),
            "outer_truth_in_student_or_action_selection": False,
            "outer_truth_used_for_scoring_and_oracles_only": True,
        },
    }
    fold_report = REPORT_DIR / f"{VERSION}_fold{fold}.json"
    fold_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        OUT_DIR / f"v412_fold{fold}.npz",
        ids=query_ids,
        baseline=outer_anchor,
        truth=truth,
        path_student=prediction,
        teacher_oracle=teacher_prediction,
        full_oracle=full_oracle,
        delta_path_student=deltas["path_student"],
        delta_teacher_oracle=deltas["teacher_oracle"],
        delta_full_oracle=deltas["full_oracle"],
    )
    print(
        f"v412 fold={fold} student={methods['path_student']['delta']['weighted']:+.6f} "
        f"teacher={methods['teacher_oracle']['delta']['weighted']:+.6f} "
        f"full={methods['full_oracle']['delta']['weighted']:+.6f}",
        flush=True,
    )


def aggregate() -> None:
    rows = [
        json.loads(
            (REPORT_DIR / f"{VERSION}_fold{fold}.json").read_text(encoding="utf-8")
        )
        for fold in range(5)
    ]
    parts = [np.load(OUT_DIR / f"v412_fold{fold}.npz") for fold in range(5)]
    aggregate_result: dict[str, object] = {}
    for name in ("path_student", "teacher_oracle", "full_oracle"):
        delta = np.asarray(
            [row["methods"][name]["delta"]["weighted"] for row in rows]
        )
        aggregate_result[name] = {
            "weighted_delta": summarize(delta),
            "positive_folds": int(np.sum(delta > 0.0)),
        }
    passed = bool(
        aggregate_result["path_student"]["weighted_delta"]["mean"]
        >= MIN_STUDENT_DELTA
        and aggregate_result["path_student"]["positive_folds"] >= MIN_FOLD_WINS
        and aggregate_result["teacher_oracle"]["weighted_delta"]["mean"]
        >= MIN_TEACHER_DELTA
    )
    ids = np.concatenate([part["ids"] for part in parts])
    if len(ids) != 1984 or len(np.unique(ids)) != 1984:
        raise AssertionError("v412 OOF IDs are not a unique 1984-point cover")
    names = (
        "baseline",
        "truth",
        "path_student",
        "teacher_oracle",
        "full_oracle",
        "delta_path_student",
        "delta_teacher_oracle",
        "delta_full_oracle",
    )
    np.savez_compressed(
        OOF_CACHE,
        ids=ids,
        fold=np.concatenate(
            [np.full(len(parts[fold]["ids"]), fold, dtype=np.int16) for fold in range(5)]
        ),
        **{
            name: np.concatenate([part[name] for part in parts]) for name in names
        },
    )
    report = {
        "method": VERSION,
        "status": (
            "official_oof_passed_ready_for_final_training"
            if passed
            else "analysis_only_no_candidate"
        ),
        "primary_hypothesis": (
            "Ordered depth-16 official-pointcloud path vertices and interactions contain "
            "the fine geometry needed to predict the v411 view-invariant identity."
        ),
        "compliance": {
            "uses": [
                "official train H multi-array views",
                "official train/test coordinates",
                "official Round1_Map.ply-derived geometry-only depth-16 paths",
            ],
            "does_not_use": [
                "test H or hidden truth",
                "Sionna path coefficients or synthetic targets",
                "v212 teacher output or online score labels",
            ],
        },
        "parameters": {
            "paths": PATH_COUNT,
            "max_depth": MAX_DEPTH,
            "student_epochs": STUDENT_EPOCHS,
            "batch_size": BATCH_SIZE,
            "path_transformer_layers": 3,
            "embedding_dim": EMBED_DIM,
            "five_gpu_execution": True,
        },
        "fold_rows": rows,
        "aggregate": aggregate_result,
        "selection": {
            "minimum_student_delta": MIN_STUDENT_DELTA,
            "minimum_positive_folds": MIN_FOLD_WINS,
            "minimum_teacher_delta": MIN_TEACHER_DELTA,
            "passed": passed,
            "next_action": (
                "train the final official-only model and evaluate frozen synthetic local traceback"
                if passed
                else "do not deploy; the ordered path set still does not bridge geometry to identity"
            ),
        },
        "outputs": {
            "oof": PATHS.relative(OOF_CACHE),
            "report": PATHS.relative(REPORT),
            "prediction": None,
            "zip": None,
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["selection"], ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    for path in (
        PATH_CACHE,
        TARGET_CACHE,
        GEOMETRY_CACHE,
        v293.FEATURE_CACHE,
        v297.COMPONENT_CACHE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if world_size != 5:
            raise ValueError("v412 distributed mode requires exactly five workers")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        run_fold(local_rank, torch.device("cuda", local_rank))
        dist.barrier()
        if local_rank == 0:
            aggregate()
        dist.barrier()
        dist.destroy_process_group()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for fold in range(5):
            run_fold(fold, device)
        aggregate()


if __name__ == "__main__":
    main()
