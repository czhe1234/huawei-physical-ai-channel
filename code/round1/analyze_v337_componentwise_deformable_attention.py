#!/usr/bin/env python3
"""Strict official-only component-wise deformable spectral attention."""

from __future__ import annotations

import json
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v208_strict_historical_replay as v208
import analyze_v247_delaunay_spectral_field as v247
import analyze_v254_route_topology_spectral_field as v254
import analyze_v274_planar_barrier_attention as v274
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 337
VERSION = "v337_componentwise_deformable_attention"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
TARGETS_OUT = OUT_DIR / "v337_componentwise_targets.npz"
OUTPUT = OUT_DIR / "v337_Round1_Test_Channel.npy"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"

OUTER_SEEDS = (337, 1337, 2337, 3337, 4337)
OUTER_COUNT = 320
INNER_COUNT = 240
NEIGHBORS = 16
EPOCHS = 35
BATCH_SIZE = 12
FINAL_SEEDS = (33701, 33731, 33761)
MIN_OFFICIAL_DELTA = 0.005
ALPHA_GRID = np.linspace(0.0, 1.5, 13, dtype=np.float32)
S_SIZE = 192
H_SIZE = 16
OFFSETS = tuple(
    (ds, dh) for ds in (-2, -1, 0, 1, 2) for dh in (-1, 0, 1)
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize(x: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    total = out.reshape(len(out), -1).sum(axis=1)
    valid = total > 1e-20
    out[valid] /= total[valid, None, None]
    out[~valid] = 0.0
    return out.astype(np.float32)


def derive_d_numpy(sh: np.ndarray) -> np.ndarray:
    s_marginal = np.asarray(sh, dtype=np.float32).sum(axis=2, dtype=np.float64)
    return normalize(s_marginal[:, (-np.arange(S_SIZE)) % S_SIZE, None])[:, :, 0]


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    y = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def metrics(pred_sh: np.ndarray, truth_sh: np.ndarray, truth_d: np.ndarray) -> dict[str, float]:
    pred_d = derive_d_numpy(pred_sh)
    sh = float(np.mean(row_cos(pred_sh, truth_sh)))
    d = float(np.mean(row_cos(pred_d, truth_d)))
    return {"sh": sh, "d": d, "weighted": 0.52 * sh + 0.48 * d}


def derive_d_torch(sh: torch.Tensor) -> torch.Tensor:
    marginal = sh.sum(dim=2)
    reverse = torch.remainder(-torch.arange(S_SIZE, device=sh.device), S_SIZE)
    out = marginal[:, reverse]
    return out / out.sum(dim=1, keepdim=True).clamp_min(1e-30)


def spectral_basis() -> np.ndarray:
    s = 2.0 * np.pi * np.arange(S_SIZE, dtype=np.float32) / S_SIZE
    h = 2.0 * np.pi * np.arange(H_SIZE, dtype=np.float32) / H_SIZE
    channels = [np.ones((S_SIZE, H_SIZE), dtype=np.float32)]
    for order in range(1, 5):
        channels.append(np.broadcast_to(np.sin(order * s)[:, None], (S_SIZE, H_SIZE)))
        channels.append(np.broadcast_to(np.cos(order * s)[:, None], (S_SIZE, H_SIZE)))
    for order in range(1, 4):
        channels.append(np.broadcast_to(np.sin(order * h)[None, :], (S_SIZE, H_SIZE)))
        channels.append(np.broadcast_to(np.cos(order * h)[None, :], (S_SIZE, H_SIZE)))
    channels.extend(
        [
            np.sin(s)[:, None] * np.sin(h)[None, :],
            np.sin(s)[:, None] * np.cos(h)[None, :],
            np.cos(s)[:, None] * np.sin(h)[None, :],
            np.cos(s)[:, None] * np.cos(h)[None, :],
        ]
    )
    return np.stack(channels).astype(np.float32)


def trim_set(data: v274.ModelSet) -> v274.ModelSet:
    return v274.ModelSet(
        query_ids=data.query_ids,
        query_node=data.query_node,
        neighbor_ids=data.neighbor_ids[:, :NEIGHBORS],
        edge=data.edge[:, :NEIGHBORS],
        base_weights=(
            data.base_weights[:, :NEIGHBORS]
            / np.maximum(data.base_weights[:, :NEIGHBORS].sum(axis=1, keepdims=True), 1e-20)
        ).astype(np.float32),
    )


def base_prediction(data: v274.ModelSet, sh: np.ndarray) -> np.ndarray:
    out = np.sum(
        data.base_weights[:, :, None, None] * sh[data.neighbor_ids],
        axis=1,
        dtype=np.float64,
    )
    return normalize(out)


class ComponentwiseAttention(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, basis: np.ndarray):
        super().__init__()
        latent = 96
        basis_dim = int(basis.shape[0])
        self.query_encoder = nn.Sequential(
            nn.Linear(node_dim, 192), nn.SiLU(), nn.LayerNorm(192), nn.Linear(192, 64)
        )
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(node_dim, 192), nn.SiLU(), nn.LayerNorm(192), nn.Linear(192, 64)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 96), nn.SiLU(), nn.LayerNorm(96), nn.Linear(96, 64)
        )
        self.trunk = nn.Sequential(
            nn.Linear(64 * 4, 192),
            nn.SiLU(),
            nn.Dropout(0.04),
            nn.Linear(192, latent),
            nn.SiLU(),
        )
        self.component_coeff = nn.Linear(latent, basis_dim)
        self.offset_delta = nn.Linear(latent, len(OFFSETS))
        self.mass_gate = nn.Linear(latent, len(OFFSETS))
        self.query_gate = nn.Sequential(
            nn.Linear(64 + latent, 96), nn.SiLU(), nn.Linear(96, 1)
        )
        nn.init.zeros_(self.component_coeff.weight)
        nn.init.zeros_(self.component_coeff.bias)
        nn.init.zeros_(self.offset_delta.weight)
        nn.init.zeros_(self.offset_delta.bias)
        nn.init.zeros_(self.mass_gate.weight)
        nn.init.zeros_(self.mass_gate.bias)
        nn.init.zeros_(self.query_gate[-1].weight)
        nn.init.constant_(self.query_gate[-1].bias, -0.7)
        self.register_buffer("basis", torch.from_numpy(basis))
        offset_prior = [-7.0 * abs(ds) - 6.0 * abs(dh) for ds, dh in OFFSETS]
        self.register_buffer("offset_prior", torch.tensor(offset_prior, dtype=torch.float32))

    def forward(
        self,
        query_node: torch.Tensor,
        neighbor_node: torch.Tensor,
        edge: torch.Tensor,
        base_weights: torch.Tensor,
        neighbor_sh: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        q = self.query_encoder(query_node)
        n = self.neighbor_encoder(neighbor_node)
        e = self.edge_encoder(edge)
        q_expanded = q[:, None, :].expand_as(n)
        hidden = self.trunk(torch.cat([q_expanded, n, q_expanded * n, e], dim=2))

        component_coeff = 2.25 * torch.tanh(self.component_coeff(hidden))
        component_logit = torch.einsum("bkc,csh->bksh", component_coeff, self.basis)
        offset_logit = self.offset_prior[None, None, :] + 3.0 * torch.tanh(
            self.offset_delta(hidden)
        )
        mass_gate = 0.9 * torch.tanh(self.mass_gate(hidden))

        candidates = torch.stack(
            [torch.roll(neighbor_sh, shifts=(ds, dh), dims=(2, 3)) for ds, dh in OFFSETS],
            dim=2,
        )
        log_mass = torch.log(candidates.clamp_min(1e-9))
        log_mass = torch.tanh((log_mass + 7.0) / 4.0)
        logits = (
            torch.log(base_weights.clamp_min(1e-12))[:, :, None, None, None]
            + offset_logit[:, :, :, None, None]
            + component_logit[:, :, None, :, :]
            + mass_gate[:, :, :, None, None] * log_mass
        )
        batch = len(query_node)
        weights = torch.softmax(logits.reshape(batch, -1, S_SIZE, H_SIZE), dim=1)
        proposal = torch.sum(
            weights * candidates.reshape(batch, -1, S_SIZE, H_SIZE), dim=1
        )
        anchor = torch.sum(base_weights[:, :, None, None] * neighbor_sh, dim=1)
        pooled = torch.sum(base_weights[:, :, None] * hidden, dim=1)
        gate = 0.9 * torch.sigmoid(self.query_gate(torch.cat([q, pooled], dim=1)))
        prediction = anchor + gate[:, :, None] * (proposal - anchor)
        prediction = prediction.clamp_min(0.0)
        prediction = prediction / prediction.flatten(1).sum(dim=1)[:, None, None].clamp_min(1e-30)
        return prediction, {
            "gate": gate[:, 0],
            "center_probability": weights[:, :: len(OFFSETS)].sum(dim=1).mean(dim=(1, 2)),
        }


def model_loss(pred: torch.Tensor, truth_sh: torch.Tensor, truth_d: torch.Tensor) -> torch.Tensor:
    pred_d = derive_d_torch(pred)
    sh_loss = 1.0 - F.cosine_similarity(pred.flatten(1), truth_sh.flatten(1), dim=1).mean()
    d_loss = 1.0 - F.cosine_similarity(pred_d, truth_d, dim=1).mean()
    sqrt_loss = 1.0 - F.cosine_similarity(
        torch.sqrt(pred.clamp_min(1e-12)).flatten(1),
        torch.sqrt(truth_sh.clamp_min(1e-12)).flatten(1),
        dim=1,
    ).mean()
    return 0.50 * sh_loss + 0.46 * d_loss + 0.04 * sqrt_loss


@torch.no_grad()
def predict_model(
    model: ComponentwiseAttention,
    data: v274.ModelSet,
    node_raw: np.ndarray,
    stats: v274.Standardizer,
    sh: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    node = np.clip((node_raw - stats.node_mean) / stats.node_std, -8.0, 8.0).astype(np.float32)
    pred = np.empty((len(data.query_ids), S_SIZE, H_SIZE), dtype=np.float32)
    gates = np.empty(len(data.query_ids), dtype=np.float32)
    centers = np.empty(len(data.query_ids), dtype=np.float32)
    for start in range(0, len(data.query_ids), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(data.query_ids))
        neighbor_ids = data.neighbor_ids[start:end]
        block, diagnostics = model(
            torch.from_numpy(data.query_node[start:end]).to(device),
            torch.from_numpy(node[neighbor_ids]).to(device),
            torch.from_numpy(data.edge[start:end]).to(device),
            torch.from_numpy(data.base_weights[start:end]).to(device),
            torch.from_numpy(sh[neighbor_ids]).to(device),
        )
        pred[start:end] = block.cpu().numpy()
        gates[start:end] = diagnostics["gate"].cpu().numpy()
        centers[start:end] = diagnostics["center_probability"].cpu().numpy()
    return pred, gates, centers


def train_model(
    train_data: v274.ModelSet,
    node_raw: np.ndarray,
    stats: v274.Standardizer,
    sh: np.ndarray,
    d: np.ndarray,
    seed: int,
    device: torch.device,
    epochs: int,
    validation: v274.ModelSet | None = None,
) -> tuple[ComponentwiseAttention, dict[str, object]]:
    set_seed(seed)
    basis = spectral_basis()
    model = ComponentwiseAttention(node_raw.shape[1], train_data.edge.shape[2], basis).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    node = np.clip((node_raw - stats.node_mean) / stats.node_std, -8.0, 8.0).astype(np.float32)
    rng = np.random.default_rng(seed)
    history: list[dict[str, object]] = []
    best_epoch = epochs
    best_metric = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    if validation is not None:
        initial, _, _ = predict_model(model, validation, node_raw, stats, sh, device)
        initial_metric = metrics(
            initial, sh[validation.query_ids], d[validation.query_ids]
        )
        best_metric = initial_metric["weighted"]
        best_epoch = 0
        best_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_data.query_ids))
        losses: list[float] = []
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start : start + BATCH_SIZE]
            neighbor_ids = train_data.neighbor_ids[rows]
            pred, _ = model(
                torch.from_numpy(train_data.query_node[rows]).to(device),
                torch.from_numpy(node[neighbor_ids]).to(device),
                torch.from_numpy(train_data.edge[rows]).to(device),
                torch.from_numpy(train_data.base_weights[rows]).to(device),
                torch.from_numpy(sh[neighbor_ids]).to(device),
            )
            truth_sh = torch.from_numpy(sh[train_data.query_ids[rows]]).to(device)
            truth_d = torch.from_numpy(d[train_data.query_ids[rows]]).to(device)
            loss = model_loss(pred, truth_sh, truth_d)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            item: dict[str, object] = {
                "epoch": int(epoch),
                "loss": float(np.mean(losses)),
            }
            if validation is not None:
                prediction, _, _ = predict_model(
                    model, validation, node_raw, stats, sh, device
                )
                validation_metric = metrics(
                    prediction,
                    sh[validation.query_ids],
                    d[validation.query_ids],
                )
                item["validation"] = validation_metric
                if validation_metric["weighted"] > best_metric:
                    best_metric = validation_metric["weighted"]
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            history.append(item)
            suffix = (
                f" val={item['validation']['weighted']:.6f}"
                if "validation" in item
                else ""
            )
            print(
                f"v337 seed={seed} epoch={epoch}/{epochs} "
                f"loss={item['loss']:.6f}{suffix}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "epochs": epochs,
        "best_epoch": int(best_epoch),
        "best_weighted": float(best_metric) if np.isfinite(best_metric) else None,
        "history": history,
    }


def build_set(
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    geometry: tuple,
    node_raw: np.ndarray,
) -> tuple[v274.ModelSet, v274.Standardizer]:
    (
        all_pos,
        signatures,
        blocker_count,
        _,
        footprint,
        z_max,
        x_min,
        y_min,
        resolution,
    ) = geometry
    raw_train = v274.build_raw_set(
        bank_ids,
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
    stats = v274.fit_standardizer(node_raw, bank_ids, raw_train)
    raw_query = v274.build_raw_set(
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
    return trim_set(v274.apply_standardizer(raw_query, node_raw, stats)), stats


def training_set(
    bank_ids: np.ndarray,
    geometry: tuple,
    node_raw: np.ndarray,
    stats: v274.Standardizer,
) -> v274.ModelSet:
    (
        all_pos,
        signatures,
        blocker_count,
        _,
        footprint,
        z_max,
        x_min,
        y_min,
        resolution,
    ) = geometry
    raw = v274.build_raw_set(
        bank_ids,
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
    return trim_set(v274.apply_standardizer(raw, node_raw, stats))


def choose_alpha(base: np.ndarray, model: np.ndarray, truth_sh: np.ndarray, truth_d: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for alpha in ALPHA_GRID:
        pred = normalize(base + float(alpha) * (model - base))
        rows.append({"alpha": float(alpha), **metrics(pred, truth_sh, truth_d)})
    best = max(rows, key=lambda row: row["weighted"])
    return float(best["alpha"]), rows


def main() -> None:
    for path in (BASE, TARGET_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    target_data = np.load(TARGET_CACHE)
    sh = normalize(target_data["sh"].astype(np.float32))
    d = target_data["d"].astype(np.float32)
    d /= np.maximum(d.sum(axis=1, keepdims=True), 1e-20)
    valid_ids = np.flatnonzero(target_data["valid"].astype(bool)).astype(np.int64)
    geometry = v274.load_geometry()
    all_pos, signatures, blocker_count, scene_features = geometry[:4]
    train_pos = all_pos[: len(sh)]
    test_pos = all_pos[len(sh) :]
    node_raw = v274.build_node_features(all_pos, signatures, blocker_count, scene_features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"v337 device={device} valid={len(valid_ids)}", flush=True)

    fold_rows: list[dict[str, object]] = []
    ids_parts: list[np.ndarray] = []
    base_parts: list[np.ndarray] = []
    model_parts: list[np.ndarray] = []
    selected_epochs: list[int] = []
    for fold, seed in enumerate(OUTER_SEEDS):
        outer_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(train_pos[:, :2], test_pos[:, :2], OUTER_COUNT, seed),
            valid_ids,
        )
        fit_ids = np.setdiff1d(valid_ids, outer_ids)
        inner_ids = np.intersect1d(
            v254.choose_seeded_matched_indices(
                train_pos[:, :2], test_pos[:, :2], INNER_COUNT, seed + 50_000
            ),
            fit_ids,
        )
        inner_fit = np.setdiff1d(fit_ids, inner_ids)
        inner_val, inner_stats = build_set(inner_ids, inner_fit, geometry, node_raw)
        inner_train = training_set(inner_fit, geometry, node_raw, inner_stats)
        _, inner_training = train_model(
            inner_train,
            node_raw,
            inner_stats,
            sh,
            d,
            seed,
            device,
            EPOCHS,
            validation=inner_val,
        )
        selected_epoch = int(inner_training["best_epoch"])
        selected_epochs.append(selected_epoch)
        outer_set, stats = build_set(outer_ids, fit_ids, geometry, node_raw)
        train_set = training_set(fit_ids, geometry, node_raw, stats)
        base_sh = base_prediction(outer_set, sh)
        if selected_epoch == 0:
            model_sh = base_sh.copy()
            gates = np.zeros(len(outer_ids), dtype=np.float32)
            centers = np.ones(len(outer_ids), dtype=np.float32)
            outer_training: dict[str, object] = {
                "seed": seed + 100_000,
                "epochs": 0,
                "best_epoch": 0,
                "history": [],
            }
        else:
            model, outer_training = train_model(
                train_set,
                node_raw,
                stats,
                sh,
                d,
                seed + 100_000,
                device,
                selected_epoch,
            )
            model_sh, gates, centers = predict_model(
                model, outer_set, node_raw, stats, sh, device
            )
        base_metric = metrics(base_sh, sh[outer_ids], d[outer_ids])
        model_metric = metrics(model_sh, sh[outer_ids], d[outer_ids])
        delta = {key: model_metric[key] - base_metric[key] for key in ("sh", "d", "weighted")}
        fold_rows.append(
            {
                "fold": fold,
                "seed": seed,
                "fit_count": int(len(fit_ids)),
                "validation_count": int(len(outer_ids)),
                "selected_epoch": selected_epoch,
                "inner_training": inner_training,
                "outer_training": outer_training,
                "base": base_metric,
                "model": model_metric,
                "raw_delta": delta,
                "gate": summarize(gates),
                "center_probability": summarize(centers),
            }
        )
        ids_parts.append(outer_ids)
        base_parts.append(base_sh)
        model_parts.append(model_sh)
        print(
            f"v337 fold={fold + 1}/5 epoch={selected_epoch} "
            f"base={base_metric['weighted']:.6f} "
            f"model={model_metric['weighted']:.6f} delta={delta['weighted']:+.6f}",
            flush=True,
        )

    oof_ids = np.concatenate(ids_parts)
    oof_base = np.concatenate(base_parts)
    oof_model = np.concatenate(model_parts)
    alpha, alpha_rows = choose_alpha(oof_base, oof_model, sh[oof_ids], d[oof_ids])
    blended_deltas = []
    for row, ids, base_sh, model_sh in zip(fold_rows, ids_parts, base_parts, model_parts):
        blended = normalize(base_sh + alpha * (model_sh - base_sh))
        blended_metric = metrics(blended, sh[ids], d[ids])
        blended_delta = {key: blended_metric[key] - row["base"][key] for key in ("sh", "d", "weighted")}
        row["blended"] = blended_metric
        row["blended_delta"] = blended_delta
        blended_deltas.append(blended_delta)
    mean_delta = {
        key: float(np.mean([value[key] for value in blended_deltas]))
        for key in ("sh", "d", "weighted")
    }
    wins = int(np.sum([value["weighted"] > 0.0 for value in blended_deltas]))
    selected = bool(
        alpha > 0.0
        and mean_delta["weighted"] >= MIN_OFFICIAL_DELTA
        and mean_delta["sh"] > 0.0
        and mean_delta["d"] > 0.0
        and wins >= 4
    )

    local_score = None
    final_rows: list[dict[str, object]] = []
    projection = None
    if selected:
        test_ids = np.arange(len(sh), len(all_pos), dtype=np.int64)
        test_set, stats = build_set(test_ids, valid_ids, geometry, node_raw)
        train_set = training_set(valid_ids, geometry, node_raw, stats)
        test_base = base_prediction(test_set, sh)
        predictions = []
        positive_epochs = [value for value in selected_epochs if value > 0]
        final_epoch = max(int(np.median(positive_epochs)), 1)
        for seed in FINAL_SEEDS:
            model, training = train_model(
                train_set,
                node_raw,
                stats,
                sh,
                d,
                seed,
                device,
                final_epoch,
            )
            pred, gates, centers = predict_model(model, test_set, node_raw, stats, sh, device)
            predictions.append(pred)
            checkpoint = CHECKPOINT_DIR / f"seed_{seed}.pt"
            torch.save(
                {"version": VERSION, "seed": seed, "epochs": final_epoch, "model_state": model.state_dict()},
                checkpoint,
            )
            final_rows.append(
                {
                    "seed": seed,
                    "training": training,
                    "gate": summarize(gates),
                    "center_probability": summarize(centers),
                    "checkpoint": PATHS.relative(checkpoint),
                }
            )
        test_model = normalize(np.mean(predictions, axis=0))
        current_sh, _ = v274.extract_current_marginals(BASE)
        residual = test_model - test_base
        target_sh = normalize(current_sh + alpha * residual)
        target_d = derive_d_numpy(target_sh)
        np.savez_compressed(
            TARGETS_OUT,
            anchor_sh=test_base,
            model_sh=test_model,
            current_sh=current_sh,
            residual=residual,
            target_sh=target_sh,
            target_d=target_d,
            alpha=np.asarray(alpha, dtype=np.float32),
        )
        projection = v247.project(target_sh, target_d, OUTPUT)
        local_score = v208.score_prediction(OUTPUT)

    report = {
        "method": VERSION,
        "status": "analysis_only_no_candidate",
        "primary_hypothesis": (
            "Each delay-angle propagation component must select and spatially transport its own donor; "
            "a shared whole-spectrum KNN weight is the main absolute-spectrum bottleneck."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H S-H/PDP labels",
                "official train/test planar coordinates",
                "official point-cloud scene and barrier features",
                "strict official-only v239 carrier after official selection",
            ],
            "does_not_use": [
                "v211/v212 outputs or teacher labels",
                "synthetic test labels",
                "restored Etoile mesh/material/object data",
                "RT channel coefficients",
                "online-score optimization",
            ],
            "local_truth_usage": "one frozen diagnostic after official selection" if selected else "not read",
        },
        "parameters": {
            "outer_seeds": list(OUTER_SEEDS),
            "outer_count": OUTER_COUNT,
            "inner_count": INNER_COUNT,
            "neighbors": NEIGHBORS,
            "offsets": [list(value) for value in OFFSETS],
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "alpha_grid": ALPHA_GRID.tolist(),
            "minimum_official_delta": MIN_OFFICIAL_DELTA,
            "final_seeds": list(FINAL_SEEDS),
        },
        "outer_folds": fold_rows,
        "official_selection": {
            "alpha": alpha,
            "alpha_rows": alpha_rows,
            "mean_delta": mean_delta,
            "wins": wins,
            "selected": selected,
            "selected_epochs": selected_epochs,
        },
        "final_training": final_rows,
        "projection": (
            {key: summarize(value) for key, value in projection.items()} if projection is not None else None
        ),
        "local_gate": {
            "v211_reference": 0.6350959054536596,
            "v212_reference": 0.6456036433098431,
            "score": local_score,
        } if local_score is not None else None,
        "outputs": {
            "prediction": PATHS.relative(OUTPUT) if OUTPUT.is_file() else None,
            "targets": PATHS.relative(TARGETS_OUT) if TARGETS_OUT.is_file() else None,
            "report": PATHS.relative(REPORT),
        },
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["official_selection"], ensure_ascii=False, indent=2), flush=True)
    if local_score is not None:
        print(json.dumps(local_score, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
