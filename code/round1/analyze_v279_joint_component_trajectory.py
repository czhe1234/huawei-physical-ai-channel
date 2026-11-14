#!/usr/bin/env python3
"""Strict official-only joint delay-angle component trajectory model."""

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


VERSION_NUMBER = 279
VERSION = "v279_joint_component_trajectory"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
TARGETS_OUT = OUT_DIR / "v279_joint_component_targets.npz"
OUTPUT = OUT_DIR / "v279_Round1_Test_Channel.npy"
BASE = PATHS.cache_dir(239) / "v239_path_on_v237_Round1_Test_Channel.npy"
TARGET_CACHE = PATHS.cache_dir(218) / "v218_train_multispectrum_targets.npz"

LOCAL_GATE = 0.645604
OUTER_SEEDS = (279, 1279, 2279, 3279, 4279)
OUTER_COUNT = 320
INNER_COUNT = 240
PEAK_COUNT = 8
EPOCHS = 50
EVAL_EVERY = 5
BATCH_SIZE = 64
MIN_OFFICIAL_DELTA = 5e-4
FINAL_SEEDS = (27901, 27931, 27961)
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
    out[valid] /= total[valid].reshape(
        (int(np.sum(valid)),) + (1,) * (out.ndim - 1)
    )
    out[~valid] = 0.0
    return out.astype(np.float32)


def derive_d_numpy(sh: np.ndarray) -> np.ndarray:
    s_marginal = np.asarray(sh, dtype=np.float32).sum(axis=2, dtype=np.float64)
    return normalize(s_marginal[:, (-np.arange(S_SIZE)) % S_SIZE].astype(np.float32))


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    y = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return (
        np.sum(x * y, axis=1)
        / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
    ).astype(np.float32)


def metrics(
    pred_sh: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
) -> dict[str, float]:
    pred_d = derive_d_numpy(pred_sh)
    sh = float(np.mean(row_cos(pred_sh, truth_sh)))
    d = float(np.mean(row_cos(pred_d, truth_d)))
    return {"sh": sh, "d": d, "weighted": 0.52 * sh + 0.48 * d}


def top_peaks(sh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(sh, dtype=np.float32).reshape(len(sh), -1)
    indices = np.argpartition(flat, -PEAK_COUNT, axis=1)[:, -PEAK_COUNT:]
    values = np.take_along_axis(flat, indices, axis=1)
    order = np.argsort(values, axis=1)[:, ::-1]
    indices = np.take_along_axis(indices, order, axis=1).astype(np.int64)
    values = np.take_along_axis(values, order, axis=1).astype(np.float32)
    return indices, values


def derive_d_torch(flat_sh: torch.Tensor) -> torch.Tensor:
    sh = flat_sh.reshape(len(flat_sh), S_SIZE, H_SIZE)
    marginal = sh.sum(dim=2)
    reverse = torch.remainder(-torch.arange(S_SIZE, device=sh.device), S_SIZE)
    d = marginal[:, reverse]
    return d / d.sum(dim=1, keepdim=True).clamp_min(1e-30)


class ComponentTrajectoryNet(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        hidden = 96
        self.query_encoder = nn.Sequential(
            nn.Linear(node_dim, 192),
            nn.SiLU(),
            nn.LayerNorm(192),
            nn.Linear(192, hidden),
        )
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(node_dim, 192),
            nn.SiLU(),
            nn.LayerNorm(192),
            nn.Linear(192, hidden),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 96), nn.SiLU(), nn.Linear(96, 48)
        )
        self.peak_encoder = nn.Sequential(
            nn.Linear(12, 64), nn.SiLU(), nn.Linear(64, 48)
        )
        self.token_encoder = nn.Sequential(
            nn.Linear(hidden * 3 + 48 * 2, 192),
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
        self.offset_score = nn.Linear(hidden, len(OFFSETS))
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.SiLU(), nn.Linear(128, 1)
        )
        nn.init.constant_(self.gate[-1].bias, -1.4)

        offset_prior = [
            -0.95 * abs(ds) - 0.80 * abs(dh) for ds, dh in OFFSETS
        ]
        self.register_buffer(
            "offset_prior", torch.tensor(offset_prior, dtype=torch.float32)
        )
        self.register_buffer(
            "offset_ds",
            torch.tensor([value[0] for value in OFFSETS], dtype=torch.long),
        )
        self.register_buffer(
            "offset_dh",
            torch.tensor([value[1] for value in OFFSETS], dtype=torch.long),
        )

    def forward(
        self,
        query_node: torch.Tensor,
        neighbor_node: torch.Tensor,
        edge: torch.Tensor,
        base_weights: torch.Tensor,
        neighbor_sh: torch.Tensor,
        peak_indices: torch.Tensor,
        peak_mass: torch.Tensor,
        query_anchor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, neighbors, peaks = peak_indices.shape
        q = self.query_encoder(query_node)
        n = self.neighbor_encoder(neighbor_node)
        e = self.edge_encoder(edge)

        s_bin = torch.div(peak_indices, H_SIZE, rounding_mode="floor")
        h_bin = torch.remainder(peak_indices, H_SIZE)
        s_angle = 2.0 * math.pi * s_bin.float() / S_SIZE
        h_angle = 2.0 * math.pi * h_bin.float() / H_SIZE
        rank = torch.arange(peaks, device=peak_indices.device).float()
        rank = rank[None, None, :].expand(batch, neighbors, -1)
        top_mass = peak_mass[:, :, :1].clamp_min(1e-12)
        base = base_weights[:, :, None].expand(-1, -1, peaks)
        peak_feature = torch.stack(
            [
                torch.sin(s_angle),
                torch.cos(s_angle),
                torch.sin(2.0 * s_angle),
                torch.cos(2.0 * s_angle),
                torch.sin(h_angle),
                torch.cos(h_angle),
                torch.sin(2.0 * h_angle),
                torch.cos(2.0 * h_angle),
                torch.log(peak_mass.clamp_min(1e-12)) / 12.0,
                peak_mass / top_mass,
                rank / max(peaks - 1, 1),
                torch.log(base.clamp_min(1e-12)) / 8.0,
            ],
            dim=3,
        )
        p = self.peak_encoder(peak_feature)
        q_token = q[:, None, None, :].expand(-1, neighbors, peaks, -1)
        n_token = n[:, :, None, :].expand(-1, -1, peaks, -1)
        e_token = e[:, :, None, :].expand(-1, -1, peaks, -1)
        token = self.token_encoder(
            torch.cat([q_token, n_token, q_token * n_token, e_token, p], dim=3)
        ).reshape(batch, neighbors * peaks, -1)
        token = self.transformer(token)

        token_prior = torch.log(
            (base_weights[:, :, None] * peak_mass).reshape(batch, -1).clamp_min(1e-12)
        )
        token_delta = 2.25 * torch.tanh(self.token_score(token).squeeze(2))
        token_energy = torch.softmax(token_prior + token_delta, dim=1)
        offset_delta = 2.0 * torch.tanh(self.offset_score(token))
        offset_probability = torch.softmax(
            self.offset_prior[None, None, :] + offset_delta, dim=2
        )

        flat_indices = peak_indices.reshape(batch, -1)
        token_s = torch.div(flat_indices, H_SIZE, rounding_mode="floor")
        token_h = torch.remainder(flat_indices, H_SIZE)
        destination = (
            torch.remainder(token_s[:, :, None] + self.offset_ds, S_SIZE) * H_SIZE
            + torch.remainder(token_h[:, :, None] + self.offset_dh, H_SIZE)
        )
        sparse = torch.zeros(
            (batch, S_SIZE * H_SIZE),
            dtype=neighbor_sh.dtype,
            device=neighbor_sh.device,
        )
        sparse.scatter_add_(
            1,
            destination.reshape(batch, -1),
            (token_energy[:, :, None] * offset_probability).reshape(batch, -1),
        )
        if query_anchor is None:
            anchor = torch.sum(
                base_weights[:, :, None, None] * neighbor_sh, dim=1
            ).reshape(batch, -1)
        else:
            anchor = query_anchor.reshape(batch, -1)
        anchor = anchor / anchor.sum(dim=1, keepdim=True).clamp_min(1e-30)
        pooled = torch.sum(token_energy[:, :, None] * token, dim=1)
        gate = 0.72 * torch.sigmoid(self.gate(torch.cat([q, pooled], dim=1)))
        prediction = (1.0 - gate) * anchor + gate * sparse
        prediction = prediction / prediction.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return prediction, {
            "gate": gate.squeeze(1),
            "token_delta": token_delta,
            "offset_probability": offset_probability,
        }


def model_loss(
    prediction: torch.Tensor,
    truth_sh: torch.Tensor,
    truth_d: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
) -> torch.Tensor:
    pred_d = derive_d_torch(prediction)
    sh_loss = 1.0 - F.cosine_similarity(prediction, truth_sh, dim=1).mean()
    d_loss = 1.0 - F.cosine_similarity(pred_d, truth_d, dim=1).mean()
    offset_index = OFFSETS.index((0, 0))
    offset_zero = diagnostics["offset_probability"][:, :, offset_index]
    regularization = (
        2e-4 * diagnostics["token_delta"].square().mean()
        + 2e-4 * (1.0 - offset_zero).mean()
    )
    return 0.52 * sh_loss + 0.48 * d_loss + regularization


def standardized_nodes(
    node_raw: np.ndarray, stats: v274.Standardizer
) -> np.ndarray:
    return np.clip(
        (node_raw - stats.node_mean) / stats.node_std, -8.0, 8.0
    ).astype(np.float32)


@torch.no_grad()
def predict_model(
    model: ComponentTrajectoryNet,
    data: v274.ModelSet,
    node_raw: np.ndarray,
    stats: v274.Standardizer,
    sh: np.ndarray,
    peak_indices: np.ndarray,
    peak_mass: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    node = standardized_nodes(node_raw, stats)
    output = np.empty((len(data.query_ids), S_SIZE, H_SIZE), dtype=np.float32)
    gates = np.empty(len(data.query_ids), dtype=np.float32)
    for start in range(0, len(data.query_ids), BATCH_SIZE * 2):
        end = min(start + BATCH_SIZE * 2, len(data.query_ids))
        neighbor_ids = data.neighbor_ids[start:end]
        prediction, diagnostics = model(
            torch.from_numpy(data.query_node[start:end]).to(device),
            torch.from_numpy(node[neighbor_ids]).to(device),
            torch.from_numpy(data.edge[start:end]).to(device),
            torch.from_numpy(data.base_weights[start:end]).to(device),
            torch.from_numpy(sh[neighbor_ids]).to(device),
            torch.from_numpy(peak_indices[neighbor_ids]).to(device),
            torch.from_numpy(peak_mass[neighbor_ids]).to(device),
        )
        output[start:end] = prediction.reshape(-1, S_SIZE, H_SIZE).cpu().numpy()
        gates[start:end] = diagnostics["gate"].cpu().numpy()
    return normalize(output), gates


def base_prediction(data: v274.ModelSet, sh: np.ndarray) -> np.ndarray:
    output = np.sum(
        data.base_weights[:, :, None, None] * sh[data.neighbor_ids],
        axis=1,
        dtype=np.float64,
    ).astype(np.float32)
    return normalize(output)


def train_model(
    train_data: v274.ModelSet,
    validation: v274.ModelSet | None,
    node_raw: np.ndarray,
    stats: v274.Standardizer,
    sh: np.ndarray,
    d: np.ndarray,
    peak_indices: np.ndarray,
    peak_mass: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
) -> tuple[ComponentTrajectoryNet, dict[str, object]]:
    set_seed(seed)
    model = ComponentTrajectoryNet(node_raw.shape[1], train_data.edge.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=4e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    node = standardized_nodes(node_raw, stats)
    rng = np.random.default_rng(seed)
    baseline = None
    best_metric = -np.inf
    best_epoch = epochs
    best_state: dict[str, torch.Tensor] | None = None
    if validation is not None:
        baseline_sh = base_prediction(validation, sh)
        baseline = metrics(
            baseline_sh, sh[validation.query_ids], d[validation.query_ids]
        )
        best_metric = baseline["weighted"]
        best_epoch = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_data.query_ids))
        losses: list[float] = []
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start : start + BATCH_SIZE]
            neighbor_ids = train_data.neighbor_ids[rows]
            prediction, diagnostics = model(
                torch.from_numpy(train_data.query_node[rows]).to(device),
                torch.from_numpy(node[neighbor_ids]).to(device),
                torch.from_numpy(train_data.edge[rows]).to(device),
                torch.from_numpy(train_data.base_weights[rows]).to(device),
                torch.from_numpy(sh[neighbor_ids]).to(device),
                torch.from_numpy(peak_indices[neighbor_ids]).to(device),
                torch.from_numpy(peak_mass[neighbor_ids]).to(device),
            )
            loss = model_loss(
                prediction,
                torch.from_numpy(
                    sh[train_data.query_ids[rows]].reshape(len(rows), -1)
                ).to(device),
                torch.from_numpy(d[train_data.query_ids[rows]]).to(device),
                diagnostics,
            )
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
                pred_sh, gates = predict_model(
                    model,
                    validation,
                    node_raw,
                    stats,
                    sh,
                    peak_indices,
                    peak_mass,
                    device,
                )
                validation_metric = metrics(
                    pred_sh,
                    sh[validation.query_ids],
                    d[validation.query_ids],
                )
                item["validation"] = validation_metric
                item["gate"] = summarize(gates)
                item["baseline"] = baseline
                if validation_metric["weighted"] > best_metric:
                    best_metric = validation_metric["weighted"]
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            history.append(item)
        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"v279 seed={seed} epoch={epoch}/{epochs} loss={np.mean(losses):.6f}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_weighted": float(best_metric) if np.isfinite(best_metric) else None,
        "baseline": baseline,
        "history": history,
    }


def choose_joint_alpha(
    base: np.ndarray,
    model: np.ndarray,
    truth_sh: np.ndarray,
    truth_d: np.ndarray,
) -> tuple[float, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for alpha in ALPHA_GRID:
        prediction = normalize(base + float(alpha) * (model - base))
        rows.append({"alpha": float(alpha), **metrics(prediction, truth_sh, truth_d)})
    best = max(rows, key=lambda item: float(item["weighted"]))
    return float(best["alpha"]), rows


def build_sets(
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
) -> v274.RawSet:
    return v274.build_raw_set(
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


def main() -> None:
    for path in (BASE, TARGET_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    target_data = np.load(TARGET_CACHE)
    sh = normalize(target_data["sh"].astype(np.float32))
    d = normalize(target_data["d"].astype(np.float32))
    valid_ids = np.flatnonzero(target_data["valid"].astype(bool)).astype(np.int64)
    peak_indices, peak_mass = top_peaks(sh)
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
    train_pos = all_pos[: len(sh)]
    test_pos = all_pos[len(sh) :]
    node_raw = v274.build_node_features(
        all_pos, signatures, blocker_count, scene_features
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"v279 device={device} valid={len(valid_ids)}", flush=True)

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

        raw_inner_train = build_sets(
            inner_fit,
            inner_fit,
            all_pos,
            signatures,
            blocker_count,
            footprint,
            z_max,
            x_min,
            y_min,
            resolution,
        )
        raw_inner_val = build_sets(
            inner_ids,
            inner_fit,
            all_pos,
            signatures,
            blocker_count,
            footprint,
            z_max,
            x_min,
            y_min,
            resolution,
        )
        inner_stats = v274.fit_standardizer(
            node_raw, inner_fit, raw_inner_train
        )
        inner_train = v274.apply_standardizer(
            raw_inner_train, node_raw, inner_stats
        )
        inner_val = v274.apply_standardizer(raw_inner_val, node_raw, inner_stats)
        _, inner_training = train_model(
            inner_train,
            inner_val,
            node_raw,
            inner_stats,
            sh,
            d,
            peak_indices,
            peak_mass,
            seed,
            EPOCHS,
            device,
        )
        selected_epoch = int(inner_training["best_epoch"])
        selected_epochs.append(selected_epoch)

        raw_outer_train = build_sets(
            outer_fit,
            outer_fit,
            all_pos,
            signatures,
            blocker_count,
            footprint,
            z_max,
            x_min,
            y_min,
            resolution,
        )
        raw_outer_val = build_sets(
            outer_ids,
            outer_fit,
            all_pos,
            signatures,
            blocker_count,
            footprint,
            z_max,
            x_min,
            y_min,
            resolution,
        )
        outer_stats = v274.fit_standardizer(
            node_raw, outer_fit, raw_outer_train
        )
        outer_train = v274.apply_standardizer(
            raw_outer_train, node_raw, outer_stats
        )
        outer_val = v274.apply_standardizer(raw_outer_val, node_raw, outer_stats)
        base_sh = base_prediction(outer_val, sh)
        if selected_epoch == 0:
            model_sh = base_sh.copy()
            gates = np.zeros(len(outer_ids), dtype=np.float32)
            outer_training = {
                "seed": seed + 100_000,
                "best_epoch": 0,
                "history": [],
            }
        else:
            outer_model, outer_training = train_model(
                outer_train,
                None,
                node_raw,
                outer_stats,
                sh,
                d,
                peak_indices,
                peak_mass,
                seed + 100_000,
                selected_epoch,
                device,
            )
            model_sh, gates = predict_model(
                outer_model,
                outer_val,
                node_raw,
                outer_stats,
                sh,
                peak_indices,
                peak_mass,
                device,
            )
        baseline_metric = metrics(base_sh, sh[outer_ids], d[outer_ids])
        model_metric = metrics(model_sh, sh[outer_ids], d[outer_ids])
        delta = {
            key: model_metric[key] - baseline_metric[key]
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
                "baseline": baseline_metric,
                "model": model_metric,
                "raw_delta": delta,
                "gate": summarize(gates),
            }
        )
        oof_ids_parts.append(outer_ids)
        oof_base_parts.append(base_sh)
        oof_model_parts.append(model_sh)
        print(
            f"v279 outer {fold + 1}/{len(OUTER_SEEDS)} epoch={selected_epoch} "
            f"base={baseline_metric['weighted']:.6f} "
            f"model={model_metric['weighted']:.6f} delta={delta['weighted']:+.6f}",
            flush=True,
        )

    oof_ids = np.concatenate(oof_ids_parts)
    oof_base = np.concatenate(oof_base_parts)
    oof_model = np.concatenate(oof_model_parts)
    alpha, alpha_grid = choose_joint_alpha(
        oof_base, oof_model, sh[oof_ids], d[oof_ids]
    )
    fold_blended_delta: list[dict[str, float]] = []
    start = 0
    for row, ids, base_sh, model_sh in zip(
        outer_rows, oof_ids_parts, oof_base_parts, oof_model_parts
    ):
        blended = normalize(base_sh + alpha * (model_sh - base_sh))
        blended_metric = metrics(blended, sh[ids], d[ids])
        blended_delta = {
            key: blended_metric[key] - row["baseline"][key]
            for key in ("sh", "d", "weighted")
        }
        row["blended"] = blended_metric
        row["blended_delta"] = blended_delta
        fold_blended_delta.append(blended_delta)
        start += len(ids)
    mean_delta = {
        key: float(np.mean([value[key] for value in fold_blended_delta]))
        for key in ("sh", "d", "weighted")
    }
    weighted_wins = int(
        np.sum([value["weighted"] > 0.0 for value in fold_blended_delta])
    )
    official_selected = bool(
        alpha > 0.0
        and mean_delta["weighted"] >= MIN_OFFICIAL_DELTA
        and weighted_wins >= 3
        and mean_delta["sh"] > 0.0
        and mean_delta["d"] > 0.0
    )

    final_training: list[dict[str, object]] = []
    local_score = None
    projection_diagnostics = None
    if official_selected:
        positive_epochs = [epoch for epoch in selected_epochs if epoch > 0]
        final_epoch = max(int(np.median(positive_epochs)), 1)
        test_ids = np.arange(len(sh), len(all_pos), dtype=np.int64)
        raw_full_train = build_sets(
            valid_ids,
            valid_ids,
            all_pos,
            signatures,
            blocker_count,
            footprint,
            z_max,
            x_min,
            y_min,
            resolution,
        )
        raw_test = build_sets(
            test_ids,
            valid_ids,
            all_pos,
            signatures,
            blocker_count,
            footprint,
            z_max,
            x_min,
            y_min,
            resolution,
        )
        full_stats = v274.fit_standardizer(node_raw, valid_ids, raw_full_train)
        full_train = v274.apply_standardizer(
            raw_full_train, node_raw, full_stats
        )
        test_set = v274.apply_standardizer(raw_test, node_raw, full_stats)
        test_base_sh = base_prediction(test_set, sh)
        model_parts: list[np.ndarray] = []
        gate_parts: list[np.ndarray] = []
        for seed in FINAL_SEEDS:
            model, training = train_model(
                full_train,
                None,
                node_raw,
                full_stats,
                sh,
                d,
                peak_indices,
                peak_mass,
                seed,
                final_epoch,
                device,
            )
            test_model_sh, test_gate = predict_model(
                model,
                test_set,
                node_raw,
                full_stats,
                sh,
                peak_indices,
                peak_mass,
                device,
            )
            model_parts.append(test_model_sh)
            gate_parts.append(test_gate)
            checkpoint = CHECKPOINT_DIR / f"seed_{seed}.pt"
            torch.save(
                {
                    "version": VERSION,
                    "seed": seed,
                    "epoch": final_epoch,
                    "model_state": model.state_dict(),
                },
                checkpoint,
            )
            final_training.append(
                {**training, "checkpoint": PATHS.relative(checkpoint)}
            )
        test_model_sh = normalize(np.mean(model_parts, axis=0))
        current_sh, _ = v274.extract_current_marginals(BASE)
        residual = test_model_sh - test_base_sh
        target_sh = normalize(current_sh + alpha * residual)
        target_d = derive_d_numpy(target_sh)
        np.savez_compressed(
            TARGETS_OUT,
            base_sh=test_base_sh,
            model_sh=test_model_sh,
            current_sh=current_sh,
            target_sh=target_sh,
            target_d=target_d,
            residual=residual,
            alpha=np.asarray(alpha, dtype=np.float32),
            gate=np.mean(gate_parts, axis=0).astype(np.float32),
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
            "The official leaked S-H and PDP targets are one sparse joint component field "
            "with D=(-S) mod 192, and unseen points are best predicted by selecting and "
            "locally transporting discrete donor components rather than regressing dense spectra."
        ),
        "compliance_boundary": {
            "uses": [
                "official train H joint S-H/PDP labels",
                "official train/test XY coordinates",
                "official point-cloud wall/component/height features",
                "strict v239 official-only carrier",
            ],
            "does_not_use": [
                "synthetic or Sionna channel labels",
                "v212/v217 teacher outputs",
                "hidden labels",
                "RT MIMO H or path coefficients",
                "local replay truth for training or selection",
            ],
            "local_truth_usage": (
                "one final gate score after nested official selection"
                if official_selected
                else "not read"
            ),
        },
        "parameters": {
            "outer_seeds": list(OUTER_SEEDS),
            "outer_count": OUTER_COUNT,
            "inner_count": INNER_COUNT,
            "neighbors": v274.K_NEIGHBORS,
            "peaks_per_neighbor": PEAK_COUNT,
            "offsets": [list(value) for value in OFFSETS],
            "epochs": EPOCHS,
            "alpha_grid": ALPHA_GRID.tolist(),
            "minimum_official_delta": MIN_OFFICIAL_DELTA,
            "final_seeds": list(FINAL_SEEDS),
        },
        "target_identity": {
            "definition": "PDP is derived from S-H by summing H and applying D=(-S) mod 192",
            "train_sh_to_d_cosine": summarize(row_cos(derive_d_numpy(sh), d)),
        },
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
