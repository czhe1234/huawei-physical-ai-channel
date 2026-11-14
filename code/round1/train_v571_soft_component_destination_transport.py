#!/usr/bin/env python3
"""Learn path-ID-free component birth, death, and energy on physical destinations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v545_path_block_family_oracle as v545
import generate_v543_disagreement_conditioned_cell_fusion as v543
import train_v570_global_path_family_graph_renderer as v570
from project_paths import PATHS


VERSION_NUMBER = 571
VERSION = "v571_soft_component_destination_transport"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
FOLD_DIR = CACHE_DIR / "fold_predictions"
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = CACHE_DIR / "v571_soft_component_transport_oof.npz"

OOF_SOURCE = v545.OOF_SOURCE
BANK_CACHE = v545.BANK_CACHE
TRAIN_POS = v545.TRAIN_POS
TRAIN_POWER = v545.TRAIN_POWER

POINT_COUNT = 2000
CELL_COUNT = v545.HEIGHT * v545.WIDTH
SOURCE_COUNT = v545.SOURCE_COUNT
PHYSICAL_SOURCE_START = 1
SUPPORT_COUNT = 96
DONOR_NEIGHBORS = 24
DONOR_COMPONENTS = 8
FLOOR = 1e-10
MAX_LOG_GAIN = 10.0
SEED = 571_001
ALPHA_GRID = np.linspace(0.0, 1.0, 11, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--run-control", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def validate_inputs() -> None:
    required = (OOF_SOURCE, BANK_CACHE, TRAIN_POS, TRAIN_POWER)
    missing = [PATHS.relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing v571 inputs: {missing}")


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), FLOOR)
    return value / np.maximum(value.sum(axis=1, keepdims=True), 1e-30)


def coordinate_features(index: np.ndarray) -> np.ndarray:
    index = np.asarray(index, dtype=np.int64)
    delay = index // v545.WIDTH
    angle = index % v545.WIDTH
    values: list[np.ndarray] = []
    for harmonic in (1.0, 2.0, 4.0):
        phase = 2.0 * math.pi * harmonic * delay / v545.HEIGHT
        values.extend([np.sin(phase), np.cos(phase)])
    for harmonic in (1.0, 2.0, 4.0):
        phase = 2.0 * math.pi * harmonic * angle / v545.WIDTH
        values.extend([np.sin(phase), np.cos(phase)])
    return np.stack(values, axis=-1).astype(np.float32)


def gather_bank_cells(
    bank: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    gathered = np.take_along_axis(
        bank,
        indices[:, None, :],
        axis=2,
    )
    return np.transpose(gathered, (0, 2, 1)).astype(np.float32)


def spectral_value_features(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    return np.concatenate(
        [
            np.clip(np.log(value + FLOOR) / 12.0, -2.0, 0.5),
            np.sqrt(value * float(CELL_COUNT)).clip(0.0, 8.0) / 3.0,
        ],
        axis=-1,
    ).astype(np.float32)


def build_query_features(
    candidate: np.ndarray,
    base: np.ndarray,
    support_index: np.ndarray,
) -> np.ndarray:
    support_base = np.take_along_axis(base, support_index, axis=1)
    physical = candidate[:, :, PHYSICAL_SOURCE_START:]
    mean = physical.mean(axis=2)
    maximum = physical.max(axis=2)
    std = physical.std(axis=2)
    active = np.mean(physical > 1e-7, axis=2)
    summary = np.stack(
        [
            np.log(mean + FLOOR) / 12.0,
            np.log(maximum + FLOOR) / 12.0,
            np.log(std + FLOOR) / 12.0,
            active,
            np.log(support_base + FLOOR) / 12.0,
            np.sqrt(support_base * float(CELL_COUNT)).clip(0.0, 8.0) / 3.0,
        ],
        axis=2,
    ).astype(np.float32)
    return np.concatenate(
        [
            spectral_value_features(candidate),
            summary,
            coordinate_features(support_index),
        ],
        axis=2,
    ).astype(np.float32)


def build_donor_core(
    candidate: np.ndarray,
    truth_mass: np.ndarray,
    component_index: np.ndarray,
) -> np.ndarray:
    rank = np.arange(DONOR_COMPONENTS, dtype=np.float32)[None, :, None]
    rank = np.repeat(rank, len(candidate), axis=0) / max(DONOR_COMPONENTS - 1, 1)
    mass = truth_mass[:, :, None]
    summary = np.concatenate(
        [
            np.log(mass + FLOOR) / 12.0,
            np.sqrt(mass * float(CELL_COUNT)).clip(0.0, 8.0) / 3.0,
            rank,
        ],
        axis=2,
    ).astype(np.float32)
    return np.concatenate(
        [
            spectral_value_features(candidate),
            summary,
            coordinate_features(component_index),
        ],
        axis=2,
    ).astype(np.float32)


def compact_candidate_bank(
    bank: np.memmap,
    truth: np.ndarray,
    base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support_index = np.empty((POINT_COUNT, SUPPORT_COUNT), dtype=np.int64)
    query_feature_parts: list[np.ndarray] = []
    donor_core_parts: list[np.ndarray] = []
    component_index = np.argpartition(
        truth,
        -DONOR_COMPONENTS,
        axis=1,
    )[:, -DONOR_COMPONENTS:]
    component_mass = np.take_along_axis(truth, component_index, axis=1)
    order = np.argsort(component_mass, axis=1)[:, ::-1]
    component_index = np.take_along_axis(component_index, order, axis=1)
    component_mass = np.take_along_axis(component_mass, order, axis=1).astype(np.float32)

    for start in range(0, POINT_COUNT, 100):
        stop = min(start + 100, POINT_COUNT)
        local_bank = np.asarray(bank[start:stop], dtype=np.float32)
        physical_max = np.max(local_bank[:, PHYSICAL_SOURCE_START:, :], axis=1)
        local_support = np.argpartition(
            physical_max,
            -SUPPORT_COUNT,
            axis=1,
        )[:, -SUPPORT_COUNT:]
        local_value = np.take_along_axis(physical_max, local_support, axis=1)
        local_order = np.argsort(local_value, axis=1)[:, ::-1]
        local_support = np.take_along_axis(local_support, local_order, axis=1)
        support_index[start:stop] = local_support
        query_candidate = gather_bank_cells(local_bank, local_support)
        donor_candidate = gather_bank_cells(
            local_bank,
            component_index[start:stop],
        )
        query_feature_parts.append(
            build_query_features(
                query_candidate,
                base[start:stop],
                local_support,
            )
        )
        donor_core_parts.append(
            build_donor_core(
                donor_candidate,
                component_mass[start:stop],
                component_index[start:stop],
            )
        )
    return (
        support_index,
        np.concatenate(query_feature_parts, axis=0),
        component_index.astype(np.int64),
        np.concatenate(donor_core_parts, axis=0),
    )


def shuffle_within_regime(regimes: np.ndarray, valid: np.ndarray, seed: int) -> np.ndarray:
    mapping = np.arange(len(regimes), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for regime in v545.REGIMES:
        ids = np.flatnonzero(valid & (regimes == regime))
        mapping[ids] = rng.permutation(ids)
    return mapping


def load_data() -> dict[str, np.ndarray]:
    oof = np.load(OOF_SOURCE)
    truth = oof["truth"].astype(np.float32)
    base = oof["calibrated"].astype(np.float32)
    regimes = oof["regimes"].astype("U8")
    positions = np.load(TRAIN_POS).astype(np.float32)
    power = np.load(TRAIN_POWER).astype(np.float64)
    valid = power != 0.0
    static, _, _ = v543.load_static_features()
    bank = np.load(BANK_CACHE, mmap_mode="r")
    support_index, query_features, component_index, donor_core = compact_candidate_bank(
        bank,
        truth,
        base,
    )
    del bank
    return {
        "truth": truth,
        "base": base,
        "regimes": regimes,
        "positions": positions,
        "valid": valid,
        "folds": v570.contiguous_fold_codes(positions),
        "static": np.nan_to_num(static.astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6),
        "support_index": support_index,
        "query_features": query_features,
        "component_index": component_index,
        "donor_core": donor_core,
    }


def nearest_donors(
    query_ids: np.ndarray,
    fit_ids: np.ndarray,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(positions[fit_ids, :2].astype(np.float64))
    request = min(DONOR_NEIGHBORS + 1, len(fit_ids))
    distance, local = tree.query(
        positions[query_ids, :2].astype(np.float64),
        k=request,
    )
    if request == 1:
        distance = distance[:, None]
        local = local[:, None]
    candidates = fit_ids[local]
    donor_ids = np.empty((len(query_ids), DONOR_NEIGHBORS), dtype=np.int64)
    donor_distance = np.empty((len(query_ids), DONOR_NEIGHBORS), dtype=np.float32)
    for row, query in enumerate(query_ids):
        keep = candidates[row] != query
        ids = candidates[row, keep]
        dist = np.asarray(distance[row])[keep]
        if len(ids) < DONOR_NEIGHBORS:
            ids = np.resize(ids, DONOR_NEIGHBORS)
            dist = np.resize(dist, DONOR_NEIGHBORS)
        donor_ids[row] = ids[:DONOR_NEIGHBORS]
        donor_distance[row] = dist[:DONOR_NEIGHBORS]
    return donor_ids, donor_distance


def relative_features(
    query_ids: np.ndarray,
    donor_ids: np.ndarray,
    distance: np.ndarray,
    positions: np.ndarray,
    regimes: np.ndarray,
) -> np.ndarray:
    delta = positions[query_ids, None, :2] - positions[donor_ids, :2]
    azimuth = np.arctan2(delta[:, :, 1], delta[:, :, 0])
    same_regime = regimes[query_ids, None] == regimes[donor_ids]
    values = np.stack(
        [
            delta[:, :, 0] / 20.0,
            delta[:, :, 1] / 20.0,
            delta[:, :, 0] / 80.0,
            delta[:, :, 1] / 80.0,
            distance / 20.0,
            np.log1p(distance) / 5.0,
            np.sin(azimuth),
            np.cos(azimuth),
            same_regime.astype(np.float32),
        ],
        axis=2,
    )
    return np.clip(values, -8.0, 8.0).astype(np.float32)


@dataclass
class PreparedSet:
    query_ids: np.ndarray
    support_index: torch.Tensor
    query_feature: torch.Tensor
    donor_feature: torch.Tensor
    query_static: torch.Tensor
    donor_static: torch.Tensor
    base: torch.Tensor
    truth: torch.Tensor
    base_weighted: torch.Tensor

    def __len__(self) -> int:
        return len(self.query_ids)


def standardizer(value: np.ndarray, fit_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = value[fit_ids].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = value[fit_ids].std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-4)
    return mean, std


def prepare_set(
    query_ids: np.ndarray,
    fit_ids: np.ndarray,
    data: dict[str, np.ndarray],
    static_mean: np.ndarray,
    static_std: np.ndarray,
    device: torch.device,
    candidate_mapping: np.ndarray | None = None,
) -> PreparedSet:
    donor_ids, distance = nearest_donors(
        query_ids,
        fit_ids,
        data["positions"],
    )
    relative = relative_features(
        query_ids,
        donor_ids,
        distance,
        data["positions"],
        data["regimes"],
    )
    donor_core = data["donor_core"][donor_ids]
    relative = np.repeat(relative[:, :, None, :], DONOR_COMPONENTS, axis=2)
    donor_feature = np.concatenate([donor_core, relative], axis=3).reshape(
        len(query_ids), DONOR_NEIGHBORS * DONOR_COMPONENTS, -1
    )
    mapped = query_ids if candidate_mapping is None else candidate_mapping[query_ids]
    query_static = np.clip(
        (data["static"][query_ids] - static_mean) / static_std,
        -8.0,
        8.0,
    ).astype(np.float32)
    donor_static = np.clip(
        (data["static"][donor_ids] - static_mean) / static_std,
        -8.0,
        8.0,
    ).astype(np.float32)
    base_scores = v543.spectrum_scores(
        data["truth"][query_ids],
        data["base"][query_ids],
    )["weighted"]
    return PreparedSet(
        query_ids=query_ids,
        support_index=torch.from_numpy(data["support_index"][mapped]).to(device),
        query_feature=torch.from_numpy(data["query_features"][mapped]).to(device),
        donor_feature=torch.from_numpy(donor_feature).to(device),
        query_static=torch.from_numpy(query_static).to(device),
        donor_static=torch.from_numpy(donor_static).to(device),
        base=torch.from_numpy(data["base"][query_ids]).to(device),
        truth=torch.from_numpy(data["truth"][query_ids]).to(device),
        base_weighted=torch.from_numpy(base_scores).to(device),
    )


class SoftComponentTransportNet(nn.Module):
    def __init__(
        self,
        query_dim: int,
        donor_dim: int,
        static_dim: int,
        hidden: int,
    ) -> None:
        super().__init__()
        self.query_encoder = nn.Sequential(
            nn.Linear(query_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        self.donor_encoder = nn.Sequential(
            nn.Linear(donor_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        self.query_static = nn.Sequential(
            nn.Linear(static_dim, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
        )
        self.donor_static = nn.Sequential(
            nn.Linear(static_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        donor_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=6,
            dim_feedforward=hidden * 3,
            dropout=0.06,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.donor_transformer = nn.TransformerEncoder(donor_layer, num_layers=1)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=6,
            dim_feedforward=hidden * 4,
            dropout=0.06,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transport = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.output_norm = nn.LayerNorm(hidden)
        self.delta_head = nn.Linear(hidden, 1)
        self.gate_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, -1.0)

    def forward(
        self,
        support_index: torch.Tensor,
        query_feature: torch.Tensor,
        donor_feature: torch.Tensor,
        query_static: torch.Tensor,
        donor_static: torch.Tensor,
        base: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = len(base)
        global_query = self.query_static(query_static)
        query = self.query_encoder(query_feature) + global_query[:, None, :]
        donor = self.donor_encoder(donor_feature)
        donor_state = self.donor_static(donor_static)
        donor_state = donor_state[:, :, None, :].expand(
            -1, -1, DONOR_COMPONENTS, -1
        ).reshape(batch, DONOR_NEIGHBORS * DONOR_COMPONENTS, -1)
        donor = self.donor_transformer(donor + donor_state + global_query[:, None, :])
        transported = self.transport(query, donor)
        transported = self.output_norm(transported)
        raw_delta = MAX_LOG_GAIN * torch.tanh(self.delta_head(transported).squeeze(2) / 3.0)
        pooled = transported.mean(dim=1)
        gate = torch.sigmoid(self.gate_head(torch.cat([global_query, pooled], dim=1)))
        delta = gate * raw_delta
        full_delta = torch.zeros_like(base)
        full_delta.scatter_(1, support_index, delta.to(full_delta.dtype))
        prediction = base * torch.exp(full_delta.clamp(-MAX_LOG_GAIN, MAX_LOG_GAIN))
        prediction = prediction / prediction.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return prediction, {
            "delta": delta,
            "raw_delta": raw_delta,
            "gate": gate[:, 0],
        }


def torch_scores(truth: torch.Tensor, prediction: torch.Tensor) -> dict[str, torch.Tensor]:
    truth_grid = truth.reshape(-1, v545.HEIGHT, v545.WIDTH)
    pred_grid = prediction.reshape(-1, v545.HEIGHT, v545.WIDTH)
    joint = F.cosine_similarity(prediction, truth, dim=1)
    delay = F.cosine_similarity(pred_grid.sum(dim=2), truth_grid.sum(dim=2), dim=1)
    angle = F.cosine_similarity(pred_grid.sum(dim=1), truth_grid.sum(dim=1), dim=1)
    weighted = 0.50 * joint + 0.25 * delay + 0.25 * angle
    return {"joint": joint, "delay": delay, "angle": angle, "weighted": weighted}


def model_loss(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    base_weighted: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
) -> torch.Tensor:
    scores = torch_scores(truth, prediction)
    hard_weight = 1.0 + 1.5 * (1.0 - base_weighted).clamp(0.0, 1.0)
    score_loss = torch.sum(hard_weight * (1.0 - scores["weighted"])) / hard_weight.sum()
    root_loss = F.smooth_l1_loss(
        torch.sqrt(prediction.clamp_min(FLOOR)),
        torch.sqrt(truth.clamp_min(FLOOR)),
    )
    regularization = 5e-5 * diagnostics["raw_delta"].square().mean()
    return score_loss + 0.02 * root_loss + regularization


def batch_from(data: PreparedSet, rows: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return (
        data.support_index[rows],
        data.query_feature[rows],
        data.donor_feature[rows],
        data.query_static[rows],
        data.donor_static[rows],
        data.base[rows],
    )


@torch.no_grad()
def evaluate_model(
    model: SoftComponentTransportNet,
    data: PreparedSet,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, float]]:
    model.eval()
    prediction = np.empty((len(data), CELL_COUNT), dtype=np.float32)
    gates: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    for start in range(0, len(data), batch_size * 2):
        stop = min(start + batch_size * 2, len(data))
        rows = torch.arange(start, stop, device=data.base.device)
        with torch.autocast(
            device_type=data.base.device.type,
            dtype=torch.bfloat16,
            enabled=data.base.device.type == "cuda",
        ):
            value, diagnostics = model(*batch_from(data, rows))
        prediction[start:stop] = value.float().cpu().numpy()
        gates.append(diagnostics["gate"].float().cpu().numpy())
        deltas.append(diagnostics["delta"].float().cpu().numpy())
    metrics = v543.spectrum_scores(data.truth.float().cpu().numpy(), prediction)
    gate = np.concatenate(gates)
    delta = np.concatenate(deltas)
    diagnostics = {
        "gate_mean": float(np.mean(gate)),
        "gate_p90": float(np.quantile(gate, 0.9)),
        "mean_abs_log_gain": float(np.mean(np.abs(delta))),
        "p90_abs_log_gain": float(np.quantile(np.abs(delta), 0.9)),
    }
    return metrics, prediction, diagnostics


def state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def select_blend_alpha(
    base: np.ndarray,
    prediction: np.ndarray,
    truth: np.ndarray,
) -> tuple[float, float]:
    best_alpha = 0.0
    best_score = float(np.mean(v543.spectrum_scores(truth, base)["weighted"]))
    for alpha in ALPHA_GRID[1:]:
        blended = v543.log_blend(base, prediction, float(alpha))
        score = float(np.mean(v543.spectrum_scores(truth, blended)["weighted"]))
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha, best_score


def train_model(
    training: PreparedSet,
    validation: PreparedSet | None,
    args: argparse.Namespace,
    seed: int,
    fixed_epochs: int | None = None,
) -> tuple[SoftComponentTransportNet, dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SoftComponentTransportNet(
        query_dim=training.query_feature.shape[2],
        donor_dim=training.donor_feature.shape[2],
        static_dim=training.query_static.shape[1],
        hidden=args.hidden_dim,
    ).to(training.base.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    epochs = args.epochs if fixed_epochs is None else fixed_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=2e-5,
    )
    rng = np.random.default_rng(seed)
    best_epoch = 0
    best_score = (
        float(np.mean(training.base_weighted.float().cpu().numpy()))
        if validation is None
        else float(np.mean(validation.base_weighted.float().cpu().numpy()))
    )
    best_state = state_dict_cpu(model)
    best_alpha = 0.0
    history: list[dict[str, float]] = []
    stale = 0
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(training))
        losses: list[float] = []
        for start in range(0, len(order), args.batch_size):
            local = order[start : start + args.batch_size]
            rows = torch.from_numpy(local).to(training.base.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=training.base.device.type,
                dtype=torch.bfloat16,
                enabled=training.base.device.type == "cuda",
            ):
                prediction, diagnostics = model(*batch_from(training, rows))
                loss = model_loss(
                    prediction,
                    training.truth[rows],
                    training.base_weighted[rows],
                    diagnostics,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite v571 loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            train_metrics, _, train_diag = evaluate_model(
                model,
                training,
                args.batch_size,
            )
            train_score = float(np.mean(train_metrics["weighted"]))
            if validation is None:
                validation_score = train_score
                validation_diag = train_diag
                validation_alpha = 1.0
            else:
                _validation_metrics, validation_prediction, validation_diag = evaluate_model(
                    model,
                    validation,
                    args.batch_size,
                )
                validation_alpha, validation_score = select_blend_alpha(
                    validation.base.float().cpu().numpy(),
                    validation_prediction,
                    validation.truth.float().cpu().numpy(),
                )
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": float(np.mean(losses)),
                    "train_score": train_score,
                    "validation_score": validation_score,
                    "validation_alpha": validation_alpha,
                    "validation_gate": validation_diag["gate_mean"],
                    "validation_abs_log_gain": validation_diag["mean_abs_log_gain"],
                }
            )
            print(
                f"v571 epoch={epoch:03d} loss={np.mean(losses):.6f} "
                f"train={train_score:.6f} validation={validation_score:.6f} "
                f"alpha={validation_alpha:.2f} "
                f"gate={validation_diag['gate_mean']:.3f} "
                f"move={validation_diag['mean_abs_log_gain']:.3f}",
                flush=True,
            )
            if validation_score > best_score + 1e-5:
                best_score = validation_score
                best_epoch = epoch
                best_alpha = validation_alpha
                best_state = state_dict_cpu(model)
                stale = 0
            else:
                stale += 5
                if fixed_epochs is None and stale >= args.patience:
                    break
    if fixed_epochs is None:
        model.load_state_dict(best_state)
    else:
        best_epoch = fixed_epochs
        best_alpha = 1.0
        if history:
            best_score = history[-1]["validation_score"]
    return model, {
        "best_epoch": int(best_epoch),
        "best_validation_score": float(best_score),
        "best_alpha": float(best_alpha),
        "elapsed_seconds": float(time.monotonic() - started),
        "history": history,
    }


def run_variant(
    fold: int,
    data: dict[str, np.ndarray],
    device: torch.device,
    args: argparse.Namespace,
    candidate_mapping: np.ndarray | None,
    label: str,
) -> tuple[np.ndarray, dict[str, object], dict[str, torch.Tensor]]:
    valid = data["valid"]
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
    static_mean, static_std = standardizer(data["static"], inner_train_ids)
    print(
        f"v571 {label} fold={fold} inner_train={len(inner_train_ids)} "
        f"inner_validation={len(inner_validation_ids)} outer={len(query_ids)}",
        flush=True,
    )
    inner_train = prepare_set(
        inner_train_ids,
        inner_train_ids,
        data,
        static_mean,
        static_std,
        device,
        candidate_mapping,
    )
    inner_validation = prepare_set(
        inner_validation_ids,
        inner_train_ids,
        data,
        static_mean,
        static_std,
        device,
        candidate_mapping,
    )
    selected_model, selection = train_model(
        inner_train,
        inner_validation,
        args,
        seed=args.seed + 1000 * fold + (100 if label == "control" else 0),
    )
    selected_epoch = int(selection["best_epoch"])
    selected_alpha = float(selection["best_alpha"])
    del selected_model, inner_train, inner_validation
    if device.type == "cuda":
        torch.cuda.empty_cache()

    static_mean, static_std = standardizer(data["static"], fit_ids)
    final_train = prepare_set(
        fit_ids,
        fit_ids,
        data,
        static_mean,
        static_std,
        device,
        candidate_mapping,
    )
    outer_set = prepare_set(
        query_ids,
        fit_ids,
        data,
        static_mean,
        static_std,
        device,
        candidate_mapping,
    )
    model, final_training = train_model(
        final_train,
        None,
        args,
        seed=args.seed + 1000 * fold + 1 + (100 if label == "control" else 0),
        fixed_epochs=selected_epoch,
    )
    _raw_metrics, raw_prediction, diagnostics = evaluate_model(
        model,
        outer_set,
        args.batch_size,
    )
    prediction = v543.log_blend(
        data["base"][query_ids],
        raw_prediction,
        selected_alpha,
    )
    metrics = v543.spectrum_scores(data["truth"][query_ids], prediction)
    details: dict[str, object] = {
        "label": label,
        "selected_epoch": selected_epoch,
        "selected_alpha": selected_alpha,
        "selection": selection,
        "final_training": final_training,
        "score": {key: float(np.mean(value)) for key, value in metrics.items()},
        "diagnostics": diagnostics,
    }
    state = {
        "state_dict": state_dict_cpu(model),
        "static_mean": torch.from_numpy(static_mean),
        "static_std": torch.from_numpy(static_std),
    }
    del model, final_train, outer_set
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return prediction, details, state


def run_fold(
    fold: int,
    data: dict[str, np.ndarray],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, object]:
    fold_path = FOLD_DIR / f"fold_{fold}.npz"
    checkpoint_path = CHECKPOINT_DIR / f"fold_{fold}.pt"
    if fold_path.is_file() and not args.force:
        with np.load(fold_path) as loaded:
            return json.loads(str(loaded["details"].item()))
    prediction, training, state = run_variant(
        fold,
        data,
        device,
        args,
        candidate_mapping=None,
        label="main",
    )
    query_ids = np.flatnonzero(data["valid"] & (data["folds"] == fold))
    base_metrics = v543.spectrum_scores(
        data["truth"][query_ids],
        data["base"][query_ids],
    )
    pred_metrics = v543.spectrum_scores(data["truth"][query_ids], prediction)
    details: dict[str, object] = {
        "fold": fold,
        "query_points": int(len(query_ids)),
        "baseline": {key: float(np.mean(value)) for key, value in base_metrics.items()},
        "prediction": {key: float(np.mean(value)) for key, value in pred_metrics.items()},
        "gain": {
            key: float(np.mean(pred_metrics[key] - base_metrics[key]))
            for key in pred_metrics
        },
        "training": training,
        "control": None,
        "margin_over_control": None,
    }
    save_values: dict[str, np.ndarray] = {
        "query_ids": query_ids.astype(np.int32),
        "prediction": prediction.astype(np.float32),
    }
    if args.run_control:
        mapping = shuffle_within_regime(
            data["regimes"],
            data["valid"],
            args.seed + 20_000 + fold,
        )
        control_prediction, control_training, _ = run_variant(
            fold,
            data,
            device,
            args,
            candidate_mapping=mapping,
            label="control",
        )
        control_metrics = v543.spectrum_scores(
            data["truth"][query_ids],
            control_prediction,
        )
        details["control"] = {
            "score": {key: float(np.mean(value)) for key, value in control_metrics.items()},
            "training": control_training,
        }
        details["margin_over_control"] = float(
            np.mean(pred_metrics["weighted"] - control_metrics["weighted"])
        )
        save_values["control_prediction"] = control_prediction.astype(np.float32)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": VERSION,
            "fold": fold,
            "model": state,
            "support_count": SUPPORT_COUNT,
            "donor_neighbors": DONOR_NEIGHBORS,
            "donor_components": DONOR_COMPONENTS,
            "hidden_dim": args.hidden_dim,
            "selected_epoch": training["selected_epoch"],
        },
        checkpoint_path,
    )
    save_values["details"] = np.asarray(json.dumps(details, ensure_ascii=False))
    np.savez_compressed(fold_path, **save_values)
    print(json.dumps(details["gain"], indent=2), flush=True)
    return details


def finalize(data: dict[str, np.ndarray], fold_rows: list[dict[str, object]]) -> None:
    prediction = data["base"].copy()
    control = data["base"].copy()
    has_control = True
    for fold in range(5):
        with np.load(FOLD_DIR / f"fold_{fold}.npz") as loaded:
            ids = loaded["query_ids"].astype(np.int64)
            prediction[ids] = loaded["prediction"]
            if "control_prediction" in loaded.files:
                control[ids] = loaded["control_prediction"]
            else:
                has_control = False
    valid = data["valid"]
    base_metrics = v543.spectrum_scores(data["truth"][valid], data["base"][valid])
    pred_metrics = v543.spectrum_scores(data["truth"][valid], prediction[valid])
    control_metrics = (
        v543.spectrum_scores(data["truth"][valid], control[valid])
        if has_control
        else None
    )
    gain = {
        key: float(np.mean(pred_metrics[key] - base_metrics[key]))
        for key in pred_metrics
    }
    fold_gain = [float(row["gain"]["weighted"]) for row in fold_rows]
    control_margin = (
        float(np.mean(pred_metrics["weighted"] - control_metrics["weighted"]))
        if control_metrics is not None
        else None
    )
    strict_success = bool(
        gain["weighted"] >= 0.01
        and min(fold_gain) > 0.0
        and gain["joint"] > 0.0
        and gain["delay"] > 0.0
        and gain["angle"] > 0.0
        and (control_margin is None or control_margin >= 0.70 * gain["weighted"])
    )
    np.savez_compressed(
        OOF_CACHE,
        valid=valid,
        fold_id=data["folds"],
        regimes=data["regimes"],
        prediction=prediction.astype(np.float32),
        control=control.astype(np.float32) if has_control else np.empty(0, dtype=np.float32),
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "Physical candidates already cover almost all true delay-angle destinations; "
            "a path-ID-free set model can use local official component evidence to predict "
            "which destinations are born, disappear, or receive energy in unseen regions."
        ),
        "scope": {
            "train_points": POINT_COUNT,
            "scored_nonzero_points": int(np.sum(valid)),
            "support_tokens_per_point": SUPPORT_COUNT,
            "donor_component_tokens": DONOR_NEIGHBORS * DONOR_COMPONENTS,
            "physical_candidate_sources": SOURCE_COUNT - PHYSICAL_SOURCE_START,
            "contiguous_outer_regions": 5,
            "submission_generated": False,
        },
        "folds": fold_rows,
        "summary": {
            "baseline": {key: float(np.mean(value)) for key, value in base_metrics.items()},
            "prediction": {key: float(np.mean(value)) for key, value in pred_metrics.items()},
            "gain": gain,
            "control": (
                {key: float(np.mean(value)) for key, value in control_metrics.items()}
                if control_metrics is not None
                else None
            ),
            "margin_over_control": control_margin,
        },
        "decision": {
            "strict_success": strict_success,
            "positive_contiguous_regions": int(np.sum(np.asarray(fold_gain) > 0.0)),
            "minimum_region_gain": float(np.min(fold_gain)),
            "success_gate": (
                "weighted gain >= 0.01; all five contiguous regions positive; joint, delay, "
                "and angle positive; shuffled-destination margin >= 70% of gain when run"
            ),
            "generate_submission": False,
        },
        "outputs": {
            "oof": PATHS.relative(OOF_CACHE),
            "fold_predictions": PATHS.relative(FOLD_DIR),
            "checkpoints": PATHS.relative(CHECKPOINT_DIR),
            "report": PATHS.relative(REPORT),
            "submission_zip": None,
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(json.dumps(report["decision"], indent=2), flush=True)


def main() -> None:
    args = parse_args()
    validate_inputs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    data = load_data()
    selected = [args.fold] if args.fold is not None else list(range(5))
    rows = [run_fold(fold, data, device, args) for fold in selected]
    if args.fold is None:
        finalize(data, rows)
    else:
        print(
            json.dumps(
                {
                    "fold": args.fold,
                    "gain": rows[0]["gain"],
                    "cache": PATHS.relative(FOLD_DIR / f"fold_{args.fold}.npz"),
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
