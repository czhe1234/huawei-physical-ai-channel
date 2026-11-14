#!/usr/bin/env python3
"""Train a listwise global-local selector over broad path candidate pools."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import analyze_global_local_path_lifecycle as e79
from analyze_v4_path_track_spectral_targets import MH, MV, S_SC
from project_paths import PATHS


VERSION = "Round2-E82-soft-path-tube-set-selector"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "global_local_soft_path_set_selector"
ANALYSIS_DIR = (
    PATHS.analysis / "results" / "round2_global_local_soft_path_set_selector"
)
REPORT = ANALYSIS_DIR / "global_local_soft_path_set_selector.json"
PREDICTIONS = ARTIFACT_DIR / "oof_soft_path_set_predictions.npz"

OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
GEODESIC_CACHE = (
    PATHS.artifacts
    / "round2"
    / "regime_lifecycle_support"
    / "geodesic_corridor_features.npz"
)
COMPONENTS = (
    PATHS.artifacts
    / "round2"
    / "v4_path_component_transport"
    / "train_components_k24_os2_ref2.npz"
)
TINY = 1e-30
SOFT_TUBE_COST_LIMIT = 0.25
SOFT_TUBE_BANDWIDTH = 0.25


@dataclass(frozen=True)
class Config:
    hidden: int = 128
    heads: int = 4
    layers: int = 2
    dropout: float = 0.08
    epochs: int = 100
    minimum_epochs: int = 25
    validation_interval: int = 5
    learning_rate: float = 8e-4
    weight_decay: float = 2e-4
    candidate_bce_weight: float = 0.05
    point_bce_weight: float = 0.08
    delta_weight: float = 0.10
    port_weight: float = 0.10
    random_state: int = 2026080792


@dataclass(frozen=True)
class RenderSpec:
    top_k: int
    model_scale: float
    support_scale: float
    port_scale: float

    @property
    def name(self) -> str:
        return (
            f"k{self.top_k}_m{self.model_scale:.2f}_"
            f"s{self.support_scale:.2f}_p{self.port_scale:.2f}"
        )


class PathSetSelector(nn.Module):
    def __init__(self, input_dim: int, config: Config) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, config.hidden),
            nn.GELU(),
            nn.LayerNorm(config.hidden),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden,
            nhead=config.heads,
            dim_feedforward=2 * config.hidden,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.context = nn.Sequential(
            nn.Linear(2 * config.hidden, config.hidden),
            nn.GELU(),
            nn.LayerNorm(config.hidden),
        )
        self.score = nn.Linear(config.hidden, 1)
        self.delta = nn.Linear(config.hidden, 3)
        self.port = nn.Linear(config.hidden, e79.PORTS)
        self.point = nn.Sequential(
            nn.Linear(config.hidden, config.hidden // 2),
            nn.GELU(),
            nn.Linear(config.hidden // 2, 1),
        )

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.input(features)
        encoded = self.encoder(encoded, src_key_padding_mask=~mask)
        weight = mask.to(encoded.dtype)
        pooled = (encoded * weight[..., None]).sum(dim=1) / weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        pooled_tokens = pooled[:, None].expand(-1, encoded.shape[1], -1)
        hidden = self.context(torch.cat((encoded, pooled_tokens), dim=2))
        score = self.score(hidden)[..., 0]
        limits = torch.tensor(
            [1.5, 1.0, 3.0], dtype=hidden.dtype, device=hidden.device
        )
        delta = torch.tanh(self.delta(hidden)) * limits
        port = torch.softmax(self.port(hidden), dim=2)
        point = self.point(pooled)[:, 0]
        return score, delta, port, point


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    return parser.parse_args()


def render_specs() -> tuple[RenderSpec, ...]:
    return tuple(
        RenderSpec(top_k, model_scale, support_scale, 0.50)
        for top_k in (16, 24)
        for model_scale in (0.25, 0.50, 0.75, 1.00)
        for support_scale in (0.0, 0.50)
    )


def soft_path_tube_targets(
    cache: dict[str, np.ndarray],
    query_ids: np.ndarray,
    true_coordinates: np.ndarray,
    true_coefficients: np.ndarray,
    point_power: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distribute each truth path over all nearby equivalent candidates."""

    offsets = cache["offsets"].astype(np.int64)
    candidate_coordinates = cache["coordinates"].astype(np.float32)
    target_mass = np.zeros(len(candidate_coordinates), dtype=np.float32)
    target_delta = np.zeros((len(candidate_coordinates), 3), dtype=np.float32)
    target_port = np.full(
        (len(candidate_coordinates), e79.PORTS),
        1.0 / e79.PORTS,
        dtype=np.float32,
    )
    for row, point_id in enumerate(query_ids):
        start = int(offsets[row])
        end = int(offsets[row + 1])
        if end <= start or point_power[point_id] <= TINY:
            continue
        coordinate = candidate_coordinates[start:end]
        truth_coordinate = true_coordinates[point_id, :24]
        truth_coefficient = true_coefficients[point_id, :24]
        truth_energy = np.sum(np.abs(truth_coefficient) ** 2, axis=1, dtype=np.float64)
        truth_fraction = truth_energy / max(float(point_power[point_id]), TINY)
        valid = truth_fraction >= 1e-5
        if not np.any(valid):
            continue
        truth_coordinate = truth_coordinate[valid]
        truth_coefficient = truth_coefficient[valid]
        truth_fraction = truth_fraction[valid]
        cost = e79.support_cost(coordinate, truth_coordinate)
        kernel = np.exp(-0.5 * cost / SOFT_TUBE_BANDWIDTH)
        kernel[cost > SOFT_TUBE_COST_LIMIT] = 0.0
        column_total = kernel.sum(axis=0, keepdims=True)
        responsibility = np.divide(
            kernel,
            np.maximum(column_total, TINY),
            out=np.zeros_like(kernel),
            where=column_total > 0.0,
        )
        contribution = responsibility * truth_fraction[None, :]
        local_mass = contribution.sum(axis=1)
        target_mass[start:end] = local_mass.astype(np.float32)
        row_total = contribution.sum(axis=1, keepdims=True)
        normalized_contribution = np.divide(
            contribution,
            np.maximum(row_total, TINY),
            out=np.zeros_like(contribution),
            where=row_total > 0.0,
        )
        delta = np.stack(
            (
                e79.periodic_delta(
                    truth_coordinate[None, :, 0], coordinate[:, None, 0], MH
                ),
                e79.periodic_delta(
                    truth_coordinate[None, :, 1], coordinate[:, None, 1], MV
                ),
                e79.periodic_delta(
                    truth_coordinate[None, :, 2], coordinate[:, None, 2], S_SC
                ),
            ),
            axis=2,
        )
        target_delta[start:end] = np.einsum(
            "ct,ctd->cd", normalized_contribution, delta
        ).astype(np.float32)
        truth_port = np.abs(truth_coefficient) ** 2
        truth_port /= np.maximum(truth_port.sum(axis=1, keepdims=True), TINY)
        local_port = normalized_contribution @ truth_port
        valid_candidate = local_mass > 0.0
        target_port[start:end][valid_candidate] = local_port[valid_candidate].astype(
            np.float32
        )
    return target_mass, target_delta, target_port


def audit_soft_targets(
    cache: dict[str, np.ndarray],
    query_ids: np.ndarray,
    true_coordinates: np.ndarray,
    true_coefficients: np.ndarray,
    point_power: np.ndarray,
    target_mass: np.ndarray,
    target_delta: np.ndarray,
    target_port: np.ndarray,
) -> dict[str, float | bool]:
    """Verify that every supported truth path conserves its normalized energy."""

    offsets = cache["offsets"].astype(np.int64)
    candidate_coordinates = cache["coordinates"].astype(np.float32)
    absolute_errors = []
    supported_mass = 0.0
    total_mass = 0.0
    renderable_mass = 0.0
    zero_point_mass = 0.0
    for row, point_id in enumerate(query_ids):
        start = int(offsets[row])
        end = int(offsets[row + 1])
        actual = float(np.sum(target_mass[start:end], dtype=np.float64))
        renderable_mass += float(
            np.sum(np.sort(target_mass[start:end])[-24:], dtype=np.float64)
        )
        if point_power[point_id] <= TINY:
            zero_point_mass = max(zero_point_mass, actual)
            continue
        truth_energy = np.sum(
            np.abs(true_coefficients[point_id, :24]) ** 2,
            axis=1,
            dtype=np.float64,
        )
        truth_fraction = truth_energy / max(float(point_power[point_id]), TINY)
        valid = truth_fraction >= 1e-5
        total_mass += float(np.sum(truth_fraction[valid], dtype=np.float64))
        if end <= start or not np.any(valid):
            absolute_errors.append(abs(actual))
            continue
        cost = e79.support_cost(
            candidate_coordinates[start:end], true_coordinates[point_id, :24][valid]
        )
        covered = np.any(cost <= SOFT_TUBE_COST_LIMIT, axis=0)
        expected = float(np.sum(truth_fraction[valid][covered], dtype=np.float64))
        supported_mass += expected
        absolute_errors.append(abs(actual - expected))
    finite = bool(
        np.all(np.isfinite(target_mass))
        and np.all(np.isfinite(target_delta))
        and np.all(np.isfinite(target_port))
    )
    maximum_error = float(np.max(absolute_errors, initial=0.0))
    audit = {
        "finite": finite,
        "maximum_point_mass_conservation_error": maximum_error,
        "mean_point_mass_conservation_error": float(
            np.mean(absolute_errors) if absolute_errors else 0.0
        ),
        "zero_point_target_mass_max": float(zero_point_mass),
        "supported_truth_mass_fraction": float(
            supported_mass / max(total_mass, TINY)
        ),
        "renderable_top24_supported_mass_fraction": float(
            renderable_mass / max(supported_mass, TINY)
        ),
        "positive_candidate_fraction": float(np.mean(target_mass > 0.0)),
    }
    if not finite or maximum_error > 5e-6 or zero_point_mass > 1e-12:
        raise RuntimeError(f"invalid soft path targets: {audit}")
    return audit


def load_candidate_cache(outer: int) -> dict[str, np.ndarray]:
    path = e79.candidate_cache_path(outer, e79.Config())
    if not path.is_file():
        raise FileNotFoundError(PATHS.relative(path))
    cached = np.load(path, allow_pickle=False)
    return {key: cached[key] for key in cached.files}


def pad_cache(
    cache: dict[str, np.ndarray],
    features: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    offsets = cache["offsets"].astype(np.int64)
    counts = np.diff(offsets)
    maximum = int(np.max(counts))
    count = len(counts)
    feature_pad = np.zeros((count, maximum, features.shape[1]), dtype=np.float32)
    coordinate_pad = np.zeros((count, maximum, 3), dtype=np.float32)
    donor_port_pad = np.full(
        (count, maximum, e79.PORTS), 1.0 / e79.PORTS, dtype=np.float32
    )
    mass_pad = np.zeros((count, maximum), dtype=np.float32)
    fraction_pad = np.zeros((count, maximum), dtype=np.float32)
    presence_pad = np.zeros((count, maximum), dtype=np.float32)
    delta_pad = np.zeros((count, maximum, 3), dtype=np.float32)
    port_pad = np.full(
        (count, maximum, e79.PORTS), 1.0 / e79.PORTS, dtype=np.float32
    )
    mask = np.zeros((count, maximum), dtype=bool)
    for row in range(count):
        start = int(offsets[row])
        end = int(offsets[row + 1])
        local = end - start
        if local == 0:
            continue
        feature_pad[row, :local] = features[start:end]
        coordinate_pad[row, :local] = cache["coordinates"][start:end]
        donor_port_pad[row, :local] = cache["profiles"][start:end]
        mass_pad[row, :local] = cache["target_mass"][start:end]
        fraction_pad[row, :local] = cache["fractions"][start:end]
        presence_pad[row, :local] = cache["presences"][start:end]
        delta_pad[row, :local] = cache["target_delta"][start:end]
        port_pad[row, :local] = cache["target_port"][start:end]
        mask[row, :local] = True
    return {
        "features": torch.as_tensor(feature_pad, device=device),
        "coordinates": torch.as_tensor(coordinate_pad, device=device),
        "donor_port": torch.as_tensor(donor_port_pad, device=device),
        "mass": torch.as_tensor(mass_pad, device=device),
        "fraction": torch.as_tensor(fraction_pad, device=device),
        "presence": torch.as_tensor(presence_pad, device=device),
        "delta": torch.as_tensor(delta_pad, device=device),
        "port": torch.as_tensor(port_pad, device=device),
        "mask": torch.as_tensor(mask, device=device),
    }


def feature_normalization(
    tensors: dict[str, torch.Tensor], fit_rows: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.as_tensor(fit_rows, device=tensors["features"].device)
    selected = tensors["features"][rows]
    mask = tensors["mask"][rows]
    values = selected[mask]
    mean = values.mean(dim=0)
    std = values.std(dim=0)
    std = torch.where(std < 1e-4, torch.ones_like(std), std)
    return mean, std


def normalized_features(
    tensors: dict[str, torch.Tensor], mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    value = (tensors["features"] - mean) / std
    return torch.clamp(value, -8.0, 8.0)


def selector_loss(
    model: PathSetSelector,
    features: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    rows: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    local_features = features[rows]
    mask = tensors["mask"][rows]
    target_mass = tensors["mass"][rows]
    target_delta = tensors["delta"][rows]
    target_port = tensors["port"][rows]
    score, delta, port, point_logit = model(local_features, mask)
    masked_score = score.masked_fill(~mask, -1e4)
    target_total = target_mass.sum(dim=1)
    valid_point = target_total > 0.0
    target_distribution = target_mass / target_total[:, None].clamp_min(TINY)
    listwise = -(
        target_distribution * torch.log_softmax(masked_score, dim=1)
    ).sum(dim=1)
    listwise = listwise[valid_point].mean() if valid_point.any() else score.sum() * 0.0

    positive = target_mass > 0.0
    candidate_bce = F.binary_cross_entropy_with_logits(
        score[mask], positive[mask].to(score.dtype), reduction="none"
    )
    candidate_weight = torch.where(
        positive[mask], torch.full_like(candidate_bce, 3.0), torch.ones_like(candidate_bce)
    )
    candidate_bce = (candidate_bce * candidate_weight).mean()
    point_bce = F.binary_cross_entropy_with_logits(
        point_logit, valid_point.to(point_logit.dtype)
    )

    if positive.any():
        mass_weight = target_mass[positive]
        mass_weight = mass_weight / mass_weight.mean().clamp_min(TINY)
        delta_loss = (
            F.smooth_l1_loss(delta[positive], target_delta[positive], reduction="none")
            .mean(dim=1)
            .mul(mass_weight)
            .mean()
        )
        port_cosine = (
            torch.sqrt(port[positive].clamp_min(0.0))
            * torch.sqrt(target_port[positive].clamp_min(0.0))
        ).sum(dim=1)
        port_loss = ((1.0 - port_cosine) * mass_weight).mean()
    else:
        delta_loss = score.sum() * 0.0
        port_loss = score.sum() * 0.0
    total = (
        listwise
        + config.candidate_bce_weight * candidate_bce
        + config.point_bce_weight * point_bce
        + config.delta_weight * delta_loss
        + config.port_weight * port_loss
    )
    return total, {
        "total": float(total.detach().item()),
        "listwise": float(listwise.detach().item()),
        "candidate_bce": float(candidate_bce.detach().item()),
        "point_bce": float(point_bce.detach().item()),
        "delta": float(delta_loss.detach().item()),
        "port": float(port_loss.detach().item()),
    }


@torch.inference_mode()
def evaluate_loss(
    model: PathSetSelector,
    features: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    rows: np.ndarray,
    config: Config,
    batch_size: int,
) -> float:
    model.eval()
    values = []
    for start in range(0, len(rows), batch_size):
        batch = torch.as_tensor(
            rows[start : start + batch_size], device=features.device
        )
        loss, _ = selector_loss(model, features, tensors, batch, config)
        values.append(float(loss.item()) * len(batch))
    return float(np.sum(values) / max(len(rows), 1))


def train_model(
    tensors: dict[str, torch.Tensor],
    fit_rows: np.ndarray,
    validation_rows: np.ndarray | None,
    config: Config,
    batch_size: int,
    seed: int,
    epochs: int,
) -> tuple[PathSetSelector, torch.Tensor, torch.Tensor, list[dict[str, float]], int]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean, std = feature_normalization(tensors, fit_rows)
    features = normalized_features(tensors, mean, std)
    model = PathSetSelector(features.shape[2], config).to(features.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=config.learning_rate * 0.08
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = epochs
    best_validation = math.inf
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(fit_rows)
        totals = []
        for start in range(0, len(order), batch_size):
            rows = torch.as_tensor(
                order[start : start + batch_size], device=features.device
            )
            optimizer.zero_grad(set_to_none=True)
            loss, parts = selector_loss(model, features, tensors, rows, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            totals.append(parts["total"])
        scheduler.step()
        row = {"epoch": float(epoch), "train_loss": float(np.mean(totals))}
        if (
            validation_rows is not None
            and epoch >= config.minimum_epochs
            and epoch % config.validation_interval == 0
        ):
            validation = evaluate_loss(
                model, features, tensors, validation_rows, config, batch_size
            )
            row["validation_loss"] = validation
            if validation < best_validation:
                best_validation = validation
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
        history.append(row)
        if epoch % 20 == 0 or epoch == epochs:
            message = f"epoch={epoch}/{epochs} train={row['train_loss']:.5f}"
            if "validation_loss" in row:
                message += f" validation={row['validation_loss']:.5f}"
            print(message, flush=True)
    if validation_rows is not None:
        model.load_state_dict(best_state)
    return model, mean, std, history, best_epoch


@torch.inference_mode()
def infer_model(
    model: PathSetSelector,
    tensors: dict[str, torch.Tensor],
    mean: torch.Tensor,
    std: torch.Tensor,
    rows: np.ndarray,
    batch_size: int,
) -> dict[str, np.ndarray]:
    features = normalized_features(tensors, mean, std)
    model.eval()
    score_parts = []
    delta_parts = []
    port_parts = []
    point_parts = []
    for start in range(0, len(rows), batch_size):
        local_rows = torch.as_tensor(
            rows[start : start + batch_size], device=features.device
        )
        score, delta, port, point = model(
            features[local_rows], tensors["mask"][local_rows]
        )
        score_parts.append(score.cpu().numpy())
        delta_parts.append(delta.cpu().numpy())
        port_parts.append(port.cpu().numpy())
        point_parts.append(torch.sigmoid(point).cpu().numpy())
    return {
        "score": np.concatenate(score_parts),
        "delta": np.concatenate(delta_parts),
        "port": np.concatenate(port_parts),
        "point_probability": np.concatenate(point_parts),
    }


def assemble(
    tensors: dict[str, torch.Tensor],
    held_rows: np.ndarray,
    prediction: dict[str, np.ndarray],
    spec: RenderSpec,
) -> tuple[np.ndarray, np.ndarray]:
    coordinate_all = tensors["coordinates"].cpu().numpy()[held_rows]
    donor_port_all = tensors["donor_port"].cpu().numpy()[held_rows]
    mask_all = tensors["mask"].cpu().numpy()[held_rows]
    fraction_all = tensors["fraction"].cpu().numpy()[held_rows]
    presence_all = tensors["presence"].cpu().numpy()[held_rows]
    output_coordinate = np.zeros((len(held_rows), 24, 3), dtype=np.float32)
    output_coefficient = np.zeros((len(held_rows), 24, e79.PORTS), dtype=np.complex64)
    for row in range(len(held_rows)):
        valid = mask_all[row]
        if not np.any(valid) or prediction["point_probability"][row] < 0.30:
            continue
        model_score = prediction["score"][row, valid]
        base_score = (
            np.log(np.maximum(fraction_all[row, valid], 1e-10))
            + 0.5
            * np.log(np.maximum(presence_all[row, valid], 1e-8))
        )
        base_score = (base_score - np.mean(base_score)) / max(
            float(np.std(base_score)), 1e-6
        )
        model_score = (model_score - np.mean(model_score)) / max(
            float(np.std(model_score)), 1e-6
        )
        score = (1.0 - spec.model_scale) * base_score + spec.model_scale * model_score
        mass = np.exp(score - np.max(score))
        mass /= max(float(np.sum(mass)), TINY)
        selected = np.argsort(-mass, kind="stable")[: spec.top_k]
        count = min(len(selected), 24)
        selected = selected[:count]
        coordinate = np.array(coordinate_all[row, valid][selected], copy=True)
        coordinate += spec.support_scale * prediction["delta"][row, valid][selected]
        coordinate %= np.asarray([MH, MV, S_SC], dtype=np.float32)
        donor_amplitude = np.sqrt(
            np.maximum(donor_port_all[row, valid][selected], 0.0)
        )
        predicted_amplitude = np.sqrt(
            np.maximum(prediction["port"][row, valid][selected], 0.0)
        )
        amplitude = (
            (1.0 - spec.port_scale) * donor_amplitude
            + spec.port_scale * predicted_amplitude
        )
        profile = amplitude**2
        profile /= np.maximum(profile.sum(axis=1, keepdims=True), TINY)
        output_coordinate[row, :count] = coordinate
        output_coefficient[row, :count] = np.sqrt(
            np.maximum(mass[selected, None] * profile, 0.0)
        ).astype(np.complex64)
    return output_coordinate, output_coefficient


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    config = Config(epochs=args.epochs, hidden=args.hidden, layers=args.layers)
    device = torch.device(args.device)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    geodesic = np.load(GEODESIC_CACHE, allow_pickle=False)["train_features"].astype(
        np.float32
    )
    component = np.load(COMPONENTS, allow_pickle=False)
    true_coordinates = component["coordinates"].astype(np.float32)
    true_coefficients = component["coefficients"].astype(np.complex64)
    point_power = component["point_power"].astype(np.float64)
    specs = render_specs()
    variants = {
        f"{group}_{spec.name}": (
            np.zeros((len(query_ids), 24, 3), dtype=np.float32),
            np.zeros((len(query_ids), 24, e79.PORTS), dtype=np.complex64),
        )
        for group in ("local", "global_local")
        for spec in specs
    }
    training_rows = []
    target_audits = []
    for outer in range(5):
        cache = load_candidate_cache(outer)
        soft_mass, soft_delta, soft_port = soft_path_tube_targets(
            cache,
            query_ids,
            true_coordinates,
            true_coefficients,
            point_power,
        )
        cache["target_mass"] = soft_mass
        cache["target_delta"] = soft_delta
        cache["target_port"] = soft_port
        target_audit = audit_soft_targets(
            cache,
            query_ids,
            true_coordinates,
            true_coefficients,
            point_power,
            soft_mass,
            soft_delta,
            soft_port,
        )
        target_audits.append({"outer": outer, **target_audit})
        held_rows = np.flatnonzero(fold_id == outer)
        fit_rows = np.flatnonzero(fold_id != outer)
        validation_fold = (outer + 1) % 5
        validation_rows = np.flatnonzero(fold_id == validation_fold)
        inner_train_rows = np.flatnonzero(
            (fold_id != outer) & (fold_id != validation_fold)
        )
        for group, include_map in (("local", False), ("global_local", True)):
            feature = e79.expanded_features(cache, geodesic, query_ids, include_map)
            tensors = pad_cache(cache, feature, device)
            _, _, _, screen_history, best_epoch = train_model(
                tensors,
                inner_train_rows,
                validation_rows,
                config,
                args.batch_size,
                config.random_state + 1000 * outer + (100 if include_map else 0),
                config.epochs,
            )
            refit_epochs = max(best_epoch, config.minimum_epochs)
            model, mean, std, refit_history, _ = train_model(
                tensors,
                fit_rows,
                None,
                config,
                args.batch_size,
                config.random_state + 5000 + 1000 * outer + (100 if include_map else 0),
                refit_epochs,
            )
            prediction = infer_model(
                model, tensors, mean, std, held_rows, args.batch_size
            )
            for spec in specs:
                coordinate, coefficient = assemble(
                    tensors, held_rows, prediction, spec
                )
                target_coordinate, target_coefficient = variants[
                    f"{group}_{spec.name}"
                ]
                target_coordinate[held_rows] = coordinate
                target_coefficient[held_rows] = coefficient
            training_rows.append(
                {
                    "outer": outer,
                    "group": group,
                    "validation_fold": validation_fold,
                    "best_epoch": best_epoch,
                    "refit_epochs": refit_epochs,
                    "screen_history": screen_history,
                    "refit_history": refit_history,
                    "held_point_probability_mean": float(
                        np.mean(prediction["point_probability"])
                    ),
                }
            )
            print(
                f"outer={outer} group={group} best_epoch={best_epoch} "
                f"point_p={np.mean(prediction['point_probability']):.4f}",
                flush=True,
            )
            del tensors, model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    np.savez_compressed(
        PREDICTIONS,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        **{
            f"{name}_{part}": value[index]
            for name, value in variants.items()
            for index, part in enumerate(("coordinates", "coefficients"))
        },
    )
    results = e79.evaluate_variants(
        variants, query_ids, fold_id, device, args.batch_size
    )
    ranking = sorted(
        ({"name": name, **row} for name, row in results.items()),
        key=lambda row: float(row["pas_pdp_mean"]),
        reverse=True,
    )
    best_local = next(row for row in ranking if row["name"].startswith("local_"))
    best_global = next(
        row for row in ranking if row["name"].startswith("global_local_")
    )
    payload = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "Soft path-tube labels remove contradictory supervision among equivalent "
            "nearby candidates, while the neural set ranker remains anchored to the "
            "fit-only lifecycle heuristic."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "validation": "nested fixed spatial holes with one inner fold selecting epoch",
        "config": asdict(config),
        "soft_path_tube": {
            "cost_limit": SOFT_TUBE_COST_LIMIT,
            "bandwidth": SOFT_TUBE_BANDWIDTH,
        },
        "target_audits": target_audits,
        "training": training_rows,
        "best_local": best_local,
        "best_global_local": best_global,
        "global_map_margin": float(
            best_global["pas_pdp_mean"] - best_local["pas_pdp_mean"]
        ),
        "top_candidates": ranking[:20],
        "prediction_cache": PATHS.relative(PREDICTIONS),
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "best_local": best_local,
                "best_global_local": best_global,
                "global_map_margin": payload["global_map_margin"],
                "seconds": payload["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
