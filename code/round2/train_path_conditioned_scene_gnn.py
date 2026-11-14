#!/usr/bin/env python3
"""Train a path-conditioned scene GNN over fit-only lifecycle candidates."""

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

import analyze_global_local_path_lifecycle as e79
import train_global_local_path_set_selector as e82
from data import load_positions
from project_paths import PATHS


VERSION = "Round2-E83-path-conditioned-scene-gnn"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "path_conditioned_scene_gnn"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_path_conditioned_scene_gnn"
REPORT = ANALYSIS_DIR / "path_conditioned_scene_gnn.json"
PREDICTIONS = ARTIFACT_DIR / "oof_path_conditioned_scene_gnn_predictions.npz"

ROUTE_CACHE = (
    PATHS.artifacts
    / "round2"
    / "route_topology"
    / "round2_route_topology_features.npz"
)
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
COMPONENTS = (
    PATHS.artifacts
    / "round2"
    / "v4_path_component_transport"
    / "train_components_k24_os2_ref2.npz"
)
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    hidden: int = 64
    heads: int = 4
    graph_layers: int = 2
    dropout: float = 0.08
    epochs: int = 80
    minimum_epochs: int = 25
    validation_interval: int = 5
    learning_rate: float = 7e-4
    weight_decay: float = 2e-4
    residual_limit: float = 2.0
    residual_penalty: float = 0.02
    random_state: int = 2026080793


@dataclass(frozen=True)
class RenderSpec:
    top_k: int
    residual_scale: float

    @property
    def name(self) -> str:
        return f"k{self.top_k}_r{self.residual_scale:.2f}"


class GridGraphBlock(nn.Module):
    """Shared message passing on a small regular graph."""

    def __init__(self, hidden: int, dropout: float, circular_columns: bool) -> None:
        super().__init__()
        self.circular_columns = circular_columns
        self.self_linear = nn.Linear(hidden, hidden)
        self.neighbor_linear = nn.Linear(hidden, hidden)
        self.output = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        neighbor = torch.zeros_like(value)
        degree = torch.zeros_like(value[..., :1])
        neighbor[:, 1:] += value[:, :-1]
        degree[:, 1:] += 1.0
        neighbor[:, :-1] += value[:, 1:]
        degree[:, :-1] += 1.0
        if self.circular_columns:
            neighbor += torch.roll(value, 1, dims=2)
            neighbor += torch.roll(value, -1, dims=2)
            degree += 2.0
        else:
            neighbor[:, :, 1:] += value[:, :, :-1]
            degree[:, :, 1:] += 1.0
            neighbor[:, :, :-1] += value[:, :, 1:]
            degree[:, :, :-1] += 1.0
        message = self.self_linear(value) + self.neighbor_linear(
            neighbor / degree.clamp_min(1.0)
        )
        return self.norm(value + self.output(message))


class PathConditionedRanker(nn.Module):
    def __init__(
        self,
        candidate_dim: int,
        route_dim: int,
        panorama_dim: int,
        config: Config,
        use_scene: bool,
    ) -> None:
        super().__init__()
        self.use_scene = use_scene
        self.residual_limit = config.residual_limit
        self.candidate_input = nn.Sequential(
            nn.Linear(candidate_dim, config.hidden),
            nn.GELU(),
            nn.LayerNorm(config.hidden),
        )
        if use_scene:
            self.route_input = nn.Sequential(
                nn.Linear(route_dim, config.hidden),
                nn.GELU(),
                nn.LayerNorm(config.hidden),
            )
            self.panorama_input = nn.Sequential(
                nn.Linear(panorama_dim, config.hidden),
                nn.GELU(),
                nn.LayerNorm(config.hidden),
            )
            self.route_blocks = nn.ModuleList(
                GridGraphBlock(config.hidden, config.dropout, False)
                for _ in range(config.graph_layers)
            )
            self.panorama_blocks = nn.ModuleList(
                GridGraphBlock(config.hidden, config.dropout, True)
                for _ in range(config.graph_layers)
            )
            self.cross_attention = nn.MultiheadAttention(
                config.hidden,
                config.heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.cross_norm = nn.LayerNorm(config.hidden)
        self.context = nn.Sequential(
            nn.Linear(4 * config.hidden, 2 * config.hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * config.hidden, config.hidden),
            nn.GELU(),
            nn.LayerNorm(config.hidden),
        )
        self.score = nn.Linear(config.hidden, 1)

    def forward(
        self,
        candidate: torch.Tensor,
        candidate_mask: torch.Tensor,
        route: torch.Tensor,
        panorama: torch.Tensor,
    ) -> torch.Tensor:
        candidate_hidden = self.candidate_input(candidate)
        weight = candidate_mask.to(candidate_hidden.dtype)
        candidate_pool = (
            (candidate_hidden * weight[..., None]).sum(dim=1)
            / weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        )
        candidate_pool = candidate_pool[:, None].expand_as(candidate_hidden)
        if self.use_scene:
            route_hidden = self.route_input(route)
            panorama_hidden = self.panorama_input(panorama)
            for block in self.route_blocks:
                route_hidden = block(route_hidden)
            for block in self.panorama_blocks:
                panorama_hidden = block(panorama_hidden)
            scene_nodes = torch.cat(
                (
                    route_hidden.flatten(1, 2),
                    panorama_hidden.flatten(1, 2),
                ),
                dim=1,
            )
            attended, _ = self.cross_attention(
                candidate_hidden,
                scene_nodes,
                scene_nodes,
                need_weights=False,
            )
            attended = self.cross_norm(candidate_hidden + attended)
            scene_pool = scene_nodes.mean(dim=1)[:, None].expand_as(candidate_hidden)
        else:
            attended = torch.zeros_like(candidate_hidden)
            scene_pool = torch.zeros_like(candidate_hidden)
        hidden = self.context(
            torch.cat(
                (candidate_hidden, candidate_pool, attended, scene_pool),
                dim=2,
            )
        )
        return self.residual_limit * torch.tanh(self.score(hidden)[..., 0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--graph-layers", type=int, default=2)
    return parser.parse_args()


def render_specs() -> tuple[RenderSpec, ...]:
    return tuple(
        RenderSpec(top_k, residual_scale)
        for top_k in (16, 24)
        for residual_scale in (0.25, 0.50, 1.00)
    )


def route_node_features(
    query_ids: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    if split not in {"train", "test"}:
        raise ValueError(f"unsupported route feature split: {split}")
    cached = np.load(ROUTE_CACHE, allow_pickle=False)
    hit = cached[f"{split}_route_hit"][query_ids].astype(np.float32)
    occupancy = cached[f"{split}_route_occupancy"][query_ids].astype(np.float32)
    count = cached[f"{split}_route_count"][query_ids].astype(np.float32)
    offsets = cached["fan_offsets_m"].astype(np.float32)
    panorama = cached[f"{split}_panorama"][query_ids].astype(np.float32)
    heights = cached["panorama_heights_m"].astype(np.float32)

    query = positions[query_ids].astype(np.float32)
    source = bs_positions[site[query_ids]].astype(np.float32)
    delta = query[:, :2] - source[:, :2]
    direct = np.maximum(np.linalg.norm(delta, axis=1), 1e-6)
    unit = delta / direct[:, None]
    perpendicular = np.column_stack((-unit[:, 1], unit[:, 0])).astype(np.float32)
    fraction = (
        (np.arange(hit.shape[2], dtype=np.float32) + 0.5) / hit.shape[2]
    )
    target = query[:, None, :2] + perpendicular[:, None] * offsets[None, :, None]
    node_xy = (
        source[:, None, None, :2]
        + fraction[None, None, :, None]
        * (target[:, :, None, :] - source[:, None, None, :2])
    )
    cumulative_hit = np.cumsum(hit, axis=2) / np.maximum(
        np.sum(hit, axis=2, keepdims=True), 1.0
    )
    route = np.stack(
        (
            hit,
            occupancy,
            np.broadcast_to(
                np.log1p(count)[:, :, None] / np.log(12.0), hit.shape
            ),
            np.broadcast_to(offsets[None, :, None] / 12.0, hit.shape),
            np.broadcast_to(fraction[None, None, :], hit.shape),
            np.broadcast_to(
                fraction[None, None, :] * direct[:, None, None] / 200.0,
                hit.shape,
            ),
            node_xy[..., 0] / 200.0,
            node_xy[..., 1] / 200.0,
            (node_xy[..., 0] - query[:, None, None, 0]) / 100.0,
            (node_xy[..., 1] - query[:, None, None, 1]) / 100.0,
            cumulative_hit,
        ),
        axis=3,
    ).astype(np.float32)

    angles = np.linspace(0.0, 2.0 * np.pi, panorama.shape[2], endpoint=False)
    direction = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
    endpoint = (
        query[:, None, None, :2]
        + 48.0
        * panorama[..., None]
        * direction[None, None, :, :]
    )
    panorama_nodes = np.stack(
        (
            panorama,
            1.0 - panorama,
            np.broadcast_to(heights[None, :, None] / 8.0, panorama.shape),
            np.broadcast_to(direction[None, None, :, 0], panorama.shape),
            np.broadcast_to(direction[None, None, :, 1], panorama.shape),
            endpoint[..., 0] / 200.0,
            endpoint[..., 1] / 200.0,
            (endpoint[..., 0] - query[:, None, None, 0]) / 48.0,
            (endpoint[..., 1] - query[:, None, None, 1]) / 48.0,
        ),
        axis=3,
    ).astype(np.float32)
    return route, panorama_nodes


def add_base_score(tensors: dict[str, torch.Tensor]) -> None:
    mask = tensors["mask"]
    raw = torch.log(tensors["fraction"].clamp_min(1e-10))
    raw = raw + 0.5 * torch.log(tensors["presence"].clamp_min(1e-8))
    weight = mask.to(raw.dtype)
    count = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (raw * weight).sum(dim=1, keepdim=True) / count
    variance = ((raw - mean).square() * weight).sum(dim=1, keepdim=True) / count
    base = (raw - mean) / torch.sqrt(variance.clamp_min(1e-6))
    tensors["base"] = base.masked_fill(~mask, 0.0)


def normalize_values(
    value: torch.Tensor, fit_rows: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = torch.as_tensor(fit_rows, device=value.device)
    selected = value[rows].reshape(-1, value.shape[-1])
    mean = selected.mean(dim=0)
    std = selected.std(dim=0)
    std = torch.where(std < 1e-4, torch.ones_like(std), std)
    return torch.clamp((value - mean) / std, -8.0, 8.0), mean, std


def apply_normalization(
    value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return torch.clamp((value - mean) / std, -8.0, 8.0)


def selector_loss(
    model: PathConditionedRanker,
    candidate: torch.Tensor,
    route: torch.Tensor,
    panorama: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    rows: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = tensors["mask"][rows]
    residual = model(candidate[rows], mask, route[rows], panorama[rows])
    score = (tensors["base"][rows] + residual).masked_fill(~mask, -1e4)
    target_mass = tensors["mass"][rows]
    target_total = target_mass.sum(dim=1)
    valid = target_total > 0.0
    target = target_mass / target_total[:, None].clamp_min(TINY)
    listwise = -(target * torch.log_softmax(score, dim=1)).sum(dim=1)
    listwise = listwise[valid].mean() if valid.any() else score.sum() * 0.0
    residual_penalty = residual[mask].square().mean()
    total = listwise + config.residual_penalty * residual_penalty
    return total, {
        "total": float(total.detach().item()),
        "listwise": float(listwise.detach().item()),
        "residual_penalty": float(residual_penalty.detach().item()),
    }


@torch.inference_mode()
def evaluate_loss(
    model: PathConditionedRanker,
    candidate: torch.Tensor,
    route: torch.Tensor,
    panorama: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    rows: np.ndarray,
    config: Config,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    for start in range(0, len(rows), batch_size):
        batch = torch.as_tensor(
            rows[start : start + batch_size], device=candidate.device
        )
        loss, _ = selector_loss(
            model, candidate, route, panorama, tensors, batch, config
        )
        total += float(loss.item()) * len(batch)
    return total / max(len(rows), 1)


def train_model(
    tensors: dict[str, torch.Tensor],
    route_raw: torch.Tensor,
    panorama_raw: torch.Tensor,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray | None,
    config: Config,
    batch_size: int,
    seed: int,
    epochs: int,
    use_scene: bool,
) -> tuple[
    PathConditionedRanker,
    dict[str, torch.Tensor],
    list[dict[str, float]],
    int,
]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    candidate_mean, candidate_std = e82.feature_normalization(tensors, fit_rows)
    candidate = e82.normalized_features(tensors, candidate_mean, candidate_std)
    route, route_mean, route_std = normalize_values(route_raw, fit_rows)
    panorama, panorama_mean, panorama_std = normalize_values(panorama_raw, fit_rows)
    model = PathConditionedRanker(
        candidate.shape[-1],
        route.shape[-1],
        panorama.shape[-1],
        config,
        use_scene,
    ).to(candidate.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=config.learning_rate * 0.08,
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = config.minimum_epochs
    best_validation = math.inf
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(fit_rows)
        losses = []
        for start in range(0, len(order), batch_size):
            rows = torch.as_tensor(
                order[start : start + batch_size], device=candidate.device
            )
            optimizer.zero_grad(set_to_none=True)
            loss, parts = selector_loss(
                model, candidate, route, panorama, tensors, rows, config
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(parts["total"])
        scheduler.step()
        row = {"epoch": float(epoch), "train_loss": float(np.mean(losses))}
        if (
            validation_rows is not None
            and epoch >= config.minimum_epochs
            and epoch % config.validation_interval == 0
        ):
            validation = evaluate_loss(
                model,
                candidate,
                route,
                panorama,
                tensors,
                validation_rows,
                config,
                batch_size,
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
    stats = {
        "candidate_mean": candidate_mean,
        "candidate_std": candidate_std,
        "route_mean": route_mean,
        "route_std": route_std,
        "panorama_mean": panorama_mean,
        "panorama_std": panorama_std,
    }
    return model, stats, history, best_epoch


@torch.inference_mode()
def infer_model(
    model: PathConditionedRanker,
    tensors: dict[str, torch.Tensor],
    route_raw: torch.Tensor,
    panorama_raw: torch.Tensor,
    stats: dict[str, torch.Tensor],
    candidate_rows: np.ndarray,
    scene_rows: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    candidate = e82.normalized_features(
        tensors, stats["candidate_mean"], stats["candidate_std"]
    )
    route = apply_normalization(
        route_raw, stats["route_mean"], stats["route_std"]
    )
    panorama = apply_normalization(
        panorama_raw, stats["panorama_mean"], stats["panorama_std"]
    )
    model.eval()
    output = []
    for start in range(0, len(candidate_rows), batch_size):
        rows = torch.as_tensor(
            candidate_rows[start : start + batch_size], device=candidate.device
        )
        graph_rows = torch.as_tensor(
            scene_rows[start : start + batch_size], device=candidate.device
        )
        residual = model(
            candidate[rows],
            tensors["mask"][rows],
            route[graph_rows],
            panorama[graph_rows],
        )
        output.append(residual.cpu().numpy())
    return np.concatenate(output)


def shuffled_scene_rows(
    rows: np.ndarray,
    query_site: np.ndarray,
    seed: int,
) -> np.ndarray:
    output = np.array(rows, copy=True)
    rng = np.random.default_rng(seed)
    for site_value in np.unique(query_site[rows]):
        selected = np.flatnonzero(query_site[rows] == site_value)
        if len(selected) <= 1:
            continue
        shift = int(rng.integers(1, len(selected)))
        output[selected] = rows[selected[np.roll(np.arange(len(selected)), shift)]]
    return output


def assemble(
    tensors: dict[str, torch.Tensor],
    held_rows: np.ndarray,
    residual: np.ndarray,
    spec: RenderSpec,
) -> tuple[np.ndarray, np.ndarray]:
    coordinate = tensors["coordinates"].cpu().numpy()[held_rows]
    profile = tensors["donor_port"].cpu().numpy()[held_rows]
    mask = tensors["mask"].cpu().numpy()[held_rows]
    base = tensors["base"].cpu().numpy()[held_rows]
    output_coordinate = np.zeros((len(held_rows), 24, 3), dtype=np.float32)
    output_coefficient = np.zeros(
        (len(held_rows), 24, e79.PORTS), dtype=np.complex64
    )
    for row in range(len(held_rows)):
        valid = mask[row]
        if not np.any(valid):
            continue
        score = base[row, valid] + spec.residual_scale * residual[row, valid]
        mass = np.exp(score - np.max(score))
        mass /= max(float(np.sum(mass)), TINY)
        selected = np.argsort(-mass, kind="stable")[: spec.top_k]
        count = min(len(selected), 24)
        selected = selected[:count]
        local_profile = np.maximum(profile[row, valid][selected], 0.0)
        local_profile /= np.maximum(local_profile.sum(axis=1, keepdims=True), TINY)
        output_coordinate[row, :count] = coordinate[row, valid][selected]
        output_coefficient[row, :count] = np.sqrt(
            np.maximum(mass[selected, None] * local_profile, 0.0)
        ).astype(np.complex64)
    return output_coordinate, output_coefficient


def combined_folds(row: dict[str, object]) -> np.ndarray:
    return 0.5 * (
        np.asarray(row["pas"]["fold"], dtype=np.float64)
        + np.asarray(row["pdp"]["fold"], dtype=np.float64)
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    config = Config(
        epochs=args.epochs,
        hidden=args.hidden,
        graph_layers=args.graph_layers,
    )
    device = torch.device(args.device)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    setup, positions, _, site, _ = load_positions()
    query_site = site[query_ids]
    route_raw, panorama_raw = route_node_features(
        query_ids, positions, site, setup.bs_positions
    )
    route_tensor = torch.as_tensor(route_raw, device=device)
    panorama_tensor = torch.as_tensor(panorama_raw, device=device)
    component = np.load(COMPONENTS, allow_pickle=False)
    true_coordinates = component["coordinates"].astype(np.float32)
    true_coefficients = component["coefficients"].astype(np.complex64)
    point_power = component["point_power"].astype(np.float64)

    specs = render_specs()
    names = ["heuristic_k24"] + [
        f"{group}_{spec.name}"
        for group in ("local", "scene", "scene_shuffle")
        for spec in specs
    ]
    variants = {
        name: (
            np.zeros((len(query_ids), 24, 3), dtype=np.float32),
            np.zeros((len(query_ids), 24, e79.PORTS), dtype=np.complex64),
        )
        for name in names
    }
    training_rows: list[dict[str, object]] = []
    target_audits: list[dict[str, object]] = []
    for outer in range(5):
        cache = e82.load_candidate_cache(outer)
        soft_mass, soft_delta, soft_port = e82.soft_path_tube_targets(
            cache,
            query_ids,
            true_coordinates,
            true_coefficients,
            point_power,
        )
        cache["target_mass"] = soft_mass
        cache["target_delta"] = soft_delta
        cache["target_port"] = soft_port
        audit = e82.audit_soft_targets(
            cache,
            query_ids,
            true_coordinates,
            true_coefficients,
            point_power,
            soft_mass,
            soft_delta,
            soft_port,
        )
        target_audits.append({"outer": outer, **audit})
        feature = e79.expanded_features(
            cache,
            np.empty((len(positions), 0), dtype=np.float32),
            query_ids,
            False,
        )
        tensors = e82.pad_cache(cache, feature, device)
        add_base_score(tensors)
        held_rows = np.flatnonzero(fold_id == outer)
        fit_rows = np.flatnonzero(fold_id != outer)
        validation_fold = (outer + 1) % 5
        validation_rows = np.flatnonzero(fold_id == validation_fold)
        inner_train_rows = np.flatnonzero(
            (fold_id != outer) & (fold_id != validation_fold)
        )
        zero_residual = np.zeros((len(held_rows), tensors["mask"].shape[1]), np.float32)
        heuristic_spec = RenderSpec(24, 0.0)
        coordinate, coefficient = assemble(
            tensors, held_rows, zero_residual, heuristic_spec
        )
        variants["heuristic_k24"][0][held_rows] = coordinate
        variants["heuristic_k24"][1][held_rows] = coefficient

        for group, use_scene in (("local", False), ("scene", True)):
            _, _, screen_history, best_epoch = train_model(
                tensors,
                route_tensor,
                panorama_tensor,
                inner_train_rows,
                validation_rows,
                config,
                args.batch_size,
                config.random_state + 1000 * outer + (100 if use_scene else 0),
                config.epochs,
                use_scene,
            )
            refit_epochs = max(best_epoch, config.minimum_epochs)
            model, stats, refit_history, _ = train_model(
                tensors,
                route_tensor,
                panorama_tensor,
                fit_rows,
                None,
                config,
                args.batch_size,
                config.random_state + 5000 + 1000 * outer + (100 if use_scene else 0),
                refit_epochs,
                use_scene,
            )
            residual = infer_model(
                model,
                tensors,
                route_tensor,
                panorama_tensor,
                stats,
                held_rows,
                held_rows,
                args.batch_size,
            )
            if use_scene:
                shuffled_rows = shuffled_scene_rows(
                    held_rows,
                    query_site,
                    config.random_state + 9000 + outer,
                )
                shuffled_residual = infer_model(
                    model,
                    tensors,
                    route_tensor,
                    panorama_tensor,
                    stats,
                    held_rows,
                    shuffled_rows,
                    args.batch_size,
                )
            for spec in specs:
                coordinate, coefficient = assemble(
                    tensors, held_rows, residual, spec
                )
                variants[f"{group}_{spec.name}"][0][held_rows] = coordinate
                variants[f"{group}_{spec.name}"][1][held_rows] = coefficient
                if use_scene:
                    coordinate, coefficient = assemble(
                        tensors, held_rows, shuffled_residual, spec
                    )
                    variants[f"scene_shuffle_{spec.name}"][0][held_rows] = coordinate
                    variants[f"scene_shuffle_{spec.name}"][1][held_rows] = coefficient
            training_rows.append(
                {
                    "outer": outer,
                    "group": group,
                    "validation_fold": validation_fold,
                    "best_epoch": best_epoch,
                    "refit_epochs": refit_epochs,
                    "screen_history": screen_history,
                    "refit_history": refit_history,
                    "held_residual_mean_abs": float(np.mean(np.abs(residual))),
                }
            )
            print(
                f"outer={outer} group={group} best_epoch={best_epoch} "
                f"residual={np.mean(np.abs(residual)):.4f}",
                flush=True,
            )
            del model, stats
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del tensors

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
    fixed_names = {
        "heuristic": "heuristic_k24",
        "local": "local_k24_r0.25",
        "scene": "scene_k24_r0.25",
        "scene_shuffle": "scene_shuffle_k24_r0.25",
    }
    fixed = {key: results[name] for key, name in fixed_names.items()}
    scene_fold = combined_folds(fixed["scene"])
    local_fold = combined_folds(fixed["local"])
    shuffle_fold = combined_folds(fixed["scene_shuffle"])
    scene_local_delta = float(
        fixed["scene"]["pas_pdp_mean"] - fixed["local"]["pas_pdp_mean"]
    )
    scene_shuffle_delta = float(
        fixed["scene"]["pas_pdp_mean"]
        - fixed["scene_shuffle"]["pas_pdp_mean"]
    )
    deployment_gate = {
        "minimum_scene_vs_local_delta": 0.002,
        "minimum_scene_vs_local_positive_folds": 4,
        "minimum_scene_vs_shuffle_delta": 0.002,
        "minimum_scene_vs_shuffle_positive_folds": 4,
        "minimum_scene_score": 0.6023741895074426,
        "scene_vs_local_delta": scene_local_delta,
        "scene_vs_local_fold_delta": (scene_fold - local_fold).tolist(),
        "scene_vs_local_positive_folds": int(np.sum(scene_fold > local_fold)),
        "scene_vs_shuffle_delta": scene_shuffle_delta,
        "scene_vs_shuffle_fold_delta": (scene_fold - shuffle_fold).tolist(),
        "scene_vs_shuffle_positive_folds": int(np.sum(scene_fold > shuffle_fold)),
    }
    deployment_gate["passed"] = bool(
        scene_local_delta >= deployment_gate["minimum_scene_vs_local_delta"]
        and deployment_gate["scene_vs_local_positive_folds"]
        >= deployment_gate["minimum_scene_vs_local_positive_folds"]
        and scene_shuffle_delta >= deployment_gate["minimum_scene_vs_shuffle_delta"]
        and deployment_gate["scene_vs_shuffle_positive_folds"]
        >= deployment_gate["minimum_scene_vs_shuffle_positive_folds"]
        and fixed["scene"]["pas_pdp_mean"]
        >= deployment_gate["minimum_scene_score"]
    )
    payload = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "Candidate-specific message passing over the full route and panorama graphs "
            "predicts path survival better than repeating point-level map summaries."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "validation": "nested fixed 500-point five-fold spatial holes",
        "config": asdict(config),
        "scene_graph": {
            "route_shape": list(route_raw.shape),
            "panorama_shape": list(panorama_raw.shape),
            "route_nodes_per_point": int(np.prod(route_raw.shape[1:3])),
            "panorama_nodes_per_point": int(np.prod(panorama_raw.shape[1:3])),
        },
        "target_audits": target_audits,
        "training": training_rows,
        "fixed_comparison": fixed,
        "deployment_gate": deployment_gate,
        "top_candidates": ranking[:20],
        "prediction_cache": PATHS.relative(PREDICTIONS),
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "fixed_comparison": fixed,
                "deployment_gate": deployment_gate,
                "best": ranking[0],
                "seconds": payload["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
