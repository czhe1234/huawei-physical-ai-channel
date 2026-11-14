#!/usr/bin/env python3
"""Learn fit-only path-component transport across Round 2 spatial holes."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.optimize import linear_sum_assignment
from torch import nn

from analyze_v4_path_track_spectral_targets import (
    MH,
    MV,
    S_SC,
    cosine,
    render_noncoherent,
)
from data import load_positions
from project_paths import PATHS
from run_path_component_transport import circular_delta, component_energy
from run_v3_anchored_neural_rbf import (
    Config as ContextConfig,
    FOLD_CACHE,
    OOF_META,
    build_train_contexts,
    set_seed,
    validation_context,
)
from run_v4_group_conditioned_link_field import load_state_features, standardize_state


ARTIFACT_DIR = PATHS.artifacts / "round2" / "v4_learned_component_transport"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v4_learned_component_transport"
REPORT = ANALYSIS_DIR / "learned_component_transport.json"

COMPONENT_CACHE = (
    PATHS.artifacts
    / "round2"
    / "v4_path_component_transport"
    / "train_components_k24_os2_ref2.npz"
)
DESCRIPTORS = (
    PATHS.artifacts / "round2" / "pre_sionna" / "round2_train_spectral_descriptors.npz"
)
TRAIN_PAS = (
    PATHS.artifacts / "round2" / "pre_sionna" / "round2_train_pas_link_support_f16.npy"
)
TRAIN_PDP = (
    PATHS.artifacts / "round2" / "pre_sionna" / "round2_train_pdp_link_support_f16.npy"
)


@dataclass(frozen=True)
class Config:
    paths: int = 16
    state_hidden: int = 128
    delta_hidden: int = 96
    path_hidden: int = 96
    trunk_hidden: int = 192
    dropout: float = 0.05
    horizontal_bound: float = 4.0
    vertical_bound: float = 2.5
    delay_bound: float = 12.0
    log_energy_bound: float = 5.0
    epochs: int = 30
    batch_size: int = 96
    learning_rate: float = 6e-4
    weight_decay: float = 2e-5
    coordinate_weight: float = 0.45
    energy_weight: float = 0.25
    port_weight: float = 0.20
    spectral_weight: float = 0.10
    seed: int = 20_260_805


@dataclass(frozen=True)
class PairTable:
    query_ids: np.ndarray
    donor_ids: np.ndarray
    distance: np.ndarray
    target_coordinates: np.ndarray
    target_energy: np.ndarray
    match_cost: np.ndarray


def periodic_features(coordinates: torch.Tensor) -> torch.Tensor:
    periods = coordinates.new_tensor([MH, MV, S_SC])
    phase = 2.0 * math.pi * coordinates / periods
    return torch.cat(
        [coordinates / periods, torch.sin(phase), torch.cos(phase)], dim=-1
    )


def match_components(
    anchor_coordinates: np.ndarray,
    truth_coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = circular_delta(
        truth_coordinates[None, :, 0], anchor_coordinates[:, None, 0], MH
    )
    v = circular_delta(
        truth_coordinates[None, :, 1], anchor_coordinates[:, None, 1], MV
    )
    d = circular_delta(
        truth_coordinates[None, :, 2], anchor_coordinates[:, None, 2], S_SC
    )
    cost = (h / 1.5) ** 2 + v**2 + (d / 2.0) ** 2
    anchor_rows, truth_columns = linear_sum_assignment(cost)
    order = np.argsort(anchor_rows)
    anchor_rows = anchor_rows[order]
    truth_columns = truth_columns[order]
    delta = np.column_stack(
        [
            h[anchor_rows, truth_columns],
            v[anchor_rows, truth_columns],
            d[anchor_rows, truth_columns],
        ]
    ).astype(np.float32)
    return truth_columns.astype(np.int64), delta, cost[anchor_rows, truth_columns].astype(np.float32)


def pair_cache_path(fold: int, paths: int) -> PATHS.__class__:
    return ARTIFACT_DIR / f"fold{fold}_component_pairs_k{paths}.npz"


def build_pair_table(
    fold: int,
    context_config: ContextConfig,
    positions: np.ndarray,
    site: np.ndarray,
    valid: np.ndarray,
    fit_ids: np.ndarray,
    coordinates: np.ndarray,
    coefficients: np.ndarray,
    paths: int,
) -> PairTable:
    cache_path = pair_cache_path(fold, paths)
    context = build_train_contexts(
        fold, positions, site, valid, fit_ids, context_config
    )
    query_ids = np.repeat(context.query_ids, len(context_config.guard_radii_m))
    donor_ids = context.donor_ids[:, :, 0].reshape(-1)
    distance = context.distances[:, :, 0].reshape(-1)
    if cache_path.is_file():
        cached = np.load(cache_path)
        if np.array_equal(cached["query_ids"], query_ids) and np.array_equal(
            cached["donor_ids"], donor_ids
        ):
            return PairTable(
                query_ids=query_ids.astype(np.int64),
                donor_ids=donor_ids.astype(np.int64),
                distance=distance.astype(np.float32),
                target_coordinates=cached["target_coordinates"].astype(np.float32),
                target_energy=cached["target_energy"].astype(np.float32),
                match_cost=cached["match_cost"].astype(np.float32),
            )
    target_coordinates = np.empty((len(query_ids), paths, 3), dtype=np.float32)
    target_energy = np.empty((len(query_ids), paths, coefficients.shape[2]), dtype=np.float32)
    match_cost = np.empty((len(query_ids), paths), dtype=np.float32)
    for row, (query_id, donor_id) in enumerate(zip(query_ids, donor_ids)):
        anchor = coordinates[donor_id, :paths]
        truth = coordinates[query_id, :paths]
        truth_columns, delta, cost = match_components(anchor, truth)
        target_coordinates[row] = np.mod(
            anchor + delta,
            np.asarray([MH, MV, S_SC], dtype=np.float32),
        )
        target_energy[row] = np.abs(coefficients[query_id, truth_columns]) ** 2
        match_cost[row] = cost
        if (row + 1) % 4000 == 0:
            print(f"component matches fold={fold} {row + 1}/{len(query_ids)}", flush=True)
    np.savez_compressed(
        cache_path,
        query_ids=query_ids.astype(np.int32),
        donor_ids=donor_ids.astype(np.int32),
        distance=distance.astype(np.float32),
        target_coordinates=target_coordinates,
        target_energy=target_energy.astype(np.float16),
        match_cost=match_cost.astype(np.float16),
    )
    return PairTable(
        query_ids,
        donor_ids,
        distance.astype(np.float32),
        target_coordinates,
        target_energy,
        match_cost,
    )


class ComponentTransport(nn.Module):
    def __init__(self, state_size: int, config: Config) -> None:
        super().__init__()
        self.config = config
        self.query_encoder = nn.Sequential(
            nn.Linear(state_size, config.state_hidden),
            nn.SiLU(),
            nn.LayerNorm(config.state_hidden),
            nn.Linear(config.state_hidden, config.state_hidden),
        )
        self.delta_encoder = nn.Sequential(
            nn.Linear(state_size, config.delta_hidden),
            nn.SiLU(),
            nn.LayerNorm(config.delta_hidden),
            nn.Linear(config.delta_hidden, config.delta_hidden),
        )
        self.path_encoder = nn.Sequential(
            nn.Linear(9 + 1 + 8 + 1, config.path_hidden),
            nn.SiLU(),
            nn.LayerNorm(config.path_hidden),
            nn.Linear(config.path_hidden, config.path_hidden),
        )
        spatial_size = 7
        trunk_input = (
            config.state_hidden + config.delta_hidden + config.path_hidden + spatial_size
        )
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input, config.trunk_hidden),
            nn.SiLU(),
            nn.LayerNorm(config.trunk_hidden),
            nn.Dropout(config.dropout),
            nn.Linear(config.trunk_hidden, config.trunk_hidden),
            nn.SiLU(),
        )
        self.coordinate_head = nn.Linear(config.trunk_hidden, 3)
        self.energy_head = nn.Linear(config.trunk_hidden, 1)
        self.port_head = nn.Linear(config.trunk_hidden, 8)
        for head in (self.coordinate_head, self.energy_head, self.port_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        query_state: torch.Tensor,
        donor_state: torch.Tensor,
        query_position: torch.Tensor,
        donor_position: torch.Tensor,
        anchor_coordinates: torch.Tensor,
        anchor_energy: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        query = self.query_encoder(query_state)
        state_delta = self.delta_encoder(donor_state - query_state)
        offset = donor_position - query_position
        distance = torch.linalg.norm(offset, dim=1).clamp_min(1e-4)
        direction = offset / distance[:, None]
        spatial = torch.cat(
            [
                offset / 20.0,
                distance[:, None] / 20.0,
                torch.log1p(distance)[:, None],
                direction,
                (distance[:, None] / 10.0).square(),
            ],
            dim=1,
        )
        total_energy = anchor_energy.sum(dim=2).clamp_min(1e-20)
        point_energy = total_energy.sum(dim=1, keepdim=True).clamp_min(1e-20)
        fraction = total_energy / point_energy
        port_fraction = anchor_energy / total_energy[:, :, None]
        rank = torch.linspace(
            0.0, 1.0, anchor_coordinates.shape[1], device=anchor_coordinates.device
        )[None, :, None].expand(len(anchor_coordinates), -1, -1)
        path = self.path_encoder(
            torch.cat(
                [
                    periodic_features(anchor_coordinates),
                    torch.log10(total_energy.clamp_min(1e-20))[:, :, None].clamp(-12.0, 12.0) / 12.0,
                    port_fraction,
                    rank,
                ],
                dim=2,
            )
        )
        count = anchor_coordinates.shape[1]
        shared = torch.cat(
            [
                query[:, None].expand(-1, count, -1),
                state_delta[:, None].expand(-1, count, -1),
                path,
                spatial[:, None].expand(-1, count, -1),
            ],
            dim=2,
        )
        latent = self.trunk(shared)
        raw_coordinate = torch.tanh(self.coordinate_head(latent))
        bounds = anchor_coordinates.new_tensor(
            [
                self.config.horizontal_bound,
                self.config.vertical_bound,
                self.config.delay_bound,
            ]
        )
        delta = raw_coordinate * bounds
        periods = anchor_coordinates.new_tensor([MH, MV, S_SC])
        coordinates = torch.remainder(anchor_coordinates + delta, periods)
        log_energy = self.config.log_energy_bound * torch.tanh(
            self.energy_head(latent).squeeze(-1)
        )
        predicted_total = total_energy * torch.exp(log_energy)
        anchor_logits = torch.log(port_fraction.clamp_min(1e-8))
        port = torch.softmax(anchor_logits + self.port_head(latent), dim=2)
        energy = predicted_total[:, :, None] * port
        return {
            "coordinates": coordinates,
            "energy": energy,
            "delta": delta,
            "log_energy": log_energy,
            "port": port,
            "anchor_fraction": fraction,
        }


def forward_rows(
    model: ComponentTransport,
    rows: torch.Tensor,
    table: PairTable,
    state: torch.Tensor,
    positions: torch.Tensor,
    coordinates: torch.Tensor,
    energy: torch.Tensor,
) -> dict[str, torch.Tensor]:
    query_ids = torch.as_tensor(table.query_ids[rows.cpu().numpy()], device=state.device)
    donor_ids = torch.as_tensor(table.donor_ids[rows.cpu().numpy()], device=state.device)
    return model(
        state[query_ids],
        state[donor_ids],
        positions[query_ids, :2],
        positions[donor_ids, :2],
        coordinates[donor_ids],
        energy[donor_ids],
    )


def training_loss(
    output: dict[str, torch.Tensor],
    target_coordinates: torch.Tensor,
    target_energy: torch.Tensor,
    match_cost: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    true_total = target_energy.sum(dim=2).clamp_min(1e-20)
    point_total = true_total.sum(dim=1, keepdim=True).clamp_min(1e-20)
    weight = torch.sqrt(true_total / point_total).clamp_min(0.02)
    weight /= weight.sum(dim=1, keepdim=True)
    periods = target_coordinates.new_tensor([MH, MV, S_SC])
    scales = target_coordinates.new_tensor([1.5, 1.0, 2.0])
    coordinate_delta = torch.remainder(
        output["coordinates"] - target_coordinates + periods / 2.0, periods
    ) - periods / 2.0
    coordinate_loss = functional.smooth_l1_loss(
        coordinate_delta / scales,
        torch.zeros_like(coordinate_delta),
        reduction="none",
        beta=0.5,
    ).mean(dim=2)
    coordinate_loss = torch.sum(weight * coordinate_loss, dim=1).mean()
    predicted_total = output["energy"].sum(dim=2).clamp_min(1e-20)
    energy_delta = torch.log(predicted_total) - torch.log(true_total)
    energy_loss = torch.sum(
        weight * functional.smooth_l1_loss(
            energy_delta,
            torch.zeros_like(energy_delta),
            reduction="none",
            beta=0.5,
        ),
        dim=1,
    ).mean()
    true_port = target_energy / true_total[:, :, None]
    port_loss = -torch.sum(true_port * torch.log(output["port"].clamp_min(1e-8)), dim=2)
    port_loss = torch.sum(weight * port_loss, dim=1).mean()
    # A matched path with very large support cost is intrinsically ambiguous; cap its leverage.
    ambiguity = torch.mean(torch.clamp(match_cost, max=25.0)) / 25.0
    loss = (
        config.coordinate_weight * coordinate_loss
        + config.energy_weight * energy_loss
        + config.port_weight * port_loss
        + 0.01 * ambiguity
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "coordinate": float(coordinate_loss.detach().cpu()),
        "energy": float(energy_loss.detach().cpu()),
        "port": float(port_loss.detach().cpu()),
    }


@torch.no_grad()
def predict_components(
    model: ComponentTransport,
    query_ids: np.ndarray,
    donor_ids: np.ndarray,
    state: torch.Tensor,
    positions: torch.Tensor,
    coordinates: torch.Tensor,
    energy: torch.Tensor,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    output_coordinates = np.empty((len(query_ids), coordinates.shape[1], 3), dtype=np.float32)
    output_energy = np.empty((len(query_ids), coordinates.shape[1], 8), dtype=np.float32)
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        query = torch.as_tensor(query_ids[start:end], device=state.device)
        donor = torch.as_tensor(donor_ids[start:end], device=state.device)
        value = model(
            state[query],
            state[donor],
            positions[query, :2],
            positions[donor, :2],
            coordinates[donor],
            energy[donor],
        )
        output_coordinates[start:end] = value["coordinates"].cpu().numpy()
        output_energy[start:end] = value["energy"].cpu().numpy()
    return output_coordinates, output_energy


def evaluate_components(
    predicted_coordinates: np.ndarray,
    predicted_energy: np.ndarray,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    truth_pas = np.load(TRAIN_PAS, mmap_mode="r")
    truth_pdp = np.load(TRAIN_PDP, mmap_mode="r")
    pas_values = np.empty(len(query_ids), dtype=np.float64)
    pdp_values = np.empty(len(query_ids), dtype=np.float64)
    coefficients = np.sqrt(np.maximum(predicted_energy, 0.0)).astype(np.complex64)
    for start in range(0, len(query_ids), batch_size):
        end = min(start + batch_size, len(query_ids))
        _, pas, pdp = render_noncoherent(
            predicted_coordinates[start:end], coefficients[start:end], device
        )
        pas_values[start:end] = cosine(
            pas, np.asarray(truth_pas[query_ids[start:end]], dtype=np.float32), 1
        ).mean(axis=(1, 2), dtype=np.float64)
        pdp_values[start:end] = cosine(
            pdp, np.asarray(truth_pdp[query_ids[start:end]], dtype=np.float32), 3
        ).mean(axis=(1, 2), dtype=np.float64)
    return {
        "pas": float(np.mean(pas_values)),
        "pdp": float(np.mean(pdp_values)),
        "mean": float(np.mean(0.5 * (pas_values + pdp_values))),
        "fold_pas": [
            float(np.mean(pas_values[fold_id == fold])) for fold in np.unique(fold_id)
        ],
        "fold_pdp": [
            float(np.mean(pdp_values[fold_id == fold])) for fold in np.unique(fold_id)
        ],
    }


def train_fold(
    fold: int,
    config: Config,
    context_config: ContextConfig,
    device: torch.device,
    train_state_raw: np.ndarray,
    test_state_raw: np.ndarray,
    coordinates_numpy: np.ndarray,
    energy_numpy: np.ndarray,
) -> dict[str, object]:
    set_seed(config.seed + fold * 1009)
    _, positions_numpy, _, site, _ = load_positions()
    valid = np.load(DESCRIPTORS)["valid"].astype(bool)
    folds = np.load(FOLD_CACHE)
    fit_ids = folds[f"fold{fold}_fit"].astype(np.int64)
    metadata = np.load(OOF_META)
    all_query_ids = metadata["query_ids"].astype(np.int64)
    all_fold_id = metadata["fold_id"].astype(np.int8)
    validation = validation_context(fold, all_query_ids, all_fold_id)
    table = build_pair_table(
        fold,
        context_config,
        positions_numpy,
        site,
        valid,
        fit_ids,
        coordinates_numpy,
        np.sqrt(np.maximum(energy_numpy, 0.0)).astype(np.complex64),
        config.paths,
    )
    state_numpy, _, feature_mean, feature_std = standardize_state(
        train_state_raw, test_state_raw, fit_ids
    )
    state = torch.as_tensor(state_numpy, dtype=torch.float32, device=device)
    positions = torch.as_tensor(positions_numpy, dtype=torch.float32, device=device)
    coordinates = torch.as_tensor(
        coordinates_numpy[:, : config.paths], dtype=torch.float32, device=device
    )
    energy = torch.as_tensor(
        energy_numpy[:, : config.paths], dtype=torch.float32, device=device
    )
    model = ComponentTransport(state.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    rng = np.random.default_rng(config.seed + fold * 1009)
    validation_rows = np.flatnonzero(all_fold_id == fold)
    validation_query = all_query_ids[validation_rows]
    validation_donor = validation.donor_ids[:, 0]
    baseline = evaluate_components(
        coordinates_numpy[validation_donor, : config.paths],
        energy_numpy[validation_donor, : config.paths],
        validation_query,
        np.zeros(len(validation_query), dtype=np.int8),
        device,
        config.batch_size,
    )
    history: list[dict[str, object]] = [{"epoch": 0, "validation": baseline}]
    best_score = baseline["mean"]
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    print(f"component fold={fold} epoch=0 validation={baseline}", flush=True)

    target_coordinates = torch.as_tensor(
        table.target_coordinates, dtype=torch.float32, device=device
    )
    target_energy = torch.as_tensor(table.target_energy, dtype=torch.float32, device=device)
    match_cost = torch.as_tensor(table.match_cost, dtype=torch.float32, device=device)
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(len(table.query_ids))
        train_rows: list[dict[str, float]] = []
        for start in range(0, len(order), config.batch_size):
            selected = order[start : start + config.batch_size]
            rows = torch.as_tensor(selected, dtype=torch.long, device=device)
            output = forward_rows(
                model, rows, table, state, positions, coordinates, energy
            )
            loss, diagnostic = training_loss(
                output,
                target_coordinates[rows],
                target_energy[rows],
                match_cost[rows],
                config,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_rows.append(diagnostic)
        predicted_coordinates, predicted_energy = predict_components(
            model,
            validation_query,
            validation_donor,
            state,
            positions,
            coordinates,
            energy,
            config.batch_size,
        )
        validation_score = evaluate_components(
            predicted_coordinates,
            predicted_energy,
            validation_query,
            np.zeros(len(validation_query), dtype=np.int8),
            device,
            config.batch_size,
        )
        train_summary = {
            key: float(np.mean([row[key] for row in train_rows]))
            for key in train_rows[0]
        }
        history.append(
            {"epoch": epoch, "train": train_summary, "validation": validation_score}
        )
        if validation_score["mean"] > best_score:
            best_score = validation_score["mean"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"component fold={fold} epoch={epoch}/{config.epochs} "
            f"train={train_summary['loss']:.5f} "
            f"valid={validation_score['pas']:.6f}/{validation_score['pdp']:.6f}/"
            f"{validation_score['mean']:.6f} best={best_score:.6f}@{best_epoch}",
            flush=True,
        )
    checkpoint = ARTIFACT_DIR / f"fold{fold}_best.pt"
    torch.save(
        {
            "state_dict": best_state,
            "config": asdict(config),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "best_epoch": best_epoch,
            "best_score": best_score,
        },
        checkpoint,
    )
    return {
        "fold": fold,
        "training_pairs": int(len(table.query_ids)),
        "validation_points": int(len(validation_query)),
        "baseline": baseline,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "delta_vs_anchor": float(best_score - baseline["mean"]),
        "history": history,
        "checkpoint": PATHS.relative(checkpoint),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--folds", default="3")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    config = Config(epochs=args.epochs, batch_size=args.batch_size)
    context_config = ContextConfig()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    for path in (COMPONENT_CACHE, DESCRIPTORS, TRAIN_PAS, TRAIN_PDP, FOLD_CACHE, OOF_META):
        if not path.is_file():
            raise FileNotFoundError(path)
    component = np.load(COMPONENT_CACHE)
    coordinates = component["coordinates"].astype(np.float32)
    coefficient = component["coefficients"].astype(np.complex64)
    energy = np.abs(coefficient) ** 2
    train_state, test_state = load_state_features()
    device = torch.device(args.device)
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    records = [
        train_fold(
            fold,
            config,
            context_config,
            device,
            train_state,
            test_state,
            coordinates,
            energy.astype(np.float32),
        )
        for fold in folds
    ]
    payload = {
        "experiment": "Round2-E20-learned-component-transport",
        "status": "analysis_only",
        "primary_hypothesis": (
            "A path-level model conditioned on official map state can learn support motion and "
            "birth/death across complete spatial holes better than whole-spectrum regression."
        ),
        "official_inputs_only": True,
        "sionna_used": False,
        "config": asdict(config),
        "context_config": asdict(context_config),
        "folds": records,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
