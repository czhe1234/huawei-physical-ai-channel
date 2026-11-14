#!/usr/bin/env python3
"""Train a geometry-conditioned neighborhood field for Round 2 PAS/PDP spectra."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

from complex_baselines import affine_coefficients
from data import load_positions
from project_paths import PATHS


ARTIFACT_DIR = PATHS.artifacts / "round2" / "learnable_spectral_field"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_learnable_spectral_field"
REPORT = ANALYSIS_DIR / "round2_learnable_spectral_field.json"
OOF_OUTPUT = ARTIFACT_DIR / "round2_learnable_spectral_field_oof.npz"

GEOMETRY_CACHE = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_geometry_visibility_features.npz"
)
SPECTRAL_CACHE = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_train_spectral_descriptors.npz"
)
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
FOLD_CACHE = (
    PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
)

PAS_SIZE = 256
PDP_SIZE = 192
SPECTRAL_SIZE = PAS_SIZE + PDP_SIZE


@dataclass(frozen=True)
class Config:
    neighbors: int = 48
    pas_neighbors: int = 24
    candidate_neighbors: int = 192
    guard_radii_m: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
    epochs: int = 48
    batch_size: int = 128
    learning_rate: float = 1.5e-3
    weight_decay: float = 2e-5
    hidden: int = 128
    residual_limit: float = 0.28
    seed: int = 20_260_805


@dataclass(frozen=True)
class ContextTable:
    query_ids: np.ndarray
    donor_ids: np.ndarray
    distances: np.ndarray
    pas_coefficients: np.ndarray
    pdp_coefficients: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def positional_features(geometry: np.ndarray) -> np.ndarray:
    value = np.asarray(geometry, dtype=np.float32)
    xy = value[:, :2]
    features = [value]
    for scale in (5.0, 10.0, 20.0, 40.0, 80.0, 160.0):
        features.extend((np.sin(xy / scale), np.cos(xy / scale)))
    return np.concatenate(features, axis=1).astype(np.float32)


def standardize_features(
    features: np.ndarray,
    fit_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(features[fit_ids], axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(features[fit_ids], axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-4)
    return ((features - mean) / std).astype(np.float32), mean, std


def _select_context(
    candidate_ids: np.ndarray,
    candidate_distances: np.ndarray,
    query_id: int,
    guard_m: float,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    keep = (candidate_ids != query_id) & (candidate_distances >= guard_m)
    selected_ids = candidate_ids[keep][:count]
    selected_distances = candidate_distances[keep][:count]
    if len(selected_ids) < count:
        fallback = candidate_ids[candidate_ids != query_id]
        fallback_distances = candidate_distances[candidate_ids != query_id]
        needed = count - len(selected_ids)
        selected_ids = np.concatenate([selected_ids, fallback[-needed:]])
        selected_distances = np.concatenate(
            [selected_distances, fallback_distances[-needed:]]
        )
    if len(selected_ids) != count:
        raise RuntimeError("not enough same-site spectral donors")
    return selected_ids.astype(np.int64), selected_distances.astype(np.float32)


def build_context_table(
    positions: np.ndarray,
    site: np.ndarray,
    valid: np.ndarray,
    query_ids: np.ndarray,
    bank_ids: np.ndarray,
    guards: tuple[float, ...],
    config: Config,
) -> ContextTable:
    query_ids = np.asarray(query_ids, dtype=np.int64)
    guard_count = len(guards)
    shape = (len(query_ids), guard_count, config.neighbors)
    donor_ids = np.empty(shape, dtype=np.int32)
    distances = np.empty(shape, dtype=np.float32)
    pas_coefficients = np.zeros(shape, dtype=np.float32)
    pdp_coefficients = np.zeros(shape, dtype=np.float32)

    for site_value in np.unique(site[query_ids]):
        rows = np.flatnonzero(site[query_ids] == site_value)
        bank = bank_ids[(site[bank_ids] == site_value) & valid[bank_ids]]
        candidate_count = min(config.candidate_neighbors, len(bank))
        candidate_distances, local = cKDTree(positions[bank, :2]).query(
            positions[query_ids[rows], :2],
            k=candidate_count,
            workers=-1,
        )
        if candidate_count == 1:
            candidate_distances = candidate_distances[:, None]
            local = local[:, None]
        candidates = bank[local]
        for local_row, output_row in enumerate(rows):
            query_id = int(query_ids[output_row])
            for guard_index, guard_m in enumerate(guards):
                selected, selected_distance = _select_context(
                    candidates[local_row],
                    candidate_distances[local_row],
                    query_id,
                    guard_m,
                    config.neighbors,
                )
                donor_ids[output_row, guard_index] = selected
                distances[output_row, guard_index] = selected_distance
                pas_count = min(config.pas_neighbors, config.neighbors)
                pas_coefficients[output_row, guard_index, :pas_count] = affine_coefficients(
                    positions[selected[:pas_count], :2]
                    - positions[query_id, None, :2],
                    selected_distance[:pas_count],
                    distance_power=1.5,
                    ridge=0.30,
                )
                pdp_coefficients[output_row, guard_index] = affine_coefficients(
                    positions[selected, :2] - positions[query_id, None, :2],
                    selected_distance,
                    distance_power=2.0,
                    ridge=0.10,
                )
    return ContextTable(
        query_ids=query_ids,
        donor_ids=donor_ids,
        distances=distances,
        pas_coefficients=pas_coefficients,
        pdp_coefficients=pdp_coefficients,
    )


def _unit_sqrt(value: torch.Tensor) -> torch.Tensor:
    return value / torch.sqrt(value.square().sum(dim=-1, keepdim=True).clamp_min(1e-20))


class SpectralNeighborField(nn.Module):
    def __init__(self, feature_dim: int, config: Config) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden
        self.query_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden * 2),
            nn.SiLU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
        )
        self.spectrum_encoder = nn.Sequential(
            nn.Linear(SPECTRAL_SIZE, hidden * 2),
            nn.SiLU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
        )
        self.donor_encoder = nn.Sequential(
            nn.Linear(feature_dim + hidden, hidden * 2),
            nn.SiLU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
        )
        self.pair_head = nn.Sequential(
            nn.Linear(hidden * 2 + 7, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        self.pas_residual = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, PAS_SIZE),
        )
        self.pdp_residual = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, PDP_SIZE),
        )
        for head in (self.pair_head, self.gate_head, self.pas_residual, self.pdp_residual):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(
        self,
        query_feature: torch.Tensor,
        query_position: torch.Tensor,
        donor_feature: torch.Tensor,
        donor_position: torch.Tensor,
        donor_sqrt_spectrum: torch.Tensor,
        distances: torch.Tensor,
        pas_coefficients: torch.Tensor,
        pdp_coefficients: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        query_latent = self.query_encoder(query_feature)
        donor_spectral_latent = self.spectrum_encoder(donor_sqrt_spectrum)
        donor_latent = self.donor_encoder(
            torch.cat([donor_feature, donor_spectral_latent], dim=-1)
        )
        delta = donor_position - query_position[:, None, :]
        direction = delta[:, :, :2] / distances[:, :, None].clamp_min(1e-4)
        bearing = torch.atan2(delta[:, :, 1], delta[:, :, 0])
        pair_numeric = torch.stack(
            [
                delta[:, :, 0] / 20.0,
                delta[:, :, 1] / 20.0,
                distances / 20.0,
                torch.log1p(distances),
                direction[:, :, 0],
                direction[:, :, 1],
                torch.sin(2.0 * bearing),
            ],
            dim=-1,
        )
        query_pair = query_latent[:, None].expand(-1, donor_latent.shape[1], -1)
        pair_logits = self.pair_head(
            torch.cat([query_pair, donor_latent, pair_numeric], dim=-1)
        )
        prior = -1.5 * torch.log(distances.clamp_min(0.05))
        pas_weight = torch.softmax(prior + 2.0 * torch.tanh(pair_logits[:, :, 0]), dim=1)
        pdp_weight = torch.softmax(prior + 2.0 * torch.tanh(pair_logits[:, :, 1]), dim=1)

        donor_pas = donor_sqrt_spectrum[:, :, :PAS_SIZE]
        donor_pdp = donor_sqrt_spectrum[:, :, PAS_SIZE:]
        base_pas = torch.relu(
            torch.sum(pas_coefficients[:, :, None] * donor_pas, dim=1)
        )
        base_pdp = torch.relu(
            torch.sum(pdp_coefficients[:, :, None] * donor_pdp, dim=1)
        )
        base_pas = _unit_sqrt(base_pas)
        base_pdp = _unit_sqrt(base_pdp)
        attention_pas = _unit_sqrt(torch.sum(pas_weight[:, :, None] * donor_pas, dim=1))
        attention_pdp = _unit_sqrt(torch.sum(pdp_weight[:, :, None] * donor_pdp, dim=1))

        pooled_pas = torch.sum(pas_weight[:, :, None] * donor_latent, dim=1)
        pooled_pdp = torch.sum(pdp_weight[:, :, None] * donor_latent, dim=1)
        base_latent = self.spectrum_encoder(torch.cat([base_pas, base_pdp], dim=1))
        joint_pas = torch.cat([query_latent, pooled_pas, base_latent], dim=1)
        joint_pdp = torch.cat([query_latent, pooled_pdp, base_latent], dim=1)
        gates = torch.sigmoid(self.gate_head(torch.cat([query_latent, pooled_pas, base_latent], dim=1)))
        pas_start = _unit_sqrt((1.0 - gates[:, :1]) * base_pas + gates[:, :1] * attention_pas)
        pdp_start = _unit_sqrt((1.0 - gates[:, 1:]) * base_pdp + gates[:, 1:] * attention_pdp)
        pas = torch.relu(
            pas_start
            + self.config.residual_limit * torch.tanh(self.pas_residual(joint_pas))
        )
        pdp = torch.relu(
            pdp_start
            + self.config.residual_limit * torch.tanh(self.pdp_residual(joint_pdp))
        )
        return _unit_sqrt(pas), _unit_sqrt(pdp), {
            "pair_logits": pair_logits,
            "gates": gates,
            "pas_residual": pas - pas_start,
            "pdp_residual": pdp - pdp_start,
            "base_pas": base_pas,
            "base_pdp": base_pdp,
        }


def spectral_cosine(predicted_sqrt: torch.Tensor, truth_sqrt: torch.Tensor) -> torch.Tensor:
    predicted = predicted_sqrt.square()
    truth = truth_sqrt.square()
    numerator = torch.sum(predicted * truth, dim=1)
    denominator = torch.sqrt(
        torch.sum(predicted.square(), dim=1) * torch.sum(truth.square(), dim=1)
    )
    return numerator / denominator.clamp_min(1e-20)


def model_batch(
    model: SpectralNeighborField,
    rows: torch.Tensor,
    contexts: torch.Tensor,
    table: dict[str, torch.Tensor],
    feature: torch.Tensor,
    position: torch.Tensor,
    sqrt_spectrum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    donor_ids = table["donor_ids"][rows, contexts]
    query_ids = table["query_ids"][rows]
    return model(
        feature[query_ids],
        position[query_ids],
        feature[donor_ids],
        position[donor_ids],
        sqrt_spectrum[donor_ids],
        table["distances"][rows, contexts],
        table["pas_coefficients"][rows, contexts],
        table["pdp_coefficients"][rows, contexts],
    )


def tensor_table(table: ContextTable, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "query_ids": torch.as_tensor(table.query_ids, dtype=torch.long, device=device),
        "donor_ids": torch.as_tensor(table.donor_ids, dtype=torch.long, device=device),
        "distances": torch.as_tensor(table.distances, dtype=torch.float32, device=device),
        "pas_coefficients": torch.as_tensor(
            table.pas_coefficients, dtype=torch.float32, device=device
        ),
        "pdp_coefficients": torch.as_tensor(
            table.pdp_coefficients, dtype=torch.float32, device=device
        ),
    }


@torch.no_grad()
def evaluate(
    model: SpectralNeighborField,
    table: ContextTable,
    feature: torch.Tensor,
    position: torch.Tensor,
    sqrt_spectrum: torch.Tensor,
    valid: np.ndarray,
    batch_size: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    device = feature.device
    tensor_context = tensor_table(table, device)
    predicted_pas: list[np.ndarray] = []
    predicted_pdp: list[np.ndarray] = []
    baseline_pas: list[np.ndarray] = []
    baseline_pdp: list[np.ndarray] = []
    for start in range(0, len(table.query_ids), batch_size):
        end = min(start + batch_size, len(table.query_ids))
        rows = torch.arange(start, end, device=device)
        contexts = torch.zeros(end - start, dtype=torch.long, device=device)
        pas, pdp, diagnostics = model_batch(
            model,
            rows,
            contexts,
            tensor_context,
            feature,
            position,
            sqrt_spectrum,
        )
        predicted_pas.append(pas.square().cpu().numpy())
        predicted_pdp.append(pdp.square().cpu().numpy())
        baseline_pas.append(diagnostics["base_pas"].square().cpu().numpy())
        baseline_pdp.append(diagnostics["base_pdp"].square().cpu().numpy())
    pas_array = np.concatenate(predicted_pas)
    pdp_array = np.concatenate(predicted_pdp)
    base_pas_array = np.concatenate(baseline_pas)
    base_pdp_array = np.concatenate(baseline_pdp)
    truth_pas = sqrt_spectrum[
        torch.as_tensor(table.query_ids, dtype=torch.long, device=device), :PAS_SIZE
    ].square().cpu().numpy()
    truth_pdp = sqrt_spectrum[
        torch.as_tensor(table.query_ids, dtype=torch.long, device=device), PAS_SIZE:
    ].square().cpu().numpy()

    def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        numerator = np.sum(left * right, axis=1, dtype=np.float64)
        denominator = np.sqrt(
            np.sum(left * left, axis=1, dtype=np.float64)
            * np.sum(right * right, axis=1, dtype=np.float64)
        )
        return np.divide(
            numerator,
            np.where(denominator == 0.0, 1e-12, denominator),
        )

    point_valid = np.asarray(valid[table.query_ids], dtype=bool)
    pas_cosine = cosine(pas_array, truth_pas)
    pdp_cosine = cosine(pdp_array, truth_pdp)
    base_pas_cosine = cosine(base_pas_array, truth_pas)
    base_pdp_cosine = cosine(base_pdp_array, truth_pdp)
    report = {
        "count": int(len(table.query_ids)),
        "valid_count": int(np.sum(point_valid)),
        "pas": float(np.mean(pas_cosine)),
        "pdp": float(np.mean(pdp_cosine)),
        "mean": float(0.5 * (np.mean(pas_cosine) + np.mean(pdp_cosine))),
        "valid_pas": float(np.mean(pas_cosine[point_valid])),
        "valid_pdp": float(np.mean(pdp_cosine[point_valid])),
        "baseline_pas": float(np.mean(base_pas_cosine)),
        "baseline_pdp": float(np.mean(base_pdp_cosine)),
    }
    return report, pas_array.astype(np.float32), pdp_array.astype(np.float32)


def train_fold(
    fold: int,
    positions: np.ndarray,
    site: np.ndarray,
    valid: np.ndarray,
    raw_features: np.ndarray,
    sqrt_spectrum_numpy: np.ndarray,
    folds: np.lib.npyio.NpzFile,
    oof_query_ids: np.ndarray,
    oof_fold_ids: np.ndarray,
    config: Config,
    device: torch.device,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    fit_ids = folds[f"fold{fold}_fit"].astype(np.int64)
    train_ids = fit_ids[valid[fit_ids]]
    validation_rows = np.flatnonzero(oof_fold_ids == fold)
    validation_ids = oof_query_ids[validation_rows]
    features, feature_mean, feature_std = standardize_features(raw_features, fit_ids)
    train_table = build_context_table(
        positions,
        site,
        valid,
        train_ids,
        fit_ids,
        config.guard_radii_m,
        config,
    )
    validation_table = build_context_table(
        positions,
        site,
        valid,
        validation_ids,
        fit_ids,
        (0.0,),
        config,
    )
    feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    position_tensor = torch.as_tensor(positions, dtype=torch.float32, device=device)
    sqrt_spectrum = torch.as_tensor(
        sqrt_spectrum_numpy, dtype=torch.float32, device=device
    )
    train_context = tensor_table(train_table, device)
    model = SpectralNeighborField(features.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.seed + 1009 * fold)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(len(train_ids))
        running: list[tuple[float, float, float]] = []
        for start in range(0, len(order), config.batch_size):
            local_rows = order[start : start + config.batch_size]
            rows = torch.as_tensor(local_rows, dtype=torch.long, device=device)
            contexts = torch.as_tensor(
                rng.integers(0, len(config.guard_radii_m), size=len(local_rows)),
                dtype=torch.long,
                device=device,
            )
            predicted_pas, predicted_pdp, diagnostics = model_batch(
                model,
                rows,
                contexts,
                train_context,
                feature_tensor,
                position_tensor,
                sqrt_spectrum,
            )
            truth_ids = train_context["query_ids"][rows]
            truth_pas = sqrt_spectrum[truth_ids, :PAS_SIZE]
            truth_pdp = sqrt_spectrum[truth_ids, PAS_SIZE:]
            pas_cosine = spectral_cosine(predicted_pas, truth_pas)
            pdp_cosine = spectral_cosine(predicted_pdp, truth_pdp)
            regularization = (
                1e-3 * diagnostics["pair_logits"].square().mean()
                + 2e-3 * diagnostics["pas_residual"].square().mean()
                + 2e-3 * diagnostics["pdp_residual"].square().mean()
            )
            loss = 1.0 - 0.5 * (pas_cosine.mean() + pdp_cosine.mean()) + regularization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running.append(
                (
                    float(loss.detach().cpu()),
                    float(pas_cosine.mean().detach().cpu()),
                    float(pdp_cosine.mean().detach().cpu()),
                )
            )
        row: dict[str, float | int] = {
            "epoch": epoch,
            "loss": float(np.mean([value[0] for value in running])),
            "train_pas": float(np.mean([value[1] for value in running])),
            "train_pdp": float(np.mean([value[2] for value in running])),
        }
        history.append(row)
        if epoch == 1 or epoch % 8 == 0 or epoch == config.epochs:
            print(
                f"spectral field fold={fold} epoch={epoch}/{config.epochs} "
                f"loss={row['loss']:.6f} pas={row['train_pas']:.6f} "
                f"pdp={row['train_pdp']:.6f}",
                flush=True,
            )
    validation, predicted_pas, predicted_pdp = evaluate(
        model,
        validation_table,
        feature_tensor,
        position_tensor,
        sqrt_spectrum,
        valid,
        config.batch_size,
    )
    return {
        "fold": fold,
        "fit_count": int(len(fit_ids)),
        "train_valid_count": int(len(train_ids)),
        "validation": validation,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "history": history,
    }, predicted_pas, predicted_pdp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument(
        "--folds",
        default="0,1,2,3,4",
        help="Comma-separated outer folds to train; use one fold for a pilot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    config = Config(epochs=args.epochs, batch_size=args.batch_size)
    set_seed(config.seed)
    device = torch.device(args.device)
    _, positions, _, site, _ = load_positions()
    geometry = np.load(GEOMETRY_CACHE)
    raw_features = positional_features(geometry["train"].astype(np.float32))
    descriptors = np.load(SPECTRAL_CACHE)
    valid = descriptors["valid"].astype(bool)
    sqrt_spectrum = np.concatenate(
        [
            np.sqrt(np.maximum(descriptors["global_pas"].astype(np.float32), 0.0)),
            np.sqrt(np.maximum(descriptors["global_pdp"].astype(np.float32), 0.0)),
        ],
        axis=1,
    ).astype(np.float32)
    metadata = np.load(OOF_META)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_ids = metadata["fold_id"].astype(np.int8)
    folds = np.load(FOLD_CACHE)
    selected_folds = [int(value) for value in args.folds.split(",") if value.strip()]
    oof_pas = np.full((len(query_ids), PAS_SIZE), np.nan, dtype=np.float32)
    oof_pdp = np.full((len(query_ids), PDP_SIZE), np.nan, dtype=np.float32)
    records: list[dict[str, object]] = []
    for fold in selected_folds:
        record, predicted_pas, predicted_pdp = train_fold(
            fold,
            positions,
            site,
            valid,
            raw_features,
            sqrt_spectrum,
            folds,
            query_ids,
            fold_ids,
            config,
            device,
        )
        rows = np.flatnonzero(fold_ids == fold)
        oof_pas[rows] = predicted_pas
        oof_pdp[rows] = predicted_pdp
        records.append(record)
        print(f"fold={fold} validation={record['validation']}", flush=True)
    np.savez_compressed(
        OOF_OUTPUT,
        pas=oof_pas,
        pdp=oof_pdp,
        query_ids=query_ids,
        fold_id=fold_ids,
        completed_folds=np.asarray(selected_folds, dtype=np.int8),
    )
    report = {
        "experiment": "round2_learnable_spectral_field",
        "primary_hypothesis": (
            "A geometry-conditioned neural neighborhood field can improve the square-root "
            "PAS/PDP affine field by selecting propagation-compatible donors across spatial holes."
        ),
        "config": asdict(config),
        "device": str(device),
        "folds": records,
        "outputs": {"oof_spectra": PATHS.relative(OOF_OUTPUT)},
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
