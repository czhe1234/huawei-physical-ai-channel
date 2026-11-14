#!/usr/bin/env python3
"""Learn bounded donor-weight corrections around the Round 2 V2 RBF anchor."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

from data import load_positions
from project_paths import PATHS
from run_rbf_spectral_field import solve_linear_rbf_weights


ARTIFACT_DIR = PATHS.artifacts / "round2" / "v3_anchored_neural_rbf"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v3_anchored_neural_rbf"
REPORT = ANALYSIS_DIR / "anchored_neural_rbf.json"

PRE_DIR = PATHS.artifacts / "round2" / "pre_sionna"
DESCRIPTORS = PRE_DIR / "round2_train_spectral_descriptors.npz"
GEOMETRY = PRE_DIR / "round2_geometry_visibility_features.npz"
OOF_META = PRE_DIR / "round2_pre_sionna_oof_foundation_meta.npz"
FOLD_CACHE = (
    PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
)
V2_WEIGHTS = (
    PATHS.artifacts
    / "round2"
    / "rbf_spectral_field"
    / "round2_rbf_oof_weights.npz"
)

PAS_SIZE = 256
PDP_SIZE = 192
SPECTRAL_SIZE = PAS_SIZE + PDP_SIZE


@dataclass(frozen=True)
class Config:
    neighbors: int = 64
    candidate_neighbors: int = 320
    guard_radii_m: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0)
    smoothing: float = 10.0
    hidden: int = 128
    spectral_hidden: int = 64
    correction_l1_bound: float = 0.40
    correction_regularization: float = 0.03
    epochs: int = 32
    batch_size: int = 128
    learning_rate: float = 8e-4
    weight_decay: float = 2e-5
    seed: int = 20_260_805


@dataclass(frozen=True)
class ContextTable:
    query_ids: np.ndarray
    donor_ids: np.ndarray
    weights: np.ndarray
    distances: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def context_cache_path(fold: int, config: Config) -> Path:
    guards = "_".join(f"{value:g}" for value in config.guard_radii_m)
    return ARTIFACT_DIR / (
        f"fold{fold}_k{config.neighbors}_s{config.smoothing:g}_guards_{guards}.npz"
    )


def build_train_contexts(
    fold: int,
    positions: np.ndarray,
    site: np.ndarray,
    valid: np.ndarray,
    fit_ids: np.ndarray,
    config: Config,
) -> ContextTable:
    cache_path = context_cache_path(fold, config)
    query_ids = fit_ids[valid[fit_ids]]
    if cache_path.is_file():
        cached = np.load(cache_path)
        if (
            np.array_equal(cached["query_ids"], query_ids)
            and int(cached["neighbors"]) == config.neighbors
            and np.isclose(float(cached["smoothing"]), config.smoothing)
            and np.allclose(cached["guard_radii_m"], config.guard_radii_m)
        ):
            return ContextTable(
                query_ids=query_ids,
                donor_ids=cached["donor_ids"].astype(np.int64),
                weights=cached["weights"].astype(np.float32),
                distances=cached["distances"].astype(np.float32),
            )

    shape = (len(query_ids), len(config.guard_radii_m), config.neighbors)
    donor_ids = np.empty(shape, dtype=np.int32)
    weights = np.empty(shape, dtype=np.float32)
    distances = np.empty(shape, dtype=np.float32)
    for site_value in np.unique(site[query_ids]):
        rows = np.flatnonzero(site[query_ids] == site_value)
        bank = fit_ids[(site[fit_ids] == site_value) & valid[fit_ids]]
        count = min(config.candidate_neighbors, len(bank))
        candidate_distance, local = cKDTree(positions[bank, :2]).query(
            positions[query_ids[rows], :2], k=count, workers=-1
        )
        candidate_ids = bank[local]
        for local_row, output_row in enumerate(rows):
            query_id = query_ids[output_row]
            for guard_index, guard in enumerate(config.guard_radii_m):
                keep = (
                    (candidate_ids[local_row] != query_id)
                    & (candidate_distance[local_row] >= guard)
                )
                selected_ids = candidate_ids[local_row, keep][: config.neighbors]
                selected_distance = candidate_distance[local_row, keep][: config.neighbors]
                if len(selected_ids) != config.neighbors:
                    raise RuntimeError(
                        f"query {query_id} guard {guard} has only {len(selected_ids)} donors"
                    )
                donor_ids[output_row, guard_index] = selected_ids
                distances[output_row, guard_index] = selected_distance
                weights[output_row, guard_index] = solve_linear_rbf_weights(
                    positions[selected_ids, :2],
                    positions[query_id, :2],
                    config.smoothing,
                )
        print(
            f"context fold={fold} site={int(site_value)} "
            f"{len(rows)}/{len(query_ids)}",
            flush=True,
        )
    np.savez_compressed(
        cache_path,
        query_ids=query_ids.astype(np.int32),
        donor_ids=donor_ids,
        weights=weights,
        distances=distances,
        neighbors=np.int32(config.neighbors),
        smoothing=np.float32(config.smoothing),
        guard_radii_m=np.asarray(config.guard_radii_m, dtype=np.float32),
    )
    return ContextTable(query_ids, donor_ids, weights, distances)


def validation_context(
    fold: int,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
) -> ContextTable:
    rows = np.flatnonzero(fold_id == fold)
    cached = np.load(V2_WEIGHTS)
    donor_ids = cached["neighbor_ids"][rows].astype(np.int64)
    weights = cached["weights"][rows].astype(np.float32)
    _, positions, _, _, _ = load_positions()
    distances = np.linalg.norm(
        positions[donor_ids, :2] - positions[query_ids[rows], None, :2], axis=2
    ).astype(np.float32)
    return ContextTable(query_ids[rows], donor_ids, weights, distances)


def standardize_geometry(
    geometry: np.ndarray,
    fit_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Raw x/y are omitted; relative-to-BS and official-map features remain.
    selected = np.asarray(geometry[:, 3:], dtype=np.float32)
    mean = np.mean(selected[fit_ids], axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(selected[fit_ids], axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-4)
    return ((selected - mean) / std).astype(np.float32), mean, std


def unit_amplitude(value: torch.Tensor) -> torch.Tensor:
    return value / torch.sqrt(value.square().sum(dim=-1, keepdim=True).clamp_min(1e-20))


def power_cosine(predicted_amplitude: torch.Tensor, truth_amplitude: torch.Tensor) -> torch.Tensor:
    predicted = predicted_amplitude.square()
    truth = truth_amplitude.square()
    numerator = torch.sum(predicted * truth, dim=-1)
    denominator = torch.sqrt(
        torch.sum(predicted.square(), dim=-1) * torch.sum(truth.square(), dim=-1)
    )
    return numerator / denominator.clamp_min(1e-20)


class AnchoredWeightCorrector(nn.Module):
    def __init__(self, geometry_size: int, config: Config) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden
        self.query_encoder = nn.Sequential(
            nn.Linear(geometry_size, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        self.spectral_encoder = nn.Sequential(
            nn.Linear(SPECTRAL_SIZE, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, config.spectral_hidden),
        )
        pair_input = hidden + geometry_size + config.spectral_hidden + 9
        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_input, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.correction_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

    def forward(
        self,
        query_geometry: torch.Tensor,
        query_position: torch.Tensor,
        donor_geometry: torch.Tensor,
        donor_position: torch.Tensor,
        donor_spectrum: torch.Tensor,
        distances: torch.Tensor,
        base_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        query_latent = self.query_encoder(query_geometry)
        donor_latent = self.spectral_encoder(donor_spectrum)
        geometry_delta = donor_geometry - query_geometry[:, None, :]
        position_delta = donor_position - query_position[:, None, :]
        direction = position_delta / distances[:, :, None].clamp_min(1e-4)
        rank = torch.linspace(
            0.0, 1.0, donor_spectrum.shape[1], device=donor_spectrum.device
        )[None, :, None].expand(len(query_geometry), -1, -1)
        numeric = torch.cat(
            [
                position_delta / 20.0,
                distances[:, :, None] / 20.0,
                torch.log1p(distances[:, :, None]),
                direction,
                base_weights[:, :, None].clamp(-2.0, 2.0),
                base_weights.abs()[:, :, None].clamp(0.0, 2.0),
                rank,
            ],
            dim=-1,
        )
        query_expanded = query_latent[:, None, :].expand(-1, donor_spectrum.shape[1], -1)
        pair = self.pair_encoder(
            torch.cat([query_expanded, geometry_delta, donor_latent, numeric], dim=-1)
        )
        pooled_mean = pair.mean(dim=1, keepdim=True).expand_as(pair)
        pooled_anchor = torch.sum(
            base_weights[:, :, None] * pair, dim=1, keepdim=True
        ).expand_as(pair)
        raw = torch.tanh(
            self.correction_head(torch.cat([pair, pooled_mean, pooled_anchor], dim=-1))
        )
        raw = raw - raw.mean(dim=1, keepdim=True)
        correction = (
            self.config.correction_l1_bound / self.config.neighbors * raw
        )
        pas_weights = base_weights + correction[:, :, 0]
        pdp_weights = base_weights + correction[:, :, 1]
        donor_pas = donor_spectrum[:, :, :PAS_SIZE]
        donor_pdp = donor_spectrum[:, :, PAS_SIZE:]
        pas = unit_amplitude(
            torch.relu(torch.sum(pas_weights[:, :, None] * donor_pas, dim=1))
        )
        pdp = unit_amplitude(
            torch.relu(torch.sum(pdp_weights[:, :, None] * donor_pdp, dim=1))
        )
        return pas, pdp, {
            "correction": correction,
            "pas_weights": pas_weights,
            "pdp_weights": pdp_weights,
        }


def tensor_context(table: ContextTable, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "query_ids": torch.as_tensor(table.query_ids, dtype=torch.long, device=device),
        "donor_ids": torch.as_tensor(table.donor_ids, dtype=torch.long, device=device),
        "weights": torch.as_tensor(table.weights, dtype=torch.float32, device=device),
        "distances": torch.as_tensor(table.distances, dtype=torch.float32, device=device),
    }


def forward_rows(
    model: AnchoredWeightCorrector,
    context: dict[str, torch.Tensor],
    rows: torch.Tensor,
    guard: torch.Tensor,
    geometry: torch.Tensor,
    positions: torch.Tensor,
    spectra: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    query_ids = context["query_ids"][rows]
    if context["donor_ids"].ndim == 3:
        donor_ids = context["donor_ids"][rows, guard]
        weights = context["weights"][rows, guard]
        distances = context["distances"][rows, guard]
    else:
        donor_ids = context["donor_ids"][rows]
        weights = context["weights"][rows]
        distances = context["distances"][rows]
    return model(
        geometry[query_ids],
        positions[query_ids, :2],
        geometry[donor_ids],
        positions[donor_ids, :2],
        spectra[donor_ids],
        distances,
        weights,
    )


@torch.no_grad()
def evaluate(
    model: AnchoredWeightCorrector,
    table: ContextTable,
    geometry: torch.Tensor,
    positions: torch.Tensor,
    spectra: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    context = tensor_context(table, geometry.device)
    pas_values: list[np.ndarray] = []
    pdp_values: list[np.ndarray] = []
    correction_l1: list[np.ndarray] = []
    for start in range(0, len(table.query_ids), batch_size):
        end = min(start + batch_size, len(table.query_ids))
        rows = torch.arange(start, end, device=geometry.device)
        guard = torch.zeros(end - start, dtype=torch.long, device=geometry.device)
        pas, pdp, diagnostics = forward_rows(
            model, context, rows, guard, geometry, positions, spectra
        )
        truth_ids = context["query_ids"][rows]
        pas_values.append(
            power_cosine(pas, spectra[truth_ids, :PAS_SIZE]).cpu().numpy()
        )
        pdp_values.append(
            power_cosine(pdp, spectra[truth_ids, PAS_SIZE:]).cpu().numpy()
        )
        correction_l1.append(
            diagnostics["correction"].abs().sum(dim=1).cpu().numpy()
        )
    pas = np.concatenate(pas_values)
    pdp = np.concatenate(pdp_values)
    l1 = np.concatenate(correction_l1)
    return {
        "pas": float(np.mean(pas)),
        "pdp": float(np.mean(pdp)),
        "mean": float(np.mean(0.5 * (pas + pdp))),
        "correction_l1_pas": float(np.mean(l1[:, 0])),
        "correction_l1_pdp": float(np.mean(l1[:, 1])),
    }


def train_fold(
    fold: int,
    config: Config,
    device: torch.device,
) -> dict[str, object]:
    set_seed(config.seed + fold * 1009)
    setup, positions_numpy, _, site, _ = load_positions()
    del setup
    descriptors = np.load(DESCRIPTORS)
    valid = descriptors["valid"].astype(bool)
    spectra_numpy = np.concatenate(
        [
            np.sqrt(np.maximum(descriptors["global_pas"].astype(np.float32), 0.0)),
            np.sqrt(np.maximum(descriptors["global_pdp"].astype(np.float32), 0.0)),
        ],
        axis=1,
    ).astype(np.float32)
    geometry_numpy = np.load(GEOMETRY)["train"].astype(np.float32)
    folds = np.load(FOLD_CACHE)
    fit_ids = folds[f"fold{fold}_fit"].astype(np.int64)
    metadata = np.load(OOF_META)
    oof_query_ids = metadata["query_ids"].astype(np.int64)
    oof_fold_id = metadata["fold_id"].astype(np.int8)
    train_table = build_train_contexts(
        fold, positions_numpy, site, valid, fit_ids, config
    )
    validation_table = validation_context(fold, oof_query_ids, oof_fold_id)
    standardized, feature_mean, feature_std = standardize_geometry(
        geometry_numpy, fit_ids
    )

    geometry = torch.as_tensor(standardized, dtype=torch.float32, device=device)
    positions = torch.as_tensor(positions_numpy, dtype=torch.float32, device=device)
    spectra = torch.as_tensor(spectra_numpy, dtype=torch.float32, device=device)
    train_context = tensor_context(train_table, device)
    model = AnchoredWeightCorrector(geometry.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.seed + fold * 1009)
    history: list[dict[str, object]] = []
    baseline = evaluate(
        model, validation_table, geometry, positions, spectra, config.batch_size
    )
    best_score = baseline["mean"]
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history.append({"epoch": 0, "validation": baseline})
    print(f"fold={fold} epoch=0 validation={baseline}", flush=True)

    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(len(train_table.query_ids))
        train_rows: list[tuple[float, float, float, float]] = []
        for start in range(0, len(order), config.batch_size):
            local = order[start : start + config.batch_size]
            rows = torch.as_tensor(local, dtype=torch.long, device=device)
            guard = torch.as_tensor(
                rng.integers(0, len(config.guard_radii_m), size=len(local)),
                dtype=torch.long,
                device=device,
            )
            pas, pdp, diagnostics = forward_rows(
                model,
                train_context,
                rows,
                guard,
                geometry,
                positions,
                spectra,
            )
            truth_ids = train_context["query_ids"][rows]
            pas_cosine = power_cosine(pas, spectra[truth_ids, :PAS_SIZE])
            pdp_cosine = power_cosine(pdp, spectra[truth_ids, PAS_SIZE:])
            correction = diagnostics["correction"]
            regularization = config.correction_regularization * (
                correction.square().sum(dim=1).mean()
                * config.neighbors
                / max(config.correction_l1_bound**2, 1e-8)
            )
            loss = 1.0 - 0.5 * (pas_cosine.mean() + pdp_cosine.mean()) + regularization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            train_rows.append(
                (
                    float(loss.detach().cpu()),
                    float(pas_cosine.mean().detach().cpu()),
                    float(pdp_cosine.mean().detach().cpu()),
                    float(correction.abs().sum(dim=1).mean().detach().cpu()),
                )
            )
        validation = evaluate(
            model, validation_table, geometry, positions, spectra, config.batch_size
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean([value[0] for value in train_rows])),
            "train_pas": float(np.mean([value[1] for value in train_rows])),
            "train_pdp": float(np.mean([value[2] for value in train_rows])),
            "train_correction_l1": float(np.mean([value[3] for value in train_rows])),
            "validation": validation,
        }
        history.append(row)
        if validation["mean"] > best_score:
            best_score = validation["mean"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"fold={fold} epoch={epoch}/{config.epochs} "
            f"train={row['train_pas']:.6f}/{row['train_pdp']:.6f} "
            f"valid={validation['pas']:.6f}/{validation['pdp']:.6f}/"
            f"{validation['mean']:.6f} best={best_score:.6f}@{best_epoch}",
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
    final_checkpoint = ARTIFACT_DIR / f"fold{fold}_final.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "epoch": config.epochs,
        },
        final_checkpoint,
    )
    return {
        "fold": fold,
        "fit_count": int(len(fit_ids)),
        "train_valid_count": int(len(train_table.query_ids)),
        "validation_count": int(len(validation_table.query_ids)),
        "baseline": baseline,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "delta_vs_v2": float(best_score - baseline["mean"]),
        "history": history,
        "checkpoint": PATHS.relative(checkpoint),
        "final_checkpoint": PATHS.relative(final_checkpoint),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--folds", default="3")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--correction-l1-bound", type=float, default=Config.correction_l1_bound)
    parser.add_argument(
        "--correction-regularization",
        type=float,
        default=Config.correction_regularization,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    config = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        correction_l1_bound=args.correction_l1_bound,
        correction_regularization=args.correction_regularization,
    )
    set_seed(config.seed)
    device = torch.device(args.device)
    selected_folds = [int(value) for value in args.folds.split(",") if value.strip()]
    records = [train_fold(fold, config, device) for fold in selected_folds]
    payload = {
        "experiment": "Round2-E4-V3-anchored-neural-RBF",
        "status": "analysis_only",
        "primary_hypothesis": (
            "A bounded zero-sum correction to the exact V2 signed-RBF donor weights can "
            "learn propagation-compatible donor competition without discarding the V2 anchor."
        ),
        "sionna_used": False,
        "official_inputs_only": True,
        "validation": "fixed Round 2 spatial-hole folds",
        "device": str(device),
        "config": asdict(config),
        "folds": records,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
