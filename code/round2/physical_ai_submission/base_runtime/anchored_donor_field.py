"""Frozen runtime primitives for anchored donor field."""

from __future__ import annotations
import random
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from .dataset_compat import load_positions
from project_paths import PATHS
from .radial_spectral_field import solve_linear_rbf_weights

ARTIFACT_DIR = PATHS.artifacts / "round2" / "donor_anchored_neural_rbf"
PRE_DIR = PATHS.artifacts / "round2" / "pre_physical"
DESCRIPTORS = PRE_DIR / "round2_train_spectral_descriptors.npz"
GEOMETRY = PRE_DIR / "round2_geometry_visibility_features.npz"
OOF_META = PRE_DIR / "round2_pre_physical_oof_foundation_meta.npz"
FOLD_CACHE = PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
FOUNDATION_WEIGHTS = (
    PATHS.artifacts / "round2" / "rbf_spectral_field" / "round2_rbf_oof_weights.npz"
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
    correction_l1_bound: float = 0.4
    correction_regularization: float = 0.03
    epochs: int = 32
    batch_size: int = 128
    learning_rate: float = 0.0008
    weight_decay: float = 2e-05
    seed: int = 20260805


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
    guards = "_".join((f"{value:g}" for value in config.guard_radii_m))
    return (
        ARTIFACT_DIR
        / f"fold{fold}_k{config.neighbors}_s{config.smoothing:g}_guards_{guards}.npz"
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
                keep = (candidate_ids[local_row] != query_id) & (
                    candidate_distance[local_row] >= guard
                )
                selected_ids = candidate_ids[local_row, keep][: config.neighbors]
                selected_distance = candidate_distance[local_row, keep][
                    : config.neighbors
                ]
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
            f"context fold={fold} site={int(site_value)} {len(rows)}/{len(query_ids)}",
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
    fold: int, query_ids: np.ndarray, fold_id: np.ndarray
) -> ContextTable:
    rows = np.flatnonzero(fold_id == fold)
    cached = np.load(FOUNDATION_WEIGHTS)
    donor_ids = cached["neighbor_ids"][rows].astype(np.int64)
    weights = cached["weights"][rows].astype(np.float32)
    _, positions, _, _, _ = load_positions()
    distances = np.linalg.norm(
        positions[donor_ids, :2] - positions[query_ids[rows], None, :2], axis=2
    ).astype(np.float32)
    return ContextTable(query_ids[rows], donor_ids, weights, distances)


def standardize_geometry(
    geometry: np.ndarray, fit_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = np.asarray(geometry[:, 3:], dtype=np.float32)
    mean = np.mean(selected[fit_ids], axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(selected[fit_ids], axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 0.0001)
    return (((selected - mean) / std).astype(np.float32), mean, std)


def unit_amplitude(value: torch.Tensor) -> torch.Tensor:
    return value / torch.sqrt(value.square().sum(dim=-1, keepdim=True).clamp_min(1e-20))


def power_cosine(
    predicted_amplitude: torch.Tensor, truth_amplitude: torch.Tensor
) -> torch.Tensor:
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
            nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Linear(hidden, 2)
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
        direction = position_delta / distances[:, :, None].clamp_min(0.0001)
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
        query_expanded = query_latent[:, None, :].expand(
            -1, donor_spectrum.shape[1], -1
        )
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
        correction = self.config.correction_l1_bound / self.config.neighbors * raw
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
        return (
            pas,
            pdp,
            {
                "correction": correction,
                "pas_weights": pas_weights,
                "pdp_weights": pdp_weights,
            },
        )


def tensor_context(
    table: ContextTable, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "query_ids": torch.as_tensor(table.query_ids, dtype=torch.long, device=device),
        "donor_ids": torch.as_tensor(table.donor_ids, dtype=torch.long, device=device),
        "weights": torch.as_tensor(table.weights, dtype=torch.float32, device=device),
        "distances": torch.as_tensor(
            table.distances, dtype=torch.float32, device=device
        ),
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
