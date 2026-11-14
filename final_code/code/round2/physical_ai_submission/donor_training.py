"""Train donor-weight correction and rebuild deployment donor weights."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

from .config import PipelineConfig, RuntimePaths
from .data import load_official_data
from .geometry import build_geometry_regime
from .io_utils import atomic_json, atomic_npz, portable_path
from .training import prepare_training_inputs
from .training_common import (
    FoldMetadata,
    atomic_torch_save,
    load_fold_metadata,
    set_seed,
    solve_linear_rbf_weights,
)


PAS_SIZE = 256
PDP_SIZE = 192
SPECTRAL_SIZE = PAS_SIZE + PDP_SIZE
TINY = 1e-20


@dataclass(frozen=True)
class DonorTrainingConfig:
    neighbors: int = 64
    candidate_neighbors: int = 320
    guard_radii_m: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0)
    smoothing: float = 10.0
    hidden: int = 128
    spectral_hidden: int = 64
    correction_l1_bound: float = 0.8
    correction_regularization: float = 0.003
    epochs: int = 6
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


class AnchoredWeightCorrector(nn.Module):
    """Learn a bounded PAS/PDP correction around deterministic RBF weights."""

    def __init__(self, geometry_size: int, config: DonorTrainingConfig) -> None:
        super().__init__()
        hidden = config.hidden
        self.config = config
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
            (
                position_delta / 20.0,
                distances[:, :, None] / 20.0,
                torch.log1p(distances[:, :, None]),
                direction,
                base_weights[:, :, None].clamp(-2.0, 2.0),
                base_weights.abs()[:, :, None].clamp(0.0, 2.0),
                rank,
            ),
            dim=-1,
        )
        query_expanded = query_latent[:, None].expand(-1, donor_spectrum.shape[1], -1)
        pair = self.pair_encoder(
            torch.cat((query_expanded, geometry_delta, donor_latent, numeric), dim=-1)
        )
        pooled_mean = pair.mean(dim=1, keepdim=True).expand_as(pair)
        pooled_anchor = torch.sum(
            base_weights[:, :, None] * pair, dim=1, keepdim=True
        ).expand_as(pair)
        raw = torch.tanh(
            self.correction_head(torch.cat((pair, pooled_mean, pooled_anchor), dim=-1))
        )
        raw = raw - raw.mean(dim=1, keepdim=True)
        correction = self.config.correction_l1_bound / self.config.neighbors * raw
        pas_weights = base_weights + correction[:, :, 0]
        pdp_weights = base_weights + correction[:, :, 1]
        donor_pas = donor_spectrum[:, :, :PAS_SIZE]
        donor_pdp = donor_spectrum[:, :, PAS_SIZE:]
        pas = _unit_amplitude(torch.relu(torch.sum(pas_weights[:, :, None] * donor_pas, dim=1)))
        pdp = _unit_amplitude(torch.relu(torch.sum(pdp_weights[:, :, None] * donor_pdp, dim=1)))
        return pas, pdp, {
            "correction": correction,
            "pas_weights": pas_weights,
            "pdp_weights": pdp_weights,
        }


def _unit_amplitude(value: torch.Tensor) -> torch.Tensor:
    return value / torch.sqrt(value.square().sum(dim=-1, keepdim=True).clamp_min(TINY))


def _power_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_power = left.square()
    right_power = right.square()
    numerator = (left_power * right_power).sum(dim=-1)
    denominator = torch.sqrt(
        left_power.square().sum(dim=-1) * right_power.square().sum(dim=-1)
    )
    return numerator / denominator.clamp_min(TINY)


def _root(paths: RuntimePaths) -> Path:
    return paths.work_dir / "training" / "donor"


def _fold_checkpoint(paths: RuntimePaths, fold: int) -> Path:
    return _root(paths) / "fold_models" / f"fold_{fold}.pt"


def _fold_weights(paths: RuntimePaths, fold: int) -> Path:
    return _root(paths) / "fold_weights" / f"fold_{fold}.npz"


def oof_weights_path(paths: RuntimePaths) -> Path:
    return _root(paths) / "oof_weights.npz"


def deployment_weights_path(paths: RuntimePaths) -> Path:
    return paths.model_dir / "base_field" / "deployment_donor_weights.npz"


def donor_checkpoint_path(paths: RuntimePaths) -> Path:
    return paths.model_dir / "base_field" / "donor_weight_model.pt"


def _context_path(paths: RuntimePaths, name: str) -> Path:
    return _root(paths) / "contexts" / f"{name}.npz"


def _load_context(path: Path) -> ContextTable:
    with np.load(path, allow_pickle=False) as loaded:
        return ContextTable(
            loaded["query_ids"].astype(np.int64),
            loaded["donor_ids"].astype(np.int64),
            loaded["weights"].astype(np.float32),
            loaded["distances"].astype(np.float32),
        )


def _context_is_compatible(
    path: Path,
    expected_query_ids: np.ndarray,
    expected_shape: tuple[int, ...],
    config: DonorTrainingConfig,
) -> bool:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            required = {
                "query_ids",
                "donor_ids",
                "weights",
                "distances",
                "neighbors",
                "candidate_neighbors",
                "guard_radii_m",
                "smoothing",
            }
            if not required.issubset(loaded.files):
                return False
            return (
                np.array_equal(
                    loaded["query_ids"].astype(np.int64), expected_query_ids
                )
                and loaded["donor_ids"].shape == expected_shape
                and loaded["weights"].shape == expected_shape
                and loaded["distances"].shape == expected_shape
                and int(loaded["neighbors"]) == config.neighbors
                and int(loaded["candidate_neighbors"])
                == config.candidate_neighbors
                and np.array_equal(
                    loaded["guard_radii_m"].astype(np.float64),
                    np.asarray(config.guard_radii_m, dtype=np.float64),
                )
                and np.isclose(
                    float(loaded["smoothing"]),
                    config.smoothing,
                    rtol=0.0,
                    atol=1e-7,
                )
            )
    except (KeyError, OSError, ValueError):
        return False


def _save_context(path: Path, table: ContextTable, config: DonorTrainingConfig) -> None:
    atomic_npz(
        path,
        query_ids=table.query_ids.astype(np.int32),
        donor_ids=table.donor_ids.astype(np.int32),
        weights=table.weights.astype(np.float32),
        distances=table.distances.astype(np.float32),
        neighbors=np.int32(config.neighbors),
        candidate_neighbors=np.int32(config.candidate_neighbors),
        guard_radii_m=np.asarray(config.guard_radii_m, dtype=np.float32),
        smoothing=np.float32(config.smoothing),
    )


def _training_context(
    paths: RuntimePaths,
    name: str,
    positions: np.ndarray,
    sites: np.ndarray,
    valid: np.ndarray,
    fit_ids: np.ndarray,
    config: DonorTrainingConfig,
    force: bool,
) -> ContextTable:
    path = _context_path(paths, name)
    expected_query = fit_ids[valid[fit_ids]]
    if path.is_file() and not force:
        expected_shape = (
            len(expected_query),
            len(config.guard_radii_m),
            config.neighbors,
        )
        if _context_is_compatible(path, expected_query, expected_shape, config):
            return _load_context(path)

    shape = (len(expected_query), len(config.guard_radii_m), config.neighbors)
    donor_ids = np.empty(shape, dtype=np.int32)
    weights = np.empty(shape, dtype=np.float32)
    distances = np.empty(shape, dtype=np.float32)
    for site in np.unique(sites[expected_query]):
        rows = np.flatnonzero(sites[expected_query] == site)
        bank = fit_ids[(sites[fit_ids] == site) & valid[fit_ids]]
        candidate_count = min(config.candidate_neighbors, len(bank))
        candidate_distance, local = cKDTree(positions[bank, :2]).query(
            positions[expected_query[rows], :2], k=candidate_count, workers=-1
        )
        candidate_distance = np.atleast_2d(candidate_distance)
        local = np.atleast_2d(local)
        candidate_ids = bank[local]
        for local_row, destination in enumerate(rows):
            query_id = expected_query[destination]
            for guard_index, guard in enumerate(config.guard_radii_m):
                keep = (candidate_ids[local_row] != query_id) & (
                    candidate_distance[local_row] >= guard
                )
                selected = candidate_ids[local_row, keep][: config.neighbors]
                selected_distance = candidate_distance[local_row, keep][
                    : config.neighbors
                ]
                if len(selected) != config.neighbors:
                    raise RuntimeError(
                        f"donor context {name}: query {query_id} guard {guard:g} "
                        f"has {len(selected)} donors"
                    )
                donor_ids[destination, guard_index] = selected
                distances[destination, guard_index] = selected_distance
                weights[destination, guard_index] = solve_linear_rbf_weights(
                    positions[selected, :2], positions[query_id, :2], config.smoothing
                )
        print(f"donor training context {name} site={int(site)} {len(rows)}", flush=True)
    table = ContextTable(expected_query, donor_ids, weights, distances)
    _save_context(path, table, config)
    return table


def _query_context(
    paths: RuntimePaths,
    name: str,
    query_ids: np.ndarray,
    query_positions: np.ndarray,
    query_sites: np.ndarray,
    train_positions: np.ndarray,
    train_sites: np.ndarray,
    valid: np.ndarray,
    fit_ids: np.ndarray,
    config: DonorTrainingConfig,
    force: bool,
) -> ContextTable:
    path = _context_path(paths, name)
    expected_shape = (len(query_ids), config.neighbors)
    if path.is_file() and not force:
        if _context_is_compatible(path, query_ids, expected_shape, config):
            return _load_context(path)

    donor_ids = np.empty(expected_shape, dtype=np.int32)
    weights = np.empty(expected_shape, dtype=np.float32)
    distances = np.empty(expected_shape, dtype=np.float32)
    for site in np.unique(query_sites):
        rows = np.flatnonzero(query_sites == site)
        bank = fit_ids[(train_sites[fit_ids] == site) & valid[fit_ids]]
        local_distance, local = cKDTree(train_positions[bank, :2]).query(
            query_positions[rows, :2], k=config.neighbors, workers=-1
        )
        selected = bank[local]
        donor_ids[rows] = selected
        distances[rows] = local_distance
        for local_row, destination in enumerate(rows):
            weights[destination] = solve_linear_rbf_weights(
                train_positions[selected[local_row], :2],
                query_positions[destination, :2],
                config.smoothing,
            )
    table = ContextTable(query_ids, donor_ids, weights, distances)
    _save_context(path, table, config)
    return table


def _standardize_geometry(
    geometry: np.ndarray, fit_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = np.asarray(geometry[:, 3:], dtype=np.float32)
    mean = selected[fit_ids].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = selected[fit_ids].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-4] = 1.0
    return ((selected - mean) / std).astype(np.float32), mean, std


def _tensor_context(table: ContextTable, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "query_ids": torch.as_tensor(table.query_ids, dtype=torch.long, device=device),
        "donor_ids": torch.as_tensor(table.donor_ids, dtype=torch.long, device=device),
        "weights": torch.as_tensor(table.weights, dtype=torch.float32, device=device),
        "distances": torch.as_tensor(table.distances, dtype=torch.float32, device=device),
    }


def _forward_rows(
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


def _train_model(
    table: ContextTable,
    geometry_numpy: np.ndarray,
    positions_numpy: np.ndarray,
    spectra_numpy: np.ndarray,
    fit_ids: np.ndarray,
    config: DonorTrainingConfig,
    device: torch.device,
    seed: int,
) -> tuple[AnchoredWeightCorrector, dict[str, Any]]:
    set_seed(seed)
    standardized, feature_mean, feature_std = _standardize_geometry(
        geometry_numpy, fit_ids
    )
    geometry = torch.as_tensor(standardized, dtype=torch.float32, device=device)
    positions = torch.as_tensor(positions_numpy, dtype=torch.float32, device=device)
    spectra = torch.as_tensor(spectra_numpy, dtype=torch.float32, device=device)
    context = _tensor_context(table, device)
    model = AnchoredWeightCorrector(geometry.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(len(table.query_ids))
        rows_epoch: list[tuple[float, float, float, float]] = []
        for start in range(0, len(order), config.batch_size):
            local = order[start : start + config.batch_size]
            rows = torch.as_tensor(local, dtype=torch.long, device=device)
            guard = torch.as_tensor(
                rng.integers(0, len(config.guard_radii_m), size=len(local)),
                dtype=torch.long,
                device=device,
            )
            pas, pdp, diagnostics = _forward_rows(
                model, context, rows, guard, geometry, positions, spectra
            )
            truth_ids = context["query_ids"][rows]
            pas_cosine = _power_cosine(pas, spectra[truth_ids, :PAS_SIZE])
            pdp_cosine = _power_cosine(pdp, spectra[truth_ids, PAS_SIZE:])
            correction = diagnostics["correction"]
            regularization = config.correction_regularization * (
                correction.square().sum(dim=1).mean()
                * config.neighbors
                / max(config.correction_l1_bound**2, 1e-8)
            )
            loss = 1.0 - 0.5 * (pas_cosine.mean() + pdp_cosine.mean()) + regularization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            rows_epoch.append(
                (
                    float(loss.detach().cpu()),
                    float(pas_cosine.mean().detach().cpu()),
                    float(pdp_cosine.mean().detach().cpu()),
                    float(correction.abs().sum(dim=1).mean().detach().cpu()),
                )
            )
        row = {
            "epoch": epoch,
            "loss": float(np.mean([value[0] for value in rows_epoch])),
            "pas": float(np.mean([value[1] for value in rows_epoch])),
            "pdp": float(np.mean([value[2] for value in rows_epoch])),
            "correction_l1": float(np.mean([value[3] for value in rows_epoch])),
        }
        history.append(row)
        print(f"donor epoch={epoch}/{config.epochs} {row}", flush=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "epoch": config.epochs,
        "seed": seed,
        "history": history,
    }
    return model, payload


@torch.inference_mode()
def _predict_weights(
    model: AnchoredWeightCorrector,
    checkpoint: dict[str, Any],
    table: ContextTable,
    train_geometry: np.ndarray,
    query_geometry: np.ndarray,
    train_positions: np.ndarray,
    query_positions: np.ndarray,
    spectra_numpy: np.ndarray,
    config: DonorTrainingConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    selected_train = np.asarray(train_geometry[:, 3:], dtype=np.float32)
    selected_query = np.asarray(query_geometry[:, 3:], dtype=np.float32)
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    train_geometry_tensor = torch.as_tensor(
        (selected_train - mean) / std, dtype=torch.float32, device=device
    )
    query_geometry_tensor = torch.as_tensor(
        (selected_query - mean) / std, dtype=torch.float32, device=device
    )
    train_xy = torch.as_tensor(train_positions[:, :2], dtype=torch.float32, device=device)
    query_xy = torch.as_tensor(query_positions[:, :2], dtype=torch.float32, device=device)
    spectra = torch.as_tensor(spectra_numpy, dtype=torch.float32, device=device)
    pas_weights = np.empty_like(table.weights, dtype=np.float32)
    pdp_weights = np.empty_like(table.weights, dtype=np.float32)
    model.eval()
    for start in range(0, len(table.query_ids), config.batch_size):
        end = min(start + config.batch_size, len(table.query_ids))
        local_ids = torch.as_tensor(
            table.donor_ids[start:end], dtype=torch.long, device=device
        )
        _, _, diagnostics = model(
            query_geometry_tensor[start:end],
            query_xy[start:end],
            train_geometry_tensor[local_ids],
            train_xy[local_ids],
            spectra[local_ids],
            torch.as_tensor(table.distances[start:end], device=device),
            torch.as_tensor(table.weights[start:end], device=device),
        )
        pas_weights[start:end] = diagnostics["pas_weights"].cpu().numpy()
        pdp_weights[start:end] = diagnostics["pdp_weights"].cpu().numpy()
    return pas_weights, pdp_weights


def _load_training_arrays(
    paths: RuntimePaths,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    geometry = build_geometry_regime(paths, force=False)
    train_geometry = np.asarray(geometry["train_visibility"], dtype=np.float32)
    test_geometry = np.asarray(geometry["test_visibility"], dtype=np.float32)
    with np.load(
        paths.base_dir / "official_spectra" / "descriptors.npz", allow_pickle=False
    ) as descriptors:
        valid = descriptors["valid"].astype(bool)
        spectra = np.concatenate(
            (
                np.sqrt(np.maximum(descriptors["global_pas"], 0.0)),
                np.sqrt(np.maximum(descriptors["global_pdp"], 0.0)),
            ),
            axis=1,
        ).astype(np.float32)
    return train_geometry, test_geometry, valid, spectra


def _load_fold_model(
    path: Path, geometry_size: int, device: torch.device
) -> tuple[AnchoredWeightCorrector, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = DonorTrainingConfig(**checkpoint["config"])
    model = AnchoredWeightCorrector(geometry_size, config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def train_donor_model(
    paths: RuntimePaths,
    pipeline_config: PipelineConfig,
    device: torch.device,
    batch_size: int,
    force: bool = False,
    training_config: DonorTrainingConfig | None = None,
) -> dict[str, Any]:
    """Build strict OOF donor weights and the full production donor model."""

    started = time.time()
    config = training_config or DonorTrainingConfig()
    data, layout, preparation = prepare_training_inputs(
        paths, pipeline_config, device, batch_size, force=False
    )
    folds: FoldMetadata = load_fold_metadata(layout.fold_cache, layout.fold_metadata)
    train_geometry, test_geometry, valid, spectra = _load_training_arrays(paths)
    lookup = folds.query_lookup
    fold_reports: list[dict[str, Any]] = []

    for fold in range(5):
        checkpoint_path = _fold_checkpoint(paths, fold)
        weights_path = _fold_weights(paths, fold)
        if checkpoint_path.is_file() and weights_path.is_file() and not force:
            fold_reports.append(
                {
                    "fold": fold,
                    "status": "reused",
                    "checkpoint": portable_path(checkpoint_path),
                    "weights": portable_path(weights_path),
                }
            )
            continue
        fit_ids = folds.fit_indices[fold]
        train_table = _training_context(
            paths,
            f"fold_{fold}_train",
            data.train_positions,
            data.train_sites,
            valid,
            fit_ids,
            config,
            force,
        )
        model, checkpoint = _train_model(
            train_table,
            train_geometry,
            data.train_positions,
            spectra,
            fit_ids,
            config,
            device,
            config.seed + fold * 1009,
        )
        atomic_torch_save(checkpoint_path, checkpoint)
        query_ids = folds.query_indices[fold]
        held_context = _query_context(
            paths,
            f"fold_{fold}_held",
            query_ids,
            data.train_positions[query_ids],
            data.train_sites[query_ids],
            data.train_positions,
            data.train_sites,
            valid,
            fit_ids,
            config,
            force,
        )
        pas_weights, pdp_weights = _predict_weights(
            model,
            checkpoint,
            held_context,
            train_geometry,
            train_geometry[query_ids],
            data.train_positions,
            data.train_positions[query_ids],
            spectra,
            config,
            device,
        )
        atomic_npz(
            weights_path,
            query_ids=query_ids.astype(np.int32),
            donor_ids=held_context.donor_ids.astype(np.int32),
            base_weights=held_context.weights.astype(np.float32),
            pas_weights=pas_weights,
            pdp_weights=pdp_weights,
        )
        fold_reports.append(
            {
                "fold": fold,
                "status": "trained",
                "fit_count": int(len(fit_ids)),
                "train_valid_count": int(len(train_table.query_ids)),
                "held_count": int(len(query_ids)),
                "checkpoint": portable_path(checkpoint_path),
                "weights": portable_path(weights_path),
                "history": checkpoint["history"],
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof_donor = np.empty((len(folds.query_ids), config.neighbors), dtype=np.int32)
    oof_base = np.empty_like(oof_donor, dtype=np.float32)
    oof_pas = np.empty_like(oof_base)
    oof_pdp = np.empty_like(oof_base)
    for fold in range(5):
        with np.load(_fold_weights(paths, fold), allow_pickle=False) as loaded:
            rows = lookup[loaded["query_ids"].astype(np.int64)]
            oof_donor[rows] = loaded["donor_ids"].astype(np.int32)
            oof_base[rows] = loaded["base_weights"].astype(np.float32)
            oof_pas[rows] = loaded["pas_weights"].astype(np.float32)
            oof_pdp[rows] = loaded["pdp_weights"].astype(np.float32)
    atomic_npz(
        oof_weights_path(paths),
        query_ids=folds.query_ids.astype(np.int32),
        fold_id=folds.fold_id,
        donor_ids=oof_donor,
        base_weights=oof_base,
        pas_weights=oof_pas,
        pdp_weights=oof_pdp,
    )

    full_fit = np.arange(len(data.train_positions), dtype=np.int64)
    full_checkpoint = donor_checkpoint_path(paths)
    deployment = deployment_weights_path(paths)
    if full_checkpoint.is_file() and deployment.is_file() and not force:
        production_report: dict[str, Any] = {
            "status": "reused",
            "checkpoint": portable_path(full_checkpoint),
            "deployment": portable_path(deployment),
        }
    else:
        full_table = _training_context(
            paths,
            "full_train",
            data.train_positions,
            data.train_sites,
            valid,
            full_fit,
            config,
            force,
        )
        model, checkpoint = _train_model(
            full_table,
            train_geometry,
            data.train_positions,
            spectra,
            full_fit,
            config,
            device,
            config.seed + 90_000,
        )
        atomic_torch_save(full_checkpoint, checkpoint)
        test_ids = np.arange(len(data.test_positions), dtype=np.int64)
        test_context = _query_context(
            paths,
            "test",
            test_ids,
            data.test_positions,
            data.test_sites,
            data.train_positions,
            data.train_sites,
            valid,
            full_fit,
            config,
            force,
        )
        pas_weights, pdp_weights = _predict_weights(
            model,
            checkpoint,
            test_context,
            train_geometry,
            test_geometry,
            data.train_positions,
            data.test_positions,
            spectra,
            config,
            device,
        )
        atomic_npz(
            deployment,
            donor_ids=test_context.donor_ids.astype(np.int32),
            base_weights=test_context.weights.astype(np.float32),
            pas_weights=pas_weights,
            pdp_weights=pdp_weights,
            epoch=np.int32(config.epochs),
        )
        production_report = {
            "status": "trained",
            "fit_count": int(len(full_fit)),
            "train_valid_count": int(len(full_table.query_ids)),
            "checkpoint": portable_path(full_checkpoint),
            "deployment": portable_path(deployment),
            "history": checkpoint["history"],
        }

    report = {
        "stage": "train_donor_model",
        "status": "complete",
        "config": asdict(config),
        "preparation": preparation,
        "folds": fold_reports,
        "oof_weights": portable_path(oof_weights_path(paths)),
        "production": production_report,
        "wall_seconds": float(time.time() - started),
        "device": str(device),
        "peak_torch_memory_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
        "measured_bottleneck": "donor context construction and six neural fits",
    }
    atomic_json(paths.report_dir / "train_donor_model.json", report)
    return report
