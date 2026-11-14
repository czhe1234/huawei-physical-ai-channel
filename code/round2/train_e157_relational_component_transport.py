#!/usr/bin/env python3
"""Learn train-to-query transport of official 24-component channel states."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

import analyze_e153_v7_v12_peak_actions as e153
import run_e154_local_spectral_transport as e154
from data import load_positions
from project_paths import PATHS
from run_rbf_spectral_field import OOF_META, statistics
from run_v9_joint_sionna_migration import point_metrics


VERSION = "Round2-E157-relational-component-transport"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "relational_component_transport"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_relational_component_transport"
REPORT = ANALYSIS_DIR / "relational_component_transport.json"
METRICS = ARTIFACT_DIR / "relational_component_transport_metrics.npz"

COMPONENTS = (
    PATHS.artifacts
    / "round2"
    / "v4_path_component_transport"
    / "train_components_k24_os2_ref2.npz"
)
FOLD_CACHE = (
    PATHS.artifacts / "round2" / "foundation" / "round2_spatial_hole_folds.npz"
)
GEOMETRY_CACHE = (
    PATHS.artifacts
    / "round2"
    / "regime_lifecycle_support"
    / "geodesic_corridor_features.npz"
)
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
V7_CHANNEL = e153.V7_CHANNEL

MP = 2
MH = 16
MV = 8
N_UE = 4
S_SC = 192
PATHS_PER_POINT = 24
PORTS = MP * N_UE
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    neighbors: int = 8
    minimum_train_distance: float = 4.0
    hidden: int = 128
    heads: int = 4
    layers: int = 2
    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    maximum_h_shift: float = 2.0
    maximum_v_shift: float = 1.5
    maximum_delay_shift: float = 8.0
    maximum_log_mass_shift: float = 2.0
    maximum_port_logit_shift: float = 1.5
    action_alphas: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.30)
    projection_cycles: int = 4
    seed: int = 20_260_810_157


@dataclass(frozen=True)
class PointTensors:
    coordinates: torch.Tensor
    fractions: torch.Tensor
    port_share: torch.Tensor
    geometry: torch.Tensor
    positions: torch.Tensor
    site: torch.Tensor
    bs_relative: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--folds", default="all")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_numpy(value: np.ndarray, axes: int | tuple[int, ...]) -> np.ndarray:
    output = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = output.sum(axis=axes, keepdims=True, dtype=np.float64)
    return np.divide(
        output,
        np.maximum(total, TINY),
        out=np.zeros_like(output),
        where=total > 0.0,
    )


def normalize_torch(value: torch.Tensor, dims: int | tuple[int, ...]) -> torch.Tensor:
    output = value.clamp_min(0.0)
    total = output.sum(dim=dims, keepdim=True)
    return torch.where(total > TINY, output / total.clamp_min(TINY), 0.0)


def periodic_features(coordinates: torch.Tensor) -> torch.Tensor:
    period = coordinates.new_tensor((MH, MV, S_SC))
    phase = 2.0 * math.pi * coordinates / period
    return torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)


def spectral_kernels(
    coordinates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    horizontal_index = torch.arange(MH, dtype=torch.float32, device=coordinates.device)
    vertical_index = torch.arange(MV, dtype=torch.float32, device=coordinates.device)
    delay_index = torch.arange(S_SC, dtype=torch.float32, device=coordinates.device)
    horizontal_wave = torch.exp(
        2j
        * math.pi
        * coordinates[..., 0, None]
        * horizontal_index
        / MH
    )
    vertical_wave = torch.exp(
        2j * math.pi * coordinates[..., 1, None] * vertical_index / MV
    )
    delay_wave = torch.exp(
        -2j * math.pi * coordinates[..., 2, None] * delay_index / S_SC
    )
    horizontal = torch.fft.fft(horizontal_wave, dim=-1).abs().square().float()
    vertical = torch.fft.fft(vertical_wave, dim=-1).abs().square().float()
    delay = torch.fft.ifft(delay_wave, dim=-1).abs().square().float()
    return (
        normalize_torch(horizontal, -1),
        normalize_torch(vertical, -1),
        normalize_torch(delay, -1),
    )


def render_compact_spectra(
    coordinates: torch.Tensor,
    mass: torch.Tensor,
    port_share: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    horizontal, vertical, delay = spectral_kernels(coordinates)
    port_energy = mass[..., None] * port_share
    port_energy = port_energy.reshape(*port_energy.shape[:-1], MP, N_UE)
    angle = torch.einsum("btpu,bth,btv->buhv", port_energy, horizontal, vertical)
    delay_spectrum = torch.einsum("btpu,btd->bpud", port_energy, delay)
    return normalize_torch(angle, (2, 3)), normalize_torch(delay_spectrum, 3)


def mean_cosine(
    prediction: torch.Tensor, truth: torch.Tensor, dims: tuple[int, ...]
) -> torch.Tensor:
    numerator = (prediction * truth).sum(dim=dims, dtype=torch.float64)
    norm_product = (
        prediction.square().sum(dim=dims, dtype=torch.float64)
        * truth.square().sum(dim=dims, dtype=torch.float64)
    )
    stable_root = torch.sqrt(norm_product.clamp_min(1e-24))
    denominator = torch.where(norm_product == 0.0, 1e-12, stable_root)
    cosine = numerator / denominator
    return cosine.mean()


class RelationalComponentTransport(nn.Module):
    def __init__(self, config: Config, geometry_dim: int) -> None:
        super().__init__()
        self.config = config
        component_dim = 6 + PORTS + 2
        edge_dim = 8 + 2 * geometry_dim
        query_dim = geometry_dim + 6
        self.token_encoder = nn.Sequential(
            nn.Linear(component_dim + edge_dim, config.hidden),
            nn.GELU(),
            nn.Linear(config.hidden, config.hidden),
            nn.LayerNorm(config.hidden),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(query_dim, config.hidden),
            nn.GELU(),
            nn.Linear(config.hidden, config.hidden),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden,
            nhead=config.heads,
            dim_feedforward=4 * config.hidden,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.competition = nn.TransformerEncoder(
            layer, num_layers=config.layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(config.hidden)
        self.output = nn.Linear(config.hidden, 1 + 3 + 1 + PORTS)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        points: PointTensors,
        query_ids: torch.Tensor,
        neighbor_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, neighbors = neighbor_ids.shape
        paths = points.coordinates.shape[1]
        source_coordinate = points.coordinates[neighbor_ids]
        source_fraction = points.fractions[neighbor_ids]
        source_port = points.port_share[neighbor_ids]

        source_geometry = points.geometry[neighbor_ids]
        target_geometry = points.geometry[query_ids]
        geometry_delta = source_geometry - target_geometry[:, None]
        source_position = points.positions[neighbor_ids]
        target_position = points.positions[query_ids]
        offset = source_position - target_position[:, None]
        distance = torch.linalg.vector_norm(offset, dim=-1).clamp_min(1e-3)
        direction = offset / distance[..., None]
        source_radial = torch.linalg.vector_norm(points.bs_relative[neighbor_ids], dim=-1)
        target_radial = torch.linalg.vector_norm(points.bs_relative[query_ids], dim=-1)
        radial_delta = source_radial - target_radial[:, None]
        neighbor_rank = torch.arange(
            neighbors, dtype=torch.float32, device=query_ids.device
        )[None].expand(batch, -1)

        edge = torch.cat(
            (
                offset / 30.0,
                distance[..., None] / 30.0,
                direction,
                radial_delta[..., None] / 30.0,
                neighbor_rank[..., None] / max(neighbors - 1, 1),
                torch.exp(-distance[..., None] / 12.0),
                source_geometry,
                geometry_delta,
            ),
            dim=-1,
        )
        edge = edge[:, :, None].expand(-1, -1, paths, -1)
        rank = torch.arange(paths, dtype=torch.float32, device=query_ids.device)
        rank = rank[None, None, :, None].expand(batch, neighbors, -1, -1)
        component = torch.cat(
            (
                periodic_features(source_coordinate),
                torch.sqrt(source_port.clamp_min(0.0)),
                torch.log10(source_fraction.clamp_min(1e-10))[..., None] / 10.0,
                rank / max(paths - 1, 1),
            ),
            dim=-1,
        )

        target_radial_vector = points.bs_relative[query_ids]
        target_radial_norm = torch.linalg.vector_norm(
            target_radial_vector, dim=-1, keepdim=True
        ).clamp_min(1e-3)
        query_context = torch.cat(
            (
                target_geometry,
                target_radial_vector / 150.0,
                target_radial_norm / 200.0,
                target_radial_vector / target_radial_norm,
                points.site[query_ids, None].float(),
            ),
            dim=-1,
        )
        token = torch.cat((component, edge), dim=-1).reshape(batch, -1, component.shape[-1] + edge.shape[-1])
        hidden = self.token_encoder(token) + self.query_encoder(query_context)[:, None]
        hidden = self.final_norm(self.competition(hidden))
        raw = self.output(hidden)

        source_coordinate = source_coordinate.reshape(batch, -1, 3)
        source_fraction = source_fraction.reshape(batch, -1)
        source_port = source_port.reshape(batch, -1, PORTS)
        distance_weight = torch.softmax(-distance / 8.0, dim=1)
        distance_weight = distance_weight[:, :, None].expand(-1, -1, paths).reshape(batch, -1)

        gate = 0.05 + 0.95 * torch.sigmoid(raw[..., 0])
        shift_limit = raw.new_tensor(
            (
                self.config.maximum_h_shift,
                self.config.maximum_v_shift,
                self.config.maximum_delay_shift,
            )
        )
        coordinate = source_coordinate + torch.tanh(raw[..., 1:4]) * shift_limit
        mass = (
            source_fraction
            * distance_weight
            * gate
            * torch.exp(
                self.config.maximum_log_mass_shift * torch.tanh(raw[..., 4])
            )
        )
        port_logits = (
            torch.log(source_port.clamp_min(1e-8))
            + self.config.maximum_port_logit_shift * torch.tanh(raw[..., 5:])
        )
        port_share = torch.softmax(port_logits, dim=-1)
        return render_compact_spectra(coordinate, mass, port_share)


def load_fold(cache: np.lib.npyio.NpzFile, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        cache[f"fold{fold}_fit"].astype(np.int64),
        cache[f"fold{fold}_query"].astype(np.int64),
        cache[f"fold{fold}_ignored"].astype(np.int64),
    )


def build_neighbor_ids(
    target_ids: np.ndarray,
    bank_ids: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    count: int,
    minimum_distance: float,
) -> np.ndarray:
    output = np.empty((len(target_ids), count), dtype=np.int64)
    for site_value in (0, 1):
        rows = np.flatnonzero(site[target_ids] == site_value)
        bank = bank_ids[site[bank_ids] == site_value]
        tree = cKDTree(positions[bank, :2])
        request = min(len(bank), max(count + 24, 64))
        distance, local = tree.query(positions[target_ids[rows], :2], k=request, workers=-1)
        distance = np.atleast_2d(distance)
        local = np.atleast_2d(local)
        for destination, target, values, indices in zip(
            rows, target_ids[rows], distance, local, strict=True
        ):
            candidates = bank[np.asarray(indices, dtype=np.int64)]
            valid = (candidates != target) & (np.asarray(values) >= minimum_distance)
            selected = candidates[valid][:count]
            if len(selected) < count:
                fallback = candidates[candidates != target]
                selected = np.unique(np.concatenate((selected, fallback)), return_index=False)[:count]
            if len(selected) != count:
                raise RuntimeError(f"Only {len(selected)} neighbors for point {target}")
            output[destination] = selected
    return output


def prepare_point_tensors(
    device: torch.device,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    fit_ids: np.ndarray,
) -> PointTensors:
    cache = np.load(COMPONENTS, allow_pickle=False)
    coordinate = cache["coordinates"].astype(np.float32)
    coefficient = cache["coefficients"].astype(np.complex64)
    path_energy = np.sum(np.abs(coefficient) ** 2, axis=-1, dtype=np.float64)
    point_power = cache["point_power"].astype(np.float64)
    fraction = np.divide(
        path_energy,
        np.maximum(point_power[:, None], TINY),
        out=np.zeros_like(path_energy),
        where=point_power[:, None] > 0.0,
    ).astype(np.float32)
    port_energy = np.abs(coefficient) ** 2
    port_share = np.divide(
        port_energy,
        np.maximum(path_energy[..., None], TINY),
        out=np.full_like(port_energy, 1.0 / PORTS, dtype=np.float32),
        where=path_energy[..., None] > 0.0,
    ).astype(np.float32)

    geometry_raw = np.load(GEOMETRY_CACHE, allow_pickle=False)["train_features"].astype(np.float32)
    center = np.median(geometry_raw[fit_ids], axis=0)
    scale = np.quantile(np.abs(geometry_raw[fit_ids] - center), 0.75, axis=0)
    scale = np.maximum(scale, 1e-3)
    geometry = np.clip((geometry_raw - center) / scale, -6.0, 6.0).astype(np.float32)
    relative = positions[:, :2] - bs_positions[site, :2]
    return PointTensors(
        coordinates=torch.as_tensor(coordinate, device=device),
        fractions=torch.as_tensor(fraction, device=device),
        port_share=torch.as_tensor(port_share, device=device),
        geometry=torch.as_tensor(geometry, device=device),
        positions=torch.as_tensor(positions[:, :2], dtype=torch.float32, device=device),
        site=torch.as_tensor(site, dtype=torch.int64, device=device),
        bs_relative=torch.as_tensor(relative, dtype=torch.float32, device=device),
    )


def predict_compact(
    model: nn.Module,
    points: PointTensors,
    target_ids: np.ndarray,
    neighbor_ids: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    angle = np.empty((len(target_ids), N_UE, MH, MV), dtype=np.float32)
    delay = np.empty((len(target_ids), MP, N_UE, S_SC), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(target_ids), batch_size):
            stop = min(start + batch_size, len(target_ids))
            query = torch.as_tensor(target_ids[start:stop], dtype=torch.long, device=device)
            neighbor = torch.as_tensor(
                neighbor_ids[start:stop], dtype=torch.long, device=device
            )
            pred_angle, pred_delay = model(points, query, neighbor)
            angle[start:stop] = pred_angle.cpu().numpy()
            delay[start:stop] = pred_delay.cpu().numpy()
    return angle, delay


def compact_cosines(
    angle: np.ndarray,
    delay: np.ndarray,
    true_angle: np.ndarray,
    true_delay: np.ndarray,
) -> tuple[float, float]:
    return (
        float(np.mean(e154.endpoint_cosine(angle, true_angle, (2, 3)))),
        float(np.mean(e154.endpoint_cosine(delay, true_delay, (3,)))),
    )


def blend_compact(base: np.ndarray, endpoint: np.ndarray, alpha: float, axes: tuple[int, ...]) -> np.ndarray:
    if alpha <= 0.0:
        return np.asarray(base, dtype=np.float32)
    mixed = ((1.0 - alpha) * np.sqrt(np.maximum(base, 0.0)) + alpha * np.sqrt(np.maximum(endpoint, 0.0))) ** 2
    return normalize_numpy(mixed, axes)


def score_action(
    base: np.ndarray,
    truth_all: np.ndarray,
    query_ids: np.ndarray,
    base_angle: np.ndarray,
    base_delay: np.ndarray,
    endpoint_angle: np.ndarray,
    endpoint_delay: np.ndarray,
    alpha: float,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    arrays = tuple(np.empty(len(query_ids), dtype=np.float64) for _ in range(4))
    for start in range(0, len(query_ids), batch_size):
        stop = min(start + batch_size, len(query_ids))
        base_tensor = torch.as_tensor(
            np.array(base[start:stop], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:stop]], dtype=np.complex64, copy=True),
            device=device,
        )
        if alpha <= 0.0:
            prediction = base_tensor
        else:
            angle = blend_compact(
                base_angle[start:stop], endpoint_angle[start:stop], alpha, (2, 3)
            )
            delay = blend_compact(
                base_delay[start:stop], endpoint_delay[start:stop], alpha, (3,)
            )
            with torch.inference_mode():
                desired_pas, desired_pdp = e154.full_targets(
                    base_tensor,
                    torch.as_tensor(angle, device=device),
                    torch.as_tensor(delay, device=device),
                )
                projection_config = e154.Config(cycles=4, order="pas_pdp")
                prediction = e154.project_variant(
                    base_tensor,
                    desired_pas,
                    desired_pdp,
                    "joint_pas_pdp",
                    projection_config,
                )
        with torch.inference_mode():
            values = point_metrics(
                prediction,
                truth,
                e154.pas_distribution(truth),
                e154.pdp_distribution(truth),
            )
        for destination, source in zip(arrays, values, strict=True):
            destination[start:stop] = source
    result = statistics(*arrays)
    return {key: float(value) for key, value in result.items()}, arrays


def canonical_rows(query_ids: np.ndarray) -> dict[int, int]:
    return {int(point): row for row, point in enumerate(query_ids)}


def train_fold(
    fold: int,
    config: Config,
    device: torch.device,
    setup: object,
    positions: np.ndarray,
    site: np.ndarray,
    folds: np.lib.npyio.NpzFile,
    canonical_query: np.ndarray,
    canonical_row: dict[int, int],
    v7_all: np.ndarray,
    v7_angle_all: np.ndarray,
    v7_delay_all: np.ndarray,
    truth_all: np.ndarray,
    true_angle_all: np.ndarray,
    true_delay_all: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    inner_fold = (fold + 1) % 5
    _, outer_ids, outer_ignored = load_fold(folds, fold)
    _, inner_ids, inner_ignored = load_fold(folds, inner_fold)
    excluded = np.unique(
        np.concatenate((outer_ids, outer_ignored, inner_ids, inner_ignored))
    )
    fit_ids = np.setdiff1d(np.arange(len(positions), dtype=np.int64), excluded)
    train_ids = fit_ids[true_angle_all[fit_ids].sum(axis=(1, 2, 3)) > 0.0]
    train_neighbors = build_neighbor_ids(
        train_ids,
        fit_ids,
        positions,
        site,
        config.neighbors,
        config.minimum_train_distance,
    )
    inner_neighbors = build_neighbor_ids(
        inner_ids,
        fit_ids,
        positions,
        site,
        config.neighbors,
        config.minimum_train_distance,
    )
    outer_neighbors = build_neighbor_ids(
        outer_ids,
        fit_ids,
        positions,
        site,
        config.neighbors,
        config.minimum_train_distance,
    )
    points = prepare_point_tensors(device, positions, site, setup.bs_positions, fit_ids)
    model = RelationalComponentTransport(config, points.geometry.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_angle = torch.as_tensor(true_angle_all, device=device)
    train_delay = torch.as_tensor(true_delay_all, device=device)
    rng = np.random.default_rng(config.seed + fold)
    row_lookup = canonical_rows(canonical_query)
    inner_rows = np.asarray([row_lookup[int(point)] for point in inner_ids], dtype=np.int64)
    outer_rows = np.asarray([row_lookup[int(point)] for point in outer_ids], dtype=np.int64)
    history: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    checkpoint = ARTIFACT_DIR / f"outer{fold}_relational_transport.pt"

    for epoch in range(config.epochs + 1):
        if epoch > 0:
            model.train()
            order = rng.permutation(len(train_ids))
            running_loss = 0.0
            steps = 0
            for start in range(0, len(order), config.batch_size):
                rows = order[start : start + config.batch_size]
                query = torch.as_tensor(
                    train_ids[rows], dtype=torch.long, device=device
                )
                neighbor = torch.as_tensor(
                    train_neighbors[rows], dtype=torch.long, device=device
                )
                optimizer.zero_grad(set_to_none=True)
                pred_angle, pred_delay = model(points, query, neighbor)
                target_angle = train_angle[query]
                target_delay = train_delay[query]
                pas_cosine = mean_cosine(pred_angle, target_angle, (2, 3))
                pdp_cosine = mean_cosine(pred_delay, target_delay, (3,))
                loss = 0.55 * (1.0 - pas_cosine) + 0.45 * (1.0 - pdp_cosine)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss in fold {fold} epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                if not all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ):
                    raise RuntimeError(
                        f"Non-finite gradient in fold {fold} epoch {epoch}"
                    )
                optimizer.step()
                running_loss += float(loss.detach())
                steps += 1
            train_loss = running_loss / max(steps, 1)
        else:
            train_loss = 0.0

        inner_angle, inner_delay = predict_compact(
            model,
            points,
            inner_ids,
            inner_neighbors,
            config.batch_size,
            device,
        )
        endpoint_pas, endpoint_pdp = compact_cosines(
            inner_angle,
            inner_delay,
            true_angle_all[inner_ids],
            true_delay_all[inner_ids],
        )
        alpha_rows = []
        for alpha in config.action_alphas:
            score, _ = score_action(
                v7_all[inner_rows],
                truth_all,
                inner_ids,
                v7_angle_all[inner_rows],
                v7_delay_all[inner_rows],
                inner_angle,
                inner_delay,
                alpha,
                device,
                config.batch_size,
            )
            alpha_rows.append({"alpha": alpha, **score})
        selected = max(alpha_rows, key=lambda row: float(row["score"]))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "inner_endpoint_pas": endpoint_pas,
            "inner_endpoint_pdp": endpoint_pdp,
            "selected_alpha": float(selected["alpha"]),
            "selected_score": float(selected["score"]),
            "alpha_scores": alpha_rows,
        }
        history.append(row)
        if best is None or float(selected["score"]) > float(best["selected_score"]):
            best = row.copy()
            torch.save(
                {
                    "version": VERSION,
                    "fold": fold,
                    "config": asdict(config),
                    "model": model.state_dict(),
                    "selection": best,
                },
                checkpoint,
            )
        print(
            f"E157 fold={fold} epoch={epoch:02d} loss={train_loss:.6f} "
            f"inner_endpoint={endpoint_pas:.6f}/{endpoint_pdp:.6f} "
            f"alpha={selected['alpha']:.2f} score={selected['score']:.6f}",
            flush=True,
        )

    if best is None:
        raise RuntimeError("No checkpoint selected")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    outer_angle, outer_delay = predict_compact(
        model,
        points,
        outer_ids,
        outer_neighbors,
        config.batch_size,
        device,
    )
    endpoint_pas, endpoint_pdp = compact_cosines(
        outer_angle,
        outer_delay,
        true_angle_all[outer_ids],
        true_delay_all[outer_ids],
    )
    outer_score, outer_arrays = score_action(
        v7_all[outer_rows],
        truth_all,
        outer_ids,
        v7_angle_all[outer_rows],
        v7_delay_all[outer_rows],
        outer_angle,
        outer_delay,
        float(best["selected_alpha"]),
        device,
        config.batch_size,
    )
    baseline_score, baseline_arrays = score_action(
        v7_all[outer_rows],
        truth_all,
        outer_ids,
        v7_angle_all[outer_rows],
        v7_delay_all[outer_rows],
        outer_angle,
        outer_delay,
        0.0,
        device,
        config.batch_size,
    )
    report = {
        "fold": fold,
        "inner_fold": inner_fold,
        "fit_points": int(len(fit_ids)),
        "train_points": int(len(train_ids)),
        "inner_points": int(len(inner_ids)),
        "outer_points": int(len(outer_ids)),
        "selected_epoch": int(best["epoch"]),
        "selected_alpha": float(best["selected_alpha"]),
        "outer_endpoint_pas": endpoint_pas,
        "outer_endpoint_pdp": endpoint_pdp,
        "baseline": baseline_score,
        "candidate": outer_score,
        "delta": float(outer_score["score"] - baseline_score["score"]),
        "history": history,
        "checkpoint": PATHS.relative(checkpoint),
    }
    arrays = {
        "query_ids": outer_ids,
        "canonical_rows": outer_rows,
        "endpoint_angle": outer_angle,
        "endpoint_delay": outer_delay,
        **{
            f"candidate_{name}": value
            for name, value in zip(
                ("pas", "pdp", "error", "energy"), outer_arrays, strict=True
            )
        },
        **{
            f"baseline_{name}": value
            for name, value in zip(
                ("pas", "pdp", "error", "energy"), baseline_arrays, strict=True
            )
        },
    }
    return report, arrays


def finite_payload(value: object) -> bool:
    if isinstance(value, dict):
        return all(finite_payload(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_payload(item) for item in value)
    if isinstance(value, float):
        return math.isnan(value) or math.isfinite(value)
    return True


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    config = Config(epochs=args.epochs, batch_size=args.batch_size)
    set_seed(config.seed)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    required = (
        COMPONENTS,
        FOLD_CACHE,
        GEOMETRY_CACHE,
        TRAIN_CHANNEL,
        V7_CHANNEL,
        OOF_META,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    setup, positions, _, site, _ = load_positions()
    folds = np.load(FOLD_CACHE, allow_pickle=False)
    metadata = np.load(OOF_META, allow_pickle=False)
    canonical_query = metadata["query_ids"].astype(np.int64)
    canonical_fold = metadata["fold_id"].astype(np.int8)
    if len(np.unique(canonical_query)) != 500:
        raise RuntimeError("Canonical query ids are not 500 unique points")
    fold_union = np.concatenate([folds[f"fold{fold}_query"] for fold in range(5)])
    if set(map(int, fold_union)) != set(map(int, canonical_query)):
        raise RuntimeError("Fold queries do not match canonical 500 points")

    v7_all = np.load(V7_CHANNEL, mmap_mode="r")
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    true_angle_all, true_delay_all = e154.load_compact_train_spectra()
    v7_angle_all, v7_delay_all, _ = e153.channel_group_maps(
        v7_all, device, args.batch_size
    )
    canonical_row = canonical_rows(canonical_query)
    requested = range(5) if args.folds == "all" else [int(value) for value in args.folds.split(",")]
    fold_reports = []
    fold_arrays = []
    for fold in requested:
        report, arrays = train_fold(
            fold,
            config,
            device,
            setup,
            positions,
            site,
            folds,
            canonical_query,
            canonical_row,
            v7_all,
            v7_angle_all,
            v7_delay_all,
            truth_all,
            true_angle_all,
            true_delay_all,
        )
        fold_reports.append(report)
        fold_arrays.append(arrays)

    if args.folds != "all":
        payload = {
            "version": VERSION,
            "status": "partial_screen",
            "config": asdict(config),
            "folds": fold_reports,
            "runtime_seconds": time.time() - started,
        }
        REPORT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    candidate = [np.empty(500, dtype=np.float64) for _ in range(4)]
    baseline = [np.empty(500, dtype=np.float64) for _ in range(4)]
    endpoint_angle = np.empty((500, N_UE, MH, MV), dtype=np.float32)
    endpoint_delay = np.empty((500, MP, N_UE, S_SC), dtype=np.float32)
    for arrays in fold_arrays:
        rows = arrays["canonical_rows"]
        endpoint_angle[rows] = arrays["endpoint_angle"]
        endpoint_delay[rows] = arrays["endpoint_delay"]
        for index, name in enumerate(("pas", "pdp", "error", "energy")):
            candidate[index][rows] = arrays[f"candidate_{name}"]
            baseline[index][rows] = arrays[f"baseline_{name}"]

    overall_candidate = statistics(*candidate)
    overall_baseline = statistics(*baseline)
    if abs(float(overall_baseline["score"]) - 0.6444563456) > 1e-8:
        raise RuntimeError(f"V7 score mismatch: {overall_baseline}")
    fold_deltas = []
    for fold in range(5):
        selected = canonical_fold == fold
        fold_deltas.append(
            float(
                statistics(*candidate, selected)["score"]
                - statistics(*baseline, selected)["score"]
            )
        )
    gain = float(overall_candidate["score"] - overall_baseline["score"])
    positive_folds = int(np.sum(np.asarray(fold_deltas) > 0.0))
    payload = {
        "version": VERSION,
        "status": "strict_complete",
        "config": asdict(config),
        "baseline": overall_baseline,
        "candidate": overall_candidate,
        "delta_vs_v7": gain,
        "fold_deltas": fold_deltas,
        "positive_folds": positive_folds,
        "decision": {
            "basic_gate_passed": bool(gain > 0.0 and positive_folds >= 3),
            "material_gate_passed": bool(gain >= 0.005 and positive_folds >= 3),
            "minimum_gain": 0.005,
            "minimum_positive_folds": 3,
            "generate_candidate": False,
        },
        "folds": fold_reports,
        "runtime_seconds": time.time() - started,
    }
    if not finite_payload(payload):
        raise RuntimeError("Non-finite report payload")
    np.savez_compressed(
        METRICS,
        query_ids=canonical_query,
        fold_id=canonical_fold,
        endpoint_angle=endpoint_angle.astype(np.float16),
        endpoint_delay=endpoint_delay.astype(np.float16),
        **{
            f"candidate_{name}": candidate[index]
            for index, name in enumerate(("pas", "pdp", "error", "energy"))
        },
        **{
            f"baseline_{name}": baseline[index]
            for index, name in enumerate(("pas", "pdp", "error", "energy"))
        },
    )
    REPORT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
