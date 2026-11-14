#!/usr/bin/env python3
"""Test whether complete boundary H can predict a held complex wavefield."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch import nn

import analyze_e153_v7_v12_peak_actions as e153
from data import load_positions
from official_metrics import aggregate_statistics, point_metric_statistics
from project_paths import PATHS
from run_foundation_benchmark import POWER_CACHE
from run_rbf_spectral_field import FOLD_CACHE, OOF_META


VERSION = "Round2-E158-full-boundary-wavefield-operator"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "full_boundary_wavefield_operator"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_full_boundary_wavefield_operator"
REPORT = ANALYSIS_DIR / "full_boundary_wavefield_operator.json"

TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
MAP_FEATURES = (
    PATHS.artifacts
    / "round2"
    / "regime_lifecycle_support"
    / "geodesic_corridor_features.npz"
)

OUTER = 3
MP = 2
MH = 16
MV = 8
N_UE = 4
S_SC = 192
OUTPUT_CHANNELS = MP * N_UE * 2
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    neighbors: int = 8
    train_guard_m: float = 6.0
    hidden: int = 32
    blocks: int = 3
    batch_size: int = 2
    capacity_points: int = 16
    capacity_steps: int = 200
    capacity_threshold: float = 0.75
    screen_steps: int = 1200
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.5
    log_interval: int = 100
    seed: int = 20260810158


@dataclass(frozen=True)
class NeighborTable:
    query_ids: np.ndarray
    donor_ids: np.ndarray
    query_features: np.ndarray
    donor_features: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("identity", "capacity", "screen", "all"), default="all"
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def channel_to_angle_delay(channel: torch.Tensor) -> torch.Tensor:
    """Map [...,256,4,192] H to [...,2,16,8,4,192] complex AD."""

    leading = channel.shape[:-3]
    shaped = channel.reshape(*leading, MP, MH, MV, N_UE, S_SC)
    angle = torch.fft.fft2(shaped, dim=(-4, -3), norm="ortho")
    return torch.fft.ifft(angle, dim=-1, norm="ortho")


def angle_delay_to_channel(angle_delay: torch.Tensor) -> torch.Tensor:
    frequency = torch.fft.fft(angle_delay, dim=-1, norm="ortho")
    channel = torch.fft.ifft2(frequency, dim=(-4, -3), norm="ortho")
    leading = channel.shape[:-5]
    return channel.reshape(*leading, MP * MH * MV, N_UE, S_SC)


def identity_check(train_channel: np.ndarray, fit_ids: np.ndarray, device: torch.device) -> dict[str, float]:
    point_power = np.load(POWER_CACHE).astype(np.float64)
    rows = fit_ids[point_power[fit_ids] > 0.0][:4]
    source_numpy = np.asarray(train_channel[rows], dtype=np.complex64)
    source = torch.as_tensor(source_numpy, dtype=torch.complex64, device=device)
    restored = angle_delay_to_channel(channel_to_angle_delay(source))
    restored_numpy = restored.cpu().numpy().astype(np.complex64)
    difference = restored_numpy - source_numpy
    relative_l2 = float(
        np.sqrt(np.sum(np.abs(difference) ** 2, dtype=np.float64))
        / max(np.sqrt(np.sum(np.abs(source_numpy) ** 2, dtype=np.float64)), TINY)
    )
    maximum_absolute = float(np.max(np.abs(difference)))
    source_metric = aggregate_statistics(
        point_metric_statistics(source_numpy, source_numpy)
    )
    restored_metric = aggregate_statistics(
        point_metric_statistics(restored_numpy, source_numpy)
    )
    score_difference = float(restored_metric["score"] - source_metric["score"])
    passed = relative_l2 <= 1e-6 and abs(score_difference) <= 1e-8
    if not passed:
        raise RuntimeError(
            "lossless angle-delay identity failed: "
            f"relative_l2={relative_l2:.3e}, score_delta={score_difference:.3e}"
        )
    return {
        "relative_l2": relative_l2,
        "maximum_absolute": maximum_absolute,
        "source_score": float(source_metric["score"]),
        "restored_score": float(restored_metric["score"]),
        "score_difference": score_difference,
        "passed": True,
    }


def point_rms(point_power: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(point_power, 0.0) / float(MP * MH * MV * N_UE * S_SC))


def build_site_scales(
    rms: np.ndarray, site: np.ndarray, fit_ids: np.ndarray
) -> np.ndarray:
    output = np.empty(2, dtype=np.float32)
    for site_value in range(2):
        local = rms[fit_ids[(site[fit_ids] == site_value) & (rms[fit_ids] > 0.0)]]
        if len(local) == 0:
            raise RuntimeError(f"site {site_value} has no nonzero fit points")
        output[site_value] = np.float32(np.quantile(local, 0.90))
    return output


def build_query_features(
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    map_features: np.ndarray,
) -> np.ndarray:
    serving = bs_positions[site]
    delta = positions - serving
    distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-6)
    bearing = np.arctan2(delta[:, 1], delta[:, 0])
    return np.concatenate(
        (
            delta[:, :2] / 200.0,
            distance[:, None] / 250.0,
            np.sin(bearing)[:, None],
            np.cos(bearing)[:, None],
            (site == 0)[:, None].astype(np.float32),
            (site == 1)[:, None].astype(np.float32),
            map_features,
        ),
        axis=1,
    ).astype(np.float32)


def build_neighbor_table(
    query_ids: np.ndarray,
    donor_pool: np.ndarray,
    positions: np.ndarray,
    site: np.ndarray,
    bs_positions: np.ndarray,
    map_features: np.ndarray,
    query_features: np.ndarray,
    rms: np.ndarray,
    site_scales: np.ndarray,
    neighbors: int,
    minimum_distance: float,
) -> NeighborTable:
    donor_ids = np.empty((len(query_ids), neighbors), dtype=np.int32)
    donor_features = np.empty((len(query_ids), neighbors, 33), dtype=np.float32)
    radial = np.linalg.norm(positions - bs_positions[site], axis=1)
    for site_value in np.unique(site[query_ids]):
        site_value = int(site_value)
        rows = np.flatnonzero(site[query_ids] == site_value)
        bank = donor_pool[site[donor_pool] == site_value]
        tree = cKDTree(positions[bank, :2])
        search = min(len(bank), max(96, neighbors * 12))
        distances, local = tree.query(
            positions[query_ids[rows], :2], k=search, workers=-1
        )
        distances = np.atleast_2d(distances)
        local = np.atleast_2d(local)
        for local_row, output_row in enumerate(rows):
            query_id = int(query_ids[output_row])
            candidates = bank[local[local_row]]
            candidate_distance = distances[local_row]
            keep = (candidates != query_id) & (candidate_distance >= minimum_distance)
            candidates = candidates[keep]
            candidate_distance = candidate_distance[keep]
            if len(candidates) < neighbors:
                raise RuntimeError(
                    f"query {query_id} has only {len(candidates)} donors beyond "
                    f"{minimum_distance}m"
                )
            selected = candidates[:neighbors]
            selected_distance = candidate_distance[:neighbors]
            donor_ids[output_row] = selected.astype(np.int32)
            relative = positions[selected, :2] - positions[query_id, :2]
            feature = np.concatenate(
                (
                    relative / 32.0,
                    selected_distance[:, None] / 32.0,
                    ((radial[selected] - radial[query_id]) / 32.0)[:, None],
                    np.clip(
                        map_features[selected] - map_features[query_id][None, :],
                        -6.0,
                        6.0,
                    ),
                    np.log1p(
                        rms[selected, None]
                        / max(float(site_scales[site_value]), 1e-12)
                    ),
                ),
                axis=1,
            )
            donor_features[output_row] = feature.astype(np.float32)
    return NeighborTable(
        query_ids=query_ids.astype(np.int64),
        donor_ids=donor_ids,
        query_features=query_features[query_ids].astype(np.float32),
        donor_features=donor_features,
    )


def spectral_coordinates() -> torch.Tensor:
    h = 2.0 * math.pi * torch.arange(MH, dtype=torch.float32) / float(MH)
    v = 2.0 * math.pi * torch.arange(MV, dtype=torch.float32) / float(MV)
    d = 2.0 * math.pi * torch.arange(S_SC, dtype=torch.float32) / float(S_SC)
    hh, vv, dd = torch.meshgrid(h, v, d, indexing="ij")
    return torch.stack(
        (
            torch.sin(hh),
            torch.cos(hh),
            torch.sin(vv),
            torch.cos(vv),
            torch.sin(dd),
            torch.cos(dd),
            torch.arange(S_SC, dtype=torch.float32)[None, None, :]
            .expand(MH, MV, S_SC)
            / float(S_SC - 1),
        ),
        dim=0,
    )


def pack_donor_field(angle_delay: torch.Tensor) -> torch.Tensor:
    """[B,K,P,H,V,U,D] complex -> [B,K*16,H,V,D] real."""

    value = torch.stack((angle_delay.real, angle_delay.imag), dim=4)
    value = value.permute(0, 1, 2, 6, 4, 3, 5, 7).contiguous()
    return value.reshape(
        len(value), value.shape[1] * OUTPUT_CHANNELS, MH, MV, S_SC
    )


def unpack_prediction(value: torch.Tensor) -> torch.Tensor:
    shaped = value.reshape(len(value), MP, N_UE, 2, MH, MV, S_SC)
    complex_value = torch.complex(shaped[:, :, :, 0], shaped[:, :, :, 1])
    return complex_value.permute(0, 1, 3, 4, 2, 5).contiguous()


def padded_depthwise(block: nn.Conv3d, value: torch.Tensor) -> torch.Tensor:
    value = torch.cat((value[:, :, -1:], value, value[:, :, :1]), dim=2)
    value = torch.cat((value[:, :, :, -1:], value, value[:, :, :, :1]), dim=3)
    value = F.pad(value, (2, 2, 0, 0, 0, 0))
    return block(value)


class WavefieldBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        groups = 8 if hidden % 8 == 0 else 4
        self.norm = nn.GroupNorm(groups, hidden)
        self.depthwise = nn.Conv3d(
            hidden, hidden, kernel_size=(3, 3, 5), groups=hidden, bias=False
        )
        self.pointwise = nn.Conv3d(hidden, hidden, kernel_size=1)
        self.film = nn.Linear(hidden, hidden * 2)
        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, value: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        update = F.gelu(self.norm(value))
        update = padded_depthwise(self.depthwise, update)
        update = self.pointwise(update)
        scale, bias = self.film(context).chunk(2, dim=1)
        update = update * (1.0 + 0.1 * torch.tanh(scale)[:, :, None, None, None])
        update = update + bias[:, :, None, None, None]
        return value + update


class FullBoundaryWavefieldOperator(nn.Module):
    def __init__(
        self,
        query_feature_size: int,
        donor_feature_size: int,
        config: Config,
    ) -> None:
        super().__init__()
        self.config = config
        self.donor_encoder = nn.Sequential(
            nn.Linear(donor_feature_size, config.hidden),
            nn.GELU(),
            nn.LayerNorm(config.hidden),
            nn.Linear(config.hidden, config.hidden),
        )
        self.donor_gate = nn.Linear(config.hidden, 1)
        nn.init.zeros_(self.donor_gate.weight)
        nn.init.zeros_(self.donor_gate.bias)
        self.query_encoder = nn.Sequential(
            nn.Linear(query_feature_size, config.hidden),
            nn.GELU(),
            nn.LayerNorm(config.hidden),
            nn.Linear(config.hidden, config.hidden),
        )
        input_channels = config.neighbors * OUTPUT_CHANNELS + 7
        self.stem = nn.Conv3d(input_channels, config.hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            WavefieldBlock(config.hidden) for _ in range(config.blocks)
        )
        self.output = nn.Conv3d(config.hidden, OUTPUT_CHANNELS, kernel_size=1)
        self.register_buffer("coordinate", spectral_coordinates(), persistent=False)
        self._initialize_average_transport()

    def _initialize_average_transport(self) -> None:
        nn.init.zeros_(self.stem.weight)
        nn.init.zeros_(self.stem.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if self.config.hidden < OUTPUT_CHANNELS:
            raise ValueError("hidden must be at least OUTPUT_CHANNELS")
        with torch.no_grad():
            for channel in range(OUTPUT_CHANNELS):
                for neighbor in range(self.config.neighbors):
                    source = neighbor * OUTPUT_CHANNELS + channel
                    self.stem.weight[channel, source, 0, 0, 0] = (
                        1.0 / self.config.neighbors
                    )
                self.output.weight[channel, channel, 0, 0, 0] = 1.0

    def forward(
        self,
        donor_field: torch.Tensor,
        donor_features: torch.Tensor,
        query_features: torch.Tensor,
    ) -> torch.Tensor:
        donor_code = self.donor_encoder(donor_features)
        gate = torch.softmax(self.donor_gate(donor_code).squeeze(-1), dim=1)
        batch = len(donor_field)
        grouped = donor_field.reshape(
            batch, self.config.neighbors, OUTPUT_CHANNELS, MH, MV, S_SC
        )
        grouped = grouped * (
            gate * float(self.config.neighbors)
        )[:, :, None, None, None, None]
        flattened = grouped.reshape(
            batch, self.config.neighbors * OUTPUT_CHANNELS, MH, MV, S_SC
        )
        coordinate = self.coordinate[None].expand(batch, -1, -1, -1, -1)
        value = self.stem(torch.cat((flattened, coordinate), dim=1))
        context = self.query_encoder(query_features) + torch.sum(
            gate[:, :, None] * donor_code, dim=1
        )
        for block in self.blocks:
            value = block(value, context)
        return unpack_prediction(self.output(value).float())


def official_score_torch(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    physical_scale: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scale = physical_scale[:, None, None, None, None, None]
    predicted = prediction * scale
    target = truth * scale

    predicted_pas = torch.fft.fft(predicted, dim=-1, norm="ortho").abs().square()
    target_pas = torch.fft.fft(target, dim=-1, norm="ortho").abs().square()
    predicted_pas = predicted_pas.permute(0, 4, 5, 1, 2, 3).reshape(
        len(predicted), N_UE, S_SC, MP * MH * MV
    )
    target_pas = target_pas.permute(0, 4, 5, 1, 2, 3).reshape_as(predicted_pas)
    pas_numerator = torch.sum(predicted_pas * target_pas, dim=-1)
    pas_product = torch.sum(predicted_pas.square(), dim=-1) * torch.sum(
        target_pas.square(), dim=-1
    )
    pas_denominator = torch.sqrt(pas_product)
    pas_denominator = torch.where(
        pas_denominator == 0.0,
        torch.full_like(pas_denominator, 1e-12),
        pas_denominator,
    )
    pas = torch.mean(pas_numerator / pas_denominator)

    predicted_pdp = torch.fft.ifft2(
        predicted, dim=(2, 3), norm="ortho"
    ).abs().square()
    target_pdp = torch.fft.ifft2(target, dim=(2, 3), norm="ortho").abs().square()
    pdp_numerator = torch.sum(predicted_pdp * target_pdp, dim=-1)
    pdp_product = torch.sum(predicted_pdp.square(), dim=-1) * torch.sum(
        target_pdp.square(), dim=-1
    )
    pdp_denominator = torch.sqrt(pdp_product)
    pdp_denominator = torch.where(
        pdp_denominator == 0.0,
        torch.full_like(pdp_denominator, 1e-12),
        pdp_denominator,
    )
    pdp = torch.mean(pdp_numerator / pdp_denominator)

    predicted_h = angle_delay_to_channel(predicted)
    target_h = angle_delay_to_channel(target)
    error = torch.sum((predicted_h - target_h).abs().square())
    energy = torch.sum(target_h.abs().square())
    nmse = error / energy.clamp_min(1e-30)
    nmse_score = 1.0 / (1.0 + nmse)
    score = 0.4 * pas + 0.4 * pdp + 0.2 * nmse_score
    return score, {
        "pas": pas,
        "pdp": pdp,
        "nmse": nmse,
        "nmse_score": nmse_score,
    }


class WavefieldDataset:
    def __init__(
        self,
        train_channel: np.ndarray,
        table: NeighborTable,
        site: np.ndarray,
        site_scales: np.ndarray,
        device: torch.device,
    ) -> None:
        self.train_channel = train_channel
        self.table = table
        self.site = site
        self.site_scales = site_scales
        self.device = device

    def batch(
        self, rows: np.ndarray, boundary_enabled: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = np.asarray(rows, dtype=np.int64)
        query_ids = self.table.query_ids[rows]
        target_numpy = np.asarray(self.train_channel[query_ids], dtype=np.complex64)
        target = torch.as_tensor(target_numpy, dtype=torch.complex64, device=self.device)
        scale_numpy = self.site_scales[self.site[query_ids]].astype(np.float32)
        scale = torch.as_tensor(scale_numpy, dtype=torch.float32, device=self.device)
        target_ad = channel_to_angle_delay(
            target / scale[:, None, None, None]
        )

        donor_feature_numpy = self.table.donor_features[rows].copy()
        if boundary_enabled:
            donor_numpy = np.asarray(
                self.train_channel[self.table.donor_ids[rows]], dtype=np.complex64
            )
            donor = torch.as_tensor(
                donor_numpy, dtype=torch.complex64, device=self.device
            )
            donor = donor / scale[:, None, None, None, None]
            donor_field = pack_donor_field(channel_to_angle_delay(donor))
        else:
            donor_feature_numpy[:, :, -1] = 0.0
            donor_field = torch.zeros(
                (
                    len(rows),
                    self.table.donor_ids.shape[1] * OUTPUT_CHANNELS,
                    MH,
                    MV,
                    S_SC,
                ),
                dtype=torch.float32,
                device=self.device,
            )
        donor_features = torch.as_tensor(
            donor_feature_numpy, dtype=torch.float32, device=self.device
        )
        query_features = torch.as_tensor(
            self.table.query_features[rows], dtype=torch.float32, device=self.device
        )
        return donor_field, donor_features, query_features, target_ad


def train_model(
    name: str,
    dataset: WavefieldDataset,
    config: Config,
    steps: int,
    boundary_enabled: bool,
    device: torch.device,
    allowed_rows: np.ndarray | None = None,
) -> tuple[FullBoundaryWavefieldOperator, list[dict[str, float]]]:
    set_seed(config.seed + (0 if boundary_enabled else 17) + steps)
    model = FullBoundaryWavefieldOperator(
        dataset.table.query_features.shape[1],
        dataset.table.donor_features.shape[2],
        config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    rng = np.random.default_rng(config.seed + (0 if boundary_enabled else 17) + steps)
    available = (
        np.arange(len(dataset.table.query_ids), dtype=np.int64)
        if allowed_rows is None
        else np.asarray(allowed_rows, dtype=np.int64)
    )
    history: list[dict[str, float]] = []
    rolling: list[tuple[float, float, float, float, float]] = []
    for step in range(1, steps + 1):
        replace = len(available) < config.batch_size
        rows = rng.choice(available, size=config.batch_size, replace=replace)
        donor_field, donor_features, query_features, target = dataset.batch(
            rows, boundary_enabled
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(donor_field, donor_features, query_features)
        prediction = prediction.to(torch.complex64)
        query_ids = dataset.table.query_ids[rows]
        scale = torch.as_tensor(
            dataset.site_scales[dataset.site[query_ids]],
            dtype=torch.float32,
            device=device,
        )
        score, values = official_score_torch(prediction, target, scale)
        loss = 1.0 - score + 0.02 * torch.log1p(values["nmse"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite {name} loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        rolling.append(
            (
                float(loss.detach().cpu()),
                float(values["pas"].detach().cpu()),
                float(values["pdp"].detach().cpu()),
                float(values["nmse"].detach().cpu()),
                float(score.detach().cpu()),
            )
        )
        if step % config.log_interval == 0 or step == steps:
            record = {
                "step": float(step),
                "loss": float(np.mean([row[0] for row in rolling])),
                "pas": float(np.mean([row[1] for row in rolling])),
                "pdp": float(np.mean([row[2] for row in rolling])),
                "nmse": float(np.mean([row[3] for row in rolling])),
                "score": float(np.mean([row[4] for row in rolling])),
            }
            history.append(record)
            print(
                f"E158 {name} step={step}/{steps} "
                f"PAS={record['pas']:.6f} PDP={record['pdp']:.6f} "
                f"NMSE={record['nmse']:.6f} score={record['score']:.6f}",
                flush=True,
            )
            rolling.clear()
    return model, history


@torch.no_grad()
def evaluate_model(
    model: FullBoundaryWavefieldOperator,
    dataset: WavefieldDataset,
    boundary_enabled: bool,
    batch_size: int = 1,
) -> tuple[np.ndarray, dict[str, float | int]]:
    model.eval()
    prediction = np.empty(
        (len(dataset.table.query_ids), MP * MH * MV, N_UE, S_SC),
        dtype=np.complex64,
    )
    for start in range(0, len(prediction), batch_size):
        end = min(start + batch_size, len(prediction))
        rows = np.arange(start, end, dtype=np.int64)
        donor_field, donor_features, query_features, _ = dataset.batch(
            rows, boundary_enabled
        )
        predicted_ad = model(donor_field, donor_features, query_features)
        predicted_h = angle_delay_to_channel(predicted_ad.to(torch.complex64))
        query_ids = dataset.table.query_ids[rows]
        scale = torch.as_tensor(
            dataset.site_scales[dataset.site[query_ids]],
            dtype=torch.float32,
            device=dataset.device,
        )
        predicted_h = predicted_h * scale[:, None, None, None]
        prediction[start:end] = predicted_h.cpu().numpy().astype(np.complex64)
    truth = np.asarray(
        dataset.train_channel[dataset.table.query_ids], dtype=np.complex64
    )
    metrics = aggregate_statistics(point_metric_statistics(prediction, truth))
    return prediction, metrics


def capacity_rows(point_power: np.ndarray, table: NeighborTable, site: np.ndarray) -> np.ndarray:
    rows: list[int] = []
    for site_value in range(2):
        local = np.flatnonzero(site[table.query_ids] == site_value)
        local = local[point_power[table.query_ids[local]] > 0.0]
        order = np.argsort(point_power[table.query_ids[local]])[::-1]
        rows.extend(local[order[:8]].tolist())
    return np.asarray(rows, dtype=np.int64)


def save_checkpoint(
    path: Path,
    model: FullBoundaryWavefieldOperator,
    config: Config,
    boundary_enabled: bool,
) -> None:
    torch.save(
        {
            "state_dict": copy.deepcopy(model.state_dict()),
            "config": asdict(config),
            "boundary_enabled": boundary_enabled,
            "outer": OUTER,
        },
        path,
    )


def v7_outer_metrics(
    train_channel: np.ndarray,
    held_ids: np.ndarray,
) -> dict[str, float | int]:
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    row_map = np.full(len(train_channel), -1, dtype=np.int64)
    row_map[query_ids] = np.arange(len(query_ids), dtype=np.int64)
    rows = row_map[held_ids]
    if np.any(rows < 0):
        raise RuntimeError("outer query missing from V7 OOF channel")
    v7 = np.load(e153.V7_CHANNEL, mmap_mode="r")
    truth = np.asarray(train_channel[held_ids], dtype=np.complex64)
    return aggregate_statistics(
        point_metric_statistics(np.asarray(v7[rows], dtype=np.complex64), truth)
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    config = Config()
    set_seed(config.seed)
    device = torch.device(args.device)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    required = (TRAIN_CHANNEL, MAP_FEATURES, FOLD_CACHE, OOF_META, POWER_CACHE, e153.V7_CHANNEL)
    missing = [PATHS.relative(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing E158 inputs: {missing}")

    setup, positions, _, site, _ = load_positions()
    positions = positions.astype(np.float64)
    site = site.astype(np.int8)
    train_channel = np.load(TRAIN_CHANNEL, mmap_mode="r")
    point_power = np.load(POWER_CACHE).astype(np.float64)
    rms = point_rms(point_power)
    folds = np.load(FOLD_CACHE, allow_pickle=False)
    fit_ids = folds[f"fold{OUTER}_fit"].astype(np.int64)
    held_ids = folds[f"fold{OUTER}_query"].astype(np.int64)

    map_raw = np.load(MAP_FEATURES, allow_pickle=False)["train_features"].astype(
        np.float32
    )
    map_mean = np.mean(map_raw[fit_ids], axis=0, dtype=np.float64).astype(np.float32)
    map_std = np.maximum(
        np.std(map_raw[fit_ids], axis=0, dtype=np.float64).astype(np.float32),
        1e-4,
    )
    map_features = np.clip((map_raw - map_mean) / map_std, -6.0, 6.0).astype(
        np.float32
    )
    query_features = build_query_features(
        positions, site, setup.bs_positions, map_features
    )
    site_scales = build_site_scales(rms, site, fit_ids)

    identity = identity_check(train_channel, fit_ids, device)
    payload: dict[str, object] = {
        "experiment": VERSION,
        "status": "identity_complete",
        "official_inputs_only": True,
        "sionna_used": False,
        "v12_used": False,
        "v7_role": "official-score comparator only",
        "outer": OUTER,
        "config": asdict(config),
        "identity": identity,
        "site_scales": site_scales.tolist(),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"E158 identity={identity}", flush=True)
    if args.stage == "identity":
        return

    train_table = build_neighbor_table(
        fit_ids,
        fit_ids,
        positions,
        site,
        setup.bs_positions,
        map_features,
        query_features,
        rms,
        site_scales,
        config.neighbors,
        config.train_guard_m,
    )
    train_dataset = WavefieldDataset(
        train_channel, train_table, site, site_scales, device
    )
    selected_capacity_rows = capacity_rows(point_power, train_table, site)
    capacity_model, capacity_history = train_model(
        "capacity",
        train_dataset,
        config,
        config.capacity_steps,
        True,
        device,
        selected_capacity_rows,
    )
    capacity_table = NeighborTable(
        query_ids=train_table.query_ids[selected_capacity_rows],
        donor_ids=train_table.donor_ids[selected_capacity_rows],
        query_features=train_table.query_features[selected_capacity_rows],
        donor_features=train_table.donor_features[selected_capacity_rows],
    )
    capacity_dataset = WavefieldDataset(
        train_channel, capacity_table, site, site_scales, device
    )
    _, capacity_metrics = evaluate_model(
        capacity_model, capacity_dataset, True, batch_size=1
    )
    capacity_passed = float(capacity_metrics["score"]) >= config.capacity_threshold
    payload.update(
        {
            "status": "capacity_complete" if capacity_passed else "capacity_failed",
            "capacity": {
                "query_ids": capacity_table.query_ids.tolist(),
                "history": capacity_history,
                "metrics": capacity_metrics,
                "threshold": config.capacity_threshold,
                "passed": capacity_passed,
            },
        }
    )
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_checkpoint(
        ARTIFACT_DIR / "capacity_operator.pt", capacity_model, config, True
    )
    print(f"E158 capacity={capacity_metrics} passed={capacity_passed}", flush=True)
    if not capacity_passed or args.stage == "capacity":
        return

    boundary_model, boundary_history = train_model(
        "boundary",
        train_dataset,
        config,
        config.screen_steps,
        True,
        device,
    )
    control_model, control_history = train_model(
        "no_boundary_control",
        train_dataset,
        config,
        config.screen_steps,
        False,
        device,
    )
    held_table = build_neighbor_table(
        held_ids,
        fit_ids,
        positions,
        site,
        setup.bs_positions,
        map_features,
        query_features,
        rms,
        site_scales,
        config.neighbors,
        0.0,
    )
    held_dataset = WavefieldDataset(
        train_channel, held_table, site, site_scales, device
    )
    boundary_prediction, boundary_metrics = evaluate_model(
        boundary_model, held_dataset, True
    )
    control_prediction, control_metrics = evaluate_model(
        control_model, held_dataset, False
    )
    v7_metrics = v7_outer_metrics(train_channel, held_ids)

    boundary_path = ARTIFACT_DIR / "outer3_boundary_prediction.npy"
    control_path = ARTIFACT_DIR / "outer3_no_boundary_prediction.npy"
    np.save(boundary_path, boundary_prediction)
    np.save(control_path, control_prediction)
    save_checkpoint(
        ARTIFACT_DIR / "outer3_boundary_operator.pt", boundary_model, config, True
    )
    save_checkpoint(
        ARTIFACT_DIR / "outer3_no_boundary_operator.pt", control_model, config, False
    )

    delta_v7 = float(boundary_metrics["score"] - v7_metrics["score"])
    delta_control = float(boundary_metrics["score"] - control_metrics["score"])
    all_components_non_worse = bool(
        boundary_metrics["pas"] >= v7_metrics["pas"]
        and boundary_metrics["pdp"] >= v7_metrics["pdp"]
        and boundary_metrics["nmse"] <= v7_metrics["nmse"]
    )
    gate_passed = bool(
        delta_v7 >= 0.005
        and delta_control >= 0.003
        and all_components_non_worse
    )
    payload.update(
        {
            "status": "screen_gate_passed" if gate_passed else "screen_gate_failed",
            "capacity": payload["capacity"],
            "screen": {
                "held_count": int(len(held_ids)),
                "held_query_ids": held_ids.tolist(),
                "v7": v7_metrics,
                "boundary": boundary_metrics,
                "no_boundary_control": control_metrics,
                "delta_vs_v7": delta_v7,
                "delta_vs_no_boundary": delta_control,
                "all_components_non_worse": all_components_non_worse,
                "required_delta_vs_v7": 0.005,
                "required_delta_vs_no_boundary": 0.003,
                "gate_passed": gate_passed,
                "boundary_history": boundary_history,
                "control_history": control_history,
            },
            "outputs": {
                "boundary_prediction": PATHS.relative(boundary_path),
                "control_prediction": PATHS.relative(control_path),
                "boundary_checkpoint": PATHS.relative(
                    ARTIFACT_DIR / "outer3_boundary_operator.pt"
                ),
                "control_checkpoint": PATHS.relative(
                    ARTIFACT_DIR / "outer3_no_boundary_operator.pt"
                ),
            },
            "seconds": float(time.time() - started),
        }
    )
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
