#!/usr/bin/env python3
"""Fit a complete complex channel function from distance and official geometry."""

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

from common.official_metrics import OfficialMetricAccumulator
from project_paths import PATHS
from round2.data import load_positions
from round2.spatial_hole_cv import build_spatial_hole_folds
import round2.train_e109_geometry_conditioned_transfer_operator as e109
import round2.train_offline_sionna_geometry_student as e87


VERSION = "Round2-E110-geometry-full-channel-function"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "geometry_full_channel_function"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_geometry_full_channel_function"
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"

MP = 2
MH = 16
MV = 8
N_UE = 4
S_SC = 192
REAL_CHANNELS = 2 * MP * N_UE
COMPLEX_SIZE = MP * MH * MV * N_UE * S_SC
REAL_SCALE = math.sqrt(2.0 * COMPLEX_SIZE)
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    latent_dim: int = 512
    ae_width: int = 32
    geometry_hidden: int = 512
    geometry_branch: int = 96
    geometry_blocks: int = 5
    dropout: float = 0.06
    autoencoder_epochs: int = 24
    operator_epochs: int = 64
    finetune_epochs: int = 10
    validation_interval: int = 2
    batch_size: int = 8
    evaluation_batch_size: int = 8
    autoencoder_lr: float = 3e-4
    operator_lr: float = 3e-4
    finetune_lr: float = 3e-5
    weight_decay: float = 2e-4
    beta: float = 140.30
    gamma: float = 0.10
    latent_weight: float = 0.35
    metric_weight: float = 0.65
    power_weight: float = 0.08
    zero_weight: float = 0.08
    random_state: int = 20260808110


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "screen"), default="screen")
    parser.add_argument("--outer", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--autoencoder-epochs", type=int, default=None)
    parser.add_argument("--operator-epochs", type=int, default=None)
    parser.add_argument("--finetune-epochs", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    compatible_seed = int(seed % (2**32 - 1))
    np.random.seed(compatible_seed)
    torch.manual_seed(compatible_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(compatible_seed)


def required_inputs() -> tuple:
    return (
        TRAIN_CHANNEL,
        PATHS.round2 / "Round2_Train_Pos.npy",
        PATHS.round2 / "Round2_Test_Pos.npy",
        PATHS.round2 / "Round2_Setup.json",
        PATHS.round2 / "Round2_Map.ply",
        e87.GEOMETRY_CACHE,
        e87.ROUTE_CACHE,
        e87.GEODESIC_CACHE,
        e87.VISIBILITY_CACHE,
        e109.DELAY_ANGLE_CACHE,
    )


def ensure_inputs() -> None:
    missing = [PATHS.relative(path) for path in required_inputs() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing E110 inputs: {missing}")


def split_rows(outer: int) -> dict[str, np.ndarray]:
    _, train_positions, test_positions, train_site, test_site = load_positions()
    folds = build_spatial_hole_folds(
        train_positions, test_positions, train_site, test_site
    )
    if not 0 <= outer < len(folds):
        raise ValueError(f"outer must be in [0,{len(folds) - 1}], got {outer}")
    held = folds[outer]
    validation_fold = folds[(outer + 1) % len(folds)]
    validation = np.intersect1d(
        held.fit_indices, validation_fold.query_indices, assume_unique=False
    ).astype(np.int64)
    train = held.fit_indices[~np.isin(held.fit_indices, validation)].astype(np.int64)
    return {
        "train": train,
        "validation": validation,
        "held": held.query_indices.astype(np.int64),
        "ignored": held.ignored_indices.astype(np.int64),
    }


class ChannelTransform:
    """Remove the distance carrier baseline and expose a beam-delay tensor."""

    def __init__(
        self,
        distance: np.ndarray,
        beta: float,
        gamma: float,
        device: torch.device,
    ) -> None:
        self.distance = torch.as_tensor(distance.astype(np.float32), device=device)
        self.frequency = torch.linspace(-0.5, 0.5, S_SC, device=device)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.device = device

    def carrier(self, rows: torch.Tensor, sign: float) -> torch.Tensor:
        phase = self.distance[rows, None] * (
            self.beta + self.gamma * self.frequency[None]
        )
        return torch.exp(sign * 1j * phase).to(torch.complex64)

    def to_beam_delay(
        self, channel: torch.Tensor, rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        power = channel.abs().square().sum(dim=(1, 2, 3))
        nonzero = power > 0.0
        normalized = channel / torch.sqrt(power.clamp_min(TINY))[:, None, None, None]
        aligned = normalized * self.carrier(rows, +1.0)[:, None, None, :]
        shaped = aligned.reshape(-1, MP, MH, MV, N_UE, S_SC)
        beam_delay = torch.fft.ifft(
            torch.fft.fft2(shaped, dim=(2, 3), norm="ortho"),
            dim=-1,
            norm="ortho",
        )
        beam_delay = beam_delay.permute(0, 1, 4, 5, 2, 3)
        real = torch.view_as_real(beam_delay.contiguous())
        real = (
            real.permute(0, 1, 2, 6, 3, 4, 5)
            .reshape(-1, REAL_CHANNELS, S_SC, MH, MV)
            .float()
        )
        return real * REAL_SCALE, power, nonzero

    def from_beam_delay(
        self,
        beam_delay: torch.Tensor,
        rows: torch.Tensor,
        power: torch.Tensor | None = None,
    ) -> torch.Tensor:
        value = (beam_delay.float() / REAL_SCALE).reshape(
            -1, MP, N_UE, 2, S_SC, MH, MV
        )
        value = value.permute(0, 1, 2, 4, 5, 6, 3).contiguous()
        complex_value = torch.view_as_complex(value)
        complex_value = complex_value.permute(0, 1, 4, 5, 2, 3)
        channel = torch.fft.ifft2(
            torch.fft.fft(complex_value, dim=-1, norm="ortho"),
            dim=(2, 3),
            norm="ortho",
        ).reshape(-1, MP * MH * MV, N_UE, S_SC)
        channel = channel * self.carrier(rows, -1.0)[:, None, None, :]
        if power is not None:
            channel = channel * torch.sqrt(power.clamp_min(0.0))[:, None, None, None]
        return channel.to(torch.complex64)


class ConvBlock3d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = max(1, min(8, channels // 4))
        while channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 0.5 * self.block(value)


class ChannelAutoencoder(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        width = config.ae_width
        self.encoder_grid = nn.Sequential(
            nn.Conv3d(REAL_CHANNELS, width, 4, stride=2, padding=1),
            ConvBlock3d(width),
            nn.Conv3d(width, 2 * width, 4, stride=2, padding=1),
            ConvBlock3d(2 * width),
            nn.Conv3d(
                2 * width,
                4 * width,
                kernel_size=(4, 4, 2),
                stride=(2, 2, 2),
                padding=(1, 1, 0),
            ),
            ConvBlock3d(4 * width),
        )
        grid_size = 4 * width * 24 * 2
        self.encoder_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(grid_size, 2 * config.latent_dim),
            nn.SiLU(),
            nn.LayerNorm(2 * config.latent_dim),
            nn.Linear(2 * config.latent_dim, config.latent_dim),
        )
        self.decoder_head = nn.Sequential(
            nn.Linear(config.latent_dim, 2 * config.latent_dim),
            nn.SiLU(),
            nn.LayerNorm(2 * config.latent_dim),
            nn.Linear(2 * config.latent_dim, grid_size),
            nn.SiLU(),
        )
        self.decoder_grid = nn.Sequential(
            ConvBlock3d(4 * width),
            nn.ConvTranspose3d(
                4 * width,
                2 * width,
                kernel_size=(4, 4, 2),
                stride=(2, 2, 2),
                padding=(1, 1, 0),
            ),
            ConvBlock3d(2 * width),
            nn.ConvTranspose3d(2 * width, width, 4, stride=2, padding=1),
            ConvBlock3d(width),
            nn.ConvTranspose3d(width, REAL_CHANNELS, 4, stride=2, padding=1),
        )
        self.width = width
        self.latent_dim = config.latent_dim

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        return self.encoder_head(self.encoder_grid(value))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        grid = self.decoder_head(latent).reshape(-1, 4 * self.width, 24, 2, 1)
        output = self.decoder_grid(grid).float()
        rms = torch.sqrt(output.square().mean(dim=(1, 2, 3, 4), keepdim=True))
        return output / rms.clamp_min(1e-6)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(value)
        return self.decode(latent), latent


class GeometryFullChannelFunction(nn.Module):
    def __init__(
        self,
        scalar_dim: int,
        map_dim: int,
        config: Config,
    ) -> None:
        super().__init__()
        branch = config.geometry_branch
        hidden = config.geometry_hidden
        self.scalar = nn.Sequential(
            nn.Linear(scalar_dim + map_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, branch),
            nn.SiLU(),
        )
        self.route = e109.conv2d_encoder(5, branch)
        self.polar = e109.conv2d_encoder(4, branch)
        self.corridor = e109.conv2d_encoder(4, branch)
        self.delay_angle = e109.conv2d_encoder(e109.MAP_CHANNELS, branch)
        self.panorama = nn.Sequential(
            nn.Conv1d(4, 64, 5, padding=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(128 * 4, branch),
            nn.SiLU(),
        )
        self.distance_base = nn.Sequential(
            nn.Linear(scalar_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        self.geometry_fusion = nn.Sequential(
            nn.Linear(6 * branch, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        self.scale = nn.Linear(hidden, hidden)
        self.shift = nn.Linear(hidden, hidden)
        self.blocks = nn.ModuleList(
            e109.ResidualBlock(hidden, config.dropout)
            for _ in range(config.geometry_blocks)
        )
        self.latent_head = nn.Linear(hidden, config.latent_dim)
        self.power_head = nn.Linear(hidden, 1)
        self.nonzero_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.latent_head.weight)
        nn.init.zeros_(self.latent_head.bias)
        nn.init.zeros_(self.power_head.weight)
        nn.init.zeros_(self.power_head.bias)
        nn.init.zeros_(self.nonzero_head.weight)
        nn.init.constant_(self.nonzero_head.bias, 2.5)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        scalar = batch["scalar"]
        geometry = torch.cat(
            (
                self.scalar(torch.cat((scalar, batch["map_scalar"]), dim=1)),
                self.route(batch["route"]),
                self.panorama(batch["panorama"]),
                self.polar(batch["polar"]),
                self.corridor(batch["corridor"]),
                self.delay_angle(batch["delay_angle"]),
            ),
            dim=1,
        )
        geometry = self.geometry_fusion(geometry)
        distance = self.distance_base(scalar)
        hidden = distance * (1.0 + 0.5 * torch.tanh(self.scale(geometry)))
        hidden = hidden + self.shift(geometry)
        for block in self.blocks:
            hidden = block(hidden)
        return {
            "latent": self.latent_head(hidden),
            "log_power": 6.0 * torch.tanh(
                self.power_head(hidden).squeeze(1) / 6.0
            ),
            "nonzero_logit": self.nonzero_head(hidden).squeeze(1),
        }


class StableSqrt(torch.autograd.Function):
    """Use the exact square root forward value and a bounded zero derivative."""

    @staticmethod
    def forward(ctx: object, value: torch.Tensor, epsilon: float) -> torch.Tensor:
        ctx.save_for_backward(value)
        ctx.epsilon = epsilon
        return torch.sqrt(value.clamp_min(0.0))

    @staticmethod
    def backward(
        ctx: object, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        (value,) = ctx.saved_tensors
        denominator = 2.0 * torch.sqrt(value.clamp_min(0.0) + ctx.epsilon)
        return gradient / denominator, None


def stable_sqrt(value: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    return StableSqrt.apply(value, epsilon)


def cosine(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    dims: tuple[int, ...],
) -> torch.Tensor:
    numerator = (prediction * truth).sum(dim=dims)
    denominator_squared = (
        prediction.square().sum(dim=dims) * truth.square().sum(dim=dims)
    )
    denominator = stable_sqrt(denominator_squared, 1e-24)
    denominator = torch.where(
        denominator_squared == 0.0,
        torch.full_like(denominator, 1e-12),
        denominator,
    )
    return numerator / denominator


def differentiable_score(
    prediction: torch.Tensor, truth: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_shaped = prediction.reshape(-1, MP, MH, MV, N_UE, S_SC)
    truth_shaped = truth.reshape_as(pred_shaped)
    pred_pas = torch.fft.fft2(pred_shaped, dim=(2, 3)).abs().square().float()
    truth_pas = torch.fft.fft2(truth_shaped, dim=(2, 3)).abs().square().float()
    pas = cosine(pred_pas, truth_pas, (1, 2, 3)).mean(dim=(1, 2))
    pred_pdp = torch.fft.ifft(prediction, dim=-1).abs().square().float()
    truth_pdp = torch.fft.ifft(truth, dim=-1).abs().square().float()
    pdp = cosine(pred_pdp, truth_pdp, (3,)).mean(dim=(1, 2))
    error = (prediction - truth).abs().square().sum(dim=(1, 2, 3))
    energy = truth.abs().square().sum(dim=(1, 2, 3))
    nmse = error.sum() / energy.sum().clamp_min(TINY)
    nmse_score = 1.0 / (1.0 + nmse)
    score = 0.4 * pas.mean() + 0.4 * pdp.mean() + 0.2 * nmse_score
    return score, {
        "pas": pas.mean(),
        "pdp": pdp.mean(),
        "nmse": nmse,
        "nmse_score": nmse_score,
    }


def channel_batch(
    channel: np.ndarray,
    rows: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    return torch.as_tensor(
        np.array(channel[rows], dtype=np.complex64, copy=True), device=device
    )


def metric_payload(values: dict[str, float | int]) -> dict[str, float | int]:
    return {
        key: int(value) if key == "count" else float(value)
        for key, value in values.items()
    }


@torch.inference_mode()
def evaluate_autoencoder(
    model: ChannelAutoencoder,
    transform: ChannelTransform,
    channel: np.ndarray,
    rows: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    accumulator = OfficialMetricAccumulator()
    for start in range(0, len(rows), batch_size):
        local = rows[start : start + batch_size]
        row_tensor = torch.as_tensor(local, device=device)
        truth = channel_batch(channel, local, device)
        target, power, nonzero = transform.to_beam_delay(truth, row_tensor)
        output = torch.zeros_like(truth)
        if torch.any(nonzero):
            reconstructed, _ = model(target[nonzero])
            output[nonzero] = transform.from_beam_delay(
                reconstructed, row_tensor[nonzero], power[nonzero]
            )
        accumulator.add(output.cpu().numpy(), truth.cpu().numpy())
    return metric_payload(accumulator.result())


def autoencoder_loss(
    model: ChannelAutoencoder,
    transform: ChannelTransform,
    truth: torch.Tensor,
    rows: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    target, _, nonzero = transform.to_beam_delay(truth, rows)
    if not torch.all(nonzero):
        target = target[nonzero]
        truth = truth[nonzero]
        rows = rows[nonzero]
    reconstructed, latent = model(target)
    predicted = transform.from_beam_delay(reconstructed, rows)
    normalized_truth = truth / torch.sqrt(
        truth.abs().square().sum(dim=(1, 2, 3)).clamp_min(TINY)
    )[:, None, None, None]
    score, parts = differentiable_score(predicted, normalized_truth)
    reconstruction = (reconstructed - target).square().mean()
    loss = 0.45 * reconstruction + 0.55 * (1.0 - score)
    loss = loss + 1e-5 * latent.square().mean()
    return loss, {
        "loss": float(loss.detach().item()),
        "reconstruction": float(reconstruction.detach().item()),
        "score": float(score.detach().item()),
        **{key: float(value.detach().item()) for key, value in parts.items()},
    }


def train_autoencoder(
    model: ChannelAutoencoder,
    transform: ChannelTransform,
    channel: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    nonzero_mask: np.ndarray,
    config: Config,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    nonzero_train = train_rows[np.asarray(nonzero_mask[train_rows], dtype=bool)]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.autoencoder_lr,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.random_state + 1)
    history: list[dict[str, object]] = []
    initial = evaluate_autoencoder(
        model,
        transform,
        channel,
        validation_rows,
        config.evaluation_batch_size,
        device,
    )
    best_score = float(initial["score"])
    best_state = copy.deepcopy(model.state_dict())
    history.append({"epoch": 0, "validation": initial})
    for epoch in range(1, config.autoencoder_epochs + 1):
        model.train()
        order = rng.permutation(nonzero_train)
        sums: dict[str, float] = {}
        batches = 0
        for start in range(0, len(order), config.batch_size):
            local = order[start : start + config.batch_size]
            rows = torch.as_tensor(local, device=device)
            truth = channel_batch(channel, local, device)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = autoencoder_loss(model, transform, truth, rows)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite E110 autoencoder loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + value
            batches += 1
        record: dict[str, object] = {
            "epoch": epoch,
            "train": {key: value / max(batches, 1) for key, value in sums.items()},
        }
        if epoch % config.validation_interval == 0 or epoch == config.autoencoder_epochs:
            validation = evaluate_autoencoder(
                model,
                transform,
                channel,
                validation_rows,
                config.evaluation_batch_size,
                device,
            )
            record["validation"] = validation
            if float(validation["score"]) > best_score:
                best_score = float(validation["score"])
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"E110 autoencoder epoch={epoch}/{config.autoencoder_epochs} "
                f"train={record['train']['score']:.6f} validation={validation['score']:.6f}",
                flush=True,
            )
        history.append(record)
    model.load_state_dict(best_state)
    return best_state, history


@torch.inference_mode()
def encode_latents(
    model: ChannelAutoencoder,
    transform: ChannelTransform,
    channel: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    count = len(channel)
    latent = np.zeros((count, model.latent_dim), dtype=np.float32)
    log_power = np.full(count, math.log(TINY), dtype=np.float32)
    nonzero = np.zeros(count, dtype=bool)
    all_rows = np.arange(count, dtype=np.int64)
    for start in range(0, count, batch_size):
        local = all_rows[start : start + batch_size]
        rows = torch.as_tensor(local, device=device)
        truth = channel_batch(channel, local, device)
        target, power, valid = transform.to_beam_delay(truth, rows)
        if torch.any(valid):
            latent[local[valid.cpu().numpy()]] = (
                model.encode(target[valid]).cpu().numpy().astype(np.float32)
            )
        local_power = power.cpu().numpy().astype(np.float64)
        local_valid = local_power > 0.0
        log_power[local[local_valid]] = np.log(local_power[local_valid]).astype(np.float32)
        nonzero[local] = local_valid
    return {"latent": latent, "log_power": log_power, "nonzero": nonzero}


def normalization(
    value: np.ndarray,
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(value[rows], dtype=np.float64)
    mean = selected.mean(axis=0)
    std = selected.std(axis=0)
    std = np.where(std < 1e-5, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def geometry_batch(
    store: e109.FeatureStore,
    rows: torch.Tensor,
    scalar: torch.Tensor,
    map_scalar: torch.Tensor,
) -> dict[str, torch.Tensor]:
    identity = torch.arange(len(store.positions), device=store.device)
    return store.batch(rows, scalar, map_scalar, "full_geometry", identity)


def decode_function_output(
    prediction: dict[str, torch.Tensor],
    autoencoder: ChannelAutoencoder,
    transform: ChannelTransform,
    rows: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    power_mean: torch.Tensor,
    power_std: torch.Tensor,
    hard_zero: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    latent = prediction["latent"] * latent_std + latent_mean
    log_power = prediction["log_power"] * power_std + power_mean
    power = torch.exp(log_power.clamp(-40.0, 10.0))
    beam_delay = autoencoder.decode(latent)
    channel = transform.from_beam_delay(beam_delay, rows, power)
    nonzero_probability = torch.sigmoid(prediction["nonzero_logit"])
    if hard_zero:
        channel = torch.where(
            (nonzero_probability >= 0.5)[:, None, None, None],
            channel,
            torch.zeros_like(channel),
        )
    return channel, latent, nonzero_probability


@torch.inference_mode()
def evaluate_function(
    model: GeometryFullChannelFunction,
    autoencoder: ChannelAutoencoder,
    transform: ChannelTransform,
    store: e109.FeatureStore,
    channel: np.ndarray,
    rows: np.ndarray,
    scalar: torch.Tensor,
    map_scalar: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    power_mean: torch.Tensor,
    power_std: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    autoencoder.eval()
    accumulator = OfficialMetricAccumulator()
    for start in range(0, len(rows), batch_size):
        local = rows[start : start + batch_size]
        row_tensor = torch.as_tensor(local, device=device)
        prediction = model(geometry_batch(store, row_tensor, scalar, map_scalar))
        output, _, _ = decode_function_output(
            prediction,
            autoencoder,
            transform,
            row_tensor,
            latent_mean,
            latent_std,
            power_mean,
            power_std,
            True,
        )
        truth = channel_batch(channel, local, device)
        accumulator.add(output.cpu().numpy(), truth.cpu().numpy())
    return metric_payload(accumulator.result())


def function_loss(
    model: GeometryFullChannelFunction,
    autoencoder: ChannelAutoencoder,
    transform: ChannelTransform,
    store: e109.FeatureStore,
    channel: np.ndarray,
    local: np.ndarray,
    scalar: torch.Tensor,
    map_scalar: torch.Tensor,
    targets: dict[str, torch.Tensor],
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    power_mean: torch.Tensor,
    power_std: torch.Tensor,
    config: Config,
    device: torch.device,
    finetune: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    rows = torch.as_tensor(local, device=device)
    prediction = model(geometry_batch(store, rows, scalar, map_scalar))
    output, latent, _ = decode_function_output(
        prediction,
        autoencoder,
        transform,
        rows,
        latent_mean,
        latent_std,
        power_mean,
        power_std,
        False,
    )
    truth = channel_batch(channel, local, device)
    valid = targets["nonzero"][rows]
    zero_target = valid.float()
    zero_raw = torch.nn.functional.binary_cross_entropy_with_logits(
        prediction["nonzero_logit"], zero_target, reduction="none"
    )
    zero_weights = torch.where(valid, torch.ones_like(zero_raw), 6.0 * torch.ones_like(zero_raw))
    zero_loss = (zero_raw * zero_weights).mean()
    if torch.any(valid):
        if finetune:
            score, parts = differentiable_score(output[valid], truth[valid])
        else:
            predicted_energy = output[valid].abs().square().sum(dim=(1, 2, 3))
            truth_energy = truth[valid].abs().square().sum(dim=(1, 2, 3))
            normalized_output = output[valid] / torch.sqrt(
                predicted_energy.clamp_min(TINY)
            )[:, None, None, None]
            normalized_truth = truth[valid] / torch.sqrt(
                truth_energy.clamp_min(TINY)
            )[:, None, None, None]
            score, parts = differentiable_score(normalized_output, normalized_truth)
        latent_target = targets["latent"][rows][valid]
        latent_loss = torch.nn.functional.smooth_l1_loss(
            prediction["latent"][valid], latent_target
        )
        power_loss = torch.nn.functional.smooth_l1_loss(
            prediction["log_power"][valid], targets["log_power"][rows][valid]
        )
        metric_loss = 1.0 - score
    else:
        zero = prediction["latent"].sum() * 0.0
        score = zero
        latent_loss = zero
        power_loss = zero
        metric_loss = zero
        parts = {key: zero for key in ("pas", "pdp", "nmse", "nmse_score")}
    metric_weight = 0.80 if finetune else config.metric_weight
    latent_weight = 0.15 if finetune else config.latent_weight
    loss = (
        metric_weight * metric_loss
        + latent_weight * latent_loss
        + config.power_weight * power_loss
        + config.zero_weight * zero_loss
    )
    return loss, {
        "loss": float(loss.detach().item()),
        "score": float(score.detach().item()),
        "latent": float(latent_loss.detach().item()),
        "power": float(power_loss.detach().item()),
        "zero": float(zero_loss.detach().item()),
        **{key: float(value.detach().item()) for key, value in parts.items()},
    }


def train_function_stage(
    model: GeometryFullChannelFunction,
    autoencoder: ChannelAutoencoder,
    transform: ChannelTransform,
    store: e109.FeatureStore,
    channel: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    scalar: torch.Tensor,
    map_scalar: torch.Tensor,
    targets: dict[str, torch.Tensor],
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    power_mean: torch.Tensor,
    power_std: torch.Tensor,
    config: Config,
    device: torch.device,
    finetune: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[dict[str, object]]]:
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)
    if finetune:
        for parameter in autoencoder.decoder_head.parameters():
            parameter.requires_grad_(True)
        for parameter in autoencoder.decoder_grid.parameters():
            parameter.requires_grad_(True)
    trainable = list(model.parameters()) + [
        parameter for parameter in autoencoder.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.finetune_lr if finetune else config.operator_lr,
        weight_decay=config.weight_decay,
    )
    epochs = config.finetune_epochs if finetune else config.operator_epochs
    seed_offset = 200 if finetune else 100
    rng = np.random.default_rng(config.random_state + seed_offset)
    initial = evaluate_function(
        model,
        autoencoder,
        transform,
        store,
        channel,
        validation_rows,
        scalar,
        map_scalar,
        latent_mean,
        latent_std,
        power_mean,
        power_std,
        config.evaluation_batch_size,
        device,
    )
    best_score = float(initial["score"])
    best_model = copy.deepcopy(model.state_dict())
    best_autoencoder = copy.deepcopy(autoencoder.state_dict())
    history: list[dict[str, object]] = [{"epoch": 0, "validation": initial}]
    label = "finetune" if finetune else "operator"
    for epoch in range(1, epochs + 1):
        model.train()
        autoencoder.decoder_head.train(finetune)
        autoencoder.decoder_grid.train(finetune)
        order = rng.permutation(train_rows)
        sums: dict[str, float] = {}
        batches = 0
        for start in range(0, len(order), config.batch_size):
            local = order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss, parts = function_loss(
                model,
                autoencoder,
                transform,
                store,
                channel,
                local,
                scalar,
                map_scalar,
                targets,
                latent_mean,
                latent_std,
                power_mean,
                power_std,
                config,
                device,
                finetune,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite E110 {label} loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + value
            batches += 1
        record: dict[str, object] = {
            "epoch": epoch,
            "train": {key: value / max(batches, 1) for key, value in sums.items()},
        }
        if epoch % config.validation_interval == 0 or epoch == epochs:
            validation = evaluate_function(
                model,
                autoencoder,
                transform,
                store,
                channel,
                validation_rows,
                scalar,
                map_scalar,
                latent_mean,
                latent_std,
                power_mean,
                power_std,
                config.evaluation_batch_size,
                device,
            )
            record["validation"] = validation
            if float(validation["score"]) > best_score:
                best_score = float(validation["score"])
                best_model = copy.deepcopy(model.state_dict())
                best_autoencoder = copy.deepcopy(autoencoder.state_dict())
            print(
                f"E110 {label} epoch={epoch}/{epochs} "
                f"train={record['train']['score']:.6f} validation={validation['score']:.6f}",
                flush=True,
            )
        history.append(record)
    model.load_state_dict(best_model)
    autoencoder.load_state_dict(best_autoencoder)
    return best_model, best_autoencoder, history


def tensor_targets(
    encoded: dict[str, np.ndarray],
    train_rows: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    valid_train = train_rows[encoded["nonzero"][train_rows]]
    latent_mean, latent_std = normalization(encoded["latent"], valid_train)
    power_mean, power_std = normalization(encoded["log_power"][:, None], valid_train)
    targets = {
        "latent": torch.as_tensor(
            (encoded["latent"] - latent_mean) / latent_std, device=device
        ),
        "log_power": torch.as_tensor(
            ((encoded["log_power"] - power_mean[0]) / power_std[0]).astype(np.float32),
            device=device,
        ),
        "nonzero": torch.as_tensor(encoded["nonzero"], device=device),
    }
    constants = {
        "latent_mean": latent_mean,
        "latent_std": latent_std,
        "power_mean": power_mean,
        "power_std": power_std,
    }
    return targets, constants


def best_epoch(history: list[dict[str, object]]) -> int:
    rows = [row for row in history if "validation" in row]
    return int(max(rows, key=lambda row: float(row["validation"]["score"]))["epoch"])


def main() -> None:
    args = parse_args()
    ensure_inputs()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    config = Config(
        autoencoder_epochs=(
            args.autoencoder_epochs
            if args.autoencoder_epochs is not None
            else (2 if args.stage == "smoke" else Config.autoencoder_epochs)
        ),
        operator_epochs=(
            args.operator_epochs
            if args.operator_epochs is not None
            else (3 if args.stage == "smoke" else Config.operator_epochs)
        ),
        finetune_epochs=(
            args.finetune_epochs
            if args.finetune_epochs is not None
            else (1 if args.stage == "smoke" else Config.finetune_epochs)
        ),
        validation_interval=1 if args.stage == "smoke" else Config.validation_interval,
        batch_size=args.batch_size,
    )
    set_seed(config.random_state + args.outer)
    rows = split_rows(args.outer)
    if args.stage == "smoke":
        rows = {
            "train": rows["train"][:192],
            "validation": rows["validation"][:32],
            "held": rows["held"][:24],
            "ignored": rows["ignored"],
        }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"outer{args.outer}_{args.stage}"
    report_path = ANALYSIS_DIR / f"{suffix}.json"
    checkpoint_path = ARTIFACT_DIR / f"{suffix}.pt"
    latent_path = ARTIFACT_DIR / f"{suffix}_latents.npz"
    started = time.time()

    setup, positions, _, site, _ = load_positions()
    serving = setup.bs_positions[site]
    distance = np.linalg.norm(positions - serving, axis=1)
    transform = ChannelTransform(distance, config.beta, config.gamma, device)
    channel = np.load(TRAIN_CHANNEL, mmap_mode="r")

    store = e109.FeatureStore(device)
    extra_position = e87.position_features(positions, site, setup.bs_positions)
    store.scalar_raw = np.concatenate((store.scalar_raw, extra_position), axis=1)
    scalar, map_scalar, scalar_normalization = store.scalar_tensors(rows["train"])

    autoencoder = ChannelAutoencoder(config).to(device)
    autoencoder_state, autoencoder_history = train_autoencoder(
        autoencoder,
        transform,
        channel,
        rows["train"],
        rows["validation"],
        store.nonzero_numpy,
        config,
        device,
    )
    autoencoder_scores = {
        name: evaluate_autoencoder(
            autoencoder,
            transform,
            channel,
            selected,
            config.evaluation_batch_size,
            device,
        )
        for name, selected in rows.items()
        if name != "ignored"
    }

    encoded = encode_latents(
        autoencoder,
        transform,
        channel,
        config.evaluation_batch_size,
        device,
    )
    np.savez_compressed(latent_path, **encoded)
    targets, constants = tensor_targets(encoded, rows["train"], device)
    latent_mean = torch.as_tensor(constants["latent_mean"], device=device)
    latent_std = torch.as_tensor(constants["latent_std"], device=device)
    power_mean = torch.as_tensor(constants["power_mean"][0], device=device)
    power_std = torch.as_tensor(constants["power_std"][0], device=device)

    model = GeometryFullChannelFunction(
        scalar.shape[1], map_scalar.shape[1], config
    ).to(device)
    operator_state, _, operator_history = train_function_stage(
        model,
        autoencoder,
        transform,
        store,
        channel,
        rows["train"],
        rows["validation"],
        scalar,
        map_scalar,
        targets,
        latent_mean,
        latent_std,
        power_mean,
        power_std,
        config,
        device,
        False,
    )
    operator_scores = {
        name: evaluate_function(
            model,
            autoencoder,
            transform,
            store,
            channel,
            selected,
            scalar,
            map_scalar,
            latent_mean,
            latent_std,
            power_mean,
            power_std,
            config.evaluation_batch_size,
            device,
        )
        for name, selected in rows.items()
        if name != "ignored"
    }

    finetune_model_state, finetune_autoencoder_state, finetune_history = train_function_stage(
        model,
        autoencoder,
        transform,
        store,
        channel,
        rows["train"],
        rows["validation"],
        scalar,
        map_scalar,
        targets,
        latent_mean,
        latent_std,
        power_mean,
        power_std,
        config,
        device,
        True,
    )
    final_scores = {
        name: evaluate_function(
            model,
            autoencoder,
            transform,
            store,
            channel,
            selected,
            scalar,
            map_scalar,
            latent_mean,
            latent_std,
            power_mean,
            power_std,
            config.evaluation_batch_size,
            device,
        )
        for name, selected in rows.items()
        if name != "ignored"
    }

    torch.save(
        {
            "version": VERSION,
            "config": asdict(config),
            "outer": args.outer,
            "stage": args.stage,
            "autoencoder": finetune_autoencoder_state,
            "operator": finetune_model_state,
            "pre_finetune_autoencoder": autoencoder_state,
            "pre_finetune_operator": operator_state,
            "constants": constants,
            "scalar_normalization": scalar_normalization,
        },
        checkpoint_path,
    )
    report = {
        "experiment": VERSION,
        "status": f"{args.stage}_complete",
        "primary_hypothesis": (
            "A distance-aligned baseline and always-on official geometry can be fitted "
            "as a function that directly generates the complete complex channel."
        ),
        "outer": int(args.outer),
        "config": asdict(config),
        "split": {key: int(len(value)) for key, value in rows.items()},
        "inputs": {
            "official_train_channel": PATHS.relative(TRAIN_CHANNEL),
            "official_geometry": PATHS.relative(PATHS.round2 / "Round2_Map.ply"),
            "uses_v11_or_v12": False,
            "uses_sionna_channel": False,
            "geometry_is_always_enabled": True,
        },
        "autoencoder": {
            "best_epoch": best_epoch(autoencoder_history),
            "scores": autoencoder_scores,
            "history": autoencoder_history,
        },
        "geometry_function_before_finetune": {
            "best_epoch": best_epoch(operator_history),
            "scores": operator_scores,
            "history": operator_history,
        },
        "geometry_function_after_finetune": {
            "best_epoch": best_epoch(finetune_history),
            "scores": final_scores,
            "history": finetune_history,
        },
        "diagnosis": {
            "representation_ceiling_held": float(autoencoder_scores["held"]["score"]),
            "function_fit_train": float(final_scores["train"]["score"]),
            "function_generalization_held": float(final_scores["held"]["score"]),
            "held_gap_to_representation_ceiling": float(
                autoencoder_scores["held"]["score"] - final_scores["held"]["score"]
            ),
        },
        "outputs": {
            "checkpoint": PATHS.relative(checkpoint_path),
            "latents": PATHS.relative(latent_path),
            "candidate_generated": False,
        },
        "seconds": time.time() - started,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
