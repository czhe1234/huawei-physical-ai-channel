#!/usr/bin/env python3
"""Planar geometry-conditioned residual field on official joint support."""

from __future__ import annotations

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v283_joint_geometry_path_renderer as v283
from project_paths import PATHS


VERSION_NUMBER = 284
VERSION = "v284_planar_geometry_residual_field"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / f"{VERSION}.json"
TARGETS_OUT = OUT_DIR / "v284_planar_geometry_targets.npz"
OUTPUT = OUT_DIR / "v284_Round1_Test_Channel.npy"

PATH_CHANNELS = 24
FIELD_CHANNELS = 48
MAX_LOG_GAIN = 2.5
MAX_SOURCE_MASS = 0.18
DILATIONS = (1, 2, 4, 8, 16, 4)


class CircularConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        dilation: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.pad_s = dilation[0] * (kernel_size[0] // 2)
        self.pad_h = dilation[1] * (kernel_size[1] // 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(
            x,
            (self.pad_h, self.pad_h, self.pad_s, self.pad_s),
            mode="circular",
        )
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation_s: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = CircularConv2d(
            channels, channels, dilation=(dilation_s, 1)
        )
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = CircularConv2d(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(F.silu(self.norm1(x)))
        residual = self.conv2(F.silu(self.norm2(residual)))
        return x + 0.55 * residual


class PlanarGeometryResidualField(nn.Module):
    def __init__(self, global_dim: int, path_dim: int, patch_count: int) -> None:
        super().__init__()
        self.patch_embedding = nn.Embedding(patch_count + 1, 20)
        self.type_embedding = nn.Embedding(2, 6)
        self.path_encoder = nn.Sequential(
            nn.Linear(path_dim + 26, 96),
            nn.SiLU(),
            nn.LayerNorm(96),
            nn.Linear(96, PATH_CHANNELS),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, 192),
            nn.SiLU(),
            nn.LayerNorm(192),
            nn.Linear(192, FIELD_CHANNELS),
        )
        self.input_projection = CircularConv2d(
            PATH_CHANNELS + 3, FIELD_CHANNELS
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(FIELD_CHANNELS, dilation) for dilation in DILATIONS]
        )
        self.output_norm = nn.GroupNorm(8, FIELD_CHANNELS)
        self.output_head = nn.Conv2d(FIELD_CHANNELS, 2, kernel_size=1)
        self.source_gate = nn.Sequential(
            nn.Linear(FIELD_CHANNELS * 2, 96),
            nn.SiLU(),
            nn.Linear(96, 1),
        )
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
        nn.init.zeros_(self.source_gate[-1].weight)
        nn.init.constant_(self.source_gate[-1].bias, -3.4)

    @staticmethod
    def geometry_support(occupancy: torch.Tensor) -> torch.Tensor:
        support = torch.zeros_like(occupancy)
        for ds, dh in v283.OFFSETS:
            support = support + torch.roll(
                occupancy, shifts=(int(ds), int(dh)), dims=(2, 3)
            )
        return support > 0.0

    def forward(
        self,
        global_feature: torch.Tensor,
        path_numeric: torch.Tensor,
        patch_id: torch.Tensor,
        path_type: torch.Tensor,
        path_valid: torch.Tensor,
        center_s: torch.Tensor,
        center_h: torch.Tensor,
        anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, path_count = center_s.shape
        path_latent = self.path_encoder(
            torch.cat(
                [
                    path_numeric,
                    self.patch_embedding(patch_id),
                    self.type_embedding(path_type),
                ],
                dim=2,
            )
        )
        valid = path_valid[:, None, :].to(path_latent.dtype)
        destination = center_s * v283.v279.H_SIZE + center_h
        destination = destination[:, None, :].expand(-1, PATH_CHANNELS, -1)
        field_flat = torch.zeros(
            (batch, PATH_CHANNELS, v283.v279.S_SIZE * v283.v279.H_SIZE),
            dtype=anchor.dtype,
            device=anchor.device,
        )
        field_flat.scatter_add_(
            2, destination, path_latent.transpose(1, 2) * valid
        )
        count_flat = torch.zeros(
            (batch, 1, v283.v279.S_SIZE * v283.v279.H_SIZE),
            dtype=anchor.dtype,
            device=anchor.device,
        )
        count_flat.scatter_add_(
            2,
            destination[:, :1],
            path_valid[:, None, :].to(anchor.dtype),
        )
        field_flat = field_flat / count_flat.clamp_min(1.0)
        geometry = field_flat.reshape(
            batch, PATH_CHANNELS, v283.v279.S_SIZE, v283.v279.H_SIZE
        )
        occupancy = count_flat.reshape(
            batch, 1, v283.v279.S_SIZE, v283.v279.H_SIZE
        )

        anchor_map = anchor.reshape(
            batch, v283.v279.S_SIZE, v283.v279.H_SIZE
        )
        sqrt_anchor = torch.sqrt(anchor_map.clamp_min(0.0))
        log_anchor = torch.log(anchor_map.clamp_min(1e-10))
        log_anchor = log_anchor - log_anchor.mean(dim=(1, 2), keepdim=True)
        log_anchor = log_anchor / log_anchor.std(
            dim=(1, 2), keepdim=True
        ).clamp_min(0.2)
        occupancy_feature = torch.log1p(occupancy)
        network_input = torch.cat(
            [
                sqrt_anchor[:, None],
                log_anchor[:, None],
                occupancy_feature,
                geometry,
            ],
            dim=1,
        )
        global_latent = self.global_encoder(global_feature)
        hidden = self.input_projection(network_input)
        hidden = hidden + global_latent[:, :, None, None]
        for block in self.blocks:
            hidden = block(hidden)
        raw = self.output_head(F.silu(self.output_norm(hidden)))

        correction = MAX_LOG_GAIN * torch.tanh(raw[:, 0])
        corrected_anchor = anchor_map * torch.exp(correction)
        corrected_anchor = corrected_anchor / corrected_anchor.sum(
            dim=(1, 2), keepdim=True
        ).clamp_min(1e-30)

        support = self.geometry_support(occupancy)
        source_logits = raw[:, 1].masked_fill(~support[:, 0], -1e4)
        source = torch.softmax(source_logits.flatten(1), dim=1).reshape_as(anchor_map)
        pooled = hidden.mean(dim=(2, 3))
        source_mass = MAX_SOURCE_MASS * torch.sigmoid(
            self.source_gate(torch.cat([global_latent, pooled], dim=1))
        )
        prediction = (
            (1.0 - source_mass[:, :, None]) * corrected_anchor
            + source_mass[:, :, None] * source
        )
        prediction = prediction / prediction.sum(
            dim=(1, 2), keepdim=True
        ).clamp_min(1e-30)
        return prediction.flatten(1), {
            "gate": source_mass.squeeze(1),
            "source_mass": source_mass.squeeze(1),
            "correction": correction,
            "support_fraction": support.float().mean(dim=(1, 2, 3)),
        }


def residual_field_loss(
    prediction: torch.Tensor,
    truth_sh: torch.Tensor,
    truth_d: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
) -> torch.Tensor:
    pred_d = v283.v279.derive_d_torch(prediction)
    sh_loss = 1.0 - F.cosine_similarity(prediction, truth_sh, dim=1).mean()
    d_loss = 1.0 - F.cosine_similarity(pred_d, truth_d, dim=1).mean()
    correction_penalty = diagnostics["correction"].square().mean()
    source_penalty = diagnostics["source_mass"].mean()
    return (
        0.52 * sh_loss
        + 0.48 * d_loss
        + 2e-6 * correction_penalty
        + 2e-5 * source_penalty
    )


def configure_runner() -> None:
    v283.VERSION_NUMBER = VERSION_NUMBER
    v283.VERSION = VERSION
    v283.OUT_DIR = OUT_DIR
    v283.REPORT_DIR = REPORT_DIR
    v283.CHECKPOINT_DIR = CHECKPOINT_DIR
    v283.REPORT = REPORT
    v283.TARGETS_OUT = TARGETS_OUT
    v283.OUTPUT = OUTPUT
    v283.OUTER_SEEDS = (284, 1284, 2284, 3284, 4284)
    v283.EPOCHS = 60
    v283.FINAL_SEEDS = (28401, 28431, 28461)
    v283.JointPathRenderer = PlanarGeometryResidualField
    v283.loss_fn = residual_field_loss


def rewrite_report() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    report["method"] = VERSION
    report["primary_hypothesis"] = (
        "The full wall-path candidates are useful as a planar joint delay-angle coordinate "
        "field, but should condition bounded local corrections to the official anchor rather "
        "than reconstruct the complete spectrum as a free sparse path mixture."
    )
    report["architecture"] = {
        "joint_shape": [v283.v279.S_SIZE, v283.v279.H_SIZE],
        "path_channels": PATH_CHANNELS,
        "field_channels": FIELD_CHANNELS,
        "dilations": list(DILATIONS),
        "maximum_log_gain": MAX_LOG_GAIN,
        "maximum_geometry_source_mass": MAX_SOURCE_MASS,
        "circular_axes": ["delay-S", "H-angle"],
        "zero_initialized_residual": True,
    }
    report["compliance_boundary"]["uses"] = [
        "official train H joint S-H/PDP labels",
        "official train/test planar coordinates",
        "official point-cloud full wall-reflection geometry rasterized as AI features",
        "strict v239 official-only carrier after official selection",
    ]
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    configure_runner()
    v283.main()
    rewrite_report()


if __name__ == "__main__":
    main()
