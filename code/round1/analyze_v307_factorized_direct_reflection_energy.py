#!/usr/bin/env python3
"""v307: factor direct-path mass from conditional reflection energy."""

from __future__ import annotations

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v297_continuous_delay_component_transport as v297
import analyze_v306_matched_patch_energy_transformer as v306
from project_paths import PATHS


VERSION_NUMBER = 307
VERSION = "v307_factorized_direct_reflection_energy"


class FactorizedPatchEnergyTransformer(v306.MatchedPatchEnergyTransformer):
    def __init__(self, global_dim: int, path_dim: int, patch_count: int):
        super().__init__(global_dim, path_dim, patch_count)
        self.direct_fraction_head = nn.Sequential(
            nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1)
        )
        nn.init.constant_(self.direct_fraction_head[-1].bias, 1.0)

    def forward(
        self,
        feature: v306.v264.FeatureSet,
        ids,
        anchor: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        global_feature = torch.from_numpy(feature.global_features[ids]).to(device)
        numeric = torch.from_numpy(feature.path_numeric[ids]).to(device)
        patch_id = torch.from_numpy(feature.patch_id[ids]).to(device)
        path_type = torch.from_numpy(feature.path_type[ids]).to(device)
        valid = torch.from_numpy(feature.path_valid[ids]).to(device)
        delay = torch.from_numpy(feature.delay[ids]).to(device)
        query = self.global_encoder(global_feature)
        path = self.path_encoder(
            torch.cat(
                [numeric, self.patch_embedding(patch_id), self.type_embedding(path_type)],
                dim=2,
            )
        )
        encoded = self.transformer(path + query[:, None, :], src_key_padding_mask=~valid)
        reflection_logits = self.energy_head(encoded[:, 1:]).squeeze(2)
        reflection_valid = valid[:, 1:]
        reflection_logits = reflection_logits.masked_fill(~reflection_valid, -1e9)
        reflection_weight = torch.softmax(reflection_logits, dim=1)
        reflection_pool = torch.sum(reflection_weight[:, :, None] * encoded[:, 1:], dim=1)
        direct_fraction = torch.sigmoid(
            self.direct_fraction_head(torch.cat([query, reflection_pool], dim=1)).squeeze(1)
        )
        weights = torch.cat(
            [
                direct_fraction[:, None],
                (1.0 - direct_fraction[:, None]) * reflection_weight,
            ],
            dim=1,
        )
        bins = torch.arange(v297.N_DELAY, device=device, dtype=delay.dtype)[None, None, :]
        delta = bins - delay[:, :, None]
        delta = torch.remainder(delta + v297.N_DELAY / 2.0, v297.N_DELAY) - v297.N_DELAY / 2.0
        atoms = torch.square(torch.sinc(delta) / torch.sinc(delta / v297.N_DELAY))
        atoms /= atoms.sum(dim=2, keepdim=True).clamp_min(1e-30)
        path_pdp = torch.sum(weights[:, :, None] * atoms, dim=1)
        pooled = direct_fraction[:, None] * encoded[:, 0] + (
            1.0 - direct_fraction[:, None]
        ) * reflection_pool
        gate = 0.70 * torch.sigmoid(self.gate(torch.cat([query, pooled], dim=1)))
        prediction = (1.0 - gate) * anchor + gate * path_pdp
        prediction /= prediction.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return prediction, {
            "weights": weights,
            "gate": gate.squeeze(1),
            "direct_fraction": direct_fraction,
            "reflection_weight": reflection_weight,
        }


def factorized_supervision_loss(
    diagnostics: dict[str, torch.Tensor],
    labels: torch.Tensor,
    quality: torch.Tensor,
) -> torch.Tensor:
    direct_target = labels[:, 0].clamp(0.0, 1.0)
    direct_loss = F.binary_cross_entropy(
        diagnostics["direct_fraction"], direct_target, reduction="none"
    )
    direct_loss = torch.sum(direct_loss * quality) / quality.sum().clamp_min(1e-6)
    reflection_mass = labels[:, 1:].sum(dim=1)
    reflection_target = labels[:, 1:] / reflection_mass[:, None].clamp_min(1e-8)
    reflection_loss = -torch.sum(
        reflection_target
        * torch.log(diagnostics["reflection_weight"].clamp_min(1e-10)),
        dim=1,
    )
    reflection_quality = quality * torch.sqrt(reflection_mass.clamp_min(0.0))
    reflection_loss = torch.sum(reflection_loss * reflection_quality) / reflection_quality.sum().clamp_min(1e-6)
    return direct_loss + 0.45 * reflection_loss


def configure() -> None:
    v306.VERSION_NUMBER = VERSION_NUMBER
    v306.VERSION = VERSION
    v306.OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
    v306.REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
    v306.CHECKPOINT_DIR = PATHS.checkpoints / VERSION
    v306.OUT_DIR.mkdir(parents=True, exist_ok=True)
    v306.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    v306.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    v306.REPORT = v306.REPORT_DIR / f"{VERSION}.json"
    v306.TARGETS_OUT = v306.OUT_DIR / "v307_factorized_delay_targets.npz"
    v306.OUTPUT = v306.OUT_DIR / "v307_Round1_Test_Channel.npy"
    v306.MODEL_CLASS = FactorizedPatchEnergyTransformer
    v306.SUPERVISION_LOSS = factorized_supervision_loss


def add_report_context() -> None:
    report = json.loads(v306.REPORT.read_text(encoding="utf-8"))
    report["factorization"] = {
        "direct": "independent continuous fraction",
        "reflection": "conditional softmax over 128 official-pointcloud candidates",
        "supervision": "soft direct BCE plus reflection-normalized cross entropy",
    }
    v306.REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    configure()
    v306.main()
    add_report_context()
