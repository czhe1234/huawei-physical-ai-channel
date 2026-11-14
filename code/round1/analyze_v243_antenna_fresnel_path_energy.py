#!/usr/bin/env python3
"""v243: antenna-pattern and Fresnel constrained path-energy model."""

from __future__ import annotations

import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import analyze_v208_strict_historical_replay as v208
import analyze_v225_official_graph_residual_field as v225
import analyze_v239_nnls_path_energy_decoder as v239
import analyze_v240_plane_family_path_energy as v240
import analyze_v242_hierarchical_physical_path_energy as v242
import generate_v215_pointcloud_physics_neural_field as v215
from build_map_features import BS_POS
from generate_v87_easy_field_projection import summarize
from project_paths import PATHS


VERSION_NUMBER = 243
VERSION = "v243_antenna_fresnel_path_energy"
OUT_DIR = PATHS.cache_dir(VERSION_NUMBER)
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = REPORT_DIR / "v243_antenna_fresnel_path_energy.json"
TARGET_OUT = OUT_DIR / "v243_test_antenna_fresnel_pdp.npy"
ACTION_OUT = OUT_DIR / "v243_antenna_fresnel_actions.npz"

_train_pos = np.load(PATHS.round1 / "Round1_Train_Pos.npy").astype(np.float64)
_sector = np.mean(_train_pos - BS_POS[None, :], axis=0)
INIT_YAW = float(math.atan2(_sector[1], _sector[0]))
INIT_PITCH = float(math.atan2(_sector[2], max(math.hypot(_sector[0], _sector[1]), 1e-9)))


def physical_features(prefix: str, indices: np.ndarray | None) -> np.ndarray:
    cache = np.load(v215.PATH_CACHE)

    def take(name: str) -> np.ndarray:
        value = cache[f"{prefix}_{name}"].astype(np.float32)
        return value if indices is None else value[indices]

    base = np.stack(
        [
            np.nan_to_num(take("path_length"), nan=400.0, posinf=400.0),
            np.nan_to_num(take("score"), nan=-4.0, posinf=4.0, neginf=-4.0),
            np.clip(np.nan_to_num(take("incidence_cos"), nan=1e-3), 1e-3, 1.0),
            np.maximum(np.nan_to_num(take("excess_length"), nan=300.0), 0.0),
            np.clip(np.nan_to_num(take("edge_margin"), nan=0.0), -1.0, 1.0),
        ],
        axis=2,
    ).astype(np.float32)
    return np.concatenate(
        [
            base,
            np.nan_to_num(take("tx_dir"), nan=0.0),
            np.nan_to_num(take("rx_dir"), nan=0.0),
        ],
        axis=2,
    ).astype(np.float32)


class AntennaFresnelPathEnergy(nn.Module):
    def __init__(self, global_dim: int, path_dim: int):
        super().__init__()
        self.plane_coeff = nn.Embedding(v240.PLANE_COUNT, 1, padding_idx=0)
        self.pair_coeff = nn.Embedding(v240.PAIR_COUNT, 1, padding_idx=0)
        self.plane_emb = nn.Embedding(v240.PLANE_COUNT, 10, padding_idx=0)
        self.type_bias = nn.Parameter(torch.zeros(3))
        self.path_loss_raw = nn.Parameter(torch.full((3,), 0.55))
        self.score_coef = nn.Parameter(torch.full((3,), 0.40))
        self.incidence_raw = nn.Parameter(torch.full((3,), -0.60))
        self.excess_raw = nn.Parameter(torch.full((3,), -2.0))
        self.edge_coef = nn.Parameter(torch.full((3,), 0.10))
        self.yaw = nn.Parameter(torch.tensor(INIT_YAW, dtype=torch.float32))
        self.pitch = nn.Parameter(torch.tensor(INIT_PITCH, dtype=torch.float32))
        self.beamwidth_raw = nn.Parameter(torch.tensor(0.45, dtype=torch.float32))
        self.pattern_scale_raw = nn.Parameter(torch.tensor(0.20, dtype=torch.float32))
        self.epsilon_raw = nn.Parameter(torch.tensor(3.8, dtype=torch.float32))
        self.fresnel_scale_raw = nn.Parameter(torch.tensor(0.20, dtype=torch.float32))
        self.global_net = nn.Sequential(
            nn.Linear(global_dim, 256), nn.SiLU(), nn.LayerNorm(256), nn.Dropout(0.05), nn.Linear(256, 96), nn.SiLU()
        )
        residual_dim = path_dim + 96 + 20
        self.path_residual = nn.Sequential(
            nn.Linear(residual_dim, 192), nn.SiLU(), nn.LayerNorm(192), nn.Dropout(0.05), nn.Linear(192, 96), nn.SiLU()
        )
        self.energy_residual = nn.Linear(96, 1)
        self.shift_head = nn.Linear(96, len(v239.SHIFTS))
        self.tail_head = nn.Sequential(nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, v239.TAIL_COUNT))
        nn.init.zeros_(self.plane_coeff.weight)
        nn.init.zeros_(self.pair_coeff.weight)
        nn.init.zeros_(self.energy_residual.weight)
        nn.init.zeros_(self.energy_residual.bias)

    def forward(
        self,
        global_features: torch.Tensor,
        path_numeric: torch.Tensor,
        path_type: torch.Tensor,
        path_valid: torch.Tensor,
        raw_delay: torch.Tensor,
        physical: torch.Tensor,
        plane1: torch.Tensor,
        plane2: torch.Tensor,
        pair_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        type_idx = torch.clamp(path_type - 1, 0, 2)
        length = physical[..., 0].clamp_min(1.0)
        score = physical[..., 1]
        incidence = physical[..., 2].clamp(1e-3, 1.0)
        excess = physical[..., 3]
        edge = physical[..., 4]
        tx_dir = F.normalize(physical[..., 5:8], dim=2, eps=1e-8)

        path_loss = 1.0 + F.softplus(self.path_loss_raw[type_idx])
        incidence_coef = F.softplus(self.incidence_raw[type_idx])
        excess_coef = F.softplus(self.excess_raw[type_idx])
        base = (
            self.type_bias[type_idx]
            - path_loss * torch.log(length)
            + self.score_coef[type_idx] * score
            + incidence_coef * torch.log(incidence)
            - excess_coef * (excess / 100.0)
            + self.edge_coef[type_idx] * edge
            + self.plane_coeff(plane1).squeeze(2)
            + self.plane_coeff(plane2).squeeze(2)
            + self.pair_coeff(pair_id).squeeze(2)
        )

        cp = torch.cos(self.pitch)
        boresight = torch.stack([cp * torch.cos(self.yaw), cp * torch.sin(self.yaw), torch.sin(self.pitch)])
        cos_angle = torch.sum(tx_dir * boresight[None, None, :], dim=2).clamp(-1.0, 1.0)
        angle = torch.acos(cos_angle)
        beamwidth = math.radians(35.0) + F.softplus(self.beamwidth_raw)
        attenuation_db = torch.clamp(12.0 * (angle / beamwidth) ** 2, max=30.0)
        pattern_log_power = -attenuation_db * (math.log(10.0) / 10.0)

        eps_r = 1.0 + F.softplus(self.epsilon_raw)
        sin2 = (1.0 - incidence**2).clamp(0.0, 1.0)
        root = torch.sqrt((eps_r - sin2).clamp_min(1e-6))
        rs = (incidence - root) / (incidence + root).clamp_min(1e-6)
        rp = (eps_r * incidence - root) / (eps_r * incidence + root).clamp_min(1e-6)
        reflect_power = (0.5 * (rs**2 + rp**2)).clamp_min(1e-5)
        reflection_count = type_idx.float()
        fresnel_log_power = reflection_count * torch.log(reflect_power)
        physical_logit = (
            base
            + F.softplus(self.pattern_scale_raw) * pattern_log_power
            + F.softplus(self.fresnel_scale_raw) * fresnel_log_power
        )

        global_latent = self.global_net(global_features)
        residual_input = torch.cat(
            [
                path_numeric,
                global_latent[:, None, :].expand(-1, path_numeric.shape[1], -1),
                self.plane_emb(plane1),
                self.plane_emb(plane2),
            ],
            dim=2,
        )
        latent = self.path_residual(residual_input)
        residual = 0.45 * torch.tanh(self.energy_residual(latent).squeeze(2))
        path_logits = physical_logit[..., None] + residual[..., None] + self.shift_head(latent)
        path_logits = path_logits.masked_fill(~path_valid[..., None], -1e9).flatten(1)
        logits = torch.cat([path_logits, self.tail_head(global_latent)], dim=1)
        weights = torch.softmax(logits, dim=1)
        atoms = v239.torch_atoms(raw_delay, path_type, path_valid)
        pred = torch.sum(weights[..., None] * atoms, dim=1)
        pred /= pred.sum(dim=1, keepdim=True).clamp_min(1e-30)
        return pred, weights, residual


def run_variant(name: str, source, predicted_d: np.ndarray, alpha: np.ndarray, regimes: np.ndarray, clip: float) -> dict[str, object]:
    current = v225.current_delay_marginals(source)
    target_d = v239.normalize_d((1.0 - alpha[:, None]) * current["d"] + alpha[:, None] * predicted_d)
    targets = {"d": target_d, "du": current["du"], "dh": current["dh"]}
    output = OUT_DIR / f"v243_{name}_Round1_Test_Channel.npy"
    diag = v225.apply_delay_targets(source, output, targets, regimes, clip)
    return {
        "name": name,
        "source": PATHS.relative(source),
        "output": PATHS.relative(output),
        "score": v208.score_prediction(output),
        "alpha": summarize(alpha),
        "target_l1": summarize(v225.l1_rows(target_d, current["d"])),
        "diagnostics": {key: summarize(value) for key, value in diag.items()},
    }


def main() -> None:
    v242.VERSION_NUMBER = VERSION_NUMBER
    v242.VERSION = VERSION
    v242.OUT_DIR = OUT_DIR
    v242.REPORT_DIR = REPORT_DIR
    v242.CHECKPOINT_DIR = CHECKPOINT_DIR
    v242.REPORT = REPORT
    v242.TARGET_OUT = TARGET_OUT
    v242.ACTION_OUT = ACTION_OUT
    v242.TRAIN_EPOCHS = 320
    v242.SEEDS = (24301, 24337, 24373)
    v242.HierarchicalPhysicalPathEnergy = AntennaFresnelPathEnergy
    v242.physical_features = physical_features
    v242.run_variant = run_variant
    v242.main()
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    report["method"] = VERSION
    report["primary_hypothesis"] = (
        "Explicit TR 38.901 directional gain and learnable Fresnel reflection terms explain path-energy ordering missing from length/plane-only models."
    )
    report["antenna_fresnel"] = {
        "boresight_initialization": "official train-coordinate sector mean",
        "initial_yaw_rad": INIT_YAW,
        "initial_pitch_rad": INIT_PITCH,
        "element_pattern": "learnable-scale capped quadratic TR 38.901 approximation",
        "reflection": "learnable effective permittivity, averaged s/p Fresnel power",
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
