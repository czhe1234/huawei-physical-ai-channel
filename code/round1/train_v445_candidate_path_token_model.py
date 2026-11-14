#!/usr/bin/env python3
"""Train a power-free candidate-path token branch on top of frozen v443."""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_v438_geometry_path_support_large_model as v438
import train_v439_component_route_long_model as trainer
from project_paths import PATHS


VERSION_NUMBER = 445
VERSION = "v445_candidate_path_token_model"
BASE_CACHE = PATHS.cache_dir(443)
PATH_CACHE = PATHS.cache_dir(445)
WARMSTART = (
    PATHS.checkpoints
    / "v443_dense_component_route_model"
    / "component_route_best.pt"
)
FEATURES = 28
MAX_PATHS = 128
DIM = 256

BaseComponentModel = trainer.ComponentRouteTransformer
base_model_loss = trainer.model_loss


class CandidateTrainingData(trainer.TrainingData):
    def __init__(self) -> None:
        super().__init__()
        self.path = {
            name: np.load(PATH_CACHE / f"{name}.npy", mmap_mode="r")
            for name in (
                "path_candidate_feature",
                "path_candidate_valid",
                "path_candidate_mass",
            )
        }

    def batch(
        self,
        ids: np.ndarray,
        device: torch.device,
        *,
        geometry_ids: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        output = super().batch(ids, device, geometry_ids=geometry_ids)
        source_ids = ids if geometry_ids is None else geometry_ids
        output["path_candidate_feature"] = torch.from_numpy(
            np.asarray(self.path["path_candidate_feature"][source_ids], dtype=np.float32)
        ).to(device)
        output["path_candidate_valid"] = torch.from_numpy(
            np.asarray(self.path["path_candidate_valid"][source_ids], dtype=np.bool_)
        ).to(device)
        output["path_candidate_mass"] = torch.from_numpy(
            np.asarray(self.path["path_candidate_mass"][ids], dtype=np.float32)
        ).to(device)
        return output


class CandidatePathTokenModel(BaseComponentModel):
    uses_path_candidates = True

    def __init__(self, coordinate_dropout: float) -> None:
        super().__init__(coordinate_dropout)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(FEATURES, DIM),
            nn.LayerNorm(DIM),
            nn.GELU(),
            nn.Linear(DIM, DIM),
        )
        self.candidate_position = nn.Sequential(
            nn.Linear(32, DIM),
            nn.LayerNorm(DIM),
            nn.GELU(),
            nn.Linear(DIM, DIM),
        )
        self.candidate_special = nn.Parameter(torch.randn(1, 1, DIM) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=DIM,
            nhead=8,
            dim_feedforward=4 * DIM,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=6
        )
        self.candidate_mass_head = nn.Linear(DIM, 1)
        self.candidate_query = nn.Parameter(
            torch.randn(1, v438.DELAY_BINS, DIM) * 0.02
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=DIM,
            nhead=8,
            dim_feedforward=4 * DIM,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=4
        )
        self.candidate_joint_head = nn.Linear(DIM, trainer.H_BINS)
        self.candidate_delay_head = nn.Linear(DIM, 1)
        nn.init.zeros_(self.candidate_joint_head.weight)
        nn.init.zeros_(self.candidate_joint_head.bias)
        nn.init.zeros_(self.candidate_delay_head.weight)
        nn.init.zeros_(self.candidate_delay_head.bias)

    def forward(
        self,
        route: torch.Tensor,
        position: torch.Tensor,
        component_feature: torch.Tensor,
        *,
        path_candidate_feature: torch.Tensor,
        path_candidate_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = super().forward(route, position, component_feature)
        position_token = self.candidate_position(position)[:, None, :]
        path = self.candidate_encoder(path_candidate_feature) + position_token
        special = self.candidate_special.expand(len(route), -1, -1) + position_token
        memory = torch.cat([special, path], dim=1)
        memory_valid = torch.cat(
            [
                torch.ones(
                    len(route), 1, dtype=torch.bool, device=path_candidate_valid.device
                ),
                path_candidate_valid,
            ],
            dim=1,
        )
        memory = self.candidate_transformer(
            memory, src_key_padding_mask=~memory_valid
        )
        encoded_path = memory[:, 1:]
        mass_logits = self.candidate_mass_head(encoded_path).squeeze(-1)
        mass_logits = mass_logits.masked_fill(~path_candidate_valid, -1e4)
        mass = F.softmax(mass_logits, dim=1) * path_candidate_valid.to(mass_logits.dtype)
        mass = mass / mass.sum(dim=1, keepdim=True).clamp_min(1e-8)
        weighted_path = encoded_path * (1.0 + 4.0 * mass[:, :, None])
        decoder_memory = torch.cat([memory[:, :1], weighted_path], dim=1)
        query = self.candidate_query.expand(len(route), -1, -1) + position_token
        delay = self.candidate_decoder(
            query,
            decoder_memory,
            memory_key_padding_mask=~memory_valid,
        )
        output["joint_logits"] = (
            output["joint_logits"] + self.candidate_joint_head(delay)
        )
        output["path_delay_logits"] = (
            output["path_delay_logits"]
            + self.candidate_delay_head(delay).squeeze(-1)
        )
        output["candidate_mass_logits"] = mass_logits
        return output


def warmstart_and_freeze(model: nn.Module) -> dict[str, object]:
    checkpoint = torch.load(WARMSTART, map_location="cpu", weights_only=True)
    source = checkpoint["model"]
    target = model.state_dict()
    compatible = {
        name: value
        for name, value in source.items()
        if name in target and target[name].shape == value.shape
    }
    if len(compatible) != len(source):
        raise RuntimeError(
            f"Only {len(compatible)}/{len(source)} v443 tensors match v445"
        )
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected warmstart tensors: {unexpected}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("candidate_"))
    return {
        "source": PATHS.relative(WARMSTART),
        "mode": "full_v443_plus_frozen_zero_initialized_candidate_branch",
        "loaded_tensor_count": len(compatible),
        "new_tensor_count": len(missing),
        "loaded_parameter_count": int(
            sum(value.numel() for value in compatible.values())
        ),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }


def candidate_model_loss(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    base, parts = base_model_loss(output, batch)
    valid_row = torch.any(batch["path_candidate_valid"], dim=1) & (batch["zero"] < 0.5)
    row = -torch.sum(
        batch["path_candidate_mass"]
        * F.log_softmax(output["candidate_mass_logits"], dim=1),
        dim=1,
    )
    mass_loss = (
        torch.mean(row[valid_row])
        if torch.any(valid_row)
        else torch.zeros((), device=row.device, dtype=row.dtype)
    )
    return base + 0.30 * mass_loss, {
        **parts,
        "candidate_mass": float(mass_loss.detach()),
    }


def main() -> None:
    trainer.VERSION_NUMBER = VERSION_NUMBER
    trainer.VERSION = VERSION
    trainer.BASE_CACHE = BASE_CACHE
    trainer.COMPONENT_CACHE = BASE_CACHE
    trainer.CHECKPOINT_DIR = PATHS.checkpoints / VERSION
    trainer.REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
    trainer.REPORT = trainer.REPORT_DIR / f"{VERSION}.json"
    trainer.WARMSTART = WARMSTART
    trainer.TrainingData = CandidateTrainingData
    trainer.ComponentRouteTransformer = CandidatePathTokenModel
    trainer.warmstart = warmstart_and_freeze
    trainer.model_loss = candidate_model_loss
    v438.CACHE = BASE_CACHE

    args = trainer.parse_args()
    required = (
        BASE_CACHE / "route_raster.npy",
        BASE_CACHE / "component_features.npy",
        PATH_CACHE / "path_candidate_feature.npy",
        PATH_CACHE / "path_candidate_valid.npy",
        PATH_CACHE / "path_candidate_mass.npy",
        WARMSTART,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    trainer.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    model, training, data = trainer.train(args)
    if args.smoke:
        print(json.dumps({"status": "smoke_pass", "training": training}, indent=2))
        return

    device = next(model.parameters()).device
    rng = np.random.default_rng(args.seed + 1)
    holdout = trainer.evaluate_model(
        model, data, data.holdout_ids, device, args.eval_batch_size
    )
    shuffled_ids = rng.permutation(data.holdout_ids)
    shuffled = trainer.evaluate_model(
        model,
        data,
        data.holdout_ids,
        device,
        args.eval_batch_size,
        geometry_ids=shuffled_ids,
    )
    baseline_model = BaseComponentModel(args.coordinate_dropout).to(device)
    baseline_checkpoint = torch.load(WARMSTART, map_location=device, weights_only=True)
    baseline_model.load_state_dict(baseline_checkpoint["model"])
    baseline = trainer.evaluate_model(
        baseline_model, data, data.holdout_ids, device, args.eval_batch_size
    )
    report = {
        "method": VERSION,
        "status": "candidate_path_holdout_evaluated",
        "primary_hypothesis": (
            "Power-free candidate existence, delay, angles, and interactions resolve "
            "path birth/death that route rasters and local interpolation miss."
        ),
        "leakage_control": {
            "input_power": False,
            "input_complex_coefficient": False,
            "input_power_rank": False,
            "candidate_order": "tau_then_depth_then_departure_and_arrival_angle",
        },
        "data": {
            "train_count": int(len(data.train_ids)),
            "validation_count": int(len(data.validation_ids)),
            "holdout_count": int(len(data.holdout_ids)),
        },
        "parameters": trainer.serializable_args(args),
        "single_gpu": {
            "logical_device_count": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(0),
        },
        "training": training,
        "holdout": {
            "v443": baseline,
            "v445": holdout,
            "v445_shuffled_all_geometry": shuffled,
            "v445_minus_v443": {
                name: float(holdout[name] - baseline[name])
                for name in ("score", "joint_sh_cosine", "pdp_cosine")
            },
            "v445_minus_shuffled": {
                name: float(holdout[name] - shuffled[name])
                for name in ("score", "joint_sh_cosine", "pdp_cosine")
            },
        },
    }
    trainer.REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["holdout"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
