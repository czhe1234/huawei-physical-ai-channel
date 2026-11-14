#!/usr/bin/env python3
"""Add a high-resolution visibility branch to the frozen v443 model."""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_v438_geometry_path_support_large_model as v438
import train_v439_component_route_long_model as trainer
from project_paths import PATHS


VERSION_NUMBER = 444
VERSION = "v444_high_resolution_visibility_model"
CACHE = PATHS.cache_dir(443)
WARMSTART = (
    PATHS.checkpoints
    / "v443_dense_component_route_model"
    / "component_route_best.pt"
)

BaseComponentModel = trainer.ComponentRouteTransformer


class HighResolutionVisibilityModel(BaseComponentModel):
    """Preserve one-metre route changes that the v443 16x downsampling loses."""

    def __init__(self, coordinate_dropout: float) -> None:
        super().__init__(coordinate_dropout)
        dim = 256
        self.visibility_stem = nn.Sequential(
            nn.Conv2d(12, 64, 5, padding=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 5, stride=(1, 2), padding=2, bias=False),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.Conv2d(128, dim, 5, stride=(1, 2), padding=2, bias=False),
            nn.GroupNorm(32, dim),
            nn.GELU(),
        )
        self.visibility_position = nn.Sequential(
            nn.Linear(32, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.visibility_axis = nn.Parameter(torch.randn(1, 160, dim) * 0.01)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=8,
            dim_feedforward=4 * dim,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.visibility_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.visibility_query = nn.Parameter(
            torch.randn(1, v438.DELAY_BINS, dim) * 0.02
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=8,
            dim_feedforward=4 * dim,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.visibility_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.visibility_joint_head = nn.Linear(dim, trainer.H_BINS)
        self.visibility_path_head = nn.Linear(dim, 1)
        nn.init.zeros_(self.visibility_joint_head.weight)
        nn.init.zeros_(self.visibility_joint_head.bias)
        nn.init.zeros_(self.visibility_path_head.weight)
        nn.init.zeros_(self.visibility_path_head.bias)

    def forward(
        self,
        route: torch.Tensor,
        position: torch.Tensor,
        component_feature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = super().forward(route, position, component_feature)
        feature = self.visibility_stem(route)
        # Average captures broad corridors; max preserves thin blockers.
        feature = 0.5 * (torch.mean(feature, dim=3) + torch.amax(feature, dim=3))
        memory = feature.transpose(1, 2)
        position_token = self.visibility_position(position)[:, None, :]
        memory = self.visibility_encoder(
            memory + self.visibility_axis + position_token
        )
        query = self.visibility_query.expand(len(route), -1, -1) + position_token
        delay = self.visibility_decoder(query, memory)
        output["joint_logits"] = (
            output["joint_logits"] + self.visibility_joint_head(delay)
        )
        output["path_delay_logits"] = (
            output["path_delay_logits"]
            + self.visibility_path_head(delay).squeeze(-1)
        )
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
            f"Only {len(compatible)}/{len(source)} v443 tensors match v444"
        )
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected warmstart tensors: {unexpected}")
    trainable_prefix = "visibility_"
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(trainable_prefix))
    return {
        "source": PATHS.relative(WARMSTART),
        "mode": "full_v443_plus_frozen_zero_initialized_visibility_branch",
        "loaded_tensor_count": len(compatible),
        "new_tensor_count": len(missing),
        "loaded_parameter_count": int(
            sum(value.numel() for value in compatible.values())
        ),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }


def main() -> None:
    trainer.VERSION_NUMBER = VERSION_NUMBER
    trainer.VERSION = VERSION
    trainer.BASE_CACHE = CACHE
    trainer.COMPONENT_CACHE = CACHE
    trainer.CHECKPOINT_DIR = PATHS.checkpoints / VERSION
    trainer.REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
    trainer.REPORT = trainer.REPORT_DIR / f"{VERSION}.json"
    trainer.WARMSTART = WARMSTART
    trainer.ComponentRouteTransformer = HighResolutionVisibilityModel
    trainer.warmstart = warmstart_and_freeze
    v438.CACHE = CACHE

    args = trainer.parse_args()
    for path in (
        CACHE / "route_raster.npy",
        CACHE / "component_features.npy",
        CACHE / "component_mass.npy",
        WARMSTART,
    ):
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
        "status": "high_resolution_visibility_holdout_evaluated",
        "primary_hypothesis": (
            "Preserving the 160-sample BS-UE obstruction profile predicts path "
            "birth and death beyond v443's 16x-downsampled route encoder."
        ),
        "data": {
            "source": "v442 Sionna labels and official point-cloud route raster",
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
            "v444": holdout,
            "v444_shuffled_geometry": shuffled,
            "v444_minus_v443": {
                name: float(holdout[name] - baseline[name])
                for name in ("score", "joint_sh_cosine", "pdp_cosine")
            },
            "v444_minus_shuffled": {
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
