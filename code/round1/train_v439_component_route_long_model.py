#!/usr/bin/env python3
"""Long-train a component-supervised global route model on v437 labels."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_v438_geometry_path_support_large_model as v438
from project_paths import PATHS


VERSION_NUMBER = 439
VERSION = "v439_component_route_long_model"
BASE_CACHE = PATHS.cache_dir(438)
COMPONENT_CACHE = PATHS.cache_dir(439)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
WARMSTART = (
    PATHS.checkpoints
    / "v438_geometry_path_support_large_model"
    / "route_transformer_best.pt"
)

COMPONENT_COUNT = 80
TOTAL_COMPONENTS = 82
COMPONENT_FEATURES = 32
ACTIVE_DELAY_BINS = 32
H_BINS = 16
DIM = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--eval-batch-size", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=1.2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=2.0e-4)
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--min-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--coordinate-dropout", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=43_901)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--protect-warmstart", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


class TrainingData(v438.TrainingData):
    def __init__(self) -> None:
        super().__init__()
        names = (
            "component_features",
            "component_mass",
            "component_delay",
            "component_h",
            "component_match_fraction",
        )
        self.component = {
            name: np.load(COMPONENT_CACHE / f"{name}.npy", mmap_mode="r")
            for name in names
        }

    def batch(
        self,
        ids: np.ndarray,
        device: torch.device,
        *,
        geometry_ids: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        source_ids = ids if geometry_ids is None else geometry_ids
        output = super().batch(
            ids,
            device,
            include_route=True,
            route_ids=source_ids,
        )
        output["component_feature"] = torch.from_numpy(
            np.asarray(self.component["component_features"][source_ids], dtype=np.float32)
        ).to(device)
        for name in ("component_mass", "component_delay", "component_h"):
            output[name] = torch.from_numpy(
                np.asarray(self.component[name][ids], dtype=np.float32)
            ).to(device)
        return output


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = max(1, channels // 16)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.gelu(value + self.block(value))


class ComponentRouteTransformer(nn.Module):
    def __init__(self, coordinate_dropout: float) -> None:
        super().__init__()
        self.coordinate_dropout = coordinate_dropout
        self.base = v438.RouteSupportTransformer()
        self.component_feature = nn.Sequential(
            nn.Linear(COMPONENT_FEATURES, DIM),
            nn.LayerNorm(DIM),
            nn.GELU(),
            nn.Linear(DIM, DIM),
        )
        self.special_component = nn.Parameter(torch.randn(1, 2, DIM) * 0.02)
        self.component_id = nn.Parameter(torch.randn(1, TOTAL_COMPONENTS, DIM) * 0.02)
        self.component_type = nn.Parameter(torch.randn(1, 1, DIM) * 0.01)
        fusion_encoder_layer = nn.TransformerEncoderLayer(
            d_model=DIM,
            nhead=16,
            dim_feedforward=4 * DIM,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(
            fusion_encoder_layer, num_layers=4
        )
        self.fusion_delay_query = nn.Parameter(
            torch.randn(1, v438.DELAY_BINS, DIM) * 0.02
        )
        fusion_decoder_layer = nn.TransformerDecoderLayer(
            d_model=DIM,
            nhead=16,
            dim_feedforward=4 * DIM,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_decoder = nn.TransformerDecoder(
            fusion_decoder_layer, num_layers=4
        )
        self.residual_joint_head = nn.Linear(DIM, H_BINS)
        self.residual_path_delay_head = nn.Linear(DIM, 1)
        self.fusion_global = nn.Sequential(
            nn.LayerNorm(DIM),
            nn.Linear(DIM, DIM),
            nn.GELU(),
        )
        self.residual_depth_head = nn.Linear(DIM, v438.DEPTH_BINS)
        self.residual_scalar_head = nn.Linear(DIM, 4)
        self.residual_scale = nn.Parameter(torch.zeros(4))
        self.component_mass_head = nn.Linear(DIM, 1)
        self.component_delay_head = nn.Linear(DIM, ACTIVE_DELAY_BINS)
        self.component_h_head = nn.Linear(DIM, H_BINS)

    def forward(
        self,
        route: torch.Tensor,
        position: torch.Tensor,
        component_feature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.training and self.coordinate_dropout > 0.0:
            keep = (
                torch.rand(len(position), 1, device=position.device)
                >= self.coordinate_dropout
            ).to(position.dtype)
            position = torch.cat([position[:, :8], position[:, 8:] * keep], dim=1)
        route_feature = self.base.route(route)
        if route_feature.shape[-2:] != (10, 6):
            raise ValueError(f"Unexpected route feature map {route_feature.shape}")
        route_token = route_feature.flatten(2).transpose(1, 2)
        position_token = self.base.position(position)[:, None, :]
        base_memory = torch.cat([position_token, route_token], dim=1)
        base_memory = self.base.encoder(base_memory + self.base.route_position)
        base_query = (
            self.base.delay_query.expand(len(route), -1, -1) + base_memory[:, :1]
        )
        base_delay = self.base.decoder(base_query, base_memory)
        base_pooled = self.base.global_head(base_memory[:, 0])
        component = self.component_feature(component_feature)
        special = self.special_component.expand(len(route), -1, -1)
        component = torch.cat([component, special], dim=1)
        component = component + self.component_id + self.component_type
        fusion_memory = self.fusion_encoder(
            torch.cat([base_memory, component], dim=1)
        )
        component_memory = fusion_memory[:, 61:]
        fusion_query = (
            self.fusion_delay_query.expand(len(route), -1, -1)
            + fusion_memory[:, :1]
        )
        fusion_delay = self.fusion_decoder(fusion_query, fusion_memory)
        fusion_pooled = self.fusion_global(fusion_memory[:, 0])
        scalar_residual = self.residual_scalar_head(fusion_pooled)
        scale = torch.tanh(self.residual_scale)
        return {
            "joint_logits": self.base.joint_head(base_delay)
            + scale[0] * self.residual_joint_head(fusion_delay),
            "path_delay_logits": self.base.path_delay_head(base_delay).squeeze(-1)
            + scale[1] * self.residual_path_delay_head(fusion_delay).squeeze(-1),
            "depth_logits": self.base.depth_head(base_pooled)
            + scale[2] * self.residual_depth_head(fusion_pooled),
            "power": self.base.power_head(base_pooled).squeeze(-1)
            + scale[3] * scalar_residual[:, 0],
            "count": self.base.count_head(base_pooled).squeeze(-1)
            + scale[3] * scalar_residual[:, 1],
            "zero_logit": self.base.zero_head(base_pooled).squeeze(-1)
            + scale[3] * scalar_residual[:, 2],
            "los_logit": self.base.los_head(base_pooled).squeeze(-1)
            + scale[3] * scalar_residual[:, 3],
            "component_mass_logits": self.component_mass_head(component_memory).squeeze(-1),
            "component_delay_logits": self.component_delay_head(component_memory),
            "component_h_logits": self.component_h_head(component_memory),
        }


def cosine_probability_loss(
    probability: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    prediction = probability.flatten(1)
    truth = target.flatten(1)
    cosine = torch.sum(prediction * truth, dim=1) / torch.clamp(
        torch.linalg.vector_norm(prediction, dim=1)
        * torch.linalg.vector_norm(truth, dim=1),
        min=1e-8,
    )
    if torch.any(mask):
        return torch.mean(1.0 - cosine[mask])
    return torch.zeros((), device=probability.device, dtype=probability.dtype)


def model_loss(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    base, base_parts = v438.model_loss(output, batch)
    nonzero = batch["zero"] < 0.5
    joint = F.softmax(output["joint_logits"].flatten(1), dim=1).reshape(
        -1, v438.DELAY_BINS, H_BINS
    )
    reverse = torch.cat(
        [
            torch.zeros(1, device=joint.device, dtype=torch.long),
            torch.arange(v438.DELAY_BINS - 1, 0, -1, device=joint.device),
        ]
    )
    pdp = torch.sum(joint[:, reverse], dim=2)
    metric_joint = cosine_probability_loss(joint, batch["joint_sh"], nonzero)
    metric_pdp = cosine_probability_loss(pdp, batch["pdp"], nonzero)
    component_mass = v438.distribution_loss(
        output["component_mass_logits"], batch["component_mass"], nonzero
    )
    component_delay = v438.distribution_loss(
        output["component_delay_logits"], batch["component_delay"], nonzero
    )
    component_h = v438.distribution_loss(
        output["component_h_logits"], batch["component_h"], nonzero
    )
    total = (
        base
        + 0.70 * metric_joint
        + 0.45 * metric_pdp
        + 0.30 * component_mass
        + 0.22 * component_delay
        + 0.18 * component_h
    )
    parts = {
        **base_parts,
        "metric_joint": float(metric_joint.detach()),
        "metric_pdp": float(metric_pdp.detach()),
        "component_mass": float(component_mass.detach()),
        "component_delay": float(component_delay.detach()),
        "component_h": float(component_h.detach()),
    }
    return total, parts


def warmstart(model: nn.Module) -> dict[str, object]:
    checkpoint = torch.load(WARMSTART, map_location="cpu", weights_only=True)
    source = checkpoint["model"]
    target = model.state_dict()
    if set(source) == set(target) and all(
        source[name].shape == target[name].shape for name in source
    ):
        model.load_state_dict(source, strict=True)
        mode = "full_model"
    else:
        model.base.load_state_dict(source, strict=True)
        mode = "base_model"
    return {
        "source": PATHS.relative(WARMSTART),
        "mode": mode,
        "loaded_tensor_count": len(source),
        "source_tensor_count": len(source),
        "loaded_parameter_count": int(sum(value.numel() for value in source.values())),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data: TrainingData,
    ids: np.ndarray,
    device: torch.device,
    batch_size: int,
    *,
    geometry_ids: np.ndarray | None = None,
) -> dict[str, float]:
    model.eval()
    joint_parts: list[np.ndarray] = []
    path_parts: list[np.ndarray] = []
    depth_parts: list[np.ndarray] = []
    power_parts: list[np.ndarray] = []
    zero_parts: list[np.ndarray] = []
    component_parts: list[np.ndarray] = []
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        batch_ids = ids[start:end]
        source_ids = None if geometry_ids is None else geometry_ids[start:end]
        batch = data.batch(batch_ids, device, geometry_ids=source_ids)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = forward_model(model, batch)
        joint_parts.append(
            F.softmax(output["joint_logits"].flatten(1), dim=1)
            .reshape(-1, v438.DELAY_BINS, H_BINS)
            .float()
            .cpu()
            .numpy()
        )
        path_parts.append(
            F.softmax(output["path_delay_logits"], dim=1).float().cpu().numpy()
        )
        depth_parts.append(
            F.softmax(output["depth_logits"], dim=1).float().cpu().numpy()
        )
        power_parts.append(output["power"].float().cpu().numpy())
        zero_parts.append(torch.sigmoid(output["zero_logit"]).float().cpu().numpy())
        component_parts.append(
            F.softmax(output["component_mass_logits"], dim=1).float().cpu().numpy()
        )
    joint = np.concatenate(joint_parts)
    reverse = np.concatenate([[0], np.arange(v438.DELAY_BINS - 1, 0, -1)])
    pdp = np.sum(joint[:, reverse], axis=2)
    path_delay = np.concatenate(path_parts)
    depth = np.concatenate(depth_parts)
    power = np.concatenate(power_parts) * data.power_std + data.power_mean
    zero_probability = np.concatenate(zero_parts)
    component = np.concatenate(component_parts)
    zero = np.asarray(data.values["zero_channel"][ids], dtype=bool)
    nonzero = ~zero
    joint_cosine = float(
        np.mean(
            v438.cosine_rows(
                joint[nonzero],
                np.asarray(data.values["joint_sh"][ids[nonzero]], dtype=np.float32),
            )
        )
    )
    pdp_cosine = float(
        np.mean(
            v438.cosine_rows(
                pdp[nonzero],
                np.asarray(data.values["pdp"][ids[nonzero]], dtype=np.float32),
            )
        )
    )
    path_cosine = float(
        np.mean(
            v438.cosine_rows(
                path_delay[nonzero],
                np.asarray(data.values["path_delay"][ids[nonzero]], dtype=np.float32),
            )
        )
    )
    depth_cosine = float(
        np.mean(
            v438.cosine_rows(
                depth[nonzero],
                np.asarray(data.values["path_depth"][ids[nonzero]], dtype=np.float32),
            )
        )
    )
    component_cosine = float(
        np.mean(
            v438.cosine_rows(
                component[nonzero],
                np.asarray(data.component["component_mass"][ids[nonzero]], dtype=np.float32),
            )
        )
    )
    truth_power = np.asarray(data.values["log_power"][ids], dtype=np.float32)
    power_mae = float(np.mean(np.abs(power[nonzero] - truth_power[nonzero])))
    power_spearman = v438.spearman(power[nonzero], truth_power[nonzero])
    if np.any(zero) and np.any(nonzero):
        order = np.argsort(zero_probability)
        rank = np.empty(len(order), dtype=np.float64)
        rank[order] = np.arange(len(order), dtype=np.float64)
        zero_auc = float(
            (np.sum(rank[zero]) - np.sum(np.arange(np.sum(zero), dtype=np.float64)))
            / (np.sum(zero) * np.sum(nonzero))
        )
    else:
        zero_auc = 0.5
    return {
        "score": 0.60 * joint_cosine + 0.40 * pdp_cosine,
        "joint_sh_cosine": joint_cosine,
        "pdp_cosine": pdp_cosine,
        "path_delay_cosine": path_cosine,
        "path_depth_cosine": depth_cosine,
        "component_mass_cosine": component_cosine,
        "power_log_mae": power_mae,
        "power_spearman": power_spearman,
        "zero_auc": zero_auc,
        "count": int(len(ids)),
        "nonzero_count": int(np.sum(nonzero)),
    }


def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for target, source in zip(ema.parameters(), model.parameters(), strict=True):
            target.lerp_(source, 1.0 - decay)
        for target, source in zip(ema.buffers(), model.buffers(), strict=True):
            target.copy_(source)


def forward_model(
    model: nn.Module, batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Run the common model interface, including optional path-token inputs."""
    optional = {}
    if getattr(model, "uses_path_candidates", False):
        optional = {
            name: batch[name]
            for name in ("path_candidate_feature", "path_candidate_valid")
        }
    return model(
        batch["route"],
        batch["position"],
        batch["component_feature"],
        **optional,
    )


def learning_rate_scale(
    step: int, total_steps: int, warmup_steps: int, min_scale: float
) -> float:
    if step < warmup_steps:
        return max((step + 1) / max(warmup_steps, 1), 1e-3)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return min_scale + (1.0 - min_scale) * cosine


def train(args: argparse.Namespace) -> tuple[nn.Module, dict[str, object], TrainingData]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"v439 requires exactly one visible CUDA GPU, got {torch.cuda.device_count()}"
        )
    data = TrainingData()
    model = ComponentRouteTransformer(args.coordinate_dropout)
    warmstart_report = warmstart(model)
    resume_report: dict[str, object] | None = None
    epoch_offset = 0
    resume_validation: dict[str, float] | None = None
    resume_source = ""
    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(resume_checkpoint["model"])
        epoch_offset = int(resume_checkpoint["epoch"])
        resume_validation = resume_checkpoint["validation"]
        resume_source = str(resume_checkpoint["source"])
        resume_report = {
            "checkpoint": PATHS.relative(args.resume),
            "epoch": epoch_offset,
            "source": resume_source,
            "validation": resume_validation,
        }
    model = model.to(device)
    ema = copy.deepcopy(model).to(device).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    steps_per_epoch = math.ceil(len(data.train_ids) / args.batch_size)
    total_steps = max(args.epochs * steps_per_epoch, 1)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    minimum_scale = args.min_learning_rate / args.learning_rate
    rng = np.random.default_rng(args.seed)
    history: list[dict[str, object]] = []
    best_score = (
        float(resume_validation["score"])
        if resume_validation is not None
        else -np.inf
    )
    best_epoch = epoch_offset
    best_source = resume_source
    stale = 0
    global_step = 0
    checkpoint = CHECKPOINT_DIR / "component_route_best.pt"
    started = time.time()
    epochs = 2 if args.smoke else args.epochs
    train_ids = data.train_ids[: min(len(data.train_ids), 2048)] if args.smoke else data.train_ids
    validation_ids = (
        data.validation_ids[: min(len(data.validation_ids), 512)]
        if args.smoke
        else data.validation_ids
    )
    initial_validation: dict[str, float] | None = None
    if args.protect_warmstart and resume_validation is None:
        initial_validation = evaluate_model(
            model,
            data,
            validation_ids,
            device,
            args.eval_batch_size,
        )
        best_score = float(initial_validation["score"])
        best_epoch = 0
        best_source = "warmstart"
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": 0,
                "source": "warmstart",
                "validation": initial_validation,
                "args": serializable_args(args),
                "power_mean": data.power_mean,
                "power_std": data.power_std,
                "count_mean": data.count_mean,
                "count_std": data.count_std,
            },
            checkpoint,
        )
        print(
            f"{VERSION} epoch=0 baseline={initial_validation['score']:.6f} "
            f"joint={initial_validation['joint_sh_cosine']:.6f} "
            f"pdp={initial_validation['pdp_cosine']:.6f}",
            flush=True,
        )
    for stage_epoch in range(1, epochs + 1):
        epoch = epoch_offset + stage_epoch
        model.train()
        order = rng.permutation(train_ids)
        epoch_loss = 0.0
        part_sum: dict[str, float] = {}
        seen = 0
        epoch_started = time.time()
        for start in range(0, len(order), args.batch_size):
            ids = order[start : start + args.batch_size]
            scale = learning_rate_scale(
                global_step, total_steps, warmup_steps, minimum_scale
            )
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * scale
            batch = data.batch(ids, device)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                output = forward_model(model, batch)
                loss, parts = model_loss(output, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            update_ema(ema, model, args.ema_decay)
            global_step += 1
            epoch_loss += float(loss.detach()) * len(ids)
            seen += len(ids)
            for name, value in parts.items():
                part_sum[name] = part_sum.get(name, 0.0) + value * len(ids)
        raw_validation = evaluate_model(
            model,
            data,
            validation_ids,
            device,
            args.eval_batch_size,
        )
        ema_validation = evaluate_model(
            ema,
            data,
            validation_ids,
            device,
            args.eval_batch_size,
        )
        candidates = (("raw", raw_validation, model), ("ema", ema_validation, ema))
        source, validation, selected_model = max(
            candidates, key=lambda item: item[1]["score"]
        )
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(seen, 1),
            "train_parts": {
                name: value / max(seen, 1) for name, value in part_sum.items()
            },
            "raw_validation": raw_validation,
            "ema_validation": ema_validation,
            "selected_source": source,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": float(time.time() - epoch_started),
        }
        history.append(row)
        print(
            f"{VERSION} epoch={epoch} loss={row['train_loss']:.5f} "
            f"raw={raw_validation['score']:.6f} ema={ema_validation['score']:.6f} "
            f"joint={validation['joint_sh_cosine']:.6f} pdp={validation['pdp_cosine']:.6f} "
            f"component={validation['component_mass_cosine']:.6f} "
            f"lr={row['learning_rate']:.3e} seconds={row['seconds']:.1f}",
            flush=True,
        )
        if validation["score"] > best_score + 1e-5:
            best_score = validation["score"]
            best_epoch = epoch
            best_source = source
            stale = 0
            torch.save(
                {
                    "model": selected_model.state_dict(),
                    "epoch": epoch,
                    "source": source,
                    "validation": validation,
                    "args": serializable_args(args),
                    "power_mean": data.power_mean,
                    "power_std": data.power_std,
                    "count_mean": data.count_mean,
                    "count_std": data.count_std,
                },
                checkpoint,
            )
        else:
            stale += 1
        if (
            not args.smoke
            and stage_epoch >= args.min_epochs
            and stale >= args.patience
        ):
            break
    best = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(best["model"])
    return model, {
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "best_epoch": int(best_epoch),
        "best_source": best_source,
        "best_validation": best["validation"],
        "checkpoint": PATHS.relative(checkpoint),
        "seconds": float(time.time() - started),
        "warmstart": warmstart_report,
        "resume": resume_report,
        "initial_validation": initial_validation,
        "history": history,
    }, data


def main() -> None:
    args = parse_args()
    for path in (
        BASE_CACHE / "route_raster.npy",
        COMPONENT_CACHE / "component_features.npy",
        COMPONENT_CACHE / "component_mass.npy",
        WARMSTART,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    model, training, data = train(args)
    if args.smoke:
        print(json.dumps({"status": "smoke_pass", "training": training}, indent=2))
        return
    device = next(model.parameters()).device
    rng = np.random.default_rng(args.seed + 1)
    train_sample = np.sort(rng.choice(data.train_ids, 10_000, replace=False))
    train_metrics = evaluate_model(
        model, data, train_sample, device, args.eval_batch_size
    )
    permutation = rng.permutation(data.holdout_ids)
    holdout = evaluate_model(
        model, data, data.holdout_ids, device, args.eval_batch_size
    )
    shuffled = evaluate_model(
        model,
        data,
        data.holdout_ids,
        device,
        args.eval_batch_size,
        geometry_ids=permutation,
    )
    v438_report_path = (
        PATHS.version_report_dir(438) / "v438_geometry_path_support_large_model.json"
    )
    v438_report = json.loads(v438_report_path.read_text(encoding="utf-8"))
    v438_holdout = v438_report["holdout"]["route_transformer"]
    report = {
        "method": VERSION,
        "status": "synthetic_spatial_holdout_evaluated",
        "primary_hypothesis": (
            "Point-cloud component tokens with direct path-family supervision and long "
            "metric-aligned training generalize beyond v438's coordinate shortcut."
        ),
        "single_gpu": {
            "logical_device_count": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(0),
        },
        "parameters": serializable_args(args),
        "training": training,
        "train_sample": train_metrics,
        "holdout": {
            "v438": v438_holdout,
            "v439": holdout,
            "v439_shuffled_all_geometry": shuffled,
            "v439_minus_v438": {
                name: float(holdout[name] - v438_holdout[name])
                for name in ("score", "joint_sh_cosine", "pdp_cosine")
            },
            "v439_minus_shuffled": {
                name: float(holdout[name] - shuffled[name])
                for name in ("score", "joint_sh_cosine", "pdp_cosine")
            },
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["holdout"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
