#!/usr/bin/env python3
"""Train v438 coordinate and route-transformer models on the v437 dataset."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata
import torch
import torch.nn as nn
import torch.nn.functional as F

from project_paths import PATHS


VERSION_NUMBER = 438
VERSION = "v438_geometry_path_support_large_model"
CACHE = PATHS.cache_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
BS = np.asarray([50.0, 0.0], dtype=np.float32)
DELAY_BINS = 192
H_BINS = 16
DEPTH_BINS = 17


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs-coordinate", type=int, default=24)
    parser.add_argument("--epochs-full", type=int, default=36)
    parser.add_argument("--batch-size-coordinate", type=int, default=512)
    parser.add_argument("--batch-size-full", type=int, default=128)
    parser.add_argument("--learning-rate-coordinate", type=float, default=4e-4)
    parser.add_argument("--learning-rate-full", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=43_801)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    return parser.parse_args()


class TrainingData:
    def __init__(self) -> None:
        names = (
            "positions",
            "split",
            "route_raster",
            "joint_sh",
            "pdp",
            "path_delay",
            "path_depth",
            "path_count",
            "los_fraction",
            "log_power",
            "zero_channel",
        )
        self.values = {
            name: np.load(CACHE / f"{name}.npy", mmap_mode="r") for name in names
        }
        self.positions = self.values["positions"]
        self.split = self.values["split"]
        self.train_ids = np.flatnonzero(self.split == 0).astype(np.int64)
        self.validation_ids = np.flatnonzero(self.split == 1).astype(np.int64)
        self.holdout_ids = np.flatnonzero(self.split == 2).astype(np.int64)
        nonzero_train = self.train_ids[~self.values["zero_channel"][self.train_ids]]
        self.power_mean = float(np.mean(self.values["log_power"][nonzero_train]))
        self.power_std = float(np.std(self.values["log_power"][nonzero_train]))
        log_count = np.log1p(self.values["path_count"][self.train_ids].astype(np.float32))
        self.count_mean = float(np.mean(log_count))
        self.count_std = float(np.std(log_count))

    def position_features(self, ids: np.ndarray) -> np.ndarray:
        xy = np.asarray(self.positions[ids, :2], dtype=np.float32)
        delta = xy - BS[None, :]
        distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-4)
        angle = np.arctan2(delta[:, 1], delta[:, 0])
        base = np.column_stack(
            [
                (xy[:, 0] - 150.0) / 120.0,
                (xy[:, 1] + 50.0) / 160.0,
                delta[:, 0] / 250.0,
                delta[:, 1] / 350.0,
                distance / 350.0,
                np.log1p(distance) / 6.0,
                np.sin(angle),
                np.cos(angle),
            ]
        )
        harmonics = []
        for scale in (5.0, 10.0, 20.0, 40.0, 80.0, 160.0):
            phase_x = 2.0 * np.pi * xy[:, 0] / scale
            phase_y = 2.0 * np.pi * xy[:, 1] / scale
            harmonics.extend(
                [np.sin(phase_x), np.cos(phase_x), np.sin(phase_y), np.cos(phase_y)]
            )
        return np.column_stack([base, *harmonics]).astype(np.float32)

    def batch(
        self,
        ids: np.ndarray,
        device: torch.device,
        *,
        include_route: bool,
        route_ids: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        source_ids = ids if route_ids is None else route_ids
        output = {
            "position": torch.from_numpy(self.position_features(ids)).to(device),
            "joint_sh": torch.from_numpy(
                np.asarray(self.values["joint_sh"][ids], dtype=np.float32)
            ).to(device),
            "pdp": torch.from_numpy(
                np.asarray(self.values["pdp"][ids], dtype=np.float32)
            ).to(device),
            "path_delay": torch.from_numpy(
                np.asarray(self.values["path_delay"][ids], dtype=np.float32)
            ).to(device),
            "path_depth": torch.from_numpy(
                np.asarray(self.values["path_depth"][ids], dtype=np.float32)
            ).to(device),
            "zero": torch.from_numpy(
                np.asarray(self.values["zero_channel"][ids], dtype=np.float32)
            ).to(device),
            "power": torch.from_numpy(
                (
                    np.asarray(self.values["log_power"][ids], dtype=np.float32)
                    - self.power_mean
                )
                / max(self.power_std, 1e-6)
            ).to(device),
            "count": torch.from_numpy(
                (
                    np.log1p(
                        np.asarray(self.values["path_count"][ids], dtype=np.float32)
                    )
                    - self.count_mean
                )
                / max(self.count_std, 1e-6)
            ).to(device),
            "los": torch.from_numpy(
                np.asarray(self.values["los_fraction"][ids], dtype=np.float32)
            ).to(device),
        }
        if include_route:
            output["route"] = torch.from_numpy(
                np.asarray(self.values["route_raster"][source_ids], dtype=np.float32)
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


class RouteSupportTransformer(nn.Module):
    def __init__(self, position_dim: int = 32, dim: int = 512) -> None:
        super().__init__()
        self.route = nn.Sequential(
            nn.Conv2d(12, 64, 5, stride=(2, 2), padding=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            ResidualConvBlock(64),
            nn.Conv2d(64, 128, 3, stride=(2, 2), padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            ResidualConvBlock(128),
            nn.Conv2d(128, 192, 3, stride=(2, 2), padding=1, bias=False),
            nn.GroupNorm(24, 192),
            nn.GELU(),
            ResidualConvBlock(192),
            nn.Conv2d(192, dim, 3, stride=(2, 1), padding=1, bias=False),
            nn.GroupNorm(32, dim),
            nn.GELU(),
        )
        self.position = nn.Sequential(
            nn.Linear(position_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.route_position = nn.Parameter(torch.randn(1, 61, dim) * 0.01)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=16,
            dim_feedforward=4 * dim,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.delay_query = nn.Parameter(torch.randn(1, DELAY_BINS, dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=16,
            dim_feedforward=4 * dim,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.joint_head = nn.Linear(dim, H_BINS)
        self.path_delay_head = nn.Linear(dim, 1)
        self.global_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        self.depth_head = nn.Linear(dim, DEPTH_BINS)
        self.power_head = nn.Linear(dim, 1)
        self.count_head = nn.Linear(dim, 1)
        self.zero_head = nn.Linear(dim, 1)
        self.los_head = nn.Linear(dim, 1)

    def forward(self, route: torch.Tensor, position: torch.Tensor) -> dict[str, torch.Tensor]:
        feature = self.route(route)
        if feature.shape[-2:] != (10, 6):
            raise ValueError(f"Unexpected route feature map {feature.shape}")
        route_token = feature.flatten(2).transpose(1, 2)
        position_token = self.position(position)[:, None, :]
        memory = torch.cat([position_token, route_token], dim=1)
        memory = self.encoder(memory + self.route_position)
        query = self.delay_query.expand(len(route), -1, -1) + memory[:, :1, :]
        delay = self.decoder(query, memory)
        pooled = self.global_head(memory[:, 0])
        return {
            "joint_logits": self.joint_head(delay),
            "path_delay_logits": self.path_delay_head(delay).squeeze(-1),
            "depth_logits": self.depth_head(pooled),
            "power": self.power_head(pooled).squeeze(-1),
            "count": self.count_head(pooled).squeeze(-1),
            "zero_logit": self.zero_head(pooled).squeeze(-1),
            "los_logit": self.los_head(pooled).squeeze(-1),
        }


class CoordinateSpectrumNet(nn.Module):
    def __init__(self, position_dim: int = 32, hidden: int = 1024) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(position_dim, hidden), nn.LayerNorm(hidden), nn.GELU()]
        for _ in range(5):
            layers.extend(
                [
                    nn.Linear(hidden, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(0.05),
                ]
            )
        self.trunk = nn.Sequential(*layers)
        self.joint_head = nn.Linear(hidden, DELAY_BINS * H_BINS)
        self.path_delay_head = nn.Linear(hidden, DELAY_BINS)
        self.depth_head = nn.Linear(hidden, DEPTH_BINS)
        self.scalar_head = nn.Linear(hidden, 4)

    def forward(self, position: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(position)
        scalar = self.scalar_head(hidden)
        return {
            "joint_logits": self.joint_head(hidden).reshape(-1, DELAY_BINS, H_BINS),
            "path_delay_logits": self.path_delay_head(hidden),
            "depth_logits": self.depth_head(hidden),
            "power": scalar[:, 0],
            "count": scalar[:, 1],
            "zero_logit": scalar[:, 2],
            "los_logit": scalar[:, 3],
        }


def distribution_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    flat_logits = logits.flatten(1)
    flat_target = target.flatten(1)
    row = -torch.sum(flat_target * F.log_softmax(flat_logits, dim=1), dim=1)
    if torch.any(mask):
        return torch.mean(row[mask])
    return torch.zeros((), device=logits.device, dtype=logits.dtype)


def model_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    nonzero = batch["zero"] < 0.5
    joint_loss = distribution_loss(output["joint_logits"], batch["joint_sh"], nonzero)
    joint_probability = F.softmax(output["joint_logits"].flatten(1), dim=1).reshape(
        -1, DELAY_BINS, H_BINS
    )
    reverse = torch.cat(
        [
            torch.zeros(1, device=joint_probability.device, dtype=torch.long),
            torch.arange(
                DELAY_BINS - 1,
                0,
                -1,
                device=joint_probability.device,
            ),
        ]
    )
    predicted_pdp = torch.sum(joint_probability[:, reverse], dim=2)
    pdp_loss = distribution_loss(
        torch.log(torch.clamp(predicted_pdp, min=1e-12)), batch["pdp"], nonzero
    )
    path_delay_loss = distribution_loss(
        output["path_delay_logits"], batch["path_delay"], nonzero
    )
    depth_loss = distribution_loss(output["depth_logits"], batch["path_depth"], nonzero)
    zero_loss = F.binary_cross_entropy_with_logits(output["zero_logit"], batch["zero"])
    if torch.any(nonzero):
        power_loss = F.smooth_l1_loss(output["power"][nonzero], batch["power"][nonzero])
        count_loss = F.smooth_l1_loss(output["count"][nonzero], batch["count"][nonzero])
        los_loss = F.binary_cross_entropy_with_logits(
            output["los_logit"][nonzero], batch["los"][nonzero]
        )
    else:
        power_loss = count_loss = los_loss = torch.zeros((), device=joint_loss.device)
    total = (
        joint_loss
        + 0.45 * pdp_loss
        + 0.25 * path_delay_loss
        + 0.10 * depth_loss
        + 0.06 * power_loss
        + 0.06 * count_loss
        + 0.10 * zero_loss
        + 0.04 * los_loss
    )
    parts = {
        "joint": float(joint_loss.detach()),
        "pdp": float(pdp_loss.detach()),
        "path_delay": float(path_delay_loss.detach()),
        "depth": float(depth_loss.detach()),
        "power": float(power_loss.detach()),
        "count": float(count_loss.detach()),
        "zero": float(zero_loss.detach()),
        "los": float(los_loss.detach()),
    }
    return total, parts


def cosine_rows(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = prediction.reshape(len(prediction), -1).astype(np.float64)
    truth = target.reshape(len(target), -1).astype(np.float64)
    numerator = np.sum(pred * truth, axis=1)
    denominator = np.sqrt(np.sum(pred * pred, axis=1) * np.sum(truth * truth, axis=1))
    return (numerator / np.maximum(denominator, 1e-30)).astype(np.float32)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data: TrainingData,
    ids: np.ndarray,
    device: torch.device,
    batch_size: int,
    *,
    include_route: bool,
    route_ids: np.ndarray | None = None,
) -> dict[str, float]:
    model.eval()
    joint_parts = []
    pdp_parts = []
    path_parts = []
    depth_parts = []
    power_parts = []
    zero_parts = []
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        batch_ids = ids[start:end]
        batch_route_ids = None if route_ids is None else route_ids[start:end]
        batch = data.batch(
            batch_ids,
            device,
            include_route=include_route,
            route_ids=batch_route_ids,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = (
                model(batch["route"], batch["position"])
                if include_route
                else model(batch["position"])
            )
        joint = F.softmax(output["joint_logits"].flatten(1), dim=1).reshape(
            -1, DELAY_BINS, H_BINS
        )
        reverse = torch.cat(
            [
                torch.zeros(1, device=device, dtype=torch.long),
                torch.arange(DELAY_BINS - 1, 0, -1, device=device),
            ]
        )
        pdp = torch.sum(joint[:, reverse], dim=2)
        joint_parts.append(joint.float().cpu().numpy())
        pdp_parts.append(pdp.float().cpu().numpy())
        path_parts.append(F.softmax(output["path_delay_logits"], dim=1).float().cpu().numpy())
        depth_parts.append(F.softmax(output["depth_logits"], dim=1).float().cpu().numpy())
        power_parts.append(output["power"].float().cpu().numpy())
        zero_parts.append(torch.sigmoid(output["zero_logit"]).float().cpu().numpy())
    joint = np.concatenate(joint_parts)
    pdp = np.concatenate(pdp_parts)
    path_delay = np.concatenate(path_parts)
    depth = np.concatenate(depth_parts)
    power = np.concatenate(power_parts) * data.power_std + data.power_mean
    zero_probability = np.concatenate(zero_parts)
    zero = np.asarray(data.values["zero_channel"][ids], dtype=bool)
    nonzero = ~zero
    truth_joint = np.asarray(data.values["joint_sh"][ids], dtype=np.float32)
    truth_pdp = np.asarray(data.values["pdp"][ids], dtype=np.float32)
    truth_path = np.asarray(data.values["path_delay"][ids], dtype=np.float32)
    truth_depth = np.asarray(data.values["path_depth"][ids], dtype=np.float32)
    truth_power = np.asarray(data.values["log_power"][ids], dtype=np.float32)
    joint_cosine = float(np.mean(cosine_rows(joint[nonzero], truth_joint[nonzero])))
    pdp_cosine = float(np.mean(cosine_rows(pdp[nonzero], truth_pdp[nonzero])))
    path_cosine = float(np.mean(cosine_rows(path_delay[nonzero], truth_path[nonzero])))
    depth_cosine = float(np.mean(cosine_rows(depth[nonzero], truth_depth[nonzero])))
    power_mae = float(np.mean(np.abs(power[nonzero] - truth_power[nonzero])))
    power_spearman = spearman(power[nonzero], truth_power[nonzero])
    if np.any(zero) and np.any(nonzero):
        order = np.argsort(zero_probability)
        rank = np.empty(len(order), dtype=np.float64)
        rank[order] = np.arange(len(order), dtype=np.float64)
        zero_auc = float(
            (
                np.sum(rank[zero])
                - np.sum(np.arange(np.sum(zero), dtype=np.float64))
            )
            / (np.sum(zero) * np.sum(nonzero))
        )
    else:
        zero_auc = 0.5
    score = 0.60 * joint_cosine + 0.40 * pdp_cosine
    return {
        "score": score,
        "joint_sh_cosine": joint_cosine,
        "pdp_cosine": pdp_cosine,
        "path_delay_cosine": path_cosine,
        "path_depth_cosine": depth_cosine,
        "power_log_mae": power_mae,
        "power_spearman": power_spearman,
        "zero_auc": zero_auc,
        "count": int(len(ids)),
        "nonzero_count": int(np.sum(nonzero)),
    }


def train_one_model(
    *,
    name: str,
    model: nn.Module,
    data: TrainingData,
    device: torch.device,
    include_route: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    min_epochs: int,
    seed: int,
    eval_batch_size: int,
) -> tuple[nn.Module, dict[str, object]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    checkpoint = CHECKPOINT_DIR / f"{name}_best.pt"
    history = []
    best_score = -np.inf
    best_epoch = 0
    stale = 0
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(data.train_ids)
        epoch_loss = 0.0
        part_sum: dict[str, float] = {}
        seen = 0
        epoch_started = time.time()
        for start in range(0, len(order), batch_size):
            ids = order[start : start + batch_size]
            batch = data.batch(ids, device, include_route=include_route)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = (
                    model(batch["route"], batch["position"])
                    if include_route
                    else model(batch["position"])
                )
                loss, parts = model_loss(output, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(ids)
            seen += len(ids)
            for key, value in parts.items():
                part_sum[key] = part_sum.get(key, 0.0) + value * len(ids)
        validation = evaluate_model(
            model,
            data,
            data.validation_ids,
            device,
            eval_batch_size,
            include_route=include_route,
        )
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / seen,
            "train_parts": {key: value / seen for key, value in part_sum.items()},
            "validation": validation,
            "seconds": float(time.time() - epoch_started),
        }
        history.append(row)
        print(
            f"v438 {name} epoch={epoch} loss={row['train_loss']:.5f} "
            f"val={validation['score']:.6f} joint={validation['joint_sh_cosine']:.6f} "
            f"pdp={validation['pdp_cosine']:.6f} seconds={row['seconds']:.1f}",
            flush=True,
        )
        if validation["score"] > best_score + 1e-5:
            best_score = validation["score"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation": validation,
                    "power_mean": data.power_mean,
                    "power_std": data.power_std,
                    "count_mean": data.count_mean,
                    "count_std": data.count_std,
                },
                checkpoint,
            )
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            break
    best = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(best["model"])
    return model, {
        "name": name,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_epoch": int(best_epoch),
        "best_validation": best["validation"],
        "seconds": float(time.time() - started),
        "checkpoint": PATHS.relative(checkpoint),
        "history": history,
    }


def knn_metrics(data: TrainingData, query_ids: np.ndarray, k: int = 8) -> dict[str, float]:
    train_xy = np.asarray(data.positions[data.train_ids, :2], dtype=np.float32)
    query_xy = np.asarray(data.positions[query_ids, :2], dtype=np.float32)
    tree = cKDTree(train_xy)
    distance, local = tree.query(query_xy, k=k, workers=-1)
    distance = np.maximum(np.asarray(distance), 1e-3)
    neighbor = data.train_ids[np.asarray(local)]
    weight = 1.0 / distance**2
    weight /= np.sum(weight, axis=1, keepdims=True)
    joint_prediction = []
    pdp_prediction = []
    for start in range(0, len(query_ids), 256):
        end = min(start + 256, len(query_ids))
        w = weight[start:end]
        n = neighbor[start:end]
        joint_prediction.append(
            np.sum(
                np.asarray(data.values["joint_sh"][n], dtype=np.float32)
                * w[:, :, None, None],
                axis=1,
            )
        )
        pdp_prediction.append(
            np.sum(
                np.asarray(data.values["pdp"][n], dtype=np.float32)
                * w[:, :, None],
                axis=1,
            )
        )
    joint = np.concatenate(joint_prediction)
    pdp = np.concatenate(pdp_prediction)
    zero = np.asarray(data.values["zero_channel"][query_ids], dtype=bool)
    nonzero = ~zero
    joint_cosine = float(
        np.mean(
            cosine_rows(
                joint[nonzero],
                np.asarray(data.values["joint_sh"][query_ids[nonzero]], dtype=np.float32),
            )
        )
    )
    pdp_cosine = float(
        np.mean(
            cosine_rows(
                pdp[nonzero],
                np.asarray(data.values["pdp"][query_ids[nonzero]], dtype=np.float32),
            )
        )
    )
    return {
        "k": k,
        "score": 0.60 * joint_cosine + 0.40 * pdp_cosine,
        "joint_sh_cosine": joint_cosine,
        "pdp_cosine": pdp_cosine,
        "count": int(len(query_ids)),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v438 requires CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"v438 requires exactly one visible GPU, got {torch.cuda.device_count()}")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = TrainingData()
    baseline_started = time.time()
    knn_validation = knn_metrics(data, data.validation_ids)
    knn_seconds = time.time() - baseline_started

    torch.manual_seed(args.seed)
    coordinate, coordinate_report = train_one_model(
        name="coordinate",
        model=CoordinateSpectrumNet(),
        data=data,
        device=device,
        include_route=False,
        epochs=args.epochs_coordinate,
        batch_size=args.batch_size_coordinate,
        learning_rate=args.learning_rate_coordinate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_epochs=args.min_epochs,
        seed=args.seed,
        eval_batch_size=args.eval_batch_size,
    )
    torch.manual_seed(args.seed + 100)
    full, full_report = train_one_model(
        name="route_transformer",
        model=RouteSupportTransformer(),
        data=data,
        device=device,
        include_route=True,
        epochs=args.epochs_full,
        batch_size=args.batch_size_full,
        learning_rate=args.learning_rate_full,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_epochs=args.min_epochs,
        seed=args.seed + 100,
        eval_batch_size=args.eval_batch_size,
    )

    holdout_knn = knn_metrics(data, data.holdout_ids)
    holdout_coordinate = evaluate_model(
        coordinate,
        data,
        data.holdout_ids,
        device,
        args.eval_batch_size,
        include_route=False,
    )
    holdout_full = evaluate_model(
        full,
        data,
        data.holdout_ids,
        device,
        args.eval_batch_size,
        include_route=True,
    )
    rng = np.random.default_rng(args.seed + 999)
    shuffled_route_ids = rng.permutation(data.holdout_ids)
    holdout_shuffled = evaluate_model(
        full,
        data,
        data.holdout_ids,
        device,
        args.eval_batch_size,
        include_route=True,
        route_ids=shuffled_route_ids,
    )
    report = {
        "method": VERSION,
        "status": "synthetic_spatial_holdout_evaluated",
        "primary_hypothesis": (
            "A global BS-UE point-cloud route encoder plus coordinate-conditioned delay decoder "
            "learns path birth/death and joint delay-angle support beyond coordinate-only and KNN baselines."
        ),
        "data": {
            "source": "v437 random-position Sionna labels plus official point-cloud route raster",
            "train_count": int(len(data.train_ids)),
            "validation_count": int(len(data.validation_ids)),
            "holdout_count": int(len(data.holdout_ids)),
            "holdout_read_policy": "only after validation-selected coordinate and full checkpoints",
        },
        "single_gpu": {
            "logical_device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
        },
        "parameters": vars(args),
        "normalization": {
            "power_mean": data.power_mean,
            "power_std": data.power_std,
            "count_mean": data.count_mean,
            "count_std": data.count_std,
        },
        "knn_validation": knn_validation,
        "knn_seconds": knn_seconds,
        "coordinate": coordinate_report,
        "route_transformer": full_report,
        "holdout": {
            "knn": holdout_knn,
            "coordinate": holdout_coordinate,
            "route_transformer": holdout_full,
            "route_transformer_shuffled_geometry": holdout_shuffled,
            "full_minus_coordinate": {
                key: float(holdout_full[key] - holdout_coordinate[key])
                for key in ("score", "joint_sh_cosine", "pdp_cosine")
            },
            "full_minus_knn": {
                key: float(holdout_full[key] - holdout_knn[key])
                for key in ("score", "joint_sh_cosine", "pdp_cosine")
            },
            "full_minus_shuffled": {
                key: float(holdout_full[key] - holdout_shuffled[key])
                for key in ("score", "joint_sh_cosine", "pdp_cosine")
            },
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["holdout"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
