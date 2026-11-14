"""Path-state calibration network and five-fold production ensemble."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import PipelineConfig, RuntimePaths
from .data import load_official_data
from .features import load_model_inputs
from .io_utils import atomic_json, atomic_npz, portable_path


@dataclass(frozen=True)
class ModelConfig:
    hidden: int = 96
    heads: int = 4
    sequence_layers: int = 2
    set_layers: int = 2
    feedforward: int = 256
    dropout: float = 0.08
    score_limit: float = 4.0
    port_limit: float = 2.0
    sequence_hash_buckets: int = 32749

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any]) -> "ModelConfig":
        names = {field.name for field in fields(cls)}
        return cls(**{name: payload[name] for name in names if name in payload})


class PathStateModel(nn.Module):
    def __init__(self, feature_dim: int, sequence_dim: int, config: ModelConfig):
        super().__init__()
        hidden = config.hidden
        self.config = config
        self.base_input = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        self.sequence_input = nn.Linear(sequence_dim, hidden)
        self.interaction_embedding = nn.Embedding(8, hidden)
        self.object_embedding = nn.Embedding(16, hidden)
        self.primitive_embedding = nn.Embedding(24038, hidden)
        self.step_embedding = nn.Embedding(16, hidden)
        self.hash_embedding_a = nn.Embedding(config.sequence_hash_buckets, hidden)
        self.hash_embedding_b = nn.Embedding(config.sequence_hash_buckets, hidden)
        sequence_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.heads,
            dim_feedforward=config.feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(
            sequence_layer, num_layers=config.sequence_layers
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        set_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.heads,
            dim_feedforward=config.feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            set_layer, num_layers=config.set_layers
        )
        self.score_head = nn.Linear(hidden, 1)
        self.coordinate_head = nn.Linear(hidden, 3)
        self.port_head = nn.Linear(hidden, 8)

    def sequence_state(
        self,
        continuous: torch.Tensor,
        interactions: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        batch, paths, steps, _ = continuous.shape
        flat = batch * paths
        step_ids = torch.arange(steps, device=continuous.device)
        hidden = (
            self.sequence_input(continuous)
            + self.interaction_embedding(interactions.clamp(0, 7))
            + self.step_embedding(step_ids)[None, None]
        )
        hidden = hidden.reshape(flat, steps, self.config.hidden)
        token_mask = step_ids[None] < depth.reshape(flat, 1).clamp_min(1)
        hidden = self.sequence_encoder(hidden, src_key_padding_mask=~token_mask)
        weight = token_mask.to(hidden.dtype)
        pooled = (hidden * weight[..., None]).sum(dim=1) / weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        return pooled.reshape(batch, paths, self.config.hidden)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        base = self.base_input(batch["base"])
        sequence = self.sequence_state(
            batch["sequence_geometry"], batch["interactions"], batch["depth"]
        )
        hidden = self.fusion(torch.cat((base, sequence), dim=-1))
        hidden = self.set_encoder(hidden)
        score_residual = self.config.score_limit * torch.tanh(
            self.score_head(hidden)[..., 0]
        )
        coordinate_residual = torch.tanh(self.coordinate_head(hidden)) * torch.tensor(
            (2.0, 4.0, 2.0), device=hidden.device
        )
        port_residual = self.config.port_limit * torch.tanh(self.port_head(hidden))
        mass = torch.softmax(batch["base_log_mass"] + score_residual, dim=1)
        port = torch.softmax(batch["base_log_port"] + port_residual, dim=2).reshape(
            len(hidden), hidden.shape[1], 2, 4
        )
        return {
            "mass": mass,
            "port": port,
            "coordinate": batch["coordinate"] + coordinate_residual,
        }


def _decode_mapping(latent: torch.Tensor) -> dict[str, torch.Tensor]:
    bounded = torch.tanh(latent)
    yaw = (
        torch.deg2rad(torch.tensor((-44.0, 73.5), device=latent.device))
        + math.radians(30.0) * bounded[0:2]
    )
    tilt = (
        torch.deg2rad(torch.tensor((-11.5, -10.5), device=latent.device))
        + math.radians(18.0) * bounded[2:4]
    )
    return {
        "yaw": yaw,
        "tilt": tilt,
        "horizontal_spacing": 0.50 + 0.18 * bounded[4],
        "vertical_spacing": 4.00 + 1.20 * bounded[5],
        "horizontal_offset": 4.0 * bounded[6:8],
        "vertical_offset": 2.0 + 4.0 * bounded[8:10],
    }


@torch.inference_mode()
def _mapped_coordinates(
    auxiliary: dict[str, np.ndarray],
    sites: np.ndarray,
    latent: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    parameters = _decode_mapping(
        torch.as_tensor(latent, dtype=torch.float32, device=device)
    )
    site = torch.as_tensor(sites.astype(np.int64), device=device)
    angles = np.array(auxiliary["angles"], dtype=np.float32, copy=True)
    theta = torch.as_tensor(angles[..., 0], device=device)
    phi = torch.as_tensor(angles[..., 1], device=device)
    yaw = parameters["yaw"][site][:, None]
    tilt = parameters["tilt"][site][:, None]
    horizontal = (
        16.0
        * parameters["horizontal_spacing"]
        * torch.sin(theta)
        * torch.sin(phi - yaw)
        + parameters["horizontal_offset"][site][:, None]
    )
    vertical = (
        8.0
        * parameters["vertical_spacing"]
        * (
            torch.sin(tilt) * torch.sin(theta) * torch.cos(phi - yaw)
            - torch.cos(tilt) * torch.cos(theta)
        )
        + parameters["vertical_offset"][site][:, None]
    )
    coordinate = np.array(auxiliary["coordinate"], dtype=np.float32, copy=True)
    coordinate[..., 0] = torch.remainder(horizontal, 16.0).cpu().numpy()
    coordinate[..., 1] = torch.remainder(vertical, 8.0).cpu().numpy()
    return coordinate


def _batch(
    rows: np.ndarray,
    features: np.ndarray,
    auxiliary: dict[str, np.ndarray],
    coordinate: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    seed_count: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    fraction = auxiliary["fraction"][rows]
    presence = auxiliary["presence"][rows]
    anchored = np.maximum(fraction, 1e-12) * np.sqrt(
        np.maximum(presence, 1.0) / float(seed_count)
    )
    port = auxiliary["port"][rows].reshape(len(rows), fraction.shape[1], 8)
    return {
        "base": torch.as_tensor(
            (features[rows].astype(np.float32) - mean) / std,
            dtype=torch.float32,
            device=device,
        ),
        "base_log_mass": torch.as_tensor(
            np.log(np.maximum(anchored, 1e-12)),
            dtype=torch.float32,
            device=device,
        ),
        "coordinate": torch.as_tensor(
            coordinate[rows], dtype=torch.float32, device=device
        ),
        "base_log_port": torch.as_tensor(
            np.log(np.maximum(port, 1e-8)),
            dtype=torch.float32,
            device=device,
        ),
        "sequence_geometry": torch.as_tensor(
            auxiliary["geometry"][rows], dtype=torch.float32, device=device
        ),
        "depth": torch.as_tensor(auxiliary["depth"][rows], device=device),
        "interactions": torch.as_tensor(auxiliary["interactions"][rows], device=device),
    }


def _circular_mean(
    values: np.ndarray, weights: np.ndarray, periods: np.ndarray
) -> np.ndarray:
    phase = values * (2.0 * np.pi / periods)[None, None, None]
    expanded = weights[..., None]
    sine = np.sum(expanded * np.sin(phase), axis=0, dtype=np.float64)
    cosine = np.sum(expanded * np.cos(phase), axis=0, dtype=np.float64)
    angle = np.mod(np.arctan2(sine, cosine), 2.0 * np.pi)
    return (angle * periods / (2.0 * np.pi)).astype(np.float32)


def checkpoint_path(model_dir: Path, fold: int) -> Path:
    return model_dir / "path_calibrator" / f"fold_{fold}.pt"


@torch.inference_mode()
def predict_path_states(
    paths: RuntimePaths,
    pipeline_config: PipelineConfig,
    split: str,
    device: torch.device,
    batch_size: int | None = None,
) -> dict[str, Any]:
    started = time.time()
    torch.set_float32_matmul_precision("high")
    features, auxiliary = load_model_inputs(paths, split)
    data = load_official_data(paths.data_dir)
    sites = data.train_sites if split == "train" else data.test_sites
    rows = np.arange(len(features), dtype=np.int64)
    batch_size = batch_size or pipeline_config.inference_batch_size
    outputs: list[dict[str, np.ndarray]] = []
    fold_reports: list[dict[str, Any]] = []
    for fold in range(5):
        path = checkpoint_path(paths.model_dir, fold)
        if not path.is_file():
            raise FileNotFoundError(f"missing path calibrator checkpoint: {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = ModelConfig.from_checkpoint(checkpoint["config"])
        model = PathStateModel(
            features.shape[-1], auxiliary["geometry"].shape[-1], config
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        coordinate = _mapped_coordinates(
            auxiliary,
            sites,
            np.asarray(checkpoint["mapping_latent"], dtype=np.float32),
            device,
        )
        mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
        local: list[dict[str, torch.Tensor]] = []
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            local.append(
                model(
                    _batch(
                        batch_rows,
                        features,
                        auxiliary,
                        coordinate,
                        mean,
                        std,
                        pipeline_config.seed_count,
                        device,
                    )
                )
            )
        outputs.append(
            {
                name: np.concatenate(
                    [block[name].detach().cpu().numpy() for block in local], axis=0
                )
                for name in ("mass", "port", "coordinate")
            }
        )
        fold_reports.append(
            {
                "fold": fold,
                "checkpoint": portable_path(path),
                "best_epoch": int(checkpoint.get("best_epoch", -1)),
            }
        )
        del model, checkpoint, local
        if device.type == "cuda":
            torch.cuda.empty_cache()
    periods = np.asarray((16.0, 8.0, 192.0), dtype=np.float32)
    mass_stack = np.stack([value["mass"] for value in outputs], axis=0)
    port_stack = np.stack([value["port"] for value in outputs], axis=0)
    coordinate_stack = np.stack(
        [np.mod(value["coordinate"], periods) for value in outputs], axis=0
    )
    mass = mass_stack.mean(axis=0, dtype=np.float64)
    mass /= np.maximum(mass.sum(axis=1, keepdims=True), 1e-30)
    port = port_stack.mean(axis=0, dtype=np.float64)
    port /= np.maximum(port.sum(axis=(2, 3), keepdims=True), 1e-30)
    weights = mass_stack / np.maximum(mass_stack.sum(axis=0, keepdims=True), 1e-30)
    coordinate = _circular_mean(coordinate_stack, weights, periods)
    output_path = paths.work_dir / "path_state" / f"{split}.npz"
    atomic_npz(
        output_path,
        mass=mass.astype(np.float32),
        port=port.astype(np.float32),
        coordinate=coordinate,
    )
    report = {
        "stage": "path_state_inference",
        "status": "complete",
        "split": split,
        "folds": fold_reports,
        "mass_shape": list(mass.shape),
        "port_shape": list(port.shape),
        "coordinate_shape": list(coordinate.shape),
        "finite": bool(
            np.isfinite(mass).all()
            and np.isfinite(port).all()
            and np.isfinite(coordinate).all()
        ),
        "wall_seconds": float(time.time() - started),
        "device": str(device),
        "peak_torch_memory_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
        "measured_bottleneck": "five-fold transformer inference",
    }
    atomic_json(paths.report_dir / f"path_state_{split}.json", report)
    return report
