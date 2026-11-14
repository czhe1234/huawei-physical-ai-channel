"""Frozen runtime primitives for path link model."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from project_paths import PATHS
from .path_link_core import (
    LinkGate,
    spectral_features,
)

ARTIFACT_DIR = PATHS.artifacts / "round2" / "full_path_pdp_link_gate"


def checkpoint_path(fold: int) -> Path:
    return ARTIFACT_DIR / f"fold{fold}_full_link_gate.pt"


@torch.inference_mode()
def predict_alpha(
    model: LinkGate,
    point_ids: np.ndarray,
    base_pdp: np.ndarray,
    raw_pdp: np.ndarray,
    context: np.ndarray,
    context_mean: np.ndarray,
    context_std: np.ndarray,
    link_tensor: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output = np.zeros((len(point_ids), 1024), dtype=np.float32)
    model.eval()
    for start in range(0, len(point_ids), batch_size):
        end = min(start + batch_size, len(point_ids))
        ids = point_ids[start:end]
        base = torch.as_tensor(
            np.asarray(base_pdp[ids], dtype=np.float32).reshape(len(ids), 1024, 192),
            device=device,
        )
        raw = torch.as_tensor(
            np.asarray(raw_pdp[ids], dtype=np.float32).reshape(len(ids), 1024, 192),
            device=device,
        )
        local_context = torch.as_tensor(
            (np.asarray(context[ids]) - context_mean) / context_std, device=device
        )
        features = spectral_features(base, raw, local_context, link_tensor)
        output[start:end] = model(features).cpu().numpy()
    return output


def load_model(fold: int, device: torch.device) -> tuple[LinkGate, dict[str, object]]:
    checkpoint = torch.load(
        checkpoint_path(fold), map_location=device, weights_only=False
    )
    config = checkpoint["config"]
    model = LinkGate(
        int(checkpoint["input_dim"]), int(config["hidden"]), float(config["dropout"])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return (model, checkpoint)
