#!/usr/bin/env python3
"""Audit whether official-coordinate Sionna path tokens improve joint S-H prediction."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

import train_v469_neural_sionna_calibrator as v469
from project_paths import PATHS


VERSION_NUMBER = 476
VERSION = "v476_path_conditioned_joint_audit"
CACHE = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
REPORT = REPORT_DIR / f"{VERSION}.json"
PATH_FEATURES = CACHE / "v476_path_token_features.npy"
PATH_VALID = CACHE / "v476_path_token_valid.npy"
CHECKPOINT = CHECKPOINT_DIR / "path_joint_best.pt"
MAX_PATHS = 96
WIDTH = 128
PATH_LATENT = 128
BS_POSITION = np.asarray([50.0, 0.0, 25.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def require(path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def gather_last(vertices: np.ndarray, count: np.ndarray) -> np.ndarray:
    index = np.maximum(count - 1, 0)
    row = np.arange(vertices.shape[0])[:, None]
    path = np.arange(vertices.shape[1])[None, :]
    return vertices[row, path, index]


def shard_path_features(
    path_file,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path_file)
    valid = data["valid"][:, :MAX_PATHS].astype(bool)
    tau = data["tau"][:, :MAX_PATHS].astype(np.float32)
    angles = [
        data[name][:, :MAX_PATHS].astype(np.float32)
        for name in ("theta_t", "phi_t", "theta_r", "phi_r")
    ]
    log_power = data["log_power"][:, :MAX_PATHS].astype(np.float32)
    normalized_power = data["normalized_power"][:, :MAX_PATHS].astype(np.float32)
    interaction_count = data["interaction_count"][:, :MAX_PATHS].astype(np.int64)
    interaction_mask = data["interaction_mask"][:, :MAX_PATHS].astype(np.int64)
    interactions = data["interactions"][:, :MAX_PATHS].astype(np.int8)
    vertices = data["vertices"][:, :MAX_PATHS].astype(np.float32)
    fingerprints = [
        data[name][:, :MAX_PATHS].astype(np.float32)
        for name in (
            "rx_fingerprint",
            "tx_pol_fingerprint",
            "tx_h_fingerprint",
            "tx_v_fingerprint",
        )
    ]

    count = interaction_count
    step_valid = (interactions != 0) & np.isfinite(vertices).all(axis=-1)
    safe_vertices = np.where(step_valid[..., None], vertices, 0.0)
    first = safe_vertices[:, :, 0]
    last = gather_last(safe_vertices, count)
    has_interaction = (count > 0)[..., None]
    first = np.where(has_interaction, first, 0.0)
    last = np.where(has_interaction, last, 0.0)
    ue = positions[:, None, :]

    direct = np.linalg.norm(ue - BS_POSITION[None, None, :], axis=-1)
    path_length = direct.copy()
    active = count > 0
    if np.any(active):
        length = np.linalg.norm(first - BS_POSITION[None, None, :], axis=-1)
        for depth in range(1, interactions.shape[2]):
            use = step_valid[:, :, depth]
            segment = np.linalg.norm(
                safe_vertices[:, :, depth] - safe_vertices[:, :, depth - 1],
                axis=-1,
            )
            length += np.where(use, segment, 0.0)
        length += np.linalg.norm(ue - last, axis=-1)
        path_length = np.where(active, length, direct)

    vertex_denominator = np.maximum(count[..., None], 1)
    vertex_mean = np.sum(safe_vertices, axis=2) / vertex_denominator
    centered = np.where(
        step_valid[..., None],
        safe_vertices - vertex_mean[:, :, None, :],
        0.0,
    )
    vertex_std = np.sqrt(
        np.sum(centered * centered, axis=2) / vertex_denominator
    )
    max_log_power = np.max(np.where(valid, log_power, -1e9), axis=1, keepdims=True)
    relative_log_power = np.where(valid, log_power - max_log_power, 0.0)

    feature_parts = [
        (tau * (192.0 * 30_000.0))[..., None],
        *[function(value)[..., None] for value in angles for function in (np.sin, np.cos)],
        relative_log_power[..., None],
        np.log(np.maximum(normalized_power, 1e-8))[..., None],
        normalized_power[..., None],
        (count.astype(np.float32) / 16.0)[..., None],
        *[
            (((interaction_mask & bit) != 0).astype(np.float32))[..., None]
            for bit in (1, 2, 4, 8)
        ],
        *[
            (np.sum(interactions == kind, axis=2).astype(np.float32) / 16.0)[..., None]
            for kind in (1, 2, 4, 8)
        ],
        ((path_length - direct) / np.maximum(direct, 1.0))[..., None],
        (path_length / 500.0)[..., None],
        ((first - ue) / 200.0),
        ((last - ue) / 200.0),
        ((vertex_mean - ue) / 200.0),
        (vertex_std / 100.0),
        *fingerprints,
    ]
    features = np.concatenate(feature_parts, axis=-1).astype(np.float32)
    features[~np.isfinite(features)] = 0.0
    features *= valid[..., None]
    return features, valid


def build_path_cache() -> tuple[np.ndarray, np.ndarray]:
    if PATH_FEATURES.is_file() and PATH_VALID.is_file():
        return np.load(PATH_FEATURES, mmap_mode="r"), np.load(PATH_VALID, mmap_mode="r")
    source = PATHS.cache_dir(475)
    positions = np.load(source / "positions.npz")["positions"].astype(np.float32)
    feature_rows = []
    valid_rows = []
    offset = 0
    for shard in sorted((source / "shards").glob("shard_*/path_targets.npz")):
        metadata = np.load(shard.parent / "metadata.npz")
        ids = metadata["position_id"].astype(np.int64)
        expected = np.arange(offset, offset + len(ids), dtype=np.int64)
        if not np.array_equal(ids, expected):
            raise ValueError(f"Unexpected position order in {shard}")
        features, valid = shard_path_features(shard, positions[ids])
        feature_rows.append(features)
        valid_rows.append(valid)
        offset += len(ids)
    if offset != 2500:
        raise ValueError(f"Expected 2500 path rows, found {offset}")
    features = np.concatenate(feature_rows, axis=0)
    valid = np.concatenate(valid_rows, axis=0)
    train_valid = valid[:2000]
    selected = features[:2000][train_valid]
    mean = selected.mean(axis=0, keepdims=True)
    std = np.maximum(selected.std(axis=0, keepdims=True), 1e-4)
    features = ((features - mean) / std).astype(np.float32)
    features *= valid[..., None]
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(PATH_FEATURES, features)
    np.save(PATH_VALID, valid)
    return np.load(PATH_FEATURES, mmap_mode="r"), np.load(PATH_VALID, mmap_mode="r")


class PathEncoder(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(feature_dim, WIDTH),
            nn.LayerNorm(WIDTH),
            nn.GELU(),
            nn.Linear(WIDTH, WIDTH),
        )
        self.summary = nn.Parameter(torch.randn(1, 4, WIDTH) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=WIDTH,
            nhead=8,
            dim_feedforward=4 * WIDTH,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=4)
        self.output = nn.Sequential(
            nn.Linear(4 * WIDTH, 2 * PATH_LATENT),
            nn.LayerNorm(2 * PATH_LATENT),
            nn.GELU(),
            nn.Linear(2 * PATH_LATENT, PATH_LATENT),
        )

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        token = self.input(features)
        summary = self.summary.expand(len(features), -1, -1)
        memory = torch.cat([summary, token], dim=1)
        summary_valid = torch.ones(
            len(features), 4, dtype=torch.bool, device=features.device
        )
        memory_valid = torch.cat([summary_valid, valid], dim=1)
        memory = self.transformer(memory, src_key_padding_mask=~memory_valid)
        return self.output(memory[:, :4].flatten(1))


class PathJointModel(nn.Module):
    def __init__(self, point_dim: int, path_dim: int) -> None:
        super().__init__()
        self.path = PathEncoder(path_dim)
        self.calibrator = v469.JointCalibrator(
            point_dim + PATH_LATENT,
            width=96,
            blocks=8,
        )

    def forward(
        self,
        raw: torch.Tensor,
        base: torch.Tensor,
        quality: torch.Tensor,
        point: torch.Tensor,
        path: torch.Tensor,
        valid: torch.Tensor,
        *,
        path_mode: str = "normal",
    ) -> torch.Tensor:
        raw = raw.reshape(len(raw), 192, 16)
        base = base.reshape(len(base), 192, 16)
        quality = quality.reshape(len(quality), 192, 16)
        latent = self.path(path, valid)
        if path_mode == "zero":
            latent = torch.zeros_like(latent)
        elif path_mode != "normal":
            raise ValueError(path_mode)
        return self.calibrator(raw, base, quality, torch.cat([point, latent], dim=1))


def cosine(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(
        prediction.flatten(1), target.flatten(1), dim=1
    )


@dataclass
class Evaluation:
    cosine: float
    loss: float


@torch.no_grad()
def evaluate(
    model: PathJointModel,
    ids: np.ndarray,
    data: dict[str, np.ndarray],
    point_features: np.ndarray,
    path_features: np.ndarray,
    path_valid: np.ndarray,
    device: torch.device,
    batch_size: int,
    *,
    mode: str,
    geometry_ids: np.ndarray | None = None,
) -> Evaluation:
    model.eval()
    values = []
    losses = []
    source_ids = ids if geometry_ids is None else geometry_ids
    for start in range(0, len(ids), batch_size):
        stop = min(start + batch_size, len(ids))
        row = ids[start:stop]
        geometry = source_ids[start:stop]
        prediction = model(
            torch.from_numpy(data["train_raw"][row]).to(device),
            torch.from_numpy(data["train_base"][row]).to(device),
            torch.from_numpy(data["train_cell_quality"][row]).to(device),
            torch.from_numpy(point_features[row]).to(device),
            torch.from_numpy(np.asarray(path_features[geometry])).to(device),
            torch.from_numpy(np.asarray(path_valid[geometry])).to(device),
            path_mode=mode,
        )
        target = torch.from_numpy(data["train_target"][row]).to(device).reshape(-1, 192, 16)
        values.extend(cosine(prediction, target).cpu().numpy().astype(float).tolist())
        losses.append(float(v469.spectral_loss(prediction, target).cpu()))
    return Evaluation(cosine=float(np.mean(values)), loss=float(np.mean(losses)))


def train(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    data = v469.build_joint_dataset()
    point_features = np.load(v469.POINT_FEATURE_CACHE).astype(np.float32)[:2000]
    path_features, path_valid = build_path_cache()
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(2000)
    fit_ids = order[:1500]
    validation_ids = order[1500:]
    model = PathJointModel(point_features.shape[1], path_features.shape[2]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    epochs = 2 if args.smoke else args.epochs
    patience = 2 if args.smoke else args.patience
    history = []
    best = -math.inf
    best_epoch = -1
    stale = 0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = rng.permutation(fit_ids)
        losses = []
        for start in range(0, len(shuffled), args.batch_size):
            row = shuffled[start : start + args.batch_size]
            prediction = model(
                torch.from_numpy(data["train_raw"][row]).to(device),
                torch.from_numpy(data["train_base"][row]).to(device),
                torch.from_numpy(data["train_cell_quality"][row]).to(device),
                torch.from_numpy(point_features[row]).to(device),
                torch.from_numpy(np.asarray(path_features[row])).to(device),
                torch.from_numpy(np.asarray(path_valid[row])).to(device),
            )
            target = torch.from_numpy(data["train_target"][row]).to(device).reshape(-1, 192, 16)
            loss = v469.spectral_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        normal = evaluate(
            model,
            validation_ids,
            data,
            point_features,
            path_features,
            path_valid,
            device,
            args.batch_size,
            mode="normal",
        )
        zero = evaluate(
            model,
            validation_ids,
            data,
            point_features,
            path_features,
            path_valid,
            device,
            args.batch_size,
            mode="zero",
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_normal": asdict(normal),
            "validation_zero": asdict(zero),
            "path_gain": normal.cosine - zero.cosine,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if normal.cosine > best + 1e-6:
            best = normal.cosine
            best_epoch = epoch
            stale = 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, CHECKPOINT)
        else:
            stale += 1
        if stale >= patience:
            break

    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    normal = evaluate(
        model, validation_ids, data, point_features, path_features, path_valid,
        device, args.batch_size, mode="normal"
    )
    zero = evaluate(
        model, validation_ids, data, point_features, path_features, path_valid,
        device, args.batch_size, mode="zero"
    )
    shuffled_results = []
    for _ in range(5):
        geometry_ids = rng.permutation(validation_ids)
        shuffled_results.append(
            evaluate(
                model, validation_ids, data, point_features, path_features, path_valid,
                device, args.batch_size, mode="normal", geometry_ids=geometry_ids
            )
        )
    target_flat = data["train_target"][validation_ids].reshape(len(validation_ids), -1)
    raw_flat = data["train_raw"][validation_ids].reshape(len(validation_ids), -1)
    base_flat = data["train_base"][validation_ids].reshape(len(validation_ids), -1)
    raw_cosine = np.mean(
        np.sum(raw_flat * target_flat, axis=1)
        / np.maximum(
            np.linalg.norm(raw_flat, axis=1) * np.linalg.norm(target_flat, axis=1),
            1e-12,
        )
    )
    base_cosine = np.mean(
        np.sum(base_flat * target_flat, axis=1)
        / np.maximum(
            np.linalg.norm(base_flat, axis=1) * np.linalg.norm(target_flat, axis=1),
            1e-12,
        )
    )
    shuffled_cosine = float(np.mean([item.cosine for item in shuffled_results]))
    decision = (
        normal.cosine - zero.cosine >= 0.003
        and normal.cosine - shuffled_cosine >= 0.003
    )
    return {
        "method": VERSION,
        "status": "path_signal_passed" if decision else "path_signal_rejected",
        "primary_hypothesis": (
            "Per-path delay, angle, interaction, vertex, power, and array fingerprints "
            "contain joint S-H information beyond aggregated Sionna spectra and position."
        ),
        "split": {
            "seed": args.seed,
            "fit_count": len(fit_ids),
            "validation_count": len(validation_ids),
        },
        "parameters": vars(args),
        "path_cache": {
            "feature_shape": list(path_features.shape),
            "valid_shape": list(path_valid.shape),
            "mean_paths": float(np.mean(np.sum(path_valid, axis=1))),
            "zero_path_points": int(np.sum(np.sum(path_valid, axis=1) == 0)),
        },
        "training": {
            "best_epoch": best_epoch,
            "history": history,
        },
        "validation": {
            "raw_sionna_cosine": float(raw_cosine),
            "base_cosine": float(base_cosine),
            "normal": asdict(normal),
            "zero_path": asdict(zero),
            "shuffled_path_mean": shuffled_cosine,
            "normal_minus_zero": normal.cosine - zero.cosine,
            "normal_minus_shuffled": normal.cosine - shuffled_cosine,
        },
        "decision_rule": {
            "minimum_normal_minus_zero": 0.003,
            "minimum_normal_minus_shuffled": 0.003,
            "passed": decision,
        },
        "outputs": {
            "path_features": PATHS.relative(PATH_FEATURES),
            "path_valid": PATHS.relative(PATH_VALID),
            "checkpoint": PATHS.relative(CHECKPOINT),
            "report": PATHS.relative(REPORT),
        },
    }


def main() -> None:
    args = parse_args()
    require(PATHS.cache_dir(475) / "positions.npz")
    CACHE.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = train(args)
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["validation"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
