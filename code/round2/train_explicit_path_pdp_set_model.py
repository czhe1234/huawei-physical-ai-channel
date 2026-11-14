#!/usr/bin/env python3
"""Train a cross-fitted set model that directly optimizes explicit-path PDP."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from analyze_joint_path_token_oracle import FREQUENCIES, OOF_META, REGIME_CACHE, v9_channel
from analyze_joint_path_token_pdp_isolation import evaluate_grid
from analyze_v4_exact_joint_projection import pdp_distribution
from prepare_sionna_calibration_dataset import TRUTH_PDP
from project_paths import PATHS
from run_rbf_spectral_field import statistics


VERSION = "Round2-E63-explicit-path-pdp-set-model"
PATH_CACHE = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_oracle"
    / "n469_s24_k128_b096_d064_scatter1_sel106"
)
FEATURES = (
    PATHS.artifacts
    / "round2"
    / "explicit_path_pdp_predictor"
    / "path_features_n469.npy"
)
ORACLE_METRICS = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_pdp_isolation"
    / "pdp_isolation_metrics_n469.npz"
)
ARTIFACT_DIR = PATHS.artifacts / "round2" / "explicit_path_pdp_set_model"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_explicit_path_pdp_set_model"
REPORT = ANALYSIS_DIR / "explicit_path_pdp_set_model.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--regularization", type=float, default=2e-3)
    parser.add_argument("--links-per-step", type=int, default=96)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--log-amplitude-limit", type=float, default=2.5)
    parser.add_argument("--random-state", type=int, default=106)
    return parser.parse_args()


class PathEnergySetModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        heads: int,
        layers: int,
        feedforward: int,
        dropout: float,
        limit: float,
    ) -> None:
        super().__init__()
        self.limit = limit
        self.input = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.input(features)
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        valid = mask.to(hidden.dtype)[..., None]
        context = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        context = context[:, None].expand_as(hidden)
        raw = self.limit * torch.tanh(self.head(torch.cat((hidden, context), dim=-1)))[
            ..., 0
        ]
        raw = raw * mask
        center = raw.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return (raw - center) * mask


def feature_statistics(
    features: np.ndarray, mask: np.ndarray, point_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    valid = mask & point_mask[:, None]
    values = np.asarray(features)[valid].astype(np.float64)
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std[std < 1e-5] = 1.0
    return mean, std


def pdp_cosine(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    numerator = (prediction * truth).sum(dim=2, dtype=torch.float64)
    denominator = torch.sqrt(
        prediction.square().sum(dim=2, dtype=torch.float64)
        * truth.square().sum(dim=2, dtype=torch.float64)
    )
    return numerator / torch.where(denominator == 0.0, 1e-12, denominator)


def batch_loss(
    model: PathEnergySetModel,
    point_ids: np.ndarray,
    features: np.ndarray,
    coefficient: np.ndarray,
    tau: np.ndarray,
    mask: np.ndarray,
    truth_pdp: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    frequency: torch.Tensor,
    device: torch.device,
    links_per_step: int,
    regularization: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, float]:
    local_features = torch.as_tensor(
        (np.asarray(features[point_ids], dtype=np.float32) - mean) / std,
        device=device,
    )
    local_mask = torch.as_tensor(np.asarray(mask[point_ids]), device=device)
    local_features = local_features * local_mask[..., None]
    raw = model(local_features, local_mask)
    amplitude = torch.exp(raw)
    link_count = coefficient.shape[2] * coefficient.shape[3]
    if links_per_step <= 0 or links_per_step >= link_count:
        links = np.arange(link_count, dtype=np.int64)
    else:
        links = np.sort(rng.choice(link_count, size=links_per_step, replace=False))
    coefficient_batch = np.asarray(coefficient[point_ids]).reshape(
        len(point_ids), coefficient.shape[1], link_count
    )[:, :, links]
    coefficient_tensor = torch.as_tensor(
        np.array(coefficient_batch, copy=True), device=device
    )
    tau_tensor = torch.as_tensor(
        np.array(tau[point_ids], dtype=np.float32, copy=True), device=device
    )
    phase = torch.exp(-2j * torch.pi * tau_tensor[:, :, None] * frequency[None, None])
    channel = torch.einsum(
        "bk,bkl,bks->bls", amplitude.to(coefficient_tensor.dtype), coefficient_tensor, phase
    )
    delay_power = torch.fft.ifft(channel, dim=-1).abs().square().float()
    prediction = delay_power / delay_power.sum(dim=2, keepdim=True).clamp_min(1e-30)
    truth = torch.as_tensor(
        np.asarray(truth_pdp[point_ids]).reshape(len(point_ids), link_count, 192)[
            :, links
        ],
        device=device,
        dtype=torch.float32,
    )
    cosine = pdp_cosine(prediction, truth)
    loss = -cosine.mean() + regularization * raw.square().sum() / local_mask.sum().clamp_min(1)
    return loss, float(cosine.mean().detach().item())


def train_fold(
    fold: int,
    args: argparse.Namespace,
    features: np.ndarray,
    coefficient: np.ndarray,
    tau: np.ndarray,
    mask: np.ndarray,
    truth_pdp: np.ndarray,
    fold_id: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    train_points = np.flatnonzero(fold_id != fold).astype(np.int64)
    held_points = np.flatnonzero(fold_id == fold).astype(np.int64)
    point_mask = fold_id != fold
    mean, std = feature_statistics(features, mask, point_mask)
    model = PathEnergySetModel(
        input_dim=features.shape[-1],
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        feedforward=args.feedforward,
        dropout=args.dropout,
        limit=args.log_amplitude_limit,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.1
    )
    frequency = torch.as_tensor(FREQUENCIES, device=device, dtype=torch.float32)
    rng = np.random.default_rng(args.random_state + 1000 * fold)
    history = []
    model.train()
    for epoch in range(args.epochs):
        order = rng.permutation(train_points)
        epoch_loss = 0.0
        epoch_cosine = 0.0
        batches = 0
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss, cosine = batch_loss(
                model,
                batch,
                features,
                coefficient,
                tau,
                mask,
                truth_pdp,
                mean,
                std,
                frequency,
                device,
                args.links_per_step,
                args.regularization,
                rng,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            epoch_cosine += cosine
            batches += 1
        scheduler.step()
        history.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss / max(batches, 1),
                "sampled_link_pdp_cosine": epoch_cosine / max(batches, 1),
            }
        )
        if (epoch + 1) % 20 == 0 or epoch + 1 == args.epochs:
            print(
                f"set PDP fold {fold} epoch {epoch + 1}/{args.epochs} "
                f"cosine={history[-1]['sampled_link_pdp_cosine']:.6f}",
                flush=True,
            )
    model.eval()
    held_prediction = np.zeros((len(held_points), mask.shape[1]), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(held_points), args.batch_size):
            end = min(start + args.batch_size, len(held_points))
            local = held_points[start:end]
            local_features = torch.as_tensor(
                (np.asarray(features[local], dtype=np.float32) - mean) / std,
                device=device,
            )
            local_mask = torch.as_tensor(np.asarray(mask[local]), device=device)
            local_features = local_features * local_mask[..., None]
            held_prediction[start:end] = model(local_features, local_mask).cpu().numpy()
    checkpoint = ARTIFACT_DIR / f"fold{fold}_set_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "config": vars(args),
            "input_dim": int(features.shape[-1]),
        },
        checkpoint,
    )
    return held_prediction, {
        "fold": fold,
        "train_points": int(len(train_points)),
        "held_points": int(len(held_points)),
        "checkpoint": PATHS.relative(checkpoint),
        "history": history,
    }


def render_pdp(
    log_amplitude: np.ndarray,
    coefficient: np.ndarray,
    tau: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    frequency = torch.as_tensor(FREQUENCIES, device=device, dtype=torch.float32)
    output = []
    for start in range(0, len(log_amplitude), batch_size):
        end = min(start + batch_size, len(log_amplitude))
        coefficient_tensor = torch.as_tensor(
            np.array(coefficient[start:end], copy=True), device=device
        )
        tau_tensor = torch.as_tensor(
            np.array(tau[start:end], dtype=np.float32, copy=True), device=device
        )
        local_mask = torch.as_tensor(np.asarray(mask[start:end]), device=device)
        amplitude = torch.exp(
            torch.as_tensor(log_amplitude[start:end], device=device)
        ) * local_mask
        phase = torch.exp(-2j * torch.pi * tau_tensor[:, :, None] * frequency[None, None])
        channel = torch.einsum(
            "bk,bkmu,bks->bmus",
            amplitude.to(coefficient_tensor.dtype),
            coefficient_tensor,
            phase,
        )
        output.append(pdp_distribution(channel).cpu().numpy().astype(np.float32))
    return np.concatenate(output)


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    required = (PATH_CACHE, FEATURES, ORACLE_METRICS, OOF_META, REGIME_CACHE, TRUTH_PDP)
    missing = [PATHS.relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    oracle = np.load(ORACLE_METRICS, allow_pickle=False)
    rows = oracle["rows"].astype(np.int64)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)[rows]
    features = np.load(FEATURES, mmap_mode="r")
    coefficient = np.load(PATH_CACHE / "coefficient.npy", mmap_mode="r")
    tau = np.load(PATH_CACHE / "tau.npy", mmap_mode="r")
    mask = np.asarray(np.load(PATH_CACHE / "mask.npy", mmap_mode="r"), dtype=bool)
    truth_pdp_all = np.load(TRUTH_PDP, mmap_mode="r")
    truth_pdp = truth_pdp_all[query_ids[rows]]

    prediction = np.zeros(mask.shape, dtype=np.float32)
    folds = []
    for fold in range(5):
        held = fold_id == fold
        values, diagnostics = train_fold(
            fold,
            args,
            features,
            coefficient,
            tau,
            mask,
            truth_pdp,
            fold_id,
            device,
        )
        prediction[held] = values
        folds.append(diagnostics)
    np.save(ARTIFACT_DIR / "oof_log_amplitude.npy", prediction)

    base = v9_channel(rows, device, args.batch_size)
    train_h = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    truth = np.asarray(train_h[query_ids[rows]], dtype=np.complex64)
    grids = []
    metrics_by_blend = {}
    endpoint_cosines = {}
    truth_pdp_tensor = torch.as_tensor(
        np.asarray(truth_pdp, dtype=np.float32), device=device
    )
    for blend in (0.0, 0.25, 0.50, 0.75, 1.0):
        path_pdp = render_pdp(
            blend * prediction,
            coefficient,
            tau,
            mask,
            device,
            args.batch_size,
        )
        path_pdp_tensor = torch.as_tensor(path_pdp, device=device)
        endpoint_cosines[f"blend_{blend:.2f}"] = float(
            pdp_cosine(
                path_pdp_tensor.reshape(len(path_pdp), -1, 192),
                truth_pdp_tensor.reshape(len(path_pdp), -1, 192),
            ).mean().item()
        )
        projection_rows, metrics = evaluate_grid(base, truth, path_pdp, device)
        for candidate in projection_rows:
            candidate["amplitude_blend"] = blend
            candidate["compound_name"] = f"blend{blend:.2f}_{candidate['candidate']}"
        grids.extend(projection_rows)
        metrics_by_blend[f"blend_{blend:.2f}"] = metrics
        print(f"set PDP blend {blend:.2f} evaluated", flush=True)
    grids.sort(key=lambda row: float(row["score"]), reverse=True)
    candidates = [
        row
        for row in grids
        if str(row["candidate"]).startswith("path_pdp_")
        and float(row["amplitude_blend"]) > 0.0
    ]
    selected = candidates[0]
    metrics = metrics_by_blend[f"blend_{float(selected['amplitude_blend']):.2f}"]
    names = metrics["candidate_names"].astype(str)
    selected_index = int(np.flatnonzero(names == selected["candidate"])[0])
    baseline_index = int(np.flatnonzero(names == "v9_identity")[0])
    fold_deltas = []
    for fold in range(5):
        local = fold_id == fold
        current = statistics(
            metrics["pas"][selected_index],
            metrics["pdp"][selected_index],
            metrics["error"][selected_index],
            metrics["energy"][selected_index],
            local,
        )
        baseline = statistics(
            metrics["pas"][baseline_index],
            metrics["pdp"][baseline_index],
            metrics["error"][baseline_index],
            metrics["energy"][baseline_index],
            local,
        )
        fold_deltas.append(
            {
                "fold": fold,
                "result": current,
                "baseline": baseline,
                "delta": float(current["score"] - baseline["score"]),
            }
        )
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "A point-level set model can infer coherent relative path energies directly from "
            "the full candidate path set and improve held-hole PDP without imitating a "
            "non-unique oracle amplitude vector."
        ),
        "configuration": vars(args),
        "point_count": int(len(rows)),
        "endpoint_cosines": endpoint_cosines,
        "fold_training": folds,
        "selected": selected,
        "selected_fold_deltas_vs_v9": fold_deltas,
        "positive_folds": int(sum(row["delta"] > 0.0 for row in fold_deltas)),
        "top_candidates": grids[:20],
        "artifacts": {
            "prediction": PATHS.relative(ARTIFACT_DIR / "oof_log_amplitude.npy"),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
