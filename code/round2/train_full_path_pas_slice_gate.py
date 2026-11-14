#!/usr/bin/env python3
"""Train a 4000-point cross-fitted PAS slice gate for the explicit V10 paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from analyze_joint_path_token_oracle import OOF_META
from analyze_v11_explicit_path_pas_upper_bound import build_raw_pas, v10_oof_channel
from analyze_v4_exact_joint_projection import (
    exact_pas_projection,
    exact_pdp_projection,
    pas_distribution,
    pdp_distribution,
    preserve_point_power,
    truth_views,
)
from prepare_sionna_calibration_dataset import TRUTH_PAS
from project_paths import PATHS
from round2.data import load_positions
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics
from train_full_path_pdp_link_gate import FOLD_IDS, rbf_donors


VERSION = "Round2-E69-full-4000-path-PAS-slice-gate"
PATH_CACHE = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_oracle"
    / "n4000_s24_k128_b096_d064_scatter1_sel106"
)
OOF_PATH_CACHE = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_oracle"
    / "n469_s24_k128_b096_d064_scatter1_sel106"
)
PDP_GATE_DIR = PATHS.artifacts / "round2" / "full_path_pdp_link_gate"
PAS_RELIABILITY = (
    PATHS.artifacts
    / "round2"
    / "v9_joint_sionna_endpoints"
    / "train_pas_spectral_reliability.npy"
)
ARTIFACT_DIR = PATHS.artifacts / "round2" / "full_path_pas_slice_gate"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_full_path_pas_slice_gate"
BASE_PAS = ARTIFACT_DIR / "train_rbf_base_pas_f16.npy"
RAW_PAS = ARTIFACT_DIR / "train_raw_path_pas_f16.npy"
POINT_CONTEXT = ARTIFACT_DIR / "train_point_context.npy"
OOF_ALPHA = ARTIFACT_DIR / "train_oof_alpha_f16.npy"
TRAIN_REPORT = ANALYSIS_DIR / "full_path_pas_slice_gate_train.json"
V10_REPORT = ANALYSIS_DIR / "full_path_pas_slice_gate_v10_oof.json"
V10_ALPHA = ARTIFACT_DIR / "v10_oof_pretrained_alpha_f16.npy"
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("build", "train", "evaluate", "all"), default="all"
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--smoothing", type=float, default=1e-5)
    parser.add_argument("--build-batch-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gate-penalty", type=float, default=3e-4)
    parser.add_argument("--smoothness-penalty", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=320)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--random-state", type=int, default=2026080769)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalize_pas(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(0.0)
    total = value.sum(dim=(1, 2, 3), keepdim=True)
    return torch.where(total > TINY, value / total.clamp_min(TINY), 0.0)


def cosine_slices(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    numerator = (prediction * truth).sum(dim=(1, 2, 3), dtype=torch.float64)
    denominator = torch.sqrt(
        prediction.square().sum(dim=(1, 2, 3), dtype=torch.float64)
        * truth.square().sum(dim=(1, 2, 3), dtype=torch.float64)
    )
    return numerator / torch.where(denominator == 0.0, 1e-12, denominator)


def slice_coordinates() -> np.ndarray:
    ue = np.arange(4, dtype=np.int64)
    frequency = np.arange(192, dtype=np.float32)
    u = np.broadcast_to(ue[:, None], (4, 192)).reshape(-1)
    f = np.broadcast_to(frequency[None, :], (4, 192)).reshape(-1)
    phase = 2.0 * np.pi * f / 192.0
    return np.concatenate(
        (
            np.eye(4, dtype=np.float32)[u],
            ((f - 95.5) / 95.5)[:, None].astype(np.float32),
            np.sin(phase)[:, None].astype(np.float32),
            np.cos(phase)[:, None].astype(np.float32),
            np.sin(2.0 * phase)[:, None].astype(np.float32),
            np.cos(2.0 * phase)[:, None].astype(np.float32),
        ),
        axis=1,
    )


def spectral_features(
    base: torch.Tensor,
    raw: torch.Tensor,
    context: torch.Tensor,
    coordinates: torch.Tensor,
) -> torch.Tensor:
    count = len(base)
    base_view = base.permute(0, 4, 5, 1, 2, 3).reshape(count, 768, 2, 16, 8)
    raw_view = raw.permute(0, 4, 5, 1, 2, 3).reshape(count, 768, 2, 16, 8)
    base_h = torch.sqrt(base_view.sum(dim=4).clamp_min(0.0))
    raw_h = torch.sqrt(raw_view.sum(dim=4).clamp_min(0.0))
    base_v = torch.sqrt(base_view.sum(dim=3).clamp_min(0.0))
    raw_v = torch.sqrt(raw_view.sum(dim=3).clamp_min(0.0))
    base_pol = torch.sqrt(base_view.sum(dim=(3, 4)).clamp_min(0.0))
    raw_pol = torch.sqrt(raw_view.sum(dim=(3, 4)).clamp_min(0.0))
    blocks = []
    for left, right in ((base_h, raw_h), (base_v, raw_v), (base_pol, raw_pol)):
        blocks.extend((left.flatten(2), right.flatten(2)))
        blocks.append((right - left).flatten(2))
        blocks.append((right * left).flatten(2))
    base_flat = base_view.flatten(2)
    raw_flat = raw_view.flatten(2)
    base_entropy = -(base_flat * torch.log(base_flat.clamp_min(1e-12))).sum(
        dim=2, keepdim=True
    )
    raw_entropy = -(raw_flat * torch.log(raw_flat.clamp_min(1e-12))).sum(
        dim=2, keepdim=True
    )
    agreement = (torch.sqrt(base_flat) * torch.sqrt(raw_flat)).sum(
        dim=2, keepdim=True
    )
    summary = torch.cat(
        (
            base_entropy,
            raw_entropy,
            agreement,
            base_flat.max(dim=2, keepdim=True).values,
            raw_flat.max(dim=2, keepdim=True).values,
        ),
        dim=2,
    )
    context_block = context[:, None].expand(-1, 768, -1)
    coordinate_block = coordinates[None].expand(count, -1, -1)
    return torch.cat((*blocks, summary, context_block, coordinate_block), dim=2)


class PasSliceGate(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, -4.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(features)[..., 0])


def build_rbf_base(args: argparse.Namespace, device: torch.device) -> None:
    if BASE_PAS.is_file() and not args.force:
        return
    fold_id = np.load(FOLD_IDS)
    _, train_positions, _, train_site, _ = load_positions()
    train_h = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    power = np.sum(np.abs(train_h) ** 2, axis=(1, 2, 3), dtype=np.float64)
    valid = power > 0.0
    donor_ids, weights = rbf_donors(
        fold_id,
        train_positions,
        train_site,
        valid,
        args.neighbors,
        args.smoothing,
    )
    truth = np.load(TRUTH_PAS, mmap_mode="r")
    output = np.lib.format.open_memmap(
        BASE_PAS, mode="w+", dtype=np.float16, shape=truth.shape
    )
    for start in range(0, len(truth), args.build_batch_size):
        end = min(start + args.build_batch_size, len(truth))
        donor = torch.as_tensor(
            np.asarray(truth[donor_ids[start:end]], dtype=np.float32), device=device
        )
        local_weights = torch.as_tensor(weights[start:end], device=device)
        interpolated = torch.einsum(
            "bk,bkphvus->bphvus", local_weights, torch.sqrt(donor.clamp_min(0.0))
        )
        result = normalize_pas(torch.square(interpolated.clamp_min(0.0)))
        output[start:end] = result.cpu().numpy().astype(np.float16)
        if end % 100 == 0 or end == len(truth):
            output.flush()
            print(f"RBF base PAS {end}/{len(truth)}", flush=True)
    output.flush()


def build_raw_path_pas(args: argparse.Namespace, device: torch.device) -> None:
    if RAW_PAS.is_file() and not args.force:
        return
    coefficient = np.load(PATH_CACHE / "coefficient.npy", mmap_mode="r")
    tau = np.load(PATH_CACHE / "tau.npy", mmap_mode="r")
    mask = np.load(PATH_CACHE / "mask.npy", mmap_mode="r")
    from analyze_joint_path_token_oracle import FREQUENCIES

    frequency = torch.as_tensor(FREQUENCIES, device=device, dtype=torch.float32)
    output = np.lib.format.open_memmap(
        RAW_PAS,
        mode="w+",
        dtype=np.float16,
        shape=(len(coefficient), 2, 16, 8, 4, 192),
    )
    for start in range(0, len(coefficient), args.build_batch_size):
        end = min(start + args.build_batch_size, len(coefficient))
        coefficient_tensor = torch.as_tensor(
            np.array(coefficient[start:end], dtype=np.complex64, copy=True), device=device
        )
        tau_tensor = torch.as_tensor(
            np.array(tau[start:end], dtype=np.float32, copy=True), device=device
        )
        mask_tensor = torch.as_tensor(np.asarray(mask[start:end]), device=device)
        phase = torch.exp(
            -2j * torch.pi * tau_tensor[:, :, None] * frequency[None, None]
        )
        channel = torch.einsum(
            "bk,bkmu,bks->bmus",
            mask_tensor.to(coefficient_tensor.dtype),
            coefficient_tensor,
            phase,
        )
        output[start:end] = pas_distribution(channel).cpu().numpy().astype(np.float16)
        if end % 100 == 0 or end == len(coefficient):
            output.flush()
            print(f"raw path PAS {end}/{len(coefficient)}", flush=True)
    output.flush()


def build_point_context() -> np.ndarray:
    if POINT_CONTEXT.is_file():
        return np.load(POINT_CONTEXT, mmap_mode="r")
    old = np.asarray(
        np.load(PDP_GATE_DIR / "train_point_context.npy", mmap_mode="r"),
        dtype=np.float32,
    )
    reliability = np.asarray(np.load(PAS_RELIABILITY, mmap_mode="r"), dtype=np.float32)
    context = np.concatenate((old, reliability), axis=1).astype(np.float32)
    np.save(POINT_CONTEXT, context)
    return np.load(POINT_CONTEXT, mmap_mode="r")


def checkpoint_path(fold: int) -> Path:
    return ARTIFACT_DIR / f"fold{fold}_full_pas_slice_gate.pt"


@torch.inference_mode()
def predict_alpha(
    model: PasSliceGate,
    point_ids: np.ndarray,
    base_pas: np.ndarray,
    raw_pas: np.ndarray,
    context: np.ndarray,
    context_mean: np.ndarray,
    context_std: np.ndarray,
    coordinates: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output = np.zeros((len(point_ids), 4, 192), dtype=np.float32)
    model.eval()
    for start in range(0, len(point_ids), batch_size):
        end = min(start + batch_size, len(point_ids))
        ids = point_ids[start:end]
        base = torch.as_tensor(
            np.asarray(base_pas[ids], dtype=np.float32), device=device
        )
        raw = torch.as_tensor(
            np.asarray(raw_pas[ids], dtype=np.float32), device=device
        )
        local_context = torch.as_tensor(
            (np.asarray(context[ids]) - context_mean) / context_std, device=device
        )
        features = spectral_features(base, raw, local_context, coordinates)
        output[start:end] = model(features).reshape(end - start, 4, 192).cpu().numpy()
    return output


def train_fold(
    fold: int,
    args: argparse.Namespace,
    fold_id: np.ndarray,
    nonzero: np.ndarray,
    base_pas: np.ndarray,
    raw_pas: np.ndarray,
    truth_pas: np.ndarray,
    context: np.ndarray,
    coordinates: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    train_points = np.flatnonzero((fold_id != fold) & nonzero).astype(np.int64)
    held_points = np.flatnonzero((fold_id == fold) & nonzero).astype(np.int64)
    context_mean = context[train_points].mean(axis=0).astype(np.float32)
    context_std = context[train_points].std(axis=0).astype(np.float32)
    context_std[context_std < 1e-5] = 1.0
    sample_base = torch.zeros((1, 2, 16, 8, 4, 192), device=device)
    sample_context = torch.zeros((1, context.shape[1]), device=device)
    input_dim = spectral_features(
        sample_base, sample_base, sample_context, coordinates
    ).shape[-1]
    del sample_base, sample_context
    model = PasSliceGate(input_dim, args.hidden, args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.1
    )
    rng = np.random.default_rng(args.random_state + 1000 * fold)
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(train_points)
        total_cosine = 0.0
        total_alpha = 0.0
        batches = 0
        for start in range(0, len(order), args.batch_size):
            ids = order[start : start + args.batch_size]
            base = torch.as_tensor(
                np.asarray(base_pas[ids], dtype=np.float32), device=device
            )
            raw = torch.as_tensor(
                np.asarray(raw_pas[ids], dtype=np.float32), device=device
            )
            truth = normalize_pas(
                torch.as_tensor(
                    np.asarray(truth_pas[ids], dtype=np.float32), device=device
                )
            )
            local_context = torch.as_tensor(
                (np.asarray(context[ids]) - context_mean) / context_std, device=device
            )
            features = spectral_features(base, raw, local_context, coordinates)
            alpha = model(features).reshape(len(ids), 4, 192)
            desired = normalize_pas(
                (1.0 - alpha[:, None, None, None]) * base
                + alpha[:, None, None, None] * raw
            )
            cosine = cosine_slices(desired, truth)
            smoothness = torch.mean(torch.square(alpha[:, :, 1:] - alpha[:, :, :-1]))
            loss = (
                -cosine.mean()
                + args.gate_penalty * alpha.mean()
                + args.smoothness_penalty * smoothness
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_cosine += float(cosine.mean().detach().item())
            total_alpha += float(alpha.mean().detach().item())
            batches += 1
        scheduler.step()
        history.append(
            {
                "epoch": epoch + 1,
                "train_pas_cosine": total_cosine / max(batches, 1),
                "train_alpha_mean": total_alpha / max(batches, 1),
            }
        )
        if (epoch + 1) % 3 == 0 or epoch + 1 == args.epochs:
            print(
                f"PAS slice gate fold {fold} epoch {epoch + 1}/{args.epochs} "
                f"cosine={history[-1]['train_pas_cosine']:.6f} "
                f"alpha={history[-1]['train_alpha_mean']:.4f}",
                flush=True,
            )
    prediction = predict_alpha(
        model,
        held_points,
        base_pas,
        raw_pas,
        context,
        context_mean,
        context_std,
        coordinates,
        device,
        args.batch_size,
    )
    checkpoint = checkpoint_path(fold)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "context_mean": context_mean,
            "context_std": context_std,
            "input_dim": input_dim,
            "config": vars(args),
        },
        checkpoint,
    )
    return prediction, {
        "fold": fold,
        "train_points": int(len(train_points)),
        "held_points": int(len(held_points)),
        "checkpoint": PATHS.relative(checkpoint),
        "history": history,
    }


def load_model(fold: int, device: torch.device) -> tuple[PasSliceGate, dict[str, object]]:
    checkpoint = torch.load(
        checkpoint_path(fold), map_location=device, weights_only=False
    )
    config = checkpoint["config"]
    model = PasSliceGate(
        int(checkpoint["input_dim"]), int(config["hidden"]), float(config["dropout"])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def train_models(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    fold_id = np.load(FOLD_IDS)
    base_pas = np.load(BASE_PAS, mmap_mode="r")
    raw_pas = np.load(RAW_PAS, mmap_mode="r")
    truth_pas = np.load(TRUTH_PAS, mmap_mode="r")
    context = build_point_context()
    coordinates = torch.as_tensor(slice_coordinates(), device=device)
    train_h = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    power = np.sum(np.abs(train_h) ** 2, axis=(1, 2, 3), dtype=np.float64)
    nonzero = power > 0.0
    alpha = np.zeros((len(fold_id), 4, 192), dtype=np.float32)
    training = []
    for fold in range(5):
        held = np.flatnonzero((fold_id == fold) & nonzero)
        values, diagnostics = train_fold(
            fold,
            args,
            fold_id,
            nonzero,
            base_pas,
            raw_pas,
            truth_pas,
            context,
            coordinates,
            device,
        )
        alpha[held] = values
        training.append(diagnostics)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    np.save(OOF_ALPHA, alpha.astype(np.float16))

    valid_ids = np.flatnonzero(nonzero)
    rows = []
    for scale in (0.25, 0.50, 0.75, 1.00, 1.25, 1.50):
        fold_rows = []
        weighted_base = 0.0
        weighted_selected = 0.0
        for fold in range(5):
            ids = np.flatnonzero((fold_id == fold) & nonzero)
            base = torch.as_tensor(
                np.asarray(base_pas[ids], dtype=np.float32), device=device
            )
            raw = torch.as_tensor(
                np.asarray(raw_pas[ids], dtype=np.float32), device=device
            )
            truth = normalize_pas(
                torch.as_tensor(
                    np.asarray(truth_pas[ids], dtype=np.float32), device=device
                )
            )
            local_alpha = torch.clamp(
                scale * torch.as_tensor(alpha[ids], device=device), 0.0, 1.0
            )
            desired = normalize_pas(
                (1.0 - local_alpha[:, None, None, None]) * base
                + local_alpha[:, None, None, None] * raw
            )
            baseline = float(cosine_slices(base, truth).mean().item())
            selected = float(cosine_slices(desired, truth).mean().item())
            fold_rows.append(
                {
                    "fold": fold,
                    "baseline": baseline,
                    "selected": selected,
                    "delta": selected - baseline,
                }
            )
            weighted_base += baseline * len(ids)
            weighted_selected += selected * len(ids)
        base_mean = weighted_base / len(valid_ids)
        selected_mean = weighted_selected / len(valid_ids)
        rows.append(
            {
                "scale": scale,
                "baseline": base_mean,
                "selected": selected_mean,
                "delta": selected_mean - base_mean,
                "positive_folds": int(sum(row["delta"] > 0.0 for row in fold_rows)),
                "folds": fold_rows,
            }
        )
    winner = max(rows, key=lambda row: float(row["delta"]))
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "configuration": vars(args),
        "nonzero_points": int(np.sum(nonzero)),
        "training": training,
        "scale_grid": rows,
        "selected": winner,
        "alpha": {
            "mean": float(np.mean(alpha[nonzero])),
            "median": float(np.median(alpha[nonzero])),
            "p90": float(np.quantile(alpha[nonzero], 0.90)),
            "fraction_below_0.05": float(np.mean(alpha[nonzero] < 0.05)),
        },
    }
    TRAIN_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["selected"], ensure_ascii=False, indent=2), flush=True)
    return report


def project_joint(
    base: torch.Tensor,
    desired_pas: torch.Tensor,
    desired_pdp: torch.Tensor,
    cycles: int,
    order: str,
) -> torch.Tensor:
    reference_power = base.abs().square().sum(dim=(1, 2, 3))
    output = base
    for _ in range(cycles):
        for domain in order.split("_"):
            if domain == "pas":
                output = exact_pas_projection(output, desired_pas)
            elif domain == "pdp":
                output = exact_pdp_projection(output, desired_pdp)
            else:
                raise ValueError(domain)
        output = preserve_point_power(output, reference_power)
    return output


def evaluate_v10(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    rows = np.load(OOF_PATH_CACHE / "rows.npy").astype(np.int64)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    test_hole_fold = metadata["fold_id"].astype(np.int8)[rows]
    global_ids = query_ids[rows]
    broad_fold = np.load(FOLD_IDS)[global_ids]
    base = v10_oof_channel(device)
    base_pas_array = pas_distribution(torch.as_tensor(base, device=device)).cpu().numpy()
    upper_args = argparse.Namespace(
        batch_size=args.build_batch_size, force=False
    )
    raw_pas = build_raw_pas(upper_args, device)
    full_context = np.load(POINT_CONTEXT, mmap_mode="r")
    context = np.asarray(full_context[global_ids], dtype=np.float32)
    coordinates = torch.as_tensor(slice_coordinates(), device=device)
    alpha = np.zeros((len(rows), 4, 192), dtype=np.float32)
    for fold in range(5):
        local_rows = np.flatnonzero(broad_fold == fold)
        if not len(local_rows):
            continue
        model, checkpoint = load_model(fold, device)
        proxy_context = np.lib.format.open_memmap(
            ARTIFACT_DIR / f"_oof_context_fold{fold}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(local_rows), context.shape[1]),
        )
        proxy_context[...] = context[local_rows]
        alpha[local_rows] = predict_alpha(
            model,
            np.arange(len(local_rows), dtype=np.int64),
            np.asarray(base_pas_array[local_rows]),
            np.asarray(raw_pas[local_rows]),
            proxy_context,
            np.asarray(checkpoint["context_mean"], dtype=np.float32),
            np.asarray(checkpoint["context_std"], dtype=np.float32),
            coordinates,
            device,
            args.batch_size,
        )
        del proxy_context
        (ARTIFACT_DIR / f"_oof_context_fold{fold}.npy").unlink(missing_ok=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    np.save(V10_ALPHA, alpha.astype(np.float16))

    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    truth = np.asarray(truth_all[global_ids], dtype=np.complex64)
    base_tensor = torch.as_tensor(base, device=device)
    truth_tensor = torch.as_tensor(truth, device=device)
    truth_pas, truth_pdp = truth_views(truth_tensor)
    base_pas = pas_distribution(base_tensor)
    base_pdp = pdp_distribution(base_tensor)
    raw_pas_tensor = torch.as_tensor(
        np.asarray(raw_pas, dtype=np.float32), device=device
    )
    baseline_values = point_metrics(
        base_tensor, truth_tensor, truth_pas, truth_pdp
    )
    result_rows = []
    metric_cache = {}
    for scale in (0.25, 0.50, 0.75, 1.00, 1.25):
        local_alpha = torch.clamp(
            scale * torch.as_tensor(alpha, device=device), 0.0, 1.0
        )
        desired_pas = normalize_pas(
            (1.0 - local_alpha[:, None, None, None]) * base_pas
            + local_alpha[:, None, None, None] * raw_pas_tensor
        )
        for order in ("pas_pdp", "pdp_pas"):
            for cycles in (1, 2, 4):
                with torch.inference_mode():
                    prediction = project_joint(
                        base_tensor, desired_pas, base_pdp, cycles, order
                    )
                    values = point_metrics(
                        prediction, truth_tensor, truth_pas, truth_pdp
                    )
                name = f"scale{scale:.2f}_{order}_c{cycles}"
                result_rows.append(
                    {
                        "candidate": name,
                        "scale": scale,
                        "order": order,
                        "cycles": cycles,
                        **statistics(*values),
                    }
                )
                metric_cache[name] = values
    result_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    selected = result_rows[0]
    selected_values = metric_cache[str(selected["candidate"])]
    baseline = statistics(*baseline_values)
    fold_results = []
    for fold in range(5):
        mask = test_hole_fold == fold
        current = statistics(*selected_values, mask)
        anchor = statistics(*baseline_values, mask)
        fold_results.append(
            {
                "fold": fold,
                "result": current,
                "baseline": anchor,
                "delta": float(current["score"] - anchor["score"]),
            }
        )
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "point_count": int(len(rows)),
        "baseline": baseline,
        "selected": selected,
        "fold_results": fold_results,
        "positive_folds": int(sum(row["delta"] > 0.0 for row in fold_results)),
        "alpha": {
            "mean": float(np.mean(alpha)),
            "median": float(np.median(alpha)),
            "p90": float(np.quantile(alpha, 0.90)),
            "fraction_below_0.05": float(np.mean(alpha < 0.05)),
        },
        "top_candidates": result_rows[:20],
    }
    V10_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    required = (
        PATH_CACHE,
        OOF_PATH_CACHE,
        FOLD_IDS,
        TRUTH_PAS,
        PDP_GATE_DIR / "train_point_context.npy",
        PAS_RELIABILITY,
        OOF_META,
    )
    missing = [PATHS.relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if args.phase in ("build", "all"):
        build_rbf_base(args, device)
        build_raw_path_pas(args, device)
        build_point_context()
        if args.phase == "build":
            return
    if args.phase in ("train", "all"):
        train_models(args, device)
        if args.phase == "train":
            return
    if args.phase in ("evaluate", "all"):
        evaluate_v10(args, device)
    print(f"{VERSION} completed in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
