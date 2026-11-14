#!/usr/bin/env python3
"""Train full-4000 E88 spectral gates and strictly transfer them to fixed holes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from prepare_sionna_calibration_dataset import TRUTH_PAS, TRUTH_PDP
from project_paths import PATHS
from round2.data import load_positions
from train_full_path_pas_slice_gate import (
    BASE_PAS,
    PasSliceGate,
    slice_coordinates,
    spectral_features as pas_spectral_features,
)
from train_full_path_pdp_link_gate import BASE_PDP, FOLD_IDS
from train_path_pdp_link_gate import (
    LinkGate,
    link_coordinates,
    spectral_features as pdp_spectral_features,
)
import train_e88_dense_path_residual_student as e88
import train_e90_e88_slice_reliability_gates as e90
import train_offline_sionna_geometry_student as e87
import train_surface_interaction_energy_model as e85


VERSION = "Round2-E92-full4000-E88-transfer-gates"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "full4000_e88_transfer_gates"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_full4000_e88_transfer_gates"
REPORT = ANALYSIS_DIR / "full4000_e88_transfer_gates.json"
FULL_ENDPOINTS = (
    PATHS.artifacts
    / "round2"
    / "e88_full4000_oof_endpoints"
    / "full4000_oof_e88_endpoints.npz"
)
OOF_ALPHA = ARTIFACT_DIR / "full4000_oof_alpha.npz"
FIXED_ALPHA = ARTIFACT_DIR / "fixed500_transferred_alpha.npz"
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pas-epochs", type=int, default=18)
    parser.add_argument("--pdp-epochs", type=int, default=36)
    parser.add_argument("--pas-batch-size", type=int, default=16)
    parser.add_argument("--pdp-batch-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=320)
    return parser.parse_args()


def context_features(confidence: np.ndarray, point_ids: np.ndarray) -> np.ndarray:
    _, _, _, train_site, _ = load_positions()
    return np.concatenate(
        (
            confidence.astype(np.float32),
            np.eye(2, dtype=np.float32)[train_site[point_ids]],
        ),
        axis=1,
    )


def normalize_context(
    context: np.ndarray, train_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = context[train_points].mean(axis=0).astype(np.float32)
    std = context[train_points].std(axis=0).astype(np.float32)
    std[std < 1e-5] = 1.0
    return ((context - mean) / std).astype(np.float32), mean, std


def pas_checkpoint(fold: int) -> Path:
    return ARTIFACT_DIR / f"fold{fold}_pas_gate.pt"


def pdp_checkpoint(fold: int) -> Path:
    return ARTIFACT_DIR / f"fold{fold}_pdp_gate.pt"


@torch.inference_mode()
def predict_pas_source(
    model: PasSliceGate,
    points: np.ndarray,
    base_pas: np.ndarray,
    endpoint_pas: np.ndarray,
    context: np.ndarray,
    coordinates: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output = np.empty((len(points), 4, 192), dtype=np.float32)
    for start in range(0, len(points), batch_size):
        end = min(start + batch_size, len(points))
        ids = points[start:end]
        base = torch.as_tensor(
            np.asarray(base_pas[ids], dtype=np.float32), device=device
        )
        raw = torch.as_tensor(
            endpoint_pas[ids].astype(np.float32), device=device
        )[..., None].expand(-1, -1, -1, -1, -1, 192)
        local_context = torch.as_tensor(context[ids], device=device)
        features = pas_spectral_features(base, raw, local_context, coordinates)
        output[start:end] = (
            model(features).reshape(len(ids), 4, 192).cpu().numpy()
        )
    return output


@torch.inference_mode()
def predict_pdp_source(
    model: LinkGate,
    points: np.ndarray,
    base_pdp: np.ndarray,
    endpoint_pdp: np.ndarray,
    context: np.ndarray,
    coordinates: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output = np.empty((len(points), 1024), dtype=np.float32)
    for start in range(0, len(points), batch_size):
        end = min(start + batch_size, len(points))
        ids = points[start:end]
        base = torch.as_tensor(
            np.asarray(base_pdp[ids], dtype=np.float32).reshape(len(ids), 1024, 192),
            device=device,
        )
        raw = torch.as_tensor(endpoint_pdp[ids].astype(np.float32), device=device)
        raw = raw[:, :, None, None, :, :].expand(-1, -1, 16, 8, -1, -1)
        raw = raw.reshape(len(ids), 1024, 192)
        local_context = torch.as_tensor(context[ids], device=device)
        features = pdp_spectral_features(base, raw, local_context, coordinates)
        output[start:end] = model(features).cpu().numpy()
    return output


def train_pas_fold(
    fold: int,
    fold_id: np.ndarray,
    nonzero: np.ndarray,
    base_pas: np.ndarray,
    truth_pas: np.ndarray,
    endpoint_pas: np.ndarray,
    context: np.ndarray,
    coordinates: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_points = np.flatnonzero((fold_id != fold) & nonzero)
    held_points = np.flatnonzero(fold_id == fold)
    normalized_context, mean, std = normalize_context(context, train_points)
    sample = torch.zeros((1, 2, 16, 8, 4, 192), device=device)
    sample_context = torch.zeros((1, context.shape[1]), device=device)
    input_dim = pas_spectral_features(
        sample, sample, sample_context, coordinates
    ).shape[-1]
    seed = 2026080792 + 1000 * fold
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = PasSliceGate(input_dim, args.hidden, 0.10).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=8e-4, weight_decay=1e-4, fused=device.type == "cuda"
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.pas_epochs, eta_min=8e-5
    )
    rng = np.random.default_rng(seed + 1)
    history = []
    for epoch in range(1, args.pas_epochs + 1):
        model.train()
        order = rng.permutation(train_points)
        scores = []
        alphas = []
        for start in range(0, len(order), args.pas_batch_size):
            ids = order[start : start + args.pas_batch_size]
            base = torch.as_tensor(
                np.asarray(base_pas[ids], dtype=np.float32), device=device
            )
            truth = torch.as_tensor(
                np.asarray(truth_pas[ids], dtype=np.float32), device=device
            )
            raw = torch.as_tensor(endpoint_pas[ids].astype(np.float32), device=device)
            raw = raw[..., None].expand(-1, -1, -1, -1, -1, 192)
            local_context = torch.as_tensor(normalized_context[ids], device=device)
            features = pas_spectral_features(base, raw, local_context, coordinates)
            alpha = model(features).reshape(len(ids), 4, 192)
            desired = e90.root_blend(
                base, raw, alpha[:, None, None, None], (1, 2, 3)
            )
            cosine = e90.cosine_pas(desired, truth)
            smoothness = (alpha[:, :, 1:] - alpha[:, :, :-1]).square().mean()
            loss = -cosine.mean() + 3e-4 * alpha.mean() + 1e-4 * smoothness
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            if not torch.isfinite(gradient):
                raise FloatingPointError(
                    f"non-finite E92 PAS gradient fold={fold} epoch={epoch}"
                )
            optimizer.step()
            scores.append(float(cosine.mean().detach().item()))
            alphas.append(float(alpha.mean().detach().item()))
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "cosine": float(np.mean(scores)),
                "alpha": float(np.mean(alphas)),
            }
        )
        if epoch % 3 == 0 or epoch == args.pas_epochs:
            print(
                f"E92 PAS fold={fold} epoch={epoch}/{args.pas_epochs} "
                f"cosine={history[-1]['cosine']:.6f} alpha={history[-1]['alpha']:.4f}",
                flush=True,
            )
    prediction = predict_pas_source(
        model,
        held_points,
        base_pas,
        endpoint_pas,
        normalized_context,
        coordinates,
        device,
        args.pas_batch_size,
    )
    checkpoint = pas_checkpoint(fold)
    torch.save(
        {
            "experiment": VERSION,
            "fold": fold,
            "input_dim": input_dim,
            "hidden": args.hidden,
            "state_dict": model.state_dict(),
            "context_mean": mean,
            "context_std": std,
        },
        checkpoint,
    )
    return prediction, {
        "fold": fold,
        "train_count": int(len(train_points)),
        "held_count": int(len(held_points)),
        "history": history,
        "checkpoint": PATHS.relative(checkpoint),
    }


def train_pdp_fold(
    fold: int,
    fold_id: np.ndarray,
    nonzero: np.ndarray,
    base_pdp: np.ndarray,
    truth_pdp: np.ndarray,
    endpoint_pdp: np.ndarray,
    context: np.ndarray,
    coordinates: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_points = np.flatnonzero((fold_id != fold) & nonzero)
    held_points = np.flatnonzero(fold_id == fold)
    normalized_context, mean, std = normalize_context(context, train_points)
    input_dim = 24 * 4 + 3 + context.shape[1] + coordinates.shape[1]
    seed = 2026080793 + 1000 * fold
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = LinkGate(input_dim, args.hidden, 0.10).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=8e-4, weight_decay=1e-4, fused=device.type == "cuda"
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.pdp_epochs, eta_min=8e-5
    )
    rng = np.random.default_rng(seed + 1)
    history = []
    for epoch in range(1, args.pdp_epochs + 1):
        model.train()
        order = rng.permutation(train_points)
        scores = []
        alphas = []
        for start in range(0, len(order), args.pdp_batch_size):
            ids = order[start : start + args.pdp_batch_size]
            base = torch.as_tensor(
                np.asarray(base_pdp[ids], dtype=np.float32).reshape(len(ids), 1024, 192),
                device=device,
            )
            truth = torch.as_tensor(
                np.asarray(truth_pdp[ids], dtype=np.float32).reshape(len(ids), 1024, 192),
                device=device,
            )
            raw = torch.as_tensor(endpoint_pdp[ids].astype(np.float32), device=device)
            raw = raw[:, :, None, None, :, :].expand(-1, -1, 16, 8, -1, -1)
            raw = raw.reshape(len(ids), 1024, 192)
            local_context = torch.as_tensor(normalized_context[ids], device=device)
            features = pdp_spectral_features(base, raw, local_context, coordinates)
            alpha = model(features)
            desired = e90.root_blend(base, raw, alpha[..., None], (2,))
            cosine = e90.cosine_pdp(desired, truth)
            loss = -cosine.mean() + 2e-4 * alpha.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            if not torch.isfinite(gradient):
                raise FloatingPointError(
                    f"non-finite E92 PDP gradient fold={fold} epoch={epoch}"
                )
            optimizer.step()
            scores.append(float(cosine.mean().detach().item()))
            alphas.append(float(alpha.mean().detach().item()))
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "cosine": float(np.mean(scores)),
                "alpha": float(np.mean(alphas)),
            }
        )
        if epoch % 6 == 0 or epoch == args.pdp_epochs:
            print(
                f"E92 PDP fold={fold} epoch={epoch}/{args.pdp_epochs} "
                f"cosine={history[-1]['cosine']:.6f} alpha={history[-1]['alpha']:.4f}",
                flush=True,
            )
    prediction = predict_pdp_source(
        model,
        held_points,
        base_pdp,
        endpoint_pdp,
        normalized_context,
        coordinates,
        device,
        args.pdp_batch_size,
    )
    checkpoint = pdp_checkpoint(fold)
    torch.save(
        {
            "experiment": VERSION,
            "fold": fold,
            "input_dim": input_dim,
            "hidden": args.hidden,
            "state_dict": model.state_dict(),
            "context_mean": mean,
            "context_std": std,
        },
        checkpoint,
    )
    return prediction, {
        "fold": fold,
        "train_count": int(len(train_points)),
        "held_count": int(len(held_points)),
        "history": history,
        "checkpoint": PATHS.relative(checkpoint),
    }


@torch.inference_mode()
def transfer_fixed500(
    fold_id: np.ndarray,
    context: np.ndarray,
    base_pas: torch.Tensor,
    base_pdp: torch.Tensor,
    endpoint_pas: torch.Tensor,
    endpoint_pdp: torch.Tensor,
    pas_coordinates: torch.Tensor,
    pdp_coordinates: torch.Tensor,
    query_ids: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha_pas = torch.zeros((len(query_ids), 4, 192), device=device)
    alpha_pdp = torch.zeros((len(query_ids), 1024), device=device)
    # The broad 4000-point folds select a gate that did not train on this point.
    # They are intentionally independent of the five fixed-hole reporting folds.
    query_gate_fold = fold_id[query_ids]
    assigned = np.zeros(len(query_ids), dtype=bool)
    for fold in range(5):
        rows = np.flatnonzero(query_gate_fold == fold)
        assigned[rows] = True
        ids = torch.as_tensor(rows, device=device)
        pas_payload = torch.load(pas_checkpoint(fold), map_location="cpu", weights_only=False)
        pas_model = PasSliceGate(
            pas_payload["input_dim"], pas_payload["hidden"], 0.10
        ).to(device)
        pas_model.load_state_dict(pas_payload["state_dict"])
        pdp_payload = torch.load(pdp_checkpoint(fold), map_location="cpu", weights_only=False)
        pdp_model = LinkGate(
            pdp_payload["input_dim"], pdp_payload["hidden"], 0.10
        ).to(device)
        pdp_model.load_state_dict(pdp_payload["state_dict"])
        pas_model.eval()
        pdp_model.eval()
        pas_mean = torch.as_tensor(pas_payload["context_mean"], device=device)
        pas_std = torch.as_tensor(pas_payload["context_std"], device=device)
        pdp_mean = torch.as_tensor(pdp_payload["context_mean"], device=device)
        pdp_std = torch.as_tensor(pdp_payload["context_std"], device=device)
        for start in range(0, len(rows), args.pdp_batch_size):
            end = min(start + args.pdp_batch_size, len(rows))
            local = ids[start:end]
            pas_context = (torch.as_tensor(context[rows[start:end]], device=device) - pas_mean) / pas_std
            pas_features = pas_spectral_features(
                base_pas[local], endpoint_pas[local], pas_context, pas_coordinates
            )
            alpha_pas[local] = pas_model(pas_features).reshape(len(local), 4, 192)
            pdp_context = (torch.as_tensor(context[rows[start:end]], device=device) - pdp_mean) / pdp_std
            local_base = base_pdp[local].reshape(len(local), 1024, 192)
            local_raw = endpoint_pdp[local].reshape(len(local), 1024, 192)
            pdp_features = pdp_spectral_features(
                local_base, local_raw, pdp_context, pdp_coordinates
            )
            alpha_pdp[local] = pdp_model(pdp_features)
        del pas_model, pdp_model
    if not np.all(assigned):
        raise RuntimeError("some fixed-500 rows were not assigned a held-out gate")
    return alpha_pas, alpha_pdp


@torch.inference_mode()
def evaluate_fixed500(
    alpha_pas: torch.Tensor,
    alpha_pdp: torch.Tensor,
    base_channel: torch.Tensor,
    truth_channel: torch.Tensor,
    truth_pas: torch.Tensor,
    truth_pdp: torch.Tensor,
    base_pas: torch.Tensor,
    base_pdp: torch.Tensor,
    endpoint_pas: torch.Tensor,
    endpoint_pdp: torch.Tensor,
    fold_id: np.ndarray,
) -> dict[str, Any]:
    baseline_values = e87.point_metrics(
        base_channel, truth_channel, truth_pas, truth_pdp
    )
    baseline = e87.statistics(*baseline_values)
    rows = []
    for scale in (0.25, 0.50, 0.75, 1.00):
        desired_pas = e90.root_blend(
            base_pas,
            endpoint_pas,
            (scale * alpha_pas)[:, None, None, None],
            (1, 2, 3),
        )
        desired_pdp = e90.root_blend(
            base_pdp,
            endpoint_pdp,
            (scale * alpha_pdp).reshape(len(base_channel), 256, 4, 1),
            (3,),
        )
        for order in ("pas_pdp", "pdp_pas"):
            prediction = e87.project_joint(
                base_channel, desired_pas, desired_pdp, 4, order
            )
            values = e87.point_metrics(
                prediction, truth_channel, truth_pas, truth_pdp
            )
            metrics = e87.statistics(*values)
            folds = []
            for fold in range(5):
                mask = fold_id == fold
                local = e87.statistics(*values, mask)
                local_base = e87.statistics(*baseline_values, mask)
                folds.append(
                    {
                        "fold": fold,
                        "delta": float(local["score"] - local_base["score"]),
                    }
                )
            rows.append(
                {
                    "scale": scale,
                    "order": order,
                    "metrics": metrics,
                    "delta": float(metrics["score"] - baseline["score"]),
                    "positive_folds": int(sum(row["delta"] > 0.0 for row in folds)),
                    "worst_fold_delta": float(min(row["delta"] for row in folds)),
                    "folds": folds,
                }
            )
    return {
        "V12": baseline,
        "primary_scale_050_pas_pdp": next(
            row for row in rows if row["scale"] == 0.50 and row["order"] == "pas_pdp"
        ),
        "best_scan": max(rows, key=lambda row: float(row["metrics"]["score"])),
        "scan": rows,
    }


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    fold_id = np.load(FOLD_IDS).astype(np.int8)
    full_endpoint = np.load(FULL_ENDPOINTS, allow_pickle=False)
    endpoint_fold = full_endpoint["fold_id"].astype(np.int8)
    if not np.array_equal(endpoint_fold, fold_id):
        raise ValueError("E91 endpoint folds do not match the full-4000 gate folds")
    endpoint_pas = full_endpoint["final_pas"].astype(np.float32)
    endpoint_pdp = full_endpoint["final_pdp"].astype(np.float32)
    confidence = full_endpoint["confidence"].astype(np.float32)
    if endpoint_pas.shape != (len(fold_id), 2, 16, 8, 4):
        raise ValueError(f"unexpected E91 PAS shape: {endpoint_pas.shape}")
    if endpoint_pdp.shape != (len(fold_id), 2, 4, 192):
        raise ValueError(f"unexpected E91 PDP shape: {endpoint_pdp.shape}")
    context = context_features(confidence, np.arange(len(fold_id), dtype=np.int64))
    power = np.asarray(np.load(e85.POINT_POWER), dtype=np.float64)
    nonzero = power > 0.0
    base_pas = np.load(BASE_PAS, mmap_mode="r")
    base_pdp = np.load(BASE_PDP, mmap_mode="r")
    truth_pas = np.load(TRUTH_PAS, mmap_mode="r")
    truth_pdp = np.load(TRUTH_PDP, mmap_mode="r")
    pas_coordinates = torch.as_tensor(slice_coordinates(), device=device)
    pdp_coordinates = torch.as_tensor(link_coordinates(), device=device)
    oof_pas = np.zeros((len(fold_id), 4, 192), dtype=np.float16)
    oof_pdp = np.zeros((len(fold_id), 1024), dtype=np.float16)
    pas_training = []
    pdp_training = []
    started = time.time()
    for fold in range(5):
        prediction, record = train_pas_fold(
            fold,
            fold_id,
            nonzero,
            base_pas,
            truth_pas,
            endpoint_pas,
            context,
            pas_coordinates,
            args,
            device,
        )
        oof_pas[fold_id == fold] = prediction.astype(np.float16)
        pas_training.append(record)
        torch.cuda.empty_cache()
    for fold in range(5):
        prediction, record = train_pdp_fold(
            fold,
            fold_id,
            nonzero,
            base_pdp,
            truth_pdp,
            endpoint_pdp,
            context,
            pdp_coordinates,
            args,
            device,
        )
        oof_pdp[fold_id == fold] = prediction.astype(np.float16)
        pdp_training.append(record)
        torch.cuda.empty_cache()
    np.savez_compressed(
        OOF_ALPHA,
        fold_id=fold_id,
        alpha_pas=oof_pas,
        alpha_pdp=oof_pdp,
    )

    fixed_meta = np.load(e88.OOF_META, allow_pickle=False)
    query_ids = fixed_meta["query_ids"].astype(np.int64)
    fixed_fold = fixed_meta["fold_id"].astype(np.int8)
    if (
        query_ids.shape != (500,)
        or len(np.unique(query_ids)) != len(query_ids)
        or np.any(query_ids < 0)
        or np.any(query_ids >= len(fold_id))
    ):
        raise ValueError("fixed-500 query IDs are invalid")
    if fixed_fold.shape != (500,) or np.any((fixed_fold < 0) | (fixed_fold >= 5)):
        raise ValueError("fixed-500 reporting folds are invalid")
    query_gate_fold = fold_id[query_ids]
    fixed_endpoint = np.load(e88.OOF_PREDICTIONS, allow_pickle=False)
    if not np.array_equal(fixed_endpoint["query_ids"].astype(np.int64), query_ids):
        raise ValueError("fixed E88 endpoints are not aligned with fixed-500 query IDs")
    if not np.array_equal(fixed_endpoint["fold_id"].astype(np.int8), fixed_fold):
        raise ValueError("fixed E88 endpoint folds do not match reporting folds")
    fixed_context = context_features(
        fixed_endpoint["confidence"].astype(np.float32), query_ids
    )
    base_channel = torch.as_tensor(
        np.array(np.load(e87.V12_CHANNEL, mmap_mode="r"), copy=True), device=device
    )
    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    truth_channel = torch.as_tensor(
        np.asarray(truth_all[query_ids], dtype=np.complex64), device=device
    )
    fixed_base_pas = e87.pas_distribution(base_channel)
    fixed_base_pdp = e87.pdp_distribution(base_channel)
    fixed_truth_pas = e87.pas_distribution(truth_channel)
    fixed_truth_pdp = e87.pdp_distribution(truth_channel)
    fixed_endpoint_pas, fixed_endpoint_pdp = e87.expanded_endpoint(
        fixed_endpoint["final_pas"].astype(np.float32),
        fixed_endpoint["final_pdp"].astype(np.float32),
        device,
    )
    fixed_alpha_pas, fixed_alpha_pdp = transfer_fixed500(
        fold_id,
        fixed_context,
        fixed_base_pas,
        fixed_base_pdp,
        fixed_endpoint_pas,
        fixed_endpoint_pdp,
        pas_coordinates,
        pdp_coordinates,
        query_ids,
        args,
        device,
    )
    np.savez_compressed(
        FIXED_ALPHA,
        query_ids=query_ids.astype(np.int32),
        fold_id=fixed_fold,
        gate_fold_id=query_gate_fold,
        alpha_pas=fixed_alpha_pas.cpu().numpy().astype(np.float16),
        alpha_pdp=fixed_alpha_pdp.cpu().numpy().astype(np.float16),
    )
    fixed_evaluation = evaluate_fixed500(
        fixed_alpha_pas,
        fixed_alpha_pdp,
        base_channel,
        truth_channel,
        fixed_truth_pas,
        fixed_truth_pdp,
        fixed_base_pas,
        fixed_base_pdp,
        fixed_endpoint_pas,
        fixed_endpoint_pdp,
        fixed_fold,
    )
    primary = fixed_evaluation["primary_scale_050_pas_pdp"]
    report = {
        "experiment": VERSION,
        "status": "strict_complete",
        "configuration": vars(args),
        "pas_training": pas_training,
        "pdp_training": pdp_training,
        "fixed500": fixed_evaluation,
        "decision": {
            "primary_passed": bool(
                primary["delta"] > 0.0 and primary["positive_folds"] == 5
            ),
            "primary": primary,
        },
        "outputs": {
            "full4000_oof_alpha": PATHS.relative(OOF_ALPHA),
            "fixed500_alpha": PATHS.relative(FIXED_ALPHA),
        },
        "seconds": time.time() - started,
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
