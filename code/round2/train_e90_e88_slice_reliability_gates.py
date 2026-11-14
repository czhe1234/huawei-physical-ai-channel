#!/usr/bin/env python3
"""Train strict PAS-slice and PDP-link reliability gates for the E88 endpoint."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
import time
from typing import Any

import numpy as np
import torch

from project_paths import PATHS
from round2.data import load_positions
from train_full_path_pas_slice_gate import (
    PasSliceGate,
    normalize_pas,
    slice_coordinates,
    spectral_features as pas_spectral_features,
)
from train_path_pdp_link_gate import (
    LinkGate,
    link_coordinates,
    spectral_features as pdp_spectral_features,
)
import train_e88_dense_path_residual_student as e88
import train_offline_sionna_geometry_student as e87


VERSION = "Round2-E90-E88-slice-reliability-gates"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e88_slice_reliability_gates"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e88_slice_reliability_gates"
REPORT = ANALYSIS_DIR / "e88_slice_reliability_gates.json"
OOF_ALPHA = ARTIFACT_DIR / "strict_oof_slice_alpha.npz"
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    epochs: int = 24
    point_batch_size: int = 4
    hidden: int = 320
    dropout: float = 0.10
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    max_alpha: float = 0.60
    gate_penalty: float = 2e-4
    smoothness_penalty: float = 1e-4
    zero_penalty: float = 0.02
    validation_interval: int = 2
    projection_cycles: int = 4
    projection_order: str = "pas_pdp"
    random_state: int = 2026080790


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--point-batch-size", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=320)
    return parser.parse_args()


def cosine_pas(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    numerator = (prediction * truth).sum(dim=(1, 2, 3))
    norm_product = (
        prediction.square().sum(dim=(1, 2, 3))
        * truth.square().sum(dim=(1, 2, 3))
    )
    denominator = torch.sqrt(norm_product.clamp_min(1e-24))
    return numerator / torch.where(denominator == 0.0, 1e-12, denominator)


def cosine_pdp(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    numerator = (prediction * truth).sum(dim=2)
    norm_product = (
        prediction.square().sum(dim=2) * truth.square().sum(dim=2)
    )
    denominator = torch.sqrt(norm_product.clamp_min(1e-24))
    return numerator / torch.where(denominator == 0.0, 1e-12, denominator)


def root_blend(
    base: torch.Tensor,
    endpoint: torch.Tensor,
    alpha: torch.Tensor,
    norm_dims: tuple[int, ...],
) -> torch.Tensor:
    root = (1.0 - alpha) * torch.sqrt(base.clamp_min(0.0)) + alpha * torch.sqrt(
        endpoint.clamp_min(0.0)
    )
    value = root.square()
    total = value.sum(dim=norm_dims, keepdim=True)
    return torch.where(total > 0.0, value / total.clamp_min(TINY), 0.0)


def point_context(confidence: np.ndarray, query_ids: np.ndarray) -> np.ndarray:
    _, _, _, train_site, _ = load_positions()
    return np.concatenate(
        (
            confidence.astype(np.float32),
            np.eye(2, dtype=np.float32)[train_site[query_ids]],
        ),
        axis=1,
    )


def combined_pas_features(
    base: torch.Tensor,
    endpoint: torch.Tensor,
    path: torch.Tensor,
    context: torch.Tensor,
    coordinates: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        (
            pas_spectral_features(base, endpoint, context, coordinates),
            pas_spectral_features(base, path, context, coordinates),
        ),
        dim=2,
    )


def combined_pdp_features(
    base: torch.Tensor,
    endpoint: torch.Tensor,
    path: torch.Tensor,
    context: torch.Tensor,
    coordinates: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        (
            pdp_spectral_features(base, endpoint, context, coordinates),
            pdp_spectral_features(base, path, context, coordinates),
        ),
        dim=2,
    )


def build_models(
    context_dim: int,
    pas_coordinates: torch.Tensor,
    pdp_coordinates: torch.Tensor,
    config: Config,
    device: torch.device,
) -> tuple[PasSliceGate, LinkGate]:
    sample_pas = torch.zeros((1, 2, 16, 8, 4, 192), device=device)
    sample_pdp = torch.zeros((1, 1024, 192), device=device)
    sample_context = torch.zeros((1, context_dim), device=device)
    pas_dim = combined_pas_features(
        sample_pas, sample_pas, sample_pas, sample_context, pas_coordinates
    ).shape[-1]
    pdp_dim = combined_pdp_features(
        sample_pdp, sample_pdp, sample_pdp, sample_context, pdp_coordinates
    ).shape[-1]
    return (
        PasSliceGate(pas_dim, config.hidden, config.dropout).to(device),
        LinkGate(pdp_dim, config.hidden, config.dropout).to(device),
    )


@torch.inference_mode()
def predict_alpha(
    pas_model: PasSliceGate,
    pdp_model: LinkGate,
    points: np.ndarray,
    base_pas: torch.Tensor,
    base_pdp: torch.Tensor,
    endpoint_pas: torch.Tensor,
    endpoint_pdp: torch.Tensor,
    path_pas: torch.Tensor,
    path_pdp: torch.Tensor,
    context: torch.Tensor,
    pas_coordinates: torch.Tensor,
    pdp_coordinates: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    pas_model.eval()
    pdp_model.eval()
    output_pas = torch.empty((len(points), 4, 192), device=base_pas.device)
    output_pdp = torch.empty((len(points), 1024), device=base_pas.device)
    for start in range(0, len(points), config.point_batch_size):
        end = min(start + config.point_batch_size, len(points))
        ids = torch.as_tensor(points[start:end], device=base_pas.device)
        local_context = context[ids]
        pas_features = combined_pas_features(
            base_pas[ids],
            endpoint_pas[ids],
            path_pas[ids],
            local_context,
            pas_coordinates,
        )
        output_pas[start:end] = config.max_alpha * pas_model(pas_features).reshape(
            len(ids), 4, 192
        )
        local_base_pdp = base_pdp[ids].reshape(len(ids), 1024, 192)
        local_endpoint_pdp = endpoint_pdp[ids].reshape(len(ids), 1024, 192)
        local_path_pdp = path_pdp[ids].reshape(len(ids), 1024, 192)
        pdp_features = combined_pdp_features(
            local_base_pdp,
            local_endpoint_pdp,
            local_path_pdp,
            local_context,
            pdp_coordinates,
        )
        output_pdp[start:end] = config.max_alpha * pdp_model(pdp_features)
    return output_pas, output_pdp


@torch.inference_mode()
def evaluate_points(
    pas_model: PasSliceGate,
    pdp_model: LinkGate,
    points: np.ndarray,
    base_channel: torch.Tensor,
    truth_channel: torch.Tensor,
    truth_pas: torch.Tensor,
    truth_pdp: torch.Tensor,
    base_pas: torch.Tensor,
    base_pdp: torch.Tensor,
    endpoint_pas: torch.Tensor,
    endpoint_pdp: torch.Tensor,
    path_pas: torch.Tensor,
    path_pdp: torch.Tensor,
    context: torch.Tensor,
    pas_coordinates: torch.Tensor,
    pdp_coordinates: torch.Tensor,
    config: Config,
) -> tuple[dict[str, float | int], torch.Tensor, torch.Tensor]:
    alpha_pas, alpha_pdp = predict_alpha(
        pas_model,
        pdp_model,
        points,
        base_pas,
        base_pdp,
        endpoint_pas,
        endpoint_pdp,
        path_pas,
        path_pdp,
        context,
        pas_coordinates,
        pdp_coordinates,
        config,
    )
    ids = torch.as_tensor(points, device=base_channel.device)
    desired_pas = root_blend(
        base_pas[ids],
        endpoint_pas[ids],
        alpha_pas[:, None, None, None],
        (1, 2, 3),
    )
    desired_pdp = root_blend(
        base_pdp[ids],
        endpoint_pdp[ids],
        alpha_pdp.reshape(len(points), 256, 4, 1),
        (3,),
    )
    prediction = e87.project_joint(
        base_channel[ids],
        desired_pas,
        desired_pdp,
        config.projection_cycles,
        config.projection_order,
    )
    metrics = e87.statistics(
        *e87.point_metrics(
            prediction,
            truth_channel[ids],
            truth_pas[ids],
            truth_pdp[ids],
        )
    )
    return metrics, alpha_pas, alpha_pdp


def train_fold(
    outer: int,
    fold_id: np.ndarray,
    valid: np.ndarray,
    base_channel: torch.Tensor,
    truth_channel: torch.Tensor,
    truth_pas: torch.Tensor,
    truth_pdp: torch.Tensor,
    base_pas: torch.Tensor,
    base_pdp: torch.Tensor,
    endpoint_pas: torch.Tensor,
    endpoint_pdp: torch.Tensor,
    path_pas: torch.Tensor,
    path_pdp: torch.Tensor,
    context: torch.Tensor,
    pas_coordinates: torch.Tensor,
    pdp_coordinates: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    validation_fold = (outer + 1) % 5
    fit = np.flatnonzero((fold_id != outer) & (fold_id != validation_fold))
    validation = np.flatnonzero(fold_id == validation_fold)
    held = np.flatnonzero(fold_id == outer)
    seed = config.random_state + 1000 * outer
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pas_model, pdp_model = build_models(
        context.shape[1], pas_coordinates, pdp_coordinates, config, base_channel.device
    )
    parameters = tuple(pas_model.parameters()) + tuple(pdp_model.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=base_channel.device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.learning_rate * 0.1,
    )
    rng = np.random.default_rng(seed + 1)
    best_score = -math.inf
    best_epoch = 0
    best_pas = copy.deepcopy(pas_model.state_dict())
    best_pdp = copy.deepcopy(pdp_model.state_dict())
    history = []
    for epoch in range(1, config.epochs + 1):
        pas_model.train()
        pdp_model.train()
        order = rng.permutation(fit)
        losses = []
        pas_scores = []
        pdp_scores = []
        alpha_means = []
        for start in range(0, len(order), config.point_batch_size):
            local_np = order[start : start + config.point_batch_size]
            ids = torch.as_tensor(local_np, device=base_channel.device)
            local_context = context[ids]
            pas_features = combined_pas_features(
                base_pas[ids],
                endpoint_pas[ids],
                path_pas[ids],
                local_context,
                pas_coordinates,
            )
            alpha_pas = config.max_alpha * pas_model(pas_features).reshape(
                len(ids), 4, 192
            )
            desired_pas = root_blend(
                base_pas[ids],
                endpoint_pas[ids],
                alpha_pas[:, None, None, None],
                (1, 2, 3),
            )
            pas_cosine = cosine_pas(desired_pas, truth_pas[ids])

            local_base_pdp = base_pdp[ids].reshape(len(ids), 1024, 192)
            local_endpoint_pdp = endpoint_pdp[ids].reshape(len(ids), 1024, 192)
            local_path_pdp = path_pdp[ids].reshape(len(ids), 1024, 192)
            pdp_features = combined_pdp_features(
                local_base_pdp,
                local_endpoint_pdp,
                local_path_pdp,
                local_context,
                pdp_coordinates,
            )
            alpha_pdp = config.max_alpha * pdp_model(pdp_features)
            desired_pdp = root_blend(
                local_base_pdp,
                local_endpoint_pdp,
                alpha_pdp[..., None],
                (2,),
            )
            pdp_cosine = cosine_pdp(
                desired_pdp, truth_pdp[ids].reshape(len(ids), 1024, 192)
            )
            local_valid = torch.as_tensor(valid[local_np], device=base_channel.device)
            if torch.any(local_valid):
                pas_loss = (1.0 - pas_cosine[local_valid]).mean()
                pdp_loss = (1.0 - pdp_cosine[local_valid]).mean()
            else:
                pas_loss = alpha_pas.sum() * 0.0
                pdp_loss = alpha_pdp.sum() * 0.0
            zero_loss = alpha_pas.sum() * 0.0
            if torch.any(~local_valid):
                zero_loss = 0.5 * (
                    alpha_pas[~local_valid].square().mean()
                    + alpha_pdp[~local_valid].square().mean()
                )
            smoothness = (alpha_pas[:, :, 1:] - alpha_pas[:, :, :-1]).square().mean()
            loss = (
                pas_loss
                + pdp_loss
                + config.gate_penalty * (alpha_pas.mean() + alpha_pdp.mean())
                + config.smoothness_penalty * smoothness
                + config.zero_penalty * zero_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 2.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"non-finite E90 gradient outer={outer} epoch={epoch}"
                )
            optimizer.step()
            losses.append(float(loss.detach().item()))
            if torch.any(local_valid):
                pas_scores.append(float(pas_cosine[local_valid].mean().detach().item()))
                pdp_scores.append(float(pdp_cosine[local_valid].mean().detach().item()))
            alpha_means.append(
                0.5
                * (
                    float(alpha_pas.mean().detach().item())
                    + float(alpha_pdp.mean().detach().item())
                )
            )
        scheduler.step()
        if epoch % config.validation_interval == 0 or epoch == config.epochs:
            validation_metrics, _, _ = evaluate_points(
                pas_model,
                pdp_model,
                validation,
                base_channel,
                truth_channel,
                truth_pas,
                truth_pdp,
                base_pas,
                base_pdp,
                endpoint_pas,
                endpoint_pdp,
                path_pas,
                path_pdp,
                context,
                pas_coordinates,
                pdp_coordinates,
                config,
            )
            row = {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "train_pas": float(np.mean(pas_scores)),
                "train_pdp": float(np.mean(pdp_scores)),
                "alpha_mean": float(np.mean(alpha_means)),
                "validation": validation_metrics,
            }
            history.append(row)
            if float(validation_metrics["score"]) > best_score:
                best_score = float(validation_metrics["score"])
                best_epoch = epoch
                best_pas = copy.deepcopy(pas_model.state_dict())
                best_pdp = copy.deepcopy(pdp_model.state_dict())
            print(
                f"E90 outer={outer} epoch={epoch}/{config.epochs} "
                f"loss={row['loss']:.6f} alpha={row['alpha_mean']:.4f} "
                f"validation={validation_metrics['score']:.6f}",
                flush=True,
            )
    pas_model.load_state_dict(best_pas)
    pdp_model.load_state_dict(best_pdp)
    held_metrics, held_pas, held_pdp = evaluate_points(
        pas_model,
        pdp_model,
        held,
        base_channel,
        truth_channel,
        truth_pas,
        truth_pdp,
        base_pas,
        base_pdp,
        endpoint_pas,
        endpoint_pdp,
        path_pas,
        path_pdp,
        context,
        pas_coordinates,
        pdp_coordinates,
        config,
    )
    checkpoint_path = ARTIFACT_DIR / f"outer{outer}_slice_gates.pt"
    torch.save(
        {
            "experiment": VERSION,
            "outer": outer,
            "config": asdict(config),
            "pas_state_dict": pas_model.state_dict(),
            "pdp_state_dict": pdp_model.state_dict(),
            "context_dim": int(context.shape[1]),
            "sionna_inputs_required_for_inference": False,
        },
        checkpoint_path,
    )
    return held_pas, held_pdp, {
        "outer": outer,
        "fit_count": int(len(fit)),
        "validation_count": int(len(validation)),
        "held_count": int(len(held)),
        "best_epoch": best_epoch,
        "best_validation": best_score,
        "held_metrics": held_metrics,
        "history": history,
        "checkpoint": PATHS.relative(checkpoint_path),
    }


@torch.inference_mode()
def evaluate_oof(
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
    config: Config,
) -> dict[str, Any]:
    baseline_values = e87.point_metrics(
        base_channel, truth_channel, truth_pas, truth_pdp
    )
    baseline = e87.statistics(*baseline_values)
    branches = {}
    for name, pas_scale, pdp_scale in (
        ("pas_only", 1.0, 0.0),
        ("pdp_only", 0.0, 1.0),
        ("joint", 1.0, 1.0),
        ("joint_scale_075", 0.75, 0.75),
        ("joint_scale_050", 0.50, 0.50),
    ):
        desired_pas = root_blend(
            base_pas,
            endpoint_pas,
            (pas_scale * alpha_pas)[:, None, None, None],
            (1, 2, 3),
        )
        desired_pdp = root_blend(
            base_pdp,
            endpoint_pdp,
            (pdp_scale * alpha_pdp).reshape(len(base_pdp), 256, 4, 1),
            (3,),
        )
        prediction = e87.project_joint(
            base_channel,
            desired_pas,
            desired_pdp,
            config.projection_cycles,
            config.projection_order,
        )
        values = e87.point_metrics(prediction, truth_channel, truth_pas, truth_pdp)
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
        branches[name] = {
            "metrics": metrics,
            "delta": float(metrics["score"] - baseline["score"]),
            "positive_folds": int(sum(row["delta"] > 0.0 for row in folds)),
            "worst_fold_delta": float(min(row["delta"] for row in folds)),
            "folds": folds,
        }
    return {"V12": baseline, "branches": branches}


def main() -> None:
    args = parse_args()
    config = Config(
        epochs=args.epochs,
        point_batch_size=args.point_batch_size,
        hidden=args.hidden,
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    metadata = np.load(e88.OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    endpoint = np.load(e88.OOF_PREDICTIONS, allow_pickle=False)
    confidence = endpoint["confidence"].astype(np.float32)
    context_array = point_context(confidence, query_ids)
    context = torch.as_tensor(context_array, device=device)
    base_channel = torch.as_tensor(
        np.array(np.load(e87.V12_CHANNEL, mmap_mode="r"), copy=True), device=device
    )
    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    truth_channel = torch.as_tensor(
        np.asarray(truth_all[query_ids], dtype=np.complex64), device=device
    )
    valid = (
        truth_channel.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64) > 0.0
    ).cpu().numpy()
    base_pas = e87.pas_distribution(base_channel)
    base_pdp = e87.pdp_distribution(base_channel)
    truth_pas = e87.pas_distribution(truth_channel)
    truth_pdp = e87.pdp_distribution(truth_channel)
    endpoint_pas, endpoint_pdp = e87.expanded_endpoint(
        endpoint["final_pas"].astype(np.float32),
        endpoint["final_pdp"].astype(np.float32),
        device,
    )
    path_pas, path_pdp = e87.expanded_endpoint(
        endpoint["path_pas"].astype(np.float32),
        endpoint["path_pdp"].astype(np.float32),
        device,
    )
    pas_coordinates = torch.as_tensor(slice_coordinates(), device=device)
    pdp_coordinates = torch.as_tensor(link_coordinates(), device=device)
    oof_pas = torch.zeros((len(query_ids), 4, 192), device=device)
    oof_pdp = torch.zeros((len(query_ids), 1024), device=device)
    training = []
    started = time.time()
    for outer in range(5):
        held_pas, held_pdp, record = train_fold(
            outer,
            fold_id,
            valid,
            base_channel,
            truth_channel,
            truth_pas,
            truth_pdp,
            base_pas,
            base_pdp,
            endpoint_pas,
            endpoint_pdp,
            path_pas,
            path_pdp,
            context,
            pas_coordinates,
            pdp_coordinates,
            config,
        )
        held = np.flatnonzero(fold_id == outer)
        oof_pas[torch.as_tensor(held, device=device)] = held_pas
        oof_pdp[torch.as_tensor(held, device=device)] = held_pdp
        training.append(record)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    evaluation = evaluate_oof(
        oof_pas,
        oof_pdp,
        base_channel,
        truth_channel,
        truth_pas,
        truth_pdp,
        base_pas,
        base_pdp,
        endpoint_pas,
        endpoint_pdp,
        fold_id,
        config,
    )
    np.savez_compressed(
        OOF_ALPHA,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        alpha_pas=oof_pas.cpu().numpy().astype(np.float16),
        alpha_pdp=oof_pdp.cpu().numpy().astype(np.float16),
    )
    selected = evaluation["branches"]["joint"]
    report = {
        "experiment": VERSION,
        "status": "strict_complete",
        "primary_hypothesis": (
            "E88 reliability is slice-specific and can be predicted from deployable "
            "base/final/path spectral disagreement features."
        ),
        "configuration": asdict(config),
        "training": training,
        "evaluation": evaluation,
        "decision": {
            "passed": bool(
                selected["delta"] > 0.0 and selected["positive_folds"] == 5
            ),
            "selected_branch": "joint",
        },
        "oof_alpha": PATHS.relative(OOF_ALPHA),
        "seconds": time.time() - started,
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
