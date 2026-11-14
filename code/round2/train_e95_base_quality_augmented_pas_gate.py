#!/usr/bin/env python3
"""Train an E88 PAS gate across a curriculum of base-spectrum qualities."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import torch

from prepare_sionna_calibration_dataset import TRUTH_PAS
from project_paths import PATHS
from train_full_path_pas_slice_gate import (
    BASE_PAS,
    PasSliceGate,
    slice_coordinates,
    spectral_features,
)
from train_full_path_pdp_link_gate import FOLD_IDS
import train_e88_dense_path_residual_student as e88
import train_e90_e88_slice_reliability_gates as e90
import train_e92_full4000_e88_transfer_gates as e92
import train_offline_sionna_geometry_student as e87
import train_surface_interaction_energy_model as e85


VERSION = "Round2-E95-base-quality-augmented-PAS-gate"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "base_quality_augmented_pas_gate"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_base_quality_augmented_pas_gate"
ALPHA_OUTPUT = ARTIFACT_DIR / "fixed500_augmented_pas_alpha.npz"
REPORT = ANALYSIS_DIR / "base_quality_augmented_pas_gate.json"
QUALITY_LEVELS = (0.0, 0.1, 0.2, 0.3)
PRIMARY = {
    "pas_source": "augmented_gate",
    "pas_scale": 1.0,
    "pdp_source": "E92_gate",
    "pdp_scale": 0.25,
    "order": "pas_pdp",
    "cycles": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=320)
    return parser.parse_args()


def checkpoint_path(fold: int) -> Any:
    return ARTIFACT_DIR / f"fold{fold}_augmented_pas_gate.pt"


def train_fold(
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
) -> dict[str, Any]:
    train_points = np.flatnonzero((fold_id != fold) & nonzero)
    normalized_context, mean, std = e92.normalize_context(context, train_points)
    sample = torch.zeros((1, 2, 16, 8, 4, 192), device=device)
    sample_context = torch.zeros((1, context.shape[1]), device=device)
    input_dim = spectral_features(
        sample, sample, sample_context, coordinates
    ).shape[-1]
    seed = 2026080795 + 1000 * fold
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = PasSliceGate(input_dim, args.hidden, 0.10).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=8e-4, weight_decay=1e-4, fused=device.type == "cuda"
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=8e-5
    )
    rng = np.random.default_rng(seed + 1)
    history = []
    quality_array = np.asarray(QUALITY_LEVELS, dtype=np.float32)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train_points)
        scores = []
        alphas = []
        quality_alpha: dict[str, list[float]] = {
            f"{value:.1f}": [] for value in QUALITY_LEVELS
        }
        for start in range(0, len(order), args.batch_size):
            ids = order[start : start + args.batch_size]
            base = torch.as_tensor(
                np.asarray(base_pas[ids], dtype=np.float32), device=device
            )
            truth = torch.as_tensor(
                np.asarray(truth_pas[ids], dtype=np.float32), device=device
            )
            raw = torch.as_tensor(endpoint_pas[ids].astype(np.float32), device=device)
            raw = raw[..., None].expand(-1, -1, -1, -1, -1, 192)
            selected_quality = rng.choice(quality_array, size=len(ids), replace=True)
            quality = torch.as_tensor(selected_quality, device=device).reshape(
                len(ids), 1, 1, 1, 1, 1
            )
            augmented_base = e90.root_blend(base, truth, quality, (1, 2, 3))
            local_context = torch.as_tensor(normalized_context[ids], device=device)
            features = spectral_features(
                augmented_base, raw, local_context, coordinates
            )
            alpha = model(features).reshape(len(ids), 4, 192)
            desired = e90.root_blend(
                augmented_base, raw, alpha[:, None, None, None], (1, 2, 3)
            )
            cosine = e90.cosine_pas(desired, truth)
            smoothness = (alpha[:, :, 1:] - alpha[:, :, :-1]).square().mean()
            loss = -cosine.mean() + 3e-4 * alpha.mean() + 1e-4 * smoothness
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            if not torch.isfinite(gradient):
                raise FloatingPointError(
                    f"non-finite E95 PAS gradient fold={fold} epoch={epoch}"
                )
            optimizer.step()
            scores.append(float(cosine.mean().detach().item()))
            point_alpha = alpha.mean(dim=(1, 2)).detach().cpu().numpy()
            for value in QUALITY_LEVELS:
                mask = selected_quality == np.float32(value)
                if np.any(mask):
                    quality_alpha[f"{value:.1f}"].extend(point_alpha[mask].tolist())
            alphas.append(float(alpha.mean().detach().item()))
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "cosine": float(np.mean(scores)),
                "alpha": float(np.mean(alphas)),
                "alpha_by_quality": {
                    key: float(np.mean(value)) if value else None
                    for key, value in quality_alpha.items()
                },
            }
        )
        if epoch % 3 == 0 or epoch == args.epochs:
            print(
                f"E95 PAS fold={fold} epoch={epoch}/{args.epochs} "
                f"cosine={history[-1]['cosine']:.6f} "
                f"alpha={history[-1]['alpha']:.4f} "
                f"by_quality={history[-1]['alpha_by_quality']}",
                flush=True,
            )
    checkpoint = checkpoint_path(fold)
    torch.save(
        {
            "experiment": VERSION,
            "fold": fold,
            "input_dim": input_dim,
            "hidden": args.hidden,
            "quality_levels": QUALITY_LEVELS,
            "state_dict": model.state_dict(),
            "context_mean": mean,
            "context_std": std,
        },
        checkpoint,
    )
    return {
        "fold": fold,
        "train_count": int(len(train_points)),
        "history": history,
        "checkpoint": PATHS.relative(checkpoint),
    }


@torch.inference_mode()
def predict_fixed_alpha(
    fold_id: np.ndarray,
    query_ids: np.ndarray,
    context: np.ndarray,
    base_pas: torch.Tensor,
    endpoint_pas: torch.Tensor,
    coordinates: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    output = torch.zeros((len(query_ids), 4, 192), device=device)
    query_gate_fold = fold_id[query_ids]
    assigned = np.zeros(len(query_ids), dtype=bool)
    for fold in range(5):
        rows = np.flatnonzero(query_gate_fold == fold)
        assigned[rows] = True
        payload = torch.load(
            checkpoint_path(fold), map_location="cpu", weights_only=False
        )
        model = PasSliceGate(
            payload["input_dim"], payload["hidden"], 0.10
        ).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        mean = torch.as_tensor(payload["context_mean"], device=device)
        std = torch.as_tensor(payload["context_std"], device=device)
        for start in range(0, len(rows), args.batch_size):
            local_rows = rows[start : start + args.batch_size]
            ids = torch.as_tensor(local_rows, device=device)
            local_context = (
                torch.as_tensor(context[local_rows], device=device) - mean
            ) / std
            features = spectral_features(
                base_pas[ids], endpoint_pas[ids], local_context, coordinates
            )
            output[ids] = model(features).reshape(len(local_rows), 4, 192)
        del model
    if not np.all(assigned):
        raise RuntimeError("some fixed rows were not assigned an E95 gate")
    return output


@torch.inference_mode()
def evaluate_action(
    name: str,
    alpha_pas: torch.Tensor,
    pas_scale: float,
    alpha_pdp: torch.Tensor,
    pdp_scale: float,
    order: str,
    base_channel: torch.Tensor,
    truth_channel: torch.Tensor,
    truth_pas: torch.Tensor,
    truth_pdp: torch.Tensor,
    base_pas: torch.Tensor,
    base_pdp: torch.Tensor,
    endpoint_pas: torch.Tensor,
    endpoint_pdp: torch.Tensor,
    baseline_values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    fold_id: np.ndarray,
) -> dict[str, Any]:
    desired_pas = e90.root_blend(
        base_pas,
        endpoint_pas,
        (pas_scale * alpha_pas)[:, None, None, None],
        (1, 2, 3),
    )
    desired_pdp = e90.root_blend(
        base_pdp,
        endpoint_pdp,
        (pdp_scale * alpha_pdp).reshape(len(base_channel), 256, 4, 1),
        (3,),
    )
    prediction = e87.project_joint(base_channel, desired_pas, desired_pdp, 4, order)
    values = e87.point_metrics(
        prediction, truth_channel, truth_pas, truth_pdp
    )
    metrics = e87.statistics(*values)
    baseline = e87.statistics(*baseline_values)
    folds = []
    for fold in range(5):
        mask = fold_id == fold
        local = e87.statistics(*values, mask)
        local_base = e87.statistics(*baseline_values, mask)
        folds.append(
            {"fold": fold, "delta": float(local["score"] - local_base["score"])}
        )
    return {
        "name": name,
        "pas_scale": pas_scale,
        "pdp_scale": pdp_scale,
        "order": order,
        "metrics": metrics,
        "delta": float(metrics["score"] - baseline["score"]),
        "positive_folds": int(sum(row["delta"] > 0.0 for row in folds)),
        "worst_fold_delta": float(min(row["delta"] for row in folds)),
        "folds": folds,
    }


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    started = time.time()
    fold_id = np.load(FOLD_IDS).astype(np.int8)
    full_endpoint = np.load(e92.FULL_ENDPOINTS, allow_pickle=False)
    if not np.array_equal(full_endpoint["fold_id"].astype(np.int8), fold_id):
        raise ValueError("E91 endpoint folds changed")
    endpoint_pas = full_endpoint["final_pas"].astype(np.float32)
    context = e92.context_features(
        full_endpoint["confidence"].astype(np.float32),
        np.arange(len(fold_id), dtype=np.int64),
    )
    power = np.asarray(np.load(e85.POINT_POWER), dtype=np.float64)
    nonzero = power > 0.0
    base_pas = np.load(BASE_PAS, mmap_mode="r")
    truth_pas = np.load(TRUTH_PAS, mmap_mode="r")
    coordinates = torch.as_tensor(slice_coordinates(), device=device)
    training = []
    for fold in range(5):
        training.append(
            train_fold(
                fold,
                fold_id,
                nonzero,
                base_pas,
                truth_pas,
                endpoint_pas,
                context,
                coordinates,
                args,
                device,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fixed_meta = np.load(e88.OOF_META, allow_pickle=False)
    query_ids = fixed_meta["query_ids"].astype(np.int64)
    fixed_fold = fixed_meta["fold_id"].astype(np.int8)
    fixed_endpoint = np.load(e88.OOF_PREDICTIONS, allow_pickle=False)
    if not np.array_equal(fixed_endpoint["query_ids"].astype(np.int64), query_ids):
        raise ValueError("fixed E88 endpoint order changed")
    fixed_context = e92.context_features(
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
    augmented_alpha = predict_fixed_alpha(
        fold_id,
        query_ids,
        fixed_context,
        fixed_base_pas,
        fixed_endpoint_pas,
        coordinates,
        args,
        device,
    )
    e92_alpha_payload = np.load(e92.FIXED_ALPHA, allow_pickle=False)
    if not np.array_equal(
        e92_alpha_payload["query_ids"].astype(np.int64), query_ids
    ):
        raise ValueError("E92 alpha order changed")
    old_alpha_pas = torch.as_tensor(
        e92_alpha_payload["alpha_pas"].astype(np.float32), device=device
    )
    alpha_pdp = torch.as_tensor(
        e92_alpha_payload["alpha_pdp"].astype(np.float32), device=device
    )
    zero_pas = torch.zeros_like(augmented_alpha)
    zero_pdp = torch.zeros_like(alpha_pdp)
    baseline_values = e87.point_metrics(
        base_channel,
        truth_channel,
        fixed_truth_pas,
        fixed_truth_pdp,
    )
    actions = [
        evaluate_action(
            "augmented_pas_only",
            augmented_alpha,
            1.0,
            zero_pdp,
            0.0,
            "pas_pdp",
            base_channel,
            truth_channel,
            fixed_truth_pas,
            fixed_truth_pdp,
            fixed_base_pas,
            fixed_base_pdp,
            fixed_endpoint_pas,
            fixed_endpoint_pdp,
            baseline_values,
            fixed_fold,
        ),
        evaluate_action(
            "primary_augmented_pas_plus_e92_pdp",
            augmented_alpha,
            1.0,
            alpha_pdp,
            0.25,
            "pas_pdp",
            base_channel,
            truth_channel,
            fixed_truth_pas,
            fixed_truth_pdp,
            fixed_base_pas,
            fixed_base_pdp,
            fixed_endpoint_pas,
            fixed_endpoint_pdp,
            baseline_values,
            fixed_fold,
        ),
        evaluate_action(
            "E93_best_stable_control",
            old_alpha_pas,
            0.125,
            alpha_pdp,
            0.25,
            "pas_pdp",
            base_channel,
            truth_channel,
            fixed_truth_pas,
            fixed_truth_pdp,
            fixed_base_pas,
            fixed_base_pdp,
            fixed_endpoint_pas,
            fixed_endpoint_pdp,
            baseline_values,
            fixed_fold,
        ),
    ]
    primary = next(
        row for row in actions if row["name"] == "primary_augmented_pas_plus_e92_pdp"
    )
    control = next(
        row for row in actions if row["name"] == "E93_best_stable_control"
    )
    np.savez_compressed(
        ALPHA_OUTPUT,
        query_ids=query_ids.astype(np.int32),
        fold_id=fixed_fold,
        gate_fold_id=fold_id[query_ids],
        alpha_pas=augmented_alpha.cpu().numpy().astype(np.float16),
    )
    report = {
        "experiment": VERSION,
        "status": "strict_complete",
        "primary_hypothesis": (
            "Training the PAS gate on a continuum of base qualities teaches it to "
            "condition E88 reliability on a strong V12-like base."
        ),
        "configuration": {
            **vars(args),
            "quality_levels": QUALITY_LEVELS,
            "primary": PRIMARY,
        },
        "training": training,
        "V12": e87.statistics(*baseline_values),
        "alpha": {
            "augmented_pas_mean": float(augmented_alpha.mean().item()),
            "E92_pas_mean": float(old_alpha_pas.mean().item()),
            "E92_pdp_mean": float(alpha_pdp.mean().item()),
        },
        "actions": actions,
        "decision": {
            "primary_passed": bool(
                primary["metrics"]["score"] > control["metrics"]["score"]
                and primary["positive_folds"] == 5
            ),
            "beats_E93_control": float(primary["metrics"]["score"] - control["metrics"]["score"]),
            "candidate_allowed": False,
        },
        "output": PATHS.relative(ALPHA_OUTPUT),
        "seconds": time.time() - started,
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
