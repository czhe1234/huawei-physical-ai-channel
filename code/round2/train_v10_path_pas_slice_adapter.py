#!/usr/bin/env python3
"""Adapt the full path PAS gate to the actual V10 OOF carrier domain."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from analyze_joint_path_token_oracle import OOF_META
from analyze_v11_explicit_path_pas_upper_bound import build_raw_pas, v10_oof_channel
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution, truth_views
from prepare_sionna_calibration_dataset import TRUTH_PAS
from project_paths import PATHS
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics
from train_full_path_pas_slice_gate import (
    ARTIFACT_DIR as PRETRAIN_DIR,
    FOLD_IDS,
    OOF_PATH_CACHE,
    POINT_CONTEXT,
    PasSliceGate,
    checkpoint_path as pretrain_checkpoint_path,
    cosine_slices,
    normalize_pas,
    predict_alpha,
    project_joint,
    slice_coordinates,
    spectral_features,
)


VERSION = "Round2-E70-V10-carrier-path-PAS-slice-adapter"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v10_path_pas_slice_adapter"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v10_path_pas_slice_adapter"
OOF_ALPHA = ARTIFACT_DIR / "oof_alpha_f16.npy"
REPORT = ANALYSIS_DIR / "v10_path_pas_slice_adapter.json"
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--gate-penalty", type=float, default=5e-4)
    parser.add_argument("--smoothness-penalty", type=float, default=2e-4)
    parser.add_argument("--bias-shift", type=float, default=-1.0)
    parser.add_argument("--random-state", type=int, default=2026080770)
    return parser.parse_args()


def checkpoint_path(fold: int) -> object:
    return ARTIFACT_DIR / f"fold{fold}_v10_pas_adapter.pt"


def averaged_pretrained(device: torch.device) -> tuple[dict[str, torch.Tensor], dict]:
    checkpoints = [
        torch.load(pretrain_checkpoint_path(fold), map_location="cpu", weights_only=False)
        for fold in range(5)
    ]
    state = {}
    for key in checkpoints[0]["state_dict"]:
        state[key] = torch.stack(
            [checkpoint["state_dict"][key].float() for checkpoint in checkpoints]
        ).mean(dim=0)
    return {key: value.to(device) for key, value in state.items()}, checkpoints[0]


def train_fold(
    fold: int,
    args: argparse.Namespace,
    fold_id: np.ndarray,
    base_pas: np.ndarray,
    raw_pas: np.ndarray,
    truth_pas: np.ndarray,
    context: np.ndarray,
    coordinates: torch.Tensor,
    pretrained_state: dict[str, torch.Tensor],
    pretrained_meta: dict,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    fit = np.flatnonzero(fold_id != fold).astype(np.int64)
    held = np.flatnonzero(fold_id == fold).astype(np.int64)
    context_mean = context[fit].mean(axis=0).astype(np.float32)
    context_std = context[fit].std(axis=0).astype(np.float32)
    context_std[context_std < 1e-5] = 1.0
    config = pretrained_meta["config"]
    model = PasSliceGate(
        int(pretrained_meta["input_dim"]),
        int(config["hidden"]),
        float(config["dropout"]),
    ).to(device)
    model.load_state_dict(pretrained_state)
    with torch.no_grad():
        model.network[-1].bias.add_(args.bias_shift)
    for index in (0, 2, 4):
        for parameter in model.network[index].parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.1
    )
    rng = np.random.default_rng(args.random_state + 1000 * fold)
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(fit)
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
                (context[ids] - context_mean) / context_std, device=device
            )
            alpha = model(
                spectral_features(base, raw, local_context, coordinates)
            ).reshape(len(ids), 4, 192)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_cosine += float(cosine.mean().detach().item())
            total_alpha += float(alpha.mean().detach().item())
            batches += 1
        scheduler.step()
        history.append(
            {
                "epoch": epoch + 1,
                "fit_pas_cosine": total_cosine / max(batches, 1),
                "fit_alpha_mean": total_alpha / max(batches, 1),
            }
        )
        if (epoch + 1) % 2 == 0 or epoch + 1 == args.epochs:
            print(
                f"V10 PAS adapter fold {fold} epoch {epoch + 1}/{args.epochs} "
                f"cosine={history[-1]['fit_pas_cosine']:.6f} "
                f"alpha={history[-1]['fit_alpha_mean']:.4f}",
                flush=True,
            )
    prediction = predict_alpha(
        model,
        held,
        base_pas,
        raw_pas,
        context,
        context_mean,
        context_std,
        coordinates,
        device,
        args.batch_size,
    )
    path = checkpoint_path(fold)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "context_mean": context_mean,
            "context_std": context_std,
            "input_dim": int(pretrained_meta["input_dim"]),
            "config": {**config, **vars(args)},
        },
        path,
    )
    return prediction, {
        "fold": fold,
        "fit_count": int(len(fit)),
        "held_count": int(len(held)),
        "checkpoint": PATHS.relative(path),
        "history": history,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    required = (
        OOF_PATH_CACHE,
        OOF_META,
        FOLD_IDS,
        POINT_CONTEXT,
        TRUTH_PAS,
        *tuple(pretrain_checkpoint_path(fold) for fold in range(5)),
    )
    missing = [PATHS.relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    rows = np.load(OOF_PATH_CACHE / "rows.npy").astype(np.int64)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)[rows]
    global_ids = query_ids[rows]
    base = v10_oof_channel(device)
    base_pas = pas_distribution(torch.as_tensor(base, device=device)).cpu().numpy()
    raw_pas = build_raw_pas(
        argparse.Namespace(batch_size=args.batch_size, force=False), device
    )
    truth_pas_all = np.load(TRUTH_PAS, mmap_mode="r")
    truth_pas = np.asarray(truth_pas_all[global_ids], dtype=np.float16)
    full_context = np.load(POINT_CONTEXT, mmap_mode="r")
    context = np.asarray(full_context[global_ids], dtype=np.float32)
    coordinates = torch.as_tensor(slice_coordinates(), device=device)
    pretrained_state, pretrained_meta = averaged_pretrained(device)

    alpha = np.zeros((len(rows), 4, 192), dtype=np.float32)
    training = []
    for fold in range(5):
        held = np.flatnonzero(fold_id == fold)
        prediction, diagnostics = train_fold(
            fold,
            args,
            fold_id,
            base_pas,
            raw_pas,
            truth_pas,
            context,
            coordinates,
            pretrained_state,
            pretrained_meta,
            device,
        )
        alpha[held] = prediction
        training.append(diagnostics)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    np.save(OOF_ALPHA, alpha.astype(np.float16))

    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    truth = np.asarray(truth_all[global_ids], dtype=np.complex64)
    base_tensor = torch.as_tensor(base, device=device)
    truth_tensor = torch.as_tensor(truth, device=device)
    truth_pas_tensor, truth_pdp_tensor = truth_views(truth_tensor)
    base_pas_tensor = torch.as_tensor(base_pas, device=device)
    raw_pas_tensor = torch.as_tensor(
        np.asarray(raw_pas, dtype=np.float32), device=device
    )
    base_pdp = pdp_distribution(base_tensor)
    baseline_values = point_metrics(
        base_tensor, truth_tensor, truth_pas_tensor, truth_pdp_tensor
    )
    candidates = []
    metric_cache = {}
    for scale in (0.25, 0.50, 0.75, 1.00, 1.25):
        local_alpha = torch.clamp(
            scale * torch.as_tensor(alpha, device=device), 0.0, 1.0
        )
        desired_pas = normalize_pas(
            (1.0 - local_alpha[:, None, None, None]) * base_pas_tensor
            + local_alpha[:, None, None, None] * raw_pas_tensor
        )
        for order in ("pas_pdp", "pdp_pas"):
            for cycles in (1, 2, 4):
                with torch.inference_mode():
                    prediction = project_joint(
                        base_tensor, desired_pas, base_pdp, cycles, order
                    )
                    values = point_metrics(
                        prediction, truth_tensor, truth_pas_tensor, truth_pdp_tensor
                    )
                name = f"scale{scale:.2f}_{order}_c{cycles}"
                candidates.append(
                    {
                        "candidate": name,
                        "scale": scale,
                        "order": order,
                        "cycles": cycles,
                        **statistics(*values),
                    }
                )
                metric_cache[name] = values
    candidates.sort(key=lambda row: float(row["score"]), reverse=True)
    selected = candidates[0]
    selected_values = metric_cache[str(selected["candidate"])]
    baseline = statistics(*baseline_values)
    fold_rows = []
    for fold in range(5):
        mask = fold_id == fold
        result = statistics(*selected_values, mask)
        anchor = statistics(*baseline_values, mask)
        fold_rows.append(
            {
                "fold": fold,
                "baseline": anchor,
                "result": result,
                "delta": float(result["score"] - anchor["score"]),
            }
        )
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "configuration": vars(args),
        "point_count": int(len(rows)),
        "training": training,
        "baseline": baseline,
        "selected": selected,
        "fold_results": fold_rows,
        "positive_folds": int(sum(row["delta"] > 0.0 for row in fold_rows)),
        "alpha": {
            "mean": float(np.mean(alpha)),
            "median": float(np.median(alpha)),
            "p90": float(np.quantile(alpha, 0.90)),
            "fraction_below_0.05": float(np.mean(alpha < 0.05)),
        },
        "top_candidates": candidates[:20],
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
