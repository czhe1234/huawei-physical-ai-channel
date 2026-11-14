#!/usr/bin/env python3
"""Train the explicit-path PDP set model on every allowed spatial-fold fit point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from analyze_joint_path_token_oracle import OOF_META, v9_channel
from analyze_joint_path_token_pdp_isolation import evaluate_grid
from prepare_sionna_calibration_dataset import TRUTH_PDP
from project_paths import PATHS
from round2.data import load_positions
from round2.spatial_hole_cv import build_spatial_hole_folds
from run_rbf_spectral_field import statistics
from train_explicit_path_pdp_set_model import (
    PathEnergySetModel,
    batch_loss,
    feature_statistics,
    pdp_cosine,
    render_pdp,
)


VERSION = "Round2-E64-full-fit-explicit-path-pdp-set-model"
PATH_CACHE = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_oracle"
    / "n4000_s24_k128_b096_d064_scatter1_sel106"
)
FEATURES = (
    PATHS.artifacts
    / "round2"
    / "explicit_path_pdp_predictor"
    / "path_features_train_n4000.npy"
)
ARTIFACT_DIR = PATHS.artifacts / "round2" / "full_explicit_path_pdp_set_model"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_full_explicit_path_pdp_set_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--folds", default="0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--regularization", type=float, default=2e-3)
    parser.add_argument("--links-per-step", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--log-amplitude-limit", type=float, default=2.5)
    parser.add_argument("--random-state", type=int, default=106)
    return parser.parse_args()


def parse_folds(value: str) -> list[int]:
    folds = sorted({int(item) for item in value.split(",") if item.strip()})
    if not folds or any(fold < 0 or fold > 4 for fold in folds):
        raise ValueError(f"invalid folds: {value}")
    return folds


def train_one_fold(
    fold: int,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
    nonzero: np.ndarray,
    args: argparse.Namespace,
    features: np.ndarray,
    coefficient: np.ndarray,
    tau: np.ndarray,
    mask: np.ndarray,
    truth_pdp: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    train_points = fit_indices[nonzero[fit_indices]]
    point_mask = np.zeros(len(features), dtype=bool)
    point_mask[train_points] = True
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
    frequency = torch.as_tensor(
        np.arange(-96, 96, dtype=np.float32) * 30_000.0,
        device=device,
    )
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
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(
                f"full path PDP fold {fold} epoch {epoch + 1}/{args.epochs} "
                f"cosine={history[-1]['sampled_link_pdp_cosine']:.6f}",
                flush=True,
            )
    model.eval()
    prediction = np.zeros((len(query_indices), mask.shape[1]), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(query_indices), args.batch_size):
            end = min(start + args.batch_size, len(query_indices))
            local = query_indices[start:end]
            local_features = torch.as_tensor(
                (np.asarray(features[local], dtype=np.float32) - mean) / std,
                device=device,
            )
            local_mask = torch.as_tensor(np.asarray(mask[local]), device=device)
            local_features = local_features * local_mask[..., None]
            prediction[start:end] = model(local_features, local_mask).cpu().numpy()
    checkpoint = ARTIFACT_DIR / f"fold{fold}_full_set_model.pt"
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
    return prediction, {
        "fold": fold,
        "fit_count": int(len(fit_indices)),
        "train_nonzero_count": int(len(train_points)),
        "query_count": int(len(query_indices)),
        "checkpoint": PATHS.relative(checkpoint),
        "history": history,
    }


def main() -> None:
    args = parse_args()
    requested_folds = parse_folds(args.folds)
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    required = (PATH_CACHE, FEATURES, TRUTH_PDP, OOF_META)
    missing = [PATHS.relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    _, train_positions, test_positions, train_site, test_site = load_positions()
    folds = build_spatial_hole_folds(
        train_positions, test_positions, train_site, test_site
    )
    features = np.load(FEATURES, mmap_mode="r")
    coefficient = np.load(PATH_CACHE / "coefficient.npy", mmap_mode="r")
    tau = np.load(PATH_CACHE / "tau.npy", mmap_mode="r")
    mask = np.asarray(np.load(PATH_CACHE / "mask.npy", mmap_mode="r"), dtype=bool)
    truth_pdp = np.load(TRUTH_PDP, mmap_mode="r")
    train_h = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    power = np.sum(np.abs(train_h) ** 2, axis=(1, 2, 3), dtype=np.float64)
    nonzero = power > 0.0
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    query_fold = metadata["fold_id"].astype(np.int8)
    global_to_row = np.full(len(train_positions), -1, dtype=np.int64)
    global_to_row[query_ids] = np.arange(len(query_ids), dtype=np.int64)

    prediction = np.zeros((len(query_ids), mask.shape[1]), dtype=np.float32)
    trained_rows = np.zeros(len(query_ids), dtype=bool)
    training = []
    for fold in requested_folds:
        definition = folds[fold]
        values, diagnostics = train_one_fold(
            fold,
            definition.fit_indices,
            definition.query_indices,
            nonzero,
            args,
            features,
            coefficient,
            tau,
            mask,
            truth_pdp,
            device,
        )
        rows = global_to_row[definition.query_indices]
        if np.any(rows < 0):
            raise RuntimeError(f"fold {fold} query mapping is incomplete")
        prediction[rows] = values
        trained_rows[rows] = True
        training.append(diagnostics)
    local_rows = np.flatnonzero(trained_rows)
    local_ids = query_ids[local_rows]
    np.save(
        ARTIFACT_DIR / f"oof_log_amplitude_folds_{'-'.join(map(str, requested_folds))}.npy",
        prediction,
    )

    local_coefficient = coefficient[local_ids]
    local_tau = tau[local_ids]
    local_mask = mask[local_ids]
    base = v9_channel(local_rows, device, args.batch_size)
    truth = np.asarray(train_h[local_ids], dtype=np.complex64)
    truth_pdp_tensor = torch.as_tensor(
        np.asarray(truth_pdp[local_ids], dtype=np.float32), device=device
    )
    grids = []
    metrics_by_blend = {}
    endpoint_cosines = {}
    for blend in (0.0, 0.25, 0.50, 0.75, 1.0):
        path_pdp = render_pdp(
            blend * prediction[local_rows],
            local_coefficient,
            local_tau,
            local_mask,
            device,
            args.batch_size,
        )
        endpoint_cosines[f"blend_{blend:.2f}"] = float(
            pdp_cosine(
                torch.as_tensor(path_pdp, device=device).reshape(len(path_pdp), -1, 192),
                truth_pdp_tensor.reshape(len(path_pdp), -1, 192),
            ).mean().item()
        )
        projection_rows, metrics = evaluate_grid(base, truth, path_pdp, device)
        for candidate in projection_rows:
            candidate["amplitude_blend"] = blend
            candidate["compound_name"] = f"blend{blend:.2f}_{candidate['candidate']}"
        grids.extend(projection_rows)
        metrics_by_blend[f"blend_{blend:.2f}"] = metrics
        print(f"full path PDP blend {blend:.2f} evaluated", flush=True)
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
    fold_results = []
    for fold in requested_folds:
        local = query_fold[local_rows] == fold
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
        fold_results.append(
            {
                "fold": fold,
                "result": current,
                "baseline": baseline,
                "delta": float(current["score"] - baseline["score"]),
            }
        )
    fold_tag = "-".join(map(str, requested_folds))
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "requested_folds": requested_folds,
        "configuration": vars(args),
        "point_count": int(len(local_rows)),
        "endpoint_cosines": endpoint_cosines,
        "training": training,
        "selected": selected,
        "fold_results": fold_results,
        "positive_folds": int(sum(row["delta"] > 0.0 for row in fold_results)),
        "top_candidates": grids[:20],
        "seconds": float(time.time() - started),
    }
    report_path = ANALYSIS_DIR / f"full_explicit_path_pdp_set_model_folds_{fold_tag}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
