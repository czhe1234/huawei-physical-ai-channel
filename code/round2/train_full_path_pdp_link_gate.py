#!/usr/bin/env python3
"""Train a 4000-point cross-fitted gate from V9 PDP to explicit-path PDP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
import torch

from analyze_joint_path_token_oracle import FREQUENCIES, OOF_META, v9_channel
from analyze_v4_exact_joint_projection import pdp_distribution, truth_views
from prepare_sionna_calibration_dataset import TRUTH_PDP
from project_paths import PATHS
from round2.data import load_positions
from run_rbf_spectral_field import solve_linear_rbf_weights, statistics
from run_v9_joint_sionna_migration import point_metrics
from train_path_pdp_link_gate import (
    LinkGate,
    cosine_by_link,
    link_coordinates,
    project,
    spectral_features,
)


VERSION = "Round2-E66-full-4000-path-PDP-link-gate"
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
TOKEN_FEATURES = (
    PATHS.artifacts
    / "round2"
    / "explicit_path_pdp_predictor"
    / "path_features_train_n4000.npy"
)
TRAIN_REGIME = (
    PATHS.artifacts
    / "round2"
    / "explicit_path_pdp_predictor"
    / "path_regime_train_n4000.npy"
)
RELIABILITY = (
    PATHS.artifacts
    / "round2"
    / "v9_joint_sionna_endpoints"
    / "train_pdp_spectral_reliability.npy"
)
E65_DIR = PATHS.artifacts / "round2" / "path_pdp_link_gate"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "full_path_pdp_link_gate"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_full_path_pdp_link_gate"
FOLD_IDS = ARTIFACT_DIR / "train_fold_id.npy"
FOLD_REPORT = ANALYSIS_DIR / "matched_spatial_folds.json"
BASE_PDP = ARTIFACT_DIR / "train_rbf_base_pdp_f16.npy"
RAW_PDP = ARTIFACT_DIR / "train_raw_path_pdp_f16.npy"
POINT_CONTEXT = ARTIFACT_DIR / "train_point_context.npy"
OOF_ALPHA = ARTIFACT_DIR / "train_oof_alpha_f16.npy"
TRAIN_REPORT = ANALYSIS_DIR / "full_path_pdp_link_gate_train.json"
V9_REPORT = ANALYSIS_DIR / "full_path_pdp_link_gate_v9_oof.json"
V9_ALPHA = ARTIFACT_DIR / "v9_oof_pretrained_alpha_f16.npy"
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("build", "train", "evaluate", "all"), default="all"
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--micro-clusters", type=int, default=160)
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--smoothing", type=float, default=1e-5)
    parser.add_argument("--build-batch-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gate-penalty", type=float, default=2e-4)
    parser.add_argument("--hidden", type=int, default=320)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--random-state", type=int, default=2026080666)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalize_links(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = value.sum(axis=-1, keepdims=True, dtype=np.float64)
    return np.divide(
        value,
        np.maximum(total, TINY),
        out=np.zeros_like(value),
        where=total > 0.0,
    )


def build_fold_ids(args: argparse.Namespace) -> np.ndarray:
    if FOLD_IDS.is_file() and not args.force:
        return np.load(FOLD_IDS)
    _, train_positions, test_positions, train_site, test_site = load_positions()
    fold_id = np.full(len(train_positions), -1, dtype=np.int8)
    rows: list[dict[str, object]] = []
    for site in sorted(np.unique(train_site).tolist()):
        site_ids = np.flatnonzero(train_site == site)
        xy = np.asarray(train_positions[site_ids, :2], dtype=np.float64)
        model = KMeans(
            n_clusters=args.micro_clusters,
            random_state=args.random_state + int(site),
            n_init=16,
        ).fit(xy)
        order = np.lexsort(
            (model.cluster_centers_[:, 1], model.cluster_centers_[:, 0])
        )
        cluster_fold = np.empty(args.micro_clusters, dtype=np.int8)
        cluster_fold[order] = np.arange(args.micro_clusters, dtype=np.int64) % 5
        fold_id[site_ids] = cluster_fold[model.labels_]
    if np.any(fold_id < 0):
        raise RuntimeError("fold assignment is incomplete")
    for fold in range(5):
        for site in sorted(np.unique(train_site).tolist()):
            query = np.flatnonzero((fold_id == fold) & (train_site == site))
            fit = np.flatnonzero((fold_id != fold) & (train_site == site))
            distance = cKDTree(train_positions[fit, :2]).query(
                train_positions[query, :2], k=1, workers=-1
            )[0]
            test_query = np.flatnonzero(test_site == site)
            test_distance = cKDTree(train_positions[train_site == site, :2]).query(
                test_positions[test_query, :2], k=1, workers=-1
            )[0]
            rows.append(
                {
                    "fold": fold,
                    "site": int(site),
                    "query_count": int(len(query)),
                    "nearest_median_m": float(np.median(distance)),
                    "nearest_p90_m": float(np.quantile(distance, 0.90)),
                    "test_nearest_median_m": float(np.median(test_distance)),
                    "test_nearest_p90_m": float(np.quantile(test_distance, 0.90)),
                }
            )
    np.save(FOLD_IDS, fold_id)
    FOLD_REPORT.parent.mkdir(parents=True, exist_ok=True)
    FOLD_REPORT.write_text(
        json.dumps(
            {
                "experiment": VERSION,
                "micro_clusters_per_site": args.micro_clusters,
                "folds": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return fold_id


def rbf_donors(
    fold_id: np.ndarray,
    train_positions: np.ndarray,
    train_site: np.ndarray,
    valid: np.ndarray,
    neighbors: int,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    donor_ids = np.empty((len(train_positions), neighbors), dtype=np.int32)
    weights = np.empty((len(train_positions), neighbors), dtype=np.float32)
    for fold in range(5):
        for site in sorted(np.unique(train_site).tolist()):
            query = np.flatnonzero((fold_id == fold) & (train_site == site))
            bank = np.flatnonzero(
                (fold_id != fold) & (train_site == site) & valid
            )
            if len(bank) < neighbors:
                raise RuntimeError(f"fold {fold} site {site} has too few donors")
            _, local = cKDTree(train_positions[bank, :2]).query(
                train_positions[query, :2], k=neighbors, workers=-1
            )
            selected = bank[local]
            donor_ids[query] = selected
            for row, point_id in enumerate(query):
                weights[point_id] = solve_linear_rbf_weights(
                    train_positions[selected[row], :2],
                    train_positions[point_id, :2],
                    smoothing,
                )
        print(f"RBF donor weights fold {fold + 1}/5", flush=True)
    return donor_ids, weights


def build_rbf_base(args: argparse.Namespace, fold_id: np.ndarray) -> None:
    if BASE_PDP.is_file() and not args.force:
        return
    _, train_positions, _, train_site, _ = load_positions()
    train_h = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    power = np.sum(np.abs(train_h) ** 2, axis=(1, 2, 3), dtype=np.float64)
    valid = power > 0.0
    truth = np.load(TRUTH_PDP, mmap_mode="r")
    donor_ids, weights = rbf_donors(
        fold_id,
        train_positions,
        train_site,
        valid,
        args.neighbors,
        args.smoothing,
    )
    output = np.lib.format.open_memmap(
        BASE_PDP, mode="w+", dtype=np.float16, shape=truth.shape
    )
    for start in range(0, len(truth), args.build_batch_size):
        end = min(start + args.build_batch_size, len(truth))
        donor = np.asarray(truth[donor_ids[start:end]], dtype=np.float32)
        root = np.sqrt(np.maximum(donor, 0.0))
        interpolated = np.einsum(
            "bk,bkmus->bmus", weights[start:end], root, optimize=True
        )
        output[start:end] = normalize_links(
            np.square(np.maximum(interpolated, 0.0))
        ).astype(np.float16)
        if end % 100 == 0 or end == len(truth):
            output.flush()
            print(f"RBF base PDP {end}/{len(truth)}", flush=True)
    output.flush()


def build_raw_path_pdp(args: argparse.Namespace, device: torch.device) -> None:
    if RAW_PDP.is_file() and not args.force:
        return
    coefficient = np.load(PATH_CACHE / "coefficient.npy", mmap_mode="r")
    tau = np.load(PATH_CACHE / "tau.npy", mmap_mode="r")
    mask = np.load(PATH_CACHE / "mask.npy", mmap_mode="r")
    output = np.lib.format.open_memmap(
        RAW_PDP,
        mode="w+",
        dtype=np.float16,
        shape=(len(coefficient), 256, 4, 192),
    )
    frequency = torch.as_tensor(FREQUENCIES, device=device, dtype=torch.float32)
    for start in range(0, len(coefficient), args.build_batch_size):
        end = min(start + args.build_batch_size, len(coefficient))
        coefficient_tensor = torch.as_tensor(
            np.array(coefficient[start:end], copy=True), device=device
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
        with torch.inference_mode():
            local = pdp_distribution(channel).cpu().numpy()
        output[start:end] = local.astype(np.float16)
        if end % 100 == 0 or end == len(coefficient):
            output.flush()
            print(f"raw path PDP {end}/{len(coefficient)}", flush=True)
    output.flush()


def build_point_context(args: argparse.Namespace) -> np.ndarray:
    if POINT_CONTEXT.is_file() and not args.force:
        return np.load(POINT_CONTEXT, mmap_mode="r")
    features = np.load(TOKEN_FEATURES, mmap_mode="r")
    mask = np.asarray(np.load(PATH_CACHE / "mask.npy", mmap_mode="r"), dtype=bool)
    token_mean = np.empty((len(features), features.shape[2]), dtype=np.float32)
    for start in range(0, len(features), 64):
        end = min(start + 64, len(features))
        valid = mask[start:end, :, None].astype(np.float32)
        token_mean[start:end] = (
            np.asarray(features[start:end], dtype=np.float32) * valid
        ).sum(axis=1) / valid.sum(axis=1).clip(min=1.0)
    setup, train_positions, _, train_site, _ = load_positions()
    xy = train_positions[:, :2]
    center = xy.mean(axis=0)
    scale = xy.std(axis=0) + 1e-6
    relative = xy - setup.bs_positions[train_site, :2]
    relative_scale = relative.std(axis=0) + 1e-6
    bearing = np.arctan2(relative[:, 1], relative[:, 0])
    regime = np.load(TRAIN_REGIME).astype(np.int64)
    reliability = np.asarray(np.load(RELIABILITY, mmap_mode="r"), dtype=np.float32)
    extra = np.concatenate(
        (
            ((xy - center) / scale).astype(np.float32),
            (relative / relative_scale).astype(np.float32),
            np.sin(bearing)[:, None].astype(np.float32),
            np.cos(bearing)[:, None].astype(np.float32),
            np.eye(2, dtype=np.float32)[train_site],
            np.eye(4, dtype=np.float32)[regime],
            reliability,
        ),
        axis=1,
    )
    context = np.concatenate((token_mean, extra), axis=1).astype(np.float32)
    np.save(POINT_CONTEXT, context)
    return np.load(POINT_CONTEXT, mmap_mode="r")


def checkpoint_path(fold: int) -> Path:
    return ARTIFACT_DIR / f"fold{fold}_full_link_gate.pt"


def train_fold(
    fold: int,
    args: argparse.Namespace,
    fold_id: np.ndarray,
    nonzero: np.ndarray,
    base_pdp: np.ndarray,
    raw_pdp: np.ndarray,
    truth_pdp: np.ndarray,
    context: np.ndarray,
    link: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    train_points = np.flatnonzero((fold_id != fold) & nonzero).astype(np.int64)
    held_points = np.flatnonzero((fold_id == fold) & nonzero).astype(np.int64)
    context_mean = context[train_points].mean(axis=0).astype(np.float32)
    context_std = context[train_points].std(axis=0).astype(np.float32)
    context_std[context_std < 1e-5] = 1.0
    input_dim = 24 * 4 + 3 + context.shape[1] + link.shape[1]
    model = LinkGate(input_dim, args.hidden, args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.1
    )
    link_tensor = torch.as_tensor(link, device=device)
    rng = np.random.default_rng(args.random_state + 1000 * fold)
    history = []
    model.train()
    for epoch in range(args.epochs):
        order = rng.permutation(train_points)
        total_cosine = 0.0
        total_alpha = 0.0
        batches = 0
        for start in range(0, len(order), args.batch_size):
            ids = order[start : start + args.batch_size]
            base = torch.as_tensor(
                np.asarray(base_pdp[ids], dtype=np.float32).reshape(
                    len(ids), 1024, 192
                ),
                device=device,
            )
            raw = torch.as_tensor(
                np.asarray(raw_pdp[ids], dtype=np.float32).reshape(
                    len(ids), 1024, 192
                ),
                device=device,
            )
            truth = torch.as_tensor(
                np.asarray(truth_pdp[ids], dtype=np.float32).reshape(
                    len(ids), 1024, 192
                ),
                device=device,
            )
            local_context = torch.as_tensor(
                (np.asarray(context[ids]) - context_mean) / context_std,
                device=device,
            )
            features = spectral_features(base, raw, local_context, link_tensor)
            alpha = model(features)
            desired = (1.0 - alpha[..., None]) * base + alpha[..., None] * raw
            desired = desired / desired.sum(dim=2, keepdim=True).clamp_min(TINY)
            cosine = cosine_by_link(desired, truth)
            loss = -cosine.mean() + args.gate_penalty * alpha.mean()
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
                "train_pdp_cosine": total_cosine / max(batches, 1),
                "train_alpha_mean": total_alpha / max(batches, 1),
            }
        )
        if (epoch + 1) % 6 == 0 or epoch + 1 == args.epochs:
            print(
                f"full link gate fold {fold} epoch {epoch + 1}/{args.epochs} "
                f"cosine={history[-1]['train_pdp_cosine']:.6f} "
                f"alpha={history[-1]['train_alpha_mean']:.4f}",
                flush=True,
            )
    prediction = predict_alpha(
        model,
        held_points,
        base_pdp,
        raw_pdp,
        context,
        context_mean,
        context_std,
        link_tensor,
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
            (np.asarray(context[ids]) - context_mean) / context_std,
            device=device,
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
    return model, checkpoint


def train_models(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    fold_id = np.load(FOLD_IDS)
    base_pdp = np.load(BASE_PDP, mmap_mode="r")
    raw_pdp = np.load(RAW_PDP, mmap_mode="r")
    truth_pdp = np.load(TRUTH_PDP, mmap_mode="r")
    context = build_point_context(args)
    train_h = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    power = np.sum(np.abs(train_h) ** 2, axis=(1, 2, 3), dtype=np.float64)
    nonzero = power > 0.0
    link = link_coordinates()
    alpha = np.zeros((len(fold_id), 1024), dtype=np.float32)
    training = []
    for fold in range(5):
        held = np.flatnonzero((fold_id == fold) & nonzero)
        values, diagnostics = train_fold(
            fold,
            args,
            fold_id,
            nonzero,
            base_pdp,
            raw_pdp,
            truth_pdp,
            context,
            link,
            device,
        )
        alpha[held] = values
        training.append(diagnostics)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    np.save(OOF_ALPHA, alpha.astype(np.float16))
    valid_ids = np.flatnonzero(nonzero)
    baseline_scores = []
    selected_scores = []
    scale_rows = []
    for scale in (0.25, 0.50, 0.75, 1.00, 1.25, 1.50):
        fold_rows = []
        all_baseline = []
        all_selected = []
        for fold in range(5):
            ids = np.flatnonzero((fold_id == fold) & nonzero)
            base = torch.as_tensor(
                np.asarray(base_pdp[ids], dtype=np.float32).reshape(len(ids), 1024, 192),
                device=device,
            )
            raw = torch.as_tensor(
                np.asarray(raw_pdp[ids], dtype=np.float32).reshape(len(ids), 1024, 192),
                device=device,
            )
            truth = torch.as_tensor(
                np.asarray(truth_pdp[ids], dtype=np.float32).reshape(len(ids), 1024, 192),
                device=device,
            )
            local_alpha = torch.clamp(
                scale * torch.as_tensor(alpha[ids], device=device), 0.0, 1.0
            )
            desired = (1.0 - local_alpha[..., None]) * base + local_alpha[..., None] * raw
            desired = desired / desired.sum(dim=2, keepdim=True).clamp_min(TINY)
            baseline = cosine_by_link(base, truth).mean().item()
            selected = cosine_by_link(desired, truth).mean().item()
            fold_rows.append(
                {
                    "fold": fold,
                    "baseline": float(baseline),
                    "selected": float(selected),
                    "delta": float(selected - baseline),
                }
            )
            all_baseline.append(baseline * len(ids))
            all_selected.append(selected * len(ids))
        base_mean = float(sum(all_baseline) / len(valid_ids))
        selected_mean = float(sum(all_selected) / len(valid_ids))
        scale_rows.append(
            {
                "scale": scale,
                "baseline": base_mean,
                "selected": selected_mean,
                "delta": selected_mean - base_mean,
                "positive_folds": int(sum(row["delta"] > 0.0 for row in fold_rows)),
                "folds": fold_rows,
            }
        )
        baseline_scores.append(base_mean)
        selected_scores.append(selected_mean)
    winner = max(scale_rows, key=lambda row: float(row["delta"]))
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "configuration": vars(args),
        "nonzero_points": int(np.sum(nonzero)),
        "training": training,
        "scale_grid": scale_rows,
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


def evaluate_v9(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    test_hole_fold = metadata["fold_id"].astype(np.int8)
    rows = np.load(OOF_PATH_CACHE / "rows.npy").astype(np.int64)
    global_ids = query_ids[rows]
    broad_fold = np.load(FOLD_IDS)[global_ids]
    context = np.load(POINT_CONTEXT, mmap_mode="r")
    raw_pdp = np.load(E65_DIR / "raw_path_pdp_f16.npy", mmap_mode="r")
    base_pdp = np.load(E65_DIR / "v9_base_pdp_f16.npy", mmap_mode="r")
    base_channel = np.load(E65_DIR / "v9_base_channel.npy", mmap_mode="r")
    link_tensor = torch.as_tensor(link_coordinates(), device=device)
    alpha = np.zeros((len(rows), 1024), dtype=np.float32)
    for fold in range(5):
        local_rows = np.flatnonzero(broad_fold == fold)
        if not len(local_rows):
            continue
        model, checkpoint = load_model(fold, device)
        local_context = np.asarray(context[global_ids[local_rows]], dtype=np.float32)
        proxy_context = np.lib.format.open_memmap(
            ARTIFACT_DIR / f"_context_fold{fold}.npy",
            mode="w+",
            dtype=np.float32,
            shape=local_context.shape,
        )
        proxy_context[...] = local_context
        alpha[local_rows] = predict_alpha(
            model,
            np.arange(len(local_rows), dtype=np.int64),
            np.asarray(base_pdp[local_rows]),
            np.asarray(raw_pdp[local_rows]),
            proxy_context,
            np.asarray(checkpoint["context_mean"], dtype=np.float32),
            np.asarray(checkpoint["context_std"], dtype=np.float32),
            link_tensor,
            device,
            args.batch_size,
        )
        del proxy_context
        (ARTIFACT_DIR / f"_context_fold{fold}.npy").unlink(missing_ok=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    np.save(V9_ALPHA, alpha.astype(np.float16))
    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    truth_channel = np.asarray(truth_all[global_ids], dtype=np.complex64)
    base_tensor = torch.as_tensor(np.asarray(base_channel), device=device)
    truth_tensor = torch.as_tensor(truth_channel, device=device)
    truth_pas, truth_pdp = truth_views(truth_tensor)
    base_pdp_tensor = torch.as_tensor(
        np.asarray(base_pdp, dtype=np.float32), device=device
    )
    raw_pdp_tensor = torch.as_tensor(
        np.asarray(raw_pdp, dtype=np.float32), device=device
    )
    result_rows = []
    metric_cache = {}
    for scale in (0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00):
        local_alpha = torch.clamp(
            scale * torch.as_tensor(alpha, device=device), 0.0, 1.0
        ).reshape(len(rows), 256, 4)
        desired = (
            (1.0 - local_alpha[..., None]) * base_pdp_tensor
            + local_alpha[..., None] * raw_pdp_tensor
        )
        desired = desired / desired.sum(dim=3, keepdim=True).clamp_min(TINY)
        for order in ("pas_pdp", "pdp_pas"):
            for cycles in (1, 2, 4, 8):
                with torch.inference_mode():
                    prediction = project(base_tensor, desired, cycles, order)
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
    baseline_values = point_metrics(
        base_tensor, truth_tensor, truth_pas, truth_pdp
    )
    baseline = statistics(*baseline_values)
    result_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    selected = result_rows[0]
    selected_values = metric_cache[str(selected["candidate"])]
    fold_results = []
    for fold in range(5):
        mask = test_hole_fold[rows] == fold
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
    V9_REPORT.write_text(
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
        TOKEN_FEATURES,
        TRAIN_REGIME,
        RELIABILITY,
        TRUTH_PDP,
        OOF_META,
    )
    missing = [PATHS.relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    fold_id = build_fold_ids(args)
    if args.phase in ("build", "all"):
        build_rbf_base(args, fold_id)
        build_raw_path_pdp(args, device)
        build_point_context(args)
        if args.phase == "build":
            return
    if args.phase in ("train", "all"):
        train_models(args, device)
        if args.phase == "train":
            return
    if args.phase in ("evaluate", "all"):
        evaluate_v9(args, device)
    print(f"{VERSION} completed in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
