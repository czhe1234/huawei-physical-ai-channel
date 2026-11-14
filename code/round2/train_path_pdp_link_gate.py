#!/usr/bin/env python3
"""Learn a cross-fitted per-link gate between V9 PDP and raw explicit-path PDP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from analyze_joint_path_token_oracle import FREQUENCIES, OOF_META, REGIME_CACHE, v9_channel
from analyze_v4_exact_joint_projection import (
    exact_pas_projection,
    exact_pdp_projection,
    pas_distribution,
    pdp_distribution,
    preserve_point_power,
    truth_views,
)
from project_paths import PATHS
from round2.data import load_positions
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics


VERSION = "Round2-E65-path-pdp-link-gate"
PATH_CACHE = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_oracle"
    / "n469_s24_k128_b096_d064_scatter1_sel106"
)
TOKEN_FEATURES = (
    PATHS.artifacts
    / "round2"
    / "explicit_path_pdp_predictor"
    / "path_features_n469.npy"
)
ARTIFACT_DIR = PATHS.artifacts / "round2" / "path_pdp_link_gate"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_path_pdp_link_gate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--folds", default="0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gate-penalty", type=float, default=2e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--random-state", type=int, default=106)
    parser.add_argument("--reuse-spectra", action="store_true")
    return parser.parse_args()


def parse_folds(value: str) -> list[int]:
    result = sorted({int(item) for item in value.split(",") if item.strip()})
    if not result or any(fold < 0 or fold > 4 for fold in result):
        raise ValueError(f"invalid folds: {value}")
    return result


class LinkGate(nn.Module):
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
        nn.init.constant_(self.network[-1].bias, -3.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(features)[..., 0])


def link_coordinates() -> np.ndarray:
    antenna = np.arange(256, dtype=np.int64)
    polarization = antenna // 128
    local = antenna % 128
    horizontal = local // 8
    vertical = local % 8
    ue = np.arange(4, dtype=np.int64)
    p = np.broadcast_to(polarization[:, None], (256, 4)).reshape(-1)
    h = np.broadcast_to(horizontal[:, None], (256, 4)).reshape(-1)
    v = np.broadcast_to(vertical[:, None], (256, 4)).reshape(-1)
    u = np.broadcast_to(ue[None, :], (256, 4)).reshape(-1)
    return np.concatenate(
        (
            np.eye(2, dtype=np.float32)[p],
            np.eye(4, dtype=np.float32)[u],
            (h / 15.0)[:, None].astype(np.float32),
            (v / 7.0)[:, None].astype(np.float32),
            np.sin(2.0 * np.pi * h / 16.0)[:, None].astype(np.float32),
            np.cos(2.0 * np.pi * h / 16.0)[:, None].astype(np.float32),
            np.sin(2.0 * np.pi * v / 8.0)[:, None].astype(np.float32),
            np.cos(2.0 * np.pi * v / 8.0)[:, None].astype(np.float32),
        ),
        axis=1,
    )


def point_context(rows: np.ndarray, query_ids: np.ndarray) -> np.ndarray:
    token = np.load(TOKEN_FEATURES, mmap_mode="r")
    mask = np.asarray(np.load(PATH_CACHE / "mask.npy", mmap_mode="r"), dtype=bool)
    valid = mask[..., None].astype(np.float32)
    token_mean = (np.asarray(token) * valid).sum(axis=1) / valid.sum(axis=1).clip(min=1.0)
    _, train_positions, _, train_site, _ = load_positions()
    ids = query_ids[rows]
    positions = train_positions[ids, :2]
    site = train_site[ids]
    center = train_positions[:, :2].mean(axis=0)
    scale = train_positions[:, :2].std(axis=0) + 1e-6
    regime = np.load(REGIME_CACHE, allow_pickle=False)["oof_regime"].astype(np.int8)[rows]
    extra = np.concatenate(
        (
            ((positions - center) / scale).astype(np.float32),
            np.eye(2, dtype=np.float32)[site],
            np.eye(4, dtype=np.float32)[regime],
        ),
        axis=1,
    )
    return np.concatenate((token_mean.astype(np.float32), extra), axis=1)


def spectral_features(
    base: torch.Tensor,
    raw: torch.Tensor,
    context: torch.Tensor,
    link: torch.Tensor,
) -> torch.Tensor:
    count = len(base)
    base_group = base.reshape(count, 1024, 24, 8).sum(dim=3)
    raw_group = raw.reshape(count, 1024, 24, 8).sum(dim=3)
    base_root = torch.sqrt(base_group.clamp_min(0.0))
    raw_root = torch.sqrt(raw_group.clamp_min(0.0))
    difference = raw_root - base_root
    product = raw_root * base_root
    base_entropy = -(base * torch.log(base.clamp_min(1e-12))).sum(dim=2, keepdim=True)
    raw_entropy = -(raw * torch.log(raw.clamp_min(1e-12))).sum(dim=2, keepdim=True)
    agreement = (base_root * raw_root).sum(dim=2, keepdim=True)
    summary = torch.cat((base_entropy, raw_entropy, agreement), dim=2)
    context_block = context[:, None].expand(-1, 1024, -1)
    link_block = link[None].expand(count, -1, -1)
    return torch.cat(
        (base_root, raw_root, difference, product, summary, context_block, link_block),
        dim=2,
    )


def cosine_by_link(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    numerator = (prediction * truth).sum(dim=2, dtype=torch.float64)
    denominator = torch.sqrt(
        prediction.square().sum(dim=2, dtype=torch.float64)
        * truth.square().sum(dim=2, dtype=torch.float64)
    )
    return numerator / torch.where(denominator == 0.0, 1e-12, denominator)


def build_spectra(
    rows: np.ndarray,
    query_ids: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_path = ARTIFACT_DIR / "v9_base_channel.npy"
    base_pdp_path = ARTIFACT_DIR / "v9_base_pdp_f16.npy"
    raw_pdp_path = ARTIFACT_DIR / "raw_path_pdp_f16.npy"
    if base_path.is_file() and base_pdp_path.is_file() and raw_pdp_path.is_file():
        base = np.load(base_path, mmap_mode="r")
        base_pdp = np.load(base_pdp_path, mmap_mode="r")
        raw_pdp = np.load(raw_pdp_path, mmap_mode="r")
    else:
        base_array = v9_channel(rows, device, batch_size)
        np.save(base_path, base_array)
        base = np.load(base_path, mmap_mode="r")
        base_blocks = []
        raw_blocks = []
        coefficient = np.load(PATH_CACHE / "coefficient.npy", mmap_mode="r")
        tau = np.load(PATH_CACHE / "tau.npy", mmap_mode="r")
        mask = np.load(PATH_CACHE / "mask.npy", mmap_mode="r")
        frequency = torch.as_tensor(FREQUENCIES, device=device, dtype=torch.float32)
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            base_tensor = torch.as_tensor(
                np.array(base[start:end], copy=True), device=device
            )
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
            base_blocks.append(pdp_distribution(base_tensor).cpu().numpy())
            raw_blocks.append(pdp_distribution(channel).cpu().numpy())
        np.save(base_pdp_path, np.concatenate(base_blocks).astype(np.float16))
        np.save(raw_pdp_path, np.concatenate(raw_blocks).astype(np.float16))
        base_pdp = np.load(base_pdp_path, mmap_mode="r")
        raw_pdp = np.load(raw_pdp_path, mmap_mode="r")
    train_h = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    truth = np.asarray(train_h[query_ids[rows]], dtype=np.complex64)
    with torch.inference_mode():
        _, truth_pdp = truth_views(torch.as_tensor(truth, device=device))
    return base, base_pdp, raw_pdp, truth_pdp.cpu().numpy().astype(np.float32)


def train_fold(
    fold: int,
    args: argparse.Namespace,
    base_pdp: np.ndarray,
    raw_pdp: np.ndarray,
    truth_pdp: np.ndarray,
    context: np.ndarray,
    fold_id: np.ndarray,
    link: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    train_points = np.flatnonzero(fold_id != fold).astype(np.int64)
    held_points = np.flatnonzero(fold_id == fold).astype(np.int64)
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
                np.asarray(base_pdp[ids], dtype=np.float32).reshape(len(ids), 1024, 192),
                device=device,
            )
            raw = torch.as_tensor(
                np.asarray(raw_pdp[ids], dtype=np.float32).reshape(len(ids), 1024, 192),
                device=device,
            )
            truth = torch.as_tensor(
                truth_pdp[ids].reshape(len(ids), 1024, 192), device=device
            )
            local_context = torch.as_tensor(
                (context[ids] - context_mean) / context_std, device=device
            )
            features = spectral_features(base, raw, local_context, link_tensor)
            alpha = model(features)
            desired = (1.0 - alpha[..., None]) * base + alpha[..., None] * raw
            desired = desired / desired.sum(dim=2, keepdim=True).clamp_min(1e-30)
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
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(
                f"link gate fold {fold} epoch {epoch + 1}/{args.epochs} "
                f"cosine={history[-1]['train_pdp_cosine']:.6f} "
                f"alpha={history[-1]['train_alpha_mean']:.4f}",
                flush=True,
            )
    prediction = np.zeros((len(held_points), 1024), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(held_points), args.batch_size):
            end = min(start + args.batch_size, len(held_points))
            ids = held_points[start:end]
            base = torch.as_tensor(
                np.asarray(base_pdp[ids], dtype=np.float32).reshape(len(ids), 1024, 192),
                device=device,
            )
            raw = torch.as_tensor(
                np.asarray(raw_pdp[ids], dtype=np.float32).reshape(len(ids), 1024, 192),
                device=device,
            )
            local_context = torch.as_tensor(
                (context[ids] - context_mean) / context_std, device=device
            )
            features = spectral_features(base, raw, local_context, link_tensor)
            prediction[start:end] = model(features).cpu().numpy()
    checkpoint = ARTIFACT_DIR / f"fold{fold}_link_gate.pt"
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


def project(
    base: torch.Tensor,
    desired_pdp: torch.Tensor,
    cycles: int,
    order: str,
) -> torch.Tensor:
    reference_power = base.abs().square().sum(dim=(1, 2, 3))
    desired_pas = pas_distribution(base)
    output = base
    for _ in range(cycles):
        for domain in order.split("_"):
            if domain == "pas":
                output = exact_pas_projection(output, desired_pas)
            else:
                output = exact_pdp_projection(output, desired_pdp)
        output = preserve_point_power(output, reference_power)
    return output


def main() -> None:
    args = parse_args()
    requested_folds = parse_folds(args.folds)
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    required = (PATH_CACHE, TOKEN_FEATURES, OOF_META, REGIME_CACHE)
    missing = [PATHS.relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_all = metadata["fold_id"].astype(np.int8)
    rows = np.load(PATH_CACHE / "rows.npy").astype(np.int64)
    fold_id = fold_all[rows]
    base, base_pdp, raw_pdp, truth_pdp = build_spectra(
        rows, query_ids, device, args.batch_size
    )
    context = point_context(rows, query_ids)
    link = link_coordinates()
    alpha = np.zeros((len(rows), 1024), dtype=np.float32)
    active = np.zeros(len(rows), dtype=bool)
    training = []
    for fold in requested_folds:
        held = fold_id == fold
        values, diagnostics = train_fold(
            fold,
            args,
            base_pdp,
            raw_pdp,
            truth_pdp,
            context,
            fold_id,
            link,
            device,
        )
        alpha[held] = values
        active[held] = True
        training.append(diagnostics)
    local = np.flatnonzero(active)
    np.save(
        ARTIFACT_DIR / f"oof_alpha_folds_{'-'.join(map(str, requested_folds))}.npy",
        alpha,
    )

    base_tensor = torch.as_tensor(np.asarray(base[local]), device=device)
    truth_channel = np.asarray(
        np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")[
            query_ids[rows[local]]
        ],
        dtype=np.complex64,
    )
    truth_tensor = torch.as_tensor(truth_channel, device=device)
    truth_pas, truth_pdp_view = truth_views(truth_tensor)
    base_pdp_tensor = torch.as_tensor(
        np.asarray(base_pdp[local], dtype=np.float32), device=device
    )
    raw_pdp_tensor = torch.as_tensor(
        np.asarray(raw_pdp[local], dtype=np.float32), device=device
    )
    result_rows = []
    metric_cache = {}
    for scale in (0.50, 0.75, 1.00, 1.25, 1.50):
        local_alpha = torch.clamp(
            scale * torch.as_tensor(alpha[local], device=device), 0.0, 1.0
        ).reshape(len(local), 256, 4)
        desired = (
            (1.0 - local_alpha[..., None]) * base_pdp_tensor
            + local_alpha[..., None] * raw_pdp_tensor
        )
        desired = desired / desired.sum(dim=3, keepdim=True).clamp_min(1e-30)
        for order in ("pas_pdp", "pdp_pas"):
            for cycles in (1, 2, 4, 8):
                with torch.inference_mode():
                    prediction = project(base_tensor, desired, cycles, order)
                    values = point_metrics(
                        prediction, truth_tensor, truth_pas, truth_pdp_view
                    )
                name = f"scale{scale:.2f}_{order}_c{cycles}"
                row = {"candidate": name, "scale": scale, "order": order, "cycles": cycles, **statistics(*values)}
                result_rows.append(row)
                metric_cache[name] = values
    baseline_values = point_metrics(
        base_tensor, truth_tensor, truth_pas, truth_pdp_view
    )
    baseline = statistics(*baseline_values)
    result_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    selected = result_rows[0]
    selected_values = metric_cache[str(selected["candidate"])]
    fold_results = []
    for fold in requested_folds:
        fold_mask = fold_id[local] == fold
        current = statistics(*selected_values, fold_mask)
        anchor = statistics(*baseline_values, fold_mask)
        fold_results.append(
            {
                "fold": fold,
                "result": current,
                "baseline": anchor,
                "delta": float(current["score"] - anchor["score"]),
            }
        )
    fold_tag = "-".join(map(str, requested_folds))
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "requested_folds": requested_folds,
        "configuration": vars(args),
        "point_count": int(len(local)),
        "baseline": baseline,
        "selected": selected,
        "fold_results": fold_results,
        "positive_folds": int(sum(row["delta"] > 0.0 for row in fold_results)),
        "alpha": {
            "mean": float(np.mean(alpha[local])),
            "median": float(np.median(alpha[local])),
            "p90": float(np.quantile(alpha[local], 0.90)),
            "fraction_below_0.05": float(np.mean(alpha[local] < 0.05)),
        },
        "training": training,
        "top_candidates": result_rows[:20],
        "seconds": float(time.time() - started),
    }
    report_path = ANALYSIS_DIR / f"path_pdp_link_gate_folds_{fold_tag}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
