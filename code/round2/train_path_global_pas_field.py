#!/usr/bin/env python3
"""Learn a low-dimensional global PAS field from explicit propagation paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from analyze_joint_path_token_oracle import OOF_META
from analyze_v11_explicit_path_pas_upper_bound import v10_oof_channel
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution, truth_views
from prepare_sionna_calibration_dataset import TRUTH_PAS
from project_paths import PATHS
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics
from train_full_path_pas_slice_gate import (
    BASE_PAS,
    FOLD_IDS,
    OOF_PATH_CACHE,
    POINT_CONTEXT,
    project_joint,
)


VERSION = "Round2-E73-explicit-path-global-PAS-field"
PATH_CACHE = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_oracle"
    / "n4000_s24_k128_b096_d064_scatter1_sel106"
)
TOKEN_FEATURES = (
    PATHS.artifacts
    / "round2"
    / "explicit_path_pdp_predictor"
    / "path_features_train_n4000.npy"
)
ARTIFACT_DIR = PATHS.artifacts / "round2" / "path_global_pas_field"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_path_global_pas_field"
OOF_TARGET = ARTIFACT_DIR / "train_oof_global_pas_f16.npy"
OOF_REPORT = ANALYSIS_DIR / "path_global_pas_field_train.json"
V10_TARGET = ARTIFACT_DIR / "v10_oof_global_pas_f16.npy"
V10_REPORT = ANALYSIS_DIR / "path_global_pas_field_v10_oof.json"
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("train", "evaluate", "all"), default="all"
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--token-hidden", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--residual-scale", type=float, default=3.0)
    parser.add_argument("--hellinger-weight", type=float, default=0.08)
    parser.add_argument("--residual-weight", type=float, default=2e-4)
    parser.add_argument("--random-state", type=int, default=2026080773)
    return parser.parse_args()


def normalize(value: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    value = value.clamp_min(0.0)
    total = value.sum(dim=dims, keepdim=True)
    return torch.where(total > TINY, value / total.clamp_min(TINY), 0.0)


def global_pas(value: torch.Tensor) -> torch.Tensor:
    return normalize(value.sum(dim=(4, 5)), (1, 2, 3)).reshape(len(value), 256)


def cosine(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    numerator = (prediction * truth).sum(dim=1, dtype=torch.float64)
    denominator = torch.sqrt(
        prediction.square().sum(dim=1, dtype=torch.float64)
        * truth.square().sum(dim=1, dtype=torch.float64)
    )
    return numerator / torch.where(denominator == 0.0, 1e-12, denominator)


class PathGlobalPasModel(nn.Module):
    def __init__(
        self,
        token_dim: int,
        context_dim: int,
        hidden: int,
        blocks: int,
        heads: int,
        dropout: float,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.token_stem = nn.Sequential(
            nn.Linear(token_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=blocks)
        self.attention = nn.Linear(hidden, 1)
        self.base_encoder = nn.Sequential(
            nn.Linear(256, hidden), nn.SiLU(), nn.LayerNorm(hidden)
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden // 2),
            nn.SiLU(),
            nn.LayerNorm(hidden // 2),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + hidden // 2, hidden * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 256),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        token: torch.Tensor,
        mask: torch.Tensor,
        base: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(
            self.token_stem(token), src_key_padding_mask=~mask
        )
        logits = self.attention(encoded).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(weights[..., None] * encoded, dim=1)
        residual = self.residual_scale * torch.tanh(
            self.head(
                torch.cat(
                    (pooled, self.base_encoder(torch.sqrt(base)), self.context_encoder(context)),
                    dim=1,
                )
            )
        )
        target = torch.softmax(torch.log(base.clamp_min(1e-8)) + residual, dim=1)
        return target, residual


def checkpoint_path(fold: int) -> Path:
    return ARTIFACT_DIR / f"fold{fold}_path_global_pas.pt"


def token_statistics(
    features: np.ndarray, mask: np.ndarray, fit: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(features.shape[2], dtype=np.float64)
    square = np.zeros(features.shape[2], dtype=np.float64)
    count = 0
    for start in range(0, len(fit), 64):
        ids = fit[start : start + 64]
        local = np.asarray(features[ids], dtype=np.float32)
        valid = mask[ids]
        values = local[valid]
        total += values.sum(axis=0, dtype=np.float64)
        square += np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)
        count += len(values)
    mean = total / max(count, 1)
    variance = square / max(count, 1) - mean * mean
    std = np.sqrt(np.maximum(variance, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


@torch.inference_mode()
def predict(
    model: PathGlobalPasModel,
    ids: np.ndarray,
    features: np.ndarray,
    mask: np.ndarray,
    base: np.ndarray,
    context: np.ndarray,
    token_mean: np.ndarray,
    token_std: np.ndarray,
    context_mean: np.ndarray,
    context_std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output = np.empty((len(ids), 256), dtype=np.float32)
    model.eval()
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        local_ids = ids[start:end]
        token = torch.as_tensor(
            (np.asarray(features[local_ids], dtype=np.float32) - token_mean) / token_std,
            device=device,
        )
        valid = torch.as_tensor(mask[local_ids], device=device)
        local_base = torch.as_tensor(base[local_ids], device=device)
        local_context = torch.as_tensor(
            (context[local_ids] - context_mean) / context_std, device=device
        )
        output[start:end] = model(token, valid, local_base, local_context)[0].cpu().numpy()
    return output


def train_fold(
    fold: int,
    args: argparse.Namespace,
    fold_id: np.ndarray,
    nonzero: np.ndarray,
    features: np.ndarray,
    mask: np.ndarray,
    base: np.ndarray,
    truth: np.ndarray,
    context: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    fit = np.flatnonzero((fold_id != fold) & nonzero).astype(np.int64)
    held = np.flatnonzero((fold_id == fold) & nonzero).astype(np.int64)
    token_mean, token_std = token_statistics(features, mask, fit)
    context_mean = context[fit].mean(axis=0).astype(np.float32)
    context_std = context[fit].std(axis=0).astype(np.float32)
    context_std[context_std < 1e-5] = 1.0
    model = PathGlobalPasModel(
        features.shape[2],
        context.shape[1],
        args.token_hidden,
        args.blocks,
        args.heads,
        args.dropout,
        args.residual_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.05
    )
    rng = np.random.default_rng(args.random_state + 1000 * fold)
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(fit)
        total_loss = 0.0
        total_cosine = 0.0
        batches = 0
        for start in range(0, len(order), args.batch_size):
            ids = order[start : start + args.batch_size]
            token = torch.as_tensor(
                (np.asarray(features[ids], dtype=np.float32) - token_mean) / token_std,
                device=device,
            )
            valid = torch.as_tensor(mask[ids], device=device)
            local_base = torch.as_tensor(base[ids], device=device)
            local_truth = torch.as_tensor(truth[ids], device=device)
            local_context = torch.as_tensor(
                (context[ids] - context_mean) / context_std, device=device
            )
            prediction, residual = model(token, valid, local_base, local_context)
            local_cosine = cosine(prediction, local_truth)
            hellinger = torch.mean(
                torch.square(torch.sqrt(prediction) - torch.sqrt(local_truth))
            )
            loss = (
                1.0
                - local_cosine.mean()
                + args.hellinger_weight * hellinger
                + args.residual_weight * residual.square().mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.detach().item())
            total_cosine += float(local_cosine.mean().detach().item())
            batches += 1
        scheduler.step()
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss / max(batches, 1),
                "fit_global_pas_cosine": total_cosine / max(batches, 1),
            }
        )
        if (epoch + 1) % 8 == 0 or epoch + 1 == args.epochs:
            print(
                f"global PAS fold {fold} epoch {epoch + 1}/{args.epochs} "
                f"cosine={history[-1]['fit_global_pas_cosine']:.6f}",
                flush=True,
            )
    held_prediction = predict(
        model,
        held,
        features,
        mask,
        base,
        context,
        token_mean,
        token_std,
        context_mean,
        context_std,
        device,
        args.batch_size,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(features.shape[2]),
            "context_dim": int(context.shape[1]),
            "token_mean": token_mean,
            "token_std": token_std,
            "context_mean": context_mean,
            "context_std": context_std,
            "config": vars(args),
        },
        checkpoint_path(fold),
    )
    held_base = torch.as_tensor(base[held], device=device)
    held_truth = torch.as_tensor(truth[held], device=device)
    held_pred = torch.as_tensor(held_prediction, device=device)
    return held_prediction, {
        "fold": fold,
        "fit_count": int(len(fit)),
        "held_count": int(len(held)),
        "baseline": float(cosine(held_base, held_truth).mean().item()),
        "selected": float(cosine(held_pred, held_truth).mean().item()),
        "checkpoint": PATHS.relative(checkpoint_path(fold)),
        "history": history,
    }


def load_model(fold: int, device: torch.device) -> tuple[PathGlobalPasModel, dict]:
    checkpoint = torch.load(checkpoint_path(fold), map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = PathGlobalPasModel(
        int(checkpoint["input_dim"]),
        int(checkpoint["context_dim"]),
        int(config["token_hidden"]),
        int(config["blocks"]),
        int(config["heads"]),
        float(config["dropout"]),
        float(config["residual_scale"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def train_models(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    fold_id = np.load(FOLD_IDS)
    features = np.load(TOKEN_FEATURES, mmap_mode="r")
    mask = np.asarray(np.load(PATH_CACHE / "mask.npy", mmap_mode="r"), dtype=bool)
    base_pas = np.load(BASE_PAS, mmap_mode="r")
    truth_pas = np.load(TRUTH_PAS, mmap_mode="r")
    context = np.load(POINT_CONTEXT, mmap_mode="r")
    train_h = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    power = np.sum(np.abs(train_h) ** 2, axis=(1, 2, 3), dtype=np.float64)
    nonzero = power > 0.0
    base = np.empty((len(base_pas), 256), dtype=np.float32)
    truth = np.empty_like(base)
    for start in range(0, len(base_pas), 16):
        end = min(start + 16, len(base_pas))
        base[start:end] = global_pas(
            torch.as_tensor(np.asarray(base_pas[start:end], dtype=np.float32), device=device)
        ).cpu().numpy()
        truth[start:end] = global_pas(
            torch.as_tensor(np.asarray(truth_pas[start:end], dtype=np.float32), device=device)
        ).cpu().numpy()
    output = np.zeros_like(base)
    rows = []
    for fold in range(5):
        held = np.flatnonzero((fold_id == fold) & nonzero)
        prediction, diagnostics = train_fold(
            fold,
            args,
            fold_id,
            nonzero,
            features,
            mask,
            base,
            truth,
            np.asarray(context),
            device,
        )
        output[held] = prediction
        diagnostics["delta"] = diagnostics["selected"] - diagnostics["baseline"]
        rows.append(diagnostics)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    np.save(OOF_TARGET, output.astype(np.float16))
    valid = np.flatnonzero(nonzero)
    baseline = float(
        cosine(torch.as_tensor(base[valid], device=device), torch.as_tensor(truth[valid], device=device)).mean().item()
    )
    selected = float(
        cosine(torch.as_tensor(output[valid], device=device), torch.as_tensor(truth[valid], device=device)).mean().item()
    )
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "configuration": vars(args),
        "baseline": baseline,
        "selected": selected,
        "delta": selected - baseline,
        "positive_folds": int(sum(row["delta"] > 0.0 for row in rows)),
        "folds": rows,
        "output": PATHS.relative(OOF_TARGET),
    }
    OOF_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("baseline", "selected", "delta", "positive_folds")}, indent=2), flush=True)
    return report


def apply_global_target(
    base_pas: torch.Tensor, target: torch.Tensor, eta: float
) -> torch.Tensor:
    current_global = global_pas(base_pas).reshape(len(base_pas), 2, 16, 8)
    target_view = target.reshape(len(base_pas), 2, 16, 8)
    ratio = torch.pow(
        target_view.clamp_min(1e-8) / current_global.clamp_min(1e-8), eta
    ).clamp(0.25, 4.0)
    desired = base_pas * ratio[:, :, :, :, None, None]
    return normalize(desired, (1, 2, 3))


def evaluate_v10(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    rows = np.load(OOF_PATH_CACHE / "rows.npy").astype(np.int64)
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    test_hole_fold = metadata["fold_id"].astype(np.int8)[rows]
    global_ids = query_ids[rows]
    broad_fold = np.load(FOLD_IDS)[global_ids]
    features_full = np.load(TOKEN_FEATURES, mmap_mode="r")
    mask_full = np.asarray(np.load(PATH_CACHE / "mask.npy", mmap_mode="r"), dtype=bool)
    context_full = np.load(POINT_CONTEXT, mmap_mode="r")
    features = np.asarray(features_full[global_ids], dtype=np.float32)
    mask = mask_full[global_ids]
    context = np.asarray(context_full[global_ids], dtype=np.float32)
    base = v10_oof_channel(device)
    base_tensor = torch.as_tensor(base, device=device)
    base_pas = pas_distribution(base_tensor)
    base_global = global_pas(base_pas).cpu().numpy()
    target = np.zeros((len(rows), 256), dtype=np.float32)
    for fold in range(5):
        local = np.flatnonzero(broad_fold == fold)
        model, checkpoint = load_model(fold, device)
        target[local] = predict(
            model,
            local,
            features,
            mask,
            base_global,
            context,
            np.asarray(checkpoint["token_mean"], dtype=np.float32),
            np.asarray(checkpoint["token_std"], dtype=np.float32),
            np.asarray(checkpoint["context_mean"], dtype=np.float32),
            np.asarray(checkpoint["context_std"], dtype=np.float32),
            device,
            args.batch_size,
        )
        del model
    np.save(V10_TARGET, target.astype(np.float16))
    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    truth = np.asarray(truth_all[global_ids], dtype=np.complex64)
    truth_tensor = torch.as_tensor(truth, device=device)
    truth_pas, truth_pdp = truth_views(truth_tensor)
    base_pdp = pdp_distribution(base_tensor)
    baseline_values = point_metrics(base_tensor, truth_tensor, truth_pas, truth_pdp)
    candidates = []
    metric_cache = {}
    target_tensor = torch.as_tensor(target, device=device)
    for eta in (0.10, 0.20, 0.35, 0.50, 0.75, 1.00):
        desired_pas = apply_global_target(base_pas, target_tensor, eta)
        for order in ("pas_pdp", "pdp_pas"):
            for cycles in (1, 2, 4):
                with torch.inference_mode():
                    prediction = project_joint(
                        base_tensor, desired_pas, base_pdp, cycles, order
                    )
                    values = point_metrics(prediction, truth_tensor, truth_pas, truth_pdp)
                name = f"eta{eta:.2f}_{order}_c{cycles}"
                candidates.append(
                    {"candidate": name, "eta": eta, "order": order, "cycles": cycles, **statistics(*values)}
                )
                metric_cache[name] = values
    candidates.sort(key=lambda row: float(row["score"]), reverse=True)
    selected = candidates[0]
    selected_values = metric_cache[str(selected["candidate"])]
    baseline = statistics(*baseline_values)
    fold_rows = []
    for fold in range(5):
        local = test_hole_fold == fold
        anchor = statistics(*baseline_values, local)
        result = statistics(*selected_values, local)
        fold_rows.append(
            {"fold": fold, "baseline": anchor, "result": result, "delta": result["score"] - anchor["score"]}
        )
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "baseline": baseline,
        "selected": selected,
        "fold_results": fold_rows,
        "positive_folds": int(sum(row["delta"] > 0.0 for row in fold_rows)),
        "top_candidates": candidates[:20],
        "output": PATHS.relative(V10_TARGET),
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
    required = (PATH_CACHE, TOKEN_FEATURES, BASE_PAS, TRUTH_PAS, POINT_CONTEXT, FOLD_IDS, OOF_META)
    missing = [PATHS.relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if args.phase in ("train", "all"):
        train_models(args, device)
        if args.phase == "train":
            return
    if args.phase in ("evaluate", "all"):
        evaluate_v10(args, device)
    print(f"{VERSION} completed in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
