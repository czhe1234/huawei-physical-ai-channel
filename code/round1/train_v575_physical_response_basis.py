#!/usr/bin/env python3
"""Fit and strictly evaluate the v575 paired physical-response basis."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn

import generate_v543_disagreement_conditioned_cell_fusion as v543
import generate_v575_physical_response_basis as source
from project_paths import PATHS


VERSION_NUMBER = 575
VERSION = "v575_physical_response_basis"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = CACHE_DIR / "v575_physical_response_basis_oof.npz"
CHECKPOINT_DIR = PATHS.checkpoints / VERSION

PAS_NAMES = ("sh", "psh", "shue", "shv")
PAS_WEIGHTS = {"sh": 0.4, "psh": 0.2, "shue": 0.2, "shv": 0.2}
PAIR_NAMES = tuple(item.name for item in source.PAIRS)
PAIR_COUNT = len(PAIR_NAMES)
FLOOR = 1e-10
TINY = 1e-30
SEED = 57_575

PAS_TRUTH = {
    name: (
        PATHS.cache_dir(549)
        / "pas_calibration"
        / "official_train"
        / f"official_train_{name}.npy"
    )
    for name in PAS_NAMES
}
PAS_ANCHOR = {
    "sh": PATHS.cache_dir(543) / "v543_cellwise_fusion_oof.npz",
    "psh": PATHS.cache_dir(549) / "pas_calibration" / "n24" / "psh_calibrated_oof.npy",
    "shue": PATHS.cache_dir(549) / "pas_calibration" / "n24" / "shue_calibrated_oof.npy",
    "shv": PATHS.cache_dir(549) / "pas_calibration" / "n24" / "shv_calibrated_oof.npy",
}
PDP_ANCHOR = (
    PATHS.cache_dir(554)
    / "linkwise_pdp_calibration"
    / "old_calibrated_linkwise_pdp_oof.npy"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--coefficient-cap", type=float, default=1.5)
    parser.add_argument("--coefficient-penalty", type=float, default=2e-3)
    parser.add_argument("--oracle-steps", type=int, default=100)
    parser.add_argument("--skip-control", action="store_true")
    return parser.parse_args()


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_flat(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(FLOOR)
    return value / value.sum(dim=1, keepdim=True).clamp_min(TINY)


def normalize_pdp(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(0.0)
    return value / value.sum(dim=-1, keepdim=True).clamp_min(TINY)


def cosine_flat(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    numerator = (left * right).flatten(1).sum(dim=1)
    left_norm = left.square().flatten(1).sum(dim=1).sqrt()
    right_norm = right.square().flatten(1).sum(dim=1).sqrt()
    product = left_norm * right_norm
    denominator = torch.where(product == 0.0, torch.full_like(product, 1e-12), product)
    return numerator / denominator


def pdp_point_score(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    numerator = (left * right).sum(dim=-1)
    product = left.square().sum(dim=-1).sqrt() * right.square().sum(dim=-1).sqrt()
    denominator = torch.where(product == 0.0, torch.full_like(product, 1e-12), product)
    return (numerator / denominator).mean(dim=1)


def load_selection() -> dict[str, np.ndarray]:
    require(source.SELECTION)
    value = np.load(source.SELECTION)
    return {
        "selected_ids": value["selected_ids"].astype(np.int64),
        "fold_id": value["fold_id"].astype(np.int16),
        "regimes": value["regimes"].astype("U8"),
        "replacements": value["replacements"].astype(np.int64),
    }


def load_pas_arrays(selected_ids: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    truth: dict[str, np.ndarray] = {}
    anchor: dict[str, np.ndarray] = {}
    for name in PAS_NAMES:
        require(PAS_TRUTH[name])
        truth[name] = np.asarray(
            np.load(PAS_TRUTH[name], mmap_mode="r")[selected_ids], dtype=np.float32
        ).reshape(len(selected_ids), -1)
        if name == "sh":
            require(PAS_ANCHOR[name])
            anchor[name] = np.asarray(
                np.load(PAS_ANCHOR[name])["calibrated"][selected_ids],
                dtype=np.float32,
            ).reshape(len(selected_ids), -1)
        else:
            require(PAS_ANCHOR[name])
            anchor[name] = np.asarray(
                np.load(PAS_ANCHOR[name], mmap_mode="r")[selected_ids],
                dtype=np.float32,
            ).reshape(len(selected_ids), -1)
    return truth, anchor


def official_linkwise_pdp(selected_ids: np.ndarray) -> np.ndarray:
    channel = np.load(PATHS.round1 / "Round1_Train_Channel.npy", mmap_mode="r")
    local = np.asarray(channel[selected_ids], dtype=np.complex64)
    cir = np.fft.ifft(local, axis=-1)
    power = np.abs(cir).astype(np.float32) ** 2
    total = power.sum(axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)
    output = power / np.maximum(total, TINY)
    output[total[..., 0] <= TINY] = 0.0
    return output.reshape(len(selected_ids), 1024, 192).astype(np.float32)


def load_pdp_arrays(selected_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    require(PDP_ANCHOR)
    anchor = np.asarray(
        np.load(PDP_ANCHOR, mmap_mode="r")[selected_ids], dtype=np.float32
    ).reshape(len(selected_ids), 1024, 192)
    return official_linkwise_pdp(selected_ids), anchor


def load_response_bank() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    responses: dict[str, np.ndarray] = {}
    summaries: dict[str, list[np.ndarray]] = {}
    for name in (*PAS_NAMES, "pdp"):
        local_shape = (
            (400, PAIR_COUNT, 1024, 192)
            if name == "pdp"
            else (400, PAIR_COUNT, int(np.prod(source.PAS_SHAPES[name])))
        )
        responses[name] = np.empty(local_shape, dtype=np.float32)

    for pair_index, item in enumerate(source.PAIRS):
        seed_arrays: dict[str, list[np.ndarray]] = {name: [] for name in responses}
        seed_summaries = []
        for seed in source.PAIR_SEEDS:
            paths = source.response_paths(item, seed)
            for path in paths.values():
                require(path)
            for name in responses:
                seed_arrays[name].append(
                    np.asarray(np.load(paths[name], mmap_mode="r"), dtype=np.float32)
                )
            seed_summaries.append(np.load(paths["summary"]))
        local_columns = []
        for name in responses:
            first, second = seed_arrays[name]
            responses[name][:, pair_index] = 0.5 * (first + second)
            difference = first - second
            instability = np.sqrt(
                np.mean(difference.astype(np.float64) ** 2, axis=tuple(range(1, difference.ndim)))
            ).astype(np.float32)
            local_columns.append(instability[:, None])
        for key in (
            "minus_path_count",
            "plus_path_count",
            "minus_log_power",
            "plus_log_power",
            "sh_response_rms",
            "psh_response_rms",
            "shue_response_rms",
            "shv_response_rms",
            "pdp_response_rms",
        ):
            first = np.asarray(seed_summaries[0][key], dtype=np.float32)
            second = np.asarray(seed_summaries[1][key], dtype=np.float32)
            mean = 0.5 * (first + second)
            spread = np.abs(first - second)
            if key == "plus_path_count":
                minus = 0.5 * (
                    np.asarray(seed_summaries[0]["minus_path_count"], dtype=np.float32)
                    + np.asarray(seed_summaries[1]["minus_path_count"], dtype=np.float32)
                )
                local_columns.extend(
                    [
                        np.log1p(np.maximum(mean, 0.0))[:, None],
                        ((mean - minus) / np.maximum(1.0, mean + minus))[:, None],
                    ]
                )
            elif key == "plus_log_power":
                minus = 0.5 * (
                    np.asarray(seed_summaries[0]["minus_log_power"], dtype=np.float32)
                    + np.asarray(seed_summaries[1]["minus_log_power"], dtype=np.float32)
                )
                local_columns.append((mean - minus)[:, None])
            elif key.startswith("minus_"):
                continue
            else:
                local_columns.extend([mean[:, None], spread[:, None]])
        summaries[item.name] = local_columns
    summary_matrix = np.concatenate(
        [np.concatenate(summaries[name], axis=1) for name in PAIR_NAMES], axis=1
    ).astype(np.float32)
    return responses, {"matrix": summary_matrix}


def build_features(summary: np.ndarray) -> tuple[np.ndarray, list[str]]:
    static_train, _, static_names = v543.load_static_features()
    selected_ids = np.load(source.SELECTION)["selected_ids"].astype(np.int64)
    static = static_train[selected_ids]
    features = np.concatenate([static, summary], axis=1).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    names = list(static_names) + [
        f"response_summary_{index}" for index in range(summary.shape[1])
    ]
    return features, names


def shuffled_row_map(regimes: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED + 911)
    mapping = np.arange(len(regimes), dtype=np.int64)
    for regime in np.unique(regimes):
        local = np.flatnonzero(regimes == regime)
        permuted = local.copy()
        rng.shuffle(permuted)
        mapping[local] = permuted
    return mapping


class CoefficientModel(nn.Module):
    def __init__(self, input_dim: int, width: int, coefficient_cap: float) -> None:
        super().__init__()
        self.coefficient_cap = float(coefficient_cap)
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, PAIR_COUNT),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.coefficient_cap * torch.tanh(self.net(features))


def apply_flat_response(
    anchor: torch.Tensor,
    response: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    delta = torch.einsum("bj,bjd->bd", coefficients, response)
    amplitude = (anchor.clamp_min(FLOOR).sqrt() + delta).clamp_min(0.0)
    return normalize_flat(amplitude.square())


def apply_pdp_response(
    anchor: torch.Tensor,
    response: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    delta = torch.einsum("bj,bjld->bld", coefficients, response)
    amplitude = (anchor.clamp_min(FLOOR).sqrt() + delta).clamp_min(0.0)
    return normalize_pdp(amplitude.square())


def batch_scores(
    indices: torch.Tensor,
    response_indices: torch.Tensor,
    coefficients: torch.Tensor,
    tensors: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pas_score = torch.zeros(len(indices), device=indices.device)
    for name in PAS_NAMES:
        prediction = apply_flat_response(
            tensors["pas_anchor"][name][indices],
            tensors["responses"][name][response_indices],
            coefficients,
        )
        pas_score = pas_score + PAS_WEIGHTS[name] * cosine_flat(
            tensors["pas_truth"][name][indices], prediction
        )
    pdp_prediction = apply_pdp_response(
        tensors["pdp_anchor"][indices],
        tensors["responses"]["pdp"][response_indices],
        coefficients,
    )
    pdp_score = pdp_point_score(tensors["pdp_truth"][indices], pdp_prediction)
    combined = 0.5 * pas_score + 0.5 * pdp_score
    return pas_score, pdp_score, combined


@torch.no_grad()
def baseline_scores(indices: torch.Tensor, tensors: dict[str, object]):
    pas_score = torch.zeros(len(indices), device=indices.device)
    for name in PAS_NAMES:
        pas_score = pas_score + PAS_WEIGHTS[name] * cosine_flat(
            tensors["pas_truth"][name][indices],
            tensors["pas_anchor"][name][indices],
        )
    pdp_score = pdp_point_score(
        tensors["pdp_truth"][indices], tensors["pdp_anchor"][indices]
    )
    return pas_score, pdp_score, 0.5 * (pas_score + pdp_score)


def standardize(
    features: np.ndarray, fit_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features[fit_ids].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = features[fit_ids].std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-5, 1.0, std).astype(np.float32)
    output = np.clip((features - mean) / std, -8.0, 8.0).astype(np.float32)
    return output, mean, std


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_ids: np.ndarray,
    features: torch.Tensor,
    row_map: torch.Tensor,
    tensors: dict[str, object],
    batch_size: int,
    coefficient_penalty: float,
    rng: np.random.Generator,
) -> float:
    model.train()
    order = np.asarray(train_ids, dtype=np.int64).copy()
    rng.shuffle(order)
    total = 0.0
    count = 0
    for start in range(0, len(order), batch_size):
        local = torch.as_tensor(order[start : start + batch_size], device=features.device)
        coefficients = model(features[local])
        _, _, score = batch_scores(local, row_map[local], coefficients, tensors)
        loss = 1.0 - score.mean() + coefficient_penalty * coefficients.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.detach()) * len(local)
        count += len(local)
    return total / max(count, 1)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    ids: np.ndarray,
    features: torch.Tensor,
    row_map: torch.Tensor,
    tensors: dict[str, object],
    batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    outputs = {
        "pas": np.zeros(len(ids), dtype=np.float32),
        "pdp": np.zeros(len(ids), dtype=np.float32),
        "combined": np.zeros(len(ids), dtype=np.float32),
        "coefficients": np.zeros((len(ids), PAIR_COUNT), dtype=np.float32),
    }
    for start in range(0, len(ids), batch_size):
        stop = min(start + batch_size, len(ids))
        local_np = np.asarray(ids[start:stop], dtype=np.int64)
        local = torch.as_tensor(local_np, device=features.device)
        coefficients = model(features[local])
        pas, pdp, combined = batch_scores(
            local, row_map[local], coefficients, tensors
        )
        outputs["pas"][start:stop] = pas.cpu().numpy()
        outputs["pdp"][start:stop] = pdp.cpu().numpy()
        outputs["combined"][start:stop] = combined.cpu().numpy()
        outputs["coefficients"][start:stop] = coefficients.cpu().numpy()
    return outputs


def fit_one_model(
    train_ids: np.ndarray,
    validation_ids: np.ndarray,
    features_np: np.ndarray,
    row_map: torch.Tensor,
    tensors: dict[str, object],
    args: argparse.Namespace,
    seed: int,
) -> tuple[int, list[dict[str, float]]]:
    standardized, _, _ = standardize(features_np, train_ids)
    features = torch.from_numpy(standardized).to(tensors["device"])
    model = CoefficientModel(features.shape[1], args.width, args.coefficient_cap).to(
        tensors["device"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    rng = np.random.default_rng(seed)
    best_epoch = 0
    best_score = -np.inf
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            optimizer,
            train_ids,
            features,
            row_map,
            tensors,
            args.batch_size,
            args.coefficient_penalty,
            rng,
        )
        validation = evaluate_model(
            model,
            validation_ids,
            features,
            row_map,
            tensors,
            args.batch_size,
        )
        score = float(np.mean(validation["combined"]))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_score": score,
                "validation_pas": float(np.mean(validation["pas"])),
                "validation_pdp": float(np.mean(validation["pdp"])),
            }
        )
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    return max(best_epoch, 1), history


def refit_and_evaluate(
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
    best_epoch: int,
    features_np: np.ndarray,
    row_map: torch.Tensor,
    tensors: dict[str, object],
    args: argparse.Namespace,
    seed: int,
    checkpoint: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    standardized, mean, std = standardize(features_np, fit_ids)
    features = torch.from_numpy(standardized).to(tensors["device"])
    model = CoefficientModel(features.shape[1], args.width, args.coefficient_cap).to(
        tensors["device"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    rng = np.random.default_rng(seed)
    for _ in range(best_epoch):
        train_epoch(
            model,
            optimizer,
            fit_ids,
            features,
            row_map,
            tensors,
            args.batch_size,
            args.coefficient_penalty,
            rng,
        )
    result = evaluate_model(
        model, query_ids, features, row_map, tensors, args.batch_size
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "best_epoch": best_epoch,
            "pair_names": PAIR_NAMES,
            "row_map_identity": bool(torch.equal(row_map, torch.arange(len(row_map), device=row_map.device))),
        },
        checkpoint,
    )
    return result, {"mean": mean, "std": std}


def run_crossfit(
    label: str,
    features: np.ndarray,
    fold_id: np.ndarray,
    row_map_np: np.ndarray,
    tensors: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    row_map = torch.from_numpy(row_map_np.astype(np.int64)).to(tensors["device"])
    oof = {
        "pas": np.zeros(len(fold_id), dtype=np.float32),
        "pdp": np.zeros(len(fold_id), dtype=np.float32),
        "combined": np.zeros(len(fold_id), dtype=np.float32),
        "coefficients": np.zeros((len(fold_id), PAIR_COUNT), dtype=np.float32),
    }
    folds = []
    for outer in range(5):
        query_ids = np.flatnonzero(fold_id == outer)
        fit_ids = np.flatnonzero(fold_id != outer)
        inner = (outer + 4) % 5
        train_ids = np.flatnonzero((fold_id != outer) & (fold_id != inner))
        validation_ids = np.flatnonzero(fold_id == inner)
        seed_everything(SEED + outer + (10_000 if label == "control" else 0))
        best_epoch, history = fit_one_model(
            train_ids,
            validation_ids,
            features,
            row_map,
            tensors,
            args,
            SEED + outer * 101,
        )
        result, _ = refit_and_evaluate(
            fit_ids,
            query_ids,
            best_epoch,
            features,
            row_map,
            tensors,
            args,
            SEED + outer * 211 + 17,
            CHECKPOINT_DIR / f"{label}_fold_{outer}.pt",
        )
        for key in oof:
            oof[key][query_ids] = result[key]
        baseline = baseline_scores(
            torch.as_tensor(query_ids, device=tensors["device"]), tensors
        )
        baseline_pas = baseline[0].cpu().numpy()
        baseline_pdp = baseline[1].cpu().numpy()
        baseline_combined = baseline[2].cpu().numpy()
        fold_row = {
            "fold": outer,
            "inner_validation_fold": inner,
            "fit_count": int(len(fit_ids)),
            "query_count": int(len(query_ids)),
            "best_epoch": int(best_epoch),
            "baseline_pas": float(np.mean(baseline_pas)),
            "prediction_pas": float(np.mean(result["pas"])),
            "pas_gain": float(np.mean(result["pas"] - baseline_pas)),
            "baseline_pdp": float(np.mean(baseline_pdp)),
            "prediction_pdp": float(np.mean(result["pdp"])),
            "pdp_gain": float(np.mean(result["pdp"] - baseline_pdp)),
            "baseline_combined": float(np.mean(baseline_combined)),
            "prediction_combined": float(np.mean(result["combined"])),
            "combined_gain": float(np.mean(result["combined"] - baseline_combined)),
            "selection_history": history,
        }
        folds.append(fold_row)
        print(
            f"v575 {label} fold={outer} epoch={best_epoch} "
            f"gain={fold_row['combined_gain']:+.6f} "
            f"pas={fold_row['pas_gain']:+.6f} pdp={fold_row['pdp_gain']:+.6f}",
            flush=True,
        )
    all_ids = torch.arange(len(fold_id), device=tensors["device"])
    baseline = baseline_scores(all_ids, tensors)
    baseline_np = {
        "pas": baseline[0].cpu().numpy(),
        "pdp": baseline[1].cpu().numpy(),
        "combined": baseline[2].cpu().numpy(),
    }
    return {
        "oof": oof,
        "baseline": baseline_np,
        "folds": folds,
        "summary": {
            "baseline_pas": float(np.mean(baseline_np["pas"])),
            "prediction_pas": float(np.mean(oof["pas"])),
            "pas_gain": float(np.mean(oof["pas"] - baseline_np["pas"])),
            "baseline_pdp": float(np.mean(baseline_np["pdp"])),
            "prediction_pdp": float(np.mean(oof["pdp"])),
            "pdp_gain": float(np.mean(oof["pdp"] - baseline_np["pdp"])),
            "baseline_combined": float(np.mean(baseline_np["combined"])),
            "prediction_combined": float(np.mean(oof["combined"])),
            "combined_gain": float(
                np.mean(oof["combined"] - baseline_np["combined"])
            ),
            "fold_wins": int(sum(row["combined_gain"] > 0.0 for row in folds)),
        },
    }


def pointwise_oracle(
    tensors: dict[str, object], args: argparse.Namespace
) -> dict[str, object]:
    all_scores = []
    all_coefficients = []
    for start in range(0, 400, 20):
        stop = min(start + 20, 400)
        ids = torch.arange(start, stop, device=tensors["device"])
        raw = torch.zeros((stop - start, PAIR_COUNT), device=tensors["device"], requires_grad=True)
        optimizer = torch.optim.Adam([raw], lr=0.06)
        best_score = torch.full((stop - start,), -torch.inf, device=tensors["device"])
        best_coefficients = torch.zeros_like(raw)
        for _ in range(args.oracle_steps):
            coefficients = args.coefficient_cap * torch.tanh(raw)
            _, _, score = batch_scores(ids, ids, coefficients, tensors)
            loss = -score.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                improved = score > best_score
                best_score[improved] = score[improved]
                best_coefficients[improved] = coefficients[improved]
        all_scores.append(best_score.detach().cpu().numpy())
        all_coefficients.append(best_coefficients.detach().cpu().numpy())
    score = np.concatenate(all_scores)
    coefficients = np.concatenate(all_coefficients)
    all_ids = torch.arange(400, device=tensors["device"])
    baseline = baseline_scores(all_ids, tensors)[2].cpu().numpy()
    return {
        "score": score,
        "coefficients": coefficients,
        "baseline": float(np.mean(baseline)),
        "oracle": float(np.mean(score)),
        "gain": float(np.mean(score - baseline)),
        "positive_fraction": float(np.mean(score > baseline)),
    }


def summarize_by_regime(
    regimes: np.ndarray,
    baseline: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, object]:
    output = {}
    for regime in ("clear", "light", "medium", "heavy"):
        local = regimes == regime
        output[regime] = {
            "count": int(np.sum(local)),
            "baseline": float(np.mean(baseline[local])),
            "prediction": float(np.mean(prediction[local])),
            "gain": float(np.mean(prediction[local] - baseline[local])),
        }
    return output


def main() -> None:
    args = parse_args()
    for path in (
        source.SELECTION,
        PATHS.round1 / "Round1_Train_Channel.npy",
        PDP_ANCHOR,
        *PAS_TRUTH.values(),
        *PAS_ANCHOR.values(),
    ):
        require(path)
    for item in source.PAIRS:
        for seed in source.PAIR_SEEDS:
            if not source.completed(item, seed):
                raise RuntimeError(f"incomplete physical response: {item.name} seed={seed}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("v575 training requires CUDA")
    device = torch.device(f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    started = time.time()

    selection = load_selection()
    selected_ids = selection["selected_ids"]
    pas_truth_np, pas_anchor_np = load_pas_arrays(selected_ids)
    pdp_truth_np, pdp_anchor_np = load_pdp_arrays(selected_ids)
    response_np, summary = load_response_bank()
    features, feature_names = build_features(summary["matrix"])

    tensors = {
        "device": device,
        "pas_truth": {
            name: torch.from_numpy(value).to(device) for name, value in pas_truth_np.items()
        },
        "pas_anchor": {
            name: torch.from_numpy(value).to(device) for name, value in pas_anchor_np.items()
        },
        "pdp_truth": torch.from_numpy(pdp_truth_np).to(device),
        "pdp_anchor": torch.from_numpy(pdp_anchor_np).to(device),
        "responses": {
            name: torch.from_numpy(value).to(device) for name, value in response_np.items()
        },
    }
    del pas_truth_np, pas_anchor_np, pdp_truth_np, pdp_anchor_np, response_np

    identity = np.arange(400, dtype=np.int64)
    main_result = run_crossfit(
        "main", features, selection["fold_id"], identity, tensors, args
    )
    control_result = None
    control_map = shuffled_row_map(selection["regimes"])
    if not args.skip_control:
        static_dim = features.shape[1] - summary["matrix"].shape[1]
        control_features = features.copy()
        control_features[:, static_dim:] = summary["matrix"][control_map]
        control_result = run_crossfit(
            "control",
            control_features,
            selection["fold_id"],
            control_map,
            tensors,
            args,
        )
    oracle = pointwise_oracle(tensors, args)
    main_summary = main_result["summary"]
    control_gain = (
        float(control_result["summary"]["combined_gain"])
        if control_result is not None
        else float("nan")
    )
    control_margin = float(main_summary["combined_gain"] - control_gain)
    passed = bool(
        main_summary["combined_gain"] >= 0.02
        and main_summary["fold_wins"] >= 4
        and control_result is not None
        and control_margin >= 0.01
    )

    np.savez_compressed(
        OOF_CACHE,
        selected_ids=selected_ids,
        fold_id=selection["fold_id"],
        regimes=selection["regimes"],
        pair_names=np.asarray(PAIR_NAMES),
        main_coefficients=main_result["oof"]["coefficients"],
        main_pas=main_result["oof"]["pas"],
        main_pdp=main_result["oof"]["pdp"],
        main_combined=main_result["oof"]["combined"],
        baseline_pas=main_result["baseline"]["pas"],
        baseline_pdp=main_result["baseline"]["pdp"],
        baseline_combined=main_result["baseline"]["combined"],
        control_coefficients=(
            control_result["oof"]["coefficients"]
            if control_result is not None
            else np.zeros((400, PAIR_COUNT), dtype=np.float32)
        ),
        control_combined=(
            control_result["oof"]["combined"]
            if control_result is not None
            else np.zeros(400, dtype=np.float32)
        ),
        oracle_coefficients=oracle["coefficients"],
        oracle_combined=oracle["score"],
        control_row_map=control_map,
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete_passed" if passed else "analysis_complete_failed",
        "primary_hypothesis": (
            "The response of PAS/PDP to controlled Sionna physics perturbations is a new "
            "same-position information source that can correct a strong official-data anchor."
        ),
        "selection": {
            "count": 400,
            "rows_per_fold": [
                int(np.sum(selection["fold_id"] == fold)) for fold in range(5)
            ],
            "zero_power_replacements": selection["replacements"].tolist(),
            "path": PATHS.relative(source.SELECTION),
        },
        "response_basis": {
            "pair_count": PAIR_COUNT,
            "pair_names": list(PAIR_NAMES),
            "pairs": [asdict(item) for item in source.PAIRS],
            "seeds": list(source.PAIR_SEEDS),
            "definition": (
                "mean over two seeds of 0.5*(sqrt(plus normalized spectrum) - "
                "sqrt(minus normalized spectrum))"
            ),
            "modalities": [*PAS_NAMES, "1024-link PDP"],
        },
        "anchors": {
            "sh": "v543 strict OOF calibrated joint S-H",
            "psh_shue_shv": "v549 24-seed strict OOF calibrated PAS marginals",
            "pdp": "v554 old calibrated OOF, corresponding to the v549 PDP carrier",
        },
        "model": {
            "architecture": "three-layer geometry/response-summary MLP predicting 20 bounded response coefficients",
            "width": args.width,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "coefficient_cap": args.coefficient_cap,
            "coefficient_penalty": args.coefficient_penalty,
            "feature_count": int(features.shape[1]),
            "static_feature_count": int(features.shape[1] - summary["matrix"].shape[1]),
            "response_summary_feature_count": int(summary["matrix"].shape[1]),
            "feature_names": feature_names,
        },
        "protocol": {
            "outer": "five fixed continuous spatial folds, each exactly 80 points",
            "inner": "one deterministic held-out fit fold selects epoch; outer fold is never used",
            "control": "same architecture with complete response fields and summaries permuted within propagation regime",
            "score": "0.5*PAS + 0.5*exact 1024-link PDP; PAS=0.4*SH+0.2*PSH+0.2*SHUE+0.2*SHV",
        },
        "main": {
            "summary": main_summary,
            "folds": main_result["folds"],
            "by_regime": summarize_by_regime(
                selection["regimes"],
                main_result["baseline"]["combined"],
                main_result["oof"]["combined"],
            ),
        },
        "control": (
            {
                "summary": control_result["summary"],
                "folds": control_result["folds"],
                "real_minus_shuffled_gain": control_margin,
            }
            if control_result is not None
            else None
        ),
        "pointwise_oracle": {
            key: value for key, value in oracle.items() if key not in ("score", "coefficients")
        },
        "decision": {
            "required_combined_gain": 0.02,
            "required_fold_wins": 4,
            "required_real_minus_shuffled_margin": 0.01,
            "passed": passed,
            "next_step": (
                "generate all-2000 train and 500-test response fields, refit, and deploy one candidate"
                if passed
                else "stop the physical-response-basis route; do not tune coefficient caps or MLP width"
            ),
        },
        "runtime": {
            "seconds": float(time.time() - started),
            "device": str(device),
            "torch": torch.__version__,
        },
        "outputs": {
            "oof": PATHS.relative(OOF_CACHE),
            "checkpoints": PATHS.relative(CHECKPOINT_DIR),
            "report": PATHS.relative(REPORT),
        },
    }
    atomic_json(REPORT, report)
    print(
        json.dumps(
            {
                "report": PATHS.relative(REPORT),
                "main_gain": main_summary["combined_gain"],
                "fold_wins": main_summary["fold_wins"],
                "control_gain": control_gain,
                "control_margin": control_margin,
                "oracle_gain": oracle["gain"],
                "passed": passed,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
