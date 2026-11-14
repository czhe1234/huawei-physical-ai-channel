#!/usr/bin/env python3
"""Evaluate separate, weakly tied PAS/PDP coefficients on the v575 basis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn

import train_v575_physical_response_basis as v575
from project_paths import PATHS


VERSION_NUMBER = 576
VERSION = "v576_dual_metric_physical_response"
CACHE_DIR = PATHS.cache_dir(VERSION_NUMBER)
REPORT_DIR = PATHS.version_report_dir(VERSION_NUMBER)
REPORT = REPORT_DIR / f"{VERSION}.json"
OOF_CACHE = CACHE_DIR / "v576_dual_metric_physical_response_oof.npz"
CHECKPOINT_DIR = PATHS.checkpoints / VERSION
SEED = 57_576


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
    parser.add_argument("--head-tie-penalty", type=float, default=2e-4)
    parser.add_argument("--oracle-steps", type=int, default=100)
    parser.add_argument("--skip-control", action="store_true")
    return parser.parse_args()


class CoefficientBranch(nn.Module):
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
            nn.Linear(width, v575.PAIR_COUNT),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.coefficient_cap * torch.tanh(self.net(features))


class DualCoefficientModel(nn.Module):
    def __init__(self, input_dim: int, width: int, coefficient_cap: float) -> None:
        super().__init__()
        self.pas = CoefficientBranch(input_dim, width, coefficient_cap)
        self.pdp = CoefficientBranch(input_dim, width, coefficient_cap)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pas(features), self.pdp(features)


def batch_scores(
    indices: torch.Tensor,
    response_indices: torch.Tensor,
    pas_coefficients: torch.Tensor,
    pdp_coefficients: torch.Tensor,
    tensors: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pas_score = torch.zeros(len(indices), device=indices.device)
    for name in v575.PAS_NAMES:
        prediction = v575.apply_flat_response(
            tensors["pas_anchor"][name][indices],
            tensors["responses"][name][response_indices],
            pas_coefficients,
        )
        pas_score = pas_score + v575.PAS_WEIGHTS[name] * v575.cosine_flat(
            tensors["pas_truth"][name][indices], prediction
        )
    pdp_prediction = v575.apply_pdp_response(
        tensors["pdp_anchor"][indices],
        tensors["responses"]["pdp"][response_indices],
        pdp_coefficients,
    )
    pdp_score = v575.pdp_point_score(
        tensors["pdp_truth"][indices], pdp_prediction
    )
    return pas_score, pdp_score, 0.5 * (pas_score + pdp_score)


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_ids: np.ndarray,
    features: torch.Tensor,
    row_map: torch.Tensor,
    tensors: dict[str, object],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> float:
    model.train()
    order = np.asarray(train_ids, dtype=np.int64).copy()
    rng.shuffle(order)
    total = 0.0
    count = 0
    for start in range(0, len(order), args.batch_size):
        local = torch.as_tensor(
            order[start : start + args.batch_size], device=features.device
        )
        pas_coefficients, pdp_coefficients = model(features[local])
        _, _, score = batch_scores(
            local,
            row_map[local],
            pas_coefficients,
            pdp_coefficients,
            tensors,
        )
        coefficient_cost = 0.5 * (
            pas_coefficients.square().mean() + pdp_coefficients.square().mean()
        )
        tie_cost = (pas_coefficients - pdp_coefficients).square().mean()
        loss = (
            1.0
            - score.mean()
            + args.coefficient_penalty * coefficient_cost
            + args.head_tie_penalty * tie_cost
        )
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
    output = {
        "pas": np.zeros(len(ids), dtype=np.float32),
        "pdp": np.zeros(len(ids), dtype=np.float32),
        "combined": np.zeros(len(ids), dtype=np.float32),
        "pas_coefficients": np.zeros(
            (len(ids), v575.PAIR_COUNT), dtype=np.float32
        ),
        "pdp_coefficients": np.zeros(
            (len(ids), v575.PAIR_COUNT), dtype=np.float32
        ),
    }
    for start in range(0, len(ids), batch_size):
        stop = min(start + batch_size, len(ids))
        local_np = np.asarray(ids[start:stop], dtype=np.int64)
        local = torch.as_tensor(local_np, device=features.device)
        pas_coefficients, pdp_coefficients = model(features[local])
        pas, pdp, combined = batch_scores(
            local,
            row_map[local],
            pas_coefficients,
            pdp_coefficients,
            tensors,
        )
        output["pas"][start:stop] = pas.cpu().numpy()
        output["pdp"][start:stop] = pdp.cpu().numpy()
        output["combined"][start:stop] = combined.cpu().numpy()
        output["pas_coefficients"][start:stop] = pas_coefficients.cpu().numpy()
        output["pdp_coefficients"][start:stop] = pdp_coefficients.cpu().numpy()
    return output


def select_epoch(
    train_ids: np.ndarray,
    validation_ids: np.ndarray,
    features_np: np.ndarray,
    row_map: torch.Tensor,
    tensors: dict[str, object],
    args: argparse.Namespace,
    seed: int,
) -> tuple[int, list[dict[str, float]]]:
    standardized, _, _ = v575.standardize(features_np, train_ids)
    features = torch.from_numpy(standardized).to(tensors["device"])
    model = DualCoefficientModel(
        features.shape[1], args.width, args.coefficient_cap
    ).to(tensors["device"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    rng = np.random.default_rng(seed)
    best_epoch = 1
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
            args,
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
    return best_epoch, history


def refit(
    fit_ids: np.ndarray,
    query_ids: np.ndarray,
    best_epoch: int,
    features_np: np.ndarray,
    row_map: torch.Tensor,
    tensors: dict[str, object],
    args: argparse.Namespace,
    seed: int,
    checkpoint: Path,
) -> dict[str, np.ndarray]:
    standardized, mean, std = v575.standardize(features_np, fit_ids)
    features = torch.from_numpy(standardized).to(tensors["device"])
    model = DualCoefficientModel(
        features.shape[1], args.width, args.coefficient_cap
    ).to(tensors["device"])
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
            args,
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
            "pair_names": v575.PAIR_NAMES,
            "head_tie_penalty": args.head_tie_penalty,
        },
        checkpoint,
    )
    return result


def crossfit(
    label: str,
    features: np.ndarray,
    fold_id: np.ndarray,
    row_map_np: np.ndarray,
    tensors: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    row_map = torch.from_numpy(row_map_np.astype(np.int64)).to(tensors["device"])
    oof = {
        "pas": np.zeros(400, dtype=np.float32),
        "pdp": np.zeros(400, dtype=np.float32),
        "combined": np.zeros(400, dtype=np.float32),
        "pas_coefficients": np.zeros((400, v575.PAIR_COUNT), dtype=np.float32),
        "pdp_coefficients": np.zeros((400, v575.PAIR_COUNT), dtype=np.float32),
    }
    folds = []
    for outer in range(5):
        query_ids = np.flatnonzero(fold_id == outer)
        fit_ids = np.flatnonzero(fold_id != outer)
        inner = (outer + 4) % 5
        train_ids = np.flatnonzero((fold_id != outer) & (fold_id != inner))
        validation_ids = np.flatnonzero(fold_id == inner)
        v575.seed_everything(SEED + outer + (10_000 if label == "control" else 0))
        best_epoch, history = select_epoch(
            train_ids,
            validation_ids,
            features,
            row_map,
            tensors,
            args,
            SEED + outer * 101,
        )
        result = refit(
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
        baseline = v575.baseline_scores(
            torch.as_tensor(query_ids, device=tensors["device"]), tensors
        )
        base_pas, base_pdp, base_combined = [value.cpu().numpy() for value in baseline]
        row = {
            "fold": outer,
            "inner_validation_fold": inner,
            "best_epoch": best_epoch,
            "baseline_pas": float(np.mean(base_pas)),
            "prediction_pas": float(np.mean(result["pas"])),
            "pas_gain": float(np.mean(result["pas"] - base_pas)),
            "baseline_pdp": float(np.mean(base_pdp)),
            "prediction_pdp": float(np.mean(result["pdp"])),
            "pdp_gain": float(np.mean(result["pdp"] - base_pdp)),
            "baseline_combined": float(np.mean(base_combined)),
            "prediction_combined": float(np.mean(result["combined"])),
            "combined_gain": float(np.mean(result["combined"] - base_combined)),
            "selection_history": history,
        }
        folds.append(row)
        print(
            f"v576 {label} fold={outer} epoch={best_epoch} "
            f"gain={row['combined_gain']:+.6f} "
            f"pas={row['pas_gain']:+.6f} pdp={row['pdp_gain']:+.6f}",
            flush=True,
        )
    all_ids = torch.arange(400, device=tensors["device"])
    baseline = v575.baseline_scores(all_ids, tensors)
    baseline_np = {
        "pas": baseline[0].cpu().numpy(),
        "pdp": baseline[1].cpu().numpy(),
        "combined": baseline[2].cpu().numpy(),
    }
    summary = {
        "baseline_pas": float(np.mean(baseline_np["pas"])),
        "prediction_pas": float(np.mean(oof["pas"])),
        "pas_gain": float(np.mean(oof["pas"] - baseline_np["pas"])),
        "baseline_pdp": float(np.mean(baseline_np["pdp"])),
        "prediction_pdp": float(np.mean(oof["pdp"])),
        "pdp_gain": float(np.mean(oof["pdp"] - baseline_np["pdp"])),
        "baseline_combined": float(np.mean(baseline_np["combined"])),
        "prediction_combined": float(np.mean(oof["combined"])),
        "combined_gain": float(np.mean(oof["combined"] - baseline_np["combined"])),
        "fold_wins": int(sum(row["combined_gain"] > 0.0 for row in folds)),
    }
    return {"oof": oof, "baseline": baseline_np, "folds": folds, "summary": summary}


def dual_oracle(tensors: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    scores = []
    pas_coefficients_out = []
    pdp_coefficients_out = []
    for start in range(0, 400, 20):
        stop = min(start + 20, 400)
        ids = torch.arange(start, stop, device=tensors["device"])
        raw_pas = torch.zeros(
            (stop - start, v575.PAIR_COUNT), device=tensors["device"], requires_grad=True
        )
        raw_pdp = torch.zeros_like(raw_pas, requires_grad=True)
        optimizer = torch.optim.Adam([raw_pas, raw_pdp], lr=0.06)
        best_score = torch.full((stop - start,), -torch.inf, device=tensors["device"])
        best_pas = torch.zeros_like(raw_pas)
        best_pdp = torch.zeros_like(raw_pdp)
        for _ in range(args.oracle_steps):
            pas_coefficients = args.coefficient_cap * torch.tanh(raw_pas)
            pdp_coefficients = args.coefficient_cap * torch.tanh(raw_pdp)
            _, _, score = batch_scores(
                ids, ids, pas_coefficients, pdp_coefficients, tensors
            )
            optimizer.zero_grad(set_to_none=True)
            (-score.mean()).backward()
            optimizer.step()
            with torch.no_grad():
                improved = score > best_score
                best_score[improved] = score[improved]
                best_pas[improved] = pas_coefficients[improved]
                best_pdp[improved] = pdp_coefficients[improved]
        scores.append(best_score.detach().cpu().numpy())
        pas_coefficients_out.append(best_pas.detach().cpu().numpy())
        pdp_coefficients_out.append(best_pdp.detach().cpu().numpy())
    score = np.concatenate(scores)
    all_ids = torch.arange(400, device=tensors["device"])
    baseline = v575.baseline_scores(all_ids, tensors)[2].cpu().numpy()
    return {
        "score": score,
        "pas_coefficients": np.concatenate(pas_coefficients_out),
        "pdp_coefficients": np.concatenate(pdp_coefficients_out),
        "baseline": float(np.mean(baseline)),
        "oracle": float(np.mean(score)),
        "gain": float(np.mean(score - baseline)),
        "positive_fraction": float(np.mean(score > baseline)),
    }


def main() -> None:
    args = parse_args()
    for path in (
        v575.source.SELECTION,
        v575.PDP_ANCHOR,
        *v575.PAS_TRUTH.values(),
        *v575.PAS_ANCHOR.values(),
    ):
        v575.require(path)
    for item in v575.source.PAIRS:
        for seed in v575.source.PAIR_SEEDS:
            if not v575.source.completed(item, seed):
                raise RuntimeError(f"incomplete v575 response {item.name} seed={seed}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    v575.seed_everything(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("v576 requires CUDA")
    device = torch.device(f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    started = time.time()

    selection = v575.load_selection()
    selected_ids = selection["selected_ids"]
    pas_truth_np, pas_anchor_np = v575.load_pas_arrays(selected_ids)
    pdp_truth_np, pdp_anchor_np = v575.load_pdp_arrays(selected_ids)
    response_np, summary = v575.load_response_bank()
    features, feature_names = v575.build_features(summary["matrix"])
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
    main_result = crossfit(
        "main", features, selection["fold_id"], identity, tensors, args
    )
    control_result = None
    control_map = v575.shuffled_row_map(selection["regimes"])
    if not args.skip_control:
        static_dim = features.shape[1] - summary["matrix"].shape[1]
        control_features = features.copy()
        control_features[:, static_dim:] = summary["matrix"][control_map]
        control_result = crossfit(
            "control",
            control_features,
            selection["fold_id"],
            control_map,
            tensors,
            args,
        )
    oracle = dual_oracle(tensors, args)
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
        pair_names=np.asarray(v575.PAIR_NAMES),
        main_pas_coefficients=main_result["oof"]["pas_coefficients"],
        main_pdp_coefficients=main_result["oof"]["pdp_coefficients"],
        main_pas=main_result["oof"]["pas"],
        main_pdp=main_result["oof"]["pdp"],
        main_combined=main_result["oof"]["combined"],
        baseline_pas=main_result["baseline"]["pas"],
        baseline_pdp=main_result["baseline"]["pdp"],
        baseline_combined=main_result["baseline"]["combined"],
        control_combined=(
            control_result["oof"]["combined"]
            if control_result is not None
            else np.zeros(400, dtype=np.float32)
        ),
        oracle_pas_coefficients=oracle["pas_coefficients"],
        oracle_pdp_coefficients=oracle["pdp_coefficients"],
        oracle_combined=oracle["score"],
        control_row_map=control_map,
    )
    report = {
        "method": VERSION,
        "status": "analysis_complete_passed" if passed else "analysis_complete_failed",
        "primary_hypothesis": (
            "The v575 response source is valid, but simulator-to-official effective "
            "coefficients differ between PAS and PDP; separate weakly tied heads should "
            "preserve the five-fold PDP gain without fold-4 PAS overcorrection."
        ),
        "single_change_from_v575": (
            "replace one 20-coefficient head with independent PAS and PDP branches; "
            "all data, responses, folds, anchors, optimizer settings and pass criteria remain fixed"
        ),
        "model": {
            "branches": "independent three-layer PAS and PDP coefficient MLPs",
            "width": args.width,
            "coefficient_cap": args.coefficient_cap,
            "coefficient_penalty": args.coefficient_penalty,
            "head_tie_penalty": args.head_tie_penalty,
            "learning_rate": args.learning_rate,
            "max_epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "feature_count": int(features.shape[1]),
            "feature_names": feature_names,
        },
        "main": {
            "summary": main_summary,
            "folds": main_result["folds"],
            "by_regime": v575.summarize_by_regime(
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
            key: value
            for key, value in oracle.items()
            if key not in ("score", "pas_coefficients", "pdp_coefficients")
        },
        "decision": {
            "required_combined_gain": 0.02,
            "required_fold_wins": 4,
            "required_real_minus_shuffled_margin": 0.01,
            "passed": passed,
            "next_step": (
                "expand the paired response generator to all 2000 train and 500 test points"
                if passed
                else "stop the paired physical-response route and retain its PDP response as diagnostic evidence"
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
    v575.atomic_json(REPORT, report)
    print(
        json.dumps(
            {
                "report": PATHS.relative(REPORT),
                "main_gain": main_summary["combined_gain"],
                "pas_gain": main_summary["pas_gain"],
                "pdp_gain": main_summary["pdp_gain"],
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
