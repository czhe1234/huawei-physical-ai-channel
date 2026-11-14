#!/usr/bin/env python3
"""Refine V3 donor weights with grouped official linkwise PAS/PDP objectives."""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict

import numpy as np
import torch

from data import load_positions
from project_paths import PATHS
from run_v3_anchored_neural_rbf import (
    ARTIFACT_DIR,
    DESCRIPTORS,
    FOLD_CACHE,
    GEOMETRY,
    OOF_META,
    PAS_SIZE,
    AnchoredWeightCorrector,
    Config,
    build_train_contexts,
    forward_rows,
    power_cosine,
    set_seed,
    standardize_geometry,
    tensor_context,
    unit_amplitude,
    validation_context,
)


ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v3_anchored_neural_rbf"
REPORT = ANALYSIS_DIR / "grouped_link_refinement.json"
PRE_DIR = PATHS.artifacts / "round2" / "pre_sionna"
TRAIN_PAS = PRE_DIR / "round2_train_pas_link_support_f16.npy"
TRAIN_PDP = PRE_DIR / "round2_train_pdp_link_support_f16.npy"
GROUPED_PAS = ARTIFACT_DIR / "grouped_pas_amplitude_f16.npy"
GROUPED_PDP = ARTIFACT_DIR / "grouped_pdp_amplitude_f16.npy"

PAS_FREQUENCY_GROUPS = 8
PAS_GROUP_SIZE = 192 // PAS_FREQUENCY_GROUPS


def build_grouped_descriptors() -> tuple[np.ndarray, np.ndarray]:
    train_pas = np.load(TRAIN_PAS, mmap_mode="r")
    train_pdp = np.load(TRAIN_PDP, mmap_mode="r")
    count = len(train_pas)
    expected_pas = (count, 4 * PAS_FREQUENCY_GROUPS, 256)
    expected_pdp = (count, 2 * 4, 192)
    if GROUPED_PAS.is_file() and GROUPED_PDP.is_file():
        pas = np.load(GROUPED_PAS, mmap_mode="r")
        pdp = np.load(GROUPED_PDP, mmap_mode="r")
        if pas.shape == expected_pas and pdp.shape == expected_pdp:
            return pas, pdp

    pas_output = np.lib.format.open_memmap(
        GROUPED_PAS, mode="w+", dtype=np.float16, shape=expected_pas
    )
    pdp_output = np.lib.format.open_memmap(
        GROUPED_PDP, mode="w+", dtype=np.float16, shape=expected_pdp
    )
    for start in range(0, count, 16):
        end = min(start + 16, count)
        pas = np.asarray(train_pas[start:end], dtype=np.float32)
        pas = pas.reshape(end - start, 256, 4, PAS_FREQUENCY_GROUPS, PAS_GROUP_SIZE)
        pas = pas.mean(axis=4, dtype=np.float64).astype(np.float32)
        pas = np.transpose(pas, (0, 2, 3, 1)).reshape(end - start, -1, 256)
        pdp = np.asarray(train_pdp[start:end], dtype=np.float32)
        pdp = pdp.reshape(end - start, 2, 16, 8, 4, 192)
        pdp = pdp.mean(axis=(2, 3), dtype=np.float64).astype(np.float32)
        pdp = pdp.reshape(end - start, -1, 192)
        pas_output[start:end] = np.sqrt(np.maximum(pas, 0.0)).astype(np.float16)
        pdp_output[start:end] = np.sqrt(np.maximum(pdp, 0.0)).astype(np.float16)
    pas_output.flush()
    pdp_output.flush()
    return np.load(GROUPED_PAS, mmap_mode="r"), np.load(GROUPED_PDP, mmap_mode="r")


def grouped_predictions(
    weights: torch.Tensor,
    donor_ids: torch.Tensor,
    pas_amplitude: torch.Tensor,
    pdp_amplitude: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pas = torch.relu(
        torch.sum(weights[:, :, 0, None, None] * pas_amplitude[donor_ids], dim=1)
    )
    pdp = torch.relu(
        torch.sum(weights[:, :, 1, None, None] * pdp_amplitude[donor_ids], dim=1)
    )
    return unit_amplitude(pas), unit_amplitude(pdp)


def stable_power_cosine(
    predicted_amplitude: torch.Tensor,
    truth_amplitude: torch.Tensor,
) -> torch.Tensor:
    predicted = predicted_amplitude.square()
    truth = truth_amplitude.square()
    numerator = torch.sum(predicted * truth, dim=-1)
    product = torch.sum(predicted.square(), dim=-1) * torch.sum(
        truth.square(), dim=-1
    )
    return numerator / torch.sqrt(product.clamp_min(1e-20))


@torch.no_grad()
def evaluate_grouped(
    model: AnchoredWeightCorrector,
    table,
    geometry: torch.Tensor,
    positions: torch.Tensor,
    spectra: torch.Tensor,
    pas_amplitude: torch.Tensor,
    pdp_amplitude: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    context = tensor_context(table, geometry.device)
    pas_values: list[np.ndarray] = []
    pdp_values: list[np.ndarray] = []
    correction_values: list[np.ndarray] = []
    for start in range(0, len(table.query_ids), batch_size):
        end = min(start + batch_size, len(table.query_ids))
        rows = torch.arange(start, end, device=geometry.device)
        guard = torch.zeros(end - start, dtype=torch.long, device=geometry.device)
        _, _, diagnostics = forward_rows(
            model, context, rows, guard, geometry, positions, spectra
        )
        donor_ids = context["donor_ids"][rows]
        query_ids = context["query_ids"][rows]
        predicted_pas, predicted_pdp = grouped_predictions(
            diagnostics["correction"] * 0.0
            + torch.stack(
                [diagnostics["pas_weights"], diagnostics["pdp_weights"]], dim=2
            ),
            donor_ids,
            pas_amplitude,
            pdp_amplitude,
        )
        pas_values.append(
            stable_power_cosine(predicted_pas, pas_amplitude[query_ids])
            .mean(dim=1)
            .cpu()
            .numpy()
        )
        pdp_values.append(
            stable_power_cosine(predicted_pdp, pdp_amplitude[query_ids])
            .mean(dim=1)
            .cpu()
            .numpy()
        )
        correction_values.append(
            diagnostics["correction"].abs().sum(dim=1).cpu().numpy()
        )
    pas = np.concatenate(pas_values)
    pdp = np.concatenate(pdp_values)
    correction = np.concatenate(correction_values)
    return {
        "pas": float(np.mean(pas)),
        "pdp": float(np.mean(pdp)),
        "mean": float(np.mean(0.5 * (pas + pdp))),
        "correction_l1_pas": float(np.mean(correction[:, 0])),
        "correction_l1_pdp": float(np.mean(correction[:, 1])),
    }


def refine_fold(
    fold: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: torch.device,
    pas_amplitude: torch.Tensor,
    pdp_amplitude: torch.Tensor,
) -> dict[str, object]:
    checkpoint = torch.load(
        ARTIFACT_DIR / f"fold{fold}_final.pt",
        map_location=device,
        weights_only=False,
    )
    config = Config(**checkpoint["config"])
    set_seed(config.seed + 50_000 + fold * 1009)
    _, positions_numpy, _, site, _ = load_positions()
    descriptors = np.load(DESCRIPTORS)
    valid = descriptors["valid"].astype(bool)
    spectra_numpy = np.concatenate(
        [
            np.sqrt(np.maximum(descriptors["global_pas"].astype(np.float32), 0.0)),
            np.sqrt(np.maximum(descriptors["global_pdp"].astype(np.float32), 0.0)),
        ],
        axis=1,
    ).astype(np.float32)
    geometry_numpy = np.load(GEOMETRY)["train"].astype(np.float32)
    folds = np.load(FOLD_CACHE)
    fit_ids = folds[f"fold{fold}_fit"].astype(np.int64)
    metadata = np.load(OOF_META)
    oof_query_ids = metadata["query_ids"].astype(np.int64)
    oof_fold_id = metadata["fold_id"].astype(np.int8)
    train_table = build_train_contexts(
        fold, positions_numpy, site, valid, fit_ids, config
    )
    validation_table = validation_context(fold, oof_query_ids, oof_fold_id)
    standardized, _, _ = standardize_geometry(geometry_numpy, fit_ids)
    geometry = torch.as_tensor(standardized, dtype=torch.float32, device=device)
    positions = torch.as_tensor(positions_numpy, dtype=torch.float32, device=device)
    spectra = torch.as_tensor(spectra_numpy, dtype=torch.float32, device=device)
    train_context = tensor_context(train_table, device)

    model = AnchoredWeightCorrector(geometry.shape[1], config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.seed + 50_000 + fold * 1009)
    history: list[dict[str, object]] = []
    baseline = evaluate_grouped(
        model,
        validation_table,
        geometry,
        positions,
        spectra,
        pas_amplitude,
        pdp_amplitude,
        batch_size,
    )
    history.append({"epoch": 0, "validation": baseline})
    best_score = baseline["mean"]
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    print(f"group fold={fold} epoch=0 validation={baseline}", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_table.query_ids))
        train_rows: list[tuple[float, float, float]] = []
        for start in range(0, len(order), batch_size):
            local = order[start : start + batch_size]
            rows = torch.as_tensor(local, dtype=torch.long, device=device)
            guard = torch.as_tensor(
                rng.integers(0, len(config.guard_radii_m), size=len(local)),
                dtype=torch.long,
                device=device,
            )
            global_pas, global_pdp, diagnostics = forward_rows(
                model,
                train_context,
                rows,
                guard,
                geometry,
                positions,
                spectra,
            )
            query_ids = train_context["query_ids"][rows]
            donor_ids = train_context["donor_ids"][rows, guard]
            combined_weights = torch.stack(
                [diagnostics["pas_weights"], diagnostics["pdp_weights"]], dim=2
            )
            grouped_pas, grouped_pdp = grouped_predictions(
                combined_weights,
                donor_ids,
                pas_amplitude,
                pdp_amplitude,
            )
            pas_cosine = stable_power_cosine(
                grouped_pas, pas_amplitude[query_ids]
            ).mean(dim=1)
            pdp_cosine = stable_power_cosine(
                grouped_pdp, pdp_amplitude[query_ids]
            ).mean(dim=1)
            global_cosine = 0.5 * (
                power_cosine(global_pas, spectra[query_ids, :PAS_SIZE])
                + power_cosine(global_pdp, spectra[query_ids, PAS_SIZE:])
            )
            correction = diagnostics["correction"]
            regularization = config.correction_regularization * (
                correction.square().sum(dim=1).mean()
                * config.neighbors
                / max(config.correction_l1_bound**2, 1e-8)
            )
            grouped_cosine = 0.5 * (pas_cosine + pdp_cosine)
            loss = 1.0 - (0.85 * grouped_cosine.mean() + 0.15 * global_cosine.mean())
            loss = loss + regularization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            train_rows.append(
                (
                    float(loss.detach().cpu()),
                    float(pas_cosine.mean().detach().cpu()),
                    float(pdp_cosine.mean().detach().cpu()),
                )
            )
        validation = evaluate_grouped(
            model,
            validation_table,
            geometry,
            positions,
            spectra,
            pas_amplitude,
            pdp_amplitude,
            batch_size,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean([row[0] for row in train_rows])),
                "train_pas": float(np.mean([row[1] for row in train_rows])),
                "train_pdp": float(np.mean([row[2] for row in train_rows])),
                "validation": validation,
            }
        )
        if validation["mean"] > best_score:
            best_score = validation["mean"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        epoch_checkpoint = ARTIFACT_DIR / f"fold{fold}_group_epoch{epoch}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": asdict(config),
                "feature_mean": checkpoint["feature_mean"],
                "feature_std": checkpoint["feature_std"],
                "base_epoch": int(checkpoint["epoch"]),
                "group_epoch": epoch,
            },
            epoch_checkpoint,
        )
        print(
            f"group fold={fold} epoch={epoch}/{epochs} "
            f"train={history[-1]['train_pas']:.6f}/{history[-1]['train_pdp']:.6f} "
            f"valid={validation['pas']:.6f}/{validation['pdp']:.6f}/"
            f"{validation['mean']:.6f} best={best_score:.6f}@{best_epoch}",
            flush=True,
        )

    best_checkpoint = ARTIFACT_DIR / f"fold{fold}_group_best.pt"
    torch.save(
        {
            "state_dict": best_state,
            "config": asdict(config),
            "feature_mean": checkpoint["feature_mean"],
            "feature_std": checkpoint["feature_std"],
            "base_epoch": int(checkpoint["epoch"]),
            "group_epoch": best_epoch,
        },
        best_checkpoint,
    )
    return {
        "fold": fold,
        "baseline": baseline,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "history": history,
        "best_checkpoint": PATHS.relative(best_checkpoint),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    pas_numpy, pdp_numpy = build_grouped_descriptors()
    device = torch.device(args.device)
    pas_amplitude = torch.as_tensor(
        np.asarray(pas_numpy, dtype=np.float32), device=device
    )
    pdp_amplitude = torch.as_tensor(
        np.asarray(pdp_numpy, dtype=np.float32), device=device
    )
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    records = [
        refine_fold(
            fold,
            args.epochs,
            args.learning_rate,
            args.batch_size,
            device,
            pas_amplitude,
            pdp_amplitude,
        )
        for fold in folds
    ]
    payload = {
        "experiment": "Round2-E4-V3-grouped-link-refinement",
        "status": "analysis_only",
        "primary_hypothesis": (
            "Official linkwise PAS/PDP structure grouped by UE, frequency and BS "
            "polarization provides a closer training objective than collapsed spectra."
        ),
        "sionna_used": False,
        "official_inputs_only": True,
        "base_epoch": 6,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "records": records,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
