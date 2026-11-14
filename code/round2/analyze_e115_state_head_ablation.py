#!/usr/bin/env python3
"""Attribute E115 gains and failures to individual predicted path states."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

import train_e115_complete_path_state_operator as e115
from project_paths import PATHS


VERSION = "Round2-E115-state-head-ablation"
REPORT = (
    PATHS.analysis
    / "results"
    / "round2_complete_path_state_operator"
    / "state_head_ablation.json"
)

STATE_COMBINATIONS: dict[str, tuple[str, ...]] = {
    "raw": (),
    "mass": ("mass",),
    "coordinate": ("coordinate",),
    "port": ("port",),
    "mass_coordinate": ("mass", "coordinate"),
    "mass_port": ("mass", "port"),
    "coordinate_port": ("coordinate", "port"),
    "mass_coordinate_port": ("mass", "coordinate", "port"),
    "mass_coordinate_port_phase": ("mass", "coordinate", "port", "phase"),
    "mass_coordinate_port_power": ("mass", "coordinate", "port", "power"),
    "full_without_zero": ("mass", "coordinate", "port", "phase", "power"),
    "full_soft_zero": (
        "mass",
        "coordinate",
        "port",
        "phase",
        "power",
        "soft_zero",
    ),
    "full_hard_zero": (
        "mass",
        "coordinate",
        "port",
        "phase",
        "power",
        "hard_zero",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=12)
    return parser.parse_args()


def combine_states(
    raw: dict[str, torch.Tensor],
    learned: dict[str, torch.Tensor],
    states: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    output = dict(raw)
    if "mass" in states:
        for key in ("mass", "mass_logits", "score_residual"):
            output[key] = learned[key]
    if "coordinate" in states:
        for key in ("coordinate", "coordinate_residual"):
            output[key] = learned[key]
    if "port" in states:
        for key in ("port", "port_residual"):
            output[key] = learned[key]
    if "phase" in states:
        for key in ("phase", "phase_residual"):
            output[key] = learned[key]
    if "power" in states:
        output["log_power"] = learned["log_power"]
    if "soft_zero" in states:
        nonzero_probability = 1.0 - torch.sigmoid(learned["zero_logit"])
        output["log_power"] = output["log_power"] + torch.log(
            nonzero_probability.clamp_min(1e-6)
        )
    if "hard_zero" in states:
        is_zero = torch.sigmoid(learned["zero_logit"]) >= 0.5
        output["log_power"] = torch.where(
            is_zero,
            torch.full_like(output["log_power"], -60.0),
            output["log_power"],
        )
    return output


@torch.inference_mode()
def evaluate_combinations(
    model: e115.CompletePathStateOperator,
    store: e115.TensorStore,
    rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    batch_size: int,
) -> dict[str, object]:
    train_h = np.load(e115.TRAIN_CHANNEL, mmap_mode="r")
    totals: dict[str, dict[str, float]] = {
        name: defaultdict(float) for name in STATE_COMBINATIONS
    }
    zero_probabilities: list[np.ndarray] = []
    zero_targets: list[np.ndarray] = []
    power_errors: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(rows), batch_size):
        local = rows[start : start + batch_size]
        row_tensor = torch.as_tensor(local, device=store.device)
        batch = store.batch(row_tensor, normalization)
        raw = e115.raw_output(batch)
        learned = model(batch)
        truth = torch.as_tensor(
            np.asarray(train_h[local], dtype=np.complex64), device=store.device
        )
        zero_probability = torch.sigmoid(learned["zero_logit"])
        zero_probabilities.append(zero_probability.cpu().numpy())
        zero_targets.append((batch["point_power"] <= 0.0).cpu().numpy())
        valid_power = batch["point_power"] > 0.0
        local_power_error = torch.full(
            (len(local),), np.nan, dtype=torch.float64, device=store.device
        )
        local_power_error[valid_power] = (
            learned["log_power"][valid_power].double()
            - torch.log(batch["point_power"][valid_power])
        ).abs()
        power_errors.append(local_power_error.cpu().numpy())
        target_total = batch["target_mass"].sum(dim=1, keepdim=True)
        target_distribution = batch["target_mass"] / target_total.clamp_min(e115.TINY)
        for name, states in STATE_COMBINATIONS.items():
            output = combine_states(raw, learned, states)
            pas, pdp, _, _ = e115.sufficient_scores(output, batch)
            mass_cosine = (
                torch.sqrt(output["mass"].clamp_min(1e-12))
                * torch.sqrt(target_distribution.clamp_min(0.0))
            ).sum(dim=1)
            prediction = e115.render_complex_channel(output)
            complex_pas, complex_pdp, error, energy = e115.torch_official_metrics(
                prediction, truth
            )
            accumulator = totals[name]
            accumulator["path_pas"] += float(pas.sum().item())
            accumulator["path_pdp"] += float(pdp.sum().item())
            accumulator["mass_cosine"] += float(mass_cosine.sum().item())
            accumulator["complex_pas"] += float(complex_pas.sum().item())
            accumulator["complex_pdp"] += float(complex_pdp.sum().item())
            accumulator["squared_error"] += float(error.sum().item())
            accumulator["truth_energy"] += float(energy.sum().item())

    result: dict[str, object] = {}
    for name, accumulator in totals.items():
        nmse = accumulator["squared_error"] / max(
            accumulator["truth_energy"], e115.TINY
        )
        complex_pas = accumulator["complex_pas"] / len(rows)
        complex_pdp = accumulator["complex_pdp"] / len(rows)
        nmse_score = 1.0 / (1.0 + nmse)
        result[name] = {
            "states": list(STATE_COMBINATIONS[name]),
            "path_endpoint_pas": accumulator["path_pas"] / len(rows),
            "path_endpoint_pdp": accumulator["path_pdp"] / len(rows),
            "path_endpoint_mean": (
                accumulator["path_pas"] + accumulator["path_pdp"]
            )
            / (2.0 * len(rows)),
            "mass_hellinger_cosine": accumulator["mass_cosine"] / len(rows),
            "complex_pas": complex_pas,
            "complex_pdp": complex_pdp,
            "complex_nmse": nmse,
            "complex_nmse_score": nmse_score,
            "complex_score": 0.4 * complex_pas + 0.4 * complex_pdp + 0.2 * nmse_score,
        }
    zero_probability = np.concatenate(zero_probabilities)
    is_zero = np.concatenate(zero_targets).astype(bool)
    power_error = np.concatenate(power_errors)
    return {
        "count": int(len(rows)),
        "predicted_zero_probability_zero_mean": float(
            np.mean(zero_probability[is_zero])
        )
        if np.any(is_zero)
        else None,
        "predicted_zero_probability_nonzero_mean": float(
            np.mean(zero_probability[~is_zero])
        )
        if np.any(~is_zero)
        else None,
        "predicted_log_power_mae": float(np.nanmean(power_error)),
        "combinations": result,
        "best_path_endpoint": max(
            result, key=lambda key: float(result[key]["path_endpoint_mean"])
        ),
        "best_complex_score": max(
            result, key=lambda key: float(result[key]["complex_score"])
        ),
    }


def load_model(
    checkpoint_path: Path,
    store: e115.TensorStore,
    device: torch.device,
) -> tuple[
    e115.CompletePathStateOperator,
    dict[str, torch.Tensor],
    dict[str, object],
]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = e115.Config(**checkpoint["config"])
    variant = e115.Variant(**checkpoint["variant"])
    model = e115.CompletePathStateOperator(
        store.candidate_feature.shape[-1],
        store.context.shape[-1],
        store.graph_feature.shape[-1],
        config,
        variant,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    normalization = {
        key: value.to(device) for key, value in checkpoint["normalization"].items()
    }
    metadata = {
        "variant": checkpoint["variant"],
        "config": checkpoint["config"],
        "effective_seed": int(checkpoint["effective_seed"]),
        "best_epoch": int(checkpoint["best_epoch"]),
        "checkpoint": PATHS.relative(checkpoint_path),
    }
    return model, normalization, metadata


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    store = e115.TensorStore(e115.cache_dir("screen_block01"), device)
    fold_id = store.fold_id.cpu().numpy()
    split_rows = {
        "validation_fold1": np.flatnonzero(fold_id == 1),
        "held_fold0": np.flatnonzero(fold_id == 0),
    }
    variants: dict[str, object] = {}
    for variant in e115.VARIANTS:
        checkpoint_path = e115.ARTIFACT_DIR / f"outer0_screen_{variant.name}.pt"
        model, normalization, metadata = load_model(checkpoint_path, store, device)
        variants[variant.name] = {
            **metadata,
            "splits": {
                split_name: evaluate_combinations(
                    model, store, rows, normalization, args.batch_size
                )
                for split_name, rows in split_rows.items()
            },
        }
        print(f"finished variant={variant.name}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "source_experiment": e115.VERSION,
        "purpose": (
            "Isolate whether learned mass, coordinate, port, phase, power, or an "
            "unapplied zero-coverage head causes the E115 screen failure."
        ),
        "official_h_only": True,
        "sionna_used": False,
        "variants": variants,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
