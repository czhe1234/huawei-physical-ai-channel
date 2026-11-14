#!/usr/bin/env python3
"""Fit a low-dimensional global Sionna-to-official angle mapping."""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import numpy as np
import torch

import analyze_path_candidate_information_audit as e84
from data import load_positions
from project_paths import PATHS


VERSION = "Round2-E98-global-array-angle-mapping"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "global_array_angle_mapping"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_global_array_angle_mapping"
PARAMETERS = ARTIFACT_DIR / "strict_fold_angle_parameters.npz"
REPORT = ANALYSIS_DIR / "global_array_angle_mapping.json"
TINY = 1e-30
CURRENT_YAW_DEG = (-44.0, 73.5)
CURRENT_TILT_DEG = (-11.5, -10.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    return parser.parse_args()


def periodic_delta(value: torch.Tensor, period: float) -> torch.Tensor:
    return torch.remainder(value + period / 2.0, period) - period / 2.0


def decode_parameters(latent: torch.Tensor) -> dict[str, torch.Tensor]:
    bounded = torch.tanh(latent)
    yaw = torch.deg2rad(
        torch.tensor(CURRENT_YAW_DEG, device=latent.device)
    ) + math.radians(30.0) * bounded[0:2]
    tilt = torch.deg2rad(
        torch.tensor(CURRENT_TILT_DEG, device=latent.device)
    ) + math.radians(18.0) * bounded[2:4]
    horizontal_spacing = 0.50 + 0.18 * bounded[4]
    vertical_spacing = 4.00 + 1.20 * bounded[5]
    horizontal_offset = 4.0 * bounded[6:8]
    vertical_offset = 2.0 + 4.0 * bounded[8:10]
    return {
        "yaw": yaw,
        "tilt": tilt,
        "horizontal_spacing": horizontal_spacing,
        "vertical_spacing": vertical_spacing,
        "horizontal_offset": horizontal_offset,
        "vertical_offset": vertical_offset,
    }


def serializable_parameters(latent: torch.Tensor) -> dict[str, Any]:
    decoded = decode_parameters(latent.detach())
    return {
        "yaw_deg": torch.rad2deg(decoded["yaw"]).cpu().numpy().tolist(),
        "tilt_deg": torch.rad2deg(decoded["tilt"]).cpu().numpy().tolist(),
        "horizontal_spacing_lambda": float(decoded["horizontal_spacing"].item()),
        "vertical_spacing_lambda": float(decoded["vertical_spacing"].item()),
        "horizontal_offset_bins": decoded["horizontal_offset"].cpu().numpy().tolist(),
        "vertical_offset_bins": decoded["vertical_offset"].cpu().numpy().tolist(),
        "latent": latent.detach().cpu().numpy().tolist(),
    }


class MappingData:
    def __init__(self, device: torch.device) -> None:
        component = np.load(e84.COMPONENTS, allow_pickle=False)
        coordinates = component["coordinates"].astype(np.float32)
        coefficients = component["coefficients"].astype(np.complex64)
        point_power = component["point_power"].astype(np.float64)
        energy = np.sum(np.abs(coefficients) ** 2, axis=2, dtype=np.float64)
        mass = np.divide(
            energy,
            np.maximum(point_power[:, None], TINY),
            out=np.zeros_like(energy),
            where=point_power[:, None] > TINY,
        )
        valid = (point_power[:, None] > TINY) & (mass >= 1e-5)
        angles = np.array(
            np.load(e84.PATH_CACHE / "angles.npy", mmap_mode="r"),
            dtype=np.float32,
            copy=True,
        )
        tau = np.array(
            np.load(e84.PATH_CACHE / "tau.npy", mmap_mode="r"),
            dtype=np.float32,
            copy=True,
        )
        mask = np.array(
            np.load(e84.PATH_CACHE / "mask.npy", mmap_mode="r"),
            dtype=bool,
            copy=True,
        )
        _, _, _, site, _ = load_positions()
        self.truth = torch.as_tensor(coordinates, device=device)
        self.mass = torch.as_tensor(mass.astype(np.float32), device=device)
        self.truth_valid = torch.as_tensor(valid, device=device)
        self.theta = torch.as_tensor(angles[..., 0], device=device)
        self.phi = torch.as_tensor(angles[..., 1], device=device)
        self.delay = torch.as_tensor(
            np.remainder(tau * 30_000.0 * 192.0, 192.0).astype(np.float32),
            device=device,
        )
        self.candidate_valid = torch.as_tensor(mask, device=device)
        self.site = torch.as_tensor(site.astype(np.int64), device=device)
        self.fold = np.load(e84.FULL_FOLD).astype(np.int8)

    def candidate_angles(
        self, rows: torch.Tensor, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parameters = decode_parameters(latent)
        site = self.site[rows]
        yaw = parameters["yaw"][site][:, None]
        tilt = parameters["tilt"][site][:, None]
        theta = self.theta[rows]
        phi = self.phi[rows]
        horizontal = (
            16.0
            * parameters["horizontal_spacing"]
            * torch.sin(theta)
            * torch.sin(phi - yaw)
            + parameters["horizontal_offset"][site][:, None]
        )
        vertical = (
            8.0
            * parameters["vertical_spacing"]
            * (
                torch.sin(tilt) * torch.sin(theta) * torch.cos(phi - yaw)
                - torch.cos(tilt) * torch.cos(theta)
            )
            + parameters["vertical_offset"][site][:, None]
        )
        return horizontal, vertical

    def costs(
        self, rows: torch.Tensor, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        horizontal, vertical = self.candidate_angles(rows, latent)
        truth = self.truth[rows]
        dh = periodic_delta(
            horizontal[:, None, :] - truth[:, :, 0, None], 16.0
        )
        dv = periodic_delta(vertical[:, None, :] - truth[:, :, 1, None], 8.0)
        dd = periodic_delta(
            self.delay[rows, None, :] - truth[:, :, 2, None], 192.0
        )
        horizontal_cost = torch.square(dh / 1.5)
        vertical_cost = torch.square(dv)
        delay_cost = torch.square(dd / 2.0)
        invalid = ~self.candidate_valid[rows, None, :]
        return (
            (horizontal_cost + vertical_cost + delay_cost).masked_fill(invalid, 1e6),
            (horizontal_cost + delay_cost).masked_fill(invalid, 1e6),
            (vertical_cost + delay_cost).masked_fill(invalid, 1e6),
            delay_cost.masked_fill(invalid, 1e6),
        )


def batch_loss(data: MappingData, rows: np.ndarray, latent: torch.Tensor) -> torch.Tensor:
    ids = torch.as_tensor(rows, device=latent.device)
    full_cost, _, _, _ = data.costs(ids, latent)
    minimum = full_cost.amin(dim=2)
    weight = data.mass[ids] * data.truth_valid[ids]
    data_loss = (weight * torch.log1p(minimum)).sum() / weight.sum().clamp_min(TINY)
    return data_loss + 2e-4 * latent.square().mean()


@torch.inference_mode()
def evaluate(
    data: MappingData,
    rows: np.ndarray,
    latent: torch.Tensor,
    batch_size: int = 256,
) -> dict[str, float | int]:
    covered = {name: 0.0 for name in ("full", "horizontal_delay", "vertical_delay", "delay")}
    thresholds = (0.25, 1.0, 4.0, 9.0)
    threshold_mass = {threshold: 0.0 for threshold in thresholds}
    total_mass = 0.0
    weighted_cost = 0.0
    weighted_soft = 0.0
    valid_components = 0
    for start in range(0, len(rows), batch_size):
        local = torch.as_tensor(rows[start : start + batch_size], device=latent.device)
        costs = data.costs(local, latent)
        minima = [value.amin(dim=2) for value in costs]
        valid = data.truth_valid[local]
        mass = data.mass[local] * valid
        local_total = mass.sum(dtype=torch.float64)
        total_mass += float(local_total.item())
        valid_components += int(valid.sum().item())
        for name, minimum in zip(covered, minima):
            covered[name] += float(
                (mass * (minimum <= 0.25)).sum(dtype=torch.float64).item()
            )
        for threshold in thresholds:
            threshold_mass[threshold] += float(
                (mass * (minima[0] <= threshold)).sum(dtype=torch.float64).item()
            )
        weighted_cost += float(
            (mass * torch.log1p(minima[0])).sum(dtype=torch.float64).item()
        )
        weighted_soft += float(
            (mass * torch.exp(-0.5 * minima[0].clamp_max(50.0)))
            .sum(dtype=torch.float64)
            .item()
        )
    denominator = max(total_mass, TINY)
    return {
        "point_count": int(len(rows)),
        "valid_components": valid_components,
        "mass": total_mass,
        "full_tight_coverage": covered["full"] / denominator,
        "horizontal_delay_tight_coverage": covered["horizontal_delay"] / denominator,
        "vertical_delay_tight_coverage": covered["vertical_delay"] / denominator,
        "delay_tight_coverage": covered["delay"] / denominator,
        "full_coverage_thresholds": {
            str(threshold): threshold_mass[threshold] / denominator
            for threshold in thresholds
        },
        "weighted_log1p_min_cost": weighted_cost / denominator,
        "weighted_soft_coverage": weighted_soft / denominator,
    }


def optimize_fold(
    outer: int,
    data: MappingData,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    train_rows = np.flatnonzero(data.fold != outer)
    held_rows = np.flatnonzero(data.fold == outer)
    baseline = torch.zeros(10, device=device)
    baseline_train = evaluate(data, train_rows, baseline)
    baseline_held = evaluate(data, held_rows, baseline)
    attempts = []
    best_latent = None
    best_objective = math.inf
    for restart in range(args.restarts):
        seed = 2026080798 + 1000 * outer + 100 * restart
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        initial = torch.zeros(10, device=device)
        if restart > 0:
            initial = 0.15 * torch.randn(10, device=device)
        latent = torch.nn.Parameter(initial)
        optimizer = torch.optim.AdamW(
            [latent], lr=args.learning_rate, weight_decay=1e-4
        )
        history = []
        for step in range(1, args.steps + 1):
            batch = rng.choice(
                train_rows,
                size=min(args.batch_size, len(train_rows)),
                replace=False,
            )
            loss = batch_loss(data, batch, latent)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(latent.grad).all():
                raise FloatingPointError(
                    f"non-finite E98 gradient fold={outer} restart={restart} step={step}"
                )
            optimizer.step()
            if step % 50 == 0 or step == args.steps:
                history.append({"step": step, "loss": float(loss.detach().item())})
                print(
                    f"E98 fold={outer} restart={restart} step={step}/{args.steps} "
                    f"loss={loss.detach().item():.6f}",
                    flush=True,
                )
        train_metrics = evaluate(data, train_rows, latent)
        objective = float(train_metrics["weighted_log1p_min_cost"])
        attempt = {
            "restart": restart,
            "history": history,
            "parameters": serializable_parameters(latent),
            "train": train_metrics,
        }
        attempts.append(attempt)
        if objective < best_objective:
            best_objective = objective
            best_latent = latent.detach().clone()
    if best_latent is None:
        raise RuntimeError(f"E98 fold {outer} did not produce parameters")
    held = evaluate(data, held_rows, best_latent)
    return best_latent, {
        "fold": outer,
        "train_count": int(len(train_rows)),
        "held_count": int(len(held_rows)),
        "baseline_train": baseline_train,
        "baseline_held": baseline_held,
        "selected_parameters": serializable_parameters(best_latent),
        "held": held,
        "held_tight_delta": float(
            held["full_tight_coverage"] - baseline_held["full_tight_coverage"]
        ),
        "attempts": attempts,
    }


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    started = time.time()
    data = MappingData(device)
    latent_by_fold = []
    folds = []
    for outer in range(5):
        latent, record = optimize_fold(outer, data, args, device)
        latent_by_fold.append(latent.cpu().numpy().astype(np.float32))
        folds.append(record)
    baseline = torch.zeros(10, device=device)
    baseline_full = evaluate(data, np.arange(len(data.fold)), baseline)
    oof_covered = 0.0
    oof_total = 0.0
    for fold, latent_np in enumerate(latent_by_fold):
        rows = np.flatnonzero(data.fold == fold)
        local = evaluate(data, rows, torch.as_tensor(latent_np, device=device))
        oof_covered += float(local["full_tight_coverage"] * local["mass"])
        oof_total += float(local["mass"])
    oof_coverage = oof_covered / max(oof_total, TINY)
    oof_delta = float(oof_coverage - baseline_full["full_tight_coverage"])
    latent_array = np.stack(latent_by_fold)
    np.savez_compressed(
        PARAMETERS,
        fold_id=np.arange(5, dtype=np.int8),
        latent=latent_array,
    )
    report = {
        "experiment": VERSION,
        "status": "strict_complete",
        "primary_hypothesis": (
            "A residual global Sionna-to-official array-coordinate mismatch is a major "
            "cause of the PAS gap and can be identified from official path components."
        ),
        "configuration": {
            **vars(args),
            "current_yaw_deg": CURRENT_YAW_DEG,
            "current_tilt_deg": CURRENT_TILT_DEG,
            "current_horizontal_spacing_lambda": 0.50,
            "current_vertical_spacing_lambda": 4.00,
            "current_horizontal_offset_bins": (0.0, 0.0),
            "current_vertical_offset_bins": (2.0, 2.0),
            "tight_cost": 0.25,
            "support_scales": (1.5, 1.0, 2.0),
        },
        "baseline_full": baseline_full,
        "folds": folds,
        "strict_oof": {
            "full_tight_coverage": oof_coverage,
            "delta": oof_delta,
            "positive_folds": int(sum(row["held_tight_delta"] > 0.0 for row in folds)),
            "worst_fold_delta": float(min(row["held_tight_delta"] for row in folds)),
        },
        "parameter_stability": {
            "latent_std": latent_array.std(axis=0).tolist(),
            "maximum_latent_std": float(latent_array.std(axis=0).max()),
        },
        "decision": {
            "passed": bool(
                oof_delta >= 0.02
                and all(row["held_tight_delta"] > 0.0 for row in folds)
            ),
            "minimum_oof_delta": 0.02,
            "required_positive_folds": 5,
            "retrain_surface_model": False,
        },
        "output": PATHS.relative(PARAMETERS),
        "seconds": float(time.time() - started),
    }
    report["decision"]["retrain_surface_model"] = report["decision"]["passed"]
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
