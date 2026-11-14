#!/usr/bin/env python3
"""Screen frozen-physics calibration variants for the E88 dense student."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
import time
from typing import Any

import numpy as np
import torch

from project_paths import PATHS
import train_e88_dense_path_residual_student as e88
import train_offline_sionna_geometry_student as e87
import train_surface_interaction_energy_model as e85


VERSION = "Round2-E89-frozen-physics-calibrator"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "frozen_physics_calibrator"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_frozen_physics_calibrator"
SCREEN_REPORT = ANALYSIS_DIR / "frozen_physics_outer0_screen.json"
MODES = ("freeze_grid", "freeze_geometry", "heads_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def configure_trainable(model: e88.DensePathResidualStudent, mode: str) -> int:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if mode == "freeze_grid":
        for parameter in model.spatial_field.grids.parameters():
            parameter.requires_grad_(False)
    elif mode == "freeze_geometry":
        for module in (
            model.scalar,
            model.polar,
            model.corridor,
            model.spatial_field,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    elif mode == "heads_only":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for module in (
            model.coordinate_head,
            model.mass_head,
            model.port_head,
            model.pas_residual,
            model.pdp_residual,
            model.valid_head,
            model.reliability_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    else:
        raise ValueError(mode)
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def train_outer0(
    mode: str,
    config: e88.Config,
    geometry: e88.GeometryStore,
    official_pas: torch.Tensor,
    official_pdp: torch.Tensor,
    teacher_pas: torch.Tensor,
    teacher_pdp: torch.Tensor,
    power: np.ndarray,
    fit: np.ndarray,
    validation: np.ndarray,
    epochs: int,
) -> tuple[e88.DensePathResidualStudent, dict[str, Any]]:
    checkpoint = torch.load(e88.PRETRAIN_CHECKPOINT, map_location="cpu", weights_only=False)
    seed = config.random_state + 89_017
    torch.manual_seed(seed)
    if geometry.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = e88.build_model(geometry, config)
    model.load_state_dict(checkpoint["state_dict"])
    trainable = configure_trainable(model, mode)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.finetune_learning_rate,
        weight_decay=config.weight_decay,
        fused=geometry.device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=config.finetune_learning_rate * 0.05,
    )
    rng = np.random.default_rng(seed + 1)
    best_score = -math.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        order = rng.permutation(fit)
        for start in range(0, len(order), config.finetune_batch_size):
            local_np = order[start : start + config.finetune_batch_size]
            local = torch.as_tensor(local_np, device=geometry.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=geometry.device.type,
                dtype=torch.bfloat16,
                enabled=geometry.device.type == "cuda",
            ):
                output = model(geometry.batch(local_np))
                loss = e88.official_objective(
                    output,
                    official_pas[local],
                    official_pdp[local],
                    teacher_pas[local],
                    teacher_pdp[local],
                    torch.as_tensor(power[local_np] > 0.0, device=geometry.device),
                    config,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite E89 loss mode={mode} epoch={epoch}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"non-finite E89 gradient mode={mode} epoch={epoch}"
                )
            optimizer.step()
            losses.append(float(loss.detach().item()))
        scheduler.step()
        if epoch % config.validation_interval == 0 or epoch == epochs:
            metrics = e88.official_endpoint_score(
                model, geometry, validation, official_pas, official_pdp, config
            )
            row = {"epoch": epoch, "loss": float(np.mean(losses)), **metrics}
            history.append(row)
            if row["mean"] > best_score:
                best_score = row["mean"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"E89 mode={mode} epoch={epoch}/{epochs} "
                f"loss={row['loss']:.6f} validation={row['mean']:.6f}",
                flush=True,
            )
    model.load_state_dict(best_state)
    return model, {
        "mode": mode,
        "trainable_parameters": trainable,
        "best_epoch": best_epoch,
        "best_validation": best_score,
        "history": history,
    }


@torch.inference_mode()
def held_fusion_scan(
    prediction: dict[str, np.ndarray],
    held_mask: np.ndarray,
    held_ids: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    truth = torch.as_tensor(
        np.asarray(truth_all[held_ids], dtype=np.complex64), device=device
    )
    truth_pas = e87.pas_distribution(truth)
    truth_pdp = e87.pdp_distribution(truth)
    endpoint_pas, endpoint_pdp = e87.expanded_endpoint(
        prediction["pas"], prediction["pdp"], device
    )
    output = {}
    for name, path in (("V11", e87.V11_CHANNEL), ("V12", e87.V12_CHANNEL)):
        base = torch.as_tensor(
            np.asarray(np.load(path, mmap_mode="r")[held_mask], dtype=np.complex64),
            device=device,
        )
        base_pas = e87.pas_distribution(base)
        base_pdp = e87.pdp_distribution(base)
        base_values = e87.point_metrics(base, truth, truth_pas, truth_pdp)
        baseline = e87.statistics(*base_values)
        rows = []
        for beta_pas in (0.0, 0.05, 0.10, 0.15, 0.20):
            desired_pas = e87.endpoint_target(
                base_pas, endpoint_pas, beta_pas, "sqrt", (1, 2, 3)
            )
            for beta_pdp in (0.05, 0.10, 0.20, 0.30, 0.40):
                desired_pdp = e87.endpoint_target(
                    base_pdp, endpoint_pdp, beta_pdp, "sqrt", (3,)
                )
                for order in ("pas_pdp", "pdp_pas"):
                    channel = e87.project_joint(
                        base, desired_pas, desired_pdp, 4, order
                    )
                    metrics = e87.statistics(
                        *e87.point_metrics(
                            channel, truth, truth_pas, truth_pdp
                        )
                    )
                    rows.append(
                        {
                            "beta_pas": beta_pas,
                            "beta_pdp": beta_pdp,
                            "order": order,
                            "metrics": metrics,
                            "delta": float(metrics["score"] - baseline["score"]),
                        }
                    )
        output[name] = {
            "baseline": baseline,
            "best": max(rows, key=lambda row: float(row["metrics"]["score"])),
            "fixed_v11_asymmetric": next(
                row
                for row in rows
                if row["beta_pas"] == 0.15
                and row["beta_pdp"] == 0.30
                and row["order"] == "pas_pdp"
            ),
            "fixed_v12_conservative": next(
                row
                for row in rows
                if row["beta_pas"] == 0.05
                and row["beta_pdp"] == 0.20
                and row["order"] == "pdp_pas"
            ),
        }
    return output


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    checkpoint = torch.load(e88.PRETRAIN_CHECKPOINT, map_location="cpu", weights_only=False)
    config = e88.Config(
        **{
            **checkpoint["config"],
            "finetune_epochs": args.epochs,
            "finetune_batch_size": args.batch_size,
        }
    )
    geometry = e88.GeometryStore("train", device)
    teacher = np.load(
        PATHS.artifacts
        / "round2"
        / "offline_sionna_geometry_student"
        / "raw_sionna_sufficient_teacher.npz",
        allow_pickle=False,
    )
    official_pas = torch.as_tensor(
        np.array(np.load(e85.TARGET_PAS, mmap_mode="r"), dtype=np.float32, copy=True),
        device=device,
    )
    official_pdp = torch.as_tensor(
        np.array(np.load(e85.TARGET_PDP, mmap_mode="r"), dtype=np.float32, copy=True),
        device=device,
    )
    teacher_pas = torch.as_tensor(teacher["pas"].astype(np.float32), device=device)
    teacher_pdp = torch.as_tensor(teacher["pdp"].astype(np.float32), device=device)
    power = np.asarray(np.load(e85.POINT_POWER), dtype=np.float64)
    metadata = np.load(e88.OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    held_mask = fold_id == 0
    validation_mask = fold_id == 1
    held = query_ids[held_mask]
    validation = query_ids[validation_mask]
    excluded = np.zeros(len(geometry.positions), dtype=bool)
    excluded[held] = True
    excluded[validation] = True
    fit = np.flatnonzero(~excluded)
    started = time.time()
    results = []
    for mode in MODES:
        model, training = train_outer0(
            mode,
            config,
            geometry,
            official_pas,
            official_pdp,
            teacher_pas,
            teacher_pdp,
            power,
            fit,
            validation,
            args.epochs,
        )
        prediction = e88.predict_official(
            model, geometry, held, config.finetune_batch_size
        )
        endpoint = e88.endpoint_summary(
            prediction["pas"],
            prediction["pdp"],
            official_pas[torch.as_tensor(held, device=device)],
            official_pdp[torch.as_tensor(held, device=device)],
            np.zeros(len(held), dtype=np.int8),
            device,
        )
        fusion = held_fusion_scan(prediction, held_mask, held, device)
        checkpoint_path = ARTIFACT_DIR / f"outer0_{mode}.pt"
        torch.save(
            {
                "experiment": VERSION,
                "mode": mode,
                "config": asdict(config),
                "state_dict": model.state_dict(),
                "sionna_inputs_required_for_inference": False,
            },
            checkpoint_path,
        )
        results.append(
            {
                "mode": mode,
                "training": training,
                "held_endpoint": endpoint,
                "held_fusion": fusion,
                "checkpoint": PATHS.relative(checkpoint_path),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "experiment": VERSION,
        "status": "outer0_screen_complete",
        "primary_hypothesis": (
            "Freezing the dense Sionna spatial field during official-H calibration "
            "preserves learned path continuity and improves unseen-hole transfer."
        ),
        "configuration": asdict(config),
        "results": results,
        "seconds": time.time() - started,
    }
    SCREEN_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
