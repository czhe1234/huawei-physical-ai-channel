#!/usr/bin/env python3
"""Train an identity-supervised path action gate on the frozen E150 interface."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

import analyze_e119_v12_unique_path_complement as e119
import analyze_e148_v7_pathwise_neural_residual as e148
from data import load_positions
from project_paths import PATHS
from run_rbf_spectral_field import statistics
import train_e115_complete_path_state_operator as e115
import train_e150_v7_pathwise_carrier_operator as e150


VERSION = "Round2-E151-identity-supervised-action-gate"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "identity_supervised_action_gate"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_identity_supervised_action_gate"
REPORT = ANALYSIS_DIR / "identity_supervised_action_gate.json"
METRICS = ARTIFACT_DIR / "identity_supervised_action_gate_metrics.npz"
TINY = 1e-30


@dataclass(frozen=True)
class Config(e150.Config):
    action_warmup_epochs: int = 12
    finetune_epochs: int = 12
    action_learning_rate: float = 2e-4
    finetune_learning_rate: float = 5e-5
    action_limit: float = 0.50
    action_supervision_weight: float = 0.35
    action_gate_weight: float = 0.25
    action_delta_weight: float = 0.75
    negative_action_weight: float = 0.08
    seed: int = 20_260_810_151


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--outer", default="0")
    parser.add_argument("--phase", choices=("train", "all"), default="all")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=18)
    parser.add_argument("--action-warmup-epochs", type=int, default=12)
    parser.add_argument("--finetune-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--metric-batch-size", type=int, default=8)
    return parser.parse_args()


def checkpoint_path(outer: int):
    return ARTIFACT_DIR / f"outer{outer}_identity_supervised_action.pt"


def action_supervision_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["mask"]
    positive = batch["target_positive"] & mask
    negative = (~batch["target_positive"]) & mask
    local = batch["source_type"] == 0
    base = torch.where(
        local[..., None],
        batch["v7_coefficient"],
        torch.zeros_like(batch["v7_coefficient"]),
    )
    target = torch.where(
        positive[..., None],
        batch["target_coefficient"],
        base,
    )
    predicted_delta = output["action"][..., None] * (
        output["state_coefficient"] - base
    )
    target_delta = target - base
    point_scale = torch.sqrt(batch["target_point_power"].clamp_min(1e-12)).to(
        predicted_delta.dtype
    )
    normalized_prediction = predicted_delta / point_scale[:, None, None]
    normalized_target = target_delta / point_scale[:, None, None]

    if positive.any():
        mass = batch["target_mass"]
        maximum = mass.max(dim=1, keepdim=True).values.clamp_min(1e-8)
        relative_mass = torch.sqrt((mass / maximum).clamp_min(0.0))
        positive_weight = relative_mass[positive]
        complex_error = (
            normalized_prediction[positive] - normalized_target[positive]
        ).abs().square().mean(dim=1)
        delta_loss = (
            complex_error * positive_weight
        ).sum() / positive_weight.sum().clamp_min(1e-8)
        target_action = config.action_limit * relative_mass[positive]
        gate_loss = F.smooth_l1_loss(
            output["action"][positive], target_action
        )
    else:
        delta_loss = output["action"].sum() * 0.0
        gate_loss = output["action"].sum() * 0.0
    if negative.any():
        negative_energy = normalized_prediction[negative].abs().square().mean(dim=1)
        delta_loss = delta_loss + config.negative_action_weight * negative_energy.mean()
        gate_loss = gate_loss + config.negative_action_weight * F.smooth_l1_loss(
            output["action"][negative],
            torch.zeros_like(output["action"][negative]),
        )
    loss = (
        config.action_delta_weight * delta_loss
        + config.action_gate_weight * gate_loss
    )
    return loss, {
        "action_supervision": float(loss.detach().cpu()),
        "delta_loss": float(delta_loss.detach().cpu()),
        "gate_loss": float(gate_loss.detach().cpu()),
        "positive_action_mean": float(output["action"][positive].mean().detach().cpu())
        if positive.any()
        else 0.0,
        "negative_action_abs_mean": float(
            output["action"][negative].abs().mean().detach().cpu()
        )
        if negative.any()
        else 0.0,
    }


@torch.inference_mode()
def validation_curve(
    model: e150.PathwiseCarrierOperator,
    store: e150.FoldStore,
    rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    config: Config,
) -> list[dict[str, object]]:
    output = []
    for alpha in (0.0, 0.10, 0.25, 0.50, 0.75, 1.0):
        values, diagnostics = e150.evaluate_model_rows(
            model, store, rows, normalization, alpha, config
        )
        output.append(
            {
                "alpha": float(alpha),
                "metrics": statistics(*values),
                "diagnostics": diagnostics,
            }
        )
    return output


def train_action_warmup(
    model: e150.PathwiseCarrierOperator,
    store: e150.FoldStore,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    config: Config,
    seed: int,
) -> tuple[dict[str, torch.Tensor], float, dict[str, object]]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.action_head.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        model.action_head.parameters(),
        lr=config.action_learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.action_warmup_epochs, 1),
        eta_min=config.action_learning_rate * 0.05,
    )
    baseline_values = tuple(
        value[validation_rows].cpu().numpy()
        for value in (
            store.base_pas,
            store.base_pdp,
            store.base_error,
            store.base_energy,
        )
    )
    baseline = statistics(*baseline_values)
    best_score = float(baseline["score"])
    best_epoch = 0
    best_alpha = 0.0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, object]] = [
        {
            "epoch": 0,
            "selected_alpha": 0.0,
            "selected_validation": baseline,
        }
    ]
    rng = np.random.default_rng(seed)
    for epoch in range(1, config.action_warmup_epochs + 1):
        model.train()
        order = rng.permutation(fit_rows)
        losses = []
        for start in range(0, len(order), config.batch_size):
            local = torch.as_tensor(
                order[start : start + config.batch_size], device=store.device
            )
            batch = store.batch(local, normalization)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss, _ = action_supervision_loss(output, batch, config)
            loss.backward()
            e150.stable_clip_gradients(model.action_head.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        row: dict[str, object] = {
            "epoch": epoch,
            "train_action_loss": float(np.mean(losses)),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        if epoch == 1 or epoch % config.validation_interval == 0 or epoch == config.action_warmup_epochs:
            curve = validation_curve(
                model, store, validation_rows, normalization, config
            )
            winner = max(curve, key=lambda item: float(item["metrics"]["score"]))
            row["alpha_curve"] = curve
            row["selected_alpha"] = float(winner["alpha"])
            row["selected_validation"] = winner["metrics"]
            score = float(winner["metrics"]["score"])
            if score > best_score + 1e-10:
                best_score = score
                best_epoch = epoch
                best_alpha = float(winner["alpha"])
                best_state = copy.deepcopy(model.state_dict())
        history.append(row)
        if epoch == 1 or epoch % 2 == 0 or epoch == config.action_warmup_epochs:
            message = (
                f"E151 outer={store.outer} action={epoch}/{config.action_warmup_epochs} "
                f"train={row['train_action_loss']:.6f}"
            )
            if "selected_validation" in row:
                message += (
                    f" inner={row['selected_validation']['score']:.6f} "
                    f"alpha={row['selected_alpha']:.2f}"
                )
            print(message, flush=True)
    model.load_state_dict(best_state)
    for parameter in model.parameters():
        parameter.requires_grad = True
    return best_state, best_alpha, {
        "best_epoch": best_epoch,
        "best_alpha": best_alpha,
        "best_validation_score": best_score,
        "history": history,
    }


def train_finetune(
    model: e150.PathwiseCarrierOperator,
    store: e150.FoldStore,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    config: Config,
    initial_alpha: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], float, dict[str, object]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.finetune_learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.finetune_epochs, 1),
        eta_min=config.finetune_learning_rate * 0.05,
    )
    initial_curve = validation_curve(
        model, store, validation_rows, normalization, config
    )
    initial = max(initial_curve, key=lambda item: float(item["metrics"]["score"]))
    best_score = float(initial["metrics"]["score"])
    best_epoch = 0
    best_alpha = float(initial["alpha"] if initial_alpha > 0.0 else initial["alpha"])
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, object]] = [
        {
            "epoch": 0,
            "alpha_curve": initial_curve,
            "selected_alpha": best_alpha,
            "selected_validation": initial["metrics"],
        }
    ]
    rng = np.random.default_rng(seed)
    for epoch in range(1, config.finetune_epochs + 1):
        model.train()
        order = rng.permutation(fit_rows)
        losses = []
        scores = []
        for start in range(0, len(order), config.batch_size):
            local = torch.as_tensor(
                order[start : start + config.batch_size], device=store.device
            )
            batch = store.batch(local, normalization)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            state_value, _ = e150.state_loss(output, batch, config)
            action_value, _ = action_supervision_loss(output, batch, config)
            official_value, official = e150.official_loss(
                model, output, batch, config
            )
            loss = (
                official_value
                + 0.15 * state_value
                + config.action_supervision_weight * action_value
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite E151 loss epoch={epoch}")
            loss.backward()
            e150.stable_clip_gradients(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            scores.append(official["score"])
        scheduler.step()
        row: dict[str, object] = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_batch_score": float(np.mean(scores)),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        if epoch == 1 or epoch % config.validation_interval == 0 or epoch == config.finetune_epochs:
            curve = validation_curve(
                model, store, validation_rows, normalization, config
            )
            winner = max(curve, key=lambda item: float(item["metrics"]["score"]))
            row["alpha_curve"] = curve
            row["selected_alpha"] = float(winner["alpha"])
            row["selected_validation"] = winner["metrics"]
            score = float(winner["metrics"]["score"])
            if score > best_score + 1e-10:
                best_score = score
                best_epoch = epoch
                best_alpha = float(winner["alpha"])
                best_state = copy.deepcopy(model.state_dict())
        history.append(row)
        if epoch == 1 or epoch % 2 == 0 or epoch == config.finetune_epochs:
            message = (
                f"E151 outer={store.outer} finetune={epoch}/{config.finetune_epochs} "
                f"train={row['train_batch_score']:.6f}"
            )
            if "selected_validation" in row:
                message += (
                    f" inner={row['selected_validation']['score']:.6f} "
                    f"alpha={row['selected_alpha']:.2f}"
                )
            print(message, flush=True)
    model.load_state_dict(best_state)
    return best_state, best_alpha, {
        "best_epoch": best_epoch,
        "best_alpha": best_alpha,
        "best_validation_score": best_score,
        "history": history,
    }


def train_fold(
    outer: int,
    config: Config,
    device: torch.device,
) -> tuple[
    dict[str, object],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    inner = (outer + 1) % 5
    store = e150.FoldStore(outer, device)
    fit_rows = np.flatnonzero(
        (store.query_fold_np != outer) & (store.query_fold_np != inner)
    )
    validation_rows = np.flatnonzero(store.query_fold_np == inner)
    held_rows = np.flatnonzero(store.query_fold_np == outer)
    normalization = store.normalization(fit_rows)
    torch.manual_seed(config.seed + outer * 1009)
    model = e150.PathwiseCarrierOperator(
        store.candidate_feature.shape[-1], store.point_feature.shape[-1], config
    ).to(device)
    _, pretrain = e150.train_pretrain(
        model,
        store,
        fit_rows,
        validation_rows,
        normalization,
        config,
        config.seed + outer * 2027,
    )
    _, action_alpha, action = train_action_warmup(
        model,
        store,
        fit_rows,
        validation_rows,
        normalization,
        config,
        config.seed + outer * 4099,
    )
    best_state, alpha, finetune = train_finetune(
        model,
        store,
        fit_rows,
        validation_rows,
        normalization,
        config,
        action_alpha,
        config.seed + outer * 8191,
    )
    selected_values, selected_diagnostics = e150.evaluate_model_rows(
        model, store, held_rows, normalization, alpha, config
    )
    selected = statistics(*selected_values)
    baseline = statistics(
        *(value[held_rows].cpu().numpy() for value in (
            store.base_pas,
            store.base_pdp,
            store.base_error,
            store.base_energy,
        ))
    )
    override, control_record = e150.geometry_control_override(
        store,
        held_rows,
        config.seed + outer * 16_381,
        config.geometry_control_strata,
    )
    control_values, control_diagnostics = e150.evaluate_model_rows(
        model,
        store,
        held_rows,
        normalization,
        alpha,
        config,
        override,
    )
    control = statistics(*control_values)
    path = checkpoint_path(outer)
    torch.save(
        {
            "experiment": VERSION,
            "outer": outer,
            "inner": inner,
            "config": asdict(config),
            "state_dict": best_state,
            "normalization": {
                key: value.detach().cpu() for key, value in normalization.items()
            },
            "selected_alpha": alpha,
            "official_h_only": True,
            "sionna_used": False,
            "v12_used": False,
        },
        path,
    )
    record = {
        "outer": outer,
        "inner": inner,
        "fit_count": int(len(fit_rows)),
        "validation_count": int(len(validation_rows)),
        "held_count": int(len(held_rows)),
        "assignment": store.assignment,
        "pretrain": pretrain,
        "action_warmup": action,
        "finetune": finetune,
        "selected_alpha": alpha,
        "held_baseline": baseline,
        "held_selected": selected,
        "held_delta": float(selected["score"] - baseline["score"]),
        "selected_diagnostics": selected_diagnostics,
        "geometry_control": {
            **control_record,
            "metrics": control,
            "diagnostics": control_diagnostics,
        },
        "geometry_margin": float(selected["score"] - control["score"]),
        "checkpoint": PATHS.relative(path),
    }
    print(
        f"E151 outer={outer} held={selected['score']:.6f} "
        f"delta={record['held_delta']:+.6f} "
        f"geometry_margin={record['geometry_margin']:+.6f}",
        flush=True,
    )
    del model, store
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record, selected_values, control_values


def main() -> None:
    args = parse_args()
    started = time.time()
    config = Config(
        pretrain_epochs=args.pretrain_epochs,
        action_warmup_epochs=args.action_warmup_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        metric_batch_size=args.metric_batch_size,
    )
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed % (2**32 - 1))
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    outers = e150.selected_outers(args.outer)
    for outer in outers:
        if not (e150.cache_dir(outer) / "state.json").is_file():
            e150.build_cache(outer, config, device, args.force_cache)

    baseline = e150.full_baseline(config, device)
    fold_all = np.load(e115.FULL_FOLD).astype(np.int8)
    query_fold = fold_all[baseline["query_ids"]]
    selected_values = tuple(np.empty(500, dtype=np.float64) for _ in range(4))
    control_values = tuple(np.empty(500, dtype=np.float64) for _ in range(4))
    fold_records = []
    for outer in outers:
        record, values, control = train_fold(outer, config, device)
        held = np.flatnonzero(query_fold == outer)
        for destination, value in zip(selected_values, values, strict=True):
            destination[held] = value
        for destination, value in zip(control_values, control, strict=True):
            destination[held] = value
        fold_records.append(record)

    complete = len(outers) == 5
    selected = statistics(*selected_values) if complete else None
    control = statistics(*control_values) if complete else None
    gate = None
    if complete and selected is not None and control is not None:
        fold_delta = [float(record["held_delta"]) for record in fold_records]
        gate = {
            "required_fixed500_score": config.required_score,
            "required_v7_gain": config.required_v7_gain,
            "required_positive_folds": config.required_positive_folds,
            "required_geometry_margin": config.required_geometry_margin,
            "fixed500_score": float(selected["score"]),
            "v7_gain": float(selected["score"] - baseline["metrics"]["score"]),
            "positive_folds": int(sum(value > 0.0 for value in fold_delta)),
            "minimum_fold_delta": float(min(fold_delta)),
            "geometry_margin": float(selected["score"] - control["score"]),
            "pas_non_worse": bool(selected["pas"] >= baseline["metrics"]["pas"]),
            "pdp_non_worse": bool(selected["pdp"] >= baseline["metrics"]["pdp"]),
            "nmse_non_worse": bool(selected["nmse"] <= baseline["metrics"]["nmse"]),
        }
        gate["passed"] = bool(
            gate["fixed500_score"] > gate["required_fixed500_score"]
            and gate["v7_gain"] >= gate["required_v7_gain"]
            and gate["positive_folds"] >= gate["required_positive_folds"]
            and gate["geometry_margin"] >= gate["required_geometry_margin"]
            and gate["pas_non_worse"]
            and gate["pdp_non_worse"]
            and gate["nmse_non_worse"]
        )
        np.savez_compressed(
            METRICS,
            query_ids=baseline["query_ids"].astype(np.int32),
            baseline_pas=baseline["values"][0],
            baseline_pdp=baseline["values"][1],
            baseline_error=baseline["values"][2],
            baseline_energy=baseline["values"][3],
            selected_pas=selected_values[0],
            selected_pdp=selected_values[1],
            selected_error=selected_values[2],
            selected_energy=selected_values[3],
            control_pas=control_values[0],
            control_pdp=control_values[1],
            control_error=control_values[2],
            control_energy=control_values[3],
        )

    report = {
        "experiment": VERSION,
        "status": (
            "analysis_only_positive"
            if gate is not None and gate["passed"]
            else "analysis_only_negative"
            if gate is not None
            else "outer_screen_complete"
        ),
        "single_experimental_goal": (
            "Test whether one-to-one official-H path identity can directly supervise "
            "the E150 action gate and turn the high path-state interface ceiling into "
            "a transferable V7 correction."
        ),
        "protocol": {
            "e150_cache_and_interface_frozen": True,
            "only_new_mechanism": "identity-supervised path action gate",
            "official_h_only": True,
            "sionna_used": False,
            "v12_used": False,
            "submission_generated": False,
        },
        "config": asdict(config),
        "baseline_v7": baseline["metrics"],
        "folds": fold_records,
        "selected": selected,
        "geometry_control": control,
        "gate": gate,
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "baseline_v7": baseline["metrics"],
                "folds": [
                    {
                        "outer": row["outer"],
                        "held_selected": row["held_selected"],
                        "held_delta": row["held_delta"],
                        "geometry_margin": row["geometry_margin"],
                    }
                    for row in fold_records
                ],
                "selected": selected,
                "gate": gate,
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
