#!/usr/bin/env python3
"""Screen an identity-first path operator with one-to-one supervision."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

import analyze_e114_path_state_variable_ceiling as e114
import analyze_e124_one_to_one_path_identity_ceiling as e124
import train_e115_complete_path_state_operator as e115
import train_e118_hierarchical_unique_path_operator as e118
from common.official_metrics import PointMetricStatistics, aggregate_statistics
from project_paths import PATHS


VERSION = "Round2-E125-identity-first-path-operator"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "identity_first_path_operator"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_identity_first_path_operator"
REPORT = ANALYSIS_DIR / "identity_first_outer0_screen.json"
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"
OUTER = 0
INNER = 1
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--core-epochs", type=int, default=30)
    parser.add_argument("--scene-epochs", type=int, default=16)
    parser.add_argument("--graph-epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--validation-interval", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20_260_808_125)
    return parser.parse_args()


def hard_batch(
    store: e118.ClusterTensorStore,
    rows: torch.Tensor,
    normalization: dict[str, torch.Tensor],
    hard_mass: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch = store.batch(rows, normalization)
    batch["target_mass"] = hard_mass[rows]
    return batch


def base_output(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    logits = batch["base_log_mass"].masked_fill(~batch["mask"], -1e4)
    base_complex = torch.complex(
        batch["base_phase"][..., : e115.PORTS],
        batch["base_phase"][..., e115.PORTS :],
    )
    return {
        "mass": torch.softmax(logits, dim=1),
        "mass_logits": logits,
        "coordinate": batch["coordinate"],
        "coordinate_residual": torch.zeros_like(batch["coordinate"]),
        "port": torch.softmax(batch["base_log_port"], dim=2),
        "port_residual": torch.zeros_like(batch["base_log_port"]),
        "phase": torch.angle(base_complex),
        "phase_residual": torch.zeros_like(batch["base_log_port"]),
        "score_residual": torch.zeros_like(logits),
        "zero_logit": torch.zeros(len(logits), device=logits.device),
        "log_power": batch["base_log_power"],
    }


def mass_only_output(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    isolated = dict(output)
    base_complex = torch.complex(
        batch["base_phase"][..., : e115.PORTS],
        batch["base_phase"][..., e115.PORTS :],
    )
    isolated["coordinate"] = batch["coordinate"]
    isolated["coordinate_residual"] = torch.zeros_like(batch["coordinate"])
    isolated["port"] = torch.softmax(batch["base_log_port"], dim=2)
    isolated["port_residual"] = torch.zeros_like(batch["base_log_port"])
    isolated["phase"] = torch.angle(base_complex)
    isolated["phase_residual"] = torch.zeros_like(batch["base_log_port"])
    isolated["log_power"] = batch["base_log_power"]
    return isolated


def identity_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    target_absolute = batch["target_mass"]
    target_total = target_absolute.sum(dim=1)
    nonzero = target_total > 0.0
    target = target_absolute / target_total[:, None].clamp_min(TINY)
    valid = batch["mask"]
    positive = target_absolute > 0.0
    logits = output["mass_logits"]
    selected_positive = positive[valid]
    positive_count = selected_positive.sum().clamp_min(1)
    negative_count = (~selected_positive).sum().clamp_min(1)
    positive_weight = (negative_count / positive_count).clamp(1.0, 4.0).float()
    bce = F.binary_cross_entropy_with_logits(
        logits[valid], selected_positive.to(logits.dtype), pos_weight=positive_weight
    )
    listwise_per_point = -(
        target * torch.log_softmax(logits, dim=1)
    ).sum(dim=1)
    listwise = (
        listwise_per_point[nonzero].mean()
        if nonzero.any()
        else logits.sum() * 0.0
    )
    mass_affinity = (
        torch.sqrt(output["mass"].clamp_min(1e-12))
        * torch.sqrt(target.clamp_min(0.0))
    ).sum(dim=1)
    hellinger = (
        (1.0 - mass_affinity[nonzero]).mean()
        if nonzero.any()
        else logits.sum() * 0.0
    )
    predicted_count = (torch.sigmoid(logits) * valid).sum(dim=1)
    target_count = positive.sum(dim=1).to(predicted_count.dtype)
    cardinality = (
        torch.square(predicted_count - target_count)
        / torch.square(target_count + 1.0)
    ).mean()
    pas, pdp, _, _ = e115.sufficient_scores(output, batch)
    spectral = (
        1.0 - 0.5 * (pas[nonzero] + pdp[nonzero]).mean()
        if nonzero.any()
        else logits.sum() * 0.0
    )
    residual = output["score_residual"].square().mean()
    loss = (
        1.00 * spectral
        + 0.55 * listwise
        + 0.30 * hellinger
        + 0.20 * bce
        + 0.03 * cardinality
        + 0.002 * residual
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "spectral": float(spectral.detach().cpu()),
        "listwise": float(listwise.detach().cpu()),
        "hellinger": float(hellinger.detach().cpu()),
        "bce": float(bce.detach().cpu()),
        "cardinality": float(cardinality.detach().cpu()),
    }


def top_mass_recall(
    prediction: torch.Tensor,
    target: torch.Tensor,
    count: int,
) -> torch.Tensor:
    take = min(count, prediction.shape[1])
    selected = torch.topk(prediction, take, dim=1).indices
    return torch.gather(target, 1, selected).sum(dim=1)


@torch.inference_mode()
def evaluate(
    model: e118.HierarchicalUniquePathOperator | None,
    store: e118.ClusterTensorStore,
    rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    hard_mass: torch.Tensor,
    batch_size: int,
    include_complex: bool,
) -> dict[str, object]:
    if model is not None:
        model.eval()
    train_h = np.load(TRAIN_CHANNEL, mmap_mode="r") if include_complex else None
    logits_parts: list[np.ndarray] = []
    positive_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    affinity_parts: list[np.ndarray] = []
    top16_parts: list[np.ndarray] = []
    top24_parts: list[np.ndarray] = []
    adaptive_parts: list[np.ndarray] = []
    pas_parts: list[np.ndarray] = []
    pdp_parts: list[np.ndarray] = []
    error_parts: list[np.ndarray] = []
    energy_parts: list[np.ndarray] = []
    complex_pas_parts: list[np.ndarray] = []
    complex_pdp_parts: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        local = rows[start : start + batch_size]
        row_tensor = torch.as_tensor(local, dtype=torch.long, device=store.device)
        batch = hard_batch(store, row_tensor, normalization, hard_mass)
        output = (
            base_output(batch)
            if model is None
            else mass_only_output(model(batch), batch)
        )
        target_absolute = batch["target_mass"]
        target_total = target_absolute.sum(dim=1)
        target = target_absolute / target_total[:, None].clamp_min(TINY)
        affinity = (
            torch.sqrt(output["mass"].clamp_min(1e-12))
            * torch.sqrt(target.clamp_min(0.0))
        ).sum(dim=1)
        positive = target_absolute > 0.0
        valid = batch["mask"]
        logits_parts.append(output["mass_logits"].cpu().numpy())
        positive_parts.append(positive.cpu().numpy())
        valid_parts.append(valid.cpu().numpy())
        nonzero = target_total > 0.0
        affinity_parts.append(affinity[nonzero].cpu().numpy())
        top16_parts.append(top_mass_recall(output["mass"], target, 16)[nonzero].cpu().numpy())
        top24_parts.append(top_mass_recall(output["mass"], target, 24)[nonzero].cpu().numpy())
        adaptive = torch.zeros(len(local), dtype=target.dtype, device=target.device)
        for index in range(len(local)):
            take = max(int(positive[index].sum().item()), 1)
            ids = torch.topk(output["mass"][index], min(take, target.shape[1])).indices
            adaptive[index] = target[index, ids].sum()
        adaptive_parts.append(adaptive[nonzero].cpu().numpy())
        pas, pdp, _, _ = e115.sufficient_scores(output, batch)
        pas_parts.append(pas.cpu().numpy())
        pdp_parts.append(pdp.cpu().numpy())
        if include_complex:
            prediction = e115.render_complex_channel(output)
            truth = torch.as_tensor(
                np.asarray(train_h[local], dtype=np.complex64), device=store.device
            )
            complex_pas, complex_pdp, error, energy = e115.torch_official_metrics(
                prediction, truth
            )
            complex_pas_parts.append(complex_pas.cpu().numpy())
            complex_pdp_parts.append(complex_pdp.cpu().numpy())
            error_parts.append(error.cpu().numpy())
            energy_parts.append(energy.cpu().numpy())

    logits = np.concatenate(logits_parts)
    positive = np.concatenate(positive_parts)
    valid = np.concatenate(valid_parts)
    existence_ap = float(
        average_precision_score(positive[valid].astype(np.int8), logits[valid])
    )
    pas = np.concatenate(pas_parts)
    pdp = np.concatenate(pdp_parts)
    affinity = np.concatenate(affinity_parts)
    result: dict[str, object] = {
        "count": int(len(rows)),
        "existence_prevalence": float(np.mean(positive[valid])),
        "existence_ap": existence_ap,
        "mass_bhattacharyya": float(np.mean(affinity)),
        "top16_target_mass_recall": float(np.mean(np.concatenate(top16_parts))),
        "top24_target_mass_recall": float(np.mean(np.concatenate(top24_parts))),
        "adaptive_count_target_mass_recall": float(
            np.mean(np.concatenate(adaptive_parts))
        ),
        "endpoint_pas": float(np.mean(pas)),
        "endpoint_pdp": float(np.mean(pdp)),
        "endpoint_mean": float(np.mean(0.5 * (pas + pdp))),
    }
    result["selection_score"] = float(
        0.4 * result["endpoint_pas"]
        + 0.4 * result["endpoint_pdp"]
        + 0.2 * result["mass_bhattacharyya"]
    )
    if include_complex:
        values = PointMetricStatistics(
            pas=np.concatenate(complex_pas_parts),
            pdp=np.concatenate(complex_pdp_parts),
            squared_error=np.concatenate(error_parts),
            truth_energy=np.concatenate(energy_parts),
        )
        result["full_official"] = aggregate_statistics(values)
    return result


def train_stage(
    stage: str,
    model: e118.HierarchicalUniquePathOperator,
    variant: e115.Variant,
    store: e118.ClusterTensorStore,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    normalization: dict[str, torch.Tensor],
    hard_mass: torch.Tensor,
    epochs: int,
    batch_size: int,
    validation_interval: int,
    seed: int,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    model.variant = variant
    parameters = e118.configure_stage(model, stage)
    learning_rate = 5e-4 if stage == "core" else 3e-4
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=2e-4
    )
    initial = evaluate(
        model,
        store,
        validation_rows,
        normalization,
        hard_mass,
        batch_size,
        include_complex=False,
    )
    best_score = float(initial["selection_score"])
    best_epoch = 0
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    history = [{"epoch": 0, "validation": initial}]
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        shuffled = rng.permutation(fit_rows)
        loss_sum = 0.0
        batches = 0
        for start in range(0, len(shuffled), batch_size):
            local = shuffled[start : start + batch_size]
            rows = torch.as_tensor(local, dtype=torch.long, device=store.device)
            batch = hard_batch(store, rows, normalization, hard_mass)
            optimizer.zero_grad(set_to_none=True)
            output = mass_only_output(model(batch), batch)
            loss, _ = identity_loss(output, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            batches += 1
        if epoch % validation_interval == 0 or epoch == epochs:
            validation = evaluate(
                model,
                store,
                validation_rows,
                normalization,
                hard_mass,
                batch_size,
                include_complex=False,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": loss_sum / max(batches, 1),
                    "validation": validation,
                }
            )
            print(
                f"stage={stage} epoch={epoch}/{epochs} "
                f"loss={loss_sum / max(batches, 1):.6f} "
                f"selection={validation['selection_score']:.6f} "
                f"ap={validation['existence_ap']:.6f}",
                flush=True,
            )
            if float(validation["selection_score"]) > best_score + 1e-8:
                best_score = float(validation["selection_score"])
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
    model.load_state_dict(best_state)
    return (
        {
            "stage": stage,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "best_validation_selection": best_score,
            "history": history,
        },
        best_state,
    )


def save_checkpoint(
    path,
    state: dict[str, torch.Tensor],
    config: e118.Config,
    stage: str,
    normalization: dict[str, torch.Tensor],
) -> None:
    torch.save(
        {
            "experiment": VERSION,
            "stage": stage,
            "config": asdict(config),
            "state_dict": state,
            "normalization": {
                key: value.detach().cpu() for key, value in normalization.items()
            },
            "official_h_only": True,
            "one_to_one_identity_supervision": True,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    directory = e118.cache_dir(e124.e123.cache_tag(OUTER))
    store = e118.ClusterTensorStore(directory, device)
    fold_id = np.load(e115.FULL_FOLD).astype(np.int8)
    fit_rows = np.flatnonzero((fold_id != OUTER) & (fold_id != INNER))
    validation_rows = np.flatnonzero(fold_id == INNER)
    held_rows = np.flatnonzero(fold_id == OUTER)
    normalization = store.normalization(fit_rows)

    component = np.load(e114.COMPONENTS, allow_pickle=False)
    base_coordinate = np.asarray(
        np.load(directory / "coordinate.npy", mmap_mode="r"), np.float32
    )
    mask = np.asarray(np.load(directory / "mask.npy", mmap_mode="r"), bool)
    targets, assignment = e124.build_one_to_one_targets(
        base_coordinate,
        mask,
        component["coordinates"].astype(np.float32),
        component["coefficients"].astype(np.complex64),
        component["point_power"].astype(np.float64),
    )
    hard_mass = torch.as_tensor(
        targets["mass_absolute"], dtype=torch.float32, device=device
    )

    config = e118.Config(
        hidden=192,
        heads=8,
        token_layers=3,
        candidate_layers=3,
        graph_layers=2,
        dropout=0.08,
        score_limit=6.0,
        random_state=args.seed,
    )
    model = e118.HierarchicalUniquePathOperator(
        store.raw.candidate_feature.shape[-1],
        store.cluster_feature.shape[-1],
        store.raw.context.shape[-1],
        store.graph_feature.shape[-1],
        config,
        e115.VARIANTS[0],
    ).to(device)

    baseline_validation = evaluate(
        None,
        store,
        validation_rows,
        normalization,
        hard_mass,
        args.batch_size,
        include_complex=True,
    )
    baseline_held = evaluate(
        None,
        store,
        held_rows,
        normalization,
        hard_mass,
        args.batch_size,
        include_complex=True,
    )
    stage_specs = (
        ("core", e115.VARIANTS[0], args.core_epochs),
        ("scene", e115.VARIANTS[1], args.scene_epochs),
        ("graph", e115.VARIANTS[2], args.graph_epochs),
    )
    stages = []
    held_results = []
    for stage_index, (stage, variant, epochs) in enumerate(stage_specs):
        stage_record, state = train_stage(
            stage,
            model,
            variant,
            store,
            fit_rows,
            validation_rows,
            normalization,
            hard_mass,
            epochs,
            args.batch_size,
            args.validation_interval,
            args.seed + 1009 * stage_index,
        )
        checkpoint = ARTIFACT_DIR / f"outer0_identity_{stage}.pt"
        save_checkpoint(checkpoint, state, config, stage, normalization)
        model.variant = variant
        held = evaluate(
            model,
            store,
            held_rows,
            normalization,
            hard_mass,
            args.batch_size,
            include_complex=True,
        )
        stage_record["checkpoint"] = PATHS.relative(checkpoint)
        stages.append(stage_record)
        held_results.append({"stage": stage, "metrics": held})
        print(
            f"stage={stage} held_full={held['full_official']['score']:.6f} "
            f"held_ap={held['existence_ap']:.6f}",
            flush=True,
        )

    core_held = held_results[0]["metrics"]
    best_index = int(
        np.argmax(
            [item["metrics"]["full_official"]["score"] for item in held_results]
        )
    )
    best_held = held_results[best_index]
    gate = {
        "minimum_core_existence_ap_gain": 0.03,
        "minimum_core_top16_mass_recall_gain": 0.03,
        "minimum_core_full_score_gain": 0.005,
        "core_existence_ap_gain": float(
            core_held["existence_ap"] - baseline_held["existence_ap"]
        ),
        "core_top16_mass_recall_gain": float(
            core_held["top16_target_mass_recall"]
            - baseline_held["top16_target_mass_recall"]
        ),
        "core_full_score_gain": float(
            core_held["full_official"]["score"]
            - baseline_held["full_official"]["score"]
        ),
    }
    gate["passed"] = bool(
        gate["core_existence_ap_gain"] >= gate["minimum_core_existence_ap_gain"]
        and gate["core_top16_mass_recall_gain"]
        >= gate["minimum_core_top16_mass_recall_gain"]
        and gate["core_full_score_gain"] >= gate["minimum_core_full_score_gain"]
    )
    report = {
        "experiment": VERSION,
        "status": "outer0_screen_complete",
        "primary_hypothesis": (
            "One-to-one identity targets make candidate existence and coefficient-mass "
            "competition learnable before downstream coordinate, port, and phase heads."
        ),
        "strict_protocol": {
            "outer": OUTER,
            "inner": INNER,
            "fit_folds": [2, 3, 4],
            "fit_count": int(len(fit_rows)),
            "validation_count": int(len(validation_rows)),
            "held_count": int(len(held_rows)),
            "held_used_for_epoch_selection": False,
            "official_h_only": True,
            "sionna_or_v12_used": False,
        },
        "config": asdict(config),
        "loss": {
            "spectral": 1.0,
            "listwise": 0.55,
            "hellinger": 0.30,
            "existence_bce": 0.20,
            "cardinality": 0.03,
            "residual": 0.002,
        },
        "assignment": assignment,
        "baseline_validation": baseline_validation,
        "baseline_held": baseline_held,
        "stages": stages,
        "held_results": held_results,
        "best_held_stage_diagnostic_only": best_held,
        "gate": gate,
        "decision": {
            "identity_stage_passed": bool(gate["passed"]),
            "next_if_passed": (
                "Freeze the identity core and train teacher-forced then rollout "
                "coordinate, port, common-phase, relative-phase, and power heads."
            ),
            "next_if_failed": (
                "Do not train downstream heads; add globally stable lifecycle identity "
                "features or redesign unique-path candidates."
            ),
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "baseline_held": baseline_held,
                "held_results": held_results,
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
