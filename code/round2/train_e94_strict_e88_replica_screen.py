#!/usr/bin/env python3
"""Train a second strictly held-out E88 model per fixed hole and audit uncertainty."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import time
from typing import Any

import numpy as np
from scipy.stats import spearmanr
import torch

from project_paths import PATHS
import train_e88_dense_path_residual_student as e88
import train_offline_sionna_geometry_student as e87
import train_surface_interaction_energy_model as e85


VERSION = "Round2-E94-strict-E88-replica-screen"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "strict_e88_replica_screen"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_strict_e88_replica_screen"
ENDPOINTS = ARTIFACT_DIR / "fixed500_strict_replica_endpoints.npz"
REPORT = ANALYSIS_DIR / "strict_e88_replica_screen.json"
PRIMARY_POLICY = {
    "beta_pas": 0.05,
    "beta_pdp": 0.20,
    "order": "pdp_pas",
    "cycles": 4,
}
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def normalized_root_mean(
    first: np.ndarray, second: np.ndarray, axes: tuple[int, ...]
) -> np.ndarray:
    root = 0.5 * (
        np.sqrt(np.maximum(first, 0.0)) + np.sqrt(np.maximum(second, 0.0))
    )
    value = np.square(root)
    total = value.sum(axis=axes, keepdims=True, dtype=np.float64)
    return np.divide(
        value,
        np.maximum(total, TINY),
        out=np.zeros_like(value),
        where=total > 0.0,
    ).astype(np.float32)


def checkpoint_path(fold: int) -> Any:
    return ARTIFACT_DIR / f"fold{fold}_strict_replica.pt"


def load_or_train_replica(
    *,
    fold: int,
    fit: np.ndarray,
    validation: np.ndarray,
    geometry: e88.GeometryStore,
    official_pas: torch.Tensor,
    official_pdp: torch.Tensor,
    teacher_pas: torch.Tensor,
    teacher_pdp: torch.Tensor,
    power: np.ndarray,
    config: e88.Config,
) -> tuple[e88.DensePathResidualStudent, dict[str, Any]]:
    checkpoint = checkpoint_path(fold)
    expected = {
        "experiment": VERSION,
        "held_fold": fold,
        "validation_fold": (fold + 2) % 5,
        "config": asdict(config),
    }
    if checkpoint.is_file():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if all(payload.get(key) == value for key, value in expected.items()):
            model = e88.build_model(geometry, config)
            model.load_state_dict(payload["state_dict"])
            return model, {
                "status": "reused",
                "held_fold": fold,
                "validation_fold": (fold + 2) % 5,
                "checkpoint": PATHS.relative(checkpoint),
            }

    model, record = e88.train_official_fold(
        outer=100 + fold,
        fit=fit,
        validation=validation,
        geometry=geometry,
        official_pas=official_pas,
        official_pdp=official_pdp,
        teacher_pas=teacher_pas,
        teacher_pdp=teacher_pdp,
        power=power,
        config=config,
    )
    torch.save(
        {
            **expected,
            "state_dict": model.state_dict(),
            "sionna_inputs_required_for_inference": False,
        },
        checkpoint,
    )
    return model, {
        "status": "trained",
        "held_fold": fold,
        "validation_fold": (fold + 2) % 5,
        "checkpoint": PATHS.relative(checkpoint),
        "training": record,
    }


def endpoint_point_scores(
    pas: np.ndarray,
    pdp: np.ndarray,
    truth_pas: torch.Tensor,
    truth_pdp: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = e88.spectral_point_score(
        torch.as_tensor(pas, device=device),
        torch.as_tensor(pdp, device=device),
        truth_pas,
        truth_pdp,
    )
    return tuple(value.cpu().numpy() for value in values)


@torch.inference_mode()
def evaluate_policy(
    name: str,
    pas: np.ndarray,
    pdp: np.ndarray,
    base: torch.Tensor,
    truth: torch.Tensor,
    truth_pas: torch.Tensor,
    truth_pdp: torch.Tensor,
    baseline_values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    fold_id: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    endpoint_pas, endpoint_pdp = e87.expanded_endpoint(pas, pdp, device)
    desired_pas = e87.endpoint_target(
        e87.pas_distribution(base),
        endpoint_pas,
        PRIMARY_POLICY["beta_pas"],
        "sqrt",
        (1, 2, 3),
    )
    desired_pdp = e87.endpoint_target(
        e87.pdp_distribution(base),
        endpoint_pdp,
        PRIMARY_POLICY["beta_pdp"],
        "sqrt",
        (3,),
    )
    prediction = e87.project_joint(
        base,
        desired_pas,
        desired_pdp,
        PRIMARY_POLICY["cycles"],
        PRIMARY_POLICY["order"],
    )
    values = e87.point_metrics(prediction, truth, truth_pas, truth_pdp)
    metrics = e87.statistics(*values)
    baseline = e87.statistics(*baseline_values)
    folds = []
    for fold in range(5):
        mask = fold_id == fold
        local = e87.statistics(*values, mask)
        local_base = e87.statistics(*baseline_values, mask)
        folds.append(
            {"fold": fold, "delta": float(local["score"] - local_base["score"])}
        )
    point_nmse = np.divide(
        values[2],
        np.maximum(values[3], TINY),
        out=np.zeros_like(values[2]),
        where=values[3] > 0.0,
    )
    point_score = 0.4 * values[0] + 0.4 * values[1] + 0.2 / (1.0 + point_nmse)
    return {
        "name": name,
        "policy": PRIMARY_POLICY,
        "metrics": metrics,
        "delta": float(metrics["score"] - baseline["score"]),
        "positive_folds": int(sum(row["delta"] > 0.0 for row in folds)),
        "worst_fold_delta": float(min(row["delta"] for row in folds)),
        "folds": folds,
    }, point_score


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    started = time.time()
    config = e88.Config(
        finetune_epochs=args.epochs,
        finetune_batch_size=args.batch_size,
    )
    geometry = e88.GeometryStore("train", device)
    metadata = np.load(e88.OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    first = np.load(e88.OOF_PREDICTIONS, allow_pickle=False)
    if not np.array_equal(first["query_ids"].astype(np.int64), query_ids):
        raise ValueError("E88 first endpoint order changed")

    official_pas = torch.as_tensor(
        np.array(np.load(e85.TARGET_PAS, mmap_mode="r"), dtype=np.float32, copy=True),
        device=device,
    )
    official_pdp = torch.as_tensor(
        np.array(np.load(e85.TARGET_PDP, mmap_mode="r"), dtype=np.float32, copy=True),
        device=device,
    )
    teacher_cache = np.load(e87.TEACHER_CACHE, allow_pickle=False)
    teacher_pas = torch.as_tensor(teacher_cache["pas"].astype(np.float32), device=device)
    teacher_pdp = torch.as_tensor(teacher_cache["pdp"].astype(np.float32), device=device)
    power = np.asarray(np.load(e85.POINT_POWER), dtype=np.float64)
    all_rows = np.arange(len(geometry.positions), dtype=np.int64)

    replica_pas = np.zeros_like(first["final_pas"], dtype=np.float32)
    replica_pdp = np.zeros_like(first["final_pdp"], dtype=np.float32)
    replica_confidence = np.zeros_like(first["confidence"], dtype=np.float32)
    training = []
    global_to_query = np.full(len(geometry.positions), -1, dtype=np.int64)
    global_to_query[query_ids] = np.arange(len(query_ids))
    for fold in range(5):
        held = query_ids[fold_id == fold]
        validation_fold = (fold + 2) % 5
        validation = query_ids[fold_id == validation_fold]
        excluded = np.zeros(len(geometry.positions), dtype=bool)
        excluded[held] = True
        excluded[validation] = True
        fit = all_rows[~excluded]
        model, record = load_or_train_replica(
            fold=fold,
            fit=fit,
            validation=validation,
            geometry=geometry,
            official_pas=official_pas,
            official_pdp=official_pdp,
            teacher_pas=teacher_pas,
            teacher_pdp=teacher_pdp,
            power=power,
            config=config,
        )
        prediction = e88.predict_official(
            model, geometry, held, config.finetune_batch_size
        )
        target = global_to_query[held]
        replica_pas[target] = prediction["pas"]
        replica_pdp[target] = prediction["pdp"]
        replica_confidence[target, 0] = prediction["valid"]
        replica_confidence[target, 1] = prediction["reliability"]
        training.append(record)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    first_pas = first["final_pas"].astype(np.float32)
    first_pdp = first["final_pdp"].astype(np.float32)
    ensemble_pas = normalized_root_mean(first_pas, replica_pas, (1, 2, 3))
    ensemble_pdp = normalized_root_mean(first_pdp, replica_pdp, (3,))
    truth_pas_fixed = official_pas[torch.as_tensor(query_ids, device=device)]
    truth_pdp_fixed = official_pdp[torch.as_tensor(query_ids, device=device)]
    endpoints = {
        "first": (first_pas, first_pdp),
        "replica": (replica_pas, replica_pdp),
        "root_ensemble": (ensemble_pas, ensemble_pdp),
    }
    endpoint_reports = {
        name: e88.endpoint_summary(
            pas, pdp, truth_pas_fixed, truth_pdp_fixed, fold_id, device
        )
        for name, (pas, pdp) in endpoints.items()
    }

    agreement_pas, agreement_pdp, agreement = endpoint_point_scores(
        first_pas,
        first_pdp,
        torch.as_tensor(replica_pas, device=device),
        torch.as_tensor(replica_pdp, device=device),
        device,
    )
    quality_pas, quality_pdp, quality = endpoint_point_scores(
        ensemble_pas,
        ensemble_pdp,
        truth_pas_fixed,
        truth_pdp_fixed,
        device,
    )

    base = torch.as_tensor(
        np.array(np.load(e87.V12_CHANNEL, mmap_mode="r"), copy=True), device=device
    )
    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    truth = torch.as_tensor(
        np.asarray(truth_all[query_ids], dtype=np.complex64), device=device
    )
    truth_channel_pas = e87.pas_distribution(truth)
    truth_channel_pdp = e87.pdp_distribution(truth)
    baseline_values = e87.point_metrics(
        base, truth, truth_channel_pas, truth_channel_pdp
    )
    baseline = e87.statistics(*baseline_values)
    policies = {}
    point_scores = {}
    for name, (pas, pdp) in endpoints.items():
        policies[name], point_scores[name] = evaluate_policy(
            name,
            pas,
            pdp,
            base,
            truth,
            truth_channel_pas,
            truth_channel_pdp,
            baseline_values,
            fold_id,
            device,
        )
    base_nmse = np.divide(
        baseline_values[2],
        np.maximum(baseline_values[3], TINY),
        out=np.zeros_like(baseline_values[2]),
        where=baseline_values[3] > 0.0,
    )
    base_point_score = (
        0.4 * baseline_values[0]
        + 0.4 * baseline_values[1]
        + 0.2 / (1.0 + base_nmse)
    )
    action_gain = point_scores["root_ensemble"] - base_point_score
    uncertainty = 1.0 - agreement
    uncertainty_report = {
        "agreement_pas_mean": float(agreement_pas.mean()),
        "agreement_pdp_mean": float(agreement_pdp.mean()),
        "agreement_mean": float(agreement.mean()),
        "ensemble_quality_mean": float(quality.mean()),
        "uncertainty_vs_endpoint_error_spearman": float(
            spearmanr(uncertainty, 1.0 - quality).statistic
        ),
        "uncertainty_vs_action_gain_spearman": float(
            spearmanr(uncertainty, action_gain).statistic
        ),
    }
    np.savez_compressed(
        ENDPOINTS,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        replica_pas=replica_pas.astype(np.float16),
        replica_pdp=replica_pdp.astype(np.float16),
        replica_confidence=replica_confidence.astype(np.float16),
        ensemble_pas=ensemble_pas.astype(np.float16),
        ensemble_pdp=ensemble_pdp.astype(np.float16),
        agreement_pas=agreement_pas.astype(np.float32),
        agreement_pdp=agreement_pdp.astype(np.float32),
    )
    first_policy = policies["first"]
    ensemble_policy = policies["root_ensemble"]
    passed = bool(
        endpoint_reports["root_ensemble"]["mean"]
        > max(endpoint_reports["first"]["mean"], endpoint_reports["replica"]["mean"])
        and ensemble_policy["metrics"]["score"] > first_policy["metrics"]["score"]
        and ensemble_policy["delta"] > 0.0
        and ensemble_policy["positive_folds"] == 5
        and uncertainty_report["uncertainty_vs_endpoint_error_spearman"] > 0.10
    )
    report = {
        "experiment": VERSION,
        "status": "screen_complete",
        "primary_hypothesis": (
            "A strictly independent E88 replica supplies ensemble improvement and "
            "deployable epistemic uncertainty missing from the single-model gate."
        ),
        "configuration": asdict(config),
        "replica_protocol": {
            "held_fold": "fixed fold f",
            "validation_fold": "fixed fold (f+2)%5",
            "seed_outer_id": "100+f",
            "policy": PRIMARY_POLICY,
        },
        "V12": baseline,
        "endpoints": endpoint_reports,
        "policies": policies,
        "uncertainty": uncertainty_report,
        "decision": {
            "passed": passed,
            "expand_to_full4000": passed,
        },
        "training": training,
        "output": PATHS.relative(ENDPOINTS),
        "seconds": time.time() - started,
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
