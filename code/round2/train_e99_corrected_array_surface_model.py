#!/usr/bin/env python3
"""Retrain the E85 surface model with a strictly fitted array-angle mapping."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import time
from typing import Any

import numpy as np
import torch

import analyze_e86_surface_endpoint_v11_blend as e86
import optimize_e98_global_array_angle_mapping as e98
import train_surface_interaction_energy_model as e85
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from project_paths import PATHS
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics


VERSION = "Round2-E99-corrected-array-surface-model"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "corrected_array_surface_model"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_corrected_array_surface_model"
REPORT = ANALYSIS_DIR / "corrected_array_surface_model.json"
SMOKE_REPORT = ANALYSIS_DIR / "smoke.json"
ENDPOINT = ARTIFACT_DIR / "strict_oof_corrected_geometry_sequence_endpoint.npz"
RENDERED_PAS = ARTIFACT_DIR / "strict_oof_corrected_rendered_pas_f16.npy"
RENDERED_PDP = ARTIFACT_DIR / "strict_oof_corrected_rendered_pdp_f16.npy"
OOF_CHANNEL = ARTIFACT_DIR / "strict_oof_corrected_v11_blend_channel.npy"
MAPPING_PARAMETERS = ARTIFACT_DIR / "strict_mapping_parameters.npz"
V12_CHANNEL = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "e86_frozen_geometry_sequence_oof_channel.npy"
)
OLD_ENDPOINT = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "strict_oof_geometry_sequence_path_endpoint.npz"
)
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--sequence-layers", type=int, default=2)
    parser.add_argument("--set-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--mapping-steps", type=int, default=250)
    parser.add_argument("--mapping-batch-size", type=int, default=384)
    parser.add_argument("--mapping-restarts", type=int, default=2)
    parser.add_argument("--mapping-learning-rate", type=float, default=0.025)
    return parser.parse_args()


def parse_folds(value: str) -> tuple[int, ...]:
    folds = tuple(int(item) for item in value.split(",") if item.strip())
    if not folds or len(set(folds)) != len(folds) or any(fold not in range(5) for fold in folds):
        raise ValueError("folds must be a unique subset of 0,1,2,3,4")
    return folds


def fit_mapping(
    data: e98.MappingData,
    rows: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    label: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    best_latent: torch.Tensor | None = None
    best_objective = math.inf
    for restart in range(args.mapping_restarts):
        local_seed = seed + 100 * restart
        torch.manual_seed(local_seed)
        rng = np.random.default_rng(local_seed)
        initial = torch.zeros(10, device=device)
        if restart > 0:
            initial = 0.15 * torch.randn(10, device=device)
        latent = torch.nn.Parameter(initial)
        optimizer = torch.optim.AdamW(
            [latent], lr=args.mapping_learning_rate, weight_decay=1e-4
        )
        history = []
        for step in range(1, args.mapping_steps + 1):
            batch = rng.choice(
                rows,
                size=min(args.mapping_batch_size, len(rows)),
                replace=False,
            )
            loss = e98.batch_loss(data, batch, latent)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if latent.grad is None or not torch.isfinite(latent.grad).all():
                raise FloatingPointError(
                    f"non-finite mapping gradient label={label} restart={restart} step={step}"
                )
            optimizer.step()
            if step % 50 == 0 or step == args.mapping_steps:
                history.append({"step": step, "loss": float(loss.detach().item())})
                print(
                    f"E99 mapping={label} restart={restart} "
                    f"step={step}/{args.mapping_steps} loss={loss.detach().item():.6f}",
                    flush=True,
                )
        metrics = e98.evaluate(data, rows, latent)
        objective = float(metrics["weighted_log1p_min_cost"])
        attempts.append(
            {
                "restart": restart,
                "history": history,
                "parameters": e98.serializable_parameters(latent),
                "fit": metrics,
            }
        )
        if objective < best_objective:
            best_objective = objective
            best_latent = latent.detach().clone()
    if best_latent is None:
        raise RuntimeError(f"mapping fit produced no parameters for {label}")
    return best_latent, {
        "label": label,
        "fit_count": int(len(rows)),
        "selected_parameters": e98.serializable_parameters(best_latent),
        "attempts": attempts,
    }


@torch.inference_mode()
def coordinates_from_mapping(
    base_coordinate: np.ndarray,
    data: e98.MappingData,
    latent: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    coordinate = np.array(base_coordinate, dtype=np.float32, copy=True)
    rows = np.arange(len(coordinate), dtype=np.int64)
    for start in range(0, len(rows), batch_size):
        local = rows[start : start + batch_size]
        local_tensor = torch.as_tensor(local, device=device)
        horizontal, vertical = data.candidate_angles(local_tensor, latent)
        coordinate[local, :, 0] = torch.remainder(horizontal, 16.0).cpu().numpy()
        coordinate[local, :, 1] = torch.remainder(vertical, 8.0).cpu().numpy()
    if not np.isfinite(coordinate).all():
        raise FloatingPointError("corrected coordinates contain non-finite values")
    return coordinate


def endpoint_summary(
    pas: np.ndarray, pdp: np.ndarray, fold_id: np.ndarray
) -> dict[str, object]:
    rows = np.arange(len(pas), dtype=np.int64)
    return e85.summarize_prediction(pas, pdp, rows, fold_id)


def metric_comparison(
    selected: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    fold_id: np.ndarray,
) -> dict[str, object]:
    selected_summary = statistics(*selected)
    baseline_summary = statistics(*baseline)
    folds = []
    for outer in range(5):
        mask = fold_id == outer
        local = statistics(*selected, mask)
        anchor = statistics(*baseline, mask)
        folds.append(
            {
                "fold": outer,
                "selected": local,
                "baseline": anchor,
                "delta": float(local["score"] - anchor["score"]),
            }
        )
    return {
        "selected": selected_summary,
        "baseline": baseline_summary,
        "delta": float(selected_summary["score"] - baseline_summary["score"]),
        "positive_folds": int(sum(row["delta"] > 0.0 for row in folds)),
        "worst_fold_delta": float(min(row["delta"] for row in folds)),
        "folds": folds,
    }


def run_fold(
    outer: int,
    store: e85.DataStore,
    base_coordinate: np.ndarray,
    mapping_data: e98.MappingData,
    config: e85.Config,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    query_ids = store.strict_ids
    fold_id = store.strict_fold
    validation_fold = (outer + 1) % 5
    held = query_ids[fold_id == outer]
    validation = query_ids[fold_id == validation_fold]
    all_points = np.arange(len(store.base), dtype=np.int64)
    held_mask = np.zeros(len(store.base), dtype=bool)
    validation_mask = np.zeros(len(store.base), dtype=bool)
    held_mask[held] = True
    validation_mask[validation] = True
    screen_fit = all_points[~held_mask & ~validation_mask]
    refit = all_points[~held_mask]

    screen_latent, screen_mapping = fit_mapping(
        mapping_data,
        screen_fit,
        args,
        device,
        2026080899 + 10000 * outer,
        f"fold{outer}_screen",
    )
    store.coordinate = coordinates_from_mapping(
        base_coordinate, mapping_data, screen_latent, device
    )
    mean_screen, std_screen = store.normalization(screen_fit)
    screen_model, screen_history, best_epoch = e85.train_model(
        store,
        screen_fit,
        validation,
        mean_screen,
        std_screen,
        config,
        device,
        args.batch_size,
        config.epochs,
        config.random_state + 210000 + 1000 * outer,
        "geometry",
    )
    del screen_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    refit_latent, refit_mapping = fit_mapping(
        mapping_data,
        refit,
        args,
        device,
        2026081899 + 10000 * outer,
        f"fold{outer}_refit",
    )
    store.coordinate = coordinates_from_mapping(
        base_coordinate, mapping_data, refit_latent, device
    )
    mean_refit, std_refit = store.normalization(refit)
    model, refit_history, _ = e85.train_model(
        store,
        refit,
        None,
        mean_refit,
        std_refit,
        config,
        device,
        args.batch_size,
        best_epoch,
        config.random_state + 260000 + 1000 * outer,
        "geometry",
    )
    prediction = e85.predict_rows(
        model,
        store,
        held,
        mean_refit,
        std_refit,
        device,
        args.batch_size,
    )
    raw = e85.evaluate_rows(
        None,
        store,
        held,
        mean_refit,
        std_refit,
        device,
        args.batch_size,
    )
    shuffled_rows = e85.shuffled_sequence_rows(
        store, held, config.random_state + 390000 + outer
    )
    shuffled = e85.predict_rows(
        model,
        store,
        held,
        mean_refit,
        std_refit,
        device,
        args.batch_size,
        shuffled_rows,
    )
    checkpoint = ARTIFACT_DIR / f"strict_fold{outer}_corrected_geometry_sequence.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "feature_mean": mean_refit,
            "feature_std": std_refit,
            "best_epoch": best_epoch,
            "sequence_mode": "geometry",
            "outer": outer,
            "validation_fold": validation_fold,
            "mapping_latent": refit_latent.cpu().numpy().astype(np.float32),
            "mapping_parameters": e98.serializable_parameters(refit_latent),
        },
        checkpoint,
    )
    mapping_baseline = e98.evaluate(
        mapping_data, held, torch.zeros(10, device=device)
    )
    mapping_held = e98.evaluate(mapping_data, held, refit_latent)
    fold_endpoint = ARTIFACT_DIR / f"strict_fold{outer}_corrected_endpoint.npz"
    np.savez_compressed(
        fold_endpoint,
        held_ids=held.astype(np.int32),
        mass=prediction["mass"],
        coordinate=prediction["coordinate"],
        port=prediction["port"],
        pas=prediction["pas"],
        pdp=prediction["pdp"],
        raw_pas=raw["per_point_pas"],
        raw_pdp=raw["per_point_pdp"],
        shuffled_pas=shuffled["pas"],
        shuffled_pdp=shuffled["pdp"],
        screen_mapping_latent=screen_latent.cpu().numpy().astype(np.float32),
        refit_mapping_latent=refit_latent.cpu().numpy().astype(np.float32),
    )
    print(
        f"E99 fold={outer} epoch={best_epoch} "
        f"endpoint={np.mean(0.5 * (prediction['pas'] + prediction['pdp'])):.6f} "
        f"raw={raw['pas_pdp_mean']:.6f}",
        flush=True,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "outer": outer,
        "validation_fold": validation_fold,
        "held_ids": held,
        "best_epoch": best_epoch,
        "prediction": prediction,
        "raw": raw,
        "shuffled": shuffled,
        "screen_mapping_latent": screen_latent.cpu().numpy().astype(np.float32),
        "refit_mapping_latent": refit_latent.cpu().numpy().astype(np.float32),
        "mapping": {
            "screen": screen_mapping,
            "refit": refit_mapping,
            "held_baseline": mapping_baseline,
            "held_corrected": mapping_held,
            "held_tight_delta": float(
                mapping_held["full_tight_coverage"]
                - mapping_baseline["full_tight_coverage"]
            ),
        },
        "screen_fit_count": int(len(screen_fit)),
        "refit_count": int(len(refit)),
        "held_count": int(len(held)),
        "screen_history": screen_history,
        "refit_history": refit_history,
        "checkpoint": PATHS.relative(checkpoint),
        "fold_endpoint": PATHS.relative(fold_endpoint),
    }


def main() -> None:
    args = parse_args()
    folds = parse_folds(args.folds)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    missing = [
        path
        for path in (*e85.required_inputs(), e85.BASE_COORDINATE, OLD_ENDPOINT, V12_CHANNEL)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError([PATHS.relative(path) for path in missing])
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    started = time.time()
    config = e85.Config(
        hidden=args.hidden,
        sequence_layers=args.sequence_layers,
        set_layers=args.set_layers,
        epochs=args.epochs,
        minimum_epochs=min(16, args.epochs),
        validation_interval=min(4, args.epochs),
        learning_rate=args.learning_rate,
    )
    store = e85.DataStore()
    base_coordinate = np.array(store.coordinate, dtype=np.float32, copy=True)
    mapping_data = e98.MappingData(device)
    fold_results = [
        run_fold(
            outer,
            store,
            base_coordinate,
            mapping_data,
            config,
            args,
            device,
        )
        for outer in folds
    ]
    store.coordinate = base_coordinate

    if folds != tuple(range(5)):
        payload = {
            "experiment": VERSION,
            "status": "smoke_complete",
            "configuration": {**vars(args), "model": asdict(config)},
            "folds": [
                {
                    key: value
                    for key, value in row.items()
                    if key
                    not in (
                        "held_ids",
                        "prediction",
                        "raw",
                        "shuffled",
                        "screen_mapping_latent",
                        "refit_mapping_latent",
                    )
                }
                for row in fold_results
            ],
            "seconds": float(time.time() - started),
        }
        SMOKE_REPORT.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return

    query_ids = store.strict_ids
    fold_id = store.strict_fold
    global_to_row = np.full(len(store.base), -1, dtype=np.int64)
    global_to_row[query_ids] = np.arange(len(query_ids), dtype=np.int64)
    mass = np.zeros((len(query_ids), e85.PATHS_PER_POINT), dtype=np.float32)
    coordinate = np.zeros((len(query_ids), e85.PATHS_PER_POINT, 3), dtype=np.float32)
    port = np.zeros(
        (len(query_ids), e85.PATHS_PER_POINT, e85.MP, e85.N_UE), dtype=np.float32
    )
    pas = np.zeros(len(query_ids), dtype=np.float64)
    pdp = np.zeros(len(query_ids), dtype=np.float64)
    raw_pas = np.zeros(len(query_ids), dtype=np.float64)
    raw_pdp = np.zeros(len(query_ids), dtype=np.float64)
    shuffled_pas = np.zeros(len(query_ids), dtype=np.float64)
    shuffled_pdp = np.zeros(len(query_ids), dtype=np.float64)
    for result in fold_results:
        rows = global_to_row[result["held_ids"]]
        prediction = result["prediction"]
        mass[rows] = prediction["mass"]
        coordinate[rows] = prediction["coordinate"]
        port[rows] = prediction["port"]
        pas[rows] = prediction["pas"]
        pdp[rows] = prediction["pdp"]
        raw_pas[rows] = result["raw"]["per_point_pas"]
        raw_pdp[rows] = result["raw"]["per_point_pdp"]
        shuffled_pas[rows] = result["shuffled"]["pas"]
        shuffled_pdp[rows] = result["shuffled"]["pdp"]
    np.savez_compressed(
        ENDPOINT,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        mass=mass,
        coordinate=coordinate,
        port=port,
        pas=pas,
        pdp=pdp,
        raw_pas=raw_pas,
        raw_pdp=raw_pdp,
        shuffled_pas=shuffled_pas,
        shuffled_pdp=shuffled_pdp,
    )
    screen_latent = np.stack([row["screen_mapping_latent"] for row in fold_results])
    refit_latent = np.stack([row["refit_mapping_latent"] for row in fold_results])
    np.savez_compressed(
        MAPPING_PARAMETERS,
        fold_id=np.arange(5, dtype=np.int8),
        screen_latent=screen_latent,
        refit_latent=refit_latent,
    )

    endpoint_pas_np, endpoint_pdp_np = e86.render_cache(
        str(ENDPOINT),
        str(RENDERED_PAS),
        str(RENDERED_PDP),
        device,
        batch_size=8,
        force=True,
    )
    base_np = np.load(e86.V11_CHANNEL, mmap_mode="r")
    truth_np = np.asarray(
        np.load(e86.TRAIN_CHANNEL, mmap_mode="r")[query_ids], dtype=np.complex64
    )
    base = torch.as_tensor(np.array(base_np, copy=True), device=device)
    truth = torch.as_tensor(truth_np, device=device)
    truth_pas = pas_distribution(truth)
    truth_pdp = pdp_distribution(truth)
    base_pas = pas_distribution(base)
    base_pdp = pdp_distribution(base)
    endpoint_pas = torch.as_tensor(
        np.asarray(endpoint_pas_np, dtype=np.float32), device=device
    )
    endpoint_pdp = torch.as_tensor(
        np.asarray(endpoint_pdp_np, dtype=np.float32), device=device
    )
    spec = e86.Candidate(
        endpoint="corrected_geometry_sequence",
        mode="sqrt",
        pas_beta=0.20,
        pdp_beta=0.20,
        order="pas_pdp",
        cycles=4,
    )
    with torch.inference_mode():
        prediction = e86.make_candidate(
            base, base_pas, base_pdp, endpoint_pas, endpoint_pdp, spec
        )
        new_metrics = point_metrics(prediction, truth, truth_pas, truth_pdp)
        v11_metrics = point_metrics(base, truth, truth_pas, truth_pdp)
    np.save(OOF_CHANNEL, prediction.cpu().numpy().astype(np.complex64))
    v12 = torch.as_tensor(
        np.array(np.load(V12_CHANNEL, mmap_mode="r"), dtype=np.complex64, copy=True),
        device=device,
    )
    with torch.inference_mode():
        v12_metrics = point_metrics(v12, truth, truth_pas, truth_pdp)
    versus_v11 = metric_comparison(new_metrics, v11_metrics, fold_id)
    versus_v12 = metric_comparison(new_metrics, v12_metrics, fold_id)

    old_endpoint = np.load(OLD_ENDPOINT, allow_pickle=False)
    old_pas = old_endpoint["pas"].astype(np.float64)
    old_pdp = old_endpoint["pdp"].astype(np.float64)
    endpoint_new = endpoint_summary(pas, pdp, fold_id)
    endpoint_old = endpoint_summary(old_pas, old_pdp, fold_id)
    endpoint_raw = endpoint_summary(raw_pas, raw_pdp, fold_id)
    endpoint_shuffle = endpoint_summary(shuffled_pas, shuffled_pdp, fold_id)
    endpoint_gain = float(
        endpoint_new["pas_pdp_mean"] - endpoint_old["pas_pdp_mean"]
    )
    gate = {
        "minimum_endpoint_gain_vs_e85": 0.005,
        "minimum_score_gain_vs_v12": 0.001,
        "minimum_positive_folds_vs_v12": 4,
        "minimum_worst_fold_delta_vs_v12": -0.001,
        "endpoint_gain_vs_e85": endpoint_gain,
        "score_gain_vs_v12": versus_v12["delta"],
        "positive_folds_vs_v12": versus_v12["positive_folds"],
        "worst_fold_delta_vs_v12": versus_v12["worst_fold_delta"],
    }
    gate["passed"] = bool(
        endpoint_gain >= gate["minimum_endpoint_gain_vs_e85"]
        and versus_v12["delta"] >= gate["minimum_score_gain_vs_v12"]
        and versus_v12["positive_folds"] >= gate["minimum_positive_folds_vs_v12"]
        and versus_v12["worst_fold_delta"]
        >= gate["minimum_worst_fold_delta_vs_v12"]
    )
    fold_records = []
    for row in fold_results:
        fold_records.append(
            {
                key: value
                for key, value in row.items()
                if key
                not in (
                    "held_ids",
                    "prediction",
                    "raw",
                    "shuffled",
                    "screen_mapping_latent",
                    "refit_mapping_latent",
                )
            }
        )
    output = np.load(OOF_CHANNEL, mmap_mode="r")
    report = {
        "experiment": VERSION,
        "status": "strict_complete",
        "primary_hypothesis": (
            "The E98 global array-angle correction improves the E85 surface path "
            "endpoint and the frozen V12 spectral projection without pointwise tuning."
        ),
        "validation": (
            "Canonical 500-point contiguous spatial holes. Each screen/refit mapping "
            "and surface model excludes the held hole before predicting it."
        ),
        "configuration": {**vars(args), "model": asdict(config)},
        "endpoint": {
            "old_e85": endpoint_old,
            "corrected_raw": endpoint_raw,
            "corrected_model": endpoint_new,
            "corrected_sequence_shuffle": endpoint_shuffle,
            "gain_vs_e85": endpoint_gain,
            "margin_over_shuffle": float(
                endpoint_new["pas_pdp_mean"] - endpoint_shuffle["pas_pdp_mean"]
            ),
        },
        "fixed_policy": asdict(spec),
        "complete_score": {
            "versus_v11": versus_v11,
            "versus_v12": versus_v12,
        },
        "folds": fold_records,
        "mapping_parameter_stability": {
            "screen_latent_std": screen_latent.std(axis=0).tolist(),
            "refit_latent_std": refit_latent.std(axis=0).tolist(),
            "maximum_refit_latent_std": float(refit_latent.std(axis=0).max()),
        },
        "decision": {
            **gate,
            "generate_v13": gate["passed"],
        },
        "outputs": {
            "endpoint": PATHS.relative(ENDPOINT),
            "rendered_pas": PATHS.relative(RENDERED_PAS),
            "rendered_pdp": PATHS.relative(RENDERED_PDP),
            "oof_channel": PATHS.relative(OOF_CHANNEL),
            "mapping_parameters": PATHS.relative(MAPPING_PARAMETERS),
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "finite": bool(
                np.isfinite(output.real).all() and np.isfinite(output.imag).all()
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
                "endpoint": report["endpoint"],
                "complete_score": report["complete_score"],
                "decision": report["decision"],
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
