#!/usr/bin/env python3
"""Train the E155 PAS operator on exact cross-fitted V7 anchors."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import time

import numpy as np
import torch

from project_paths import PATHS
from run_rbf_spectral_field import OOF_META, statistics
import run_e155_scene_conditioned_pas_operator as e155


VERSION = "Round2-E156-v7-domain-matched-pas"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "v7_domain_matched_pas"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_v7_domain_matched_pas"
REPORT = ANALYSIS_DIR / "v7_domain_matched_pas.json"
METRICS = ARTIFACT_DIR / "v7_domain_matched_pas_metrics.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--folds", default="all")
    parser.add_argument("--steps", type=int, default=e155.Config.steps)
    parser.add_argument(
        "--validation-interval", type=int, default=e155.Config.validation_interval
    )
    parser.add_argument("--metric-batch-size", type=int, default=8)
    parser.add_argument("--force-context", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    return parser.parse_args()


def train_fold(
    outer: int,
    config: e155.Config,
    store: e155.SceneStore,
    official_probability: np.ndarray,
    truth_amplitude: np.ndarray,
    v7_probability: np.ndarray,
    teacher_probability: np.ndarray,
    fixed_query_ids: np.ndarray,
    fixed_fold_id: np.ndarray,
    device: torch.device,
    force_context: bool,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Use exact fixed-500 V7 predictions as every train/inner/outer main anchor."""

    started = time.time()
    inner = (outer + 1) % 5
    e155.set_seed(config.seed + 100_000 + outer * 1009)
    outer_rows = np.flatnonzero(fixed_fold_id == outer)
    inner_rows = np.flatnonzero(fixed_fold_id == inner)
    train_rows = np.flatnonzero(
        (fixed_fold_id != outer) & (fixed_fold_id != inner)
    )
    all_ids = np.arange(len(official_probability), dtype=np.int64)
    nonzero = (
        official_probability.reshape(len(official_probability), -1).sum(axis=1) > 0
    )
    train_query_ids = fixed_query_ids[train_rows]
    train_query_ids = train_query_ids[nonzero[train_query_ids]]
    train_rows = train_rows[nonzero[fixed_query_ids[train_rows]]]
    inner_query_ids = fixed_query_ids[inner_rows]
    outer_query_ids = fixed_query_ids[outer_rows]
    excluded = np.concatenate((inner_query_ids, outer_query_ids))
    nested_donor_pool = all_ids[~np.isin(all_ids, excluded) & nonzero]
    outer_donor_pool = all_ids[~np.isin(all_ids, outer_query_ids) & nonzero]

    train_table = e155.build_context_table(
        outer,
        inner,
        "train_exact",
        train_query_ids,
        nested_donor_pool,
        store,
        truth_amplitude,
        config,
        force_context,
    )
    inner_table = e155.build_context_table(
        outer,
        inner,
        "inner_exact",
        inner_query_ids,
        nested_donor_pool,
        store,
        truth_amplitude,
        config,
        force_context,
    )
    outer_table = e155.build_context_table(
        outer,
        inner,
        "outer_exact",
        outer_query_ids,
        outer_donor_pool,
        store,
        truth_amplitude,
        config,
        force_context,
    )

    spectral = e155.fit_spectral_pca(
        truth_amplitude.reshape(len(truth_amplitude), e155.PAS_SIZE),
        nested_donor_pool,
        config.spectrum_rank,
        config.seed + 100_000 + outer,
    )
    teacher_delta = np.sqrt(np.maximum(teacher_probability[train_rows], 0.0)) - np.sqrt(
        np.maximum(v7_probability[train_rows], 0.0)
    )
    basis, basis_explained = e155.fit_correction_basis(
        teacher_delta, config.correction_rank
    )

    training_main = np.sqrt(np.maximum(v7_probability[train_rows], 0.0)).reshape(
        len(train_rows), e155.PAS_SIZE
    )
    training_truth = truth_amplitude[train_query_ids].reshape(
        len(train_query_ids), e155.PAS_SIZE
    )
    training_coefficients = (training_truth - training_main) @ basis.T
    coefficient_scale = np.std(
        training_coefficients, axis=0, dtype=np.float64
    ).astype(np.float32)
    coefficient_scale = np.maximum(coefficient_scale, 2e-3)

    model = e155.SceneConditionedPASOperator(
        store.position.shape[1],
        store.scalar.shape[1],
        config.spectrum_rank,
        config.correction_rank,
        len(config.guard_radii_m),
        config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    basis_t = torch.as_tensor(basis, device=device)
    coefficient_scale_t = torch.as_tensor(coefficient_scale, device=device)
    truth_amplitude_t = torch.as_tensor(truth_amplitude, device=device)
    training_main_t = torch.as_tensor(training_main, device=device)
    builder = e155.EpisodeBuilder(
        train_table,
        store.positions,
        store.site,
        config.hole_sizes,
        config.seed + 100_000 + outer * 2003,
    )

    inner_anchor = np.sqrt(np.maximum(v7_probability[inner_rows], 0.0)).reshape(
        len(inner_rows), e155.PAS_SIZE
    )
    inner_truth = official_probability[inner_query_ids]
    initial_prediction = e155.predict_table(
        model, inner_table, store, spectral, inner_anchor, device, config
    )
    initial_validation = e155.validation_summary(
        initial_prediction,
        inner_anchor,
        inner_truth,
        basis,
        coefficient_scale,
    )
    best_score = float(initial_validation["best"]["pas"])
    best_alpha = float(initial_validation["best"]["alpha"])
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, object]] = [
        {"step": 0, "validation": initial_validation}
    ]
    print(
        f"E156 outer={outer} step=0 inner_pas={best_score:.6f} "
        f"alpha={best_alpha:.2f}",
        flush=True,
    )

    rolling: list[tuple[float, float, float, float]] = []
    for step in range(1, config.steps + 1):
        model.train()
        rows = builder.sample()
        main_anchor = training_main_t[torch.as_tensor(rows, device=device)]
        prediction = e155.run_model_rows(
            model,
            rows,
            train_table,
            store,
            spectral,
            main_anchor,
            device,
            config,
        )
        query_ids = train_table.query_ids[rows]
        truth = truth_amplitude_t[query_ids].reshape(len(rows), e155.PAS_SIZE)
        target_coefficients = ((truth - main_anchor) @ basis_t.T) / coefficient_scale_t
        predicted_amplitude = e155.reconstruct_amplitude_torch(
            main_anchor,
            prediction,
            basis_t,
            coefficient_scale_t,
            1.0,
        )
        cosine = e155.endpoint_cosine_torch(predicted_amplitude, truth).mean()
        coefficient_loss = torch.nn.functional.smooth_l1_loss(
            prediction, target_coefficients
        )
        action = prediction.square().mean()
        loss = (
            1.0
            - cosine
            + config.coefficient_weight * coefficient_loss
            + config.action_regularization * action
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
        optimizer.step()
        rolling.append(
            (
                float(loss.detach().cpu()),
                float(cosine.detach().cpu()),
                float(coefficient_loss.detach().cpu()),
                float(action.detach().sqrt().cpu()),
            )
        )
        if step % config.validation_interval == 0 or step == config.steps:
            prediction_inner = e155.predict_table(
                model, inner_table, store, spectral, inner_anchor, device, config
            )
            validation = e155.validation_summary(
                prediction_inner,
                inner_anchor,
                inner_truth,
                basis,
                coefficient_scale,
            )
            train_summary = {
                "loss": float(np.mean([item[0] for item in rolling])),
                "pas": float(np.mean([item[1] for item in rolling])),
                "coefficient_loss": float(np.mean([item[2] for item in rolling])),
                "action_rms": float(np.mean([item[3] for item in rolling])),
            }
            history.append(
                {"step": step, "train": train_summary, "validation": validation}
            )
            current = float(validation["best"]["pas"])
            if current > best_score:
                best_score = current
                best_alpha = float(validation["best"]["alpha"])
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"E156 outer={outer} step={step}/{config.steps} "
                f"train_pas={train_summary['pas']:.6f} "
                f"inner_pas={current:.6f} alpha={validation['best']['alpha']:.2f} "
                f"best={best_score:.6f}@{best_step}",
                flush=True,
            )
            rolling.clear()

    model.load_state_dict(best_state)
    outer_main = np.sqrt(np.maximum(v7_probability[outer_rows], 0.0)).reshape(
        len(outer_rows), e155.PAS_SIZE
    )
    outer_prediction = e155.predict_table(
        model, outer_table, store, spectral, outer_main, device, config
    )
    outer_target_amplitude = e155.reconstruct_amplitude_numpy(
        outer_main,
        outer_prediction,
        basis,
        coefficient_scale,
        best_alpha,
    )
    outer_target = np.square(outer_target_amplitude)
    outer_truth = official_probability[outer_query_ids]
    endpoint_baseline = e155.endpoint_cosine_numpy(
        np.square(outer_main.reshape(-1, e155.MP, e155.MH, e155.MV, e155.N_UE)),
        outer_truth,
    )
    endpoint_prediction = e155.endpoint_cosine_numpy(outer_target, outer_truth)

    oracle_coefficients = (
        truth_amplitude[outer_query_ids].reshape(len(outer_query_ids), e155.PAS_SIZE)
        - outer_main
    ) @ basis.T
    oracle_standardized = oracle_coefficients / coefficient_scale[None]
    oracle_target = e155.reconstruct_amplitude_numpy(
        outer_main,
        oracle_standardized,
        basis,
        coefficient_scale,
        1.0,
    ) ** 2

    rbf_main = np.asarray(train_table.anchors()[:, 0], dtype=np.float32).reshape(
        len(train_query_ids), e155.PAS_SIZE
    )
    anchor_cosine = e155.endpoint_cosine_numpy(
        np.square(rbf_main.reshape(-1, e155.MP, e155.MH, e155.MV, e155.N_UE)),
        np.square(training_main.reshape(-1, e155.MP, e155.MH, e155.MV, e155.N_UE)),
    )
    checkpoint = ARTIFACT_DIR / f"outer{outer}_v7_domain_pas.pt"
    torch.save(
        {
            "state_dict": best_state,
            "config": asdict(config),
            "outer": outer,
            "inner": inner,
            "best_step": best_step,
            "best_alpha": best_alpha,
            "spectral_mean": spectral.mean,
            "spectral_components": spectral.components,
            "spectral_scale": spectral.scale,
            "correction_basis": basis,
            "coefficient_scale": coefficient_scale,
        },
        checkpoint,
    )
    record = {
        "outer": outer,
        "inner": inner,
        "train_count": int(len(train_query_ids)),
        "inner_count": int(len(inner_query_ids)),
        "outer_count": int(len(outer_query_ids)),
        "teacher_basis_explained_energy": basis_explained,
        "rbf_to_exact_v7_pas_cosine": float(np.mean(anchor_cosine)),
        "best_step": best_step,
        "best_alpha": best_alpha,
        "best_inner_pas": best_score,
        "outer_endpoint": {
            "baseline_pas": float(np.mean(endpoint_baseline)),
            "prediction_pas": float(np.mean(endpoint_prediction)),
            "delta": float(np.mean(endpoint_prediction - endpoint_baseline)),
            "positive_fraction": float(np.mean(endpoint_prediction >= endpoint_baseline)),
        },
        "history": history,
        "checkpoint": PATHS.relative(checkpoint),
        "seconds": float(time.time() - started),
    }
    return record, outer_target.astype(np.float32), oracle_target.astype(np.float32)


def evaluate_complete_masked(
    endpoint_pas: np.ndarray,
    action_mask: np.ndarray,
    truth_all: np.ndarray,
    query_ids: np.ndarray,
    device: torch.device,
    batch_size: int,
    config: e155.Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep V7 bit-for-bit unchanged on rows whose selected action is zero."""

    base_all = np.load(e155.V7_CHANNEL, mmap_mode="r")
    arrays = tuple(np.empty(len(query_ids), dtype=np.float64) for _ in range(4))
    for start in range(0, len(query_ids), batch_size):
        stop = min(start + batch_size, len(query_ids))
        base = torch.as_tensor(
            np.array(base_all[start:stop], dtype=np.complex64, copy=True), device=device
        )
        truth = torch.as_tensor(
            np.array(truth_all[query_ids[start:stop]], dtype=np.complex64, copy=True),
            device=device,
        )
        prediction = base.clone()
        active = np.flatnonzero(action_mask[start:stop])
        if len(active):
            compact = torch.as_tensor(
                np.array(endpoint_pas[start:stop][active], dtype=np.float32, copy=True),
                device=device,
            )
            full_endpoint = compact[..., None].expand(
                -1, -1, -1, -1, -1, e155.S_SC
            )
            with torch.inference_mode():
                active_base = base[active]
                base_pas = e155.pas_distribution(active_base)
                desired_pas = e155.endpoint_target(
                    base_pas,
                    full_endpoint,
                    config.project_beta,
                    "sqrt",
                    (1, 2, 3),
                )
                reference_power = active_base.abs().square().sum(dim=(1, 2, 3))
                active_prediction = active_base
                for _ in range(config.project_cycles):
                    active_prediction = e155.exact_pas_projection(
                        active_prediction, desired_pas
                    )
                    active_prediction = e155.preserve_point_power(
                        active_prediction, reference_power
                    )
                prediction[active] = active_prediction
        with torch.inference_mode():
            values = e155.point_metrics(
                prediction,
                truth,
                e155.pas_distribution(truth),
                e155.pdp_distribution(truth),
            )
        for destination, source in zip(arrays, values, strict=True):
            destination[start:stop] = source
    return arrays


def main() -> None:
    args = parse_args()
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    e155.ARTIFACT_DIR = ARTIFACT_DIR
    e155.ANALYSIS_DIR = ANALYSIS_DIR
    config = e155.Config(
        steps=args.steps,
        validation_interval=args.validation_interval,
        seed=260_810_156,
    )
    device = torch.device(args.device)
    selected_folds = (
        list(range(5))
        if args.folds == "all"
        else [int(value) for value in args.folds.split(",")]
    )
    if any(value not in range(5) for value in selected_folds):
        raise ValueError(selected_folds)

    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    store = e155.SceneStore()
    official_probability, v7_probability, teacher_probability = e155.load_pas_arrays(
        device, args.metric_batch_size
    )
    truth_amplitude = np.sqrt(np.maximum(official_probability, 0.0)).astype(np.float32)
    if args.score_only:
        if not REPORT.is_file() or not METRICS.is_file():
            raise FileNotFoundError("E156 score-only requires the completed report and metrics")
        previous = json.loads(REPORT.read_text(encoding="utf-8"))
        cached = np.load(METRICS, allow_pickle=False)
        predicted_pas = cached["predicted_pas"].astype(np.float32)
        oracle_pas = cached["oracle_pas"].astype(np.float32)
        records = previous["fold_training"]
        selected_folds = previous["selected_folds"]
    else:
        predicted_pas = np.array(v7_probability, dtype=np.float32, copy=True)
        oracle_pas = np.array(v7_probability, dtype=np.float32, copy=True)
        records = []
        for outer in selected_folds:
            record, outer_target, outer_oracle = train_fold(
                outer,
                config,
                store,
                official_probability,
                truth_amplitude,
                v7_probability,
                teacher_probability,
                query_ids,
                fold_id,
                device,
                args.force_context,
            )
            rows = np.flatnonzero(fold_id == outer)
            predicted_pas[rows] = outer_target
            oracle_pas[rows] = outer_oracle
            records.append(record)

    payload: dict[str, object] = {
        "experiment": VERSION,
        "status": "partial_screen" if len(selected_folds) < 5 else "strict_screen_complete",
        "official_inputs_only_at_inference": True,
        "sionna_or_v12_held_values_used_as_input": False,
        "primary_hypothesis": (
            "E155 failed mainly because its train-time main anchor was a guarded RBF "
            "field while held inference corrected the exact V7 PAS. E156 uses exact "
            "cross-fitted V7 PAS as every train, inner, and outer main anchor."
        ),
        "config": asdict(config),
        "selected_folds": selected_folds,
        "fold_training": records,
    }
    if len(selected_folds) == 5:
        truth_all = np.load(e155.TRAIN_CHANNEL, mmap_mode="r")
        selected_action = np.zeros(len(query_ids), dtype=bool)
        for record in records:
            selected_action[fold_id == int(record["outer"])] = (
                float(record["best_alpha"]) > 0.0
            )
        prediction_values = evaluate_complete_masked(
            predicted_pas,
            selected_action,
            truth_all,
            query_ids,
            device,
            args.metric_batch_size,
            config,
        )
        baseline_values = evaluate_complete_masked(
            v7_probability,
            np.zeros(len(query_ids), dtype=bool),
            truth_all,
            query_ids,
            device,
            args.metric_batch_size,
            config,
        )
        oracle_values = evaluate_complete_masked(
            oracle_pas,
            np.ones(len(query_ids), dtype=bool),
            truth_all,
            query_ids,
            device,
            args.metric_batch_size,
            config,
        )
        result = e155.summarize_complete(prediction_values, baseline_values, fold_id)
        oracle = e155.summarize_complete(oracle_values, baseline_values, fold_id)
        passed = bool(result["delta_to_v7"] >= 0.005 and result["positive_folds"] == 5)
        payload.update(
            {
                "status": "analysis_complete_gate_passed" if passed else "analysis_complete_gate_failed",
                "complete_result": result,
                "oracle": oracle,
                "deployment_gate": {
                    "minimum_delta_to_v7": 0.005,
                    "minimum_positive_folds": 5,
                    "actual_delta_to_v7": result["delta_to_v7"],
                    "actual_positive_folds": result["positive_folds"],
                    "passed": passed,
                    "expand_pdp_nmse": passed,
                    "generate_candidate": False,
                },
            }
        )
        np.savez_compressed(
            METRICS,
            query_ids=query_ids.astype(np.int32),
            fold_id=fold_id,
            predicted_pas=predicted_pas.astype(np.float16),
            oracle_pas=oracle_pas.astype(np.float16),
            prediction_pas=prediction_values[0],
            prediction_pdp=prediction_values[1],
            prediction_error=prediction_values[2],
            prediction_energy=prediction_values[3],
            baseline_pas=baseline_values[0],
            baseline_pdp=baseline_values[1],
            baseline_error=baseline_values[2],
            baseline_energy=baseline_values[3],
            oracle_metric_pas=oracle_values[0],
            oracle_metric_pdp=oracle_values[1],
            oracle_metric_error=oracle_values[2],
            oracle_metric_energy=oracle_values[3],
        )
        payload["outputs"] = {
            "report": PATHS.relative(REPORT),
            "metrics": PATHS.relative(METRICS),
        }
    payload["seconds"] = float(time.time() - started)
    if not e155.finite_payload(payload):
        raise ValueError("non-finite E156 report")
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
