#!/usr/bin/env python3
"""Measure strict held-hole performance at fixed E107 adaptation epochs."""

from __future__ import annotations

from dataclasses import asdict
import json
import time

import numpy as np
import torch

import analyze_e100_locked_energy_angle_correction as e100
import analyze_e86_surface_endpoint_v11_blend as e86
import train_e104_locked_pdp_angle_transport as e104
import train_e107_v11_conditioned_transport as e107
import train_surface_interaction_energy_model as e85
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from project_paths import PATHS
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics


VERSION = "Round2-E108-fixed-epoch-transport-trajectory"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "fixed_epoch_transport_trajectory"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_fixed_epoch_transport_trajectory"
REPORT = ANALYSIS_DIR / "fixed_epoch_transport_trajectory.json"
METRICS = ARTIFACT_DIR / "fixed_epoch_transport_trajectory_metrics.npz"
CHECKPOINT_EPOCHS = (0, 1, 2, 4, 8, 12)


def main() -> None:
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = e107.AdaptConfig()
    required = (
        e104.FULL_ENDPOINT,
        e104.STRICT_ENDPOINT,
        e104.VALID_MASK,
        e107.OLD_PDP,
        e86.V11_CHANNEL,
        e86.TRAIN_CHANNEL,
        e100.V12_CHANNEL,
    )
    missing = [PATHS.relative(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")

    endpoint_store = e85.DataStore()
    full_endpoint = e104.EndpointData(e104.FULL_ENDPOINT, strict=False)
    strict_endpoint = e104.EndpointData(e104.STRICT_ENDPOINT, strict=True)
    query_ids = endpoint_store.strict_ids
    fold_id = endpoint_store.strict_fold
    canonical = e107.CanonicalStore(query_ids)
    valid = np.asarray(np.load(e104.VALID_MASK, mmap_mode="r"), dtype=bool)
    all_rows = np.arange(len(endpoint_store.base), dtype=np.int64)

    truth = torch.as_tensor(
        np.asarray(canonical.truth[query_ids], dtype=np.complex64), device=device
    )
    v12 = torch.as_tensor(
        np.array(canonical.v12, dtype=np.complex64, copy=True), device=device
    )
    with torch.inference_mode():
        v12_metrics = point_metrics(
            v12, truth, pas_distribution(truth), pdp_distribution(truth)
        )
    del truth, v12

    trajectory = {
        epoch: [np.zeros(len(query_ids), dtype=np.float64) for _ in range(4)]
        for epoch in CHECKPOINT_EPOCHS
    }
    fold_training = []
    for outer in range(5):
        held = query_ids[fold_id == outer]
        fit_canonical = query_ids[fold_id != outer]
        held_mask = np.zeros(len(endpoint_store.base), dtype=bool)
        held_mask[held] = True
        fit_rows = all_rows[~held_mask]
        mean, std = endpoint_store.normalization(fit_rows)
        model = e107.pretrain(
            endpoint_store,
            full_endpoint,
            fit_rows,
            mean,
            std,
            valid,
            config.pretrain_epochs,
            config.random_state + 100000 + 10000 * outer,
            device,
        )
        trainable = e107.trainable_transport_heads(model)
        optimizer = torch.optim.AdamW(
            trainable, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        rng = np.random.default_rng(
            config.random_state + 120000 + 10000 * outer
        )
        local_held = np.flatnonzero(fold_id == outer)
        held_rows = []

        for epoch in range(0, max(CHECKPOINT_EPOCHS) + 1):
            if epoch > 0:
                train = e107.train_adapt_epoch(
                    model,
                    optimizer,
                    endpoint_store,
                    strict_endpoint,
                    canonical,
                    fit_canonical,
                    mean,
                    std,
                    valid,
                    config,
                    device,
                    rng,
                )
            else:
                train = None
            if epoch not in CHECKPOINT_EPOCHS:
                continue
            prediction = e107.evaluate_model(
                model,
                endpoint_store,
                strict_endpoint,
                canonical,
                held,
                mean,
                std,
                valid,
                config,
                device,
                False,
            )
            for destination, source in zip(trajectory[epoch], prediction["metrics"]):
                destination[local_held] = source
            baseline = statistics(
                *(value[local_held] for value in v12_metrics)
            )
            held_rows.append(
                {
                    "epoch": epoch,
                    "train": train,
                    "held": prediction["summary"],
                    "held_delta": float(
                        prediction["summary"]["score"] - baseline["score"]
                    ),
                }
            )
            print(
                f"E108 outer={outer} epoch={epoch} "
                f"held_delta={held_rows[-1]['held_delta']:.6f}",
                flush=True,
            )
        fold_training.append(
            {
                "outer": outer,
                "pretrain_count": int(len(fit_rows)),
                "adapt_count": int(len(fit_canonical)),
                "trajectory": held_rows,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fixed_epochs = []
    save_arrays: dict[str, np.ndarray] = {
        "query_ids": query_ids.astype(np.int32),
        "fold_id": fold_id,
    }
    for epoch in CHECKPOINT_EPOCHS:
        values = tuple(trajectory[epoch])
        comparison = e100.metric_comparison(values, v12_metrics, fold_id)
        eligible = bool(
            comparison["delta"] >= 0.001
            and comparison["positive_folds"] >= 4
            and comparison["worst_fold_delta"] >= -0.001
        )
        fixed_epochs.append(
            {
                "epoch": epoch,
                "versus_v12": comparison,
                "passes_stage_gate": eligible,
            }
        )
        for name, value in zip(("pas", "pdp", "error", "energy"), values):
            save_arrays[f"epoch{epoch}_{name}"] = value.astype(
                np.float64 if name in ("error", "energy") else np.float32
            )
    winner = max(
        fixed_epochs, key=lambda row: float(row["versus_v12"]["delta"])
    )
    stage_passed = any(row["passes_stage_gate"] for row in fixed_epochs)
    np.savez_compressed(METRICS, **save_arrays)
    report = {
        "experiment": VERSION,
        "status": "strict_diagnostic_complete",
        "primary_hypothesis": (
            "A fixed adaptation stage, independent of an alternate validation hole, "
            "can make the V11-conditioned transport gain consistent across holes."
        ),
        "validation": (
            "Each fixed epoch is evaluated on five disjoint held holes. Held results "
            "are diagnostic only and cannot directly select a V13 candidate."
        ),
        "config": {
            "e107": {
                "transport": asdict(e104.Config()),
                "adapt": asdict(config),
            },
            "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
            "stage_gate": {
                "minimum_score_gain": 0.001,
                "minimum_positive_folds": 4,
                "minimum_worst_fold_delta": -0.001,
            },
        },
        "fold_training": fold_training,
        "fixed_epochs": fixed_epochs,
        "best_posthoc_diagnostic": winner,
        "decision": {
            "stage_passed": stage_passed,
            "generate_v13": False,
            "reason": (
                "A passing fixed stage requires a separate nested confirmation."
                if stage_passed
                else "No fixed stage has stable material gain across holes."
            ),
        },
        "outputs": {
            "metrics": PATHS.relative(METRICS),
            "finite": bool(
                all(np.isfinite(value).all() for value in save_arrays.values())
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
                "fixed_epochs": fixed_epochs,
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
