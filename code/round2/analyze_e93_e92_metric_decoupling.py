#!/usr/bin/env python3
"""Evaluate PAS/PDP-specific deployment scales for the frozen E92 gates."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import torch

from project_paths import PATHS
import train_e88_dense_path_residual_student as e88
import train_e90_e88_slice_reliability_gates as e90
import train_e92_full4000_e88_transfer_gates as e92
import train_offline_sionna_geometry_student as e87


VERSION = "Round2-E93-E92-metric-decoupling"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e92_metric_decoupling"
REPORT = ANALYSIS_DIR / "e92_metric_decoupling.json"
PRIMARY = {"pas_scale": 0.25, "pdp_scale": 0.50, "order": "pas_pdp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def fold_metrics(
    values: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    baseline_values: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    fold_id: np.ndarray,
) -> list[dict[str, float | int]]:
    rows = []
    for fold in range(5):
        mask = fold_id == fold
        current = e87.statistics(*values, mask)
        baseline = e87.statistics(*baseline_values, mask)
        rows.append(
            {
                "fold": fold,
                "delta": float(current["score"] - baseline["score"]),
            }
        )
    return rows


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    started = time.time()

    alpha_payload = np.load(e92.FIXED_ALPHA, allow_pickle=False)
    query_ids = alpha_payload["query_ids"].astype(np.int64)
    fold_id = alpha_payload["fold_id"].astype(np.int8)
    alpha_pas = torch.as_tensor(
        alpha_payload["alpha_pas"].astype(np.float32), device=device
    )
    alpha_pdp = torch.as_tensor(
        alpha_payload["alpha_pdp"].astype(np.float32), device=device
    )
    if alpha_pas.shape != (500, 4, 192) or alpha_pdp.shape != (500, 1024):
        raise ValueError("unexpected E92 fixed-alpha shapes")

    metadata = np.load(e88.OOF_META, allow_pickle=False)
    if not np.array_equal(metadata["query_ids"].astype(np.int64), query_ids):
        raise ValueError("E92 alpha rows do not match the fixed-500 metadata")
    if not np.array_equal(metadata["fold_id"].astype(np.int8), fold_id):
        raise ValueError("E92 alpha reporting folds do not match the fixed holes")

    base_channel = torch.as_tensor(
        np.array(np.load(e87.V12_CHANNEL, mmap_mode="r"), copy=True), device=device
    )
    truth_all = np.load(PATHS.round2 / "Round2_Train_Channel.npy", mmap_mode="r")
    truth_channel = torch.as_tensor(
        np.asarray(truth_all[query_ids], dtype=np.complex64), device=device
    )
    endpoint_payload = np.load(e88.OOF_PREDICTIONS, allow_pickle=False)
    endpoint_pas, endpoint_pdp = e87.expanded_endpoint(
        endpoint_payload["final_pas"].astype(np.float32),
        endpoint_payload["final_pdp"].astype(np.float32),
        device,
    )
    base_pas = e87.pas_distribution(base_channel)
    base_pdp = e87.pdp_distribution(base_channel)
    truth_pas = e87.pas_distribution(truth_channel)
    truth_pdp = e87.pdp_distribution(truth_channel)
    baseline_values = e87.point_metrics(
        base_channel, truth_channel, truth_pas, truth_pdp
    )
    baseline = e87.statistics(*baseline_values)

    pas_scales = (0.0, 0.125, 0.25, 0.375, 0.50)
    pdp_scales = (0.0, 0.25, 0.50, 0.75, 1.00)
    rows: list[dict[str, Any]] = []
    for pas_scale in pas_scales:
        desired_pas = e90.root_blend(
            base_pas,
            endpoint_pas,
            (pas_scale * alpha_pas)[:, None, None, None],
            (1, 2, 3),
        )
        for pdp_scale in pdp_scales:
            if pas_scale == 0.0 and pdp_scale == 0.0:
                continue
            desired_pdp = e90.root_blend(
                base_pdp,
                endpoint_pdp,
                (pdp_scale * alpha_pdp).reshape(len(base_channel), 256, 4, 1),
                (3,),
            )
            for order in ("pas_pdp", "pdp_pas"):
                prediction = e87.project_joint(
                    base_channel, desired_pas, desired_pdp, 4, order
                )
                values = e87.point_metrics(
                    prediction, truth_channel, truth_pas, truth_pdp
                )
                metrics = e87.statistics(*values)
                folds = fold_metrics(values, baseline_values, fold_id)
                rows.append(
                    {
                        "pas_scale": pas_scale,
                        "pdp_scale": pdp_scale,
                        "order": order,
                        "metrics": metrics,
                        "delta": float(metrics["score"] - baseline["score"]),
                        "positive_folds": int(
                            sum(float(row["delta"]) > 0.0 for row in folds)
                        ),
                        "worst_fold_delta": float(
                            min(float(row["delta"]) for row in folds)
                        ),
                        "folds": folds,
                    }
                )

    primary = next(
        row
        for row in rows
        if row["pas_scale"] == PRIMARY["pas_scale"]
        and row["pdp_scale"] == PRIMARY["pdp_scale"]
        and row["order"] == PRIMARY["order"]
    )
    stable = [row for row in rows if row["positive_folds"] == 5]
    best_stable = max(stable, key=lambda row: float(row["delta"]))
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "E92 over-corrects V12 PAS before its PDP gate is fully used; "
            "metric-specific scales should preserve PAS and release more PDP gain."
        ),
        "configuration": {
            "projection_cycles": 4,
            "pas_scales": pas_scales,
            "pdp_scales": pdp_scales,
            "orders": ("pas_pdp", "pdp_pas"),
            "primary": PRIMARY,
        },
        "V12": baseline,
        "alpha": {
            "pas_mean": float(alpha_pas.mean().item()),
            "pdp_mean": float(alpha_pdp.mean().item()),
        },
        "primary": primary,
        "best_stable_scan": best_stable,
        "scan": rows,
        "decision": {
            "primary_passed": bool(
                primary["delta"] > 0.0 and primary["positive_folds"] == 5
            ),
            "candidate_allowed": False,
            "reason": (
                "This experiment isolates metric decoupling. A final candidate also "
                "requires a frozen test-time training and inference path."
            ),
        },
        "seconds": time.time() - started,
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
