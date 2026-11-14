#!/usr/bin/env python3
"""Audit paired sequence/control endpoint ensembles around the V12 policy."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import torch

import analyze_e86_surface_endpoint_v11_blend as e86
from analyze_v4_exact_joint_projection import pas_distribution, pdp_distribution
from project_paths import PATHS
from run_rbf_spectral_field import statistics
from run_v9_joint_sionna_migration import point_metrics


VERSION = "Round2-E96-surface-control-endpoint-ensemble"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_surface_control_endpoint_ensemble"
REPORT = ANALYSIS_DIR / "surface_control_endpoint_ensemble.json"
WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)
PRIMARY = {"pas_geometry_weight": 0.75, "pdp_geometry_weight": 0.75}
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def root_ensemble(
    control: torch.Tensor,
    sequence: torch.Tensor,
    sequence_weight: float,
    dims: tuple[int, ...],
) -> torch.Tensor:
    root = (
        (1.0 - sequence_weight) * torch.sqrt(control.clamp_min(0.0))
        + sequence_weight * torch.sqrt(sequence.clamp_min(0.0))
    )
    value = root.square()
    total = value.sum(dim=dims, keepdim=True)
    return torch.where(total > TINY, value / total.clamp_min(TINY), 0.0)


def metric_record(
    name: str,
    pas_weight: float,
    pdp_weight: float,
    values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    fold_id: np.ndarray,
) -> dict[str, Any]:
    selected = statistics(*values)
    anchor = statistics(*baseline)
    folds = []
    for fold in range(5):
        mask = fold_id == fold
        local = statistics(*values, mask)
        local_anchor = statistics(*baseline, mask)
        folds.append(
            {
                "fold": fold,
                "delta": float(local["score"] - local_anchor["score"]),
            }
        )
    return {
        "name": name,
        "pas_geometry_weight": pas_weight,
        "pdp_geometry_weight": pdp_weight,
        "metrics": selected,
        "delta": float(selected["score"] - anchor["score"]),
        "positive_folds": int(sum(row["delta"] > 0.0 for row in folds)),
        "worst_fold_delta": float(min(row["delta"] for row in folds)),
        "folds": folds,
    }


def nested_global_selection(
    values_by_name: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    fold_id: np.ndarray,
) -> dict[str, Any]:
    stitched = tuple(np.array(value, copy=True) for value in baseline)
    decisions = []
    for held_fold in range(5):
        fit = fold_id != held_fold
        winner = max(
            values_by_name,
            key=lambda name: statistics(*values_by_name[name], fit)["score"],
        )
        held = fold_id == held_fold
        for output, source in zip(stitched, values_by_name[winner]):
            output[held] = source[held]
        decisions.append(
            {
                "fold": held_fold,
                "candidate": winner,
                "fit_score": float(statistics(*values_by_name[winner], fit)["score"]),
                "held_delta": float(
                    statistics(*values_by_name[winner], held)["score"]
                    - statistics(*baseline, held)["score"]
                ),
            }
        )
    summary = statistics(*stitched)
    anchor = statistics(*baseline)
    return {
        "metrics": summary,
        "delta": float(summary["score"] - anchor["score"]),
        "positive_folds": int(sum(row["held_delta"] > 0.0 for row in decisions)),
        "worst_fold_delta": float(min(row["held_delta"] for row in decisions)),
        "decisions": decisions,
    }


def pointwise_oracle(
    names: list[str],
    values_by_name: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
    baseline: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    values = np.stack(
        [np.stack(values_by_name[name], axis=0) for name in names], axis=0
    )
    pas = values[:, 0]
    pdp = values[:, 1]
    error = values[:, 2]
    energy = values[:, 3]
    proxy = 0.4 * pas + 0.4 * pdp + 0.2 / (1.0 + error / np.maximum(energy, TINY))
    winner = np.argmax(proxy, axis=0)
    point = np.arange(values.shape[2])
    stitched = tuple(values[winner, metric, point] for metric in range(4))
    summary = statistics(*stitched)
    anchor = statistics(*baseline)
    counts = np.bincount(winner, minlength=len(names))
    return {
        "metrics": summary,
        "delta": float(summary["score"] - anchor["score"]),
        "selection_counts": {
            name: int(count) for name, count in zip(names, counts.tolist())
        },
    }


def main() -> None:
    args = parse_args()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    started = time.time()
    metadata = np.load(e86.OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    base_np = np.load(e86.V11_CHANNEL, mmap_mode="r")
    truth_all = np.load(e86.TRAIN_CHANNEL, mmap_mode="r")
    truth_np = np.asarray(truth_all[query_ids], dtype=np.complex64)
    base = torch.as_tensor(np.array(base_np, copy=True), device=device)
    truth = torch.as_tensor(truth_np, device=device)
    base_pas = pas_distribution(base)
    base_pdp = pdp_distribution(base)
    truth_pas = pas_distribution(truth)
    truth_pdp = pdp_distribution(truth)
    control_pas = torch.as_tensor(
        np.asarray(np.load(e86.RENDERED["no_sequence"][0], mmap_mode="r"), dtype=np.float32),
        device=device,
    )
    control_pdp = torch.as_tensor(
        np.asarray(np.load(e86.RENDERED["no_sequence"][1], mmap_mode="r"), dtype=np.float32),
        device=device,
    )
    sequence_pas = torch.as_tensor(
        np.asarray(
            np.load(e86.RENDERED["geometry_sequence"][0], mmap_mode="r"),
            dtype=np.float32,
        ),
        device=device,
    )
    sequence_pdp = torch.as_tensor(
        np.asarray(
            np.load(e86.RENDERED["geometry_sequence"][1], mmap_mode="r"),
            dtype=np.float32,
        ),
        device=device,
    )
    baseline_values = None
    values_by_name: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    rows = []
    for pas_weight in WEIGHTS:
        endpoint_pas = root_ensemble(
            control_pas, sequence_pas, pas_weight, (1, 2, 3)
        )
        for pdp_weight in WEIGHTS:
            endpoint_pdp = root_ensemble(
                control_pdp, sequence_pdp, pdp_weight, (3,)
            )
            spec = e86.Candidate(
                endpoint="paired_control",
                mode="sqrt",
                pas_beta=0.20,
                pdp_beta=0.20,
                order="pas_pdp",
                cycles=4,
            )
            with torch.inference_mode():
                prediction = e86.make_candidate(
                    base,
                    base_pas,
                    base_pdp,
                    endpoint_pas,
                    endpoint_pdp,
                    spec,
                )
                values = point_metrics(prediction, truth, truth_pas, truth_pdp)
            name = f"geometry_weight_p{pas_weight:.2f}_d{pdp_weight:.2f}"
            values_by_name[name] = values
            if pas_weight == 1.0 and pdp_weight == 1.0:
                baseline_values = values
            print(f"E96 evaluated {name}", flush=True)
    if baseline_values is None:
        raise RuntimeError("V12 geometry endpoint baseline was not evaluated")
    for pas_weight in WEIGHTS:
        for pdp_weight in WEIGHTS:
            name = f"geometry_weight_p{pas_weight:.2f}_d{pdp_weight:.2f}"
            rows.append(
                metric_record(
                    name,
                    pas_weight,
                    pdp_weight,
                    values_by_name[name],
                    baseline_values,
                    fold_id,
                )
            )
    rows.sort(key=lambda row: float(row["metrics"]["score"]), reverse=True)
    primary_name = (
        f"geometry_weight_p{PRIMARY['pas_geometry_weight']:.2f}_"
        f"d{PRIMARY['pdp_geometry_weight']:.2f}"
    )
    primary = next(row for row in rows if row["name"] == primary_name)
    geometry_name = "geometry_weight_p1.00_d1.00"
    control_name = "geometry_weight_p0.00_d0.00"
    nested = nested_global_selection(values_by_name, baseline_values, fold_id)
    oracle_all = pointwise_oracle(
        list(values_by_name), values_by_name, baseline_values
    )
    oracle_pair = pointwise_oracle(
        [geometry_name, control_name], values_by_name, baseline_values
    )
    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "primary_hypothesis": (
            "The paired no-sequence endpoint identifies where ordered surface-sequence "
            "corrections are unstable, so a conservative endpoint ensemble can improve V12."
        ),
        "configuration": {
            "device": args.device,
            "weights": WEIGHTS,
            "primary": PRIMARY,
            "base_beta": 0.20,
            "order": "pas_pdp",
            "cycles": 4,
        },
        "V12": statistics(*baseline_values),
        "primary": primary,
        "top_candidates": rows[:10],
        "nested_global_selection": nested,
        "pointwise_oracle_pair": oracle_pair,
        "pointwise_oracle_grid": oracle_all,
        "decision": {
            "primary_passed": bool(
                primary["delta"] > 0.0 and primary["positive_folds"] == 5
            ),
            "train_slice_selector": bool(oracle_pair["delta"] >= 0.002),
            "candidate_allowed": False,
        },
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
