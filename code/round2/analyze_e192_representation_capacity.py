#!/usr/bin/env python3
"""Check the independent E192 event-state renderer on canonical fixed500 holes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import resource
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import analyze_global_local_path_lifecycle as e79
from common.official_metrics import (
    PointMetricStatistics,
    aggregate_statistics,
    point_metric_statistics_torch,
)
from project_paths import PATHS
from run_path_component_transport import component_energy, synthesize_batch


VERSION = "Round2-E192-representation-capacity"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e192_multiseed_path_generator"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e192_multiseed_path_generator"
ANCHOR_DIR = ARTIFACT_DIR / "event_anchors_train"
HISTORY_DIR = ARTIFACT_DIR / "event_history_train"
REPORT = ANALYSIS_DIR / "representation_capacity.json"
METRICS = ARTIFACT_DIR / "representation_capacity_metrics.npz"

PATH_CACHE = (
    PATHS.artifacts
    / "round2"
    / "joint_path_token_oracle"
    / "n4000_s24_k128_b096_d064_scatter1_sel106"
)
COMPONENTS = (
    PATHS.artifacts
    / "round2"
    / "v4_path_component_transport"
    / "train_components_k24_os2_ref2.npz"
)
OOF_META = (
    PATHS.artifacts
    / "round2"
    / "pre_sionna"
    / "round2_pre_sionna_oof_foundation_meta.npz"
)
TRAIN_CHANNEL = PATHS.round2 / "Round2_Train_Channel.npy"

PERIODS = np.asarray((16.0, 8.0, 192.0), dtype=np.float32)
TINY = 1e-30


@dataclass(frozen=True)
class Config:
    minimum_fraction: float = 1e-5
    match_cost_limit: float = 1.0
    match_bandwidth: float = 0.25
    batch_size: int = 16
    minimum_score: float = 0.800000
    v15_pas: float = 0.6769415758539015
    v15_pdp: float = 0.7803473980573273
    v15_nmse_score: float = 0.5172970702149067


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=Config().batch_size)
    return parser.parse_args()


def required_inputs() -> tuple:
    return (
        ANCHOR_DIR / "mask.npy",
        ANCHOR_DIR / "representative_slot.npy",
        HISTORY_DIR / "complete.json",
        HISTORY_DIR / "mask.npy",
        PATH_CACHE / "mask.npy",
        COMPONENTS,
        OOF_META,
        TRAIN_CHANNEL,
    )


def event_coordinates(
    anchor_mask: np.ndarray,
    representative_slot: np.ndarray,
    raw_coordinate: np.ndarray,
) -> np.ndarray:
    rows, events = np.nonzero(anchor_mask)
    slots = representative_slot[rows, events]
    if np.any(slots < 0):
        raise RuntimeError("active E192 event has no representative slot")
    output = np.zeros((*anchor_mask.shape, 3), dtype=np.float32)
    output[rows, events] = raw_coordinate[rows, slots]
    return output


def build_oracle_state(
    candidate_coordinate: np.ndarray,
    candidate_mask: np.ndarray,
    truth_coordinate: np.ndarray,
    truth_coefficient: np.ndarray,
    truth_power: np.ndarray,
    config: Config,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    count, candidates, _ = candidate_coordinate.shape
    output_coordinate = np.array(candidate_coordinate, copy=True)
    output_coefficient = np.zeros((count, candidates, 8), dtype=np.complex64)
    matched_count = np.zeros(count, dtype=np.int16)
    matched_fraction = np.zeros(count, dtype=np.float64)
    available_fraction = np.zeros(count, dtype=np.float64)
    matched_cost_parts: list[np.ndarray] = []

    for row in range(count):
        candidate_ids = np.flatnonzero(candidate_mask[row])
        if len(candidate_ids) == 0 or truth_power[row] <= TINY:
            continue
        truth_energy = component_energy(truth_coefficient[row])
        truth_fraction = truth_energy / max(float(truth_power[row]), TINY)
        truth_ids = np.flatnonzero(truth_fraction >= config.minimum_fraction)
        available_fraction[row] = float(np.sum(truth_fraction[truth_ids]))
        if len(truth_ids) == 0:
            continue

        cost = e79.support_cost(
            candidate_coordinate[row, candidate_ids], truth_coordinate[row, truth_ids]
        )
        utility = truth_fraction[truth_ids][None] * np.exp(
            -0.5 * cost / config.match_bandwidth
        )
        utility[cost > config.match_cost_limit] = 0.0
        candidate_local, truth_local = linear_sum_assignment(-utility)
        accepted = utility[candidate_local, truth_local] > 0.0
        candidate_local = candidate_local[accepted]
        truth_local = truth_local[accepted]
        if len(candidate_local) == 0:
            continue

        assigned_candidate = candidate_ids[candidate_local]
        assigned_truth = truth_ids[truth_local]
        output_coordinate[row, assigned_candidate] = truth_coordinate[
            row, assigned_truth
        ]
        output_coefficient[row, assigned_candidate] = truth_coefficient[
            row, assigned_truth
        ]
        matched_count[row] = len(assigned_candidate)
        matched_fraction[row] = float(np.sum(truth_fraction[assigned_truth]))
        matched_cost_parts.append(cost[candidate_local, truth_local])

    matched_cost = (
        np.concatenate(matched_cost_parts)
        if matched_cost_parts
        else np.empty(0, dtype=np.float32)
    )
    nonzero = truth_power > TINY
    audit = {
        "point_count": int(count),
        "nonzero_count": int(np.sum(nonzero)),
        "matched_path_count_nonzero_mean": float(np.mean(matched_count[nonzero])),
        "matched_path_count_nonzero_median": float(np.median(matched_count[nonzero])),
        "matched_fraction_nonzero_mean": float(np.mean(matched_fraction[nonzero])),
        "matched_fraction_of_available": float(
            np.sum(matched_fraction) / max(float(np.sum(available_fraction)), TINY)
        ),
        "matched_cost_mean": float(np.mean(matched_cost)) if len(matched_cost) else 0.0,
        "matched_cost_p90": float(np.quantile(matched_cost, 0.90))
        if len(matched_cost)
        else 0.0,
    }
    return {
        "coordinate": output_coordinate,
        "coefficient": output_coefficient,
        "matched_count": matched_count,
        "matched_fraction": matched_fraction,
    }, audit


@torch.inference_mode()
def evaluate(
    coordinate: np.ndarray,
    coefficient: np.ndarray,
    query_ids: np.ndarray,
    fold_id: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, object], PointMetricStatistics]:
    truth_all = np.load(TRAIN_CHANNEL, mmap_mode="r")
    pas_parts: list[np.ndarray] = []
    pdp_parts: list[np.ndarray] = []
    error_parts: list[np.ndarray] = []
    energy_parts: list[np.ndarray] = []
    for start in range(0, len(query_ids), batch_size):
        stop = min(start + batch_size, len(query_ids))
        prediction = synthesize_batch(
            coordinate[start:stop], coefficient[start:stop], device
        )
        truth = torch.as_tensor(
            np.asarray(truth_all[query_ids[start:stop]], dtype=np.complex64),
            device=device,
        )
        values = point_metric_statistics_torch(prediction, truth)
        pas_parts.append(values.pas)
        pdp_parts.append(values.pdp)
        error_parts.append(values.squared_error)
        energy_parts.append(values.truth_energy)

    values = PointMetricStatistics(
        pas=np.concatenate(pas_parts),
        pdp=np.concatenate(pdp_parts),
        squared_error=np.concatenate(error_parts),
        truth_energy=np.concatenate(energy_parts),
    )
    summary = {
        "overall": aggregate_statistics(values),
        "folds": [
            {"fold": fold, **aggregate_statistics(values, fold_id == fold)}
            for fold in range(5)
        ],
    }
    return summary, values


def main() -> None:
    args = parse_args()
    started = time.time()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [PATHS.relative(path) for path in required_inputs() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing E192 capacity inputs: {missing}")

    config = Config(batch_size=args.batch_size)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()

    load_started = time.time()
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    anchor_mask_all = np.load(ANCHOR_DIR / "mask.npy", mmap_mode="r")
    history_mask_all = np.load(HISTORY_DIR / "mask.npy", mmap_mode="r")
    representative_all = np.load(
        ANCHOR_DIR / "representative_slot.npy", mmap_mode="r"
    )
    raw_mask = np.load(PATH_CACHE / "mask.npy", mmap_mode="r")
    component = np.load(COMPONENTS, allow_pickle=False)

    # E85's deployed coordinate convention is [H,V,D] with +2 vertical alignment.
    import train_surface_interaction_energy_model as e85

    raw_coordinate = np.asarray(
        np.load(e85.BASE_COORDINATE, mmap_mode="r"), dtype=np.float32
    )
    if raw_coordinate.shape[:2] != raw_mask.shape:
        raise RuntimeError("raw E192 coordinate/candidate identity mismatch")
    anchor_mask = np.asarray(anchor_mask_all[query_ids], dtype=bool)
    history_active = np.any(
        np.asarray(history_mask_all[:, query_ids], dtype=bool), axis=0
    )
    candidate_mask = anchor_mask & history_active
    representative = np.asarray(representative_all[query_ids], dtype=np.int16)
    coordinate = event_coordinates(
        candidate_mask, representative, raw_coordinate[query_ids]
    )
    truth_coordinate = component["coordinates"][query_ids].astype(np.float32)
    truth_coefficient = component["coefficients"][query_ids].astype(np.complex64)
    truth_power = component["point_power"][query_ids].astype(np.float64)
    load_seconds = time.time() - load_started

    label_started = time.time()
    state, assignment = build_oracle_state(
        coordinate,
        candidate_mask,
        truth_coordinate,
        truth_coefficient,
        truth_power,
        config,
    )
    label_seconds = time.time() - label_started

    score_started = time.time()
    summary, values = evaluate(
        state["coordinate"],
        state["coefficient"],
        query_ids,
        fold_id,
        device,
        config.batch_size,
    )
    score_seconds = time.time() - score_started
    overall = summary["overall"]
    component_gate = {
        "pas_above_v15": bool(float(overall["pas"]) > config.v15_pas),
        "pdp_above_v15": bool(float(overall["pdp"]) > config.v15_pdp),
        "nmse_score_above_v15": bool(
            float(overall["nmse_score"]) > config.v15_nmse_score
        ),
    }
    passed = bool(
        float(overall["score"]) >= config.minimum_score
        and all(component_gate.values())
    )
    np.savez_compressed(
        METRICS,
        query_ids=query_ids.astype(np.int32),
        fold_id=fold_id,
        matched_count=state["matched_count"],
        matched_fraction=state["matched_fraction"],
        pas=values.pas,
        pdp=values.pdp,
        squared_error=values.squared_error,
        truth_energy=values.truth_energy,
    )

    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(device)
        peak_gpu = float(torch.cuda.max_memory_allocated() / 2**30)
    else:
        gpu = "cpu"
        peak_gpu = 0.0
    report = {
        "experiment": VERSION,
        "status": "representation_capacity_passed"
        if passed
        else "representation_capacity_failed",
        "configuration": asdict(config),
        "data_lineage": {
            "candidate_identity": "E192 event anchors from raw 24-seed external paths",
            "held_state": "official Train H 24-path decomposition, oracle-only",
            "renderer": "independent analytic 24-path synthesis",
            "historical_version_channel_used": False,
        },
        "assignment": assignment,
        "candidate_availability": {
            "anchor_events": int(np.sum(anchor_mask)),
            "history_backed_events": int(np.sum(candidate_mask)),
            "history_backed_ratio": float(
                np.sum(candidate_mask) / max(int(np.sum(anchor_mask)), 1)
            ),
        },
        "metrics": summary,
        "gate": {
            "minimum_score": config.minimum_score,
            "component_gate": component_gate,
            "passed": passed,
        },
        "performance": {
            "device": gpu,
            "peak_torch_memory_gib": peak_gpu,
            "peak_rss_gib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            ),
            "timing_seconds": {
                "load": float(load_seconds),
                "hungarian_labels": float(label_seconds),
                "gpu_synthesis_and_official_scoring": float(score_seconds),
            },
            "measured_bottleneck": max(
                {
                    "load": load_seconds,
                    "hungarian_labels": label_seconds,
                    "gpu_synthesis_and_official_scoring": score_seconds,
                },
                key={
                    "load": load_seconds,
                    "hungarian_labels": label_seconds,
                    "gpu_synthesis_and_official_scoring": score_seconds,
                }.get,
            ),
        },
        "outputs": {"metrics": PATHS.relative(METRICS)},
        "seconds": float(time.time() - started),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
