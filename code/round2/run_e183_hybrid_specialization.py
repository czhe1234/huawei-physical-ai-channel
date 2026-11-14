#!/usr/bin/env python3
"""Run staged gates for E183's nine-hole Clean specialization branch."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
import torch

import analyze_e173_v7_clean_carrier_oracle as e173
import analyze_e180_clean_point_action_oracle as e180
from common.official_metrics import PointMetricStatistics, aggregate_statistics
from project_paths import PATHS


VERSION = "Round2-E183-hybrid-nine-hole-specialization"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e183_hybrid_specialization"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e183_hybrid_specialization"

IDENTITY_METRICS = ARTIFACT_DIR / "canonical_13_action_identity_metrics.npz"
IDENTITY_REPORT = ANALYSIS_DIR / "canonical_13_action_identity_gate.json"
IDENTITY_CSV = ANALYSIS_DIR / "canonical_13动作身份重放.csv"

EXPECTED_ACTION_IDS = (
    "V7",
    "V4",
    "E121_path",
    "E122_path",
    "E142_raw",
    "E142_projected",
    "E161_knn",
    "E162_knn",
    "carrier_00_nearest",
    "carrier_07_raw_idw_k24",
    "carrier_08_radial_idw_k24",
    "carrier_11_radial_affine_k24_r0.1",
    "carrier_16_affine_k24_steer_axis90_v8pi",
)
EXPECTED_V7_SCORE = 0.6444563457343335
EXPECTED_ORACLE_SCORE = 0.6772011759697127
ACTION_TOLERANCE = 5e-5
ORACLE_TOLERANCE = 2e-5
N_POINT = 500
TINY = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("identity",), default="identity")
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_compatible(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_compatible(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def statistics_from_arrays(
    pas: np.ndarray,
    pdp: np.ndarray,
    error: np.ndarray,
    energy: np.ndarray,
) -> PointMetricStatistics:
    return PointMetricStatistics(
        pas=np.asarray(pas, dtype=np.float64),
        pdp=np.asarray(pdp, dtype=np.float64),
        squared_error=np.asarray(error, dtype=np.float64),
        truth_energy=np.asarray(energy, dtype=np.float64),
    )


def score_difference(
    current: dict[str, float | int], reference: dict[str, float | int]
) -> dict[str, float]:
    return {
        name: abs(float(current[name]) - float(reference[name]))
        for name in ("pas", "pdp", "nmse", "score")
    }


def device_report(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        return {"device": str(device), "cuda": False}
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "device": str(device),
        "cuda": True,
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
    }


def load_identity_context() -> tuple[
    list[e180.ActionSpec],
    list[str],
    np.ndarray,
    dict[str, PointMetricStatistics],
    e180.ActionProvider,
]:
    metadata = np.load(e180.OOF_META, allow_pickle=False)
    query_ids = np.asarray(metadata["query_ids"], dtype=np.int64)
    fold_id = np.asarray(metadata["fold_id"], dtype=np.int8)
    e168 = np.load(e180.E168_METRICS, allow_pickle=False)
    if not np.array_equal(query_ids, np.asarray(e168["query_ids"], dtype=np.int64)):
        raise RuntimeError("canonical fixed500 order changed")
    site = np.asarray(e168["site"], dtype=np.int8)
    actions = [action for action in e180.build_actions(fold_id, site) if action.primary]
    action_ids = [action.action_id for action in actions]
    if tuple(action_ids) != EXPECTED_ACTION_IDS:
        raise RuntimeError(f"E180 compact action order changed: {action_ids}")
    cached = e180.load_cached_metrics(actions)
    missing = set(action_ids) - set(cached)
    if missing:
        raise RuntimeError(f"missing E180 cached metrics: {sorted(missing)}")
    return actions, action_ids, query_ids, cached, e180.ActionProvider(actions)


def compute_action_metrics(
    action_ids: list[str],
    query_ids: np.ndarray,
    provider: e180.ActionProvider,
    truth_all: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, PointMetricStatistics], dict[str, float]]:
    truth_host_started = time.perf_counter()
    truth_host = np.asarray(truth_all[query_ids], dtype=np.complex64)
    truth_host_seconds = time.perf_counter() - truth_host_started
    transfer_started = time.perf_counter()
    truth_device = torch.as_tensor(truth_host, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    truth_transfer_seconds = time.perf_counter() - transfer_started

    output: dict[str, PointMetricStatistics] = {}
    io_seconds = 0.0
    transfer_seconds = truth_transfer_seconds
    compute_seconds = 0.0
    for action_index, action_id in enumerate(action_ids):
        arrays = {
            name: np.empty(N_POINT, dtype=np.float64)
            for name in ("pas", "pdp", "error", "energy")
        }
        for start in range(0, N_POINT, batch_size):
            end = min(start + batch_size, N_POINT)
            io_started = time.perf_counter()
            prediction_host = provider.block(action_id, start, end)
            io_seconds += time.perf_counter() - io_started
            if not (
                np.isfinite(prediction_host.real).all()
                and np.isfinite(prediction_host.imag).all()
            ):
                raise ValueError(f"non-finite action block: {action_id} {start}:{end}")

            transfer_started = time.perf_counter()
            prediction = torch.as_tensor(prediction_host, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            transfer_seconds += time.perf_counter() - transfer_started

            compute_started = time.perf_counter()
            pas, pdp, error, energy, _ = e173.point_metrics_torch(
                prediction, truth_device[start:end]
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            compute_seconds += time.perf_counter() - compute_started
            arrays["pas"][start:end] = pas
            arrays["pdp"][start:end] = pdp
            arrays["error"][start:end] = error
            arrays["energy"][start:end] = energy
        output[action_id] = statistics_from_arrays(
            arrays["pas"], arrays["pdp"], arrays["error"], arrays["energy"]
        )
        current = aggregate_statistics(output[action_id])
        print(
            f"E183 identity {action_index + 1}/{len(action_ids)} {action_id}: "
            f"score={current['score']:.9f}",
            flush=True,
        )
    return output, {
        "truth_host_load_seconds": truth_host_seconds,
        "host_action_io_seconds": io_seconds,
        "host_to_device_seconds": transfer_seconds,
        "gpu_metric_seconds": compute_seconds,
    }


def save_point_metrics(
    action_ids: list[str], metrics: dict[str, PointMetricStatistics]
) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        IDENTITY_METRICS,
        action_ids=np.asarray(action_ids, dtype="U64"),
        pas=np.stack([metrics[action_id].pas for action_id in action_ids]),
        pdp=np.stack([metrics[action_id].pdp for action_id in action_ids]),
        error=np.stack(
            [metrics[action_id].squared_error for action_id in action_ids]
        ),
        energy=np.stack([metrics[action_id].truth_energy for action_id in action_ids]),
    )


def run_identity(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    actions, action_ids, query_ids, cached, provider = load_identity_context()
    truth_all = np.load(e180.TRAIN_CHANNEL, mmap_mode="r")
    metrics, timing = compute_action_metrics(
        action_ids,
        query_ids,
        provider,
        truth_all,
        device,
        args.batch_size,
    )
    save_point_metrics(action_ids, metrics)

    rows: list[dict[str, object]] = []
    all_action_pass = True
    for action in actions:
        actual = aggregate_statistics(metrics[action.action_id])
        reference = aggregate_statistics(cached[action.action_id])
        difference = score_difference(actual, reference)
        passed = max(difference.values()) <= ACTION_TOLERANCE
        all_action_pass &= passed
        rows.append(
            {
                "action_id": action.action_id,
                "family": action.family,
                "actual_pas": actual["pas"],
                "reference_pas": reference["pas"],
                "abs_delta_pas": difference["pas"],
                "actual_pdp": actual["pdp"],
                "reference_pdp": reference["pdp"],
                "abs_delta_pdp": difference["pdp"],
                "actual_nmse": actual["nmse"],
                "reference_nmse": reference["nmse"],
                "abs_delta_nmse": difference["nmse"],
                "actual_score": actual["score"],
                "reference_score": reference["score"],
                "abs_delta_score": difference["score"],
                "maximum_abs_delta": max(difference.values()),
                "identity_pass": passed,
            }
        )

    selection, selected_stats, oracle = e180.oracle_selection(action_ids, metrics)
    oracle_delta = abs(float(oracle["score"]) - EXPECTED_ORACLE_SCORE)
    v7 = aggregate_statistics(metrics["V7"])
    v7_delta = abs(float(v7["score"]) - EXPECTED_V7_SCORE)
    oracle_pass = oracle_delta <= ORACLE_TOLERANCE
    v7_pass = v7_delta <= ACTION_TOLERANCE
    overall_pass = bool(all_action_pass and oracle_pass and v7_pass)

    peak_torch_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    elapsed = time.perf_counter() - started
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0**2)
    report = {
        "experiment": VERSION,
        "phase": "canonical_action_identity_gate",
        "status": "identity_gate_passed" if overall_pass else "identity_gate_failed",
        "official_inputs_only": True,
        "sionna_used": False,
        "v8_v12_used": False,
        "model_training": False,
        "action_ids": action_ids,
        "action_tolerance": ACTION_TOLERANCE,
        "oracle_tolerance": ORACLE_TOLERANCE,
        "per_action": rows,
        "v7": {
            "actual": v7,
            "expected_score": EXPECTED_V7_SCORE,
            "absolute_score_delta": v7_delta,
            "passed": v7_pass,
        },
        "oracle": {
            "recomputed_point_statistics": oracle,
            "expected_score": EXPECTED_ORACLE_SCORE,
            "absolute_score_delta": oracle_delta,
            "passed": oracle_pass,
            "selection_counts": {
                action_id: int(np.sum(selection == index))
                for index, action_id in enumerate(action_ids)
            },
            "truth_energy_identity_max_abs": float(
                max(
                    np.max(
                        np.abs(
                            metrics[action_id].truth_energy
                            - selected_stats.truth_energy
                        )
                    )
                    for action_id in action_ids
                )
            ),
        },
        "gate": {
            "all_13_actions_pass": bool(all_action_pass),
            "v7_pass": v7_pass,
            "oracle_pass": oracle_pass,
            "overall_pass": overall_pass,
        },
        "performance": {
            **timing,
            "wall_seconds": elapsed,
            "device": device_report(device),
            "peak_torch_bytes": peak_torch_bytes,
            "peak_rss_gib": peak_rss_gib,
            "measured_bottleneck": max(
                (
                    ("host_action_io", timing["host_action_io_seconds"]),
                    ("host_to_device", timing["host_to_device_seconds"]),
                    ("gpu_metric", timing["gpu_metric_seconds"]),
                ),
                key=lambda item: item[1],
            )[0],
        },
        "outputs": {
            "point_metrics": PATHS.relative(IDENTITY_METRICS),
            "action_csv": PATHS.relative(IDENTITY_CSV),
            "report": PATHS.relative(IDENTITY_REPORT),
        },
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(IDENTITY_CSV, rows)
    IDENTITY_REPORT.write_text(
        json.dumps(
            json_compatible(report), ensure_ascii=False, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.phase == "identity":
        report = run_identity(args)
    else:
        raise ValueError(args.phase)
    print(
        json.dumps(
            json_compatible(report), ensure_ascii=False, indent=2, allow_nan=False
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
