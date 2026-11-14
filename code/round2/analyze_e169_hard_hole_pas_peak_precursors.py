#!/usr/bin/env python3
"""Decompose Clean V7 PAS errors in hard pseudo-holes and audit outside-H precursors."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr
from scipy.spatial import cKDTree

import analyze_e167_clean_test_hole_difficulty as e167
from common.official_metrics import (
    PointMetricStatistics,
    aggregate_statistics,
    point_metric_statistics,
)
from project_paths import PATHS


VERSION = "Round2-E169-hard-hole-pas-peak-precursors"
ARTIFACT_DIR = PATHS.artifacts / "round2" / "e169_hard_hole_pas_peak_precursors"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e169_hard_hole_pas_peak_precursors"
VIS_DIR = PATHS.analysis / "visualizations" / "复赛困难洞V7_PAS峰错误与洞外预兆"

REPORT_JSON = ANALYSIS_DIR / "hard_hole_pas_peak_precursors.json"
REPORT_MD = ANALYSIS_DIR / "困难pseudo洞V7_PAS峰错误与洞外预兆.md"
HOLE_CSV = ANALYSIS_DIR / "逐洞PAS峰错误分解.csv"
EVENT_CSV = ANALYSIS_DIR / "峰错误事件与洞外预兆.csv"
POINT_CSV = ANALYSIS_DIR / "逐点PAS表示与主错误.csv"
PRECURSOR_CSV = ANALYSIS_DIR / "洞外预兆与负控制汇总.csv"
TRAIN_COMPACT = ARTIFACT_DIR / "official_train_compact_pas.npy"
V7_COMPACT = ARTIFACT_DIR / "canonical_v7_compact_pas.npy"
EVENT_CACHE = ARTIFACT_DIR / "pas_peak_event_metrics.npz"

TRAIN_CHANNEL = e167.TRAIN_CHANNEL
V7_CHANNEL = e167.V7_COMPLETE
OOF_META = e167.OOF_META
HARD_HOLES = ("P_S1F3", "P_S1F1", "P_S1F2", "P_S1F4")

N_UE = 4
N_POL = 2
N_H = 16
N_V = 8
N_SC = 192
PEAK_MIN = 0.010
MAX_PEAKS = 8
MATCH_H = 2.0
MATCH_V = 1.0
MASS_ABS_TOL = 0.002
NEIGHBORS = 32
NEAR_COUNT = 8
FAR_START = 16
CONTROL_REPEATS = 20
TINY = 1e-30
EXPECTED_V7 = e167.EXPECTED_V7
IDENTITY_TOLERANCE = e167.IDENTITY_TOLERANCE
SEED = 20_260_811_169

ERROR_LABELS = {
    "missing_peak": "漏峰",
    "spurious_peak": "假峰",
    "angle_shift": "角度偏移",
    "power_mismatch": "功率不匹配",
    "correct_peak": "基本正确",
}


@dataclass(frozen=True)
class Peak:
    h: int
    v: int
    value: float
    mass: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--fft-batch-size", type=int, default=16)
    parser.add_argument("--metric-batch-size", type=int, default=8)
    parser.add_argument("--rebuild-compact", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def json_compatible(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_compatible(value.tolist())
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


def normalize_maps(value: np.ndarray) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = array.sum(axis=(-2, -1), keepdims=True, dtype=np.float64)
    return np.divide(
        array,
        np.maximum(total, TINY),
        out=np.zeros_like(array),
        where=total > 0.0,
    )


def build_compact_pas(
    channel: np.ndarray,
    output_path: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.empty((len(channel), N_UE, N_H, N_V), dtype=np.float32)
    for start in range(0, len(channel), batch_size):
        stop = min(start + batch_size, len(channel))
        value = torch.as_tensor(
            np.array(channel[start:stop], dtype=np.complex64, copy=True), device=device
        )
        with torch.inference_mode():
            shaped = value.reshape(len(value), N_POL, N_H, N_V, N_UE, N_SC)
            power = torch.fft.fft2(shaped, dim=(2, 3)).abs().square()
            compact = power.sum(dim=1).mean(dim=-1).permute(0, 3, 1, 2)
            compact = compact / compact.sum(dim=(2, 3), keepdim=True).clamp_min(TINY)
            output[start:stop] = compact.cpu().numpy()
        if stop == len(channel) or stop % 256 == 0:
            print(f"E169 compact PAS {output_path.name}: {stop}/{len(channel)}", flush=True)
    np.save(output_path, output)
    return output


def load_or_build_compact(
    channel: np.ndarray,
    path: Path,
    device: torch.device,
    batch_size: int,
    rebuild: bool,
) -> np.ndarray:
    if path.exists() and not rebuild:
        value = np.load(path, mmap_mode="r")
        expected = (len(channel), N_UE, N_H, N_V)
        if value.shape != expected or value.dtype != np.float32:
            raise ValueError(f"invalid compact cache {path}: {value.shape} {value.dtype}")
        return np.asarray(value)
    return build_compact_pas(channel, path, device, batch_size)


def official_v7_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    query_ids: np.ndarray,
    batch_size: int,
) -> tuple[PointMetricStatistics, dict[str, float | int], dict[str, float]]:
    point_pas = np.empty(len(query_ids), dtype=np.float64)
    point_pdp = np.empty(len(query_ids), dtype=np.float64)
    point_error = np.empty(len(query_ids), dtype=np.float64)
    point_energy = np.empty(len(query_ids), dtype=np.float64)
    for start in range(0, len(query_ids), batch_size):
        stop = min(start + batch_size, len(query_ids))
        stats = point_metric_statistics(
            np.asarray(prediction[start:stop], dtype=np.complex64),
            np.asarray(truth[query_ids[start:stop]], dtype=np.complex64),
        )
        point_pas[start:stop] = stats.pas
        point_pdp[start:stop] = stats.pdp
        point_error[start:stop] = stats.squared_error
        point_energy[start:stop] = stats.truth_energy
        print(f"E169 official metrics: {stop}/{len(query_ids)}", flush=True)
    stats = PointMetricStatistics(point_pas, point_pdp, point_error, point_energy)
    overall = aggregate_statistics(stats)
    errors = {
        key: abs(float(overall[key]) - float(expected))
        for key, expected in EXPECTED_V7.items()
    }
    if max(errors.values()) > IDENTITY_TOLERANCE:
        raise RuntimeError(f"V7 identity gate failed: {errors}")
    return stats, overall, errors


def compact_point_cosine(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    pred = prediction.reshape(len(prediction), N_UE, -1).astype(np.float64)
    target = truth.reshape(len(truth), N_UE, -1).astype(np.float64)
    numerator = np.sum(pred * target, axis=-1)
    denominator = np.sqrt(np.sum(pred * pred, axis=-1) * np.sum(target * target, axis=-1))
    denominator = np.where(denominator == 0.0, 1e-12, denominator)
    return np.mean(numerator / denominator, axis=1)


def periodic_distance(left: int, right: int, period: int) -> float:
    delta = abs(left - right)
    return float(min(delta, period - delta))


def periodic_signed(target: int, anchor: int, period: int) -> float:
    return float((target - anchor + period / 2.0) % period - period / 2.0)


def scaled_distance(left: Peak, right: Peak) -> tuple[float, float, float]:
    dh = periodic_distance(left.h, right.h, N_H)
    dv = periodic_distance(left.v, right.v, N_V)
    return math.sqrt((dh / MATCH_H) ** 2 + (dv / MATCH_V) ** 2), dh, dv


def angle_indices(center: int, radius: int, period: int) -> list[int]:
    return [(center + offset) % period for offset in range(-radius, radius + 1)]


def window_mass(value: np.ndarray, h: int, v: int) -> float:
    hi = angle_indices(h, 1, N_H)
    vi = angle_indices(v, 1, N_V)
    return float(value[np.ix_(hi, vi)].sum(dtype=np.float64))


def window_mass_batch(value: np.ndarray, h: int, v: int) -> np.ndarray:
    hi = angle_indices(h, 1, N_H)
    vi = angle_indices(v, 1, N_V)
    return value[:, hi][:, :, vi].sum(axis=(1, 2), dtype=np.float64)


def extract_peaks(value: np.ndarray) -> list[Peak]:
    array = np.asarray(value, dtype=np.float32)
    if float(array.sum(dtype=np.float64)) <= TINY:
        return []
    local = maximum_filter(array, size=(3, 3), mode="wrap")
    candidates = np.argwhere((array >= local - 1e-12) & (array >= PEAK_MIN))
    if len(candidates) == 0:
        candidates = np.asarray([np.unravel_index(int(np.argmax(array)), array.shape)])
    order = np.argsort([-float(array[tuple(index)]) for index in candidates])[:MAX_PEAKS]
    return [
        Peak(
            h=int(candidates[index][0]),
            v=int(candidates[index][1]),
            value=float(array[tuple(candidates[index])]),
            mass=window_mass(array, int(candidates[index][0]), int(candidates[index][1])),
        )
        for index in order
    ]


def match_peaks(prediction: np.ndarray, truth: np.ndarray) -> list[dict[str, object]]:
    pred_peaks = extract_peaks(prediction)
    truth_peaks = extract_peaks(truth)
    accepted: list[tuple[int, int, float, float, float]] = []
    if pred_peaks and truth_peaks:
        cost = np.empty((len(pred_peaks), len(truth_peaks)), dtype=np.float64)
        axes: dict[tuple[int, int], tuple[float, float]] = {}
        for pred_index, pred in enumerate(pred_peaks):
            for truth_index, target in enumerate(truth_peaks):
                distance, dh, dv = scaled_distance(pred, target)
                cost[pred_index, truth_index] = distance
                axes[(pred_index, truth_index)] = (dh, dv)
        rows, columns = linear_sum_assignment(cost)
        for pred_index, truth_index in zip(rows, columns, strict=True):
            if cost[pred_index, truth_index] <= 1.0:
                dh, dv = axes[(int(pred_index), int(truth_index))]
                accepted.append(
                    (int(pred_index), int(truth_index), float(cost[pred_index, truth_index]), dh, dv)
                )

    pred_matched = {item[0] for item in accepted}
    truth_matched = {item[1] for item in accepted}
    events: list[dict[str, object]] = []
    for pred_index, truth_index, distance, dh, dv in accepted:
        pred = pred_peaks[pred_index]
        target = truth_peaks[truth_index]
        tolerance = max(MASS_ABS_TOL, 0.05 * max(pred.mass, target.mass))
        if dh > 0.0 or dv > 0.0:
            event_type = "angle_shift"
            error_weight = min(pred.mass, target.mass)
        elif abs(pred.mass - target.mass) > tolerance:
            event_type = "power_mismatch"
            error_weight = abs(pred.mass - target.mass)
        else:
            event_type = "correct_peak"
            error_weight = min(pred.mass, target.mass)
        events.append(
            {
                "event_type": event_type,
                "pred_h": pred.h,
                "pred_v": pred.v,
                "truth_h": target.h,
                "truth_v": target.v,
                "pred_peak_value": pred.value,
                "truth_peak_value": target.value,
                "pred_mass": pred.mass,
                "truth_mass": target.mass,
                "mass_difference_truth_minus_v7": target.mass - pred.mass,
                "shift_h": periodic_signed(target.h, pred.h, N_H),
                "shift_v": periodic_signed(target.v, pred.v, N_V),
                "scaled_shift": distance,
                "error_weight": error_weight,
            }
        )
    for truth_index, target in enumerate(truth_peaks):
        if truth_index in truth_matched:
            continue
        events.append(
            {
                "event_type": "missing_peak",
                "pred_h": -1,
                "pred_v": -1,
                "truth_h": target.h,
                "truth_v": target.v,
                "pred_peak_value": 0.0,
                "truth_peak_value": target.value,
                "pred_mass": 0.0,
                "truth_mass": target.mass,
                "mass_difference_truth_minus_v7": target.mass,
                "shift_h": float("nan"),
                "shift_v": float("nan"),
                "scaled_shift": float("nan"),
                "error_weight": target.mass,
            }
        )
    for pred_index, pred in enumerate(pred_peaks):
        if pred_index in pred_matched:
            continue
        events.append(
            {
                "event_type": "spurious_peak",
                "pred_h": pred.h,
                "pred_v": pred.v,
                "truth_h": -1,
                "truth_v": -1,
                "pred_peak_value": pred.value,
                "truth_peak_value": 0.0,
                "pred_mass": pred.mass,
                "truth_mass": 0.0,
                "mass_difference_truth_minus_v7": -pred.mass,
                "shift_h": float("nan"),
                "shift_v": float("nan"),
                "scaled_shift": float("nan"),
                "error_weight": pred.mass,
            }
        )
    return events


def build_peak_mask(compact: np.ndarray) -> np.ndarray:
    output = np.zeros_like(compact, dtype=bool)
    for ue in range(N_UE):
        local = maximum_filter(compact[:, ue], size=(1, 3, 3), mode="wrap")
        output[:, ue] = (compact[:, ue] >= local - 1e-12) & (compact[:, ue] >= PEAK_MIN)
    return output


def support_near_coordinate(mask: np.ndarray, h: int, v: int) -> np.ndarray:
    hi = angle_indices(h, int(MATCH_H), N_H)
    vi = angle_indices(v, int(MATCH_V), N_V)
    return mask[:, hi][:, :, vi].any(axis=(1, 2)).astype(np.float64)


def local_peak_offsets(
    maps: np.ndarray,
    masks: np.ndarray,
    anchor_h: int,
    anchor_v: int,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.full((len(maps), 2), np.nan, dtype=np.float64)
    valid = np.zeros(len(maps), dtype=bool)
    h_offsets = range(-3, 4)
    v_offsets = range(-2, 3)
    for row in range(len(maps)):
        best: tuple[float, int, int] | None = None
        for dh in h_offsets:
            for dv in v_offsets:
                h = (anchor_h + dh) % N_H
                v = (anchor_v + dv) % N_V
                if not masks[row, h, v]:
                    continue
                candidate = (float(maps[row, h, v]), dh, dv)
                if best is None or candidate[0] > best[0]:
                    best = candidate
        if best is not None:
            valid[row] = True
            offsets[row] = (best[1], best[2])
    return offsets, valid


def fit_local_affine(
    donor_xy: np.ndarray,
    query_xy: np.ndarray,
    values: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    response = np.asarray(values, dtype=np.float64)
    valid = np.all(np.isfinite(response), axis=1) if response.ndim == 2 else np.isfinite(response)
    if int(np.sum(valid)) < 6:
        width = response.shape[1] if response.ndim == 2 else 1
        return np.full(width, np.nan, dtype=np.float64)
    delta = (donor_xy[valid] - query_xy[None]) / 20.0
    design = np.column_stack([np.ones(np.sum(valid)), delta])
    weight = 1.0 / np.square(distances[valid] + 2.0)
    target = response[valid]
    if target.ndim == 1:
        target = target[:, None]
    gram = design.T @ (weight[:, None] * design)
    ridge = np.diag([1e-5, 2e-3, 2e-3])
    rhs = design.T @ (weight[:, None] * target)
    coefficient = np.linalg.solve(gram + ridge, rhs)
    return coefficient[0]


def angle_control_coordinates(h: int, v: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    while len(output) < CONTROL_REPEATS:
        dh = int(rng.integers(3, N_H - 2))
        if bool(rng.integers(0, 2)):
            dh = -dh
        dv = int(rng.integers(-N_V // 2, N_V // 2 + 1))
        candidate = ((h + dh) % N_H, (v + dv) % N_V)
        if periodic_distance(candidate[0], h, N_H) <= MATCH_H and periodic_distance(candidate[1], v, N_V) <= MATCH_V:
            continue
        output.append(candidate)
    return output


def add_precursor_features(
    event: dict[str, object],
    train_maps: np.ndarray,
    train_peak_mask: np.ndarray,
    donor_ids: np.ndarray,
    donor_distances: np.ndarray,
    train_xy: np.ndarray,
    query_xy: np.ndarray,
    rng: np.random.Generator,
) -> None:
    ue = int(event["ue"])
    maps = train_maps[donor_ids, ue]
    masks = train_peak_mask[donor_ids, ue]
    donor_xy = train_xy[donor_ids]
    event_type = str(event["event_type"])
    if event_type == "missing_peak":
        anchor_h, anchor_v = int(event["truth_h"]), int(event["truth_v"])
    else:
        anchor_h, anchor_v = int(event["pred_h"]), int(event["pred_v"])
    masses = window_mass_batch(maps, anchor_h, anchor_v)
    support = support_near_coordinate(masks, anchor_h, anchor_v)
    raw_delta = float(np.mean(masses[:NEAR_COUNT]) - np.mean(masses[FAR_START:]))
    support_delta = float(np.mean(support[:NEAR_COUNT]) - np.mean(support[FAR_START:]))
    position_controls = np.empty(CONTROL_REPEATS, dtype=np.float64)
    support_controls = np.empty(CONTROL_REPEATS, dtype=np.float64)
    for index in range(CONTROL_REPEATS):
        order = rng.permutation(len(masses))
        shuffled_mass = masses[order]
        shuffled_support = support[order]
        position_controls[index] = float(
            np.mean(shuffled_mass[:NEAR_COUNT]) - np.mean(shuffled_mass[FAR_START:])
        )
        support_controls[index] = float(
            np.mean(shuffled_support[:NEAR_COUNT]) - np.mean(shuffled_support[FAR_START:])
        )
    angle_controls = []
    for control_h, control_v in angle_control_coordinates(anchor_h, anchor_v, rng):
        control_mass = window_mass_batch(maps, control_h, control_v)
        angle_controls.append(float(np.mean(control_mass[:NEAR_COUNT]) - np.mean(control_mass[FAR_START:])))

    event.update(
        {
            "outside_near_mass": float(np.mean(masses[:NEAR_COUNT])),
            "outside_far_mass": float(np.mean(masses[FAR_START:])),
            "outside_near_minus_far_mass": raw_delta,
            "outside_near_support": float(np.mean(support[:NEAR_COUNT])),
            "outside_far_support": float(np.mean(support[FAR_START:])),
            "outside_near_minus_far_support": support_delta,
            "position_shuffle_mass_delta_mean": float(np.mean(position_controls)),
            "position_shuffle_mass_delta_q90": float(np.quantile(position_controls, 0.90)),
            "position_shuffle_support_delta_mean": float(np.mean(support_controls)),
            "angle_control_mass_delta_mean": float(np.mean(angle_controls)),
            "angle_control_mass_delta_q90": float(np.quantile(angle_controls, 0.90)),
        }
    )

    if event_type in {"missing_peak", "spurious_peak"}:
        sign = 1.0 if event_type == "missing_peak" else -1.0
        precursor = sign * raw_delta
        control_values = sign * position_controls
        angle_values = sign * np.asarray(angle_controls)
        event.update(
            {
                "lifecycle_precursor_score": precursor,
                "lifecycle_position_control_mean": float(np.mean(control_values)),
                "lifecycle_angle_control_mean": float(np.mean(angle_values)),
                "lifecycle_direction_success": bool(precursor > 0.0),
                "lifecycle_position_control_success_rate": float(np.mean(control_values > 0.0)),
                "lifecycle_angle_control_success_rate": float(np.mean(angle_values > 0.0)),
                "strong_lifecycle_precursor": bool(
                    precursor > float(np.quantile(control_values, 0.90))
                    and precursor > float(np.quantile(angle_values, 0.90))
                ),
            }
        )

    if event_type == "angle_shift":
        offsets, valid = local_peak_offsets(maps, masks, anchor_h, anchor_v)
        estimate = fit_local_affine(donor_xy, query_xy, offsets, donor_distances)
        truth_offset = np.asarray([event["shift_h"], event["shift_v"]], dtype=np.float64)
        baseline_distance = math.sqrt((truth_offset[0] / MATCH_H) ** 2 + (truth_offset[1] / MATCH_V) ** 2)
        if np.all(np.isfinite(estimate)):
            estimate = np.clip(estimate, [-MATCH_H, -MATCH_V], [MATCH_H, MATCH_V])
            predicted_distance = math.sqrt(
                ((truth_offset[0] - estimate[0]) / MATCH_H) ** 2
                + ((truth_offset[1] - estimate[1]) / MATCH_V) ** 2
            )
            improvement = baseline_distance - predicted_distance
        else:
            improvement = float("nan")
        controls = []
        for _ in range(CONTROL_REPEATS):
            shuffled = offsets.copy()
            shuffled[valid] = shuffled[valid][rng.permutation(np.sum(valid))]
            control_estimate = fit_local_affine(donor_xy, query_xy, shuffled, donor_distances)
            if np.all(np.isfinite(control_estimate)):
                control_estimate = np.clip(
                    control_estimate, [-MATCH_H, -MATCH_V], [MATCH_H, MATCH_V]
                )
                control_distance = math.sqrt(
                    ((truth_offset[0] - control_estimate[0]) / MATCH_H) ** 2
                    + ((truth_offset[1] - control_estimate[1]) / MATCH_V) ** 2
                )
                controls.append(baseline_distance - control_distance)
        event.update(
            {
                "shift_neighbor_peak_count": int(np.sum(valid)),
                "shift_predicted_h": float(estimate[0]) if len(estimate) > 0 else float("nan"),
                "shift_predicted_v": float(estimate[1]) if len(estimate) > 1 else float("nan"),
                "shift_distance_improvement": improvement,
                "shift_direction_success": bool(np.isfinite(improvement) and improvement > 0.0),
                "shift_control_improvement_mean": float(np.mean(controls)) if controls else float("nan"),
                "shift_control_success_rate": float(np.mean(np.asarray(controls) > 0.0)) if controls else float("nan"),
            }
        )

    if event_type == "power_mismatch":
        estimate = fit_local_affine(donor_xy, query_xy, masses, donor_distances)
        target = float(event["truth_mass"])
        baseline = abs(float(event["pred_mass"]) - target)
        predicted = float(np.clip(estimate[0], 0.0, 1.0)) if np.isfinite(estimate[0]) else float("nan")
        improvement = baseline - abs(predicted - target) if np.isfinite(predicted) else float("nan")
        controls = []
        for _ in range(CONTROL_REPEATS):
            shuffled = masses[rng.permutation(len(masses))]
            control_estimate = fit_local_affine(donor_xy, query_xy, shuffled, donor_distances)
            if np.isfinite(control_estimate[0]):
                control_predicted = float(np.clip(control_estimate[0], 0.0, 1.0))
                controls.append(baseline - abs(control_predicted - target))
        event.update(
            {
                "power_predicted_mass": predicted,
                "power_error_reduction": improvement,
                "power_direction_success": bool(np.isfinite(improvement) and improvement > 0.0),
                "power_control_error_reduction_mean": float(np.mean(controls)) if controls else float("nan"),
                "power_control_success_rate": float(np.mean(np.asarray(controls) > 0.0)) if controls else float("nan"),
            }
        )


def finite_values(rows: list[dict[str, object]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key, float("nan"))
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return np.asarray(values, dtype=np.float64)


def summarize_event_subset(events: list[dict[str, object]]) -> dict[str, object]:
    error_events = [row for row in events if row["event_type"] != "correct_peak"]
    total_weight = sum(float(row["error_weight"]) for row in error_events)
    output: dict[str, object] = {
        "event_count": len(events),
        "error_event_count": len(error_events),
        "error_weight_sum": total_weight,
    }
    for event_type in ERROR_LABELS:
        selected = [row for row in events if row["event_type"] == event_type]
        weight = sum(float(row["error_weight"]) for row in selected)
        output[f"{event_type}_count"] = len(selected)
        output[f"{event_type}_weight"] = weight
        output[f"{event_type}_error_share"] = (
            weight / total_weight if event_type != "correct_peak" and total_weight > 0.0 else 0.0
        )

    for event_type, prefix in (("missing_peak", "birth"), ("spurious_peak", "death")):
        selected = [row for row in events if row["event_type"] == event_type]
        score = finite_values(selected, "lifecycle_precursor_score")
        position = finite_values(selected, "lifecycle_position_control_mean")
        angle = finite_values(selected, "lifecycle_angle_control_mean")
        position_success = finite_values(selected, "lifecycle_position_control_success_rate")
        angle_success = finite_values(selected, "lifecycle_angle_control_success_rate")
        output[f"{prefix}_event_count"] = len(score)
        output[f"{prefix}_direction_success_rate"] = float(np.mean(score > 0.0)) if len(score) else float("nan")
        output[f"{prefix}_position_control_success_rate"] = float(np.mean(position_success)) if len(position_success) else float("nan")
        output[f"{prefix}_angle_control_success_rate"] = float(np.mean(angle_success)) if len(angle_success) else float("nan")
        output[f"{prefix}_success_margin_vs_position"] = (
            float(np.mean(score > 0.0) - np.mean(position_success)) if len(score) and len(position_success) else float("nan")
        )
        output[f"{prefix}_median_precursor"] = float(np.median(score)) if len(score) else float("nan")
        output[f"{prefix}_median_margin_vs_position"] = (
            float(np.median(score - position)) if len(score) == len(position) and len(score) else float("nan")
        )
        output[f"{prefix}_median_margin_vs_angle"] = (
            float(np.median(score - angle)) if len(score) == len(angle) and len(score) else float("nan")
        )
        strong = [bool(row.get("strong_lifecycle_precursor", False)) for row in selected]
        output[f"{prefix}_strong_precursor_rate"] = float(np.mean(strong)) if strong else float("nan")

    shifted = [row for row in events if row["event_type"] == "angle_shift"]
    shift_improvement = finite_values(shifted, "shift_distance_improvement")
    shift_control = finite_values(shifted, "shift_control_improvement_mean")
    shift_control_success = finite_values(shifted, "shift_control_success_rate")
    output["shift_evaluable_count"] = len(shift_improvement)
    output["shift_success_rate"] = float(np.mean(shift_improvement > 0.0)) if len(shift_improvement) else float("nan")
    output["shift_control_success_rate"] = float(np.mean(shift_control_success)) if len(shift_control_success) else float("nan")
    output["shift_success_margin"] = (
        float(np.mean(shift_improvement > 0.0) - np.mean(shift_control_success))
        if len(shift_improvement) and len(shift_control_success)
        else float("nan")
    )
    output["shift_mean_distance_improvement"] = float(np.mean(shift_improvement)) if len(shift_improvement) else float("nan")
    output["shift_mean_margin_vs_control"] = (
        float(np.mean(shift_improvement - shift_control))
        if len(shift_improvement) == len(shift_control) and len(shift_improvement)
        else float("nan")
    )

    powered = [row for row in events if row["event_type"] == "power_mismatch"]
    power_improvement = finite_values(powered, "power_error_reduction")
    power_control = finite_values(powered, "power_control_error_reduction_mean")
    power_control_success = finite_values(powered, "power_control_success_rate")
    output["power_evaluable_count"] = len(power_improvement)
    output["power_success_rate"] = float(np.mean(power_improvement > 0.0)) if len(power_improvement) else float("nan")
    output["power_control_success_rate"] = float(np.mean(power_control_success)) if len(power_control_success) else float("nan")
    output["power_success_margin"] = (
        float(np.mean(power_improvement > 0.0) - np.mean(power_control_success))
        if len(power_improvement) and len(power_control_success)
        else float("nan")
    )
    output["power_mean_error_reduction"] = float(np.mean(power_improvement)) if len(power_improvement) else float("nan")
    output["power_mean_margin_vs_control"] = (
        float(np.mean(power_improvement - power_control))
        if len(power_improvement) == len(power_control) and len(power_improvement)
        else float("nan")
    )
    return output


def mechanism_gate(
    hard_summary: dict[str, object],
    per_hole: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    specifications = [
        (
            "birth",
            "missing_peak_error_share",
            "birth_success_margin_vs_position",
            "birth_median_margin_vs_position",
        ),
        (
            "death",
            "spurious_peak_error_share",
            "death_success_margin_vs_position",
            "death_median_margin_vs_position",
        ),
        (
            "angle_shift",
            "angle_shift_error_share",
            "shift_success_margin",
            "shift_mean_margin_vs_control",
        ),
        (
            "power_mismatch",
            "power_mismatch_error_share",
            "power_success_margin",
            "power_mean_margin_vs_control",
        ),
    ]
    for mechanism, share_key, success_key, margin_key in specifications:
        share = float(hard_summary.get(share_key, 0.0))
        success_margin = float(hard_summary.get(success_key, float("nan")))
        value_margin = float(hard_summary.get(margin_key, float("nan")))
        positive_holes = 0
        eligible_holes = 0
        for row in per_hole:
            if not bool(row["is_hard_hole"]):
                continue
            count_key = {
                "birth": "birth_event_count",
                "death": "death_event_count",
                "angle_shift": "shift_evaluable_count",
                "power_mismatch": "power_evaluable_count",
            }[mechanism]
            if int(row.get(count_key, 0)) < 3:
                continue
            eligible_holes += 1
            hole_margin = float(row.get(margin_key, float("nan")))
            if math.isfinite(hole_margin) and hole_margin > 0.0:
                positive_holes += 1
        signal_gate = (
            (math.isfinite(success_margin) and success_margin >= 0.10)
            or (math.isfinite(value_margin) and value_margin >= 0.01)
        )
        passed = share >= 0.20 and signal_gate and positive_holes >= 3
        rows.append(
            {
                "mechanism": mechanism,
                "hard_error_share": share,
                "success_rate_margin_vs_control": success_margin,
                "value_margin_vs_control": value_margin,
                "positive_hard_holes": positive_holes,
                "eligible_hard_holes": eligible_holes,
                "gate_passed": passed,
            }
        )
    return rows


def plot_error_composition(rows: list[dict[str, object]]) -> Path:
    selected = sorted(rows, key=lambda row: float(row["official_pas"]))
    labels = [str(row["pseudo_hole_id"]) for row in selected]
    keys = ("missing_peak", "spurious_peak", "angle_shift", "power_mismatch")
    colors = ("#d1495b", "#edae49", "#30638e", "#2a9d8f")
    fig, ax = plt.subplots(figsize=(12, 6.5))
    bottom = np.zeros(len(selected), dtype=np.float64)
    for key, color in zip(keys, colors, strict=True):
        values = np.asarray([float(row[f"{key}_error_share"]) for row in selected])
        ax.bar(labels, values, bottom=bottom, label=ERROR_LABELS[key], color=color)
        bottom += values
    for index, row in enumerate(selected):
        ax.text(index, 1.02, f"PAS={float(row['official_pas']):.3f}", ha="center", va="bottom", fontsize=9, rotation=35)
    ax.set_ylim(0.0, 1.14)
    ax.set_ylabel("峰错误事件质量占比")
    ax.set_title("十个pseudo-hole的V7 PAS峰错误构成")
    ax.legend(ncol=4, loc="upper center")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = VIS_DIR / "01_十洞V7_PAS峰错误构成.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_precursors(hard: dict[str, object]) -> Path:
    labels = ["漏峰出生", "假峰消失", "角度漂移", "功率变化"]
    actual = [
        float(hard["birth_direction_success_rate"]),
        float(hard["death_direction_success_rate"]),
        float(hard["shift_success_rate"]),
        float(hard["power_success_rate"]),
    ]
    control = [
        float(hard["birth_position_control_success_rate"]),
        float(hard["death_position_control_success_rate"]),
        float(hard["shift_control_success_rate"]),
        float(hard["power_control_success_rate"]),
    ]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.bar(x - width / 2, actual, width, label="真实洞外空间顺序", color="#2a9d8f")
    ax.bar(x + width / 2, control, width, label="邻居位置打乱控制", color="#9aa0a6")
    ax.axhline(0.5, color="#444444", linewidth=1, linestyle="--")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("方向成功率")
    ax.set_title("四个最难pseudo-hole：洞外Train H预兆是否优于控制")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = VIS_DIR / "02_困难洞外TrainH预兆与位置打乱控制.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_examples(
    events: list[dict[str, object]],
    v7_maps: np.ndarray,
    truth_maps: np.ndarray,
    train_maps: np.ndarray,
    neighbor_ids: np.ndarray,
) -> Path:
    chosen = []
    for hole_id in HARD_HOLES:
        candidates = [
            row for row in events if row["pseudo_hole_id"] == hole_id and row["event_type"] != "correct_peak"
        ]
        chosen.append(max(candidates, key=lambda row: float(row["error_weight"])))
    fig, axes = plt.subplots(len(chosen), 3, figsize=(11.5, 13.5), constrained_layout=True)
    for row_index, event in enumerate(chosen):
        oof_row = int(event["oof_row"])
        ue = int(event["ue"])
        donors = neighbor_ids[oof_row]
        near = np.mean(train_maps[donors[:NEAR_COUNT], ue], axis=0)
        far = np.mean(train_maps[donors[FAR_START:], ue], axis=0)
        maps = (v7_maps[oof_row, ue], truth_maps[oof_row, ue], near - far)
        titles = ("V7", "隐藏真值", "洞外近邻 - 远邻")
        limits = ((0.0, max(float(np.max(maps[0])), float(np.max(maps[1])))),) * 2 + ((-float(np.max(np.abs(maps[2]))), float(np.max(np.abs(maps[2])))),)
        for column, (value, title, limit) in enumerate(zip(maps, titles, limits, strict=True)):
            cmap = "magma" if column < 2 else "coolwarm"
            image = axes[row_index, column].imshow(value.T, origin="lower", aspect="auto", cmap=cmap, vmin=limit[0], vmax=max(limit[1], 1e-6))
            axes[row_index, column].set_title(f"{event['pseudo_hole_id']} UE{ue} {title}")
            axes[row_index, column].set_xlabel("水平角格")
            axes[row_index, column].set_ylabel("垂直角格")
            if int(event["pred_h"]) >= 0:
                axes[row_index, column].scatter(event["pred_h"], event["pred_v"], marker="x", color="#00ffff", s=60, linewidths=2)
            if int(event["truth_h"]) >= 0:
                axes[row_index, column].scatter(event["truth_h"], event["truth_v"], marker="o", facecolors="none", edgecolors="#39ff14", s=70, linewidths=2)
            fig.colorbar(image, ax=axes[row_index, column], fraction=0.046)
        axes[row_index, 0].text(
            0.02,
            0.96,
            ERROR_LABELS[str(event["event_type"])],
            transform=axes[row_index, 0].transAxes,
            va="top",
            color="white",
            fontsize=10,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 3},
        )
    path = VIS_DIR / "03_四个困难洞典型PAS峰错误.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def markdown_report(report: dict[str, object], hole_rows: list[dict[str, object]], gate_rows: list[dict[str, object]], figures: list[Path]) -> None:
    hard = report["hard_hole_summary"]
    lines = [
        "# 困难pseudo-hole的V7 PAS峰错误与洞外预兆",
        "",
        "## 固定背景",
        "",
        f"Clean V7固定500点官方PAS/PDP/NMSE/总分=`{report['official_v7']['pas']:.6f}/{report['official_v7']['pdp']:.6f}/{report['official_v7']['nmse']:.6f}/{report['official_v7']['score']:.6f}`。",
        f"峰图代理与逐点官方PAS的Pearson/Spearman=`{report['representation_gate']['pearson']:.6f}/{report['representation_gate']['spearman']:.6f}`，困难洞保留数=`{report['representation_gate']['hard_holes_in_bottom_five']}/4`，表示门=`{report['representation_gate']['passed']}`。",
        "",
        "峰错误使用每点每UE的频率平均、极化合并`16x8`角域图；它用于解释方向结构，官方PAS仍按项目正式实现计算。出生/消失是漏峰/假峰的空间机制子类，不重复计入错误质量。",
        "",
        "## 逐洞错误分解",
        "",
        "| pseudo-hole | 官方PAS | 漏峰 | 假峰 | 角度偏移 | 功率不匹配 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(hole_rows, key=lambda item: float(item["official_pas"])):
        lines.append(
            f"| {row['pseudo_hole_id']} | {row['official_pas']:.6f} | "
            f"{row['missing_peak_error_share']:.1%} | {row['spurious_peak_error_share']:.1%} | "
            f"{row['angle_shift_error_share']:.1%} | {row['power_mismatch_error_share']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## 四个最难洞的洞外预兆",
            "",
            "| 机制 | 真实方向成功率 | 位置打乱控制 | 成功率margin | 数值margin |",
            "|---|---:|---:|---:|---:|",
            f"| 漏峰出生 | {hard['birth_direction_success_rate']:.1%} | {hard['birth_position_control_success_rate']:.1%} | {hard['birth_success_margin_vs_position']:+.1%} | {hard['birth_median_margin_vs_position']:+.4f} |",
            f"| 假峰消失 | {hard['death_direction_success_rate']:.1%} | {hard['death_position_control_success_rate']:.1%} | {hard['death_success_margin_vs_position']:+.1%} | {hard['death_median_margin_vs_position']:+.4f} |",
            f"| 角度漂移 | {hard['shift_success_rate']:.1%} | {hard['shift_control_success_rate']:.1%} | {hard['shift_success_margin']:+.1%} | {hard['shift_mean_margin_vs_control']:+.4f} |",
            f"| 功率变化 | {hard['power_success_rate']:.1%} | {hard['power_control_success_rate']:.1%} | {hard['power_success_margin']:+.1%} | {hard['power_mean_margin_vs_control']:+.4f} |",
            "",
            "## 预注册机制门",
            "",
            "| 机制 | 困难洞错误质量占比 | 成功率margin | 数值margin | 正向困难洞 | 是否通过 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in gate_rows:
        lines.append(
            f"| {row['mechanism']} | {row['hard_error_share']:.1%} | "
            f"{row['success_rate_margin_vs_control']:+.1%} | {row['value_margin_vs_control']:+.4f} | "
            f"{row['positive_hard_holes']}/{row['eligible_hard_holes']} | {row['gate_passed']} |"
        )
    lines.extend(["", "## 图", ""])
    for path in figures:
        relative = Path("../../visualizations/复赛困难洞V7_PAS峰错误与洞外预兆") / path.name
        lines.append(f"![{path.stem}]({relative.as_posix()})")
        lines.append("")
    lines.extend(
        [
            "## 结论",
            "",
            str(report["conclusion"]),
            "",
            "本轮只做Clean诊断，没有训练模型、修改V7、生成测试信道或提交包。",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    e167.configure_chinese_font()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        gpu_name = torch.cuda.get_device_name(device)
    else:
        gpu_name = "CPU"

    _, pseudo_holes, context = e167.build_holes()
    pseudo_by_id = {hole.hole_id: hole for hole in pseudo_holes}
    metadata = np.load(OOF_META, allow_pickle=False)
    query_ids = metadata["query_ids"].astype(np.int64)
    fold_id = metadata["fold_id"].astype(np.int8)
    train_positions = np.asarray(context["train_positions"], dtype=np.float64)
    train_xy = train_positions[:, :2]
    truth_channel = np.load(TRAIN_CHANNEL, mmap_mode="r")
    v7_channel = np.load(V7_CHANNEL, mmap_mode="r")
    if v7_channel.shape != (500, 256, 4, 192):
        raise ValueError(f"unexpected V7 channel shape: {v7_channel.shape}")

    train_compact = load_or_build_compact(
        truth_channel, TRAIN_COMPACT, device, args.fft_batch_size, args.rebuild_compact
    )
    v7_compact = load_or_build_compact(
        v7_channel, V7_COMPACT, device, args.fft_batch_size, args.rebuild_compact
    )
    truth_compact = np.asarray(train_compact[query_ids], dtype=np.float32)
    official_stats, official_overall, identity_errors = official_v7_metrics(
        v7_channel, truth_channel, query_ids, args.metric_batch_size
    )
    compact_cosine = compact_point_cosine(v7_compact, truth_compact)
    representation_pearson = float(pearsonr(compact_cosine, official_stats.pas).statistic)
    representation_spearman = float(spearmanr(compact_cosine, official_stats.pas).statistic)

    global_to_oof = np.full(len(train_positions), -1, dtype=np.int64)
    global_to_oof[query_ids] = np.arange(len(query_ids), dtype=np.int64)
    hole_for_oof = np.empty(len(query_ids), dtype=object)
    neighbor_ids = np.empty((len(query_ids), NEIGHBORS), dtype=np.int64)
    neighbor_distances = np.empty((len(query_ids), NEIGHBORS), dtype=np.float64)
    for hole in pseudo_holes:
        rows = global_to_oof[hole.point_ids]
        if np.any(rows < 0) or np.any(fold_id[rows] != hole.ordinal):
            raise RuntimeError(f"canonical hole identity mismatch: {hole.hole_id}")
        hole_for_oof[rows] = hole.hole_id
        tree = cKDTree(train_xy[hole.fit_ids])
        distance, local = tree.query(train_xy[hole.point_ids], k=NEIGHBORS)
        neighbor_ids[rows] = hole.fit_ids[local]
        neighbor_distances[rows] = distance

    hole_official_pas: dict[str, float] = {}
    hole_compact_pas: dict[str, float] = {}
    for hole in pseudo_holes:
        rows = global_to_oof[hole.point_ids]
        hole_official_pas[hole.hole_id] = float(np.mean(official_stats.pas[rows]))
        hole_compact_pas[hole.hole_id] = float(np.mean(compact_cosine[rows]))
    bottom_five = {
        key for key, _ in sorted(hole_compact_pas.items(), key=lambda item: item[1])[:5]
    }
    hard_retained = sum(hole_id in bottom_five for hole_id in HARD_HOLES)
    representation_gate = {
        "pearson": representation_pearson,
        "spearman": representation_spearman,
        "hard_holes_in_bottom_five": hard_retained,
        "passed": bool(representation_pearson >= 0.70 and hard_retained >= 3),
    }
    if not representation_gate["passed"]:
        report = {
            "experiment": VERSION,
            "status": "representation_gate_failed",
            "official_v7": official_overall,
            "identity_errors": identity_errors,
            "representation_gate": representation_gate,
            "seconds": time.perf_counter() - started,
        }
        REPORT_JSON.write_text(
            json.dumps(json_compatible(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"compact PAS representation gate failed: {representation_gate}")

    train_peak_mask = build_peak_mask(train_compact)
    rng = np.random.default_rng(SEED)
    events: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    for oof_row, global_id in enumerate(query_ids):
        hole_id = str(hole_for_oof[oof_row])
        per_point_weights = {key: 0.0 for key in ERROR_LABELS}
        point_event_count = 0
        for ue in range(N_UE):
            local_events = match_peaks(v7_compact[oof_row, ue], truth_compact[oof_row, ue])
            for event in local_events:
                event.update(
                    {
                        "event_id": len(events),
                        "pseudo_hole_id": hole_id,
                        "is_hard_hole": hole_id in HARD_HOLES,
                        "fold": int(fold_id[oof_row]),
                        "site": int(pseudo_by_id[hole_id].site),
                        "oof_row": oof_row,
                        "global_train_id": int(global_id),
                        "ue": ue,
                        "official_point_pas": float(official_stats.pas[oof_row]),
                        "compact_point_pas": float(compact_cosine[oof_row]),
                        "nearest_outside_train_m": float(neighbor_distances[oof_row, 0]),
                        "far_group_mean_distance_m": float(np.mean(neighbor_distances[oof_row, FAR_START:])),
                    }
                )
                if event["event_type"] != "correct_peak":
                    add_precursor_features(
                        event,
                        train_compact,
                        train_peak_mask,
                        neighbor_ids[oof_row],
                        neighbor_distances[oof_row],
                        train_xy,
                        train_xy[global_id],
                        rng,
                    )
                per_point_weights[str(event["event_type"])] += float(event["error_weight"])
                point_event_count += 1
                events.append(event)
        error_only = {key: value for key, value in per_point_weights.items() if key != "correct_peak"}
        dominant = max(error_only, key=error_only.get) if sum(error_only.values()) > 0.0 else "correct_peak"
        point_rows.append(
            {
                "pseudo_hole_id": hole_id,
                "is_hard_hole": hole_id in HARD_HOLES,
                "fold": int(fold_id[oof_row]),
                "site": int(pseudo_by_id[hole_id].site),
                "oof_row": oof_row,
                "global_train_id": int(global_id),
                "official_pas": float(official_stats.pas[oof_row]),
                "compact_pas": float(compact_cosine[oof_row]),
                "event_count": point_event_count,
                "dominant_error": dominant,
                **{f"{key}_weight": value for key, value in per_point_weights.items()},
            }
        )
        if (oof_row + 1) % 50 == 0:
            print(f"E169 peak events: {oof_row + 1}/500", flush=True)

    hole_rows: list[dict[str, object]] = []
    for hole in pseudo_holes:
        selected = [row for row in events if row["pseudo_hole_id"] == hole.hole_id]
        summary = summarize_event_subset(selected)
        hole_rows.append(
            {
                "pseudo_hole_id": hole.hole_id,
                "is_hard_hole": hole.hole_id in HARD_HOLES,
                "site": hole.site,
                "fold": hole.ordinal,
                "point_count": len(hole.point_ids),
                "official_pas": hole_official_pas[hole.hole_id],
                "compact_pas": hole_compact_pas[hole.hole_id],
                **summary,
            }
        )
    hard_events = [row for row in events if bool(row["is_hard_hole"])]
    other_events = [row for row in events if not bool(row["is_hard_hole"])]
    hard_summary = summarize_event_subset(hard_events)
    other_summary = summarize_event_subset(other_events)
    gate_rows = mechanism_gate(hard_summary, hole_rows)

    precursor_rows = []
    for label, summary in (("四个最难洞", hard_summary), ("其余六洞", other_summary)):
        precursor_rows.append({"group": label, **summary})
    for row in gate_rows:
        precursor_rows.append({"group": "预注册机制门", **row})

    figures = [
        plot_error_composition(hole_rows),
        plot_precursors(hard_summary),
        plot_examples(events, v7_compact, truth_compact, train_compact, neighbor_ids),
    ]
    passed = [str(row["mechanism"]) for row in gate_rows if bool(row["gate_passed"])]
    if passed:
        conclusion = (
            "通过预注册门的洞外预兆机制为："
            + "、".join(passed)
            + "。这表示对应峰错误不仅在困难洞占比足够大，而且洞外官方Train H的空间变化方向明显优于保持边际控制。"
            "但四个困难洞最大的错误仍是假峰，错误质量占比为"
            f"{hard_summary['spurious_peak_error_share']:.1%}，其消失预兆没有过门；"
            "漏峰出生审计又是在隐藏真值峰角度上检查洞外趋势，因此它只证明正确角度附近存在可利用的能量上升信号，"
            "尚未证明仅凭Clean可见量能够先发现这个角度。"
        )
    else:
        conclusion = (
            "没有任何一种峰错误机制同时通过占比、负控制margin和3/4困难洞方向一致性门。"
            "洞外Train H可以提示这些位置不稳定，但现有局部峰趋势不足以可靠指出应补哪一峰、删哪一峰、往哪里移动或如何改功率。"
        )

    np.savez_compressed(
        EVENT_CACHE,
        query_ids=query_ids,
        fold_id=fold_id,
        neighbor_ids=neighbor_ids,
        neighbor_distances=neighbor_distances.astype(np.float32),
        compact_point_pas=compact_cosine.astype(np.float32),
        official_point_pas=official_stats.pas.astype(np.float32),
    )
    write_csv(HOLE_CSV, hole_rows)
    write_csv(EVENT_CSV, events)
    write_csv(POINT_CSV, point_rows)
    write_csv(PRECURSOR_CSV, precursor_rows)

    report = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "diagnostic_only": True,
        "official_inputs_only": True,
        "sionna_or_v8_v12_read": False,
        "device": str(device),
        "gpu_name": gpu_name,
        "official_v7": official_overall,
        "identity_errors": identity_errors,
        "representation_gate": representation_gate,
        "hard_holes": list(HARD_HOLES),
        "definitions": {
            "compact_pas": "frequency-averaged, transmit-polarization-collapsed, per-UE 16x8 normalized PAS",
            "peak_min": PEAK_MIN,
            "max_peaks": MAX_PEAKS,
            "match_tolerance": {"horizontal_bins": MATCH_H, "vertical_bins": MATCH_V},
            "outside_neighbors": NEIGHBORS,
            "near_count": NEAR_COUNT,
            "far_ranks_zero_based": [FAR_START, NEIGHBORS - 1],
            "control_repeats": CONTROL_REPEATS,
        },
        "per_hole": hole_rows,
        "hard_hole_summary": hard_summary,
        "other_hole_summary": other_summary,
        "mechanism_gates": gate_rows,
        "conclusion": conclusion,
        "outputs": {
            "report_markdown": str(REPORT_MD.relative_to(PATHS.root)),
            "hole_csv": str(HOLE_CSV.relative_to(PATHS.root)),
            "event_csv": str(EVENT_CSV.relative_to(PATHS.root)),
            "point_csv": str(POINT_CSV.relative_to(PATHS.root)),
            "precursor_csv": str(PRECURSOR_CSV.relative_to(PATHS.root)),
            "event_cache": str(EVENT_CACHE.relative_to(PATHS.root)),
            "figures": [str(path.relative_to(PATHS.root)) for path in figures],
        },
        "seconds": time.perf_counter() - started,
    }
    REPORT_JSON.write_text(
        json.dumps(json_compatible(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_report(report, hole_rows, gate_rows, figures)
    print(json.dumps(json_compatible(report), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
