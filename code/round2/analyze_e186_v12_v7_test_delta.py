#!/usr/bin/env python3
"""Measure how the submitted V12 changes V7 on the 500 real Test points."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter, maximum_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import torch

from project_paths import PATHS
from round2.data import load_positions


VERSION = "Round2-E186-v12-v7-real-test-delta"
ANALYSIS_DIR = PATHS.analysis / "results" / "round2_e186_v12_v7_diagnostic"
VIS_DIR = PATHS.analysis / "visualizations" / "复赛E186_V12相对V7真实Test改动"
V7_CHANNEL = (
    PATHS.artifacts
    / "round2"
    / "complex_carrier_lifecycle_overlay"
    / "Round2_V7_Stacked_Lifecycle_Test_Channel.npy"
)
V12_CHANNEL = (
    PATHS.artifacts
    / "round2"
    / "surface_interaction_energy_model"
    / "Round2_E86_Surface_Endpoint_V11_Blend_Test_Channel.npy"
)
TEST_SUMMARY = (
    PATHS.analysis
    / "results"
    / "round2_initial_data_audit"
    / "round2_test_point_summary.csv"
)

POINT_CSV = ANALYSIS_DIR / "真实Test500点_V12相对V7改动.csv"
HOLE_CSV = ANALYSIS_DIR / "真实Test十洞_V12相对V7改动.csv"
EVENT_CSV = ANALYSIS_DIR / "真实Test_V12相对V7峰动作.csv"
PAIR_CSV = ANALYSIS_DIR / "真实Test改动最相似点对.csv"
REPORT_JSON = ANALYSIS_DIR / "test_delta_report.json"
REPORT_MD = ANALYSIS_DIR / "V12相对V7真实Test改动诊断.md"

MP = 2
MH = 16
MV = 8
N_UE = 4
S_SC = 192
TINY = 1e-30

ANGLE_PEAK_MIN = 0.010
ANGLE_MAX_PEAKS = 8
ANGLE_MATCH_H = 2.0
ANGLE_MATCH_V = 1.0
DELAY_PEAK_MIN = 0.001
DELAY_MAX_PEAKS = 8
DELAY_MATCH_BINS = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shuffle-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260812186)
    return parser.parse_args()


def normalize(value: np.ndarray, axes: tuple[int, ...] | int) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = array.sum(axis=axes, keepdims=True, dtype=np.float64)
    return np.divide(
        array,
        np.maximum(total, TINY),
        out=np.zeros_like(array),
        where=total > 0.0,
    )


def cosine(left: np.ndarray, right: np.ndarray, axes: tuple[int, ...] | int) -> np.ndarray:
    numerator = np.sum(left * right, axis=axes, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(left * left, axis=axes, dtype=np.float64)
        * np.sum(right * right, axis=axes, dtype=np.float64)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > TINY,
    )


def load_holes(test_positions: np.ndarray, test_sites: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw_cluster = np.full(len(test_positions), -1, dtype=np.int16)
    with TEST_SUMMARY.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(row["index"])
            if int(row["site"]) != int(test_sites[index]):
                raise RuntimeError(f"site mismatch for Test row {index}")
            raw_cluster[index] = int(row["cluster"])
    labels = np.empty(len(test_positions), dtype="U8")
    ordinal = np.full(len(test_positions), -1, dtype=np.int8)
    for site in (0, 1):
        clusters = sorted(np.unique(raw_cluster[test_sites == site]).tolist())
        if len(clusters) != 5:
            raise RuntimeError(f"site {site} has {len(clusters)} holes")
        for local, cluster in enumerate(clusters):
            mask = (test_sites == site) & (raw_cluster == cluster)
            labels[mask] = f"S{site}H{local}"
            ordinal[mask] = local
    if np.any(ordinal < 0):
        raise RuntimeError("incomplete real Test hole assignment")
    return labels, ordinal


def channel_views(
    channel: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    count = len(channel)
    pas = np.empty((count, N_UE, MH, MV), dtype=np.float32)
    pdp = np.empty((count, MP, N_UE, S_SC), dtype=np.float32)
    joint = np.empty((count, MH, MV, S_SC), dtype=np.float32)
    power = np.empty(count, dtype=np.float64)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        value = torch.as_tensor(
            np.array(channel[start:end], dtype=np.complex64, copy=True), device=device
        )
        with torch.inference_mode():
            shaped = value.reshape(end - start, MP, MH, MV, N_UE, S_SC)
            spatial = torch.fft.fft2(shaped, dim=(2, 3))
            pas_power = spatial.abs().square().float()
            local_pas = pas_power.sum(dim=1).mean(dim=-1).permute(0, 3, 1, 2)
            delay_power = torch.fft.ifft(shaped, dim=-1).abs().square().float()
            local_pdp = delay_power.mean(dim=(2, 3))
            local_joint = torch.fft.ifft(spatial, dim=-1).abs().square().sum(
                dim=(1, 4), dtype=torch.float32
            )
            local_power = value.abs().square().sum(dim=(1, 2, 3), dtype=torch.float64)
        pas[start:end] = local_pas.cpu().numpy()
        pdp[start:end] = local_pdp.cpu().numpy()
        joint[start:end] = local_joint.cpu().numpy()
        power[start:end] = local_power.cpu().numpy()
    return {
        "pas": normalize(pas, (2, 3)),
        "pdp": normalize(pdp, 3),
        "joint": normalize(joint, (1, 2, 3)),
        "power": power,
    }


def periodic_distance(left: int, right: int, period: int) -> float:
    delta = abs(left - right)
    return float(min(delta, period - delta))


def angle_peaks(value: np.ndarray) -> list[tuple[int, int, float, float]]:
    if float(value.sum()) <= TINY:
        return []
    local = maximum_filter(value, size=(3, 3), mode="wrap")
    candidates = np.argwhere((value >= local - 1e-12) & (value >= ANGLE_PEAK_MIN))
    if len(candidates) == 0:
        candidates = np.asarray([np.unravel_index(np.argmax(value), value.shape)])
    order = np.argsort([-float(value[tuple(index)]) for index in candidates])
    output = []
    for slot in order[:ANGLE_MAX_PEAKS]:
        h, v = (int(item) for item in candidates[slot])
        hs = [(h + delta) % MH for delta in (-1, 0, 1)]
        vs = [(v + delta) % MV for delta in (-1, 0, 1)]
        output.append((h, v, float(value[h, v]), float(value[np.ix_(hs, vs)].sum())))
    return output


def delay_peaks(value: np.ndarray) -> list[tuple[int, float, float]]:
    if float(value.sum()) <= TINY:
        return []
    local = maximum_filter1d(value, size=5, mode="wrap")
    candidates = np.flatnonzero((value >= local - 1e-12) & (value >= DELAY_PEAK_MIN))
    if len(candidates) == 0:
        candidates = np.asarray([int(np.argmax(value))])
    order = candidates[np.argsort(-value[candidates])][:DELAY_MAX_PEAKS]
    output = []
    for delay in order:
        indices = [(int(delay) + delta) % S_SC for delta in range(-2, 3)]
        output.append((int(delay), float(value[int(delay)]), float(value[indices].sum())))
    return output


def mass_action(old: float, new: float, absolute_tolerance: float) -> str:
    tolerance = max(absolute_tolerance, 0.05 * max(old, new))
    if new - old > tolerance:
        return "增强"
    if old - new > tolerance:
        return "减弱"
    return "基本不变"


def peak_events(
    v7: dict[str, np.ndarray],
    v12: dict[str, np.ndarray],
    hole_labels: np.ndarray,
    sites: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    events: list[dict[str, object]] = []
    per_point = {
        name: np.zeros(len(hole_labels), dtype=np.float64)
        for name in (
            "angle_new_count",
            "angle_removed_count",
            "angle_moved_count",
            "angle_strengthened_count",
            "angle_weakened_count",
            "angle_new_mass",
            "angle_removed_mass",
            "angle_moved_mass",
            "delay_new_count",
            "delay_removed_count",
            "delay_moved_count",
            "delay_strengthened_count",
            "delay_weakened_count",
            "delay_new_mass",
            "delay_removed_mass",
            "delay_moved_mass",
        )
    }

    for point in range(len(hole_labels)):
        for ue in range(N_UE):
            old = angle_peaks(v7["pas"][point, ue])
            new = angle_peaks(v12["pas"][point, ue])
            cost = np.empty((len(old), len(new)), dtype=np.float64)
            for i, left in enumerate(old):
                for j, right in enumerate(new):
                    dh = periodic_distance(left[0], right[0], MH)
                    dv = periodic_distance(left[1], right[1], MV)
                    cost[i, j] = math.sqrt((dh / ANGLE_MATCH_H) ** 2 + (dv / ANGLE_MATCH_V) ** 2)
            accepted: list[tuple[int, int, float, float, float]] = []
            if len(old) and len(new):
                rows, columns = linear_sum_assignment(cost)
                for i, j in zip(rows, columns, strict=True):
                    if cost[i, j] <= 1.0:
                        dh = periodic_distance(old[i][0], new[j][0], MH)
                        dv = periodic_distance(old[i][1], new[j][1], MV)
                        accepted.append((int(i), int(j), float(cost[i, j]), dh, dv))
            old_used = {item[0] for item in accepted}
            new_used = {item[1] for item in accepted}
            for i, j, match_cost, dh, dv in accepted:
                left, right = old[i], new[j]
                position = "移动" if dh > 0.0 or dv > 0.0 else "保留"
                mass = mass_action(left[3], right[3], 0.002)
                if position == "移动":
                    per_point["angle_moved_count"][point] += 1
                    per_point["angle_moved_mass"][point] += min(left[3], right[3])
                if mass == "增强":
                    per_point["angle_strengthened_count"][point] += 1
                elif mass == "减弱":
                    per_point["angle_weakened_count"][point] += 1
                events.append(
                    {
                        "point": point,
                        "site": int(sites[point]),
                        "hole_id": str(hole_labels[point]),
                        "domain": "角度",
                        "group": f"UE{ue}",
                        "position_action": position,
                        "mass_action": mass,
                        "old_h": left[0],
                        "old_v": left[1],
                        "old_delay": -1,
                        "new_h": right[0],
                        "new_v": right[1],
                        "new_delay": -1,
                        "shift_h": dh,
                        "shift_v": dv,
                        "shift_delay": 0.0,
                        "old_mass": left[3],
                        "new_mass": right[3],
                        "mass_delta": right[3] - left[3],
                        "match_cost": match_cost,
                    }
                )
            for i, left in enumerate(old):
                if i in old_used:
                    continue
                per_point["angle_removed_count"][point] += 1
                per_point["angle_removed_mass"][point] += left[3]
                events.append(
                    {
                        "point": point,
                        "site": int(sites[point]),
                        "hole_id": str(hole_labels[point]),
                        "domain": "角度",
                        "group": f"UE{ue}",
                        "position_action": "消失",
                        "mass_action": "减弱",
                        "old_h": left[0],
                        "old_v": left[1],
                        "old_delay": -1,
                        "new_h": -1,
                        "new_v": -1,
                        "new_delay": -1,
                        "shift_h": float("nan"),
                        "shift_v": float("nan"),
                        "shift_delay": float("nan"),
                        "old_mass": left[3],
                        "new_mass": 0.0,
                        "mass_delta": -left[3],
                        "match_cost": float("nan"),
                    }
                )
            for j, right in enumerate(new):
                if j in new_used:
                    continue
                per_point["angle_new_count"][point] += 1
                per_point["angle_new_mass"][point] += right[3]
                events.append(
                    {
                        "point": point,
                        "site": int(sites[point]),
                        "hole_id": str(hole_labels[point]),
                        "domain": "角度",
                        "group": f"UE{ue}",
                        "position_action": "新增",
                        "mass_action": "增强",
                        "old_h": -1,
                        "old_v": -1,
                        "old_delay": -1,
                        "new_h": right[0],
                        "new_v": right[1],
                        "new_delay": -1,
                        "shift_h": float("nan"),
                        "shift_v": float("nan"),
                        "shift_delay": float("nan"),
                        "old_mass": 0.0,
                        "new_mass": right[3],
                        "mass_delta": right[3],
                        "match_cost": float("nan"),
                    }
                )

        for polarization in range(MP):
            for ue in range(N_UE):
                old = delay_peaks(v7["pdp"][point, polarization, ue])
                new = delay_peaks(v12["pdp"][point, polarization, ue])
                cost = np.empty((len(old), len(new)), dtype=np.float64)
                for i, left in enumerate(old):
                    for j, right in enumerate(new):
                        cost[i, j] = periodic_distance(left[0], right[0], S_SC)
                accepted: list[tuple[int, int, float]] = []
                if len(old) and len(new):
                    rows, columns = linear_sum_assignment(cost)
                    for i, j in zip(rows, columns, strict=True):
                        if cost[i, j] <= DELAY_MATCH_BINS:
                            accepted.append((int(i), int(j), float(cost[i, j])))
                old_used = {item[0] for item in accepted}
                new_used = {item[1] for item in accepted}
                for i, j, shift in accepted:
                    left, right = old[i], new[j]
                    position = "移动" if shift > 0.0 else "保留"
                    mass = mass_action(left[2], right[2], 0.001)
                    if position == "移动":
                        per_point["delay_moved_count"][point] += 1
                        per_point["delay_moved_mass"][point] += min(left[2], right[2])
                    if mass == "增强":
                        per_point["delay_strengthened_count"][point] += 1
                    elif mass == "减弱":
                        per_point["delay_weakened_count"][point] += 1
                    events.append(
                        {
                            "point": point,
                            "site": int(sites[point]),
                            "hole_id": str(hole_labels[point]),
                            "domain": "时延",
                            "group": f"P{polarization}-UE{ue}",
                            "position_action": position,
                            "mass_action": mass,
                            "old_h": -1,
                            "old_v": -1,
                            "old_delay": left[0],
                            "new_h": -1,
                            "new_v": -1,
                            "new_delay": right[0],
                            "shift_h": 0.0,
                            "shift_v": 0.0,
                            "shift_delay": shift,
                            "old_mass": left[2],
                            "new_mass": right[2],
                            "mass_delta": right[2] - left[2],
                            "match_cost": shift,
                        }
                    )
                for i, left in enumerate(old):
                    if i in old_used:
                        continue
                    per_point["delay_removed_count"][point] += 1
                    per_point["delay_removed_mass"][point] += left[2]
                    events.append(
                        {
                            "point": point,
                            "site": int(sites[point]),
                            "hole_id": str(hole_labels[point]),
                            "domain": "时延",
                            "group": f"P{polarization}-UE{ue}",
                            "position_action": "消失",
                            "mass_action": "减弱",
                            "old_h": -1,
                            "old_v": -1,
                            "old_delay": left[0],
                            "new_h": -1,
                            "new_v": -1,
                            "new_delay": -1,
                            "shift_h": float("nan"),
                            "shift_v": float("nan"),
                            "shift_delay": float("nan"),
                            "old_mass": left[2],
                            "new_mass": 0.0,
                            "mass_delta": -left[2],
                            "match_cost": float("nan"),
                        }
                    )
                for j, right in enumerate(new):
                    if j in new_used:
                        continue
                    per_point["delay_new_count"][point] += 1
                    per_point["delay_new_mass"][point] += right[2]
                    events.append(
                        {
                            "point": point,
                            "site": int(sites[point]),
                            "hole_id": str(hole_labels[point]),
                            "domain": "时延",
                            "group": f"P{polarization}-UE{ue}",
                            "position_action": "新增",
                            "mass_action": "增强",
                            "old_h": -1,
                            "old_v": -1,
                            "old_delay": -1,
                            "new_h": -1,
                            "new_v": -1,
                            "new_delay": right[0],
                            "shift_h": float("nan"),
                            "shift_v": float("nan"),
                            "shift_delay": float("nan"),
                            "old_mass": 0.0,
                            "new_mass": right[2],
                            "mass_delta": right[2],
                            "match_cost": float("nan"),
                        }
                    )
    return events, per_point


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


def correction_embedding(v7: dict[str, np.ndarray], v12: dict[str, np.ndarray]) -> np.ndarray:
    pas_delta = np.sqrt(v12["pas"].clip(min=0.0)) - np.sqrt(v7["pas"].clip(min=0.0))
    pdp_delta = np.sqrt(v12["pdp"].clip(min=0.0)) - np.sqrt(v7["pdp"].clip(min=0.0))
    joint_delta = np.sqrt(v12["joint"].clip(min=0.0)) - np.sqrt(v7["joint"].clip(min=0.0))
    # Joint angle-delay is reduced only by fixed pooling, preserving signed transport direction.
    joint_delta = joint_delta.reshape(len(joint_delta), MH, MV, 24, 8).sum(axis=4)
    output = np.concatenate(
        (
            pas_delta.reshape(len(pas_delta), -1),
            pdp_delta.reshape(len(pdp_delta), -1),
            joint_delta.reshape(len(joint_delta), -1),
        ),
        axis=1,
    ).astype(np.float32)
    norm = np.linalg.norm(output, axis=1, keepdims=True)
    return np.divide(output, np.maximum(norm, 1e-12), out=np.zeros_like(output), where=norm > 0.0)


def pair_similarity_analysis(
    embedding: np.ndarray,
    positions: np.ndarray,
    sites: np.ndarray,
    holes: np.ndarray,
    change: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    similarity = embedding @ embedding.T
    np.fill_diagonal(similarity, -np.inf)
    distance = np.linalg.norm(positions[:, None, :2] - positions[None, :, :2], axis=2)
    same_site = sites[:, None] == sites[None, :]
    same_hole = holes[:, None] == holes[None, :]

    nearest_same_hole = np.full(len(positions), -1, dtype=np.int64)
    nearest_cross_hole = np.full(len(positions), -1, dtype=np.int64)
    most_similar = np.full(len(positions), -1, dtype=np.int64)
    rows: list[dict[str, object]] = []
    for point in range(len(positions)):
        local = np.flatnonzero(same_hole[point] & (np.arange(len(positions)) != point))
        nearest_same_hole[point] = local[np.argmin(distance[point, local])]
        cross = np.flatnonzero(same_site[point] & ~same_hole[point])
        nearest_cross_hole[point] = cross[np.argmin(distance[point, cross])]
        candidates = np.flatnonzero(same_site[point] & (np.arange(len(positions)) != point))
        most_similar[point] = candidates[np.argmax(similarity[point, candidates])]
        for kind, other in (
            ("同洞最近邻", nearest_same_hole[point]),
            ("同站跨洞最近邻", nearest_cross_hole[point]),
            ("同站改动最相似", most_similar[point]),
        ):
            rows.append(
                {
                    "point": point,
                    "hole_id": str(holes[point]),
                    "pair_type": kind,
                    "other_point": int(other),
                    "other_hole_id": str(holes[other]),
                    "distance_m": float(distance[point, other]),
                    "correction_cosine": float(similarity[point, other]),
                    "change_ratio_left": float(change[point]),
                    "change_ratio_right": float(change[other]),
                }
            )

    observed = {
        "same_hole_nearest_mean": float(np.mean(similarity[np.arange(len(positions)), nearest_same_hole])),
        "cross_hole_nearest_mean": float(np.mean(similarity[np.arange(len(positions)), nearest_cross_hole])),
        "same_site_best_mean": float(np.mean(similarity[np.arange(len(positions)), most_similar])),
        "same_site_best_cross_hole_fraction": float(np.mean(holes[most_similar] != holes)),
    }

    # Shuffle only within site and change-magnitude quintile. This controls for both
    # propagation site and the trivial fact that near-zero corrections resemble each other.
    quantile = np.zeros(len(change), dtype=np.int8)
    for site in np.unique(sites):
        ids = np.flatnonzero(sites == site)
        edges = np.quantile(change[ids], (0.2, 0.4, 0.6, 0.8))
        quantile[ids] = np.digitize(change[ids], edges).astype(np.int8)
    rng = np.random.default_rng(seed)
    control = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        permuted = np.arange(len(positions))
        for site in np.unique(sites):
            for bucket in range(5):
                ids = np.flatnonzero((sites == site) & (quantile == bucket))
                if len(ids) <= 1:
                    continue
                local = rng.permutation(ids)
                for _ in range(8):
                    collision = local == ids
                    if not np.any(collision):
                        break
                    local[collision] = np.roll(local[collision], 1)
                permuted[ids] = local
        collision = permuted == nearest_same_hole
        if np.any(collision):
            for point in np.flatnonzero(collision):
                candidates = np.flatnonzero(
                    (sites == sites[point])
                    & (quantile == quantile[point])
                    & (np.arange(len(positions)) != nearest_same_hole[point])
                )
                if len(candidates):
                    permuted[point] = int(rng.choice(candidates))
        control[repeat] = np.mean(
            similarity[permuted, nearest_same_hole]
        )
    observed["shuffle_control_mean"] = float(np.mean(control))
    observed["shuffle_control_std"] = float(np.std(control))
    observed["same_hole_margin_over_control"] = float(
        observed["same_hole_nearest_mean"] - np.mean(control)
    )
    observed["shuffle_exceed_fraction"] = float(
        np.mean(control >= observed["same_hole_nearest_mean"])
    )
    return observed, rows


def pca_summary(embedding: np.ndarray) -> dict[str, object]:
    centered = embedding - embedding.mean(axis=0, keepdims=True)
    _, singular, _ = np.linalg.svd(centered, full_matrices=False)
    variance = np.square(singular)
    ratio = variance / np.maximum(variance.sum(), TINY)
    cumulative = np.cumsum(ratio)
    return {
        "explained_first_1": float(cumulative[0]),
        "explained_first_2": float(cumulative[1]),
        "explained_first_3": float(cumulative[2]),
        "explained_first_5": float(cumulative[4]),
        "explained_first_10": float(cumulative[9]),
        "components_for_80_percent": int(np.searchsorted(cumulative, 0.80) + 1),
        "components_for_90_percent": int(np.searchsorted(cumulative, 0.90) + 1),
    }


def summarize_events(events: list[dict[str, object]], mask: np.ndarray | None = None) -> dict[str, object]:
    if mask is None:
        selected = events
    else:
        selected = [event for event in events if mask[int(event["point"])]]
    output: dict[str, object] = {}
    for domain in ("角度", "时延"):
        local = [event for event in selected if event["domain"] == domain]
        output[domain] = {
            "event_count": len(local),
            "position": dict(Counter(str(event["position_action"]) for event in local)),
            "mass": dict(Counter(str(event["mass_action"]) for event in local)),
            "absolute_mass_changed": float(
                np.sum([abs(float(event["mass_delta"])) for event in local], dtype=np.float64)
            ),
        }
    return output


def distribution_summary(value: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(value)),
        "median": float(np.median(value)),
        "p75": float(np.quantile(value, 0.75)),
        "p90": float(np.quantile(value, 0.90)),
        "p95": float(np.quantile(value, 0.95)),
        "max": float(np.max(value)),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    for path in (V7_CHANNEL, V12_CHANNEL, TEST_SUMMARY):
        if not path.exists():
            raise FileNotFoundError(path)
    v7_channel = np.load(V7_CHANNEL, mmap_mode="r")
    v12_channel = np.load(V12_CHANNEL, mmap_mode="r")
    if v7_channel.shape != (500, 256, 4, 192) or v12_channel.shape != v7_channel.shape:
        raise ValueError((v7_channel.shape, v12_channel.shape))
    _, _, test_positions, _, test_sites = load_positions()
    holes, _ = load_holes(test_positions, test_sites)

    view_started = time.time()
    v7 = channel_views(v7_channel, device, args.batch_size)
    v12 = channel_views(v12_channel, device, args.batch_size)
    view_seconds = time.time() - view_started

    delta_energy = np.empty(len(v7_channel), dtype=np.float64)
    reference_energy = np.empty(len(v7_channel), dtype=np.float64)
    for start in range(0, len(v7_channel), args.batch_size):
        end = min(start + args.batch_size, len(v7_channel))
        left = np.asarray(v7_channel[start:end], dtype=np.complex64)
        right = np.asarray(v12_channel[start:end], dtype=np.complex64)
        delta_energy[start:end] = np.sum(np.abs(right - left) ** 2, axis=(1, 2, 3), dtype=np.float64)
        reference_energy[start:end] = np.sum(np.abs(left) ** 2, axis=(1, 2, 3), dtype=np.float64)
    relative_h_change = np.divide(
        delta_energy,
        np.maximum(reference_energy, TINY),
        out=np.zeros_like(delta_energy),
        where=reference_energy > 0.0,
    )
    pas_cosine = cosine(v7["pas"], v12["pas"], (2, 3)).mean(axis=1)
    pdp_cosine = cosine(v7["pdp"], v12["pdp"], 3).mean(axis=(1, 2))
    joint_cosine = cosine(v7["joint"], v12["joint"], (1, 2, 3))
    pas_hellinger = 1.0 - np.sum(
        np.sqrt(np.maximum(v7["pas"] * v12["pas"], 0.0)), axis=(2, 3), dtype=np.float64
    ).mean(axis=1)
    pdp_hellinger = 1.0 - np.sum(
        np.sqrt(np.maximum(v7["pdp"] * v12["pdp"], 0.0)), axis=3, dtype=np.float64
    ).mean(axis=(1, 2))
    power_ratio = np.divide(
        v12["power"],
        np.maximum(v7["power"], TINY),
        out=np.ones_like(v12["power"]),
        where=v7["power"] > 0.0,
    )

    events, event_point = peak_events(v7, v12, holes, test_sites)
    embedding = correction_embedding(v7, v12)
    similarity, pair_rows = pair_similarity_analysis(
        embedding,
        test_positions,
        test_sites,
        holes,
        relative_h_change,
        args.shuffle_repeats,
        args.seed,
    )
    pca = pca_summary(embedding)

    point_rows = []
    for point in range(len(test_positions)):
        row = {
            "point": point,
            "site": int(test_sites[point]),
            "hole_id": str(holes[point]),
            "x": float(test_positions[point, 0]),
            "y": float(test_positions[point, 1]),
            "relative_h_change": float(relative_h_change[point]),
            "pas_cosine_v7_v12": float(pas_cosine[point]),
            "pas_hellinger_change": float(pas_hellinger[point]),
            "pdp_cosine_v7_v12": float(pdp_cosine[point]),
            "pdp_hellinger_change": float(pdp_hellinger[point]),
            "joint_angle_delay_cosine": float(joint_cosine[point]),
            "power_ratio_v12_v7": float(power_ratio[point]),
        }
        row.update({key: float(value[point]) for key, value in event_point.items()})
        point_rows.append(row)

    hole_rows = []
    for hole in sorted(np.unique(holes).tolist()):
        mask = holes == hole
        hole_rows.append(
            {
                "hole_id": hole,
                "site": int(test_sites[mask][0]),
                "count": int(np.sum(mask)),
                "relative_h_change_mean": float(np.mean(relative_h_change[mask])),
                "relative_h_change_median": float(np.median(relative_h_change[mask])),
                "pas_cosine_mean": float(np.mean(pas_cosine[mask])),
                "pas_hellinger_change_mean": float(np.mean(pas_hellinger[mask])),
                "pdp_cosine_mean": float(np.mean(pdp_cosine[mask])),
                "pdp_hellinger_change_mean": float(np.mean(pdp_hellinger[mask])),
                "joint_angle_delay_cosine_mean": float(np.mean(joint_cosine[mask])),
                "power_ratio_mean": float(np.mean(power_ratio[mask])),
                "angle_moved_per_point": float(event_point["angle_moved_count"][mask].mean()),
                "angle_new_per_point": float(event_point["angle_new_count"][mask].mean()),
                "angle_removed_per_point": float(event_point["angle_removed_count"][mask].mean()),
                "delay_moved_per_point": float(event_point["delay_moved_count"][mask].mean()),
                "delay_new_per_point": float(event_point["delay_new_count"][mask].mean()),
                "delay_removed_per_point": float(event_point["delay_removed_count"][mask].mean()),
            }
        )

    write_csv(POINT_CSV, point_rows)
    write_csv(HOLE_CSV, hole_rows)
    write_csv(EVENT_CSV, events)
    write_csv(PAIR_CSV, pair_rows)

    event_summary = summarize_events(events)
    changed_thresholds = {
        "relative_h_change_gt_0_01": int(np.sum(relative_h_change > 0.01)),
        "relative_h_change_gt_0_05": int(np.sum(relative_h_change > 0.05)),
        "pas_hellinger_gt_0_01": int(np.sum(pas_hellinger > 0.01)),
        "pas_hellinger_gt_0_02": int(np.sum(pas_hellinger > 0.02)),
        "pdp_hellinger_gt_0_01": int(np.sum(pdp_hellinger > 0.01)),
        "any_angle_peak_position_change": int(
            np.sum(
                event_point["angle_moved_count"]
                + event_point["angle_new_count"]
                + event_point["angle_removed_count"]
                > 0
            )
        ),
        "any_delay_peak_position_change": int(
            np.sum(
                event_point["delay_moved_count"]
                + event_point["delay_new_count"]
                + event_point["delay_removed_count"]
                > 0
            )
        ),
    }
    payload = {
        "experiment": VERSION,
        "status": "analysis_complete",
        "scope": "real Test has no GT; all statements describe V12 relative to V7 only",
        "inputs": {
            "v7": str(PATHS.relative(V7_CHANNEL)),
            "v12": str(PATHS.relative(V12_CHANNEL)),
            "test_positions": "datasets/round2/Round2_Test_Pos.npy",
        },
        "point_count": 500,
        "hole_count": 10,
        "overall_change": {
            "relative_h_change": distribution_summary(relative_h_change),
            "pas_cosine": distribution_summary(pas_cosine),
            "pas_hellinger_change": distribution_summary(pas_hellinger),
            "pdp_cosine": distribution_summary(pdp_cosine),
            "pdp_hellinger_change": distribution_summary(pdp_hellinger),
            "joint_angle_delay_cosine": distribution_summary(joint_cosine),
            "power_ratio": distribution_summary(power_ratio),
            "threshold_counts": changed_thresholds,
        },
        "peak_actions": event_summary,
        "holes": hole_rows,
        "correction_similarity": similarity,
        "correction_pca": pca,
        "performance": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "view_seconds": float(view_seconds),
            "wall_seconds": float(time.time() - started),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "measured_bottleneck": "sequential reading of two 0.73-GiB channel files and CPU peak matching",
        },
        "outputs": {
            "point_csv": str(PATHS.relative(POINT_CSV)),
            "hole_csv": str(PATHS.relative(HOLE_CSV)),
            "event_csv": str(PATHS.relative(EVENT_CSV)),
            "pair_csv": str(PATHS.relative(PAIR_CSV)),
        },
    }
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    labels = [row["hole_id"] for row in hole_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes[0, 0].bar(x, [row["pas_hellinger_change_mean"] for row in hole_rows], color="#b23a48")
    axes[0, 0].set_title("各真实Test洞 PAS改动幅度")
    axes[0, 0].set_ylabel("Hellinger变化")
    axes[0, 1].bar(x, [row["pdp_hellinger_change_mean"] for row in hole_rows], color="#287271")
    axes[0, 1].set_title("各真实Test洞 PDP改动幅度")
    axes[0, 1].set_ylabel("Hellinger变化")
    axes[1, 0].bar(x - 0.25, [row["angle_moved_per_point"] for row in hole_rows], width=0.25, label="移动")
    axes[1, 0].bar(x, [row["angle_removed_per_point"] for row in hole_rows], width=0.25, label="删除")
    axes[1, 0].bar(x + 0.25, [row["angle_new_per_point"] for row in hole_rows], width=0.25, label="新增")
    axes[1, 0].set_title("每点角度峰动作")
    axes[1, 0].legend()
    axes[1, 1].bar(x - 0.25, [row["delay_moved_per_point"] for row in hole_rows], width=0.25, label="移动")
    axes[1, 1].bar(x, [row["delay_removed_per_point"] for row in hole_rows], width=0.25, label="删除")
    axes[1, 1].bar(x + 0.25, [row["delay_new_per_point"] for row in hole_rows], width=0.25, label="新增")
    axes[1, 1].set_title("每点时延峰动作")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(VIS_DIR / "十个真实Test洞改动概览.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for hole in labels:
        mask = holes == hole
        ax.scatter(
            test_positions[mask, 0],
            test_positions[mask, 1],
            c=pas_hellinger[mask],
            cmap="viridis",
            vmin=float(np.min(pas_hellinger)),
            vmax=float(np.max(pas_hellinger)),
            s=28,
            edgecolors="none",
        )
    image = ax.scatter([], [], c=[], cmap="viridis")
    image.set_clim(float(np.min(pas_hellinger)), float(np.max(pas_hellinger)))
    fig.colorbar(image, ax=ax, label="V12相对V7 PAS改动幅度")
    ax.set_title("真实500个Test点的V12改动空间分布")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    fig.savefig(VIS_DIR / "真实Test点PAS改动空间分布.png", dpi=180)
    plt.close(fig)

    lines = [
        "# V12相对V7真实Test改动诊断",
        "",
        "> 真实Test没有GT。本文只回答V12相对V7改了什么，不判断改对或改错。",
        "",
        "## 结果数据、图",
        "",
        f"- 逐点表：`{PATHS.relative(POINT_CSV)}`",
        f"- 十洞表：`{PATHS.relative(HOLE_CSV)}`",
        f"- 峰动作表：`{PATHS.relative(EVENT_CSV)}`",
        f"- 相似点对：`{PATHS.relative(PAIR_CSV)}`",
        f"- 图：`{PATHS.relative(VIS_DIR / '十个真实Test洞改动概览.png')}`",
        f"- 图：`{PATHS.relative(VIS_DIR / '真实Test点PAS改动空间分布.png')}`",
        "",
        "## 现象",
        "",
        f"V12相对V7的复信道相对改变量中位数为`{np.median(relative_h_change):.6f}`，p90为`{np.quantile(relative_h_change, 0.9):.6f}`。",
        f"PAS改动幅度中位数为`{np.median(pas_hellinger):.6f}`，PDP为`{np.median(pdp_hellinger):.6f}`。",
        f"角度峰动作：`{json.dumps(event_summary['角度']['position'], ensure_ascii=False)}`。",
        f"时延峰动作：`{json.dumps(event_summary['时延']['position'], ensure_ascii=False)}`。",
        f"同洞最近点的改动方向余弦均值为`{similarity['same_hole_nearest_mean']:.6f}`，保持site和改动幅度边际的打乱对照为`{similarity['shuffle_control_mean']:.6f}`，差值`{similarity['same_hole_margin_over_control']:+.6f}`。",
        f"改动PCA前3维解释`{pca['explained_first_3']:.2%}`，达到80%需要`{pca['components_for_80_percent']}`维。",
        "",
        "## 能说明什么问题",
        "",
        "若同洞近邻显著高于打乱对照，说明V12不是逐点无规律乱改，而是在连续空间洞内重复相似的谱运输动作。若差值接近0，则只能说明改动幅度类似，不能证明修正规律具有空间连续性。",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
